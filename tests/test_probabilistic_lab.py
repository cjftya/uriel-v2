from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from uriel_v2.probabilistic_lab.budget import checkpoint_steps
from uriel_v2.probabilistic_lab.cli import build_parser
from uriel_v2.probabilistic_lab.phase2 import build_phase2_jobs, run_phase2
from uriel_v2.probabilistic_lab.pilot import build_pilot_jobs, run_pilot
from uriel_v2.probabilistic_lab.phase3 import run_phase3, validate_phase3_dataset
from uriel_v2.probabilistic_lab.phase4 import run_phase4
from uriel_v2.probabilistic_lab.phase5 import run_phase5
from uriel_v2.probabilistic_lab.phase6 import run_phase6
from uriel_v2.probabilistic_lab.phase7 import run_phase7
from uriel_v2.probabilistic_lab.phase8 import run_phase8
from uriel_v2.probabilistic_lab.phase9 import (
    _fit_failure_models,
    _type_metrics,
    run_phase9,
)
from uriel_v2.probabilistic_lab.phase10 import (
    _fit_survival_distribution,
    _survival_metrics,
    run_phase10,
)
from uriel_v2.probabilistic_lab.phase11 import (
    _fit_continuous_hierarchy,
    run_phase11,
)
from uriel_v2.probabilistic_lab.phase12 import _fit_gate, run_phase12
from uriel_v2.probabilistic_lab.phase13 import (
    _estimate_copula,
    _recalibration_levels,
    run_phase13,
)
from uriel_v2.probabilistic_lab.phase14 import (
    _fit_runtime_scales,
    _rank_candidates,
    run_phase14,
)
from uriel_v2.probabilistic_lab.phase15 import (
    _bootstrap_metric_summary,
    _build_sensitivity_scenarios,
    _scenario_selection_rows,
    run_phase15,
)
from uriel_v2.probabilistic_lab.phase16 import _decide_verdict, run_phase16
from uriel_v2.probabilistic_lab.problems import (
    build_pilot_problems,
    evaluate_objective,
    objective_transformation,
)
from uriel_v2.probabilistic_lab.runner import execute_job
from uriel_v2.probabilistic_lab.schema import AlgorithmSpec, BudgetSpec, JobSpec
from uriel_v2.probabilistic_lab.synthetic import (
    SYNTHETIC_FAMILIES,
    build_synthetic_benchmark,
    materialize_matrix,
    materialize_stream,
)
from uriel_v2.probabilistic_lab.validation import validate_dataset


def _jobs(seed_replicates: int = 1):
    return build_pilot_jobs(
        instances_per_family=1,
        seed_replicates=seed_replicates,
        master_seed=20260819,
        monte_carlo_budget=64,
        random_search_budget=48,
    )


def _scientific_result(bundle):
    result = asdict(bundle.result)
    result.pop("runtime")
    result.pop("failure_time")
    traces = []
    for trace in bundle.traces:
        row = asdict(trace)
        row.pop("elapsed_time")
        traces.append(row)
    return result, traces


def test_budget_checkpoints_are_monotone_and_end_at_budget() -> None:
    budget = BudgetSpec(budget_type="evaluations", total=17)
    assert checkpoint_steps(budget) == (1, 2, 4, 9, 17)


def test_phase9_fits_binary_and_multiclass_failure_models() -> None:
    generator = np.random.default_rng(20260825)
    x_train = generator.normal(size=(120, 6))
    x_validation = generator.normal(size=(24, 6))
    y_failure_train = np.zeros(120, dtype=bool)
    y_failure_train[::4] = True
    y_type_train = np.full(120, None, dtype=object)
    positive_indices = np.flatnonzero(y_failure_train)
    y_type_train[positive_indices[::2]] = "FAIL_NUMERIC"
    y_type_train[positive_indices[1::2]] = "FAIL_TIMEOUT"

    _, failure_probability, conditional, binary_status, type_status = _fit_failure_models(
        x_train,
        x_validation,
        y_failure_train,
        y_type_train,
        ["FAIL_NUMERIC", "FAIL_TIMEOUT"],
        job_id="unit__fold0__failure_distribution",
        master_seed=20260825,
        gradient_boosting_iterations=5,
        beta_prior=(0.5, 0.5),
    )

    assert binary_status == "fitted"
    assert type_status == "fitted"
    assert np.isfinite(failure_probability).all()
    assert np.logical_and(failure_probability >= 0.0, failure_probability <= 1.0).all()
    assert np.allclose(conditional.sum(axis=1), 1.0)
    observed_failure = np.asarray([True, True, False] * 8, dtype=bool)
    observed_type = np.asarray(
        ["FAIL_NUMERIC", "FAIL_TIMEOUT", None] * 8,
        dtype=object,
    )
    metrics = _type_metrics(
        observed_failure,
        observed_type,
        conditional,
        ["FAIL_NUMERIC", "FAIL_TIMEOUT"],
        model_status=type_status,
    )
    assert metrics["type_metric_status"] == "available:fitted"
    assert np.isfinite(metrics["type_log_loss"])


def test_phase10_fits_monotone_censoring_aware_survival() -> None:
    generator = np.random.default_rng(20260826)
    x_train = generator.normal(size=(140, 6))
    x_validation = generator.normal(size=(28, 6))
    budget_train = np.full(140, 100.0)
    event_train = np.zeros(140, dtype=bool)
    event_train[::3] = True
    duration_train = np.full(140, 100.0)
    event_steps = np.asarray([1, 2, 5, 10, 20, 50, 100], dtype=float)
    duration_train[event_train] = np.resize(event_steps, int(event_train.sum()))

    _, reach, _, _ = _fit_survival_distribution(
        x_train,
        x_validation,
        event_train,
        duration_train,
        budget_train,
        job_id="unit__fold0__runtime_survival",
        master_seed=20260826,
        gradient_boosting_iterations=5,
        beta_prior=(0.5, 0.5),
    )

    assert np.isfinite(reach).all()
    assert np.logical_and(reach >= 0.0, reach <= 1.0).all()
    assert (np.diff(reach, axis=1) >= 0.0).all()
    event_validation = np.asarray([True, False] * 14, dtype=bool)
    duration_validation = np.where(event_validation, np.resize(event_steps, 28), 100.0)
    metrics = _survival_metrics(
        event_validation,
        duration_validation,
        np.full(28, 100.0),
        reach,
    )
    assert np.isfinite(metrics["survival_nll"])
    assert np.isfinite(metrics["integrated_brier"])
    assert metrics["c_index_comparable_pair_count"] > 0


def test_phase11_continuous_hierarchy_shrinks_and_handles_unseen_groups() -> None:
    training_groups = pd.DataFrame(
        {
            "domain": ["sampling"] * 8,
            "problem_family": ["a"] * 4 + ["b"] * 4,
            "algorithm_family": ["iid"] * 2 + ["rqmc"] * 2 + ["iid"] * 2 + ["rqmc"] * 2,
        }
    )
    validation_groups = pd.DataFrame(
        {
            "domain": ["sampling", "sampling"],
            "problem_family": ["a", "unseen"],
            "algorithm_family": ["iid", "iid"],
        }
    )
    observed = np.asarray([-2.2, -1.8, -1.2, -0.8, 0.8, 1.2, 1.8, 2.2])
    mean, std, _model, support, diagnostics = _fit_continuous_hierarchy(
        observed,
        np.zeros(len(observed)),
        np.zeros(len(validation_groups)),
        training_groups,
        validation_groups,
        target="unit",
    )

    assert np.isfinite(mean).all()
    assert (std > 0.0).all()
    assert support["shrinkage_weight"].between(0.0, 1.0).all()
    assert support["shrinkage_weight"].between(0.0, 1.0, inclusive="neither").any()
    assert diagnostics["unseen_domain_problem_family_count"] == 1


def test_phase12_gate_is_bounded_and_uses_cross_fitted_loss_labels() -> None:
    generator = np.random.default_rng(20260827)
    x_train = generator.normal(size=(160, 8))
    x_validation = generator.normal(size=(32, 8))
    base_loss = np.square(x_train[:, 0] - 0.25)
    hierarchical_loss = np.square(x_train[:, 0] + 0.25)

    estimator, weight, diagnostics = _fit_gate(
        x_train,
        x_validation,
        base_loss,
        hierarchical_loss,
        seed=20260827,
        gate_iterations=5,
        weight_clip=(0.02, 0.98),
        beta_prior=(0.5, 0.5),
    )

    assert estimator is not None
    assert diagnostics["fit_status"] == "fitted"
    assert diagnostics["training_row_count"] == len(x_train)
    assert np.isfinite(weight).all()
    assert np.logical_and(weight >= 0.02, weight <= 0.98).all()


def test_phase13_recalibration_levels_are_monotone_and_regularized() -> None:
    pit = np.linspace(0.15, 0.95, 300)
    levels, diagnostics = _recalibration_levels(
        pit,
        calibration_strength=50.0,
        lower_bound=0.0,
        upper_bound=1.0,
    )

    assert np.isfinite(levels).all()
    assert np.logical_and(levels >= 0.0, levels <= 1.0).all()
    assert (np.diff(levels) >= 0.0).all()
    assert diagnostics["training_row_count"] == len(pit)
    assert diagnostics["maximum_level_shift"] > 0.0


def test_phase13_copula_is_positive_semidefinite_with_failure_fallback() -> None:
    generator = np.random.default_rng(20260828)
    quality_pit = np.clip(generator.uniform(size=240), 1e-6, 1.0 - 1e-6)
    runtime_pit = np.clip(0.75 * quality_pit + 0.25 * generator.uniform(size=240), 1e-6, 1.0 - 1e-6)
    correlation, diagnostics = _estimate_copula(
        quality_pit,
        runtime_pit,
        np.full(240, 0.01),
        np.zeros(240, dtype=bool),
        pd.Series([f"feature-{index}" for index in range(240)]),
        shrinkage=50.0,
        minimum_class_rows=10,
        seed=20260828,
    )

    assert np.allclose(correlation, correlation.T)
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.linalg.eigvalsh(correlation).min() >= 0.0
    assert correlation[0, 1] > 0.0
    assert correlation[0, 2] == pytest.approx(0.0)
    assert correlation[1, 2] == pytest.approx(0.0)
    assert diagnostics["failure_dependence_status"] == "unavailable_no_failure_events"


def test_phase14_runtime_scales_use_positive_training_domain_medians() -> None:
    training = pd.DataFrame(
        {
            "domain": ["sampling", "sampling", "optimization", "optimization"],
            "observed_runtime": [1.0, 3.0, 4.0, 8.0],
        }
    )
    scales, support = _fit_runtime_scales(training, minimum_runtime_scale=1e-6)

    assert scales == {"optimization": 6.0, "sampling": 2.0}
    assert (support["runtime_scale_seconds"] > 0.0).all()
    assert set(support["support_type"]) == {
        "runtime_scale_global",
        "runtime_scale_domain",
    }


def test_phase14_candidate_ties_are_ranked_by_algorithm_name() -> None:
    candidates = pd.DataFrame(
        {
            "split_name": ["instance_holdout", "instance_holdout"],
            "fold": [0, 0],
            "problem_id": ["problem", "problem"],
            "cutoff": [0.05, 0.05],
            "utility_profile": ["balanced", "balanced"],
            "algorithm": ["z_algorithm", "a_algorithm"],
            "expected_utility": [0.5, 0.5],
            "realized_utility": [0.4, 0.4],
        }
    )
    ranked = _rank_candidates(candidates)

    winner = ranked.loc[ranked["predicted_rank"] == 1].iloc[0]
    oracle = ranked.loc[ranked["realized_rank"] == 1].iloc[0]
    assert winner["algorithm"] == "a_algorithm"
    assert oracle["algorithm"] == "a_algorithm"


def test_phase15_sensitivity_scenarios_are_symmetric_and_preregistered() -> None:
    scenarios = _build_sensitivity_scenarios(0.25)

    assert len(scenarios) == 44
    assert len({scenario["scenario_id"] for scenario in scenarios}) == len(scenarios)
    assert sum(bool(scenario["is_nominal"]) for scenario in scenarios) == 4
    balanced_runtime = [
        scenario
        for scenario in scenarios
        if scenario["base_profile"] == "balanced"
        and scenario["perturbed_component"] == "runtime_weight"
    ]
    assert sorted(scenario["multiplier"] for scenario in balanced_runtime) == [0.75, 1.25]


def test_phase15_cluster_bootstrap_is_deterministic_and_finite() -> None:
    frame = pd.DataFrame(
        {
            "problem_id": ["a", "a", "b", "b", "c", "c"],
            "selection_correct": [True, False, True, True, False, False],
            "oracle_regret": [0.1, 0.2, 0.0, 0.1, 0.3, 0.2],
            "selected_vs_training_global_best_utility_gain": [0.1, 0.0, 0.2, 0.1, -0.1, 0.0],
            "selected_vs_random_utility_gain": [0.2, 0.1, 0.3, 0.2, -0.1, 0.0],
            "selection_entropy": [0.5, 0.6, 0.3, 0.4, 0.8, 0.7],
        }
    )
    first = _bootstrap_metric_summary(
        frame, iterations=100, confidence_level=0.95, seed=20260830
    )
    second = _bootstrap_metric_summary(
        frame, iterations=100, confidence_level=0.95, seed=20260830
    )

    assert first == second
    assert np.isfinite(list(first.values())).all()
    assert first["mean_oracle_regret_ci_lower"] <= first["mean_oracle_regret_ci_upper"]


def test_phase15_sensitivity_ties_use_algorithm_name() -> None:
    candidates = pd.DataFrame(
        {
            "split_name": ["instance_holdout", "instance_holdout"],
            "fold": [0, 0],
            "problem_id": ["problem", "problem"],
            "problem_family": ["family", "family"],
            "domain": ["sampling", "sampling"],
            "cutoff": [0.05, 0.05],
            "utility_profile": ["balanced", "balanced"],
            "algorithm": ["z_algorithm", "a_algorithm"],
            "training_global_best": [False, True],
            "predicted_quality": [0.5, 0.5],
            "predicted_runtime_ratio": [1.0, 1.0],
            "predicted_failure_probability": [0.0, 0.0],
            "predicted_quality_uncertainty": [0.1, 0.1],
            "predicted_sla_probability": [0.5, 0.5],
            "realized_quality": [0.5, 0.5],
            "realized_runtime_ratio": [1.0, 1.0],
            "realized_failure": [0.0, 0.0],
            "realized_sla": [0.0, 0.0],
        }
    )
    scenario = next(
        item
        for item in _build_sensitivity_scenarios(0.25)
        if item["scenario_id"] == "balanced__nominal"
    )

    selection = _scenario_selection_rows(candidates, scenario).iloc[0]
    assert selection["selected_algorithm"] == "a_algorithm"
    assert selection["oracle_algorithm"] == "a_algorithm"


def test_phase16_deployment_requires_every_criterion_to_pass() -> None:
    criteria = pd.DataFrame(
        {
            "criterion_id": [
                "point_prediction_skill",
                "marginal_distribution_calibration",
                "survival_prediction",
                "joint_probability_generalization",
                "failure_risk_estimability",
                "domain_expert_coverage",
                "selection_value_vs_random",
                "selection_value_vs_training_global_best",
                "selection_regret_vs_baselines",
                "selection_policy_robustness",
            ],
            "status": ["PASS"] * 10,
        }
    )
    success = _decide_verdict(criteria)
    assert success["verdict"] == "DEPLOYABLE_SUCCESS"
    assert success["deployment_ready"] is True

    criteria.loc[
        criteria["criterion_id"] == "selection_value_vs_random", "status"
    ] = "FAIL"
    partial = _decide_verdict(criteria)
    assert partial["verdict"] == "PARTIAL_SUCCESS_RESEARCH_ONLY"
    assert partial["deployment_ready"] is False


def test_problem_generation_is_reproducible_and_ids_are_unique() -> None:
    first = build_pilot_problems(3, 20260819)
    second = build_pilot_problems(3, 20260819)
    assert first == second
    assert len(first) == 18
    assert len({problem.problem_id for problem in first}) == len(first)


def test_ill_conditioned_rosenbrock_changes_axis_scaling() -> None:
    problems = build_pilot_problems(6, 20260820)
    ill = next(problem for problem in problems if problem.problem_id == "rosenbrock-0005-ill_conditioned")
    base = replace(
        ill,
        problem_id="rosenbrock-base-control",
        condition_number=1.0,
        extension={**ill.extension, "variant": "base"},
    )
    point = np.full((1, int(ill.dimension)), 0.5)
    assert float(evaluate_objective(ill, point)[0]) > float(evaluate_objective(base, point)[0])


@pytest.mark.parametrize("domain", ["sampling", "optimization"])
def test_same_seed_reproduces_scientific_result(domain: str) -> None:
    job = next(job for job in _jobs() if job.problem.domain == domain)
    first = execute_job(job)
    second = execute_job(job)
    assert _scientific_result(first) == _scientific_result(second)
    assert first.result.status == "SUCCESS"
    assert first.traces[-1].budget_fraction == 1.0


def test_run_id_changes_with_seed_and_does_not_use_runtime() -> None:
    jobs = _jobs(seed_replicates=2)
    paired = [job for job in jobs if job.problem.problem_id == jobs[0].problem.problem_id]
    assert len(paired) == 2
    assert paired[0].run_id != paired[1].run_id
    assert paired[0].run_id == paired[0].run_id


def test_worker_checkpoint_parquet_and_resume(tmp_path) -> None:
    run_directory = tmp_path / "pilot"
    summary = run_pilot(
        run_directory,
        instances_per_family=1,
        seed_replicates=1,
        master_seed=20260819,
        monte_carlo_budget=64,
        random_search_budget=48,
        workers=2,
    )
    assert summary["status"] == "PASS"
    assert summary["problem_count"] == 6
    assert summary["run_count"] == 6
    assert summary["failure_count"] == 0

    checkpoint_before = (run_directory / "checkpoint.jsonl").read_text(encoding="utf-8")
    resumed = run_pilot(
        run_directory,
        instances_per_family=1,
        seed_replicates=1,
        master_seed=20260819,
        monte_carlo_budget=64,
        random_search_budget=48,
        workers=2,
    )
    assert resumed["status"] == "PASS"
    assert (run_directory / "checkpoint.jsonl").read_text(encoding="utf-8") == checkpoint_before

    runs = pd.read_parquet(run_directory / "data/runs/runs.parquet")
    traces = pd.read_parquet(run_directory / "data/traces/common/trace_common.parquet")
    features = pd.read_parquet(run_directory / "data/features/trajectory_features.parquet")
    problem_features = pd.read_parquet(run_directory / "data/features/problem_features.parquet")
    assert runs["run_id"].is_unique
    assert set(traces["run_id"]) == set(runs["run_id"])
    assert len(features) == 18
    assert "problem_seed" not in problem_features.columns
    assert validate_dataset(run_directory)["status"] == "PASS"


def test_resume_rejects_changed_job_configuration(tmp_path) -> None:
    run_directory = tmp_path / "pilot"
    run_pilot(
        run_directory,
        instances_per_family=1,
        seed_replicates=1,
        master_seed=20260819,
        monte_carlo_budget=32,
        random_search_budget=32,
        workers=1,
    )
    with pytest.raises(ValueError, match="job_ids_sha256"):
        run_pilot(
            run_directory,
            instances_per_family=1,
            seed_replicates=1,
            master_seed=20260819,
            monte_carlo_budget=64,
            random_search_budget=32,
            workers=1,
        )


def _phase2_jobs():
    return build_phase2_jobs(
        instances_per_family=1,
        seed_replicates=1,
        master_seed=20260820,
        sampling_budget=64,
        optimization_budget=64,
    )


@pytest.mark.parametrize("algorithm", ["rqmc_sobol", "cma_es"])
def test_phase2_algorithm_reproduces_scientific_result(algorithm: str) -> None:
    job = next(job for job in _phase2_jobs() if job.algorithm.algorithm == algorithm)
    first = execute_job(job)
    second = execute_job(job)
    assert first.result.status == "SUCCESS"
    assert first.result.failure is False
    assert first.traces[-1].step == job.budget.total
    assert _scientific_result(first) == _scientific_result(second)


def test_phase2_jobs_use_complete_paired_seeds() -> None:
    jobs = _phase2_jobs()
    assert len(jobs) == 12
    pairs = {}
    for job in jobs:
        pairs.setdefault((job.problem.problem_id, job.seed), set()).add(job.algorithm.algorithm)
    assert len(pairs) == 6
    for (problem_id, _seed), algorithms in pairs.items():
        expected = {"monte_carlo_iid", "rqmc_sobol"} if "mean" in problem_id else {"random_search", "cma_es"}
        assert algorithms == expected


def test_phase2_pipeline_writes_paired_comparisons(tmp_path) -> None:
    run_directory = tmp_path / "phase2"
    summary = run_phase2(
        run_directory,
        instances_per_family=1,
        seed_replicates=2,
        master_seed=20260820,
        sampling_budget=64,
        optimization_budget=64,
        bootstrap_iterations=100,
        workers=2,
    )
    assert summary["status"] == "PHASE_2_PASS"
    assert summary["problem_count"] == 6
    assert summary["run_count"] == 24
    assert summary["pair_count"] == 12
    assert summary["failure_count"] == 0
    assert set(summary["comparison_results"]) == {
        "sampling_rqmc_vs_iid",
        "optimization_cmaes_vs_random",
    }
    pairs = pd.read_parquet(run_directory / "data/comparisons/paired_runs.parquet")
    problems = pd.read_parquet(run_directory / "data/comparisons/paired_problem_summary.parquet")
    assert pairs["pair_id"].is_unique
    assert len(problems) == 6
    assert set(pairs["winner"]) <= {"baseline", "challenger", "tie"}
    assert validate_dataset(run_directory)["status"] == "PASS"


def test_cli_exposes_phase2_without_changing_phase1() -> None:
    parser = build_parser()
    action = next(item for item in parser._actions if getattr(item, "dest", None) == "command")
    assert {
        "pilot",
        "phase2",
        "phase3",
        "phase4",
        "phase5",
        "phase6",
        "phase7",
        "phase8",
        "phase9",
        "phase10",
        "phase11",
        "phase12",
        "phase13",
        "phase14",
        "phase15",
        "phase16",
        "validate",
    } <= set(action.choices)


def test_phase3_synthetic_benchmark_is_balanced_and_reproducible() -> None:
    first = build_synthetic_benchmark(12, 20260821)
    second = build_synthetic_benchmark(12, 20260821)
    family_count = sum(len(families) for families in SYNTHETIC_FAMILIES.values())
    assert first == second
    assert len(first) == 12 * family_count
    assert len({problem.problem_id for problem in first}) == len(first)
    assert len({problem.problem_seed for problem in first}) == len(first)
    assert {problem.domain for problem in first} == set(SYNTHETIC_FAMILIES)
    for family in {problem.problem_family for problem in first}:
        dimensions = [problem.dimension for problem in first if problem.problem_family == family]
        counts = pd.Series(dimensions).value_counts()
        assert int(counts.max() - counts.min()) <= 1


@pytest.mark.parametrize("family", ["ackley", "griewank", "schwefel"])
def test_phase3_optimization_family_has_zero_at_transformed_optimum(family: str) -> None:
    problems = build_synthetic_benchmark(6, 20260821)
    problem = next(item for item in problems if item.problem_family == family)
    shift, _rotation = objective_transformation(problem)
    objective = float(evaluate_objective(problem, shift.reshape(1, -1))[0])
    assert objective == pytest.approx(0.0, abs=1e-5)


def test_phase3_optimization_objectives_are_finite_and_nonnegative() -> None:
    rng = np.random.Generator(np.random.PCG64(20260821))
    problems = build_synthetic_benchmark(6, 20260821)
    for problem in problems:
        if problem.domain != "optimization":
            continue
        lower = float(problem.extension["lower_bound"])
        upper = float(problem.extension["upper_bound"])
        points = rng.uniform(lower, upper, size=(32, int(problem.dimension)))
        objectives = evaluate_objective(problem, points)
        assert np.isfinite(objectives).all(), problem.problem_id
        assert (objectives >= -1e-12).all(), problem.problem_id


@pytest.mark.parametrize("algorithm", ["monte_carlo_iid", "rqmc_sobol"])
def test_phase3_lognormal_sampling_is_executable(algorithm: str) -> None:
    problem = next(item for item in build_synthetic_benchmark(2, 20260821) if item.problem_family == "lognormal_mean")
    spec = AlgorithmSpec(
        algorithm=algorithm,
        algorithm_family="independent_sampling" if algorithm == "monte_carlo_iid" else "structured_sampling",
        random_mechanism="Independent Sampling" if algorithm == "monte_carlo_iid" else "Structured Sampling",
    )
    job = JobSpec(problem=problem, algorithm=spec, seed=1234, budget=BudgetSpec("samples", 64))
    first = execute_job(job)
    second = execute_job(job)
    assert first.result.status == "SUCCESS"
    assert _scientific_result(first) == _scientific_result(second)


def test_phase3_all_ready_families_execute() -> None:
    problems = build_synthetic_benchmark(1, 20260821)
    for problem in problems:
        if problem.domain not in {"sampling", "optimization"}:
            continue
        algorithms = (
            (
                AlgorithmSpec("monte_carlo_iid", "independent_sampling", "Independent Sampling"),
                AlgorithmSpec("rqmc_sobol", "structured_sampling", "Structured Sampling"),
            )
            if problem.domain == "sampling"
            else (
                AlgorithmSpec("random_search", "random_search", "Independent Sampling"),
                AlgorithmSpec("cma_es", "adaptive_distribution", "Adaptive Distribution"),
            )
        )
        for algorithm in algorithms:
            budget_type = "samples" if problem.domain == "sampling" else "evaluations"
            job = JobSpec(problem, algorithm, 9876, BudgetSpec(budget_type, 64))
            bundle = execute_job(job)
            assert bundle.result.status == "SUCCESS", (problem.problem_family, algorithm.algorithm)
            assert np.isfinite(bundle.result.quality_final)


def test_phase3_staged_problems_materialize_deterministically() -> None:
    problems = build_synthetic_benchmark(1, 20260821)
    matrix_problem = next(item for item in problems if item.domain == "matrix")
    stream_problem = next(item for item in problems if item.problem_family == "concept_drift_stream")
    matrix = materialize_matrix(matrix_problem)
    stream = materialize_stream(stream_problem, size=256)
    assert matrix.shape == (int(matrix_problem.extension["rows"]), int(matrix_problem.extension["columns"]))
    assert np.array_equal(matrix, materialize_matrix(matrix_problem))
    assert stream.shape == (256,)
    assert np.array_equal(stream, materialize_stream(stream_problem, size=256))


def test_phase3_pipeline_writes_valid_benchmark(tmp_path) -> None:
    run_directory = tmp_path / "phase3"
    summary = run_phase3(
        run_directory,
        instances_per_family=6,
        master_seed=20260821,
        folds=3,
        minimum_problem_count=100,
    )
    assert summary["status"] == "PHASE_3_PASS"
    assert summary["problem_count"] == 102
    assert summary["family_count"] == 17
    assert summary["domain_count"] == 4
    assert summary["phase3_checks"]["required_structure_axes_varied"] is True
    assert validate_phase3_dataset(run_directory)["status"] == "PASS"
    index = pd.read_parquet(run_directory / "data/benchmark/benchmark_index.parquet")
    assert set(index["execution_tier"]) == {"ready", "staged"}
    assert set(index["instance_fold"]) == {0, 1, 2}
    problem_features = pd.read_parquet(run_directory / "data/features/problem_features.parquet")
    assert "problem_seed" not in problem_features.columns


def test_phase4_and_phase5_pipeline_pass_quality_gate(tmp_path) -> None:
    benchmark_directory = tmp_path / "phase3"
    phase4_directory = tmp_path / "phase4"
    phase5_directory = tmp_path / "phase5"
    phase3 = run_phase3(
        benchmark_directory,
        instances_per_family=6,
        master_seed=20260821,
        folds=3,
        minimum_problem_count=100,
    )
    assert phase3["status"] == "PHASE_3_PASS"
    phase4 = run_phase4(
        phase4_directory,
        benchmark_directory,
        seed_replicates=2,
        master_seed=20260822,
        sampling_budget=64,
        optimization_budget=64,
        bootstrap_iterations=100,
        workers=2,
    )
    assert phase4["status"] == "PHASE_4_PASS"
    assert phase4["problem_count"] == 60
    assert phase4["run_count"] == 240
    assert phase4["pair_count"] == 120
    phase5 = run_phase5(
        phase5_directory,
        phase4_directory,
        reproducibility_samples_per_algorithm_family=1,
    )
    assert phase5["status"] == "PHASE_5_PASS"
    assert phase5["critical_issue_count"] == 0
    assert phase5["reproducibility_check_count"] == 20
    assert phase5["reproducibility_mismatch_count"] == 0

    phase6_directory = tmp_path / "phase6"
    phase6 = run_phase6(phase6_directory, phase4_directory, phase5_directory)
    assert phase6["status"] == "PHASE_6_PASS"
    assert phase6["feature_row_count"] == 720
    assert phase6["target_row_count"] == 720
    assert phase6["preprocessing_fold_count"] == 6
    features = pd.read_parquet(phase6_directory / "data/features/model_features.parquet")
    targets = pd.read_parquet(phase6_directory / "data/targets/model_targets.parquet")
    assert features["feature_id"].is_unique
    assert not any("seed" in column.lower() for column in features.columns)
    assert not set(features.columns) & {
        "target_quality_final",
        "target_runtime",
        "target_failure",
    }
    assert set(features["feature_id"]) == set(targets["feature_id"])

    phase7_directory = tmp_path / "phase7"
    phase7 = run_phase7(
        phase7_directory,
        phase6_directory,
        random_forest_estimators=4,
        gradient_boosting_iterations=10,
    )
    assert phase7["status"] == "PHASE_7_PASS"
    assert phase7["job_count"] == 54
    assert phase7["prediction_row_count"] == 12_960
    assert phase7["model_artifact_count"] == 54
    assert phase7["fit_status_counts"] == {"fitted": 36, "constant_fallback": 18}
    predictions = pd.read_parquet(phase7_directory / "data/predictions/oof_predictions.parquet")
    assert not predictions.duplicated(["feature_id", "split_name", "target", "model"]).any()
    assert predictions.loc[predictions["target"] == "quality", "prediction"].between(0.0, 1.0).all()
    assert predictions.loc[predictions["target"] == "runtime", "prediction"].ge(0.0).all()
    assert predictions.loc[predictions["target"] == "failure", "prediction"].between(0.0, 1.0).all()

    phase7_manifest_before = (phase7_directory / "manifest.json").read_bytes()
    resumed_phase7 = run_phase7(
        phase7_directory,
        phase6_directory,
        random_forest_estimators=4,
        gradient_boosting_iterations=10,
    )
    assert resumed_phase7["status"] == "PHASE_7_PASS"
    assert (phase7_directory / "manifest.json").read_bytes() == phase7_manifest_before

    phase8_directory = tmp_path / "phase8"
    phase8 = run_phase8(
        phase8_directory,
        phase6_directory,
        phase7_directory,
        gradient_boosting_iterations=8,
    )
    assert phase8["status"] == "PHASE_8_PASS"
    assert phase8["job_count"] == 6
    assert phase8["prediction_row_count"] == 1_440
    assert phase8["model_artifact_count"] == 6
    assert phase8["quantile_count"] == 7
    assert phase8["aggregate_metric_row_count"] == 8
    distribution = pd.read_parquet(
        phase8_directory / "data/predictions/oof_quality_distribution.parquet"
    )
    quantile_columns = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
    assert not distribution.duplicated(["feature_id", "split_name"]).any()
    assert np.isfinite(distribution[quantile_columns].to_numpy(dtype=float)).all()
    assert (np.diff(distribution[quantile_columns].to_numpy(dtype=float), axis=1) >= 0.0).all()
    assert distribution["pit"].between(0.0, 1.0).all()

    phase9_directory = tmp_path / "phase9"
    phase9 = run_phase9(
        phase9_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        gradient_boosting_iterations=8,
        calibration_bins=5,
    )
    assert phase9["status"] == "PHASE_9_PASS"
    assert phase9["job_count"] == 6
    assert phase9["prediction_row_count"] == 1_440
    assert phase9["model_artifact_count"] == 6
    assert phase9["observed_failure_count"] == 0
    assert phase9["estimability_status"] == "NO_OBSERVED_FAILURES"
    assert phase9["binary_fit_status_counts"] == {"beta_binomial_fallback": 6}
    assert phase9["type_fit_status_counts"] == {
        "unavailable_no_observed_failure_types": 6
    }
    failure_probability = pd.read_parquet(
        phase9_directory / "data/predictions/oof_failure_probability.parquet"
    )
    failure_types = pd.read_parquet(
        phase9_directory / "data/predictions/oof_failure_type_probability.parquet"
    )
    assert not failure_probability.duplicated(["feature_id", "split_name"]).any()
    assert failure_probability["failure_probability"].between(0.0, 1.0).all()
    assert (failure_probability["failure_probability"] > 0.0).all()
    assert failure_types.empty

    phase9_manifest_before = (phase9_directory / "manifest.json").read_bytes()
    resumed_phase9 = run_phase9(
        phase9_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        gradient_boosting_iterations=8,
        calibration_bins=5,
    )
    assert resumed_phase9["status"] == "PHASE_9_PASS"
    assert (phase9_directory / "manifest.json").read_bytes() == phase9_manifest_before

    phase10_directory = tmp_path / "phase10"
    phase10 = run_phase10(
        phase10_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        gradient_boosting_iterations=5,
    )
    assert phase10["status"] == "PHASE_10_PASS"
    assert phase10["job_count"] == 6
    assert phase10["prediction_row_count"] == 1_440
    assert phase10["model_artifact_count"] == 6
    assert phase10["quantile_count"] == 7
    assert phase10["survival_horizon_count"] == 7
    assert phase10["observed_event_count"] + phase10["censored_count"] == 720
    runtime_survival = pd.read_parquet(
        phase10_directory / "data/predictions/oof_runtime_survival.parquet"
    )
    runtime_columns = [
        "runtime_q05",
        "runtime_q10",
        "runtime_q25",
        "runtime_q50",
        "runtime_q75",
        "runtime_q90",
        "runtime_q95",
    ]
    reach_columns = [
        "reach_by_p001",
        "reach_by_p002",
        "reach_by_p005",
        "reach_by_p010",
        "reach_by_p020",
        "reach_by_p050",
        "reach_by_p100",
    ]
    assert not runtime_survival.duplicated(["feature_id", "split_name"]).any()
    assert (runtime_survival[runtime_columns].to_numpy(dtype=float) > 0.0).all()
    assert (np.diff(runtime_survival[runtime_columns].to_numpy(dtype=float), axis=1) >= 0.0).all()
    assert (np.diff(runtime_survival[reach_columns].to_numpy(dtype=float), axis=1) >= 0.0).all()

    phase11_directory = tmp_path / "phase11"
    phase11 = run_phase11(
        phase11_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        phase10_directory,
        ridge_alpha=5.0,
        prior_strength=10.0,
    )
    assert phase11["status"] == "PHASE_11_PASS"
    assert phase11["job_count"] == 6
    assert phase11["prediction_row_count"] == 1_440
    assert phase11["model_artifact_count"] == 6
    assert phase11["posterior_row_count"] > 0
    assert phase11["checks"]["family_holdout_uses_unseen_family_fallback"] is True
    hierarchical = pd.read_parquet(
        phase11_directory / "data/predictions/oof_hierarchical_predictions.parquet"
    )
    quality_hierarchical_columns = [
        "quality_q05",
        "quality_q10",
        "quality_q25",
        "quality_q50",
        "quality_q75",
        "quality_q90",
        "quality_q95",
    ]
    assert not hierarchical.duplicated(["feature_id", "split_name"]).any()
    assert (
        np.diff(
            hierarchical[quality_hierarchical_columns].to_numpy(dtype=float), axis=1
        )
        >= 0.0
    ).all()
    assert hierarchical["failure_probability"].between(0.0, 1.0).all()

    phase11_manifest_before = (phase11_directory / "manifest.json").read_bytes()
    resumed_phase11 = run_phase11(
        phase11_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        phase10_directory,
        ridge_alpha=5.0,
        prior_strength=10.0,
    )
    assert resumed_phase11["status"] == "PHASE_11_PASS"
    assert (phase11_directory / "manifest.json").read_bytes() == phase11_manifest_before

    phase12_directory = tmp_path / "phase12"
    phase12 = run_phase12(
        phase12_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        phase10_directory,
        phase11_directory,
        gate_iterations=5,
        minimum_gate_rows=20,
    )
    assert phase12["status"] == "PHASE_12_PASS"
    assert phase12["job_count"] == 6
    assert phase12["prediction_row_count"] == 1_440
    assert phase12["model_artifact_count"] == 6
    assert phase12["gate_support_row_count"] == 144
    assert phase12["expert_slot_count"] == 6
    assert phase12["checks"]["meta_gate_training_is_cross_fitted"] is True
    assert phase12["checks"]["unexecuted_experts_explicit"] is True
    assert phase12["checks"]["mixture_identity_exact"] is True
    assert phase12["unavailable_expert_slots"] == [
        "matrix",
        "natural_process",
        "stream",
    ]
    mixture = pd.read_parquet(
        phase12_directory / "data/predictions/oof_mixture_predictions.parquet"
    )
    mixture_quality_columns = [f"moe_{column}" for column in quality_hierarchical_columns]
    mixture_runtime_columns = [f"moe_{column}" for column in runtime_columns]
    mixture_reach_columns = [f"moe_{column}" for column in reach_columns]
    assert not mixture.duplicated(["feature_id", "split_name"]).any()
    assert (
        np.diff(mixture[mixture_quality_columns].to_numpy(dtype=float), axis=1) >= 0.0
    ).all()
    assert (
        np.diff(mixture[mixture_runtime_columns].to_numpy(dtype=float), axis=1) >= 0.0
    ).all()
    assert (
        np.diff(mixture[mixture_reach_columns].to_numpy(dtype=float), axis=1) >= 0.0
    ).all()
    for target in ("quality", "runtime", "failure", "survival"):
        assert mixture[f"{target}_hierarchical_weight"].between(0.0, 1.0).all()

    phase12_manifest_before = (phase12_directory / "manifest.json").read_bytes()
    resumed_phase12 = run_phase12(
        phase12_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        phase10_directory,
        phase11_directory,
        gate_iterations=5,
        minimum_gate_rows=20,
    )
    assert resumed_phase12["status"] == "PHASE_12_PASS"
    assert (phase12_directory / "manifest.json").read_bytes() == phase12_manifest_before

    phase13_directory = tmp_path / "phase13"
    phase13 = run_phase13(
        phase13_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        phase10_directory,
        phase11_directory,
        phase12_directory,
        calibration_strength=20.0,
        minimum_class_rows=5,
        copula_shrinkage=20.0,
    )
    assert phase13["status"] == "PHASE_13_PASS"
    assert phase13["job_count"] == 6
    assert phase13["prediction_row_count"] == 1_440
    assert phase13["model_artifact_count"] == 6
    assert phase13["calibration_support_row_count"] == 150
    assert phase13["failure_dependence_estimability"] == "UNAVAILABLE_NO_OBSERVED_FAILURES"
    assert phase13["checks"]["calibration_training_is_cross_fitted"] is True
    assert phase13["checks"]["copula_correlation_matrices_valid"] is True
    assert phase13["checks"]["joint_probabilities_nested"] is True
    joint = pd.read_parquet(
        phase13_directory / "data/predictions/oof_joint_calibrated_predictions.parquet"
    )
    joint_quality_columns = [f"quality_{column.removeprefix('quality_')}" for column in quality_hierarchical_columns]
    joint_runtime_columns = runtime_columns
    joint_reach_columns = reach_columns
    assert not joint.duplicated(["feature_id", "split_name"]).any()
    assert (np.diff(joint[joint_quality_columns].to_numpy(dtype=float), axis=1) >= 0.0).all()
    assert (np.diff(joint[joint_runtime_columns].to_numpy(dtype=float), axis=1) >= 0.0).all()
    assert (np.diff(joint[joint_reach_columns].to_numpy(dtype=float), axis=1) >= 0.0).all()
    assert joint["joint_nll"].map(np.isfinite).all()
    assert (
        joint["joint_quality_ge_q090_runtime_within_budget_no_failure_probability"]
        <= joint["joint_quality_ge_q075_runtime_within_budget_no_failure_probability"]
    ).all()

    phase13_manifest_before = (phase13_directory / "manifest.json").read_bytes()
    resumed_phase13 = run_phase13(
        phase13_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        phase10_directory,
        phase11_directory,
        phase12_directory,
        calibration_strength=20.0,
        minimum_class_rows=5,
        copula_shrinkage=20.0,
    )
    assert resumed_phase13["status"] == "PHASE_13_PASS"
    assert (phase13_directory / "manifest.json").read_bytes() == phase13_manifest_before

    phase14_directory = tmp_path / "phase14"
    phase14 = run_phase14(
        phase14_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        phase10_directory,
        phase11_directory,
        phase12_directory,
        phase13_directory,
    )
    assert phase14["status"] == "PHASE_14_PASS"
    assert phase14["job_count"] == 6
    assert phase14["candidate_row_count"] == 2_880
    assert phase14["selection_row_count"] == 1_440
    assert phase14["fold_metric_row_count"] == 24
    assert phase14["aggregate_metric_row_count"] == 96
    assert phase14["utility_profile_count"] == 4
    assert phase14["primary_utility_profile"] == "balanced"
    assert phase14["checks"]["decision_training_is_cross_fitted"] is True
    assert phase14["checks"]["runtime_units_not_mixed_with_budget"] is True
    assert phase14["checks"]["rank_permutations_and_flags_valid"] is True
    candidates = pd.read_parquet(
        phase14_directory / "data/decisions/oof_algorithm_candidates.parquet"
    )
    selections = pd.read_parquet(
        phase14_directory / "data/decisions/oof_algorithm_selections.parquet"
    )
    assert not candidates.duplicated(
        ["split_name", "problem_id", "cutoff", "utility_profile", "algorithm"]
    ).any()
    assert (candidates.groupby(
        ["split_name", "problem_id", "cutoff", "utility_profile"]
    )["selected_by_policy"].sum() == 1).all()
    assert (selections["oracle_regret"] >= 0.0).all()
    assert selections["selection_entropy"].between(0.0, 1.0).all()

    phase14_manifest_before = (phase14_directory / "manifest.json").read_bytes()
    resumed_phase14 = run_phase14(
        phase14_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        phase10_directory,
        phase11_directory,
        phase12_directory,
        phase13_directory,
    )
    assert resumed_phase14["status"] == "PHASE_14_PASS"
    assert (phase14_directory / "manifest.json").read_bytes() == phase14_manifest_before

    phase15_directory = tmp_path / "phase15"
    phase15 = run_phase15(
        phase15_directory,
        phase14_directory,
        bootstrap_iterations=100,
    )
    assert phase15["status"] == "PHASE_15_PASS"
    assert phase15["scenario_count"] == 44
    assert phase15["sensitivity_selection_row_count"] == 15_840
    assert phase15["sensitivity_metric_row_count"] == 88
    assert phase15["perturbation_stability_row_count"] == 1_440
    assert phase15["profile_stability_row_count"] == 360
    assert phase15["cross_split_agreement_row_count"] == 720
    assert phase15["heldout_metric_row_count"] == 8
    assert phase15["checks"]["phase14_nominal_selection_reproduced"] is True
    assert phase15["checks"]["performance_not_used_as_gate"] is True
    sensitivity = pd.read_parquet(
        phase15_directory / "data/robustness/utility_sensitivity_selections.parquet"
    )
    assert not sensitivity.duplicated(
        ["split_name", "problem_id", "cutoff", "base_profile", "scenario_id"]
    ).any()
    assert sensitivity["oracle_regret"].ge(0.0).all()

    phase15_manifest_before = (phase15_directory / "manifest.json").read_bytes()
    resumed_phase15 = run_phase15(
        phase15_directory,
        phase14_directory,
        bootstrap_iterations=100,
    )
    assert resumed_phase15["status"] == "PHASE_15_PASS"
    assert (phase15_directory / "manifest.json").read_bytes() == phase15_manifest_before

    phase16_directory = tmp_path / "phase16"
    phase16 = run_phase16(
        phase16_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        phase10_directory,
        phase11_directory,
        phase12_directory,
        phase13_directory,
        phase14_directory,
        phase15_directory,
    )
    assert phase16["status"] == "PHASE_16_PASS"
    assert phase16["project_complete"] is True
    assert phase16["final_decision"]["verdict"] == "PARTIAL_SUCCESS_RESEARCH_ONLY"
    assert phase16["final_decision"]["deployment_ready"] is False
    assert phase16["criterion_count"] == 10
    assert phase16["metric_row_count"] == 31
    assert phase16["checks"]["verdict_recomputed_exact"] is True
    assert "failure_risk_estimability" in phase16["failed_deployment_criteria"]
    final_assessment = json.loads(
        (phase16_directory / "final_assessment.json").read_text(encoding="utf-8")
    )
    assert final_assessment["final_decision"] == phase16["final_decision"]

    phase16_manifest_before = (phase16_directory / "manifest.json").read_bytes()
    resumed_phase16 = run_phase16(
        phase16_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        phase10_directory,
        phase11_directory,
        phase12_directory,
        phase13_directory,
        phase14_directory,
        phase15_directory,
    )
    assert resumed_phase16["status"] == "PHASE_16_PASS"
    assert (phase16_directory / "manifest.json").read_bytes() == phase16_manifest_before

    (phase16_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 16 resume manifest hash mismatch"):
        run_phase16(
            phase16_directory,
            phase6_directory,
            phase7_directory,
            phase8_directory,
            phase9_directory,
            phase10_directory,
            phase11_directory,
            phase12_directory,
            phase13_directory,
            phase14_directory,
            phase15_directory,
        )

    (phase15_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 15 resume manifest hash mismatch"):
        run_phase15(
            phase15_directory,
            phase14_directory,
            bootstrap_iterations=100,
        )

    (phase14_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 14 resume manifest hash mismatch"):
        run_phase14(
            phase14_directory,
            phase6_directory,
            phase7_directory,
            phase8_directory,
            phase9_directory,
            phase10_directory,
            phase11_directory,
            phase12_directory,
            phase13_directory,
        )

    (phase13_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 13 resume manifest hash mismatch"):
        run_phase13(
            phase13_directory,
            phase6_directory,
            phase7_directory,
            phase8_directory,
            phase9_directory,
            phase10_directory,
            phase11_directory,
            phase12_directory,
            calibration_strength=20.0,
            minimum_class_rows=5,
            copula_shrinkage=20.0,
        )

    (phase12_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 12 resume manifest hash mismatch"):
        run_phase12(
            phase12_directory,
            phase6_directory,
            phase7_directory,
            phase8_directory,
            phase9_directory,
            phase10_directory,
            phase11_directory,
            gate_iterations=5,
            minimum_gate_rows=20,
        )

    (phase11_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 11 resume manifest hash mismatch"):
        run_phase11(
            phase11_directory,
            phase6_directory,
            phase7_directory,
            phase8_directory,
            phase9_directory,
            phase10_directory,
            ridge_alpha=5.0,
            prior_strength=10.0,
        )

    phase10_manifest_before = (phase10_directory / "manifest.json").read_bytes()
    resumed_phase10 = run_phase10(
        phase10_directory,
        phase6_directory,
        phase7_directory,
        phase8_directory,
        phase9_directory,
        gradient_boosting_iterations=5,
    )
    assert resumed_phase10["status"] == "PHASE_10_PASS"
    assert (phase10_directory / "manifest.json").read_bytes() == phase10_manifest_before

    (phase10_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 10 resume manifest hash mismatch"):
        run_phase10(
            phase10_directory,
            phase6_directory,
            phase7_directory,
            phase8_directory,
            phase9_directory,
            gradient_boosting_iterations=5,
        )

    (phase9_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 9 resume manifest hash mismatch"):
        run_phase9(
            phase9_directory,
            phase6_directory,
            phase7_directory,
            phase8_directory,
            gradient_boosting_iterations=8,
            calibration_bins=5,
        )

    phase8_manifest_before = (phase8_directory / "manifest.json").read_bytes()
    resumed_phase8 = run_phase8(
        phase8_directory,
        phase6_directory,
        phase7_directory,
        gradient_boosting_iterations=8,
    )
    assert resumed_phase8["status"] == "PHASE_8_PASS"
    assert (phase8_directory / "manifest.json").read_bytes() == phase8_manifest_before

    (phase8_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 8 resume manifest hash mismatch"):
        run_phase8(
            phase8_directory,
            phase6_directory,
            phase7_directory,
            gradient_boosting_iterations=8,
        )

    (phase7_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 7 resume manifest hash mismatch"):
        run_phase7(
            phase7_directory,
            phase6_directory,
            random_forest_estimators=4,
            gradient_boosting_iterations=10,
        )

    manifest_before = (phase6_directory / "manifest.json").read_bytes()
    resumed = run_phase6(phase6_directory, phase4_directory, phase5_directory)
    assert resumed["status"] == "PHASE_6_PASS"
    assert (phase6_directory / "manifest.json").read_bytes() == manifest_before

    (phase6_directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed Phase 6 resume manifest hash mismatch"):
        run_phase6(phase6_directory, phase4_directory, phase5_directory)
