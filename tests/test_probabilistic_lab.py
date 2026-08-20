from __future__ import annotations

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
