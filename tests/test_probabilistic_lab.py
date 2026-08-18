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
    assert runs["run_id"].is_unique
    assert set(traces["run_id"]) == set(runs["run_id"])
    assert len(features) == 18
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
    assert {"pilot", "phase2", "phase3", "validate"} <= set(action.choices)


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
