from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import pytest

from uriel_v2.probabilistic_lab.budget import checkpoint_steps
from uriel_v2.probabilistic_lab.pilot import build_pilot_jobs, run_pilot
from uriel_v2.probabilistic_lab.problems import build_pilot_problems
from uriel_v2.probabilistic_lab.runner import execute_job
from uriel_v2.probabilistic_lab.schema import BudgetSpec
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
