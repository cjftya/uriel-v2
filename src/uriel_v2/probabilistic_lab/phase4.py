from __future__ import annotations

import hashlib
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.phase2 import (
    COMPARISONS,
    build_paired_comparisons,
    core_algorithms,
)
from uriel_v2.probabilistic_lab.phase3 import load_phase3_problems, validate_phase3_dataset
from uriel_v2.probabilistic_lab.schema import BudgetSpec, JobSpec, canonical_json
from uriel_v2.probabilistic_lab.storage import write_dataset
from uriel_v2.probabilistic_lab.validation import validate_dataset
from uriel_v2.probabilistic_lab.worker import run_jobs
from uriel_v2.provenance import current_git_commit


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def build_phase4_jobs(
    benchmark_directory: str | Path,
    *,
    seed_replicates: int,
    master_seed: int,
    sampling_budget: int,
    optimization_budget: int,
) -> list[JobSpec]:
    if seed_replicates <= 0:
        raise ValueError("seed_replicates must be positive")
    problems = load_phase3_problems(benchmark_directory, execution_tier="ready")
    algorithms = core_algorithms()
    jobs: list[JobSpec] = []
    for problem in problems:
        algorithm_names = (
            ("monte_carlo_iid", "rqmc_sobol")
            if problem.domain == "sampling"
            else ("random_search", "cma_es")
        )
        budget = BudgetSpec(
            "samples" if problem.domain == "sampling" else "evaluations",
            sampling_budget if problem.domain == "sampling" else optimization_budget,
        )
        for replicate in range(seed_replicates):
            sequence = np.random.SeedSequence([master_seed, problem.problem_seed, replicate, 4])
            paired_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
            for algorithm_name in algorithm_names:
                jobs.append(
                    JobSpec(
                        problem=problem,
                        algorithm=algorithms[algorithm_name],
                        seed=paired_seed,
                        budget=budget,
                        rng_version=f"NumPy {np.__version__}",
                    )
                )
    return jobs


def _job_ids_sha256(jobs: list[JobSpec]) -> str:
    return hashlib.sha256("\n".join(sorted(job.run_id for job in jobs)).encode("utf-8")).hexdigest()


def run_phase4(
    run_directory: str | Path,
    benchmark_directory: str | Path,
    *,
    seed_replicates: int = 10,
    master_seed: int = 20_260_822,
    sampling_budget: int = 1_024,
    optimization_budget: int = 1_024,
    bootstrap_iterations: int = 10_000,
    workers: str | int = "auto",
    resume: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    benchmark_path = Path(benchmark_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    benchmark_validation = validate_phase3_dataset(benchmark_path)
    if benchmark_validation["status"] != "PASS":
        raise ValueError("Phase 4 requires a valid Phase 3 benchmark")
    benchmark_manifest = json.loads((benchmark_path / "manifest.json").read_text(encoding="utf-8"))
    parameters = {
        "seed_replicates": seed_replicates,
        "master_seed": master_seed,
        "sampling_budget": sampling_budget,
        "optimization_budget": optimization_budget,
        "bootstrap_iterations": bootstrap_iterations,
    }
    jobs = build_phase4_jobs(
        benchmark_path,
        seed_replicates=seed_replicates,
        master_seed=master_seed,
        sampling_budget=sampling_budget,
        optimization_budget=optimization_budget,
    )
    preregistration = {
        "phase": 4,
        "scope": "all Phase 3 ready-tier problems, two paired algorithms per domain",
        "pair_key": ["problem_id", "seed", "budget_type", "budget"],
        "inference_unit": "problem-instance mean across paired seeds",
        "seed_policy": "same numeric seed within each algorithm pair; seed is excluded from model feature tables",
        "completion_criteria": {
            "all_planned_jobs_present": True,
            "all_problem_algorithm_seed_replicates_complete": True,
            "paired_comparison_coverage_complete": True,
            "generic_dataset_validation_pass": True,
            "no_execution_failures": True,
        },
    }
    configuration = {
        "phase": 4,
        "parameters": parameters,
        "benchmark_directory": str(benchmark_path.resolve()),
        "benchmark_version": benchmark_manifest["benchmark_version"],
        "benchmark_problem_metadata_sha256": benchmark_manifest["problem_metadata_sha256"],
        "benchmark_index_sha256": benchmark_manifest["benchmark_index_sha256"],
        "benchmark_ready_problem_count": benchmark_validation["ready_count"],
        "comparisons": COMPARISONS,
        "job_count": len(jobs),
        "job_ids_sha256": _job_ids_sha256(jobs),
        "preregistration_sha256": hashlib.sha256(
            canonical_json(preregistration).encode("utf-8")
        ).hexdigest(),
        "git_commit": current_git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    config_path = run_path / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("job_ids_sha256", "preregistration_sha256", "benchmark_problem_metadata_sha256"):
            if existing[key] != configuration[key]:
                raise ValueError(f"resume configuration mismatch: {key}")
    else:
        _json_dump(config_path, configuration)
        _json_dump(run_path / "preregistration.json", preregistration)

    progress_interval = max(1, len(jobs) // 100)
    records = run_jobs(
        jobs,
        run_path,
        workers=workers,
        resume=resume,
        logger=logger,
        progress_interval=progress_interval,
    )
    paths = write_dataset(run_path, jobs, records)
    validation = validate_dataset(run_path)
    runs = pd.read_parquet(paths["runs"])
    problems = pd.read_parquet(paths["problems"])
    pairs, problem_summary, comparisons = build_paired_comparisons(
        runs,
        problems,
        bootstrap_seed=master_seed,
        bootstrap_iterations=bootstrap_iterations,
    )
    comparison_directory = run_path / "data/comparisons"
    comparison_directory.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(comparison_directory / "paired_runs.parquet", index=False)
    problem_summary.to_parquet(comparison_directory / "paired_problem_summary.parquet", index=False)

    benchmark_index = pd.read_parquet(benchmark_path / "data/benchmark/benchmark_index.parquet")
    benchmark_index = benchmark_index[benchmark_index["problem_id"].isin(set(problems["problem_id"]))].copy()
    benchmark_output = run_path / "data/benchmark"
    benchmark_output.mkdir(parents=True, exist_ok=True)
    benchmark_index.to_parquet(benchmark_output / "benchmark_index.parquet", index=False)
    run_splits = runs[["run_id", "problem_id", "algorithm", "seed"]].merge(
        benchmark_index[["problem_id", "instance_fold", "family_holdout_fold"]],
        on="problem_id",
        how="left",
        validate="many_to_one",
    )
    split_directory = run_path / "data/splits"
    split_directory.mkdir(parents=True, exist_ok=True)
    run_splits.to_parquet(split_directory / "run_splits.parquet", index=False)

    expected_pairs = len(jobs) // 2
    replicate_counts = runs.groupby(["problem_id", "algorithm"]).size()
    numeric_pair_columns = [
        "quality_final_baseline",
        "quality_final_challenger",
        "quality_difference",
        "objective_ratio",
        "log10_objective_gain",
        "runtime_ratio",
    ]
    phase4_checks = {
        "benchmark_validation_pass": benchmark_validation["status"] == "PASS",
        "dataset_validation_pass": validation["status"] == "PASS",
        "all_planned_jobs_present": len(runs) == len(jobs) and set(runs["run_id"]) == {job.run_id for job in jobs},
        "all_problem_algorithm_seed_replicates_complete": bool((replicate_counts == seed_replicates).all()),
        "complete_pair_coverage": len(pairs) == expected_pairs and pairs["pair_id"].is_unique,
        "all_pair_numeric_values_finite": bool(
            np.isfinite(pairs[numeric_pair_columns].to_numpy(dtype=float)).all()
        ),
        "all_problem_splits_present": bool(
            run_splits[["instance_fold", "family_holdout_fold"]].notna().all().all()
        ),
        "seed_excluded_from_problem_features": "problem_seed"
        not in pd.read_parquet(paths["problem_features"]).columns,
        "no_execution_failures": int(runs["failure"].sum()) == 0,
    }
    status = "PHASE_4_PASS" if all(phase4_checks.values()) else "PHASE_4_FAIL"
    summary = {
        "status": status,
        "scope": "Phase 4 large-scale paired repeated execution",
        "executed_at": datetime.now(_timezone()).isoformat(timespec="seconds"),
        "elapsed_seconds": perf_counter() - started_at,
        "problem_count": validation["problem_count"],
        "run_count": validation["run_count"],
        "pair_count": int(len(pairs)),
        "trace_count": validation["trace_count"],
        "trajectory_feature_count": validation["trajectory_feature_count"],
        "failure_count": validation["failure_count"],
        "phase4_checks": phase4_checks,
        "comparison_results": comparisons,
        "configuration": configuration,
    }
    _json_dump(run_path / "validation.json", validation)
    _json_dump(run_path / "comparison_summary.json", comparisons)
    _json_dump(run_path / "summary.json", summary)
    return summary
