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
from uriel_v2.probabilistic_lab.problems import build_pilot_problems
from uriel_v2.probabilistic_lab.schema import AlgorithmSpec, BudgetSpec, JobSpec
from uriel_v2.probabilistic_lab.storage import write_dataset
from uriel_v2.probabilistic_lab.validation import validate_dataset
from uriel_v2.probabilistic_lab.worker import run_jobs
from uriel_v2.provenance import current_git_commit


def build_pilot_jobs(
    *,
    instances_per_family: int,
    seed_replicates: int,
    master_seed: int,
    monte_carlo_budget: int,
    random_search_budget: int,
) -> list[JobSpec]:
    if seed_replicates <= 0:
        raise ValueError("seed_replicates must be positive")
    problems = build_pilot_problems(instances_per_family, master_seed)
    monte_carlo = AlgorithmSpec(
        algorithm="monte_carlo_iid",
        algorithm_family="independent_sampling",
        random_mechanism="Independent Sampling",
        configuration={"estimator": "sample_mean"},
    )
    random_search = AlgorithmSpec(
        algorithm="random_search",
        algorithm_family="random_search",
        random_mechanism="Independent Sampling",
        configuration={"proposal": "uniform_within_bounds"},
    )
    jobs: list[JobSpec] = []
    for problem_index, problem in enumerate(problems):
        algorithm = monte_carlo if problem.domain == "sampling" else random_search
        budget = BudgetSpec(
            budget_type="samples" if problem.domain == "sampling" else "evaluations",
            total=monte_carlo_budget if problem.domain == "sampling" else random_search_budget,
        )
        for replicate in range(seed_replicates):
            sequence = np.random.SeedSequence([master_seed, problem_index, replicate])
            run_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
            jobs.append(
                JobSpec(
                    problem=problem,
                    algorithm=algorithm,
                    seed=run_seed,
                    budget=budget,
                    rng_version=f"NumPy {np.__version__}",
                )
            )
    return jobs


def _configuration(jobs: list[JobSpec], parameters: dict[str, Any]) -> dict[str, Any]:
    job_ids = sorted(job.run_id for job in jobs)
    return {
        "parameters": parameters,
        "job_count": len(jobs),
        "job_ids_sha256": hashlib.sha256("\n".join(job_ids).encode("utf-8")).hexdigest(),
        "git_commit": current_git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def run_pilot(
    run_directory: str | Path,
    *,
    instances_per_family: int = 4,
    seed_replicates: int = 3,
    master_seed: int = 20_260_819,
    monte_carlo_budget: int = 4_096,
    random_search_budget: int = 2_048,
    workers: str | int = "auto",
    resume: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    parameters = {
        "instances_per_family": instances_per_family,
        "seed_replicates": seed_replicates,
        "master_seed": master_seed,
        "monte_carlo_budget": monte_carlo_budget,
        "random_search_budget": random_search_budget,
    }
    jobs = build_pilot_jobs(**parameters)
    configuration = _configuration(jobs, parameters)
    config_path = run_path / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing["job_ids_sha256"] != configuration["job_ids_sha256"]:
            raise ValueError("resume configuration mismatch: job_ids_sha256")
    else:
        config_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
    records = run_jobs(jobs, run_path, workers=workers, resume=resume, logger=logger)
    paths = write_dataset(run_path, jobs, records)
    validation = validate_dataset(run_path)
    (run_path / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    runs = pd.read_parquet(paths["runs"])
    group_rows = []
    for (algorithm, family), group in runs.groupby(["algorithm", "problem_family"], sort=True):
        group_rows.append(
            {
                "algorithm": algorithm,
                "problem_family": family,
                "runs": int(len(group)),
                "quality_mean": float(group["quality_final"].mean()),
                "quality_std": float(group["quality_final"].std(ddof=0)),
                "target_rate": float(group["target_reached"].mean()),
                "failure_rate": float(group["failure"].mean()),
                "runtime_mean": float(group["runtime"].mean()),
            }
        )
    summary = {
        "status": validation["status"],
        "scope": "Phase 1 infrastructure smoke pilot",
        "executed_at": datetime.now(_timezone()).isoformat(timespec="seconds"),
        "elapsed_seconds": perf_counter() - started_at,
        "problem_count": validation["problem_count"],
        "run_count": validation["run_count"],
        "trace_count": validation["trace_count"],
        "trajectory_feature_count": validation["trajectory_feature_count"],
        "failure_count": validation["failure_count"],
        "algorithms": sorted(runs["algorithm"].unique().tolist()),
        "random_mechanisms": sorted(runs["random_mechanism"].unique().tolist()),
        "by_algorithm_problem_family": group_rows,
        "configuration": configuration,
    }
    (run_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
