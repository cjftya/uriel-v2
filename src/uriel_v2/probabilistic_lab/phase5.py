from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.phase4 import build_phase4_jobs
from uriel_v2.probabilistic_lab.runner import execute_job
from uriel_v2.probabilistic_lab.schema import ExperimentBundle, canonical_json
from uriel_v2.probabilistic_lab.storage import read_checkpoint
from uriel_v2.probabilistic_lab.validation import validate_dataset
from uriel_v2.provenance import current_git_commit


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


_INTEGER_COLUMNS = {
    "seed",
    "budget",
    "steps",
    "first_passage_time",
    "t50",
    "t75",
    "t90",
    "t95",
    "t99",
    "step",
}


def _normalize(value: Any, column: str) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if column in _INTEGER_COLUMNS and float(value).is_integer():
            return int(value)
        return float(value)
    return value


def _scientific_hash_from_bundle(bundle: ExperimentBundle) -> str:
    result = bundle.result.to_record()
    result.pop("runtime", None)
    result.pop("failure_time", None)
    traces = []
    for trace in bundle.traces:
        record = trace.to_record()
        record.pop("elapsed_time", None)
        traces.append(record)
    return hashlib.sha256(canonical_json({"result": result, "traces": traces}).encode("utf-8")).hexdigest()


def _scientific_hash_from_frames(runs: pd.DataFrame, traces: pd.DataFrame, run_id: str) -> str:
    result = {
        column: _normalize(value, column)
        for column, value in runs.loc[runs["run_id"] == run_id].iloc[0].items()
    }
    result.pop("runtime", None)
    result.pop("failure_time", None)
    trace_records = []
    selected = traces.loc[traces["run_id"] == run_id].sort_values("step")
    for row in selected.to_dict("records"):
        normalized = {column: _normalize(value, column) for column, value in row.items()}
        normalized.pop("elapsed_time", None)
        trace_records.append(normalized)
    return hashlib.sha256(
        canonical_json({"result": result, "traces": trace_records}).encode("utf-8")
    ).hexdigest()


def run_phase5(
    run_directory: str | Path,
    phase4_directory: str | Path,
    *,
    reproducibility_samples_per_algorithm_family: int = 1,
) -> dict[str, Any]:
    if reproducibility_samples_per_algorithm_family <= 0:
        raise ValueError("reproducibility sample count must be positive")
    started_at = perf_counter()
    run_path = Path(run_directory)
    phase4_path = Path(phase4_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    phase4_summary = json.loads((phase4_path / "summary.json").read_text(encoding="utf-8"))
    phase4_config = json.loads((phase4_path / "config.json").read_text(encoding="utf-8"))
    validation = validate_dataset(phase4_path)
    problems = pd.read_parquet(phase4_path / "data/problems/problem_metadata.parquet")
    runs = pd.read_parquet(phase4_path / "data/runs/runs.parquet")
    traces = pd.read_parquet(phase4_path / "data/traces/common/trace_common.parquet")
    problem_features = pd.read_parquet(phase4_path / "data/features/problem_features.parquet")
    trajectory_features = pd.read_parquet(phase4_path / "data/features/trajectory_features.parquet")
    pairs = pd.read_parquet(phase4_path / "data/comparisons/paired_runs.parquet")
    benchmark_index = pd.read_parquet(phase4_path / "data/benchmark/benchmark_index.parquet")
    run_splits = pd.read_parquet(phase4_path / "data/splits/run_splits.parquet")
    checkpoint_records = read_checkpoint(phase4_path / "checkpoint.jsonl")

    issues: list[dict[str, Any]] = []

    def add_issue(severity: str, issue_type: str, count: int, details: Any = None) -> None:
        if count:
            issues.append(
                {
                    "severity": severity,
                    "type": issue_type,
                    "count": int(count),
                    "details": details,
                }
            )

    add_issue("critical", "generic_validation_issue", len(validation["issues"]), validation["issues"])
    add_issue("critical", "phase4_status_not_pass", int(phase4_summary["status"] != "PHASE_4_PASS"))

    parameters = phase4_config["parameters"]
    planned_jobs = build_phase4_jobs(
        phase4_config["benchmark_directory"],
        seed_replicates=int(parameters["seed_replicates"]),
        master_seed=int(parameters["master_seed"]),
        sampling_budget=int(parameters["sampling_budget"]),
        optimization_budget=int(parameters["optimization_budget"]),
    )
    planned_ids = {job.run_id for job in planned_jobs}
    actual_ids = set(runs["run_id"])
    planned_hash = hashlib.sha256("\n".join(sorted(planned_ids)).encode("utf-8")).hexdigest()
    add_issue("critical", "job_hash_mismatch", int(planned_hash != phase4_config["job_ids_sha256"]))
    add_issue("critical", "missing_planned_run", len(planned_ids - actual_ids), sorted(planned_ids - actual_ids)[:10])
    add_issue("critical", "unexpected_run", len(actual_ids - planned_ids), sorted(actual_ids - planned_ids)[:10])
    checkpoint_ids = {record["result"]["run_id"] for record in checkpoint_records}
    add_issue("critical", "checkpoint_run_mismatch", len(checkpoint_ids ^ actual_ids))

    expected_replicates = int(parameters["seed_replicates"])
    replicate_counts = runs.groupby(["problem_id", "algorithm"]).size()
    bad_replicates = replicate_counts[replicate_counts != expected_replicates]
    bad_replicate_details = [
        {"problem_id": str(key[0]), "algorithm": str(key[1]), "count": int(value)}
        for key, value in bad_replicates.head(10).items()
    ]
    add_issue("critical", "incomplete_seed_replicates", len(bad_replicates), bad_replicate_details)
    add_issue("critical", "duplicate_run_id", int(runs["run_id"].duplicated().sum()))
    add_issue("critical", "duplicate_pair_id", int(pairs["pair_id"].duplicated().sum()))
    add_issue("critical", "incomplete_pair_coverage", int(len(pairs) != len(runs) // 2))

    forbidden_feature_columns = sorted(
        {
            column
            for column in [*problem_features.columns, *trajectory_features.columns]
            if column.lower() in {"seed", "problem_seed", "rng_algorithm", "rng_version"}
        }
    )
    add_issue("critical", "seed_leakage_feature", len(forbidden_feature_columns), forbidden_feature_columns)

    split_fold_counts = run_splits.groupby("problem_id")["instance_fold"].nunique()
    split_leakage = split_fold_counts[split_fold_counts != 1]
    add_issue("critical", "problem_instance_split_leakage", len(split_leakage), split_leakage.head(10).to_dict())
    index_ids = set(benchmark_index["problem_id"])
    add_issue("critical", "split_index_problem_mismatch", len(index_ids ^ set(problems["problem_id"])))
    add_issue(
        "critical",
        "missing_run_split",
        int(run_splits[["instance_fold", "family_holdout_fold"]].isna().any(axis=1).sum()),
    )

    inconsistent_status = runs[
        (runs["success"] == runs["failure"])
        | (runs["timeout"] & ~runs["failure"])
        | (runs["failure"] & runs["failure_type"].isna())
        | (runs["success"] & runs["quality_final"].isna())
    ]
    add_issue("critical", "inconsistent_run_status", len(inconsistent_status))
    add_issue("critical", "nonpositive_runtime", int((runs["runtime"] <= 0.0).sum()))

    ordered_traces = traces.sort_values(["run_id", "step"])
    nonmonotone = ordered_traces.groupby("run_id")["step"].diff().fillna(1) <= 0
    add_issue("critical", "nonmonotone_trace_step", int(nonmonotone.sum()))
    final_counts = traces.loc[(traces["budget_fraction"] - 1.0).abs() <= 1e-12].groupby("run_id").size()
    successful_ids = set(runs.loc[runs["success"], "run_id"])
    bad_final = [run_id for run_id in successful_ids if int(final_counts.get(run_id, 0)) != 1]
    add_issue("critical", "invalid_final_checkpoint_count", len(bad_final), bad_final[:10])

    runtime_outliers = 0
    runtime_details: dict[str, Any] = {}
    for algorithm, group in runs.groupby("algorithm"):
        values = group["runtime"].astype(float)
        median = float(values.median())
        mad = float(np.median(np.abs(values - median)))
        threshold = max(median * 5.0, median + 8.0 * max(mad, 1e-12))
        count = int((values > threshold).sum())
        runtime_outliers += count
        runtime_details[str(algorithm)] = {
            "median": median,
            "mad": mad,
            "outlier_threshold": threshold,
            "outlier_count": count,
        }
    add_issue("warning", "runtime_outlier", runtime_outliers, runtime_details)

    selected_jobs: list[Any] = []
    selected_counts: dict[tuple[str, str], int] = {}
    for job in planned_jobs:
        key = (job.algorithm.algorithm, job.problem.problem_family)
        count = selected_counts.get(key, 0)
        if count >= reproducibility_samples_per_algorithm_family:
            continue
        selected_jobs.append(job)
        selected_counts[key] = count + 1
    reproduction_rows: list[dict[str, Any]] = []
    for job in selected_jobs:
        rerun = execute_job(job)
        expected_hash = _scientific_hash_from_frames(runs, traces, job.run_id)
        actual_hash = _scientific_hash_from_bundle(rerun)
        reproduction_rows.append(
            {
                "run_id": job.run_id,
                "problem_id": job.problem.problem_id,
                "problem_family": job.problem.problem_family,
                "algorithm": job.algorithm.algorithm,
                "match": expected_hash == actual_hash,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            }
        )
    reproduction = pd.DataFrame(reproduction_rows)
    reproduction_mismatches = int((~reproduction["match"]).sum()) if not reproduction.empty else 0
    add_issue("critical", "scientific_reproduction_mismatch", reproduction_mismatches)

    issue_frame = pd.DataFrame(
        [
            {
                "severity": issue["severity"],
                "type": issue["type"],
                "count": issue["count"],
                "details_json": canonical_json(issue["details"]),
            }
            for issue in issues
        ],
        columns=["severity", "type", "count", "details_json"],
    )
    quality_directory = run_path / "data/quality"
    quality_directory.mkdir(parents=True, exist_ok=True)
    issue_frame.to_parquet(quality_directory / "quality_issues.parquet", index=False)
    reproduction.to_parquet(quality_directory / "reproducibility_checks.parquet", index=False)

    critical_count = int(sum(issue["count"] for issue in issues if issue["severity"] == "critical"))
    warning_count = int(sum(issue["count"] for issue in issues if issue["severity"] == "warning"))
    checks = {
        "generic_dataset_validation_pass": validation["status"] == "PASS",
        "planned_job_identity_complete": planned_ids == actual_ids,
        "checkpoint_identity_complete": checkpoint_ids == actual_ids,
        "paired_seed_replicates_complete": len(bad_replicates) == 0 and len(pairs) == len(runs) // 2,
        "seed_leakage_absent": not forbidden_feature_columns,
        "problem_split_leakage_absent": len(split_leakage) == 0,
        "run_status_consistent": inconsistent_status.empty,
        "trace_integrity_pass": int(nonmonotone.sum()) == 0 and not bad_final,
        "scientific_reproduction_pass": reproduction_mismatches == 0 and not reproduction.empty,
    }
    status = "PHASE_5_PASS" if critical_count == 0 and all(checks.values()) else "PHASE_5_FAIL"
    configuration = {
        "phase": 5,
        "phase4_directory": str(phase4_path.resolve()),
        "phase4_git_commit": phase4_config["git_commit"],
        "phase4_job_ids_sha256": phase4_config["job_ids_sha256"],
        "reproducibility_samples_per_algorithm_family": reproducibility_samples_per_algorithm_family,
        "git_commit": current_git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    audit = {
        "status": status,
        "scope": "Phase 5 data quality, leakage, trace, runtime, and scientific reproducibility audit",
        "executed_at": datetime.now(_timezone()).isoformat(timespec="seconds"),
        "elapsed_seconds": perf_counter() - started_at,
        "problem_count": int(len(problems)),
        "run_count": int(len(runs)),
        "pair_count": int(len(pairs)),
        "trace_count": int(len(traces)),
        "critical_issue_count": critical_count,
        "warning_count": warning_count,
        "reproducibility_check_count": int(len(reproduction)),
        "reproducibility_mismatch_count": reproduction_mismatches,
        "checks": checks,
        "issues": issues,
        "runtime_diagnostics": runtime_details,
        "performance_snapshot": phase4_summary["comparison_results"],
        "configuration": configuration,
    }
    _json_dump(run_path / "config.json", configuration)
    _json_dump(run_path / "quality_audit.json", audit)
    _json_dump(run_path / "summary.json", audit)
    return audit
