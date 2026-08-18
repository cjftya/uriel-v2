from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _invalid_numeric_rows(frame: pd.DataFrame, excluded: set[str]) -> dict[str, int]:
    invalid: dict[str, int] = {}
    for column in frame.select_dtypes(include="number").columns:
        if column in excluded:
            continue
        count = int(frame[column].dropna().map(lambda value: not math.isfinite(float(value))).sum())
        if count:
            invalid[column] = count
    return invalid


def validate_dataset(run_directory: str | Path) -> dict[str, Any]:
    run_path = Path(run_directory)
    problems = pd.read_parquet(run_path / "data/problems/problem_metadata.parquet")
    runs = pd.read_parquet(run_path / "data/runs/runs.parquet")
    traces = pd.read_parquet(run_path / "data/traces/common/trace_common.parquet")
    trajectory = pd.read_parquet(run_path / "data/features/trajectory_features.parquet")
    issues: list[dict[str, Any]] = []

    duplicate_runs = int(runs["run_id"].duplicated().sum())
    if duplicate_runs:
        issues.append({"type": "duplicate_run", "count": duplicate_runs})
    duplicate_problems = int(problems["problem_id"].duplicated().sum())
    if duplicate_problems:
        issues.append({"type": "duplicate_problem", "count": duplicate_problems})

    successful_ids = set(runs.loc[runs["success"], "run_id"])
    trace_ids = set(traces["run_id"])
    missing_trace = sorted(successful_ids - trace_ids)
    if missing_trace:
        issues.append({"type": "missing_trace", "count": len(missing_trace), "examples": missing_trace[:5]})
    unknown_trace = sorted(trace_ids - set(runs["run_id"]))
    if unknown_trace:
        issues.append({"type": "orphan_trace", "count": len(unknown_trace), "examples": unknown_trace[:5]})

    completed_ids = set(traces.loc[(traces["budget_fraction"] - 1.0).abs() <= 1e-12, "run_id"])
    incomplete = sorted(successful_ids - completed_ids)
    if incomplete:
        issues.append({"type": "missing_final_checkpoint", "count": len(incomplete), "examples": incomplete[:5]})

    invalid_runs = _invalid_numeric_rows(runs, {"failure_time"})
    invalid_traces = _invalid_numeric_rows(traces, set())
    invalid_problems = _invalid_numeric_rows(problems, set())
    if invalid_runs:
        issues.append({"type": "invalid_run_numeric", "columns": invalid_runs})
    if invalid_traces:
        issues.append({"type": "invalid_trace_numeric", "columns": invalid_traces})
    if invalid_problems:
        issues.append({"type": "invalid_problem_numeric", "columns": invalid_problems})

    for frame_name, frame, columns in (
        ("problems", problems, ("extension_json",)),
        ("runs", runs, ("algorithm_config_json", "extension_json")),
        ("traces", traces, ("extension_json",)),
    ):
        for column in columns:
            try:
                frame[column].map(json.loads)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                issues.append({"type": "invalid_extension_json", "frame": frame_name, "column": column, "error": str(exc)})

    expected_feature_rows = len(successful_ids) * 3
    if len(trajectory) != expected_feature_rows:
        issues.append(
            {"type": "trajectory_feature_count", "expected": expected_feature_rows, "actual": len(trajectory)}
        )
    return {
        "status": "PASS" if not issues else "FAIL",
        "problem_count": int(len(problems)),
        "run_count": int(len(runs)),
        "success_count": int(runs["success"].sum()),
        "failure_count": int(runs["failure"].sum()),
        "trace_count": int(len(traces)),
        "trajectory_feature_count": int(len(trajectory)),
        "issues": issues,
    }
