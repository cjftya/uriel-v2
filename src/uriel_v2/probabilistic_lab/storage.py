from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from uriel_v2.probabilistic_lab.features import extract_trajectory_features
from uriel_v2.probabilistic_lab.schema import JobSpec, SCHEMA_VERSION


def read_checkpoint(path: str | Path) -> list[dict[str, Any]]:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return []
    records: list[dict[str, Any]] = []
    with checkpoint.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid checkpoint JSON at line {line_number}") from exc
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"checkpoint schema mismatch at line {line_number}")
            records.append(record)
    return records


def append_checkpoint(path: str | Path, record: dict[str, Any]) -> None:
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with checkpoint.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()


def write_dataset(run_directory: str | Path, jobs: Iterable[JobSpec], checkpoint_records: list[dict[str, Any]]) -> dict[str, Path]:
    run_path = Path(run_directory)
    job_list = list(jobs)
    problems_by_id = {job.problem.problem_id: job.problem for job in job_list}
    result_records = [record["result"] for record in checkpoint_records]
    trace_records = [trace for record in checkpoint_records for trace in record["traces"]]
    if not result_records:
        raise ValueError("cannot write an empty experiment dataset")

    problems = pd.DataFrame([problem.to_record() for problem in problems_by_id.values()]).sort_values("problem_id")
    runs = pd.DataFrame(result_records).sort_values("run_id")
    traces = pd.DataFrame(trace_records).sort_values(["run_id", "step"]) if trace_records else pd.DataFrame(
        columns=["run_id", "step", "budget_fraction"]
    )
    trajectory_features = extract_trajectory_features(traces)

    destinations = {
        "problems": run_path / "data/problems/problem_metadata.parquet",
        "runs": run_path / "data/runs/runs.parquet",
        "traces": run_path / "data/traces/common/trace_common.parquet",
        "problem_features": run_path / "data/features/problem_features.parquet",
        "trajectory_features": run_path / "data/features/trajectory_features.parquet",
    }
    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    problems.to_parquet(destinations["problems"], index=False)
    runs.to_parquet(destinations["runs"], index=False)
    traces.to_parquet(destinations["traces"], index=False)
    problems.to_parquet(destinations["problem_features"], index=False)
    trajectory_features.to_parquet(destinations["trajectory_features"], index=False)
    return destinations
