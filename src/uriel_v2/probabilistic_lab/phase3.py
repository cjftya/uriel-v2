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
from uriel_v2.probabilistic_lab.schema import ProblemSpec, canonical_json
from uriel_v2.probabilistic_lab.synthetic import (
    BENCHMARK_VERSION,
    READY_DOMAINS,
    SYNTHETIC_FAMILIES,
    benchmark_index_records,
    build_synthetic_benchmark,
)
from uriel_v2.provenance import current_git_commit


REQUIRED_STRUCTURE_AXIS_LEVELS = {
    "dimension": 6,
    "noise": 8,
    "entropy": 8,
    "skewness": 4,
    "condition_number": 5,
    "spectral_decay": 4,
    "multimodality": 3,
    "ruggedness": 5,
    "sparsity": 4,
    "effective_dimension": 6,
}


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _normalized_scalar(value: Any, column: str) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if column in {"dimension", "size"} and float(value).is_integer():
            return int(value)
        return float(value)
    return value


def _frame_sha256(frame: pd.DataFrame, sort_columns: list[str]) -> str:
    ordered = frame.sort_values(sort_columns).reset_index(drop=True)
    records = [
        {column: _normalized_scalar(value, column) for column, value in row.items()}
        for row in ordered.to_dict("records")
    ]
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def _invalid_numeric(frame: pd.DataFrame) -> dict[str, int]:
    invalid: dict[str, int] = {}
    for column in frame.select_dtypes(include="number").columns:
        count = int(frame[column].dropna().map(lambda value: not math.isfinite(float(value))).sum())
        if count:
            invalid[column] = count
    return invalid


def _axis_coverage(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    coverage: dict[str, dict[str, float | int | None]] = {}
    for column, minimum_levels in REQUIRED_STRUCTURE_AXIS_LEVELS.items():
        values = frame[column].dropna().astype(float)
        coverage[column] = {
            "required_unique_levels": minimum_levels,
            "non_null_count": int(len(values)),
            "unique_levels": int(values.nunique()),
            "minimum": None if values.empty else float(values.min()),
            "maximum": None if values.empty else float(values.max()),
        }
    return coverage


def validate_phase3_frames(
    problems: pd.DataFrame,
    benchmark_index: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    expected_count = int(manifest["problem_count"])
    if len(problems) != expected_count:
        issues.append({"type": "problem_count", "expected": expected_count, "actual": int(len(problems))})
    duplicate_ids = int(problems["problem_id"].duplicated().sum())
    duplicate_seeds = int(problems["problem_seed"].duplicated().sum())
    if duplicate_ids:
        issues.append({"type": "duplicate_problem_id", "count": duplicate_ids})
    if duplicate_seeds:
        issues.append({"type": "duplicate_problem_seed", "count": duplicate_seeds})

    numeric_issues = _invalid_numeric(problems)
    if numeric_issues:
        issues.append({"type": "invalid_problem_numeric", "columns": numeric_issues})

    extension_errors = 0
    extension_versions: set[str] = set()
    extension_tiers: set[str] = set()
    for payload in problems["extension_json"]:
        try:
            extension = json.loads(payload)
            extension_versions.add(str(extension.get("benchmark_version")))
            extension_tiers.add(str(extension.get("execution_tier")))
        except (TypeError, ValueError, json.JSONDecodeError):
            extension_errors += 1
    if extension_errors:
        issues.append({"type": "invalid_extension_json", "count": extension_errors})
    if extension_versions != {BENCHMARK_VERSION}:
        issues.append({"type": "benchmark_version", "actual": sorted(extension_versions)})
    if extension_tiers != {"ready", "staged"}:
        issues.append({"type": "execution_tiers", "actual": sorted(extension_tiers)})

    actual_family_counts = {str(key): int(value) for key, value in problems["problem_family"].value_counts().items()}
    if actual_family_counts != manifest["family_counts"]:
        issues.append({"type": "family_counts", "actual": actual_family_counts})
    actual_domain_counts = {str(key): int(value) for key, value in problems["domain"].value_counts().items()}
    if actual_domain_counts != manifest["domain_counts"]:
        issues.append({"type": "domain_counts", "actual": actual_domain_counts})

    unbalanced: dict[str, dict[str, int]] = {}
    for family, group in problems.groupby("problem_family"):
        dimension_counts = group["dimension"].value_counts()
        if int(dimension_counts.max() - dimension_counts.min()) > 1:
            unbalanced[str(family)] = {str(key): int(value) for key, value in dimension_counts.items()}
    if unbalanced:
        issues.append({"type": "dimension_axis_imbalance", "families": unbalanced})

    axis_coverage = _axis_coverage(problems)
    insufficient_axes = {
        axis: details
        for axis, details in axis_coverage.items()
        if int(details["unique_levels"]) < min(int(details["required_unique_levels"]), expected_count)
    }
    if insufficient_axes:
        issues.append({"type": "insufficient_structure_axis_coverage", "axes": insufficient_axes})

    problem_ids = set(problems["problem_id"])
    index_ids = set(benchmark_index["problem_id"])
    if problem_ids != index_ids or benchmark_index["problem_id"].duplicated().any():
        issues.append(
            {
                "type": "benchmark_index_coverage",
                "missing": len(problem_ids - index_ids),
                "unknown": len(index_ids - problem_ids),
                "duplicates": int(benchmark_index["problem_id"].duplicated().sum()),
            }
        )
    folds = int(manifest["folds"])
    invalid_folds = benchmark_index.loc[
        ~benchmark_index["instance_fold"].between(0, folds - 1)
        | ~benchmark_index["family_holdout_fold"].between(0, folds - 1)
    ]
    if not invalid_folds.empty:
        issues.append({"type": "invalid_split_fold", "count": int(len(invalid_folds))})
    if int(manifest["instances_per_family"]) >= folds:
        incomplete_folds = [
            str(family)
            for family, group in benchmark_index.groupby("problem_family")
            if set(group["instance_fold"]) != set(range(folds))
        ]
        if incomplete_folds:
            issues.append({"type": "incomplete_instance_folds", "families": incomplete_folds})

    problem_hash = _frame_sha256(problems, ["problem_id"])
    index_hash = _frame_sha256(benchmark_index, ["problem_id"])
    if problem_hash != manifest["problem_metadata_sha256"]:
        issues.append({"type": "problem_hash_mismatch", "actual": problem_hash})
    if index_hash != manifest["benchmark_index_sha256"]:
        issues.append({"type": "index_hash_mismatch", "actual": index_hash})

    return {
        "status": "PASS" if not issues else "FAIL",
        "problem_count": int(len(problems)),
        "family_count": int(problems["problem_family"].nunique()),
        "domain_count": int(problems["domain"].nunique()),
        "ready_count": int((benchmark_index["execution_tier"] == "ready").sum()),
        "staged_count": int((benchmark_index["execution_tier"] == "staged").sum()),
        "duplicate_problem_count": duplicate_ids,
        "duplicate_seed_count": duplicate_seeds,
        "axis_coverage": axis_coverage,
        "issues": issues,
    }


def validate_phase3_dataset(run_directory: str | Path) -> dict[str, Any]:
    run_path = Path(run_directory)
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    problems = pd.read_parquet(run_path / "data/problems/problem_metadata.parquet")
    benchmark_index = pd.read_parquet(run_path / "data/benchmark/benchmark_index.parquet")
    return validate_phase3_frames(problems, benchmark_index, manifest)


def load_phase3_problems(
    run_directory: str | Path,
    *,
    execution_tier: str | None = None,
) -> list[ProblemSpec]:
    run_path = Path(run_directory)
    validation = validate_phase3_dataset(run_path)
    if validation["status"] != "PASS":
        raise ValueError(f"invalid Phase 3 benchmark: {validation['issues'][:3]}")
    problems = pd.read_parquet(run_path / "data/problems/problem_metadata.parquet")
    if execution_tier is not None:
        tiers = pd.read_parquet(run_path / "data/benchmark/benchmark_index.parquet")[
            ["problem_id", "execution_tier"]
        ]
        problems = problems.merge(tiers, on="problem_id", how="left", validate="one_to_one")
        problems = problems[problems["execution_tier"] == execution_tier].drop(columns=["execution_tier"])
    return [ProblemSpec.from_record(record) for record in problems.sort_values("problem_id").to_dict("records")]


def run_phase3(
    run_directory: str | Path,
    *,
    instances_per_family: int = 128,
    master_seed: int = 20_260_821,
    folds: int = 5,
    minimum_problem_count: int = 1_000,
    logger: Any | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    problems = build_synthetic_benchmark(instances_per_family, master_seed)
    problem_frame = pd.DataFrame([problem.to_record() for problem in problems]).sort_values("problem_id")
    index_frame = pd.DataFrame(benchmark_index_records(problems, folds)).sort_values("problem_id")

    family_counts = {str(key): int(value) for key, value in problem_frame["problem_family"].value_counts().items()}
    domain_counts = {str(key): int(value) for key, value in problem_frame["domain"].value_counts().items()}
    manifest = {
        "phase": 3,
        "benchmark_version": BENCHMARK_VERSION,
        "master_seed": int(master_seed),
        "instances_per_family": int(instances_per_family),
        "problem_count": int(len(problem_frame)),
        "family_count": int(problem_frame["problem_family"].nunique()),
        "domain_count": int(problem_frame["domain"].nunique()),
        "family_counts": family_counts,
        "domain_counts": domain_counts,
        "families": {domain: list(families) for domain, families in SYNTHETIC_FAMILIES.items()},
        "ready_domains": sorted(READY_DOMAINS),
        "folds": int(folds),
        "split_unit": "problem_id; all future algorithm/seed runs inherit the same problem split",
        "required_structure_axis_levels": REQUIRED_STRUCTURE_AXIS_LEVELS,
        "problem_metadata_sha256": _frame_sha256(problem_frame, ["problem_id"]),
        "benchmark_index_sha256": _frame_sha256(index_frame, ["problem_id"]),
    }

    destinations = {
        "problems": run_path / "data/problems/problem_metadata.parquet",
        "problem_features": run_path / "data/features/problem_features.parquet",
        "benchmark_index": run_path / "data/benchmark/benchmark_index.parquet",
    }
    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    problem_frame.to_parquet(destinations["problems"], index=False)
    problem_frame.drop(columns=["problem_seed"]).to_parquet(destinations["problem_features"], index=False)
    index_frame.to_parquet(destinations["benchmark_index"], index=False)
    _json_dump(run_path / "manifest.json", manifest)

    validation = validate_phase3_dataset(run_path)
    scale_target_met = len(problem_frame) >= minimum_problem_count
    phase3_checks = {
        "benchmark_validation_pass": validation["status"] == "PASS",
        "minimum_problem_count_met": scale_target_met,
        "four_domains_present": validation["domain_count"] == len(SYNTHETIC_FAMILIES),
        "all_families_present": validation["family_count"] == sum(len(value) for value in SYNTHETIC_FAMILIES.values()),
        "problem_ids_and_seeds_unique": validation["duplicate_problem_count"] == 0
        and validation["duplicate_seed_count"] == 0,
        "ready_and_staged_tiers_present": validation["ready_count"] > 0 and validation["staged_count"] > 0,
        "required_structure_axes_varied": all(
            int(details["unique_levels"]) >= int(details["required_unique_levels"])
            for details in validation["axis_coverage"].values()
        ),
    }
    status = "PHASE_3_PASS" if all(phase3_checks.values()) else "PHASE_3_FAIL"
    config = {
        "phase": 3,
        "parameters": {
            "instances_per_family": instances_per_family,
            "master_seed": master_seed,
            "folds": folds,
            "minimum_problem_count": minimum_problem_count,
        },
        "git_commit": current_git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    summary = {
        "status": status,
        "scope": "Phase 3 balanced synthetic problem benchmark",
        "executed_at": datetime.now(_timezone()).isoformat(timespec="seconds"),
        "elapsed_seconds": perf_counter() - started_at,
        "problem_count": int(len(problem_frame)),
        "family_count": validation["family_count"],
        "domain_count": validation["domain_count"],
        "ready_count": validation["ready_count"],
        "staged_count": validation["staged_count"],
        "domain_counts": domain_counts,
        "family_counts": family_counts,
        "phase3_checks": phase3_checks,
        "configuration": config,
    }
    _json_dump(run_path / "config.json", config)
    _json_dump(run_path / "validation.json", validation)
    _json_dump(run_path / "summary.json", summary)
    if logger is not None:
        logger.info(
            "[PHASE3] status=%s problems=%d families=%d domains=%d ready=%d staged=%d",
            status,
            len(problem_frame),
            validation["family_count"],
            validation["domain_count"],
            validation["ready_count"],
            validation["staged_count"],
        )
    return summary
