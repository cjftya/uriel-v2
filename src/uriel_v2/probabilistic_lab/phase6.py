from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.features import EARLY_CUTOFFS
from uriel_v2.probabilistic_lab.schema import canonical_json
from uriel_v2.probabilistic_lab.validation import validate_dataset
from uriel_v2.provenance import current_git_commit


PHASE6_SCHEMA_VERSION = "phase6-v1"
SPLIT_COLUMNS = {
    "instance_holdout": "instance_fold",
    "family_holdout": "family_holdout_fold",
}
IDENTIFIER_COLUMNS = ("feature_id", "run_id", "problem_id")
CATEGORICAL_COLUMNS = (
    "problem_family",
    "domain",
    "algorithm",
    "algorithm_family",
    "random_mechanism",
    "budget_type",
)
PROBLEM_NUMERIC_COLUMNS = (
    "dimension",
    "size",
    "density",
    "sparsity",
    "noise",
    "entropy",
    "skewness",
    "kurtosis",
    "autocorrelation",
    "condition_number",
    "spectral_decay",
    "multimodality",
    "ruggedness",
    "effective_dimension",
)
TRAJECTORY_NUMERIC_COLUMNS = (
    "observed_fraction",
    "observed_steps",
    "objective_last",
    "best_so_far",
    "improvement_sum",
    "improvement_slope",
    "variance_mean",
    "entropy_mean",
    "diversity_mean",
    "autocorrelation_lag1",
    "transition_magnitude_mean",
    "stagnation_fraction",
    "failure_signal_max",
)
NUMERIC_FEATURE_COLUMNS = (*PROBLEM_NUMERIC_COLUMNS, "budget", "cutoff", *TRAJECTORY_NUMERIC_COLUMNS)
TARGET_COLUMNS = (
    "target_quality_final",
    "target_quality_best",
    "target_runtime",
    "target_failure",
    "target_timeout",
    "target_success",
    "target_first_passage_time",
)
FORBIDDEN_FEATURE_TOKENS = (
    "seed",
    "rng",
    "quality_final",
    "quality_best",
    "runtime",
    "failure",
    "timeout",
    "success",
    "first_passage",
    "target_",
    "t50",
    "t75",
    "t90",
    "t95",
    "t99",
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _append_progress(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": datetime.now(_timezone()).isoformat(timespec="seconds"),
        **event,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _relative_hashes(run_path: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.relative_to(run_path)): _file_sha256(path) for path in paths}


def _input_paths(phase4_path: Path, phase5_path: Path) -> dict[str, Path]:
    return {
        "phase4_config": phase4_path / "config.json",
        "phase4_summary": phase4_path / "summary.json",
        "phase4_problems": phase4_path / "data/problems/problem_metadata.parquet",
        "phase4_runs": phase4_path / "data/runs/runs.parquet",
        "phase4_traces": phase4_path / "data/traces/common/trace_common.parquet",
        "phase4_problem_features": phase4_path / "data/features/problem_features.parquet",
        "phase4_trajectory_features": phase4_path / "data/features/trajectory_features.parquet",
        "phase4_benchmark_index": phase4_path / "data/benchmark/benchmark_index.parquet",
        "phase4_run_splits": phase4_path / "data/splits/run_splits.parquet",
        "phase5_summary": phase5_path / "summary.json",
    }


def _input_fingerprints(paths: dict[str, Path]) -> dict[str, str]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 6 input files are missing: {missing[:5]}")
    return {name: _file_sha256(path) for name, path in sorted(paths.items())}


def _configuration_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stage_is_complete(run_path: Path, state: dict[str, Any], stage: str) -> bool:
    recorded = state.get("completed_stages", {}).get(stage)
    if not recorded:
        return False
    for relative, expected_hash in recorded["outputs"].items():
        path = run_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 6 resume output hash mismatch: stage={stage} path={relative}")
    return True


def _complete_stage(
    run_path: Path,
    state: dict[str, Any],
    stage: str,
    outputs: Iterable[Path],
    progress_path: Path,
) -> None:
    state.setdefault("completed_stages", {})[stage] = {
        "completed_at": datetime.now(_timezone()).isoformat(timespec="seconds"),
        "outputs": _relative_hashes(run_path, outputs),
    }
    state["last_completed_stage"] = stage
    _atomic_json(run_path / "run_state.json", state)
    _append_progress(progress_path, {"event": "stage_completed", "stage": stage})


def _run_stage(
    run_path: Path,
    state: dict[str, Any],
    stage: str,
    outputs: tuple[Path, ...],
    action: Callable[[], None],
    progress_path: Path,
    logger: logging.Logger,
) -> None:
    if _stage_is_complete(run_path, state, stage):
        logger.info("[PHASE6][RESUME] stage=%s status=SKIP_VERIFIED", stage)
        _append_progress(progress_path, {"event": "stage_resumed", "stage": stage})
        return
    logger.info("[PHASE6][START] stage=%s", stage)
    _append_progress(progress_path, {"event": "stage_started", "stage": stage})
    action()
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"Phase 6 stage did not create required outputs: {missing}")
    _complete_stage(run_path, state, stage, outputs, progress_path)
    logger.info("[PHASE6][DONE] stage=%s", stage)


def _feature_id(run_id: str, cutoff: float) -> str:
    return f"{run_id}:p{int(round(cutoff * 100.0)):02d}"


def _load_phase6_frames(phase4_path: Path) -> dict[str, pd.DataFrame]:
    return {
        "problems": pd.read_parquet(phase4_path / "data/problems/problem_metadata.parquet"),
        "runs": pd.read_parquet(phase4_path / "data/runs/runs.parquet"),
        "problem_features": pd.read_parquet(phase4_path / "data/features/problem_features.parquet"),
        "trajectory_features": pd.read_parquet(
            phase4_path / "data/features/trajectory_features.parquet"
        ),
        "benchmark_index": pd.read_parquet(
            phase4_path / "data/benchmark/benchmark_index.parquet"
        ),
        "run_splits": pd.read_parquet(phase4_path / "data/splits/run_splits.parquet"),
    }


def _build_raw_tables(
    run_path: Path,
    phase4_path: Path,
    logger: logging.Logger,
) -> None:
    frames = _load_phase6_frames(phase4_path)
    runs = frames["runs"].copy()
    problem_features = frames["problem_features"].drop(
        columns=["problem_seed", "extension_json", "schema_version"], errors="ignore"
    )
    trajectory = frames["trajectory_features"].copy()
    run_columns = [
        "run_id",
        "problem_id",
        "problem_family",
        "domain",
        "algorithm",
        "algorithm_family",
        "random_mechanism",
        "budget_type",
        "budget",
    ]
    missing_run_columns = sorted(set(run_columns) - set(runs.columns))
    if missing_run_columns:
        raise ValueError(f"Phase 4 runs missing Phase 6 columns: {missing_run_columns}")
    missing_trajectory = sorted(set(("run_id", "cutoff", *TRAJECTORY_NUMERIC_COLUMNS)) - set(trajectory.columns))
    if missing_trajectory:
        raise ValueError(f"Phase 4 trajectory features missing columns: {missing_trajectory}")

    metadata = runs[run_columns]
    features = trajectory[["run_id", "cutoff", *TRAJECTORY_NUMERIC_COLUMNS]].merge(
        metadata,
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    structural_columns = ["problem_id", *PROBLEM_NUMERIC_COLUMNS]
    features = features.merge(
        problem_features[structural_columns],
        on="problem_id",
        how="left",
        validate="many_to_one",
    )
    features.insert(
        0,
        "feature_id",
        [_feature_id(str(run_id), float(cutoff)) for run_id, cutoff in zip(features["run_id"], features["cutoff"], strict=True)],
    )
    feature_columns = [
        *IDENTIFIER_COLUMNS,
        *CATEGORICAL_COLUMNS,
        *NUMERIC_FEATURE_COLUMNS,
    ]
    features = features[feature_columns].sort_values("feature_id").reset_index(drop=True)

    target_source = runs[
        [
            "run_id",
            "quality_final",
            "quality_best",
            "runtime",
            "failure",
            "timeout",
            "success",
            "first_passage_time",
        ]
    ].rename(
        columns={
            "quality_final": "target_quality_final",
            "quality_best": "target_quality_best",
            "runtime": "target_runtime",
            "failure": "target_failure",
            "timeout": "target_timeout",
            "success": "target_success",
            "first_passage_time": "target_first_passage_time",
        }
    )
    targets = features[["feature_id", "run_id", "problem_id", "cutoff"]].merge(
        target_source,
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    targets = targets.sort_values("feature_id").reset_index(drop=True)

    split_source = frames["run_splits"][
        ["run_id", "problem_id", "instance_fold", "family_holdout_fold"]
    ]
    splits = features[["feature_id", "run_id", "problem_id", "cutoff"]].merge(
        split_source,
        on=["run_id", "problem_id"],
        how="left",
        validate="many_to_one",
    )
    splits = splits.sort_values("feature_id").reset_index(drop=True)

    normalized_problem_features = frames["problem_features"].drop(
        columns=["problem_seed"], errors="ignore"
    ).sort_values("problem_id")
    normalized_trajectory = trajectory[
        ["run_id", "cutoff", *TRAJECTORY_NUMERIC_COLUMNS]
    ].sort_values(["run_id", "cutoff"])

    _atomic_parquet(run_path / "data/features/problem_features.parquet", normalized_problem_features)
    _atomic_parquet(run_path / "data/features/trajectory_features.parquet", normalized_trajectory)
    _atomic_parquet(run_path / "data/features/model_features.parquet", features)
    _atomic_parquet(run_path / "data/targets/model_targets.parquet", targets)
    _atomic_parquet(run_path / "data/splits/model_splits.parquet", splits)
    schema = {
        "schema_version": PHASE6_SCHEMA_VERSION,
        "grain": "one row per run_id and early-trajectory cutoff",
        "cutoffs": list(EARLY_CUTOFFS),
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "categorical_feature_columns": list(CATEGORICAL_COLUMNS),
        "numeric_feature_columns": list(NUMERIC_FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "split_columns": SPLIT_COLUMNS,
        "seed_policy": "seed, problem_seed, RNG identifiers, and algorithm configuration are excluded from model features",
        "target_policy": "targets are stored in a physically separate table and never merged into feature artifacts",
    }
    _atomic_json(run_path / "feature_schema.json", schema)
    logger.info(
        "[PHASE6][RAW] feature_rows=%s target_rows=%s split_rows=%s",
        len(features),
        len(targets),
        len(splits),
    )


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _numeric_statistics(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))]
    if finite.empty:
        return {
            "non_null_count": 0,
            "all_missing": True,
            "impute_median": 0.0,
            "mean": 0.0,
            "scale_std": 1.0,
            "q01": None,
            "q99": None,
            "add_missing_indicator": True,
        }
    median = float(finite.median())
    standard_deviation = float(finite.std(ddof=0))
    return {
        "non_null_count": int(len(finite)),
        "all_missing": False,
        "impute_median": median,
        "mean": float(finite.mean()),
        "scale_std": standard_deviation if standard_deviation > 1e-12 else 1.0,
        "q01": _finite_or_none(finite.quantile(0.01)),
        "q99": _finite_or_none(finite.quantile(0.99)),
        "add_missing_indicator": bool(numeric.isna().any()),
    }


def _build_preprocessing_specs(run_path: Path, logger: logging.Logger) -> None:
    features = pd.read_parquet(run_path / "data/features/model_features.parquet")
    splits = pd.read_parquet(run_path / "data/splits/model_splits.parquet")
    feature_splits = features[["feature_id"]].merge(
        splits[["feature_id", "problem_id", "instance_fold", "family_holdout_fold"]],
        on="feature_id",
        how="left",
        validate="one_to_one",
    )
    specifications: dict[str, Any] = {
        "schema_version": PHASE6_SCHEMA_VERSION,
        "fit_policy": "all statistics and categorical vocabularies are fit on training problem instances only",
        "unknown_category": "__UNKNOWN__",
        "missing_category": "__MISSING__",
        "splits": {},
    }
    for split_name, fold_column in SPLIT_COLUMNS.items():
        fold_values = sorted(int(value) for value in feature_splits[fold_column].dropna().unique())
        specifications["splits"][split_name] = {}
        for fold in fold_values:
            train_mask = feature_splits[fold_column].astype(int) != fold
            validation_mask = ~train_mask
            train_features = features.loc[train_mask.to_numpy()]
            validation_features = features.loc[validation_mask.to_numpy()]
            train_problem_ids = sorted(set(feature_splits.loc[train_mask, "problem_id"].astype(str)))
            validation_problem_ids = sorted(set(feature_splits.loc[validation_mask, "problem_id"].astype(str)))
            overlap = set(train_problem_ids) & set(validation_problem_ids)
            if overlap:
                raise ValueError(
                    f"problem split leakage in {split_name} fold {fold}: {sorted(overlap)[:5]}"
                )
            numeric = {
                column: _numeric_statistics(train_features[column])
                for column in NUMERIC_FEATURE_COLUMNS
            }
            categorical = {}
            for column in CATEGORICAL_COLUMNS:
                vocabulary = sorted(
                    value
                    for value in train_features[column].dropna().astype(str).unique()
                    if value not in {"__UNKNOWN__", "__MISSING__"}
                )
                categorical[column] = {
                    "vocabulary": [*vocabulary, "__MISSING__", "__UNKNOWN__"],
                    "unseen_validation_count": int(
                        (~validation_features[column].fillna("__MISSING__").astype(str).isin(vocabulary + ["__MISSING__"])).sum()
                    ),
                }
            specifications["splits"][split_name][str(fold)] = {
                "fold_column": fold_column,
                "training_row_count": int(train_mask.sum()),
                "validation_row_count": int(validation_mask.sum()),
                "training_problem_count": len(train_problem_ids),
                "validation_problem_count": len(validation_problem_ids),
                "training_problem_ids_sha256": hashlib.sha256(
                    "\n".join(train_problem_ids).encode("utf-8")
                ).hexdigest(),
                "validation_problem_ids_sha256": hashlib.sha256(
                    "\n".join(validation_problem_ids).encode("utf-8")
                ).hexdigest(),
                "numeric": numeric,
                "categorical": categorical,
            }
            logger.info(
                "[PHASE6][PREPROCESS] split=%s fold=%s train_rows=%s validation_rows=%s",
                split_name,
                fold,
                int(train_mask.sum()),
                int(validation_mask.sum()),
            )
    _atomic_json(run_path / "data/preprocessing/preprocessing_specs.json", specifications)


def _feature_quality(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in NUMERIC_FEATURE_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values = numeric.dropna().astype(float)
        nonfinite_count = int((~np.isfinite(values)).sum())
        finite = values[np.isfinite(values)]
        q1 = float(finite.quantile(0.25)) if not finite.empty else math.nan
        q3 = float(finite.quantile(0.75)) if not finite.empty else math.nan
        iqr = q3 - q1
        extreme_count = 0
        if not finite.empty and iqr > 0.0:
            extreme_count = int(((finite < q1 - 10.0 * iqr) | (finite > q3 + 10.0 * iqr)).sum())
        rows.append(
            {
                "feature": column,
                "row_count": int(len(frame)),
                "missing_count": int(numeric.isna().sum()),
                "missing_rate": float(numeric.isna().mean()),
                "nonfinite_count": nonfinite_count,
                "unique_count": int(finite.nunique()),
                "minimum": None if finite.empty else float(finite.min()),
                "median": None if finite.empty else float(finite.median()),
                "maximum": None if finite.empty else float(finite.max()),
                "iqr": None if finite.empty else float(iqr),
                "extreme_iqr10_count": extreme_count,
            }
        )
    return pd.DataFrame(rows)


def _seed_stability(
    phase4_path: Path,
    expected_replicates: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    runs = pd.read_parquet(phase4_path / "data/runs/runs.parquet")
    trajectory = pd.read_parquet(phase4_path / "data/features/trajectory_features.parquet")
    frame = trajectory[["run_id", "cutoff", *TRAJECTORY_NUMERIC_COLUMNS]].merge(
        runs[["run_id", "problem_id", "algorithm"]],
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    group_columns = ["problem_id", "algorithm", "cutoff"]
    group_sizes = frame.groupby(group_columns, sort=True).size().rename("seed_replicate_count")
    records: list[pd.DataFrame] = []
    for cutoff in EARLY_CUTOFFS:
        logger.info("[PHASE6][STABILITY] cutoff=%.2f", cutoff)
        cutoff_frame = frame.loc[np.isclose(frame["cutoff"].astype(float), cutoff)]
        grouped = cutoff_frame.groupby(group_columns, sort=True)
        for feature in TRAJECTORY_NUMERIC_COLUMNS:
            aggregate = grouped[feature].agg(["mean", "std", "min", "max", "count"]).reset_index()
            aggregate["feature"] = feature
            aggregate["normalized_std"] = aggregate["std"].fillna(0.0) / np.maximum(
                aggregate["mean"].abs(), 1e-12
            )
            records.append(aggregate)
    stability = pd.concat(records, ignore_index=True)
    stability = stability.merge(group_sizes.reset_index(), on=group_columns, how="left", validate="many_to_one")
    stability = stability[
        [
            *group_columns,
            "feature",
            "seed_replicate_count",
            "count",
            "mean",
            "std",
            "min",
            "max",
            "normalized_std",
        ]
    ].sort_values(["cutoff", "problem_id", "algorithm", "feature"])
    summary_rows = []
    for feature, group in stability.groupby("feature", sort=True):
        finite = group["normalized_std"].replace([np.inf, -np.inf], np.nan).dropna()
        summary_rows.append(
            {
                "feature": str(feature),
                "group_count": int(len(group)),
                "median_normalized_std": 0.0 if finite.empty else float(finite.median()),
                "p95_normalized_std": 0.0 if finite.empty else float(finite.quantile(0.95)),
                "maximum_normalized_std": 0.0 if finite.empty else float(finite.max()),
            }
        )
    summary = {
        "schema_version": PHASE6_SCHEMA_VERSION,
        "expected_seed_replicates": expected_replicates,
        "group_count": int(len(group_sizes)),
        "incomplete_group_count": int((group_sizes != expected_replicates).sum()),
        "feature_summaries": summary_rows,
        "interpretation": "diagnostic only; stochastic dispersion is not used as a model-performance gate",
    }
    return stability, summary


def _build_diagnostics(
    run_path: Path,
    phase4_path: Path,
    expected_replicates: int,
    logger: logging.Logger,
) -> None:
    features = pd.read_parquet(run_path / "data/features/model_features.parquet")
    quality = _feature_quality(features)
    stability, stability_summary = _seed_stability(phase4_path, expected_replicates, logger)
    _atomic_parquet(run_path / "data/quality/feature_quality.parquet", quality)
    _atomic_parquet(run_path / "data/quality/seed_stability.parquet", stability)
    _atomic_json(run_path / "data/quality/seed_stability_summary.json", stability_summary)


def _validate_phase6(
    run_path: Path,
    phase4_path: Path,
    phase5_path: Path,
    expected_replicates: int,
    original_input_fingerprints: dict[str, str],
) -> dict[str, Any]:
    features = pd.read_parquet(run_path / "data/features/model_features.parquet")
    targets = pd.read_parquet(run_path / "data/targets/model_targets.parquet")
    splits = pd.read_parquet(run_path / "data/splits/model_splits.parquet")
    trajectory = pd.read_parquet(run_path / "data/features/trajectory_features.parquet")
    feature_quality = pd.read_parquet(run_path / "data/quality/feature_quality.parquet")
    stability_summary = json.loads(
        (run_path / "data/quality/seed_stability_summary.json").read_text(encoding="utf-8")
    )
    preprocessing = json.loads(
        (run_path / "data/preprocessing/preprocessing_specs.json").read_text(encoding="utf-8")
    )
    phase4_summary = json.loads((phase4_path / "summary.json").read_text(encoding="utf-8"))
    phase5_summary = json.loads((phase5_path / "summary.json").read_text(encoding="utf-8"))
    current_inputs = _input_fingerprints(_input_paths(phase4_path, phase5_path))
    expected_rows = int(phase4_summary["run_count"]) * len(EARLY_CUTOFFS)
    forbidden_columns = sorted(
        column
        for column in features.columns
        if column not in IDENTIFIER_COLUMNS
        and column not in TRAJECTORY_NUMERIC_COLUMNS
        and any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    nonfinite_features = int(feature_quality["nonfinite_count"].sum())
    required_missing = int(feature_quality.loc[
        feature_quality["feature"].isin(("budget", "cutoff", *TRAJECTORY_NUMERIC_COLUMNS)),
        "missing_count",
    ].sum())
    split_problem_leakage = 0
    for fold_column in SPLIT_COLUMNS.values():
        split_problem_leakage += int(
            (splits.groupby("problem_id")[fold_column].nunique() != 1).sum()
        )
    cutoff_values = tuple(sorted(float(value) for value in features["cutoff"].unique()))
    observed_after_cutoff = int(
        (trajectory["observed_fraction"].astype(float) > trajectory["cutoff"].astype(float) + 1e-12).sum()
    )
    preprocessing_fold_count = sum(len(folds) for folds in preprocessing["splits"].values())
    checks = {
        "phase4_quality_pass": phase4_summary["status"] == "PHASE_4_PASS",
        "phase5_quality_pass": phase5_summary["status"] == "PHASE_5_PASS",
        "source_inputs_unchanged": current_inputs == original_input_fingerprints,
        "expected_feature_rows": len(features) == expected_rows,
        "expected_target_rows": len(targets) == expected_rows,
        "expected_split_rows": len(splits) == expected_rows,
        "feature_ids_unique": bool(features["feature_id"].is_unique),
        "feature_target_coverage_exact": set(features["feature_id"]) == set(targets["feature_id"]),
        "feature_split_coverage_exact": set(features["feature_id"]) == set(splits["feature_id"]),
        "early_cutoffs_exact": np.allclose(cutoff_values, EARLY_CUTOFFS),
        "no_future_trajectory_observation": observed_after_cutoff == 0,
        "forbidden_feature_columns_absent": not forbidden_columns,
        "target_columns_physically_separate": not (set(TARGET_COLUMNS) & set(features.columns)),
        "feature_numeric_values_finite": nonfinite_features == 0,
        "required_feature_values_present": required_missing == 0,
        "problem_split_leakage_absent": split_problem_leakage == 0,
        "fold_preprocessing_complete": preprocessing_fold_count > 0
        and set(preprocessing["splits"]) == set(SPLIT_COLUMNS),
        "seed_replicate_groups_complete": stability_summary["incomplete_group_count"] == 0
        and stability_summary["expected_seed_replicates"] == expected_replicates,
    }
    issues = []
    if forbidden_columns:
        issues.append({"type": "feature_leakage", "columns": forbidden_columns})
    if nonfinite_features:
        issues.append({"type": "nonfinite_feature", "count": nonfinite_features})
    if required_missing:
        issues.append({"type": "missing_required_feature", "count": required_missing})
    if split_problem_leakage:
        issues.append({"type": "problem_split_leakage", "count": split_problem_leakage})
    if observed_after_cutoff:
        issues.append({"type": "future_trajectory_observation", "count": observed_after_cutoff})
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    if failed_checks:
        issues.append({"type": "failed_quality_checks", "checks": failed_checks})
    return {
        "status": "PHASE_6_PASS" if all(checks.values()) else "PHASE_6_FAIL",
        "schema_version": PHASE6_SCHEMA_VERSION,
        "feature_row_count": int(len(features)),
        "target_row_count": int(len(targets)),
        "split_row_count": int(len(splits)),
        "problem_count": int(splits["problem_id"].nunique()),
        "run_count": int(features["run_id"].nunique()),
        "cutoffs": list(cutoff_values),
        "numeric_feature_count": len(NUMERIC_FEATURE_COLUMNS),
        "categorical_feature_count": len(CATEGORICAL_COLUMNS),
        "preprocessing_fold_count": preprocessing_fold_count,
        "checks": checks,
        "issues": issues,
        "diagnostics": {
            "nonfinite_feature_count": nonfinite_features,
            "required_missing_count": required_missing,
            "expected_seed_replicates": expected_replicates,
            "seed_stability_group_count": stability_summary["group_count"],
            "incomplete_seed_stability_group_count": stability_summary["incomplete_group_count"],
        },
    }


def run_phase6(
    run_directory: str | Path,
    phase4_directory: str | Path,
    phase5_directory: str | Path,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    phase4_path = Path(phase4_directory)
    phase5_path = Path(phase5_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"

    generic_validation = validate_dataset(phase4_path)
    phase4_summary = json.loads((phase4_path / "summary.json").read_text(encoding="utf-8"))
    phase4_config = json.loads((phase4_path / "config.json").read_text(encoding="utf-8"))
    phase5_summary = json.loads((phase5_path / "summary.json").read_text(encoding="utf-8"))
    if generic_validation["status"] != "PASS" or phase4_summary["status"] != "PHASE_4_PASS":
        raise ValueError("Phase 6 requires a validated PHASE_4_PASS dataset")
    if phase5_summary["status"] != "PHASE_5_PASS":
        raise ValueError("Phase 6 requires a PHASE_5_PASS quality audit")
    expected_replicates = int(phase4_config["parameters"]["seed_replicates"])
    input_paths = _input_paths(phase4_path, phase5_path)
    fingerprints = _input_fingerprints(input_paths)
    stable_configuration = {
        "phase": 6,
        "schema_version": PHASE6_SCHEMA_VERSION,
        "input_fingerprints": fingerprints,
        "early_cutoffs": list(EARLY_CUTOFFS),
        "split_columns": SPLIT_COLUMNS,
        "expected_seed_replicates": expected_replicates,
    }
    config_sha256 = _configuration_hash(stable_configuration)
    configuration = {
        **stable_configuration,
        "phase4_directory": str(phase4_path.resolve()),
        "phase5_directory": str(phase5_path.resolve()),
        "git_commit": current_git_commit(),
        "configuration_sha256": config_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != config_sha256:
            raise ValueError("Phase 6 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_6_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 6 resume manifest hash mismatch")
            active_logger.info("[PHASE6][RESUME] status=PHASE_6_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info(
            "[PHASE6][RESUME] last_completed_stage=%s",
            state.get("last_completed_stage"),
        )
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE6_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": config_sha256,
            "input_fingerprints": fingerprints,
            "completed_stages": {},
            "last_completed_stage": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase6_started"})

    raw_outputs = (
        run_path / "data/features/problem_features.parquet",
        run_path / "data/features/trajectory_features.parquet",
        run_path / "data/features/model_features.parquet",
        run_path / "data/targets/model_targets.parquet",
        run_path / "data/splits/model_splits.parquet",
        run_path / "feature_schema.json",
    )
    _run_stage(
        run_path,
        state,
        "raw_feature_tables",
        raw_outputs,
        lambda: _build_raw_tables(run_path, phase4_path, active_logger),
        progress_path,
        active_logger,
    )
    preprocessing_outputs = (run_path / "data/preprocessing/preprocessing_specs.json",)
    _run_stage(
        run_path,
        state,
        "fold_preprocessing",
        preprocessing_outputs,
        lambda: _build_preprocessing_specs(run_path, active_logger),
        progress_path,
        active_logger,
    )
    diagnostic_outputs = (
        run_path / "data/quality/feature_quality.parquet",
        run_path / "data/quality/seed_stability.parquet",
        run_path / "data/quality/seed_stability_summary.json",
    )
    _run_stage(
        run_path,
        state,
        "feature_quality_diagnostics",
        diagnostic_outputs,
        lambda: _build_diagnostics(run_path, phase4_path, expected_replicates, active_logger),
        progress_path,
        active_logger,
    )

    validation = _validate_phase6(
        run_path,
        phase4_path,
        phase5_path,
        expected_replicates,
        fingerprints,
    )
    validation["executed_at"] = datetime.now(_timezone()).isoformat(timespec="seconds")
    validation["elapsed_seconds"] = perf_counter() - started_at
    validation["configuration"] = configuration
    _atomic_json(run_path / "validation.json", validation)
    output_files = [
        *raw_outputs,
        *preprocessing_outputs,
        *diagnostic_outputs,
        run_path / "validation.json",
    ]
    manifest = {
        "phase": 6,
        "status": validation["status"],
        "schema_version": PHASE6_SCHEMA_VERSION,
        "input_fingerprints": fingerprints,
        "output_sha256": _relative_hashes(run_path, output_files),
        "row_counts": {
            "features": validation["feature_row_count"],
            "targets": validation["target_row_count"],
            "splits": validation["split_row_count"],
        },
        "phase7_allowed": validation["status"] == "PHASE_6_PASS",
    }
    _atomic_json(run_path / "manifest.json", manifest)
    state["status"] = validation["status"]
    state["last_completed_stage"] = "validation"
    state["manifest_sha256"] = _file_sha256(run_path / "manifest.json")
    state["validation_sha256"] = _file_sha256(run_path / "validation.json")
    _atomic_json(state_path, state)
    _append_progress(
        progress_path,
        {"event": "phase6_finished", "status": validation["status"]},
    )
    active_logger.info(
        "[PHASE6][SUMMARY] status=%s features=%s runs=%s folds=%s directory=%s",
        validation["status"],
        validation["feature_row_count"],
        validation["run_count"],
        validation["preprocessing_fold_count"],
        run_path,
    )
    return validation
