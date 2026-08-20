from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import uriel_v2.probabilistic_lab.phase7 as phase7_module
from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.phase7 import (
    _append_progress,
    _atomic_json,
    _atomic_parquet,
    _atomic_pickle,
    _configuration_hash,
    _file_sha256,
    _load_phase6_tables,
    _relative_hashes,
    _transform_features,
    _validate_phase6_input,
    _verify_fold_contract,
)
from uriel_v2.provenance import current_git_commit


PHASE8_SCHEMA_VERSION = "phase8-v1"
MODEL_NAME = "quantile_gradient_boosting"
QUALITY_TARGET_COLUMN = "target_quality_final"
QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
BRIER_THRESHOLDS = (0.25, 0.50, 0.75, 0.90)


def _quantile_column(level: float) -> str:
    return f"q{int(round(level * 100.0)):02d}"


def _threshold_suffix(threshold: float) -> str:
    return f"{int(round(threshold * 100.0)):02d}"


def _phase7_required_paths(phase7_path: Path) -> dict[str, Path]:
    return {
        "phase7_config": phase7_path / "config.json",
        "phase7_manifest": phase7_path / "manifest.json",
        "phase7_validation": phase7_path / "validation.json",
        "phase7_predictions": phase7_path / "data/predictions/oof_predictions.parquet",
        "phase7_fold_metrics": phase7_path / "data/metrics/fold_metrics.parquet",
        "phase7_aggregate_metrics": phase7_path / "data/metrics/aggregate_metrics.parquet",
        "phase7_fold_schemas": phase7_path / "data/preprocessing/fold_feature_schemas.json",
        "phase7_model_registry": phase7_path / "model_registry.json",
    }


def _validate_phase7_input(
    phase7_path: Path,
    phase6_fingerprints: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    required = _phase7_required_paths(phase7_path)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 8 Phase 7 input files are missing: {missing[:5]}")
    validation = json.loads(required["phase7_validation"].read_text(encoding="utf-8"))
    manifest = json.loads(required["phase7_manifest"].read_text(encoding="utf-8"))
    if validation.get("status") != "PHASE_7_PASS":
        raise ValueError("Phase 8 requires a PHASE_7_PASS baseline run")
    if manifest.get("status") != "PHASE_7_PASS" or not manifest.get("phase8_allowed"):
        raise ValueError("Phase 7 manifest does not allow Phase 8")
    if validation.get("configuration", {}).get("input_fingerprints") != phase6_fingerprints:
        raise ValueError("Phase 7 was not built from the supplied Phase 6 dataset")
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = phase7_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 7 manifest output hash mismatch: {relative}")
    fingerprints = {name: _file_sha256(path) for name, path in sorted(required.items())}
    return validation, manifest, fingerprints


def _combined_input_fingerprints(
    phase6_fingerprints: dict[str, str],
    phase7_fingerprints: dict[str, str],
) -> dict[str, str]:
    return {
        **{f"phase6/{name}": value for name, value in phase6_fingerprints.items()},
        **{f"phase7/{name}": value for name, value in phase7_fingerprints.items()},
    }


def _job_id(split_name: str, fold: int) -> str:
    return f"{split_name}__fold{fold}__quality_distribution"


def _job_paths(run_path: Path, job_id: str) -> dict[str, Path]:
    return {
        "predictions": run_path / "checkpoints/predictions" / f"{job_id}.parquet",
        "metrics": run_path / "checkpoints/metrics" / f"{job_id}.json",
        "model": run_path / "models" / f"{job_id}.pkl.gz",
        "marker": run_path / "checkpoints/jobs" / f"{job_id}.json",
    }


def _verify_job(run_path: Path, job_id: str) -> dict[str, Any] | None:
    paths = _job_paths(run_path, job_id)
    if not paths["marker"].is_file():
        return None
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    for relative, expected_hash in marker.get("output_sha256", {}).items():
        path = run_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 8 job output hash mismatch: job={job_id} path={relative}")
    return marker


def _job_seed(master_seed: int, job_id: str, level: float) -> int:
    token = f"{job_id}:{level:.6f}"
    offset = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
    return int((master_seed + offset) % (2**31 - 1))


def _support_values(quantile_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray((0.0, *QUANTILE_LEVELS, 1.0), dtype=float)
    values = np.column_stack(
        (
            np.zeros(len(quantile_values), dtype=float),
            quantile_values,
            np.ones(len(quantile_values), dtype=float),
        )
    )
    return values, probabilities


def _cdf_at(
    quantile_values: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    support, probabilities = _support_values(quantile_values)
    clipped = np.clip(np.asarray(points, dtype=float), 0.0, 1.0)
    indices = (support <= clipped[:, None]).sum(axis=1) - 1
    indices = np.clip(indices, 0, support.shape[1] - 2)
    rows = np.arange(len(support))
    left_value = support[rows, indices]
    right_value = support[rows, indices + 1]
    left_probability = probabilities[indices]
    right_probability = probabilities[indices + 1]
    width = right_value - left_value
    fraction = np.divide(
        clipped - left_value,
        width,
        out=np.zeros_like(clipped),
        where=width > 1e-12,
    )
    cdf = left_probability + np.clip(fraction, 0.0, 1.0) * (
        right_probability - left_probability
    )
    cdf[clipped <= 0.0] = 0.0
    cdf[clipped >= 1.0] = 1.0
    return np.clip(cdf, 0.0, 1.0)


def _piecewise_nll(observed: np.ndarray, quantile_values: np.ndarray) -> np.ndarray:
    support, probabilities = _support_values(quantile_values)
    evaluation_points = np.clip(np.asarray(observed, dtype=float), 1e-6, 1.0 - 1e-6)
    indices = (support <= evaluation_points[:, None]).sum(axis=1) - 1
    indices = np.clip(indices, 0, support.shape[1] - 2)
    rows = np.arange(len(support))
    value_width = support[rows, indices + 1] - support[rows, indices]
    probability_width = probabilities[indices + 1] - probabilities[indices]
    density = probability_width / np.maximum(value_width, 1e-4)
    return -np.log(np.clip(density, 1e-12, 1e6))


def _crps(observed: np.ndarray, quantile_values: np.ndarray) -> np.ndarray:
    support, probabilities = _support_values(quantile_values)
    residual = np.asarray(observed, dtype=float)[:, None] - support
    pinball = np.maximum(
        probabilities[None, :] * residual,
        (probabilities[None, :] - 1.0) * residual,
    )
    return 2.0 * np.trapezoid(pinball, probabilities, axis=1)


def _predictive_moments(quantile_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support, probabilities = _support_values(quantile_values)
    mean = np.trapezoid(support, probabilities, axis=1)
    second_moment = np.trapezoid(support**2, probabilities, axis=1)
    standard_deviation = np.sqrt(np.maximum(second_moment - mean**2, 0.0))
    return mean, standard_deviation


def _distribution_metrics(
    observed: np.ndarray,
    quantile_values: np.ndarray,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    pit = _cdf_at(quantile_values, observed)
    nll = _piecewise_nll(observed, quantile_values)
    crps = _crps(observed, quantile_values)
    predictive_mean, _predictive_std = _predictive_moments(quantile_values)
    calibration_errors = []
    result: dict[str, Any] = {
        "nll": float(nll.mean()),
        "crps": float(crps.mean()),
        "pit_mean": float(pit.mean()),
        "pit_variance": float(pit.var()),
        "coverage_80": float(
            np.mean((observed >= quantile_values[:, 1]) & (observed <= quantile_values[:, 5]))
        ),
        "coverage_90": float(
            np.mean((observed >= quantile_values[:, 0]) & (observed <= quantile_values[:, 6]))
        ),
        "interval_width_80": float(np.mean(quantile_values[:, 5] - quantile_values[:, 1])),
        "interval_width_90": float(np.mean(quantile_values[:, 6] - quantile_values[:, 0])),
        "median_mae": float(mean_absolute_error(observed, quantile_values[:, 3])),
        "median_rmse": float(math.sqrt(mean_squared_error(observed, quantile_values[:, 3]))),
        "predictive_mean_mae": float(mean_absolute_error(observed, predictive_mean)),
        "predictive_mean_rmse": float(math.sqrt(mean_squared_error(observed, predictive_mean))),
    }
    result["coverage_error_80"] = abs(result["coverage_80"] - 0.80)
    result["coverage_error_90"] = abs(result["coverage_90"] - 0.90)
    brier_values = []
    for threshold in BRIER_THRESHOLDS:
        cdf = _cdf_at(quantile_values, np.full(len(observed), threshold, dtype=float))
        event = (observed <= threshold).astype(float)
        brier = float(np.mean((event - cdf) ** 2))
        result[f"brier_le_{_threshold_suffix(threshold)}"] = brier
        brier_values.append(brier)
    result["brier_threshold_mean"] = float(np.mean(brier_values))
    for index, level in enumerate(QUANTILE_LEVELS):
        empirical = float(np.mean(observed <= quantile_values[:, index]))
        error = empirical - level
        result[f"empirical_cdf_{_quantile_column(level)}"] = empirical
        result[f"calibration_error_{_quantile_column(level)}"] = error
        calibration_errors.append(abs(error))
    result["calibration_mae"] = float(np.mean(calibration_errors))
    result["calibration_max_abs"] = float(np.max(calibration_errors))
    return result


def _fit_distribution(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_train: np.ndarray,
    *,
    job_id: str,
    master_seed: int,
    gradient_boosting_iterations: int,
) -> tuple[dict[str, HistGradientBoostingRegressor], np.ndarray, dict[str, Any]]:
    estimators: dict[str, HistGradientBoostingRegressor] = {}
    raw_predictions = np.empty((len(x_validation), len(QUANTILE_LEVELS)), dtype=float)
    for index, level in enumerate(QUANTILE_LEVELS):
        estimator = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=level,
            max_iter=gradient_boosting_iterations,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=_job_seed(master_seed, job_id, level),
        )
        estimator.fit(x_train, y_train)
        column = _quantile_column(level)
        estimators[column] = estimator
        raw_predictions[:, index] = np.clip(estimator.predict(x_validation), 0.0, 1.0)
    differences = np.diff(raw_predictions, axis=1)
    crossing_rows = np.any(differences < 0.0, axis=1)
    crossing = {
        "raw_crossing_row_count": int(crossing_rows.sum()),
        "raw_crossing_adjacent_pair_count": int((differences < 0.0).sum()),
        "maximum_raw_crossing_violation": float(
            np.maximum(-differences, 0.0).max(initial=0.0)
        ),
        "postprocessing": "row-wise increasing rearrangement followed by [0, 1] clipping",
    }
    return estimators, np.sort(raw_predictions, axis=1), crossing


def _prediction_frame(
    validation_features: pd.DataFrame,
    observed: np.ndarray,
    quantile_values: np.ndarray,
    *,
    split_name: str,
    fold: int,
) -> pd.DataFrame:
    predictive_mean, predictive_std = _predictive_moments(quantile_values)
    frame = validation_features[["feature_id", "cutoff"]].reset_index(drop=True).copy()
    frame["split_name"] = split_name
    frame["fold"] = fold
    frame["observed_quality"] = observed.astype(float)
    for index, level in enumerate(QUANTILE_LEVELS):
        frame[_quantile_column(level)] = quantile_values[:, index]
    frame["predictive_mean"] = predictive_mean
    frame["predictive_std"] = predictive_std
    frame["pit"] = _cdf_at(quantile_values, observed)
    return frame


def _all_jobs(preprocessing: dict[str, Any]) -> list[tuple[str, int]]:
    jobs = []
    for split_name in sorted(preprocessing["splits"]):
        for fold in sorted(int(value) for value in preprocessing["splits"][split_name]):
            jobs.append((split_name, fold))
    return jobs


def _run_jobs(
    run_path: Path,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
    state: dict[str, Any],
    *,
    master_seed: int,
    gradient_boosting_iterations: int,
    progress_path: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[Path]]:
    fold_schemas: dict[str, Any] = {
        "schema_version": PHASE8_SCHEMA_VERSION,
        "source_preprocessing_schema_version": preprocessing["schema_version"],
        "transform_policy": (
            "reuse Phase 6 training-fold imputation, clipping, scaling, missing indicators, "
            "and categorical vocabularies through the verified Phase 7 transformer"
        ),
        "folds": {},
    }
    output_paths: list[Path] = []
    quality_values = pd.to_numeric(targets[QUALITY_TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(quality_values).all() or not np.logical_and(
        quality_values >= 0.0, quality_values <= 1.0
    ).all():
        raise ValueError("Phase 8 quality target must be finite and bounded in [0, 1]")
    for split_name in sorted(preprocessing["splits"]):
        fold_column = feature_schema["split_columns"][split_name]
        for fold in sorted(int(value) for value in preprocessing["splits"][split_name]):
            fold_key = f"{split_name}/fold{fold}"
            fold_specification = preprocessing["splits"][split_name][str(fold)]
            training_mask, validation_mask, fold_contract = _verify_fold_contract(
                features,
                splits,
                fold_column,
                fold,
                fold_specification,
            )
            x_train, feature_names = _transform_features(
                features.loc[training_mask], fold_specification, feature_schema
            )
            x_validation, validation_feature_names = _transform_features(
                features.loc[validation_mask], fold_specification, feature_schema
            )
            if feature_names != validation_feature_names:
                raise ValueError(f"Phase 8 transformed schema mismatch: {fold_key}")
            fold_schemas["folds"][fold_key] = {
                **fold_contract,
                "fold_column": fold_column,
                "transformed_feature_count": len(feature_names),
                "transformed_feature_names": feature_names,
                "transformed_feature_names_sha256": hashlib.sha256(
                    "\n".join(feature_names).encode("utf-8")
                ).hexdigest(),
            }
            job_id = _job_id(split_name, fold)
            paths = _job_paths(run_path, job_id)
            marker = _verify_job(run_path, job_id)
            if marker is not None:
                state.setdefault("completed_jobs", {})[job_id] = _file_sha256(paths["marker"])
                output_paths.extend(paths.values())
                logger.info("[PHASE8][RESUME] job=%s status=SKIP_VERIFIED", job_id)
                continue
            logger.info(
                "[PHASE8][FOLD] split=%s fold=%s train_rows=%s validation_rows=%s",
                split_name,
                fold,
                int(training_mask.sum()),
                int(validation_mask.sum()),
            )
            y_train = quality_values[training_mask]
            y_validation = quality_values[validation_mask]
            estimators, quantile_values, crossing = _fit_distribution(
                x_train,
                x_validation,
                y_train,
                job_id=job_id,
                master_seed=master_seed,
                gradient_boosting_iterations=gradient_boosting_iterations,
            )
            predictions = _prediction_frame(
                features.loc[validation_mask],
                y_validation,
                quantile_values,
                split_name=split_name,
                fold=fold,
            )
            metrics = {
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "model": MODEL_NAME,
                "target": "quality_distribution",
                "training_row_count": int(training_mask.sum()),
                "validation_row_count": int(validation_mask.sum()),
                "training_problem_count": fold_contract["training_problem_count"],
                "validation_problem_count": fold_contract["validation_problem_count"],
                **crossing,
                **_distribution_metrics(y_validation, quantile_values),
            }
            model_artifact = {
                "schema_version": PHASE8_SCHEMA_VERSION,
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "model": MODEL_NAME,
                "target": QUALITY_TARGET_COLUMN,
                "quantile_levels": list(QUANTILE_LEVELS),
                "feature_names": feature_names,
                "fold_contract": fold_contract,
                "postprocessing": crossing["postprocessing"],
                "estimators": estimators,
            }
            _atomic_parquet(paths["predictions"], predictions)
            _atomic_json(paths["metrics"], metrics)
            _atomic_pickle(paths["model"], model_artifact)
            marker = {
                "schema_version": PHASE8_SCHEMA_VERSION,
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "model": MODEL_NAME,
                "output_sha256": _relative_hashes(
                    run_path,
                    (paths["predictions"], paths["metrics"], paths["model"]),
                ),
            }
            _atomic_json(paths["marker"], marker)
            state.setdefault("completed_jobs", {})[job_id] = _file_sha256(paths["marker"])
            state["last_completed_job"] = job_id
            _atomic_json(run_path / "run_state.json", state)
            _append_progress(progress_path, {"event": "job_completed", "job_id": job_id})
            output_paths.extend(paths.values())
            logger.info(
                "[PHASE8][JOB] job=%s status=FITTED raw_crossing_rows=%s",
                job_id,
                crossing["raw_crossing_row_count"],
            )
    fold_schema_path = run_path / "data/preprocessing/fold_feature_schemas.json"
    _atomic_json(fold_schema_path, fold_schemas)
    output_paths.append(fold_schema_path)
    return fold_schemas, output_paths


def _aggregate_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quantile_columns = [_quantile_column(level) for level in QUANTILE_LEVELS]
    metric_rows = []
    calibration_rows = []
    groups: list[tuple[str, tuple[str, ...]]] = [
        ("overall", ("split_name",)),
        ("cutoff", ("split_name", "cutoff")),
    ]
    for scope, columns in groups:
        for keys, group in predictions.groupby(list(columns), sort=True):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            identifiers = dict(zip(columns, key_values, strict=True))
            observed = group["observed_quality"].to_numpy(dtype=float)
            quantiles = group[quantile_columns].to_numpy(dtype=float)
            row = {
                "scope": scope,
                "split_name": identifiers["split_name"],
                "cutoff": float(identifiers["cutoff"]) if "cutoff" in identifiers else None,
                "row_count": int(len(group)),
                "fold_count": int(group["fold"].nunique()),
                "model": MODEL_NAME,
                **_distribution_metrics(observed, quantiles),
            }
            metric_rows.append(row)
            for index, level in enumerate(QUANTILE_LEVELS):
                empirical = float(np.mean(observed <= quantiles[:, index]))
                calibration_rows.append(
                    {
                        "scope": scope,
                        "split_name": identifiers["split_name"],
                        "cutoff": row["cutoff"],
                        "quantile": level,
                        "empirical_cdf": empirical,
                        "calibration_error": empirical - level,
                        "row_count": int(len(group)),
                    }
                )
    aggregate = pd.DataFrame(metric_rows).sort_values(
        ["scope", "split_name", "cutoff"], na_position="first"
    ).reset_index(drop=True)
    calibration = pd.DataFrame(calibration_rows).sort_values(
        ["scope", "split_name", "cutoff", "quantile"], na_position="first"
    ).reset_index(drop=True)
    return aggregate, calibration


def _aggregate_job_outputs(
    run_path: Path,
    jobs: list[tuple[str, int]],
) -> tuple[list[Path], dict[str, Any]]:
    prediction_frames = []
    metric_rows = []
    registry = {
        "schema_version": PHASE8_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "target": QUALITY_TARGET_COLUMN,
        "quantile_levels": list(QUANTILE_LEVELS),
        "artifact_policy": "one seven-quantile distribution model artifact per split/fold",
        "jobs": [],
    }
    for split_name, fold in jobs:
        job_id = _job_id(split_name, fold)
        marker = _verify_job(run_path, job_id)
        if marker is None:
            raise ValueError(f"Phase 8 job is incomplete: {job_id}")
        paths = _job_paths(run_path, job_id)
        prediction_frames.append(pd.read_parquet(paths["predictions"]))
        metric_rows.append(json.loads(paths["metrics"].read_text(encoding="utf-8")))
        registry["jobs"].append(
            {
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "artifact": str(paths["model"].relative_to(run_path)),
                "artifact_sha256": marker["output_sha256"][
                    str(paths["model"].relative_to(run_path))
                ],
            }
        )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["split_name", "feature_id"]
    ).reset_index(drop=True)
    fold_metrics = pd.DataFrame(metric_rows).sort_values(
        ["split_name", "fold"]
    ).reset_index(drop=True)
    aggregate_metrics, calibration = _aggregate_metrics(predictions)
    predictions_path = run_path / "data/predictions/oof_quality_distribution.parquet"
    fold_metrics_path = run_path / "data/metrics/fold_distribution_metrics.parquet"
    aggregate_metrics_path = run_path / "data/metrics/aggregate_distribution_metrics.parquet"
    calibration_path = run_path / "data/calibration/quantile_calibration.parquet"
    registry_path = run_path / "model_registry.json"
    _atomic_parquet(predictions_path, predictions)
    _atomic_parquet(fold_metrics_path, fold_metrics)
    _atomic_parquet(aggregate_metrics_path, aggregate_metrics)
    _atomic_parquet(calibration_path, calibration)
    _atomic_json(registry_path, registry)
    return [
        predictions_path,
        fold_metrics_path,
        aggregate_metrics_path,
        calibration_path,
        registry_path,
    ], registry


def _validate_phase8(
    run_path: Path,
    phase6_path: Path,
    phase7_path: Path,
    original_input_fingerprints: dict[str, str],
    phase6_validation: dict[str, Any],
    phase7_validation: dict[str, Any],
    features: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    _, _, current_phase6_fingerprints = _validate_phase6_input(phase6_path)
    _, _, current_phase7_fingerprints = _validate_phase7_input(
        phase7_path, current_phase6_fingerprints
    )
    current_inputs = _combined_input_fingerprints(
        current_phase6_fingerprints, current_phase7_fingerprints
    )
    predictions = pd.read_parquet(
        run_path / "data/predictions/oof_quality_distribution.parquet"
    )
    fold_metrics = pd.read_parquet(
        run_path / "data/metrics/fold_distribution_metrics.parquet"
    )
    aggregate_metrics = pd.read_parquet(
        run_path / "data/metrics/aggregate_distribution_metrics.parquet"
    )
    calibration = pd.read_parquet(
        run_path / "data/calibration/quantile_calibration.parquet"
    )
    registry = json.loads((run_path / "model_registry.json").read_text(encoding="utf-8"))
    fold_schemas = json.loads(
        (run_path / "data/preprocessing/fold_feature_schemas.json").read_text(encoding="utf-8")
    )
    jobs = _all_jobs(preprocessing)
    expected_prediction_rows = len(features) * len(preprocessing["splits"])
    quantile_columns = [_quantile_column(level) for level in QUANTILE_LEVELS]
    quantile_values = predictions[quantile_columns].to_numpy(dtype=float)
    coverage_ok = True
    for split_name in preprocessing["splits"]:
        group = predictions.loc[predictions["split_name"] == split_name]
        if len(group) != len(features) or set(group["feature_id"]) != set(features["feature_id"]):
            coverage_ok = False
    fold_assignment_ok = True
    for split_name, fold_column in feature_schema["split_columns"].items():
        expected_folds = pd.Series(
            splits[fold_column].to_numpy(dtype=int),
            index=splits["feature_id"].astype(str),
        )
        observed = predictions.loc[predictions["split_name"] == split_name]
        mapped = observed["feature_id"].astype(str).map(expected_folds)
        if mapped.isna().any() or not np.array_equal(
            mapped.to_numpy(dtype=int), observed["fold"].to_numpy(dtype=int)
        ):
            fold_assignment_ok = False
    expected_quality = pd.Series(
        pd.to_numeric(targets[QUALITY_TARGET_COLUMN], errors="coerce").to_numpy(dtype=float),
        index=targets["feature_id"].astype(str),
    )
    mapped_quality = predictions["feature_id"].astype(str).map(expected_quality).to_numpy(dtype=float)
    quality_target_exact = bool(
        np.allclose(mapped_quality, predictions["observed_quality"].to_numpy(dtype=float))
    )
    required_metric_columns = [
        "nll",
        "crps",
        "calibration_mae",
        "calibration_max_abs",
        "coverage_80",
        "coverage_90",
        "brier_threshold_mean",
        "median_mae",
        "predictive_mean_mae",
    ]
    fold_metrics_finite = bool(
        np.isfinite(fold_metrics[required_metric_columns].to_numpy(dtype=float)).all()
    )
    aggregate_metrics_finite = bool(
        np.isfinite(aggregate_metrics[required_metric_columns].to_numpy(dtype=float)).all()
    )
    expected_aggregate_rows = len(preprocessing["splits"]) * (1 + len(feature_schema["cutoffs"]))
    checks = {
        "phase6_quality_pass": phase6_validation["status"] == "PHASE_6_PASS",
        "phase7_quality_pass": phase7_validation["status"] == "PHASE_7_PASS",
        "source_inputs_unchanged": current_inputs == original_input_fingerprints,
        "expected_job_count": len(fold_metrics) == len(jobs) == len(registry["jobs"]),
        "all_job_markers_verified": all(
            _verify_job(run_path, _job_id(*job)) is not None for job in jobs
        ),
        "expected_prediction_rows": len(predictions) == expected_prediction_rows,
        "prediction_keys_unique": not predictions.duplicated(["feature_id", "split_name"]).any(),
        "oof_coverage_exact": coverage_ok,
        "fold_assignments_exact": fold_assignment_ok,
        "quality_target_exact": quality_target_exact,
        "quantiles_finite": bool(np.isfinite(quantile_values).all()),
        "quantiles_bounded": bool(
            np.logical_and(quantile_values >= 0.0, quantile_values <= 1.0).all()
        ),
        "quantiles_nondecreasing": bool((np.diff(quantile_values, axis=1) >= 0.0).all()),
        "predictive_moments_finite": bool(
            np.isfinite(predictions[["predictive_mean", "predictive_std"]].to_numpy(dtype=float)).all()
        ),
        "predictive_std_nonnegative": bool((predictions["predictive_std"] >= 0.0).all()),
        "pit_bounded": bool(predictions["pit"].between(0.0, 1.0).all()),
        "crps_nonnegative": bool(
            (fold_metrics["crps"] >= 0.0).all() and (aggregate_metrics["crps"] >= 0.0).all()
        ),
        "required_metrics_finite": fold_metrics_finite and aggregate_metrics_finite,
        "aggregate_metrics_complete": len(aggregate_metrics) == expected_aggregate_rows,
        "calibration_rows_complete": len(calibration)
        == expected_aggregate_rows * len(QUANTILE_LEVELS),
        "fold_preprocessing_contract_complete": len(fold_schemas["folds"]) == len(jobs),
        "phase8_scope_quality_only": registry["target"] == QUALITY_TARGET_COLUMN,
    }
    issues = []
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    if failed_checks:
        issues.append({"type": "failed_quality_checks", "checks": failed_checks})
    return {
        "status": "PHASE_8_PASS" if all(checks.values()) else "PHASE_8_FAIL",
        "schema_version": PHASE8_SCHEMA_VERSION,
        "scope": "conditional quality distribution via monotone quantile gradient boosting",
        "performance_gate_policy": (
            "construction, leakage, distribution validity, and integrity only; NLL, CRPS, "
            "calibration, coverage, and Brier values are not Phase 8 pass thresholds"
        ),
        "phase9_boundary": "failure probability and failure types are deferred to Phase 9",
        "phase6_directory": str(phase6_path.resolve()),
        "phase7_directory": str(phase7_path.resolve()),
        "feature_row_count": int(len(features)),
        "prediction_row_count": int(len(predictions)),
        "job_count": int(len(fold_metrics)),
        "model_artifact_count": int(len(registry["jobs"])),
        "aggregate_metric_row_count": int(len(aggregate_metrics)),
        "calibration_row_count": int(len(calibration)),
        "split_count": len(preprocessing["splits"]),
        "fold_count": len(jobs),
        "quantile_count": len(QUANTILE_LEVELS),
        "quantile_levels": list(QUANTILE_LEVELS),
        "brier_thresholds": list(BRIER_THRESHOLDS),
        "raw_crossing_row_count": int(fold_metrics["raw_crossing_row_count"].sum()),
        "checks": checks,
        "issues": issues,
    }


def run_phase8(
    run_directory: str | Path,
    phase6_directory: str | Path,
    phase7_directory: str | Path,
    *,
    master_seed: int = 20_260_824,
    gradient_boosting_iterations: int = 80,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    phase6_path = Path(phase6_directory)
    phase7_path = Path(phase7_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"
    if gradient_boosting_iterations < 1:
        raise ValueError("Phase 8 gradient boosting iterations must be positive")
    phase6_validation, _phase6_manifest, phase6_fingerprints = _validate_phase6_input(
        phase6_path
    )
    phase7_validation, _phase7_manifest, phase7_fingerprints = _validate_phase7_input(
        phase7_path, phase6_fingerprints
    )
    input_fingerprints = _combined_input_fingerprints(
        phase6_fingerprints, phase7_fingerprints
    )
    features, targets, splits, feature_schema, preprocessing = _load_phase6_tables(phase6_path)
    stable_configuration = {
        "phase": 8,
        "schema_version": PHASE8_SCHEMA_VERSION,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "preprocessing_implementation_sha256": _file_sha256(Path(phase7_module.__file__)),
        "input_fingerprints": input_fingerprints,
        "master_seed": int(master_seed),
        "model": MODEL_NAME,
        "target": QUALITY_TARGET_COLUMN,
        "quantile_levels": list(QUANTILE_LEVELS),
        "brier_thresholds": list(BRIER_THRESHOLDS),
        "gradient_boosting_iterations": int(gradient_boosting_iterations),
        "split_columns": feature_schema["split_columns"],
        "cutoffs": feature_schema["cutoffs"],
        "distribution_policy": (
            "quantile models with increasing rearrangement and bounded piecewise-linear CDF"
        ),
    }
    configuration_sha256 = _configuration_hash(stable_configuration)
    configuration = {
        **stable_configuration,
        "phase6_directory": str(phase6_path.resolve()),
        "phase7_directory": str(phase7_path.resolve()),
        "git_commit": current_git_commit(),
        "configuration_sha256": configuration_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Phase 8 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_8_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 8 resume manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest.get("output_sha256", {}).items():
                path = run_path / relative
                if not path.is_file() or _file_sha256(path) != expected_hash:
                    raise ValueError(f"completed Phase 8 output hash mismatch: {relative}")
            active_logger.info("[PHASE8][RESUME] status=PHASE_8_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info(
            "[PHASE8][RESUME] completed_jobs=%s",
            len(state.get("completed_jobs", {})),
        )
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE8_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": configuration_sha256,
            "input_fingerprints": input_fingerprints,
            "completed_jobs": {},
            "last_completed_job": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase8_started"})
    fold_schemas, job_outputs = _run_jobs(
        run_path,
        features,
        targets,
        splits,
        feature_schema,
        preprocessing,
        state,
        master_seed=master_seed,
        gradient_boosting_iterations=gradient_boosting_iterations,
        progress_path=progress_path,
        logger=active_logger,
    )
    jobs = _all_jobs(preprocessing)
    aggregate_outputs, _registry = _aggregate_job_outputs(run_path, jobs)
    validation = _validate_phase8(
        run_path,
        phase6_path,
        phase7_path,
        input_fingerprints,
        phase6_validation,
        phase7_validation,
        features,
        targets,
        splits,
        feature_schema,
        preprocessing,
    )
    validation["executed_at"] = datetime.now(_timezone()).isoformat(timespec="seconds")
    validation["elapsed_seconds"] = perf_counter() - started_at
    validation["configuration"] = configuration
    validation_path = run_path / "validation.json"
    _atomic_json(validation_path, validation)
    output_paths = sorted(
        {
            path
            for path in [
                run_path / "config.json",
                *job_outputs,
                *aggregate_outputs,
                validation_path,
            ]
        },
        key=lambda path: str(path),
    )
    manifest = {
        "phase": 8,
        "status": validation["status"],
        "schema_version": PHASE8_SCHEMA_VERSION,
        "input_fingerprints": input_fingerprints,
        "output_sha256": _relative_hashes(run_path, output_paths),
        "row_counts": {
            "features": validation["feature_row_count"],
            "predictions": validation["prediction_row_count"],
            "jobs": validation["job_count"],
            "calibration": validation["calibration_row_count"],
        },
        "phase9_allowed": validation["status"] == "PHASE_8_PASS",
        "performance_gate_policy": validation["performance_gate_policy"],
    }
    manifest_path = run_path / "manifest.json"
    _atomic_json(manifest_path, manifest)
    state["status"] = validation["status"]
    state["last_completed_stage"] = "validation"
    state["manifest_sha256"] = _file_sha256(manifest_path)
    state["validation_sha256"] = _file_sha256(validation_path)
    state["fold_schema_count"] = len(fold_schemas["folds"])
    _atomic_json(state_path, state)
    _append_progress(progress_path, {"event": "phase8_finished", "status": validation["status"]})
    active_logger.info(
        "[PHASE8][SUMMARY] status=%s jobs=%s predictions=%s artifacts=%s directory=%s",
        validation["status"],
        validation["job_count"],
        validation["prediction_row_count"],
        validation["model_artifact_count"],
        run_path,
    )
    return validation
