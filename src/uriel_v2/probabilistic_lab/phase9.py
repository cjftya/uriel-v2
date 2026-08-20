from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

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
from uriel_v2.probabilistic_lab.phase8 import (
    _combined_input_fingerprints as _phase8_input_fingerprints,
)
from uriel_v2.probabilistic_lab.phase8 import _validate_phase7_input
from uriel_v2.provenance import current_git_commit


PHASE9_SCHEMA_VERSION = "phase9-v1"
BINARY_MODEL_NAME = "hist_gradient_boosting_failure"
TYPE_MODEL_NAME = "hist_gradient_boosting_failure_type"
FAILURE_TARGET_COLUMN = "target_failure"
DEFAULT_CALIBRATION_BINS = 10
DEFAULT_BETA_PRIOR = (0.5, 0.5)
PROBABILITY_EPSILON = 1e-12


def _phase8_required_paths(phase8_path: Path) -> dict[str, Path]:
    return {
        "phase8_config": phase8_path / "config.json",
        "phase8_manifest": phase8_path / "manifest.json",
        "phase8_validation": phase8_path / "validation.json",
        "phase8_predictions": phase8_path
        / "data/predictions/oof_quality_distribution.parquet",
        "phase8_fold_metrics": phase8_path
        / "data/metrics/fold_distribution_metrics.parquet",
        "phase8_aggregate_metrics": phase8_path
        / "data/metrics/aggregate_distribution_metrics.parquet",
        "phase8_calibration": phase8_path
        / "data/calibration/quantile_calibration.parquet",
        "phase8_fold_schemas": phase8_path
        / "data/preprocessing/fold_feature_schemas.json",
        "phase8_model_registry": phase8_path / "model_registry.json",
    }


def _validate_phase8_input(
    phase8_path: Path,
    phase6_phase7_fingerprints: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    required = _phase8_required_paths(phase8_path)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 9 Phase 8 input files are missing: {missing[:5]}")
    validation = json.loads(required["phase8_validation"].read_text(encoding="utf-8"))
    manifest = json.loads(required["phase8_manifest"].read_text(encoding="utf-8"))
    if validation.get("status") != "PHASE_8_PASS":
        raise ValueError("Phase 9 requires a PHASE_8_PASS distribution run")
    if manifest.get("status") != "PHASE_8_PASS" or not manifest.get("phase9_allowed"):
        raise ValueError("Phase 8 manifest does not allow Phase 9")
    if validation.get("configuration", {}).get("input_fingerprints") != phase6_phase7_fingerprints:
        raise ValueError("Phase 8 was not built from the supplied Phase 6 and Phase 7 inputs")
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = phase8_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 8 manifest output hash mismatch: {relative}")
    fingerprints = {name: _file_sha256(path) for name, path in sorted(required.items())}
    return validation, manifest, fingerprints


def _combined_input_fingerprints(
    phase6_fingerprints: dict[str, str],
    phase7_fingerprints: dict[str, str],
    phase8_fingerprints: dict[str, str],
    failure_label_source_sha256: str,
) -> dict[str, str]:
    return {
        **_phase8_input_fingerprints(phase6_fingerprints, phase7_fingerprints),
        **{f"phase8/{name}": value for name, value in phase8_fingerprints.items()},
        "phase4/failure_label_source": failure_label_source_sha256,
    }


def _load_failure_labels(
    phase6_path: Path,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], Path, str]:
    phase6_config = json.loads((phase6_path / "config.json").read_text(encoding="utf-8"))
    phase4_path = Path(phase6_config["phase4_directory"])
    runs_path = phase4_path / "data/runs/runs.parquet"
    if not runs_path.is_file():
        raise FileNotFoundError(f"Phase 9 failure label source is missing: {runs_path}")
    source_sha256 = _file_sha256(runs_path)
    expected_sha256 = phase6_config.get("input_fingerprints", {}).get("phase4_runs")
    if source_sha256 != expected_sha256:
        raise ValueError("Phase 9 failure label source no longer matches the frozen Phase 6 input")
    runs = pd.read_parquet(
        runs_path,
        columns=["run_id", "failure", "failure_type", "timeout", "status"],
    )
    if not runs["run_id"].is_unique:
        raise ValueError("Phase 9 failure label source must have unique run_id values")
    indexed = runs.set_index(runs["run_id"].astype(str))
    run_ids = targets["run_id"].astype(str)
    source_failure = run_ids.map(indexed["failure"])
    if source_failure.isna().any():
        raise ValueError("Phase 9 failure label source does not cover every Phase 6 target")
    observed_failure = targets[FAILURE_TARGET_COLUMN].astype(bool).reset_index(drop=True)
    if not np.array_equal(source_failure.to_numpy(dtype=bool), observed_failure.to_numpy(dtype=bool)):
        raise ValueError("Phase 9 Phase 4 and Phase 6 failure labels disagree")
    source_types = run_ids.map(indexed["failure_type"]).reset_index(drop=True)
    normalized_types = source_types.where(observed_failure, None)
    missing_positive_type = observed_failure & (
        normalized_types.isna() | normalized_types.astype(str).str.strip().eq("")
    )
    if missing_positive_type.any():
        raise ValueError("Phase 9 observed failures require a non-empty failure_type")
    labels = targets[["feature_id", "run_id", "problem_id", "cutoff"]].copy()
    labels["observed_failure"] = observed_failure.to_numpy(dtype=bool)
    labels["observed_failure_type"] = normalized_types
    global_types = sorted(
        str(value)
        for value in labels.loc[labels["observed_failure"], "observed_failure_type"].dropna().unique()
    )
    return labels, global_types, runs_path, source_sha256


def _job_id(split_name: str, fold: int) -> str:
    return f"{split_name}__fold{fold}__failure_distribution"


def _job_paths(run_path: Path, job_id: str) -> dict[str, Path]:
    return {
        "binary_predictions": run_path
        / "checkpoints/predictions/binary"
        / f"{job_id}.parquet",
        "type_predictions": run_path
        / "checkpoints/predictions/types"
        / f"{job_id}.parquet",
        "metrics": run_path / "checkpoints/metrics" / f"{job_id}.json",
        "support": run_path / "checkpoints/support" / f"{job_id}.parquet",
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
            raise ValueError(f"Phase 9 job output hash mismatch: job={job_id} path={relative}")
    return marker


def _job_seed(master_seed: int, job_id: str, target: str) -> int:
    offset = int(
        hashlib.sha256(f"{job_id}:{target}".encode("utf-8")).hexdigest()[:8],
        16,
    )
    return int((master_seed + offset) % (2**31 - 1))


def _binary_metrics(
    observed: np.ndarray,
    probability: np.ndarray,
    *,
    calibration_bins: int,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=bool)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    clipped = np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    numeric = observed.astype(float)
    brier = float(np.mean((numeric - probability) ** 2))
    binary_log_loss = float(
        -np.mean(numeric * np.log(clipped) + (1.0 - numeric) * np.log(1.0 - clipped))
    )
    bin_index = np.minimum((probability * calibration_bins).astype(int), calibration_bins - 1)
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index in range(calibration_bins):
        mask = bin_index == index
        if not mask.any():
            continue
        gap = abs(float(numeric[mask].mean()) - float(probability[mask].mean()))
        weighted_gap += float(mask.mean()) * gap
        maximum_gap = max(maximum_gap, gap)
    diverse = bool(observed.any() and (~observed).any())
    return {
        "failure_prevalence": float(numeric.mean()),
        "mean_failure_probability": float(probability.mean()),
        "brier": brier,
        "log_loss": binary_log_loss,
        "ece": float(weighted_gap),
        "calibration_max_abs": float(maximum_gap),
        "roc_auc": float(roc_auc_score(observed, probability)) if diverse else None,
        "average_precision": (
            float(average_precision_score(observed, probability)) if diverse else None
        ),
        "binary_discrimination_available": diverse,
    }


def _calibration_rows(
    observed: np.ndarray,
    probability: np.ndarray,
    *,
    calibration_bins: int,
    scope: str,
    split_name: str,
    cutoff: float | None,
) -> list[dict[str, Any]]:
    observed = np.asarray(observed, dtype=bool).astype(float)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    bin_index = np.minimum((probability * calibration_bins).astype(int), calibration_bins - 1)
    rows = []
    for index in range(calibration_bins):
        mask = bin_index == index
        count = int(mask.sum())
        mean_probability = float(probability[mask].mean()) if count else None
        empirical_rate = float(observed[mask].mean()) if count else None
        rows.append(
            {
                "scope": scope,
                "split_name": split_name,
                "cutoff": cutoff,
                "bin": index,
                "lower_bound": index / calibration_bins,
                "upper_bound": (index + 1) / calibration_bins,
                "row_count": count,
                "mean_failure_probability": mean_probability,
                "empirical_failure_rate": empirical_rate,
                "absolute_gap": (
                    abs(empirical_rate - mean_probability) if count else None
                ),
            }
        )
    return rows


def _type_metrics(
    observed_failure: np.ndarray,
    observed_type: np.ndarray,
    conditional_probability: np.ndarray,
    global_types: list[str],
    *,
    model_status: str,
) -> dict[str, Any]:
    observed_failure = np.asarray(observed_failure, dtype=bool)
    positive_count = int(observed_failure.sum())
    if positive_count == 0:
        return {
            "type_metric_status": "unavailable_no_positive_validation_rows",
            "type_validation_positive_count": 0,
            "type_log_loss": None,
            "type_brier": None,
            "type_top1_accuracy": None,
            "type_macro_f1": None,
        }
    if not global_types or conditional_probability.shape[1] == 0:
        return {
            "type_metric_status": "unavailable_no_observed_failure_types",
            "type_validation_positive_count": positive_count,
            "type_log_loss": None,
            "type_brier": None,
            "type_top1_accuracy": None,
            "type_macro_f1": None,
        }
    type_to_index = {value: index for index, value in enumerate(global_types)}
    actual = np.asarray(observed_type, dtype=object)[observed_failure]
    actual_index = np.asarray([type_to_index[str(value)] for value in actual], dtype=int)
    probability = conditional_probability[observed_failure]
    rows = np.arange(positive_count)
    true_probability = np.clip(
        probability[rows, actual_index], PROBABILITY_EPSILON, 1.0
    )
    one_hot = np.zeros_like(probability)
    one_hot[rows, actual_index] = 1.0
    predicted_index = probability.argmax(axis=1)
    return {
        "type_metric_status": f"available:{model_status}",
        "type_validation_positive_count": positive_count,
        "type_log_loss": float(-np.log(true_probability).mean()),
        "type_brier": float(np.mean(np.sum((one_hot - probability) ** 2, axis=1))),
        "type_top1_accuracy": float(np.mean(predicted_index == actual_index)),
        "type_macro_f1": float(
            f1_score(
                actual_index,
                predicted_index,
                labels=np.arange(len(global_types)),
                average="macro",
                zero_division=0,
            )
        ),
    }


def _fit_failure_models(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_failure_train: np.ndarray,
    y_type_train: np.ndarray,
    global_types: list[str],
    *,
    job_id: str,
    master_seed: int,
    gradient_boosting_iterations: int,
    beta_prior: tuple[float, float],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str, str]:
    y_failure_train = np.asarray(y_failure_train, dtype=bool)
    alpha, beta = beta_prior
    binary_estimator: HistGradientBoostingClassifier | None = None
    if np.unique(y_failure_train).size < 2:
        constant_probability = float(
            (int(y_failure_train.sum()) + alpha) / (len(y_failure_train) + alpha + beta)
        )
        failure_probability = np.full(len(x_validation), constant_probability, dtype=float)
        binary_status = "beta_binomial_fallback"
    else:
        binary_estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=gradient_boosting_iterations,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=_job_seed(master_seed, job_id, "failure"),
        )
        binary_estimator.fit(x_train, y_failure_train.astype(int))
        positive_index = list(binary_estimator.classes_).index(1)
        failure_probability = binary_estimator.predict_proba(x_validation)[:, positive_index]
        constant_probability = None
        binary_status = "fitted"
    failure_probability = np.clip(failure_probability, 0.0, 1.0)

    type_estimator: HistGradientBoostingClassifier | None = None
    conditional_probability = np.empty((len(x_validation), len(global_types)), dtype=float)
    positive_mask = y_failure_train
    training_types = np.asarray(y_type_train, dtype=object)[positive_mask]
    observed_training_types = sorted(str(value) for value in np.unique(training_types))
    if not global_types:
        conditional_probability = np.empty((len(x_validation), 0), dtype=float)
        type_status = "unavailable_no_observed_failure_types"
    elif len(training_types) == 0:
        conditional_probability.fill(1.0 / len(global_types))
        type_status = "uniform_fallback_no_positive_training_rows"
    elif len(observed_training_types) == 1:
        conditional_probability.fill(0.0)
        conditional_probability[:, global_types.index(observed_training_types[0])] = 1.0
        type_status = "constant_type_fallback"
    else:
        type_estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=gradient_boosting_iterations,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=_job_seed(master_seed, job_id, "failure_type"),
        )
        type_estimator.fit(x_train[positive_mask], training_types.astype(str))
        trained_probability = type_estimator.predict_proba(x_validation)
        conditional_probability.fill(0.0)
        for source_index, label in enumerate(type_estimator.classes_):
            conditional_probability[:, global_types.index(str(label))] = trained_probability[
                :, source_index
            ]
        type_status = "fitted"
    model = {
        "binary_estimator": binary_estimator,
        "binary_status": binary_status,
        "constant_failure_probability": constant_probability,
        "beta_prior": {"alpha": alpha, "beta": beta},
        "type_estimator": type_estimator,
        "type_status": type_status,
        "global_failure_types": list(global_types),
        "training_failure_types": observed_training_types,
    }
    return model, failure_probability, conditional_probability, binary_status, type_status


def _empty_type_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_id": pd.Series(dtype="object"),
            "cutoff": pd.Series(dtype="float64"),
            "split_name": pd.Series(dtype="object"),
            "fold": pd.Series(dtype="int64"),
            "observed_failure": pd.Series(dtype="bool"),
            "observed_failure_type": pd.Series(dtype="object"),
            "failure_type": pd.Series(dtype="object"),
            "conditional_probability": pd.Series(dtype="float64"),
            "joint_probability": pd.Series(dtype="float64"),
            "type_fit_status": pd.Series(dtype="object"),
        }
    )


def _prediction_frames(
    validation_features: pd.DataFrame,
    validation_labels: pd.DataFrame,
    failure_probability: np.ndarray,
    conditional_probability: np.ndarray,
    global_types: list[str],
    *,
    split_name: str,
    fold: int,
    binary_status: str,
    type_status: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    binary = validation_features[["feature_id", "cutoff"]].reset_index(drop=True).copy()
    label_frame = validation_labels.reset_index(drop=True)
    binary["split_name"] = split_name
    binary["fold"] = fold
    binary["observed_failure"] = label_frame["observed_failure"].to_numpy(dtype=bool)
    binary["observed_failure_type"] = label_frame["observed_failure_type"]
    binary["failure_probability"] = failure_probability
    binary["binary_fit_status"] = binary_status
    binary["type_fit_status"] = type_status
    if not global_types:
        return binary, _empty_type_predictions()
    frames = []
    for index, failure_type in enumerate(global_types):
        frame = binary[
            [
                "feature_id",
                "cutoff",
                "split_name",
                "fold",
                "observed_failure",
                "observed_failure_type",
            ]
        ].copy()
        frame["failure_type"] = failure_type
        frame["conditional_probability"] = conditional_probability[:, index]
        frame["joint_probability"] = failure_probability * conditional_probability[:, index]
        frame["type_fit_status"] = type_status
        frames.append(frame)
    return binary, pd.concat(frames, ignore_index=True)


def _support_frame(
    job_id: str,
    split_name: str,
    fold: int,
    y_failure_train: np.ndarray,
    y_failure_validation: np.ndarray,
    y_type_train: np.ndarray,
    y_type_validation: np.ndarray,
    global_types: list[str],
) -> pd.DataFrame:
    rows = []
    for label, value in (("no_failure", False), ("failure", True)):
        rows.append(
            {
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "target": "failure",
                "class_label": label,
                "training_count": int(np.sum(y_failure_train == value)),
                "validation_count": int(np.sum(y_failure_validation == value)),
            }
        )
    for failure_type in global_types:
        rows.append(
            {
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "target": "failure_type_given_failure",
                "class_label": failure_type,
                "training_count": int(
                    np.sum(y_failure_train & (y_type_train.astype(str) == failure_type))
                ),
                "validation_count": int(
                    np.sum(y_failure_validation & (y_type_validation.astype(str) == failure_type))
                ),
            }
        )
    return pd.DataFrame(rows)


def _all_jobs(preprocessing: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        (split_name, fold)
        for split_name in sorted(preprocessing["splits"])
        for fold in sorted(int(value) for value in preprocessing["splits"][split_name])
    ]


def _run_jobs(
    run_path: Path,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
    global_types: list[str],
    state: dict[str, Any],
    *,
    master_seed: int,
    gradient_boosting_iterations: int,
    calibration_bins: int,
    beta_prior: tuple[float, float],
    progress_path: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[Path]]:
    fold_schemas: dict[str, Any] = {
        "schema_version": PHASE9_SCHEMA_VERSION,
        "source_preprocessing_schema_version": preprocessing["schema_version"],
        "transform_policy": (
            "reuse Phase 6 training-fold preprocessing through the verified Phase 7 transformer"
        ),
        "folds": {},
    }
    output_paths: list[Path] = []
    failure_values = labels["observed_failure"].to_numpy(dtype=bool)
    type_values = labels["observed_failure_type"].to_numpy(dtype=object)
    for split_name in sorted(preprocessing["splits"]):
        fold_column = feature_schema["split_columns"][split_name]
        for fold in sorted(int(value) for value in preprocessing["splits"][split_name]):
            fold_key = f"{split_name}/fold{fold}"
            fold_specification = preprocessing["splits"][split_name][str(fold)]
            training_mask, validation_mask, fold_contract = _verify_fold_contract(
                features, splits, fold_column, fold, fold_specification
            )
            x_train, feature_names = _transform_features(
                features.loc[training_mask], fold_specification, feature_schema
            )
            x_validation, validation_feature_names = _transform_features(
                features.loc[validation_mask], fold_specification, feature_schema
            )
            if feature_names != validation_feature_names:
                raise ValueError(f"Phase 9 transformed schema mismatch: {fold_key}")
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
                logger.info("[PHASE9][RESUME] job=%s status=SKIP_VERIFIED", job_id)
                continue
            y_failure_train = failure_values[training_mask]
            y_failure_validation = failure_values[validation_mask]
            y_type_train = type_values[training_mask]
            y_type_validation = type_values[validation_mask]
            logger.info(
                "[PHASE9][FOLD] split=%s fold=%s train_rows=%s validation_rows=%s train_failures=%s",
                split_name,
                fold,
                int(training_mask.sum()),
                int(validation_mask.sum()),
                int(y_failure_train.sum()),
            )
            model, failure_probability, conditional_probability, binary_status, type_status = (
                _fit_failure_models(
                    x_train,
                    x_validation,
                    y_failure_train,
                    y_type_train,
                    global_types,
                    job_id=job_id,
                    master_seed=master_seed,
                    gradient_boosting_iterations=gradient_boosting_iterations,
                    beta_prior=beta_prior,
                )
            )
            binary_predictions, type_predictions = _prediction_frames(
                features.loc[validation_mask],
                labels.loc[validation_mask],
                failure_probability,
                conditional_probability,
                global_types,
                split_name=split_name,
                fold=fold,
                binary_status=binary_status,
                type_status=type_status,
            )
            metrics = {
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "binary_model": BINARY_MODEL_NAME,
                "type_model": TYPE_MODEL_NAME,
                "binary_fit_status": binary_status,
                "type_fit_status": type_status,
                "training_row_count": int(training_mask.sum()),
                "validation_row_count": int(validation_mask.sum()),
                "training_problem_count": fold_contract["training_problem_count"],
                "validation_problem_count": fold_contract["validation_problem_count"],
                "training_positive_count": int(y_failure_train.sum()),
                "validation_positive_count": int(y_failure_validation.sum()),
                "global_failure_type_count": len(global_types),
                "training_failure_type_count": len(
                    set(str(value) for value in y_type_train[y_failure_train])
                ),
                **_binary_metrics(
                    y_failure_validation,
                    failure_probability,
                    calibration_bins=calibration_bins,
                ),
                **_type_metrics(
                    y_failure_validation,
                    y_type_validation,
                    conditional_probability,
                    global_types,
                    model_status=type_status,
                ),
            }
            support = _support_frame(
                job_id,
                split_name,
                fold,
                y_failure_train,
                y_failure_validation,
                y_type_train,
                y_type_validation,
                global_types,
            )
            model_artifact = {
                "schema_version": PHASE9_SCHEMA_VERSION,
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "binary_model": BINARY_MODEL_NAME,
                "type_model": TYPE_MODEL_NAME,
                "feature_names": feature_names,
                "fold_contract": fold_contract,
                **model,
            }
            _atomic_parquet(paths["binary_predictions"], binary_predictions)
            _atomic_parquet(paths["type_predictions"], type_predictions)
            _atomic_json(paths["metrics"], metrics)
            _atomic_parquet(paths["support"], support)
            _atomic_pickle(paths["model"], model_artifact)
            marker = {
                "schema_version": PHASE9_SCHEMA_VERSION,
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "binary_fit_status": binary_status,
                "type_fit_status": type_status,
                "output_sha256": _relative_hashes(
                    run_path,
                    (
                        paths["binary_predictions"],
                        paths["type_predictions"],
                        paths["metrics"],
                        paths["support"],
                        paths["model"],
                    ),
                ),
            }
            _atomic_json(paths["marker"], marker)
            state.setdefault("completed_jobs", {})[job_id] = _file_sha256(paths["marker"])
            state["last_completed_job"] = job_id
            _atomic_json(run_path / "run_state.json", state)
            _append_progress(
                progress_path,
                {
                    "event": "job_completed",
                    "job_id": job_id,
                    "binary_fit_status": binary_status,
                    "type_fit_status": type_status,
                },
            )
            output_paths.extend(paths.values())
            logger.info(
                "[PHASE9][JOB] job=%s binary=%s type=%s",
                job_id,
                binary_status,
                type_status,
            )
    fold_schema_path = run_path / "data/preprocessing/fold_feature_schemas.json"
    _atomic_json(fold_schema_path, fold_schemas)
    output_paths.append(fold_schema_path)
    return fold_schemas, output_paths


def _conditional_matrix_from_long(
    binary_group: pd.DataFrame,
    type_predictions: pd.DataFrame,
    global_types: list[str],
) -> np.ndarray:
    if not global_types:
        return np.empty((len(binary_group), 0), dtype=float)
    selected = type_predictions.loc[
        type_predictions["feature_id"].isin(binary_group["feature_id"])
        & type_predictions["split_name"].eq(binary_group["split_name"].iloc[0])
    ]
    pivot = selected.pivot(
        index="feature_id", columns="failure_type", values="conditional_probability"
    )
    pivot = pivot.reindex(
        index=binary_group["feature_id"].astype(str), columns=global_types, fill_value=0.0
    )
    return pivot.to_numpy(dtype=float)


def _aggregate_metrics(
    binary_predictions: pd.DataFrame,
    type_predictions: pd.DataFrame,
    global_types: list[str],
    calibration_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    calibration_rows = []
    groups: list[tuple[str, tuple[str, ...]]] = [
        ("overall", ("split_name",)),
        ("cutoff", ("split_name", "cutoff")),
    ]
    for scope, columns in groups:
        for keys, group in binary_predictions.groupby(list(columns), sort=True):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            identifiers = dict(zip(columns, key_values, strict=True))
            split_name = str(identifiers["split_name"])
            cutoff = float(identifiers["cutoff"]) if "cutoff" in identifiers else None
            observed = group["observed_failure"].to_numpy(dtype=bool)
            probability = group["failure_probability"].to_numpy(dtype=float)
            conditional = _conditional_matrix_from_long(group, type_predictions, global_types)
            row = {
                "scope": scope,
                "split_name": split_name,
                "cutoff": cutoff,
                "row_count": int(len(group)),
                "fold_count": int(group["fold"].nunique()),
                "positive_count": int(observed.sum()),
                "global_failure_type_count": len(global_types),
                "binary_model": BINARY_MODEL_NAME,
                "type_model": TYPE_MODEL_NAME,
                **_binary_metrics(observed, probability, calibration_bins=calibration_bins),
                **_type_metrics(
                    observed,
                    group["observed_failure_type"].to_numpy(dtype=object),
                    conditional,
                    global_types,
                    model_status="aggregate_oof",
                ),
            }
            metric_rows.append(row)
            calibration_rows.extend(
                _calibration_rows(
                    observed,
                    probability,
                    calibration_bins=calibration_bins,
                    scope=scope,
                    split_name=split_name,
                    cutoff=cutoff,
                )
            )
    aggregate = pd.DataFrame(metric_rows).sort_values(
        ["scope", "split_name", "cutoff"], na_position="first"
    ).reset_index(drop=True)
    calibration = pd.DataFrame(calibration_rows).sort_values(
        ["scope", "split_name", "cutoff", "bin"], na_position="first"
    ).reset_index(drop=True)
    return aggregate, calibration


def _aggregate_job_outputs(
    run_path: Path,
    jobs: list[tuple[str, int]],
    labels: pd.DataFrame,
    global_types: list[str],
    calibration_bins: int,
) -> tuple[list[Path], dict[str, Any]]:
    binary_frames = []
    type_frames = []
    metric_rows = []
    support_frames = []
    registry = {
        "schema_version": PHASE9_SCHEMA_VERSION,
        "targets": ["failure_probability", "failure_type_given_failure"],
        "binary_model": BINARY_MODEL_NAME,
        "type_model": TYPE_MODEL_NAME,
        "global_failure_types": global_types,
        "artifact_policy": "one binary and conditional-type model bundle per split/fold",
        "jobs": [],
    }
    for split_name, fold in jobs:
        job_id = _job_id(split_name, fold)
        marker = _verify_job(run_path, job_id)
        if marker is None:
            raise ValueError(f"Phase 9 job is incomplete: {job_id}")
        paths = _job_paths(run_path, job_id)
        binary_frames.append(pd.read_parquet(paths["binary_predictions"]))
        type_frames.append(pd.read_parquet(paths["type_predictions"]))
        metric_rows.append(json.loads(paths["metrics"].read_text(encoding="utf-8")))
        support_frames.append(pd.read_parquet(paths["support"]))
        registry["jobs"].append(
            {
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "binary_fit_status": marker["binary_fit_status"],
                "type_fit_status": marker["type_fit_status"],
                "artifact": str(paths["model"].relative_to(run_path)),
                "artifact_sha256": marker["output_sha256"][
                    str(paths["model"].relative_to(run_path))
                ],
            }
        )
    binary_predictions = pd.concat(binary_frames, ignore_index=True).sort_values(
        ["split_name", "feature_id"]
    ).reset_index(drop=True)
    nonempty_type_frames = [frame for frame in type_frames if not frame.empty]
    type_predictions = (
        pd.concat(nonempty_type_frames, ignore_index=True).sort_values(
            ["split_name", "feature_id", "failure_type"]
        ).reset_index(drop=True)
        if nonempty_type_frames
        else _empty_type_predictions()
    )
    fold_metrics = pd.DataFrame(metric_rows).sort_values(
        ["split_name", "fold"]
    ).reset_index(drop=True)
    support = pd.concat(support_frames, ignore_index=True).sort_values(
        ["split_name", "fold", "target", "class_label"]
    ).reset_index(drop=True)
    aggregate_metrics, calibration = _aggregate_metrics(
        binary_predictions, type_predictions, global_types, calibration_bins
    )
    binary_path = run_path / "data/predictions/oof_failure_probability.parquet"
    type_path = run_path / "data/predictions/oof_failure_type_probability.parquet"
    labels_path = run_path / "data/targets/failure_labels.parquet"
    fold_metrics_path = run_path / "data/metrics/fold_failure_metrics.parquet"
    aggregate_metrics_path = run_path / "data/metrics/aggregate_failure_metrics.parquet"
    calibration_path = run_path / "data/calibration/failure_calibration.parquet"
    support_path = run_path / "data/support/fold_class_support.parquet"
    registry_path = run_path / "model_registry.json"
    _atomic_parquet(binary_path, binary_predictions)
    _atomic_parquet(type_path, type_predictions)
    _atomic_parquet(labels_path, labels)
    _atomic_parquet(fold_metrics_path, fold_metrics)
    _atomic_parquet(aggregate_metrics_path, aggregate_metrics)
    _atomic_parquet(calibration_path, calibration)
    _atomic_parquet(support_path, support)
    _atomic_json(registry_path, registry)
    return [
        binary_path,
        type_path,
        labels_path,
        fold_metrics_path,
        aggregate_metrics_path,
        calibration_path,
        support_path,
        registry_path,
    ], registry


def _validate_optional_metrics(frame: pd.DataFrame) -> bool:
    valid = True
    for row in frame.itertuples(index=False):
        diverse = bool(row.binary_discrimination_available)
        roc_finite = row.roc_auc is not None and np.isfinite(row.roc_auc)
        ap_finite = row.average_precision is not None and np.isfinite(row.average_precision)
        if diverse != (roc_finite and ap_finite):
            valid = False
        type_available = str(row.type_metric_status).startswith("available:")
        values = [row.type_log_loss, row.type_brier, row.type_top1_accuracy, row.type_macro_f1]
        type_finite = all(value is not None and np.isfinite(value) for value in values)
        if type_available != type_finite:
            valid = False
    return valid


def _validate_phase9(
    run_path: Path,
    phase6_path: Path,
    phase7_path: Path,
    phase8_path: Path,
    original_input_fingerprints: dict[str, str],
    phase6_validation: dict[str, Any],
    phase7_validation: dict[str, Any],
    phase8_validation: dict[str, Any],
    features: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
    source_labels: pd.DataFrame,
    global_types: list[str],
    calibration_bins: int,
) -> dict[str, Any]:
    _, _, current_phase6 = _validate_phase6_input(phase6_path)
    _, _, current_phase7 = _validate_phase7_input(phase7_path, current_phase6)
    phase6_phase7 = _phase8_input_fingerprints(current_phase6, current_phase7)
    _, _, current_phase8 = _validate_phase8_input(phase8_path, phase6_phase7)
    current_labels, current_types, _runs_path, current_label_sha = _load_failure_labels(
        phase6_path, targets
    )
    current_inputs = _combined_input_fingerprints(
        current_phase6, current_phase7, current_phase8, current_label_sha
    )
    binary = pd.read_parquet(run_path / "data/predictions/oof_failure_probability.parquet")
    type_predictions = pd.read_parquet(
        run_path / "data/predictions/oof_failure_type_probability.parquet"
    )
    saved_labels = pd.read_parquet(run_path / "data/targets/failure_labels.parquet")
    fold_metrics = pd.read_parquet(run_path / "data/metrics/fold_failure_metrics.parquet")
    aggregate = pd.read_parquet(run_path / "data/metrics/aggregate_failure_metrics.parquet")
    calibration = pd.read_parquet(run_path / "data/calibration/failure_calibration.parquet")
    support = pd.read_parquet(run_path / "data/support/fold_class_support.parquet")
    registry = json.loads((run_path / "model_registry.json").read_text(encoding="utf-8"))
    fold_schemas = json.loads(
        (run_path / "data/preprocessing/fold_feature_schemas.json").read_text(encoding="utf-8")
    )
    jobs = _all_jobs(preprocessing)
    expected_binary_rows = len(features) * len(preprocessing["splits"])
    coverage_ok = True
    for split_name in preprocessing["splits"]:
        group = binary.loc[binary["split_name"] == split_name]
        if len(group) != len(features) or set(group["feature_id"]) != set(features["feature_id"]):
            coverage_ok = False
    fold_assignment_ok = True
    for split_name, fold_column in feature_schema["split_columns"].items():
        expected_folds = pd.Series(
            splits[fold_column].to_numpy(dtype=int), index=splits["feature_id"].astype(str)
        )
        observed = binary.loc[binary["split_name"] == split_name]
        mapped = observed["feature_id"].astype(str).map(expected_folds)
        if mapped.isna().any() or not np.array_equal(
            mapped.to_numpy(dtype=int), observed["fold"].to_numpy(dtype=int)
        ):
            fold_assignment_ok = False
    expected_failure = pd.Series(
        targets[FAILURE_TARGET_COLUMN].astype(bool).to_numpy(),
        index=targets["feature_id"].astype(str),
    )
    mapped_failure = binary["feature_id"].astype(str).map(expected_failure)
    target_exact = bool(
        np.array_equal(mapped_failure.to_numpy(dtype=bool), binary["observed_failure"].to_numpy(dtype=bool))
    )
    expected_type_rows = len(binary) * len(global_types)
    if global_types:
        type_keys_unique = not type_predictions.duplicated(
            ["feature_id", "split_name", "failure_type"]
        ).any()
        probability_sums = type_predictions.groupby(
            ["feature_id", "split_name"], sort=False
        )["conditional_probability"].sum()
        joint_sums = type_predictions.groupby(
            ["feature_id", "split_name"], sort=False
        )["joint_probability"].sum()
        binary_index = binary.set_index(["feature_id", "split_name"])["failure_probability"]
        type_distribution_valid = bool(
            len(type_predictions) == expected_type_rows
            and type_keys_unique
            and np.allclose(probability_sums.to_numpy(dtype=float), 1.0)
            and np.allclose(
                joint_sums.sort_index().to_numpy(dtype=float),
                binary_index.sort_index().to_numpy(dtype=float),
            )
        )
    else:
        type_distribution_valid = bool(type_predictions.empty and not binary["observed_failure"].any())
    required_columns = [
        "failure_prevalence",
        "mean_failure_probability",
        "brier",
        "log_loss",
        "ece",
        "calibration_max_abs",
    ]
    required_metrics_finite = bool(
        np.isfinite(fold_metrics[required_columns].to_numpy(dtype=float)).all()
        and np.isfinite(aggregate[required_columns].to_numpy(dtype=float)).all()
    )
    single_class = (fold_metrics["training_positive_count"] == 0) | (
        fold_metrics["training_positive_count"] == fold_metrics["training_row_count"]
    )
    expected_aggregate_rows = len(preprocessing["splits"]) * (
        1 + len(feature_schema["cutoffs"])
    )
    labels_exact = bool(
        saved_labels.equals(source_labels)
        and current_labels.equals(source_labels)
        and current_types == global_types
    )
    checks = {
        "phase6_quality_pass": phase6_validation["status"] == "PHASE_6_PASS",
        "phase7_quality_pass": phase7_validation["status"] == "PHASE_7_PASS",
        "phase8_quality_pass": phase8_validation["status"] == "PHASE_8_PASS",
        "source_inputs_unchanged": current_inputs == original_input_fingerprints,
        "failure_labels_frozen_and_exact": labels_exact,
        "expected_job_count": len(fold_metrics) == len(jobs) == len(registry["jobs"]),
        "all_job_markers_verified": all(
            _verify_job(run_path, _job_id(*job)) is not None for job in jobs
        ),
        "expected_binary_prediction_rows": len(binary) == expected_binary_rows,
        "binary_prediction_keys_unique": not binary.duplicated(
            ["feature_id", "split_name"]
        ).any(),
        "oof_coverage_exact": coverage_ok,
        "fold_assignments_exact": fold_assignment_ok,
        "failure_target_exact": target_exact,
        "failure_probabilities_finite": bool(
            np.isfinite(binary["failure_probability"].to_numpy(dtype=float)).all()
        ),
        "failure_probabilities_bounded": bool(
            binary["failure_probability"].between(0.0, 1.0).all()
        ),
        "conditional_type_distribution_valid": type_distribution_valid,
        "required_metrics_finite": required_metrics_finite,
        "optional_metric_availability_explicit": (
            _validate_optional_metrics(fold_metrics) and _validate_optional_metrics(aggregate)
        ),
        "single_class_failure_handled": bool(
            single_class.empty
            or (fold_metrics.loc[single_class, "binary_fit_status"] == "beta_binomial_fallback").all()
        ),
        "aggregate_metrics_complete": len(aggregate) == expected_aggregate_rows,
        "calibration_rows_complete": len(calibration)
        == expected_aggregate_rows * calibration_bins,
        "class_support_complete": len(support) == len(jobs) * (2 + len(global_types)),
        "fold_preprocessing_contract_complete": len(fold_schemas["folds"]) == len(jobs),
        "phase9_scope_failure_only": registry["targets"]
        == ["failure_probability", "failure_type_given_failure"],
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    failure_count = int(source_labels["observed_failure"].sum())
    if failure_count == 0:
        estimability_status = "NO_OBSERVED_FAILURES"
    elif len(global_types) < 2:
        estimability_status = "SINGLE_FAILURE_TYPE_ONLY"
    else:
        estimability_status = "FAILURE_AND_TYPE_MODELS_ESTIMABLE"
    issues = []
    if failed_checks:
        issues.append({"type": "failed_quality_checks", "checks": failed_checks})
    caveats = []
    if failure_count == 0:
        caveats.append(
            {
                "type": "failure_model_not_estimable",
                "detail": (
                    "The frozen benchmark contains no observed failures; binary folds use a "
                    "Jeffreys-prior beta-binomial fallback and conditional failure types are unavailable."
                ),
            }
        )
    return {
        "status": "PHASE_9_PASS" if all(checks.values()) else "PHASE_9_FAIL",
        "schema_version": PHASE9_SCHEMA_VERSION,
        "scope": "leakage-safe binary failure probability and conditional failure-type distribution",
        "performance_gate_policy": (
            "construction, leakage, probability validity, class-support transparency, and integrity only; "
            "Brier, log loss, ECE, AUROC, AUPRC, and type metrics are not Phase 9 pass thresholds"
        ),
        "phase10_boundary": "runtime and censoring-aware survival modelling are deferred to Phase 10",
        "phase6_directory": str(phase6_path.resolve()),
        "phase7_directory": str(phase7_path.resolve()),
        "phase8_directory": str(phase8_path.resolve()),
        "feature_row_count": int(len(features)),
        "prediction_row_count": int(len(binary)),
        "failure_type_prediction_row_count": int(len(type_predictions)),
        "job_count": int(len(fold_metrics)),
        "model_artifact_count": int(len(registry["jobs"])),
        "aggregate_metric_row_count": int(len(aggregate)),
        "calibration_row_count": int(len(calibration)),
        "class_support_row_count": int(len(support)),
        "observed_failure_count": failure_count,
        "observed_failure_type_count": len(global_types),
        "observed_failure_types": global_types,
        "estimability_status": estimability_status,
        "failure_probability_estimable": bool(0 < failure_count < len(source_labels)),
        "failure_type_estimable": bool(failure_count > 0 and len(global_types) > 1),
        "binary_fit_status_counts": {
            str(key): int(value)
            for key, value in fold_metrics["binary_fit_status"].value_counts().items()
        },
        "type_fit_status_counts": {
            str(key): int(value)
            for key, value in fold_metrics["type_fit_status"].value_counts().items()
        },
        "checks": checks,
        "issues": issues,
        "caveats": caveats,
    }


def run_phase9(
    run_directory: str | Path,
    phase6_directory: str | Path,
    phase7_directory: str | Path,
    phase8_directory: str | Path,
    *,
    master_seed: int = 20_260_825,
    gradient_boosting_iterations: int = 80,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    beta_prior: tuple[float, float] = DEFAULT_BETA_PRIOR,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    phase6_path = Path(phase6_directory)
    phase7_path = Path(phase7_directory)
    phase8_path = Path(phase8_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"
    if gradient_boosting_iterations < 1 or calibration_bins < 2:
        raise ValueError("Phase 9 model iterations must be positive and calibration bins at least two")
    if len(beta_prior) != 2 or any(value <= 0.0 for value in beta_prior):
        raise ValueError("Phase 9 beta prior alpha and beta must be positive")
    phase6_validation, _phase6_manifest, phase6_fingerprints = _validate_phase6_input(
        phase6_path
    )
    phase7_validation, _phase7_manifest, phase7_fingerprints = _validate_phase7_input(
        phase7_path, phase6_fingerprints
    )
    phase6_phase7 = _phase8_input_fingerprints(phase6_fingerprints, phase7_fingerprints)
    phase8_validation, _phase8_manifest, phase8_fingerprints = _validate_phase8_input(
        phase8_path, phase6_phase7
    )
    features, targets, splits, feature_schema, preprocessing = _load_phase6_tables(phase6_path)
    labels, global_types, failure_label_source, failure_label_sha = _load_failure_labels(
        phase6_path, targets
    )
    input_fingerprints = _combined_input_fingerprints(
        phase6_fingerprints,
        phase7_fingerprints,
        phase8_fingerprints,
        failure_label_sha,
    )
    stable_configuration = {
        "phase": 9,
        "schema_version": PHASE9_SCHEMA_VERSION,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "preprocessing_implementation_sha256": _file_sha256(Path(phase7_module.__file__)),
        "input_fingerprints": input_fingerprints,
        "master_seed": int(master_seed),
        "binary_model": BINARY_MODEL_NAME,
        "type_model": TYPE_MODEL_NAME,
        "gradient_boosting_iterations": int(gradient_boosting_iterations),
        "calibration_bins": int(calibration_bins),
        "beta_prior": {"alpha": float(beta_prior[0]), "beta": float(beta_prior[1])},
        "global_failure_types": global_types,
        "split_columns": feature_schema["split_columns"],
        "cutoffs": feature_schema["cutoffs"],
        "single_class_policy": (
            "Jeffreys-prior beta-binomial probability; no synthetic failures or label injection"
        ),
    }
    configuration_sha256 = _configuration_hash(stable_configuration)
    configuration = {
        **stable_configuration,
        "phase6_directory": str(phase6_path.resolve()),
        "phase7_directory": str(phase7_path.resolve()),
        "phase8_directory": str(phase8_path.resolve()),
        "failure_label_source": str(failure_label_source.resolve()),
        "git_commit": current_git_commit(),
        "configuration_sha256": configuration_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Phase 9 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_9_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 9 resume manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest.get("output_sha256", {}).items():
                path = run_path / relative
                if not path.is_file() or _file_sha256(path) != expected_hash:
                    raise ValueError(f"completed Phase 9 output hash mismatch: {relative}")
            active_logger.info("[PHASE9][RESUME] status=PHASE_9_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info(
            "[PHASE9][RESUME] completed_jobs=%s", len(state.get("completed_jobs", {}))
        )
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE9_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": configuration_sha256,
            "input_fingerprints": input_fingerprints,
            "completed_jobs": {},
            "last_completed_job": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase9_started"})
    fold_schemas, job_outputs = _run_jobs(
        run_path,
        features,
        labels,
        splits,
        feature_schema,
        preprocessing,
        global_types,
        state,
        master_seed=master_seed,
        gradient_boosting_iterations=gradient_boosting_iterations,
        calibration_bins=calibration_bins,
        beta_prior=beta_prior,
        progress_path=progress_path,
        logger=active_logger,
    )
    jobs = _all_jobs(preprocessing)
    aggregate_outputs, _registry = _aggregate_job_outputs(
        run_path, jobs, labels, global_types, calibration_bins
    )
    validation = _validate_phase9(
        run_path,
        phase6_path,
        phase7_path,
        phase8_path,
        input_fingerprints,
        phase6_validation,
        phase7_validation,
        phase8_validation,
        features,
        targets,
        splits,
        feature_schema,
        preprocessing,
        labels,
        global_types,
        calibration_bins,
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
        "phase": 9,
        "status": validation["status"],
        "schema_version": PHASE9_SCHEMA_VERSION,
        "input_fingerprints": input_fingerprints,
        "output_sha256": _relative_hashes(run_path, output_paths),
        "row_counts": {
            "features": validation["feature_row_count"],
            "binary_predictions": validation["prediction_row_count"],
            "failure_type_predictions": validation["failure_type_prediction_row_count"],
            "jobs": validation["job_count"],
            "calibration": validation["calibration_row_count"],
            "class_support": validation["class_support_row_count"],
        },
        "estimability_status": validation["estimability_status"],
        "phase10_allowed": validation["status"] == "PHASE_9_PASS",
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
    _append_progress(progress_path, {"event": "phase9_finished", "status": validation["status"]})
    active_logger.info(
        "[PHASE9][SUMMARY] status=%s estimability=%s failures=%s jobs=%s predictions=%s directory=%s",
        validation["status"],
        validation["estimability_status"],
        validation["observed_failure_count"],
        validation["job_count"],
        validation["prediction_row_count"],
        run_path,
    )
    return validation
