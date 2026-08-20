from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import pickle
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.schema import canonical_json
from uriel_v2.provenance import current_git_commit


PHASE7_SCHEMA_VERSION = "phase7-v1"
MODEL_NAMES = ("linear", "random_forest", "gradient_boosting")
TARGETS = {
    "quality": {
        "column": "target_quality_final",
        "task": "regression",
        "transform": "identity",
    },
    "runtime": {
        "column": "target_runtime",
        "task": "regression",
        "transform": "log1p",
    },
    "failure": {
        "column": "target_failure",
        "task": "classification",
        "transform": "identity",
    },
}


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


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as compressed:
            pickle.dump(value, compressed, protocol=pickle.HIGHEST_PROTOCOL)
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


def _configuration_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _relative_hashes(run_path: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.relative_to(run_path)): _file_sha256(path) for path in paths}


def _finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _problem_ids_sha256(values: Iterable[Any]) -> str:
    problem_ids = sorted({str(value) for value in values})
    return hashlib.sha256("\n".join(problem_ids).encode("utf-8")).hexdigest()


def _phase6_required_paths(phase6_path: Path) -> dict[str, Path]:
    return {
        "phase6_config": phase6_path / "config.json",
        "phase6_manifest": phase6_path / "manifest.json",
        "phase6_validation": phase6_path / "validation.json",
        "phase6_feature_schema": phase6_path / "feature_schema.json",
        "phase6_features": phase6_path / "data/features/model_features.parquet",
        "phase6_targets": phase6_path / "data/targets/model_targets.parquet",
        "phase6_splits": phase6_path / "data/splits/model_splits.parquet",
        "phase6_preprocessing": phase6_path / "data/preprocessing/preprocessing_specs.json",
    }


def _validate_phase6_input(phase6_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    required = _phase6_required_paths(phase6_path)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 7 input files are missing: {missing[:5]}")
    validation = json.loads(required["phase6_validation"].read_text(encoding="utf-8"))
    manifest = json.loads(required["phase6_manifest"].read_text(encoding="utf-8"))
    if validation.get("status") != "PHASE_6_PASS":
        raise ValueError("Phase 7 requires a PHASE_6_PASS feature dataset")
    if manifest.get("status") != "PHASE_6_PASS" or not manifest.get("phase7_allowed"):
        raise ValueError("Phase 6 manifest does not allow Phase 7")
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = phase6_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 6 manifest output hash mismatch: {relative}")
    fingerprints = {name: _file_sha256(path) for name, path in sorted(required.items())}
    return validation, manifest, fingerprints


def _load_phase6_tables(
    phase6_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    features = pd.read_parquet(phase6_path / "data/features/model_features.parquet")
    targets = pd.read_parquet(phase6_path / "data/targets/model_targets.parquet")
    splits = pd.read_parquet(phase6_path / "data/splits/model_splits.parquet")
    feature_schema = json.loads((phase6_path / "feature_schema.json").read_text(encoding="utf-8"))
    preprocessing = json.loads(
        (phase6_path / "data/preprocessing/preprocessing_specs.json").read_text(encoding="utf-8")
    )
    for name, frame in (("features", features), ("targets", targets), ("splits", splits)):
        if "feature_id" not in frame or not frame["feature_id"].is_unique:
            raise ValueError(f"Phase 6 {name} must have unique feature_id values")
    feature_ids = features["feature_id"].astype(str)
    if set(feature_ids) != set(targets["feature_id"].astype(str)):
        raise ValueError("Phase 6 feature/target coverage mismatch")
    if set(feature_ids) != set(splits["feature_id"].astype(str)):
        raise ValueError("Phase 6 feature/split coverage mismatch")
    targets = targets.set_index("feature_id").loc[feature_ids].reset_index()
    splits = splits.set_index("feature_id").loc[feature_ids].reset_index()
    for column in ("run_id", "problem_id", "cutoff"):
        if not features[column].reset_index(drop=True).equals(targets[column].reset_index(drop=True)):
            raise ValueError(f"Phase 6 feature/target identifier mismatch: {column}")
        if not features[column].reset_index(drop=True).equals(splits[column].reset_index(drop=True)):
            raise ValueError(f"Phase 6 feature/split identifier mismatch: {column}")
    return features, targets, splits, feature_schema, preprocessing


def _transform_features(
    frame: pd.DataFrame,
    fold_specification: dict[str, Any],
    feature_schema: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for column in feature_schema["numeric_feature_columns"]:
        statistics = fold_specification["numeric"][column]
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        values[missing] = float(statistics["impute_median"])
        lower = statistics.get("q01")
        upper = statistics.get("q99")
        if lower is not None and upper is not None:
            values = np.clip(values, float(lower), float(upper))
        values = (values - float(statistics["mean"])) / float(statistics["scale_std"])
        arrays.append(values.astype(np.float32, copy=False).reshape(-1, 1))
        names.append(column)
        if statistics["add_missing_indicator"]:
            arrays.append(missing.astype(np.float32).reshape(-1, 1))
            names.append(f"{column}__missing")
    for column in feature_schema["categorical_feature_columns"]:
        vocabulary = list(fold_specification["categorical"][column]["vocabulary"])
        known = set(vocabulary) - {"__UNKNOWN__"}
        values = frame[column].fillna("__MISSING__").astype(str).to_numpy()
        mapped = np.asarray(
            [value if value in known else "__UNKNOWN__" for value in values],
            dtype=object,
        )
        for category in vocabulary:
            arrays.append((mapped == category).astype(np.float32).reshape(-1, 1))
            names.append(f"{column}=={category}")
    transformed = np.column_stack(arrays).astype(np.float32, copy=False)
    if not np.isfinite(transformed).all():
        raise ValueError("Phase 7 preprocessing produced non-finite values")
    return transformed, names


def _verify_fold_contract(
    features: pd.DataFrame,
    splits: pd.DataFrame,
    fold_column: str,
    fold: int,
    fold_specification: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    validation_mask = splits[fold_column].to_numpy(dtype=int) == fold
    training_mask = ~validation_mask
    train_problem_ids = features.loc[training_mask, "problem_id"].astype(str)
    validation_problem_ids = features.loc[validation_mask, "problem_id"].astype(str)
    overlap = set(train_problem_ids) & set(validation_problem_ids)
    if overlap:
        raise ValueError(f"Phase 7 split leakage: {fold_column} fold {fold}")
    observed = {
        "training_row_count": int(training_mask.sum()),
        "validation_row_count": int(validation_mask.sum()),
        "training_problem_count": int(train_problem_ids.nunique()),
        "validation_problem_count": int(validation_problem_ids.nunique()),
        "training_problem_ids_sha256": _problem_ids_sha256(train_problem_ids),
        "validation_problem_ids_sha256": _problem_ids_sha256(validation_problem_ids),
    }
    for key, value in observed.items():
        if value != fold_specification[key]:
            raise ValueError(
                f"Phase 7 fold contract mismatch: {fold_column} fold {fold} {key}"
            )
    return training_mask, validation_mask, observed


def _job_seed(master_seed: int, job_id: str) -> int:
    offset = int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8], 16)
    return int((master_seed + offset) % (2**31 - 1))


def _regression_estimator(
    model_name: str,
    seed: int,
    random_forest_estimators: int,
    gradient_boosting_iterations: int,
) -> Any:
    if model_name == "linear":
        return Ridge(alpha=1.0, solver="lsqr")
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=random_forest_estimators,
            max_depth=12,
            min_samples_leaf=20,
            max_features=0.75,
            n_jobs=1,
            random_state=seed,
        )
    if model_name == "gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=gradient_boosting_iterations,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"Unsupported Phase 7 model: {model_name}")


def _classification_estimator(
    model_name: str,
    seed: int,
    random_forest_estimators: int,
    gradient_boosting_iterations: int,
) -> Any:
    if model_name == "linear":
        return LogisticRegression(max_iter=1_000, solver="lbfgs", random_state=seed)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=random_forest_estimators,
            max_depth=12,
            min_samples_leaf=20,
            max_features=0.75,
            n_jobs=1,
            random_state=seed,
        )
    if model_name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=gradient_boosting_iterations,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"Unsupported Phase 7 model: {model_name}")


def _fit_predict(
    model_name: str,
    target_name: str,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    random_forest_estimators: int,
    gradient_boosting_iterations: int,
) -> tuple[Any | None, np.ndarray, np.ndarray, str]:
    target_specification = TARGETS[target_name]
    if target_specification["task"] == "classification":
        labels = y_train.astype(int)
        prevalence = float(labels.mean())
        reference = np.full(len(x_validation), prevalence, dtype=float)
        if np.unique(labels).size < 2:
            return None, reference.copy(), reference, "constant_fallback"
        estimator = _classification_estimator(
            model_name,
            seed,
            random_forest_estimators,
            gradient_boosting_iterations,
        )
        estimator.fit(x_train, labels)
        classes = list(estimator.classes_)
        positive_index = classes.index(1)
        prediction = estimator.predict_proba(x_validation)[:, positive_index]
        return estimator, np.clip(prediction, 0.0, 1.0), reference, "fitted"

    raw_targets = y_train.astype(float)
    fit_targets = (
        np.log1p(raw_targets)
        if target_specification["transform"] == "log1p"
        else raw_targets
    )
    estimator = _regression_estimator(
        model_name,
        seed,
        random_forest_estimators,
        gradient_boosting_iterations,
    )
    estimator.fit(x_train, fit_targets)
    prediction = np.asarray(estimator.predict(x_validation), dtype=float)
    if target_specification["transform"] == "log1p":
        prediction = np.maximum(np.expm1(prediction), 0.0)
    elif target_name == "quality":
        prediction = np.clip(prediction, 0.0, 1.0)
    reference = np.full(len(x_validation), float(raw_targets.mean()), dtype=float)
    return estimator, prediction, reference, "fitted"


def _safe_skill(loss: float, reference_loss: float) -> float | None:
    if not math.isfinite(reference_loss) or reference_loss <= 0.0:
        return None
    return _finite_or_none(1.0 - loss / reference_loss)


def _regression_metrics(
    observed: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    mae = float(mean_absolute_error(observed, prediction))
    rmse = float(math.sqrt(mean_squared_error(observed, prediction)))
    reference_mae = float(mean_absolute_error(observed, reference))
    reference_rmse = float(math.sqrt(mean_squared_error(observed, reference)))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": _finite_or_none(r2_score(observed, prediction)),
        "reference_mae": reference_mae,
        "reference_rmse": reference_rmse,
        "mae_skill_vs_train_mean": _safe_skill(mae, reference_mae),
        "rmse_skill_vs_train_mean": _safe_skill(rmse, reference_rmse),
        "brier": None,
        "log_loss": None,
        "roc_auc": None,
        "average_precision": None,
        "reference_brier": None,
        "brier_skill_vs_prevalence": None,
        "discrimination_status": "not_applicable",
    }


def _classification_metrics(
    observed: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    labels = observed.astype(int)
    probabilities = np.clip(prediction.astype(float), 1e-15, 1.0 - 1e-15)
    reference_probabilities = np.clip(reference.astype(float), 1e-15, 1.0 - 1e-15)
    brier = float(brier_score_loss(labels, probabilities))
    reference_brier = float(brier_score_loss(labels, reference_probabilities))
    has_two_classes = np.unique(labels).size == 2
    return {
        "mae": None,
        "rmse": None,
        "r2": None,
        "reference_mae": None,
        "reference_rmse": None,
        "mae_skill_vs_train_mean": None,
        "rmse_skill_vs_train_mean": None,
        "brier": brier,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_two_classes else None,
        "average_precision": (
            float(average_precision_score(labels, probabilities)) if has_two_classes else None
        ),
        "reference_brier": reference_brier,
        "brier_skill_vs_prevalence": _safe_skill(brier, reference_brier),
        "discrimination_status": "estimated" if has_two_classes else "single_class_unavailable",
    }


def _metrics(
    target_name: str,
    observed: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    if TARGETS[target_name]["task"] == "classification":
        return _classification_metrics(observed, prediction, reference)
    return _regression_metrics(observed, prediction, reference)


def _job_id(split_name: str, fold: int, target_name: str, model_name: str) -> str:
    return f"{split_name}__fold{fold}__{target_name}__{model_name}"


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
            raise ValueError(f"Phase 7 job output hash mismatch: job={job_id} path={relative}")
    return marker


def _all_jobs(
    preprocessing: dict[str, Any],
    model_names: tuple[str, ...],
) -> list[tuple[str, int, str, str]]:
    jobs = []
    for split_name in sorted(preprocessing["splits"]):
        folds = sorted(int(value) for value in preprocessing["splits"][split_name])
        for fold in folds:
            for target_name in TARGETS:
                for model_name in model_names:
                    jobs.append((split_name, fold, target_name, model_name))
    return jobs


def _run_baseline_jobs(
    run_path: Path,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
    state: dict[str, Any],
    *,
    model_names: tuple[str, ...],
    master_seed: int,
    random_forest_estimators: int,
    gradient_boosting_iterations: int,
    progress_path: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[Path]]:
    fold_schemas: dict[str, Any] = {
        "schema_version": PHASE7_SCHEMA_VERSION,
        "source_preprocessing_schema_version": preprocessing["schema_version"],
        "transform_policy": (
            "training-fold median imputation, q01/q99 clipping, mean/std scaling, "
            "missing indicators, and train-vocabulary one-hot encoding"
        ),
        "folds": {},
    }
    output_paths: list[Path] = []
    for split_name in sorted(preprocessing["splits"]):
        fold_column = feature_schema["split_columns"][split_name]
        folds = sorted(int(value) for value in preprocessing["splits"][split_name])
        for fold in folds:
            fold_key = f"{split_name}/fold{fold}"
            fold_specification = preprocessing["splits"][split_name][str(fold)]
            training_mask, validation_mask, fold_contract = _verify_fold_contract(
                features,
                splits,
                fold_column,
                fold,
                fold_specification,
            )
            logger.info(
                "[PHASE7][FOLD] split=%s fold=%s train_rows=%s validation_rows=%s",
                split_name,
                fold,
                int(training_mask.sum()),
                int(validation_mask.sum()),
            )
            x_train, feature_names = _transform_features(
                features.loc[training_mask], fold_specification, feature_schema
            )
            x_validation, validation_feature_names = _transform_features(
                features.loc[validation_mask], fold_specification, feature_schema
            )
            if feature_names != validation_feature_names:
                raise ValueError(f"Phase 7 transformed schema mismatch: {fold_key}")
            fold_schemas["folds"][fold_key] = {
                **fold_contract,
                "fold_column": fold_column,
                "transformed_feature_count": len(feature_names),
                "transformed_feature_names": feature_names,
                "transformed_feature_names_sha256": hashlib.sha256(
                    "\n".join(feature_names).encode("utf-8")
                ).hexdigest(),
            }
            for target_name, target_specification in TARGETS.items():
                target_column = target_specification["column"]
                raw_targets = targets[target_column]
                if target_specification["task"] == "classification":
                    target_values = raw_targets.astype(int).to_numpy()
                else:
                    target_values = pd.to_numeric(raw_targets, errors="coerce").to_numpy(dtype=float)
                    if not np.isfinite(target_values).all():
                        raise ValueError(f"Phase 7 target contains non-finite values: {target_name}")
                y_train = target_values[training_mask]
                y_validation = target_values[validation_mask]
                for model_name in model_names:
                    job_id = _job_id(split_name, fold, target_name, model_name)
                    paths = _job_paths(run_path, job_id)
                    marker = _verify_job(run_path, job_id)
                    if marker is not None:
                        state.setdefault("completed_jobs", {})[job_id] = _file_sha256(paths["marker"])
                        output_paths.extend(paths.values())
                        logger.info("[PHASE7][RESUME] job=%s status=SKIP_VERIFIED", job_id)
                        continue
                    seed = _job_seed(master_seed, job_id)
                    estimator, prediction, reference, fit_status = _fit_predict(
                        model_name,
                        target_name,
                        x_train,
                        x_validation,
                        y_train,
                        seed=seed,
                        random_forest_estimators=random_forest_estimators,
                        gradient_boosting_iterations=gradient_boosting_iterations,
                    )
                    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
                        raise ValueError(f"Phase 7 model produced non-finite predictions: {job_id}")
                    validation_rows = features.loc[
                        validation_mask,
                        ["feature_id", "cutoff"],
                    ].reset_index(drop=True)
                    prediction_frame = validation_rows.assign(
                        split_name=split_name,
                        fold=fold,
                        target=target_name,
                        model=model_name,
                        observed=y_validation.astype(float),
                        prediction=prediction,
                        reference_prediction=reference,
                        fit_status=fit_status,
                    )
                    metric_row = {
                        "job_id": job_id,
                        "split_name": split_name,
                        "fold": fold,
                        "target": target_name,
                        "target_column": target_column,
                        "task": target_specification["task"],
                        "target_transform": target_specification["transform"],
                        "model": model_name,
                        "fit_status": fit_status,
                        "seed": seed,
                        "training_row_count": int(training_mask.sum()),
                        "validation_row_count": int(validation_mask.sum()),
                        "training_problem_count": fold_contract["training_problem_count"],
                        "validation_problem_count": fold_contract["validation_problem_count"],
                        "training_positive_count": (
                            int(y_train.sum())
                            if target_specification["task"] == "classification"
                            else None
                        ),
                        "validation_positive_count": (
                            int(y_validation.sum())
                            if target_specification["task"] == "classification"
                            else None
                        ),
                        **_metrics(target_name, y_validation, prediction, reference),
                    }
                    model_artifact = {
                        "schema_version": PHASE7_SCHEMA_VERSION,
                        "job_id": job_id,
                        "split_name": split_name,
                        "fold": fold,
                        "target": target_name,
                        "target_specification": target_specification,
                        "model": model_name,
                        "fit_status": fit_status,
                        "seed": seed,
                        "feature_names": feature_names,
                        "fold_contract": fold_contract,
                        "estimator": estimator,
                        "constant_probability": (
                            float(reference[0]) if fit_status == "constant_fallback" else None
                        ),
                    }
                    _atomic_parquet(paths["predictions"], prediction_frame)
                    _atomic_json(paths["metrics"], metric_row)
                    _atomic_pickle(paths["model"], model_artifact)
                    marker = {
                        "schema_version": PHASE7_SCHEMA_VERSION,
                        "job_id": job_id,
                        "split_name": split_name,
                        "fold": fold,
                        "target": target_name,
                        "model": model_name,
                        "fit_status": fit_status,
                        "output_sha256": _relative_hashes(
                            run_path,
                            (paths["predictions"], paths["metrics"], paths["model"]),
                        ),
                    }
                    _atomic_json(paths["marker"], marker)
                    state.setdefault("completed_jobs", {})[job_id] = _file_sha256(paths["marker"])
                    state["last_completed_job"] = job_id
                    _atomic_json(run_path / "run_state.json", state)
                    _append_progress(
                        progress_path,
                        {"event": "job_completed", "job_id": job_id, "fit_status": fit_status},
                    )
                    output_paths.extend(paths.values())
                    logger.info("[PHASE7][JOB] job=%s status=%s", job_id, fit_status)
    fold_schema_path = run_path / "data/preprocessing/fold_feature_schemas.json"
    _atomic_json(fold_schema_path, fold_schemas)
    output_paths.append(fold_schema_path)
    return fold_schemas, output_paths


def _aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["split_name", "target", "model"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        split_name, target_name, model_name = keys
        rows.append(
            {
                "scope": "overall",
                "split_name": split_name,
                "target": target_name,
                "model": model_name,
                "cutoff": None,
                "row_count": int(len(group)),
                "fold_count": int(group["fold"].nunique()),
                **_metrics(
                    target_name,
                    group["observed"].to_numpy(dtype=float),
                    group["prediction"].to_numpy(dtype=float),
                    group["reference_prediction"].to_numpy(dtype=float),
                ),
            }
        )
    cutoff_group_columns = [*group_columns, "cutoff"]
    for keys, group in predictions.groupby(cutoff_group_columns, sort=True):
        split_name, target_name, model_name, cutoff = keys
        rows.append(
            {
                "scope": "cutoff",
                "split_name": split_name,
                "target": target_name,
                "model": model_name,
                "cutoff": float(cutoff),
                "row_count": int(len(group)),
                "fold_count": int(group["fold"].nunique()),
                **_metrics(
                    target_name,
                    group["observed"].to_numpy(dtype=float),
                    group["prediction"].to_numpy(dtype=float),
                    group["reference_prediction"].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["scope", "split_name", "target", "model", "cutoff"],
        na_position="first",
    ).reset_index(drop=True)


def _aggregate_job_outputs(
    run_path: Path,
    jobs: list[tuple[str, int, str, str]],
) -> tuple[list[Path], dict[str, Any]]:
    prediction_frames = []
    metric_rows = []
    registry = {
        "schema_version": PHASE7_SCHEMA_VERSION,
        "artifact_policy": "one frozen estimator artifact per split/fold/target/model job",
        "jobs": [],
    }
    for split_name, fold, target_name, model_name in jobs:
        job_id = _job_id(split_name, fold, target_name, model_name)
        marker = _verify_job(run_path, job_id)
        if marker is None:
            raise ValueError(f"Phase 7 job is incomplete: {job_id}")
        paths = _job_paths(run_path, job_id)
        prediction_frames.append(pd.read_parquet(paths["predictions"]))
        metric_rows.append(json.loads(paths["metrics"].read_text(encoding="utf-8")))
        registry["jobs"].append(
            {
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "target": target_name,
                "model": model_name,
                "fit_status": marker["fit_status"],
                "artifact": str(paths["model"].relative_to(run_path)),
                "artifact_sha256": marker["output_sha256"][str(paths["model"].relative_to(run_path))],
            }
        )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["split_name", "target", "model", "feature_id"]
    ).reset_index(drop=True)
    fold_metrics = pd.DataFrame(metric_rows).sort_values(
        ["split_name", "fold", "target", "model"]
    ).reset_index(drop=True)
    aggregate_metrics = _aggregate_metrics(predictions)
    predictions_path = run_path / "data/predictions/oof_predictions.parquet"
    fold_metrics_path = run_path / "data/metrics/fold_metrics.parquet"
    aggregate_metrics_path = run_path / "data/metrics/aggregate_metrics.parquet"
    registry_path = run_path / "model_registry.json"
    _atomic_parquet(predictions_path, predictions)
    _atomic_parquet(fold_metrics_path, fold_metrics)
    _atomic_parquet(aggregate_metrics_path, aggregate_metrics)
    _atomic_json(registry_path, registry)
    return [predictions_path, fold_metrics_path, aggregate_metrics_path, registry_path], registry


def _validate_phase7(
    run_path: Path,
    phase6_path: Path,
    original_input_fingerprints: dict[str, str],
    phase6_validation: dict[str, Any],
    features: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
    model_names: tuple[str, ...],
) -> dict[str, Any]:
    current_input_fingerprints = {
        name: _file_sha256(path)
        for name, path in sorted(_phase6_required_paths(phase6_path).items())
    }
    predictions = pd.read_parquet(run_path / "data/predictions/oof_predictions.parquet")
    fold_metrics = pd.read_parquet(run_path / "data/metrics/fold_metrics.parquet")
    aggregate_metrics = pd.read_parquet(run_path / "data/metrics/aggregate_metrics.parquet")
    registry = json.loads((run_path / "model_registry.json").read_text(encoding="utf-8"))
    fold_schemas = json.loads(
        (run_path / "data/preprocessing/fold_feature_schemas.json").read_text(encoding="utf-8")
    )
    jobs = _all_jobs(preprocessing, model_names)
    expected_prediction_rows = len(features) * len(preprocessing["splits"]) * len(TARGETS) * len(model_names)
    key_columns = ["feature_id", "split_name", "target", "model"]
    coverage_ok = True
    for split_name in preprocessing["splits"]:
        for target_name in TARGETS:
            for model_name in model_names:
                group = predictions.loc[
                    (predictions["split_name"] == split_name)
                    & (predictions["target"] == target_name)
                    & (predictions["model"] == model_name)
                ]
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
    numeric_prediction_columns = ["observed", "prediction", "reference_prediction"]
    numeric_predictions_finite = bool(
        np.isfinite(predictions[numeric_prediction_columns].to_numpy(dtype=float)).all()
    )
    quality_prediction = predictions.loc[predictions["target"] == "quality", "prediction"]
    runtime_prediction = predictions.loc[predictions["target"] == "runtime", "prediction"]
    failure_prediction = predictions.loc[predictions["target"] == "failure", "prediction"]
    regression_metrics_finite = bool(
        np.isfinite(
            fold_metrics.loc[fold_metrics["task"] == "regression", ["mae", "rmse"]].to_numpy(dtype=float)
        ).all()
    )
    classification_metrics_finite = bool(
        np.isfinite(
            fold_metrics.loc[
                fold_metrics["task"] == "classification", ["brier", "log_loss"]
            ].to_numpy(dtype=float)
        ).all()
    )
    classification_rows = fold_metrics.loc[fold_metrics["task"] == "classification"]
    single_class_rows = classification_rows.loc[
        (classification_rows["training_positive_count"] == 0)
        | (
            classification_rows["training_positive_count"]
            == classification_rows["training_row_count"]
        )
    ]
    checks = {
        "phase6_quality_pass": phase6_validation["status"] == "PHASE_6_PASS",
        "source_inputs_unchanged": current_input_fingerprints == original_input_fingerprints,
        "expected_job_count": len(fold_metrics) == len(jobs) == len(registry["jobs"]),
        "all_job_markers_verified": all(
            _verify_job(run_path, _job_id(*job)) is not None for job in jobs
        ),
        "expected_prediction_rows": len(predictions) == expected_prediction_rows,
        "prediction_keys_unique": not predictions.duplicated(key_columns).any(),
        "oof_coverage_exact": coverage_ok,
        "fold_assignments_exact": fold_assignment_ok,
        "prediction_values_finite": numeric_predictions_finite,
        "quality_predictions_bounded": bool(quality_prediction.between(0.0, 1.0).all()),
        "runtime_predictions_nonnegative": bool((runtime_prediction >= 0.0).all()),
        "failure_probabilities_bounded": bool(failure_prediction.between(0.0, 1.0).all()),
        "required_fold_metrics_finite": regression_metrics_finite and classification_metrics_finite,
        "single_class_failure_handled": bool(
            single_class_rows.empty
            or (single_class_rows["fit_status"] == "constant_fallback").all()
        ),
        "fold_preprocessing_contract_complete": len(fold_schemas["folds"])
        == sum(len(values) for values in preprocessing["splits"].values()),
        "aggregate_metrics_complete": len(aggregate_metrics)
        == len(preprocessing["splits"]) * len(TARGETS) * len(model_names) * (1 + len(feature_schema["cutoffs"])),
    }
    issues = []
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    if failed_checks:
        issues.append({"type": "failed_quality_checks", "checks": failed_checks})
    return {
        "status": "PHASE_7_PASS" if all(checks.values()) else "PHASE_7_FAIL",
        "schema_version": PHASE7_SCHEMA_VERSION,
        "scope": "leakage-safe LR/RF/GBM point-prediction baselines",
        "performance_gate_policy": (
            "construction and integrity only; predictive adequacy is intentionally not a Phase 7 pass gate"
        ),
        "phase6_directory": str(phase6_path.resolve()),
        "feature_row_count": int(len(features)),
        "prediction_row_count": int(len(predictions)),
        "job_count": int(len(fold_metrics)),
        "model_artifact_count": int(len(registry["jobs"])),
        "split_count": len(preprocessing["splits"]),
        "fold_count": sum(len(values) for values in preprocessing["splits"].values()),
        "models": list(model_names),
        "targets": list(TARGETS),
        "cutoffs": feature_schema["cutoffs"],
        "fit_status_counts": {
            str(key): int(value) for key, value in fold_metrics["fit_status"].value_counts().items()
        },
        "failure_target_positive_count": int(
            predictions.loc[
                (predictions["target"] == "failure")
                & (predictions["split_name"] == sorted(preprocessing["splits"])[0])
                & (predictions["model"] == model_names[0]),
                "observed",
            ].sum()
        ),
        "checks": checks,
        "issues": issues,
    }


def run_phase7(
    run_directory: str | Path,
    phase6_directory: str | Path,
    *,
    master_seed: int = 20_260_823,
    random_forest_estimators: int = 48,
    gradient_boosting_iterations: int = 100,
    model_names: tuple[str, ...] = MODEL_NAMES,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    phase6_path = Path(phase6_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"
    if random_forest_estimators < 1 or gradient_boosting_iterations < 1:
        raise ValueError("Phase 7 model iteration counts must be positive")
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("Phase 7 model_names must be unique and non-empty")
    unsupported_models = sorted(set(model_names) - set(MODEL_NAMES))
    if unsupported_models:
        raise ValueError(f"Unsupported Phase 7 models: {unsupported_models}")
    phase6_validation, _phase6_manifest, input_fingerprints = _validate_phase6_input(phase6_path)
    features, targets, splits, feature_schema, preprocessing = _load_phase6_tables(phase6_path)
    stable_configuration = {
        "phase": 7,
        "schema_version": PHASE7_SCHEMA_VERSION,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "input_fingerprints": input_fingerprints,
        "master_seed": int(master_seed),
        "models": list(model_names),
        "targets": TARGETS,
        "random_forest_estimators": int(random_forest_estimators),
        "gradient_boosting_iterations": int(gradient_boosting_iterations),
        "split_columns": feature_schema["split_columns"],
        "cutoffs": feature_schema["cutoffs"],
    }
    configuration_sha256 = _configuration_hash(stable_configuration)
    configuration = {
        **stable_configuration,
        "phase6_directory": str(phase6_path.resolve()),
        "git_commit": current_git_commit(),
        "configuration_sha256": configuration_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Phase 7 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_7_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 7 resume manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest.get("output_sha256", {}).items():
                path = run_path / relative
                if not path.is_file() or _file_sha256(path) != expected_hash:
                    raise ValueError(f"completed Phase 7 output hash mismatch: {relative}")
            active_logger.info("[PHASE7][RESUME] status=PHASE_7_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info(
            "[PHASE7][RESUME] completed_jobs=%s",
            len(state.get("completed_jobs", {})),
        )
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE7_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": configuration_sha256,
            "input_fingerprints": input_fingerprints,
            "completed_jobs": {},
            "last_completed_job": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase7_started"})
    fold_schemas, job_outputs = _run_baseline_jobs(
        run_path,
        features,
        targets,
        splits,
        feature_schema,
        preprocessing,
        state,
        model_names=model_names,
        master_seed=master_seed,
        random_forest_estimators=random_forest_estimators,
        gradient_boosting_iterations=gradient_boosting_iterations,
        progress_path=progress_path,
        logger=active_logger,
    )
    jobs = _all_jobs(preprocessing, model_names)
    aggregate_outputs, _registry = _aggregate_job_outputs(run_path, jobs)
    validation = _validate_phase7(
        run_path,
        phase6_path,
        input_fingerprints,
        phase6_validation,
        features,
        splits,
        feature_schema,
        preprocessing,
        model_names,
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
        "phase": 7,
        "status": validation["status"],
        "schema_version": PHASE7_SCHEMA_VERSION,
        "input_fingerprints": input_fingerprints,
        "output_sha256": _relative_hashes(run_path, output_paths),
        "row_counts": {
            "features": validation["feature_row_count"],
            "predictions": validation["prediction_row_count"],
            "jobs": validation["job_count"],
        },
        "phase8_allowed": validation["status"] == "PHASE_7_PASS",
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
    _append_progress(progress_path, {"event": "phase7_finished", "status": validation["status"]})
    active_logger.info(
        "[PHASE7][SUMMARY] status=%s jobs=%s predictions=%s artifacts=%s directory=%s",
        validation["status"],
        validation["job_count"],
        validation["prediction_row_count"],
        validation["model_artifact_count"],
        run_path,
    )
    return validation
