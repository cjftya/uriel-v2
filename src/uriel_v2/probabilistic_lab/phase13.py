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
from scipy.stats import kstest, multivariate_normal, norm, rankdata
from sklearn.linear_model import LogisticRegression

import uriel_v2.probabilistic_lab.phase12 as phase12_module
from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.phase7 import (
    _append_progress,
    _atomic_json,
    _atomic_parquet,
    _atomic_pickle,
    _configuration_hash,
    _file_sha256,
    _relative_hashes,
    _verify_fold_contract,
)
from uriel_v2.probabilistic_lab.phase8 import (
    QUANTILE_LEVELS,
    _cdf_at,
    _crps as _quality_crps,
    _distribution_metrics,
    _piecewise_nll,
    _predictive_moments,
)
from uriel_v2.probabilistic_lab.phase9 import _binary_metrics
from uriel_v2.probabilistic_lab.phase10 import (
    SURVIVAL_HORIZONS,
    _horizon_observation,
    _runtime_cdf,
    _runtime_crps,
    _runtime_metrics,
    _runtime_moments,
    _runtime_nll,
    _runtime_support,
    _survival_metrics,
    _survival_summaries,
)
from uriel_v2.probabilistic_lab.phase12 import (
    _all_jobs,
    _frames_equal,
    _load_inputs as _load_phase12_inputs,
    _ordered_predictions,
)
from uriel_v2.provenance import current_git_commit


PHASE13_SCHEMA_VERSION = "phase13-v1"
JOINT_MODEL_NAME = "cross_fitted_recalibrated_gaussian_copula"
DEFAULT_MASTER_SEED = 20_260_828
DEFAULT_CALIBRATION_STRENGTH = 200.0
DEFAULT_MINIMUM_CLASS_ROWS = 20
DEFAULT_COPULA_SHRINKAGE = 200.0
DEFAULT_BETA_PRIOR = (0.5, 0.5)
DEFAULT_CALIBRATION_BINS = 10
JOINT_QUALITY_THRESHOLDS = (0.75, 0.90)
PROBABILITY_EPSILON = 1e-9


def _quantile_suffix(level: float) -> str:
    return f"q{int(round(level * 100.0)):02d}"


def _horizon_suffix(horizon: float) -> str:
    return f"p{int(round(horizon * 100.0)):03d}"


def _threshold_suffix(threshold: float) -> str:
    return f"q{int(round(threshold * 100.0)):03d}"


def _phase12_required_paths(phase12_path: Path) -> dict[str, Path]:
    return {
        "phase12_config": phase12_path / "config.json",
        "phase12_manifest": phase12_path / "manifest.json",
        "phase12_validation": phase12_path / "validation.json",
        "phase12_predictions": phase12_path
        / "data/predictions/oof_mixture_predictions.parquet",
        "phase12_labels": phase12_path / "data/targets/mixture_labels.parquet",
        "phase12_fold_metrics": phase12_path
        / "data/metrics/fold_mixture_metrics.parquet",
        "phase12_aggregate_metrics": phase12_path
        / "data/metrics/aggregate_mixture_metrics.parquet",
        "phase12_support": phase12_path / "data/support/expert_gate_support.parquet",
        "phase12_fold_schemas": phase12_path
        / "data/preprocessing/fold_feature_schemas.json",
        "phase12_model_registry": phase12_path / "model_registry.json",
    }


def _validate_phase12_input(
    phase12_path: Path,
    expected_input_fingerprints: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    required = _phase12_required_paths(phase12_path)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 13 Phase 12 input files are missing: {missing[:5]}")
    validation = json.loads(required["phase12_validation"].read_text(encoding="utf-8"))
    manifest = json.loads(required["phase12_manifest"].read_text(encoding="utf-8"))
    if validation.get("status") != "PHASE_12_PASS":
        raise ValueError("Phase 13 requires a PHASE_12_PASS Mixture-of-Experts run")
    if manifest.get("status") != "PHASE_12_PASS" or not manifest.get("phase13_allowed"):
        raise ValueError("Phase 12 manifest does not allow Phase 13")
    if validation.get("configuration", {}).get("input_fingerprints") != expected_input_fingerprints:
        raise ValueError("Phase 12 was not built from the supplied Phase 6 through 11 inputs")
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = phase12_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 12 manifest output hash mismatch: {relative}")
    fingerprints = {name: _file_sha256(path) for name, path in sorted(required.items())}
    return validation, manifest, fingerprints


def _combined_input_fingerprints(
    phase12_expected_inputs: dict[str, str],
    phase12_fingerprints: dict[str, str],
) -> dict[str, str]:
    return {
        **phase12_expected_inputs,
        **{f"phase12/{name}": value for name, value in phase12_fingerprints.items()},
    }


def _load_inputs(
    phase6_path: Path,
    phase7_path: Path,
    phase8_path: Path,
    phase9_path: Path,
    phase10_path: Path,
    phase11_path: Path,
    phase12_path: Path,
) -> dict[str, Any]:
    phase12_source = _load_phase12_inputs(
        phase6_path,
        phase7_path,
        phase8_path,
        phase9_path,
        phase10_path,
        phase11_path,
    )
    phase12_expected = phase12_source["input_fingerprints"]
    phase12_validation, _phase12_manifest, phase12_fingerprints = _validate_phase12_input(
        phase12_path, phase12_expected
    )
    predictions = pd.read_parquet(
        phase12_path / "data/predictions/oof_mixture_predictions.parquet"
    )
    labels = pd.read_parquet(phase12_path / "data/targets/mixture_labels.parquet")
    expected_rows = len(phase12_source["features"]) * len(
        phase12_source["preprocessing"]["splits"]
    )
    if len(predictions) != expected_rows or predictions.duplicated(
        ["feature_id", "split_name"]
    ).any():
        raise ValueError("Phase 13 requires exact unique Phase 12 OOF coverage")
    return {
        **phase12_source,
        "validations": {
            **phase12_source["validations"],
            "phase12": phase12_validation,
        },
        "input_fingerprints": _combined_input_fingerprints(
            phase12_expected, phase12_fingerprints
        ),
        "phase12_predictions": predictions,
        "phase12_labels": labels,
    }


def _rank_uniform(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Phase 13 rank transform requires finite non-empty values")
    ranks = rankdata(values, method="average")
    return np.clip((ranks - 0.5) / len(values), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def _recalibration_levels(
    pit: np.ndarray,
    *,
    calibration_strength: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pit = np.clip(np.asarray(pit, dtype=float), 0.0, 1.0)
    if not len(pit) or not np.isfinite(pit).all():
        raise ValueError("Phase 13 calibration PIT values must be finite and non-empty")
    nominal = np.asarray(QUANTILE_LEVELS, dtype=float)
    empirical = np.quantile(pit, nominal, method="linear")
    adjusted = (len(pit) * empirical + calibration_strength * nominal) / (
        len(pit) + calibration_strength
    )
    adjusted = np.maximum.accumulate(np.clip(adjusted, lower_bound, upper_bound))
    diagnostics = {
        "training_row_count": int(len(pit)),
        "training_pit_mean": float(pit.mean()),
        "training_pit_variance": float(pit.var()),
        "calibration_strength": float(calibration_strength),
        "maximum_level_shift": float(np.max(np.abs(adjusted - nominal))),
    }
    return adjusted, diagnostics


def _interpolate_support(
    support: np.ndarray,
    probabilities: np.ndarray,
    target_probabilities: np.ndarray,
) -> np.ndarray:
    support = np.asarray(support, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(target_probabilities, dtype=float)
    output = np.empty((len(support), len(targets)), dtype=float)
    for column, probability in enumerate(targets):
        right = int(np.searchsorted(probabilities, probability, side="right"))
        right = min(max(right, 1), len(probabilities) - 1)
        left = right - 1
        width = probabilities[right] - probabilities[left]
        fraction = (probability - probabilities[left]) / max(width, 1e-12)
        output[:, column] = support[:, left] + fraction * (
            support[:, right] - support[:, left]
        )
    return output


def _apply_quality_recalibration(
    quantiles: np.ndarray,
    adjusted_levels: np.ndarray,
) -> np.ndarray:
    quantiles = np.asarray(quantiles, dtype=float)
    support = np.column_stack(
        (np.zeros(len(quantiles), dtype=float), quantiles, np.ones(len(quantiles), dtype=float))
    )
    probabilities = np.asarray((0.0, *QUANTILE_LEVELS, 1.0), dtype=float)
    return np.clip(
        _interpolate_support(support, probabilities, adjusted_levels), 0.0, 1.0
    )


def _apply_runtime_recalibration(
    quantiles: np.ndarray,
    adjusted_levels: np.ndarray,
) -> np.ndarray:
    support_logs, probabilities = _runtime_support(np.asarray(quantiles, dtype=float))
    calibrated_logs = _interpolate_support(
        support_logs, probabilities, adjusted_levels
    )
    return np.exp(np.clip(calibrated_logs, -30.0, 30.0))


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _fit_binary_calibrator(
    raw_training: np.ndarray,
    observed_training: np.ndarray,
    *,
    minimum_class_rows: int,
    beta_prior: tuple[float, float],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = np.clip(np.asarray(raw_training, dtype=float), 0.0, 1.0)
    observed = np.asarray(observed_training, dtype=bool)
    if len(raw) != len(observed) or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Phase 13 binary calibration inputs are invalid")
    positive_count = int(observed.sum())
    negative_count = int((~observed).sum())
    alpha, beta = beta_prior
    if positive_count == 0:
        status = "beta_binomial_fallback_no_positive_rows"
        constant = float(alpha / (len(observed) + alpha + beta))
        estimator = None
    elif negative_count == 0:
        status = "beta_binomial_fallback_no_negative_rows"
        constant = float((positive_count + alpha) / (len(observed) + alpha + beta))
        estimator = None
    elif min(positive_count, negative_count) < minimum_class_rows:
        status = "beta_binomial_fallback_insufficient_class_support"
        constant = float((positive_count + alpha) / (len(observed) + alpha + beta))
        estimator = None
    else:
        estimator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=200,
            random_state=seed,
        )
        estimator.fit(_logit(raw)[:, None], observed.astype(int))
        status = "fitted_logistic_recalibration"
        constant = None
    model = {
        "status": status,
        "estimator": estimator,
        "constant_probability": constant,
        "beta_prior": {"alpha": float(alpha), "beta": float(beta)},
    }
    diagnostics = {
        "fit_status": status,
        "training_row_count": int(len(observed)),
        "training_positive_count": positive_count,
        "training_negative_count": negative_count,
        "training_raw_probability_mean": float(raw.mean()),
        "constant_probability": constant,
    }
    return model, diagnostics


def _predict_binary_calibrator(model: dict[str, Any], raw: np.ndarray) -> np.ndarray:
    raw = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
    estimator = model["estimator"]
    if estimator is None:
        calibrated = np.full(len(raw), model["constant_probability"], dtype=float)
    else:
        positive_index = list(estimator.classes_).index(1)
        calibrated = estimator.predict_proba(_logit(raw)[:, None])[:, positive_index]
    return np.clip(calibrated, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def _deterministic_uniform(feature_ids: pd.Series, seed: int) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(feature_ids.astype(str), index=False).to_numpy(
        dtype=np.uint64
    )
    hashes ^= np.uint64(seed)
    mantissa = hashes >> np.uint64(11)
    return (mantissa.astype(np.float64) + 0.5) / float(2**53)


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    positive = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-6)) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(positive), 1e-12))
    correlation = positive / np.outer(scale, scale)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _estimate_copula(
    quality_pit: np.ndarray,
    runtime_pit: np.ndarray,
    failure_probability: np.ndarray,
    observed_failure: np.ndarray,
    feature_ids: pd.Series,
    *,
    shrinkage: float,
    minimum_class_rows: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    quality_u = _rank_uniform(quality_pit)
    runtime_u = _rank_uniform(runtime_pit)
    quality_z = norm.ppf(quality_u)
    runtime_z = norm.ppf(runtime_u)
    observed = np.asarray(observed_failure, dtype=bool)
    positive_count = int(observed.sum())
    negative_count = int((~observed).sum())
    if positive_count == 0:
        failure_status = "unavailable_no_failure_events"
    elif negative_count == 0:
        failure_status = "unavailable_no_nonfailure_events"
    elif min(positive_count, negative_count) < minimum_class_rows:
        failure_status = "unavailable_insufficient_class_support"
    else:
        failure_status = "available_randomized_binary_pit"
    if failure_status.startswith("available"):
        jitter = _deterministic_uniform(feature_ids, seed)
        probability = np.clip(
            np.asarray(failure_probability, dtype=float),
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )
        failure_u = np.where(
            observed,
            (1.0 - probability) + jitter * probability,
            jitter * (1.0 - probability),
        )
        failure_z = norm.ppf(_rank_uniform(failure_u))
        raw = np.corrcoef(
            np.column_stack((quality_z, runtime_z, failure_z)), rowvar=False
        )
    else:
        quality_runtime = float(np.corrcoef(quality_z, runtime_z)[0, 1])
        raw = np.asarray(
            [[1.0, quality_runtime, 0.0], [quality_runtime, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
    factor = len(quality_z) / (len(quality_z) + shrinkage)
    shrunk = np.eye(3) + factor * (raw - np.eye(3))
    shrunk = np.clip(shrunk, -0.95, 0.95)
    np.fill_diagonal(shrunk, 1.0)
    correlation = _nearest_correlation(shrunk)
    diagnostics = {
        "training_row_count": int(len(quality_z)),
        "failure_dependence_status": failure_status,
        "training_failure_count": positive_count,
        "training_nonfailure_count": negative_count,
        "shrinkage": float(shrinkage),
        "shrinkage_factor": float(factor),
        "raw_quality_runtime_correlation": float(raw[0, 1]),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(correlation).min()),
        "maximum_eigenvalue": float(np.linalg.eigvalsh(correlation).max()),
    }
    return correlation, diagnostics


def _copula_log_density(
    quality_z: np.ndarray,
    runtime_z: np.ndarray,
    rho: float,
) -> np.ndarray:
    rho = float(np.clip(rho, -0.95, 0.95))
    determinant = max(1.0 - rho**2, 1e-12)
    numerator = rho**2 * (quality_z**2 + runtime_z**2) - 2.0 * rho * quality_z * runtime_z
    return -0.5 * math.log(determinant) - numerator / (2.0 * determinant)


def _conditional_failure_probability(
    quality_z: np.ndarray,
    runtime_z: np.ndarray,
    marginal_probability: np.ndarray,
    correlation: np.ndarray,
    failure_dependence_status: str,
) -> np.ndarray:
    marginal = np.clip(
        np.asarray(marginal_probability, dtype=float),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    if not failure_dependence_status.startswith("available"):
        return marginal
    predictor_correlation = correlation[:2, :2]
    cross = correlation[2, :2]
    inverse = np.linalg.inv(predictor_correlation)
    coefficients = cross @ inverse
    mean = coefficients[0] * quality_z + coefficients[1] * runtime_z
    variance = max(float(1.0 - coefficients @ cross.T), 1e-6)
    threshold = norm.ppf(1.0 - marginal)
    conditional = norm.sf((threshold - mean) / math.sqrt(variance))
    return np.clip(conditional, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def _bivariate_tail_probability(
    quality_cdf: np.ndarray,
    runtime_cdf: np.ndarray,
    rho: float,
    *,
    seed: int,
) -> np.ndarray:
    quality_u = np.clip(quality_cdf, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    runtime_u = np.clip(runtime_cdf, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    points = np.column_stack((norm.ppf(quality_u), norm.ppf(runtime_u)))
    lower = multivariate_normal.cdf(
        points,
        mean=np.zeros(2),
        cov=np.asarray([[1.0, rho], [rho, 1.0]], dtype=float),
        rng=np.random.default_rng(seed),
    )
    return np.clip(runtime_u - lower, 0.0, 1.0)


def _binary_nll(observed: np.ndarray, probability: np.ndarray) -> np.ndarray:
    numeric = np.asarray(observed, dtype=bool).astype(float)
    clipped = np.clip(
        np.asarray(probability, dtype=float),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    return -(numeric * np.log(clipped) + (1.0 - numeric) * np.log(1.0 - clipped))


def _raw_quantiles(frame: pd.DataFrame, target: str) -> np.ndarray:
    return frame[
        [f"moe_{target}_{_quantile_suffix(level)}" for level in QUANTILE_LEVELS]
    ].to_numpy(dtype=float)


def _calibrated_prediction_frame(
    training_source: pd.DataFrame,
    validation_source: pd.DataFrame,
    *,
    job_id: str,
    master_seed: int,
    calibration_strength: float,
    minimum_class_rows: int,
    copula_shrinkage: float,
    beta_prior: tuple[float, float],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    quality_training_raw = _raw_quantiles(training_source, "quality")
    quality_validation_raw = _raw_quantiles(validation_source, "quality")
    runtime_training_raw = _raw_quantiles(training_source, "runtime")
    runtime_validation_raw = _raw_quantiles(validation_source, "runtime")
    quality_training_pit = _cdf_at(
        quality_training_raw,
        training_source["observed_quality"].to_numpy(dtype=float),
    )
    runtime_training_pit = _runtime_cdf(
        runtime_training_raw,
        training_source["observed_runtime"].to_numpy(dtype=float),
    )
    quality_levels, quality_diagnostics = _recalibration_levels(
        quality_training_pit,
        calibration_strength=calibration_strength,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    runtime_levels, runtime_diagnostics = _recalibration_levels(
        runtime_training_pit,
        calibration_strength=calibration_strength,
        lower_bound=0.001,
        upper_bound=0.999,
    )
    quality_training = _apply_quality_recalibration(
        quality_training_raw, quality_levels
    )
    quality_validation = _apply_quality_recalibration(
        quality_validation_raw, quality_levels
    )
    runtime_training = _apply_runtime_recalibration(
        runtime_training_raw, runtime_levels
    )
    runtime_validation = _apply_runtime_recalibration(
        runtime_validation_raw, runtime_levels
    )
    failure_seed = int(
        (master_seed + int(hashlib.sha256(f"{job_id}:failure".encode()).hexdigest()[:8], 16))
        % (2**31 - 1)
    )
    failure_model, failure_diagnostics = _fit_binary_calibrator(
        training_source["moe_failure_probability"].to_numpy(dtype=float),
        training_source["observed_failure"].to_numpy(dtype=bool),
        minimum_class_rows=minimum_class_rows,
        beta_prior=beta_prior,
        seed=failure_seed,
    )
    failure_training = _predict_binary_calibrator(
        failure_model,
        training_source["moe_failure_probability"].to_numpy(dtype=float),
    )
    failure_validation = _predict_binary_calibrator(
        failure_model,
        validation_source["moe_failure_probability"].to_numpy(dtype=float),
    )
    survival_models: dict[str, Any] = {}
    survival_diagnostics: dict[str, Any] = {}
    calibrated_reach = np.empty((len(validation_source), len(SURVIVAL_HORIZONS)))
    for index, horizon in enumerate(SURVIVAL_HORIZONS):
        suffix = _horizon_suffix(horizon)
        observable, outcome = _horizon_observation(
            training_source["event_observed"].to_numpy(dtype=bool),
            training_source["duration_step"].to_numpy(dtype=float),
            training_source["budget"].to_numpy(dtype=float),
            horizon,
        )
        survival_seed = int(
            (
                master_seed
                + int(
                    hashlib.sha256(f"{job_id}:survival:{horizon}".encode()).hexdigest()[:8],
                    16,
                )
            )
            % (2**31 - 1)
        )
        model, diagnostics = _fit_binary_calibrator(
            training_source.loc[observable, f"moe_reach_by_{suffix}"].to_numpy(dtype=float),
            outcome[observable],
            minimum_class_rows=minimum_class_rows,
            beta_prior=beta_prior,
            seed=survival_seed,
        )
        survival_models[suffix] = model
        survival_diagnostics[suffix] = {
            **diagnostics,
            "observable_training_count": int(observable.sum()),
        }
        calibrated_reach[:, index] = _predict_binary_calibrator(
            model,
            validation_source[f"moe_reach_by_{suffix}"].to_numpy(dtype=float),
        )
    calibrated_reach = np.maximum.accumulate(calibrated_reach, axis=1)
    calibrated_quality_training_pit = _cdf_at(
        quality_training,
        training_source["observed_quality"].to_numpy(dtype=float),
    )
    calibrated_runtime_training_pit = _runtime_cdf(
        runtime_training,
        training_source["observed_runtime"].to_numpy(dtype=float),
    )
    copula_seed = int(
        (master_seed + int(hashlib.sha256(f"{job_id}:copula".encode()).hexdigest()[:8], 16))
        % (2**31 - 1)
    )
    correlation, copula_diagnostics = _estimate_copula(
        calibrated_quality_training_pit,
        calibrated_runtime_training_pit,
        failure_training,
        training_source["observed_failure"].to_numpy(dtype=bool),
        training_source["feature_id"],
        shrinkage=copula_shrinkage,
        minimum_class_rows=minimum_class_rows,
        seed=copula_seed,
    )
    identity_columns = [
        "feature_id",
        "cutoff",
        "domain",
        "problem_family",
        "algorithm_family",
        "split_name",
        "fold",
        "observed_quality",
        "observed_runtime",
        "observed_failure",
        "event_observed",
        "first_passage_step",
        "censor_step",
        "duration_step",
        "duration_fraction",
        "budget",
    ]
    frame = validation_source[identity_columns].reset_index(drop=True).copy()
    for index, level in enumerate(QUANTILE_LEVELS):
        suffix = _quantile_suffix(level)
        frame[f"reference_quality_{suffix}"] = quality_validation_raw[:, index]
        frame[f"quality_{suffix}"] = quality_validation[:, index]
        frame[f"reference_runtime_{suffix}"] = runtime_validation_raw[:, index]
        frame[f"runtime_{suffix}"] = runtime_validation[:, index]
    frame["reference_failure_probability"] = validation_source[
        "moe_failure_probability"
    ].to_numpy(dtype=float)
    frame["failure_probability"] = failure_validation
    for index, horizon in enumerate(SURVIVAL_HORIZONS):
        suffix = _horizon_suffix(horizon)
        frame[f"reference_reach_by_{suffix}"] = validation_source[
            f"moe_reach_by_{suffix}"
        ].to_numpy(dtype=float)
        frame[f"reach_by_{suffix}"] = calibrated_reach[:, index]
    quality_pit = _cdf_at(
        quality_validation,
        frame["observed_quality"].to_numpy(dtype=float),
    )
    runtime_pit = _runtime_cdf(
        runtime_validation,
        frame["observed_runtime"].to_numpy(dtype=float),
    )
    quality_z = norm.ppf(np.clip(quality_pit, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON))
    runtime_z = norm.ppf(np.clip(runtime_pit, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON))
    reference_quality_pit = _cdf_at(
        quality_validation_raw,
        frame["observed_quality"].to_numpy(dtype=float),
    )
    reference_runtime_pit = _runtime_cdf(
        runtime_validation_raw,
        frame["observed_runtime"].to_numpy(dtype=float),
    )
    reference_quality_z = norm.ppf(
        np.clip(reference_quality_pit, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    )
    reference_runtime_z = norm.ppf(
        np.clip(reference_runtime_pit, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    )
    failure_status = copula_diagnostics["failure_dependence_status"]
    conditional_failure = _conditional_failure_probability(
        quality_z,
        runtime_z,
        failure_validation,
        correlation,
        failure_status,
    )
    reference_conditional_failure = _conditional_failure_probability(
        reference_quality_z,
        reference_runtime_z,
        frame["reference_failure_probability"].to_numpy(dtype=float),
        correlation,
        failure_status,
    )
    rho = float(correlation[0, 1])
    copula_log_density = _copula_log_density(quality_z, runtime_z, rho)
    reference_copula_log_density = _copula_log_density(
        reference_quality_z, reference_runtime_z, rho
    )
    observed_failure = frame["observed_failure"].to_numpy(dtype=bool)
    quality_nll = _piecewise_nll(
        frame["observed_quality"].to_numpy(dtype=float), quality_validation
    )
    runtime_nll = _runtime_nll(
        frame["observed_runtime"].to_numpy(dtype=float), runtime_validation
    )
    failure_nll = _binary_nll(observed_failure, failure_validation)
    conditional_failure_nll = _binary_nll(observed_failure, conditional_failure)
    reference_quality_nll = _piecewise_nll(
        frame["observed_quality"].to_numpy(dtype=float), quality_validation_raw
    )
    reference_runtime_nll = _runtime_nll(
        frame["observed_runtime"].to_numpy(dtype=float), runtime_validation_raw
    )
    reference_failure_nll = _binary_nll(
        observed_failure, frame["reference_failure_probability"].to_numpy(dtype=float)
    )
    reference_conditional_failure_nll = _binary_nll(
        observed_failure, reference_conditional_failure
    )
    frame["quality_pit"] = quality_pit
    frame["runtime_pit"] = runtime_pit
    frame["reference_quality_pit"] = reference_quality_pit
    frame["reference_runtime_pit"] = reference_runtime_pit
    frame["copula_quality_runtime_rho"] = rho
    frame["copula_quality_failure_rho"] = float(correlation[0, 2])
    frame["copula_runtime_failure_rho"] = float(correlation[1, 2])
    frame["copula_minimum_eigenvalue"] = copula_diagnostics["minimum_eigenvalue"]
    frame["failure_dependence_status"] = failure_status
    frame["conditional_failure_probability"] = conditional_failure
    frame["copula_log_density"] = copula_log_density
    frame["independent_joint_nll"] = quality_nll + runtime_nll + failure_nll
    frame["joint_nll"] = (
        quality_nll + runtime_nll - copula_log_density + conditional_failure_nll
    )
    frame["reference_copula_log_density"] = reference_copula_log_density
    frame["reference_independent_joint_nll"] = (
        reference_quality_nll + reference_runtime_nll + reference_failure_nll
    )
    frame["reference_joint_nll"] = (
        reference_quality_nll
        + reference_runtime_nll
        - reference_copula_log_density
        + reference_conditional_failure_nll
    )
    quality_mean, quality_std = _predictive_moments(quality_validation)
    runtime_mean, runtime_std = _runtime_moments(runtime_validation)
    restricted_mean, expected_par10 = _survival_summaries(
        calibrated_reach, frame["budget"].to_numpy(dtype=float)
    )
    frame["quality_predictive_mean"] = quality_mean
    frame["quality_predictive_std"] = quality_std
    frame["runtime_predictive_mean"] = runtime_mean
    frame["runtime_predictive_std"] = runtime_std
    frame["predicted_restricted_mean_step"] = restricted_mean
    frame["predicted_par10"] = expected_par10
    runtime_within_budget = _runtime_cdf(
        runtime_validation, frame["budget"].to_numpy(dtype=float)
    )
    frame["runtime_within_budget_probability"] = runtime_within_budget
    for threshold_index, threshold in enumerate(JOINT_QUALITY_THRESHOLDS):
        suffix = _threshold_suffix(threshold)
        quality_cdf = _cdf_at(
            quality_validation, np.full(len(frame), threshold, dtype=float)
        )
        quality_above = 1.0 - quality_cdf
        qr_probability = _bivariate_tail_probability(
            quality_cdf,
            runtime_within_budget,
            rho,
            seed=copula_seed + threshold_index + 1,
        )
        frame[f"quality_ge_{suffix}_probability"] = quality_above
        frame[f"joint_quality_ge_{suffix}_runtime_within_budget_probability"] = qr_probability
        frame[
            f"joint_quality_ge_{suffix}_runtime_within_budget_no_failure_probability"
        ] = np.clip(qr_probability * (1.0 - failure_validation), 0.0, 1.0)
    support_rows: list[dict[str, Any]] = []
    for target, levels, diagnostics in (
        ("quality", quality_levels, quality_diagnostics),
        ("runtime", runtime_levels, runtime_diagnostics),
    ):
        for nominal, adjusted in zip(QUANTILE_LEVELS, levels, strict=True):
            support_rows.append(
                {
                    "component": "marginal_quantile_recalibration",
                    "target": target,
                    "component_name": _quantile_suffix(nominal),
                    "nominal_level": float(nominal),
                    "adjusted_level": float(adjusted),
                    "fit_status": "empirical_pit_recalibration",
                    "training_row_count": diagnostics["training_row_count"],
                    "training_positive_count": None,
                    "training_negative_count": None,
                    "dependence_status": None,
                    "estimated_correlation": None,
                }
            )
    support_rows.append(
        {
            "component": "binary_recalibration",
            "target": "failure",
            "component_name": "failure_probability",
            "nominal_level": None,
            "adjusted_level": None,
            **failure_diagnostics,
            "dependence_status": None,
            "estimated_correlation": None,
        }
    )
    for horizon in SURVIVAL_HORIZONS:
        suffix = _horizon_suffix(horizon)
        support_rows.append(
            {
                "component": "binary_recalibration",
                "target": "survival",
                "component_name": suffix,
                "nominal_level": float(horizon),
                "adjusted_level": None,
                **survival_diagnostics[suffix],
                "dependence_status": None,
                "estimated_correlation": None,
            }
        )
    for first, second, row, column in (
        ("quality", "runtime", 0, 1),
        ("quality", "failure", 0, 2),
        ("runtime", "failure", 1, 2),
    ):
        dependence_status = (
            "available_continuous_gaussian_copula"
            if "failure" not in {first, second}
            else failure_status
        )
        support_rows.append(
            {
                "component": "gaussian_copula_dependence",
                "target": "joint",
                "component_name": f"{first}_{second}",
                "nominal_level": None,
                "adjusted_level": None,
                "fit_status": JOINT_MODEL_NAME,
                "training_row_count": copula_diagnostics["training_row_count"],
                "training_positive_count": copula_diagnostics["training_failure_count"],
                "training_negative_count": copula_diagnostics["training_nonfailure_count"],
                "dependence_status": dependence_status,
                "estimated_correlation": float(correlation[row, column]),
            }
        )
    model_artifact = {
        "schema_version": PHASE13_SCHEMA_VERSION,
        "joint_model": JOINT_MODEL_NAME,
        "job_id": job_id,
        "quality_adjusted_levels": quality_levels,
        "runtime_adjusted_levels": runtime_levels,
        "failure_calibrator": failure_model,
        "survival_calibrators": survival_models,
        "copula_correlation": correlation,
        "copula_diagnostics": copula_diagnostics,
        "joint_probability_failure_policy": (
            "exact marginal independence fallback because failure dependence is unavailable"
            if not failure_status.startswith("available")
            else "marginal failure approximation for threshold summaries"
        ),
    }
    return frame, model_artifact, pd.DataFrame(support_rows)


def _pit_payload(values: np.ndarray) -> dict[str, float]:
    pit = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return {
        "pit_mean": float(pit.mean()),
        "pit_variance": float(pit.var()),
        "pit_ks_uniform": float(kstest(pit, "uniform").statistic),
    }


def _short_runtime_metrics(values: dict[str, Any]) -> dict[str, Any]:
    return {
        (key.removeprefix("runtime_") if key.startswith("runtime_") else key): value
        for key, value in values.items()
    }


def _metric_payload(frame: pd.DataFrame) -> dict[str, Any]:
    observed_quality = frame["observed_quality"].to_numpy(dtype=float)
    observed_runtime = frame["observed_runtime"].to_numpy(dtype=float)
    observed_failure = frame["observed_failure"].to_numpy(dtype=bool)
    event = frame["event_observed"].to_numpy(dtype=bool)
    duration = frame["duration_step"].to_numpy(dtype=float)
    budget = frame["budget"].to_numpy(dtype=float)
    result: dict[str, Any] = {}
    source_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for source in ("calibrated", "reference"):
        prefix = "" if source == "calibrated" else "reference_"
        quality_quantiles = frame[
            [f"{prefix}quality_{_quantile_suffix(level)}" for level in QUANTILE_LEVELS]
        ].to_numpy(dtype=float)
        runtime_quantiles = frame[
            [f"{prefix}runtime_{_quantile_suffix(level)}" for level in QUANTILE_LEVELS]
        ].to_numpy(dtype=float)
        failure_probability = frame[f"{prefix}failure_probability"].to_numpy(dtype=float)
        reach = frame[
            [f"{prefix}reach_by_{_horizon_suffix(horizon)}" for horizon in SURVIVAL_HORIZONS]
        ].to_numpy(dtype=float)
        metrics = {
            "quality": _distribution_metrics(observed_quality, quality_quantiles),
            "runtime": _short_runtime_metrics(
                _runtime_metrics(observed_runtime, runtime_quantiles)
            ),
            "failure": _binary_metrics(
                observed_failure,
                failure_probability,
                calibration_bins=DEFAULT_CALIBRATION_BINS,
            ),
            "survival": _survival_metrics(event, duration, budget, reach),
        }
        source_metrics[source] = metrics
        for target, target_metrics in metrics.items():
            result.update(
                {
                    f"{source}_{target}_{key}": value
                    for key, value in target_metrics.items()
                }
            )
    result.update(
        {
            f"calibrated_quality_{key}": value
            for key, value in _pit_payload(frame["quality_pit"].to_numpy(dtype=float)).items()
        }
    )
    result.update(
        {
            f"reference_quality_{key}": value
            for key, value in _pit_payload(
                frame["reference_quality_pit"].to_numpy(dtype=float)
            ).items()
        }
    )
    result.update(
        {
            f"calibrated_runtime_{key}": value
            for key, value in _pit_payload(frame["runtime_pit"].to_numpy(dtype=float)).items()
        }
    )
    result.update(
        {
            f"reference_runtime_{key}": value
            for key, value in _pit_payload(
                frame["reference_runtime_pit"].to_numpy(dtype=float)
            ).items()
        }
    )
    for column in (
        "joint_nll",
        "independent_joint_nll",
        "reference_joint_nll",
        "reference_independent_joint_nll",
        "copula_log_density",
        "reference_copula_log_density",
    ):
        result[column] = float(frame[column].mean())
    result["joint_nll_delta_vs_reference"] = (
        result["joint_nll"] - result["reference_joint_nll"]
    )
    result["copula_nll_gain_vs_independence"] = (
        result["joint_nll"] - result["independent_joint_nll"]
    )
    result["reference_copula_nll_gain_vs_independence"] = (
        result["reference_joint_nll"] - result["reference_independent_joint_nll"]
    )
    result["copula_quality_runtime_rho_mean"] = float(
        frame["copula_quality_runtime_rho"].mean()
    )
    for threshold in JOINT_QUALITY_THRESHOLDS:
        suffix = _threshold_suffix(threshold)
        probability_column = (
            f"joint_quality_ge_{suffix}_runtime_within_budget_no_failure_probability"
        )
        probability = frame[probability_column].to_numpy(dtype=float)
        observed = (
            (observed_quality >= threshold)
            & (observed_runtime <= budget)
            & (~observed_failure)
        ).astype(float)
        result[f"joint_{suffix}_observed_rate"] = float(observed.mean())
        result[f"joint_{suffix}_predicted_rate"] = float(probability.mean())
        result[f"joint_{suffix}_brier"] = float(np.mean((observed - probability) ** 2))
        result[f"joint_{suffix}_calibration_error"] = float(
            probability.mean() - observed.mean()
        )
    for target, metric in (
        ("quality", "nll"),
        ("runtime", "nll"),
        ("failure", "brier"),
        ("survival", "integrated_brier"),
    ):
        result[f"calibrated_{target}_{metric}_delta_vs_reference"] = (
            source_metrics["calibrated"][target][metric]
            - source_metrics["reference"][target][metric]
        )
    return result


def _job_id(split_name: str, fold: int) -> str:
    return f"{split_name}__fold{fold}__joint_calibration"


def _job_paths(run_path: Path, job_id: str) -> dict[str, Path]:
    return {
        "predictions": run_path / "checkpoints/predictions" / f"{job_id}.parquet",
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
            raise ValueError(f"Phase 13 job output hash mismatch: job={job_id} path={relative}")
    return marker


def _run_jobs(
    run_path: Path,
    features: pd.DataFrame,
    phase12_predictions: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
    state: dict[str, Any],
    *,
    master_seed: int,
    calibration_strength: float,
    minimum_class_rows: int,
    copula_shrinkage: float,
    beta_prior: tuple[float, float],
    progress_path: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[Path]]:
    fold_schemas: dict[str, Any] = {
        "schema_version": PHASE13_SCHEMA_VERSION,
        "source_preprocessing_schema_version": preprocessing["schema_version"],
        "transform_policy": (
            "no feature refit; verify Phase 6 fold contract and fit Phase 13 calibration plus "
            "copula only on non-validation Phase 12 OOF predictions"
        ),
        "folds": {},
    }
    output_paths: list[Path] = []
    for split_name in sorted(preprocessing["splits"]):
        fold_column = feature_schema["split_columns"][split_name]
        for fold in sorted(int(value) for value in preprocessing["splits"][split_name]):
            fold_key = f"{split_name}/fold{fold}"
            fold_specification = preprocessing["splits"][split_name][str(fold)]
            training_mask, validation_mask, fold_contract = _verify_fold_contract(
                features, splits, fold_column, fold, fold_specification
            )
            training_source = _ordered_predictions(
                phase12_predictions,
                split_name,
                features.loc[training_mask, "feature_id"],
            )
            validation_source = _ordered_predictions(
                phase12_predictions,
                split_name,
                features.loc[validation_mask, "feature_id"],
            )
            meta_training_current_fold_count = int((training_source["fold"] == fold).sum())
            validation_wrong_fold_count = int((validation_source["fold"] != fold).sum())
            if meta_training_current_fold_count or validation_wrong_fold_count:
                raise ValueError(f"Phase 13 cross-fitting contract failed: {fold_key}")
            fold_schemas["folds"][fold_key] = {
                **fold_contract,
                "fold_column": fold_column,
                "meta_training_current_fold_count": meta_training_current_fold_count,
                "validation_wrong_fold_count": validation_wrong_fold_count,
                "calibration_source": "Phase 12 OOF predictions from non-validation folds",
            }
            job_id = _job_id(split_name, fold)
            paths = _job_paths(run_path, job_id)
            marker = _verify_job(run_path, job_id)
            if marker is not None:
                state.setdefault("completed_jobs", {})[job_id] = _file_sha256(paths["marker"])
                output_paths.extend(paths.values())
                logger.info("[PHASE13][RESUME] job=%s status=SKIP_VERIFIED", job_id)
                continue
            logger.info(
                "[PHASE13][FOLD] split=%s fold=%s train_rows=%s validation_rows=%s",
                split_name,
                fold,
                int(training_mask.sum()),
                int(validation_mask.sum()),
            )
            predictions, model_artifact, support = _calibrated_prediction_frame(
                training_source,
                validation_source,
                job_id=job_id,
                master_seed=master_seed,
                calibration_strength=calibration_strength,
                minimum_class_rows=minimum_class_rows,
                copula_shrinkage=copula_shrinkage,
                beta_prior=beta_prior,
            )
            support.insert(0, "fold", fold)
            support.insert(0, "split_name", split_name)
            support.insert(0, "job_id", job_id)
            metrics = {
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "training_row_count": int(training_mask.sum()),
                "validation_row_count": int(validation_mask.sum()),
                "training_problem_count": fold_contract["training_problem_count"],
                "validation_problem_count": fold_contract["validation_problem_count"],
                "meta_training_current_fold_count": meta_training_current_fold_count,
                "validation_wrong_fold_count": validation_wrong_fold_count,
                **_metric_payload(predictions),
            }
            model_artifact.update(
                {
                    "split_name": split_name,
                    "fold": fold,
                    "fold_contract": fold_contract,
                }
            )
            _atomic_parquet(paths["predictions"], predictions)
            _atomic_json(paths["metrics"], metrics)
            _atomic_parquet(paths["support"], support)
            _atomic_pickle(paths["model"], model_artifact)
            marker = {
                "schema_version": PHASE13_SCHEMA_VERSION,
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "output_sha256": _relative_hashes(
                    run_path,
                    (
                        paths["predictions"],
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
            _append_progress(progress_path, {"event": "job_completed", "job_id": job_id})
            output_paths.extend(paths.values())
            logger.info(
                "[PHASE13][JOB] job=%s rho_qr=%.4f joint_nll=%.4f failure_dependence=%s",
                job_id,
                float(predictions["copula_quality_runtime_rho"].iloc[0]),
                float(predictions["joint_nll"].mean()),
                str(predictions["failure_dependence_status"].iloc[0]),
            )
    fold_schema_path = run_path / "data/preprocessing/fold_calibration_schemas.json"
    _atomic_json(fold_schema_path, fold_schemas)
    output_paths.append(fold_schema_path)
    return fold_schemas, output_paths


def _aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, columns in (
        ("overall", ("split_name",)),
        ("cutoff", ("split_name", "cutoff")),
        ("domain", ("split_name", "domain")),
        ("domain_cutoff", ("split_name", "domain", "cutoff")),
    ):
        for keys, group in predictions.groupby(list(columns), sort=True):
            values = keys if isinstance(keys, tuple) else (keys,)
            identifiers = dict(zip(columns, values, strict=True))
            rows.append(
                {
                    "scope": scope,
                    "split_name": str(identifiers["split_name"]),
                    "domain": identifiers.get("domain"),
                    "cutoff": (
                        float(identifiers["cutoff"]) if "cutoff" in identifiers else None
                    ),
                    "row_count": int(len(group)),
                    "fold_count": int(group["fold"].nunique()),
                    **_metric_payload(group),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "split_name", "domain", "cutoff"], na_position="first"
    ).reset_index(drop=True)


def _aggregate_job_outputs(
    run_path: Path,
    jobs: list[tuple[str, int]],
    labels: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any]]:
    prediction_frames = []
    metric_rows = []
    support_frames = []
    registry = {
        "schema_version": PHASE13_SCHEMA_VERSION,
        "joint_model": JOINT_MODEL_NAME,
        "marginals": ["quality", "runtime", "failure", "first_passage_survival"],
        "dependence_order": ["quality", "runtime", "failure"],
        "calibration_policy": "cross-fitted empirical PIT and logistic/Beta-Binomial calibration",
        "jobs": [],
    }
    for split_name, fold in jobs:
        job_id = _job_id(split_name, fold)
        marker = _verify_job(run_path, job_id)
        if marker is None:
            raise ValueError(f"Phase 13 job is incomplete: {job_id}")
        paths = _job_paths(run_path, job_id)
        prediction_frames.append(pd.read_parquet(paths["predictions"]))
        metric_rows.append(json.loads(paths["metrics"].read_text(encoding="utf-8")))
        support_frames.append(pd.read_parquet(paths["support"]))
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
    support = pd.concat(support_frames, ignore_index=True).sort_values(
        ["split_name", "fold", "component", "target", "component_name"]
    ).reset_index(drop=True)
    aggregate = _aggregate_metrics(predictions)
    predictions_path = run_path / "data/predictions/oof_joint_calibrated_predictions.parquet"
    labels_path = run_path / "data/targets/joint_calibration_labels.parquet"
    fold_metrics_path = run_path / "data/metrics/fold_joint_calibration_metrics.parquet"
    aggregate_path = run_path / "data/metrics/aggregate_joint_calibration_metrics.parquet"
    support_path = run_path / "data/calibration/calibration_copula_support.parquet"
    registry_path = run_path / "model_registry.json"
    _atomic_parquet(predictions_path, predictions)
    _atomic_parquet(labels_path, labels)
    _atomic_parquet(fold_metrics_path, fold_metrics)
    _atomic_parquet(aggregate_path, aggregate)
    _atomic_parquet(support_path, support)
    _atomic_json(registry_path, registry)
    return [
        predictions_path,
        labels_path,
        fold_metrics_path,
        aggregate_path,
        support_path,
        registry_path,
    ], registry


def _reference_identity_exact(
    predictions: pd.DataFrame,
    phase12_predictions: pd.DataFrame,
) -> bool:
    for split_name, group in predictions.groupby("split_name"):
        ordered = _ordered_predictions(
            phase12_predictions, split_name, group["feature_id"]
        )
        for level in QUANTILE_LEVELS:
            suffix = _quantile_suffix(level)
            for target in ("quality", "runtime"):
                if not np.allclose(
                    group[f"reference_{target}_{suffix}"].to_numpy(dtype=float),
                    ordered[f"moe_{target}_{suffix}"].to_numpy(dtype=float),
                ):
                    return False
        if not np.allclose(
            group["reference_failure_probability"].to_numpy(dtype=float),
            ordered["moe_failure_probability"].to_numpy(dtype=float),
        ):
            return False
        for horizon in SURVIVAL_HORIZONS:
            suffix = _horizon_suffix(horizon)
            if not np.allclose(
                group[f"reference_reach_by_{suffix}"].to_numpy(dtype=float),
                ordered[f"moe_reach_by_{suffix}"].to_numpy(dtype=float),
            ):
                return False
    return True


def _copula_support_valid(support: pd.DataFrame) -> bool:
    rows = support.loc[support["component"] == "gaussian_copula_dependence"]
    for _keys, group in rows.groupby(["split_name", "fold"]):
        values = dict(
            zip(group["component_name"], group["estimated_correlation"], strict=True)
        )
        if set(values) != {"quality_runtime", "quality_failure", "runtime_failure"}:
            return False
        matrix = np.asarray(
            [
                [1.0, values["quality_runtime"], values["quality_failure"]],
                [values["quality_runtime"], 1.0, values["runtime_failure"]],
                [values["quality_failure"], values["runtime_failure"], 1.0],
            ],
            dtype=float,
        )
        if not np.isfinite(matrix).all() or np.linalg.eigvalsh(matrix).min() < -1e-8:
            return False
    return True


def _validate_phase13(
    run_path: Path,
    paths: tuple[Path, Path, Path, Path, Path, Path, Path],
    original_inputs: dict[str, str],
    source: dict[str, Any],
) -> dict[str, Any]:
    current = _load_inputs(*paths)
    features = source["features"]
    splits = source["splits"]
    labels = source["phase12_labels"]
    preprocessing = source["preprocessing"]
    feature_schema = source["feature_schema"]
    predictions = pd.read_parquet(
        run_path / "data/predictions/oof_joint_calibrated_predictions.parquet"
    )
    saved_labels = pd.read_parquet(
        run_path / "data/targets/joint_calibration_labels.parquet"
    )
    fold_metrics = pd.read_parquet(
        run_path / "data/metrics/fold_joint_calibration_metrics.parquet"
    )
    aggregate = pd.read_parquet(
        run_path / "data/metrics/aggregate_joint_calibration_metrics.parquet"
    )
    support = pd.read_parquet(
        run_path / "data/calibration/calibration_copula_support.parquet"
    )
    registry = json.loads((run_path / "model_registry.json").read_text(encoding="utf-8"))
    fold_schemas = json.loads(
        (run_path / "data/preprocessing/fold_calibration_schemas.json").read_text(
            encoding="utf-8"
        )
    )
    jobs = _all_jobs(preprocessing)
    expected_rows = len(features) * len(preprocessing["splits"])
    coverage_ok = all(
        len(group) == len(features) and set(group["feature_id"]) == set(features["feature_id"])
        for _split_name, group in predictions.groupby("split_name")
    )
    fold_assignment_ok = True
    for split_name, fold_column in feature_schema["split_columns"].items():
        expected_folds = pd.Series(
            splits[fold_column].to_numpy(dtype=int), index=splits["feature_id"].astype(str)
        )
        observed = predictions.loc[predictions["split_name"] == split_name]
        mapped = observed["feature_id"].astype(str).map(expected_folds)
        if mapped.isna().any() or not np.array_equal(
            mapped.to_numpy(dtype=int), observed["fold"].to_numpy(dtype=int)
        ):
            fold_assignment_ok = False
    quality = predictions[
        [f"quality_{_quantile_suffix(level)}" for level in QUANTILE_LEVELS]
    ].to_numpy(dtype=float)
    runtime = predictions[
        [f"runtime_{_quantile_suffix(level)}" for level in QUANTILE_LEVELS]
    ].to_numpy(dtype=float)
    reach = predictions[
        [f"reach_by_{_horizon_suffix(horizon)}" for horizon in SURVIVAL_HORIZONS]
    ].to_numpy(dtype=float)
    probability_columns = [
        "failure_probability",
        "conditional_failure_probability",
        "runtime_within_budget_probability",
    ]
    for threshold in JOINT_QUALITY_THRESHOLDS:
        suffix = _threshold_suffix(threshold)
        probability_columns.extend(
            [
                f"quality_ge_{suffix}_probability",
                f"joint_quality_ge_{suffix}_runtime_within_budget_probability",
                f"joint_quality_ge_{suffix}_runtime_within_budget_no_failure_probability",
            ]
        )
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    calibration_levels = support.loc[
        support["component"] == "marginal_quantile_recalibration", "adjusted_level"
    ].to_numpy(dtype=float)
    support_components_exact = all(
        len(group) == 25
        and (group["component"] == "marginal_quantile_recalibration").sum() == 14
        and (group["component"] == "binary_recalibration").sum() == 8
        and (group["component"] == "gaussian_copula_dependence").sum() == 3
        for _keys, group in support.groupby(["split_name", "fold"])
    )
    failure_rows = support.loc[
        (support["component"] == "gaussian_copula_dependence")
        & support["component_name"].str.contains("failure")
    ]
    source_has_no_failures = not bool(source["phase12_predictions"]["observed_failure"].any())
    fallback_explicit = (
        bool((failure_rows["dependence_status"] == "unavailable_no_failure_events").all())
        and bool(
            (
                predictions["failure_dependence_status"]
                == "unavailable_no_failure_events"
            ).all()
        )
        if source_has_no_failures
        else True
    )
    joint_nested = True
    for threshold in JOINT_QUALITY_THRESHOLDS:
        suffix = _threshold_suffix(threshold)
        quality_probability = predictions[f"quality_ge_{suffix}_probability"]
        qr = predictions[
            f"joint_quality_ge_{suffix}_runtime_within_budget_probability"
        ]
        qrf = predictions[
            f"joint_quality_ge_{suffix}_runtime_within_budget_no_failure_probability"
        ]
        if not (
            (qr <= quality_probability + 1e-12).all()
            and (qr <= predictions["runtime_within_budget_probability"] + 1e-12).all()
            and (qrf <= qr + 1e-12).all()
            and (qrf <= 1.0 - predictions["failure_probability"] + 1e-12).all()
        ):
            joint_nested = False
    q75 = predictions[
        "joint_quality_ge_q075_runtime_within_budget_no_failure_probability"
    ]
    q90 = predictions[
        "joint_quality_ge_q090_runtime_within_budget_no_failure_probability"
    ]
    required_metrics = [
        "calibrated_quality_nll",
        "calibrated_quality_crps",
        "calibrated_quality_pit_ks_uniform",
        "calibrated_runtime_nll",
        "calibrated_runtime_crps",
        "calibrated_runtime_pit_ks_uniform",
        "calibrated_failure_brier",
        "calibrated_survival_survival_nll",
        "calibrated_survival_integrated_brier",
        "joint_nll",
        "independent_joint_nll",
        "joint_q075_brier",
        "joint_q090_brier",
    ]
    metrics_finite = bool(
        np.isfinite(fold_metrics[required_metrics].to_numpy(dtype=float)).all()
        and np.isfinite(aggregate[required_metrics].to_numpy(dtype=float)).all()
    )
    expected_aggregate_rows = len(preprocessing["splits"]) * (
        1
        + len(feature_schema["cutoffs"])
        + features["domain"].nunique()
        + features["domain"].nunique() * len(feature_schema["cutoffs"])
    )
    checks = {
        **{
            f"{phase}_quality_pass": validation["status"] == f"PHASE_{phase[5:]}_PASS"
            for phase, validation in current["validations"].items()
        },
        "source_inputs_unchanged": current["input_fingerprints"] == original_inputs,
        "joint_labels_frozen_and_exact": _frames_equal(saved_labels, labels),
        "phase12_reference_predictions_exact": _reference_identity_exact(
            predictions, source["phase12_predictions"]
        ),
        "expected_job_count": len(fold_metrics) == len(jobs) == len(registry["jobs"]),
        "all_job_markers_verified": all(
            _verify_job(run_path, _job_id(*job)) is not None for job in jobs
        ),
        "expected_prediction_rows": len(predictions) == expected_rows,
        "prediction_keys_unique": not predictions.duplicated(
            ["feature_id", "split_name"]
        ).any(),
        "oof_coverage_exact": coverage_ok,
        "fold_assignments_exact": fold_assignment_ok,
        "calibration_training_is_cross_fitted": bool(
            (fold_metrics["meta_training_current_fold_count"] == 0).all()
            and (fold_metrics["validation_wrong_fold_count"] == 0).all()
        ),
        "quality_quantiles_finite_bounded": bool(
            np.isfinite(quality).all()
            and np.logical_and(quality >= 0.0, quality <= 1.0).all()
        ),
        "quality_quantiles_nondecreasing": bool((np.diff(quality, axis=1) >= 0.0).all()),
        "runtime_quantiles_finite_positive": bool(
            np.isfinite(runtime).all() and (runtime > 0.0).all()
        ),
        "runtime_quantiles_nondecreasing": bool((np.diff(runtime, axis=1) >= 0.0).all()),
        "reach_probabilities_finite_bounded": bool(
            np.isfinite(reach).all()
            and np.logical_and(reach >= 0.0, reach <= 1.0).all()
        ),
        "reach_probabilities_nondecreasing": bool((np.diff(reach, axis=1) >= 0.0).all()),
        "all_probabilities_finite_bounded": bool(
            np.isfinite(probabilities).all()
            and np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all()
        ),
        "calibration_levels_finite_bounded": bool(
            np.isfinite(calibration_levels).all()
            and np.logical_and(calibration_levels >= 0.0, calibration_levels <= 1.0).all()
        ),
        "calibration_support_components_exact": support_components_exact,
        "copula_correlation_matrices_valid": _copula_support_valid(support),
        "failure_dependence_fallback_explicit": fallback_explicit,
        "joint_probabilities_nested": joint_nested,
        "higher_quality_joint_probability_nondecreasing": bool((q90 <= q75 + 1e-12).all()),
        "joint_nll_finite": bool(
            np.isfinite(predictions[["joint_nll", "reference_joint_nll"]].to_numpy()).all()
        ),
        "required_metrics_finite": metrics_finite,
        "aggregate_metrics_complete": len(aggregate) == expected_aggregate_rows,
        "fold_calibration_contract_complete": len(fold_schemas["folds"]) == len(jobs),
        "phase13_scope_exact": registry["joint_model"] == JOINT_MODEL_NAME,
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    issues = []
    if failed_checks:
        issues.append({"type": "failed_quality_checks", "checks": failed_checks})
    return {
        "status": "PHASE_13_PASS" if all(checks.values()) else "PHASE_13_FAIL",
        "schema_version": PHASE13_SCHEMA_VERSION,
        "scope": "cross-fitted marginal recalibration and joint Gaussian-copula modelling",
        "performance_gate_policy": (
            "construction, cross-fitting, calibration validity, joint coherence, OOF coverage, "
            "and integrity only; metric improvements are recorded but deferred to Phase 16"
        ),
        "phase14_boundary": "decision utility and algorithm selection are deferred to Phase 14",
        "phase6_directory": str(paths[0].resolve()),
        "phase7_directory": str(paths[1].resolve()),
        "phase8_directory": str(paths[2].resolve()),
        "phase9_directory": str(paths[3].resolve()),
        "phase10_directory": str(paths[4].resolve()),
        "phase11_directory": str(paths[5].resolve()),
        "phase12_directory": str(paths[6].resolve()),
        "feature_row_count": int(len(features)),
        "prediction_row_count": int(len(predictions)),
        "job_count": int(len(fold_metrics)),
        "model_artifact_count": int(len(registry["jobs"])),
        "aggregate_metric_row_count": int(len(aggregate)),
        "calibration_support_row_count": int(len(support)),
        "joint_quality_thresholds": list(JOINT_QUALITY_THRESHOLDS),
        "failure_dependence_estimability": (
            "UNAVAILABLE_NO_OBSERVED_FAILURES"
            if source_has_no_failures
            else "AVAILABLE_OR_FOLD_SPECIFIC_FALLBACK"
        ),
        "checks": checks,
        "issues": issues,
    }


def run_phase13(
    run_directory: str | Path,
    phase6_directory: str | Path,
    phase7_directory: str | Path,
    phase8_directory: str | Path,
    phase9_directory: str | Path,
    phase10_directory: str | Path,
    phase11_directory: str | Path,
    phase12_directory: str | Path,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
    calibration_strength: float = DEFAULT_CALIBRATION_STRENGTH,
    minimum_class_rows: int = DEFAULT_MINIMUM_CLASS_ROWS,
    copula_shrinkage: float = DEFAULT_COPULA_SHRINKAGE,
    beta_prior: tuple[float, float] = DEFAULT_BETA_PRIOR,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    paths = (
        Path(phase6_directory),
        Path(phase7_directory),
        Path(phase8_directory),
        Path(phase9_directory),
        Path(phase10_directory),
        Path(phase11_directory),
        Path(phase12_directory),
    )
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"
    if calibration_strength < 0.0:
        raise ValueError("Phase 13 calibration strength must be non-negative")
    if minimum_class_rows < 1:
        raise ValueError("Phase 13 minimum class rows must be positive")
    if copula_shrinkage < 0.0:
        raise ValueError("Phase 13 copula shrinkage must be non-negative")
    if len(beta_prior) != 2 or any(value <= 0.0 for value in beta_prior):
        raise ValueError("Phase 13 beta prior alpha and beta must be positive")
    source = _load_inputs(*paths)
    stable_configuration = {
        "phase": 13,
        "schema_version": PHASE13_SCHEMA_VERSION,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "phase12_implementation_sha256": _file_sha256(Path(phase12_module.__file__)),
        "input_fingerprints": source["input_fingerprints"],
        "master_seed": int(master_seed),
        "joint_model": JOINT_MODEL_NAME,
        "calibration_strength": float(calibration_strength),
        "minimum_class_rows": int(minimum_class_rows),
        "copula_shrinkage": float(copula_shrinkage),
        "beta_prior": {"alpha": float(beta_prior[0]), "beta": float(beta_prior[1])},
        "quantile_levels": list(QUANTILE_LEVELS),
        "survival_horizons": list(SURVIVAL_HORIZONS),
        "joint_quality_thresholds": list(JOINT_QUALITY_THRESHOLDS),
        "split_columns": source["feature_schema"]["split_columns"],
        "cutoffs": source["feature_schema"]["cutoffs"],
    }
    configuration_sha256 = _configuration_hash(stable_configuration)
    configuration = {
        **stable_configuration,
        "phase6_directory": str(paths[0].resolve()),
        "phase7_directory": str(paths[1].resolve()),
        "phase8_directory": str(paths[2].resolve()),
        "phase9_directory": str(paths[3].resolve()),
        "phase10_directory": str(paths[4].resolve()),
        "phase11_directory": str(paths[5].resolve()),
        "phase12_directory": str(paths[6].resolve()),
        "git_commit": current_git_commit(),
        "configuration_sha256": configuration_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Phase 13 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_13_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 13 resume manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest.get("output_sha256", {}).items():
                path = run_path / relative
                if not path.is_file() or _file_sha256(path) != expected_hash:
                    raise ValueError(f"completed Phase 13 output hash mismatch: {relative}")
            active_logger.info("[PHASE13][RESUME] status=PHASE_13_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info(
            "[PHASE13][RESUME] completed_jobs=%s", len(state.get("completed_jobs", {}))
        )
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE13_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": configuration_sha256,
            "input_fingerprints": source["input_fingerprints"],
            "completed_jobs": {},
            "last_completed_job": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase13_started"})
    fold_schemas, job_outputs = _run_jobs(
        run_path,
        source["features"],
        source["phase12_predictions"],
        source["splits"],
        source["feature_schema"],
        source["preprocessing"],
        state,
        master_seed=master_seed,
        calibration_strength=calibration_strength,
        minimum_class_rows=minimum_class_rows,
        copula_shrinkage=copula_shrinkage,
        beta_prior=beta_prior,
        progress_path=progress_path,
        logger=active_logger,
    )
    jobs = _all_jobs(source["preprocessing"])
    aggregate_outputs, _registry = _aggregate_job_outputs(
        run_path, jobs, source["phase12_labels"]
    )
    validation = _validate_phase13(
        run_path, paths, source["input_fingerprints"], source
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
        "phase": 13,
        "status": validation["status"],
        "schema_version": PHASE13_SCHEMA_VERSION,
        "input_fingerprints": source["input_fingerprints"],
        "output_sha256": _relative_hashes(run_path, output_paths),
        "row_counts": {
            "features": validation["feature_row_count"],
            "predictions": validation["prediction_row_count"],
            "jobs": validation["job_count"],
            "aggregate_metrics": validation["aggregate_metric_row_count"],
            "calibration_support": validation["calibration_support_row_count"],
        },
        "phase14_allowed": validation["status"] == "PHASE_13_PASS",
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
    _append_progress(
        progress_path, {"event": "phase13_finished", "status": validation["status"]}
    )
    active_logger.info(
        "[PHASE13][SUMMARY] status=%s jobs=%s predictions=%s calibration_support=%s directory=%s",
        validation["status"],
        validation["job_count"],
        validation["prediction_row_count"],
        validation["calibration_support_row_count"],
        run_path,
    )
    return validation
