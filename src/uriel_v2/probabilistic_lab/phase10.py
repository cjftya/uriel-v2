from __future__ import annotations

import hashlib
import json
import logging
import math
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

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
from uriel_v2.probabilistic_lab.phase9 import (
    _combined_input_fingerprints as _phase9_input_fingerprints,
)
from uriel_v2.probabilistic_lab.phase9 import _load_failure_labels, _validate_phase8_input
from uriel_v2.provenance import current_git_commit


PHASE10_SCHEMA_VERSION = "phase10-v1"
RUNTIME_MODEL_NAME = "log_quantile_gradient_boosting"
SURVIVAL_MODEL_NAME = "discrete_horizon_gradient_boosting"
RUNTIME_TARGET_COLUMN = "target_runtime"
FIRST_PASSAGE_TARGET_COLUMN = "target_first_passage_time"
QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
SURVIVAL_HORIZONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
DEFAULT_BETA_PRIOR = (0.5, 0.5)
PROBABILITY_EPSILON = 1e-12


def _quantile_column(level: float) -> str:
    return f"runtime_q{int(round(level * 100.0)):02d}"


def _horizon_suffix(horizon: float) -> str:
    return f"p{int(round(horizon * 100.0)):03d}"


def _reach_column(horizon: float) -> str:
    return f"reach_by_{_horizon_suffix(horizon)}"


def _survival_column(horizon: float) -> str:
    return f"survival_{_horizon_suffix(horizon)}"


def _phase9_required_paths(phase9_path: Path) -> dict[str, Path]:
    return {
        "phase9_config": phase9_path / "config.json",
        "phase9_manifest": phase9_path / "manifest.json",
        "phase9_validation": phase9_path / "validation.json",
        "phase9_binary_predictions": phase9_path
        / "data/predictions/oof_failure_probability.parquet",
        "phase9_type_predictions": phase9_path
        / "data/predictions/oof_failure_type_probability.parquet",
        "phase9_labels": phase9_path / "data/targets/failure_labels.parquet",
        "phase9_fold_metrics": phase9_path / "data/metrics/fold_failure_metrics.parquet",
        "phase9_aggregate_metrics": phase9_path
        / "data/metrics/aggregate_failure_metrics.parquet",
        "phase9_calibration": phase9_path / "data/calibration/failure_calibration.parquet",
        "phase9_support": phase9_path / "data/support/fold_class_support.parquet",
        "phase9_fold_schemas": phase9_path
        / "data/preprocessing/fold_feature_schemas.json",
        "phase9_model_registry": phase9_path / "model_registry.json",
    }


def _validate_phase9_input(
    phase9_path: Path,
    expected_input_fingerprints: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    required = _phase9_required_paths(phase9_path)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 10 Phase 9 input files are missing: {missing[:5]}")
    validation = json.loads(required["phase9_validation"].read_text(encoding="utf-8"))
    manifest = json.loads(required["phase9_manifest"].read_text(encoding="utf-8"))
    if validation.get("status") != "PHASE_9_PASS":
        raise ValueError("Phase 10 requires a PHASE_9_PASS failure-distribution run")
    if manifest.get("status") != "PHASE_9_PASS" or not manifest.get("phase10_allowed"):
        raise ValueError("Phase 9 manifest does not allow Phase 10")
    if validation.get("configuration", {}).get("input_fingerprints") != expected_input_fingerprints:
        raise ValueError("Phase 9 was not built from the supplied Phase 6, 7, and 8 inputs")
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = phase9_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 9 manifest output hash mismatch: {relative}")
    fingerprints = {name: _file_sha256(path) for name, path in sorted(required.items())}
    return validation, manifest, fingerprints


def _combined_input_fingerprints(
    phase9_expected_inputs: dict[str, str],
    phase9_fingerprints: dict[str, str],
    runtime_label_source_sha256: str,
) -> dict[str, str]:
    return {
        **phase9_expected_inputs,
        **{f"phase9/{name}": value for name, value in phase9_fingerprints.items()},
        "phase4/runtime_survival_label_source": runtime_label_source_sha256,
    }


def _load_runtime_survival_labels(
    phase6_path: Path,
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, Path, str]:
    phase6_config = json.loads((phase6_path / "config.json").read_text(encoding="utf-8"))
    phase4_path = Path(phase6_config["phase4_directory"])
    runs_path = phase4_path / "data/runs/runs.parquet"
    if not runs_path.is_file():
        raise FileNotFoundError(f"Phase 10 runtime label source is missing: {runs_path}")
    source_sha256 = _file_sha256(runs_path)
    expected_sha256 = phase6_config.get("input_fingerprints", {}).get("phase4_runs")
    if source_sha256 != expected_sha256:
        raise ValueError("Phase 10 runtime label source no longer matches the frozen Phase 6 input")
    runs = pd.read_parquet(
        runs_path,
        columns=[
            "run_id",
            "runtime",
            "target_reached",
            "first_passage_time",
            "budget",
            "steps",
            "timeout",
            "failure",
        ],
    )
    if not runs["run_id"].is_unique:
        raise ValueError("Phase 10 runtime label source must have unique run_id values")
    indexed = runs.set_index(runs["run_id"].astype(str))
    run_ids = targets["run_id"].astype(str)
    source = pd.DataFrame(
        {column: run_ids.map(indexed[column]) for column in runs.columns if column != "run_id"}
    )
    if source[["runtime", "target_reached", "budget", "steps"]].isna().any().any():
        raise ValueError("Phase 10 runtime label source does not cover every Phase 6 target")
    runtime = pd.to_numeric(source["runtime"], errors="coerce").to_numpy(dtype=float)
    target_runtime = pd.to_numeric(targets[RUNTIME_TARGET_COLUMN], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(runtime).all()
        or (runtime <= 0.0).any()
        or not np.allclose(runtime, target_runtime, rtol=0.0, atol=1e-15)
    ):
        raise ValueError("Phase 10 Phase 4 and Phase 6 runtime labels disagree")
    event = source["target_reached"].astype(bool).to_numpy()
    first_passage = pd.to_numeric(source["first_passage_time"], errors="coerce").to_numpy(
        dtype=float
    )
    target_passage = pd.to_numeric(
        targets[FIRST_PASSAGE_TARGET_COLUMN], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.allclose(first_passage, target_passage, equal_nan=True):
        raise ValueError("Phase 10 Phase 4 and Phase 6 first-passage labels disagree")
    if not np.array_equal(event, np.isfinite(first_passage)):
        raise ValueError("Phase 10 target_reached must exactly match first-passage availability")
    budget = pd.to_numeric(source["budget"], errors="coerce").to_numpy(dtype=float)
    steps = pd.to_numeric(source["steps"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(budget).all() or (budget <= 0.0).any() or not np.isfinite(steps).all():
        raise ValueError("Phase 10 budget and step labels must be finite with positive budgets")
    censor_step = np.minimum(np.maximum(steps, 0.0), budget)
    duration_step = np.where(event, first_passage, censor_step)
    if (
        not np.isfinite(duration_step).all()
        or (duration_step < 0.0).any()
        or (duration_step > budget).any()
    ):
        raise ValueError("Phase 10 survival durations must be inside the execution budget")
    labels = targets[["feature_id", "run_id", "problem_id", "cutoff"]].copy()
    labels["runtime_seconds"] = runtime
    labels["event_observed"] = event
    labels["first_passage_step"] = first_passage
    labels["censor_step"] = censor_step
    labels["duration_step"] = duration_step
    labels["budget"] = budget
    labels["duration_fraction"] = duration_step / budget
    labels["timeout"] = source["timeout"].astype(bool).to_numpy()
    labels["failure"] = source["failure"].astype(bool).to_numpy()
    if not np.allclose(
        pd.to_numeric(features["budget"], errors="coerce").to_numpy(dtype=float), budget
    ):
        raise ValueError("Phase 10 feature budget and runtime label budget disagree")
    return labels, runs_path, source_sha256


def _job_id(split_name: str, fold: int) -> str:
    return f"{split_name}__fold{fold}__runtime_survival"


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
            raise ValueError(f"Phase 10 job output hash mismatch: job={job_id} path={relative}")
    return marker


def _job_seed(master_seed: int, job_id: str, target: str, level: float) -> int:
    token = f"{job_id}:{target}:{level:.6f}"
    offset = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
    return int((master_seed + offset) % (2**31 - 1))


def _runtime_support(runtime_quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logs = np.log(np.maximum(runtime_quantiles, 1e-12))
    lower_width = np.maximum(logs[:, 1] - logs[:, 0], 0.25)
    upper_width = np.maximum(logs[:, -1] - logs[:, -2], 0.25)
    support_logs = np.column_stack((logs[:, 0] - lower_width, logs, logs[:, -1] + upper_width))
    probabilities = np.asarray((0.001, *QUANTILE_LEVELS, 0.999), dtype=float)
    return support_logs, probabilities


def _runtime_cdf(runtime_quantiles: np.ndarray, points: np.ndarray) -> np.ndarray:
    support, probabilities = _runtime_support(runtime_quantiles)
    point_logs = np.log(np.maximum(np.asarray(points, dtype=float), 1e-12))
    indices = (support <= point_logs[:, None]).sum(axis=1) - 1
    indices = np.clip(indices, 0, support.shape[1] - 2)
    rows = np.arange(len(support))
    left = support[rows, indices]
    right = support[rows, indices + 1]
    width = right - left
    fraction = np.divide(
        point_logs - left, width, out=np.zeros_like(point_logs), where=width > 1e-12
    )
    cdf = probabilities[indices] + np.clip(fraction, 0.0, 1.0) * (
        probabilities[indices + 1] - probabilities[indices]
    )
    cdf[point_logs <= support[:, 0]] = 0.0
    cdf[point_logs >= support[:, -1]] = 1.0
    return np.clip(cdf, 0.0, 1.0)


def _runtime_nll(observed: np.ndarray, runtime_quantiles: np.ndarray) -> np.ndarray:
    support, probabilities = _runtime_support(runtime_quantiles)
    observed = np.maximum(np.asarray(observed, dtype=float), 1e-12)
    point_logs = np.log(observed)
    indices = (support <= point_logs[:, None]).sum(axis=1) - 1
    indices = np.clip(indices, 0, support.shape[1] - 2)
    rows = np.arange(len(support))
    log_width = support[rows, indices + 1] - support[rows, indices]
    probability_width = probabilities[indices + 1] - probabilities[indices]
    density_log = probability_width / np.maximum(log_width, 1e-6)
    return -np.log(np.clip(density_log, PROBABILITY_EPSILON, 1e12)) + np.log(observed)


def _runtime_crps(observed: np.ndarray, runtime_quantiles: np.ndarray) -> np.ndarray:
    support_logs, probabilities = _runtime_support(runtime_quantiles)
    support = np.exp(np.clip(support_logs, -30.0, 30.0))
    residual = np.asarray(observed, dtype=float)[:, None] - support
    pinball = np.maximum(
        probabilities[None, :] * residual,
        (probabilities[None, :] - 1.0) * residual,
    )
    return 2.0 * np.trapezoid(pinball, probabilities, axis=1)


def _runtime_moments(runtime_quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support_logs, probabilities = _runtime_support(runtime_quantiles)
    support = np.exp(np.clip(support_logs, -30.0, 30.0))
    mean = np.trapezoid(support, probabilities, axis=1)
    second = np.trapezoid(support**2, probabilities, axis=1)
    return mean, np.sqrt(np.maximum(second - mean**2, 0.0))


def _runtime_metrics(observed: np.ndarray, runtime_quantiles: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    predictive_mean, _ = _runtime_moments(runtime_quantiles)
    calibration_errors = []
    result: dict[str, Any] = {
        "runtime_nll": float(_runtime_nll(observed, runtime_quantiles).mean()),
        "runtime_crps": float(_runtime_crps(observed, runtime_quantiles).mean()),
        "runtime_pit_mean": float(_runtime_cdf(runtime_quantiles, observed).mean()),
        "runtime_pit_variance": float(_runtime_cdf(runtime_quantiles, observed).var()),
        "runtime_coverage_80": float(
            np.mean((observed >= runtime_quantiles[:, 1]) & (observed <= runtime_quantiles[:, 5]))
        ),
        "runtime_coverage_90": float(
            np.mean((observed >= runtime_quantiles[:, 0]) & (observed <= runtime_quantiles[:, 6]))
        ),
        "runtime_interval_width_80": float(
            np.mean(runtime_quantiles[:, 5] - runtime_quantiles[:, 1])
        ),
        "runtime_interval_width_90": float(
            np.mean(runtime_quantiles[:, 6] - runtime_quantiles[:, 0])
        ),
        "runtime_median_mae": float(np.mean(np.abs(observed - runtime_quantiles[:, 3]))),
        "runtime_median_rmse": float(
            math.sqrt(np.mean((observed - runtime_quantiles[:, 3]) ** 2))
        ),
        "runtime_predictive_mean_mae": float(np.mean(np.abs(observed - predictive_mean))),
    }
    result["runtime_coverage_error_80"] = abs(result["runtime_coverage_80"] - 0.80)
    result["runtime_coverage_error_90"] = abs(result["runtime_coverage_90"] - 0.90)
    for index, level in enumerate(QUANTILE_LEVELS):
        empirical = float(np.mean(observed <= runtime_quantiles[:, index]))
        error = empirical - level
        result[f"runtime_empirical_cdf_{_quantile_column(level)}"] = empirical
        result[f"runtime_calibration_error_{_quantile_column(level)}"] = error
        calibration_errors.append(abs(error))
    result["runtime_calibration_mae"] = float(np.mean(calibration_errors))
    result["runtime_calibration_max_abs"] = float(np.max(calibration_errors))
    return result


def _horizon_steps(budget: np.ndarray, horizon: float) -> np.ndarray:
    budget = np.asarray(budget, dtype=float)
    return np.minimum(budget, np.maximum(1.0, np.ceil(budget * horizon)))


def _horizon_observation(
    event: np.ndarray,
    duration_step: np.ndarray,
    budget: np.ndarray,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    event = np.asarray(event, dtype=bool)
    duration_step = np.asarray(duration_step, dtype=float)
    step = _horizon_steps(budget, horizon)
    outcome = event & (duration_step <= step)
    observable = outcome | (duration_step >= step)
    return observable, outcome


def _fit_runtime_distribution(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    runtime_train: np.ndarray,
    *,
    job_id: str,
    master_seed: int,
    gradient_boosting_iterations: int,
) -> tuple[dict[str, HistGradientBoostingRegressor], np.ndarray, dict[str, Any]]:
    log_runtime = np.log(np.maximum(np.asarray(runtime_train, dtype=float), 1e-12))
    estimators: dict[str, HistGradientBoostingRegressor] = {}
    raw_log_predictions = np.empty((len(x_validation), len(QUANTILE_LEVELS)), dtype=float)
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
            random_state=_job_seed(master_seed, job_id, "runtime", level),
        )
        estimator.fit(x_train, log_runtime)
        estimators[_quantile_column(level)] = estimator
        raw_log_predictions[:, index] = estimator.predict(x_validation)
    differences = np.diff(raw_log_predictions, axis=1)
    crossing_rows = np.any(differences < 0.0, axis=1)
    sorted_predictions = np.sort(raw_log_predictions, axis=1)
    runtime_quantiles = np.exp(np.clip(sorted_predictions, -30.0, 30.0))
    crossing = {
        "runtime_raw_crossing_row_count": int(crossing_rows.sum()),
        "runtime_raw_crossing_adjacent_pair_count": int((differences < 0.0).sum()),
        "runtime_maximum_raw_crossing_violation": float(
            np.maximum(-differences, 0.0).max(initial=0.0)
        ),
        "runtime_postprocessing": "row-wise increasing rearrangement in log-runtime space",
    }
    return estimators, runtime_quantiles, crossing


def _fit_survival_distribution(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    event_train: np.ndarray,
    duration_train: np.ndarray,
    budget_train: np.ndarray,
    *,
    job_id: str,
    master_seed: int,
    gradient_boosting_iterations: int,
    beta_prior: tuple[float, float],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], pd.DataFrame]:
    estimators: dict[str, HistGradientBoostingClassifier | None] = {}
    statuses: dict[str, str] = {}
    constants: dict[str, float | None] = {}
    raw = np.empty((len(x_validation), len(SURVIVAL_HORIZONS)), dtype=float)
    support_rows = []
    alpha, beta = beta_prior
    for index, horizon in enumerate(SURVIVAL_HORIZONS):
        observable, outcome = _horizon_observation(
            event_train, duration_train, budget_train, horizon
        )
        observed_outcome = outcome[observable]
        suffix = _horizon_suffix(horizon)
        if not observable.any() or np.unique(observed_outcome).size < 2:
            probability = float(
                (int(observed_outcome.sum()) + alpha)
                / (len(observed_outcome) + alpha + beta)
            )
            raw[:, index] = probability
            estimators[suffix] = None
            statuses[suffix] = "beta_binomial_fallback"
            constants[suffix] = probability
        else:
            estimator = HistGradientBoostingClassifier(
                loss="log_loss",
                max_iter=gradient_boosting_iterations,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=_job_seed(master_seed, job_id, "survival", horizon),
            )
            estimator.fit(x_train[observable], observed_outcome.astype(int))
            positive_index = list(estimator.classes_).index(1)
            raw[:, index] = estimator.predict_proba(x_validation)[:, positive_index]
            estimators[suffix] = estimator
            statuses[suffix] = "fitted"
            constants[suffix] = None
        support_rows.append(
            {
                "horizon": horizon,
                "horizon_suffix": suffix,
                "observable_training_count": int(observable.sum()),
                "event_by_horizon_training_count": int(outcome[observable].sum()),
                "censored_before_horizon_training_count": int((~observable).sum()),
                "fit_status": statuses[suffix],
            }
        )
    raw = np.clip(raw, 0.0, 1.0)
    differences = np.diff(raw, axis=1)
    crossing_rows = np.any(differences < 0.0, axis=1)
    cumulative = np.maximum.accumulate(raw, axis=1)
    crossing = {
        "survival_raw_crossing_row_count": int(crossing_rows.sum()),
        "survival_raw_crossing_adjacent_pair_count": int((differences < 0.0).sum()),
        "survival_maximum_raw_crossing_violation": float(
            np.maximum(-differences, 0.0).max(initial=0.0)
        ),
        "survival_postprocessing": "row-wise cumulative maximum of reach probabilities",
    }
    model = {
        "estimators": estimators,
        "fit_statuses": statuses,
        "constant_probabilities": constants,
        "beta_prior": {"alpha": alpha, "beta": beta},
    }
    return model, cumulative, crossing, pd.DataFrame(support_rows)


def _harrell_c_index(
    duration_fraction: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> tuple[float | None, int]:
    duration = np.asarray(duration_fraction, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    concordant = 0.0
    comparable = 0
    for value in np.unique(duration[event]):
        event_risk = risk[event & np.isclose(duration, value, rtol=0.0, atol=1e-12)]
        later_risk = np.sort(risk[duration > value + 1e-12])
        if not len(event_risk) or not len(later_risk):
            continue
        later_list = later_risk.tolist()
        for score in event_risk:
            lower = bisect_left(later_list, score)
            upper = bisect_right(later_list, score)
            concordant += lower + 0.5 * (upper - lower)
            comparable += len(later_list)
    return (
        (float(concordant / comparable) if comparable else None),
        int(comparable),
    )


def _survival_summaries(
    reach_probability: np.ndarray,
    budget: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    budget = np.asarray(budget, dtype=float)
    steps = np.column_stack([_horizon_steps(budget, value) for value in SURVIVAL_HORIZONS])
    previous_steps = np.column_stack((np.zeros(len(budget)), steps[:, :-1]))
    previous_survival = np.column_stack(
        (np.ones(len(budget)), 1.0 - reach_probability[:, :-1])
    )
    restricted_mean_steps = np.sum(previous_survival * (steps - previous_steps), axis=1)
    increments = np.diff(
        np.column_stack((np.zeros(len(budget)), reach_probability)), axis=1
    )
    expected_par10 = np.sum(increments * steps, axis=1) + (
        1.0 - reach_probability[:, -1]
    ) * (10.0 * budget)
    return restricted_mean_steps, expected_par10


def _survival_nll(
    event: np.ndarray,
    duration_step: np.ndarray,
    budget: np.ndarray,
    reach_probability: np.ndarray,
) -> np.ndarray:
    event = np.asarray(event, dtype=bool)
    duration = np.asarray(duration_step, dtype=float)
    budget = np.asarray(budget, dtype=float)
    steps = np.column_stack([_horizon_steps(budget, value) for value in SURVIVAL_HORIZONS])
    increments = np.diff(
        np.column_stack((np.zeros(len(event)), reach_probability)), axis=1
    )
    likelihood = np.ones(len(event), dtype=float)
    for row in range(len(event)):
        if event[row]:
            candidates = np.flatnonzero(steps[row] >= duration[row])
            likelihood[row] = increments[row, candidates[0]] if len(candidates) else 0.0
        else:
            observed_horizons = np.flatnonzero(steps[row] <= duration[row])
            if len(observed_horizons):
                likelihood[row] = 1.0 - reach_probability[row, observed_horizons[-1]]
    return -np.log(np.clip(likelihood, PROBABILITY_EPSILON, 1.0))


def _survival_metrics(
    event: np.ndarray,
    duration_step: np.ndarray,
    budget: np.ndarray,
    reach_probability: np.ndarray,
) -> dict[str, Any]:
    event = np.asarray(event, dtype=bool)
    duration = np.asarray(duration_step, dtype=float)
    budget = np.asarray(budget, dtype=float)
    brier_values = []
    calibration_values = []
    result: dict[str, Any] = {
        "event_count": int(event.sum()),
        "event_rate": float(event.mean()),
        "survival_nll": float(
            _survival_nll(event, duration, budget, reach_probability).mean()
        ),
    }
    for index, horizon in enumerate(SURVIVAL_HORIZONS):
        observable, outcome = _horizon_observation(event, duration, budget, horizon)
        suffix = _horizon_suffix(horizon)
        if observable.any():
            probability = reach_probability[observable, index]
            numeric = outcome[observable].astype(float)
            brier = float(np.mean((numeric - probability) ** 2))
            empirical = float(numeric.mean())
            predicted = float(probability.mean())
            error = predicted - empirical
            brier_values.append(brier)
            calibration_values.append(abs(error))
            result[f"survival_brier_{suffix}"] = brier
            result[f"survival_empirical_reach_{suffix}"] = empirical
            result[f"survival_predicted_reach_{suffix}"] = predicted
            result[f"survival_calibration_error_{suffix}"] = error
            result[f"survival_observable_count_{suffix}"] = int(observable.sum())
        else:
            result[f"survival_brier_{suffix}"] = None
            result[f"survival_empirical_reach_{suffix}"] = None
            result[f"survival_predicted_reach_{suffix}"] = None
            result[f"survival_calibration_error_{suffix}"] = None
            result[f"survival_observable_count_{suffix}"] = 0
    result["integrated_brier"] = float(np.mean(brier_values))
    result["survival_calibration_mae"] = float(np.mean(calibration_values))
    result["survival_calibration_max_abs"] = float(np.max(calibration_values))
    restricted_mean, expected_par10 = _survival_summaries(reach_probability, budget)
    observed_par10 = np.where(event, duration, 10.0 * budget)
    risk = -restricted_mean / budget
    c_index, comparable_pairs = _harrell_c_index(duration / budget, event, risk)
    result.update(
        {
            "c_index": c_index,
            "c_index_comparable_pair_count": comparable_pairs,
            "observed_par10_mean": float(observed_par10.mean()),
            "predicted_par10_mean": float(expected_par10.mean()),
            "par10_mae": float(np.mean(np.abs(observed_par10 - expected_par10))),
            "observed_reach_by_budget": float(event.mean()),
            "predicted_reach_by_budget": float(reach_probability[:, -1].mean()),
            "restricted_mean_step_mean": float(restricted_mean.mean()),
        }
    )
    return result


def _prediction_frame(
    validation_features: pd.DataFrame,
    validation_labels: pd.DataFrame,
    runtime_quantiles: np.ndarray,
    reach_probability: np.ndarray,
    *,
    split_name: str,
    fold: int,
) -> pd.DataFrame:
    frame = validation_features[["feature_id", "cutoff"]].reset_index(drop=True).copy()
    labels = validation_labels.reset_index(drop=True)
    observed_runtime = labels["runtime_seconds"].to_numpy(dtype=float)
    budget = labels["budget"].to_numpy(dtype=float)
    event = labels["event_observed"].to_numpy(dtype=bool)
    duration = labels["duration_step"].to_numpy(dtype=float)
    predictive_mean, predictive_std = _runtime_moments(runtime_quantiles)
    restricted_mean, expected_par10 = _survival_summaries(reach_probability, budget)
    frame["split_name"] = split_name
    frame["fold"] = fold
    frame["observed_runtime"] = observed_runtime
    frame["event_observed"] = event
    frame["first_passage_step"] = labels["first_passage_step"]
    frame["censor_step"] = labels["censor_step"].to_numpy(dtype=float)
    frame["duration_step"] = duration
    frame["duration_fraction"] = labels["duration_fraction"].to_numpy(dtype=float)
    frame["budget"] = budget
    for index, level in enumerate(QUANTILE_LEVELS):
        frame[_quantile_column(level)] = runtime_quantiles[:, index]
    frame["runtime_predictive_mean"] = predictive_mean
    frame["runtime_predictive_std"] = predictive_std
    frame["runtime_pit"] = _runtime_cdf(runtime_quantiles, observed_runtime)
    for index, horizon in enumerate(SURVIVAL_HORIZONS):
        frame[_reach_column(horizon)] = reach_probability[:, index]
        frame[_survival_column(horizon)] = 1.0 - reach_probability[:, index]
    frame["probability_reach_by_budget"] = reach_probability[:, -1]
    frame["probability_not_reached_by_budget"] = 1.0 - reach_probability[:, -1]
    frame["predicted_restricted_mean_step"] = restricted_mean
    frame["observed_par10"] = np.where(event, duration, 10.0 * budget)
    frame["predicted_par10"] = expected_par10
    return frame


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
    state: dict[str, Any],
    *,
    master_seed: int,
    gradient_boosting_iterations: int,
    beta_prior: tuple[float, float],
    progress_path: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[Path]]:
    fold_schemas: dict[str, Any] = {
        "schema_version": PHASE10_SCHEMA_VERSION,
        "source_preprocessing_schema_version": preprocessing["schema_version"],
        "transform_policy": (
            "reuse Phase 6 training-fold preprocessing through the verified Phase 7 transformer"
        ),
        "folds": {},
    }
    output_paths: list[Path] = []
    runtime_values = labels["runtime_seconds"].to_numpy(dtype=float)
    event_values = labels["event_observed"].to_numpy(dtype=bool)
    duration_values = labels["duration_step"].to_numpy(dtype=float)
    budget_values = labels["budget"].to_numpy(dtype=float)
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
                raise ValueError(f"Phase 10 transformed schema mismatch: {fold_key}")
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
                logger.info("[PHASE10][RESUME] job=%s status=SKIP_VERIFIED", job_id)
                continue
            logger.info(
                "[PHASE10][FOLD] split=%s fold=%s train_rows=%s validation_rows=%s train_events=%s",
                split_name,
                fold,
                int(training_mask.sum()),
                int(validation_mask.sum()),
                int(event_values[training_mask].sum()),
            )
            runtime_estimators, runtime_quantiles, runtime_crossing = _fit_runtime_distribution(
                x_train,
                x_validation,
                runtime_values[training_mask],
                job_id=job_id,
                master_seed=master_seed,
                gradient_boosting_iterations=gradient_boosting_iterations,
            )
            survival_model, reach_probability, survival_crossing, support = (
                _fit_survival_distribution(
                    x_train,
                    x_validation,
                    event_values[training_mask],
                    duration_values[training_mask],
                    budget_values[training_mask],
                    job_id=job_id,
                    master_seed=master_seed,
                    gradient_boosting_iterations=gradient_boosting_iterations,
                    beta_prior=beta_prior,
                )
            )
            predictions = _prediction_frame(
                features.loc[validation_mask],
                labels.loc[validation_mask],
                runtime_quantiles,
                reach_probability,
                split_name=split_name,
                fold=fold,
            )
            validation_support = []
            for horizon in SURVIVAL_HORIZONS:
                observable, outcome = _horizon_observation(
                    event_values[validation_mask],
                    duration_values[validation_mask],
                    budget_values[validation_mask],
                    horizon,
                )
                validation_support.append(
                    {
                        "observable_validation_count": int(observable.sum()),
                        "event_by_horizon_validation_count": int(outcome[observable].sum()),
                        "censored_before_horizon_validation_count": int((~observable).sum()),
                    }
                )
            support = pd.concat(
                [support.reset_index(drop=True), pd.DataFrame(validation_support)], axis=1
            )
            support.insert(0, "fold", fold)
            support.insert(0, "split_name", split_name)
            support.insert(0, "job_id", job_id)
            metrics = {
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "runtime_model": RUNTIME_MODEL_NAME,
                "survival_model": SURVIVAL_MODEL_NAME,
                "training_row_count": int(training_mask.sum()),
                "validation_row_count": int(validation_mask.sum()),
                "training_problem_count": fold_contract["training_problem_count"],
                "validation_problem_count": fold_contract["validation_problem_count"],
                "training_event_count": int(event_values[training_mask].sum()),
                "validation_event_count": int(event_values[validation_mask].sum()),
                **runtime_crossing,
                **survival_crossing,
                **_runtime_metrics(runtime_values[validation_mask], runtime_quantiles),
                **_survival_metrics(
                    event_values[validation_mask],
                    duration_values[validation_mask],
                    budget_values[validation_mask],
                    reach_probability,
                ),
            }
            model_artifact = {
                "schema_version": PHASE10_SCHEMA_VERSION,
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "runtime_model": RUNTIME_MODEL_NAME,
                "survival_model": SURVIVAL_MODEL_NAME,
                "quantile_levels": list(QUANTILE_LEVELS),
                "survival_horizons": list(SURVIVAL_HORIZONS),
                "feature_names": feature_names,
                "fold_contract": fold_contract,
                "runtime_estimators": runtime_estimators,
                "survival": survival_model,
                "runtime_postprocessing": runtime_crossing["runtime_postprocessing"],
                "survival_postprocessing": survival_crossing["survival_postprocessing"],
            }
            _atomic_parquet(paths["predictions"], predictions)
            _atomic_json(paths["metrics"], metrics)
            _atomic_parquet(paths["support"], support)
            _atomic_pickle(paths["model"], model_artifact)
            marker = {
                "schema_version": PHASE10_SCHEMA_VERSION,
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "output_sha256": _relative_hashes(
                    run_path,
                    (paths["predictions"], paths["metrics"], paths["support"], paths["model"]),
                ),
            }
            _atomic_json(paths["marker"], marker)
            state.setdefault("completed_jobs", {})[job_id] = _file_sha256(paths["marker"])
            state["last_completed_job"] = job_id
            _atomic_json(run_path / "run_state.json", state)
            _append_progress(progress_path, {"event": "job_completed", "job_id": job_id})
            output_paths.extend(paths.values())
            logger.info(
                "[PHASE10][JOB] job=%s runtime_crossing=%s survival_crossing=%s",
                job_id,
                runtime_crossing["runtime_raw_crossing_row_count"],
                survival_crossing["survival_raw_crossing_row_count"],
            )
    fold_schema_path = run_path / "data/preprocessing/fold_feature_schemas.json"
    _atomic_json(fold_schema_path, fold_schemas)
    output_paths.append(fold_schema_path)
    return fold_schemas, output_paths


def _runtime_calibration_rows(
    observed: np.ndarray,
    runtime_quantiles: np.ndarray,
    *,
    scope: str,
    split_name: str,
    cutoff: float | None,
) -> list[dict[str, Any]]:
    rows = []
    for index, level in enumerate(QUANTILE_LEVELS):
        empirical = float(np.mean(observed <= runtime_quantiles[:, index]))
        rows.append(
            {
                "scope": scope,
                "split_name": split_name,
                "cutoff": cutoff,
                "quantile": level,
                "empirical_cdf": empirical,
                "calibration_error": empirical - level,
                "row_count": int(len(observed)),
            }
        )
    return rows


def _survival_calibration_rows(
    event: np.ndarray,
    duration_step: np.ndarray,
    budget: np.ndarray,
    reach_probability: np.ndarray,
    *,
    scope: str,
    split_name: str,
    cutoff: float | None,
) -> list[dict[str, Any]]:
    rows = []
    for index, horizon in enumerate(SURVIVAL_HORIZONS):
        observable, outcome = _horizon_observation(event, duration_step, budget, horizon)
        probability = reach_probability[observable, index]
        empirical = float(outcome[observable].mean()) if observable.any() else None
        predicted = float(probability.mean()) if observable.any() else None
        rows.append(
            {
                "scope": scope,
                "split_name": split_name,
                "cutoff": cutoff,
                "horizon_fraction": horizon,
                "observable_row_count": int(observable.sum()),
                "event_by_horizon_count": int(outcome[observable].sum()),
                "empirical_reach_probability": empirical,
                "predicted_reach_probability": predicted,
                "calibration_error": (
                    predicted - empirical if observable.any() else None
                ),
            }
        )
    return rows


def _aggregate_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quantile_columns = [_quantile_column(level) for level in QUANTILE_LEVELS]
    reach_columns = [_reach_column(horizon) for horizon in SURVIVAL_HORIZONS]
    metric_rows = []
    runtime_calibration = []
    survival_calibration = []
    groups: list[tuple[str, tuple[str, ...]]] = [
        ("overall", ("split_name",)),
        ("cutoff", ("split_name", "cutoff")),
    ]
    for scope, columns in groups:
        for keys, group in predictions.groupby(list(columns), sort=True):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            identifiers = dict(zip(columns, key_values, strict=True))
            split_name = str(identifiers["split_name"])
            cutoff = float(identifiers["cutoff"]) if "cutoff" in identifiers else None
            observed_runtime = group["observed_runtime"].to_numpy(dtype=float)
            runtime_quantiles = group[quantile_columns].to_numpy(dtype=float)
            event = group["event_observed"].to_numpy(dtype=bool)
            duration = group["duration_step"].to_numpy(dtype=float)
            budget = group["budget"].to_numpy(dtype=float)
            reach = group[reach_columns].to_numpy(dtype=float)
            metric_rows.append(
                {
                    "scope": scope,
                    "split_name": split_name,
                    "cutoff": cutoff,
                    "row_count": int(len(group)),
                    "fold_count": int(group["fold"].nunique()),
                    "runtime_model": RUNTIME_MODEL_NAME,
                    "survival_model": SURVIVAL_MODEL_NAME,
                    **_runtime_metrics(observed_runtime, runtime_quantiles),
                    **_survival_metrics(event, duration, budget, reach),
                }
            )
            runtime_calibration.extend(
                _runtime_calibration_rows(
                    observed_runtime,
                    runtime_quantiles,
                    scope=scope,
                    split_name=split_name,
                    cutoff=cutoff,
                )
            )
            survival_calibration.extend(
                _survival_calibration_rows(
                    event,
                    duration,
                    budget,
                    reach,
                    scope=scope,
                    split_name=split_name,
                    cutoff=cutoff,
                )
            )
    aggregate = pd.DataFrame(metric_rows).sort_values(
        ["scope", "split_name", "cutoff"], na_position="first"
    ).reset_index(drop=True)
    runtime_frame = pd.DataFrame(runtime_calibration).sort_values(
        ["scope", "split_name", "cutoff", "quantile"], na_position="first"
    ).reset_index(drop=True)
    survival_frame = pd.DataFrame(survival_calibration).sort_values(
        ["scope", "split_name", "cutoff", "horizon_fraction"], na_position="first"
    ).reset_index(drop=True)
    return aggregate, runtime_frame, survival_frame


def _aggregate_job_outputs(
    run_path: Path,
    jobs: list[tuple[str, int]],
    labels: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any]]:
    prediction_frames = []
    metric_rows = []
    support_frames = []
    registry = {
        "schema_version": PHASE10_SCHEMA_VERSION,
        "targets": ["runtime_distribution", "first_passage_survival"],
        "runtime_model": RUNTIME_MODEL_NAME,
        "survival_model": SURVIVAL_MODEL_NAME,
        "quantile_levels": list(QUANTILE_LEVELS),
        "survival_horizons": list(SURVIVAL_HORIZONS),
        "artifact_policy": "one runtime-quantile and discrete-survival model bundle per split/fold",
        "jobs": [],
    }
    for split_name, fold in jobs:
        job_id = _job_id(split_name, fold)
        marker = _verify_job(run_path, job_id)
        if marker is None:
            raise ValueError(f"Phase 10 job is incomplete: {job_id}")
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
        ["split_name", "fold", "horizon"]
    ).reset_index(drop=True)
    aggregate, runtime_calibration, survival_calibration = _aggregate_metrics(predictions)
    predictions_path = run_path / "data/predictions/oof_runtime_survival.parquet"
    labels_path = run_path / "data/targets/runtime_survival_labels.parquet"
    fold_metrics_path = run_path / "data/metrics/fold_runtime_survival_metrics.parquet"
    aggregate_path = run_path / "data/metrics/aggregate_runtime_survival_metrics.parquet"
    runtime_calibration_path = run_path / "data/calibration/runtime_quantile_calibration.parquet"
    survival_calibration_path = run_path / "data/calibration/survival_calibration.parquet"
    support_path = run_path / "data/support/survival_horizon_support.parquet"
    registry_path = run_path / "model_registry.json"
    _atomic_parquet(predictions_path, predictions)
    _atomic_parquet(labels_path, labels)
    _atomic_parquet(fold_metrics_path, fold_metrics)
    _atomic_parquet(aggregate_path, aggregate)
    _atomic_parquet(runtime_calibration_path, runtime_calibration)
    _atomic_parquet(survival_calibration_path, survival_calibration)
    _atomic_parquet(support_path, support)
    _atomic_json(registry_path, registry)
    return [
        predictions_path,
        labels_path,
        fold_metrics_path,
        aggregate_path,
        runtime_calibration_path,
        survival_calibration_path,
        support_path,
        registry_path,
    ], registry


def _labels_equal(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    if list(first.columns) != list(second.columns) or len(first) != len(second):
        return False
    for column in first.columns:
        if pd.api.types.is_numeric_dtype(first[column]):
            if not np.allclose(
                pd.to_numeric(first[column], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(second[column], errors="coerce").to_numpy(dtype=float),
                equal_nan=True,
            ):
                return False
        elif not first[column].reset_index(drop=True).equals(second[column].reset_index(drop=True)):
            return False
    return True


def _validate_phase10(
    run_path: Path,
    phase6_path: Path,
    phase7_path: Path,
    phase8_path: Path,
    phase9_path: Path,
    original_input_fingerprints: dict[str, str],
    phase6_validation: dict[str, Any],
    phase7_validation: dict[str, Any],
    phase8_validation: dict[str, Any],
    phase9_validation: dict[str, Any],
    features: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
    source_labels: pd.DataFrame,
) -> dict[str, Any]:
    _, _, current_phase6 = _validate_phase6_input(phase6_path)
    _, _, current_phase7 = _validate_phase7_input(phase7_path, current_phase6)
    phase6_phase7 = _phase8_input_fingerprints(current_phase6, current_phase7)
    _, _, current_phase8 = _validate_phase8_input(phase8_path, phase6_phase7)
    _, _, _failure_source, failure_sha = _load_failure_labels(phase6_path, targets)
    expected_phase9 = _phase9_input_fingerprints(
        current_phase6, current_phase7, current_phase8, failure_sha
    )
    _, _, current_phase9 = _validate_phase9_input(phase9_path, expected_phase9)
    current_labels, _runtime_source, runtime_sha = _load_runtime_survival_labels(
        phase6_path, features, targets
    )
    current_inputs = _combined_input_fingerprints(expected_phase9, current_phase9, runtime_sha)
    predictions = pd.read_parquet(run_path / "data/predictions/oof_runtime_survival.parquet")
    saved_labels = pd.read_parquet(run_path / "data/targets/runtime_survival_labels.parquet")
    fold_metrics = pd.read_parquet(
        run_path / "data/metrics/fold_runtime_survival_metrics.parquet"
    )
    aggregate = pd.read_parquet(
        run_path / "data/metrics/aggregate_runtime_survival_metrics.parquet"
    )
    runtime_calibration = pd.read_parquet(
        run_path / "data/calibration/runtime_quantile_calibration.parquet"
    )
    survival_calibration = pd.read_parquet(
        run_path / "data/calibration/survival_calibration.parquet"
    )
    support = pd.read_parquet(run_path / "data/support/survival_horizon_support.parquet")
    registry = json.loads((run_path / "model_registry.json").read_text(encoding="utf-8"))
    fold_schemas = json.loads(
        (run_path / "data/preprocessing/fold_feature_schemas.json").read_text(encoding="utf-8")
    )
    jobs = _all_jobs(preprocessing)
    expected_rows = len(features) * len(preprocessing["splits"])
    coverage_ok = True
    for split_name in preprocessing["splits"]:
        group = predictions.loc[predictions["split_name"] == split_name]
        if len(group) != len(features) or set(group["feature_id"]) != set(features["feature_id"]):
            coverage_ok = False
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
    expected_runtime = pd.Series(
        source_labels["runtime_seconds"].to_numpy(dtype=float),
        index=source_labels["feature_id"].astype(str),
    )
    expected_event = pd.Series(
        source_labels["event_observed"].to_numpy(dtype=bool),
        index=source_labels["feature_id"].astype(str),
    )
    expected_duration = pd.Series(
        source_labels["duration_step"].to_numpy(dtype=float),
        index=source_labels["feature_id"].astype(str),
    )
    identifiers = predictions["feature_id"].astype(str)
    label_values_exact = bool(
        np.allclose(
            identifiers.map(expected_runtime).to_numpy(dtype=float),
            predictions["observed_runtime"].to_numpy(dtype=float),
        )
        and np.array_equal(
            identifiers.map(expected_event).to_numpy(dtype=bool),
            predictions["event_observed"].to_numpy(dtype=bool),
        )
        and np.allclose(
            identifiers.map(expected_duration).to_numpy(dtype=float),
            predictions["duration_step"].to_numpy(dtype=float),
        )
    )
    quantile_columns = [_quantile_column(level) for level in QUANTILE_LEVELS]
    reach_columns = [_reach_column(horizon) for horizon in SURVIVAL_HORIZONS]
    survival_columns = [_survival_column(horizon) for horizon in SURVIVAL_HORIZONS]
    quantiles = predictions[quantile_columns].to_numpy(dtype=float)
    reach = predictions[reach_columns].to_numpy(dtype=float)
    survival = predictions[survival_columns].to_numpy(dtype=float)
    required_metrics = [
        "runtime_nll",
        "runtime_crps",
        "runtime_calibration_mae",
        "runtime_coverage_80",
        "runtime_coverage_90",
        "runtime_median_mae",
        "survival_nll",
        "integrated_brier",
        "survival_calibration_mae",
        "observed_par10_mean",
        "predicted_par10_mean",
        "par10_mae",
    ]
    required_metrics_finite = bool(
        np.isfinite(fold_metrics[required_metrics].to_numpy(dtype=float)).all()
        and np.isfinite(aggregate[required_metrics].to_numpy(dtype=float)).all()
    )
    c_index_availability = bool(
        (
            (fold_metrics["c_index_comparable_pair_count"] > 0)
            == fold_metrics["c_index"].notna()
        ).all()
        and (
            (aggregate["c_index_comparable_pair_count"] > 0)
            == aggregate["c_index"].notna()
        ).all()
    )
    expected_aggregate_rows = len(preprocessing["splits"]) * (
        1 + len(feature_schema["cutoffs"])
    )
    checks = {
        "phase6_quality_pass": phase6_validation["status"] == "PHASE_6_PASS",
        "phase7_quality_pass": phase7_validation["status"] == "PHASE_7_PASS",
        "phase8_quality_pass": phase8_validation["status"] == "PHASE_8_PASS",
        "phase9_quality_pass": phase9_validation["status"] == "PHASE_9_PASS",
        "source_inputs_unchanged": current_inputs == original_input_fingerprints,
        "runtime_survival_labels_frozen_and_exact": (
            _labels_equal(saved_labels, source_labels)
            and _labels_equal(current_labels, source_labels)
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
        "runtime_and_survival_targets_exact": label_values_exact,
        "runtime_quantiles_finite_positive": bool(
            np.isfinite(quantiles).all() and (quantiles > 0.0).all()
        ),
        "runtime_quantiles_nondecreasing": bool((np.diff(quantiles, axis=1) >= 0.0).all()),
        "runtime_pit_bounded": bool(predictions["runtime_pit"].between(0.0, 1.0).all()),
        "reach_probabilities_finite_bounded": bool(
            np.isfinite(reach).all() and np.logical_and(reach >= 0.0, reach <= 1.0).all()
        ),
        "reach_probabilities_nondecreasing": bool((np.diff(reach, axis=1) >= 0.0).all()),
        "survival_probabilities_exact": bool(np.allclose(survival, 1.0 - reach)),
        "par10_predictions_finite_nonnegative": bool(
            np.isfinite(predictions["predicted_par10"]).all()
            and (predictions["predicted_par10"] >= 0.0).all()
        ),
        "required_metrics_finite": required_metrics_finite,
        "c_index_availability_explicit": c_index_availability,
        "aggregate_metrics_complete": len(aggregate) == expected_aggregate_rows,
        "runtime_calibration_complete": len(runtime_calibration)
        == expected_aggregate_rows * len(QUANTILE_LEVELS),
        "survival_calibration_complete": len(survival_calibration)
        == expected_aggregate_rows * len(SURVIVAL_HORIZONS),
        "survival_support_complete": len(support) == len(jobs) * len(SURVIVAL_HORIZONS),
        "fold_preprocessing_contract_complete": len(fold_schemas["folds"]) == len(jobs),
        "phase10_scope_runtime_survival_only": registry["targets"]
        == ["runtime_distribution", "first_passage_survival"],
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    issues = []
    if failed_checks:
        issues.append({"type": "failed_quality_checks", "checks": failed_checks})
    return {
        "status": "PHASE_10_PASS" if all(checks.values()) else "PHASE_10_FAIL",
        "schema_version": PHASE10_SCHEMA_VERSION,
        "scope": "leakage-safe runtime distribution and censoring-aware first-passage survival",
        "performance_gate_policy": (
            "construction, leakage, censoring semantics, distribution validity, and integrity only; "
            "runtime NLL/CRPS, coverage, C-index, survival Brier, calibration, and PAR10 are not "
            "Phase 10 pass thresholds"
        ),
        "phase11_boundary": "hierarchical Bayesian partial pooling is deferred to Phase 11",
        "phase6_directory": str(phase6_path.resolve()),
        "phase7_directory": str(phase7_path.resolve()),
        "phase8_directory": str(phase8_path.resolve()),
        "phase9_directory": str(phase9_path.resolve()),
        "feature_row_count": int(len(features)),
        "prediction_row_count": int(len(predictions)),
        "job_count": int(len(fold_metrics)),
        "model_artifact_count": int(len(registry["jobs"])),
        "aggregate_metric_row_count": int(len(aggregate)),
        "runtime_calibration_row_count": int(len(runtime_calibration)),
        "survival_calibration_row_count": int(len(survival_calibration)),
        "survival_support_row_count": int(len(support)),
        "observed_event_count": int(source_labels["event_observed"].sum()),
        "censored_count": int((~source_labels["event_observed"]).sum()),
        "event_rate": float(source_labels["event_observed"].mean()),
        "quantile_count": len(QUANTILE_LEVELS),
        "survival_horizon_count": len(SURVIVAL_HORIZONS),
        "runtime_raw_crossing_row_count": int(
            fold_metrics["runtime_raw_crossing_row_count"].sum()
        ),
        "survival_raw_crossing_row_count": int(
            fold_metrics["survival_raw_crossing_row_count"].sum()
        ),
        "checks": checks,
        "issues": issues,
    }


def run_phase10(
    run_directory: str | Path,
    phase6_directory: str | Path,
    phase7_directory: str | Path,
    phase8_directory: str | Path,
    phase9_directory: str | Path,
    *,
    master_seed: int = 20_260_826,
    gradient_boosting_iterations: int = 60,
    beta_prior: tuple[float, float] = DEFAULT_BETA_PRIOR,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    phase6_path = Path(phase6_directory)
    phase7_path = Path(phase7_directory)
    phase8_path = Path(phase8_directory)
    phase9_path = Path(phase9_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"
    if gradient_boosting_iterations < 1:
        raise ValueError("Phase 10 gradient boosting iterations must be positive")
    if len(beta_prior) != 2 or any(value <= 0.0 for value in beta_prior):
        raise ValueError("Phase 10 beta prior alpha and beta must be positive")
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
    _, _, failure_source, failure_sha = _load_failure_labels(phase6_path, targets)
    expected_phase9_inputs = _phase9_input_fingerprints(
        phase6_fingerprints, phase7_fingerprints, phase8_fingerprints, failure_sha
    )
    phase9_validation, _phase9_manifest, phase9_fingerprints = _validate_phase9_input(
        phase9_path, expected_phase9_inputs
    )
    labels, runtime_label_source, runtime_label_sha = _load_runtime_survival_labels(
        phase6_path, features, targets
    )
    input_fingerprints = _combined_input_fingerprints(
        expected_phase9_inputs, phase9_fingerprints, runtime_label_sha
    )
    stable_configuration = {
        "phase": 10,
        "schema_version": PHASE10_SCHEMA_VERSION,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "preprocessing_implementation_sha256": _file_sha256(Path(phase7_module.__file__)),
        "input_fingerprints": input_fingerprints,
        "master_seed": int(master_seed),
        "runtime_model": RUNTIME_MODEL_NAME,
        "survival_model": SURVIVAL_MODEL_NAME,
        "runtime_target": RUNTIME_TARGET_COLUMN,
        "first_passage_target": FIRST_PASSAGE_TARGET_COLUMN,
        "quantile_levels": list(QUANTILE_LEVELS),
        "survival_horizons": list(SURVIVAL_HORIZONS),
        "gradient_boosting_iterations": int(gradient_boosting_iterations),
        "beta_prior": {"alpha": float(beta_prior[0]), "beta": float(beta_prior[1])},
        "split_columns": feature_schema["split_columns"],
        "cutoffs": feature_schema["cutoffs"],
        "censoring_policy": (
            "target reached is an event at first-passage step; otherwise right-censor at "
            "min(executed steps, budget); horizon steps use ceil(budget*fraction)"
        ),
    }
    configuration_sha256 = _configuration_hash(stable_configuration)
    configuration = {
        **stable_configuration,
        "phase6_directory": str(phase6_path.resolve()),
        "phase7_directory": str(phase7_path.resolve()),
        "phase8_directory": str(phase8_path.resolve()),
        "phase9_directory": str(phase9_path.resolve()),
        "failure_label_source": str(failure_source.resolve()),
        "runtime_label_source": str(runtime_label_source.resolve()),
        "git_commit": current_git_commit(),
        "configuration_sha256": configuration_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Phase 10 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_10_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 10 resume manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest.get("output_sha256", {}).items():
                path = run_path / relative
                if not path.is_file() or _file_sha256(path) != expected_hash:
                    raise ValueError(f"completed Phase 10 output hash mismatch: {relative}")
            active_logger.info("[PHASE10][RESUME] status=PHASE_10_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info(
            "[PHASE10][RESUME] completed_jobs=%s", len(state.get("completed_jobs", {}))
        )
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE10_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": configuration_sha256,
            "input_fingerprints": input_fingerprints,
            "completed_jobs": {},
            "last_completed_job": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase10_started"})
    fold_schemas, job_outputs = _run_jobs(
        run_path,
        features,
        labels,
        splits,
        feature_schema,
        preprocessing,
        state,
        master_seed=master_seed,
        gradient_boosting_iterations=gradient_boosting_iterations,
        beta_prior=beta_prior,
        progress_path=progress_path,
        logger=active_logger,
    )
    jobs = _all_jobs(preprocessing)
    aggregate_outputs, _registry = _aggregate_job_outputs(run_path, jobs, labels)
    validation = _validate_phase10(
        run_path,
        phase6_path,
        phase7_path,
        phase8_path,
        phase9_path,
        input_fingerprints,
        phase6_validation,
        phase7_validation,
        phase8_validation,
        phase9_validation,
        features,
        targets,
        splits,
        feature_schema,
        preprocessing,
        labels,
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
        "phase": 10,
        "status": validation["status"],
        "schema_version": PHASE10_SCHEMA_VERSION,
        "input_fingerprints": input_fingerprints,
        "output_sha256": _relative_hashes(run_path, output_paths),
        "row_counts": {
            "features": validation["feature_row_count"],
            "predictions": validation["prediction_row_count"],
            "jobs": validation["job_count"],
            "runtime_calibration": validation["runtime_calibration_row_count"],
            "survival_calibration": validation["survival_calibration_row_count"],
            "survival_support": validation["survival_support_row_count"],
        },
        "event_counts": {
            "observed": validation["observed_event_count"],
            "right_censored": validation["censored_count"],
        },
        "phase11_allowed": validation["status"] == "PHASE_10_PASS",
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
        progress_path, {"event": "phase10_finished", "status": validation["status"]}
    )
    active_logger.info(
        "[PHASE10][SUMMARY] status=%s events=%s censored=%s jobs=%s predictions=%s directory=%s",
        validation["status"],
        validation["observed_event_count"],
        validation["censored_count"],
        validation["job_count"],
        validation["prediction_row_count"],
        run_path,
    )
    return validation
