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
from scipy.stats import beta as beta_distribution
from scipy.stats import norm
from sklearn.linear_model import Ridge

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
    QUANTILE_LEVELS,
    _cdf_at,
    _combined_input_fingerprints as _phase8_input_fingerprints,
    _distribution_metrics,
    _predictive_moments,
    _validate_phase7_input,
)
from uriel_v2.probabilistic_lab.phase9 import (
    _binary_metrics,
    _combined_input_fingerprints as _phase9_input_fingerprints,
    _load_failure_labels,
    _validate_phase8_input,
)
from uriel_v2.probabilistic_lab.phase10 import (
    SURVIVAL_HORIZONS,
    _combined_input_fingerprints as _phase10_input_fingerprints,
    _horizon_observation,
    _load_runtime_survival_labels,
    _runtime_cdf,
    _runtime_metrics,
    _runtime_moments,
    _survival_metrics,
    _survival_summaries,
    _validate_phase9_input,
)
from uriel_v2.provenance import current_git_commit


PHASE11_SCHEMA_VERSION = "phase11-v1"
CONTINUOUS_MODEL_NAME = "ridge_normal_normal_partial_pooling"
BINARY_MODEL_NAME = "hierarchical_beta_binomial_partial_pooling"
DEFAULT_RIDGE_ALPHA = 10.0
DEFAULT_PRIOR_STRENGTH = 20.0
DEFAULT_BETA_PRIOR = (0.5, 0.5)
DEFAULT_CALIBRATION_BINS = 10
MINIMUM_VARIANCE = 1e-10
GROUP_LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("domain", ("domain",)),
    ("domain_algorithm_family", ("domain", "algorithm_family")),
    ("domain_problem_family", ("domain", "problem_family")),
    (
        "domain_problem_algorithm_family",
        ("domain", "problem_family", "algorithm_family"),
    ),
)


def _quality_quantile_column(level: float) -> str:
    return f"quality_q{int(round(level * 100.0)):02d}"


def _runtime_quantile_column(level: float) -> str:
    return f"runtime_q{int(round(level * 100.0)):02d}"


def _horizon_suffix(horizon: float) -> str:
    return f"p{int(round(horizon * 100.0)):03d}"


def _reach_column(horizon: float) -> str:
    return f"reach_by_{_horizon_suffix(horizon)}"


def _normalize_key(value: Any) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def _group_key_json(key: tuple[str, ...]) -> str:
    return json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))


def _group_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple[str, ...]]:
    values = frame.loc[:, list(columns)].fillna("__MISSING__").astype(str).to_numpy()
    return [tuple(row) for row in values]


def _phase10_required_paths(phase10_path: Path) -> dict[str, Path]:
    return {
        "phase10_config": phase10_path / "config.json",
        "phase10_manifest": phase10_path / "manifest.json",
        "phase10_validation": phase10_path / "validation.json",
        "phase10_predictions": phase10_path
        / "data/predictions/oof_runtime_survival.parquet",
        "phase10_labels": phase10_path / "data/targets/runtime_survival_labels.parquet",
        "phase10_fold_metrics": phase10_path
        / "data/metrics/fold_runtime_survival_metrics.parquet",
        "phase10_aggregate_metrics": phase10_path
        / "data/metrics/aggregate_runtime_survival_metrics.parquet",
        "phase10_runtime_calibration": phase10_path
        / "data/calibration/runtime_quantile_calibration.parquet",
        "phase10_survival_calibration": phase10_path
        / "data/calibration/survival_calibration.parquet",
        "phase10_survival_support": phase10_path
        / "data/support/survival_horizon_support.parquet",
        "phase10_fold_schemas": phase10_path
        / "data/preprocessing/fold_feature_schemas.json",
        "phase10_model_registry": phase10_path / "model_registry.json",
    }


def _validate_phase10_input(
    phase10_path: Path,
    expected_input_fingerprints: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    required = _phase10_required_paths(phase10_path)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 11 Phase 10 input files are missing: {missing[:5]}")
    validation = json.loads(required["phase10_validation"].read_text(encoding="utf-8"))
    manifest = json.loads(required["phase10_manifest"].read_text(encoding="utf-8"))
    if validation.get("status") != "PHASE_10_PASS":
        raise ValueError("Phase 11 requires a PHASE_10_PASS runtime-survival run")
    if manifest.get("status") != "PHASE_10_PASS" or not manifest.get("phase11_allowed"):
        raise ValueError("Phase 10 manifest does not allow Phase 11")
    if validation.get("configuration", {}).get("input_fingerprints") != expected_input_fingerprints:
        raise ValueError("Phase 10 was not built from the supplied Phase 6 through 9 inputs")
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = phase10_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 10 manifest output hash mismatch: {relative}")
    fingerprints = {name: _file_sha256(path) for name, path in sorted(required.items())}
    return validation, manifest, fingerprints


def _combined_input_fingerprints(
    phase10_expected_inputs: dict[str, str],
    phase10_fingerprints: dict[str, str],
) -> dict[str, str]:
    return {
        **phase10_expected_inputs,
        **{f"phase10/{name}": value for name, value in phase10_fingerprints.items()},
    }


def _build_labels(
    targets: pd.DataFrame,
    failure_labels: pd.DataFrame,
    runtime_labels: pd.DataFrame,
) -> pd.DataFrame:
    feature_ids = targets["feature_id"].astype(str)
    failure = failure_labels.set_index(failure_labels["feature_id"].astype(str)).loc[feature_ids]
    runtime = runtime_labels.set_index(runtime_labels["feature_id"].astype(str)).loc[feature_ids]
    labels = targets[["feature_id", "run_id", "problem_id", "cutoff"]].copy()
    labels["observed_quality"] = pd.to_numeric(
        targets["target_quality_final"], errors="coerce"
    ).to_numpy(dtype=float)
    labels["observed_runtime"] = pd.to_numeric(
        targets["target_runtime"], errors="coerce"
    ).to_numpy(dtype=float)
    labels["observed_failure"] = failure["observed_failure"].to_numpy(dtype=bool)
    labels["event_observed"] = runtime["event_observed"].to_numpy(dtype=bool)
    labels["first_passage_step"] = runtime["first_passage_step"].to_numpy()
    labels["censor_step"] = runtime["censor_step"].to_numpy(dtype=float)
    labels["duration_step"] = runtime["duration_step"].to_numpy(dtype=float)
    labels["duration_fraction"] = runtime["duration_fraction"].to_numpy(dtype=float)
    labels["budget"] = runtime["budget"].to_numpy(dtype=float)
    if not np.isfinite(labels[["observed_quality", "observed_runtime"]].to_numpy()).all():
        raise ValueError("Phase 11 continuous labels must be finite")
    if not labels["observed_quality"].between(0.0, 1.0).all():
        raise ValueError("Phase 11 quality labels must be bounded in [0, 1]")
    if not (labels["observed_runtime"] > 0.0).all():
        raise ValueError("Phase 11 runtime labels must be positive")
    return labels


def _load_reference_predictions(
    phase8_path: Path,
    phase9_path: Path,
    phase10_path: Path,
) -> pd.DataFrame:
    quality = pd.read_parquet(
        phase8_path / "data/predictions/oof_quality_distribution.parquet"
    )
    failure = pd.read_parquet(
        phase9_path / "data/predictions/oof_failure_probability.parquet"
    )
    runtime = pd.read_parquet(
        phase10_path / "data/predictions/oof_runtime_survival.parquet"
    )
    keys = ["feature_id", "split_name"]
    quality_columns = [f"q{int(round(level * 100.0)):02d}" for level in QUANTILE_LEVELS]
    runtime_columns = [_runtime_quantile_column(level) for level in QUANTILE_LEVELS]
    reach_columns = [_reach_column(horizon) for horizon in SURVIVAL_HORIZONS]
    reference = quality[keys + quality_columns].rename(
        columns={column: f"reference_quality_{column}" for column in quality_columns}
    )
    reference = reference.merge(
        failure[keys + ["failure_probability"]].rename(
            columns={"failure_probability": "reference_failure_probability"}
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    reference = reference.merge(
        runtime[keys + runtime_columns + reach_columns].rename(
            columns={
                **{column: f"reference_{column}" for column in runtime_columns},
                **{column: f"reference_{column}" for column in reach_columns},
            }
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if reference.duplicated(keys).any():
        raise ValueError("Phase 11 reference predictions must be unique")
    return reference.sort_values(keys).reset_index(drop=True)


def _load_inputs(
    phase6_path: Path,
    phase7_path: Path,
    phase8_path: Path,
    phase9_path: Path,
    phase10_path: Path,
) -> dict[str, Any]:
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
    failure_labels, _failure_types, failure_source, failure_sha = _load_failure_labels(
        phase6_path, targets
    )
    expected_phase9 = _phase9_input_fingerprints(
        phase6_fingerprints, phase7_fingerprints, phase8_fingerprints, failure_sha
    )
    phase9_validation, _phase9_manifest, phase9_fingerprints = _validate_phase9_input(
        phase9_path, expected_phase9
    )
    runtime_labels, runtime_source, runtime_sha = _load_runtime_survival_labels(
        phase6_path, features, targets
    )
    expected_phase10 = _phase10_input_fingerprints(
        expected_phase9, phase9_fingerprints, runtime_sha
    )
    phase10_validation, _phase10_manifest, phase10_fingerprints = _validate_phase10_input(
        phase10_path, expected_phase10
    )
    labels = _build_labels(targets, failure_labels, runtime_labels)
    references = _load_reference_predictions(phase8_path, phase9_path, phase10_path)
    expected_reference_rows = len(features) * len(preprocessing["splits"])
    if len(references) != expected_reference_rows:
        raise ValueError("Phase 11 reference predictions do not cover every OOF row")
    return {
        "validations": {
            "phase6": phase6_validation,
            "phase7": phase7_validation,
            "phase8": phase8_validation,
            "phase9": phase9_validation,
            "phase10": phase10_validation,
        },
        "input_fingerprints": _combined_input_fingerprints(
            expected_phase10, phase10_fingerprints
        ),
        "features": features,
        "targets": targets,
        "splits": splits,
        "feature_schema": feature_schema,
        "preprocessing": preprocessing,
        "labels": labels,
        "references": references,
        "failure_source": failure_source,
        "runtime_source": runtime_source,
    }


def _fit_continuous_hierarchy(
    y_train: np.ndarray,
    base_train: np.ndarray,
    base_validation: np.ndarray,
    training_groups: pd.DataFrame,
    validation_groups: pd.DataFrame,
    *,
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    y_train = np.asarray(y_train, dtype=float)
    residual = y_train - np.asarray(base_train, dtype=float)
    prediction = np.asarray(base_validation, dtype=float).copy()
    group_variance = np.zeros(len(validation_groups), dtype=float)
    model_levels: dict[str, Any] = {}
    support_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for level_name, columns in GROUP_LEVELS:
        training_keys = _group_keys(training_groups, columns)
        validation_keys = _group_keys(validation_groups, columns)
        statistics = (
            pd.DataFrame({"group_key": training_keys, "residual": residual})
            .groupby("group_key", sort=True)["residual"]
            .agg(["count", "mean"])
        )
        residual_variance = max(float(np.var(residual, ddof=1)), MINIMUM_VARIANCE)
        if len(statistics) > 1:
            mean_variance = float(np.var(statistics["mean"].to_numpy(dtype=float), ddof=1))
            sampling_variance = float(
                np.mean(residual_variance / statistics["count"].to_numpy(dtype=float))
            )
            prior_variance = max(mean_variance - sampling_variance, 0.0)
        else:
            prior_variance = 0.0
        effects: dict[tuple[str, ...], float] = {}
        posterior_variances: dict[tuple[str, ...], float] = {}
        weights: dict[tuple[str, ...], float] = {}
        for raw_key, row in statistics.iterrows():
            key = _normalize_key(raw_key)
            count = int(row["count"])
            if prior_variance <= 0.0:
                weight = 0.0
                posterior_variance = 0.0
            else:
                weight = float(
                    (count * prior_variance)
                    / (count * prior_variance + residual_variance)
                )
                posterior_variance = float(
                    1.0 / (1.0 / prior_variance + count / residual_variance)
                )
            effect = weight * float(row["mean"])
            effects[key] = effect
            posterior_variances[key] = posterior_variance
            weights[key] = weight
            support_rows.append(
                {
                    "target": target,
                    "distribution": "normal_normal",
                    "hierarchy_level": level_name,
                    "group_columns": json.dumps(list(columns), separators=(",", ":")),
                    "group_key": _group_key_json(key),
                    "training_count": count,
                    "event_count": None,
                    "prior_mean": 0.0,
                    "prior_strength": None,
                    "prior_variance": prior_variance,
                    "residual_variance": residual_variance,
                    "sample_mean": float(row["mean"]),
                    "posterior_mean": effect,
                    "posterior_std": math.sqrt(max(posterior_variance, 0.0)),
                    "posterior_alpha": None,
                    "posterior_beta": None,
                    "shrinkage_weight": weight,
                }
            )
        training_effect = np.fromiter(
            (effects[key] for key in training_keys), dtype=float, count=len(training_keys)
        )
        residual -= training_effect
        seen = np.fromiter(
            (key in effects for key in validation_keys), dtype=bool, count=len(validation_keys)
        )
        validation_effect = np.fromiter(
            (effects.get(key, 0.0) for key in validation_keys),
            dtype=float,
            count=len(validation_keys),
        )
        validation_variance = np.fromiter(
            (
                posterior_variances[key] if key in posterior_variances else prior_variance
                for key in validation_keys
            ),
            dtype=float,
            count=len(validation_keys),
        )
        prediction += validation_effect
        group_variance += validation_variance
        diagnostics[f"seen_{level_name}_count"] = int(seen.sum())
        diagnostics[f"unseen_{level_name}_count"] = int((~seen).sum())
        model_levels[level_name] = {
            "columns": columns,
            "prior_variance": prior_variance,
            "residual_variance": residual_variance,
            "effects": effects,
            "posterior_variances": posterior_variances,
            "shrinkage_weights": weights,
        }
    final_residual_variance = max(float(np.mean(residual**2)), MINIMUM_VARIANCE)
    predictive_std = np.sqrt(final_residual_variance + group_variance)
    model = {
        "target": target,
        "levels": model_levels,
        "final_residual_variance": final_residual_variance,
        "uncertainty_policy": (
            "final residual variance plus posterior group variance for seen groups and "
            "prior group variance for unseen groups"
        ),
    }
    return prediction, predictive_std, model, pd.DataFrame(support_rows), diagnostics


def _beta_parent_mean(
    level_name: str,
    key: tuple[str, ...],
    posteriors: dict[str, dict[tuple[str, ...], tuple[float, float, float]]],
    global_mean: float,
) -> float:
    domain = (key[0],)
    if level_name == "domain":
        return global_mean
    if level_name in {"domain_algorithm_family", "domain_problem_family"}:
        return posteriors["domain"].get(domain, (0.0, 0.0, global_mean))[2]
    family_key = (key[0], key[1])
    algorithm_key = (key[0], key[2])
    family_mean = posteriors["domain_problem_family"].get(
        family_key, (0.0, 0.0, global_mean)
    )[2]
    algorithm_mean = posteriors["domain_algorithm_family"].get(
        algorithm_key, (0.0, 0.0, global_mean)
    )[2]
    return 0.5 * (family_mean + algorithm_mean)


def _fit_beta_binomial_hierarchy(
    outcomes: np.ndarray,
    training_groups: pd.DataFrame,
    validation_groups: pd.DataFrame,
    *,
    target: str,
    observable: np.ndarray | None = None,
    beta_prior: tuple[float, float] = DEFAULT_BETA_PRIOR,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    outcomes = np.asarray(outcomes, dtype=bool)
    observed = (
        np.ones(len(outcomes), dtype=bool)
        if observable is None
        else np.asarray(observable, dtype=bool)
    )
    if not observed.any():
        raise ValueError(f"Phase 11 has no observable training rows for {target}")
    alpha0, beta0 = beta_prior
    events = int(outcomes[observed].sum())
    observed_count = int(observed.sum())
    global_alpha = float(alpha0 + events)
    global_beta = float(beta0 + observed_count - events)
    global_mean = global_alpha / (global_alpha + global_beta)
    posteriors: dict[str, dict[tuple[str, ...], tuple[float, float, float]]] = {}
    support_rows: list[dict[str, Any]] = [
        {
            "target": target,
            "distribution": "beta_binomial",
            "hierarchy_level": "global",
            "group_columns": "[]",
            "group_key": "__GLOBAL__",
            "training_count": observed_count,
            "event_count": events,
            "prior_mean": alpha0 / (alpha0 + beta0),
            "prior_strength": alpha0 + beta0,
            "prior_variance": None,
            "residual_variance": None,
            "sample_mean": events / observed_count,
            "posterior_mean": global_mean,
            "posterior_std": math.sqrt(
                global_alpha
                * global_beta
                / ((global_alpha + global_beta) ** 2 * (global_alpha + global_beta + 1.0))
            ),
            "posterior_alpha": global_alpha,
            "posterior_beta": global_beta,
            "shrinkage_weight": observed_count / (observed_count + alpha0 + beta0),
        }
    ]
    observed_groups = training_groups.loc[observed].reset_index(drop=True)
    observed_outcomes = outcomes[observed]
    for level_name, columns in GROUP_LEVELS:
        table = observed_groups.loc[:, list(columns)].fillna("__MISSING__").astype(str).copy()
        table["_outcome"] = observed_outcomes.astype(int)
        level_posteriors: dict[tuple[str, ...], tuple[float, float, float]] = {}
        for raw_key, group in table.groupby(list(columns), sort=True):
            key = _normalize_key(raw_key)
            count = int(len(group))
            event_count = int(group["_outcome"].sum())
            prior_mean = float(
                np.clip(
                    _beta_parent_mean(level_name, key, posteriors, global_mean),
                    1e-9,
                    1.0 - 1e-9,
                )
            )
            prior_alpha = prior_mean * prior_strength
            prior_beta = (1.0 - prior_mean) * prior_strength
            posterior_alpha = prior_alpha + event_count
            posterior_beta = prior_beta + count - event_count
            posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
            posterior_std = math.sqrt(
                posterior_alpha
                * posterior_beta
                / (
                    (posterior_alpha + posterior_beta) ** 2
                    * (posterior_alpha + posterior_beta + 1.0)
                )
            )
            level_posteriors[key] = (posterior_alpha, posterior_beta, posterior_mean)
            support_rows.append(
                {
                    "target": target,
                    "distribution": "beta_binomial",
                    "hierarchy_level": level_name,
                    "group_columns": json.dumps(list(columns), separators=(",", ":")),
                    "group_key": _group_key_json(key),
                    "training_count": count,
                    "event_count": event_count,
                    "prior_mean": prior_mean,
                    "prior_strength": prior_strength,
                    "prior_variance": None,
                    "residual_variance": None,
                    "sample_mean": event_count / count,
                    "posterior_mean": posterior_mean,
                    "posterior_std": posterior_std,
                    "posterior_alpha": posterior_alpha,
                    "posterior_beta": posterior_beta,
                    "shrinkage_weight": count / (count + prior_strength),
                }
            )
        posteriors[level_name] = level_posteriors
    probability = np.full(len(validation_groups), global_mean, dtype=float)
    posterior_alpha = np.full(len(validation_groups), global_alpha, dtype=float)
    posterior_beta = np.full(len(validation_groups), global_beta, dtype=float)
    selected_level = np.full(len(validation_groups), "global", dtype=object)
    diagnostics: dict[str, Any] = {}
    for level_name, columns in GROUP_LEVELS:
        keys = _group_keys(validation_groups, columns)
        level_posteriors = posteriors[level_name]
        seen = np.fromiter(
            (key in level_posteriors for key in keys), dtype=bool, count=len(keys)
        )
        for index in np.flatnonzero(seen):
            alpha, beta, mean = level_posteriors[keys[index]]
            probability[index] = mean
            posterior_alpha[index] = alpha
            posterior_beta[index] = beta
            selected_level[index] = level_name
        diagnostics[f"seen_{level_name}_count"] = int(seen.sum())
        diagnostics[f"unseen_{level_name}_count"] = int((~seen).sum())
    lower = beta_distribution.ppf(0.05, posterior_alpha, posterior_beta)
    upper = beta_distribution.ppf(0.95, posterior_alpha, posterior_beta)
    # Equal-tail intervals of extremely skewed Beta posteriors can exclude the mean.
    # Preserve the requested 90% mass while expanding the reported interval to contain it.
    lower = np.minimum(lower, probability)
    upper = np.maximum(upper, probability)
    diagnostics["selected_level_counts"] = {
        str(level): int((selected_level == level).sum())
        for level in sorted(set(selected_level.tolist()))
    }
    model = {
        "target": target,
        "beta_prior": {"alpha": alpha0, "beta": beta0},
        "prior_strength": prior_strength,
        "global_posterior": {
            "alpha": global_alpha,
            "beta": global_beta,
            "mean": global_mean,
        },
        "levels": posteriors,
        "selection_policy": (
            "deepest observed group posterior; unseen family falls back to "
            "domain-algorithm, domain, then global posterior"
        ),
    }
    return probability, lower, upper, model, pd.DataFrame(support_rows), diagnostics


def _job_id(split_name: str, fold: int) -> str:
    return f"{split_name}__fold{fold}__hierarchical_partial_pooling"


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
            raise ValueError(f"Phase 11 job output hash mismatch: job={job_id} path={relative}")
    return marker


def _all_jobs(preprocessing: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        (split_name, fold)
        for split_name in sorted(preprocessing["splits"])
        for fold in sorted(int(value) for value in preprocessing["splits"][split_name])
    ]


def _prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _metric_payload(predictions: pd.DataFrame) -> dict[str, Any]:
    quality_columns = [_quality_quantile_column(level) for level in QUANTILE_LEVELS]
    reference_quality_columns = [
        f"reference_quality_q{int(round(level * 100.0)):02d}"
        for level in QUANTILE_LEVELS
    ]
    runtime_columns = [_runtime_quantile_column(level) for level in QUANTILE_LEVELS]
    reference_runtime_columns = [f"reference_{column}" for column in runtime_columns]
    reach_columns = [_reach_column(horizon) for horizon in SURVIVAL_HORIZONS]
    reference_reach_columns = [f"reference_{column}" for column in reach_columns]
    quality = _distribution_metrics(
        predictions["observed_quality"].to_numpy(dtype=float),
        predictions[quality_columns].to_numpy(dtype=float),
    )
    reference_quality = _distribution_metrics(
        predictions["observed_quality"].to_numpy(dtype=float),
        predictions[reference_quality_columns].to_numpy(dtype=float),
    )
    runtime = _runtime_metrics(
        predictions["observed_runtime"].to_numpy(dtype=float),
        predictions[runtime_columns].to_numpy(dtype=float),
    )
    reference_runtime = _runtime_metrics(
        predictions["observed_runtime"].to_numpy(dtype=float),
        predictions[reference_runtime_columns].to_numpy(dtype=float),
    )
    failure = _binary_metrics(
        predictions["observed_failure"].to_numpy(dtype=bool),
        predictions["failure_probability"].to_numpy(dtype=float),
        calibration_bins=DEFAULT_CALIBRATION_BINS,
    )
    reference_failure = _binary_metrics(
        predictions["observed_failure"].to_numpy(dtype=bool),
        predictions["reference_failure_probability"].to_numpy(dtype=float),
        calibration_bins=DEFAULT_CALIBRATION_BINS,
    )
    survival = _survival_metrics(
        predictions["event_observed"].to_numpy(dtype=bool),
        predictions["duration_step"].to_numpy(dtype=float),
        predictions["budget"].to_numpy(dtype=float),
        predictions[reach_columns].to_numpy(dtype=float),
    )
    reference_survival = _survival_metrics(
        predictions["event_observed"].to_numpy(dtype=bool),
        predictions["duration_step"].to_numpy(dtype=float),
        predictions["budget"].to_numpy(dtype=float),
        predictions[reference_reach_columns].to_numpy(dtype=float),
    )
    return {
        **_prefixed("quality_", quality),
        **_prefixed("reference_quality_", reference_quality),
        **runtime,
        **_prefixed("reference_", reference_runtime),
        **_prefixed("failure_", failure),
        **_prefixed("reference_failure_", reference_failure),
        **survival,
        **_prefixed("reference_", reference_survival),
        "quality_nll_delta_vs_phase8": quality["nll"] - reference_quality["nll"],
        "quality_crps_delta_vs_phase8": quality["crps"] - reference_quality["crps"],
        "runtime_nll_delta_vs_phase10": runtime["runtime_nll"]
        - reference_runtime["runtime_nll"],
        "runtime_crps_delta_vs_phase10": runtime["runtime_crps"]
        - reference_runtime["runtime_crps"],
        "failure_brier_delta_vs_phase9": failure["brier"] - reference_failure["brier"],
        "failure_log_loss_delta_vs_phase9": failure["log_loss"]
        - reference_failure["log_loss"],
        "survival_nll_delta_vs_phase10": survival["survival_nll"]
        - reference_survival["survival_nll"],
        "integrated_brier_delta_vs_phase10": survival["integrated_brier"]
        - reference_survival["integrated_brier"],
        "par10_mae_delta_vs_phase10": survival["par10_mae"]
        - reference_survival["par10_mae"],
    }


def _prediction_frame(
    validation_features: pd.DataFrame,
    validation_labels: pd.DataFrame,
    validation_references: pd.DataFrame,
    quality_mean: np.ndarray,
    quality_std: np.ndarray,
    runtime_log_mean: np.ndarray,
    runtime_log_std: np.ndarray,
    failure_probability: np.ndarray,
    failure_lower: np.ndarray,
    failure_upper: np.ndarray,
    failure_selected_level: np.ndarray,
    reach_probability: np.ndarray,
    reach_lower: np.ndarray,
    reach_upper: np.ndarray,
    reach_selected_levels: np.ndarray,
    *,
    split_name: str,
    fold: int,
) -> pd.DataFrame:
    feature_columns = [
        "feature_id",
        "cutoff",
        "domain",
        "problem_family",
        "algorithm_family",
    ]
    frame = validation_features[feature_columns].reset_index(drop=True).copy()
    labels = validation_labels.reset_index(drop=True)
    frame["split_name"] = split_name
    frame["fold"] = fold
    for column in [
        "observed_quality",
        "observed_runtime",
        "observed_failure",
        "event_observed",
        "first_passage_step",
        "censor_step",
        "duration_step",
        "duration_fraction",
        "budget",
    ]:
        frame[column] = labels[column].to_numpy()
    z_values = norm.ppf(np.asarray(QUANTILE_LEVELS, dtype=float))
    quality_quantiles = np.clip(
        quality_mean[:, None] + quality_std[:, None] * z_values[None, :], 0.0, 1.0
    )
    quality_predictive_mean, quality_predictive_std = _predictive_moments(
        quality_quantiles
    )
    for index, level in enumerate(QUANTILE_LEVELS):
        frame[_quality_quantile_column(level)] = quality_quantiles[:, index]
    frame["quality_latent_mean"] = quality_mean
    frame["quality_latent_std"] = quality_std
    frame["quality_predictive_mean"] = quality_predictive_mean
    frame["quality_predictive_std"] = quality_predictive_std
    frame["quality_pit"] = _cdf_at(
        quality_quantiles, labels["observed_quality"].to_numpy(dtype=float)
    )
    runtime_log_quantiles = (
        runtime_log_mean[:, None] + runtime_log_std[:, None] * z_values[None, :]
    )
    runtime_quantiles = np.exp(np.clip(runtime_log_quantiles, -30.0, 30.0))
    runtime_predictive_mean, runtime_predictive_std = _runtime_moments(runtime_quantiles)
    for index, level in enumerate(QUANTILE_LEVELS):
        frame[_runtime_quantile_column(level)] = runtime_quantiles[:, index]
    frame["runtime_log_mean"] = runtime_log_mean
    frame["runtime_log_std"] = runtime_log_std
    frame["runtime_predictive_mean"] = runtime_predictive_mean
    frame["runtime_predictive_std"] = runtime_predictive_std
    frame["runtime_pit"] = _runtime_cdf(
        runtime_quantiles, labels["observed_runtime"].to_numpy(dtype=float)
    )
    frame["failure_probability"] = failure_probability
    frame["failure_probability_q05"] = failure_lower
    frame["failure_probability_q95"] = failure_upper
    frame["failure_selected_hierarchy"] = failure_selected_level
    for index, horizon in enumerate(SURVIVAL_HORIZONS):
        suffix = _horizon_suffix(horizon)
        frame[_reach_column(horizon)] = reach_probability[:, index]
        frame[f"reach_lower_{suffix}"] = reach_lower[:, index]
        frame[f"reach_upper_{suffix}"] = reach_upper[:, index]
        frame[f"reach_selected_hierarchy_{suffix}"] = reach_selected_levels[:, index]
    restricted_mean, expected_par10 = _survival_summaries(
        reach_probability, labels["budget"].to_numpy(dtype=float)
    )
    event = labels["event_observed"].to_numpy(dtype=bool)
    duration = labels["duration_step"].to_numpy(dtype=float)
    budget = labels["budget"].to_numpy(dtype=float)
    frame["predicted_restricted_mean_step"] = restricted_mean
    frame["observed_par10"] = np.where(event, duration, 10.0 * budget)
    frame["predicted_par10"] = expected_par10
    reference = validation_references.reset_index(drop=True)
    for column in reference.columns:
        if column not in {"feature_id", "split_name"}:
            frame[column] = reference[column].to_numpy()
    return frame


def _run_jobs(
    run_path: Path,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    references: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
    state: dict[str, Any],
    *,
    ridge_alpha: float,
    prior_strength: float,
    beta_prior: tuple[float, float],
    progress_path: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[Path]]:
    fold_schemas: dict[str, Any] = {
        "schema_version": PHASE11_SCHEMA_VERSION,
        "source_preprocessing_schema_version": preprocessing["schema_version"],
        "transform_policy": (
            "reuse verified Phase 6 fold preprocessing; hierarchical posteriors are fitted "
            "from training-fold rows only"
        ),
        "folds": {},
    }
    reference_by_split = {
        split_name: group.set_index("feature_id")
        for split_name, group in references.groupby("split_name", sort=True)
    }
    quality_values = labels["observed_quality"].to_numpy(dtype=float)
    runtime_log_values = np.log(labels["observed_runtime"].to_numpy(dtype=float))
    failure_values = labels["observed_failure"].to_numpy(dtype=bool)
    event_values = labels["event_observed"].to_numpy(dtype=bool)
    duration_values = labels["duration_step"].to_numpy(dtype=float)
    budget_values = labels["budget"].to_numpy(dtype=float)
    output_paths: list[Path] = []
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
                raise ValueError(f"Phase 11 transformed schema mismatch: {fold_key}")
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
                logger.info("[PHASE11][RESUME] job=%s status=SKIP_VERIFIED", job_id)
                continue
            logger.info(
                "[PHASE11][FOLD] split=%s fold=%s train_rows=%s validation_rows=%s",
                split_name,
                fold,
                int(training_mask.sum()),
                int(validation_mask.sum()),
            )
            training_groups = features.loc[training_mask].reset_index(drop=True)
            validation_groups = features.loc[validation_mask].reset_index(drop=True)
            quality_ridge = Ridge(alpha=ridge_alpha, solver="lsqr")
            quality_ridge.fit(x_train, quality_values[training_mask])
            quality_mean, quality_std, quality_hierarchy, quality_support, quality_diagnostics = (
                _fit_continuous_hierarchy(
                    quality_values[training_mask],
                    quality_ridge.predict(x_train),
                    quality_ridge.predict(x_validation),
                    training_groups,
                    validation_groups,
                    target="quality",
                )
            )
            runtime_ridge = Ridge(alpha=ridge_alpha, solver="lsqr")
            runtime_ridge.fit(x_train, runtime_log_values[training_mask])
            (
                runtime_log_mean,
                runtime_log_std,
                runtime_hierarchy,
                runtime_support,
                runtime_diagnostics,
            ) = _fit_continuous_hierarchy(
                runtime_log_values[training_mask],
                runtime_ridge.predict(x_train),
                runtime_ridge.predict(x_validation),
                training_groups,
                validation_groups,
                target="log_runtime",
            )
            (
                failure_probability,
                failure_lower,
                failure_upper,
                failure_hierarchy,
                failure_support,
                failure_diagnostics,
            ) = _fit_beta_binomial_hierarchy(
                failure_values[training_mask],
                training_groups,
                validation_groups,
                target="failure",
                beta_prior=beta_prior,
                prior_strength=prior_strength,
            )
            failure_selected = np.full(len(validation_groups), "global", dtype=object)
            for level_name, columns in GROUP_LEVELS:
                keys = _group_keys(validation_groups, columns)
                seen = np.fromiter(
                    (key in failure_hierarchy["levels"][level_name] for key in keys),
                    dtype=bool,
                    count=len(keys),
                )
                failure_selected[seen] = level_name
            reach_probability = np.empty(
                (int(validation_mask.sum()), len(SURVIVAL_HORIZONS)), dtype=float
            )
            reach_lower = np.empty_like(reach_probability)
            reach_upper = np.empty_like(reach_probability)
            reach_selected_levels = np.empty(reach_probability.shape, dtype=object)
            survival_models: dict[str, Any] = {}
            survival_diagnostics: dict[str, Any] = {}
            support_frames = [quality_support, runtime_support, failure_support]
            for index, horizon in enumerate(SURVIVAL_HORIZONS):
                observable, outcome = _horizon_observation(
                    event_values[training_mask],
                    duration_values[training_mask],
                    budget_values[training_mask],
                    horizon,
                )
                suffix = _horizon_suffix(horizon)
                (
                    probability,
                    lower,
                    upper,
                    model,
                    support,
                    diagnostics,
                ) = _fit_beta_binomial_hierarchy(
                    outcome,
                    training_groups,
                    validation_groups,
                    target=f"reach_{suffix}",
                    observable=observable,
                    beta_prior=beta_prior,
                    prior_strength=prior_strength,
                )
                reach_probability[:, index] = probability
                reach_lower[:, index] = lower
                reach_upper[:, index] = upper
                selected = np.full(len(validation_groups), "global", dtype=object)
                for level_name, columns in GROUP_LEVELS:
                    keys = _group_keys(validation_groups, columns)
                    level_posteriors = model["levels"][level_name]
                    seen = np.fromiter(
                        (key in level_posteriors for key in keys),
                        dtype=bool,
                        count=len(keys),
                    )
                    selected[seen] = level_name
                reach_selected_levels[:, index] = selected
                survival_models[suffix] = model
                survival_diagnostics[suffix] = diagnostics
                support_frames.append(support)
            raw_reach = reach_probability.copy()
            reach_probability = np.maximum.accumulate(reach_probability, axis=1)
            reach_lower = np.maximum.accumulate(reach_lower, axis=1)
            reach_upper = np.maximum.accumulate(reach_upper, axis=1)
            reach_lower = np.minimum(reach_lower, reach_probability)
            reach_upper = np.maximum(reach_upper, reach_probability)
            validation_ids = validation_groups["feature_id"].astype(str).tolist()
            validation_references = (
                reference_by_split[split_name].loc[validation_ids].reset_index()
            )
            predictions = _prediction_frame(
                validation_groups,
                labels.loc[validation_mask],
                validation_references,
                quality_mean,
                quality_std,
                runtime_log_mean,
                runtime_log_std,
                failure_probability,
                failure_lower,
                failure_upper,
                failure_selected,
                reach_probability,
                reach_lower,
                reach_upper,
                reach_selected_levels,
                split_name=split_name,
                fold=fold,
            )
            support = pd.DataFrame.from_records(
                [
                    record
                    for support_frame in support_frames
                    for record in support_frame.to_dict(orient="records")
                ]
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
                "quality_unseen_domain_problem_family_count": quality_diagnostics[
                    "unseen_domain_problem_family_count"
                ],
                "runtime_unseen_domain_problem_family_count": runtime_diagnostics[
                    "unseen_domain_problem_family_count"
                ],
                "failure_unseen_domain_problem_family_count": failure_diagnostics[
                    "unseen_domain_problem_family_count"
                ],
                "survival_raw_crossing_row_count": int(
                    np.any(np.diff(raw_reach, axis=1) < 0.0, axis=1).sum()
                ),
                **_metric_payload(predictions),
            }
            model_artifact = {
                "schema_version": PHASE11_SCHEMA_VERSION,
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "continuous_model": CONTINUOUS_MODEL_NAME,
                "binary_model": BINARY_MODEL_NAME,
                "feature_names": feature_names,
                "fold_contract": fold_contract,
                "quality_ridge": quality_ridge,
                "quality_hierarchy": quality_hierarchy,
                "runtime_ridge": runtime_ridge,
                "runtime_hierarchy": runtime_hierarchy,
                "failure_hierarchy": failure_hierarchy,
                "survival_hierarchies": survival_models,
                "reach_postprocessing": "row-wise cumulative maximum",
            }
            _atomic_parquet(paths["predictions"], predictions)
            _atomic_json(paths["metrics"], metrics)
            _atomic_parquet(paths["support"], support)
            _atomic_pickle(paths["model"], model_artifact)
            marker = {
                "schema_version": PHASE11_SCHEMA_VERSION,
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
                "[PHASE11][JOB] job=%s family_unseen=%s reach_crossing=%s",
                job_id,
                quality_diagnostics["unseen_domain_problem_family_count"],
                metrics["survival_raw_crossing_row_count"],
            )
    fold_schema_path = run_path / "data/preprocessing/fold_feature_schemas.json"
    _atomic_json(fold_schema_path, fold_schemas)
    output_paths.append(fold_schema_path)
    return fold_schemas, output_paths


def _aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, columns in (
        ("overall", ("split_name",)),
        ("cutoff", ("split_name", "cutoff")),
    ):
        for keys, group in predictions.groupby(list(columns), sort=True):
            values = keys if isinstance(keys, tuple) else (keys,)
            identifiers = dict(zip(columns, values, strict=True))
            rows.append(
                {
                    "scope": scope,
                    "split_name": str(identifiers["split_name"]),
                    "cutoff": (
                        float(identifiers["cutoff"]) if "cutoff" in identifiers else None
                    ),
                    "row_count": int(len(group)),
                    "fold_count": int(group["fold"].nunique()),
                    **_metric_payload(group),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "split_name", "cutoff"], na_position="first"
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
        "schema_version": PHASE11_SCHEMA_VERSION,
        "targets": [
            "quality_distribution",
            "runtime_distribution",
            "failure_probability",
            "first_passage_survival",
        ],
        "continuous_model": CONTINUOUS_MODEL_NAME,
        "binary_model": BINARY_MODEL_NAME,
        "group_levels": [name for name, _columns in GROUP_LEVELS],
        "quantile_levels": list(QUANTILE_LEVELS),
        "survival_horizons": list(SURVIVAL_HORIZONS),
        "artifact_policy": "one hierarchical posterior bundle per split/fold",
        "jobs": [],
    }
    for split_name, fold in jobs:
        job_id = _job_id(split_name, fold)
        marker = _verify_job(run_path, job_id)
        if marker is None:
            raise ValueError(f"Phase 11 job is incomplete: {job_id}")
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
    support = pd.concat(support_frames, ignore_index=True, sort=False).sort_values(
        ["split_name", "fold", "target", "hierarchy_level", "group_key"]
    ).reset_index(drop=True)
    aggregate = _aggregate_metrics(predictions)
    predictions_path = run_path / "data/predictions/oof_hierarchical_predictions.parquet"
    labels_path = run_path / "data/targets/hierarchical_labels.parquet"
    fold_metrics_path = run_path / "data/metrics/fold_hierarchical_metrics.parquet"
    aggregate_path = run_path / "data/metrics/aggregate_hierarchical_metrics.parquet"
    support_path = run_path / "data/posteriors/group_posteriors.parquet"
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


def _frames_equal(first: pd.DataFrame, second: pd.DataFrame) -> bool:
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
        elif not first[column].reset_index(drop=True).equals(
            second[column].reset_index(drop=True)
        ):
            return False
    return True


def _training_group_support_exact(
    predictions: pd.DataFrame,
    support: pd.DataFrame,
    features: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
) -> bool:
    for split_name, fold_column in feature_schema["split_columns"].items():
        for fold in sorted(predictions.loc[predictions["split_name"] == split_name, "fold"].unique()):
            training_mask = splits[fold_column].to_numpy(dtype=int) != int(fold)
            expected = {
                _group_key_json(key)
                for key in _group_keys(
                    features.loc[training_mask].reset_index(drop=True),
                    ("domain", "problem_family"),
                )
            }
            observed = set(
                support.loc[
                    (support["split_name"] == split_name)
                    & (support["fold"] == int(fold))
                    & (support["target"] == "quality")
                    & (support["hierarchy_level"] == "domain_problem_family"),
                    "group_key",
                ].astype(str)
            )
            if observed != expected:
                return False
    return True


def _validate_phase11(
    run_path: Path,
    paths: tuple[Path, Path, Path, Path, Path],
    original_inputs: dict[str, str],
    source: dict[str, Any],
) -> dict[str, Any]:
    current = _load_inputs(*paths)
    features = source["features"]
    splits = source["splits"]
    labels = source["labels"]
    preprocessing = source["preprocessing"]
    feature_schema = source["feature_schema"]
    predictions = pd.read_parquet(
        run_path / "data/predictions/oof_hierarchical_predictions.parquet"
    )
    saved_labels = pd.read_parquet(run_path / "data/targets/hierarchical_labels.parquet")
    fold_metrics = pd.read_parquet(run_path / "data/metrics/fold_hierarchical_metrics.parquet")
    aggregate = pd.read_parquet(
        run_path / "data/metrics/aggregate_hierarchical_metrics.parquet"
    )
    support = pd.read_parquet(run_path / "data/posteriors/group_posteriors.parquet")
    registry = json.loads((run_path / "model_registry.json").read_text(encoding="utf-8"))
    fold_schemas = json.loads(
        (run_path / "data/preprocessing/fold_feature_schemas.json").read_text(encoding="utf-8")
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
    expected_labels = labels.set_index(labels["feature_id"].astype(str))
    ids = predictions["feature_id"].astype(str)
    labels_exact = bool(
        np.allclose(ids.map(expected_labels["observed_quality"]), predictions["observed_quality"])
        and np.allclose(ids.map(expected_labels["observed_runtime"]), predictions["observed_runtime"])
        and np.array_equal(
            ids.map(expected_labels["observed_failure"]).to_numpy(dtype=bool),
            predictions["observed_failure"].to_numpy(dtype=bool),
        )
        and np.array_equal(
            ids.map(expected_labels["event_observed"]).to_numpy(dtype=bool),
            predictions["event_observed"].to_numpy(dtype=bool),
        )
    )
    quality_columns = [_quality_quantile_column(level) for level in QUANTILE_LEVELS]
    runtime_columns = [_runtime_quantile_column(level) for level in QUANTILE_LEVELS]
    reach_columns = [_reach_column(horizon) for horizon in SURVIVAL_HORIZONS]
    quality = predictions[quality_columns].to_numpy(dtype=float)
    runtime = predictions[runtime_columns].to_numpy(dtype=float)
    reach = predictions[reach_columns].to_numpy(dtype=float)
    failure = predictions["failure_probability"].to_numpy(dtype=float)
    reference_columns = [column for column in predictions if column.startswith("reference_")]
    reference_source = current["references"].set_index(["feature_id", "split_name"])
    reference_index = pd.MultiIndex.from_frame(predictions[["feature_id", "split_name"]])
    expected_references = reference_source.loc[reference_index, reference_columns].reset_index(
        drop=True
    )
    reference_exact = bool(
        np.allclose(
            expected_references.to_numpy(dtype=float),
            predictions[reference_columns].to_numpy(dtype=float),
        )
    )
    required_metrics = [
        "quality_nll",
        "quality_crps",
        "runtime_nll",
        "runtime_crps",
        "failure_brier",
        "failure_log_loss",
        "survival_nll",
        "integrated_brier",
        "survival_calibration_mae",
        "par10_mae",
        "reference_quality_nll",
        "reference_runtime_nll",
        "reference_failure_brier",
        "reference_survival_nll",
    ]
    metrics_finite = bool(
        np.isfinite(fold_metrics[required_metrics].to_numpy(dtype=float)).all()
        and np.isfinite(aggregate[required_metrics].to_numpy(dtype=float)).all()
    )
    family_rows = fold_metrics["split_name"] == "family_holdout"
    family_fallback = bool(
        (
            fold_metrics.loc[family_rows, "quality_unseen_domain_problem_family_count"]
            == fold_metrics.loc[family_rows, "validation_row_count"]
        ).all()
        and (
            fold_metrics.loc[family_rows, "runtime_unseen_domain_problem_family_count"]
            == fold_metrics.loc[family_rows, "validation_row_count"]
        ).all()
    )
    expected_aggregate_rows = len(preprocessing["splits"]) * (
        1 + len(feature_schema["cutoffs"])
    )
    checks = {
        **{
            f"{phase}_quality_pass": validation["status"] == f"PHASE_{phase[5:]}_PASS"
            for phase, validation in current["validations"].items()
        },
        "source_inputs_unchanged": current["input_fingerprints"] == original_inputs,
        "hierarchical_labels_frozen_and_exact": _frames_equal(saved_labels, labels),
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
        "target_labels_exact": labels_exact,
        "reference_predictions_exact": reference_exact,
        "quality_quantiles_finite_bounded": bool(
            np.isfinite(quality).all()
            and np.logical_and(quality >= 0.0, quality <= 1.0).all()
        ),
        "quality_quantiles_nondecreasing": bool((np.diff(quality, axis=1) >= 0.0).all()),
        "quality_pit_bounded": bool(predictions["quality_pit"].between(0.0, 1.0).all()),
        "runtime_quantiles_finite_positive": bool(
            np.isfinite(runtime).all() and (runtime > 0.0).all()
        ),
        "runtime_quantiles_nondecreasing": bool((np.diff(runtime, axis=1) >= 0.0).all()),
        "runtime_pit_bounded": bool(predictions["runtime_pit"].between(0.0, 1.0).all()),
        "failure_probability_finite_bounded": bool(
            np.isfinite(failure).all()
            and np.logical_and(failure >= 0.0, failure <= 1.0).all()
        ),
        "failure_interval_contains_mean": bool(
            (
                predictions["failure_probability_q05"]
                <= predictions["failure_probability"]
            ).all()
            and (
                predictions["failure_probability"]
                <= predictions["failure_probability_q95"]
            ).all()
        ),
        "reach_probabilities_finite_bounded": bool(
            np.isfinite(reach).all()
            and np.logical_and(reach >= 0.0, reach <= 1.0).all()
        ),
        "reach_probabilities_nondecreasing": bool((np.diff(reach, axis=1) >= 0.0).all()),
        "posterior_statistics_finite": bool(
            np.isfinite(
                support[
                    [
                        "training_count",
                        "posterior_mean",
                        "posterior_std",
                        "shrinkage_weight",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "shrinkage_weights_bounded": bool(support["shrinkage_weight"].between(0.0, 1.0).all()),
        "training_group_support_exact": _training_group_support_exact(
            predictions, support, features, splits, feature_schema
        ),
        "family_holdout_uses_unseen_family_fallback": family_fallback,
        "required_metrics_finite": metrics_finite,
        "aggregate_metrics_complete": len(aggregate) == expected_aggregate_rows,
        "fold_preprocessing_contract_complete": len(fold_schemas["folds"]) == len(jobs),
        "phase11_scope_exact": registry["targets"]
        == [
            "quality_distribution",
            "runtime_distribution",
            "failure_probability",
            "first_passage_survival",
        ],
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    issues = []
    if failed_checks:
        issues.append({"type": "failed_quality_checks", "checks": failed_checks})
    return {
        "status": "PHASE_11_PASS" if all(checks.values()) else "PHASE_11_FAIL",
        "schema_version": PHASE11_SCHEMA_VERSION,
        "scope": "leakage-safe hierarchical Bayesian partial pooling",
        "performance_gate_policy": (
            "construction, leakage, posterior validity, shrinkage, OOF coverage, and integrity "
            "only; predictive deltas versus Phases 8 through 10 are not Phase 11 pass thresholds"
        ),
        "phase12_boundary": "Mixture-of-Experts routing is deferred to Phase 12",
        "phase6_directory": str(paths[0].resolve()),
        "phase7_directory": str(paths[1].resolve()),
        "phase8_directory": str(paths[2].resolve()),
        "phase9_directory": str(paths[3].resolve()),
        "phase10_directory": str(paths[4].resolve()),
        "feature_row_count": int(len(features)),
        "prediction_row_count": int(len(predictions)),
        "job_count": int(len(fold_metrics)),
        "model_artifact_count": int(len(registry["jobs"])),
        "aggregate_metric_row_count": int(len(aggregate)),
        "posterior_row_count": int(len(support)),
        "group_level_count": len(GROUP_LEVELS),
        "quantile_count": len(QUANTILE_LEVELS),
        "survival_horizon_count": len(SURVIVAL_HORIZONS),
        "observed_failure_count": int(labels["observed_failure"].sum()),
        "observed_event_count": int(labels["event_observed"].sum()),
        "checks": checks,
        "issues": issues,
    }


def run_phase11(
    run_directory: str | Path,
    phase6_directory: str | Path,
    phase7_directory: str | Path,
    phase8_directory: str | Path,
    phase9_directory: str | Path,
    phase10_directory: str | Path,
    *,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
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
    )
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"
    if ridge_alpha <= 0.0:
        raise ValueError("Phase 11 Ridge alpha must be positive")
    if prior_strength <= 0.0:
        raise ValueError("Phase 11 prior strength must be positive")
    if len(beta_prior) != 2 or any(value <= 0.0 for value in beta_prior):
        raise ValueError("Phase 11 beta prior alpha and beta must be positive")
    source = _load_inputs(*paths)
    stable_configuration = {
        "phase": 11,
        "schema_version": PHASE11_SCHEMA_VERSION,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "preprocessing_implementation_sha256": _file_sha256(Path(phase7_module.__file__)),
        "input_fingerprints": source["input_fingerprints"],
        "continuous_model": CONTINUOUS_MODEL_NAME,
        "binary_model": BINARY_MODEL_NAME,
        "ridge_alpha": float(ridge_alpha),
        "prior_strength": float(prior_strength),
        "beta_prior": {"alpha": float(beta_prior[0]), "beta": float(beta_prior[1])},
        "group_levels": [
            {"name": name, "columns": list(columns)} for name, columns in GROUP_LEVELS
        ],
        "quantile_levels": list(QUANTILE_LEVELS),
        "survival_horizons": list(SURVIVAL_HORIZONS),
        "split_columns": source["feature_schema"]["split_columns"],
        "cutoffs": source["feature_schema"]["cutoffs"],
        "pooling_policy": (
            "normal-normal sequential residual effects for quality/log-runtime; hierarchical "
            "beta-binomial deepest-available posterior for failure/reach"
        ),
    }
    configuration_sha256 = _configuration_hash(stable_configuration)
    configuration = {
        **stable_configuration,
        "phase6_directory": str(paths[0].resolve()),
        "phase7_directory": str(paths[1].resolve()),
        "phase8_directory": str(paths[2].resolve()),
        "phase9_directory": str(paths[3].resolve()),
        "phase10_directory": str(paths[4].resolve()),
        "failure_label_source": str(source["failure_source"].resolve()),
        "runtime_label_source": str(source["runtime_source"].resolve()),
        "git_commit": current_git_commit(),
        "configuration_sha256": configuration_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Phase 11 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_11_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 11 resume manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest.get("output_sha256", {}).items():
                path = run_path / relative
                if not path.is_file() or _file_sha256(path) != expected_hash:
                    raise ValueError(f"completed Phase 11 output hash mismatch: {relative}")
            active_logger.info("[PHASE11][RESUME] status=PHASE_11_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info(
            "[PHASE11][RESUME] completed_jobs=%s", len(state.get("completed_jobs", {}))
        )
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE11_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": configuration_sha256,
            "input_fingerprints": source["input_fingerprints"],
            "completed_jobs": {},
            "last_completed_job": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase11_started"})
    fold_schemas, job_outputs = _run_jobs(
        run_path,
        source["features"],
        source["labels"],
        source["references"],
        source["splits"],
        source["feature_schema"],
        source["preprocessing"],
        state,
        ridge_alpha=ridge_alpha,
        prior_strength=prior_strength,
        beta_prior=beta_prior,
        progress_path=progress_path,
        logger=active_logger,
    )
    jobs = _all_jobs(source["preprocessing"])
    aggregate_outputs, _registry = _aggregate_job_outputs(
        run_path, jobs, source["labels"]
    )
    validation = _validate_phase11(
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
        "phase": 11,
        "status": validation["status"],
        "schema_version": PHASE11_SCHEMA_VERSION,
        "input_fingerprints": source["input_fingerprints"],
        "output_sha256": _relative_hashes(run_path, output_paths),
        "row_counts": {
            "features": validation["feature_row_count"],
            "predictions": validation["prediction_row_count"],
            "jobs": validation["job_count"],
            "aggregate_metrics": validation["aggregate_metric_row_count"],
            "group_posteriors": validation["posterior_row_count"],
        },
        "phase12_allowed": validation["status"] == "PHASE_11_PASS",
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
        progress_path, {"event": "phase11_finished", "status": validation["status"]}
    )
    active_logger.info(
        "[PHASE11][SUMMARY] status=%s jobs=%s predictions=%s posteriors=%s directory=%s",
        validation["status"],
        validation["job_count"],
        validation["prediction_row_count"],
        validation["posterior_row_count"],
        run_path,
    )
    return validation
