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
from sklearn.ensemble import HistGradientBoostingClassifier

import uriel_v2.probabilistic_lab.phase7 as phase7_module
from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.phase7 import (
    _append_progress,
    _atomic_json,
    _atomic_parquet,
    _atomic_pickle,
    _configuration_hash,
    _file_sha256,
    _relative_hashes,
    _transform_features,
    _verify_fold_contract,
)
from uriel_v2.probabilistic_lab.phase8 import (
    QUANTILE_LEVELS,
    _cdf_at,
    _crps as _quality_crps,
    _distribution_metrics,
)
from uriel_v2.probabilistic_lab.phase9 import _binary_metrics
from uriel_v2.probabilistic_lab.phase10 import (
    SURVIVAL_HORIZONS,
    _horizon_observation,
    _runtime_cdf,
    _runtime_crps,
    _runtime_metrics,
    _runtime_moments,
    _survival_metrics,
    _survival_summaries,
)
from uriel_v2.probabilistic_lab.phase11 import (
    _combined_input_fingerprints as _phase11_input_fingerprints,
    _load_inputs as _load_phase11_inputs,
)
from uriel_v2.provenance import current_git_commit


PHASE12_SCHEMA_VERSION = "phase12-v1"
GATE_MODEL_NAME = "cross_fitted_hist_gradient_boosting_gate"
TARGET_NAMES = ("quality", "runtime", "failure", "survival")
EXPERT_SLOTS: tuple[tuple[str, str | None], ...] = (
    ("sampling", "sampling"),
    ("optimization", "optimization"),
    ("matrix", "matrix"),
    ("stream", "stream"),
    ("natural_process", "natural_process"),
    ("universal", None),
)
DEFAULT_GATE_ITERATIONS = 40
DEFAULT_MINIMUM_GATE_ROWS = 100
DEFAULT_GATE_WEIGHT_CLIP = (0.02, 0.98)
DEFAULT_BETA_PRIOR = (0.5, 0.5)
DEFAULT_CALIBRATION_BINS = 10
PROBABILITY_EPSILON = 1e-12


def _quality_column(source: str, level: float) -> str:
    return f"{source}_quality_q{int(round(level * 100.0)):02d}"


def _runtime_column(source: str, level: float) -> str:
    return f"{source}_runtime_q{int(round(level * 100.0)):02d}"


def _horizon_suffix(horizon: float) -> str:
    return f"p{int(round(horizon * 100.0)):03d}"


def _reach_column(source: str, horizon: float) -> str:
    return f"{source}_reach_by_{_horizon_suffix(horizon)}"


def _phase11_required_paths(phase11_path: Path) -> dict[str, Path]:
    return {
        "phase11_config": phase11_path / "config.json",
        "phase11_manifest": phase11_path / "manifest.json",
        "phase11_validation": phase11_path / "validation.json",
        "phase11_predictions": phase11_path
        / "data/predictions/oof_hierarchical_predictions.parquet",
        "phase11_labels": phase11_path / "data/targets/hierarchical_labels.parquet",
        "phase11_fold_metrics": phase11_path
        / "data/metrics/fold_hierarchical_metrics.parquet",
        "phase11_aggregate_metrics": phase11_path
        / "data/metrics/aggregate_hierarchical_metrics.parquet",
        "phase11_posteriors": phase11_path / "data/posteriors/group_posteriors.parquet",
        "phase11_fold_schemas": phase11_path
        / "data/preprocessing/fold_feature_schemas.json",
        "phase11_model_registry": phase11_path / "model_registry.json",
    }


def _validate_phase11_input(
    phase11_path: Path,
    expected_input_fingerprints: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    required = _phase11_required_paths(phase11_path)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 12 Phase 11 input files are missing: {missing[:5]}")
    validation = json.loads(required["phase11_validation"].read_text(encoding="utf-8"))
    manifest = json.loads(required["phase11_manifest"].read_text(encoding="utf-8"))
    if validation.get("status") != "PHASE_11_PASS":
        raise ValueError("Phase 12 requires a PHASE_11_PASS hierarchical run")
    if manifest.get("status") != "PHASE_11_PASS" or not manifest.get("phase12_allowed"):
        raise ValueError("Phase 11 manifest does not allow Phase 12")
    if validation.get("configuration", {}).get("input_fingerprints") != expected_input_fingerprints:
        raise ValueError("Phase 11 was not built from the supplied Phase 6 through 10 inputs")
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = phase11_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 11 manifest output hash mismatch: {relative}")
    fingerprints = {name: _file_sha256(path) for name, path in sorted(required.items())}
    return validation, manifest, fingerprints


def _combined_input_fingerprints(
    phase11_expected_inputs: dict[str, str],
    phase11_fingerprints: dict[str, str],
) -> dict[str, str]:
    return {
        **phase11_expected_inputs,
        **{f"phase11/{name}": value for name, value in phase11_fingerprints.items()},
    }


def _load_inputs(
    phase6_path: Path,
    phase7_path: Path,
    phase8_path: Path,
    phase9_path: Path,
    phase10_path: Path,
    phase11_path: Path,
) -> dict[str, Any]:
    phase11_source = _load_phase11_inputs(
        phase6_path, phase7_path, phase8_path, phase9_path, phase10_path
    )
    phase11_expected = phase11_source["input_fingerprints"]
    phase11_validation, _phase11_manifest, phase11_fingerprints = _validate_phase11_input(
        phase11_path, phase11_expected
    )
    predictions = pd.read_parquet(
        phase11_path / "data/predictions/oof_hierarchical_predictions.parquet"
    )
    labels = pd.read_parquet(phase11_path / "data/targets/hierarchical_labels.parquet")
    expected_rows = len(phase11_source["features"]) * len(
        phase11_source["preprocessing"]["splits"]
    )
    if len(predictions) != expected_rows or predictions.duplicated(
        ["feature_id", "split_name"]
    ).any():
        raise ValueError("Phase 12 requires exact unique Phase 11 OOF coverage")
    return {
        **phase11_source,
        "validations": {
            **phase11_source["validations"],
            "phase11": phase11_validation,
        },
        "input_fingerprints": _combined_input_fingerprints(
            phase11_expected, phase11_fingerprints
        ),
        "phase11_predictions": predictions,
        "phase11_labels": labels,
    }


def _ordered_predictions(
    predictions: pd.DataFrame,
    split_name: str,
    feature_ids: pd.Series,
) -> pd.DataFrame:
    group = predictions.loc[predictions["split_name"] == split_name].copy()
    indexed = group.set_index(group["feature_id"].astype(str))
    ordered = indexed.loc[feature_ids.astype(str)].reset_index(drop=True)
    if not np.array_equal(
        ordered["feature_id"].astype(str).to_numpy(), feature_ids.astype(str).to_numpy()
    ):
        raise ValueError("Phase 12 could not align Phase 11 OOF predictions")
    return ordered


def _survival_row_brier(frame: pd.DataFrame, source: str) -> np.ndarray:
    event = frame["event_observed"].to_numpy(dtype=bool)
    duration = frame["duration_step"].to_numpy(dtype=float)
    budget = frame["budget"].to_numpy(dtype=float)
    total = np.zeros(len(frame), dtype=float)
    count = np.zeros(len(frame), dtype=int)
    for horizon in SURVIVAL_HORIZONS:
        observable, outcome = _horizon_observation(event, duration, budget, horizon)
        probability = frame[_reach_column(source, horizon)].to_numpy(dtype=float)
        total[observable] += (
            outcome[observable].astype(float) - probability[observable]
        ) ** 2
        count[observable] += 1
    if (count == 0).any():
        raise ValueError("Phase 12 survival gate loss requires an observable horizon per row")
    return total / count


def _source_loss(frame: pd.DataFrame, target: str, source: str) -> np.ndarray:
    if target == "quality":
        columns = [_quality_column(source, level) for level in QUANTILE_LEVELS]
        return _quality_crps(
            frame["observed_quality"].to_numpy(dtype=float),
            frame[columns].to_numpy(dtype=float),
        )
    if target == "runtime":
        columns = [_runtime_column(source, level) for level in QUANTILE_LEVELS]
        return _runtime_crps(
            frame["observed_runtime"].to_numpy(dtype=float),
            frame[columns].to_numpy(dtype=float),
        )
    if target == "failure":
        observed = frame["observed_failure"].to_numpy(dtype=bool).astype(float)
        probability = frame[f"{source}_failure_probability"].to_numpy(dtype=float)
        return (observed - probability) ** 2
    return _survival_row_brier(frame, source)


def _gate_seed(master_seed: int, job_id: str, target: str, expert_slot: str) -> int:
    token = f"{job_id}:{target}:{expert_slot}"
    offset = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
    return int((master_seed + offset) % (2**31 - 1))


def _fit_gate(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    base_loss: np.ndarray,
    hierarchical_loss: np.ndarray,
    *,
    seed: int,
    gate_iterations: int,
    weight_clip: tuple[float, float],
    beta_prior: tuple[float, float],
) -> tuple[HistGradientBoostingClassifier | None, np.ndarray, dict[str, Any]]:
    base_loss = np.asarray(base_loss, dtype=float)
    hierarchical_loss = np.asarray(hierarchical_loss, dtype=float)
    if not np.isfinite(base_loss).all() or not np.isfinite(hierarchical_loss).all():
        raise ValueError("Phase 12 gate losses must be finite")
    hierarchical_win = hierarchical_loss < base_loss
    alpha, beta = beta_prior
    estimator: HistGradientBoostingClassifier | None = None
    if np.unique(hierarchical_win).size < 2:
        constant = float(
            (int(hierarchical_win.sum()) + alpha)
            / (len(hierarchical_win) + alpha + beta)
        )
        weight = np.full(len(x_validation), constant, dtype=float)
        fit_status = "beta_binomial_fallback"
    else:
        relative_gap = np.abs(base_loss - hierarchical_loss) / np.maximum(
            base_loss + hierarchical_loss, PROBABILITY_EPSILON
        )
        scale = float(np.quantile(relative_gap, 0.90))
        sample_weight = 0.10 + 0.90 * np.clip(
            relative_gap / max(scale, PROBABILITY_EPSILON), 0.0, 1.0
        )
        estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=gate_iterations,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=max(20, min(100, len(x_train) // 20)),
            l2_regularization=2.0,
            early_stopping=False,
            random_state=seed,
        )
        estimator.fit(x_train, hierarchical_win.astype(int), sample_weight=sample_weight)
        positive_index = list(estimator.classes_).index(1)
        weight = (
            estimator.predict_proba(x_validation)[:, positive_index]
            if len(x_validation)
            else np.empty(0, dtype=float)
        )
        constant = None
        fit_status = "fitted"
    weight = np.clip(weight, weight_clip[0], weight_clip[1])
    diagnostics = {
        "fit_status": fit_status,
        "constant_hierarchical_weight": constant,
        "training_row_count": int(len(x_train)),
        "training_hierarchical_win_count": int(hierarchical_win.sum()),
        "training_hierarchical_win_rate": float(hierarchical_win.mean()),
        "training_base_loss_mean": float(base_loss.mean()),
        "training_hierarchical_loss_mean": float(hierarchical_loss.mean()),
        "validation_row_count": int(len(x_validation)),
        "validation_hierarchical_weight_mean": (
            float(weight.mean()) if len(weight) else None
        ),
    }
    return estimator, weight, diagnostics


def _route_target(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    training_domains: np.ndarray,
    validation_domains: np.ndarray,
    base_loss: np.ndarray,
    hierarchical_loss: np.ndarray,
    *,
    target: str,
    job_id: str,
    master_seed: int,
    gate_iterations: int,
    minimum_gate_rows: int,
    weight_clip: tuple[float, float],
    beta_prior: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    universal_model, universal_weight, universal_diagnostics = _fit_gate(
        x_train,
        x_validation,
        base_loss,
        hierarchical_loss,
        seed=_gate_seed(master_seed, job_id, target, "universal"),
        gate_iterations=gate_iterations,
        weight_clip=weight_clip,
        beta_prior=beta_prior,
    )
    weights = universal_weight.copy()
    routed_expert = np.full(len(x_validation), "universal", dtype=object)
    models: dict[str, Any] = {
        "universal": {
            "domain": None,
            "estimator": universal_model,
            **universal_diagnostics,
        }
    }
    support_rows = [
        {
            "target": target,
            "expert_slot": "universal",
            "domain": None,
            "availability_status": "available",
            "routed_validation_count": int(len(x_validation)),
            **universal_diagnostics,
        }
    ]
    for expert_slot, domain in EXPERT_SLOTS:
        if expert_slot == "universal":
            continue
        train_mask = training_domains == domain
        validation_mask = validation_domains == domain
        training_count = int(train_mask.sum())
        validation_count = int(validation_mask.sum())
        if training_count == 0:
            diagnostics = {
                "fit_status": "unavailable_no_executed_rows",
                "constant_hierarchical_weight": None,
                "training_row_count": 0,
                "training_hierarchical_win_count": 0,
                "training_hierarchical_win_rate": None,
                "training_base_loss_mean": None,
                "training_hierarchical_loss_mean": None,
                "validation_row_count": validation_count,
                "validation_hierarchical_weight_mean": None,
            }
            models[expert_slot] = {"domain": domain, "estimator": None, **diagnostics}
            support_rows.append(
                {
                    "target": target,
                    "expert_slot": expert_slot,
                    "domain": domain,
                    "availability_status": "unavailable_no_executed_rows",
                    "routed_validation_count": 0,
                    **diagnostics,
                }
            )
            continue
        if training_count < minimum_gate_rows:
            diagnostics = {
                "fit_status": "unavailable_insufficient_training_rows",
                "constant_hierarchical_weight": None,
                "training_row_count": training_count,
                "training_hierarchical_win_count": int(
                    (hierarchical_loss[train_mask] < base_loss[train_mask]).sum()
                ),
                "training_hierarchical_win_rate": float(
                    (hierarchical_loss[train_mask] < base_loss[train_mask]).mean()
                ),
                "training_base_loss_mean": float(base_loss[train_mask].mean()),
                "training_hierarchical_loss_mean": float(
                    hierarchical_loss[train_mask].mean()
                ),
                "validation_row_count": validation_count,
                "validation_hierarchical_weight_mean": None,
            }
            models[expert_slot] = {"domain": domain, "estimator": None, **diagnostics}
            support_rows.append(
                {
                    "target": target,
                    "expert_slot": expert_slot,
                    "domain": domain,
                    "availability_status": "unavailable_insufficient_training_rows",
                    "routed_validation_count": 0,
                    **diagnostics,
                }
            )
            continue
        estimator, specialist_weight, diagnostics = _fit_gate(
            x_train[train_mask],
            x_validation[validation_mask],
            base_loss[train_mask],
            hierarchical_loss[train_mask],
            seed=_gate_seed(master_seed, job_id, target, expert_slot),
            gate_iterations=gate_iterations,
            weight_clip=weight_clip,
            beta_prior=beta_prior,
        )
        weights[validation_mask] = specialist_weight
        routed_expert[validation_mask] = expert_slot
        models[expert_slot] = {"domain": domain, "estimator": estimator, **diagnostics}
        support_rows.append(
            {
                "target": target,
                "expert_slot": expert_slot,
                "domain": domain,
                "availability_status": "available",
                "routed_validation_count": validation_count,
                **diagnostics,
            }
        )
    support = pd.DataFrame(support_rows)
    numeric_columns = (
        "constant_hierarchical_weight",
        "training_row_count",
        "training_hierarchical_win_count",
        "training_hierarchical_win_rate",
        "training_base_loss_mean",
        "training_hierarchical_loss_mean",
        "validation_row_count",
        "validation_hierarchical_weight_mean",
        "routed_validation_count",
    )
    for column in numeric_columns:
        support[column] = pd.to_numeric(support[column], errors="coerce").astype(float)
    return weights, routed_expert, models, support


def _source_frame(phase11_frame: pd.DataFrame) -> pd.DataFrame:
    frame = phase11_frame[
        [
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
    ].copy()
    for level in QUANTILE_LEVELS:
        suffix = f"q{int(round(level * 100.0)):02d}"
        frame[_quality_column("base", level)] = phase11_frame[f"reference_quality_{suffix}"]
        frame[_quality_column("hierarchical", level)] = phase11_frame[f"quality_{suffix}"]
        frame[_runtime_column("base", level)] = phase11_frame[f"reference_runtime_{suffix}"]
        frame[_runtime_column("hierarchical", level)] = phase11_frame[f"runtime_{suffix}"]
    frame["base_failure_probability"] = phase11_frame["reference_failure_probability"]
    frame["hierarchical_failure_probability"] = phase11_frame["failure_probability"]
    for horizon in SURVIVAL_HORIZONS:
        suffix = _horizon_suffix(horizon)
        frame[_reach_column("base", horizon)] = phase11_frame[f"reference_reach_by_{suffix}"]
        frame[_reach_column("hierarchical", horizon)] = phase11_frame[f"reach_by_{suffix}"]
    return frame


def _mix_predictions(
    validation_source: pd.DataFrame,
    weights: dict[str, np.ndarray],
    routed_experts: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = validation_source.reset_index(drop=True).copy()
    for target in TARGET_NAMES:
        frame[f"{target}_hierarchical_weight"] = weights[target]
        frame[f"{target}_router_expert"] = routed_experts[target]
        frame[f"{target}_selected_source"] = np.where(
            weights[target] >= 0.5, "hierarchical", "base"
        )
    quality_weight = weights["quality"]
    runtime_weight = weights["runtime"]
    for level in QUANTILE_LEVELS:
        base_quality = frame[_quality_column("base", level)].to_numpy(dtype=float)
        hierarchical_quality = frame[
            _quality_column("hierarchical", level)
        ].to_numpy(dtype=float)
        frame[_quality_column("moe", level)] = (
            (1.0 - quality_weight) * base_quality
            + quality_weight * hierarchical_quality
        )
        base_runtime = frame[_runtime_column("base", level)].to_numpy(dtype=float)
        hierarchical_runtime = frame[
            _runtime_column("hierarchical", level)
        ].to_numpy(dtype=float)
        frame[_runtime_column("moe", level)] = (
            (1.0 - runtime_weight) * base_runtime
            + runtime_weight * hierarchical_runtime
        )
    failure_weight = weights["failure"]
    frame["moe_failure_probability"] = (
        (1.0 - failure_weight) * frame["base_failure_probability"].to_numpy(dtype=float)
        + failure_weight
        * frame["hierarchical_failure_probability"].to_numpy(dtype=float)
    )
    survival_weight = weights["survival"]
    for horizon in SURVIVAL_HORIZONS:
        frame[_reach_column("moe", horizon)] = (
            (1.0 - survival_weight)
            * frame[_reach_column("base", horizon)].to_numpy(dtype=float)
            + survival_weight
            * frame[_reach_column("hierarchical", horizon)].to_numpy(dtype=float)
        )
    quality_quantiles = frame[
        [_quality_column("moe", level) for level in QUANTILE_LEVELS]
    ].to_numpy(dtype=float)
    runtime_quantiles = frame[
        [_runtime_column("moe", level) for level in QUANTILE_LEVELS]
    ].to_numpy(dtype=float)
    frame["moe_quality_pit"] = _cdf_at(
        quality_quantiles, frame["observed_quality"].to_numpy(dtype=float)
    )
    frame["moe_runtime_pit"] = _runtime_cdf(
        runtime_quantiles, frame["observed_runtime"].to_numpy(dtype=float)
    )
    runtime_mean, runtime_std = _runtime_moments(runtime_quantiles)
    frame["moe_runtime_predictive_mean"] = runtime_mean
    frame["moe_runtime_predictive_std"] = runtime_std
    reach = frame[
        [_reach_column("moe", horizon) for horizon in SURVIVAL_HORIZONS]
    ].to_numpy(dtype=float)
    restricted_mean, expected_par10 = _survival_summaries(
        reach, frame["budget"].to_numpy(dtype=float)
    )
    frame["moe_predicted_restricted_mean_step"] = restricted_mean
    frame["moe_predicted_par10"] = expected_par10
    for target in TARGET_NAMES:
        frame[f"base_{target}_gate_loss"] = _source_loss(frame, target, "base")
        frame[f"hierarchical_{target}_gate_loss"] = _source_loss(
            frame, target, "hierarchical"
        )
        frame[f"moe_{target}_gate_loss"] = _source_loss(frame, target, "moe")
        uniform_frame = frame.copy()
        if target in {"quality", "runtime"}:
            levels = QUANTILE_LEVELS
            column_builder = _quality_column if target == "quality" else _runtime_column
            for level in levels:
                uniform_frame[column_builder("uniform", level)] = 0.5 * (
                    frame[column_builder("base", level)]
                    + frame[column_builder("hierarchical", level)]
                )
        elif target == "failure":
            uniform_frame["uniform_failure_probability"] = 0.5 * (
                frame["base_failure_probability"]
                + frame["hierarchical_failure_probability"]
            )
        else:
            for horizon in SURVIVAL_HORIZONS:
                uniform_frame[_reach_column("uniform", horizon)] = 0.5 * (
                    frame[_reach_column("base", horizon)]
                    + frame[_reach_column("hierarchical", horizon)]
                )
        frame[f"uniform_{target}_gate_loss"] = _source_loss(
            uniform_frame, target, "uniform"
        )
        frame[f"oracle_{target}_gate_loss"] = np.minimum(
            frame[f"base_{target}_gate_loss"],
            frame[f"hierarchical_{target}_gate_loss"],
        )
    return frame


def _short_runtime_metrics(values: dict[str, Any]) -> dict[str, Any]:
    return {
        (key.removeprefix("runtime_") if key.startswith("runtime_") else key): value
        for key, value in values.items()
    }


def _metric_payload(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    observed_quality = frame["observed_quality"].to_numpy(dtype=float)
    observed_runtime = frame["observed_runtime"].to_numpy(dtype=float)
    observed_failure = frame["observed_failure"].to_numpy(dtype=bool)
    event = frame["event_observed"].to_numpy(dtype=bool)
    duration = frame["duration_step"].to_numpy(dtype=float)
    budget = frame["budget"].to_numpy(dtype=float)
    source_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for source in ("moe", "base", "hierarchical"):
        quality = _distribution_metrics(
            observed_quality,
            frame[
                [_quality_column(source, level) for level in QUANTILE_LEVELS]
            ].to_numpy(dtype=float),
        )
        runtime = _short_runtime_metrics(
            _runtime_metrics(
                observed_runtime,
                frame[
                    [_runtime_column(source, level) for level in QUANTILE_LEVELS]
                ].to_numpy(dtype=float),
            )
        )
        failure = _binary_metrics(
            observed_failure,
            frame[f"{source}_failure_probability"].to_numpy(dtype=float),
            calibration_bins=DEFAULT_CALIBRATION_BINS,
        )
        survival = _survival_metrics(
            event,
            duration,
            budget,
            frame[
                [_reach_column(source, horizon) for horizon in SURVIVAL_HORIZONS]
            ].to_numpy(dtype=float),
        )
        source_metrics[source] = {
            "quality": quality,
            "runtime": runtime,
            "failure": failure,
            "survival": survival,
        }
        for target, metrics in source_metrics[source].items():
            result.update(
                {f"{source}_{target}_{key}": value for key, value in metrics.items()}
            )
    for target in TARGET_NAMES:
        result[f"moe_{target}_loss_delta_vs_base"] = float(
            frame[f"moe_{target}_gate_loss"].mean()
            - frame[f"base_{target}_gate_loss"].mean()
        )
        result[f"moe_{target}_loss_delta_vs_hierarchical"] = float(
            frame[f"moe_{target}_gate_loss"].mean()
            - frame[f"hierarchical_{target}_gate_loss"].mean()
        )
        result[f"moe_{target}_oracle_regret"] = float(
            frame[f"moe_{target}_gate_loss"].mean()
            - frame[f"oracle_{target}_gate_loss"].mean()
        )
        for source in ("moe", "base", "hierarchical", "uniform", "oracle"):
            result[f"{source}_{target}_gate_loss_mean"] = float(
                frame[f"{source}_{target}_gate_loss"].mean()
            )
        weight = frame[f"{target}_hierarchical_weight"].to_numpy(dtype=float)
        oracle_hierarchical = (
            frame[f"hierarchical_{target}_gate_loss"].to_numpy(dtype=float)
            < frame[f"base_{target}_gate_loss"].to_numpy(dtype=float)
        )
        result[f"{target}_hierarchical_weight_mean"] = float(weight.mean())
        result[f"{target}_hierarchical_route_share"] = float(np.mean(weight >= 0.5))
        result[f"{target}_route_accuracy"] = float(
            np.mean((weight >= 0.5) == oracle_hierarchical)
        )
        entropy = -weight * np.log(np.clip(weight, PROBABILITY_EPSILON, 1.0)) - (
            1.0 - weight
        ) * np.log(np.clip(1.0 - weight, PROBABILITY_EPSILON, 1.0))
        result[f"{target}_gate_entropy_mean"] = float(entropy.mean())
    for target in ("quality", "runtime", "failure", "survival"):
        for metric in (
            "nll" if target in {"quality", "runtime", "survival"} else "brier",
        ):
            metric_name = (
                "survival_nll"
                if target == "survival"
                else ("nll" if target == "quality" else metric)
            )
            if target == "runtime":
                metric_name = "nll"
            result[f"moe_{target}_{metric_name}_delta_vs_base"] = (
                source_metrics["moe"][target][metric_name]
                - source_metrics["base"][target][metric_name]
            )
            result[f"moe_{target}_{metric_name}_delta_vs_hierarchical"] = (
                source_metrics["moe"][target][metric_name]
                - source_metrics["hierarchical"][target][metric_name]
            )
    return result


def _job_id(split_name: str, fold: int) -> str:
    return f"{split_name}__fold{fold}__mixture_of_experts"


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
            raise ValueError(f"Phase 12 job output hash mismatch: job={job_id} path={relative}")
    return marker


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
    phase11_predictions: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
    state: dict[str, Any],
    *,
    master_seed: int,
    gate_iterations: int,
    minimum_gate_rows: int,
    weight_clip: tuple[float, float],
    beta_prior: tuple[float, float],
    progress_path: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[Path]]:
    fold_schemas: dict[str, Any] = {
        "schema_version": PHASE12_SCHEMA_VERSION,
        "source_preprocessing_schema_version": preprocessing["schema_version"],
        "transform_policy": (
            "reuse verified Phase 6 fold preprocessing; gate labels use only cross-fitted "
            "Phase 8 through 11 OOF predictions from non-validation folds"
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
            x_train, feature_names = _transform_features(
                features.loc[training_mask], fold_specification, feature_schema
            )
            x_validation, validation_feature_names = _transform_features(
                features.loc[validation_mask], fold_specification, feature_schema
            )
            if feature_names != validation_feature_names:
                raise ValueError(f"Phase 12 transformed schema mismatch: {fold_key}")
            training_source = _source_frame(
                _ordered_predictions(
                    phase11_predictions,
                    split_name,
                    features.loc[training_mask, "feature_id"],
                )
            )
            validation_source = _source_frame(
                _ordered_predictions(
                    phase11_predictions,
                    split_name,
                    features.loc[validation_mask, "feature_id"],
                )
            )
            meta_training_current_fold_count = int((training_source["fold"] == fold).sum())
            validation_wrong_fold_count = int((validation_source["fold"] != fold).sum())
            if meta_training_current_fold_count or validation_wrong_fold_count:
                raise ValueError(f"Phase 12 cross-fitting contract failed: {fold_key}")
            fold_schemas["folds"][fold_key] = {
                **fold_contract,
                "fold_column": fold_column,
                "transformed_feature_count": len(feature_names),
                "transformed_feature_names": feature_names,
                "transformed_feature_names_sha256": hashlib.sha256(
                    "\n".join(feature_names).encode("utf-8")
                ).hexdigest(),
                "meta_training_current_fold_count": meta_training_current_fold_count,
                "validation_wrong_fold_count": validation_wrong_fold_count,
            }
            job_id = _job_id(split_name, fold)
            paths = _job_paths(run_path, job_id)
            marker = _verify_job(run_path, job_id)
            if marker is not None:
                state.setdefault("completed_jobs", {})[job_id] = _file_sha256(paths["marker"])
                output_paths.extend(paths.values())
                logger.info("[PHASE12][RESUME] job=%s status=SKIP_VERIFIED", job_id)
                continue
            logger.info(
                "[PHASE12][FOLD] split=%s fold=%s train_rows=%s validation_rows=%s",
                split_name,
                fold,
                int(training_mask.sum()),
                int(validation_mask.sum()),
            )
            training_domains = training_source["domain"].astype(str).to_numpy()
            validation_domains = validation_source["domain"].astype(str).to_numpy()
            weights: dict[str, np.ndarray] = {}
            routed_experts: dict[str, np.ndarray] = {}
            gate_models: dict[str, Any] = {}
            support_frames = []
            for target in TARGET_NAMES:
                base_loss = _source_loss(training_source, target, "base")
                hierarchical_loss = _source_loss(
                    training_source, target, "hierarchical"
                )
                weight, routed, models, support = _route_target(
                    x_train,
                    x_validation,
                    training_domains,
                    validation_domains,
                    base_loss,
                    hierarchical_loss,
                    target=target,
                    job_id=job_id,
                    master_seed=master_seed,
                    gate_iterations=gate_iterations,
                    minimum_gate_rows=minimum_gate_rows,
                    weight_clip=weight_clip,
                    beta_prior=beta_prior,
                )
                weights[target] = weight
                routed_experts[target] = routed
                gate_models[target] = models
                support_frames.append(support)
            predictions = _mix_predictions(validation_source, weights, routed_experts)
            support = pd.concat(support_frames, ignore_index=True)
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
            model_artifact = {
                "schema_version": PHASE12_SCHEMA_VERSION,
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "gate_model": GATE_MODEL_NAME,
                "expert_slots": [slot for slot, _domain in EXPERT_SLOTS],
                "feature_names": feature_names,
                "fold_contract": fold_contract,
                "gate_models": gate_models,
                "weight_clip": weight_clip,
                "routing_policy": (
                    "domain specialist when training support is available; otherwise universal "
                    "fallback; soft weight blends frozen base and hierarchical predictions"
                ),
            }
            _atomic_parquet(paths["predictions"], predictions)
            _atomic_json(paths["metrics"], metrics)
            _atomic_parquet(paths["support"], support)
            _atomic_pickle(paths["model"], model_artifact)
            marker = {
                "schema_version": PHASE12_SCHEMA_VERSION,
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
                "[PHASE12][JOB] job=%s quality_hier_weight=%.4f survival_hier_weight=%.4f",
                job_id,
                float(weights["quality"].mean()),
                float(weights["survival"].mean()),
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
        "schema_version": PHASE12_SCHEMA_VERSION,
        "targets": list(TARGET_NAMES),
        "gate_model": GATE_MODEL_NAME,
        "expert_slots": [
            {"name": name, "domain": domain} for name, domain in EXPERT_SLOTS
        ],
        "base_expert_source": "Phase 8 quality, Phase 9 failure, Phase 10 runtime-survival",
        "hierarchical_expert_source": "Phase 11 hierarchical partial pooling",
        "gate_training_policy": "cross-fitted OOF losses from non-validation folds only",
        "jobs": [],
    }
    for split_name, fold in jobs:
        job_id = _job_id(split_name, fold)
        marker = _verify_job(run_path, job_id)
        if marker is None:
            raise ValueError(f"Phase 12 job is incomplete: {job_id}")
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
        ["split_name", "fold", "target", "expert_slot"]
    ).reset_index(drop=True)
    aggregate = _aggregate_metrics(predictions)
    predictions_path = run_path / "data/predictions/oof_mixture_predictions.parquet"
    labels_path = run_path / "data/targets/mixture_labels.parquet"
    fold_metrics_path = run_path / "data/metrics/fold_mixture_metrics.parquet"
    aggregate_path = run_path / "data/metrics/aggregate_mixture_metrics.parquet"
    support_path = run_path / "data/support/expert_gate_support.parquet"
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


def _mixture_identity_exact(predictions: pd.DataFrame) -> bool:
    for target, levels, builder in (
        ("quality", QUANTILE_LEVELS, _quality_column),
        ("runtime", QUANTILE_LEVELS, _runtime_column),
    ):
        weight = predictions[f"{target}_hierarchical_weight"].to_numpy(dtype=float)
        for level in levels:
            expected = (
                (1.0 - weight) * predictions[builder("base", level)].to_numpy(dtype=float)
                + weight
                * predictions[builder("hierarchical", level)].to_numpy(dtype=float)
            )
            if not np.allclose(expected, predictions[builder("moe", level)]):
                return False
    failure_weight = predictions["failure_hierarchical_weight"].to_numpy(dtype=float)
    failure_expected = (
        (1.0 - failure_weight)
        * predictions["base_failure_probability"].to_numpy(dtype=float)
        + failure_weight
        * predictions["hierarchical_failure_probability"].to_numpy(dtype=float)
    )
    if not np.allclose(failure_expected, predictions["moe_failure_probability"]):
        return False
    survival_weight = predictions["survival_hierarchical_weight"].to_numpy(dtype=float)
    for horizon in SURVIVAL_HORIZONS:
        expected = (
            (1.0 - survival_weight)
            * predictions[_reach_column("base", horizon)].to_numpy(dtype=float)
            + survival_weight
            * predictions[_reach_column("hierarchical", horizon)].to_numpy(dtype=float)
        )
        if not np.allclose(expected, predictions[_reach_column("moe", horizon)]):
            return False
    return True


def _validate_phase12(
    run_path: Path,
    paths: tuple[Path, Path, Path, Path, Path, Path],
    original_inputs: dict[str, str],
    source: dict[str, Any],
) -> dict[str, Any]:
    current = _load_inputs(*paths)
    features = source["features"]
    splits = source["splits"]
    labels = source["phase11_labels"]
    preprocessing = source["preprocessing"]
    feature_schema = source["feature_schema"]
    predictions = pd.read_parquet(
        run_path / "data/predictions/oof_mixture_predictions.parquet"
    )
    saved_labels = pd.read_parquet(run_path / "data/targets/mixture_labels.parquet")
    fold_metrics = pd.read_parquet(run_path / "data/metrics/fold_mixture_metrics.parquet")
    aggregate = pd.read_parquet(run_path / "data/metrics/aggregate_mixture_metrics.parquet")
    support = pd.read_parquet(run_path / "data/support/expert_gate_support.parquet")
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
    quality = predictions[
        [_quality_column("moe", level) for level in QUANTILE_LEVELS]
    ].to_numpy(dtype=float)
    runtime = predictions[
        [_runtime_column("moe", level) for level in QUANTILE_LEVELS]
    ].to_numpy(dtype=float)
    reach = predictions[
        [_reach_column("moe", horizon) for horizon in SURVIVAL_HORIZONS]
    ].to_numpy(dtype=float)
    weights = predictions[
        [f"{target}_hierarchical_weight" for target in TARGET_NAMES]
    ].to_numpy(dtype=float)
    required_metrics = [
        "moe_quality_nll",
        "moe_quality_crps",
        "moe_runtime_nll",
        "moe_runtime_crps",
        "moe_failure_brier",
        "moe_failure_log_loss",
        "moe_survival_survival_nll",
        "moe_survival_integrated_brier",
        "moe_survival_par10_mae",
        "quality_route_accuracy",
        "runtime_route_accuracy",
        "failure_route_accuracy",
        "survival_route_accuracy",
    ]
    metrics_finite = bool(
        np.isfinite(fold_metrics[required_metrics].to_numpy(dtype=float)).all()
        and np.isfinite(aggregate[required_metrics].to_numpy(dtype=float)).all()
    )
    expected_support_rows = len(jobs) * len(TARGET_NAMES) * len(EXPERT_SLOTS)
    slot_sets_exact = all(
        set(group["expert_slot"]) == {slot for slot, _domain in EXPERT_SLOTS}
        for _keys, group in support.groupby(["split_name", "fold", "target"])
    )
    unavailable_slots = {"matrix", "stream", "natural_process"}
    unavailable_explicit = bool(
        (
            support.loc[
                support["expert_slot"].isin(unavailable_slots), "availability_status"
            ]
            == "unavailable_no_executed_rows"
        ).all()
    )
    routed_values = set()
    for target in TARGET_NAMES:
        routed_values.update(predictions[f"{target}_router_expert"].astype(str).unique())
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
        "mixture_labels_frozen_and_exact": _frames_equal(saved_labels, labels),
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
        "meta_gate_training_is_cross_fitted": bool(
            (fold_metrics["meta_training_current_fold_count"] == 0).all()
            and (fold_metrics["validation_wrong_fold_count"] == 0).all()
        ),
        "six_expert_slots_exact": [item["name"] for item in registry["expert_slots"]]
        == [slot for slot, _domain in EXPERT_SLOTS],
        "gate_support_complete": len(support) == expected_support_rows,
        "gate_support_slot_sets_exact": slot_sets_exact,
        "unexecuted_experts_explicit": unavailable_explicit,
        "unavailable_experts_never_routed": routed_values.isdisjoint(unavailable_slots),
        "gate_weights_finite_bounded": bool(
            np.isfinite(weights).all()
            and np.logical_and(weights >= 0.0, weights <= 1.0).all()
        ),
        "mixture_identity_exact": _mixture_identity_exact(predictions),
        "quality_quantiles_finite_bounded": bool(
            np.isfinite(quality).all()
            and np.logical_and(quality >= 0.0, quality <= 1.0).all()
        ),
        "quality_quantiles_nondecreasing": bool((np.diff(quality, axis=1) >= 0.0).all()),
        "quality_pit_bounded": bool(predictions["moe_quality_pit"].between(0.0, 1.0).all()),
        "runtime_quantiles_finite_positive": bool(
            np.isfinite(runtime).all() and (runtime > 0.0).all()
        ),
        "runtime_quantiles_nondecreasing": bool((np.diff(runtime, axis=1) >= 0.0).all()),
        "runtime_pit_bounded": bool(predictions["moe_runtime_pit"].between(0.0, 1.0).all()),
        "failure_probability_finite_bounded": bool(
            predictions["moe_failure_probability"].between(0.0, 1.0).all()
        ),
        "reach_probabilities_finite_bounded": bool(
            np.isfinite(reach).all()
            and np.logical_and(reach >= 0.0, reach <= 1.0).all()
        ),
        "reach_probabilities_nondecreasing": bool((np.diff(reach, axis=1) >= 0.0).all()),
        "required_metrics_finite": metrics_finite,
        "aggregate_metrics_complete": len(aggregate) == expected_aggregate_rows,
        "fold_preprocessing_contract_complete": len(fold_schemas["folds"]) == len(jobs),
        "phase12_scope_exact": registry["targets"] == list(TARGET_NAMES),
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    issues = []
    if failed_checks:
        issues.append({"type": "failed_quality_checks", "checks": failed_checks})
    return {
        "status": "PHASE_12_PASS" if all(checks.values()) else "PHASE_12_FAIL",
        "schema_version": PHASE12_SCHEMA_VERSION,
        "scope": "leakage-safe six-slot cross-fitted Mixture-of-Experts routing",
        "performance_gate_policy": (
            "construction, cross-fitting, routing availability, mixture validity, OOF coverage, "
            "and integrity only; expert loss and predictive metric deltas are not Phase 12 pass thresholds"
        ),
        "phase13_boundary": "joint probability modelling and calibration are deferred to Phase 13",
        "phase6_directory": str(paths[0].resolve()),
        "phase7_directory": str(paths[1].resolve()),
        "phase8_directory": str(paths[2].resolve()),
        "phase9_directory": str(paths[3].resolve()),
        "phase10_directory": str(paths[4].resolve()),
        "phase11_directory": str(paths[5].resolve()),
        "feature_row_count": int(len(features)),
        "prediction_row_count": int(len(predictions)),
        "job_count": int(len(fold_metrics)),
        "model_artifact_count": int(len(registry["jobs"])),
        "aggregate_metric_row_count": int(len(aggregate)),
        "gate_support_row_count": int(len(support)),
        "expert_slot_count": len(EXPERT_SLOTS),
        "available_expert_slots": sorted(
            set(support.loc[support["availability_status"] == "available", "expert_slot"])
        ),
        "unavailable_expert_slots": sorted(unavailable_slots),
        "checks": checks,
        "issues": issues,
    }


def run_phase12(
    run_directory: str | Path,
    phase6_directory: str | Path,
    phase7_directory: str | Path,
    phase8_directory: str | Path,
    phase9_directory: str | Path,
    phase10_directory: str | Path,
    phase11_directory: str | Path,
    *,
    master_seed: int = 20_260_827,
    gate_iterations: int = DEFAULT_GATE_ITERATIONS,
    minimum_gate_rows: int = DEFAULT_MINIMUM_GATE_ROWS,
    weight_clip: tuple[float, float] = DEFAULT_GATE_WEIGHT_CLIP,
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
    )
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"
    if gate_iterations < 1:
        raise ValueError("Phase 12 gate iterations must be positive")
    if minimum_gate_rows < 1:
        raise ValueError("Phase 12 minimum gate rows must be positive")
    if (
        len(weight_clip) != 2
        or not 0.0 <= weight_clip[0] < weight_clip[1] <= 1.0
    ):
        raise ValueError("Phase 12 gate weight clip must be ordered inside [0, 1]")
    if len(beta_prior) != 2 or any(value <= 0.0 for value in beta_prior):
        raise ValueError("Phase 12 beta prior alpha and beta must be positive")
    source = _load_inputs(*paths)
    stable_configuration = {
        "phase": 12,
        "schema_version": PHASE12_SCHEMA_VERSION,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "preprocessing_implementation_sha256": _file_sha256(Path(phase7_module.__file__)),
        "input_fingerprints": source["input_fingerprints"],
        "master_seed": int(master_seed),
        "gate_model": GATE_MODEL_NAME,
        "gate_iterations": int(gate_iterations),
        "minimum_gate_rows": int(minimum_gate_rows),
        "weight_clip": [float(value) for value in weight_clip],
        "beta_prior": {"alpha": float(beta_prior[0]), "beta": float(beta_prior[1])},
        "expert_slots": [
            {"name": name, "domain": domain} for name, domain in EXPERT_SLOTS
        ],
        "targets": list(TARGET_NAMES),
        "split_columns": source["feature_schema"]["split_columns"],
        "cutoffs": source["feature_schema"]["cutoffs"],
        "gate_loss": {
            "quality": "row CRPS",
            "runtime": "row CRPS",
            "failure": "row Brier",
            "survival": "mean observable-horizon Brier",
        },
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
        "git_commit": current_git_commit(),
        "configuration_sha256": configuration_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Phase 12 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_12_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 12 resume manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest.get("output_sha256", {}).items():
                path = run_path / relative
                if not path.is_file() or _file_sha256(path) != expected_hash:
                    raise ValueError(f"completed Phase 12 output hash mismatch: {relative}")
            active_logger.info("[PHASE12][RESUME] status=PHASE_12_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info(
            "[PHASE12][RESUME] completed_jobs=%s", len(state.get("completed_jobs", {}))
        )
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE12_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": configuration_sha256,
            "input_fingerprints": source["input_fingerprints"],
            "completed_jobs": {},
            "last_completed_job": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase12_started"})
    fold_schemas, job_outputs = _run_jobs(
        run_path,
        source["features"],
        source["phase11_labels"],
        source["phase11_predictions"],
        source["splits"],
        source["feature_schema"],
        source["preprocessing"],
        state,
        master_seed=master_seed,
        gate_iterations=gate_iterations,
        minimum_gate_rows=minimum_gate_rows,
        weight_clip=weight_clip,
        beta_prior=beta_prior,
        progress_path=progress_path,
        logger=active_logger,
    )
    jobs = _all_jobs(source["preprocessing"])
    aggregate_outputs, _registry = _aggregate_job_outputs(
        run_path, jobs, source["phase11_labels"]
    )
    validation = _validate_phase12(
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
        "phase": 12,
        "status": validation["status"],
        "schema_version": PHASE12_SCHEMA_VERSION,
        "input_fingerprints": source["input_fingerprints"],
        "output_sha256": _relative_hashes(run_path, output_paths),
        "row_counts": {
            "features": validation["feature_row_count"],
            "predictions": validation["prediction_row_count"],
            "jobs": validation["job_count"],
            "aggregate_metrics": validation["aggregate_metric_row_count"],
            "gate_support": validation["gate_support_row_count"],
        },
        "phase13_allowed": validation["status"] == "PHASE_12_PASS",
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
        progress_path, {"event": "phase12_finished", "status": validation["status"]}
    )
    active_logger.info(
        "[PHASE12][SUMMARY] status=%s jobs=%s predictions=%s gate_support=%s directory=%s",
        validation["status"],
        validation["job_count"],
        validation["prediction_row_count"],
        validation["gate_support_row_count"],
        run_path,
    )
    return validation
