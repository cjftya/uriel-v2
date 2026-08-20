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

import uriel_v2.probabilistic_lab.phase13 as phase13_module
from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.phase7 import (
    _append_progress,
    _atomic_json,
    _atomic_parquet,
    _configuration_hash,
    _file_sha256,
    _relative_hashes,
    _verify_fold_contract,
)
from uriel_v2.probabilistic_lab.phase8 import QUANTILE_LEVELS, _cdf_at
from uriel_v2.probabilistic_lab.phase10 import _runtime_cdf
from uriel_v2.probabilistic_lab.phase12 import _all_jobs, _frames_equal, _ordered_predictions
from uriel_v2.probabilistic_lab.phase13 import (
    _bivariate_tail_probability,
    _load_inputs as _load_phase13_inputs,
)
from uriel_v2.provenance import current_git_commit


PHASE14_SCHEMA_VERSION = "phase14-v1"
DECISION_MODEL_NAME = "cross_fitted_expected_utility_algorithm_selection"
DEFAULT_MASTER_SEED = 20_260_829
DEFAULT_RUNTIME_RATIO_CAP = 10.0
DEFAULT_MINIMUM_RUNTIME_SCALE = 1e-6
DEFAULT_SOFTMAX_TEMPERATURE = 0.10
UTILITY_QUALITY_THRESHOLD = 0.75
PRIMARY_UTILITY_PROFILE = "balanced"
UTILITY_PROFILES: dict[str, dict[str, float]] = {
    "quality_first": {
        "quality_weight": 1.0,
        "runtime_weight": 0.05,
        "failure_penalty": 0.50,
        "uncertainty_penalty": 0.05,
        "sla_bonus": 0.10,
    },
    "balanced": {
        "quality_weight": 1.0,
        "runtime_weight": 0.15,
        "failure_penalty": 1.00,
        "uncertainty_penalty": 0.10,
        "sla_bonus": 0.25,
    },
    "speed_sensitive": {
        "quality_weight": 1.0,
        "runtime_weight": 0.35,
        "failure_penalty": 1.00,
        "uncertainty_penalty": 0.05,
        "sla_bonus": 0.15,
    },
    "risk_averse": {
        "quality_weight": 1.0,
        "runtime_weight": 0.15,
        "failure_penalty": 2.00,
        "uncertainty_penalty": 0.25,
        "sla_bonus": 0.35,
    },
}


def _quantile_suffix(level: float) -> str:
    return f"q{int(round(level * 100.0)):02d}"


def _phase13_required_paths(phase13_path: Path) -> dict[str, Path]:
    return {
        "phase13_config": phase13_path / "config.json",
        "phase13_manifest": phase13_path / "manifest.json",
        "phase13_validation": phase13_path / "validation.json",
        "phase13_predictions": phase13_path
        / "data/predictions/oof_joint_calibrated_predictions.parquet",
        "phase13_labels": phase13_path / "data/targets/joint_calibration_labels.parquet",
        "phase13_fold_metrics": phase13_path
        / "data/metrics/fold_joint_calibration_metrics.parquet",
        "phase13_aggregate_metrics": phase13_path
        / "data/metrics/aggregate_joint_calibration_metrics.parquet",
        "phase13_support": phase13_path
        / "data/calibration/calibration_copula_support.parquet",
        "phase13_fold_schemas": phase13_path
        / "data/preprocessing/fold_calibration_schemas.json",
        "phase13_model_registry": phase13_path / "model_registry.json",
    }


def _validate_phase13_input(
    phase13_path: Path,
    expected_input_fingerprints: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    required = _phase13_required_paths(phase13_path)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 14 Phase 13 input files are missing: {missing[:5]}")
    validation = json.loads(required["phase13_validation"].read_text(encoding="utf-8"))
    manifest = json.loads(required["phase13_manifest"].read_text(encoding="utf-8"))
    if validation.get("status") != "PHASE_13_PASS":
        raise ValueError("Phase 14 requires a PHASE_13_PASS joint probability run")
    if manifest.get("status") != "PHASE_13_PASS" or not manifest.get("phase14_allowed"):
        raise ValueError("Phase 13 manifest does not allow Phase 14")
    if validation.get("configuration", {}).get("input_fingerprints") != expected_input_fingerprints:
        raise ValueError("Phase 13 was not built from the supplied Phase 6 through 12 inputs")
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = phase13_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 13 manifest output hash mismatch: {relative}")
    fingerprints = {name: _file_sha256(path) for name, path in sorted(required.items())}
    return validation, manifest, fingerprints


def _combined_input_fingerprints(
    phase13_expected_inputs: dict[str, str],
    phase13_fingerprints: dict[str, str],
) -> dict[str, str]:
    return {
        **phase13_expected_inputs,
        **{f"phase13/{name}": value for name, value in phase13_fingerprints.items()},
    }


def _load_inputs(
    phase6_path: Path,
    phase7_path: Path,
    phase8_path: Path,
    phase9_path: Path,
    phase10_path: Path,
    phase11_path: Path,
    phase12_path: Path,
    phase13_path: Path,
) -> dict[str, Any]:
    source = _load_phase13_inputs(
        phase6_path,
        phase7_path,
        phase8_path,
        phase9_path,
        phase10_path,
        phase11_path,
        phase12_path,
    )
    expected = source["input_fingerprints"]
    validation, _manifest, fingerprints = _validate_phase13_input(phase13_path, expected)
    predictions = pd.read_parquet(
        phase13_path / "data/predictions/oof_joint_calibrated_predictions.parquet"
    )
    labels = pd.read_parquet(
        phase13_path / "data/targets/joint_calibration_labels.parquet"
    )
    expected_rows = len(source["features"]) * len(source["preprocessing"]["splits"])
    if len(predictions) != expected_rows or predictions.duplicated(
        ["feature_id", "split_name"]
    ).any():
        raise ValueError("Phase 14 requires exact unique Phase 13 OOF coverage")
    return {
        **source,
        "validations": {**source["validations"], "phase13": validation},
        "input_fingerprints": _combined_input_fingerprints(expected, fingerprints),
        "phase13_predictions": predictions,
        "phase13_labels": labels,
    }


def _attach_identity(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    identity = features[
        ["feature_id", "run_id", "problem_id", "algorithm", "algorithm_family"]
    ]
    merged = predictions.merge(
        identity,
        on="feature_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    if merged[["run_id", "problem_id", "algorithm"]].isna().any().any():
        raise ValueError("Phase 14 could not attach Phase 6 decision identities")
    if "algorithm_family_feature" in merged:
        if not (
            merged["algorithm_family"].astype(str)
            == merged["algorithm_family_feature"].astype(str)
        ).all():
            raise ValueError("Phase 14 algorithm-family identity mismatch")
        merged = merged.drop(columns=["algorithm_family_feature"])
    return merged


def _fit_runtime_scales(
    training: pd.DataFrame,
    *,
    minimum_runtime_scale: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    runtime = training["observed_runtime"].to_numpy(dtype=float)
    finite_positive = runtime[np.isfinite(runtime) & (runtime > 0.0)]
    if not len(finite_positive):
        raise ValueError("Phase 14 runtime normalization has no positive training runtime")
    global_scale = max(float(np.median(finite_positive)), minimum_runtime_scale)
    scales: dict[str, float] = {}
    rows = [
        {
            "support_type": "runtime_scale_global",
            "domain": None,
            "runtime_scale_seconds": global_scale,
            "training_row_count": int(len(finite_positive)),
            "fallback_used": False,
        }
    ]
    for domain in sorted(training["domain"].astype(str).unique()):
        values = training.loc[
            training["domain"].astype(str) == domain, "observed_runtime"
        ].to_numpy(dtype=float)
        valid = values[np.isfinite(values) & (values > 0.0)]
        fallback = not len(valid)
        scale = global_scale if fallback else max(float(np.median(valid)), minimum_runtime_scale)
        scales[domain] = scale
        rows.append(
            {
                "support_type": "runtime_scale_domain",
                "domain": domain,
                "runtime_scale_seconds": scale,
                "training_row_count": int(len(valid)),
                "fallback_used": fallback,
            }
        )
    return scales, pd.DataFrame(rows)


def _profile_feature_rows(
    frame: pd.DataFrame,
    runtime_scales: dict[str, float],
    *,
    master_seed: int,
    runtime_ratio_cap: float,
) -> pd.DataFrame:
    domains = frame["domain"].astype(str)
    missing_domains = sorted(set(domains) - set(runtime_scales))
    if missing_domains:
        raise ValueError(f"Phase 14 runtime scales missing domains: {missing_domains}")
    scales = domains.map(runtime_scales).to_numpy(dtype=float)
    quality_quantiles = frame[
        [f"quality_{_quantile_suffix(level)}" for level in QUANTILE_LEVELS]
    ].to_numpy(dtype=float)
    runtime_quantiles = frame[
        [f"runtime_{_quantile_suffix(level)}" for level in QUANTILE_LEVELS]
    ].to_numpy(dtype=float)
    quality_cdf = _cdf_at(
        quality_quantiles,
        np.full(len(frame), UTILITY_QUALITY_THRESHOLD, dtype=float),
    )
    runtime_cdf = _runtime_cdf(runtime_quantiles, scales)
    rho_values = frame["copula_quality_runtime_rho"].to_numpy(dtype=float)
    joint_sla = np.empty(len(frame), dtype=float)
    source_folds = frame["fold"].to_numpy(dtype=int)
    for source_fold in sorted(set(source_folds)):
        mask = source_folds == source_fold
        fold_rho = rho_values[mask]
        if not np.allclose(fold_rho, fold_rho[0]):
            raise ValueError(
                "Phase 14 source fold has inconsistent quality-runtime copula correlation"
            )
        joint_sla[mask] = _bivariate_tail_probability(
            quality_cdf[mask],
            runtime_cdf[mask],
            float(fold_rho[0]),
            seed=master_seed + int(source_fold),
        )
    joint_sla *= 1.0 - frame["failure_probability"].to_numpy(dtype=float)
    expected_runtime_ratio = np.clip(
        frame["runtime_predictive_mean"].to_numpy(dtype=float) / scales,
        0.0,
        runtime_ratio_cap,
    )
    realized_runtime_ratio = np.clip(
        frame["observed_runtime"].to_numpy(dtype=float) / scales,
        0.0,
        runtime_ratio_cap,
    )
    observed_failure = frame["observed_failure"].to_numpy(dtype=bool)
    observed_sla = (
        (frame["observed_quality"].to_numpy(dtype=float) >= UTILITY_QUALITY_THRESHOLD)
        & (frame["observed_runtime"].to_numpy(dtype=float) <= scales)
        & ~observed_failure
    ).astype(float)
    identity_columns = [
        "feature_id",
        "run_id",
        "problem_id",
        "problem_family",
        "domain",
        "algorithm",
        "algorithm_family",
        "cutoff",
        "split_name",
        "fold",
    ]
    rows = []
    for profile_name, weights in UTILITY_PROFILES.items():
        result = frame[identity_columns].reset_index(drop=True).copy()
        result["utility_profile"] = profile_name
        result["runtime_scale_seconds"] = scales
        result["predicted_quality"] = frame["quality_predictive_mean"].to_numpy(dtype=float)
        result["predicted_quality_uncertainty"] = frame[
            "quality_predictive_std"
        ].to_numpy(dtype=float)
        result["predicted_runtime_ratio"] = expected_runtime_ratio
        result["predicted_failure_probability"] = frame[
            "failure_probability"
        ].to_numpy(dtype=float)
        result["predicted_sla_probability"] = np.clip(joint_sla, 0.0, 1.0)
        result["realized_quality"] = frame["observed_quality"].to_numpy(dtype=float)
        result["realized_runtime_ratio"] = realized_runtime_ratio
        result["realized_failure"] = observed_failure.astype(float)
        result["realized_sla"] = observed_sla
        result["expected_utility"] = (
            weights["quality_weight"] * result["predicted_quality"]
            - weights["runtime_weight"] * result["predicted_runtime_ratio"]
            - weights["failure_penalty"] * result["predicted_failure_probability"]
            - weights["uncertainty_penalty"] * result["predicted_quality_uncertainty"]
            + weights["sla_bonus"] * result["predicted_sla_probability"]
        )
        result["realized_utility"] = (
            weights["quality_weight"] * result["realized_quality"]
            - weights["runtime_weight"] * result["realized_runtime_ratio"]
            - weights["failure_penalty"] * result["realized_failure"]
            + weights["sla_bonus"] * result["realized_sla"]
        )
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _aggregate_candidates(feature_rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "split_name",
        "fold",
        "problem_id",
        "problem_family",
        "domain",
        "cutoff",
        "utility_profile",
        "algorithm",
    ]
    grouped = feature_rows.groupby(group_columns, sort=True)
    if (grouped["algorithm_family"].nunique() != 1).any():
        raise ValueError("Phase 14 candidate has inconsistent algorithm-family identities")
    if (grouped["runtime_scale_seconds"].nunique() != 1).any():
        raise ValueError("Phase 14 candidate has inconsistent runtime normalization scales")
    candidates = (
        grouped
        .agg(
            algorithm_family=("algorithm_family", "first"),
            replicate_count=("feature_id", "size"),
            runtime_scale_seconds=("runtime_scale_seconds", "first"),
            predicted_quality=("predicted_quality", "mean"),
            predicted_quality_uncertainty=("predicted_quality_uncertainty", "mean"),
            predicted_runtime_ratio=("predicted_runtime_ratio", "mean"),
            predicted_failure_probability=("predicted_failure_probability", "mean"),
            predicted_sla_probability=("predicted_sla_probability", "mean"),
            realized_quality=("realized_quality", "mean"),
            realized_runtime_ratio=("realized_runtime_ratio", "mean"),
            realized_failure=("realized_failure", "mean"),
            realized_sla=("realized_sla", "mean"),
            expected_utility=("expected_utility", "mean"),
            realized_utility=("realized_utility", "mean"),
        )
        .reset_index()
    )
    return candidates


def _rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "split_name",
        "fold",
        "problem_id",
        "cutoff",
        "utility_profile",
    ]
    predicted = candidates.sort_values(
        [*group_columns, "expected_utility", "algorithm"],
        ascending=[True] * len(group_columns) + [False, True],
        kind="mergesort",
    ).copy()
    predicted["predicted_rank"] = predicted.groupby(group_columns, sort=False).cumcount() + 1
    realized = candidates.sort_values(
        [*group_columns, "realized_utility", "algorithm"],
        ascending=[True] * len(group_columns) + [False, True],
        kind="mergesort",
    ).copy()
    realized["realized_rank"] = realized.groupby(group_columns, sort=False).cumcount() + 1
    rank_map = realized.set_index([*group_columns, "algorithm"])["realized_rank"]
    keys = pd.MultiIndex.from_frame(predicted[[*group_columns, "algorithm"]])
    predicted["realized_rank"] = rank_map.loc[keys].to_numpy(dtype=int)
    predicted["candidate_count"] = predicted.groupby(group_columns, sort=False)[
        "algorithm"
    ].transform("size")
    return predicted.reset_index(drop=True)


def _training_global_best(training_candidates: pd.DataFrame) -> tuple[dict[tuple[str, float, str], str], pd.DataFrame]:
    scores = (
        training_candidates.groupby(
            ["domain", "cutoff", "utility_profile", "algorithm"],
            sort=True,
            as_index=False,
        )
        .agg(
            training_realized_utility=("realized_utility", "mean"),
            training_candidate_rows=("problem_id", "size"),
        )
        .sort_values(
            ["domain", "cutoff", "utility_profile", "training_realized_utility", "algorithm"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
    )
    best = scores.groupby(["domain", "cutoff", "utility_profile"], sort=False).head(1)
    lookup = {
        (str(row.domain), float(row.cutoff), str(row.utility_profile)): str(row.algorithm)
        for row in best.itertuples(index=False)
    }
    return lookup, best.reset_index(drop=True)


def _normalized_entropy(values: np.ndarray, temperature: float) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    shifted = (values - float(values.max())) / temperature
    probabilities = np.exp(np.clip(shifted, -700.0, 0.0))
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-15, 1.0))))
    return entropy / math.log(len(values))


def _build_selections(
    ranked: pd.DataFrame,
    global_best: dict[tuple[str, float, str], str],
    *,
    softmax_temperature: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "split_name",
        "fold",
        "problem_id",
        "problem_family",
        "domain",
        "cutoff",
        "utility_profile",
    ]
    candidate_rows = []
    selection_rows = []
    for keys, group in ranked.groupby(group_columns, sort=True):
        identifiers = dict(zip(group_columns, keys, strict=True))
        selected = group.loc[group["predicted_rank"] == 1].iloc[0]
        oracle = group.loc[group["realized_rank"] == 1].iloc[0]
        baseline_algorithm = global_best[
            (
                str(identifiers["domain"]),
                float(identifiers["cutoff"]),
                str(identifiers["utility_profile"]),
            )
        ]
        baseline_rows = group.loc[group["algorithm"].astype(str) == baseline_algorithm]
        if len(baseline_rows) != 1:
            raise ValueError("Phase 14 training-global baseline is unavailable in a candidate set")
        baseline = baseline_rows.iloc[0]
        oracle_utility = float(oracle["realized_utility"])
        selected_regret = max(0.0, oracle_utility - float(selected["realized_utility"]))
        baseline_regret = max(0.0, oracle_utility - float(baseline["realized_utility"]))
        random_utility = float(group["realized_utility"].mean())
        random_regret = max(0.0, oracle_utility - random_utility)
        margin = 0.0
        if len(group) > 1:
            expected = np.sort(group["expected_utility"].to_numpy(dtype=float))[::-1]
            margin = float(expected[0] - expected[1])
        enriched = group.copy()
        enriched["selected_by_policy"] = enriched["predicted_rank"] == 1
        enriched["oracle_best"] = enriched["realized_rank"] == 1
        enriched["training_global_best"] = (
            enriched["algorithm"].astype(str) == baseline_algorithm
        )
        candidate_rows.append(enriched)
        selection_rows.append(
            {
                **identifiers,
                "candidate_count": int(len(group)),
                "selected_algorithm": str(selected["algorithm"]),
                "oracle_algorithm": str(oracle["algorithm"]),
                "training_global_best_algorithm": baseline_algorithm,
                "selection_correct": str(selected["algorithm"]) == str(oracle["algorithm"]),
                "selected_in_realized_top2": int(selected["realized_rank"]) <= min(2, len(group)),
                "selected_expected_utility": float(selected["expected_utility"]),
                "selected_realized_utility": float(selected["realized_utility"]),
                "oracle_realized_utility": oracle_utility,
                "training_global_best_realized_utility": float(baseline["realized_utility"]),
                "random_expected_realized_utility": random_utility,
                "oracle_regret": selected_regret,
                "training_global_best_oracle_regret": baseline_regret,
                "random_oracle_regret": random_regret,
                "selected_vs_training_global_best_utility_gain": float(
                    selected["realized_utility"] - baseline["realized_utility"]
                ),
                "selected_vs_random_utility_gain": float(
                    selected["realized_utility"] - random_utility
                ),
                "selection_margin": margin,
                "selection_entropy": _normalized_entropy(
                    group["expected_utility"].to_numpy(dtype=float), softmax_temperature
                ),
            }
        )
    return (
        pd.concat(candidate_rows, ignore_index=True).sort_values(
            [*group_columns, "predicted_rank"]
        ).reset_index(drop=True),
        pd.DataFrame(selection_rows).sort_values(group_columns).reset_index(drop=True),
    )


def _selection_metrics(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = (
        ("overall", ("split_name", "utility_profile")),
        ("cutoff", ("split_name", "utility_profile", "cutoff")),
        ("domain", ("split_name", "utility_profile", "domain")),
        ("domain_cutoff", ("split_name", "utility_profile", "domain", "cutoff")),
    )
    for scope, columns in scopes:
        for keys, group in selections.groupby(list(columns), sort=True):
            values = keys if isinstance(keys, tuple) else (keys,)
            identifiers = dict(zip(columns, values, strict=True))
            rows.append(
                {
                    "scope": scope,
                    "split_name": str(identifiers["split_name"]),
                    "utility_profile": str(identifiers["utility_profile"]),
                    "domain": identifiers.get("domain"),
                    "cutoff": (
                        float(identifiers["cutoff"]) if "cutoff" in identifiers else None
                    ),
                    "selection_group_count": int(len(group)),
                    "selection_accuracy": float(group["selection_correct"].mean()),
                    "selected_top2_coverage": float(
                        group["selected_in_realized_top2"].mean()
                    ),
                    "zero_regret_rate": float((group["oracle_regret"] <= 1e-12).mean()),
                    "mean_oracle_regret": float(group["oracle_regret"].mean()),
                    "median_oracle_regret": float(group["oracle_regret"].median()),
                    "p90_oracle_regret": float(group["oracle_regret"].quantile(0.90)),
                    "mean_training_global_best_regret": float(
                        group["training_global_best_oracle_regret"].mean()
                    ),
                    "mean_random_regret": float(group["random_oracle_regret"].mean()),
                    "mean_selected_vs_training_global_best_utility_gain": float(
                        group["selected_vs_training_global_best_utility_gain"].mean()
                    ),
                    "mean_selected_vs_random_utility_gain": float(
                        group["selected_vs_random_utility_gain"].mean()
                    ),
                    "mean_selection_margin": float(group["selection_margin"].mean()),
                    "mean_selection_entropy": float(group["selection_entropy"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope", "split_name", "utility_profile", "domain", "cutoff"],
        na_position="first",
    ).reset_index(drop=True)


def _fold_metric_rows(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, group in selections.groupby("utility_profile", sort=True):
        rows.append(
            {
                "split_name": str(group["split_name"].iloc[0]),
                "fold": int(group["fold"].iloc[0]),
                "utility_profile": str(profile),
                "selection_group_count": int(len(group)),
                "selection_accuracy": float(group["selection_correct"].mean()),
                "mean_oracle_regret": float(group["oracle_regret"].mean()),
                "mean_training_global_best_regret": float(
                    group["training_global_best_oracle_regret"].mean()
                ),
                "mean_random_regret": float(group["random_oracle_regret"].mean()),
                "mean_selection_entropy": float(group["selection_entropy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _job_id(split_name: str, fold: int) -> str:
    return f"{split_name}__fold{fold}__utility_selection"


def _job_paths(run_path: Path, job_id: str) -> dict[str, Path]:
    return {
        "candidates": run_path / "checkpoints/candidates" / f"{job_id}.parquet",
        "selections": run_path / "checkpoints/selections" / f"{job_id}.parquet",
        "metrics": run_path / "checkpoints/metrics" / f"{job_id}.parquet",
        "support": run_path / "checkpoints/support" / f"{job_id}.parquet",
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
            raise ValueError(f"Phase 14 job output hash mismatch: job={job_id} path={relative}")
    return marker


def _run_jobs(
    run_path: Path,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    feature_schema: dict[str, Any],
    preprocessing: dict[str, Any],
    state: dict[str, Any],
    *,
    master_seed: int,
    runtime_ratio_cap: float,
    minimum_runtime_scale: float,
    softmax_temperature: float,
    progress_path: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[Path]]:
    schemas: dict[str, Any] = {
        "schema_version": PHASE14_SCHEMA_VERSION,
        "source_preprocessing_schema_version": preprocessing["schema_version"],
        "decision_unit": "problem_id, cutoff, algorithm; seed replicates averaged before ranking",
        "runtime_normalization": (
            "validation runtime divided by the matching domain median wall-clock runtime fitted "
            "only on non-validation OOF rows; never divided by evaluation budget"
        ),
        "folds": {},
    }
    output_paths: list[Path] = []
    for split_name in sorted(preprocessing["splits"]):
        fold_column = feature_schema["split_columns"][split_name]
        for fold in sorted(int(value) for value in preprocessing["splits"][split_name]):
            fold_key = f"{split_name}/fold{fold}"
            specification = preprocessing["splits"][split_name][str(fold)]
            training_mask, validation_mask, contract = _verify_fold_contract(
                features, splits, fold_column, fold, specification
            )
            training = _ordered_predictions(
                predictions, split_name, features.loc[training_mask, "feature_id"]
            )
            validation = _ordered_predictions(
                predictions, split_name, features.loc[validation_mask, "feature_id"]
            )
            training = _attach_identity(training, features.loc[training_mask])
            validation = _attach_identity(validation, features.loc[validation_mask])
            meta_training_current_fold_count = int((training["fold"] == fold).sum())
            validation_wrong_fold_count = int((validation["fold"] != fold).sum())
            if meta_training_current_fold_count or validation_wrong_fold_count:
                raise ValueError(f"Phase 14 cross-fitting contract failed: {fold_key}")
            schemas["folds"][fold_key] = {
                **contract,
                "fold_column": fold_column,
                "meta_training_current_fold_count": meta_training_current_fold_count,
                "validation_wrong_fold_count": validation_wrong_fold_count,
                "utility_source": "Phase 13 OOF predictions and labels from non-validation folds",
            }
            job_id = _job_id(split_name, fold)
            paths = _job_paths(run_path, job_id)
            marker = _verify_job(run_path, job_id)
            if marker is not None:
                state.setdefault("completed_jobs", {})[job_id] = _file_sha256(paths["marker"])
                output_paths.extend(paths.values())
                logger.info("[PHASE14][RESUME] job=%s status=SKIP_VERIFIED", job_id)
                continue
            logger.info(
                "[PHASE14][FOLD] split=%s fold=%s train_rows=%s validation_rows=%s",
                split_name,
                fold,
                len(training),
                len(validation),
            )
            scales, runtime_support = _fit_runtime_scales(
                training, minimum_runtime_scale=minimum_runtime_scale
            )
            seed_digest = int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16)
            job_seed = int((master_seed + seed_digest) % (2**31 - 1))
            training_rows = _profile_feature_rows(
                training,
                scales,
                master_seed=job_seed,
                runtime_ratio_cap=runtime_ratio_cap,
            )
            validation_rows = _profile_feature_rows(
                validation,
                scales,
                master_seed=job_seed + 1,
                runtime_ratio_cap=runtime_ratio_cap,
            )
            training_candidates = _aggregate_candidates(training_rows)
            validation_candidates = _rank_candidates(_aggregate_candidates(validation_rows))
            baseline_lookup, baseline_support = _training_global_best(training_candidates)
            candidates, selections = _build_selections(
                validation_candidates,
                baseline_lookup,
                softmax_temperature=softmax_temperature,
            )
            metrics = _fold_metric_rows(selections)
            runtime_support["split_name"] = split_name
            runtime_support["fold"] = fold
            runtime_support["cutoff"] = np.nan
            runtime_support["utility_profile"] = None
            runtime_support["algorithm"] = None
            runtime_support["support_value"] = runtime_support["runtime_scale_seconds"]
            baseline_support = baseline_support.rename(
                columns={
                    "training_realized_utility": "support_value",
                    "training_candidate_rows": "training_row_count",
                }
            )
            baseline_support["support_type"] = "training_global_best_algorithm"
            baseline_support["split_name"] = split_name
            baseline_support["fold"] = fold
            baseline_support["runtime_scale_seconds"] = np.nan
            baseline_support["fallback_used"] = False
            support_columns = [
                "support_type",
                "split_name",
                "fold",
                "domain",
                "cutoff",
                "utility_profile",
                "algorithm",
                "runtime_scale_seconds",
                "support_value",
                "training_row_count",
                "fallback_used",
            ]
            support = pd.concat(
                [runtime_support[support_columns], baseline_support[support_columns]],
                ignore_index=True,
            )
            _atomic_parquet(paths["candidates"], candidates)
            _atomic_parquet(paths["selections"], selections)
            _atomic_parquet(paths["metrics"], metrics)
            _atomic_parquet(paths["support"], support)
            data_paths = [paths[name] for name in ("candidates", "selections", "metrics", "support")]
            marker = {
                "schema_version": PHASE14_SCHEMA_VERSION,
                "job_id": job_id,
                "split_name": split_name,
                "fold": fold,
                "candidate_row_count": int(len(candidates)),
                "selection_row_count": int(len(selections)),
                "output_sha256": _relative_hashes(run_path, data_paths),
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
                    "candidate_rows": len(candidates),
                    "selection_rows": len(selections),
                },
            )
            output_paths.extend([*data_paths, paths["marker"]])
    schema_path = run_path / "data/preprocessing/fold_decision_schemas.json"
    _atomic_json(schema_path, schemas)
    output_paths.append(schema_path)
    return schemas, output_paths


def _aggregate_job_outputs(
    run_path: Path,
    jobs: list[tuple[str, int]],
    labels: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any]]:
    candidates = []
    selections = []
    fold_metrics = []
    support = []
    for split_name, fold in jobs:
        job_id = _job_id(split_name, fold)
        if _verify_job(run_path, job_id) is None:
            raise ValueError(f"Phase 14 job is incomplete: {job_id}")
        paths = _job_paths(run_path, job_id)
        candidates.append(pd.read_parquet(paths["candidates"]))
        selections.append(pd.read_parquet(paths["selections"]))
        fold_metrics.append(pd.read_parquet(paths["metrics"]))
        support.append(pd.read_parquet(paths["support"]))
    candidate_frame = pd.concat(candidates, ignore_index=True).sort_values(
        ["split_name", "problem_id", "cutoff", "utility_profile", "predicted_rank"]
    ).reset_index(drop=True)
    selection_frame = pd.concat(selections, ignore_index=True).sort_values(
        ["split_name", "problem_id", "cutoff", "utility_profile"]
    ).reset_index(drop=True)
    fold_metric_frame = pd.concat(fold_metrics, ignore_index=True).sort_values(
        ["split_name", "fold", "utility_profile"]
    ).reset_index(drop=True)
    support_frame = pd.concat(support, ignore_index=True).sort_values(
        ["support_type", "split_name", "fold", "domain", "cutoff", "utility_profile"],
        na_position="first",
    ).reset_index(drop=True)
    aggregate_metrics = _selection_metrics(selection_frame)
    registry = {
        "schema_version": PHASE14_SCHEMA_VERSION,
        "decision_model": DECISION_MODEL_NAME,
        "primary_utility_profile": PRIMARY_UTILITY_PROFILE,
        "utility_profiles": UTILITY_PROFILES,
        "utility_quality_threshold": UTILITY_QUALITY_THRESHOLD,
        "decision_unit": "problem_id, cutoff, algorithm after averaging seed replicates",
        "runtime_unit_policy": (
            "wall-clock seconds divided by a training-only domain median in wall-clock seconds; "
            "evaluation budget is never used as a runtime denominator"
        ),
        "selection_tie_break": "expected utility descending, then algorithm name ascending",
        "oracle_tie_break": "realized utility descending, then algorithm name ascending",
        "jobs": [
            {"job_id": _job_id(split_name, fold), "split_name": split_name, "fold": fold}
            for split_name, fold in jobs
        ],
    }
    candidate_path = run_path / "data/decisions/oof_algorithm_candidates.parquet"
    selection_path = run_path / "data/decisions/oof_algorithm_selections.parquet"
    label_path = run_path / "data/targets/decision_labels.parquet"
    fold_metric_path = run_path / "data/metrics/fold_selection_metrics.parquet"
    aggregate_path = run_path / "data/metrics/aggregate_selection_metrics.parquet"
    support_path = run_path / "data/support/decision_policy_support.parquet"
    registry_path = run_path / "policy_registry.json"
    _atomic_parquet(candidate_path, candidate_frame)
    _atomic_parquet(selection_path, selection_frame)
    _atomic_parquet(label_path, labels)
    _atomic_parquet(fold_metric_path, fold_metric_frame)
    _atomic_parquet(aggregate_path, aggregate_metrics)
    _atomic_parquet(support_path, support_frame)
    _atomic_json(registry_path, registry)
    return [
        candidate_path,
        selection_path,
        label_path,
        fold_metric_path,
        aggregate_path,
        support_path,
        registry_path,
    ], registry


def _rank_permutations_valid(candidates: pd.DataFrame) -> bool:
    group_columns = ["split_name", "problem_id", "cutoff", "utility_profile"]
    for _keys, group in candidates.groupby(group_columns, sort=False):
        expected = set(range(1, len(group) + 1))
        if set(group["predicted_rank"].astype(int)) != expected:
            return False
        if set(group["realized_rank"].astype(int)) != expected:
            return False
        if int(group["selected_by_policy"].sum()) != 1:
            return False
        if int(group["oracle_best"].sum()) != 1:
            return False
        if int(group["training_global_best"].sum()) != 1:
            return False
    return True


def _validate_phase14(
    run_path: Path,
    paths: tuple[Path, Path, Path, Path, Path, Path, Path, Path],
    original_inputs: dict[str, str],
    source: dict[str, Any],
) -> dict[str, Any]:
    current = _load_inputs(*paths)
    features = source["features"]
    splits = source["splits"]
    preprocessing = source["preprocessing"]
    feature_schema = source["feature_schema"]
    candidates = pd.read_parquet(
        run_path / "data/decisions/oof_algorithm_candidates.parquet"
    )
    selections = pd.read_parquet(
        run_path / "data/decisions/oof_algorithm_selections.parquet"
    )
    labels = pd.read_parquet(run_path / "data/targets/decision_labels.parquet")
    fold_metrics = pd.read_parquet(run_path / "data/metrics/fold_selection_metrics.parquet")
    aggregate = pd.read_parquet(
        run_path / "data/metrics/aggregate_selection_metrics.parquet"
    )
    support = pd.read_parquet(run_path / "data/support/decision_policy_support.parquet")
    registry = json.loads((run_path / "policy_registry.json").read_text(encoding="utf-8"))
    configuration = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    schemas = json.loads(
        (run_path / "data/preprocessing/fold_decision_schemas.json").read_text(
            encoding="utf-8"
        )
    )
    jobs = _all_jobs(preprocessing)
    profile_count = len(UTILITY_PROFILES)
    split_count = len(preprocessing["splits"])
    expected_candidates_per_split = len(
        features[["problem_id", "cutoff", "algorithm"]].drop_duplicates()
    ) * profile_count
    expected_selections_per_split = len(
        features[["problem_id", "cutoff"]].drop_duplicates()
    ) * profile_count
    candidate_coverage = all(
        len(group) == expected_candidates_per_split
        for _split, group in candidates.groupby("split_name")
    )
    selection_coverage = all(
        len(group) == expected_selections_per_split
        for _split, group in selections.groupby("split_name")
    )
    fold_assignments_exact = True
    problem_folds = splits[["problem_id", *feature_schema["split_columns"].values()]].drop_duplicates()
    for split_name, fold_column in feature_schema["split_columns"].items():
        mapping = pd.Series(
            problem_folds[fold_column].to_numpy(dtype=int),
            index=problem_folds["problem_id"].astype(str),
        )
        observed = selections.loc[selections["split_name"] == split_name]
        expected = observed["problem_id"].astype(str).map(mapping)
        if expected.isna().any() or not np.array_equal(
            expected.to_numpy(dtype=int), observed["fold"].to_numpy(dtype=int)
        ):
            fold_assignments_exact = False
    utility_columns = [
        "expected_utility",
        "realized_utility",
        "predicted_quality",
        "predicted_quality_uncertainty",
        "predicted_runtime_ratio",
        "predicted_failure_probability",
        "predicted_sla_probability",
    ]
    metric_columns = [
        "selection_accuracy",
        "mean_oracle_regret",
        "mean_training_global_best_regret",
        "mean_random_regret",
        "mean_selection_entropy",
    ]
    selection_metrics_finite = bool(
        np.isfinite(fold_metrics[metric_columns].to_numpy(dtype=float)).all()
        and np.isfinite(
            aggregate[
                [
                    "selection_accuracy",
                    "mean_oracle_regret",
                    "mean_training_global_best_regret",
                    "mean_random_regret",
                    "mean_selection_entropy",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    runtime_support = support.loc[
        support["support_type"].isin(["runtime_scale_global", "runtime_scale_domain"])
    ]
    baseline_support = support.loc[
        support["support_type"] == "training_global_best_algorithm"
    ]
    expected_aggregate_rows = split_count * profile_count * (
        1
        + len(feature_schema["cutoffs"])
        + features["domain"].nunique()
        + features["domain"].nunique() * len(feature_schema["cutoffs"])
    )
    selected = candidates.loc[candidates["selected_by_policy"]]
    selected_map = selected.set_index(
        ["split_name", "problem_id", "cutoff", "utility_profile"]
    )["algorithm"]
    selection_keys = pd.MultiIndex.from_frame(
        selections[["split_name", "problem_id", "cutoff", "utility_profile"]]
    )
    selections_match_candidates = np.array_equal(
        selected_map.loc[selection_keys].astype(str).to_numpy(),
        selections["selected_algorithm"].astype(str).to_numpy(),
    )
    computed_regret = np.maximum(
        0.0,
        selections["oracle_realized_utility"].to_numpy(dtype=float)
        - selections["selected_realized_utility"].to_numpy(dtype=float),
    )
    expected_replicates = features.groupby(
        ["problem_id", "cutoff", "algorithm"], sort=True
    ).size()
    candidate_replicate_keys = pd.MultiIndex.from_frame(
        candidates[["problem_id", "cutoff", "algorithm"]]
    )
    replicate_counts_exact = np.array_equal(
        expected_replicates.loc[candidate_replicate_keys].to_numpy(dtype=int),
        candidates["replicate_count"].to_numpy(dtype=int),
    )
    runtime_ratio_cap = float(configuration["runtime_ratio_cap"])
    checks = {
        **{
            f"{phase}_quality_pass": validation["status"] == f"PHASE_{phase[5:]}_PASS"
            for phase, validation in current["validations"].items()
        },
        "source_inputs_unchanged": current["input_fingerprints"] == original_inputs,
        "decision_labels_frozen_and_exact": _frames_equal(labels, source["phase13_labels"]),
        "expected_job_count": len(registry["jobs"]) == len(jobs),
        "all_job_markers_verified": all(
            _verify_job(run_path, _job_id(*job)) is not None for job in jobs
        ),
        "candidate_keys_unique": not candidates.duplicated(
            ["split_name", "problem_id", "cutoff", "utility_profile", "algorithm"]
        ).any(),
        "selection_keys_unique": not selections.duplicated(
            ["split_name", "problem_id", "cutoff", "utility_profile"]
        ).any(),
        "candidate_oof_coverage_exact": candidate_coverage
        and len(candidates) == expected_candidates_per_split * split_count,
        "selection_oof_coverage_exact": selection_coverage
        and len(selections) == expected_selections_per_split * split_count,
        "fold_assignments_exact": fold_assignments_exact,
        "decision_training_is_cross_fitted": all(
            value["meta_training_current_fold_count"] == 0
            and value["validation_wrong_fold_count"] == 0
            for value in schemas["folds"].values()
        ),
        "utility_profiles_exact": set(candidates["utility_profile"]) == set(UTILITY_PROFILES)
        and set(selections["utility_profile"]) == set(UTILITY_PROFILES)
        and registry["utility_profiles"] == UTILITY_PROFILES,
        "primary_profile_registered": registry["primary_utility_profile"]
        == PRIMARY_UTILITY_PROFILE,
        "candidate_utilities_finite": bool(
            np.isfinite(candidates[utility_columns].to_numpy(dtype=float)).all()
        ),
        "probability_components_bounded": bool(
            candidates[
                ["predicted_failure_probability", "predicted_sla_probability"]
            ].apply(lambda column: column.between(0.0, 1.0).all()).all()
        ),
        "runtime_ratios_bounded": bool(
            candidates["predicted_runtime_ratio"].between(0.0, runtime_ratio_cap).all()
            and candidates["realized_runtime_ratio"].between(0.0, runtime_ratio_cap).all()
        ),
        "seed_replicate_aggregation_exact": replicate_counts_exact,
        "multiple_algorithms_per_decision": bool((candidates["candidate_count"] >= 2).all()),
        "rank_permutations_and_flags_valid": _rank_permutations_valid(candidates),
        "selection_rows_match_candidates": selections_match_candidates,
        "oracle_regret_nonnegative_exact": bool(
            (selections["oracle_regret"] >= 0.0).all()
            and np.allclose(selections["oracle_regret"], computed_regret)
        ),
        "selection_entropy_bounded": bool(selections["selection_entropy"].between(0.0, 1.0).all()),
        "runtime_scales_training_only_finite_positive": bool(
            len(runtime_support)
            and np.isfinite(runtime_support["runtime_scale_seconds"].to_numpy(dtype=float)).all()
            and (runtime_support["runtime_scale_seconds"] > 0.0).all()
            and (runtime_support["training_row_count"] > 0).all()
        ),
        "training_global_baselines_complete": bool(
            len(baseline_support)
            and baseline_support["algorithm"].notna().all()
            and (baseline_support["training_row_count"] > 0).all()
        ),
        "runtime_units_not_mixed_with_budget": "evaluation budget is never used"
        in registry["runtime_unit_policy"],
        "selection_metrics_finite": selection_metrics_finite,
        "aggregate_metrics_complete": len(aggregate) == expected_aggregate_rows,
        "fold_decision_contract_complete": len(schemas["folds"]) == len(jobs),
        "phase14_scope_exact": registry["decision_model"] == DECISION_MODEL_NAME,
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    issues = []
    if failed_checks:
        issues.append({"type": "failed_quality_checks", "checks": failed_checks})
    primary = aggregate.loc[
        (aggregate["scope"] == "overall")
        & (aggregate["utility_profile"] == PRIMARY_UTILITY_PROFILE)
    ]
    primary_metrics = {
        str(row.split_name): {
            "selection_accuracy": float(row.selection_accuracy),
            "mean_oracle_regret": float(row.mean_oracle_regret),
            "mean_training_global_best_regret": float(row.mean_training_global_best_regret),
            "mean_random_regret": float(row.mean_random_regret),
        }
        for row in primary.itertuples(index=False)
    }
    return {
        "status": "PHASE_14_PASS" if all(checks.values()) else "PHASE_14_FAIL",
        "schema_version": PHASE14_SCHEMA_VERSION,
        "scope": "cross-fitted expected utility and algorithm selection",
        "performance_gate_policy": (
            "construction, leakage safety, utility coherence, deterministic ranking, OOF coverage, "
            "and artifact integrity only; performance verdict is deferred to Phase 16"
        ),
        "phase15_boundary": (
            "selection robustness, utility sensitivity, and held-out generalization are deferred "
            "to Phase 15"
        ),
        "feature_row_count": int(len(features)),
        "candidate_row_count": int(len(candidates)),
        "selection_row_count": int(len(selections)),
        "job_count": int(len(jobs)),
        "fold_metric_row_count": int(len(fold_metrics)),
        "aggregate_metric_row_count": int(len(aggregate)),
        "support_row_count": int(len(support)),
        "utility_profile_count": profile_count,
        "primary_utility_profile": PRIMARY_UTILITY_PROFILE,
        "primary_metrics": primary_metrics,
        "checks": checks,
        "issues": issues,
    }


def run_phase14(
    run_directory: str | Path,
    phase6_directory: str | Path,
    phase7_directory: str | Path,
    phase8_directory: str | Path,
    phase9_directory: str | Path,
    phase10_directory: str | Path,
    phase11_directory: str | Path,
    phase12_directory: str | Path,
    phase13_directory: str | Path,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
    runtime_ratio_cap: float = DEFAULT_RUNTIME_RATIO_CAP,
    minimum_runtime_scale: float = DEFAULT_MINIMUM_RUNTIME_SCALE,
    softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    paths = tuple(
        Path(path)
        for path in (
            phase6_directory,
            phase7_directory,
            phase8_directory,
            phase9_directory,
            phase10_directory,
            phase11_directory,
            phase12_directory,
            phase13_directory,
        )
    )
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"
    if runtime_ratio_cap <= 0.0:
        raise ValueError("Phase 14 runtime ratio cap must be positive")
    if minimum_runtime_scale <= 0.0:
        raise ValueError("Phase 14 minimum runtime scale must be positive")
    if softmax_temperature <= 0.0:
        raise ValueError("Phase 14 softmax temperature must be positive")
    source = _load_inputs(*paths)
    stable_configuration = {
        "phase": 14,
        "schema_version": PHASE14_SCHEMA_VERSION,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "phase13_implementation_sha256": _file_sha256(Path(phase13_module.__file__)),
        "input_fingerprints": source["input_fingerprints"],
        "master_seed": int(master_seed),
        "decision_model": DECISION_MODEL_NAME,
        "utility_profiles": UTILITY_PROFILES,
        "primary_utility_profile": PRIMARY_UTILITY_PROFILE,
        "utility_quality_threshold": UTILITY_QUALITY_THRESHOLD,
        "runtime_ratio_cap": float(runtime_ratio_cap),
        "minimum_runtime_scale": float(minimum_runtime_scale),
        "softmax_temperature": float(softmax_temperature),
        "split_columns": source["feature_schema"]["split_columns"],
        "cutoffs": source["feature_schema"]["cutoffs"],
    }
    configuration_sha256 = _configuration_hash(stable_configuration)
    configuration = {
        **stable_configuration,
        **{f"phase{index + 6}_directory": str(path.resolve()) for index, path in enumerate(paths)},
        "git_commit": current_git_commit(),
        "configuration_sha256": configuration_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Phase 14 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_14_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 14 resume manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest.get("output_sha256", {}).items():
                path = run_path / relative
                if not path.is_file() or _file_sha256(path) != expected_hash:
                    raise ValueError(f"completed Phase 14 output hash mismatch: {relative}")
            active_logger.info("[PHASE14][RESUME] status=PHASE_14_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info(
            "[PHASE14][RESUME] completed_jobs=%s", len(state.get("completed_jobs", {}))
        )
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE14_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": configuration_sha256,
            "input_fingerprints": source["input_fingerprints"],
            "completed_jobs": {},
            "last_completed_job": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase14_started"})
    schemas, job_outputs = _run_jobs(
        run_path,
        source["features"],
        source["phase13_predictions"],
        source["splits"],
        source["feature_schema"],
        source["preprocessing"],
        state,
        master_seed=master_seed,
        runtime_ratio_cap=runtime_ratio_cap,
        minimum_runtime_scale=minimum_runtime_scale,
        softmax_temperature=softmax_temperature,
        progress_path=progress_path,
        logger=active_logger,
    )
    jobs = _all_jobs(source["preprocessing"])
    aggregate_outputs, _registry = _aggregate_job_outputs(
        run_path, jobs, source["phase13_labels"]
    )
    validation = _validate_phase14(run_path, paths, source["input_fingerprints"], source)
    validation["executed_at"] = datetime.now(_timezone()).isoformat(timespec="seconds")
    validation["elapsed_seconds"] = perf_counter() - started_at
    validation["configuration"] = configuration
    validation_path = run_path / "validation.json"
    _atomic_json(validation_path, validation)
    output_paths = sorted(
        {
            path
            for path in [run_path / "config.json", *job_outputs, *aggregate_outputs, validation_path]
        },
        key=lambda path: str(path),
    )
    manifest = {
        "phase": 14,
        "status": validation["status"],
        "schema_version": PHASE14_SCHEMA_VERSION,
        "input_fingerprints": source["input_fingerprints"],
        "output_sha256": _relative_hashes(run_path, output_paths),
        "row_counts": {
            "features": validation["feature_row_count"],
            "candidates": validation["candidate_row_count"],
            "selections": validation["selection_row_count"],
            "jobs": validation["job_count"],
            "aggregate_metrics": validation["aggregate_metric_row_count"],
            "support": validation["support_row_count"],
        },
        "phase15_allowed": validation["status"] == "PHASE_14_PASS",
        "performance_gate_policy": validation["performance_gate_policy"],
    }
    manifest_path = run_path / "manifest.json"
    _atomic_json(manifest_path, manifest)
    state["status"] = validation["status"]
    state["last_completed_stage"] = "validation"
    state["manifest_sha256"] = _file_sha256(manifest_path)
    state["validation_sha256"] = _file_sha256(validation_path)
    state["fold_schema_count"] = len(schemas["folds"])
    _atomic_json(state_path, state)
    _append_progress(
        progress_path, {"event": "phase14_finished", "status": validation["status"]}
    )
    active_logger.info(
        "[PHASE14][SUMMARY] status=%s candidates=%s selections=%s directory=%s",
        validation["status"],
        validation["candidate_row_count"],
        validation["selection_row_count"],
        run_path,
    )
    return validation
