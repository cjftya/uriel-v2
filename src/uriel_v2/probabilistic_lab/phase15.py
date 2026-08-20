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

import uriel_v2.probabilistic_lab.phase14 as phase14_module
from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.phase7 import (
    _append_progress,
    _atomic_json,
    _atomic_parquet,
    _configuration_hash,
    _file_sha256,
    _relative_hashes,
)
from uriel_v2.probabilistic_lab.phase14 import PRIMARY_UTILITY_PROFILE, UTILITY_PROFILES
from uriel_v2.provenance import current_git_commit


PHASE15_SCHEMA_VERSION = "phase15-v1"
ROBUSTNESS_MODEL_NAME = "frozen_policy_selection_robustness"
DEFAULT_MASTER_SEED = 20_260_830
DEFAULT_PERTURBATION_FRACTION = 0.25
DEFAULT_BOOTSTRAP_ITERATIONS = 2_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
REQUIRED_HELDOUT_SPLITS = {"instance_holdout", "family_holdout"}
SENSITIVITY_COMPONENTS = (
    "quality_weight",
    "runtime_weight",
    "failure_penalty",
    "uncertainty_penalty",
    "sla_bonus",
)


def _phase14_required_paths(phase14_path: Path) -> dict[str, Path]:
    return {
        "phase14_config": phase14_path / "config.json",
        "phase14_manifest": phase14_path / "manifest.json",
        "phase14_validation": phase14_path / "validation.json",
        "phase14_candidates": phase14_path
        / "data/decisions/oof_algorithm_candidates.parquet",
        "phase14_selections": phase14_path
        / "data/decisions/oof_algorithm_selections.parquet",
        "phase14_aggregate_metrics": phase14_path
        / "data/metrics/aggregate_selection_metrics.parquet",
        "phase14_policy_registry": phase14_path / "policy_registry.json",
    }


def _load_inputs(phase14_path: Path) -> dict[str, Any]:
    required = _phase14_required_paths(phase14_path)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 15 Phase 14 input files are missing: {missing[:5]}")
    validation = json.loads(required["phase14_validation"].read_text(encoding="utf-8"))
    manifest = json.loads(required["phase14_manifest"].read_text(encoding="utf-8"))
    registry = json.loads(required["phase14_policy_registry"].read_text(encoding="utf-8"))
    if validation.get("status") != "PHASE_14_PASS":
        raise ValueError("Phase 15 requires a PHASE_14_PASS expected-utility run")
    if manifest.get("status") != "PHASE_14_PASS" or not manifest.get("phase15_allowed"):
        raise ValueError("Phase 14 manifest does not allow Phase 15")
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = phase14_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 14 manifest output hash mismatch: {relative}")
    if registry.get("utility_profiles") != UTILITY_PROFILES:
        raise ValueError("Phase 15 requires the frozen Phase 14 utility profiles")
    if registry.get("primary_utility_profile") != PRIMARY_UTILITY_PROFILE:
        raise ValueError("Phase 15 requires the frozen Phase 14 primary utility profile")
    candidates = pd.read_parquet(required["phase14_candidates"])
    selections = pd.read_parquet(required["phase14_selections"])
    if candidates.empty or selections.empty:
        raise ValueError("Phase 15 requires non-empty Phase 14 OOF decisions")
    if set(selections["split_name"].astype(str)) != REQUIRED_HELDOUT_SPLITS:
        raise ValueError("Phase 15 requires exact instance_holdout and family_holdout splits")
    fingerprints = {
        f"phase14/{name}": _file_sha256(path) for name, path in sorted(required.items())
    }
    return {
        "validation": validation,
        "manifest": manifest,
        "registry": registry,
        "candidates": candidates,
        "selections": selections,
        "input_fingerprints": fingerprints,
    }


def _build_sensitivity_scenarios(
    perturbation_fraction: float,
) -> list[dict[str, Any]]:
    if not 0.0 < perturbation_fraction < 1.0:
        raise ValueError("Phase 15 perturbation fraction must be between zero and one")
    scenarios: list[dict[str, Any]] = []
    for profile_name in sorted(UTILITY_PROFILES):
        nominal = {name: float(value) for name, value in UTILITY_PROFILES[profile_name].items()}
        scenarios.append(
            {
                "scenario_id": f"{profile_name}__nominal",
                "base_profile": profile_name,
                "perturbed_component": "nominal",
                "multiplier": 1.0,
                "is_nominal": True,
                **nominal,
            }
        )
        for component in SENSITIVITY_COMPONENTS:
            for direction, multiplier in (
                ("minus", 1.0 - perturbation_fraction),
                ("plus", 1.0 + perturbation_fraction),
            ):
                weights = dict(nominal)
                weights[component] *= multiplier
                scenarios.append(
                    {
                        "scenario_id": (
                            f"{profile_name}__{component}__{direction}"
                            f"{int(round(perturbation_fraction * 100.0)):02d}"
                        ),
                        "base_profile": profile_name,
                        "perturbed_component": component,
                        "multiplier": float(multiplier),
                        "is_nominal": False,
                        **weights,
                    }
                )
    return scenarios


def _expected_utility(frame: pd.DataFrame, scenario: dict[str, Any]) -> np.ndarray:
    return (
        scenario["quality_weight"] * frame["predicted_quality"].to_numpy(dtype=float)
        - scenario["runtime_weight"]
        * frame["predicted_runtime_ratio"].to_numpy(dtype=float)
        - scenario["failure_penalty"]
        * frame["predicted_failure_probability"].to_numpy(dtype=float)
        - scenario["uncertainty_penalty"]
        * frame["predicted_quality_uncertainty"].to_numpy(dtype=float)
        + scenario["sla_bonus"]
        * frame["predicted_sla_probability"].to_numpy(dtype=float)
    )


def _realized_utility(frame: pd.DataFrame, scenario: dict[str, Any]) -> np.ndarray:
    return (
        scenario["quality_weight"] * frame["realized_quality"].to_numpy(dtype=float)
        - scenario["runtime_weight"]
        * frame["realized_runtime_ratio"].to_numpy(dtype=float)
        - scenario["failure_penalty"]
        * frame["realized_failure"].to_numpy(dtype=float)
        + scenario["sla_bonus"] * frame["realized_sla"].to_numpy(dtype=float)
    )


def _scenario_selection_rows(
    candidates: pd.DataFrame,
    scenario: dict[str, Any],
) -> pd.DataFrame:
    group_columns = ["split_name", "problem_id", "cutoff"]
    frame = candidates.loc[
        candidates["utility_profile"].astype(str) == scenario["base_profile"]
    ].copy()
    frame["scenario_expected_utility"] = _expected_utility(frame, scenario)
    frame["scenario_realized_utility"] = _realized_utility(frame, scenario)
    predicted = frame.sort_values(
        [*group_columns, "scenario_expected_utility", "algorithm"],
        ascending=[True] * len(group_columns) + [False, True],
        kind="mergesort",
    ).copy()
    predicted["scenario_predicted_rank"] = (
        predicted.groupby(group_columns, sort=False).cumcount() + 1
    )
    realized = frame.sort_values(
        [*group_columns, "scenario_realized_utility", "algorithm"],
        ascending=[True] * len(group_columns) + [False, True],
        kind="mergesort",
    ).copy()
    realized["scenario_realized_rank"] = (
        realized.groupby(group_columns, sort=False).cumcount() + 1
    )
    rank_map = realized.set_index([*group_columns, "algorithm"])[
        "scenario_realized_rank"
    ]
    predicted_keys = pd.MultiIndex.from_frame(predicted[[*group_columns, "algorithm"]])
    predicted["scenario_realized_rank"] = rank_map.loc[predicted_keys].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for keys, group in predicted.groupby(group_columns, sort=True):
        identifiers = dict(zip(group_columns, keys, strict=True))
        selected = group.loc[group["scenario_predicted_rank"] == 1].iloc[0]
        oracle = group.loc[group["scenario_realized_rank"] == 1].iloc[0]
        baseline_rows = group.loc[group["training_global_best"].astype(bool)]
        if len(baseline_rows) != 1:
            raise ValueError("Phase 15 nominal training-global baseline is not unique")
        baseline = baseline_rows.iloc[0]
        expected_values = np.sort(group["scenario_expected_utility"].to_numpy(dtype=float))[::-1]
        margin = float(expected_values[0] - expected_values[1]) if len(group) > 1 else 0.0
        oracle_utility = float(oracle["scenario_realized_utility"])
        selected_utility = float(selected["scenario_realized_utility"])
        random_utility = float(group["scenario_realized_utility"].mean())
        rows.append(
            {
                **identifiers,
                "fold": int(selected["fold"]),
                "problem_family": str(selected["problem_family"]),
                "domain": str(selected["domain"]),
                "base_profile": str(scenario["base_profile"]),
                "scenario_id": str(scenario["scenario_id"]),
                "perturbed_component": str(scenario["perturbed_component"]),
                "multiplier": float(scenario["multiplier"]),
                "is_nominal": bool(scenario["is_nominal"]),
                "candidate_count": int(len(group)),
                "selected_algorithm": str(selected["algorithm"]),
                "oracle_algorithm": str(oracle["algorithm"]),
                "nominal_training_global_best_algorithm": str(baseline["algorithm"]),
                "selection_correct": str(selected["algorithm"]) == str(oracle["algorithm"]),
                "selected_expected_utility": float(selected["scenario_expected_utility"]),
                "selected_realized_utility": selected_utility,
                "oracle_realized_utility": oracle_utility,
                "nominal_training_global_best_realized_utility": float(
                    baseline["scenario_realized_utility"]
                ),
                "random_expected_realized_utility": random_utility,
                "oracle_regret": max(0.0, oracle_utility - selected_utility),
                "selected_vs_nominal_training_global_best_utility_gain": float(
                    selected_utility - baseline["scenario_realized_utility"]
                ),
                "selected_vs_random_utility_gain": selected_utility - random_utility,
                "selection_margin": margin,
            }
        )
    return pd.DataFrame(rows)


def _build_sensitivity_outputs(
    candidates: pd.DataFrame,
    scenarios: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections = pd.concat(
        [_scenario_selection_rows(candidates, scenario) for scenario in scenarios],
        ignore_index=True,
    ).sort_values(
        ["split_name", "problem_id", "cutoff", "base_profile", "scenario_id"]
    ).reset_index(drop=True)
    key_columns = ["split_name", "problem_id", "cutoff", "base_profile"]
    nominal = selections.loc[selections["is_nominal"]]
    nominal_lookup = nominal.set_index(key_columns)["selected_algorithm"]
    selection_keys = pd.MultiIndex.from_frame(selections[key_columns])
    selections["nominal_selected_algorithm"] = nominal_lookup.loc[selection_keys].to_numpy()
    selections["selection_stable_from_nominal"] = (
        selections["selected_algorithm"].astype(str)
        == selections["nominal_selected_algorithm"].astype(str)
    )

    metric_rows = []
    metric_groups = [
        "split_name",
        "base_profile",
        "scenario_id",
        "perturbed_component",
        "multiplier",
        "is_nominal",
    ]
    for keys, group in selections.groupby(metric_groups, sort=True):
        identifiers = dict(zip(metric_groups, keys, strict=True))
        metric_rows.append(
            {
                **identifiers,
                "selection_group_count": int(len(group)),
                "selection_accuracy": float(group["selection_correct"].mean()),
                "zero_regret_rate": float((group["oracle_regret"] <= 1e-12).mean()),
                "mean_oracle_regret": float(group["oracle_regret"].mean()),
                "p90_oracle_regret": float(group["oracle_regret"].quantile(0.90)),
                "mean_selected_vs_nominal_training_global_best_utility_gain": float(
                    group["selected_vs_nominal_training_global_best_utility_gain"].mean()
                ),
                "mean_selected_vs_random_utility_gain": float(
                    group["selected_vs_random_utility_gain"].mean()
                ),
                "mean_selection_margin": float(group["selection_margin"].mean()),
                "nominal_selection_retention_rate": float(
                    group["selection_stable_from_nominal"].mean()
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(metric_groups).reset_index(drop=True)

    stability_rows = []
    stability_groups = [
        "split_name",
        "fold",
        "problem_id",
        "problem_family",
        "domain",
        "cutoff",
        "base_profile",
    ]
    for keys, group in selections.groupby(stability_groups, sort=True):
        identifiers = dict(zip(stability_groups, keys, strict=True))
        nominal_row = group.loc[group["is_nominal"]]
        if len(nominal_row) != 1:
            raise ValueError("Phase 15 sensitivity group does not have one nominal scenario")
        nominal_regret = float(nominal_row.iloc[0]["oracle_regret"])
        stability_rows.append(
            {
                **identifiers,
                "scenario_count": int(len(group)),
                "nominal_selected_algorithm": str(
                    nominal_row.iloc[0]["selected_algorithm"]
                ),
                "selected_algorithm_count": int(group["selected_algorithm"].nunique()),
                "stable_scenario_fraction": float(
                    group["selection_stable_from_nominal"].mean()
                ),
                "all_scenarios_agree": bool(group["selected_algorithm"].nunique() == 1),
                "nominal_oracle_regret": nominal_regret,
                "mean_oracle_regret": float(group["oracle_regret"].mean()),
                "worst_case_oracle_regret": float(group["oracle_regret"].max()),
                "maximum_regret_increase_from_nominal": max(
                    0.0, float(group["oracle_regret"].max()) - nominal_regret
                ),
                "minimum_selection_margin": float(group["selection_margin"].min()),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(stability_groups).reset_index(drop=True)
    return selections, metrics, stability


def _build_profile_stability(sensitivity: pd.DataFrame) -> pd.DataFrame:
    nominal = sensitivity.loc[sensitivity["is_nominal"]]
    group_columns = [
        "split_name",
        "fold",
        "problem_id",
        "problem_family",
        "domain",
        "cutoff",
    ]
    rows = []
    for keys, group in nominal.groupby(group_columns, sort=True):
        identifiers = dict(zip(group_columns, keys, strict=True))
        counts = (
            group.groupby("selected_algorithm", as_index=False)
            .size()
            .sort_values(["size", "selected_algorithm"], ascending=[False, True])
        )
        rows.append(
            {
                **identifiers,
                "utility_profile_count": int(len(group)),
                "selected_algorithm_count": int(group["selected_algorithm"].nunique()),
                "modal_selected_algorithm": str(counts.iloc[0]["selected_algorithm"]),
                "modal_profile_share": float(counts.iloc[0]["size"] / len(group)),
                "all_profiles_agree": bool(group["selected_algorithm"].nunique() == 1),
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _build_cross_split_agreement(sensitivity: pd.DataFrame) -> pd.DataFrame:
    nominal = sensitivity.loc[sensitivity["is_nominal"]]
    group_columns = ["problem_id", "problem_family", "domain", "cutoff", "base_profile"]
    rows = []
    for keys, group in nominal.groupby(group_columns, sort=True):
        identifiers = dict(zip(group_columns, keys, strict=True))
        signature = {
            str(row.split_name): str(row.selected_algorithm)
            for row in group.sort_values("split_name").itertuples(index=False)
        }
        rows.append(
            {
                **identifiers,
                "split_count": int(len(group)),
                "selected_algorithm_count": int(group["selected_algorithm"].nunique()),
                "all_splits_agree": bool(group["selected_algorithm"].nunique() == 1),
                "selection_signature": json.dumps(signature, sort_keys=True),
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _bootstrap_metric_summary(
    frame: pd.DataFrame,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    metric_columns = [
        "selection_correct",
        "oracle_regret",
        "selected_vs_training_global_best_utility_gain",
        "selected_vs_random_utility_gain",
        "selection_entropy",
    ]
    clusters = (
        frame.assign(selection_correct=frame["selection_correct"].astype(float))
        .groupby("problem_id", sort=True)[metric_columns]
        .mean()
    )
    if clusters.empty:
        raise ValueError("Phase 15 held-out bootstrap has no problem clusters")
    values = clusters.to_numpy(dtype=float)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(values), size=(iterations, len(values)))
    bootstrap = values[sampled].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower = np.quantile(bootstrap, alpha, axis=0)
    upper = np.quantile(bootstrap, 1.0 - alpha, axis=0)
    result: dict[str, float] = {}
    output_names = [
        "selection_accuracy",
        "mean_oracle_regret",
        "mean_selected_vs_training_global_best_utility_gain",
        "mean_selected_vs_random_utility_gain",
        "mean_selection_entropy",
    ]
    points = values.mean(axis=0)
    for index, name in enumerate(output_names):
        result[name] = float(points[index])
        result[f"{name}_ci_lower"] = float(lower[index])
        result[f"{name}_ci_upper"] = float(upper[index])
    result["selected_gain_positive_bootstrap_probability"] = float(
        (bootstrap[:, 2] > 0.0).mean()
    )
    result["random_gain_positive_bootstrap_probability"] = float(
        (bootstrap[:, 3] > 0.0).mean()
    )
    return result


def _build_heldout_metrics(
    selections: pd.DataFrame,
    *,
    iterations: int,
    confidence_level: float,
    master_seed: int,
) -> pd.DataFrame:
    rows = []
    for (split_name, profile), group in selections.groupby(
        ["split_name", "utility_profile"], sort=True
    ):
        digest = int(
            hashlib.sha256(f"{split_name}/{profile}".encode()).hexdigest()[:8], 16
        )
        seed = int((master_seed + digest) % (2**31 - 1))
        rows.append(
            {
                "split_name": str(split_name),
                "utility_profile": str(profile),
                "decision_group_count": int(len(group)),
                "problem_cluster_count": int(group["problem_id"].nunique()),
                "bootstrap_iterations": int(iterations),
                "confidence_level": float(confidence_level),
                **_bootstrap_metric_summary(
                    group,
                    iterations=iterations,
                    confidence_level=confidence_level,
                    seed=seed,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["split_name", "utility_profile"]
    ).reset_index(drop=True)


def _validate_phase15(
    run_path: Path,
    phase14_path: Path,
    original_inputs: dict[str, str],
    source: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    current = _load_inputs(phase14_path)
    sensitivity = pd.read_parquet(
        run_path / "data/robustness/utility_sensitivity_selections.parquet"
    )
    sensitivity_metrics = pd.read_parquet(
        run_path / "data/metrics/utility_sensitivity_metrics.parquet"
    )
    stability = pd.read_parquet(
        run_path / "data/robustness/perturbation_stability.parquet"
    )
    profile_stability = pd.read_parquet(
        run_path / "data/robustness/profile_stability.parquet"
    )
    cross_split = pd.read_parquet(
        run_path / "data/robustness/cross_split_agreement.parquet"
    )
    heldout = pd.read_parquet(
        run_path / "data/metrics/heldout_generalization_metrics.parquet"
    )
    registry = json.loads((run_path / "robustness_registry.json").read_text(encoding="utf-8"))
    phase14_candidates = source["candidates"]
    phase14_selections = source["selections"]
    scenarios_per_profile = 1 + 2 * len(SENSITIVITY_COMPONENTS)
    expected_sensitivity_rows = len(phase14_selections) * scenarios_per_profile
    sensitivity_keys = ["split_name", "problem_id", "cutoff", "base_profile", "scenario_id"]
    nominal = sensitivity.loc[sensitivity["is_nominal"]].sort_values(
        ["split_name", "problem_id", "cutoff", "base_profile"]
    )
    original = phase14_selections.rename(columns={"utility_profile": "base_profile"}).sort_values(
        ["split_name", "problem_id", "cutoff", "base_profile"]
    )
    nominal_reproduction = (
        len(nominal) == len(original)
        and np.array_equal(
            nominal["selected_algorithm"].astype(str).to_numpy(),
            original["selected_algorithm"].astype(str).to_numpy(),
        )
        and np.allclose(
            nominal["oracle_regret"].to_numpy(dtype=float),
            original["oracle_regret"].to_numpy(dtype=float),
        )
    )
    scenario_ids_by_profile = {
        profile: {
            str(scenario["scenario_id"])
            for scenario in scenarios
            if scenario["base_profile"] == profile
        }
        for profile in UTILITY_PROFILES
    }
    sensitivity_scenario_coverage = all(
        set(group["scenario_id"].astype(str))
        == scenario_ids_by_profile[str(keys[-1])]
        for keys, group in sensitivity.groupby(
            ["split_name", "problem_id", "cutoff", "base_profile"], sort=False
        )
    )
    ci_columns = [column for column in heldout if column.endswith("_ci_lower")]
    ci_ordered = True
    for lower_column in ci_columns:
        upper_column = lower_column.removesuffix("_ci_lower") + "_ci_upper"
        if upper_column not in heldout or not (
            heldout[lower_column].to_numpy(dtype=float)
            <= heldout[upper_column].to_numpy(dtype=float)
        ).all():
            ci_ordered = False
    expected_profile_rows = len(
        phase14_selections[["split_name", "problem_id", "cutoff"]].drop_duplicates()
    )
    expected_cross_split_rows = len(
        phase14_selections[["problem_id", "cutoff", "utility_profile"]].drop_duplicates()
    )
    expected_heldout_rows = (
        phase14_selections["split_name"].nunique()
        * phase14_selections["utility_profile"].nunique()
    )
    checks = {
        "phase14_quality_pass": current["validation"]["status"] == "PHASE_14_PASS",
        "source_inputs_unchanged": current["input_fingerprints"] == original_inputs,
        "required_heldout_splits_exact": set(sensitivity["split_name"].astype(str))
        == REQUIRED_HELDOUT_SPLITS,
        "scenario_registry_exact": registry["sensitivity_scenarios"] == scenarios,
        "scenario_count_exact": len(scenarios)
        == len(UTILITY_PROFILES) * scenarios_per_profile,
        "sensitivity_keys_unique": not sensitivity.duplicated(sensitivity_keys).any(),
        "sensitivity_coverage_exact": len(sensitivity) == expected_sensitivity_rows,
        "sensitivity_scenario_coverage_exact": sensitivity_scenario_coverage,
        "phase14_nominal_selection_reproduced": nominal_reproduction,
        "sensitivity_utilities_finite": bool(
            np.isfinite(
                sensitivity[
                    [
                        "selected_expected_utility",
                        "selected_realized_utility",
                        "oracle_realized_utility",
                        "oracle_regret",
                        "selection_margin",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "sensitivity_regret_nonnegative": bool((sensitivity["oracle_regret"] >= 0.0).all()),
        "nominal_retention_bounded": bool(
            sensitivity_metrics["nominal_selection_retention_rate"].between(0.0, 1.0).all()
        ),
        "perturbation_stability_complete": len(stability) == len(phase14_selections)
        and stability["stable_scenario_fraction"].between(0.0, 1.0).all(),
        "profile_stability_complete": len(profile_stability) == expected_profile_rows
        and profile_stability["modal_profile_share"].between(0.0, 1.0).all()
        and (profile_stability["utility_profile_count"] == len(UTILITY_PROFILES)).all(),
        "cross_split_agreement_complete": len(cross_split) == expected_cross_split_rows
        and (cross_split["split_count"] == len(REQUIRED_HELDOUT_SPLITS)).all(),
        "heldout_metrics_complete": len(heldout) == expected_heldout_rows,
        "heldout_metrics_finite": bool(
            np.isfinite(heldout.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
        ),
        "bootstrap_intervals_ordered": ci_ordered,
        "phase15_scope_exact": registry["robustness_model"] == ROBUSTNESS_MODEL_NAME,
        "performance_not_used_as_gate": registry["performance_gate_policy"].startswith(
            "construction"
        ),
        "phase14_candidate_source_nonempty": len(phase14_candidates) > 0,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    primary_heldout = heldout.loc[
        heldout["utility_profile"] == PRIMARY_UTILITY_PROFILE
    ]
    primary_stability = stability.loc[
        stability["base_profile"] == PRIMARY_UTILITY_PROFILE
    ]
    primary_cross_split = cross_split.loc[
        cross_split["base_profile"] == PRIMARY_UTILITY_PROFILE
    ]
    return {
        "status": "PHASE_15_PASS" if all(checks.values()) else "PHASE_15_FAIL",
        "schema_version": PHASE15_SCHEMA_VERSION,
        "scope": "frozen-policy selection robustness, utility sensitivity, and held-out generalization",
        "performance_gate_policy": (
            "construction, frozen-policy integrity, deterministic perturbation coverage, held-out "
            "coverage, bootstrap validity, and artifact integrity only; final performance verdict "
            "is deferred to Phase 16"
        ),
        "phase16_boundary": (
            "final multi-metric viability judgment and recommendation are deferred to Phase 16"
        ),
        "scenario_count": int(len(scenarios)),
        "sensitivity_selection_row_count": int(len(sensitivity)),
        "sensitivity_metric_row_count": int(len(sensitivity_metrics)),
        "perturbation_stability_row_count": int(len(stability)),
        "profile_stability_row_count": int(len(profile_stability)),
        "cross_split_agreement_row_count": int(len(cross_split)),
        "heldout_metric_row_count": int(len(heldout)),
        "primary_utility_profile": PRIMARY_UTILITY_PROFILE,
        "primary_heldout_metrics": {
            str(row.split_name): {
                "selection_accuracy": float(row.selection_accuracy),
                "mean_oracle_regret": float(row.mean_oracle_regret),
                "mean_selected_vs_training_global_best_utility_gain": float(
                    row.mean_selected_vs_training_global_best_utility_gain
                ),
                "mean_selected_vs_random_utility_gain": float(
                    row.mean_selected_vs_random_utility_gain
                ),
            }
            for row in primary_heldout.itertuples(index=False)
        },
        "primary_mean_scenario_retention": float(
            primary_stability["stable_scenario_fraction"].mean()
        ),
        "primary_cross_split_agreement_rate": float(
            primary_cross_split["all_splits_agree"].mean()
        ),
        "checks": checks,
        "issues": (
            [{"type": "failed_quality_checks", "checks": failed_checks}]
            if failed_checks
            else []
        ),
    }


def run_phase15(
    run_directory: str | Path,
    phase14_directory: str | Path,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
    perturbation_fraction: float = DEFAULT_PERTURBATION_FRACTION,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    phase14_path = Path(phase14_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"
    if bootstrap_iterations < 100:
        raise ValueError("Phase 15 bootstrap iterations must be at least 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Phase 15 confidence level must be between zero and one")
    scenarios = _build_sensitivity_scenarios(perturbation_fraction)
    source = _load_inputs(phase14_path)
    stable_configuration = {
        "phase": 15,
        "schema_version": PHASE15_SCHEMA_VERSION,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "phase14_implementation_sha256": _file_sha256(Path(phase14_module.__file__)),
        "input_fingerprints": source["input_fingerprints"],
        "master_seed": int(master_seed),
        "robustness_model": ROBUSTNESS_MODEL_NAME,
        "perturbation_fraction": float(perturbation_fraction),
        "bootstrap_iterations": int(bootstrap_iterations),
        "confidence_level": float(confidence_level),
        "sensitivity_components": list(SENSITIVITY_COMPONENTS),
        "required_heldout_splits": sorted(REQUIRED_HELDOUT_SPLITS),
    }
    configuration_sha256 = _configuration_hash(stable_configuration)
    configuration = {
        **stable_configuration,
        "phase14_directory": str(phase14_path.resolve()),
        "git_commit": current_git_commit(),
        "configuration_sha256": configuration_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Phase 15 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_15_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 15 resume manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest.get("output_sha256", {}).items():
                path = run_path / relative
                if not path.is_file() or _file_sha256(path) != expected_hash:
                    raise ValueError(f"completed Phase 15 output hash mismatch: {relative}")
            active_logger.info("[PHASE15][RESUME] status=PHASE_15_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info("[PHASE15][RESUME] restarting atomic analysis stage")
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE15_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": configuration_sha256,
            "input_fingerprints": source["input_fingerprints"],
            "last_completed_stage": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase15_started"})

    active_logger.info(
        "[PHASE15][SENSITIVITY] scenarios=%s candidates=%s",
        len(scenarios),
        len(source["candidates"]),
    )
    sensitivity, sensitivity_metrics, perturbation_stability = _build_sensitivity_outputs(
        source["candidates"], scenarios
    )
    profile_stability = _build_profile_stability(sensitivity)
    cross_split = _build_cross_split_agreement(sensitivity)
    heldout = _build_heldout_metrics(
        source["selections"],
        iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        master_seed=master_seed,
    )
    sensitivity_path = run_path / "data/robustness/utility_sensitivity_selections.parquet"
    sensitivity_metrics_path = run_path / "data/metrics/utility_sensitivity_metrics.parquet"
    perturbation_path = run_path / "data/robustness/perturbation_stability.parquet"
    profile_path = run_path / "data/robustness/profile_stability.parquet"
    cross_split_path = run_path / "data/robustness/cross_split_agreement.parquet"
    heldout_path = run_path / "data/metrics/heldout_generalization_metrics.parquet"
    registry_path = run_path / "robustness_registry.json"
    _atomic_parquet(sensitivity_path, sensitivity)
    _atomic_parquet(sensitivity_metrics_path, sensitivity_metrics)
    _atomic_parquet(perturbation_path, perturbation_stability)
    _atomic_parquet(profile_path, profile_stability)
    _atomic_parquet(cross_split_path, cross_split)
    _atomic_parquet(heldout_path, heldout)
    registry = {
        "schema_version": PHASE15_SCHEMA_VERSION,
        "robustness_model": ROBUSTNESS_MODEL_NAME,
        "primary_utility_profile": PRIMARY_UTILITY_PROFILE,
        "sensitivity_scenarios": scenarios,
        "perturbation_policy": (
            "one component at a time, symmetric configured fraction around frozen Phase 14 "
            "weights; no result-dependent tuning"
        ),
        "heldout_policy": (
            "instance_holdout and family_holdout are reported separately; bootstrap resamples "
            "problem_id clusters and does not alter the policy"
        ),
        "performance_gate_policy": (
            "construction and integrity only; performance verdict is deferred to Phase 16"
        ),
        "selection_tie_break": (
            "scenario expected utility descending, then algorithm name ascending; oracle uses "
            "scenario realized utility descending, then algorithm name ascending"
        ),
    }
    _atomic_json(registry_path, registry)
    state["last_completed_stage"] = "analysis"
    _atomic_json(state_path, state)
    _append_progress(
        progress_path,
        {"event": "analysis_completed", "sensitivity_rows": len(sensitivity)},
    )
    validation = _validate_phase15(
        run_path,
        phase14_path,
        source["input_fingerprints"],
        source,
        scenarios,
    )
    validation["executed_at"] = datetime.now(_timezone()).isoformat(timespec="seconds")
    validation["elapsed_seconds"] = perf_counter() - started_at
    validation["configuration"] = configuration
    validation_path = run_path / "validation.json"
    _atomic_json(validation_path, validation)
    output_paths = sorted(
        {
            run_path / "config.json",
            sensitivity_path,
            sensitivity_metrics_path,
            perturbation_path,
            profile_path,
            cross_split_path,
            heldout_path,
            registry_path,
            validation_path,
        },
        key=lambda path: str(path),
    )
    manifest = {
        "phase": 15,
        "status": validation["status"],
        "schema_version": PHASE15_SCHEMA_VERSION,
        "input_fingerprints": source["input_fingerprints"],
        "output_sha256": _relative_hashes(run_path, output_paths),
        "row_counts": {
            "scenarios": validation["scenario_count"],
            "sensitivity_selections": validation["sensitivity_selection_row_count"],
            "sensitivity_metrics": validation["sensitivity_metric_row_count"],
            "perturbation_stability": validation["perturbation_stability_row_count"],
            "profile_stability": validation["profile_stability_row_count"],
            "cross_split_agreement": validation["cross_split_agreement_row_count"],
            "heldout_metrics": validation["heldout_metric_row_count"],
        },
        "phase16_allowed": validation["status"] == "PHASE_15_PASS",
        "performance_gate_policy": validation["performance_gate_policy"],
    }
    manifest_path = run_path / "manifest.json"
    _atomic_json(manifest_path, manifest)
    state["status"] = validation["status"]
    state["last_completed_stage"] = "validation"
    state["manifest_sha256"] = _file_sha256(manifest_path)
    state["validation_sha256"] = _file_sha256(validation_path)
    _atomic_json(state_path, state)
    _append_progress(
        progress_path, {"event": "phase15_finished", "status": validation["status"]}
    )
    active_logger.info(
        "[PHASE15][SUMMARY] status=%s scenarios=%s selections=%s directory=%s",
        validation["status"],
        validation["scenario_count"],
        validation["sensitivity_selection_row_count"],
        run_path,
    )
    return validation
