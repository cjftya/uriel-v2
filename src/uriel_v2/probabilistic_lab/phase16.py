from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

import uriel_v2.probabilistic_lab.phase14 as phase14_module
import uriel_v2.probabilistic_lab.phase15 as phase15_module
from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.phase7 import (
    _append_progress,
    _atomic_json,
    _atomic_parquet,
    _configuration_hash,
    _file_sha256,
    _relative_hashes,
)
from uriel_v2.probabilistic_lab.phase14 import PRIMARY_UTILITY_PROFILE
from uriel_v2.probabilistic_lab.phase15 import _load_inputs as _load_phase14_input
from uriel_v2.provenance import current_git_commit


PHASE16_SCHEMA_VERSION = "phase16-v1"
ASSESSMENT_MODEL_NAME = "preregistered_multi_metric_final_viability_assessment"
REQUIRED_HELDOUT_SPLITS = {"instance_holdout", "family_holdout"}
FINAL_THRESHOLDS: dict[str, float] = {
    "minimum_point_mae_skill": 0.0,
    "maximum_marginal_calibration_mae": 0.10,
    "minimum_survival_c_index": 0.65,
    "maximum_survival_integrated_brier": 0.10,
    "maximum_joint_calibration_error": 0.10,
    "maximum_joint_nll_delta_vs_reference": 0.0,
    "minimum_selection_gain_ci_lower": 0.0,
    "maximum_selection_regret_delta": 0.0,
    "minimum_scenario_retention": 0.90,
    "minimum_cross_split_agreement": 0.75,
    "maximum_unavailable_expert_slots": 0.0,
}
CRITERION_ORDER = (
    "point_prediction_skill",
    "marginal_distribution_calibration",
    "survival_prediction",
    "joint_probability_generalization",
    "failure_risk_estimability",
    "domain_expert_coverage",
    "selection_value_vs_random",
    "selection_value_vs_training_global_best",
    "selection_regret_vs_baselines",
    "selection_policy_robustness",
)


def _phase15_required_paths(phase15_path: Path) -> dict[str, Path]:
    return {
        "phase15_config": phase15_path / "config.json",
        "phase15_manifest": phase15_path / "manifest.json",
        "phase15_validation": phase15_path / "validation.json",
        "phase15_heldout_metrics": phase15_path
        / "data/metrics/heldout_generalization_metrics.parquet",
        "phase15_sensitivity_metrics": phase15_path
        / "data/metrics/utility_sensitivity_metrics.parquet",
        "phase15_perturbation_stability": phase15_path
        / "data/robustness/perturbation_stability.parquet",
        "phase15_profile_stability": phase15_path
        / "data/robustness/profile_stability.parquet",
        "phase15_cross_split_agreement": phase15_path
        / "data/robustness/cross_split_agreement.parquet",
        "phase15_registry": phase15_path / "robustness_registry.json",
    }


def _load_phase15_input(
    phase15_path: Path,
    expected_phase14_fingerprints: dict[str, str],
) -> dict[str, Any]:
    required = _phase15_required_paths(phase15_path)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 16 Phase 15 input files are missing: {missing[:5]}")
    validation = json.loads(required["phase15_validation"].read_text(encoding="utf-8"))
    manifest = json.loads(required["phase15_manifest"].read_text(encoding="utf-8"))
    if validation.get("status") != "PHASE_15_PASS":
        raise ValueError("Phase 16 requires a PHASE_15_PASS robustness run")
    if manifest.get("status") != "PHASE_15_PASS" or not manifest.get("phase16_allowed"):
        raise ValueError("Phase 15 manifest does not allow Phase 16")
    if (
        validation.get("configuration", {}).get("input_fingerprints")
        != expected_phase14_fingerprints
    ):
        raise ValueError("Phase 15 was not built from the supplied Phase 14 input")
    for relative, expected_hash in manifest.get("output_sha256", {}).items():
        path = phase15_path / relative
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Phase 15 manifest output hash mismatch: {relative}")
    return {
        "validation": validation,
        "manifest": manifest,
        "heldout_metrics": pd.read_parquet(required["phase15_heldout_metrics"]),
        "sensitivity_metrics": pd.read_parquet(required["phase15_sensitivity_metrics"]),
        "perturbation_stability": pd.read_parquet(
            required["phase15_perturbation_stability"]
        ),
        "profile_stability": pd.read_parquet(required["phase15_profile_stability"]),
        "cross_split_agreement": pd.read_parquet(
            required["phase15_cross_split_agreement"]
        ),
        "input_fingerprints": {
            f"phase15/{name}": _file_sha256(path)
            for name, path in sorted(required.items())
        },
    }


def _load_inputs(paths: tuple[Path, ...]) -> dict[str, Any]:
    if len(paths) != 10:
        raise ValueError("Phase 16 requires Phase 6 through Phase 15 directories")
    phase6_to_13 = paths[:8]
    phase14_path, phase15_path = paths[8:]
    phase14_source = phase14_module._load_inputs(*phase6_to_13)
    phase14_input = _load_phase14_input(phase14_path)
    if (
        phase14_input["validation"].get("configuration", {}).get("input_fingerprints")
        != phase14_source["input_fingerprints"]
    ):
        raise ValueError("Phase 14 was not built from the supplied Phase 6 through 13 inputs")
    phase15_input = _load_phase15_input(
        phase15_path, phase14_input["input_fingerprints"]
    )
    validations = {
        **phase14_source["validations"],
        "phase14": phase14_input["validation"],
        "phase15": phase15_input["validation"],
    }
    fingerprints = {
        **phase14_source["input_fingerprints"],
        **phase14_input["input_fingerprints"],
        **phase15_input["input_fingerprints"],
    }
    phase7_metrics = pd.read_parquet(paths[1] / "data/metrics/aggregate_metrics.parquet")
    phase13_metrics = pd.read_parquet(
        paths[7] / "data/metrics/aggregate_joint_calibration_metrics.parquet"
    )
    phase14_metrics = pd.read_parquet(
        phase14_path / "data/metrics/aggregate_selection_metrics.parquet"
    )
    return {
        "validations": validations,
        "input_fingerprints": fingerprints,
        "phase7_metrics": phase7_metrics,
        "phase9_validation": validations["phase9"],
        "phase12_validation": validations["phase12"],
        "phase13_metrics": phase13_metrics,
        "phase14_metrics": phase14_metrics,
        "phase15": phase15_input,
    }


def _metric_row(
    *,
    criterion_id: str,
    source_phase: int,
    metric: str,
    value: float,
    threshold: float,
    direction: str,
    split_name: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    if direction == ">":
        passed = value > threshold
    elif direction == ">=":
        passed = value >= threshold
    elif direction == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"Unsupported Phase 16 threshold direction: {direction}")
    return {
        "criterion_id": criterion_id,
        "source_phase": int(source_phase),
        "split_name": split_name,
        "metric": metric,
        "value": float(value),
        "threshold": float(threshold),
        "direction": direction,
        "passes_threshold": bool(passed),
        "note": note,
    }


def _build_metric_summary(source: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    point = source["phase7_metrics"]
    point = point.loc[
        (point["scope"] == "overall")
        & point["target"].isin(["quality", "runtime"])
    ].copy()
    point = point.sort_values(
        ["split_name", "target", "mae_skill_vs_train_mean", "model"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).groupby(["split_name", "target"], sort=False).head(1)
    for row in point.itertuples(index=False):
        rows.append(
            _metric_row(
                criterion_id="point_prediction_skill",
                source_phase=7,
                split_name=str(row.split_name),
                metric=f"{row.target}_best_mae_skill_vs_train_mean",
                value=float(row.mae_skill_vs_train_mean),
                threshold=FINAL_THRESHOLDS["minimum_point_mae_skill"],
                direction=">",
                note=f"best model={row.model}",
            )
        )

    joint = source["phase13_metrics"]
    joint = joint.loc[joint["scope"] == "overall"].sort_values("split_name")
    for row in joint.itertuples(index=False):
        split_name = str(row.split_name)
        for metric in (
            "calibrated_quality_calibration_mae",
            "calibrated_runtime_calibration_mae",
        ):
            rows.append(
                _metric_row(
                    criterion_id="marginal_distribution_calibration",
                    source_phase=13,
                    split_name=split_name,
                    metric=metric,
                    value=float(getattr(row, metric)),
                    threshold=FINAL_THRESHOLDS["maximum_marginal_calibration_mae"],
                    direction="<=",
                )
            )
        rows.extend(
            [
                _metric_row(
                    criterion_id="survival_prediction",
                    source_phase=13,
                    split_name=split_name,
                    metric="calibrated_survival_c_index",
                    value=float(row.calibrated_survival_c_index),
                    threshold=FINAL_THRESHOLDS["minimum_survival_c_index"],
                    direction=">=",
                ),
                _metric_row(
                    criterion_id="survival_prediction",
                    source_phase=13,
                    split_name=split_name,
                    metric="calibrated_survival_integrated_brier",
                    value=float(row.calibrated_survival_integrated_brier),
                    threshold=FINAL_THRESHOLDS["maximum_survival_integrated_brier"],
                    direction="<=",
                ),
                _metric_row(
                    criterion_id="joint_probability_generalization",
                    source_phase=13,
                    split_name=split_name,
                    metric="joint_nll_delta_vs_reference",
                    value=float(row.joint_nll_delta_vs_reference),
                    threshold=FINAL_THRESHOLDS[
                        "maximum_joint_nll_delta_vs_reference"
                    ],
                    direction="<=",
                ),
                _metric_row(
                    criterion_id="joint_probability_generalization",
                    source_phase=13,
                    split_name=split_name,
                    metric="absolute_joint_q075_calibration_error",
                    value=abs(float(row.joint_q075_calibration_error)),
                    threshold=FINAL_THRESHOLDS["maximum_joint_calibration_error"],
                    direction="<=",
                ),
                _metric_row(
                    criterion_id="joint_probability_generalization",
                    source_phase=13,
                    split_name=split_name,
                    metric="absolute_joint_q090_calibration_error",
                    value=abs(float(row.joint_q090_calibration_error)),
                    threshold=FINAL_THRESHOLDS["maximum_joint_calibration_error"],
                    direction="<=",
                ),
            ]
        )

    phase9 = source["phase9_validation"]
    rows.extend(
        [
            _metric_row(
                criterion_id="failure_risk_estimability",
                source_phase=9,
                metric="failure_probability_estimable",
                value=float(bool(phase9["failure_probability_estimable"])),
                threshold=1.0,
                direction=">=",
                note=str(phase9["estimability_status"]),
            ),
            _metric_row(
                criterion_id="failure_risk_estimability",
                source_phase=9,
                metric="failure_type_estimable",
                value=float(bool(phase9["failure_type_estimable"])),
                threshold=1.0,
                direction=">=",
                note=str(phase9["estimability_status"]),
            ),
        ]
    )
    unavailable_slots = source["phase12_validation"].get("unavailable_expert_slots", [])
    rows.append(
        _metric_row(
            criterion_id="domain_expert_coverage",
            source_phase=12,
            metric="unavailable_expert_slot_count",
            value=float(len(unavailable_slots)),
            threshold=FINAL_THRESHOLDS["maximum_unavailable_expert_slots"],
            direction="<=",
            note=",".join(str(value) for value in unavailable_slots),
        )
    )

    heldout = source["phase15"]["heldout_metrics"]
    heldout = heldout.loc[
        heldout["utility_profile"] == PRIMARY_UTILITY_PROFILE
    ].sort_values("split_name")
    for row in heldout.itertuples(index=False):
        split_name = str(row.split_name)
        rows.extend(
            [
                _metric_row(
                    criterion_id="selection_value_vs_random",
                    source_phase=15,
                    split_name=split_name,
                    metric="selected_vs_random_utility_gain_ci_lower",
                    value=float(row.mean_selected_vs_random_utility_gain_ci_lower),
                    threshold=FINAL_THRESHOLDS["minimum_selection_gain_ci_lower"],
                    direction=">",
                ),
                _metric_row(
                    criterion_id="selection_value_vs_training_global_best",
                    source_phase=15,
                    split_name=split_name,
                    metric="selected_vs_training_global_best_utility_gain_ci_lower",
                    value=float(
                        row.mean_selected_vs_training_global_best_utility_gain_ci_lower
                    ),
                    threshold=FINAL_THRESHOLDS["minimum_selection_gain_ci_lower"],
                    direction=">",
                ),
            ]
        )

    selection = source["phase14_metrics"]
    selection = selection.loc[
        (selection["scope"] == "overall")
        & (selection["utility_profile"] == PRIMARY_UTILITY_PROFILE)
    ].sort_values("split_name")
    for row in selection.itertuples(index=False):
        split_name = str(row.split_name)
        rows.extend(
            [
                _metric_row(
                    criterion_id="selection_regret_vs_baselines",
                    source_phase=14,
                    split_name=split_name,
                    metric="oracle_regret_delta_vs_training_global_best",
                    value=float(
                        row.mean_oracle_regret
                        - row.mean_training_global_best_regret
                    ),
                    threshold=FINAL_THRESHOLDS["maximum_selection_regret_delta"],
                    direction="<=",
                ),
                _metric_row(
                    criterion_id="selection_regret_vs_baselines",
                    source_phase=14,
                    split_name=split_name,
                    metric="oracle_regret_delta_vs_random",
                    value=float(row.mean_oracle_regret - row.mean_random_regret),
                    threshold=FINAL_THRESHOLDS["maximum_selection_regret_delta"],
                    direction="<=",
                ),
            ]
        )

    phase15 = source["phase15"]["validation"]
    rows.extend(
        [
            _metric_row(
                criterion_id="selection_policy_robustness",
                source_phase=15,
                metric="primary_mean_scenario_retention",
                value=float(phase15["primary_mean_scenario_retention"]),
                threshold=FINAL_THRESHOLDS["minimum_scenario_retention"],
                direction=">=",
            ),
            _metric_row(
                criterion_id="selection_policy_robustness",
                source_phase=15,
                metric="primary_cross_split_agreement_rate",
                value=float(phase15["primary_cross_split_agreement_rate"]),
                threshold=FINAL_THRESHOLDS["minimum_cross_split_agreement"],
                direction=">=",
            ),
        ]
    )
    return pd.DataFrame(rows).sort_values(
        ["criterion_id", "split_name", "metric"], na_position="first"
    ).reset_index(drop=True)


def _build_criteria(metric_summary: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "point_prediction_skill": "Point prediction beats training-mean reference",
        "marginal_distribution_calibration": "Quality and runtime marginals are calibrated",
        "survival_prediction": "First-passage survival is discriminative and calibrated",
        "joint_probability_generalization": "Joint probability improves and calibrates on both holdouts",
        "failure_risk_estimability": "Failure probability and type are empirically estimable",
        "domain_expert_coverage": "Every preregistered domain expert is executable",
        "selection_value_vs_random": "Selected policy beats random with positive lower CI",
        "selection_value_vs_training_global_best": "Selected policy beats training-global-best with positive lower CI",
        "selection_regret_vs_baselines": "Selected policy regret is no worse than both baselines",
        "selection_policy_robustness": "Selection is stable to weight and holdout changes",
    }
    rows = []
    for order, criterion_id in enumerate(CRITERION_ORDER, start=1):
        group = metric_summary.loc[metric_summary["criterion_id"] == criterion_id]
        if group.empty:
            raise ValueError(f"Phase 16 criterion has no metrics: {criterion_id}")
        passed = bool(group["passes_threshold"].all())
        status = "PASS" if passed else "FAIL"
        if criterion_id == "failure_risk_estimability" and not passed:
            status = "UNAVAILABLE"
        elif criterion_id == "domain_expert_coverage" and not passed:
            status = "INCOMPLETE"
        failed_metrics = group.loc[~group["passes_threshold"], "metric"].astype(str).tolist()
        rows.append(
            {
                "criterion_order": order,
                "criterion_id": criterion_id,
                "label": labels[criterion_id],
                "status": status,
                "required_for_deployment": True,
                "metric_count": int(len(group)),
                "passed_metric_count": int(group["passes_threshold"].sum()),
                "failed_metrics": json.dumps(failed_metrics, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def _decide_verdict(criteria: pd.DataFrame) -> dict[str, Any]:
    status = {
        str(row.criterion_id): str(row.status)
        for row in criteria.itertuples(index=False)
    }
    deployment_ready = all(value == "PASS" for value in status.values())
    core_research = all(
        status[criterion] == "PASS"
        for criterion in (
            "point_prediction_skill",
            "marginal_distribution_calibration",
            "survival_prediction",
        )
    )
    if deployment_ready:
        return {
            "code": "A",
            "verdict": "DEPLOYABLE_SUCCESS",
            "deployment_ready": True,
            "research_value": True,
            "summary": (
                "All predictive, estimability, coverage, utility, regret, and robustness gates pass."
            ),
        }
    if core_research:
        return {
            "code": "B",
            "verdict": "PARTIAL_SUCCESS_RESEARCH_ONLY",
            "deployment_ready": False,
            "research_value": True,
            "summary": (
                "Predictive distribution modelling is useful, but automated algorithm selection "
                "is not validated for deployment."
            ),
        }
    if any(
        status[criterion] == "PASS"
        for criterion in (
            "point_prediction_skill",
            "marginal_distribution_calibration",
            "survival_prediction",
        )
    ):
        return {
            "code": "C",
            "verdict": "INCONCLUSIVE_LIMITED_SIGNAL",
            "deployment_ready": False,
            "research_value": True,
            "summary": "Only part of the predictive stack generalizes; more evidence is required.",
        }
    return {
        "code": "D",
        "verdict": "NO_PRACTICAL_SIGNAL",
        "deployment_ready": False,
        "research_value": False,
        "summary": "The predictive and selection stack does not show practical held-out value.",
    }


def _build_recommendations(criteria: pd.DataFrame) -> pd.DataFrame:
    status = dict(zip(criteria["criterion_id"], criteria["status"], strict=True))
    rows = [
        {
            "priority": 1,
            "category": "deployment",
            "action": "Do not deploy automated algorithm selection from this run",
            "rationale": "Deployment requires every Phase 16 criterion to pass.",
            "reopen_gate": "all_failed_phase16_criteria",
        }
    ]
    if status["failure_risk_estimability"] != "PASS":
        rows.append(
            {
                "priority": 2,
                "category": "data",
                "action": "Collect controlled failure and timeout events before refitting risk models",
                "rationale": "Zero observed failures cannot validate failure probability or type.",
                "reopen_gate": "failure_risk_estimability",
            }
        )
    if status["domain_expert_coverage"] != "PASS":
        rows.append(
            {
                "priority": 3,
                "category": "coverage",
                "action": "Execute matrix, stream, and natural-process expert benchmarks",
                "rationale": "Unexecuted expert slots prevent full-domain generalization claims.",
                "reopen_gate": "domain_expert_coverage",
            }
        )
    if any(
        status[name] != "PASS"
        for name in (
            "selection_value_vs_random",
            "selection_value_vs_training_global_best",
            "selection_regret_vs_baselines",
            "selection_policy_robustness",
        )
    ):
        rows.append(
            {
                "priority": 4,
                "category": "selection",
                "action": "Keep the predictor as a research diagnostic and redesign the selector",
                "rationale": (
                    "A selector must beat random and training-global baselines on both holdouts "
                    "without increasing oracle regret."
                ),
                "reopen_gate": "selection_value_and_robustness",
            }
        )
    if status["joint_probability_generalization"] != "PASS":
        rows.append(
            {
                "priority": 5,
                "category": "calibration",
                "action": "Revalidate joint calibration on a larger untouched family holdout",
                "rationale": "Marginals can be useful while their joint tail probability is miscalibrated.",
                "reopen_gate": "joint_probability_generalization",
            }
        )
    rows.append(
        {
            "priority": 6,
            "category": "validation",
            "action": "Freeze one external benchmark and rerun Phase 16 only after blockers change",
            "rationale": "Repeated threshold or weight tuning on current holdouts would invalidate them.",
            "reopen_gate": "external_untouched_validation",
        }
    )
    return pd.DataFrame(rows).sort_values("priority").reset_index(drop=True)


def _validate_phase16(
    run_path: Path,
    paths: tuple[Path, ...],
    original_inputs: dict[str, str],
    source: dict[str, Any],
) -> dict[str, Any]:
    current = _load_inputs(paths)
    metrics = pd.read_parquet(run_path / "data/final/metric_summary.parquet")
    criteria = pd.read_parquet(run_path / "data/final/criterion_results.parquet")
    recommendations = pd.read_parquet(run_path / "data/final/recommendations.parquet")
    assessment = json.loads((run_path / "final_assessment.json").read_text(encoding="utf-8"))
    registry = json.loads((run_path / "assessment_registry.json").read_text(encoding="utf-8"))
    recomputed = _decide_verdict(criteria)
    expected_statuses = {
        f"phase{phase}": f"PHASE_{phase}_PASS" for phase in range(6, 16)
    }
    checks = {
        **{
            f"{name}_quality_pass": current["validations"][name]["status"] == expected
            for name, expected in expected_statuses.items()
        },
        "source_inputs_unchanged": current["input_fingerprints"] == original_inputs,
        "metric_rows_nonempty_unique": len(metrics) > 0
        and not metrics.duplicated(["criterion_id", "split_name", "metric"]).any(),
        "metric_values_finite": bool(np.isfinite(metrics["value"].to_numpy(dtype=float)).all()),
        "metric_thresholds_finite": bool(
            np.isfinite(metrics["threshold"].to_numpy(dtype=float)).all()
        ),
        "criterion_set_exact": set(criteria["criterion_id"]) == set(CRITERION_ORDER)
        and len(criteria) == len(CRITERION_ORDER),
        "criterion_order_exact": criteria.sort_values("criterion_order")[
            "criterion_id"
        ].tolist()
        == list(CRITERION_ORDER),
        "criterion_statuses_valid": set(criteria["status"])
        <= {"PASS", "FAIL", "UNAVAILABLE", "INCOMPLETE"},
        "verdict_recomputed_exact": assessment["final_decision"] == recomputed,
        "deployment_requires_all_pass": bool(
            recomputed["deployment_ready"]
            == bool((criteria["status"] == "PASS").all())
        ),
        "recommendations_cover_blockers": set(
            recommendations.loc[
                recommendations["reopen_gate"] != "external_untouched_validation",
                "reopen_gate",
            ]
        )
        >= {"all_failed_phase16_criteria"},
        "threshold_registry_exact": registry["final_thresholds"] == FINAL_THRESHOLDS,
        "performance_is_final_gate": registry["performance_gate_policy"]
        == "Phase 16 uses held-out performance as the final viability gate",
        "phase16_scope_exact": registry["assessment_model"] == ASSESSMENT_MODEL_NAME,
        "required_heldout_splits_exact": set(
            source["phase15"]["heldout_metrics"]["split_name"].astype(str)
        )
        == REQUIRED_HELDOUT_SPLITS,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "PHASE_16_PASS" if all(checks.values()) else "PHASE_16_FAIL",
        "schema_version": PHASE16_SCHEMA_VERSION,
        "scope": "final multi-metric viability judgment and recommendation",
        "project_complete": bool(all(checks.values())),
        "final_decision": recomputed,
        "criterion_count": int(len(criteria)),
        "passed_criterion_count": int((criteria["status"] == "PASS").sum()),
        "failed_criterion_count": int((criteria["status"] == "FAIL").sum()),
        "unavailable_criterion_count": int(
            criteria["status"].isin(["UNAVAILABLE", "INCOMPLETE"]).sum()
        ),
        "metric_row_count": int(len(metrics)),
        "recommendation_count": int(len(recommendations)),
        "failed_deployment_criteria": criteria.loc[
            criteria["status"] != "PASS", "criterion_id"
        ].astype(str).tolist(),
        "checks": checks,
        "issues": (
            [{"type": "failed_quality_checks", "checks": failed_checks}]
            if failed_checks
            else []
        ),
    }


def run_phase16(
    run_directory: str | Path,
    phase6_directory: str | Path,
    phase7_directory: str | Path,
    phase8_directory: str | Path,
    phase9_directory: str | Path,
    phase10_directory: str | Path,
    phase11_directory: str | Path,
    phase12_directory: str | Path,
    phase13_directory: str | Path,
    phase14_directory: str | Path,
    phase15_directory: str | Path,
    *,
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
            phase14_directory,
            phase15_directory,
        )
    )
    run_path.mkdir(parents=True, exist_ok=True)
    active_logger = logger or logging.getLogger("uriel")
    progress_path = run_path / "progress.jsonl"
    source = _load_inputs(paths)
    stable_configuration = {
        "phase": 16,
        "schema_version": PHASE16_SCHEMA_VERSION,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "input_fingerprints": source["input_fingerprints"],
        "assessment_model": ASSESSMENT_MODEL_NAME,
        "final_thresholds": FINAL_THRESHOLDS,
        "criterion_order": list(CRITERION_ORDER),
        "primary_utility_profile": PRIMARY_UTILITY_PROFILE,
    }
    configuration_sha256 = _configuration_hash(stable_configuration)
    configuration = {
        **stable_configuration,
        **{f"phase{phase}_directory": str(path.resolve()) for phase, path in zip(range(6, 16), paths)},
        "git_commit": current_git_commit(),
        "configuration_sha256": configuration_sha256,
    }
    state_path = run_path / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Phase 16 resume configuration or input hash mismatch")
        if state.get("status") == "PHASE_16_PASS":
            manifest_path = run_path / "manifest.json"
            validation_path = run_path / "validation.json"
            if (
                not manifest_path.is_file()
                or not validation_path.is_file()
                or _file_sha256(manifest_path) != state.get("manifest_sha256")
                or _file_sha256(validation_path) != state.get("validation_sha256")
            ):
                raise ValueError("completed Phase 16 resume manifest hash mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest.get("output_sha256", {}).items():
                path = run_path / relative
                if not path.is_file() or _file_sha256(path) != expected_hash:
                    raise ValueError(f"completed Phase 16 output hash mismatch: {relative}")
            active_logger.info("[PHASE16][RESUME] status=PHASE_16_PASS outputs=VERIFIED")
            _append_progress(progress_path, {"event": "completed_run_verified"})
            return json.loads(validation_path.read_text(encoding="utf-8"))
        active_logger.info("[PHASE16][RESUME] restarting atomic assessment stage")
    else:
        _atomic_json(run_path / "config.json", configuration)
        state = {
            "schema_version": PHASE16_SCHEMA_VERSION,
            "status": "RUNNING",
            "configuration_sha256": configuration_sha256,
            "input_fingerprints": source["input_fingerprints"],
            "last_completed_stage": None,
        }
        _atomic_json(state_path, state)
        _append_progress(progress_path, {"event": "phase16_started"})

    metrics = _build_metric_summary(source)
    criteria = _build_criteria(metrics)
    decision = _decide_verdict(criteria)
    recommendations = _build_recommendations(criteria)
    metric_path = run_path / "data/final/metric_summary.parquet"
    criteria_path = run_path / "data/final/criterion_results.parquet"
    recommendations_path = run_path / "data/final/recommendations.parquet"
    assessment_path = run_path / "final_assessment.json"
    registry_path = run_path / "assessment_registry.json"
    _atomic_parquet(metric_path, metrics)
    _atomic_parquet(criteria_path, criteria)
    _atomic_parquet(recommendations_path, recommendations)
    assessment = {
        "schema_version": PHASE16_SCHEMA_VERSION,
        "final_decision": decision,
        "criterion_summary": {
            str(row.criterion_id): str(row.status)
            for row in criteria.itertuples(index=False)
        },
        "failed_deployment_criteria": criteria.loc[
            criteria["status"] != "PASS", "criterion_id"
        ].astype(str).tolist(),
        "recommendations": recommendations["action"].astype(str).tolist(),
    }
    registry = {
        "schema_version": PHASE16_SCHEMA_VERSION,
        "assessment_model": ASSESSMENT_MODEL_NAME,
        "primary_utility_profile": PRIMARY_UTILITY_PROFILE,
        "final_thresholds": FINAL_THRESHOLDS,
        "criterion_order": list(CRITERION_ORDER),
        "verdict_mapping": {
            "A": "DEPLOYABLE_SUCCESS",
            "B": "PARTIAL_SUCCESS_RESEARCH_ONLY",
            "C": "INCONCLUSIVE_LIMITED_SIGNAL",
            "D": "NO_PRACTICAL_SIGNAL",
        },
        "performance_gate_policy": (
            "Phase 16 uses held-out performance as the final viability gate"
        ),
        "no_posthoc_tuning_policy": (
            "thresholds are fixed in code and no Phase 16 result may alter the current holdouts"
        ),
    }
    _atomic_json(assessment_path, assessment)
    _atomic_json(registry_path, registry)
    state["last_completed_stage"] = "assessment"
    _atomic_json(state_path, state)
    _append_progress(
        progress_path,
        {"event": "assessment_completed", "verdict": decision["verdict"]},
    )
    validation = _validate_phase16(
        run_path, paths, source["input_fingerprints"], source
    )
    validation["executed_at"] = datetime.now(_timezone()).isoformat(timespec="seconds")
    validation["elapsed_seconds"] = perf_counter() - started_at
    validation["configuration"] = configuration
    validation_path = run_path / "validation.json"
    _atomic_json(validation_path, validation)
    output_paths = sorted(
        {
            run_path / "config.json",
            metric_path,
            criteria_path,
            recommendations_path,
            assessment_path,
            registry_path,
            validation_path,
        },
        key=lambda path: str(path),
    )
    manifest = {
        "phase": 16,
        "status": validation["status"],
        "schema_version": PHASE16_SCHEMA_VERSION,
        "input_fingerprints": source["input_fingerprints"],
        "output_sha256": _relative_hashes(run_path, output_paths),
        "row_counts": {
            "metrics": validation["metric_row_count"],
            "criteria": validation["criterion_count"],
            "recommendations": validation["recommendation_count"],
        },
        "project_complete": validation["project_complete"],
        "final_decision": validation["final_decision"],
    }
    manifest_path = run_path / "manifest.json"
    _atomic_json(manifest_path, manifest)
    state["status"] = validation["status"]
    state["last_completed_stage"] = "validation"
    state["manifest_sha256"] = _file_sha256(manifest_path)
    state["validation_sha256"] = _file_sha256(validation_path)
    _atomic_json(state_path, state)
    _append_progress(
        progress_path,
        {
            "event": "phase16_finished",
            "status": validation["status"],
            "verdict": decision["verdict"],
        },
    )
    active_logger.info(
        "[PHASE16][SUMMARY] status=%s verdict=%s passed=%s/%s directory=%s",
        validation["status"],
        decision["verdict"],
        validation["passed_criterion_count"],
        validation["criterion_count"],
        run_path,
    )
    return validation
