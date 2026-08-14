from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _decision(motif: str, regime: str) -> tuple[str, str]:
    if motif == "SUCCESS" and regime == "SUCCESS":
        return "C", "둘을 제한적으로 결합"
    if motif == "SUCCESS" or (motif == "WEAK SIGNAL" and regime == "NO SIGNAL"):
        return "A", "Multi-scale Motif 계속"
    if regime == "SUCCESS" or (regime == "WEAK SIGNAL" and motif == "NO SIGNAL"):
        return "B", "Regime Transition 계속"
    if motif == "WEAK SIGNAL" and regime == "WEAK SIGNAL":
        return "D", "둘 다 종료; 결합은 사전등록된 추가 검증 전까지 금지"
    return "D", "둘 다 종료"


def _cohort(metrics: Mapping[str, Any], algorithm: str, cohort: str) -> Mapping[str, Any]:
    if algorithm == "motif":
        return metrics["cohorts"][cohort]
    return metrics["transition_motif"][cohort]


def compare_motif_experiments(
    *,
    motif_metrics: str | Path,
    regime_metrics: str | Path,
    output_dir: Path,
) -> dict[str, Any]:
    motif = _load(motif_metrics)
    regime = _load(regime_metrics)
    code, description = _decision(str(motif["verdict"]), str(regime["verdict"]))
    rows: list[dict[str, Any]] = []
    for algorithm, metrics in (("Multi-scale Motif", motif), ("Regime Transition", regime)):
        key = "motif" if algorithm == "Multi-scale Motif" else "regime"
        for cohort in ("Historical", "Development"):
            values = _cohort(metrics, key, cohort)
            recall = values["candidate_recall"]["20"]
            opportunity = recall["opportunity"]
            rows.append(
                {
                    "algorithm": algorithm,
                    "cohort": cohort,
                    "verdict": metrics["verdict"],
                    "rounds": values["rounds"],
                    "recurrence_density": values["recurrence"]["actual_density"],
                    "followup_entropy": values["followup_entropy"]["actual_mean"],
                    "mean_hits_at_20": recall["observed_mean_hits"],
                    "random_mean_hits_at_20": recall["expected_mean_hits"],
                    "mean_hit_lift_at_20": recall["mean_hit_lift"],
                    "mean_hit_p_at_20": recall["mean_hit_p"],
                    "opportunity_coverage": values["opportunity"]["coverage"],
                    "opportunity_mean_hit_lift_at_20": opportunity["mean_hit_lift"] if opportunity else None,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "multi_scale_motif": motif["verdict"],
        "regime_transition": regime["verdict"],
        "decision": {"code": code, "description": description},
        "hybrid_allowed": code == "C",
        "locked_holdout": "SEALED",
        "additional_blind": "SEALED",
        "source_metrics": {
            "motif": str(Path(motif_metrics).resolve()),
            "regime": str(Path(regime_metrics).resolve()),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

