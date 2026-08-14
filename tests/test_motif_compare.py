from __future__ import annotations

import json

from uriel_v2.motif_compare import compare_motif_experiments


def _metrics(verdict: str, *, regime: bool = False) -> dict:
    cohort = {
        "rounds": 10,
        "recurrence": {"actual_density": 0.1},
        "followup_entropy": {"actual_mean": 0.8},
        "opportunity": {"coverage": 0.3},
        "candidate_recall": {
            "20": {
                "observed_mean_hits": 2.7,
                "expected_mean_hits": 2.67,
                "mean_hit_lift": 0.03,
                "mean_hit_p": 0.4,
                "opportunity": {"mean_hit_lift": 0.1},
            }
        },
    }
    result = {"verdict": verdict}
    result["transition_motif" if regime else "cohorts"] = {"Historical": cohort, "Development": cohort}
    return result


def test_compare_allows_hybrid_only_for_two_successes(tmp_path) -> None:
    motif = tmp_path / "motif.json"
    regime = tmp_path / "regime.json"
    motif.write_text(json.dumps(_metrics("SUCCESS")), encoding="utf-8")
    regime.write_text(json.dumps(_metrics("SUCCESS", regime=True)), encoding="utf-8")
    summary = compare_motif_experiments(motif_metrics=motif, regime_metrics=regime, output_dir=tmp_path / "out")
    assert summary["decision"]["code"] == "C"
    assert summary["hybrid_allowed"] is True
