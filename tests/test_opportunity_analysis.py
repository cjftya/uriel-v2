from __future__ import annotations

import numpy as np

from uriel_v2.irregular_motif import MotifMatch
from uriel_v2.opportunity_analysis import (
    _reweighted_matches,
    _verdict,
    classify_motif_family,
    classify_opportunity,
    cliffs_delta,
)


def _motif(similarities: dict[str, float], support_count: int = 6) -> MotifMatch:
    return MotifMatch(
        current_start=100,
        current_end=112,
        past_start=0,
        past_end=12,
        window_length=13,
        aggregate_similarity=0.55,
        support_count=support_count,
        similarities=similarities,
    )


def test_opportunity_labels_follow_preregistered_cutoffs() -> None:
    assert classify_opportunity(2) == ("FAIL_0_2", "FAIL_BELOW4")
    assert classify_opportunity(3) == ("HIT_3", "FAIL_BELOW4")
    assert classify_opportunity(4) == ("HIT_4", "SUCCESS_4PLUS")
    assert classify_opportunity(6) == ("HIT_5_PLUS", "SUCCESS_4PLUS")


def test_cliffs_delta_has_expected_direction() -> None:
    assert cliffs_delta(np.asarray([3.0, 4.0]), np.asarray([1.0, 2.0])) == 1.0
    assert cliffs_delta(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])) == -1.0


def test_view_reweighting_uses_only_selected_support_vector() -> None:
    similarities = {
        "raw": 0.9,
        "grid": 0.2,
        "circle": 0.3,
        "distribution": 0.4,
        "transition": 0.5,
        "context": 0.6,
    }
    result = _reweighted_matches([(_motif(similarities), np.asarray([1, 2, 3, 4, 5, 6]))], ("raw",))
    assert result[0].aggregate_similarity == 0.9
    assert result[0].support_count == 1


def test_motif_family_is_target_independent() -> None:
    similarities = {view: 0.55 for view in ("raw", "grid", "circle", "distribution", "transition", "context")}
    motif = _motif(similarities)
    assert classify_motif_family(motif) in {"high-agreement", "moderate-similarity-wide-support"}


def test_verdict_requires_development_stage2_improvement_for_weak_signal() -> None:
    def stage(lift: float, rounds: int = 35, four_plus: float = 0.02) -> dict:
        return {"mean_hit_lift": lift, "rounds": rounds, "four_plus_rate_lift": four_plus}

    summary = {
        "Historical": {"Stage1": {"top20": stage(0.08)}, "Stage2": {"top20": stage(0.15)}},
        "Development": {"Stage1": {"top20": stage(0.33)}, "Stage2": {"top20": stage(0.20)}},
    }
    verdict, decision, _conditions = _verdict(
        stage_summary=summary,
        selected_rule={"conditions": [{"feature": "number_entropy", "threshold": 0.9}]},
        replicated_features=[],
        ablation={"supporting_removals": []},
    )
    assert (verdict, decision) == ("NO SIGNAL", "C")
