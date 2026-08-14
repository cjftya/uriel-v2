from __future__ import annotations

import numpy as np

from uriel_v2.irregular_motif import MotifConfig, _checkpoint_payloads, _verdict, dtw_distance, retrieve_motifs
from uriel_v2.models import Draw
from uriel_v2.motif_features import build_feature_bundle


def test_dtw_accepts_time_warped_sequences() -> None:
    short = np.asarray([[0.0], [1.0], [2.0]])
    stretched = np.asarray([[0.0], [0.5], [1.0], [1.5], [2.0]])
    unrelated = np.asarray([[3.0], [3.0], [3.0]])
    assert dtw_distance(short, stretched) < dtw_distance(short, unrelated)
    assert dtw_distance(short, stretched, derivative=True) >= 0.0


def test_retrieve_motifs_enforces_temporal_separation() -> None:
    draws = [
        Draw(round_no=index + 1, numbers=tuple(sorted((((index * 3 + step * 7) % 45) + 1 for step in range(6)))))
        for index in range(80)
    ]
    bundle = build_feature_bundle(draws)
    config = MotifConfig("test", 5, (4, 5, 6), 8, 20, ("raw", "transition"))
    matches = retrieve_motifs(bundle, 79, config)
    assert matches
    assert all(match.past_end <= 59 for match in matches)
    assert len({match.past_end for match in matches}) == len(matches)


def test_weak_signal_uses_same_direction_opportunity_lift() -> None:
    def cohort(lift: float, opportunity_lift: float) -> dict:
        return {
            "candidate_recall": {
                "20": {
                    "mean_hit_lift": lift,
                    "mean_hit_p": 0.3,
                    "opportunity": {"mean_hit_lift": opportunity_lift},
                }
            },
            "followup_entropy": {
                "surrogates": {
                    name: {"reduction_actual_vs_surrogate": -0.001}
                    for name in ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving")
                }
            },
        }

    assert _verdict(cohort(0.05, 0.07), cohort(0.04, 0.33)) == "WEAK SIGNAL"
    assert _verdict(cohort(-0.05, -0.07), cohort(0.04, 0.33)) == "NO SIGNAL"


def test_checkpoint_loader_filters_selected_config(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        '{"row":{"round":1044,"config":"selected"},"predictions":[],"matches":[]}\n'
        '{"row":{"round":1045,"config":"other"},"predictions":[],"matches":[]}\n',
        encoding="utf-8",
    )
    payloads = _checkpoint_payloads(checkpoint, "selected")
    assert list(payloads) == [1044]
