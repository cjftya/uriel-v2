from __future__ import annotations

import json
import math

import numpy as np
import pytest

from uriel_v2.cli import build_parser
from uriel_v2.models import Draw
from uriel_v2.motif_features import FeatureBundle, build_feature_bundle
from uriel_v2.top30_broad_retrieval import (
    FROZEN_CONFIDENCE_THRESHOLD,
    RANDOM_EXACT6_RATE,
    RANDOM_FIVE_PLUS_RATE,
    _block_for_round,
    _preregistration,
    _reproduce_seen,
    _score_target,
    _score_with_label,
    _success_decision,
    hypergeometric_reference,
    load_resume_predictions,
)


def _draws(count: int = 170) -> list[Draw]:
    draws: list[Draw] = []
    for round_no in range(1, count + 1):
        start = ((round_no * 7) % 40) + 1
        numbers = tuple(sorted((((start + offset * 6 - 1) % 45) + 1) for offset in range(6)))
        if len(set(numbers)) != 6:
            raise AssertionError(numbers)
        draws.append(Draw(round_no=round_no, numbers=numbers))
    return draws


def test_hypergeometric_theory_matches_preregistration() -> None:
    reference = hypergeometric_reference()
    assert reference["mean_hits"] == pytest.approx(4.0, abs=1e-12)
    assert reference["five_plus_rate"] == pytest.approx(RANDOM_FIVE_PLUS_RATE, abs=5e-11)
    assert reference["exact6_rate"] == pytest.approx(RANDOM_EXACT6_RATE, abs=5e-11)


@pytest.mark.parametrize(
    ("round_no", "block"),
    [
        (852, "Seen-A"),
        (947, "Seen-A"),
        (948, "Seen-B"),
        (1043, "Seen-B"),
        (1044, "Seen-C"),
        (1235, "Seen-D"),
        (660, "Locked-A"),
        (851, "Locked-B"),
        (468, "Blind-A"),
        (659, "Blind-B"),
    ],
)
def test_preregistered_block_boundaries(round_no: int, block: str) -> None:
    assert _block_for_round(round_no) == block


def test_prediction_is_independent_of_target_winning_numbers() -> None:
    bundle = build_feature_bundle(_draws())
    target_index = 160
    original = _score_target(bundle, target_index)
    changed_numbers = bundle.numbers.copy()
    changed_numbers[target_index] = np.asarray((2, 9, 17, 26, 34, 45))
    changed_bundle = FeatureBundle(
        rounds=bundle.rounds,
        numbers=changed_numbers,
        views=bundle.views,
        feature_names=bundle.feature_names,
        grid_masks=bundle.grid_masks,
    )
    changed = _score_target(changed_bundle, target_index)
    assert original["history_end_round"] == 160
    assert original["ranking"] == changed["ranking"]
    assert original["scores"] == changed["scores"]
    assert original["confidence"] == changed["confidence"]


def test_top30_is_unique_and_labeling_happens_after_prediction() -> None:
    bundle = build_feature_bundle(_draws())
    prediction = _score_target(bundle, 160)
    prediction["round"] = 852
    prediction["history_end_round"] = 851
    labeled = _score_with_label(prediction, (2, 9, 17, 26, 34, 45))
    assert len(prediction["ranking"][:30]) == 30
    assert len(set(prediction["ranking"][:30])) == 30
    assert all(1 <= number <= 45 for number in prediction["ranking"][:30])
    assert labeled["history_end_round"] < labeled["round"]
    assert labeled["is_opportunity"] == int(prediction["confidence"] >= FROZEN_CONFIDENCE_THRESHOLD)


def test_locked_success_requires_all_eight_preregistered_conditions() -> None:
    pooled = {
        "opportunity_count": 60,
        "opportunity_coverage": 0.3125,
        "mean_hits_at_30": 4.30,
        "mean_hit_lift": 0.30,
        "mean_hit_p": 0.01,
        "five_plus_rate": 0.45,
        "five_plus_lift": 0.12,
        "five_plus_p": 0.02,
        "exact6_rate": 0.10,
    }
    blocks = [{"mean_hit_lift": 0.20}, {"mean_hit_lift": 0.40}]
    decision, criteria = _success_decision(pooled, blocks)
    assert decision == "SUCCESS"
    assert len(criteria) == 8
    assert all(criteria.values())

    pooled["exact6_rate"] = 0.05
    decision, criteria = _success_decision(pooled, blocks)
    assert decision == "WEAK SIGNAL"
    assert not criteria["exact6_guardrail"]


def test_inconclusive_coverage_does_not_open_blind() -> None:
    pooled = {
        "opportunity_count": 39,
        "opportunity_coverage": 39 / 192,
        "mean_hits_at_30": 4.5,
        "mean_hit_lift": 0.5,
        "mean_hit_p": 0.001,
        "five_plus_rate": 0.50,
        "five_plus_lift": 0.16,
        "five_plus_p": 0.001,
        "exact6_rate": 0.10,
    }
    decision, _criteria = _success_decision(pooled, [{"mean_hit_lift": 0.4}, {"mean_hit_lift": 0.6}])
    assert decision == "INCONCLUSIVE"
    assert decision != "SUCCESS"


def test_failed_mean_and_five_plus_tests_are_no_signal() -> None:
    pooled = {
        "opportunity_count": 64,
        "opportunity_coverage": 1 / 3,
        "mean_hits_at_30": 4.03125,
        "mean_hit_lift": 0.03125,
        "mean_hit_p": 0.4319,
        "five_plus_rate": 0.375,
        "five_plus_lift": 0.0397,
        "five_plus_p": 0.2913,
        "exact6_rate": 0.03125,
    }
    decision, _criteria = _success_decision(
        pooled,
        [{"mean_hit_lift": 0.0952}, {"mean_hit_lift": 0.0}],
    )
    assert decision == "NO SIGNAL"


def test_resume_aborts_on_hash_mismatch(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "config_hash": "old",
                "data_hash": "data",
                "source_hash": "source",
                "prediction": {"round": 852},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resume hash mismatch: config_hash"):
        load_resume_predictions(
            checkpoint,
            {"config_hash": "new", "data_hash": "data", "source_hash": "source"},
        )


def test_preregistration_is_deterministic_and_contains_blind_gate() -> None:
    first = _preregistration("source", "implementation")
    second = _preregistration("source", "implementation")
    assert first == second
    assert "open only if pooled Locked" in first["blind_gate"]
    assert len(first["success_criteria"]) == 8


def test_seen_published_aggregates_are_required_before_locked(tmp_path) -> None:
    records = []
    historical_hits = [5] * 5 + [4] * 53  # 237
    development_hits = [5] * 25 + [4] * 36  # 269
    for offset, hits in enumerate(historical_hits):
        records.append(
            {
                "round": 852 + offset,
                "ranking": list(range(1, 46)),
                "is_opportunity": 1,
                "hits_at_30": hits,
            }
        )
    for offset, hits in enumerate(development_hits):
        records.append(
            {
                "round": 1044 + offset,
                "ranking": list(range(45, 0, -1)),
                "is_opportunity": 1,
                "hits_at_30": hits,
            }
        )
    reproduction = _reproduce_seen(records, tmp_path / "missing-source")
    assert reproduction["status"] == "PASS"
    assert reproduction["cohorts"]["Seen-Historical"]["actual"]["hits_at_30_total"] == 237
    assert reproduction["cohorts"]["Seen-Development"]["actual"]["hits_at_30_total"] == 269


def test_cli_has_no_force_blind_escape_hatch() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "force-blind" not in help_text
    top30 = next(action for action in parser._actions if getattr(action, "dest", None) == "command")
    choices = top30.choices["top30-broad-retrieval"]
    assert all("force-blind" not in option for action in choices._actions for option in action.option_strings)
