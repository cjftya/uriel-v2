from __future__ import annotations

from uriel_v2.models import Draw
from uriel_v2.motif_features import build_feature_bundle
from uriel_v2.regime_motif import RegimeConfig, _edit_distance, _state_matrix, fit_regimes


def _draws(count: int = 60) -> list[Draw]:
    draws: list[Draw] = []
    for index in range(count):
        offset = 0 if index % 12 < 6 else 18
        values = tuple(sorted((((offset + index + step * 5) % 45) + 1 for step in range(6))))
        draws.append(Draw(index + 1, values))
    return draws


def test_edit_distance_supports_variable_length_transition_motifs() -> None:
    assert _edit_distance([0, 1, 1, 2], [0, 1, 2]) == 1
    assert _edit_distance([0, 1, 2], [0, 1, 2]) == 0


def test_kmeans_fit_exposes_soft_regime_probabilities() -> None:
    bundle = build_feature_bundle(_draws())
    fit = fit_regimes(bundle, 59, RegimeConfig("test", "KMeans", 4), 20260814)
    assert fit.labels.shape == (60,)
    assert fit.probabilities.shape == (60, 4)
    assert all(abs(float(row.sum()) - 1.0) < 1e-8 for row in fit.probabilities)


def test_state_reduction_is_deterministic_for_collinear_history() -> None:
    bundle = build_feature_bundle(_draws(80))
    first = _state_matrix(bundle, 79)
    second = _state_matrix(bundle, 79)
    assert first.shape == (80, 8)
    assert (abs(first - second) < 1e-10).all()
