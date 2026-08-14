from __future__ import annotations

import numpy as np

from uriel_v2.models import Draw
from uriel_v2.motif_features import VIEW_NAMES, build_feature_bundle, prefix_standardize


def _draws() -> list[Draw]:
    return [
        Draw(1, (1, 8, 15, 22, 29, 36)),
        Draw(2, (2, 9, 16, 23, 30, 37)),
        Draw(3, (3, 10, 17, 24, 31, 38)),
        Draw(4, (4, 11, 18, 25, 32, 39)),
    ]


def test_feature_bundle_has_all_views_and_masks() -> None:
    bundle = build_feature_bundle(_draws())
    assert tuple(bundle.views) == VIEW_NAMES
    assert bundle.grid_masks.shape == (4, 49)
    assert np.all(bundle.grid_masks.sum(axis=1) == 6)
    assert all(matrix.shape[0] == 4 for matrix in bundle.views.values())
    assert not bundle.frame().isna().any().any()


def test_prefix_standardize_never_reads_future_rows() -> None:
    matrix = np.asarray([[1.0], [2.0], [3.0], [1000.0]])
    standardized = prefix_standardize(matrix, 2)
    assert standardized.shape == (3, 1)
    assert np.isclose(float(standardized.mean()), 0.0)
    assert np.allclose(standardized.ravel(), [-1.22474487, 0.0, 1.22474487])
