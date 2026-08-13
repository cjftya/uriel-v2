import unittest

import numpy as np

from scripts.analyze_reverse_seed_dynamics import (
    best_mean_change,
    lz_phrase_count,
    ordinal_metrics,
    prediction_mae,
    recurrence_metrics,
    spectral_peak,
    transition_stat,
)


class ReverseSeedDynamicsTests(unittest.TestCase):
    def test_mean_change_finds_clear_boundary(self):
        values = np.r_[np.arange(24), np.arange(24) + 10_000]
        split, score = best_mean_change(values, minimum_segment=12)
        self.assertEqual(split, 24)
        self.assertGreater(score, 1.0)

    def test_transition_matrix_counts_every_transition(self):
        values = np.array([0, 250_000, 500_000, 750_000, 0], dtype=np.int64)
        matrix, statistic, persistence = transition_stat(values)
        self.assertEqual(int(matrix.sum()), len(values) - 1)
        self.assertGreaterEqual(statistic, 0.0)
        self.assertEqual(persistence, 0.0)

    def test_monotonic_sequence_has_zero_ordinal_entropy(self):
        _, entropy = ordinal_metrics(np.arange(20), order=3)
        self.assertAlmostEqual(entropy, 0.0)

    def test_recurrence_metrics_stay_in_unit_interval(self):
        metrics = recurrence_metrics(np.array([0, 10, 20, 100, 110, 120]), epsilon=15)
        for metric in metrics:
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)

    def test_spectral_peak_recovers_period_four(self):
        values = np.tile(np.array([0.0, 1.0, 0.0, -1.0]), 16)
        _, period, _ = spectral_peak(values)
        self.assertAlmostEqual(period, 4.0)

    def test_complexity_and_prediction_are_deterministic(self):
        values = np.array([(index * 73_939) % 1_000_000 for index in range(64)], dtype=np.int64)
        self.assertEqual(lz_phrase_count(values), lz_phrase_count(values.copy()))
        first = prediction_mae(values)
        second = prediction_mae(values.copy())
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"mean", "persistence", "ar3", "knn5"})


if __name__ == "__main__":
    unittest.main()
