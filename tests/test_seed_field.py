import tempfile
import unittest
from pathlib import Path

import numpy as np

from uriel_v2.seed_field import (
    SeedFieldConfig,
    SeedForecastRow,
    SeedPredictionRow,
    _best_shift,
    _best_xor_mask,
    _xor_indices,
    build_channels,
    select_top_seeds,
    transform_values,
    write_forecast_csv,
    write_prediction_csv,
)


class SeedFieldTests(unittest.TestCase):
    def test_transforms_stay_inside_twenty_bits(self):
        values = np.array([0, 1, 42, 999_999], dtype=np.uint32)
        for name in SeedFieldConfig().transforms:
            transformed = transform_values(values, name)
            self.assertEqual(len(set(map(int, transformed))), len(values))
            self.assertTrue(np.all(transformed >= 0))
            self.assertTrue(np.all(transformed < 1 << 20))

    def test_channel_mappings_cover_seed_space(self):
        config = SeedFieldConfig()
        channels = build_channels(config)
        self.assertEqual(len(channels), len(config.transforms) * len(config.resolutions) + len(config.prime_moduli))
        for channel in channels:
            self.assertEqual(channel.mapping.shape, (config.seed_space,))
            self.assertGreaterEqual(int(channel.mapping.min()), 0)
            self.assertLess(int(channel.mapping.max()), channel.size)

    def test_shift_and_xor_recover_known_operator(self):
        previous = np.zeros(64)
        previous[[1, 7, 18]] = [1.0, 0.5, 0.25]
        current = np.roll(previous, 9)
        self.assertEqual(_best_shift(previous, current), 9)

        table = _xor_indices(64)
        current_xor = previous[table[13]]
        self.assertEqual(_best_xor_mask(previous, current_xor, table), 13)

    def test_selection_is_deterministic_and_unique(self):
        score = np.zeros(1_000_000)
        score[[11, 22, 33]] = 1.0
        first = select_top_seeds(score, 1236, 100)
        second = select_top_seeds(score.copy(), 1236, 100)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(set(map(int, first))), 100)
        self.assertEqual(set(map(int, first[:3])), {11, 22, 33})

    def test_prediction_csv_contains_seed_and_numbers(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "predictions.csv"
            write_prediction_csv(
                output,
                [
                    SeedPredictionRow(
                        cohort="Development",
                        round_no=1235,
                        model="ensemble",
                        rank=1,
                        seed=42,
                        field_score=1.25,
                        numbers=(1, 2, 3, 4, 5, 6),
                        hits=2,
                    )
                ],
            )
            text = output.read_text(encoding="utf-8-sig")
            self.assertIn("seed,field_score,n1,n2,n3,n4,n5,n6,hits", text)
            self.assertIn("Development,1235,ensemble,1,42,1.25,1,2,3,4,5,6,2", text)

    def test_forecast_csv_contains_ranked_candidates(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "forecast.csv"
            write_forecast_csv(
                output,
                [
                    SeedForecastRow(
                        target_round=1236,
                        model="ensemble",
                        rank=1,
                        seed=99,
                        field_score=2.5,
                        numbers=(7, 8, 9, 10, 11, 12),
                    )
                ],
            )
            text = output.read_text(encoding="utf-8-sig")
            self.assertIn("target_round,model,rank,seed,field_score", text)
            self.assertIn("1236,ensemble,1,99,2.5,7,8,9,10,11,12", text)


if __name__ == "__main__":
    unittest.main()
