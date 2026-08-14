import tempfile
import unittest
from pathlib import Path

from uriel_v2.seed_basin import (
    BasinPoint,
    BasinSummary,
    canonical_generator_hash,
    forecast_basin,
    load_landscapes,
    seed_window_candidates,
    summarize_basin,
)


class SeedBasinTests(unittest.TestCase):
    def test_summary_preserves_exact_seeds_and_density(self) -> None:
        points = (
            BasinPoint(100, 4, 1.0),
            BasinPoint(200, 5, 0.5),
            BasinPoint(300, 6, 0.0),
            BasinPoint(900_000, 4, 2.0),
        )
        summary = summarize_basin(1, points)
        self.assertEqual(summary.exact_seeds, (300,))
        self.assertEqual(summary.max_hit, 6)
        self.assertAlmostEqual(summary.density_4_plus, 4 / 1_000_000)
        self.assertAlmostEqual(summary.density_5_plus, 2 / 1_000_000)

    def test_landscape_loader_deduplicates_round_seed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "landscape.csv"
            path.write_text(
                "round,seed,hits,positional_mae\n1,42,4,1.0\n1,42,5,0.5\n1,99,4,2.0\n",
                encoding="utf-8",
            )
            loaded = load_landscapes([path])
        self.assertEqual(len(loaded[1]), 2)
        self.assertEqual(loaded[1][0], BasinPoint(42, 5, 0.5))

    def test_forecast_is_deterministic(self) -> None:
        history = []
        for round_no in range(1, 41):
            center = float(100_000 + round_no * 1_000)
            history.append(
                BasinSummary(
                    round_no=round_no,
                    center=center,
                    weighted_center=center,
                    width=20_000.0,
                    density_4_plus=0.001,
                    density_5_plus=0.00001,
                    exact_6_count=0,
                    mean_hit=4.1,
                    max_hit=5,
                    entropy=0.9,
                    asymmetry=0.1,
                    nearest_5_distance=100.0,
                    nearest_4_distance=10.0,
                    exact_seeds=(),
                    scale_centers=(center, center, center, center),
                )
            )
        first = forecast_basin(history)
        self.assertEqual(first, forecast_basin(list(history)))
        self.assertEqual(first.delta_center, 141_000)

    def test_seed_window_is_unique_and_in_range(self) -> None:
        candidates = seed_window_candidates((0, 999_999), 10_000)
        self.assertEqual(len(candidates), 10_000)
        self.assertEqual(len(set(candidates)), 10_000)
        self.assertGreaterEqual(min(candidates), 0)
        self.assertLess(max(candidates), 1_000_000)

    def test_generator_hash_is_stable_sha256(self) -> None:
        value = canonical_generator_hash()
        self.assertEqual(len(value), 64)
        self.assertEqual(value, canonical_generator_hash())


if __name__ == "__main__":
    unittest.main()
