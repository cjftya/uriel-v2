import math
import random
import unittest

from uriel_v2.combinadic_rank import (
    TOTAL_COMBINATIONS,
    circular_rank_distance,
    combination_to_rank,
    forecast_rank,
    rank_to_combination,
    rank_window_candidates,
)


class CombinadicRankTests(unittest.TestCase):
    def test_boundaries_have_expected_lexicographic_ranks(self) -> None:
        self.assertEqual(combination_to_rank((1, 2, 3, 4, 5, 6)), 0)
        self.assertEqual(combination_to_rank((40, 41, 42, 43, 44, 45)), TOTAL_COMBINATIONS - 1)
        self.assertEqual(TOTAL_COMBINATIONS, math.comb(45, 6))

    def test_encode_decode_round_trip(self) -> None:
        rng = random.Random(20260814)
        cases = [0, 1, TOTAL_COMBINATIONS // 2, TOTAL_COMBINATIONS - 2, TOTAL_COMBINATIONS - 1]
        cases.extend(rng.randrange(TOTAL_COMBINATIONS) for _ in range(100))
        for rank in cases:
            combination = rank_to_combination(rank)
            self.assertEqual(combination_to_rank(combination), rank)

    def test_circular_distance_wraps_at_space_boundary(self) -> None:
        self.assertEqual(circular_rank_distance(0, TOTAL_COMBINATIONS - 1), 1)
        self.assertEqual(circular_rank_distance(10, 25), 15)

    def test_candidate_window_is_deterministic_unique_and_budgeted(self) -> None:
        first = rank_window_candidates((100, 200), 1_000)
        second = rank_window_candidates((100, 200), 1_000)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1_000)
        self.assertEqual(len(set(first)), 1_000)

    def test_forecast_uses_only_supplied_history(self) -> None:
        history = tuple((index * 73_939 + index * index * 17) % TOTAL_COMBINATIONS for index in range(80))
        first = forecast_rank(history)
        second = forecast_rank((*history, 12345))
        self.assertNotEqual(first, second)
        self.assertEqual(first, forecast_rank(history))


if __name__ == "__main__":
    unittest.main()
