import unittest

from uriel_v2.models import Draw
from uriel_v2.strategies import STRATEGIES, create_predictions, derive_seed


HISTORY = [
    Draw(round_no=round_no, numbers=tuple(range(offset, offset + 6)))
    for round_no, offset in enumerate(range(1, 21), start=1)
]


class StrategyTests(unittest.TestCase):
    def test_all_strategies_are_reproducible(self) -> None:
        for strategy in STRATEGIES:
            first = create_predictions(strategy, HISTORY, 21, candidates=3, history_window=10)
            second = create_predictions(strategy, HISTORY, 21, candidates=3, history_window=10)
            self.assertEqual(first, second)
            self.assertEqual(len({prediction.seed for prediction in first}), 3)

    def test_history_digest_ignores_history_outside_window(self) -> None:
        first = derive_seed("history-digest", HISTORY, 21, history_window=10)
        changed_old_history = [Draw(1, (30, 31, 32, 33, 34, 35)), *HISTORY[1:]]
        second = derive_seed("history-digest", changed_old_history, 21, history_window=10)

        self.assertEqual(first, second)
