from pathlib import Path
import unittest

from uriel_v2.data import find_draw, load_draws


DATA = Path(__file__).resolve().parents[1] / "lotto.xlsx"


class DataTests(unittest.TestCase):
    def test_loads_repository_workbook_in_ascending_round_order(self) -> None:
        draws = load_draws(DATA)

        self.assertEqual(len(draws), 1235)
        self.assertEqual([draw.round_no for draw in draws], list(range(1, 1236)))
        self.assertEqual(draws[0].round_no, 1)
        self.assertEqual(draws[-1].round_no, 1235)
        self.assertEqual(draws[-1].numbers, (6, 7, 11, 15, 39, 43))
        self.assertEqual(draws[-1].bonus, 20)

    def test_find_draw(self) -> None:
        draw = find_draw(load_draws(DATA), 1234)

        self.assertEqual(draw.numbers, (1, 15, 19, 31, 35, 43))
