import unittest

from uriel_v2.rng import generate_numbers, stable_seed


class RngTests(unittest.TestCase):
    def test_number_generation_is_deterministic_and_unique(self) -> None:
        first = generate_numbers(20260812)
        second = generate_numbers(20260812)

        self.assertEqual(first, second)
        self.assertEqual(first, (2, 26, 27, 29, 40, 44))
        self.assertEqual(len(first), 6)
        self.assertEqual(len(set(first)), 6)
        self.assertTrue(all(1 <= number <= 45 for number in first))

    def test_stable_seed_changes_with_namespace_and_parts(self) -> None:
        self.assertEqual(stable_seed("a", 1, 2, 3), stable_seed("a", 1, 2, 3))
        self.assertNotEqual(stable_seed("a", 1, 2, 3), stable_seed("b", 1, 2, 3))
        self.assertNotEqual(stable_seed("a", 1, 2, 3), stable_seed("a", 1, 2, 4))
