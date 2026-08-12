import logging
import unittest

from uriel_v2.models import SearchChunkTask
from uriel_v2.reverse import reverse_search, search_chunk
from uriel_v2.rng import generate_numbers


class ReverseTests(unittest.TestCase):
    def test_search_chunk_distribution_covers_every_seed(self) -> None:
        result = search_chunk(SearchChunkTask(start=0, end=100, target=(1, 2, 3, 4, 5, 6), min_hits=6, result_limit=10))

        self.assertEqual(sum(result.hit_distribution), 100)
        self.assertEqual(result.end - result.start, 100)
        self.assertTrue(result.best)

    def test_reverse_search_finds_known_exact_seed(self) -> None:
        target = generate_numbers(42)
        logger = logging.getLogger("test-reverse")
        logger.addHandler(logging.NullHandler())

        result = reverse_search(
            target=target,
            start=0,
            end=100,
            min_hits=6,
            chunk_size=20,
            result_limit=10,
            workers=2,
            logger=logger,
        )

        self.assertTrue(any(match.seed == 42 and match.numbers == target for match in result.matches))
        self.assertEqual(sum(result.hit_distribution), 100)
