import logging
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from uriel_v2.baselines import hypergeometric_hit_probabilities, maximum_hit_probabilities
from uriel_v2.models import Draw
from uriel_v2.reverse import reverse_search
from uriel_v2.reverse_batch import _execute_round, run_reverse_batch
from uriel_v2.rng import generate_numbers


class ReverseBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = logging.getLogger("test-reverse-batch")
        cls.logger.handlers.clear()
        cls.logger.addHandler(logging.NullHandler())
        cls.draw = Draw(round_no=9999, numbers=generate_numbers(42))

    def _run_batch(self, *, chunk_size: int, workers: int):
        executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            return _execute_round(
                draw=self.draw,
                round_index=1,
                total_rounds=1,
                seed_start=0,
                seed_end=2_000,
                min_hits=4,
                top_k=20,
                chunk_size=chunk_size,
                bucket_size=500,
                budgets=(100, 1_000, 2_000),
                worker_count=workers,
                executor=executor,
                logger=self.logger,
            )
        finally:
            if executor is not None:
                executor.shutdown()

    @staticmethod
    def _signature(result):
        return (
            result.hit_distribution,
            result.best,
            result.top_k,
            tuple((point.budget, point.best) for point in result.reconstruction_curve),
        )

    def test_batch_matches_single_reverse_search(self) -> None:
        batch = self._run_batch(chunk_size=100, workers=1)
        single = reverse_search(
            target=self.draw.numbers,
            start=0,
            end=2_000,
            min_hits=4,
            chunk_size=100,
            result_limit=20,
            workers=1,
            logger=self.logger,
        )

        self.assertEqual(batch.hit_distribution, single.hit_distribution)
        self.assertEqual(batch.best, single.best[0])
        self.assertEqual(batch.top_k, single.best[:20])

    def test_worker_count_does_not_change_results(self) -> None:
        one_worker = self._run_batch(chunk_size=100, workers=1)
        four_workers = self._run_batch(chunk_size=100, workers=4)

        self.assertEqual(self._signature(one_worker), self._signature(four_workers))

    def test_chunk_size_does_not_change_results(self) -> None:
        signatures = [self._signature(self._run_batch(chunk_size=size, workers=1)) for size in (10, 25, 100)]

        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])

    def test_known_exact_seed_is_preserved(self) -> None:
        result = self._run_batch(chunk_size=100, workers=4)

        self.assertTrue(any(match.seed == 42 and match.hits == 6 for match in result.top_k))
        self.assertEqual(result.best.seed, 42)

    def test_random_baseline_probabilities_are_normalized(self) -> None:
        self.assertAlmostEqual(sum(hypergeometric_hit_probabilities()), 1.0)
        self.assertAlmostEqual(sum(maximum_hit_probabilities(1_000)), 1.0)

    def test_batch_writes_complete_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary = run_reverse_batch(
                [self.draw],
                start_round=self.draw.round_no,
                end_round=self.draw.round_no,
                seed_start=0,
                seed_end=100,
                top_k=10,
                min_hits=4,
                chunk_size=25,
                workers=1,
                bucket_size=50,
                run_dir=run_dir,
                logger=self.logger,
            )

            self.assertEqual(summary["execution"]["total_evaluated"], 100)
            for filename in summary["files"]:
                self.assertTrue((run_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
