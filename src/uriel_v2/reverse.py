from __future__ import annotations

import heapq
import logging
import math
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from collections.abc import Iterator

from uriel_v2.evaluation import resolve_workers
from uriel_v2.metrics import positional_deviations
from uriel_v2.models import ReverseMatch, ReverseSearchResult, SearchChunkResult, SearchChunkTask
from uriel_v2.rng import generate_numbers, numbers_mask


RankedItem = tuple[int, float, int, tuple[int, ...]]


def _add_ranked(heap: list[RankedItem], item: RankedItem, limit: int) -> None:
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _to_matches(items: list[RankedItem], target: tuple[int, ...]) -> tuple[ReverseMatch, ...]:
    return tuple(
        _make_match(
            seed=-negative_seed,
            numbers=numbers,
            hits=hits,
            positional_mae=-negative_mae,
            target=target,
        )
        for hits, negative_mae, negative_seed, numbers in sorted(items, reverse=True)
    )


def _make_match(
    *,
    seed: int,
    numbers: tuple[int, ...],
    hits: int,
    positional_mae: float,
    target: tuple[int, ...],
) -> ReverseMatch:
    deviations = positional_deviations(numbers, target)
    return ReverseMatch(
        seed=seed,
        numbers=numbers,
        hits=hits,
        positional_mae=positional_mae,
        set_distance=12 - (2 * hits),
        signed_bias=sum(deviations) / len(deviations),
        deviations=deviations,
    )


def search_chunk(task: SearchChunkTask) -> SearchChunkResult:
    started = time.perf_counter()
    target_mask = numbers_mask(task.target)
    distribution = [0] * 7
    matches: list[RankedItem] = []
    all_matches: list[RankedItem] = []
    best: list[RankedItem] = []

    for seed in range(task.start, task.end):
        numbers = generate_numbers(seed)
        hits = (numbers_mask(numbers) & target_mask).bit_count()
        distribution[hits] += 1
        deviations = positional_deviations(numbers, task.target)
        mae = sum(abs(value) for value in deviations) / len(deviations)
        ranked = (hits, -mae, -seed, numbers)
        _add_ranked(best, ranked, task.result_limit)
        if hits >= task.min_hits:
            if task.collect_all_matches:
                all_matches.append(ranked)
            else:
                _add_ranked(matches, ranked, task.result_limit)

    return SearchChunkResult(
        start=task.start,
        end=task.end,
        elapsed_seconds=time.perf_counter() - started,
        hit_distribution=tuple(distribution),
        matches=_to_matches(all_matches if task.collect_all_matches else matches, task.target),
        best=_to_matches(best, task.target),
    )


def reverse_search(
    *,
    target: tuple[int, ...],
    start: int,
    end: int,
    min_hits: int,
    chunk_size: int,
    result_limit: int,
    workers: str | int,
    logger: logging.Logger,
) -> ReverseSearchResult:
    if start < 0 or end <= start:
        raise ValueError("시드 범위는 0 <= start < end여야 합니다")
    if min_hits < 0 or min_hits > 6:
        raise ValueError("min-hits는 0~6이어야 합니다")
    if chunk_size <= 0 or result_limit <= 0:
        raise ValueError("chunk-size와 result-limit은 양수여야 합니다")

    chunk_count = math.ceil((end - start) / chunk_size)

    def iter_tasks() -> Iterator[SearchChunkTask]:
        return (
            SearchChunkTask(
            start=chunk_start,
            end=min(chunk_start + chunk_size, end),
            target=target,
            min_hits=min_hits,
            result_limit=result_limit,
            )
            for chunk_start in range(start, end, chunk_size)
        )

    worker_count = min(resolve_workers(workers), chunk_count)
    logger.warning("역산 탐색은 정답을 사용하는 진단 실험이며 미래 회차 예측 성능이 아닙니다.")
    logger.info(
        "역산 시작 | 정답=%s | 시드=[%s, %s) | 총=%s | 최소적중=%s | 청크=%s | 워커=%s",
        target,
        f"{start:,}",
        f"{end:,}",
        f"{end - start:,}",
        min_hits,
        f"{chunk_size:,}",
        worker_count,
    )

    started = time.perf_counter()
    distribution = [0] * 7
    match_heap: list[RankedItem] = []
    best_heap: list[RankedItem] = []
    progress_step = max(1, chunk_count // 20)

    def absorb(result: SearchChunkResult) -> None:
        for index, count in enumerate(result.hit_distribution):
            distribution[index] += count
        for match in result.matches:
            _add_ranked(match_heap, (match.hits, -match.positional_mae, -match.seed, match.numbers), result_limit)
        for match in result.best:
            _add_ranked(best_heap, (match.hits, -match.positional_mae, -match.seed, match.numbers), 20)

    if worker_count == 1:
        for completed, task in enumerate(iter_tasks(), start=1):
            absorb(search_chunk(task))
            if completed % progress_step == 0 or completed == chunk_count:
                scanned = min(completed * chunk_size, end - start)
                logger.info("역산 진행 | %s/%s 청크 | %.1f%% | 탐색=%s", completed, chunk_count, scanned * 100 / (end - start), f"{scanned:,}")
    else:
        executor = ProcessPoolExecutor(max_workers=worker_count)
        pending: set[Future[SearchChunkResult]] = set()
        try:
            task_iterator = iter(iter_tasks())
            for _ in range(min(chunk_count, worker_count * 2)):
                pending.add(executor.submit(search_chunk, next(task_iterator)))
            scanned = 0
            completed = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    scanned += result.end - result.start
                    completed += 1
                    absorb(result)
                    try:
                        pending.add(executor.submit(search_chunk, next(task_iterator)))
                    except StopIteration:
                        pass
                    if completed % progress_step == 0 or completed == chunk_count:
                        logger.info("역산 진행 | %s/%s 청크 | %.1f%% | 탐색=%s", completed, chunk_count, scanned * 100 / (end - start), f"{scanned:,}")
        except BaseException:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()

    return ReverseSearchResult(
        start=start,
        end=end,
        elapsed_seconds=time.perf_counter() - started,
        hit_distribution=tuple(distribution),
        matches=_to_matches(match_heap, target),
        best=_to_matches(best_heap, target),
        chunks=chunk_count,
        workers=worker_count,
    )
