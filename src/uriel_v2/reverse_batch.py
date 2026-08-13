from __future__ import annotations

import csv
import heapq
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence, TextIO

from uriel_v2.baselines import (
    at_least_one_probability,
    hypergeometric_hit_probabilities,
    maximum_hit_probabilities,
    monte_carlo_round_max_baseline,
)
from uriel_v2.charts import write_budget_bar_chart_svg
from uriel_v2.evaluation import resolve_workers
from uriel_v2.models import Draw, ReverseMatch, SearchChunkResult, SearchChunkTask
from uriel_v2.reverse import search_chunk


RankedMatch = tuple[int, float, int, tuple[int, ...], ReverseMatch]


@dataclass(frozen=True, slots=True)
class ReconstructionPoint:
    budget: int
    seed_end: int
    best: ReverseMatch


@dataclass(frozen=True, slots=True)
class SeedBucket:
    start: int
    end: int
    evaluated: int
    hit_4_count: int
    hit_5_count: int
    hit_6_count: int
    best: ReverseMatch


@dataclass(frozen=True, slots=True)
class BatchRoundResult:
    round_no: int
    target: tuple[int, ...]
    seed_start: int
    seed_end: int
    elapsed_seconds: float
    workers: int
    chunks: int
    hit_distribution: tuple[int, ...]
    best: ReverseMatch
    top_k: tuple[ReverseMatch, ...]
    qualifying: tuple[ReverseMatch, ...]
    buckets: tuple[SeedBucket, ...]
    reconstruction_curve: tuple[ReconstructionPoint, ...]


def _ranked(match: ReverseMatch) -> RankedMatch:
    return (match.hits, -match.positional_mae, -match.seed, match.numbers, match)


def _add_ranked(heap: list[RankedMatch], match: ReverseMatch, limit: int) -> None:
    item = _ranked(match)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _ordered(heap: list[RankedMatch]) -> tuple[ReverseMatch, ...]:
    return tuple(item[-1] for item in sorted(heap, reverse=True))


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _segment_tasks(
    *,
    target: tuple[int, ...],
    seed_start: int,
    seed_end: int,
    min_hits: int,
    top_k: int,
    chunk_size: int,
    bucket_size: int,
    budgets: Sequence[int],
) -> list[SearchChunkTask]:
    cuts = {seed_start, seed_end}
    cuts.update(range(seed_start + chunk_size, seed_end, chunk_size))
    cuts.update(range(seed_start + bucket_size, seed_end, bucket_size))
    cuts.update(seed_start + budget for budget in budgets if 0 < budget < seed_end - seed_start)
    ordered = sorted(cuts)
    return [
        SearchChunkTask(
            start=start,
            end=end,
            target=target,
            min_hits=min_hits,
            result_limit=top_k,
            collect_all_matches=True,
        )
        for start, end in zip(ordered, ordered[1:])
    ]


def _absorb_live(
    result: SearchChunkResult,
    distribution: list[int],
    best_heap: list[RankedMatch],
) -> None:
    for hits, count in enumerate(result.hit_distribution):
        distribution[hits] += count
    for match in result.best:
        _add_ranked(best_heap, match, 1)


def _execute_round(
    *,
    draw: Draw,
    round_index: int,
    total_rounds: int,
    seed_start: int,
    seed_end: int,
    min_hits: int,
    top_k: int,
    chunk_size: int,
    bucket_size: int,
    budgets: Sequence[int],
    worker_count: int,
    executor: ProcessPoolExecutor | None,
    logger: logging.Logger,
) -> BatchRoundResult:
    tasks = _segment_tasks(
        target=draw.numbers,
        seed_start=seed_start,
        seed_end=seed_end,
        min_hits=min_hits,
        top_k=top_k,
        chunk_size=chunk_size,
        bucket_size=bucket_size,
        budgets=budgets,
    )
    started = time.perf_counter()
    live_distribution = [0] * 7
    live_best: list[RankedMatch] = []
    results: list[SearchChunkResult] = []
    evaluated = 0
    progress_step = max(1, len(tasks) // 10)

    def log_progress(completed: int) -> None:
        elapsed = time.perf_counter() - started
        speed = evaluated / elapsed if elapsed else 0.0
        remaining = (seed_end - seed_start) - evaluated
        eta = remaining / speed if speed else math.inf
        best = _ordered(live_best)[0]
        logger.info(
            "[Reverse Batch] 회차=%s | 진행=%s/%s | 탐색=%s/%s (%.1f%%) | 최고=%s seed=%s | 5-hit=%s | 6-hit=%s | 속도=%s seeds/s | ETA=%s | 경과=%s",
            draw.round_no,
            round_index,
            total_rounds,
            f"{evaluated:,}",
            f"{seed_end - seed_start:,}",
            evaluated * 100 / (seed_end - seed_start),
            best.hits,
            f"{best.seed:,}",
            f"{live_distribution[5]:,}",
            f"{live_distribution[6]:,}",
            f"{speed:,.0f}",
            _format_duration(eta) if math.isfinite(eta) else "--:--",
            _format_duration(elapsed),
        )

    def absorb(result: SearchChunkResult, completed: int) -> None:
        nonlocal evaluated
        results.append(result)
        evaluated += result.end - result.start
        _absorb_live(result, live_distribution, live_best)
        if completed % progress_step == 0 or completed == len(tasks):
            log_progress(completed)

    if worker_count == 1:
        for completed, task in enumerate(tasks, start=1):
            absorb(search_chunk(task), completed)
    else:
        if executor is None:
            raise RuntimeError("병렬 실행기 초기화에 실패했습니다")
        pending: set[Future[SearchChunkResult]] = set()
        task_iterator = iter(tasks)
        try:
            for _ in range(min(len(tasks), worker_count * 2)):
                pending.add(executor.submit(search_chunk, next(task_iterator)))
            completed = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    completed += 1
                    absorb(future.result(), completed)
                    try:
                        pending.add(executor.submit(search_chunk, next(task_iterator)))
                    except StopIteration:
                        pass
        except BaseException:
            for future in pending:
                future.cancel()
            raise

    results.sort(key=lambda result: result.start)
    top_heap: list[RankedMatch] = []
    qualifying: list[ReverseMatch] = []
    distribution = [0] * 7
    cumulative_best: list[RankedMatch] = []
    reconstruction: list[ReconstructionPoint] = []
    endpoints = {seed_start + budget: budget for budget in budgets}

    bucket_state: dict[int, dict[str, Any]] = {}
    for result in results:
        for hits, count in enumerate(result.hit_distribution):
            distribution[hits] += count
        for match in result.best:
            _add_ranked(top_heap, match, top_k)
            _add_ranked(cumulative_best, match, 1)
        qualifying.extend(result.matches)

        bucket_index = (result.start - seed_start) // bucket_size
        bucket_start = seed_start + bucket_index * bucket_size
        state = bucket_state.setdefault(
            bucket_index,
            {
                "start": bucket_start,
                "end": min(bucket_start + bucket_size, seed_end),
                "distribution": [0] * 7,
                "best": [],
            },
        )
        for hits, count in enumerate(result.hit_distribution):
            state["distribution"][hits] += count
        for match in result.best:
            _add_ranked(state["best"], match, 1)

        if result.end in endpoints:
            reconstruction.append(
                ReconstructionPoint(
                    budget=endpoints[result.end],
                    seed_end=result.end,
                    best=_ordered(cumulative_best)[0],
                )
            )

    ordered_top = _ordered(top_heap)
    buckets = tuple(
        SeedBucket(
            start=state["start"],
            end=state["end"],
            evaluated=sum(state["distribution"]),
            hit_4_count=state["distribution"][4],
            hit_5_count=state["distribution"][5],
            hit_6_count=state["distribution"][6],
            best=_ordered(state["best"])[0],
        )
        for _, state in sorted(bucket_state.items())
    )
    qualifying.sort(key=lambda match: (-match.hits, match.positional_mae, match.set_distance, match.seed))
    reconstruction.sort(key=lambda point: point.budget)
    return BatchRoundResult(
        round_no=draw.round_no,
        target=draw.numbers,
        seed_start=seed_start,
        seed_end=seed_end,
        elapsed_seconds=time.perf_counter() - started,
        workers=worker_count,
        chunks=len(tasks),
        hit_distribution=tuple(distribution),
        best=ordered_top[0],
        top_k=ordered_top,
        qualifying=tuple(qualifying),
        buckets=buckets,
        reconstruction_curve=tuple(reconstruction),
    )


class _DatasetWriters:
    def __init__(self, run_dir: Path) -> None:
        self._handles: list[TextIO] = []
        self.rounds = self._writer(run_dir / "reverse-rounds.csv", [
            "round", "winner", "best_seed", "best_numbers", "best_hits", "best_positional_mae",
            "best_set_distance", "best_signed_bias", "hit_0_count", "hit_1_count", "hit_2_count",
            "hit_3_count", "hit_4_count", "hit_5_count", "hit_6_count", "seed_start", "seed_end",
            "evaluated", "elapsed_seconds", "seeds_per_second", "workers", "chunks",
        ])
        match_header = [
            "round", "seed", "hits", "positional_mae", "set_distance", "signed_bias",
            "n1", "n2", "n3", "n4", "n5", "n6", "rank",
        ]
        self.top_k = self._writer(run_dir / "reverse-top-k.csv", match_header)
        self.hit_seeds = self._writer(run_dir / "reverse-hit-seeds.csv", match_header)
        self.buckets = self._writer(run_dir / "reverse-seed-buckets.csv", [
            "round", "bucket_start", "bucket_end", "evaluated", "best_seed", "best_hits",
            "best_positional_mae", "hit_4_count", "hit_5_count", "hit_6_count", "density_4_plus_per_100k",
        ])
        self.curve = self._writer(run_dir / "reverse-reconstruction-curve.csv", [
            "round", "budget", "seed_end", "best_seed", "best_hits", "best_positional_mae",
            "is_4_plus", "is_5_plus", "is_exact_6",
        ])

    def _writer(self, path: Path, header: list[str]) -> csv.writer:
        handle = path.open("w", newline="", encoding="utf-8-sig")
        self._handles.append(handle)
        writer = csv.writer(handle)
        writer.writerow(header)
        return writer

    @staticmethod
    def _match_row(round_no: int, match: ReverseMatch, rank: int) -> list[Any]:
        return [
            round_no, match.seed, match.hits, f"{match.positional_mae:.6f}", match.set_distance,
            f"{match.signed_bias:.6f}", *match.numbers, rank,
        ]

    def write(self, result: BatchRoundResult) -> None:
        evaluated = result.seed_end - result.seed_start
        self.rounds.writerow([
            result.round_no,
            "-".join(map(str, result.target)),
            result.best.seed,
            "-".join(map(str, result.best.numbers)),
            result.best.hits,
            f"{result.best.positional_mae:.6f}",
            result.best.set_distance,
            f"{result.best.signed_bias:.6f}",
            *result.hit_distribution,
            result.seed_start,
            result.seed_end,
            evaluated,
            f"{result.elapsed_seconds:.6f}",
            f"{evaluated / result.elapsed_seconds:.3f}",
            result.workers,
            result.chunks,
        ])
        for rank, match in enumerate(result.top_k, start=1):
            self.top_k.writerow(self._match_row(result.round_no, match, rank))
        for rank, match in enumerate(result.qualifying, start=1):
            self.hit_seeds.writerow(self._match_row(result.round_no, match, rank))
        for bucket in result.buckets:
            self.buckets.writerow([
                result.round_no,
                bucket.start,
                bucket.end,
                bucket.evaluated,
                bucket.best.seed,
                bucket.best.hits,
                f"{bucket.best.positional_mae:.6f}",
                bucket.hit_4_count,
                bucket.hit_5_count,
                bucket.hit_6_count,
                f"{(bucket.hit_4_count + bucket.hit_5_count + bucket.hit_6_count) * 100_000 / bucket.evaluated:.6f}",
            ])
        for point in result.reconstruction_curve:
            self.curve.writerow([
                result.round_no,
                point.budget,
                point.seed_end,
                point.best.seed,
                point.best.hits,
                f"{point.best.positional_mae:.6f}",
                int(point.best.hits >= 4),
                int(point.best.hits >= 5),
                int(point.best.hits == 6),
            ])
        self.flush()

    def flush(self) -> None:
        for handle in self._handles:
            handle.flush()
            os.fsync(handle.fileno())

    def close(self) -> None:
        for handle in self._handles:
            handle.close()


def _csv_data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _validate_dataset_files(
    run_dir: Path,
    *,
    round_count: int,
    budget: int,
    top_k: int,
    min_hits: int,
    bucket_size: int,
    reconstruction_budget_count: int,
    candidate_hit_distribution: Sequence[int],
) -> dict[str, int]:
    actual = {
        "reverse-rounds.csv": _csv_data_rows(run_dir / "reverse-rounds.csv"),
        "reverse-top-k.csv": _csv_data_rows(run_dir / "reverse-top-k.csv"),
        "reverse-hit-seeds.csv": _csv_data_rows(run_dir / "reverse-hit-seeds.csv"),
        "reverse-seed-buckets.csv": _csv_data_rows(run_dir / "reverse-seed-buckets.csv"),
        "reverse-reconstruction-curve.csv": _csv_data_rows(run_dir / "reverse-reconstruction-curve.csv"),
    }
    expected = {
        "reverse-rounds.csv": round_count,
        "reverse-top-k.csv": round_count * min(top_k, budget),
        "reverse-hit-seeds.csv": sum(candidate_hit_distribution[min_hits:]),
        "reverse-seed-buckets.csv": round_count * math.ceil(budget / bucket_size),
        "reverse-reconstruction-curve.csv": round_count * reconstruction_budget_count,
    }
    if actual != expected:
        raise RuntimeError(f"reverse dataset 행 수 검증 실패 | expected={expected} | actual={actual}")
    return actual


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _quantile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _curve_summary(points: dict[int, list[ReconstructionPoint]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget, selected in sorted(points.items()):
        rows.append({
            "budget": budget,
            "rounds": len(selected),
            "exact_6_rounds": sum(point.best.hits == 6 for point in selected),
            "exact_6_rate": sum(point.best.hits == 6 for point in selected) / len(selected),
            "hit_5_plus_rounds": sum(point.best.hits >= 5 for point in selected),
            "hit_5_plus_rate": sum(point.best.hits >= 5 for point in selected) / len(selected),
            "hit_4_plus_rounds": sum(point.best.hits >= 4 for point in selected),
            "hit_4_plus_rate": sum(point.best.hits >= 4 for point in selected) / len(selected),
            "mean_best_hit": mean(point.best.hits for point in selected),
            "median_best_positional_mae": median(point.best.positional_mae for point in selected),
        })
    return rows


def _write_curve_charts(run_dir: Path, curve: Sequence[dict[str, Any]], round_count: int) -> list[str]:
    charts = [
        ("curve-exact-6.svg", "Exact 6 reconstruction by seed budget", "exact_6_rate", "Round reconstruction rate", ".1%", 1.0),
        ("curve-hit-5-plus.svg", "5+ reconstruction by seed budget", "hit_5_plus_rate", "Round reconstruction rate", ".1%", 1.0),
        ("curve-mean-best-hit.svg", "Mean best hit by seed budget", "mean_best_hit", "Mean best hit", ".2f", 6.0),
        ("curve-median-best-mae.svg", "Median best positional MAE by seed budget", "median_best_positional_mae", "Median positional MAE", ".2f", None),
    ]
    written: list[str] = []
    budgets = [row["budget"] for row in curve]
    for filename, title, field, y_label, value_format, y_max in charts:
        write_budget_bar_chart_svg(
            run_dir / filename,
            title=title,
            subtitle=f"Answer-derived reverse reconstruction; {round_count} rounds; equal budget per round",
            budgets=budgets,
            values=[row[field] for row in curve],
            y_label=y_label,
            value_format=value_format,
            y_min=0.0,
            y_max=y_max,
        )
        written.append(filename)
    return written


def run_reverse_batch(
    draws: Sequence[Draw],
    *,
    start_round: int,
    end_round: int,
    seed_start: int,
    seed_end: int,
    top_k: int,
    min_hits: int,
    chunk_size: int,
    workers: str | int,
    bucket_size: int,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    if start_round <= 0 or end_round < start_round:
        raise ValueError("회차 범위가 올바르지 않습니다")
    if seed_start < 0 or seed_end <= seed_start:
        raise ValueError("시드 범위는 0 <= start < end여야 합니다")
    if top_k <= 0 or chunk_size <= 0 or bucket_size <= 0:
        raise ValueError("top-k, chunk-size, bucket-size는 양수여야 합니다")
    if min_hits < 0 or min_hits > 6:
        raise ValueError("min-hits는 0~6이어야 합니다")

    by_round = {draw.round_no: draw for draw in draws}
    missing = [round_no for round_no in range(start_round, end_round + 1) if round_no not in by_round]
    if missing:
        raise ValueError(f"요청한 회차 데이터가 없습니다: {missing[:10]}")
    selected = [by_round[round_no] for round_no in range(start_round, end_round + 1)]
    budget = seed_end - seed_start
    budgets = tuple(dict.fromkeys(min(value, budget) for value in (10_000, 100_000, budget)))
    segment_count = len(_segment_tasks(
        target=selected[0].numbers,
        seed_start=seed_start,
        seed_end=seed_end,
        min_hits=min_hits,
        top_k=top_k,
        chunk_size=chunk_size,
        bucket_size=bucket_size,
        budgets=budgets,
    ))
    worker_count = min(resolve_workers(workers), segment_count)

    logger.warning("정답 기반 reverse reconstruction입니다. 미래 회차 예측 성능으로 해석할 수 없습니다.")
    logger.info(
        "Reverse batch 시작 | 회차=%s~%s (%s회) | 시드=[%s, %s) | 회차당=%s | 총평가=%s | Top-K=%s | 최소적중=%s | 청크=%s | 버킷=%s | 워커=%s",
        start_round,
        end_round,
        len(selected),
        f"{seed_start:,}",
        f"{seed_end:,}",
        f"{budget:,}",
        f"{budget * len(selected):,}",
        top_k,
        min_hits,
        f"{chunk_size:,}",
        f"{bucket_size:,}",
        worker_count,
    )

    writers = _DatasetWriters(run_dir)
    executor = ProcessPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    wall_started = time.perf_counter()
    completed_rounds: list[dict[str, Any]] = []
    best_hit_distribution: Counter[int] = Counter()
    candidate_hit_distribution = [0] * 7
    curve_points: dict[int, list[ReconstructionPoint]] = defaultdict(list)
    seeds_by_hit: dict[int, list[int]] = {4: [], 5: [], 6: []}
    bucket_totals: dict[int, Counter[int]] = defaultdict(Counter)
    top_k_seeds: list[int] = []

    try:
        for index, draw in enumerate(selected, start=1):
            result = _execute_round(
                draw=draw,
                round_index=index,
                total_rounds=len(selected),
                seed_start=seed_start,
                seed_end=seed_end,
                min_hits=min_hits,
                top_k=top_k,
                chunk_size=chunk_size,
                bucket_size=bucket_size,
                budgets=budgets,
                worker_count=worker_count,
                executor=executor,
                logger=logger,
            )
            writers.write(result)
            best_hit_distribution[result.best.hits] += 1
            for hits, count in enumerate(result.hit_distribution):
                candidate_hit_distribution[hits] += count
            for point in result.reconstruction_curve:
                curve_points[point.budget].append(point)
            for match in result.qualifying:
                if match.hits in seeds_by_hit:
                    seeds_by_hit[match.hits].append(match.seed)
            for bucket in result.buckets:
                bucket_totals[bucket.start].update({4: bucket.hit_4_count, 5: bucket.hit_5_count, 6: bucket.hit_6_count})
            top_k_seeds.extend(match.seed for match in result.top_k)

            elapsed = time.perf_counter() - wall_started
            completed_rounds.append({
                "round": result.round_no,
                "target": list(result.target),
                "best": asdict(result.best),
                "hit_distribution": {str(hit): count for hit, count in enumerate(result.hit_distribution)},
                "hit_4_count": result.hit_distribution[4],
                "hit_5_count": result.hit_distribution[5],
                "hit_6_count": result.hit_distribution[6],
                "elapsed_seconds": result.elapsed_seconds,
                "seeds_per_second": budget / result.elapsed_seconds,
            })
            _atomic_json(run_dir / "reverse-progress.json", {
                "status": "running",
                "completed_rounds": [row["round"] for row in completed_rounds],
                "total_rounds": len(selected),
                "evaluated": len(completed_rounds) * budget,
                "elapsed_seconds": elapsed,
            })
            logger.info(
                "회차 완료 | 회차=%s | 진행=%s/%s | 최고=%s seed=%s MAE=%.3f | 4/5/6=%s/%s/%s | 회차속도=%s | 전체속도=%s | 전체경과=%s",
                result.round_no,
                index,
                len(selected),
                result.best.hits,
                f"{result.best.seed:,}",
                result.best.positional_mae,
                f"{result.hit_distribution[4]:,}",
                f"{result.hit_distribution[5]:,}",
                f"{result.hit_distribution[6]:,}",
                f"{budget / result.elapsed_seconds:,.0f}",
                f"{index * budget / elapsed:,.0f}",
                _format_duration(elapsed),
            )
    finally:
        writers.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    wall_elapsed = time.perf_counter() - wall_started
    total_evaluated = budget * len(selected)
    dataset_rows = _validate_dataset_files(
        run_dir,
        round_count=len(selected),
        budget=budget,
        top_k=top_k,
        min_hits=min_hits,
        bucket_size=bucket_size,
        reconstruction_budget_count=len(budgets),
        candidate_hit_distribution=candidate_hit_distribution,
    )
    curve = _curve_summary(curve_points)
    single_probabilities = hypergeometric_hit_probabilities()
    maximum_probabilities = maximum_hit_probabilities(budget)
    expected_counts = {
        str(hits): single_probabilities[hits] * total_evaluated
        for hits in range(7)
    }
    observed_vs_expected = {
        str(hits): {
            "observed": candidate_hit_distribution[hits],
            "expected": expected_counts[str(hits)],
            "ratio": candidate_hit_distribution[hits] / expected_counts[str(hits)],
        }
        for hits in (4, 5, 6)
    }

    landscape_levels: dict[str, Any] = {}
    for hits, seeds in seeds_by_hit.items():
        landscape_levels[str(hits)] = {
            "count": len(seeds),
            "unique_seed_count": len(set(seeds)),
            "min_seed": min(seeds) if seeds else None,
            "max_seed": max(seeds) if seeds else None,
            "median_seed": median(seeds) if seeds else None,
            "density_per_100k": len(seeds) * 100_000 / total_evaluated,
        }
    total_high_hits = sum(candidate_hit_distribution[4:7])
    landscape_buckets = []
    for bucket_start, counts in sorted(bucket_totals.items()):
        bucket_end = min(bucket_start + bucket_size, seed_end)
        evaluated_in_bucket = (bucket_end - bucket_start) * len(selected)
        high_hits = sum(counts[hit] for hit in (4, 5, 6))
        landscape_buckets.append({
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "hit_4_count": counts[4],
            "hit_5_count": counts[5],
            "hit_6_count": counts[6],
            "hit_4_plus_count": high_hits,
            "density_per_100k": high_hits * 100_000 / evaluated_in_bucket,
            "share_of_all_4_plus": high_hits / total_high_hits if total_high_hits else 0.0,
        })

    exact_rounds = [row["round"] for row in completed_rounds if row["best"]["hits"] == 6]
    highest_hit_5 = sorted(completed_rounds, key=lambda row: (-row["hit_5_count"], row["round"]))[:10]
    lowest_best_mae = sorted(completed_rounds, key=lambda row: (row["best"]["positional_mae"], row["round"]))[:10]
    summary: dict[str, Any] = {
        "experiment": "answer-derived reverse seed reconstruction Stage A",
        "warning": "Known winning numbers were used. Results measure reconstruction, not forward prediction.",
        "status": "completed",
        "config": {
            "start_round": start_round,
            "end_round": end_round,
            "rounds": len(selected),
            "seed_start": seed_start,
            "seed_end": seed_end,
            "budget_per_round": budget,
            "top_k": top_k,
            "min_hits": min_hits,
            "chunk_size": chunk_size,
            "bucket_size": bucket_size,
            "workers": worker_count,
            "reconstruction_budgets": list(budgets),
        },
        "execution": {
            "total_evaluated": total_evaluated,
            "elapsed_seconds": wall_elapsed,
            "seeds_per_second": total_evaluated / wall_elapsed,
            "completed_rounds": len(completed_rounds),
            "validated_dataset_rows": dataset_rows,
        },
        "reconstruction": {
            "best_hit_distribution": {str(hit): best_hit_distribution.get(hit, 0) for hit in range(7)},
            "hit_4_plus_rounds": sum(hit >= 4 for hit in best_hit_distribution.elements()),
            "hit_5_plus_rounds": sum(hit >= 5 for hit in best_hit_distribution.elements()),
            "exact_6_rounds": best_hit_distribution[6],
            "curve": curve,
        },
        "candidate_hit_distribution": {str(hit): count for hit, count in enumerate(candidate_hit_distribution)},
        "random_baseline": {
            "model": "independent uniformly random 6-of-45 sets; exact hypergeometric probabilities",
            "single_seed_probability": {str(hit): probability for hit, probability in enumerate(single_probabilities)},
            "expected_candidate_counts": expected_counts,
            "observed_vs_expected": observed_vs_expected,
            "best_hit_probability_per_round": {str(hit): probability for hit, probability in enumerate(maximum_probabilities)},
            "at_least_one_per_round": {
                "hit_4_plus": at_least_one_probability(minimum_hits=4, budget=budget),
                "hit_5_plus": at_least_one_probability(minimum_hits=5, budget=budget),
                "exact_6": at_least_one_probability(minimum_hits=6, budget=budget),
            },
            "monte_carlo_equal_budget": monte_carlo_round_max_baseline(budget=budget, rounds=len(selected)),
        },
        "seed_landscape": {
            "hit_levels": landscape_levels,
            "buckets": landscape_buckets,
            "top_k_spread": {
                "count": len(top_k_seeds),
                "unique_seed_count": len(set(top_k_seeds)),
                "min_seed": min(top_k_seeds),
                "max_seed": max(top_k_seeds),
                "median_seed": median(top_k_seeds),
                "p10_seed": _quantile(top_k_seeds, 0.1),
                "p90_seed": _quantile(top_k_seeds, 0.9),
            },
        },
        "unusual_rounds": {
            "exact_6_rounds": exact_rounds,
            "highest_hit_5_count": [
                {"round": row["round"], "hit_5_count": row["hit_5_count"], "best_seed": row["best"]["seed"]}
                for row in highest_hit_5
            ],
            "lowest_best_positional_mae": [
                {"round": row["round"], "best_positional_mae": row["best"]["positional_mae"], "best_hits": row["best"]["hits"], "best_seed": row["best"]["seed"]}
                for row in lowest_best_mae
            ],
        },
        "rounds": completed_rounds,
        "files": [
            "reverse-rounds.csv",
            "reverse-top-k.csv",
            "reverse-hit-seeds.csv",
            "reverse-seed-buckets.csv",
            "reverse-reconstruction-curve.csv",
            "reverse-summary.json",
            "reverse-progress.json",
        ],
    }
    summary["files"].extend(_write_curve_charts(run_dir, curve, len(selected)))
    _atomic_json(run_dir / "reverse-summary.json", summary)
    _atomic_json(run_dir / "reverse-progress.json", {
        "status": "completed",
        "completed_rounds": [row["round"] for row in completed_rounds],
        "total_rounds": len(selected),
        "evaluated": total_evaluated,
        "elapsed_seconds": wall_elapsed,
    })
    logger.info(
        "Reverse batch 완료 | 회차=%s/%s | 총평가=%s | 최고분포=%s | 4+=%s | 5+=%s | exact6=%s | 속도=%s seeds/s | 경과=%s",
        len(completed_rounds),
        len(selected),
        f"{total_evaluated:,}",
        dict(sorted(best_hit_distribution.items())),
        summary["reconstruction"]["hit_4_plus_rounds"],
        summary["reconstruction"]["hit_5_plus_rounds"],
        summary["reconstruction"]["exact_6_rounds"],
        f"{total_evaluated / wall_elapsed:,.0f}",
        _format_duration(wall_elapsed),
    )
    return summary
