from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from uriel_v2.metrics import hit_count, positional_deviations, positional_mae
from uriel_v2.models import Draw, EvaluationRow
from uriel_v2.strategies import create_predictions


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    history: tuple[Draw, ...]
    target: Draw
    strategies: tuple[str, ...]
    candidates: int
    history_window: int


def resolve_workers(value: str | int) -> int:
    if isinstance(value, int):
        workers = value
    elif value == "auto":
        workers = min(8, os.cpu_count() or 1)
    else:
        workers = int(value)
    if workers <= 0:
        raise ValueError("workers는 양수 또는 auto여야 합니다")
    return workers


def _evaluate_task(task: EvaluationTask) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for strategy in task.strategies:
        predictions = create_predictions(
            strategy,
            task.history,
            task.target.round_no,
            candidates=task.candidates,
            history_window=task.history_window,
        )
        ranked = []
        for prediction in predictions:
            hits = hit_count(prediction.numbers, task.target.numbers)
            mae = positional_mae(prediction.numbers, task.target.numbers)
            ranked.append((-hits, mae, prediction.seed, prediction))
        _, _, _, best = min(ranked)
        deviations = positional_deviations(best.numbers, task.target.numbers)
        hits = hit_count(best.numbers, task.target.numbers)
        rows.append(
            EvaluationRow(
                round_no=task.target.round_no,
                strategy=strategy,
                variant=best.variant,
                seed=best.seed,
                prediction=best.numbers,
                winner=task.target.numbers,
                hits=hits,
                set_distance=12 - (2 * hits),
                positional_mae=sum(abs(value) for value in deviations) / 6,
                signed_bias=sum(deviations) / 6,
                deviations=deviations,
            )
        )
    return rows


def evaluate_walk_forward(
    draws: Sequence[Draw],
    *,
    start_round: int,
    end_round: int,
    strategies: tuple[str, ...],
    candidates: int,
    history_window: int,
    minimum_history: int,
    workers: str | int,
    logger: logging.Logger,
) -> list[EvaluationRow]:
    tasks = [
        EvaluationTask(
            history=tuple(draws[:index]),
            target=draw,
            strategies=strategies,
            candidates=candidates,
            history_window=history_window,
        )
        for index, draw in enumerate(draws)
        if start_round <= draw.round_no <= end_round and index >= minimum_history
    ]
    if not tasks:
        raise ValueError("평가할 회차가 없습니다. 구간과 minimum-history를 확인하세요")

    worker_count = min(resolve_workers(workers), len(tasks))
    logger.info(
        "Walk-forward 시작 | 회차=%s~%s | 작업=%s | 전략=%s | 후보/전략=%s | 워커=%s",
        tasks[0].target.round_no,
        tasks[-1].target.round_no,
        len(tasks),
        ",".join(strategies),
        candidates,
        worker_count,
    )
    if candidates > 1:
        logger.warning("후보가 2개 이상이므로 회차별 정답 기준 oracle best를 기록합니다. 실전 선택 성능이 아닙니다.")

    rows: list[EvaluationRow] = []
    progress_step = max(1, len(tasks) // 20)
    if worker_count == 1:
        for completed, task in enumerate(tasks, start=1):
            rows.extend(_evaluate_task(task))
            if completed % progress_step == 0 or completed == len(tasks):
                logger.info("Walk-forward 진행 | %s/%s (%.1f%%)", completed, len(tasks), completed * 100 / len(tasks))
    else:
        executor = ProcessPoolExecutor(max_workers=worker_count)
        try:
            futures = [executor.submit(_evaluate_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.extend(future.result())
                if completed % progress_step == 0 or completed == len(tasks):
                    logger.info("Walk-forward 진행 | %s/%s (%.1f%%)", completed, len(tasks), completed * 100 / len(tasks))
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()

    rows.sort(key=lambda row: (row.round_no, row.strategy))
    return rows
