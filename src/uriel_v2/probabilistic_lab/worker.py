from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from uriel_v2.probabilistic_lab.runner import execute_job
from uriel_v2.probabilistic_lab.schema import ExperimentBundle, JobSpec
from uriel_v2.probabilistic_lab.storage import append_checkpoint, read_checkpoint


def resolve_workers(value: str | int) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("workers must be positive")
        return value
    if value == "auto":
        return max(1, min(8, os.cpu_count() or 1))
    try:
        return resolve_workers(int(value))
    except ValueError as exc:
        raise ValueError("workers must be a positive integer or auto") from exc


def _log_bundle(logger: logging.Logger, bundle: ExperimentBundle, completed: int, total: int) -> None:
    result = bundle.result
    logger.info(
        "[RESULT] %s/%s run=%s problem=%s algorithm=%s status=%s quality=%s runtime=%.4fs",
        completed,
        total,
        result.run_id,
        result.problem_id,
        result.algorithm,
        result.status,
        "NA" if result.quality_final is None else f"{result.quality_final:.6f}",
        result.runtime,
    )
    for trace in bundle.traces:
        logger.debug(
            "[STEP] run=%s step=%s fraction=%.2f objective=%.8f best=%.8f variance=%.8f entropy=%.6f",
            trace.run_id,
            trace.step,
            trace.budget_fraction,
            trace.objective,
            trace.best_so_far,
            trace.variance,
            trace.entropy,
        )


def run_jobs(
    jobs: Iterable[JobSpec],
    run_directory: str | Path,
    *,
    workers: str | int = "auto",
    resume: bool = True,
    logger: logging.Logger | None = None,
) -> list[dict]:
    job_list = list(jobs)
    run_ids = [job.run_id for job in job_list]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run_id in job list")
    log = logger or logging.getLogger("uriel.probabilistic")
    checkpoint_path = Path(run_directory) / "checkpoint.jsonl"
    existing = read_checkpoint(checkpoint_path) if resume else []
    completed_ids = {record["result"]["run_id"] for record in existing}
    unknown_ids = completed_ids - set(run_ids)
    if unknown_ids:
        raise ValueError(f"checkpoint contains jobs outside the current configuration: {sorted(unknown_ids)[:3]}")
    pending = [job for job in job_list if job.run_id not in completed_ids]
    worker_count = resolve_workers(workers)
    log.info(
        "[WORKER] jobs=%s pending=%s resumed=%s workers=%s checkpoint=%s",
        len(job_list),
        len(pending),
        len(existing),
        worker_count,
        checkpoint_path,
    )
    completed = len(existing)
    if worker_count == 1:
        for job in pending:
            log.info(
                "[RUN] run=%s problem=%s algorithm=%s seed=%s budget=%s:%s",
                job.run_id,
                job.problem.problem_id,
                job.algorithm.algorithm,
                job.seed,
                job.budget.budget_type,
                job.budget.total,
            )
            bundle = execute_job(job)
            append_checkpoint(checkpoint_path, bundle.to_checkpoint_record())
            completed += 1
            _log_bundle(log, bundle, completed, len(job_list))
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {executor.submit(execute_job, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    bundle = future.result()
                except Exception as exc:  # pragma: no cover - executor transport failure
                    log.exception("worker transport failure | run=%s error=%s", job.run_id, exc)
                    raise
                append_checkpoint(checkpoint_path, bundle.to_checkpoint_record())
                completed += 1
                _log_bundle(log, bundle, completed, len(job_list))
    records = read_checkpoint(checkpoint_path)
    log.info("[WORKER] DONE jobs=%s failures=%s", len(records), sum(record["result"]["failure"] for record in records))
    return records
