from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np

from uriel_v2.probabilistic_lab.schema import ExperimentBundle, JobSpec, RunResult, TracePoint


def quality_from_objective(objective: float) -> float:
    return 1.0 / (1.0 + max(0.0, objective))


def empirical_entropy(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or float(np.ptp(finite)) == 0.0:
        return 0.0
    counts, _ = np.histogram(finite, bins=min(16, max(2, int(math.sqrt(finite.size)))))
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def passage_times(qualities: list[tuple[int, float]], target_quality: float) -> dict[str, int | None]:
    thresholds = {"t50": 0.50, "t75": 0.75, "t90": 0.90, "t95": 0.95, "t99": 0.99}
    result: dict[str, int | None] = {}
    for name, fraction in thresholds.items():
        threshold = target_quality * fraction
        result[name] = next((step for step, quality in qualities if quality >= threshold), None)
    result["target"] = next((step for step, quality in qualities if quality >= target_quality), None)
    return result


def finish_success(
    job: JobSpec,
    started_at: float,
    traces: list[TracePoint],
    qualities: list[tuple[int, float]],
    best_objective: float,
    target_quality: float,
    extension: dict[str, Any],
) -> ExperimentBundle:
    passage = passage_times(qualities, target_quality)
    quality_values = np.asarray([quality for _, quality in qualities], dtype=float)
    stagnation = 0
    for trace in reversed(traces):
        if trace.improvement > 0.0:
            break
        stagnation += 1
    result = RunResult(
        run_id=job.run_id,
        problem_id=job.problem.problem_id,
        problem_family=job.problem.problem_family,
        domain=job.problem.domain,
        algorithm=job.algorithm.algorithm,
        algorithm_family=job.algorithm.algorithm_family,
        random_mechanism=job.algorithm.random_mechanism,
        algorithm_version=job.algorithm.version,
        seed=job.seed,
        rng_algorithm=job.rng_algorithm,
        rng_version=job.rng_version,
        budget_type=job.budget.budget_type,
        budget=job.budget.total,
        status="SUCCESS",
        quality_final=float(quality_values[-1]),
        quality_best=float(np.max(quality_values)),
        runtime=perf_counter() - started_at,
        steps=job.budget.total,
        success=True,
        failure=False,
        timeout=False,
        target_reached=passage["target"] is not None,
        first_passage_time=passage["target"],
        t50=passage["t50"],
        t75=passage["t75"],
        t90=passage["t90"],
        t95=passage["t95"],
        t99=passage["t99"],
        mean_quality=float(np.mean(quality_values)),
        variance_quality=float(np.var(quality_values)),
        best_so_far=float(best_objective),
        improvement_rate=(traces[0].best_so_far - traces[-1].best_so_far)
        / max(1, job.budget.total - traces[0].step),
        stagnation=stagnation,
        algorithm_config=job.algorithm.configuration,
        extension=extension,
    )
    return ExperimentBundle(result=result, traces=tuple(traces))
