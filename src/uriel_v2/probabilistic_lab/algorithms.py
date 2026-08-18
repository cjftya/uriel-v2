from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np

from uriel_v2.probabilistic_lab.budget import budget_fraction, checkpoint_steps
from uriel_v2.probabilistic_lab.problems import draw_sampling_batch, evaluate_objective
from uriel_v2.probabilistic_lab.registry import AlgorithmRegistry
from uriel_v2.probabilistic_lab.schema import ExperimentBundle, JobSpec, RunResult, TracePoint


def _quality(objective: float) -> float:
    return 1.0 / (1.0 + max(0.0, objective))


def _entropy(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or float(np.ptp(finite)) == 0.0:
        return 0.0
    counts, _ = np.histogram(finite, bins=min(16, max(2, int(math.sqrt(finite.size)))))
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def _passage_times(qualities: list[tuple[int, float]], target_quality: float) -> dict[str, int | None]:
    thresholds = {"t50": 0.50, "t75": 0.75, "t90": 0.90, "t95": 0.95, "t99": 0.99}
    result: dict[str, int | None] = {}
    for name, fraction in thresholds.items():
        threshold = target_quality * fraction
        result[name] = next((step for step, quality in qualities if quality >= threshold), None)
    result["target"] = next((step for step, quality in qualities if quality >= target_quality), None)
    return result


def _finish(
    job: JobSpec,
    started_at: float,
    traces: list[TracePoint],
    qualities: list[tuple[int, float]],
    best_objective: float,
    target_quality: float,
    extension: dict[str, Any],
) -> ExperimentBundle:
    passage = _passage_times(qualities, target_quality)
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
        improvement_rate=(traces[0].best_so_far - traces[-1].best_so_far) / max(1, job.budget.total - traces[0].step),
        stagnation=stagnation,
        algorithm_config=job.algorithm.configuration,
        extension=extension,
    )
    return ExperimentBundle(result=result, traces=tuple(traces))


def run_monte_carlo(job: JobSpec, rng: np.random.Generator) -> ExperimentBundle:
    started_at = perf_counter()
    traces: list[TracePoint] = []
    qualities: list[tuple[int, float]] = []
    samples: list[np.ndarray] = []
    previous_step = 0
    previous_best = math.inf
    best_objective = math.inf
    target_mean = float(job.problem.extension.get("target_mean", 0.0))
    target_quality = float(job.problem.extension.get("target_quality", 0.95))
    for step in checkpoint_steps(job.budget):
        new_count = step - previous_step
        samples.append(draw_sampling_batch(job.problem, rng, new_count))
        all_samples = np.concatenate(samples, axis=0)
        estimate = np.mean(all_samples, axis=0)
        objective = float(np.sqrt(np.mean((estimate - target_mean) ** 2)))
        variance = float(np.mean(np.var(all_samples, axis=0, ddof=1))) if step > 1 else 0.0
        current_quality = _quality(objective)
        best_objective = min(best_objective, objective)
        best_quality = _quality(best_objective)
        improvement = 0.0 if not math.isfinite(previous_best) else max(0.0, previous_best - best_objective)
        traces.append(
            TracePoint(
                run_id=job.run_id,
                step=step,
                budget_fraction=budget_fraction(step, job.budget),
                elapsed_time=perf_counter() - started_at,
                objective=objective,
                best_so_far=best_objective,
                improvement=improvement,
                improvement_rate=improvement / max(1, step - previous_step),
                variance=variance,
                entropy=_entropy(all_samples.ravel()),
                diversity=float(np.mean(np.std(all_samples, axis=0))),
                distance_to_best=max(0.0, objective - best_objective),
                distance_to_target=max(0.0, target_quality - current_quality),
                failure_signal=0.0,
                extension={
                    "estimate_mean": float(np.mean(estimate)),
                    "standard_error": math.sqrt(variance / step) if step else math.inf,
                    "sample_count": step,
                },
            )
        )
        qualities.append((step, current_quality))
        previous_best = best_objective
        previous_step = step
    return _finish(
        job,
        started_at,
        traces,
        qualities,
        best_objective,
        target_quality,
        {"final_estimate": float(np.mean(np.concatenate(samples, axis=0))), "estimator": "sample_mean"},
    )


def run_random_search(job: JobSpec, rng: np.random.Generator) -> ExperimentBundle:
    started_at = perf_counter()
    traces: list[TracePoint] = []
    qualities: list[tuple[int, float]] = []
    objective_chunks: list[np.ndarray] = []
    point_chunks: list[np.ndarray] = []
    previous_step = 0
    previous_best = math.inf
    best_objective = math.inf
    lower = float(job.problem.extension["lower_bound"])
    upper = float(job.problem.extension["upper_bound"])
    dimension = int(job.problem.dimension or 1)
    target_quality = float(job.problem.extension.get("target_quality", 0.95))
    for step in checkpoint_steps(job.budget):
        new_count = step - previous_step
        points = rng.uniform(lower, upper, size=(new_count, dimension))
        objectives = evaluate_objective(job.problem, points)
        point_chunks.append(points)
        objective_chunks.append(objectives)
        all_points = np.concatenate(point_chunks, axis=0)
        all_objectives = np.concatenate(objective_chunks)
        current_batch_best = float(np.min(objectives))
        best_objective = min(best_objective, current_batch_best)
        best_quality = _quality(best_objective)
        improvement = 0.0 if not math.isfinite(previous_best) else max(0.0, previous_best - best_objective)
        traces.append(
            TracePoint(
                run_id=job.run_id,
                step=step,
                budget_fraction=budget_fraction(step, job.budget),
                elapsed_time=perf_counter() - started_at,
                objective=current_batch_best,
                best_so_far=best_objective,
                improvement=improvement,
                improvement_rate=improvement / max(1, step - previous_step),
                variance=float(np.var(all_objectives)),
                entropy=_entropy(all_objectives),
                diversity=float(np.mean(np.std(all_points, axis=0))),
                distance_to_best=max(0.0, current_batch_best - best_objective),
                distance_to_target=max(0.0, target_quality - best_quality),
                failure_signal=0.0,
                extension={"trial_count": step, "coverage_fraction": min(1.0, step / max(1.0, 10.0 * dimension))},
            )
        )
        qualities.append((step, best_quality))
        previous_best = best_objective
        previous_step = step
    return _finish(
        job,
        started_at,
        traces,
        qualities,
        best_objective,
        target_quality,
        {"search_bounds": [lower, upper], "trials": job.budget.total},
    )


def default_registry() -> AlgorithmRegistry:
    registry = AlgorithmRegistry()
    registry.register("monte_carlo_iid", {"sampling"}, run_monte_carlo)
    registry.register("random_search", {"optimization"}, run_random_search)
    return registry
