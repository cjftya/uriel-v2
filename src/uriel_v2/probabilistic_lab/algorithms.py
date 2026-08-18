from __future__ import annotations

from time import perf_counter

import numpy as np

from uriel_v2.probabilistic_lab.algorithm_common import empirical_entropy, finish_success, quality_from_objective
from uriel_v2.probabilistic_lab.budget import budget_fraction, checkpoint_steps
from uriel_v2.probabilistic_lab.cmaes import run_cma_es
from uriel_v2.probabilistic_lab.problems import draw_sampling_batch, evaluate_objective
from uriel_v2.probabilistic_lab.registry import AlgorithmRegistry
from uriel_v2.probabilistic_lab.rqmc import run_rqmc_sobol
from uriel_v2.probabilistic_lab.schema import ExperimentBundle, JobSpec, TracePoint


def run_monte_carlo(job: JobSpec, rng: np.random.Generator) -> ExperimentBundle:
    started_at = perf_counter()
    traces: list[TracePoint] = []
    qualities: list[tuple[int, float]] = []
    samples: list[np.ndarray] = []
    previous_step = 0
    previous_best = np.inf
    best_objective = np.inf
    target_mean = float(job.problem.extension.get("target_mean", 0.0))
    target_quality = float(job.problem.extension.get("target_quality", 0.95))
    for step in checkpoint_steps(job.budget):
        new_count = step - previous_step
        samples.append(draw_sampling_batch(job.problem, rng, new_count))
        all_samples = np.concatenate(samples, axis=0)
        estimate = np.mean(all_samples, axis=0)
        objective = float(np.sqrt(np.mean((estimate - target_mean) ** 2)))
        variance = float(np.mean(np.var(all_samples, axis=0, ddof=1))) if step > 1 else 0.0
        current_quality = quality_from_objective(objective)
        best_objective = min(best_objective, objective)
        improvement = 0.0 if not np.isfinite(previous_best) else max(0.0, previous_best - best_objective)
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
                entropy=empirical_entropy(all_samples.ravel()),
                diversity=float(np.mean(np.std(all_samples, axis=0))),
                distance_to_best=max(0.0, objective - best_objective),
                distance_to_target=max(0.0, target_quality - current_quality),
                failure_signal=0.0,
                extension={
                    "estimate_mean": float(np.mean(estimate)),
                    "standard_error": float(np.sqrt(variance / step)) if step else np.inf,
                    "sample_count": step,
                },
            )
        )
        qualities.append((step, current_quality))
        previous_best = best_objective
        previous_step = step
    return finish_success(
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
    previous_best = np.inf
    best_objective = np.inf
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
        best_quality = quality_from_objective(best_objective)
        improvement = 0.0 if not np.isfinite(previous_best) else max(0.0, previous_best - best_objective)
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
                entropy=empirical_entropy(all_objectives),
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
    return finish_success(
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
    registry.register("rqmc_sobol", {"sampling"}, run_rqmc_sobol)
    registry.register("random_search", {"optimization"}, run_random_search)
    registry.register("cma_es", {"optimization"}, run_cma_es)
    return registry
