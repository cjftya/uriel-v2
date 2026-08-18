from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from scipy import stats
from scipy.stats import qmc

from uriel_v2.probabilistic_lab.algorithm_common import empirical_entropy, finish_success, quality_from_objective
from uriel_v2.probabilistic_lab.budget import budget_fraction, checkpoint_steps
from uriel_v2.probabilistic_lab.schema import ExperimentBundle, JobSpec, TracePoint


def _sobol_dimension(job: JobSpec) -> int:
    dimension = int(job.problem.dimension or 1)
    return dimension * 2 if job.problem.problem_family == "mixture_mean" else dimension


def _sobol_points(job: JobSpec, rng: np.random.Generator) -> np.ndarray:
    dimension = _sobol_dimension(job)
    scramble_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=scramble_seed)
    total = job.budget.total
    if total > 0 and total & (total - 1) == 0:
        return sampler.random_base2(int(math.log2(total)))
    return sampler.random(total)


def _transform(problem_family: str, extension: dict, dimension: int, points: np.ndarray) -> np.ndarray:
    epsilon = np.finfo(float).eps
    clipped = np.clip(points, epsilon, 1.0 - epsilon)
    scale = float(extension.get("scale", 1.0))
    if problem_family == "gaussian_mean":
        return stats.norm.ppf(clipped[:, :dimension]) * scale
    if problem_family == "student_t_mean":
        degrees_freedom = float(extension["degrees_freedom"])
        return stats.t.ppf(clipped[:, :dimension], df=degrees_freedom) * scale
    if problem_family == "mixture_mean":
        separation = float(extension["separation"])
        signs = np.where(clipped[:, :dimension] < 0.5, -1.0, 1.0)
        noise = stats.norm.ppf(clipped[:, dimension : 2 * dimension]) * scale
        return signs * separation + noise
    raise ValueError(f"unsupported RQMC problem: {problem_family}")


def run_rqmc_sobol(job: JobSpec, rng: np.random.Generator) -> ExperimentBundle:
    started_at = perf_counter()
    uniform = _sobol_points(job, rng)
    dimension = int(job.problem.dimension or 1)
    samples = _transform(job.problem.problem_family, dict(job.problem.extension), dimension, uniform)
    traces: list[TracePoint] = []
    qualities: list[tuple[int, float]] = []
    previous_best = np.inf
    best_objective = np.inf
    target_mean = float(job.problem.extension.get("target_mean", 0.0))
    target_quality = float(job.problem.extension.get("target_quality", 0.95))
    previous_step = 0
    for step in checkpoint_steps(job.budget):
        prefix = samples[:step]
        estimate = np.mean(prefix, axis=0)
        objective = float(np.sqrt(np.mean((estimate - target_mean) ** 2)))
        variance = float(np.mean(np.var(prefix, axis=0, ddof=1))) if step > 1 else 0.0
        current_quality = quality_from_objective(objective)
        best_objective = min(best_objective, objective)
        improvement = 0.0 if not np.isfinite(previous_best) else max(0.0, previous_best - best_objective)
        discrepancy_sample_size = min(step, 256)
        if step > discrepancy_sample_size:
            discrepancy_indices = np.linspace(0, step - 1, discrepancy_sample_size, dtype=int)
            discrepancy_points = uniform[discrepancy_indices]
        else:
            discrepancy_points = uniform[:step]
        discrepancy = float(qmc.discrepancy(discrepancy_points, method="CD")) if step > 1 else 1.0
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
                entropy=empirical_entropy(prefix.ravel()),
                diversity=float(np.mean(np.std(prefix, axis=0))),
                distance_to_best=max(0.0, objective - best_objective),
                distance_to_target=max(0.0, target_quality - current_quality),
                failure_signal=0.0,
                extension={
                    "estimate_mean": float(np.mean(estimate)),
                    "standard_error": float(np.sqrt(variance / step)),
                    "sample_count": step,
                    "discrepancy": discrepancy,
                    "discrepancy_sample_size": discrepancy_sample_size,
                    "scramble": "LMS+shift",
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
        {
            "final_estimate": float(np.mean(samples)),
            "estimator": "scrambled_sobol_sample_mean",
            "sobol_dimension": _sobol_dimension(job),
            "scramble": "LMS+shift",
        },
    )
