from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from uriel_v2.probabilistic_lab.algorithm_common import empirical_entropy, finish_success, quality_from_objective
from uriel_v2.probabilistic_lab.budget import budget_fraction, checkpoint_steps
from uriel_v2.probabilistic_lab.problems import evaluate_objective_with_transformation, objective_transformation
from uriel_v2.probabilistic_lab.schema import ExperimentBundle, JobSpec, TracePoint


def _parameters(dimension: int, population_size: int | None) -> dict[str, float | int | np.ndarray]:
    population = population_size or 4 + int(3 * math.log(dimension))
    parents = population // 2
    weights = np.log(parents + 0.5) - np.log(np.arange(1, parents + 1))
    weights /= np.sum(weights)
    effective_parents = float(1.0 / np.sum(weights**2))
    cc = (4.0 + effective_parents / dimension) / (dimension + 4.0 + 2.0 * effective_parents / dimension)
    cs = (effective_parents + 2.0) / (dimension + effective_parents + 5.0)
    c1 = 2.0 / ((dimension + 1.3) ** 2 + effective_parents)
    cmu = min(
        1.0 - c1,
        2.0 * (effective_parents - 2.0 + 1.0 / effective_parents) / ((dimension + 2.0) ** 2 + effective_parents),
    )
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((effective_parents - 1.0) / (dimension + 1.0)) - 1.0) + cs
    return {
        "population": population,
        "parents": parents,
        "weights": weights,
        "effective_parents": effective_parents,
        "cc": cc,
        "cs": cs,
        "c1": c1,
        "cmu": cmu,
        "damps": damps,
    }


def run_cma_es(job: JobSpec, rng: np.random.Generator) -> ExperimentBundle:
    started_at = perf_counter()
    dimension = int(job.problem.dimension or 1)
    lower = float(job.problem.extension["lower_bound"])
    upper = float(job.problem.extension["upper_bound"])
    requested_population = job.algorithm.configuration.get("population_size")
    parameters = _parameters(dimension, int(requested_population) if requested_population else None)
    population = int(parameters["population"])
    parents = int(parameters["parents"])
    weights = np.asarray(parameters["weights"], dtype=float)
    effective_parents = float(parameters["effective_parents"])
    cc, cs = float(parameters["cc"]), float(parameters["cs"])
    c1, cmu, damps = float(parameters["c1"]), float(parameters["cmu"]), float(parameters["damps"])
    chi_n = math.sqrt(dimension) * (1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension**2))

    mean = rng.uniform(lower, upper, size=dimension)
    sigma = float(job.algorithm.configuration.get("sigma_fraction", 0.30)) * (upper - lower)
    covariance = np.eye(dimension)
    eigenvectors = np.eye(dimension)
    eigenvalues = np.ones(dimension)
    inverse_sqrt_covariance = np.eye(dimension)
    evolution_covariance = np.zeros(dimension)
    evolution_sigma = np.zeros(dimension)

    checkpoints = set(checkpoint_steps(job.budget))
    traces: list[TracePoint] = []
    qualities: list[tuple[int, float]] = []
    all_objectives: list[float] = []
    best_objective = np.inf
    previous_trace_best = np.inf
    evaluations = 0
    generation = 0
    stagnant_generations = 0
    target_quality = float(job.problem.extension.get("target_quality", 0.95))
    transformation = objective_transformation(job.problem)

    while evaluations < job.budget.total:
        generation += 1
        generation_start_best = best_objective
        current_population = min(population, job.budget.total - evaluations)
        normal_steps = rng.normal(size=(current_population, dimension))
        transformed_steps = normal_steps @ (eigenvectors * np.sqrt(eigenvalues)).T
        candidates = np.clip(mean + sigma * transformed_steps, lower, upper)
        objectives = evaluate_objective_with_transformation(job.problem, candidates, transformation)
        for candidate_index, objective_value in enumerate(objectives):
            evaluations += 1
            objective = float(objective_value)
            all_objectives.append(objective)
            best_objective = min(best_objective, objective)
            if evaluations not in checkpoints:
                continue
            best_quality = quality_from_objective(best_objective)
            improvement = (
                0.0 if not np.isfinite(previous_trace_best) else max(0.0, previous_trace_best - best_objective)
            )
            partial_objectives = objectives[: candidate_index + 1]
            partial_candidates = candidates[: candidate_index + 1]
            covariance_condition = float(np.max(eigenvalues) / max(np.min(eigenvalues), 1e-30))
            elite_count = min(parents, len(partial_objectives))
            elite_values = np.sort(partial_objectives)[:elite_count]
            traces.append(
                TracePoint(
                    run_id=job.run_id,
                    step=evaluations,
                    budget_fraction=budget_fraction(evaluations, job.budget),
                    elapsed_time=perf_counter() - started_at,
                    objective=float(np.min(partial_objectives)),
                    best_so_far=best_objective,
                    improvement=improvement,
                    improvement_rate=improvement / max(1, evaluations - (traces[-1].step if traces else 0)),
                    variance=float(np.var(all_objectives)),
                    entropy=empirical_entropy(np.asarray(all_objectives)),
                    diversity=float(np.mean(np.linalg.norm(partial_candidates - mean, axis=1))),
                    distance_to_best=max(0.0, float(np.min(partial_objectives)) - best_objective),
                    distance_to_target=max(0.0, target_quality - best_quality),
                    failure_signal=float(covariance_condition > 1e14 or sigma < 1e-14),
                    extension={
                        "generation": generation,
                        "sigma": sigma,
                        "covariance_eigenvalue_min": float(np.min(eigenvalues)),
                        "covariance_eigenvalue_max": float(np.max(eigenvalues)),
                        "covariance_condition": covariance_condition,
                        "elite_spread": float(np.std(elite_values)) if elite_count > 1 else 0.0,
                        "mean_fitness": float(np.mean(partial_objectives)),
                        "population_size": population,
                        "stagnant_generations": stagnant_generations,
                    },
                )
            )
            qualities.append((evaluations, best_quality))
            previous_trace_best = best_objective

        if best_objective < generation_start_best:
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        if current_population < parents:
            continue

        order = np.argsort(objectives)
        selected = order[:parents]
        old_mean = mean.copy()
        mean = weights @ candidates[selected]
        weighted_step = (mean - old_mean) / max(sigma, 1e-30)
        evolution_sigma = (1.0 - cs) * evolution_sigma + math.sqrt(
            cs * (2.0 - cs) * effective_parents
        ) * (inverse_sqrt_covariance @ weighted_step)
        sigma_norm = float(np.linalg.norm(evolution_sigma))
        hsig_denominator = math.sqrt(max(1e-30, 1.0 - (1.0 - cs) ** (2.0 * generation)))
        hsig = float(sigma_norm / hsig_denominator / chi_n < (1.4 + 2.0 / (dimension + 1.0)))
        evolution_covariance = (1.0 - cc) * evolution_covariance + hsig * math.sqrt(
            cc * (2.0 - cc) * effective_parents
        ) * weighted_step
        selected_steps = (candidates[selected] - old_mean) / max(sigma, 1e-30)
        rank_mu = sum(weight * np.outer(step, step) for weight, step in zip(weights, selected_steps, strict=True))
        covariance = (
            (1.0 - c1 - cmu) * covariance
            + c1
            * (
                np.outer(evolution_covariance, evolution_covariance)
                + (1.0 - hsig) * cc * (2.0 - cc) * covariance
            )
            + cmu * rank_mu
        )
        covariance = 0.5 * (covariance + covariance.T)
        raw_eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(raw_eigenvalues, 1e-30)
        inverse_sqrt_covariance = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
        sigma *= math.exp((cs / damps) * (sigma_norm / chi_n - 1.0))
        sigma = min(max(sigma, 1e-15), 10.0 * (upper - lower))

    return finish_success(
        job,
        started_at,
        traces,
        qualities,
        best_objective,
        target_quality,
        {
            "generations": generation,
            "population_size": population,
            "parents": parents,
            "sigma_final": sigma,
            "covariance_eigenvalues": [float(value) for value in eigenvalues],
            "stagnant_generations": stagnant_generations,
        },
    )
