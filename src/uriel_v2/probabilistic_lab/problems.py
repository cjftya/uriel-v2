from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from uriel_v2.probabilistic_lab.schema import ProblemSpec


SAMPLING_FAMILIES = ("gaussian_mean", "student_t_mean", "mixture_mean")
OPTIMIZATION_FAMILIES = ("sphere", "rastrigin", "rosenbrock")
SYNTHETIC_SAMPLING_FAMILIES = (*SAMPLING_FAMILIES, "lognormal_mean")
SYNTHETIC_OPTIMIZATION_FAMILIES = (*OPTIMIZATION_FAMILIES, "ackley", "griewank", "schwefel")


def _problem_rng(master_seed: int, family_index: int, instance_index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(master_seed), int(family_index), int(instance_index)])
    return np.random.Generator(np.random.PCG64(sequence))


def build_pilot_problems(instances_per_family: int, master_seed: int = 20_260_819) -> list[ProblemSpec]:
    if instances_per_family <= 0:
        raise ValueError("instances_per_family must be positive")
    problems: list[ProblemSpec] = []
    families: Iterable[tuple[str, str]] = (
        *((family, "sampling") for family in SAMPLING_FAMILIES),
        *((family, "optimization") for family in OPTIMIZATION_FAMILIES),
    )
    dimensions = (2, 5, 10, 20)
    for family_index, (family, domain) in enumerate(families):
        for instance_index in range(instances_per_family):
            rng = _problem_rng(master_seed, family_index, instance_index)
            dimension = dimensions[instance_index % len(dimensions)]
            problem_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            if domain == "sampling":
                scale = float(rng.choice((0.5, 1.0, 2.0, 4.0)))
                if family == "gaussian_mean":
                    extension = {"scale": scale, "target_mean": 0.0, "target_quality": 0.95}
                    skewness, kurtosis, multimodality = 0.0, 3.0, 1.0
                elif family == "student_t_mean":
                    degrees_freedom = float(rng.choice((2.5, 3.0, 5.0, 10.0)))
                    extension = {
                        "scale": scale,
                        "degrees_freedom": degrees_freedom,
                        "target_mean": 0.0,
                        "target_quality": 0.90,
                    }
                    skewness = 0.0
                    kurtosis = None if degrees_freedom <= 4.0 else 3.0 + 6.0 / (degrees_freedom - 4.0)
                    multimodality = 1.0
                else:
                    separation = float(rng.choice((1.0, 2.0, 4.0)))
                    extension = {
                        "scale": scale,
                        "separation": separation,
                        "target_mean": 0.0,
                        "target_quality": 0.90,
                    }
                    skewness, kurtosis, multimodality = 0.0, 2.0, 2.0
                entropy = 0.5 * math.log(2.0 * math.pi * math.e * scale * scale)
                problems.append(
                    ProblemSpec(
                        problem_id=f"{family}-{instance_index:04d}",
                        problem_family=family,
                        domain=domain,
                        problem_seed=problem_seed,
                        dimension=dimension,
                        noise=scale,
                        entropy=entropy,
                        skewness=skewness,
                        kurtosis=kurtosis,
                        multimodality=multimodality,
                        effective_dimension=float(dimension),
                        extension=extension,
                    )
                )
                continue

            variant = ("base", "rotated", "ill_conditioned")[instance_index % 3]
            if family == "sphere":
                bounds = (-5.12, 5.12)
                multimodality, ruggedness = 1.0, 0.0
            elif family == "rastrigin":
                bounds = (-5.12, 5.12)
                multimodality, ruggedness = 10.0, 1.0
            else:
                bounds = (-2.048, 2.048)
                multimodality, ruggedness = 1.0, 0.7
            condition_number = 10_000.0 if variant == "ill_conditioned" else (100.0 if family == "rosenbrock" else 1.0)
            problems.append(
                ProblemSpec(
                    problem_id=f"{family}-{instance_index:04d}-{variant}",
                    problem_family=family,
                    domain=domain,
                    problem_seed=problem_seed,
                    dimension=dimension,
                    noise=0.0,
                    condition_number=condition_number,
                    multimodality=multimodality,
                    ruggedness=ruggedness,
                    effective_dimension=float(dimension),
                    extension={
                        "lower_bound": bounds[0],
                        "upper_bound": bounds[1],
                        "variant": variant,
                        "target_quality": 0.95,
                    },
                )
            )
    return problems


def draw_sampling_batch(problem: ProblemSpec, rng: np.random.Generator, count: int) -> np.ndarray:
    dimension = int(problem.dimension or 1)
    scale = float(problem.extension.get("scale", 1.0))
    if problem.problem_family == "gaussian_mean":
        return rng.normal(0.0, scale, size=(count, dimension))
    if problem.problem_family == "student_t_mean":
        degrees_freedom = float(problem.extension["degrees_freedom"])
        return rng.standard_t(degrees_freedom, size=(count, dimension)) * scale
    if problem.problem_family == "mixture_mean":
        separation = float(problem.extension["separation"])
        positive_weight = float(problem.extension.get("positive_weight", 0.5))
        signs = np.where(rng.random(size=(count, dimension)) < positive_weight, 1.0, -1.0)
        return rng.normal(signs * separation, scale, size=(count, dimension))
    if problem.problem_family == "lognormal_mean":
        log_mu = float(problem.extension.get("log_mu", 0.0))
        log_sigma = float(problem.extension["log_sigma"])
        return rng.lognormal(log_mu, log_sigma, size=(count, dimension))
    raise ValueError(f"unsupported sampling problem: {problem.problem_family}")


def objective_transformation(problem: ProblemSpec) -> tuple[np.ndarray, np.ndarray]:
    dimension = int(problem.dimension or 1)
    rng = np.random.Generator(np.random.PCG64(problem.problem_seed))
    shift = rng.uniform(-0.25, 0.25, size=dimension)
    rotation = np.eye(dimension)
    if problem.extension.get("variant") == "rotated":
        matrix = rng.normal(size=(dimension, dimension))
        rotation, _ = np.linalg.qr(matrix)
    return shift, rotation


def evaluate_objective_with_transformation(
    problem: ProblemSpec,
    points: np.ndarray,
    transformation: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    shift, rotation = transformation
    values = (points - shift) @ rotation
    variant = problem.extension.get("variant")
    if variant == "ill_conditioned":
        weights = np.geomspace(1.0, float(problem.condition_number or 1.0), values.shape[1])
    else:
        weights = np.ones(values.shape[1], dtype=float)
    if problem.problem_family == "sphere":
        return np.sum(weights * values * values, axis=1)
    if problem.problem_family == "rastrigin":
        return 10.0 * values.shape[1] + np.sum(weights * values * values - 10.0 * np.cos(2.0 * np.pi * values), axis=1)
    if problem.problem_family == "rosenbrock":
        conditioned_values = values * np.sqrt(weights) if variant == "ill_conditioned" else values
        shifted = conditioned_values + 1.0
        return np.sum(100.0 * (shifted[:, 1:] - shifted[:, :-1] ** 2) ** 2 + (1.0 - shifted[:, :-1]) ** 2, axis=1)
    conditioned_values = values * np.sqrt(weights) if variant == "ill_conditioned" else values
    if problem.problem_family == "ackley":
        squared_mean = np.mean(conditioned_values**2, axis=1)
        cosine_mean = np.mean(np.cos(2.0 * np.pi * conditioned_values), axis=1)
        return -20.0 * np.exp(-0.2 * np.sqrt(squared_mean)) - np.exp(cosine_mean) + 20.0 + math.e
    if problem.problem_family == "griewank":
        indices = np.sqrt(np.arange(1, conditioned_values.shape[1] + 1, dtype=float))
        return np.sum(conditioned_values**2, axis=1) / 4_000.0 - np.prod(
            np.cos(conditioned_values / indices), axis=1
        ) + 1.0
    if problem.problem_family == "schwefel":
        cumulative = np.cumsum(conditioned_values, axis=1)
        return np.sum(cumulative**2, axis=1)
    raise ValueError(f"unsupported optimization problem: {problem.problem_family}")


def evaluate_objective(problem: ProblemSpec, points: np.ndarray) -> np.ndarray:
    return evaluate_objective_with_transformation(problem, points, objective_transformation(problem))
