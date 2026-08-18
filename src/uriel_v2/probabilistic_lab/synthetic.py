from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import special

from uriel_v2.probabilistic_lab.problems import (
    SYNTHETIC_OPTIMIZATION_FAMILIES,
    SYNTHETIC_SAMPLING_FAMILIES,
)
from uriel_v2.probabilistic_lab.schema import ProblemSpec, canonical_json


BENCHMARK_VERSION = "phase3-v1"
SYNTHETIC_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "sampling": SYNTHETIC_SAMPLING_FAMILIES,
    "optimization": SYNTHETIC_OPTIMIZATION_FAMILIES,
    "matrix": ("low_rank_matrix", "power_law_matrix", "sparse_matrix"),
    "stream": ("uniform_stream", "zipf_stream", "bursty_stream", "concept_drift_stream"),
}
READY_DOMAINS = frozenset({"sampling", "optimization"})

_NUMERIC_DIMENSIONS = (2, 5, 10, 20, 50, 100)
_MATRIX_DIMENSIONS = (32, 64, 128, 256)
_STREAM_DIMENSIONS = (32, 128, 512, 2_048)


def _problem_seed(master_seed: int, family_index: int, instance_index: int) -> int:
    sequence = np.random.SeedSequence([int(master_seed), int(family_index), int(instance_index), 3])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] & np.uint64(np.iinfo(np.int64).max))


def _balanced_sequence(
    levels: Sequence[Any],
    count: int,
    *,
    master_seed: int,
    family_index: int,
    axis_index: int,
) -> list[Any]:
    if not levels:
        raise ValueError("balanced design axis must contain at least one level")
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([int(master_seed), family_index, axis_index, 31]))
    )
    values: list[Any] = []
    while len(values) < count:
        permutation = rng.permutation(len(levels))
        values.extend(levels[int(index)] for index in permutation)
    return values[:count]


def _axes(
    definitions: Mapping[str, Sequence[Any]],
    count: int,
    *,
    master_seed: int,
    family_index: int,
) -> dict[str, list[Any]]:
    return {
        name: _balanced_sequence(
            levels,
            count,
            master_seed=master_seed,
            family_index=family_index,
            axis_index=axis_index,
        )
        for axis_index, (name, levels) in enumerate(definitions.items())
    }


def _mixture_moments(scale: float, separation: float, positive_weight: float) -> tuple[float, float, float]:
    mean = separation * (2.0 * positive_weight - 1.0)
    positive_delta = separation - mean
    negative_delta = -separation - mean
    variance = scale**2 + positive_weight * positive_delta**2 + (1.0 - positive_weight) * negative_delta**2
    third = positive_weight * positive_delta**3 + (1.0 - positive_weight) * negative_delta**3
    fourth = positive_weight * (
        positive_delta**4 + 6.0 * positive_delta**2 * scale**2 + 3.0 * scale**4
    ) + (1.0 - positive_weight) * (
        negative_delta**4 + 6.0 * negative_delta**2 * scale**2 + 3.0 * scale**4
    )
    return mean, third / variance**1.5, fourth / variance**2


def _sampling_problem(
    family: str,
    family_index: int,
    instance_index: int,
    count: int,
    master_seed: int,
) -> ProblemSpec:
    axes = _axes(
        {
            "dimension": _NUMERIC_DIMENSIONS,
            "scale": (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
            "degrees_freedom": (2.1, 2.5, 3.0, 5.0, 10.0, 30.0),
            "separation": (0.5, 1.0, 2.0, 4.0, 8.0),
            "positive_weight": (0.2, 0.35, 0.5, 0.65, 0.8),
            "log_sigma": (0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
        },
        count,
        master_seed=master_seed,
        family_index=family_index,
    )
    dimension = int(axes["dimension"][instance_index])
    scale = float(axes["scale"][instance_index])
    extension: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "execution_tier": "ready",
        "target_quality": 0.95,
    }
    if family == "gaussian_mean":
        entropy = 0.5 * math.log(2.0 * math.pi * math.e * scale**2)
        extension.update({"scale": scale, "target_mean": 0.0})
        noise, skewness, kurtosis, multimodality = scale, 0.0, 3.0, 1.0
    elif family == "student_t_mean":
        degrees_freedom = float(axes["degrees_freedom"][instance_index])
        entropy = float(
            math.log(scale * math.sqrt(degrees_freedom) * special.beta(degrees_freedom / 2.0, 0.5))
            + (degrees_freedom + 1.0)
            / 2.0
            * (
                special.digamma((degrees_freedom + 1.0) / 2.0)
                - special.digamma(degrees_freedom / 2.0)
            )
        )
        extension.update({"scale": scale, "degrees_freedom": degrees_freedom, "target_mean": 0.0})
        noise = scale * math.sqrt(degrees_freedom / (degrees_freedom - 2.0))
        skewness = 0.0
        kurtosis = None if degrees_freedom <= 4.0 else 3.0 + 6.0 / (degrees_freedom - 4.0)
        multimodality = 1.0
    elif family == "mixture_mean":
        separation = float(axes["separation"][instance_index])
        positive_weight = float(axes["positive_weight"][instance_index])
        target_mean, skewness, kurtosis = _mixture_moments(scale, separation, positive_weight)
        variance = scale**2 + 4.0 * separation**2 * positive_weight * (1.0 - positive_weight)
        entropy = 0.5 * math.log(2.0 * math.pi * math.e * variance)
        extension.update(
            {
                "scale": scale,
                "separation": separation,
                "positive_weight": positive_weight,
                "target_mean": target_mean,
            }
        )
        noise, multimodality = math.sqrt(variance), 2.0
    elif family == "lognormal_mean":
        log_sigma = float(axes["log_sigma"][instance_index])
        log_mu = 0.0
        target_mean = math.exp(log_mu + 0.5 * log_sigma**2)
        variance = (math.exp(log_sigma**2) - 1.0) * math.exp(2.0 * log_mu + log_sigma**2)
        entropy = log_mu + 0.5 + math.log(log_sigma * math.sqrt(2.0 * math.pi))
        skewness = (math.exp(log_sigma**2) + 2.0) * math.sqrt(math.exp(log_sigma**2) - 1.0)
        kurtosis = (
            math.exp(4.0 * log_sigma**2)
            + 2.0 * math.exp(3.0 * log_sigma**2)
            + 3.0 * math.exp(2.0 * log_sigma**2)
            - 3.0
        )
        extension.update({"log_mu": log_mu, "log_sigma": log_sigma, "target_mean": target_mean})
        noise, multimodality = math.sqrt(variance), 1.0
    else:  # pragma: no cover - guarded by the family registry
        raise ValueError(f"unsupported synthetic sampling family: {family}")
    return ProblemSpec(
        problem_id=f"synthetic-{family}-{instance_index:05d}",
        problem_family=family,
        domain="sampling",
        problem_seed=_problem_seed(master_seed, family_index, instance_index),
        dimension=dimension,
        noise=noise,
        entropy=entropy,
        skewness=skewness,
        kurtosis=kurtosis,
        multimodality=multimodality,
        effective_dimension=float(dimension),
        extension=extension,
    )


def _optimization_problem(
    family: str,
    family_index: int,
    instance_index: int,
    count: int,
    master_seed: int,
) -> ProblemSpec:
    axes = _axes(
        {
            "dimension": _NUMERIC_DIMENSIONS,
            "variant": ("base", "rotated", "ill_conditioned"),
            "condition_number": (10.0, 100.0, 10_000.0, 1_000_000.0),
        },
        count,
        master_seed=master_seed,
        family_index=family_index,
    )
    dimension = int(axes["dimension"][instance_index])
    variant = str(axes["variant"][instance_index])
    ill_condition = float(axes["condition_number"][instance_index])
    bounds = {
        "sphere": (-5.12, 5.12),
        "rastrigin": (-5.12, 5.12),
        "rosenbrock": (-2.048, 2.048),
        "ackley": (-32.768, 32.768),
        "griewank": (-600.0, 600.0),
        "schwefel": (-500.0, 500.0),
    }[family]
    multimodality = 1.0 if family in {"sphere", "rosenbrock"} else 10.0
    ruggedness = {
        "sphere": 0.0,
        "rosenbrock": 0.7,
        "rastrigin": 1.0,
        "ackley": 0.9,
        "griewank": 0.8,
        "schwefel": 1.0,
    }[family]
    condition_number = ill_condition if variant == "ill_conditioned" else (100.0 if family == "rosenbrock" else 1.0)
    return ProblemSpec(
        problem_id=f"synthetic-{family}-{instance_index:05d}-{variant}",
        problem_family=family,
        domain="optimization",
        problem_seed=_problem_seed(master_seed, family_index, instance_index),
        dimension=dimension,
        noise=0.0,
        condition_number=condition_number,
        multimodality=multimodality,
        ruggedness=ruggedness,
        effective_dimension=float(dimension),
        extension={
            "benchmark_version": BENCHMARK_VERSION,
            "execution_tier": "ready",
            "lower_bound": bounds[0],
            "upper_bound": bounds[1],
            "variant": variant,
            "target_quality": 0.95,
        },
    )


def _matrix_problem(
    family: str,
    family_index: int,
    instance_index: int,
    count: int,
    master_seed: int,
) -> ProblemSpec:
    axes = _axes(
        {
            "dimension": _MATRIX_DIMENSIONS,
            "aspect_ratio": (1, 2, 4),
            "rank": (2, 4, 8, 16, 32),
            "condition_number": (10.0, 100.0, 10_000.0, 1_000_000.0),
            "spectral_decay": (0.5, 1.0, 2.0, 4.0),
            "sparsity": (0.0, 0.5, 0.9, 0.99),
            "noise": (0.0, 1e-4, 1e-2, 0.1),
        },
        count,
        master_seed=master_seed,
        family_index=family_index,
    )
    dimension = int(axes["dimension"][instance_index])
    rows = dimension * int(axes["aspect_ratio"][instance_index])
    columns = dimension
    rank = min(int(axes["rank"][instance_index]), dimension - 1)
    condition_number = float(axes["condition_number"][instance_index])
    spectral_decay = float(axes["spectral_decay"][instance_index])
    sparsity = float(axes["sparsity"][instance_index]) if family == "sparse_matrix" else 0.0
    noise = float(axes["noise"][instance_index])
    return ProblemSpec(
        problem_id=f"synthetic-{family}-{instance_index:05d}",
        problem_family=family,
        domain="matrix",
        problem_seed=_problem_seed(master_seed, family_index, instance_index),
        dimension=dimension,
        size=rows * columns,
        density=1.0 - sparsity,
        sparsity=sparsity,
        noise=noise,
        condition_number=condition_number,
        spectral_decay=spectral_decay,
        effective_dimension=float(rank),
        extension={
            "benchmark_version": BENCHMARK_VERSION,
            "execution_tier": "staged",
            "rows": rows,
            "columns": columns,
            "rank": rank,
            "spectrum": family,
        },
    )


def _probability_moments(probabilities: np.ndarray) -> tuple[float, float]:
    centered = probabilities - float(np.mean(probabilities))
    variance = float(np.mean(centered**2))
    if variance <= 0.0:
        return 0.0, 3.0
    return float(np.mean(centered**3) / variance**1.5), float(np.mean(centered**4) / variance**2)


def _stream_problem(
    family: str,
    family_index: int,
    instance_index: int,
    count: int,
    master_seed: int,
) -> ProblemSpec:
    axes = _axes(
        {
            "dimension": _STREAM_DIMENSIONS,
            "size": (10_000, 100_000, 1_000_000),
            "zipf_exponent": (1.05, 1.2, 1.5, 2.0),
            "burst_probability": (0.01, 0.05, 0.1, 0.2),
            "drift_fraction": (0.25, 0.5, 0.75),
        },
        count,
        master_seed=master_seed,
        family_index=family_index,
    )
    categories = int(axes["dimension"][instance_index])
    size = int(axes["size"][instance_index])
    exponent = float(axes["zipf_exponent"][instance_index])
    ranks = np.arange(1, categories + 1, dtype=float)
    if family == "uniform_stream":
        probabilities = np.full(categories, 1.0 / categories)
        autocorrelation = 0.0
    else:
        probabilities = ranks ** (-exponent)
        probabilities /= np.sum(probabilities)
        autocorrelation = 0.0 if family == "zipf_stream" else (0.85 if family == "bursty_stream" else 0.5)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    skewness, kurtosis = _probability_moments(probabilities)
    return ProblemSpec(
        problem_id=f"synthetic-{family}-{instance_index:05d}",
        problem_family=family,
        domain="stream",
        problem_seed=_problem_seed(master_seed, family_index, instance_index),
        dimension=categories,
        size=size,
        density=1.0,
        sparsity=0.0,
        noise=0.0,
        entropy=entropy,
        skewness=skewness,
        kurtosis=kurtosis,
        autocorrelation=autocorrelation,
        effective_dimension=float(categories),
        extension={
            "benchmark_version": BENCHMARK_VERSION,
            "execution_tier": "staged",
            "categories": categories,
            "zipf_exponent": exponent,
            "burst_probability": float(axes["burst_probability"][instance_index]),
            "drift_fraction": float(axes["drift_fraction"][instance_index]),
            "stream_model": family,
        },
    )


def build_synthetic_benchmark(
    instances_per_family: int = 128,
    master_seed: int = 20_260_821,
) -> list[ProblemSpec]:
    if instances_per_family <= 0:
        raise ValueError("instances_per_family must be positive")
    problems: list[ProblemSpec] = []
    family_index = 0
    for domain, families in SYNTHETIC_FAMILIES.items():
        for family in families:
            for instance_index in range(instances_per_family):
                if domain == "sampling":
                    problem = _sampling_problem(family, family_index, instance_index, instances_per_family, master_seed)
                elif domain == "optimization":
                    problem = _optimization_problem(family, family_index, instance_index, instances_per_family, master_seed)
                elif domain == "matrix":
                    problem = _matrix_problem(family, family_index, instance_index, instances_per_family, master_seed)
                else:
                    problem = _stream_problem(family, family_index, instance_index, instances_per_family, master_seed)
                problems.append(problem)
            family_index += 1
    return problems


def _regime(value: float | None, *, low: float, high: float, zero: str = "none") -> str:
    if value is None or value == 0.0:
        return zero
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def benchmark_index_records(problems: Sequence[ProblemSpec], folds: int = 5) -> list[dict[str, Any]]:
    if folds <= 1:
        raise ValueError("folds must be greater than one")
    family_order = {family: index for index, family in enumerate(sorted({p.problem_family for p in problems}))}
    family_offsets: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for problem in problems:
        within_family = family_offsets.get(problem.problem_family, 0)
        family_offsets[problem.problem_family] = within_family + 1
        structural = problem.to_record()
        structural.pop("problem_id")
        structural.pop("problem_seed")
        signature = hashlib.sha256(canonical_json(structural).encode("utf-8")).hexdigest()[:24]
        records.append(
            {
                "problem_id": problem.problem_id,
                "problem_family": problem.problem_family,
                "domain": problem.domain,
                "execution_tier": str(problem.extension["execution_tier"]),
                "dimension": problem.dimension,
                "structure_variant": str(
                    problem.extension.get("variant")
                    or problem.extension.get("stream_model")
                    or problem.extension.get("spectrum")
                    or problem.problem_family
                ),
                "noise_regime": _regime(problem.noise, low=0.01, high=1.0),
                "condition_regime": _regime(problem.condition_number, low=10.0, high=1_000.0),
                "instance_fold": within_family % folds,
                "family_holdout_fold": family_order[problem.problem_family] % folds,
                "design_signature": signature,
            }
        )
    return records


def materialize_matrix(problem: ProblemSpec) -> np.ndarray:
    if problem.domain != "matrix":
        raise ValueError("materialize_matrix requires a matrix problem")
    rows = int(problem.extension["rows"])
    columns = int(problem.extension["columns"])
    rank = int(problem.extension["rank"])
    rng = np.random.Generator(np.random.PCG64(problem.problem_seed))
    left, _ = np.linalg.qr(rng.normal(size=(rows, rank)))
    right, _ = np.linalg.qr(rng.normal(size=(columns, rank)))
    condition_number = float(problem.condition_number or 1.0)
    if problem.problem_family == "power_law_matrix":
        raw = np.arange(1, rank + 1, dtype=float) ** (-float(problem.spectral_decay or 1.0))
        if rank > 1:
            exponent = math.log(condition_number) / -math.log(raw[-1] / raw[0])
            singular_values = (raw / raw[0]) ** exponent
        else:
            singular_values = np.ones(1)
    else:
        singular_values = np.geomspace(1.0, 1.0 / condition_number, rank)
    matrix = (left * singular_values) @ right.T
    sparsity = float(problem.sparsity or 0.0)
    if sparsity > 0.0:
        matrix = np.where(rng.random(size=matrix.shape) < sparsity, 0.0, matrix)
    noise = float(problem.noise or 0.0)
    if noise > 0.0:
        matrix = matrix + rng.normal(0.0, noise, size=matrix.shape)
    return matrix


def materialize_stream(problem: ProblemSpec, size: int | None = None) -> np.ndarray:
    if problem.domain != "stream":
        raise ValueError("materialize_stream requires a stream problem")
    count = int(size if size is not None else problem.size or 0)
    if count <= 0:
        raise ValueError("stream size must be positive")
    categories = int(problem.extension["categories"])
    exponent = float(problem.extension["zipf_exponent"])
    rng = np.random.Generator(np.random.PCG64(problem.problem_seed))
    if problem.problem_family == "uniform_stream":
        return rng.integers(0, categories, size=count, dtype=np.int64)
    probabilities = np.arange(1, categories + 1, dtype=float) ** (-exponent)
    probabilities /= np.sum(probabilities)
    if problem.problem_family == "zipf_stream":
        return rng.choice(categories, size=count, p=probabilities).astype(np.int64)
    if problem.problem_family == "concept_drift_stream":
        change = int(count * float(problem.extension["drift_fraction"]))
        before = rng.choice(categories, size=change, p=probabilities)
        after = rng.choice(categories, size=count - change, p=probabilities[::-1])
        return np.concatenate((before, after)).astype(np.int64)
    stream = rng.choice(categories, size=count, p=probabilities).astype(np.int64)
    burst_probability = float(problem.extension["burst_probability"])
    hot = int(rng.integers(0, categories))
    for index in range(1, count):
        if rng.random() < burst_probability:
            hot = int(stream[index - 1])
        if rng.random() < burst_probability * 2.0:
            stream[index] = hot
    return stream
