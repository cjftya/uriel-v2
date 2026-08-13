from __future__ import annotations

import bisect
import math
import random
from statistics import mean
from typing import Any


def hypergeometric_hit_probabilities() -> tuple[float, ...]:
    """Return P(exactly k hits) for two independent Lotto 6/45 sets."""
    denominator = math.comb(45, 6)
    return tuple(
        math.comb(6, hits) * math.comb(39, 6 - hits) / denominator
        for hits in range(7)
    )


def maximum_hit_probabilities(budget: int) -> tuple[float, ...]:
    """Return the exact IID baseline distribution for the best of *budget* sets."""
    if budget <= 0:
        raise ValueError("budget은 양수여야 합니다")
    single = hypergeometric_hit_probabilities()
    cumulative = 0.0
    previous = 0.0
    result: list[float] = []
    for probability in single:
        cumulative += probability
        current = cumulative**budget
        result.append(max(0.0, current - previous))
        previous = current
    total = sum(result)
    result[-1] += 1.0 - total
    return tuple(result)


def at_least_one_probability(*, minimum_hits: int, budget: int) -> float:
    if minimum_hits < 0 or minimum_hits > 6:
        raise ValueError("minimum_hits는 0~6이어야 합니다")
    if budget <= 0:
        raise ValueError("budget은 양수여야 합니다")
    below = sum(hypergeometric_hit_probabilities()[:minimum_hits])
    return 1.0 - below**budget


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("빈 값의 백분위수는 계산할 수 없습니다")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def monte_carlo_round_max_baseline(
    *,
    budget: int,
    rounds: int,
    iterations: int = 10_000,
    seed: int = 20_260_813,
) -> dict[str, Any]:
    """Simulate equal-budget round maxima from the exact IID max distribution."""
    if rounds <= 0 or iterations <= 0:
        raise ValueError("rounds와 iterations는 양수여야 합니다")

    maximum = maximum_hit_probabilities(budget)
    cumulative: list[float] = []
    running = 0.0
    for probability in maximum:
        running += probability
        cumulative.append(running)
    cumulative[-1] = 1.0

    rng = random.Random(seed)
    exact_6_counts: list[float] = []
    hit_5_plus_counts: list[float] = []
    hit_4_plus_counts: list[float] = []
    mean_best_hits: list[float] = []

    for _ in range(iterations):
        maxima = [bisect.bisect_left(cumulative, rng.random()) for _ in range(rounds)]
        exact_6_counts.append(float(sum(hit == 6 for hit in maxima)))
        hit_5_plus_counts.append(float(sum(hit >= 5 for hit in maxima)))
        hit_4_plus_counts.append(float(sum(hit >= 4 for hit in maxima)))
        mean_best_hits.append(mean(maxima))

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "mean": mean(values),
            "p2_5": _percentile(values, 0.025),
            "p50": _percentile(values, 0.5),
            "p97_5": _percentile(values, 0.975),
        }

    return {
        "method": "exact IID maximum distribution sampled with a fixed PRNG seed",
        "budget_per_round": budget,
        "rounds_per_iteration": rounds,
        "iterations": iterations,
        "seed": seed,
        "best_hit_probability": {str(hit): probability for hit, probability in enumerate(maximum)},
        "exact_6_rounds": summarize(exact_6_counts),
        "hit_5_plus_rounds": summarize(hit_5_plus_counts),
        "hit_4_plus_rounds": summarize(hit_4_plus_counts),
        "mean_best_hit": summarize(mean_best_hits),
    }
