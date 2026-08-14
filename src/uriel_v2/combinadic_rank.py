from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from uriel_v2.metrics import hit_count
from uriel_v2.models import Draw
from uriel_v2.provenance import execution_metadata


LOTTO_MAXIMUM = 45
LOTTO_COUNT = 6
TOTAL_COMBINATIONS = math.comb(LOTTO_MAXIMUM, LOTTO_COUNT)
DEFAULT_MODULI = (7, 11, 13, 17, 31, 45, 64, 127, 257, 512, 1024, 4096, 65536)
DEFAULT_BUDGETS = (10, 100, 1_000, 10_000)


@dataclass(frozen=True, slots=True)
class RankForecast:
    delta_center: int
    pattern_center: int
    modulo_center: int
    state_center: int
    ensemble_centers: tuple[int, ...]


def _validate_combination(numbers: Sequence[int]) -> tuple[int, ...]:
    ordered = tuple(sorted(int(number) for number in numbers))
    if len(ordered) != LOTTO_COUNT:
        raise ValueError("조합은 번호 6개여야 합니다")
    if len(set(ordered)) != LOTTO_COUNT:
        raise ValueError("조합 안에 중복 번호가 있습니다")
    if ordered[0] < 1 or ordered[-1] > LOTTO_MAXIMUM:
        raise ValueError("번호는 1~45 범위여야 합니다")
    return ordered


def combination_to_rank(numbers: Sequence[int]) -> int:
    """Return the zero-based lexicographic rank of a Lotto 6/45 combination."""
    ordered = _validate_combination(numbers)
    rank = 0
    previous = 0
    for index, number in enumerate(ordered):
        remaining = LOTTO_COUNT - index - 1
        for candidate in range(previous + 1, number):
            rank += math.comb(LOTTO_MAXIMUM - candidate, remaining)
        previous = number
    return rank


def rank_to_combination(rank: int) -> tuple[int, ...]:
    """Invert :func:`combination_to_rank` without enumeration."""
    if rank < 0 or rank >= TOTAL_COMBINATIONS:
        raise ValueError(f"rank는 0~{TOTAL_COMBINATIONS - 1} 범위여야 합니다")
    remainder = int(rank)
    result: list[int] = []
    previous = 0
    for index in range(LOTTO_COUNT):
        slots_left = LOTTO_COUNT - index - 1
        maximum = LOTTO_MAXIMUM - slots_left
        for candidate in range(previous + 1, maximum + 1):
            block = math.comb(LOTTO_MAXIMUM - candidate, slots_left)
            if remainder < block:
                result.append(candidate)
                previous = candidate
                break
            remainder -= block
    return tuple(result)


def circular_rank_distance(left: int, right: int) -> int:
    distance = abs(int(left) - int(right))
    return min(distance, TOTAL_COMBINATIONS - distance)


def rank_features(ranks: Sequence[int], index: int) -> dict[str, int | float]:
    rank = int(ranks[index])
    delta_1 = rank - int(ranks[index - 1]) if index >= 1 else 0
    previous_delta = int(ranks[index - 1]) - int(ranks[index - 2]) if index >= 2 else 0
    delta_2 = delta_1 - previous_delta if index >= 2 else 0
    older_delta_2 = (
        (int(ranks[index - 1]) - int(ranks[index - 2]))
        - (int(ranks[index - 2]) - int(ranks[index - 3]))
        if index >= 3
        else 0
    )
    delta_3 = delta_2 - older_delta_2 if index >= 3 else 0
    xor_value = rank ^ int(ranks[index - 1]) if index >= 1 else 0
    bit_width = TOTAL_COMBINATIONS.bit_length()
    bit_string = f"{rank:0{bit_width}b}"
    return {
        "rank": rank,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
        "abs_delta": abs(delta_1),
        "delta_sign": (delta_1 > 0) - (delta_1 < 0),
        "normalized_delta": delta_1 / TOTAL_COMBINATIONS,
        "circular_delta": circular_rank_distance(rank, int(ranks[index - 1])) if index >= 1 else 0,
        "popcount": rank.bit_count(),
        "leading_zeros": len(bit_string) - len(bit_string.lstrip("0")),
        "trailing_zeros": len(bit_string) - len(bit_string.rstrip("0")),
        "bit_transitions": sum(left != right for left, right in zip(bit_string, bit_string[1:])),
        "xor": xor_value,
        "xor_popcount": xor_value.bit_count(),
        "rotated_xor": ((xor_value << 7) | (xor_value >> (bit_width - 7))) & ((1 << bit_width) - 1),
    }


def _circular_mean(values: Sequence[int], modulus: int = TOTAL_COMBINATIONS) -> int:
    if not values:
        raise ValueError("원형 평균에 값이 필요합니다")
    angles = np.asarray(values, dtype=float) * (2.0 * math.pi / modulus)
    angle = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    if angle < 0:
        angle += 2.0 * math.pi
    return int(round(angle * modulus / (2.0 * math.pi))) % modulus


def _trimmed_mean(values: Sequence[int], proportion: float = 0.1) -> float:
    ordered = sorted(values)
    trim = min(int(len(ordered) * proportion), max(0, (len(ordered) - 1) // 2))
    selected = ordered[trim : len(ordered) - trim] if trim else ordered
    return mean(selected)


def _delta_projections(ranks: Sequence[int]) -> list[int]:
    deltas = np.diff(np.asarray(ranks, dtype=np.int64))
    recent = deltas[-min(21, len(deltas)) :]
    weights = 0.85 ** np.arange(len(recent) - 1, -1, -1, dtype=float)
    moves = [
        float(np.median(recent)),
        float(np.average(recent, weights=weights)),
        float(_trimmed_mean([int(value) for value in recent])),
    ]
    return [(int(ranks[-1]) + round(move)) % TOTAL_COMBINATIONS for move in moves]


def _sequence_distance(left: np.ndarray, right: np.ndarray, metric: str) -> float:
    scale = float(TOTAL_COMBINATIONS)
    left = left.astype(float) / scale
    right = right.astype(float) / scale
    if metric == "l1":
        return float(np.mean(np.abs(left - right)))
    if metric == "l2":
        difference = left - right
        denominator = float(np.std(np.r_[left, right])) + 1e-12
        return float(np.sqrt(np.mean(difference * difference)) / denominator)
    if metric == "cosine":
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return 1.0 - float(left @ right / denominator) if denominator else 1.0
    raise ValueError(metric)


def _pattern_projections(ranks: Sequence[int]) -> list[int]:
    deltas = np.diff(np.asarray(ranks, dtype=np.int64))
    projections: list[int] = []
    for length in (3, 5, 8, 13, 21):
        if len(deltas) < length * 2 + 1:
            continue
        query = deltas[-length:]
        candidates = range(0, len(deltas) - length)
        for metric in ("l1", "l2", "cosine"):
            best = min(candidates, key=lambda start: (_sequence_distance(query, deltas[start : start + length], metric), start))
            next_delta = int(deltas[best + length])
            projections.append((int(ranks[-1]) + next_delta) % TOTAL_COMBINATIONS)
    return projections


def _state_vector(ranks: Sequence[int], index: int) -> np.ndarray:
    current = int(ranks[index])
    delta = current - int(ranks[index - 1])
    previous_delta = int(ranks[index - 1]) - int(ranks[index - 2])
    return np.asarray(
        [
            delta / TOTAL_COMBINATIONS,
            previous_delta / TOTAL_COMBINATIONS,
            (delta - previous_delta) / TOTAL_COMBINATIONS,
            (current ^ int(ranks[index - 1])).bit_count() / TOTAL_COMBINATIONS.bit_length(),
            (current % 127) / 127.0,
            (current % 257) / 257.0,
        ],
        dtype=float,
    )


def _state_projections(ranks: Sequence[int], neighbors: int = 5) -> list[int]:
    if len(ranks) < 12:
        return _delta_projections(ranks)
    query = _state_vector(ranks, len(ranks) - 1)
    candidates: list[tuple[float, int]] = []
    for index in range(2, len(ranks) - 1):
        distance = float(np.linalg.norm(query - _state_vector(ranks, index)))
        candidates.append((distance, index))
    selected = sorted(candidates)[:neighbors]
    return [
        (int(ranks[-1]) + int(ranks[index + 1]) - int(ranks[index])) % TOTAL_COMBINATIONS
        for _, index in selected
    ]


def _modulo_consensus(projections: Sequence[int], moduli: Sequence[int] = DEFAULT_MODULI) -> list[int]:
    if not projections:
        raise ValueError("modulo consensus에 projection이 필요합니다")
    scores: list[tuple[float, int]] = []
    for projection in projections:
        score = 0.0
        for modulus in moduli:
            residues = [value % modulus for value in projections]
            residue = projection % modulus
            tolerance = max(1, round(modulus * 0.03))
            score += sum(min(abs(residue - other), modulus - abs(residue - other)) <= tolerance for other in residues)
        scores.append((score, projection))
    return [value for _, value in sorted(scores, key=lambda item: (-item[0], item[1]))[:5]]


def forecast_rank(history_ranks: Sequence[int]) -> RankForecast:
    if len(history_ranks) < 32:
        raise ValueError("rank forecast에는 최소 32회 history가 필요합니다")
    delta = _delta_projections(history_ranks)
    pattern = _pattern_projections(history_ranks)
    state = _state_projections(history_ranks)
    modulo = _modulo_consensus([*delta, *pattern, *state])
    component_centers = (
        _circular_mean(delta),
        _circular_mean(pattern),
        _circular_mean(modulo),
        _circular_mean(state),
    )
    ensemble = tuple(dict.fromkeys((*component_centers, _circular_mean(component_centers))))
    return RankForecast(*component_centers, ensemble)


def rank_window_candidates(centers: Sequence[int], budget: int) -> tuple[int, ...]:
    if budget <= 0 or not centers:
        raise ValueError("center와 양수 budget이 필요합니다")
    output: list[int] = []
    seen: set[int] = set()
    offset = 0
    while len(output) < budget:
        signed_offsets = (0,) if offset == 0 else (offset, -offset)
        for signed in signed_offsets:
            for center in centers:
                candidate = (int(center) + signed) % TOTAL_COMBINATIONS
                if candidate not in seen:
                    seen.add(candidate)
                    output.append(candidate)
                    if len(output) == budget:
                        return tuple(output)
        offset += 1
    return tuple(output)


def _max_number_hit(candidates: Iterable[int], winner: Sequence[int]) -> int:
    winner_set = set(winner)
    return max(len(winner_set.intersection(rank_to_combination(rank))) for rank in candidates)


def _paired_permutation(left: np.ndarray, right: np.ndarray, *, greater: bool, seed: int, iterations: int) -> float:
    differences = left - right
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        value = float((differences * rng.choice((-1.0, 1.0), size=len(differences))).mean())
        exceed += value >= observed if greater else value <= observed
    return (exceed + 1) / (iterations + 1)


def _bootstrap_difference(left: np.ndarray, right: np.ndarray, *, seed: int, iterations: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    differences = left - right
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        samples[index] = float(rng.choice(differences, size=len(differences), replace=True).mean())
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _cohort(round_no: int, split_round: int | None) -> str:
    if split_round is None:
        return "All"
    return "Historical" if round_no < split_round else "Development"


def _write_rank_features(path: Path, draws: Sequence[Draw], ranks: Sequence[int]) -> None:
    feature_names = list(rank_features(ranks, 0))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "numbers", *feature_names, *(f"rank_mod_{modulus}" for modulus in DEFAULT_MODULI)])
        for index, (draw, rank) in enumerate(zip(draws, ranks, strict=True)):
            features = rank_features(ranks, index)
            writer.writerow(
                [draw.round_no, "-".join(map(str, draw.numbers)), *(features[name] for name in feature_names), *(rank % modulus for modulus in DEFAULT_MODULI)]
            )


def _summarize_rows(rows: Sequence[dict[str, Any]], budgets: Sequence[int], seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    cohorts = sorted({str(row["cohort"]) for row in rows})
    for cohort in cohorts:
        selected = [row for row in rows if row["cohort"] == cohort]
        algorithm_distance = np.asarray([row["ensemble_circular_distance"] for row in selected], dtype=float)
        random_distance = np.asarray([row["random_circular_distance"] for row in selected], dtype=float)
        cohort_result: dict[str, Any] = {
            "rounds": len(selected),
            "rank_distance": {
                "algorithm_mean_absolute": mean(row["ensemble_absolute_distance"] for row in selected),
                "algorithm_mean_circular": float(algorithm_distance.mean()),
                "random_mean_circular": float(random_distance.mean()),
                "circular_effect": float(algorithm_distance.mean() - random_distance.mean()),
                "bootstrap_95_ci": _bootstrap_difference(algorithm_distance, random_distance, seed=seed + len(selected)),
                "paired_permutation_p": _paired_permutation(
                    algorithm_distance, random_distance, greater=False, seed=seed + 101 + len(selected), iterations=10_000
                ),
            },
            "component_rank_distance": {},
            "budgets": {},
        }
        for component_index, component in enumerate(("delta", "pattern", "modulo", "state", "ensemble")):
            values = np.asarray([row[f"{component}_circular_distance"] for row in selected], dtype=float)
            cohort_result["component_rank_distance"][component] = {
                "mean_circular": float(values.mean()),
                "random_mean_circular": float(random_distance.mean()),
                "effect": float(values.mean() - random_distance.mean()),
                "bootstrap_95_ci": _bootstrap_difference(
                    values, random_distance, seed=seed + 1_000 * component_index + len(selected)
                ),
                "paired_permutation_p": _paired_permutation(
                    values,
                    random_distance,
                    greater=False,
                    seed=seed + 100 * component_index + len(selected),
                    iterations=10_000,
                ),
            }
        for budget in budgets:
            algorithm = np.asarray([row[f"algorithm_max_hit_{budget}"] for row in selected], dtype=float)
            baseline = np.asarray([row[f"random_max_hit_{budget}"] for row in selected], dtype=float)
            cohort_result["budgets"][str(budget)] = {
                "algorithm_mean_max_hit": float(algorithm.mean()),
                "random_mean_max_hit": float(baseline.mean()),
                "mean_max_hit_effect": float((algorithm - baseline).mean()),
                "paired_permutation_p": _paired_permutation(
                    algorithm, baseline, greater=True, seed=seed + budget + len(selected), iterations=10_000
                ),
                "algorithm_4_plus": int(np.sum(algorithm >= 4)),
                "random_4_plus": int(np.sum(baseline >= 4)),
                "algorithm_5_plus": int(np.sum(algorithm >= 5)),
                "random_5_plus": int(np.sum(baseline >= 5)),
                "algorithm_6": int(np.sum(algorithm >= 6)),
                "random_6": int(np.sum(baseline >= 6)),
                "exact_rank_window_hits": int(sum(row[f"rank_window_hit_{budget}"] for row in selected)),
            }
        result[cohort] = cohort_result
    return result


def _write_predictions(path: Path, rows: Sequence[dict[str, Any]], budgets: Sequence[int]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(run_dir: Path, draws: Sequence[Draw], ranks: Sequence[int], rows: Sequence[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/uriel-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(run_dir / name, dpi=140)
        plt.close()

    rounds = [draw.round_no for draw in draws]
    plt.plot(rounds, ranks, linewidth=0.8)
    plt.xlabel("Round")
    plt.ylabel("Combinadic rank")
    save("round-vs-rank.png")

    plt.plot(rounds[1:], np.diff(ranks), linewidth=0.7)
    plt.xlabel("Round")
    plt.ylabel("Delta rank")
    save("round-vs-delta-rank.png")

    actual = [row["actual_rank"] for row in rows]
    predicted = [row["ensemble_center"] for row in rows]
    plt.scatter(actual, predicted, s=8, alpha=0.5)
    plt.plot([0, TOTAL_COMBINATIONS], [0, TOTAL_COMBINATIONS], linestyle="--", linewidth=0.8)
    plt.xlabel("Actual rank")
    plt.ylabel("Predicted rank")
    save("predicted-vs-actual-rank.png")

    errors = [row["ensemble_circular_distance"] for row in rows]
    target_rounds = [row["round"] for row in rows]
    plt.plot(target_rounds, errors, linewidth=0.8)
    plt.xlabel("Round")
    plt.ylabel("Circular rank error")
    save("rank-prediction-error.png")

    window = min(32, len(errors))
    rolling = np.convolve(errors, np.ones(window) / window, mode="valid")
    plt.plot(target_rounds[window - 1 :], rolling, linewidth=1.0)
    plt.xlabel("Round")
    plt.ylabel("Rolling 32-round error")
    save("rolling-rank-error.png")


def run_combinadic_experiment(
    *,
    draws: Sequence[Draw],
    start_round: int,
    end_round: int,
    minimum_history: int,
    experiment_seed: int,
    split_round: int | None,
    run_dir: Path,
    logger: logging.Logger,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    started_at = perf_counter()
    draw_by_round = {draw.round_no: draw for draw in draws}
    ranks = [combination_to_rank(draw.numbers) for draw in draws]
    rank_by_round = dict(zip((draw.round_no for draw in draws), ranks, strict=True))
    _write_rank_features(run_dir / "ranks.csv", draws, ranks)

    rows: list[dict[str, Any]] = []
    targets = [round_no for round_no in range(start_round, end_round + 1) if round_no in draw_by_round]
    for completed, round_no in enumerate(targets, start=1):
        history_draws = [draw for draw in draws if draw.round_no < round_no]
        if len(history_draws) < minimum_history:
            continue
        history_ranks = [rank_by_round[draw.round_no] for draw in history_draws]
        forecast = forecast_rank(history_ranks)
        actual_rank = rank_by_round[round_no]
        winner = draw_by_round[round_no].numbers
        random_rng = random.Random((experiment_seed << 16) ^ round_no)
        random_centers = tuple(random_rng.randrange(TOTAL_COMBINATIONS) for _ in forecast.ensemble_centers)
        ensemble_center = _circular_mean(forecast.ensemble_centers)
        random_center = _circular_mean(random_centers)
        maximum_budget = max(budgets)
        algorithm_candidates = rank_window_candidates(forecast.ensemble_centers, maximum_budget)
        random_candidates = rank_window_candidates(random_centers, maximum_budget)
        row: dict[str, Any] = {
            "cohort": _cohort(round_no, split_round),
            "round": round_no,
            "history_last_round": history_draws[-1].round_no,
            "actual_rank": actual_rank,
            "delta_center": forecast.delta_center,
            "pattern_center": forecast.pattern_center,
            "modulo_center": forecast.modulo_center,
            "state_center": forecast.state_center,
            "ensemble_center": ensemble_center,
            "random_center": random_center,
        }
        for name, center in (
            ("delta", forecast.delta_center),
            ("pattern", forecast.pattern_center),
            ("modulo", forecast.modulo_center),
            ("state", forecast.state_center),
            ("ensemble", ensemble_center),
        ):
            row[f"{name}_absolute_distance"] = abs(center - actual_rank)
            row[f"{name}_circular_distance"] = circular_rank_distance(center, actual_rank)
        row["random_circular_distance"] = circular_rank_distance(random_center, actual_rank)
        row["ensemble_percentile_distance"] = row["ensemble_circular_distance"] / (TOTAL_COMBINATIONS / 2)
        for budget in budgets:
            algorithm_slice = algorithm_candidates[:budget]
            random_slice = random_candidates[:budget]
            row[f"rank_window_hit_{budget}"] = int(actual_rank in algorithm_slice)
            row[f"algorithm_max_hit_{budget}"] = _max_number_hit(algorithm_slice, winner)
            row[f"random_max_hit_{budget}"] = _max_number_hit(random_slice, winner)
        rows.append(row)
        if completed == 1 or completed == len(targets) or completed % 32 == 0:
            logger.info(
                "Combinadic 진행 | 회차=%s | %s/%s | circular error=%s | maxHit@100=%s",
                round_no,
                completed,
                len(targets),
                row["ensemble_circular_distance"],
                row["algorithm_max_hit_100"],
            )

    _write_predictions(run_dir / "predictions.csv", rows, budgets)
    _write_predictions(run_dir / "walk_forward.csv", rows, budgets)
    summary = {
        "experiment": "combinadic-rank-dynamics",
        "execution": execution_metadata(
            draws=draws, started_at=started_at, start_round=start_round, end_round=end_round
        ),
        "config": {
            "start_round": start_round,
            "end_round": end_round,
            "minimum_history": minimum_history,
            "experiment_seed": experiment_seed,
            "split_round": split_round,
            "moduli": list(DEFAULT_MODULI),
            "budgets": list(budgets),
            "total_combinations": TOTAL_COMBINATIONS,
            "random_simulations": 10_000,
        },
        "cohorts": _summarize_rows(rows, budgets, experiment_seed),
        "random_rank_distance_monte_carlo": {},
    }
    rng = np.random.default_rng(experiment_seed)
    left = rng.integers(0, TOTAL_COMBINATIONS, size=10_000, dtype=np.int64)
    right = rng.integers(0, TOTAL_COMBINATIONS, size=10_000, dtype=np.int64)
    distances = np.minimum(np.abs(left - right), TOTAL_COMBINATIONS - np.abs(left - right))
    summary["random_rank_distance_monte_carlo"] = {
        "iterations": 10_000,
        "mean_circular_distance": float(distances.mean()),
        "p2_5": float(np.quantile(distances, 0.025)),
        "p50": float(np.quantile(distances, 0.5)),
        "p97_5": float(np.quantile(distances, 0.975)),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_plots(run_dir, draws, ranks, rows)
    return summary
