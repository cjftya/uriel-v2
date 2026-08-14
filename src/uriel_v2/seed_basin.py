from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from uriel_v2.metrics import hit_count
from uriel_v2.models import Draw
from uriel_v2.provenance import execution_metadata
from uriel_v2.rng import generate_numbers


DEFAULT_SEED_MIN = 0
DEFAULT_SEED_MAX = 1_000_000
DEFAULT_BUDGETS = (10, 100, 1_000, 10_000)
DEFAULT_RADII = (100, 1_000, 10_000, 100_000)
DEFAULT_HIT_WEIGHTS = {3: 1.0, 4: 4.0, 5: 16.0, 6: 64.0}


@dataclass(frozen=True, slots=True)
class BasinPoint:
    seed: int
    hits: int
    positional_mae: float


@dataclass(frozen=True, slots=True)
class BasinSummary:
    round_no: int
    center: float
    weighted_center: float
    width: float
    density_4_plus: float
    density_5_plus: float
    exact_6_count: int
    mean_hit: float
    max_hit: int
    entropy: float
    asymmetry: float
    nearest_5_distance: float | None
    nearest_4_distance: float
    exact_seeds: tuple[int, ...]
    scale_centers: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BasinForecast:
    delta_center: int
    state_center: int
    multi_scale_center: int
    gradient_center: int
    ensemble_centers: tuple[int, ...]


def canonical_generator_hash() -> str:
    payload = (
        "SplitMix64(seed); Fisher-Yates first 6 of 1..45; sorted; "
        "constants=9E3779B97F4A7C15,BF58476D1CE4E5B9,94D049BB133111EB"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def load_landscapes(paths: Sequence[str | Path]) -> dict[int, tuple[BasinPoint, ...]]:
    grouped: dict[int, dict[int, BasinPoint]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"round", "seed", "hits", "positional_mae"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"landscape CSV 열이 부족합니다: {path}")
            for row in reader:
                point = BasinPoint(
                    seed=int(row["seed"]),
                    hits=int(row["hits"]),
                    positional_mae=float(row["positional_mae"]),
                )
                if point.seed < DEFAULT_SEED_MIN or point.seed >= DEFAULT_SEED_MAX:
                    raise ValueError(f"seed 범위가 잘못됐습니다: {row}")
                if point.hits < 4 or point.hits > 6:
                    raise ValueError(f"basin 입력은 4~6 hit여야 합니다: {row}")
                round_no = int(row["round"])
                existing = grouped.setdefault(round_no, {}).get(point.seed)
                if existing is None or (point.hits, -point.positional_mae) > (existing.hits, -existing.positional_mae):
                    grouped[round_no][point.seed] = point
    return {
        round_no: tuple(sorted(points.values(), key=lambda point: point.seed))
        for round_no, points in sorted(grouped.items())
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    cutoff = quantile * cumulative[-1]
    return float(ordered_values[min(int(np.searchsorted(cumulative, cutoff, side="left")), len(values) - 1)])


def summarize_basin(
    round_no: int,
    points: Sequence[BasinPoint],
    *,
    hit_weights: dict[int, float] = DEFAULT_HIT_WEIGHTS,
    radii: Sequence[int] = DEFAULT_RADII,
    seed_min: int = DEFAULT_SEED_MIN,
    seed_max: int = DEFAULT_SEED_MAX,
) -> BasinSummary:
    if not points:
        raise ValueError(f"{round_no}회 basin point가 없습니다")
    seeds = np.asarray([point.seed for point in points], dtype=float)
    hits = np.asarray([point.hits for point in points], dtype=int)
    weights = np.asarray([hit_weights[point.hits] for point in points], dtype=float)
    weighted_center = float(np.average(seeds, weights=weights))
    center = _weighted_quantile(seeds, weights, 0.5)
    lower = _weighted_quantile(seeds, weights, 0.1)
    upper = _weighted_quantile(seeds, weights, 0.9)
    width = upper - lower
    bins = np.histogram(seeds, bins=np.linspace(seed_min, seed_max, 101), weights=weights)[0]
    probabilities = bins[bins > 0] / bins.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)) / math.log(len(bins)))
    left_mass = float(weights[seeds < weighted_center].sum())
    right_mass = float(weights[seeds > weighted_center].sum())
    asymmetry = (right_mass - left_mass) / float(weights.sum())
    exact_seeds = tuple(int(seed) for seed in seeds[hits == 6])
    five_seeds = seeds[hits >= 5]
    four_seeds = seeds[hits >= 4]
    nearest_5 = float(np.min(np.abs(five_seeds - weighted_center))) if len(five_seeds) else None
    nearest_4 = float(np.min(np.abs(four_seeds - weighted_center)))
    scale_centers: list[float] = []
    for radius in radii:
        mask = np.abs(seeds - weighted_center) <= radius
        scale_centers.append(float(np.average(seeds[mask], weights=weights[mask])) if np.any(mask) else weighted_center)
    return BasinSummary(
        round_no=round_no,
        center=center,
        weighted_center=weighted_center,
        width=width,
        density_4_plus=float(np.sum(hits >= 4) / (seed_max - seed_min)),
        density_5_plus=float(np.sum(hits >= 5) / (seed_max - seed_min)),
        exact_6_count=int(np.sum(hits == 6)),
        mean_hit=float(np.average(hits, weights=weights)),
        max_hit=int(hits.max()),
        entropy=entropy,
        asymmetry=asymmetry,
        nearest_5_distance=nearest_5,
        nearest_4_distance=nearest_4,
        exact_seeds=exact_seeds,
        scale_centers=tuple(scale_centers),
    )


def _summary_state(summary: BasinSummary) -> np.ndarray:
    return np.asarray(
        [
            summary.weighted_center / DEFAULT_SEED_MAX,
            summary.width / DEFAULT_SEED_MAX,
            summary.density_4_plus * 1_000,
            summary.density_5_plus * 100_000,
            summary.entropy,
            summary.asymmetry,
        ],
        dtype=float,
    )


def _clip_seed(value: float) -> int:
    return min(DEFAULT_SEED_MAX - 1, max(DEFAULT_SEED_MIN, int(round(value))))


def forecast_basin(history: Sequence[BasinSummary], minimum_history: int = 32) -> BasinForecast:
    if len(history) < minimum_history:
        raise ValueError(f"basin forecast에는 최소 {minimum_history}회 history가 필요합니다")
    centers = np.asarray([summary.weighted_center for summary in history], dtype=float)
    recent_delta = np.diff(centers)[-min(21, len(centers) - 1) :]
    delta_center = _clip_seed(centers[-1] + float(np.median(recent_delta)))

    query = _summary_state(history[-1])
    candidates: list[tuple[float, int]] = []
    for index in range(1, len(history) - 1):
        candidates.append((float(np.linalg.norm(query - _summary_state(history[index]))), index))
    nearest = sorted(candidates)[:5]
    moves = [history[index + 1].weighted_center - history[index].weighted_center for _, index in nearest]
    state_center = _clip_seed(centers[-1] + float(np.median(moves)))

    scale_predictions: list[float] = []
    scale_count = len(history[-1].scale_centers)
    for scale_index in range(scale_count):
        values = np.asarray([summary.scale_centers[scale_index] for summary in history], dtype=float)
        moves = np.diff(values)[-min(21, len(values) - 1) :]
        scale_predictions.append(values[-1] + float(np.median(moves)))
    multi_scale_center = _clip_seed(float(np.median(scale_predictions)))

    current = history[-1]
    gradient_move = current.asymmetry * min(max(current.width / 4.0, 1_000.0), 100_000.0)
    gradient_center = _clip_seed(current.weighted_center + gradient_move)
    component_centers = (delta_center, state_center, multi_scale_center, gradient_center)
    ensemble = tuple(dict.fromkeys((*component_centers, _clip_seed(float(np.median(component_centers))))))
    return BasinForecast(*component_centers, ensemble)


def seed_window_candidates(centers: Sequence[int], budget: int) -> tuple[int, ...]:
    if not centers or budget <= 0:
        raise ValueError("center와 양수 budget이 필요합니다")
    result: list[int] = []
    seen: set[int] = set()
    offset = 0
    while len(result) < budget:
        signed_offsets = (0,) if offset == 0 else (offset, -offset)
        for signed in signed_offsets:
            for center in centers:
                candidate = int(center) + signed
                if DEFAULT_SEED_MIN <= candidate < DEFAULT_SEED_MAX and candidate not in seen:
                    seen.add(candidate)
                    result.append(candidate)
                    if len(result) == budget:
                        return tuple(result)
        offset += 1
    return tuple(result)


def _nearest_distance(center: int, points: Sequence[BasinPoint], minimum_hits: int) -> int | None:
    selected = [abs(point.seed - center) for point in points if point.hits >= minimum_hits]
    return min(selected) if selected else None


def _max_seed_hit(candidates: Iterable[int], winner: Sequence[int]) -> int:
    return max(hit_count(generate_numbers(seed), winner) for seed in candidates)


def _paired_permutation(left: np.ndarray, right: np.ndarray, *, greater: bool, seed: int, iterations: int = 10_000) -> float:
    differences = left - right
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        value = float((differences * rng.choice((-1.0, 1.0), size=len(differences))).mean())
        exceed += value >= observed if greater else value <= observed
    return (exceed + 1) / (iterations + 1)


def exact_seed_basin_test(
    landscapes: dict[int, tuple[BasinPoint, ...]],
    *,
    radii: Sequence[int],
    experiment_seed: int,
) -> dict[str, Any]:
    rng = random.Random(experiment_seed ^ 0xBA51)
    samples: dict[int, tuple[list[float], list[float]]] = {radius: ([], []) for radius in radii}
    exact_seed_count = 0
    for points in landscapes.values():
        seeds = np.asarray([point.seed for point in points], dtype=int)
        exact = [point.seed for point in points if point.hits == 6]
        exact_seed_count += len(exact)
        for center in exact:
            random_center = rng.randrange(DEFAULT_SEED_MIN, DEFAULT_SEED_MAX)
            for radius in radii:
                observed = int(np.sum(np.abs(seeds - center) <= radius)) - 1
                baseline = int(np.sum(np.abs(seeds - random_center) <= radius))
                samples[radius][0].append(float(observed))
                samples[radius][1].append(float(baseline))
    result: dict[str, Any] = {"exact_seed_count": exact_seed_count, "radii": {}}
    for radius, (observed_values, baseline_values) in samples.items():
        observed = np.asarray(observed_values, dtype=float)
        baseline = np.asarray(baseline_values, dtype=float)
        result["radii"][str(radius)] = {
            "samples": len(observed),
            "neighbor_4_plus_mean": float(observed.mean()) if len(observed) else None,
            "random_window_mean": float(baseline.mean()) if len(baseline) else None,
            "mean_effect": float((observed - baseline).mean()) if len(observed) else None,
            "paired_permutation_p": (
                _paired_permutation(observed, baseline, greater=True, seed=experiment_seed + radius)
                if len(observed)
                else None
            ),
        }
    return result


def _cohort(round_no: int, split_round: int | None) -> str:
    if split_round is None:
        return "All"
    return "Historical" if round_no < split_round else "Development"


def _write_basin_summary(path: Path, summaries: Sequence[BasinSummary], radii: Sequence[int]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "round", "center", "weighted_center", "width", "density_4_plus", "density_5_plus",
                "exact_6_count", "mean_hit", "max_hit", "entropy", "asymmetry", "nearest_5_distance",
                "nearest_4_distance", "exact_seeds", *(f"center_radius_{radius}" for radius in radii),
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.round_no, f"{summary.center:.6f}", f"{summary.weighted_center:.6f}",
                    f"{summary.width:.6f}", f"{summary.density_4_plus:.9f}", f"{summary.density_5_plus:.9f}",
                    summary.exact_6_count, f"{summary.mean_hit:.6f}", summary.max_hit, f"{summary.entropy:.6f}",
                    f"{summary.asymmetry:.6f}", "" if summary.nearest_5_distance is None else summary.nearest_5_distance,
                    summary.nearest_4_distance, ";".join(map(str, summary.exact_seeds)), *summary.scale_centers,
                ]
            )


def _write_exact_seeds(path: Path, landscapes: dict[int, tuple[BasinPoint, ...]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "seed", "hits"])
        for round_no, points in landscapes.items():
            for point in points:
                if point.hits == 6:
                    writer.writerow([round_no, point.seed, point.hits])


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_predictions(rows: Sequence[dict[str, Any]], budgets: Sequence[int], seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort in sorted({row["cohort"] for row in rows}):
        selected = [row for row in rows if row["cohort"] == cohort]
        cohort_result: dict[str, Any] = {
            "rounds": len(selected),
            "center_distance": {},
            "budgets": {},
        }
        for threshold in (4, 5, 6):
            algorithm = [row[f"ensemble_nearest_{threshold}_distance"] for row in selected]
            baseline = [row[f"random_nearest_{threshold}_distance"] for row in selected]
            paired = [(float(left), float(right)) for left, right in zip(algorithm, baseline, strict=True) if left != "" and right != ""]
            if paired:
                left = np.asarray([value[0] for value in paired])
                right = np.asarray([value[1] for value in paired])
                cohort_result["center_distance"][str(threshold)] = {
                    "rounds_with_target": len(paired),
                    "algorithm_mean": float(left.mean()),
                    "random_mean": float(right.mean()),
                    "mean_effect": float((left - right).mean()),
                    "paired_permutation_p": _paired_permutation(
                        left, right, greater=False, seed=seed + threshold + len(selected)
                    ),
                }
            else:
                cohort_result["center_distance"][str(threshold)] = {"rounds_with_target": 0}
        for budget in budgets:
            algorithm = np.asarray([row[f"algorithm_max_hit_{budget}"] for row in selected], dtype=float)
            baseline = np.asarray([row[f"random_max_hit_{budget}"] for row in selected], dtype=float)
            cohort_result["budgets"][str(budget)] = {
                "algorithm_mean_max_hit": float(algorithm.mean()),
                "random_mean_max_hit": float(baseline.mean()),
                "mean_max_hit_effect": float((algorithm - baseline).mean()),
                "paired_permutation_p": _paired_permutation(
                    algorithm, baseline, greater=True, seed=seed + budget + len(selected)
                ),
                "algorithm_4_plus": int(np.sum(algorithm >= 4)),
                "random_4_plus": int(np.sum(baseline >= 4)),
                "algorithm_5_plus": int(np.sum(algorithm >= 5)),
                "random_5_plus": int(np.sum(baseline >= 5)),
                "algorithm_6": int(np.sum(algorithm >= 6)),
                "random_6": int(np.sum(baseline >= 6)),
            }
        result[cohort] = cohort_result
    return result


def _continuity_metrics(summaries: Sequence[BasinSummary]) -> dict[str, float]:
    centers = np.asarray([summary.weighted_center for summary in summaries], dtype=float)
    deltas = np.diff(centers)
    return {
        "center_lag_1_correlation": float(np.corrcoef(centers[:-1], centers[1:])[0, 1]),
        "delta_lag_1_correlation": float(np.corrcoef(deltas[:-1], deltas[1:])[0, 1]),
        "mean_absolute_center_delta": float(np.mean(np.abs(deltas))),
        "median_absolute_center_delta": float(np.median(np.abs(deltas))),
    }


def _write_plots(run_dir: Path, summaries: Sequence[BasinSummary], rows: Sequence[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/uriel-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(run_dir / name, dpi=140)
        plt.close()

    rounds = [summary.round_no for summary in summaries]
    centers = [summary.weighted_center for summary in summaries]
    plt.plot(rounds, centers, linewidth=0.8)
    plt.xlabel("Round")
    plt.ylabel("Weighted basin center")
    save("round-vs-basin-center.png")

    target_rounds = [row["round"] for row in rows]
    predicted = [row["ensemble_center"] for row in rows]
    actual = [row["actual_weighted_center"] for row in rows]
    plt.plot(target_rounds, actual, label="actual", linewidth=0.8)
    plt.plot(target_rounds, predicted, label="predicted", linewidth=0.8, alpha=0.8)
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Seed center")
    save("round-vs-predicted-basin-center.png")

    errors = [abs(row["ensemble_center"] - row["actual_weighted_center"]) for row in rows]
    plt.plot(target_rounds, errors, linewidth=0.8)
    plt.xlabel("Round")
    plt.ylabel("Center prediction error")
    save("basin-prediction-error.png")

    plt.plot(rounds, [summary.width for summary in summaries], linewidth=0.8)
    plt.xlabel("Round")
    plt.ylabel("Basin width (P90-P10)")
    save("basin-width.png")

    plt.plot(rounds, [summary.density_4_plus for summary in summaries], label="4+", linewidth=0.8)
    plt.plot(rounds, [summary.density_5_plus for summary in summaries], label="5+", linewidth=0.8)
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Seed density")
    save("basin-density.png")

    distances = [row["ensemble_nearest_4_distance"] for row in rows]
    plt.hist(distances, bins=40)
    plt.xlabel("Nearest 4+ seed distance")
    plt.ylabel("Rounds")
    save("seed-distance-distribution.png")


def run_seed_basin_experiment(
    *,
    draws: Sequence[Draw],
    landscape_paths: Sequence[str | Path],
    start_round: int,
    end_round: int,
    minimum_history: int,
    experiment_seed: int,
    split_round: int | None,
    run_dir: Path,
    logger: logging.Logger,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    radii: Sequence[int] = DEFAULT_RADII,
) -> dict[str, Any]:
    started_at = perf_counter()
    landscapes = load_landscapes(landscape_paths)
    summaries = [summarize_basin(round_no, points, radii=radii) for round_no, points in landscapes.items()]
    summary_by_round = {summary.round_no: summary for summary in summaries}
    draw_by_round = {draw.round_no: draw for draw in draws}
    _write_exact_seeds(run_dir / "exact_seeds.csv", landscapes)
    _write_basin_summary(run_dir / "basin_summary.csv", summaries, radii)

    rows: list[dict[str, Any]] = []
    targets = [
        round_no
        for round_no in range(start_round, end_round + 1)
        if round_no in landscapes and round_no in draw_by_round
    ]
    for completed, round_no in enumerate(targets, start=1):
        history = [summary for summary in summaries if summary.round_no < round_no]
        if len(history) < minimum_history:
            continue
        forecast = forecast_basin(history, minimum_history)
        actual_points = landscapes[round_no]
        actual_summary = summary_by_round[round_no]
        winner = draw_by_round[round_no].numbers
        random_rng = random.Random((experiment_seed << 17) ^ round_no)
        random_centers = tuple(random_rng.randrange(DEFAULT_SEED_MIN, DEFAULT_SEED_MAX) for _ in forecast.ensemble_centers)
        ensemble_center = int(round(median(forecast.ensemble_centers)))
        random_center = int(round(median(random_centers)))
        algorithm_candidates = seed_window_candidates(forecast.ensemble_centers, max(budgets))
        random_candidates = seed_window_candidates(random_centers, max(budgets))
        row: dict[str, Any] = {
            "cohort": _cohort(round_no, split_round),
            "round": round_no,
            "history_last_round": history[-1].round_no,
            "actual_weighted_center": actual_summary.weighted_center,
            "delta_center": forecast.delta_center,
            "state_center": forecast.state_center,
            "multi_scale_center": forecast.multi_scale_center,
            "gradient_center": forecast.gradient_center,
            "ensemble_center": ensemble_center,
            "random_center": random_center,
        }
        for name, center in (
            ("delta", forecast.delta_center),
            ("state", forecast.state_center),
            ("multi_scale", forecast.multi_scale_center),
            ("gradient", forecast.gradient_center),
            ("ensemble", ensemble_center),
            ("random", random_center),
        ):
            for threshold in (4, 5, 6):
                distance = _nearest_distance(center, actual_points, threshold)
                row[f"{name}_nearest_{threshold}_distance"] = "" if distance is None else distance
        for budget in budgets:
            row[f"algorithm_max_hit_{budget}"] = _max_seed_hit(algorithm_candidates[:budget], winner)
            row[f"random_max_hit_{budget}"] = _max_seed_hit(random_candidates[:budget], winner)
        rows.append(row)
        if completed == 1 or completed == len(targets) or completed % 32 == 0:
            logger.info(
                "Seed Basin 진행 | 회차=%s | %s/%s | nearest4=%s | maxHit@100=%s",
                round_no,
                completed,
                len(targets),
                row["ensemble_nearest_4_distance"],
                row["algorithm_max_hit_100"],
            )

    _write_rows(run_dir / "basin_predictions.csv", rows)
    _write_rows(run_dir / "walk_forward.csv", rows)
    metrics = {
        "experiment": "reverse-seed-basin-attractor",
        "execution": execution_metadata(
            draws=draws, started_at=started_at, start_round=start_round, end_round=end_round
        ),
        "warning": "target 회차 landscape는 forecast가 끝난 뒤 거리와 적중 채점에만 사용됩니다.",
        "config": {
            "start_round": start_round,
            "end_round": end_round,
            "minimum_history": minimum_history,
            "experiment_seed": experiment_seed,
            "split_round": split_round,
            "seed_min": DEFAULT_SEED_MIN,
            "seed_max": DEFAULT_SEED_MAX,
            "budgets": list(budgets),
            "radii": list(radii),
            "hit_weights": {str(key): value for key, value in DEFAULT_HIT_WEIGHTS.items()},
            "random_simulations": 10_000,
            "generator": "Uriel SplitMix64 canonical generator",
            "generator_hash": canonical_generator_hash(),
            "landscape_paths": [str(Path(path)) for path in landscape_paths],
        },
        "landscape": {
            "rounds": len(summaries),
            "points": sum(len(points) for points in landscapes.values()),
            "continuity": _continuity_metrics(summaries),
            "exact_seed_basin_test": exact_seed_basin_test(
                landscapes, radii=radii, experiment_seed=experiment_seed
            ),
        },
        "cohorts": _summarize_predictions(rows, budgets, experiment_seed),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_plots(run_dir, summaries, rows)
    return metrics
