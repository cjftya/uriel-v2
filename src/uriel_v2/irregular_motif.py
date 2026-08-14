from __future__ import annotations

import csv
import json
import logging
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from uriel_v2.evaluation import resolve_workers
from uriel_v2.models import Draw
from uriel_v2.motif_features import FeatureBundle, VIEW_NAMES, build_feature_bundle, prefix_standardize
from uriel_v2.provenance import execution_metadata


CANDIDATE_SIZES = (10, 15, 20, 25, 30)
MONTE_CARLO_ITERATIONS = 10_000


@dataclass(frozen=True, slots=True)
class MotifConfig:
    name: str
    query_length: int
    candidate_lengths: tuple[int, ...]
    top_k: int
    separation: int
    views: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MotifMatch:
    current_start: int
    current_end: int
    past_start: int
    past_end: int
    window_length: int
    aggregate_similarity: float
    support_count: int
    similarities: Mapping[str, float]


DEFAULT_CONFIGS = (
    MotifConfig("shape_short", 5, (4, 5, 6), 20, 30, ("raw", "grid", "circle")),
    MotifConfig("state_medium", 8, (6, 8, 10), 30, 50, ("distribution", "transition", "context")),
    MotifConfig("multiview_long", 13, (10, 13, 16), 40, 100, VIEW_NAMES),
)


def _derivative(sequence: np.ndarray) -> np.ndarray:
    if len(sequence) <= 1:
        return np.zeros_like(sequence)
    derivative = np.empty_like(sequence, dtype=float)
    derivative[0] = sequence[1] - sequence[0]
    derivative[-1] = sequence[-1] - sequence[-2]
    if len(sequence) > 2:
        derivative[1:-1] = ((sequence[1:-1] - sequence[:-2]) + (sequence[2:] - sequence[:-2]) / 2.0) / 2.0
    return derivative


def dtw_distance(left: np.ndarray, right: np.ndarray, *, derivative: bool = False) -> float:
    """Exact normalized multivariate DTW distance."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("DTW 입력은 동일 feature 폭의 2차원 배열이어야 합니다")
    if not len(left) or not len(right):
        raise ValueError("DTW 입력은 비어 있을 수 없습니다")
    if derivative:
        left = _derivative(left)
        right = _derivative(right)
    scale = math.sqrt(left.shape[1])
    previous = np.full(len(right) + 1, np.inf, dtype=float)
    previous[0] = 0.0
    for left_row in left:
        current = np.full(len(right) + 1, np.inf, dtype=float)
        for column, right_row in enumerate(right, start=1):
            cost = float(np.linalg.norm(left_row - right_row) / scale)
            current[column] = cost + min(current[column - 1], previous[column], previous[column - 1])
        previous = current
    return float(previous[-1] / (len(left) + len(right)))


def _resample(sequence: np.ndarray, length: int) -> np.ndarray:
    if len(sequence) == length:
        return sequence
    source = np.linspace(0.0, 1.0, len(sequence))
    target = np.linspace(0.0, 1.0, length)
    output = np.empty((length, sequence.shape[1]), dtype=float)
    for column in range(sequence.shape[1]):
        output[:, column] = np.interp(target, source, sequence[:, column])
    return output


def _coarse_distance(left: np.ndarray, right: np.ndarray) -> float:
    aligned = _resample(right, len(left))
    return float(np.sqrt(np.mean((left - aligned) ** 2)))


def _sequence_similarities(
    views: Mapping[str, np.ndarray],
    query_slice: slice,
    candidate_slice: slice,
    selected_views: Sequence[str],
) -> dict[str, float]:
    similarities: dict[str, float] = {}
    for view in selected_views:
        query = views[view][query_slice]
        candidate = views[view][candidate_slice]
        normalized = dtw_distance(query, candidate)
        derivative = dtw_distance(query, candidate, derivative=True)
        similarities[view] = math.exp(-(normalized + derivative) / 2.0)
    return similarities


def retrieve_motifs(bundle: FeatureBundle, query_end: int, config: MotifConfig) -> list[MotifMatch]:
    if query_end < config.query_length - 1:
        return []
    standardized = {view: prefix_standardize(bundle.views[view], query_end) for view in config.views}
    query_start = query_end - config.query_length + 1
    query_slice = slice(query_start, query_end + 1)
    latest_past_end = query_end - config.separation
    if latest_past_end < max(config.candidate_lengths) - 1:
        return []

    coarse: list[tuple[float, int, int]] = []
    for window_length in config.candidate_lengths:
        first_end = window_length - 1
        for past_end in range(first_end, latest_past_end + 1):
            past_start = past_end - window_length + 1
            distance = mean(
                _coarse_distance(
                    standardized[view][query_slice],
                    standardized[view][past_start : past_end + 1],
                )
                for view in config.views
            )
            coarse.append((distance, past_end, window_length))

    pool_size = min(len(coarse), max(config.top_k * 4, config.top_k))
    fine: list[MotifMatch] = []
    for _, past_end, window_length in sorted(coarse)[:pool_size]:
        past_start = past_end - window_length + 1
        similarities = _sequence_similarities(
            standardized,
            query_slice,
            slice(past_start, past_end + 1),
            config.views,
        )
        ordered = sorted(similarities.values(), reverse=True)
        keep = max(1, math.ceil(len(ordered) * 0.67))
        aggregate = float(mean(ordered[:keep]))
        fine.append(
            MotifMatch(
                current_start=query_start,
                current_end=query_end,
                past_start=past_start,
                past_end=past_end,
                window_length=window_length,
                aggregate_similarity=aggregate,
                support_count=sum(value >= 0.50 for value in similarities.values()),
                similarities=similarities,
            )
        )

    selected: list[MotifMatch] = []
    duplicate_radius = max(2, config.query_length // 2)
    for match in sorted(fine, key=lambda item: (-item.aggregate_similarity, item.past_end, item.window_length)):
        if any(abs(match.past_end - existing.past_end) <= duplicate_radius for existing in selected):
            continue
        selected.append(match)
        if len(selected) == config.top_k:
            break
    return selected


def _normalize_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-12:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def _candidate_scores(bundle: FeatureBundle, query_end: int, matches: Sequence[MotifMatch]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    exact = np.zeros(45, dtype=float)
    grid = np.zeros(45, dtype=float)
    circle = np.zeros(45, dtype=float)
    band = np.zeros(45, dtype=float)
    transition = np.zeros(45, dtype=float)
    current_mean = float(bundle.numbers[query_end].mean())
    candidate_numbers = np.arange(1, 46, dtype=float)
    candidate_grid = np.c_[((candidate_numbers - 1) // 7), ((candidate_numbers - 1) % 7)]
    candidate_angles = (candidate_numbers - 1) * (2.0 * math.pi / 45.0)

    for match in matches:
        followup = bundle.numbers[match.past_end + 1]
        weight = match.aggregate_similarity
        exact[followup - 1] += weight
        projected_center = current_mean + float(followup.mean() - bundle.numbers[match.past_end].mean())
        transition += weight * np.exp(-np.abs(candidate_numbers - projected_center) / 9.0)
        for number in followup:
            row, column = (number - 1) // 7, (number - 1) % 7
            manhattan = np.abs(candidate_grid[:, 0] - row) + np.abs(candidate_grid[:, 1] - column)
            grid += weight * np.exp(-manhattan / 2.0)
            angle = (number - 1) * (2.0 * math.pi / 45.0)
            angular = np.abs(candidate_angles - angle)
            angular = np.minimum(angular, 2.0 * math.pi - angular)
            circle += weight * np.exp(-angular / 0.65)
            band += weight * np.exp(-np.abs(candidate_numbers - number) / 8.0)

    components = {
        "motif_followup_frequency": _normalize_score(exact),
        "grid_region_support": _normalize_score(grid),
        "circle_region_support": _normalize_score(circle),
        "distribution_band_support": _normalize_score(band),
        "transition_support": _normalize_score(transition),
    }
    stacked = np.vstack(list(components.values()))
    agreement = np.mean(stacked >= 0.70, axis=0)
    components["cross_view_agreement"] = agreement
    final = np.mean(np.vstack((*components.values(), agreement)), axis=0)
    return final, components


def _followup_entropy(bundle: FeatureBundle, matches: Sequence[MotifMatch]) -> float:
    counts = np.zeros(45, dtype=float)
    for match in matches:
        counts[bundle.numbers[match.past_end + 1] - 1] += match.aggregate_similarity
    total = float(counts.sum())
    if total <= 0:
        return 1.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(45))


def _entropy_for_draws(draws: np.ndarray, weights: np.ndarray | None = None) -> float:
    counts = np.zeros(45, dtype=float)
    if weights is None:
        weights = np.ones(len(draws), dtype=float)
    for draw, weight in zip(draws, weights, strict=True):
        counts[np.asarray(draw, dtype=int) - 1] += float(weight)
    total = float(counts.sum())
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(45))


def _surrogate_entropies(
    bundle: FeatureBundle,
    query_end: int,
    matches: Sequence[MotifMatch],
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    count = max(1, len(matches))
    weights = np.asarray([match.aggregate_similarity for match in matches], dtype=float)
    if not len(weights):
        weights = np.ones(1, dtype=float)
    indices = np.arange(query_end + 1)
    shuffled = rng.choice(indices, size=count, replace=len(indices) < count)
    random_lotto = np.sort(np.vstack([rng.choice(np.arange(1, 46), size=6, replace=False) for _ in range(count)]), axis=1)

    block_size = int(rng.choice((3, 5, 10, 20)))
    blocks = [indices[start : start + block_size] for start in range(0, len(indices), block_size)]
    rng.shuffle(blocks)
    block_order = np.concatenate(blocks)
    block_indices = block_order[:count] if len(block_order) >= count else rng.choice(indices, size=count, replace=True)

    pools: dict[tuple[int, int, int], list[int]] = {}
    for index in indices:
        numbers = bundle.numbers[index]
        key = (int(numbers.sum()) // 30, int(np.sum(numbers % 2)), int(numbers[-1] - numbers[0]) // 10)
        pools.setdefault(key, []).append(int(index))
    preserved: list[int] = []
    for match in matches or [None]:
        source = bundle.numbers[match.past_end + 1] if match is not None else bundle.numbers[query_end]
        key = (int(source.sum()) // 30, int(np.sum(source % 2)), int(source[-1] - source[0]) // 10)
        pool = pools.get(key, indices.tolist())
        preserved.append(int(rng.choice(pool)))
    return {
        "round_shuffle": _entropy_for_draws(bundle.numbers[shuffled], weights),
        "within_round_random": _entropy_for_draws(random_lotto, weights),
        "block_shuffle": _entropy_for_draws(bundle.numbers[block_indices], weights),
        "feature_preserving": _entropy_for_draws(bundle.numbers[preserved], weights),
    }


def _surrogate_recurrence(
    bundle: FeatureBundle,
    query_end: int,
    config: MotifConfig,
    match: MotifMatch,
    seed: int,
) -> dict[str, float]:
    standardized = {view: prefix_standardize(bundle.views[view], query_end) for view in config.views}
    query_slice = slice(query_end - config.query_length + 1, query_end + 1)
    sequence_indices = np.arange(match.past_start, match.past_end + 1)
    rng = np.random.default_rng(seed)

    round_shuffle = sequence_indices.copy()
    rng.shuffle(round_shuffle)
    block_size = min(3, len(sequence_indices))
    blocks = [sequence_indices[start : start + block_size] for start in range(0, len(sequence_indices), block_size)]
    rng.shuffle(blocks)
    block_shuffle = np.concatenate(blocks)
    maximum_start = max(0, query_end - config.separation - match.window_length + 1)
    random_start = int(rng.integers(0, maximum_start + 1)) if maximum_start else 0
    random_indices = np.arange(random_start, random_start + match.window_length)

    def similarity(indices_by_view: Mapping[str, np.ndarray]) -> float:
        values: list[float] = []
        for view in config.views:
            query = standardized[view][query_slice]
            candidate = standardized[view][indices_by_view[view]]
            distance = (dtw_distance(query, candidate) + dtw_distance(query, candidate, derivative=True)) / 2.0
            values.append(math.exp(-distance))
        ordered = sorted(values, reverse=True)
        return float(mean(ordered[: max(1, math.ceil(len(ordered) * 0.67))]))

    independent: dict[str, np.ndarray] = {}
    for view in config.views:
        start = int(rng.integers(0, maximum_start + 1)) if maximum_start else 0
        independent[view] = np.arange(start, start + match.window_length)
    return {
        "round_shuffle": similarity({view: round_shuffle for view in config.views}),
        "within_round_random": similarity({view: random_indices for view in config.views}),
        "block_shuffle": similarity({view: block_shuffle for view in config.views}),
        "feature_preserving": similarity(independent),
    }


def _predict_round(bundle: FeatureBundle, target_index: int, config: MotifConfig, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    query_end = target_index - 1
    matches = retrieve_motifs(bundle, query_end, config)
    if not matches:
        raise ValueError(f"{int(bundle.rounds[target_index])}회에서 motif를 찾지 못했습니다")
    scores, components = _candidate_scores(bundle, query_end, matches)
    ranking = np.argsort(-scores, kind="stable") + 1
    winner = set(bundle.numbers[target_index].tolist())
    entropy = _followup_entropy(bundle, matches)
    cross_view = float(mean(match.support_count for match in matches))
    top_similarity = float(mean(match.aggregate_similarity for match in matches[: min(5, len(matches))]))
    confidence = top_similarity * max(0.0, 1.0 - entropy) * (cross_view / len(config.views))
    surrogates = _surrogate_entropies(bundle, query_end, matches, seed ^ int(bundle.rounds[target_index]))
    recurrence_surrogates = _surrogate_recurrence(
        bundle,
        query_end,
        config,
        matches[0],
        seed ^ int(bundle.rounds[target_index]) ^ 0x5A17,
    )
    row: dict[str, Any] = {
        "round": int(bundle.rounds[target_index]),
        "config": config.name,
        "matched_motifs": len(matches),
        "median_motif_similarity": float(np.median([match.aggregate_similarity for match in matches])),
        "best_motif_similarity": matches[0].aggregate_similarity,
        "recurrence_similarity": float(mean(match.aggregate_similarity for match in matches)),
        "cross_view_agreement_count": cross_view,
        "followup_entropy": entropy,
        "candidate_concentration": float(scores[ranking[0] - 1] / max(float(scores.sum()), 1e-12)),
        "confidence": confidence,
        "ranked_numbers": ranking.tolist(),
        "number_scores": scores.tolist(),
        **{f"surrogate_{name}_entropy": value for name, value in surrogates.items()},
        **{f"surrogate_{name}_recurrence": value for name, value in recurrence_surrogates.items()},
    }
    for size in CANDIDATE_SIZES:
        selected = set(ranking[:size].tolist())
        hits = len(winner.intersection(selected))
        row[f"recall_at_{size}"] = hits / 6.0
        row[f"hits_at_{size}"] = hits

    prediction_rows: list[dict[str, Any]] = []
    for rank, number in enumerate(ranking, start=1):
        if rank > max(CANDIDATE_SIZES):
            break
        prediction_rows.append(
            {
                "round": row["round"],
                "config": config.name,
                "rank": rank,
                "number": int(number),
                "score": float(scores[number - 1]),
                **{name: float(values[number - 1]) for name, values in components.items()},
                "is_winner": int(number in winner),
            }
        )

    match_rows = [
        {
            "target_round": row["round"],
            "config": config.name,
            "rank": rank,
            "current_start_round": int(bundle.rounds[match.current_start]),
            "current_end_round": int(bundle.rounds[match.current_end]),
            "past_start_round": int(bundle.rounds[match.past_start]),
            "past_end_round": int(bundle.rounds[match.past_end]),
            "followup_round": int(bundle.rounds[match.past_end + 1]),
            "window_length": match.window_length,
            "aggregate_similarity": match.aggregate_similarity,
            "support_count": match.support_count,
            "view_similarities": json.dumps(match.similarities, sort_keys=True),
        }
        for rank, match in enumerate(matches, start=1)
    ]
    return row, prediction_rows, match_rows


def _evaluate_config_worker(
    bundle: FeatureBundle,
    target_indices: tuple[int, ...],
    config: MotifConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for target_index in target_indices:
        row, prediction_rows, match_rows = _predict_round(bundle, target_index, config, seed)
        rows.append(row)
        predictions.extend(prediction_rows)
        matches.extend(match_rows)
    return rows, predictions, matches


def _checkpoint_payloads(path: str | Path | None, config_name: str) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoint.jsonl"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint가 없습니다: {checkpoint}")
    payloads: dict[int, dict[str, Any]] = {}
    with checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if "row" not in payload or payload["row"].get("config") != config_name:
                continue
            payloads[int(payload["row"]["round"])] = payload
    return payloads


def _checkpoint_record(
    handle: Any,
    *,
    cohort: str,
    row: dict[str, Any],
    predictions: Sequence[dict[str, Any]],
    matches: Sequence[dict[str, Any]],
) -> None:
    handle.write(
        json.dumps(
            {"cohort": cohort, "row": row, "predictions": list(predictions), "matches": list(matches)},
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()


def _evaluate_checkpointed(
    *,
    bundle: FeatureBundle,
    target_indices: Sequence[int],
    config: MotifConfig,
    seed: int,
    cohort: str,
    resumed: Mapping[int, dict[str, Any]],
    checkpoint_handle: Any,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    total = len(target_indices)
    for completed, target_index in enumerate(target_indices, start=1):
        round_no = int(bundle.rounds[target_index])
        payload = resumed.get(round_no)
        if payload is not None:
            row = dict(payload["row"])
            prediction_rows = list(payload.get("predictions", []))
            match_rows = list(payload.get("matches", []))
        else:
            row, prediction_rows, match_rows = _predict_round(bundle, target_index, config, seed)
        rows.append(row)
        predictions.extend(prediction_rows)
        matches.extend(match_rows)
        _checkpoint_record(
            checkpoint_handle,
            cohort=cohort,
            row=row,
            predictions=prediction_rows,
            matches=match_rows,
        )
        if completed == 1 or completed == total or completed % 16 == 0:
            logger.info("Motif 진행 | cohort=%s | %s/%s | round=%s", cohort, completed, total, round_no)
    return rows, predictions, matches


def _selection_score(rows: Sequence[dict[str, Any]]) -> float:
    random_hits = 6.0 * 20.0 / 45.0
    mean_hits = mean(float(row["hits_at_20"]) for row in rows)
    entropy_reduction = mean(
        mean(float(row[f"surrogate_{name}_entropy"]) for name in ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving"))
        - float(row["followup_entropy"])
        for row in rows
    )
    threshold = float(np.quantile([float(row["confidence"]) for row in rows], 0.70))
    opportunity = [row for row in rows if float(row["confidence"]) >= threshold]
    opportunity_lift = mean(float(row["hits_at_20"]) for row in opportunity) - random_hits if opportunity else -random_hits
    return float((mean_hits - random_hits) + entropy_reduction + 0.25 * opportunity_lift)


def _paired_permutation(values: np.ndarray, seed: int, iterations: int = MONTE_CARLO_ITERATIONS) -> float:
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    simulated = np.empty(iterations, dtype=float)
    for index in range(iterations):
        simulated[index] = float(np.mean(values * rng.choice((-1.0, 1.0), size=len(values))))
    return float((np.sum(simulated >= observed) + 1) / (iterations + 1))


def _bootstrap_ci(values: np.ndarray, seed: int, iterations: int = MONTE_CARLO_ITERATIONS) -> list[float]:
    rng = np.random.default_rng(seed)
    simulated = np.empty(iterations, dtype=float)
    for index in range(iterations):
        simulated[index] = float(rng.choice(values, size=len(values), replace=True).mean())
    return [float(np.quantile(simulated, 0.025)), float(np.quantile(simulated, 0.975))]


def _stable_name_seed(name: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(name))


def _random_baseline(round_count: int, candidate_size: int, observed: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    simulations = rng.hypergeometric(ngood=6, nbad=39, nsample=candidate_size, size=(MONTE_CARLO_ITERATIONS, round_count))
    simulated_mean = simulations.mean(axis=1)
    result: dict[str, Any] = {
        "iterations": MONTE_CARLO_ITERATIONS,
        "expected_mean_hits": float(simulated_mean.mean()),
        "random_mean_hits_95_interval": [
            float(np.quantile(simulated_mean, 0.025)),
            float(np.quantile(simulated_mean, 0.975)),
        ],
        "observed_mean_hits": float(observed.mean()),
        "mean_hit_lift": float(observed.mean() - simulated_mean.mean()),
        "mean_hit_p": float((np.sum(simulated_mean >= observed.mean()) + 1) / (MONTE_CARLO_ITERATIONS + 1)),
        "observed_counts": {},
        "expected_counts": {},
        "count_p": {},
    }
    for threshold in (4, 5, 6):
        observed_count = int(np.sum(observed >= threshold))
        simulated_count = np.sum(simulations >= threshold, axis=1)
        result["observed_counts"][f"{threshold}_plus"] = observed_count
        result["expected_counts"][f"{threshold}_plus"] = float(simulated_count.mean())
        result["count_p"][f"{threshold}_plus"] = float(
            (np.sum(simulated_count >= observed_count) + 1) / (MONTE_CARLO_ITERATIONS + 1)
        )
    return result


def _benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def summarize_motif_rows(
    rows: Sequence[dict[str, Any]],
    *,
    opportunity_threshold: float,
    recurrence_threshold: float,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("요약할 motif 결과가 없습니다")
    surrogate_names = ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving")
    confidence = np.asarray([float(row["confidence"]) for row in rows], dtype=float)
    opportunity = [row for row in rows if float(row["confidence"]) >= opportunity_threshold]
    result: dict[str, Any] = {
        "rounds": len(rows),
        "config": rows[0]["config"],
        "confidence": {
            "opportunity_threshold": opportunity_threshold,
            "top_10_percent_mean_hits_at_20": float(mean(float(row["hits_at_20"]) for row in rows if float(row["confidence"]) >= np.quantile(confidence, 0.90))),
            "top_20_percent_mean_hits_at_20": float(mean(float(row["hits_at_20"]) for row in rows if float(row["confidence"]) >= np.quantile(confidence, 0.80))),
            "top_30_percent_mean_hits_at_20": float(mean(float(row["hits_at_20"]) for row in rows if float(row["confidence"]) >= np.quantile(confidence, 0.70))),
        },
        "opportunity": {
            "rounds": len(opportunity),
            "coverage": len(opportunity) / len(rows),
        },
        "recurrence": {
            "threshold": recurrence_threshold,
            "actual_density": float(np.mean([float(row["recurrence_similarity"]) >= recurrence_threshold for row in rows])),
            "actual_similarity_mean": float(mean(float(row["recurrence_similarity"]) for row in rows)),
            "surrogates": {},
        },
        "followup_entropy": {
            "actual_mean": float(mean(float(row["followup_entropy"]) for row in rows)),
            "surrogates": {},
        },
        "cross_view_agreement_mean": float(mean(float(row["cross_view_agreement_count"]) for row in rows)),
        "candidate_recall": {},
    }
    for name in surrogate_names:
        surrogate_recurrence = np.asarray([float(row[f"surrogate_{name}_recurrence"]) for row in rows])
        result["recurrence"]["surrogates"][name] = {
            "density": float(np.mean(surrogate_recurrence >= recurrence_threshold)),
            "similarity_mean": float(surrogate_recurrence.mean()),
        }
        difference = np.asarray(
            [float(row[f"surrogate_{name}_entropy"]) - float(row["followup_entropy"]) for row in rows],
            dtype=float,
        )
        result["followup_entropy"]["surrogates"][name] = {
            "surrogate_mean": float(mean(float(row[f"surrogate_{name}_entropy"]) for row in rows)),
            "reduction_actual_vs_surrogate": float(difference.mean()),
            "bootstrap_95_ci": _bootstrap_ci(difference, seed ^ _stable_name_seed(name)),
            "paired_permutation_p": _paired_permutation(difference, seed ^ _stable_name_seed(name)),
        }
    adjusted = _benjamini_hochberg(
        [result["followup_entropy"]["surrogates"][name]["paired_permutation_p"] for name in surrogate_names]
    )
    for name, q_value in zip(surrogate_names, adjusted, strict=True):
        result["followup_entropy"]["surrogates"][name]["fdr_q"] = q_value
    for size in CANDIDATE_SIZES:
        observed = np.asarray([int(row[f"hits_at_{size}"]) for row in rows], dtype=int)
        baseline = _random_baseline(len(rows), size, observed, seed + size)
        opportunity_observed = np.asarray([int(row[f"hits_at_{size}"]) for row in opportunity], dtype=int)
        baseline["opportunity"] = (
            _random_baseline(len(opportunity), size, opportunity_observed, seed + size + 10_000)
            if opportunity
            else None
        )
        result["candidate_recall"][str(size)] = baseline
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        return
    fields = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key, value in output.items():
                if isinstance(value, (list, tuple, dict)):
                    output[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            writer.writerow(output)


def _walk_forward_fields() -> list[str]:
    return [
        "round", "cohort", "config", "matched_motifs", "median_motif_similarity", "best_motif_similarity",
        "recurrence_similarity", "cross_view_agreement_count", "followup_entropy", "candidate_concentration",
        "confidence", "is_opportunity", "ranked_numbers", "number_scores",
        *(f"surrogate_{name}_{metric}" for name in ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving") for metric in ("entropy", "recurrence")),
        *(field for size in CANDIDATE_SIZES for field in (f"recall_at_{size}", f"hits_at_{size}")),
        "combination_best_hit_100", "combination_best_hit_1000", "combination_best_hit_10000",
    ]


def _combination_diagnostics(bundle: FeatureBundle, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    index_combinations = np.asarray(list(combinations(range(20), 6)), dtype=np.int16)
    round_to_index = {int(round_no): index for index, round_no in enumerate(bundle.rounds)}
    output: list[dict[str, Any]] = []
    for row in rows:
        ranking = np.asarray(row["ranked_numbers"][:20], dtype=int)
        scores = np.asarray(row["number_scores"], dtype=float)[ranking - 1]
        combination_scores = scores[index_combinations].sum(axis=1)
        order = np.argsort(-combination_scores, kind="stable")
        winner = bundle.numbers[round_to_index[int(row["round"])]]
        record: dict[str, Any] = {"round": int(row["round"]), "config": row["config"]}
        for budget in (100, 1_000, 10_000):
            combinations_by_number = ranking[index_combinations[order[:budget]]]
            hits = np.isin(combinations_by_number, winner).sum(axis=1)
            best_hit = int(hits.max())
            row[f"combination_best_hit_{budget}"] = best_hit
            record[f"best_hit_{budget}"] = best_hit
            record[f"hit_4_plus_{budget}"] = int(np.sum(hits >= 4))
            record[f"hit_5_plus_{budget}"] = int(np.sum(hits >= 5))
            record[f"hit_6_{budget}"] = int(np.sum(hits == 6))
        output.append(record)
    return output


def _summarize_combinations(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for budget in (100, 1_000, 10_000):
        values = np.asarray([int(row[f"combination_best_hit_{budget}"]) for row in rows], dtype=int)
        result[str(budget)] = {
            "rounds": len(values),
            "mean_best_hit": float(values.mean()),
            "rounds_4_plus": int(np.sum(values >= 4)),
            "rounds_5_plus": int(np.sum(values >= 5)),
            "rounds_6": int(np.sum(values == 6)),
            "note": "Ranks combinations inside the frozen Top-20 by summed number scores; no combination-layer retuning.",
        }
    return result


def _plot_motif_artifacts(
    bundle: FeatureBundle,
    rows: Sequence[dict[str, Any]],
    matches: Sequence[dict[str, Any]],
    run_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    palette = {"blue": "#2f6fed", "gold": "#d7a514", "ink": "#172033", "grid": "#e4e8ef"}
    recent_start = max(0, len(bundle.rounds) - 384)
    for view in ("grid", "circle", "transition"):
        matrix = bundle.views[view][recent_start:]
        matrix = (matrix - matrix.mean(axis=0)) / np.where(matrix.std(axis=0) < 1e-9, 1.0, matrix.std(axis=0))
        distances = np.sqrt(np.maximum(0.0, ((matrix[:, None, :] - matrix[None, :, :]) ** 2).mean(axis=2)))
        recurrence = np.exp(-distances)
        figure, axis = plt.subplots(figsize=(8.8, 7.2))
        image = axis.imshow(recurrence, cmap="Blues", vmin=0.0, vmax=1.0, origin="lower", aspect="auto")
        axis.set_title(f"{view.title()} state recurrence", loc="left", color=palette["ink"], fontweight="bold")
        axis.set_xlabel("Round index (recent 384)")
        axis.set_ylabel("Round index (recent 384)")
        figure.colorbar(image, ax=axis, label="State similarity")
        figure.tight_layout()
        figure.savefig(plot_dir / f"recurrence_{view}.png", dpi=150)
        plt.close(figure)

    matrices = []
    for view in ("grid", "circle", "transition"):
        matrix = bundle.views[view][recent_start:]
        matrix = (matrix - matrix.mean(axis=0)) / np.where(matrix.std(axis=0) < 1e-9, 1.0, matrix.std(axis=0))
        distances = np.sqrt(np.maximum(0.0, ((matrix[:, None, :] - matrix[None, :, :]) ** 2).mean(axis=2)))
        matrices.append(np.exp(-distances))
    multiview = np.mean(matrices, axis=0)
    figure, axis = plt.subplots(figsize=(8.8, 7.2))
    image = axis.imshow(multiview, cmap="Blues", vmin=0.0, vmax=1.0, origin="lower", aspect="auto")
    axis.set_title("Multi-view state recurrence", loc="left", color=palette["ink"], fontweight="bold")
    axis.set_xlabel("Round index (recent 384)")
    axis.set_ylabel("Round index (recent 384)")
    figure.colorbar(image, ax=axis, label="Mean state similarity")
    figure.tight_layout()
    figure.savefig(plot_dir / "recurrence_multiview.png", dpi=150)
    plt.close(figure)

    chart_specs = (
        ("motif_distance_distribution.png", "Motif similarity distribution", "Similarity", [float(row["median_motif_similarity"]) for row in rows]),
        (
            "motif_separation_histogram.png",
            "Motif temporal separation",
            "Separation (rounds)",
            [float(row["current_end_round"]) - float(row["past_end_round"]) for row in matches],
        ),
        ("followup_entropy.png", "Conditional follow-up entropy", "Entropy", [float(row["followup_entropy"]) for row in rows]),
        ("cross_view_agreement.png", "Cross-view agreement", "Supporting views", [float(row["cross_view_agreement_count"]) for row in rows]),
    )
    for filename, title, label, values in chart_specs:
        if not values:
            continue
        figure, axis = plt.subplots(figsize=(8.8, 5.2))
        axis.hist(values, bins=20, color=palette["blue"], edgecolor="#1f4fae")
        axis.set_title(title, loc="left", color=palette["ink"], fontweight="bold")
        axis.set_xlabel(label)
        axis.set_ylabel("Walk-forward rounds")
        axis.grid(axis="y", color=palette["grid"])
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.scatter(
        [float(row["confidence"]) for row in rows],
        [int(row["hits_at_20"]) for row in rows],
        color=palette["blue"], alpha=0.65, edgecolors="none",
    )
    axis.set_title("Confidence vs candidate hits", loc="left", color=palette["ink"], fontweight="bold")
    axis.set_xlabel("Motif confidence")
    axis.set_ylabel("Hits at candidate size 20")
    axis.set_yticks(range(7))
    axis.grid(color=palette["grid"])
    figure.tight_layout()
    figure.savefig(plot_dir / "confidence_vs_candidate_recall.png", dpi=150)
    plt.close(figure)

    groups = {
        "All rounds": list(rows),
        "Opportunity": [row for row in rows if int(row["is_opportunity"]) == 1],
    }
    labels = list(groups)
    rates_4 = [mean(int(row["hits_at_20"]) >= 4 for row in groups[label]) if groups[label] else 0.0 for label in labels]
    rates_5 = [mean(int(row["hits_at_20"]) >= 5 for row in groups[label]) if groups[label] else 0.0 for label in labels]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.bar(positions - 0.18, rates_4, 0.36, label="4+ rate", color=palette["blue"])
    axis.bar(positions + 0.18, rates_5, 0.36, label="5+ rate", color=palette["gold"])
    axis.set_title("Opportunity coverage and high-hit rate", loc="left", color=palette["ink"], fontweight="bold")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Share of evaluated rounds")
    axis.set_ylim(0.0, max(0.05, max(rates_4 + rates_5) * 1.25))
    axis.legend(frameon=False)
    axis.grid(axis="y", color=palette["grid"])
    figure.tight_layout()
    figure.savefig(plot_dir / "opportunity_coverage_vs_high_hit_rate.png", dpi=150)
    plt.close(figure)


def _verdict(historical: Mapping[str, Any], development: Mapping[str, Any]) -> str:
    hist_recall = historical["candidate_recall"]["20"]
    dev_recall = development["candidate_recall"]["20"]
    hist_opp = hist_recall["opportunity"]
    dev_opp = dev_recall["opportunity"]
    entropy_names = ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving")
    hist_entropy_positive = sum(historical["followup_entropy"]["surrogates"][name]["reduction_actual_vs_surrogate"] > 0 for name in entropy_names)
    dev_entropy_positive = sum(development["followup_entropy"]["surrogates"][name]["reduction_actual_vs_surrogate"] > 0 for name in entropy_names)
    recall_direction = hist_recall["mean_hit_lift"] > 0 and dev_recall["mean_hit_lift"] > 0
    opportunity_direction = bool(
        hist_opp and dev_opp and hist_opp["mean_hit_lift"] > 0 and dev_opp["mean_hit_lift"] > 0
    )
    strong = (
        hist_entropy_positive >= 3
        and dev_entropy_positive >= 3
        and recall_direction
        and opportunity_direction
        and hist_recall["mean_hit_p"] < 0.05
        and dev_recall["mean_hit_p"] < 0.05
    )
    if strong:
        return "SUCCESS"
    # The pre-registered WEAK SIGNAL rule is intentionally directional: an
    # intermittent opportunity subset may qualify even when entropy evidence
    # is not strong enough for SUCCESS. Locked/Blind still remain sealed.
    if recall_direction and opportunity_direction:
        return "WEAK SIGNAL"
    return "NO SIGNAL"


def run_irregular_motif_experiment(
    *,
    draws: Sequence[Draw],
    start_round: int,
    end_round: int,
    split_round: int,
    experiment_seed: int,
    workers: str | int,
    run_dir: Path,
    logger: logging.Logger,
    resume_from: str | Path | None = None,
    configs: Sequence[MotifConfig] = DEFAULT_CONFIGS,
) -> dict[str, Any]:
    started_at = perf_counter()
    selected_draws = [draw for draw in draws if draw.round_no <= end_round]
    if not selected_draws or selected_draws[-1].round_no < end_round:
        raise ValueError("end_round까지의 데이터가 없습니다")
    if not (start_round < split_round <= end_round):
        raise ValueError("split_round는 평가 범위 안이어야 합니다")
    bundle = build_feature_bundle(selected_draws)
    bundle.frame().to_parquet(run_dir / "round_features.parquet", index=False)
    np.savez_compressed(
        run_dir / "round_features_cache.npz",
        rounds=bundle.rounds,
        numbers=bundle.numbers,
        grid_masks=bundle.grid_masks,
        **{f"view_{name}": bundle.views[name] for name in VIEW_NAMES},
    )

    round_to_index = {int(round_no): index for index, round_no in enumerate(bundle.rounds)}
    historical_indices = tuple(round_to_index[round_no] for round_no in range(start_round, split_round))
    development_indices = tuple(round_to_index[round_no] for round_no in range(split_round, end_round + 1))
    if not historical_indices or not development_indices:
        raise ValueError("Historical과 Development 모두 비어 있지 않아야 합니다")

    logger.info("Motif Historical 설정 선택 | configs=%s | rounds=%s", len(configs), len(historical_indices))
    worker_count = min(resolve_workers(workers), len(configs))
    config_results: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_evaluate_config_worker, bundle, historical_indices, config, experiment_seed): config
                for config in configs
            }
            for future in as_completed(future_map):
                config = future_map[future]
                config_results[config.name] = future.result()
                logger.info("Motif Historical 설정 완료 | config=%s", config.name)
    else:
        for config in configs:
            config_results[config.name] = _evaluate_config_worker(bundle, historical_indices, config, experiment_seed)
            logger.info("Motif Historical 설정 완료 | config=%s", config.name)

    selection_rows = [
        {
            "config": config.name,
            "query_length": config.query_length,
            "candidate_lengths": ";".join(map(str, config.candidate_lengths)),
            "top_k": config.top_k,
            "separation": config.separation,
            "views": ";".join(config.views),
            "historical_score": _selection_score(config_results[config.name][0]),
            "historical_mean_hits_at_20": mean(float(row["hits_at_20"]) for row in config_results[config.name][0]),
            "historical_followup_entropy": mean(float(row["followup_entropy"]) for row in config_results[config.name][0]),
        }
        for config in configs
    ]
    selected_name = max(selection_rows, key=lambda row: (float(row["historical_score"]), -int(row["query_length"]), str(row["config"])))["config"]
    selected_config = next(config for config in configs if config.name == selected_name)
    historical_rows, historical_predictions, historical_matches = config_results[selected_name]
    logger.info("Motif 설정 고정 | selected=%s | Development 재튜닝 없음", selected_name)
    resumed = _checkpoint_payloads(resume_from, selected_name)
    checkpoint_path = run_dir / "checkpoint.jsonl"
    with checkpoint_path.open("w", encoding="utf-8") as checkpoint_handle:
        historical_by_round = {int(row["round"]): row for row in historical_rows}
        predictions_by_round: dict[int, list[dict[str, Any]]] = {}
        matches_by_round: dict[int, list[dict[str, Any]]] = {}
        for row in historical_predictions:
            predictions_by_round.setdefault(int(row["round"]), []).append(row)
        for row in historical_matches:
            matches_by_round.setdefault(int(row["target_round"]), []).append(row)
        for round_no in range(start_round, split_round):
            _checkpoint_record(
                checkpoint_handle,
                cohort="Historical",
                row=historical_by_round[round_no],
                predictions=predictions_by_round.get(round_no, []),
                matches=matches_by_round.get(round_no, []),
            )
        development_rows, development_predictions, development_matches = _evaluate_checkpointed(
            bundle=bundle,
            target_indices=development_indices,
            config=selected_config,
            seed=experiment_seed,
            cohort="Development",
            resumed=resumed,
            checkpoint_handle=checkpoint_handle,
            logger=logger,
        )
    opportunity_threshold = float(np.quantile([float(row["confidence"]) for row in historical_rows], 0.70))
    surrogate_recurrence_values = np.asarray(
        [
            float(row[f"surrogate_{name}_recurrence"])
            for row in historical_rows
            for name in ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving")
        ],
        dtype=float,
    )
    recurrence_threshold = float(np.quantile(surrogate_recurrence_values, 0.95))
    all_rows = []
    for cohort, source in (("Historical", historical_rows), ("Development", development_rows)):
        for row in source:
            row["cohort"] = cohort
            row["is_opportunity"] = int(float(row["confidence"]) >= opportunity_threshold)
            all_rows.append(row)
    combination_rows = _combination_diagnostics(bundle, all_rows)
    all_predictions = [*historical_predictions, *development_predictions]
    all_matches = [*historical_matches, *development_matches]

    historical_summary = summarize_motif_rows(
        historical_rows,
        opportunity_threshold=opportunity_threshold,
        recurrence_threshold=recurrence_threshold,
        seed=experiment_seed,
    )
    development_summary = summarize_motif_rows(
        development_rows,
        opportunity_threshold=opportunity_threshold,
        recurrence_threshold=recurrence_threshold,
        seed=experiment_seed + 1,
    )
    summary: dict[str, Any] = {
        "algorithm": "Multi-scale Recurrence Motif Engine",
        "verdict": _verdict(historical_summary, development_summary),
        "selected_config": asdict(selected_config),
        "selection_policy": "Historical selection; frozen for Development; no Development retuning",
        "opportunity_policy": "Historical confidence 70th percentile; frozen for Development",
        "recurrence_policy": "Historical pooled surrogate recurrence 95th percentile; frozen for Development",
        "cohorts": {"Historical": historical_summary, "Development": development_summary},
        "combination_diagnostics": {
            "Historical": _summarize_combinations(historical_rows),
            "Development": _summarize_combinations(development_rows),
        },
        "sealed_ranges": {"Locked": "660-851: SEALED", "Additional Blind": "468-659: SEALED"},
        "monte_carlo_iterations": MONTE_CARLO_ITERATIONS,
        "execution": execution_metadata(
            draws=selected_draws,
            started_at=started_at,
            start_round=start_round,
            end_round=end_round,
        ),
    }
    _write_csv(run_dir / "config_selection.csv", selection_rows)
    _write_csv(run_dir / "recurrence_candidates.csv", all_matches)
    _write_csv(run_dir / "motif_predictions.csv", all_predictions)
    _write_csv(run_dir / "combination_diagnostics.csv", combination_rows)
    _write_csv(run_dir / "walk_forward.csv", all_rows, _walk_forward_fields())
    _write_csv(run_dir / "opportunity_rounds.csv", [row for row in all_rows if row["is_opportunity"]], _walk_forward_fields())
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_motif_artifacts(bundle, all_rows, all_matches, run_dir)
    logger.info(
        "Motif 완료 | verdict=%s | config=%s | Historical @20 lift=%.4f | Development @20 lift=%.4f",
        summary["verdict"],
        selected_name,
        historical_summary["candidate_recall"]["20"]["mean_hit_lift"],
        development_summary["candidate_recall"]["20"]["mean_hit_lift"],
    )
    return summary
