from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from uriel_v2.metrics import hit_count
from uriel_v2.models import Draw
from uriel_v2.rng import generate_numbers


SEED_SPACE = 1_000_000
BIT_WIDTH = 20
BIT_MASK = (1 << BIT_WIDTH) - 1
UINT64_MASK = np.uint64((1 << 64) - 1)
MODEL_NAMES = ("persistence", "ewma", "analog", "shift", "xor", "ensemble", "random")


@dataclass(frozen=True, slots=True)
class SeedFieldConfig:
    seed_space: int = SEED_SPACE
    minimum_history: int = 32
    ewma_window: int = 16
    ewma_decay: float = 0.90
    analog_neighbors: int = 5
    resolutions: tuple[int, ...] = (64, 256)
    transforms: tuple[str, ...] = ("identity", "gray", "bit_reverse", "rotate7", "deinterleave")
    prime_moduli: tuple[int, ...] = (257, 509)
    budgets: tuple[int, ...] = (10, 100, 1_000, 10_000)
    hit_weights: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 1.0, 8.0, 64.0)
    mae_penalty: float = 1.0
    random_seed: int = 20_260_815
    null_iterations: int = 10_000

    def validate(self) -> None:
        if self.seed_space != SEED_SPACE:
            raise ValueError("Stage B는 고정 seed 공간 [0, 1,000,000)을 사용합니다")
        if self.minimum_history < 2 or self.analog_neighbors < 1:
            raise ValueError("minimum_history와 analog_neighbors가 올바르지 않습니다")
        if any(value <= 0 or value > self.seed_space for value in self.budgets):
            raise ValueError("budget은 seed 공간 안의 양수여야 합니다")
        if any(resolution <= 1 or resolution & (resolution - 1) for resolution in self.resolutions):
            raise ValueError("resolution은 2의 거듭제곱이어야 합니다")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LandscapePoint:
    seed: int
    hits: int
    positional_mae: float


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    name: str
    size: int
    mapping: np.ndarray
    xor_capable: bool


@dataclass(frozen=True, slots=True)
class FieldEvaluationRow:
    cohort: str
    round_no: int
    model: str
    budget: int
    selected_hit_4: int
    selected_hit_5: int
    selected_hit_6: int
    top10_max_hit: int
    top10_hit_distribution: tuple[int, ...]

    @property
    def selected_hit_4_plus(self) -> int:
        return self.selected_hit_4 + self.selected_hit_5 + self.selected_hit_6

    @property
    def selected_hit_5_plus(self) -> int:
        return self.selected_hit_5 + self.selected_hit_6


@dataclass(frozen=True, slots=True)
class SeedPredictionRow:
    cohort: str
    round_no: int
    model: str
    rank: int
    seed: int
    field_score: float
    numbers: tuple[int, ...]
    hits: int


@dataclass(frozen=True, slots=True)
class SeedForecastRow:
    target_round: int
    model: str
    rank: int
    seed: int
    field_score: float
    numbers: tuple[int, ...]


def _reverse_bits(values: np.ndarray, width: int = BIT_WIDTH) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.uint32)
    working = values.astype(np.uint32, copy=True)
    for _ in range(width):
        result = (result << 1) | (working & 1)
        working >>= 1
    return result


def _deinterleave_bits(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint32, copy=False)
    even = np.zeros_like(values)
    odd = np.zeros_like(values)
    for position in range(10):
        even |= ((values >> (2 * position)) & 1) << position
        odd |= ((values >> (2 * position + 1)) & 1) << position
    return even | (odd << 10)


def transform_values(values: np.ndarray, name: str) -> np.ndarray:
    values = values.astype(np.uint32, copy=False)
    if name == "identity":
        return values.copy()
    if name == "gray":
        return values ^ (values >> 1)
    if name == "bit_reverse":
        return _reverse_bits(values)
    if name == "rotate7":
        return ((values << 7) | (values >> (BIT_WIDTH - 7))) & BIT_MASK
    if name == "deinterleave":
        return _deinterleave_bits(values)
    raise ValueError(f"알 수 없는 seed field transform: {name}")


def build_channels(config: SeedFieldConfig) -> tuple[ChannelSpec, ...]:
    config.validate()
    seeds = np.arange(config.seed_space, dtype=np.uint32)
    channels: list[ChannelSpec] = []
    coordinate_size = 1 << BIT_WIDTH
    for transform in config.transforms:
        transformed = transform_values(seeds, transform)
        for resolution in config.resolutions:
            mapping = ((transformed.astype(np.uint64) * resolution) // coordinate_size).astype(np.int32)
            channels.append(ChannelSpec(f"{transform}-{resolution}", resolution, mapping, True))
    for modulus in config.prime_moduli:
        channels.append(ChannelSpec(f"mod-{modulus}", modulus, (seeds % modulus).astype(np.int32), False))
    return tuple(channels)


def load_landscapes(paths: Iterable[str | Path]) -> dict[int, tuple[LandscapePoint, ...]]:
    grouped: dict[int, dict[int, LandscapePoint]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"round", "seed", "hits", "positional_mae"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"landscape CSV 열이 부족합니다: {path}")
            for row in reader:
                point = LandscapePoint(
                    seed=int(row["seed"]),
                    hits=int(row["hits"]),
                    positional_mae=float(row["positional_mae"]),
                )
                if point.hits < 4 or point.seed < 0 or point.seed >= SEED_SPACE:
                    raise ValueError(f"잘못된 landscape 행: {row}")
                round_no = int(row["round"])
                existing = grouped.setdefault(round_no, {}).get(point.seed)
                if existing is None or (point.hits, -point.positional_mae) > (existing.hits, -existing.positional_mae):
                    grouped[round_no][point.seed] = point
    return {
        round_no: tuple(sorted(points.values(), key=lambda point: point.seed))
        for round_no, points in sorted(grouped.items())
    }


def build_fields(
    landscapes: dict[int, tuple[LandscapePoint, ...]],
    channels: Sequence[ChannelSpec],
    config: SeedFieldConfig,
) -> tuple[tuple[int, ...], tuple[tuple[np.ndarray, ...], ...]]:
    rounds = tuple(sorted(landscapes))
    fields: list[tuple[np.ndarray, ...]] = []
    for round_no in rounds:
        points = landscapes[round_no]
        seeds = np.fromiter((point.seed for point in points), dtype=np.int64)
        weights = np.fromiter(
            (
                config.hit_weights[point.hits] / (1.0 + config.mae_penalty * point.positional_mae)
                for point in points
            ),
            dtype=np.float64,
        )
        round_fields: list[np.ndarray] = []
        for channel in channels:
            histogram = np.bincount(channel.mapping[seeds], weights=weights, minlength=channel.size).astype(np.float64)
            total = float(histogram.sum())
            if total <= 0:
                raise ValueError(f"{round_no}회 field 질량이 0입니다")
            round_fields.append(histogram / total)
        fields.append(tuple(round_fields))
    return rounds, tuple(fields)


def _weighted_average(field_rows: Sequence[Sequence[np.ndarray]], weights: np.ndarray) -> tuple[np.ndarray, ...]:
    normalized = weights / weights.sum()
    return tuple(
        np.sum(np.stack([row[channel] for row in field_rows]) * normalized[:, None], axis=0)
        for channel in range(len(field_rows[0]))
    )


def _field_signature(field: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([values * math.sqrt(len(values)) for values in field])


def _best_shift(previous: np.ndarray, current: np.ndarray) -> int:
    correlation = np.fft.ifft(np.conj(np.fft.fft(previous)) * np.fft.fft(current)).real
    return int(np.argmax(correlation))


def _xor_indices(size: int) -> np.ndarray:
    values = np.arange(size, dtype=np.int32)
    return values[:, None] ^ values[None, :]


def _best_xor_mask(previous: np.ndarray, current: np.ndarray, table: np.ndarray) -> int:
    scores = previous[table] @ current
    return int(np.argmax(scores))


def forecast_fields(
    fields: Sequence[Sequence[np.ndarray]],
    current_index: int,
    channels: Sequence[ChannelSpec],
    config: SeedFieldConfig,
    xor_tables: dict[int, np.ndarray] | None = None,
) -> dict[str, tuple[np.ndarray, ...]]:
    if current_index < config.minimum_history - 1:
        raise ValueError("field forecast에 필요한 과거가 부족합니다")
    xor_tables = xor_tables or {
        channel.size: _xor_indices(channel.size) for channel in channels if channel.xor_capable
    }
    current = tuple(fields[current_index])
    previous = tuple(fields[current_index - 1])

    start = max(0, current_index - config.ewma_window + 1)
    ewma_rows = fields[start : current_index + 1]
    ewma_weights = config.ewma_decay ** np.arange(len(ewma_rows) - 1, -1, -1, dtype=float)
    ewma = _weighted_average(ewma_rows, ewma_weights)

    current_signature = _field_signature(current)
    candidate_indices = list(range(0, current_index))
    candidate_signatures = np.stack([_field_signature(fields[index]) for index in candidate_indices])
    norms = np.linalg.norm(candidate_signatures, axis=1) * np.linalg.norm(current_signature)
    cosine = np.divide(candidate_signatures @ current_signature, norms, out=np.zeros_like(norms), where=norms > 0)
    distances = 1.0 - cosine
    neighbor_positions = np.argsort(distances, kind="stable")[: config.analog_neighbors]
    neighbors = [candidate_indices[int(position)] + 1 for position in neighbor_positions]
    analog_weights = 1.0 / (distances[neighbor_positions] + 0.01)
    analog = _weighted_average([fields[index] for index in neighbors], analog_weights)

    shifts = [_best_shift(left, right) for left, right in zip(previous, current, strict=True)]
    shift = tuple(np.roll(values, amount) for values, amount in zip(current, shifts, strict=True))

    xor_values: list[np.ndarray] = []
    for channel, left, right in zip(channels, previous, current, strict=True):
        if channel.xor_capable:
            mask = _best_xor_mask(left, right, xor_tables[channel.size])
            xor_values.append(right[xor_tables[channel.size][mask]])
        else:
            xor_values.append(right.copy())
    xor = tuple(xor_values)

    ensemble = tuple(
        (ewma[index] + analog[index] + shift[index] + xor[index]) / 4.0
        for index in range(len(channels))
    )
    return {
        "persistence": current,
        "ewma": ewma,
        "analog": analog,
        "shift": shift,
        "xor": xor,
        "ensemble": ensemble,
    }


def _splitmix_hashes(seed_space: int, round_no: int, namespace: int) -> np.ndarray:
    values = np.arange(seed_space, dtype=np.uint64)
    values ^= np.uint64((round_no * 0x9E3779B1 + namespace * 0x85EBCA77) & ((1 << 64) - 1))
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & UINT64_MASK
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & UINT64_MASK
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & UINT64_MASK
    return values ^ (values >> np.uint64(31))


def score_seed_space(prediction: Sequence[np.ndarray], channels: Sequence[ChannelSpec]) -> np.ndarray:
    if len(prediction) != len(channels):
        raise ValueError("prediction과 channel 개수가 다릅니다")
    score = np.zeros(SEED_SPACE, dtype=np.float64)
    for values, channel in zip(prediction, channels, strict=True):
        score += values[channel.mapping] * channel.size
    return score / len(channels)


def select_top_seeds(score: np.ndarray, round_no: int, budget: int, namespace: int = 0) -> np.ndarray:
    if budget <= 0 or budget > len(score):
        raise ValueError("잘못된 seed selection budget입니다")
    hashes = _splitmix_hashes(len(score), round_no, namespace)
    if np.all(score == score[0]):
        candidates = np.argpartition(hashes, len(hashes) - budget)[-budget:]
        order = np.argsort(hashes[candidates], kind="stable")[::-1]
        return candidates[order].astype(np.int64)
    candidates = np.argpartition(score, len(score) - budget)[-budget:]
    order = np.lexsort((hashes[candidates], score[candidates]))[::-1]
    return candidates[order].astype(np.int64)


def evaluate_seed_field(
    *,
    draws: Sequence[Draw],
    landscapes: dict[int, tuple[LandscapePoint, ...]],
    start_round: int,
    end_round: int,
    cohort: str,
    config: SeedFieldConfig | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> tuple[list[FieldEvaluationRow], list[SeedPredictionRow], dict[str, Any]]:
    config = config or SeedFieldConfig()
    config.validate()
    channels = build_channels(config)
    rounds, fields = build_fields(landscapes, channels, config)
    round_index = {round_no: index for index, round_no in enumerate(rounds)}
    draw_map = {draw.round_no: draw for draw in draws}
    xor_tables = {channel.size: _xor_indices(channel.size) for channel in channels if channel.xor_capable}
    rows: list[FieldEvaluationRow] = []
    prediction_rows: list[SeedPredictionRow] = []
    total_rounds = end_round - start_round + 1

    for completed, target_round in enumerate(range(start_round, end_round + 1), start=1):
        if target_round not in round_index or target_round not in draw_map:
            raise ValueError(f"평가 회차가 field 또는 당첨 데이터에 없습니다: {target_round}")
        target_index = round_index[target_round]
        current_index = target_index - 1
        if current_index < config.minimum_history - 1:
            raise ValueError(f"{target_round}회 이전 warm-up field가 부족합니다")
        if rounds[current_index] != target_round - 1:
            raise ValueError(f"{target_round}회 직전 field가 연속적이지 않습니다")

        predictions = forecast_fields(fields, current_index, channels, config, xor_tables)
        max_budget = max(config.budgets)
        selected_by_model: dict[str, np.ndarray] = {}
        score_by_model: dict[str, np.ndarray] = {}
        for model, prediction in predictions.items():
            score_by_model[model] = score_seed_space(prediction, channels)
            selected_by_model[model] = select_top_seeds(
                score_by_model[model], target_round, max_budget, namespace=0
            )
        score_by_model["random"] = np.zeros(config.seed_space, dtype=np.float64)
        selected_by_model["random"] = select_top_seeds(
            score_by_model["random"], target_round, max_budget, namespace=config.random_seed
        )

        target_hits = {point.seed: point.hits for point in landscapes[target_round]}
        winner = draw_map[target_round].numbers
        top10_distributions: dict[str, tuple[int, ...]] = {}
        top10_maxima: dict[str, int] = {}
        for model, selected in selected_by_model.items():
            top10_hits: list[int] = []
            for rank, seed_value in enumerate(selected[:10], start=1):
                seed = int(seed_value)
                numbers = generate_numbers(seed)
                hits = hit_count(numbers, winner)
                top10_hits.append(hits)
                prediction_rows.append(
                    SeedPredictionRow(
                        cohort=cohort,
                        round_no=target_round,
                        model=model,
                        rank=rank,
                        seed=seed,
                        field_score=float(score_by_model[model][seed]),
                        numbers=numbers,
                        hits=hits,
                    )
                )
            top10_distributions[model] = tuple(top10_hits.count(hit) for hit in range(7))
            top10_maxima[model] = max(top10_hits)

        for model, selected in selected_by_model.items():
            selected_hits = np.fromiter((target_hits.get(int(seed), 0) for seed in selected), dtype=np.int8)
            for budget in config.budgets:
                subset = selected_hits[:budget]
                rows.append(
                    FieldEvaluationRow(
                        cohort=cohort,
                        round_no=target_round,
                        model=model,
                        budget=budget,
                        selected_hit_4=int(np.sum(subset == 4)),
                        selected_hit_5=int(np.sum(subset == 5)),
                        selected_hit_6=int(np.sum(subset == 6)),
                        top10_max_hit=top10_maxima[model],
                        top10_hit_distribution=top10_distributions[model],
                    )
                )

        if progress is not None:
            progress(completed, total_rounds, target_round)

    summary = summarize_seed_field(rows, landscapes, config)
    summary["cohort"] = cohort
    summary["start_round"] = start_round
    summary["end_round"] = end_round
    summary["rounds"] = end_round - start_round + 1
    summary["config"] = asdict(config)
    summary["config_fingerprint"] = config.fingerprint
    summary["channels"] = [channel.name for channel in channels]
    return rows, prediction_rows, summary


def predict_seed_field(
    *,
    landscapes: dict[int, tuple[LandscapePoint, ...]],
    target_round: int,
    top_k: int,
    config: SeedFieldConfig | None = None,
) -> tuple[list[SeedForecastRow], dict[str, Any]]:
    config = config or SeedFieldConfig()
    config.validate()
    if top_k <= 0 or top_k > config.seed_space:
        raise ValueError("top_k는 seed 공간 안의 양수여야 합니다")

    channels = build_channels(config)
    rounds, fields = build_fields(landscapes, channels, config)
    round_index = {round_no: index for index, round_no in enumerate(rounds)}
    previous_round = target_round - 1
    if previous_round not in round_index:
        raise ValueError(f"{target_round}회 예측에 필요한 {previous_round}회 field가 없습니다")
    current_index = round_index[previous_round]
    if current_index < config.minimum_history - 1:
        raise ValueError(f"{target_round}회 예측에 필요한 warm-up field가 부족합니다")
    if tuple(rounds[current_index - config.minimum_history + 1 : current_index + 1]) != tuple(
        range(target_round - config.minimum_history, target_round)
    ):
        raise ValueError(f"{target_round}회 직전 field가 연속적이지 않습니다")

    xor_tables = {channel.size: _xor_indices(channel.size) for channel in channels if channel.xor_capable}
    predictions = forecast_fields(fields, current_index, channels, config, xor_tables)
    scored: dict[str, np.ndarray] = {
        model: score_seed_space(prediction, channels) for model, prediction in predictions.items()
    }
    scored["random"] = np.zeros(config.seed_space, dtype=np.float64)

    rows: list[SeedForecastRow] = []
    for model in MODEL_NAMES:
        namespace = config.random_seed if model == "random" else 0
        selected = select_top_seeds(scored[model], target_round, top_k, namespace=namespace)
        for rank, seed_value in enumerate(selected, start=1):
            seed = int(seed_value)
            rows.append(
                SeedForecastRow(
                    target_round=target_round,
                    model=model,
                    rank=rank,
                    seed=seed,
                    field_score=float(scored[model][seed]),
                    numbers=generate_numbers(seed),
                )
            )
    metadata = {
        "target_round": target_round,
        "top_k": top_k,
        "config": asdict(config),
        "config_fingerprint": config.fingerprint,
        "last_observed_round": previous_round,
        "channels": [channel.name for channel in channels],
        "warning": "This is a candidate ranking, not evidence that lottery draws are predictable.",
    }
    return rows, metadata


def _null_distribution(
    landscapes: dict[int, tuple[LandscapePoint, ...]],
    rounds: Sequence[int],
    budget: int,
    minimum_hits: int,
    config: SeedFieldConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.random_seed + budget * 10 + minimum_hits)
    total = np.zeros(config.null_iterations, dtype=np.int64)
    for round_no in rounds:
        good = sum(point.hits >= minimum_hits for point in landscapes[round_no])
        total += rng.hypergeometric(good, config.seed_space - good, budget, size=config.null_iterations)
    return total


def summarize_seed_field(
    rows: Sequence[FieldEvaluationRow],
    landscapes: dict[int, tuple[LandscapePoint, ...]],
    config: SeedFieldConfig,
) -> dict[str, Any]:
    rounds = sorted({row.round_no for row in rows})
    null_cache: dict[tuple[int, int], np.ndarray] = {}
    result: dict[str, Any] = {"models": {}}
    for model in MODEL_NAMES:
        model_rows = [row for row in rows if row.model == model]
        if not model_rows:
            continue
        budgets: dict[str, Any] = {}
        for budget in config.budgets:
            selected = [row for row in model_rows if row.budget == budget]
            values: dict[str, Any] = {
                "rounds": len(selected),
                "selected_hit_4": sum(row.selected_hit_4 for row in selected),
                "selected_hit_5": sum(row.selected_hit_5 for row in selected),
                "selected_hit_6": sum(row.selected_hit_6 for row in selected),
                "rounds_with_4_plus": sum(row.selected_hit_4_plus > 0 for row in selected),
                "rounds_with_5_plus": sum(row.selected_hit_5_plus > 0 for row in selected),
                "rounds_with_6": sum(row.selected_hit_6 > 0 for row in selected),
            }
            for minimum_hits, key in [(4, "4_plus"), (5, "5_plus"), (6, "6")]:
                cache_key = (budget, minimum_hits)
                if cache_key not in null_cache:
                    null_cache[cache_key] = _null_distribution(landscapes, rounds, budget, minimum_hits, config)
                null = null_cache[cache_key]
                observed = (
                    values["selected_hit_4"] + values["selected_hit_5"] + values["selected_hit_6"]
                    if minimum_hits == 4
                    else values["selected_hit_5"] + values["selected_hit_6"]
                    if minimum_hits == 5
                    else values["selected_hit_6"]
                )
                expected = float(np.mean(null))
                values[f"expected_{key}"] = expected
                values[f"lift_{key}"] = observed / expected if expected else None
                values[f"p_{key}"] = float((np.sum(null >= observed) + 1) / (len(null) + 1))
            if budget == 10:
                hit_distribution = [0] * 7
                for row in selected:
                    for hit, count in enumerate(row.top10_hit_distribution):
                        hit_distribution[hit] += count
                values["top10_hit_distribution"] = hit_distribution
                values["top10_rounds_max_4_plus"] = sum(row.top10_max_hit >= 4 for row in selected)
                values["top10_rounds_max_5_plus"] = sum(row.top10_max_hit >= 5 for row in selected)
                values["top10_rounds_max_6"] = sum(row.top10_max_hit >= 6 for row in selected)
            budgets[str(budget)] = values
        result["models"][model] = {"budgets": budgets}
    return result


def write_evaluation_csv(path: str | Path, rows: Sequence[FieldEvaluationRow]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cohort",
                "round",
                "model",
                "budget",
                "selected_hit_4",
                "selected_hit_5",
                "selected_hit_6",
                "top10_max_hit",
                *[f"top10_hit_{hit}" for hit in range(7)],
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.cohort,
                    row.round_no,
                    row.model,
                    row.budget,
                    row.selected_hit_4,
                    row.selected_hit_5,
                    row.selected_hit_6,
                    row.top10_max_hit,
                    *row.top10_hit_distribution,
                ]
            )


def write_prediction_csv(path: str | Path, rows: Sequence[SeedPredictionRow]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cohort",
                "round",
                "model",
                "rank",
                "seed",
                "field_score",
                "n1",
                "n2",
                "n3",
                "n4",
                "n5",
                "n6",
                "hits",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.cohort,
                    row.round_no,
                    row.model,
                    row.rank,
                    row.seed,
                    f"{row.field_score:.12g}",
                    *row.numbers,
                    row.hits,
                ]
            )


def write_forecast_csv(path: str | Path, rows: Sequence[SeedForecastRow]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_round",
                "model",
                "rank",
                "seed",
                "field_score",
                "n1",
                "n2",
                "n3",
                "n4",
                "n5",
                "n6",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.target_round,
                    row.model,
                    row.rank,
                    row.seed,
                    f"{row.field_score:.12g}",
                    *row.numbers,
                ]
            )
