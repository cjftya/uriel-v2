from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from uriel_v2.models import Draw


VIEW_NAMES = ("raw", "grid", "circle", "distribution", "transition", "context")
PRIMES = frozenset((2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43))


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    rounds: np.ndarray
    numbers: np.ndarray
    views: Mapping[str, np.ndarray]
    feature_names: Mapping[str, tuple[str, ...]]
    grid_masks: np.ndarray

    def frame(self) -> pd.DataFrame:
        columns: dict[str, np.ndarray] = {"round": self.rounds}
        for index in range(6):
            columns[f"number_{index + 1}"] = self.numbers[:, index]
        for view in VIEW_NAMES:
            matrix = self.views[view]
            for index, name in enumerate(self.feature_names[view]):
                columns[f"{view}__{name}"] = matrix[:, index]
        for index in range(49):
            columns[f"grid_mask__cell_{index + 1}"] = self.grid_masks[:, index]
        return pd.DataFrame(columns)


def _safe_entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    denominator = math.log(max(2, len(values)))
    return float(-np.sum(probabilities * np.log(probabilities)) / denominator)


def _grid_coordinates(numbers: Sequence[int]) -> np.ndarray:
    return np.asarray([((number - 1) // 7, (number - 1) % 7) for number in numbers], dtype=float)


def _circle_coordinates(numbers: Sequence[int]) -> np.ndarray:
    angles = (np.asarray(numbers, dtype=float) - 1.0) * (2.0 * math.pi / 45.0)
    return np.c_[np.cos(angles), np.sin(angles)]


def _pair_distances(coordinates: np.ndarray, *, manhattan: bool = False) -> np.ndarray:
    values: list[float] = []
    for left, right in combinations(coordinates, 2):
        difference = np.abs(left - right)
        values.append(float(difference.sum() if manhattan else np.linalg.norm(difference)))
    return np.asarray(values, dtype=float)


def _components(mask: np.ndarray) -> int:
    occupied = {tuple(cell) for cell in np.argwhere(mask.reshape(7, 7) > 0)}
    count = 0
    while occupied:
        count += 1
        stack = [occupied.pop()]
        while stack:
            row, column = stack.pop()
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                if neighbor in occupied:
                    occupied.remove(neighbor)
                    stack.append(neighbor)
    return count


def _raw_features(numbers: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    gaps = np.diff(numbers)
    gap_probabilities = gaps / max(float(gaps.sum()), 1.0)
    gap_entropy = float(-np.sum(gap_probabilities * np.log(gap_probabilities + 1e-12)) / math.log(5))
    values = np.r_[
        numbers / 45.0,
        gaps / 44.0,
        [
            numbers[0] / 45.0,
            numbers[-1] / 45.0,
            (numbers[-1] - numbers[0]) / 44.0,
            numbers.sum() / 270.0,
            numbers.mean() / 45.0,
            np.median(numbers) / 45.0,
            np.var(numbers) / (44.0**2),
            np.std(numbers) / 44.0,
        ],
        np.sort(gaps) / 44.0,
        [gap_entropy, int(np.argmax(gaps)) / 4.0, int(np.argmin(gaps)) / 4.0],
    ]
    names = (
        *(f"number_{index}" for index in range(1, 7)),
        *(f"gap_{index}" for index in range(1, 6)),
        "first", "last", "range", "sum", "mean", "median", "variance", "std",
        *(f"sorted_gap_{index}" for index in range(1, 6)),
        "gap_entropy", "largest_gap_position", "smallest_gap_position",
    )
    return values.astype(float), tuple(names)


def _grid_features(numbers: np.ndarray) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    coordinates = _grid_coordinates(numbers)
    rows = coordinates[:, 0]
    columns = coordinates[:, 1]
    mask = np.zeros(49, dtype=np.uint8)
    for row, column in coordinates.astype(int):
        mask[row * 7 + column] = 1
    euclidean = _pair_distances(coordinates)
    manhattan = _pair_distances(coordinates, manhattan=True)
    row_occupancy = np.bincount(rows.astype(int), minlength=7) / 6.0
    column_occupancy = np.bincount(columns.astype(int), minlength=7) / 6.0
    grid = mask.reshape(7, 7)
    reflection = max(
        float(np.logical_and(grid, np.fliplr(grid)).sum()),
        float(np.logical_and(grid, np.flipud(grid)).sum()),
    ) / 6.0
    compactness = float(np.mean(euclidean <= math.sqrt(2.0)))
    values = np.r_[
        [
            rows.min() / 6.0,
            rows.max() / 6.0,
            columns.min() / 6.0,
            columns.max() / 6.0,
            rows.mean() / 6.0,
            columns.mean() / 6.0,
            rows.std() / 3.0,
            columns.std() / 3.0,
            np.std(rows - columns) / 6.0,
            _components(mask) / 6.0,
            euclidean.mean() / math.sqrt(72.0),
            euclidean.std() / math.sqrt(72.0),
            manhattan.mean() / 12.0,
            manhattan.std() / 12.0,
        ],
        row_occupancy,
        column_occupancy,
        [reflection, compactness, float(np.var(coordinates)) / 9.0],
    ]
    names = (
        "bbox_top", "bbox_bottom", "bbox_left", "bbox_right", "center_row", "center_column",
        "horizontal_spread", "vertical_spread", "diagonal_spread", "component_count",
        "pair_euclidean_mean", "pair_euclidean_std", "pair_manhattan_mean", "pair_manhattan_std",
        *(f"row_occupancy_{index}" for index in range(1, 8)),
        *(f"column_occupancy_{index}" for index in range(1, 8)),
        "symmetry", "compactness", "dispersion",
    )
    return values.astype(float), tuple(names), mask


def _circle_features(numbers: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    angles = (numbers - 1.0) * (2.0 * math.pi / 45.0)
    coordinates = np.c_[np.cos(angles), np.sin(angles)]
    chords = _pair_distances(coordinates)
    angular_gaps = np.diff(np.r_[angles, angles[0] + 2.0 * math.pi]) / (2.0 * math.pi)
    edges = np.linalg.norm(np.roll(coordinates, -1, axis=0) - coordinates, axis=1)
    centroid = coordinates.mean(axis=0)
    cross = coordinates[:, 0] * np.roll(coordinates[:, 1], -1) - coordinates[:, 1] * np.roll(coordinates[:, 0], -1)
    polygon_area = abs(float(cross.sum())) / 2.0
    normalized = np.mod(angles - angles[0], 2.0 * math.pi)
    values = np.r_[
        angles / (2.0 * math.pi),
        [chords.mean() / 2.0, chords.std() / 2.0, chords.min() / 2.0, chords.max() / 2.0],
        angular_gaps,
        [1.0 - float(np.linalg.norm(centroid)), centroid[0], centroid[1], polygon_area / 3.0, edges.sum() / 12.0],
        edges / 2.0,
        [angular_gaps.max()],
        np.cos(normalized),
        np.sin(normalized),
    ]
    names = (
        *(f"absolute_angle_{index}" for index in range(1, 7)),
        "pair_chord_mean", "pair_chord_std", "pair_chord_min", "pair_chord_max",
        *(f"angular_gap_{index}" for index in range(1, 7)),
        "circular_dispersion", "centroid_x", "centroid_y", "polygon_area", "polygon_perimeter",
        *(f"edge_length_{index}" for index in range(1, 7)),
        "dominant_angular_gap",
        *(f"rotation_normalized_cos_{index}" for index in range(1, 7)),
        *(f"rotation_normalized_sin_{index}" for index in range(1, 7)),
    )
    return values.astype(float), tuple(names)


def _distribution_features(numbers: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    decades = np.zeros(5, dtype=float)
    endings = np.zeros(10, dtype=float)
    for number in numbers.astype(int):
        decades[min((number - 1) // 10, 4)] += 1.0 / 6.0
        endings[number % 10] += 1.0 / 6.0
    consecutive = int(np.sum(np.diff(numbers) == 1))
    same_decade = sum((int(left) - 1) // 10 == (int(right) - 1) // 10 for left, right in combinations(numbers, 2))
    value_range = float(numbers[-1] - numbers[0])
    values = np.r_[
        [np.sum(numbers % 2 == 1) / 6.0, np.sum(numbers <= 22) / 6.0],
        decades,
        endings,
        [
            sum(int(number) in PRIMES for number in numbers) / 6.0,
            consecutive / 5.0,
            same_decade / 15.0,
            numbers.sum() / 270.0,
            value_range / 44.0,
            6.0 / max(value_range + 1.0, 6.0),
        ],
    ]
    names = (
        "odd_count", "low_count", *(f"decade_{index}" for index in range(1, 6)),
        *(f"ending_digit_{index}" for index in range(10)),
        "prime_count", "consecutive_count", "same_decade_pair_count", "sum_band", "spread_band", "density",
    )
    return values.astype(float), tuple(names)


def _transition_features(
    numbers: np.ndarray,
    previous: np.ndarray | None,
    previous_two: np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    names = (
        "overlap_t_minus_1", "overlap_t_minus_2", "number_displacement", "grid_centroid_row_delta",
        "grid_centroid_column_delta", "circle_centroid_x_delta", "circle_centroid_y_delta", "sum_delta",
        "range_delta", "dispersion_delta", "gap_profile_delta", "raw_shape_distance_t_minus_1",
        "grid_shape_distance_t_minus_1", "circle_shape_distance_t_minus_1", "raw_shape_distance_t_minus_2",
        "grid_shape_distance_t_minus_2", "circle_shape_distance_t_minus_2",
    )
    if previous is None:
        return np.zeros(len(names), dtype=float), names

    def changes(other: np.ndarray) -> tuple[float, float, float]:
        raw = float(np.linalg.norm(numbers / 45.0 - other / 45.0) / math.sqrt(6))
        grid = float(np.linalg.norm(_grid_coordinates(numbers).mean(axis=0) - _grid_coordinates(other).mean(axis=0)) / math.sqrt(72))
        circle = float(np.linalg.norm(_circle_coordinates(numbers).mean(axis=0) - _circle_coordinates(other).mean(axis=0)) / 2.0)
        return raw, grid, circle

    grid_now = _grid_coordinates(numbers).mean(axis=0)
    grid_previous = _grid_coordinates(previous).mean(axis=0)
    circle_now = _circle_coordinates(numbers).mean(axis=0)
    circle_previous = _circle_coordinates(previous).mean(axis=0)
    gap_now = np.diff(numbers)
    gap_previous = np.diff(previous)
    raw_one, grid_one, circle_one = changes(previous)
    raw_two, grid_two, circle_two = changes(previous_two) if previous_two is not None else (0.0, 0.0, 0.0)
    values = np.asarray(
        [
            len(set(numbers.astype(int)).intersection(previous.astype(int))) / 6.0,
            len(set(numbers.astype(int)).intersection(previous_two.astype(int))) / 6.0 if previous_two is not None else 0.0,
            np.mean(np.abs(numbers - previous)) / 44.0,
            (grid_now[0] - grid_previous[0]) / 6.0,
            (grid_now[1] - grid_previous[1]) / 6.0,
            (circle_now[0] - circle_previous[0]) / 2.0,
            (circle_now[1] - circle_previous[1]) / 2.0,
            (numbers.sum() - previous.sum()) / 264.0,
            ((numbers[-1] - numbers[0]) - (previous[-1] - previous[0])) / 44.0,
            (np.std(numbers) - np.std(previous)) / 44.0,
            np.linalg.norm(gap_now - gap_previous) / (44.0 * math.sqrt(5)),
            raw_one, grid_one, circle_one, raw_two, grid_two, circle_two,
        ],
        dtype=float,
    )
    return values, names


def _local_context(numbers_history: np.ndarray, index: int) -> tuple[np.ndarray, tuple[str, ...]]:
    values: list[float] = []
    names: list[str] = []
    for window in (3, 5, 8, 13):
        selected = numbers_history[max(0, index - window + 1) : index + 1]
        frequencies = np.bincount(selected.ravel().astype(int), minlength=46)[1:] / max(1.0, len(selected) * 6.0)
        bands = np.asarray([frequencies[start : start + 9].sum() for start in range(0, 45, 9)], dtype=float)
        grid_rows = np.zeros(7, dtype=float)
        grid_columns = np.zeros(7, dtype=float)
        octants = np.zeros(8, dtype=float)
        overlaps: list[float] = []
        dispersions: list[float] = []
        pair_counter: dict[tuple[int, int], int] = {}
        for row_index, draw in enumerate(selected):
            coordinates = _grid_coordinates(draw)
            grid_rows += np.bincount(coordinates[:, 0].astype(int), minlength=7)
            grid_columns += np.bincount(coordinates[:, 1].astype(int), minlength=7)
            angles = ((draw - 1.0) / 45.0 * 8.0).astype(int) % 8
            octants += np.bincount(angles, minlength=8)
            dispersions.append(float(np.std(draw) / 44.0))
            if row_index:
                overlaps.append(len(set(draw.astype(int)).intersection(selected[row_index - 1].astype(int))) / 6.0)
            for pair in combinations(draw.astype(int), 2):
                pair_counter[pair] = pair_counter.get(pair, 0) + 1
        normalizer = max(1.0, len(selected) * 6.0)
        grid_rows /= normalizer
        grid_columns /= normalizer
        octants /= normalizer
        pair_concentration = max(pair_counter.values(), default=0) / max(1.0, len(selected))
        entropy = _safe_entropy(frequencies)
        concentration = float(frequencies.max())
        slope = float(dispersions[-1] - dispersions[0]) / max(1, len(dispersions) - 1)
        acceleration = 0.0
        if len(dispersions) >= 3:
            acceleration = float((dispersions[-1] - dispersions[-2]) - (dispersions[-2] - dispersions[-3]))
        block = np.r_[
            bands,
            [pair_concentration, np.mean(overlaps) if overlaps else 0.0],
            grid_rows,
            grid_columns,
            octants,
            [entropy, concentration, np.mean(dispersions), slope, acceleration],
        ]
        block_names = (
            *(f"number_band_{band}" for band in range(1, 6)),
            "pair_concentration", "recent_overlap_pressure",
            *(f"grid_row_heat_{row}" for row in range(1, 8)),
            *(f"grid_column_heat_{column}" for column in range(1, 8)),
            *(f"circle_octant_heat_{octant}" for octant in range(1, 9)),
            "entropy", "concentration", "dispersion", "dispersion_slope", "dispersion_acceleration",
        )
        values.extend(block.tolist())
        names.extend(f"w{window}_{name}" for name in block_names)
    return np.asarray(values, dtype=float), tuple(names)


def build_feature_bundle(draws: Sequence[Draw]) -> FeatureBundle:
    if not draws:
        raise ValueError("feature 생성에는 draw가 필요합니다")
    numbers = np.asarray([draw.numbers for draw in draws], dtype=float)
    rounds = np.asarray([draw.round_no for draw in draws], dtype=int)
    if np.any(np.diff(rounds) <= 0):
        raise ValueError("draw는 중복 없이 회차 오름차순이어야 합니다")

    collected: dict[str, list[np.ndarray]] = {view: [] for view in VIEW_NAMES}
    names: dict[str, tuple[str, ...]] = {}
    masks: list[np.ndarray] = []
    for index, draw_numbers in enumerate(numbers):
        raw, raw_names = _raw_features(draw_numbers)
        grid, grid_names, mask = _grid_features(draw_numbers)
        circle, circle_names = _circle_features(draw_numbers)
        distribution, distribution_names = _distribution_features(draw_numbers)
        transition, transition_names = _transition_features(
            draw_numbers,
            numbers[index - 1] if index >= 1 else None,
            numbers[index - 2] if index >= 2 else None,
        )
        context, context_names = _local_context(numbers, index)
        for view, values, feature_names in (
            ("raw", raw, raw_names),
            ("grid", grid, grid_names),
            ("circle", circle, circle_names),
            ("distribution", distribution, distribution_names),
            ("transition", transition, transition_names),
            ("context", context, context_names),
        ):
            collected[view].append(values)
            names[view] = feature_names
        masks.append(mask)

    views = {view: np.vstack(collected[view]) for view in VIEW_NAMES}
    return FeatureBundle(
        rounds=rounds,
        numbers=numbers.astype(int),
        views=views,
        feature_names=names,
        grid_masks=np.vstack(masks),
    )


def prefix_standardize(matrix: np.ndarray, end_index: int) -> np.ndarray:
    """Standardize rows ``0..end_index`` without using a future row."""
    if end_index < 0 or end_index >= len(matrix):
        raise IndexError(end_index)
    selected = np.asarray(matrix[: end_index + 1], dtype=float)
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (selected - mean) / scale
    return np.clip(standardized, -8.0, 8.0)

