from __future__ import annotations

import csv
import json
import logging
import math
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from uriel_v2.evaluation import resolve_workers
from uriel_v2.irregular_motif import (
    CANDIDATE_SIZES,
    MotifMatch,
    _candidate_scores,
    _checkpoint_payloads,
    _checkpoint_record,
    _coarse_distance,
    _followup_entropy,
    _selection_score,
    _surrogate_entropies,
    _verdict,
    _write_csv,
    dtw_distance,
    summarize_motif_rows,
)
from uriel_v2.models import Draw
from uriel_v2.motif_features import FeatureBundle, VIEW_NAMES, build_feature_bundle, prefix_standardize
from uriel_v2.provenance import execution_metadata


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    name: str
    method: str
    clusters: int
    sequence_length: int = 5
    candidate_lengths: tuple[int, ...] = (4, 5, 7)
    top_k: int = 30
    separation: int = 50


DEFAULT_REGIME_CONFIGS = tuple(
    RegimeConfig(f"{method.lower()}_k{clusters}", method, clusters)
    for method in ("KMeans", "GMM")
    for clusters in (4, 6, 8, 12, 16)
) + (
    RegimeConfig("hdbscan_m20", "HDBSCAN", 20),
    RegimeConfig("hdbscan_m40", "HDBSCAN", 40),
)


@dataclass(frozen=True, slots=True)
class RegimeFit:
    states: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    cluster_count: int
    fitted_method: str


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -40.0, 40.0))
    return exponent / np.maximum(exponent.sum(axis=1, keepdims=True), 1e-12)


def _state_matrix(bundle: FeatureBundle, query_end: int) -> np.ndarray:
    selected = [prefix_standardize(bundle.views[view], query_end) for view in VIEW_NAMES]
    matrix = np.hstack(selected)
    components = min(8, matrix.shape[1], max(2, len(matrix) - 1))
    # A deterministic randomized solver is materially more stable than the
    # platform LAPACK full-SVD path for several long, highly collinear prefixes.
    return PCA(
        n_components=components,
        random_state=0,
        svd_solver="randomized",
        iterated_power=4,
    ).fit_transform(matrix)


def fit_regimes(bundle: FeatureBundle, query_end: int, config: RegimeConfig, seed: int) -> RegimeFit:
    states = _state_matrix(bundle, query_end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if config.method == "KMeans":
            model = KMeans(n_clusters=config.clusters, n_init=10, random_state=seed)
            labels = model.fit_predict(states)
            distances = model.transform(states)
            scale = max(float(np.median(np.min(distances, axis=1))), 1e-6)
            probabilities = _softmax(-distances / scale)
            return RegimeFit(states, labels.astype(int), probabilities, config.clusters, config.method)
        if config.method == "GMM":
            model = GaussianMixture(
                n_components=config.clusters,
                covariance_type="diag",
                reg_covar=1e-5,
                n_init=3,
                random_state=seed,
            )
            labels = model.fit_predict(states)
            probabilities = model.predict_proba(states)
            return RegimeFit(states, labels.astype(int), probabilities, config.clusters, config.method)
        if config.method == "HDBSCAN":
            model = HDBSCAN(
                min_cluster_size=config.clusters,
                min_samples=max(5, config.clusters // 4),
                copy=True,
            )
            labels = model.fit_predict(states)
            clusters = sorted(int(label) for label in set(labels) if label >= 0)
            if len(clusters) >= 2:
                centroids = np.vstack([states[labels == label].mean(axis=0) for label in clusters])
                distances = np.linalg.norm(states[:, None, :] - centroids[None, :, :], axis=2)
                scale = max(float(np.median(np.min(distances, axis=1))), 1e-6)
                probabilities = _softmax(-distances / scale)
                mapped = np.argmax(probabilities, axis=1)
                return RegimeFit(states, mapped.astype(int), probabilities, len(clusters), config.method)
            fallback = RegimeConfig(config.name, "KMeans", 4, config.sequence_length, config.candidate_lengths, config.top_k, config.separation)
            fit = fit_regimes(bundle, query_end, fallback, seed)
            return RegimeFit(fit.states, fit.labels, fit.probabilities, fit.cluster_count, "HDBSCAN→KMeans fallback")
    raise ValueError(f"지원하지 않는 regime method: {config.method}")


def _edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _dwell_profile(labels: np.ndarray) -> np.ndarray:
    if not len(labels):
        return np.zeros(1, dtype=float)
    dwell: list[int] = []
    current = int(labels[0])
    length = 1
    for label in labels[1:]:
        if int(label) == current:
            length += 1
        else:
            dwell.append(length)
            current = int(label)
            length = 1
    dwell.append(length)
    return np.asarray(dwell, dtype=float)


def _transition_similarity(
    fit: RegimeFit,
    query_indices: np.ndarray,
    candidate_indices: np.ndarray,
) -> tuple[float, dict[str, float]]:
    query_states = fit.states[query_indices]
    candidate_states = fit.states[candidate_indices]
    query_probabilities = fit.probabilities[query_indices]
    candidate_probabilities = fit.probabilities[candidate_indices]
    state_similarity = math.exp(-dtw_distance(query_states, candidate_states))
    soft_similarity = math.exp(-dtw_distance(query_probabilities, candidate_probabilities))
    edit_similarity = 1.0 - _edit_distance(
        fit.labels[query_indices].tolist(),
        fit.labels[candidate_indices].tolist(),
    ) / max(len(query_indices), len(candidate_indices))
    query_speed = float(np.mean(np.linalg.norm(np.diff(query_states, axis=0), axis=1))) if len(query_states) > 1 else 0.0
    candidate_speed = float(np.mean(np.linalg.norm(np.diff(candidate_states, axis=0), axis=1))) if len(candidate_states) > 1 else 0.0
    speed_similarity = math.exp(-abs(query_speed - candidate_speed))
    query_volatility = float(np.std(np.linalg.norm(np.diff(query_states, axis=0), axis=1))) if len(query_states) > 2 else 0.0
    candidate_volatility = float(np.std(np.linalg.norm(np.diff(candidate_states, axis=0), axis=1))) if len(candidate_states) > 2 else 0.0
    volatility_similarity = math.exp(-abs(query_volatility - candidate_volatility))
    query_dwell = _dwell_profile(fit.labels[query_indices])
    candidate_dwell = _dwell_profile(fit.labels[candidate_indices])
    dwell_similarity = math.exp(-abs(float(query_dwell.mean()) - float(candidate_dwell.mean())) / max(len(query_indices), len(candidate_indices)))
    components = {
        "regime_edit": max(0.0, edit_similarity),
        "soft_membership": soft_similarity,
        "state_trajectory": state_similarity,
        "transition_speed": speed_similarity,
        "transition_volatility": volatility_similarity,
        "dwell_time": dwell_similarity,
    }
    ordered = sorted(components.values(), reverse=True)
    aggregate = float(mean(ordered[:4]))
    return aggregate, components


def _transition_matches(
    bundle: FeatureBundle,
    query_end: int,
    fit: RegimeFit,
    config: RegimeConfig,
    seed: int,
) -> tuple[list[MotifMatch], dict[str, float]]:
    query_start = query_end - config.sequence_length + 1
    query_indices = np.arange(query_start, query_end + 1)
    latest_past_end = query_end - config.separation
    coarse: list[tuple[float, int, int]] = []
    for length in config.candidate_lengths:
        for past_end in range(length - 1, latest_past_end + 1):
            past_start = past_end - length + 1
            distance = _coarse_distance(fit.states[query_indices], fit.states[past_start : past_end + 1])
            coarse.append((distance, past_end, length))
    candidates: list[tuple[float, int, int, dict[str, float]]] = []
    for _, past_end, length in sorted(coarse)[: min(len(coarse), config.top_k * 4)]:
        past_start = past_end - length + 1
        aggregate, components = _transition_similarity(
            fit,
            query_indices,
            np.arange(past_start, past_end + 1),
        )
        candidates.append((aggregate, past_end, length, components))
    selected: list[MotifMatch] = []
    duplicate_radius = max(2, config.sequence_length // 2)
    for aggregate, past_end, length, components in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if any(abs(past_end - match.past_end) <= duplicate_radius for match in selected):
            continue
        selected.append(
            MotifMatch(
                current_start=query_start,
                current_end=query_end,
                past_start=past_end - length + 1,
                past_end=past_end,
                window_length=length,
                aggregate_similarity=aggregate,
                support_count=sum(value >= 0.50 for value in components.values()),
                similarities=components,
            )
        )
        if len(selected) == config.top_k:
            break
    rng = np.random.default_rng(seed)
    if not selected:
        surrogates = {name: 0.0 for name in ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving")}
    else:
        reference = selected[0]
        reference_indices = np.arange(reference.past_start, reference.past_end + 1)
        round_shuffle = reference_indices.copy()
        rng.shuffle(round_shuffle)
        random_states = rng.choice(
            np.arange(0, latest_past_end + 1),
            size=len(reference_indices),
            replace=latest_past_end + 1 < len(reference_indices),
        )
        block_size = min(3, len(reference_indices))
        blocks = [
            reference_indices[start : start + block_size]
            for start in range(0, len(reference_indices), block_size)
        ]
        rng.shuffle(blocks)
        block_shuffle = np.concatenate(blocks)
        maximum_start = max(0, latest_past_end - len(reference_indices) + 1)
        preserved_start = int(rng.integers(0, maximum_start + 1)) if maximum_start else 0
        feature_preserving = np.arange(preserved_start, preserved_start + len(reference_indices))
        surrogates = {
            "round_shuffle": _transition_similarity(fit, query_indices, round_shuffle)[0],
            "within_round_random": _transition_similarity(fit, query_indices, random_states)[0],
            "block_shuffle": _transition_similarity(fit, query_indices, block_shuffle)[0],
            "feature_preserving": _transition_similarity(fit, query_indices, feature_preserving)[0],
        }
    return selected, surrogates


def _regime_only_matches(query_end: int, fit: RegimeFit, config: RegimeConfig) -> list[MotifMatch]:
    current_label = int(fit.labels[query_end])
    current_state = fit.states[query_end]
    candidates: list[tuple[float, int]] = []
    for index in range(0, query_end - config.separation + 1):
        label_penalty = 0.0 if int(fit.labels[index]) == current_label else 1.0
        distance = float(np.linalg.norm(current_state - fit.states[index])) + label_penalty
        candidates.append((distance, index))
    selected: list[MotifMatch] = []
    for distance, index in sorted(candidates):
        if any(abs(index - match.past_end) <= 2 for match in selected):
            continue
        similarity = math.exp(-distance)
        selected.append(
            MotifMatch(
                current_start=query_end,
                current_end=query_end,
                past_start=index,
                past_end=index,
                window_length=1,
                aggregate_similarity=similarity,
                support_count=int(fit.labels[index] == current_label),
                similarities={"regime_state": similarity},
            )
        )
        if len(selected) == config.top_k:
            break
    return selected


def _rank_and_hits(
    bundle: FeatureBundle,
    target_index: int,
    query_end: int,
    matches: Sequence[MotifMatch],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[int, int]]:
    scores, components = _candidate_scores(bundle, query_end, matches)
    ranking = np.argsort(-scores, kind="stable") + 1
    winner = set(bundle.numbers[target_index].tolist())
    hits = {size: len(winner.intersection(ranking[:size].tolist())) for size in CANDIDATE_SIZES}
    return ranking, scores, components, hits


def _predict_regime_round(
    bundle: FeatureBundle,
    target_index: int,
    config: RegimeConfig,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    query_end = target_index - 1
    fit = fit_regimes(bundle, query_end, config, seed)
    transition_matches, recurrence_surrogates = _transition_matches(
        bundle,
        query_end,
        fit,
        config,
        seed ^ int(bundle.rounds[target_index]),
    )
    regime_matches = _regime_only_matches(query_end, fit, config)
    if not transition_matches or not regime_matches:
        raise ValueError(f"{int(bundle.rounds[target_index])}회에서 regime match를 찾지 못했습니다")
    ranking, scores, components, hits = _rank_and_hits(bundle, target_index, query_end, transition_matches)
    regime_ranking, regime_scores, regime_components, regime_hits = _rank_and_hits(
        bundle,
        target_index,
        query_end,
        regime_matches,
    )
    entropy = _followup_entropy(bundle, transition_matches)
    regime_entropy = _followup_entropy(bundle, regime_matches)
    regime_probability = float(np.max(fit.probabilities[query_end]))
    cross_view = float(mean(match.support_count for match in transition_matches))
    confidence = (
        float(mean(match.aggregate_similarity for match in transition_matches[: min(5, len(transition_matches))]))
        * max(0.0, 1.0 - entropy)
        * regime_probability
    )
    entropy_surrogates = _surrogate_entropies(
        bundle,
        query_end,
        transition_matches,
        seed ^ int(bundle.rounds[target_index]) ^ 0xA661,
    )
    regime_entropy_surrogates = _surrogate_entropies(
        bundle,
        query_end,
        regime_matches,
        seed ^ int(bundle.rounds[target_index]) ^ 0xB772,
    )
    row: dict[str, Any] = {
        "round": int(bundle.rounds[target_index]),
        "config": config.name,
        "method": config.method,
        "fitted_method": fit.fitted_method,
        "regime_count": fit.cluster_count,
        "current_regime": int(fit.labels[query_end]),
        "regime_confidence": regime_probability,
        "matched_motifs": len(transition_matches),
        "median_motif_similarity": float(np.median([match.aggregate_similarity for match in transition_matches])),
        "best_motif_similarity": transition_matches[0].aggregate_similarity,
        "recurrence_similarity": float(mean(match.aggregate_similarity for match in transition_matches)),
        "cross_view_agreement_count": cross_view,
        "followup_entropy": entropy,
        "regime_only_followup_entropy": regime_entropy,
        "candidate_concentration": float(scores[ranking[0] - 1] / max(float(scores.sum()), 1e-12)),
        "confidence": confidence,
        "ranked_numbers": ranking.tolist(),
        "number_scores": scores.tolist(),
        "regime_only_ranked_numbers": regime_ranking.tolist(),
        **{f"surrogate_{name}_entropy": value for name, value in entropy_surrogates.items()},
        **{f"regime_only_surrogate_{name}_entropy": value for name, value in regime_entropy_surrogates.items()},
        **{f"surrogate_{name}_recurrence": value for name, value in recurrence_surrogates.items()},
    }
    for size in CANDIDATE_SIZES:
        row[f"recall_at_{size}"] = hits[size] / 6.0
        row[f"hits_at_{size}"] = hits[size]
        row[f"regime_only_recall_at_{size}"] = regime_hits[size] / 6.0
        row[f"regime_only_hits_at_{size}"] = regime_hits[size]

    winner = set(bundle.numbers[target_index].tolist())
    prediction_rows: list[dict[str, Any]] = []
    for engine, current_ranking, current_scores, current_components in (
        ("transition_motif", ranking, scores, components),
        ("regime_only", regime_ranking, regime_scores, regime_components),
    ):
        for rank, number in enumerate(current_ranking[: max(CANDIDATE_SIZES)], start=1):
            prediction_rows.append(
                {
                    "round": row["round"],
                    "config": config.name,
                    "engine": engine,
                    "rank": rank,
                    "number": int(number),
                    "score": float(current_scores[number - 1]),
                    **{name: float(values[number - 1]) for name, values in current_components.items()},
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
            "transition_components": json.dumps(match.similarities, sort_keys=True),
        }
        for rank, match in enumerate(transition_matches, start=1)
    ]
    return row, prediction_rows, match_rows


def _evaluate_regime_worker(
    bundle: FeatureBundle,
    target_indices: tuple[int, ...],
    config: RegimeConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for target_index in target_indices:
        row, prediction_rows, match_rows = _predict_regime_round(bundle, target_index, config, seed)
        rows.append(row)
        predictions.extend(prediction_rows)
        matches.extend(match_rows)
    return rows, predictions, matches


def _evaluate_regime_checkpointed(
    *,
    bundle: FeatureBundle,
    target_indices: Sequence[int],
    config: RegimeConfig,
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
            row, prediction_rows, match_rows = _predict_regime_round(bundle, target_index, config, seed)
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
            logger.info("Regime 진행 | cohort=%s | %s/%s | round=%s", cohort, completed, total, round_no)
    return rows, predictions, matches


def _calibration_indices(indices: Sequence[int], maximum: int = 24) -> tuple[int, ...]:
    if len(indices) <= maximum:
        return tuple(indices)
    positions = np.linspace(0, len(indices) - 1, maximum).round().astype(int)
    return tuple(indices[index] for index in sorted(set(positions.tolist())))


def _regime_only_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["confidence"] = source["regime_confidence"]
        row["followup_entropy"] = source["regime_only_followup_entropy"]
        for name in ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving"):
            row[f"surrogate_{name}_entropy"] = source[f"regime_only_surrogate_{name}_entropy"]
        for size in CANDIDATE_SIZES:
            row[f"hits_at_{size}"] = source[f"regime_only_hits_at_{size}"]
            row[f"recall_at_{size}"] = source[f"regime_only_recall_at_{size}"]
        converted.append(row)
    return converted


def _write_regime_walk_forward(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "round", "cohort", "config", "method", "fitted_method", "regime_count", "current_regime",
        "regime_confidence", "matched_motifs", "median_motif_similarity", "best_motif_similarity",
        "recurrence_similarity", "cross_view_agreement_count", "followup_entropy", "regime_only_followup_entropy",
        "candidate_concentration", "confidence", "is_opportunity", "ranked_numbers", "regime_only_ranked_numbers",
        *(f"surrogate_{name}_{metric}" for name in ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving") for metric in ("entropy", "recurrence")),
        *(f"regime_only_surrogate_{name}_entropy" for name in ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving")),
        *(field for size in CANDIDATE_SIZES for field in (f"recall_at_{size}", f"hits_at_{size}", f"regime_only_recall_at_{size}", f"regime_only_hits_at_{size}")),
    ]
    _write_csv(path, rows, fields)


def _write_final_regimes(
    bundle: FeatureBundle,
    config: RegimeConfig,
    seed: int,
    run_dir: Path,
) -> RegimeFit:
    fit = fit_regimes(bundle, len(bundle.rounds) - 1, config, seed)
    assignment_rows = [
        {
            "round": int(round_no),
            "regime": int(label),
            "confidence": float(np.max(probability)),
            "method": fit.fitted_method,
        }
        for round_no, label, probability in zip(bundle.rounds, fit.labels, fit.probabilities, strict=True)
    ]
    probability_rows = [
        {
            "round": int(round_no),
            **{f"regime_{index}_probability": float(value) for index, value in enumerate(probability)},
        }
        for round_no, probability in zip(bundle.rounds, fit.probabilities, strict=True)
    ]
    _write_csv(run_dir / "regime_assignments.csv", assignment_rows)
    _write_csv(run_dir / "regime_probabilities.csv", probability_rows)
    return fit


def _plot_regime_artifacts(
    bundle: FeatureBundle,
    rows: Sequence[dict[str, Any]],
    fit: RegimeFit,
    run_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    blue, gold, ink, grid = "#2f6fed", "#d7a514", "#172033", "#e4e8ef"
    recent = max(0, len(bundle.rounds) - 384)
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.scatter(bundle.rounds[recent:], fit.labels[recent:], c=fit.labels[recent:], cmap="tab20", s=14)
    axis.set_title("Round vs fitted regime", loc="left", color=ink, fontweight="bold")
    axis.set_xlabel("Round")
    axis.set_ylabel("Regime")
    axis.grid(axis="y", color=grid)
    figure.tight_layout()
    figure.savefig(plot_dir / "round_vs_regime.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    for index in range(fit.probabilities.shape[1]):
        axis.plot(bundle.rounds[recent:], fit.probabilities[recent:, index], linewidth=1.0, alpha=0.8, label=f"R{index}")
    axis.set_title("Regime probability timeline", loc="left", color=ink, fontweight="bold")
    axis.set_xlabel("Round")
    axis.set_ylabel("Membership probability")
    if fit.probabilities.shape[1] <= 8:
        axis.legend(ncol=min(4, fit.probabilities.shape[1]), frameon=False)
    axis.grid(color=grid)
    figure.tight_layout()
    figure.savefig(plot_dir / "regime_probability_timeline.png", dpi=150)
    plt.close(figure)

    transition_counts = np.zeros((fit.cluster_count, fit.cluster_count), dtype=float)
    for left, right in zip(fit.labels[:-1], fit.labels[1:], strict=True):
        transition_counts[int(left), int(right)] += 1
    figure, axis = plt.subplots(figsize=(7.4, 6.4))
    image = axis.imshow(transition_counts, cmap="Blues", origin="lower")
    axis.set_title("Regime transition counts", loc="left", color=ink, fontweight="bold")
    axis.set_xlabel("To regime")
    axis.set_ylabel("From regime")
    figure.colorbar(image, ax=axis, label="Transitions")
    figure.tight_layout()
    figure.savefig(plot_dir / "transition_graph.png", dpi=150)
    plt.close(figure)

    dwell: list[int] = []
    length = 1
    for left, right in zip(fit.labels[:-1], fit.labels[1:], strict=True):
        if left == right:
            length += 1
        else:
            dwell.append(length)
            length = 1
    dwell.append(length)
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    axis.hist(dwell, bins=np.arange(0.5, max(dwell) + 1.5), color=blue, edgecolor="#1f4fae")
    axis.set_title("Regime dwell time", loc="left", color=ink, fontweight="bold")
    axis.set_xlabel("Consecutive rounds")
    axis.set_ylabel("Regime runs")
    axis.grid(axis="y", color=grid)
    figure.tight_layout()
    figure.savefig(plot_dir / "regime_dwell_time.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    axis.scatter([float(row["confidence"]) for row in rows], [int(row["hits_at_20"]) for row in rows], color=gold, alpha=0.7)
    axis.set_title("Transition confidence vs candidate hits", loc="left", color=ink, fontweight="bold")
    axis.set_xlabel("Transition confidence")
    axis.set_ylabel("Hits at candidate size 20")
    axis.set_yticks(range(7))
    axis.grid(color=grid)
    figure.tight_layout()
    figure.savefig(plot_dir / "confidence_vs_hit.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    axis.hist([float(row["recurrence_similarity"]) for row in rows], bins=20, color=blue, edgecolor="#1f4fae")
    axis.set_title("Transition motif recurrence", loc="left", color=ink, fontweight="bold")
    axis.set_xlabel("Transition motif similarity")
    axis.set_ylabel("Walk-forward rounds")
    axis.grid(axis="y", color=grid)
    figure.tight_layout()
    figure.savefig(plot_dir / "transition_motif_recurrence.png", dpi=150)
    plt.close(figure)

    groups = {
        "All rounds": list(rows),
        "Opportunity": [row for row in rows if int(row["is_opportunity"]) == 1],
    }
    labels = list(groups)
    rates_4 = [mean(int(row["hits_at_20"]) >= 4 for row in groups[label]) if groups[label] else 0.0 for label in labels]
    rates_5 = [mean(int(row["hits_at_20"]) >= 5 for row in groups[label]) if groups[label] else 0.0 for label in labels]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    axis.bar(positions - 0.18, rates_4, 0.36, label="4+ rate", color=blue)
    axis.bar(positions + 0.18, rates_5, 0.36, label="5+ rate", color=gold)
    axis.set_title("Opportunity round performance", loc="left", color=ink, fontweight="bold")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Share of evaluated rounds")
    axis.set_ylim(0.0, max(0.05, max(rates_4 + rates_5) * 1.25))
    axis.legend(frameon=False)
    axis.grid(axis="y", color=grid)
    figure.tight_layout()
    figure.savefig(plot_dir / "opportunity_round_performance.png", dpi=150)
    plt.close(figure)


def run_regime_motif_experiment(
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
    configs: Sequence[RegimeConfig] = DEFAULT_REGIME_CONFIGS,
) -> dict[str, Any]:
    started_at = perf_counter()
    selected_draws = [draw for draw in draws if draw.round_no <= end_round]
    bundle = build_feature_bundle(selected_draws)
    round_to_index = {int(round_no): index for index, round_no in enumerate(bundle.rounds)}
    historical_indices = tuple(round_to_index[round_no] for round_no in range(start_round, split_round))
    development_indices = tuple(round_to_index[round_no] for round_no in range(split_round, end_round + 1))
    calibration = _calibration_indices(historical_indices)
    logger.info(
        "Regime Historical 설정 선택 | configs=%s | calibration_rounds=%s | full_historical=%s",
        len(configs), len(calibration), len(historical_indices),
    )

    worker_count = min(resolve_workers(workers), len(configs))
    calibration_results: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_evaluate_regime_worker, bundle, calibration, config, experiment_seed): config
                for config in configs
            }
            for future in as_completed(future_map):
                config = future_map[future]
                calibration_results[config.name] = future.result()
                logger.info("Regime calibration 완료 | config=%s", config.name)
    else:
        for config in configs:
            calibration_results[config.name] = _evaluate_regime_worker(bundle, calibration, config, experiment_seed)
            logger.info("Regime calibration 완료 | config=%s", config.name)

    selection_rows = [
        {
            "config": config.name,
            "method": config.method,
            "clusters_or_min_cluster_size": config.clusters,
            "calibration_rounds": len(calibration),
            "historical_selection_score": _selection_score(calibration_results[config.name][0]),
            "mean_hits_at_20": mean(float(row["hits_at_20"]) for row in calibration_results[config.name][0]),
            "followup_entropy": mean(float(row["followup_entropy"]) for row in calibration_results[config.name][0]),
        }
        for config in configs
    ]
    selected_name = max(
        selection_rows,
        key=lambda row: (float(row["historical_selection_score"]), -int(row["clusters_or_min_cluster_size"]), str(row["config"])),
    )["config"]
    selected_config = next(config for config in configs if config.name == selected_name)
    logger.info("Regime 설정 고정 | selected=%s | Development 재튜닝 없음", selected_name)

    resumed = _checkpoint_payloads(resume_from, selected_name)
    checkpoint_path = run_dir / "checkpoint.jsonl"
    with checkpoint_path.open("w", encoding="utf-8") as checkpoint_handle:
        historical_rows, historical_predictions, historical_matches = _evaluate_regime_checkpointed(
            bundle=bundle,
            target_indices=historical_indices,
            config=selected_config,
            seed=experiment_seed,
            cohort="Historical",
            resumed=resumed,
            checkpoint_handle=checkpoint_handle,
            logger=logger,
        )
        development_rows, development_predictions, development_matches = _evaluate_regime_checkpointed(
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
    pooled_surrogate = np.asarray(
        [
            float(row[f"surrogate_{name}_recurrence"])
            for row in historical_rows
            for name in ("round_shuffle", "within_round_random", "block_shuffle", "feature_preserving")
        ],
        dtype=float,
    )
    recurrence_threshold = float(np.quantile(pooled_surrogate, 0.95))
    all_rows: list[dict[str, Any]] = []
    for cohort, source in (("Historical", historical_rows), ("Development", development_rows)):
        for row in source:
            row["cohort"] = cohort
            row["is_opportunity"] = int(float(row["confidence"]) >= opportunity_threshold)
            all_rows.append(row)

    historical_summary = summarize_motif_rows(
        historical_rows,
        opportunity_threshold=opportunity_threshold,
        recurrence_threshold=recurrence_threshold,
        seed=experiment_seed + 20,
    )
    development_summary = summarize_motif_rows(
        development_rows,
        opportunity_threshold=opportunity_threshold,
        recurrence_threshold=recurrence_threshold,
        seed=experiment_seed + 21,
    )
    historical_regime_only = summarize_motif_rows(
        _regime_only_rows(historical_rows),
        opportunity_threshold=float(np.quantile([float(row["regime_confidence"]) for row in historical_rows], 0.70)),
        recurrence_threshold=recurrence_threshold,
        seed=experiment_seed + 22,
    )
    development_regime_only = summarize_motif_rows(
        _regime_only_rows(development_rows),
        opportunity_threshold=float(np.quantile([float(row["regime_confidence"]) for row in historical_rows], 0.70)),
        recurrence_threshold=recurrence_threshold,
        seed=experiment_seed + 23,
    )
    fit = _write_final_regimes(bundle, selected_config, experiment_seed, run_dir)
    summary: dict[str, Any] = {
        "algorithm": "Regime-Switching + Motif Transition Engine",
        "verdict": _verdict(historical_summary, development_summary),
        "selected_config": asdict(selected_config),
        "selection_policy": "24 evenly spaced Historical calibration rounds; full Historical and Development use the frozen setting",
        "opportunity_policy": "Historical transition-confidence 70th percentile; frozen for Development",
        "transition_motif": {"Historical": historical_summary, "Development": development_summary},
        "regime_only": {"Historical": historical_regime_only, "Development": development_regime_only},
        "regime_only_verdict": _verdict(historical_regime_only, development_regime_only),
        "descriptive_final_fit": {
            "rounds": len(bundle.rounds),
            "regime_count": fit.cluster_count,
            "method": fit.fitted_method,
            "note": "Descriptive export only; walk-forward predictions refit on 1..t-1.",
        },
        "sealed_ranges": {"Locked": "660-851: SEALED", "Additional Blind": "468-659: SEALED"},
        "execution": execution_metadata(
            draws=selected_draws,
            started_at=started_at,
            start_round=start_round,
            end_round=end_round,
        ),
    }
    _write_csv(run_dir / "config_selection.csv", selection_rows)
    _write_csv(run_dir / "transition_motifs.csv", [*historical_matches, *development_matches])
    _write_csv(run_dir / "regime_predictions.csv", [*historical_predictions, *development_predictions])
    _write_regime_walk_forward(run_dir / "walk_forward.csv", all_rows)
    _write_regime_walk_forward(run_dir / "opportunity_rounds.csv", [row for row in all_rows if row["is_opportunity"]])
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_regime_artifacts(bundle, all_rows, fit, run_dir)
    logger.info(
        "Regime 완료 | verdict=%s | config=%s | Historical @20 lift=%.4f | Development @20 lift=%.4f",
        summary["verdict"],
        selected_name,
        historical_summary["candidate_recall"]["20"]["mean_hit_lift"],
        development_summary["candidate_recall"]["20"]["mean_hit_lift"],
    )
    return summary
