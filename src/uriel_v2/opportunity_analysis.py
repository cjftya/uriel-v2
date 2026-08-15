from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from uriel_v2.irregular_motif import (
    MONTE_CARLO_ITERATIONS,
    MotifMatch,
    _benjamini_hochberg,
    _candidate_scores,
    _random_baseline,
)
from uriel_v2.models import Draw
from uriel_v2.motif_features import FeatureBundle, VIEW_NAMES, build_feature_bundle
from uriel_v2.provenance import execution_metadata


FROZEN_CONFIDENCE_THRESHOLD = 0.011722291804
FROZEN_QUERY_LENGTH = 13
FROZEN_CANDIDATE_LENGTHS = (10, 13, 16)
FROZEN_TOP_K = 40
FROZEN_SEPARATION = 100
PRIMARY_CANDIDATE_SIZE = 20
SECONDARY_CANDIDATE_SIZE = 30
EVALUATION_START = 852
EVALUATION_END = 1235
SPLIT_ROUND = 1044

VIEW_LABELS = {
    "raw": "Raw",
    "grid": "Grid",
    "circle": "Circle",
    "distribution": "Distribution",
    "transition": "Transition",
    "context": "Context",
}

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "motif_retrieval": (
        "matched_motif_count", "valid_motif_count", "best_similarity", "median_similarity",
        "top5_mean_similarity", "top10_mean_similarity", "similarity_variance",
        "motif_separation_mean", "motif_separation_median", "motif_separation_std",
        "motif_cluster_count", "recurrence_concentration", "recurrence_dispersion",
    ),
    "cross_view": (
        *(f"{view}_similarity" for view in VIEW_NAMES),
        *(f"{view}_rank" for view in VIEW_NAMES),
        "cross_view_agreement_count", "view_variance", "view_entropy", "top2_view_gap",
    ),
    "followup_consensus": (
        "number_entropy", "grid_entropy", "circle_entropy", "gap_entropy", "transition_entropy",
        "pair_concentration", "region_concentration", "consensus_sharpness",
    ),
    "candidate_structure": (
        "candidate_score_mean", "candidate_score_median", "candidate_score_std",
        "candidate_score_entropy", "candidate_score_slope", "candidate_top5_bottom5_gap",
        "candidate_grid_coverage", "candidate_circle_coverage", "motif_source_diversity",
        "candidate_concentration",
    ),
    "opportunity_formation": (
        "opportunity_gap", "burst_length", "isolated_opportunity", "rolling_opportunity_density",
        "confidence_delta", "agreement_delta", "entropy_delta", "motif_count_delta",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") or value.startswith("{"):
            return json.loads(value)
    return value


def _normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if total <= 0 or len(values) <= 1:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(len(values)))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    fields = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key, value in output.items():
                if isinstance(value, (list, tuple, dict)):
                    output[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            writer.writerow(output)


def classify_opportunity(hits_at_20: int) -> tuple[str, str]:
    if hits_at_20 <= 2:
        detailed = "FAIL_0_2"
    elif hits_at_20 == 3:
        detailed = "HIT_3"
    elif hits_at_20 == 4:
        detailed = "HIT_4"
    else:
        detailed = "HIT_5_PLUS"
    binary = "SUCCESS_4PLUS" if hits_at_20 >= 4 else "FAIL_BELOW4"
    return detailed, binary


def _validate_inputs(
    *,
    draws: Sequence[Draw],
    motif_run: Path,
    start_round: int,
    end_round: int,
    split_round: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    required = (
        "metrics.json", "walk_forward.csv", "motif_predictions.csv", "recurrence_candidates.csv",
    )
    missing = [name for name in required if not (motif_run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"motif-run 필수 산출물이 없습니다: {', '.join(missing)}")
    if (start_round, end_round, split_round) != (EVALUATION_START, EVALUATION_END, SPLIT_ROUND):
        raise ValueError("이번 분석 구간은 852~1235, split 1044로 동결되어 있습니다")

    with (motif_run / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    selected = metrics.get("selected_config", {})
    expected = {
        "name": "multiview_long",
        "query_length": FROZEN_QUERY_LENGTH,
        "candidate_lengths": list(FROZEN_CANDIDATE_LENGTHS),
        "top_k": FROZEN_TOP_K,
        "separation": FROZEN_SEPARATION,
        "views": list(VIEW_NAMES),
    }
    if selected != expected:
        raise ValueError(f"motif-run 설정이 동결 설정과 다릅니다: {selected}")
    source_threshold = float(metrics["cohorts"]["Historical"]["confidence"]["opportunity_threshold"])
    if not math.isclose(source_threshold, FROZEN_CONFIDENCE_THRESHOLD, rel_tol=0.0, abs_tol=5e-13):
        raise ValueError(f"motif-run opportunity threshold가 다릅니다: {source_threshold:.12f}")

    walk = pd.read_csv(motif_run / "walk_forward.csv")
    predictions = pd.read_csv(motif_run / "motif_predictions.csv")
    matches = pd.read_csv(motif_run / "recurrence_candidates.csv")
    expected_rounds = list(range(start_round, end_round + 1))
    if walk["round"].astype(int).tolist() != expected_rounds:
        raise ValueError("walk_forward 회차가 852~1235 연속 범위와 일치하지 않습니다")
    if walk["round"].duplicated().any():
        raise ValueError("walk_forward에 중복 회차가 있습니다")
    expected_cohorts = np.where(walk["round"].to_numpy() < split_round, "Historical", "Development")
    if not np.array_equal(walk["cohort"].to_numpy(), expected_cohorts):
        raise ValueError("Historical/Development cohort 경계가 일치하지 않습니다")
    if predictions.groupby("round").size().reindex(expected_rounds).isna().any():
        raise ValueError("motif_predictions에 빠진 평가 회차가 있습니다")
    match_counts = matches.groupby("target_round").size().reindex(expected_rounds)
    if match_counts.isna().any() or (match_counts > FROZEN_TOP_K).any():
        raise ValueError("recurrence_candidates 회차별 match 수가 유효하지 않습니다")

    selected_draws = [draw for draw in draws if draw.round_no <= end_round]
    round_numbers = [draw.round_no for draw in selected_draws]
    if round_numbers != list(range(round_numbers[0], end_round + 1)):
        raise ValueError("입력 데이터 회차가 연속적이지 않습니다")
    invalid_numbers = [
        draw.round_no for draw in selected_draws
        if len(draw.numbers) != 6 or len(set(draw.numbers)) != 6 or min(draw.numbers) < 1 or max(draw.numbers) > 45
    ]
    if invalid_numbers:
        raise ValueError(f"유효하지 않은 당첨 번호 회차: {invalid_numbers[:5]}")

    hashes = {name: _sha256(motif_run / name) for name in required}
    quality = {
        "status": "PASS",
        "draw_rows": len(selected_draws),
        "evaluation_rows": len(walk),
        "prediction_rows": len(predictions),
        "match_rows": len(matches),
        "evaluation_round_key_unique": True,
        "evaluation_rounds_complete": True,
        "cohort_boundary_valid": True,
        "draw_rounds_contiguous": True,
        "draw_numbers_valid": True,
        "target_label_ranges": ["852-1043", "1044-1235"],
        "sealed_target_ranges": {"Locked": "660-851", "Additional Blind": "468-659"},
        "source_hashes": hashes,
    }
    return walk, predictions, matches, metrics, quality


def _matches_by_round(
    matches: pd.DataFrame,
    bundle: FeatureBundle,
) -> dict[int, list[tuple[MotifMatch, np.ndarray]]]:
    round_to_index = {int(round_no): index for index, round_no in enumerate(bundle.rounds)}
    result: dict[int, list[tuple[MotifMatch, np.ndarray]]] = defaultdict(list)
    for record in matches.to_dict("records"):
        similarities = {key: float(value) for key, value in _json_value(record["view_similarities"]).items()}
        target_round = int(record["target_round"])
        followup_round = int(record["followup_round"])
        motif = MotifMatch(
            current_start=round_to_index[int(record["current_start_round"])],
            current_end=round_to_index[int(record["current_end_round"])],
            past_start=round_to_index[int(record["past_start_round"])],
            past_end=round_to_index[int(record["past_end_round"])],
            window_length=int(record["window_length"]),
            aggregate_similarity=float(record["aggregate_similarity"]),
            support_count=int(record["support_count"]),
            similarities=similarities,
        )
        result[target_round].append((motif, bundle.numbers[round_to_index[followup_round]]))
    for target_round in result:
        result[target_round].sort(key=lambda item: (-item[0].aggregate_similarity, item[0].past_end))
    return result


def _weighted_counts(
    motif_items: Sequence[tuple[MotifMatch, np.ndarray]],
    mapper: Any,
    size: int,
) -> np.ndarray:
    counts = np.zeros(size, dtype=float)
    for motif, numbers in motif_items:
        for value in mapper(motif, numbers):
            counts[int(value)] += motif.aggregate_similarity
    return counts


def _view_number_support(
    motif_items: Sequence[tuple[MotifMatch, np.ndarray]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    supports = {view: np.zeros(45, dtype=float) for view in VIEW_NAMES}
    counts = np.zeros(45, dtype=int)
    for motif, followup in motif_items:
        for number in followup:
            counts[int(number) - 1] += 1
            for view in VIEW_NAMES:
                supports[view][int(number) - 1] += float(motif.similarities[view])
    return supports, counts


def _opportunity_feature_row(
    *,
    base_row: Mapping[str, Any],
    motif_items: Sequence[tuple[MotifMatch, np.ndarray]],
    bundle: FeatureBundle,
    target_index: int,
) -> dict[str, Any]:
    motifs = [item[0] for item in motif_items]
    similarities = np.asarray([motif.aggregate_similarity for motif in motifs], dtype=float)
    separations = np.asarray([motif.current_end - motif.past_end for motif in motifs], dtype=float)
    mean_by_view = {view: float(mean(motif.similarities[view] for motif in motifs)) for view in VIEW_NAMES}
    ordered_views = sorted(VIEW_NAMES, key=lambda view: (-mean_by_view[view], view))
    view_values = np.asarray([mean_by_view[view] for view in VIEW_NAMES], dtype=float)
    ranks = {view: ordered_views.index(view) + 1 for view in VIEW_NAMES}

    number_counts = _weighted_counts(motif_items, lambda _m, numbers: numbers - 1, 45)
    grid_counts = _weighted_counts(motif_items, lambda _m, numbers: (numbers - 1) // 7, 7)
    circle_counts = _weighted_counts(motif_items, lambda _m, numbers: ((numbers - 1) * 12) // 45, 12)
    gap_counts = np.zeros(44, dtype=float)
    transition_counts = np.zeros(13, dtype=float)
    pair_counts: Counter[tuple[int, int]] = Counter()
    for motif, numbers in motif_items:
        gaps = np.diff(numbers)
        for gap in gaps:
            gap_counts[int(gap) - 1] += motif.aggregate_similarity
        previous = bundle.numbers[motif.past_end]
        delta_bin = int(np.clip(round(float(numbers.mean() - previous.mean()) / 3.0), -6, 6)) + 6
        transition_counts[delta_bin] += motif.aggregate_similarity
        for left, right in combinations(map(int, numbers), 2):
            pair_counts[(left, right)] += motif.aggregate_similarity

    ranked_numbers = np.asarray(_json_value(base_row["ranked_numbers"]), dtype=int)
    number_scores = np.asarray(_json_value(base_row["number_scores"]), dtype=float)
    top20 = ranked_numbers[:20]
    top20_scores = number_scores[top20 - 1]
    probabilities = top20_scores / max(float(top20_scores.sum()), 1e-12)
    candidate_entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-15)) / math.log(20))
    slope = float(np.polyfit(np.arange(1, 21), top20_scores, 1)[0])
    unique_followups = len({motif.past_end + 1 for motif in motifs})
    sorted_past = sorted(motif.past_end for motif in motifs)
    cluster_count = 1 + sum(right - left > FROZEN_QUERY_LENGTH for left, right in zip(sorted_past, sorted_past[1:]))

    hits20 = int(base_row["hits_at_20"])
    detailed, binary = classify_opportunity(hits20)
    is_opportunity = int(float(base_row["confidence"]) >= FROZEN_CONFIDENCE_THRESHOLD)
    return {
        "round": int(base_row["round"]),
        "cohort": str(base_row["cohort"]),
        "is_opportunity": is_opportunity,
        "opportunity_label": detailed if is_opportunity else "NON_OPPORTUNITY",
        "binary_label": binary if is_opportunity else "NON_OPPORTUNITY",
        "hits_at_20": hits20,
        "hits_at_30": int(base_row["hits_at_30"]),
        "confidence": float(base_row["confidence"]),
        "matched_motif_count": len(motifs),
        "valid_motif_count": sum(motif.past_end + 1 < target_index for motif in motifs),
        "best_similarity": float(similarities.max()),
        "median_similarity": float(np.median(similarities)),
        "top5_mean_similarity": float(similarities[:5].mean()),
        "top10_mean_similarity": float(similarities[:10].mean()),
        "similarity_variance": float(similarities.var()),
        "motif_separation_mean": float(separations.mean()),
        "motif_separation_median": float(np.median(separations)),
        "motif_separation_std": float(separations.std()),
        "motif_cluster_count": cluster_count,
        "recurrence_concentration": float(similarities[:5].sum() / similarities.sum()),
        "recurrence_dispersion": _normalized_entropy(similarities),
        **{f"{view}_similarity": mean_by_view[view] for view in VIEW_NAMES},
        **{f"{view}_rank": ranks[view] for view in VIEW_NAMES},
        "strongest_view": VIEW_LABELS[ordered_views[0]],
        "weakest_view": VIEW_LABELS[ordered_views[-1]],
        "cross_view_agreement_count": float(mean(motif.support_count for motif in motifs)),
        "view_variance": float(view_values.var()),
        "view_entropy": _normalized_entropy(view_values),
        "top2_view_gap": float(view_values.max() - np.partition(view_values, -2)[-2]),
        "number_entropy": _normalized_entropy(number_counts),
        "grid_entropy": _normalized_entropy(grid_counts),
        "circle_entropy": _normalized_entropy(circle_counts),
        "gap_entropy": _normalized_entropy(gap_counts),
        "transition_entropy": _normalized_entropy(transition_counts),
        "pair_concentration": float(max(pair_counts.values(), default=0.0) / max(sum(pair_counts.values()), 1e-12)),
        "region_concentration": float(grid_counts.max() / max(grid_counts.sum(), 1e-12)),
        "consensus_sharpness": float(1.0 - _normalized_entropy(number_counts)),
        "candidate_score_mean": float(top20_scores.mean()),
        "candidate_score_median": float(np.median(top20_scores)),
        "candidate_score_std": float(top20_scores.std()),
        "candidate_score_entropy": candidate_entropy,
        "candidate_score_slope": slope,
        "candidate_top5_bottom5_gap": float(top20_scores[:5].mean() - top20_scores[-5:].mean()),
        "candidate_grid_coverage": len(set(((top20 - 1) // 7).tolist())) / 7.0,
        "candidate_circle_coverage": len(set((((top20 - 1) * 12) // 45).tolist())) / 12.0,
        "motif_source_diversity": unique_followups / len(motifs),
        "candidate_concentration": float(base_row["candidate_concentration"]),
    }


def _add_formation_features(rows: list[dict[str, Any]]) -> None:
    previous_opportunity_round: int | None = None
    burst = 0
    for index, row in enumerate(rows):
        if row["is_opportunity"]:
            row["opportunity_gap"] = (
                float(row["round"] - previous_opportunity_round) if previous_opportunity_round is not None else math.nan
            )
            burst = burst + 1 if index > 0 and rows[index - 1]["is_opportunity"] else 1
            previous_opportunity_round = int(row["round"])
        else:
            row["opportunity_gap"] = math.nan
            burst = 0
        row["burst_length"] = burst
        left = max(0, index - 9)
        row["rolling_opportunity_density"] = float(mean(item["is_opportunity"] for item in rows[left : index + 1]))
        previous = rows[index - 1] if index > 0 and rows[index - 1]["cohort"] == row["cohort"] else None
        row["confidence_delta"] = float(row["confidence"] - previous["confidence"]) if previous else math.nan
        row["agreement_delta"] = (
            float(row["cross_view_agreement_count"] - previous["cross_view_agreement_count"]) if previous else math.nan
        )
        row["entropy_delta"] = float(row["number_entropy"] - previous["number_entropy"]) if previous else math.nan
        row["motif_count_delta"] = float(row["matched_motif_count"] - previous["matched_motif_count"]) if previous else math.nan
    for index, row in enumerate(rows):
        previous_is = bool(rows[index - 1]["is_opportunity"]) if index > 0 and rows[index - 1]["cohort"] == row["cohort"] else False
        next_is = bool(rows[index + 1]["is_opportunity"]) if index + 1 < len(rows) and rows[index + 1]["cohort"] == row["cohort"] else False
        row["isolated_opportunity"] = int(bool(row["is_opportunity"]) and not previous_is and not next_is)


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if not len(left) or not len(right):
        return math.nan
    comparisons = np.sign(left[:, None] - right[None, :])
    return float(comparisons.mean())


def cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = len(left) + len(right) - 2
    if len(left) < 2 or len(right) < 2 or denominator <= 0:
        return math.nan
    pooled = math.sqrt(
        ((len(left) - 1) * float(left.var(ddof=1)) + (len(right) - 1) * float(right.var(ddof=1)))
        / denominator
    )
    return float((left.mean() - right.mean()) / pooled) if pooled > 1e-12 else 0.0


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    iterations: int = MONTE_CARLO_ITERATIONS,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    chunk = 500
    simulated = np.empty(iterations, dtype=float)
    for start in range(0, iterations, chunk):
        count = min(chunk, iterations - start)
        left_sample = rng.choice(left, size=(count, len(left)), replace=True).mean(axis=1)
        right_sample = rng.choice(right, size=(count, len(right)), replace=True).mean(axis=1)
        simulated[start : start + count] = left_sample - right_sample
    return float(np.quantile(simulated, 0.025)), float(np.quantile(simulated, 0.975))


def _permutation_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    iterations: int = MONTE_CARLO_ITERATIONS,
) -> float:
    observed = abs(float(left.mean() - right.mean()))
    combined = np.r_[left, right]
    left_count = len(left)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        shuffled = rng.permutation(combined)
        effect = abs(float(shuffled[:left_count].mean() - shuffled[left_count:].mean()))
        exceed += effect >= observed - 1e-15
    return float((exceed + 1) / (iterations + 1))


def _numeric_feature_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    excluded = {
        "round", "is_opportunity", "hits_at_20", "hits_at_30", "confidence",
        "opportunity_gap", "burst_length", "isolated_opportunity", "rolling_opportunity_density",
        "confidence_delta", "agreement_delta", "entropy_delta", "motif_count_delta",
    }
    ordered = [feature for family in FEATURE_FAMILIES.values() for feature in family if feature not in excluded]
    return [feature for feature in ordered if feature in rows[0]]


def _comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    comparison: str,
    feature_names: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cohort_index, cohort in enumerate(("Historical", "Development")):
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        if comparison == "OPPORTUNITY_VS_NON_OPPORTUNITY":
            left = [row for row in cohort_rows if row["is_opportunity"]]
            right = [row for row in cohort_rows if not row["is_opportunity"]]
            left_label, right_label = "OPPORTUNITY", "NON_OPPORTUNITY"
        elif comparison == "SUCCESS_4PLUS_VS_FAIL_BELOW4":
            opportunities = [row for row in cohort_rows if row["is_opportunity"]]
            left = [row for row in opportunities if row["binary_label"] == "SUCCESS_4PLUS"]
            right = [row for row in opportunities if row["binary_label"] == "FAIL_BELOW4"]
            left_label, right_label = "SUCCESS_4PLUS", "FAIL_BELOW4"
        elif comparison == "HIT_5_PLUS_VS_FAIL_0_2":
            opportunities = [row for row in cohort_rows if row["is_opportunity"]]
            left = [row for row in opportunities if row["opportunity_label"] == "HIT_5_PLUS"]
            right = [row for row in opportunities if row["opportunity_label"] == "FAIL_0_2"]
            left_label, right_label = "HIT_5_PLUS", "FAIL_0_2"
        else:
            raise ValueError(f"알 수 없는 comparison: {comparison}")

        cohort_records: list[dict[str, Any]] = []
        for feature_index, feature in enumerate(feature_names):
            left_values = np.asarray([float(row[feature]) for row in left if pd.notna(row[feature])], dtype=float)
            right_values = np.asarray([float(row[feature]) for row in right if pd.notna(row[feature])], dtype=float)
            if not len(left_values) or not len(right_values):
                continue
            feature_seed = seed + cohort_index * 100_000 + feature_index * 17
            ci_low, ci_high = _bootstrap_difference(left_values, right_values, seed=feature_seed)
            p_value = _permutation_difference(left_values, right_values, seed=feature_seed ^ 0x5A17)
            family = next(name for name, names in FEATURE_FAMILIES.items() if feature in names)
            cohort_records.append(
                {
                    "comparison": comparison,
                    "cohort": cohort,
                    "feature_family": family,
                    "feature": feature,
                    "left_label": left_label,
                    "right_label": right_label,
                    "left_n": len(left_values),
                    "right_n": len(right_values),
                    "left_mean": float(left_values.mean()),
                    "right_mean": float(right_values.mean()),
                    "left_median": float(np.median(left_values)),
                    "right_median": float(np.median(right_values)),
                    "mean_difference": float(left_values.mean() - right_values.mean()),
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                    "permutation_p": p_value,
                    "cliffs_delta": cliffs_delta(left_values, right_values),
                    "cohens_d": cohens_d(left_values, right_values),
                }
            )
        q_values = _benjamini_hochberg([row["permutation_p"] for row in cohort_records])
        for row, q_value in zip(cohort_records, q_values, strict=True):
            row["fdr_q"] = q_value
        output.extend(cohort_records)
    return output


def _same_direction_mechanisms(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [row for row in comparisons if row["comparison"] == "SUCCESS_4PLUS_VS_FAIL_BELOW4"]
    by_feature: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in primary:
        by_feature[str(row["feature"])][str(row["cohort"])] = row
    output: list[dict[str, Any]] = []
    for feature, cohorts in by_feature.items():
        if set(cohorts) != {"Historical", "Development"}:
            continue
        historical = cohorts["Historical"]
        development = cohorts["Development"]
        same_direction = float(historical["mean_difference"]) * float(development["mean_difference"]) > 0
        replicated = bool(
            same_direction
            and abs(float(historical["cliffs_delta"])) >= 0.147
            and abs(float(development["cliffs_delta"])) >= 0.147
            and float(historical["fdr_q"]) <= 0.10
            and float(development["fdr_q"]) <= 0.10
        )
        output.append(
            {
                "feature": feature,
                "feature_family": historical["feature_family"],
                "historical_mean_difference": historical["mean_difference"],
                "development_mean_difference": development["mean_difference"],
                "historical_cliffs_delta": historical["cliffs_delta"],
                "development_cliffs_delta": development["cliffs_delta"],
                "historical_fdr_q": historical["fdr_q"],
                "development_fdr_q": development["fdr_q"],
                "same_direction": same_direction,
                "replicated": replicated,
            }
        )
    return sorted(output, key=lambda row: (not row["replicated"], -min(abs(row["historical_cliffs_delta"]), abs(row["development_cliffs_delta"]))))


def _reweighted_matches(
    motif_items: Sequence[tuple[MotifMatch, np.ndarray]],
    selected_views: Sequence[str],
) -> list[MotifMatch]:
    result: list[MotifMatch] = []
    for motif, _followup in motif_items:
        values = sorted((float(motif.similarities[view]) for view in selected_views), reverse=True)
        keep = max(1, math.ceil(len(values) * 0.67))
        aggregate = float(mean(values[:keep]))
        result.append(
            replace(
                motif,
                aggregate_similarity=aggregate,
                support_count=sum(float(motif.similarities[view]) >= 0.50 for view in selected_views),
            )
        )
    return sorted(result, key=lambda motif: (-motif.aggregate_similarity, motif.past_end, motif.window_length))[:FROZEN_TOP_K]


def _variant_round_result(
    *,
    target_round: int,
    bundle: FeatureBundle,
    motif_items: Sequence[tuple[MotifMatch, np.ndarray]],
    selected_views: Sequence[str],
) -> dict[str, Any]:
    round_to_index = {int(round_no): index for index, round_no in enumerate(bundle.rounds)}
    target_index = round_to_index[target_round]
    motifs = _reweighted_matches(motif_items, selected_views)
    scores, _components = _candidate_scores(bundle, target_index - 1, motifs)
    ranking = np.argsort(-scores, kind="stable") + 1
    winner = set(bundle.numbers[target_index].tolist())
    counts = np.zeros(45, dtype=float)
    for motif in motifs:
        counts[bundle.numbers[motif.past_end + 1] - 1] += motif.aggregate_similarity
    entropy = _normalized_entropy(counts)
    top_similarity = float(mean(motif.aggregate_similarity for motif in motifs[: min(5, len(motifs))]))
    agreement = float(mean(motif.support_count for motif in motifs))
    confidence = top_similarity * max(0.0, 1.0 - entropy) * (agreement / len(selected_views))
    return {
        "round": target_round,
        "confidence": confidence,
        "is_opportunity": int(confidence >= FROZEN_CONFIDENCE_THRESHOLD),
        "hits_at_20": len(winner.intersection(ranking[:20].tolist())),
        "hits_at_30": len(winner.intersection(ranking[:30].tolist())),
    }


def _summarize_variant(
    rows: Sequence[Mapping[str, Any]],
    *,
    cohort: str,
    category: str,
    variant: str,
    selected_views: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    candidate = [row for row in rows if row["cohort"] == cohort]
    opportunity = [row for row in candidate if row["is_opportunity"]]
    hits = np.asarray([int(row["hits_at_20"]) for row in candidate], dtype=int)
    opportunity_hits = np.asarray([int(row["hits_at_20"]) for row in opportunity], dtype=int)
    baseline = _random_baseline(len(opportunity), 20, opportunity_hits, seed) if opportunity else None
    return {
        "category": category,
        "variant": variant,
        "selected_views": ";".join(selected_views),
        "cohort": cohort,
        "rounds": len(candidate),
        "mean_hit_at_20": float(hits.mean()),
        "hit_4_plus": int(np.sum(hits >= 4)),
        "hit_5_plus": int(np.sum(hits >= 5)),
        "hit_6": int(np.sum(hits >= 6)),
        "opportunity_rounds": len(opportunity),
        "opportunity_coverage": len(opportunity) / len(candidate),
        "opportunity_mean_hit_at_20": float(opportunity_hits.mean()) if len(opportunity_hits) else math.nan,
        "opportunity_random_lift": float(baseline["mean_hit_lift"]) if baseline else math.nan,
        "opportunity_mean_hit_p": float(baseline["mean_hit_p"]) if baseline else math.nan,
        "opportunity_4_plus": int(np.sum(opportunity_hits >= 4)),
        "opportunity_5_plus": int(np.sum(opportunity_hits >= 5)),
        "opportunity_6": int(np.sum(opportunity_hits >= 6)),
    }


def _view_diagnostics(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    bundle: FeatureBundle,
    motif_by_round: Mapping[int, Sequence[tuple[MotifMatch, np.ndarray]]],
    seed: int,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    specifications: list[tuple[str, str, tuple[str, ...]]] = [("ablation", "ALL", VIEW_NAMES)]
    specifications.extend(
        ("ablation", f"ALL - {VIEW_LABELS[view]}", tuple(item for item in VIEW_NAMES if item != view))
        for view in VIEW_NAMES
    )
    specifications.extend(("single_view", f"{VIEW_LABELS[view]}-only", (view,)) for view in VIEW_NAMES)
    pair_views = (
        ("grid", "circle"), ("grid", "transition"), ("circle", "transition"),
        ("transition", "context"), ("distribution", "context"),
    )
    specifications.extend(
        ("pair_interaction", " + ".join(VIEW_LABELS[view] for view in pair), pair) for pair in pair_views
    )

    all_variant_rows: dict[str, list[dict[str, Any]]] = {}
    feature_by_round = {int(row["round"]): row for row in feature_rows}
    for specification_index, (category, variant, selected_views) in enumerate(specifications, start=1):
        rows: list[dict[str, Any]] = []
        for target_round in range(EVALUATION_START, EVALUATION_END + 1):
            result = _variant_round_result(
                target_round=target_round,
                bundle=bundle,
                motif_items=motif_by_round[target_round],
                selected_views=selected_views,
            )
            result["cohort"] = feature_by_round[target_round]["cohort"]
            rows.append(result)
        all_variant_rows[f"{category}:{variant}"] = rows
        logger.info("View diagnostic 완료 | %s/%s | %s", specification_index, len(specifications), variant)

    summarized: list[dict[str, Any]] = []
    for specification_index, (category, variant, selected_views) in enumerate(specifications):
        rows = all_variant_rows[f"{category}:{variant}"]
        for cohort_index, cohort in enumerate(("Historical", "Development")):
            summarized.append(
                _summarize_variant(
                    rows,
                    cohort=cohort,
                    category=category,
                    variant=variant,
                    selected_views=selected_views,
                    seed=seed + specification_index * 100 + cohort_index,
                )
            )
    return (
        [row for row in summarized if row["category"] == "ablation"],
        [row for row in summarized if row["category"] == "single_view"],
        [row for row in summarized if row["category"] == "pair_interaction"],
        all_variant_rows,
    )


def classify_motif_family(motif: MotifMatch) -> str:
    """Assign a target-independent family using only the frozen support vector."""
    values = {view: float(motif.similarities[view]) for view in VIEW_NAMES}
    variance = float(np.var(list(values.values())))
    strongest = max(values, key=lambda view: (values[view], view))
    if motif.aggregate_similarity >= 0.57 and motif.support_count <= 3:
        return "high-similarity-low-support"
    if 0.50 <= motif.aggregate_similarity < 0.57 and motif.support_count >= 5:
        return "moderate-similarity-wide-support"
    if motif.support_count >= 5 and variance <= 0.0015:
        return "high-agreement"
    if motif.support_count <= 2:
        return "low-agreement"
    if variance <= 0.0005:
        return "balanced-multiview"
    if strongest in {"raw", "grid", "circle"}:
        return "shape-dominant"
    if strongest == "transition":
        return "transition-dominant"
    return "context-dominant"


def _family_entropy(items: Sequence[tuple[MotifMatch, np.ndarray]]) -> float:
    counts = np.zeros(45, dtype=float)
    for motif, followup in items:
        counts[followup - 1] += motif.aggregate_similarity
    return _normalized_entropy(counts)


def _motif_family_analysis(
    feature_rows: Sequence[Mapping[str, Any]],
    motif_by_round: Mapping[int, Sequence[tuple[MotifMatch, np.ndarray]]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    feature_by_round = {int(row["round"]): row for row in feature_rows}
    round_distributions: dict[int, dict[str, float]] = {}
    family_items: dict[tuple[str, str], list[tuple[int, MotifMatch, np.ndarray]]] = defaultdict(list)
    for target_round, items in motif_by_round.items():
        counts: Counter[str] = Counter()
        for motif, followup in items:
            family = classify_motif_family(motif)
            counts[family] += 1
            cohort = str(feature_by_round[target_round]["cohort"])
            family_items[(cohort, family)].append((target_round, motif, followup))
        total = max(1, sum(counts.values()))
        round_distributions[target_round] = {family: count / total for family, count in counts.items()}

    output: list[dict[str, Any]] = []
    random_mean = 6.0 * PRIMARY_CANDIDATE_SIZE / 45.0
    for (cohort, family), items in sorted(family_items.items()):
        target_rounds = sorted({target_round for target_round, _motif, _followup in items})
        opportunity_rounds = [target_round for target_round in target_rounds if feature_by_round[target_round]["is_opportunity"]]
        hits = np.asarray([int(feature_by_round[target_round]["hits_at_20"]) for target_round in target_rounds], dtype=int)
        entropy_items = [(motif, followup) for _round, motif, followup in items]
        output.append(
            {
                "cohort": cohort,
                "family": family,
                "occurrence": len(items),
                "round_count": len(target_rounds),
                "opportunity_count": len(opportunity_rounds),
                "mean_hit_at_20": float(hits.mean()),
                "hit_4_plus": int(np.sum(hits >= 4)),
                "hit_5_plus": int(np.sum(hits >= 5)),
                "hit_6": int(np.sum(hits >= 6)),
                "random_lift": float(hits.mean() - random_mean),
                "followup_entropy": _family_entropy(entropy_items),
            }
        )
    return output, round_distributions


def _second_order_analysis(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    family_distributions: Mapping[int, Mapping[str, float]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    opportunities = [dict(row) for row in feature_rows if row["is_opportunity"]]
    families = sorted({family for distribution in family_distributions.values() for family in distribution})
    meta_features = [
        "best_similarity", "median_similarity", *(f"{view}_similarity" for view in VIEW_NAMES),
        "cross_view_agreement_count", "motif_separation_mean",
        "number_entropy", "grid_entropy", "circle_entropy", "gap_entropy", "transition_entropy",
        "candidate_score_mean", "candidate_score_std", "candidate_score_entropy",
        "candidate_score_slope", "candidate_top5_bottom5_gap", "motif_source_diversity",
    ]
    matrix = np.asarray(
        [
            [
                *(float(row[feature]) for feature in meta_features),
                *(float(family_distributions[int(row["round"])].get(family, 0.0)) for family in families),
            ]
            for row in opportunities
        ],
        dtype=float,
    )
    historical_mask = np.asarray([row["cohort"] == "Historical" for row in opportunities])
    center = matrix[historical_mask].mean(axis=0)
    scale = matrix[historical_mask].std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (matrix - center) / scale
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(standardized)
    coordinate_rows = [
        {
            "round": int(row["round"]),
            "cohort": row["cohort"],
            "binary_label": row["binary_label"],
            "pc1": float(coordinates[index, 0]),
            "pc2": float(coordinates[index, 1]),
        }
        for index, row in enumerate(opportunities)
    ]

    summary_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "meta_features": [*meta_features, *(f"family_{family}" for family in families)],
        "standardization": "Historical mean/std frozen and applied to Development",
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cohorts": {},
    }
    for cohort_index, cohort in enumerate(("Historical", "Development")):
        indices = [index for index, row in enumerate(opportunities) if row["cohort"] == cohort]
        cohort_matrix = standardized[indices]
        labels = np.asarray([opportunities[index]["binary_label"] == "SUCCESS_4PLUS" for index in indices])
        distances = np.sqrt(np.maximum(0.0, ((cohort_matrix[:, None, :] - cohort_matrix[None, :, :]) ** 2).sum(axis=2)))
        categories: dict[str, list[float]] = {"SUCCESS_SUCCESS": [], "SUCCESS_FAILURE": [], "FAILURE_FAILURE": []}
        for left, right in combinations(range(len(indices)), 2):
            if labels[left] and labels[right]:
                category = "SUCCESS_SUCCESS"
            elif labels[left] != labels[right]:
                category = "SUCCESS_FAILURE"
            else:
                category = "FAILURE_FAILURE"
            categories[category].append(float(distances[left, right]))
        for category, values in categories.items():
            summary_rows.append(
                {
                    "cohort": cohort,
                    "comparison": category,
                    "pair_count": len(values),
                    "mean_distance": float(np.mean(values)) if values else math.nan,
                    "median_distance": float(np.median(values)) if values else math.nan,
                }
            )
        success_success = np.asarray(categories["SUCCESS_SUCCESS"], dtype=float)
        success_failure = np.asarray(categories["SUCCESS_FAILURE"], dtype=float)
        observed_effect = (
            float(success_failure.mean() - success_success.mean())
            if len(success_success) and len(success_failure) else math.nan
        )
        rng = np.random.default_rng(seed + cohort_index)
        simulated = np.empty(MONTE_CARLO_ITERATIONS, dtype=float)
        success_count = int(labels.sum())
        for iteration in range(MONTE_CARLO_ITERATIONS):
            permuted = np.zeros(len(labels), dtype=bool)
            permuted[rng.choice(len(labels), size=success_count, replace=False)] = True
            same_success: list[float] = []
            cross: list[float] = []
            for left, right in combinations(range(len(labels)), 2):
                if permuted[left] and permuted[right]:
                    same_success.append(float(distances[left, right]))
                elif permuted[left] != permuted[right]:
                    cross.append(float(distances[left, right]))
            simulated[iteration] = float(np.mean(cross) - np.mean(same_success))
        p_value = float((np.sum(simulated >= observed_effect) + 1) / (MONTE_CARLO_ITERATIONS + 1))
        summary["cohorts"][cohort] = {
            "opportunity_rounds": len(indices),
            "success_rounds": success_count,
            "success_failure_minus_success_success_distance": observed_effect,
            "label_permutation_p": p_value,
            "permutation_effect_95_interval": [float(np.quantile(simulated, 0.025)), float(np.quantile(simulated, 0.975))],
        }
        for row in summary_rows:
            if row["cohort"] == cohort:
                row["separation_effect"] = observed_effect
                row["label_permutation_p"] = p_value
    return summary_rows, coordinate_rows, summary


def _stage_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_size: int,
    seed: int,
) -> dict[str, Any]:
    hit_key = f"hits_at_{candidate_size}"
    hits = np.asarray([int(row[hit_key]) for row in rows], dtype=int)
    if not len(hits):
        return {
            "rounds": 0,
            "coverage_of_all_evaluation_rounds": 0.0,
            "observed_mean_hits": math.nan,
            "mean_hit_lift": math.nan,
            "mean_hit_p": math.nan,
            "observed_4_plus_rate": math.nan,
            "random_4_plus_rate": math.nan,
            "four_plus_rate_lift": math.nan,
            "observed_5_plus_rate": math.nan,
            "random_5_plus_rate": math.nan,
            "five_plus_rate_lift": math.nan,
            "observed_6_rate": math.nan,
        }
    baseline = _random_baseline(len(rows), candidate_size, hits, seed)
    output = dict(baseline)
    output.update(
        {
            "rounds": len(rows),
            "coverage_of_all_evaluation_rounds": len(rows) / 192.0,
            "observed_4_plus_rate": float(np.mean(hits >= 4)),
            "random_4_plus_rate": float(baseline["expected_counts"]["4_plus"] / len(rows)),
            "four_plus_rate_lift": float(np.mean(hits >= 4) - baseline["expected_counts"]["4_plus"] / len(rows)),
            "observed_5_plus_rate": float(np.mean(hits >= 5)),
            "random_5_plus_rate": float(baseline["expected_counts"]["5_plus"] / len(rows)),
            "five_plus_rate_lift": float(np.mean(hits >= 5) - baseline["expected_counts"]["5_plus"] / len(rows)),
            "observed_6_rate": float(np.mean(hits >= 6)),
        }
    )
    return output


def _rule_condition(
    feature_rows: Sequence[Mapping[str, Any]],
    feature: str,
    keep_fraction: float,
) -> dict[str, Any]:
    historical = [row for row in feature_rows if row["cohort"] == "Historical" and row["is_opportunity"]]
    success = np.asarray([float(row[feature]) for row in historical if row["binary_label"] == "SUCCESS_4PLUS"], dtype=float)
    failure = np.asarray([float(row[feature]) for row in historical if row["binary_label"] == "FAIL_BELOW4"], dtype=float)
    direction = ">=" if float(success.mean() - failure.mean()) >= 0 else "<="
    quantile = 1.0 - keep_fraction if direction == ">=" else keep_fraction
    threshold = float(np.quantile([float(row[feature]) for row in historical], quantile))
    return {"feature": feature, "operator": direction, "threshold": threshold, "historical_quantile": quantile}


def _passes_conditions(row: Mapping[str, Any], conditions: Sequence[Mapping[str, Any]]) -> bool:
    for condition in conditions:
        value = float(row[str(condition["feature"])])
        threshold = float(condition["threshold"])
        if condition["operator"] == ">=" and value < threshold:
            return False
        if condition["operator"] == "<=" and value > threshold:
            return False
    return True


def _quality_rules(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    # Five predeclared diagnostic candidates; only directions and quantiles use Historical labels.
    specifications = (
        ("R1_AGREEMENT", ("cross_view_agreement_count",), 0.55),
        ("R2_CONSENSUS", ("consensus_sharpness",), 0.55),
        ("R3_SHARPNESS", ("candidate_top5_bottom5_gap",), 0.55),
        ("R4_AGREEMENT_CONSENSUS", ("cross_view_agreement_count", "consensus_sharpness"), 0.78),
        ("R5_RECURRENCE_SHARPNESS", ("recurrence_concentration", "candidate_top5_bottom5_gap"), 0.78),
    )
    opportunity_rows = [row for row in feature_rows if row["is_opportunity"]]
    candidate_rules: list[dict[str, Any]] = []
    for rule_index, (rule_id, features, keep_fraction) in enumerate(specifications):
        conditions = [_rule_condition(feature_rows, feature, keep_fraction) for feature in features]
        cohort_metrics: dict[str, Any] = {}
        for cohort_index, cohort in enumerate(("Historical", "Development")):
            selected = [row for row in opportunity_rows if row["cohort"] == cohort and _passes_conditions(row, conditions)]
            cohort_metrics[cohort] = {
                "top20": _stage_metrics(selected, candidate_size=20, seed=seed + rule_index * 100 + cohort_index),
                "top30": _stage_metrics(selected, candidate_size=30, seed=seed + rule_index * 100 + cohort_index + 10_000),
            }
        candidate_rules.append(
            {
                "rule_id": rule_id,
                "condition_count": len(conditions),
                "conditions": conditions,
                "Historical": cohort_metrics["Historical"],
                "Development": cohort_metrics["Development"],
            }
        )

    # Selection never reads Development metrics: maximize Historical @20 lift, then coverage, then rule id.
    selected_rule = max(
        candidate_rules,
        key=lambda rule: (
            float(rule["Historical"]["top20"]["mean_hit_lift"]),
            int(rule["Historical"]["top20"]["rounds"]),
            str(rule["rule_id"]),
        ),
    )
    output_rows: list[dict[str, Any]] = []
    for rule in candidate_rules:
        for cohort in ("Historical", "Development"):
            top20 = rule[cohort]["top20"]
            top30 = rule[cohort]["top30"]
            output_rows.append(
                {
                    "rule_id": rule["rule_id"],
                    "selected": int(rule["rule_id"] == selected_rule["rule_id"]),
                    "selection_policy": "Historical @20 lift, then Historical coverage; Development untouched",
                    "condition_count": rule["condition_count"],
                    "conditions": rule["conditions"],
                    "cohort": cohort,
                    "coverage_rounds": top20["rounds"],
                    "coverage_of_evaluation": top20["coverage_of_all_evaluation_rounds"],
                    "mean_hit_at_20": top20["observed_mean_hits"],
                    "random_lift_at_20": top20["mean_hit_lift"],
                    "mean_hit_p_at_20": top20["mean_hit_p"],
                    "four_plus_rate_at_20": top20["observed_4_plus_rate"],
                    "four_plus_rate_lift_at_20": top20["four_plus_rate_lift"],
                    "five_plus_rate_at_20": top20["observed_5_plus_rate"],
                    "five_plus_rate_lift_at_20": top20["five_plus_rate_lift"],
                    "six_rate_at_20": top20["observed_6_rate"],
                    "mean_hit_at_30": top30["observed_mean_hits"],
                    "random_lift_at_30": top30["mean_hit_lift"],
                    "mean_hit_p_at_30": top30["mean_hit_p"],
                }
            )

    stage2_rows = [
        {
            **dict(row),
            "stage2_pass": int(_passes_conditions(row, selected_rule["conditions"])),
            "selected_rule_id": selected_rule["rule_id"],
        }
        for row in opportunity_rows
    ]
    return output_rows, selected_rule, stage2_rows


def _candidate_diagnostics(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    base_walk: pd.DataFrame,
    motif_by_round: Mapping[int, Sequence[tuple[MotifMatch, np.ndarray]]],
    bundle: FeatureBundle,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_by_round = {int(row["round"]): row for row in feature_rows}
    walk_by_round = {int(row["round"]): row for row in base_walk.to_dict("records")}
    round_to_index = {int(round_no): index for index, round_no in enumerate(bundle.rounds)}
    funnel_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    false_positive_rows: list[dict[str, Any]] = []
    for target_round in range(EVALUATION_START, EVALUATION_END + 1):
        feature = feature_by_round[target_round]
        if not feature["is_opportunity"]:
            continue
        target_index = round_to_index[target_round]
        base = walk_by_round[target_round]
        ranking = np.asarray(_json_value(base["ranked_numbers"]), dtype=int)
        scores = np.asarray(_json_value(base["number_scores"]), dtype=float)
        winner = set(map(int, bundle.numbers[target_index]))
        supports, support_counts = _view_number_support(motif_by_round[target_round])
        motif_family_support: dict[int, Counter[str]] = {number: Counter() for number in range(1, 46)}
        for motif, followup in motif_by_round[target_round]:
            family = classify_motif_family(motif)
            for number in map(int, followup):
                motif_family_support[number][family] += motif.aggregate_similarity

        rank_by_number = {int(number): rank for rank, number in enumerate(ranking, start=1)}
        for number in sorted(winner):
            rank = rank_by_number[number]
            support_values = {view: float(supports[view][number - 1]) for view in VIEW_NAMES}
            strongest = max(VIEW_NAMES, key=lambda view: (support_values[view], view))
            weakest = min(VIEW_NAMES, key=lambda view: (support_values[view], view))
            maximum_support = max(support_values.values(), default=0.0)
            supporting_views = [
                VIEW_LABELS[view] for view in VIEW_NAMES
                if support_values[view] >= maximum_support * 0.75 and maximum_support > 0
            ]
            if rank <= 10:
                drop_stage = "SURVIVES_TOP10"
            elif rank <= 15:
                drop_stage = "DROPS_AT_TOP10"
            elif rank <= 20:
                drop_stage = "DROPS_AT_TOP15"
            elif rank <= 25:
                drop_stage = "DROPS_AT_TOP20"
            elif rank <= 30:
                drop_stage = "DROPS_AT_TOP25"
            else:
                drop_stage = "OUTSIDE_TOP30"
            common = {
                "round": target_round,
                "cohort": feature["cohort"],
                "winner": number,
                "motif_rank": rank,
                "score": float(scores[number - 1]),
                "supporting_views": supporting_views,
                "supporting_motif_count": int(support_counts[number - 1]),
                "drop_stage": drop_stage,
            }
            funnel_rows.append(common)
            if rank > PRIMARY_CANDIDATE_SIZE:
                missing_rows.append(
                    {
                        **common,
                        "actual_rank": rank,
                        "score_percentile": float(1.0 - (rank - 1) / 44.0),
                        "strongest_supporting_view": VIEW_LABELS[strongest],
                        "weakest_supporting_view": VIEW_LABELS[weakest],
                        **{f"{view}_support": support_values[view] for view in VIEW_NAMES},
                    }
                )

        for rank, number in enumerate(ranking[:PRIMARY_CANDIDATE_SIZE], start=1):
            number = int(number)
            if number in winner:
                continue
            support_values = {view: float(supports[view][number - 1]) for view in VIEW_NAMES}
            strongest = max(VIEW_NAMES, key=lambda view: (support_values[view], view))
            family_counter = motif_family_support[number]
            dominant_family = family_counter.most_common(1)[0][0] if family_counter else "none"
            false_positive_rows.append(
                {
                    "round": target_round,
                    "cohort": feature["cohort"],
                    "number": number,
                    "rank": rank,
                    "score": float(scores[number - 1]),
                    "strongest_supporting_view": VIEW_LABELS[strongest],
                    "dominant_motif_family": dominant_family,
                    "motif_support_count": int(support_counts[number - 1]),
                    **{f"{view}_support": support_values[view] for view in VIEW_NAMES},
                }
            )
    return funnel_rows, missing_rows, false_positive_rows


def _stage1_stage2_summary(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    stage2_rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for cohort_index, cohort in enumerate(("Historical", "Development")):
        stage1 = [row for row in feature_rows if row["cohort"] == cohort and row["is_opportunity"]]
        stage2 = [row for row in stage2_rows if row["cohort"] == cohort and row["stage2_pass"]]
        summary[cohort] = {
            "Stage1": {
                "top20": _stage_metrics(stage1, candidate_size=20, seed=seed + cohort_index),
                "top30": _stage_metrics(stage1, candidate_size=30, seed=seed + cohort_index + 100),
            },
            "Stage2": {
                "top20": _stage_metrics(stage2, candidate_size=20, seed=seed + cohort_index + 1_000),
                "top30": _stage_metrics(stage2, candidate_size=30, seed=seed + cohort_index + 1_100),
            },
        }
    return summary


def _ablation_mechanism(view_ablation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in view_ablation:
        by_variant[str(row["variant"])][str(row["cohort"])] = row
    all_rows = by_variant["ALL"]
    supported: list[dict[str, Any]] = []
    for variant, cohorts in by_variant.items():
        if variant == "ALL" or set(cohorts) != {"Historical", "Development"}:
            continue
        changes = {
            cohort: float(all_rows[cohort]["opportunity_random_lift"] - cohorts[cohort]["opportunity_random_lift"])
            for cohort in ("Historical", "Development")
        }
        supported.append(
            {
                "variant": variant,
                "historical_lift_drop": changes["Historical"],
                "development_lift_drop": changes["Development"],
                "same_direction_collapse": changes["Historical"] > 0 and changes["Development"] > 0,
            }
        )
    return {
        "diagnostic_method": "Frozen Top-40 match pool; view-support vector reweighting without target labels",
        "removals": supported,
        "supporting_removals": [row["variant"] for row in supported if row["same_direction_collapse"]],
    }


def _verdict(
    *,
    stage_summary: Mapping[str, Any],
    selected_rule: Mapping[str, Any],
    replicated_features: Sequence[Mapping[str, Any]],
    ablation: Mapping[str, Any],
) -> tuple[str, str, dict[str, bool]]:
    historical_stage1 = stage_summary["Historical"]["Stage1"]["top20"]
    historical_stage2 = stage_summary["Historical"]["Stage2"]["top20"]
    development_stage1 = stage_summary["Development"]["Stage1"]["top20"]
    development_stage2 = stage_summary["Development"]["Stage2"]["top20"]
    conditions = {
        "historical_only_rule_definition": "Development" not in json.dumps(selected_rule["conditions"]),
        "development_stage2_at_least_30": int(development_stage2["rounds"]) >= 30,
        "stage2_lift_above_stage1_both": (
            float(historical_stage2["mean_hit_lift"]) > float(historical_stage1["mean_hit_lift"])
            and float(development_stage2["mean_hit_lift"]) > float(development_stage1["mean_hit_lift"])
        ),
        "stage2_lift_same_positive_direction": (
            float(historical_stage2["mean_hit_lift"]) > 0 and float(development_stage2["mean_hit_lift"]) > 0
        ),
        "four_plus_lift_same_positive_direction": (
            float(historical_stage2["four_plus_rate_lift"]) > 0
            and float(development_stage2["four_plus_rate_lift"]) > 0
        ),
        "replicated_structural_feature": any(bool(row["replicated"]) for row in replicated_features),
        "view_ablation_support": bool(ablation["supporting_removals"]),
    }
    mandatory = (
        conditions["historical_only_rule_definition"]
        and conditions["development_stage2_at_least_30"]
        and conditions["stage2_lift_above_stage1_both"]
        and conditions["stage2_lift_same_positive_direction"]
    )
    if mandatory and sum(conditions.values()) >= 6:
        return "SUCCESS", "A", conditions
    if conditions["stage2_lift_above_stage1_both"] and conditions["stage2_lift_same_positive_direction"]:
        return "WEAK SIGNAL", "B", conditions
    return "NO SIGNAL", "C", conditions


def _plot_artifacts(
    *,
    run_dir: Path,
    feature_rows: Sequence[Mapping[str, Any]],
    replicated_features: Sequence[Mapping[str, Any]],
    view_ablation: Sequence[Mapping[str, Any]],
    second_coordinates: Sequence[Mapping[str, Any]],
    quality_rules: Sequence[Mapping[str, Any]],
    funnel_rows: Sequence[Mapping[str, Any]],
    missing_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "blue": "#2f6fed", "gold": "#d7a514", "orange": "#d97706", "pink": "#c2417b",
        "ink": "#172033", "muted": "#667085", "grid": "#e4e8ef", "open": "#dce7ff",
    }

    top_features = list(replicated_features[:10])
    if not top_features:
        top_features = []
    figure, axis = plt.subplots(figsize=(10.8, 6.4))
    labels = [row["feature"] for row in reversed(top_features)]
    positions = np.arange(len(labels))
    if labels:
        hist = [row["historical_cliffs_delta"] for row in reversed(top_features)]
        dev = [row["development_cliffs_delta"] for row in reversed(top_features)]
        axis.scatter(hist, positions - 0.12, color=palette["blue"], label="Historical", s=52)
        axis.scatter(dev, positions + 0.12, color=palette["gold"], label="Development", s=52, marker="s")
        axis.axvline(0, color=palette["muted"], linewidth=1)
        axis.set_yticks(positions, labels)
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No comparable structural features", ha="center", va="center", transform=axis.transAxes)
        axis.set_yticks([])
    axis.set_title("Success vs failure structural effects", loc="left", color=palette["ink"], fontweight="bold")
    axis.set_xlabel("Cliff's delta (SUCCESS_4PLUS − FAIL_BELOW4)")
    axis.grid(axis="x", color=palette["grid"])
    figure.tight_layout()
    figure.savefig(plot_dir / "success_vs_failure_features.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    variants = [row["variant"] for row in view_ablation if row["cohort"] == "Historical"]
    positions = np.arange(len(variants))
    for offset, cohort, color in ((-0.18, "Historical", palette["blue"]), (0.18, "Development", palette["gold"])):
        values = [next(row["opportunity_random_lift"] for row in view_ablation if row["variant"] == variant and row["cohort"] == cohort) for variant in variants]
        axis.bar(positions + offset, values, 0.36, label=cohort, color=color)
    axis.axhline(0, color=palette["muted"], linewidth=1)
    axis.set_xticks(positions, variants, rotation=22, ha="right")
    axis.set_ylabel("Opportunity @20 lift vs random")
    axis.set_title("Frozen match-pool view ablation", loc="left", color=palette["ink"], fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", color=palette["grid"])
    figure.tight_layout()
    figure.savefig(plot_dir / "view_ablation_lift.png", dpi=160)
    plt.close(figure)

    def scatter_plot(filename: str, x_key: str, xlabel: str, title: str) -> None:
        figure, axis = plt.subplots(figsize=(9.6, 5.8))
        for cohort, color, marker in (("Historical", palette["blue"], "o"), ("Development", palette["gold"], "s")):
            selected = [row for row in feature_rows if row["is_opportunity"] and row["cohort"] == cohort]
            axis.scatter([row[x_key] for row in selected], [row["hits_at_20"] for row in selected], color=color, marker=marker, alpha=0.65, label=cohort, edgecolors="none")
        axis.set_title(title, loc="left", color=palette["ink"], fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Hits at Top20")
        axis.set_yticks(range(7))
        axis.grid(color=palette["grid"])
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=160)
        plt.close(figure)

    scatter_plot("cross_view_agreement_vs_hit.png", "cross_view_agreement_count", "Mean supporting view count", "Cross-view agreement vs opportunity hit")
    scatter_plot("followup_entropy_vs_hit.png", "number_entropy", "Weighted follow-up number entropy", "Follow-up entropy vs opportunity hit")
    scatter_plot("candidate_sharpness_vs_hit.png", "candidate_top5_bottom5_gap", "Top5 − Bottom5 score gap", "Candidate sharpness vs opportunity hit")

    figure, axis = plt.subplots(figsize=(9.6, 6.4))
    styles = {
        ("Historical", "SUCCESS_4PLUS"): (palette["blue"], "o"),
        ("Historical", "FAIL_BELOW4"): (palette["open"], "o"),
        ("Development", "SUCCESS_4PLUS"): (palette["gold"], "s"),
        ("Development", "FAIL_BELOW4"): ("#f5e8a8", "s"),
    }
    for key, (color, marker) in styles.items():
        selected = [row for row in second_coordinates if (row["cohort"], row["binary_label"]) == key]
        axis.scatter([row["pc1"] for row in selected], [row["pc2"] for row in selected], color=color, marker=marker, label=f"{key[0]} {key[1]}", alpha=0.75, edgecolors=palette["ink"] if "SUCCESS" in key[1] else "none", linewidths=0.4)
    axis.set_title("Second-order opportunity motif map", loc="left", color=palette["ink"], fontweight="bold")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(color=palette["grid"])
    figure.tight_layout()
    figure.savefig(plot_dir / "second_order_motif_map.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.6, 5.8))
    for cohort, color, marker in (("Historical", palette["blue"], "o"), ("Development", palette["gold"], "s")):
        selected = [row for row in quality_rules if row["cohort"] == cohort]
        axis.scatter([row["coverage_rounds"] for row in selected], [row["random_lift_at_20"] for row in selected], color=color, marker=marker, label=cohort, s=64)
        for row in selected:
            axis.annotate(row["rule_id"], (row["coverage_rounds"], row["random_lift_at_20"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.axhline(0, color=palette["muted"], linewidth=1)
    axis.axvline(30, color=palette["muted"], linewidth=1, linestyle="--")
    axis.set_title("Stage 2 coverage vs Top20 lift", loc="left", color=palette["ink"], fontweight="bold")
    axis.set_xlabel("Stage 2 opportunity rounds")
    axis.set_ylabel("Mean-hit lift vs random")
    axis.legend(frameon=False)
    axis.grid(color=palette["grid"])
    figure.tight_layout()
    figure.savefig(plot_dir / "coverage_vs_lift.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12.0, 5.8))
    axis.plot([row["round"] for row in feature_rows], [row["confidence"] for row in feature_rows], color=palette["muted"], linewidth=1)
    opportunity = [row for row in feature_rows if row["is_opportunity"]]
    colors = [palette["orange"] if row["hits_at_20"] >= 4 else palette["blue"] for row in opportunity]
    axis.scatter([row["round"] for row in opportunity], [row["confidence"] for row in opportunity], c=colors, s=22, alpha=0.8)
    axis.axhline(FROZEN_CONFIDENCE_THRESHOLD, color=palette["gold"], linewidth=1.5, linestyle="--", label="Frozen threshold")
    axis.axvline(SPLIT_ROUND - 0.5, color=palette["ink"], linewidth=1, linestyle=":", label="Development start")
    axis.set_title("Opportunity formation timeline", loc="left", color=palette["ink"], fontweight="bold")
    axis.set_xlabel("Round")
    axis.set_ylabel("Motif confidence")
    axis.legend(frameon=False)
    axis.grid(color=palette["grid"])
    figure.tight_layout()
    figure.savefig(plot_dir / "opportunity_timeline.png", dpi=160)
    plt.close(figure)

    stages = (30, 25, 20, 15, 10)
    figure, axis = plt.subplots(figsize=(9.6, 5.8))
    for cohort, color, marker in (("Historical", palette["blue"], "o"), ("Development", palette["gold"], "s")):
        selected = [row for row in funnel_rows if row["cohort"] == cohort]
        rates = [mean(int(row["motif_rank"]) <= stage for row in selected) for stage in stages]
        axis.plot(stages, rates, color=color, marker=marker, label=cohort)
    axis.set_xticks(stages)
    axis.invert_xaxis()
    axis.set_ylim(0, 1)
    axis.set_title("Winning-number candidate funnel", loc="left", color=palette["ink"], fontweight="bold")
    axis.set_xlabel("Candidate cutoff")
    axis.set_ylabel("Winning numbers retained")
    axis.legend(frameon=False)
    axis.grid(color=palette["grid"])
    figure.tight_layout()
    figure.savefig(plot_dir / "candidate_funnel.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.6, 5.8))
    for cohort, color in (("Historical", palette["blue"]), ("Development", palette["gold"])):
        values = [row["actual_rank"] for row in missing_rows if row["cohort"] == cohort]
        axis.hist(values, bins=np.arange(20.5, 46.5, 2), alpha=0.55, color=color, label=cohort)
    axis.set_title("Missing-winner rank outside Top20", loc="left", color=palette["ink"], fontweight="bold")
    axis.set_xlabel("Actual motif rank")
    axis.set_ylabel("Winning-number count")
    axis.legend(frameon=False)
    axis.grid(axis="y", color=palette["grid"])
    figure.tight_layout()
    figure.savefig(plot_dir / "missing_winner_rank.png", dpi=160)
    plt.close(figure)


def run_opportunity_analysis(
    *,
    draws: Sequence[Draw],
    motif_run: str | Path,
    start_round: int,
    end_round: int,
    split_round: int,
    experiment_seed: int,
    workers: str | int,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    del workers  # The frozen-match diagnostics are deterministic and intentionally lightweight.
    started_at = perf_counter()
    motif_run = Path(motif_run).resolve()
    walk, predictions, matches, motif_metrics, data_quality = _validate_inputs(
        draws=draws,
        motif_run=motif_run,
        start_round=start_round,
        end_round=end_round,
        split_round=split_round,
    )
    selected_draws = [draw for draw in draws if draw.round_no <= end_round]
    bundle = build_feature_bundle(selected_draws)
    motif_by_round = _matches_by_round(matches, bundle)
    round_to_index = {int(round_no): index for index, round_no in enumerate(bundle.rounds)}
    walk_records = walk.to_dict("records")
    feature_rows = [
        _opportunity_feature_row(
            base_row=row,
            motif_items=motif_by_round[int(row["round"])],
            bundle=bundle,
            target_index=round_to_index[int(row["round"])],
        )
        for row in walk_records
    ]
    _add_formation_features(feature_rows)
    pd.DataFrame(feature_rows).to_parquet(run_dir / "opportunity_features.parquet", index=False)
    label_fields = (
        "round", "cohort", "is_opportunity", "opportunity_label", "binary_label",
        "hits_at_20", "hits_at_30", "confidence",
    )
    opportunity_labels = [{field: row[field] for field in label_fields} for row in feature_rows if row["is_opportunity"]]
    _write_csv(run_dir / "opportunity_labels.csv", opportunity_labels, label_fields)
    logger.info("Opportunity features 완료 | rows=%s | opportunities=%s", len(feature_rows), len(opportunity_labels))

    feature_names = _numeric_feature_names(feature_rows)
    opportunity_comparison = _comparison_rows(
        feature_rows,
        comparison="OPPORTUNITY_VS_NON_OPPORTUNITY",
        feature_names=feature_names,
        seed=experiment_seed,
    )
    success_failure = [
        *_comparison_rows(
            feature_rows,
            comparison="SUCCESS_4PLUS_VS_FAIL_BELOW4",
            feature_names=feature_names,
            seed=experiment_seed + 1_000_000,
        ),
        *_comparison_rows(
            feature_rows,
            comparison="HIT_5_PLUS_VS_FAIL_0_2",
            feature_names=feature_names,
            seed=experiment_seed + 2_000_000,
        ),
    ]
    _write_csv(run_dir / "opportunity_non_opportunity_comparison.csv", opportunity_comparison)
    _write_csv(run_dir / "success_failure_comparison.csv", success_failure)
    replicated_features = _same_direction_mechanisms(success_failure)
    _write_csv(run_dir / "replicated_feature_diagnostics.csv", replicated_features)
    logger.info(
        "구조 비교 완료 | features=%s | replicated=%s",
        len(feature_names),
        sum(row["replicated"] for row in replicated_features),
    )

    view_ablation, single_view, pair_interactions, _variant_rows = _view_diagnostics(
        feature_rows=feature_rows,
        bundle=bundle,
        motif_by_round=motif_by_round,
        seed=experiment_seed,
        logger=logger,
    )
    _write_csv(run_dir / "view_ablation.csv", view_ablation)
    _write_csv(run_dir / "single_view_diagnostics.csv", single_view)
    _write_csv(run_dir / "pair_interactions.csv", pair_interactions)
    ablation_mechanism = _ablation_mechanism(view_ablation)

    family_rows, family_distributions = _motif_family_analysis(feature_rows, motif_by_round)
    _write_csv(run_dir / "motif_family_analysis.csv", family_rows)
    second_order_rows, second_coordinates, second_order_summary = _second_order_analysis(
        feature_rows=feature_rows,
        family_distributions=family_distributions,
        seed=experiment_seed,
    )
    _write_csv(run_dir / "second_order_motifs.csv", second_order_rows)
    _write_csv(run_dir / "second_order_motif_coordinates.csv", second_coordinates)

    quality_rule_rows, selected_rule, stage2_rows = _quality_rules(feature_rows, seed=experiment_seed)
    walk_by_round = {int(row["round"]): row for row in walk_records}
    for row in stage2_rows:
        ranking = list(map(int, _json_value(walk_by_round[int(row["round"])]["ranked_numbers"])))
        row["candidate_top20"] = ranking[:20]
        row["candidate_top30"] = ranking[:30]
    _write_csv(run_dir / "quality_rules.csv", quality_rule_rows)
    _write_csv(run_dir / "stage2_predictions.csv", stage2_rows)
    stage_summary = _stage1_stage2_summary(
        feature_rows=feature_rows,
        stage2_rows=stage2_rows,
        seed=experiment_seed,
    )

    funnel_rows, missing_rows, false_positive_rows = _candidate_diagnostics(
        feature_rows=feature_rows,
        base_walk=walk,
        motif_by_round=motif_by_round,
        bundle=bundle,
    )
    _write_csv(run_dir / "candidate_funnel.csv", funnel_rows)
    _write_csv(run_dir / "missing_winners.csv", missing_rows)
    _write_csv(run_dir / "false_positives.csv", false_positive_rows)

    verdict, decision, success_conditions = _verdict(
        stage_summary=stage_summary,
        selected_rule=selected_rule,
        replicated_features=replicated_features,
        ablation=ablation_mechanism,
    )
    summary: dict[str, Any] = {
        "analysis": "Opportunity Mechanism Analysis",
        "verdict": verdict,
        "decision": {
            "code": decision,
            "description": {
                "A": "Opportunity Detection을 Uriel 핵심 구조로 승격",
                "B": "WEAK SIGNAL 유지 후 추가 사전등록 검증",
                "C": "Opportunity 가설 종료",
            }[decision],
        },
        "frozen_configuration": {
            "query_length": FROZEN_QUERY_LENGTH,
            "historical_lengths": list(FROZEN_CANDIDATE_LENGTHS),
            "top_k": FROZEN_TOP_K,
            "minimum_temporal_separation": FROZEN_SEPARATION,
            "views": list(VIEW_NAMES),
            "primary_candidate_size": PRIMARY_CANDIDATE_SIZE,
            "confidence_threshold": FROZEN_CONFIDENCE_THRESHOLD,
        },
        "source_motif_run": {
            "path": str(motif_run),
            "verdict": motif_metrics["verdict"],
            "source_hashes": data_quality["source_hashes"],
        },
        "data_quality": data_quality,
        "opportunity_labels": {
            cohort: dict(Counter(row["opportunity_label"] for row in opportunity_labels if row["cohort"] == cohort))
            for cohort in ("Historical", "Development")
        },
        "stage1_stage2": stage_summary,
        "selected_quality_rule": selected_rule,
        "quality_rule_selection_policy": "Five predeclared candidates; thresholds/directions from Historical only; Development untouched",
        "replicated_structural_features": [row for row in replicated_features if row["replicated"]],
        "same_direction_structural_features": [row for row in replicated_features if row["same_direction"]],
        "structural_feature_summary": {
            "tested_feature_count": len(feature_names),
            "same_mean_direction_count": sum(bool(row["same_direction"]) for row in replicated_features),
            "fdr_replicated_count": sum(bool(row["replicated"]) for row in replicated_features),
        },
        "view_ablation_mechanism": ablation_mechanism,
        "second_order_motif": second_order_summary,
        "opportunity_formation": {
            cohort: {
                "opportunity_rounds": sum(row["is_opportunity"] for row in feature_rows if row["cohort"] == cohort),
                "isolated_opportunities": sum(row["isolated_opportunity"] for row in feature_rows if row["cohort"] == cohort),
                "maximum_burst_length": max(row["burst_length"] for row in feature_rows if row["cohort"] == cohort),
                "mean_gap": float(np.nanmean([row["opportunity_gap"] for row in feature_rows if row["cohort"] == cohort])),
                "maximum_rolling_10_round_density": max(row["rolling_opportunity_density"] for row in feature_rows if row["cohort"] == cohort),
            }
            for cohort in ("Historical", "Development")
        },
        "success_conditions": success_conditions,
        "candidate_diagnostics": {
            "funnel_winner_rows": len(funnel_rows),
            "missing_winner_rows": len(missing_rows),
            "false_positive_rows": len(false_positive_rows),
            "missing_winner_mean_rank": float(mean(row["actual_rank"] for row in missing_rows)),
        },
        "surrogates": {
            "reused_from_frozen_motif_run": [
                "round_shuffle", "within_round_random_lotto", "block_shuffle", "feature_preserving_surrogate",
            ],
            "added": "10,000 success/failure label permutations for second-order motifs",
        },
        "sealed_ranges": {
            "Locked": "660-851: SEALED",
            "Additional Blind": "468-659: SEALED",
            "opening_decision": "DO NOT OPEN in this phase",
        },
        "diagnostic_limits": {
            "view_diagnostics": "Frozen Top-40 match-pool support-vector reweighting; no alternative match retrieval or tuning",
            "secondary_5_plus": "Exploratory because each cohort has fewer than 10 HIT_5_PLUS opportunities",
            "causality": "Associational mechanism analysis; does not establish causal predictability",
        },
        "monte_carlo_iterations": MONTE_CARLO_ITERATIONS,
        "execution": execution_metadata(
            draws=selected_draws,
            started_at=started_at,
            start_round=start_round,
            end_round=end_round,
        ),
    }
    _plot_artifacts(
        run_dir=run_dir,
        feature_rows=feature_rows,
        replicated_features=replicated_features,
        view_ablation=view_ablation,
        second_coordinates=second_coordinates,
        quality_rules=quality_rule_rows,
        funnel_rows=funnel_rows,
        missing_rows=missing_rows,
    )
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Opportunity analysis 완료 | verdict=%s | decision=%s | selected_rule=%s | Historical Stage2 lift=%.4f | Development Stage2 lift=%.4f",
        verdict,
        decision,
        selected_rule["rule_id"],
        stage_summary["Historical"]["Stage2"]["top20"]["mean_hit_lift"],
        stage_summary["Development"]["Stage2"]["top20"]["mean_hit_lift"],
    )
    return summary
