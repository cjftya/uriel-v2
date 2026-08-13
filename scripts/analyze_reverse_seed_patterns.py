#!/usr/bin/env python3
"""Build a deterministic Markdown audit of Stage A representative reverse seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats


RNG_SEED = 20260813
PERMUTATIONS = 50_000
MONTE_CARLO = 50_000
SEED_SPACE = 1_000_000


def parse_numbers(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("-"))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def hamming(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def fmt_p(value: float) -> str:
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def fmt_num(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def bh_adjust(items: list[dict]) -> None:
    """Add Benjamini-Hochberg q-values to dictionaries containing `p`."""
    if not items:
        return
    order = np.argsort([item["p"] for item in items])
    m = len(items)
    adjusted = np.ones(m, dtype=float)
    running = 1.0
    for reverse_rank, idx in enumerate(order[::-1], start=1):
        rank = m - reverse_rank + 1
        value = min(running, items[int(idx)]["p"] * m / rank)
        adjusted[int(idx)] = value
        running = value
    for item, q in zip(items, adjusted):
        item["q"] = float(q)


def permutation_p(null: np.ndarray, observed: float, alternative: str) -> float:
    if alternative == "greater":
        count = np.count_nonzero(null >= observed)
    elif alternative == "less":
        count = np.count_nonzero(null <= observed)
    elif alternative == "two-sided":
        center = float(np.mean(null))
        count = np.count_nonzero(np.abs(null - center) >= abs(observed - center))
    else:
        raise ValueError(alternative)
    return float((count + 1) / (len(null) + 1))


def overlap(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return len(set(a) & set(b))


def bar(count: int, maximum: int, width: int = 24) -> str:
    blocks = max(1, round(width * count / maximum)) if count else 0
    return "█" * blocks


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["round"] = int(row["round"])
        row["best_seed"] = int(row["best_seed"])
        row["best_hits"] = int(row["best_hits"])
        row["best_positional_mae"] = float(row["best_positional_mae"])
        row["winner_tuple"] = parse_numbers(row["winner"])
        row["best_numbers_tuple"] = parse_numbers(row["best_numbers"])
    rows.sort(key=lambda row: row["round"])
    if len(rows) != 192 or rows[0]["round"] != 1044 or rows[-1]["round"] != 1235:
        raise ValueError("Expected exactly rounds 1044..1235 (192 rows)")
    if [row["round"] for row in rows] != list(range(1044, 1236)):
        raise ValueError("Round sequence has gaps or duplicates")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--top-k", type=Path, required=True)
    parser.add_argument("--hit-seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.input)
    seeds = np.array([row["best_seed"] for row in rows], dtype=np.int64)
    rounds = np.array([row["round"] for row in rows], dtype=np.int64)
    hits = np.array([row["best_hits"] for row in rows], dtype=np.int64)
    winners = [row["winner_tuple"] for row in rows]
    generated = [row["best_numbers_tuple"] for row in rows]
    n = len(seeds)
    rng = np.random.default_rng(RNG_SEED)

    with args.top_k.open(encoding="utf-8-sig", newline="") as handle:
        top_k_rows = list(csv.DictReader(handle))
    with args.hit_seeds.open(encoding="utf-8-sig", newline="") as handle:
        hit_seed_rows = list(csv.DictReader(handle))
    if len(top_k_rows) != 19_200 or len(hit_seed_rows) != 266_849:
        raise ValueError("Unexpected Top-100 or 4+ candidate row count")
    top_k_seeds = np.array([int(row["seed"]) for row in top_k_rows], dtype=np.int64)
    top_k_ranks = np.array([int(row["rank"]) for row in top_k_rows], dtype=np.int64)
    hit_candidate_seeds = np.array([int(row["seed"]) for row in hit_seed_rows], dtype=np.int64)
    top_k_by_round: dict[int, list[int]] = {}
    for row in top_k_rows:
        top_k_by_round.setdefault(int(row["round"]), []).append(int(row["seed"]))
    if any(values[0] != int(seeds[round_number - 1044]) for round_number, values in top_k_by_round.items()):
        raise ValueError("Top-100 rank 1 does not match reverse-rounds representative seed")
    top_k_round_medians = np.array([np.median(top_k_by_round[round_number]) for round_number in rounds])
    top_k_other_seeds = top_k_seeds[top_k_ranks > 1]
    representative_below_top_k_median = int(np.sum(seeds < top_k_round_medians))
    representative_equals_top_k_min = int(
        np.sum([seeds[index] == min(top_k_by_round[int(round_number)]) for index, round_number in enumerate(rounds)])
    )
    rank1_vs_median = stats.wilcoxon(seeds, top_k_round_medians)
    hit_candidate_buckets, _ = np.histogram(
        hit_candidate_seeds, bins=np.arange(0, SEED_SPACE + 100_000, 100_000)
    )
    hit_candidate_bucket_ratio = float(hit_candidate_buckets.max() / hit_candidate_buckets.min())

    # Marginal distribution diagnostics.
    desc = {
        "min": int(seeds.min()),
        "q1": float(np.quantile(seeds, 0.25)),
        "median": float(np.median(seeds)),
        "mean": float(np.mean(seeds)),
        "q3": float(np.quantile(seeds, 0.75)),
        "max": int(seeds.max()),
        "sd": float(np.std(seeds, ddof=1)),
        "unique": int(len(np.unique(seeds))),
    }
    bucket_counts, _ = np.histogram(seeds, bins=np.arange(0, SEED_SPACE + 100_000, 100_000))
    bucket_chi = stats.chisquare(bucket_counts, np.full(10, n / 10))
    ks = stats.kstest((seeds + 0.5) / SEED_SPACE, "uniform")

    # Temporal permutation tests: preserve the exact marginal seed set.
    observed = {
        "lag1_seed_autocorrelation": pearson(seeds[:-1], seeds[1:]),
        "mean_adjacent_abs_delta": float(np.mean(np.abs(np.diff(seeds)))),
        "adjacent_delta_le_10k": int(np.sum(np.abs(np.diff(seeds)) <= 10_000)),
        "adjacent_delta_le_50k": int(np.sum(np.abs(np.diff(seeds)) <= 50_000)),
        "mean_adjacent_hamming": float(np.mean([hamming(a, b) for a, b in zip(seeds[:-1], seeds[1:])])),
        "adjacent_generated_overlap": float(np.mean([overlap(a, b) for a, b in zip(generated[:-1], generated[1:])])),
        "seed_round_spearman": spearman(rounds, seeds),
    }
    null = {key: np.empty(PERMUTATIONS, dtype=float) for key in observed}
    for index in range(PERMUTATIONS):
        order = rng.permutation(n)
        shuffled_seeds = seeds[order]
        shuffled_generated = [generated[int(i)] for i in order]
        delta = np.abs(np.diff(shuffled_seeds))
        null["lag1_seed_autocorrelation"][index] = pearson(shuffled_seeds[:-1], shuffled_seeds[1:])
        null["mean_adjacent_abs_delta"][index] = np.mean(delta)
        null["adjacent_delta_le_10k"][index] = np.sum(delta <= 10_000)
        null["adjacent_delta_le_50k"][index] = np.sum(delta <= 50_000)
        null["mean_adjacent_hamming"][index] = np.mean(
            [hamming(a, b) for a, b in zip(shuffled_seeds[:-1], shuffled_seeds[1:])]
        )
        null["adjacent_generated_overlap"][index] = np.mean(
            [overlap(a, b) for a, b in zip(shuffled_generated[:-1], shuffled_generated[1:])]
        )
        null["seed_round_spearman"][index] = spearman(rounds, shuffled_seeds)

    temporal_tests = [
        {"name": "lag-1 seed 자기상관", "key": "lag1_seed_autocorrelation", "alternative": "two-sided"},
        {"name": "연속 회차 평균 |Δseed| (작을수록 군집)", "key": "mean_adjacent_abs_delta", "alternative": "less"},
        {"name": "연속 회차 |Δseed| ≤ 10,000 횟수", "key": "adjacent_delta_le_10k", "alternative": "greater"},
        {"name": "연속 회차 |Δseed| ≤ 50,000 횟수", "key": "adjacent_delta_le_50k", "alternative": "greater"},
        {"name": "연속 회차 20-bit Hamming 거리", "key": "mean_adjacent_hamming", "alternative": "two-sided"},
        {"name": "연속 회차 생성번호 평균 교집합", "key": "adjacent_generated_overlap", "alternative": "greater"},
        {"name": "회차와 seed의 Spearman 상관", "key": "seed_round_spearman", "alternative": "two-sided"},
    ]
    for test in temporal_tests:
        key = test["key"]
        test["observed"] = observed[key]
        test["null_mean"] = float(np.mean(null[key]))
        test["p"] = permutation_p(null[key], observed[key], test["alternative"])
    bh_adjust(temporal_tests)

    lag_rows = []
    for lag in range(1, 21):
        lag_rows.append({"lag": lag, "r": pearson(seeds[:-lag], seeds[lag:])})

    signed_delta = np.diff(seeds)
    abs_delta = np.abs(signed_delta)
    adjacent_pairs = sorted(
        [
            {
                "round_a": rows[i]["round"],
                "round_b": rows[i + 1]["round"],
                "seed_a": int(seeds[i]),
                "seed_b": int(seeds[i + 1]),
                "delta": int(signed_delta[i]),
                "abs_delta": int(abs_delta[i]),
                "hamming": hamming(seeds[i], seeds[i + 1]),
                "overlap": overlap(generated[i], generated[i + 1]),
            }
            for i in range(n - 1)
        ],
        key=lambda item: item["abs_delta"],
    )

    # Global numeric-neighbor gaps, compared with IID uniform seeds.
    sorted_indices = np.argsort(seeds)
    sorted_seeds = seeds[sorted_indices]
    gaps = np.diff(sorted_seeds)
    gap_cv = float(np.std(gaps, ddof=1) / np.mean(gaps))
    mc_min = np.empty(MONTE_CARLO)
    mc_max = np.empty(MONTE_CARLO)
    mc_cv = np.empty(MONTE_CARLO)
    mc_popcount = np.empty(MONTE_CARLO)
    for start in range(0, MONTE_CARLO, 1_000):
        size = min(1_000, MONTE_CARLO - start)
        samples = rng.integers(0, SEED_SPACE, size=(size, n), dtype=np.int64)
        samples.sort(axis=1)
        sample_gaps = np.diff(samples, axis=1)
        mc_min[start : start + size] = sample_gaps.min(axis=1)
        mc_max[start : start + size] = sample_gaps.max(axis=1)
        mc_cv[start : start + size] = sample_gaps.std(axis=1, ddof=1) / sample_gaps.mean(axis=1)
        mc_popcount[start : start + size] = np.mean(
            np.unpackbits(samples.astype(np.uint32).view(np.uint8), axis=1).reshape(size, n, 32).sum(axis=2),
            axis=1,
        )
    gap_tests = [
        {
            "name": "정렬 seed 최소 간격",
            "observed": float(gaps.min()),
            "null_mean": float(mc_min.mean()),
            "p": permutation_p(mc_min, float(gaps.min()), "less"),
        },
        {
            "name": "정렬 seed 최대 간격",
            "observed": float(gaps.max()),
            "null_mean": float(mc_max.mean()),
            "p": permutation_p(mc_max, float(gaps.max()), "greater"),
        },
        {
            "name": "정렬 간격 변동계수",
            "observed": gap_cv,
            "null_mean": float(mc_cv.mean()),
            "p": permutation_p(mc_cv, gap_cv, "greater"),
        },
    ]
    bh_adjust(gap_tests)
    nearest_pairs = []
    for pos in np.argsort(gaps)[:12]:
        i = int(sorted_indices[pos])
        j = int(sorted_indices[pos + 1])
        nearest_pairs.append(
            {
                "round_a": rows[i]["round"],
                "round_b": rows[j]["round"],
                "seed_a": int(seeds[i]),
                "seed_b": int(seeds[j]),
                "gap": int(gaps[pos]),
                "hamming": hamming(seeds[i], seeds[j]),
                "generated_overlap": overlap(generated[i], generated[j]),
                "winner_overlap": overlap(winners[i], winners[j]),
            }
        )

    # Residue, decimal digit, and bit diagnostics.
    modulo_tests = []
    modulo_counts = {}
    for modulus in [2, 3, 4, 5, 7, 8, 10, 16, 20, 25, 32]:
        counts = np.bincount(seeds % modulus, minlength=modulus)
        base = SEED_SPACE // modulus
        remainder = SEED_SPACE % modulus
        probabilities = np.array([base + (r < remainder) for r in range(modulus)], dtype=float) / SEED_SPACE
        result = stats.chisquare(counts, n * probabilities)
        modulo_tests.append({"name": f"mod {modulus}", "modulus": modulus, "p": float(result.pvalue), "chi2": float(result.statistic)})
        modulo_counts[modulus] = counts.tolist()
    bh_adjust(modulo_tests)

    digit_matrix = np.array([[int(ch) for ch in f"{seed:06d}"] for seed in seeds], dtype=int)
    digit_tests = []
    for position in range(6):
        counts = np.bincount(digit_matrix[:, position], minlength=10)
        result = stats.chisquare(counts, np.full(10, n / 10))
        digit_tests.append(
            {
                "name": f"10^{5-position} 자리",
                "position": position,
                "counts": counts.tolist(),
                "p": float(result.pvalue),
                "chi2": float(result.statistic),
            }
        )
    bh_adjust(digit_tests)

    universe = np.arange(SEED_SPACE, dtype=np.uint32)
    bit_tests = []
    for bit in range(20):
        count = int(np.sum((seeds >> bit) & 1))
        probability = float(np.mean((universe >> bit) & 1))
        result = stats.binomtest(count, n, probability)
        bit_tests.append(
            {"name": f"bit {bit}", "bit": bit, "ones": count, "expected": n * probability, "p": float(result.pvalue)}
        )
    bh_adjust(bit_tests)
    popcounts = np.array([int(seed).bit_count() for seed in seeds])
    popcount_mean = float(np.mean(popcounts))
    popcount_p = permutation_p(mc_popcount, popcount_mean, "two-sided")

    # Winning-number features and seed relationships; permutation p-values.
    features = {
        "당첨번호 합": np.array([sum(values) for values in winners], dtype=float),
        "당첨번호 범위(max-min)": np.array([max(values) - min(values) for values in winners], dtype=float),
        "홀수 개수": np.array([sum(value % 2 for value in values) for values in winners], dtype=float),
        "연속수 쌍 개수": np.array([sum(b - a == 1 for a, b in zip(values[:-1], values[1:])) for values in winners], dtype=float),
        "저번호(≤22) 개수": np.array([sum(value <= 22 for value in values) for values in winners], dtype=float),
        "당첨번호 표준편차": np.array([np.std(values, ddof=0) for values in winners], dtype=float),
    }
    feature_tests = []
    for name, values in features.items():
        observed_rho = spearman(seeds, values)
        null_rho = np.empty(PERMUTATIONS)
        for index in range(PERMUTATIONS):
            null_rho[index] = spearman(rng.permutation(seeds), values)
        feature_tests.append(
            {
                "name": name,
                "rho": observed_rho,
                "p": permutation_p(null_rho, observed_rho, "two-sided"),
            }
        )
    bh_adjust(feature_tests)
    range_values = features["당첨번호 범위(max-min)"]
    range_rho_first = spearman(seeds[: n // 2], range_values[: n // 2])
    range_rho_last = spearman(seeds[n // 2 :], range_values[n // 2 :])
    range_rho_top_k_median = spearman(top_k_round_medians, range_values)

    # Pairwise relationship between seed similarity and generated-number overlap.
    numeric_distances = []
    bit_distances = []
    output_overlaps = []
    target_overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            numeric_distances.append(abs(int(seeds[i]) - int(seeds[j])))
            bit_distances.append(hamming(seeds[i], seeds[j]))
            output_overlaps.append(overlap(generated[i], generated[j]))
            target_overlaps.append(overlap(winners[i], winners[j]))
    pair_numeric_rho = spearman(np.array(numeric_distances), np.array(output_overlaps))
    pair_bit_rho = spearman(np.array(bit_distances), np.array(output_overlaps))
    pair_target_rho = spearman(np.array(target_overlaps), np.array(output_overlaps))
    generated_duplicates = n - len(set(generated))

    exact_mask = hits == 6
    mw = stats.mannwhitneyu(seeds[exact_mask], seeds[~exact_mask], alternative="two-sided")

    # Runs test around the median.
    median = np.median(seeds)
    binary = seeds >= median
    runs = 1 + int(np.sum(binary[1:] != binary[:-1]))
    n1 = int(np.sum(binary))
    n0 = n - n1
    expected_runs = 1 + 2 * n1 * n0 / n
    variance_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))
    runs_z = (runs - expected_runs) / math.sqrt(variance_runs)
    runs_p = float(2 * stats.norm.sf(abs(runs_z)))

    source_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    top_k_sha = hashlib.sha256(args.top_k.read_bytes()).hexdigest()
    hit_seeds_sha = hashlib.sha256(args.hit_seeds.read_bytes()).hexdigest()
    significant_temporal = [test for test in temporal_tests if test["q"] < 0.05]
    significant_modulo = [test for test in modulo_tests if test["q"] < 0.05]
    significant_digits = [test for test in digit_tests if test["q"] < 0.05]
    significant_bits = [test for test in bit_tests if test["q"] < 0.05]
    significant_features = [test for test in feature_tests if test["q"] < 0.05]

    lines: list[str] = []
    add = lines.append
    add("# Uriel v2 회차별 정답 역산 seed 목록 및 패턴 분석")
    add("")
    add("- 작성일: 2026년 8월 13일 (KST)")
    add("- 대상: 1044~1235회, 192개 회차")
    add("- seed 공간: `[0, 1,000,000)`")
    add("- PRNG: SplitMix64")
    add("- 대표 seed: 회차별 `hits ↓ → positional_mae ↑ → set_distance ↑ → seed ↑` 순위의 1위")
    add(f"- 입력 파일: `{args.input.name}` (SHA-256 `{source_sha}`)")
    add(f"- 대조 파일: `{args.top_k.name}` (SHA-256 `{top_k_sha}`)")
    add(f"- 대조 파일: `{args.hit_seeds.name}` (SHA-256 `{hit_seeds_sha}`)")
    add(f"- 통계 재현 seed: `{RNG_SEED}`; 순열검정 {PERMUTATIONS:,}회; 균등 기준 Monte Carlo {MONTE_CARLO:,}회")
    add("")
    add("> **중요:** 이 seed들은 당첨번호를 먼저 알고 역산한 정답 종속 결과다. 패턴 분석은 선택된 seed 집합의 기술·진단이며, 미래 회차 seed 선택 규칙이나 예측력을 증명하지 않는다.")
    add("")
    add("## 결론")
    add("")
    add(f"192개 대표 seed는 모두 고유했고, 최소 `{desc['min']:,}`, 최대 `{desc['max']:,}`, 중앙값 `{desc['median']:,.0f}`였다. 10만 단위 bucket의 균등성 검정은 `p={fmt_p(float(bucket_chi.pvalue))}`, 연속 회차 순서의 runs test는 `p={fmt_p(runs_p)}`였다.")
    add("")
    if significant_temporal or significant_modulo or significant_digits or significant_bits or significant_features:
        add("다중검정 보정 후 남은 것은 세 가지 표현으로 요약된다. (1) 대표 seed가 낮은 쪽으로 치우친 현상이 10만 bucket·십진수 최상위 자리·bit 19에서 중복 검출됐고, (2) 당첨번호 범위가 넓을수록 대표 seed가 약간 낮아지는 약한 관계가 검출됐다. (3) 회차 순서 자체의 자기상관·근접 군집·Hamming·생성번호 유사성은 모두 유의하지 않았다.")
        add("")
        add(f"낮은 seed 편향은 후보 전체보다 대표값에서 강하다. 대표값 중앙값은 {desc['median']:,.0f}, Top-100의 rank 2~100 중앙값은 {np.median(top_k_other_seeds):,.0f}이고, 모든 4+ 후보의 10만 bucket 최대/최소 비는 {hit_candidate_bucket_ratio:.4f}에 불과하다. 이는 SplitMix64 출력 공간의 물리적 집중보다 정렬·tie-break 선택 효과와 일치한다.")
    else:
        add("회차 순서, modulo, 6자리 십진수, 20-bit, 당첨번호 요약 특성에 대한 사전 정의 검정에서 Benjamini–Hochberg 보정 `q<0.05`로 남는 패턴은 없었다. 관측된 근접쌍과 증감은 192개의 수를 놓으면 생기는 우연 범위와 구분되지 않았다.")
    add("")
    add("따라서 현재 데이터에서 지지되는 판정은 다음과 같다.")
    add("")
    add("> **REVERSE-SEED CATALOG COMPLETE / FORWARD-SEED PATTERN NOT ESTABLISHED**")
    add("")
    add("## 핵심 수치")
    add("")
    add("| 항목 | 관측값 | 기준/검정 | 판정 |")
    add("|---|---:|---:|---|")
    add(f"| 고유 seed | {desc['unique']} / {n} | 중복 pair 기대값 약 {n*(n-1)/(2*SEED_SPACE):.4f} | 전부 고유, 자연스러움 |")
    add(f"| 평균 seed | {desc['mean']:,.1f} | 균등 기대 약 499,999.5 | 차이 {desc['mean']-499999.5:+,.1f} |")
    add(f"| 10만 bucket 균등성 | χ²={bucket_chi.statistic:.3f} | p={fmt_p(float(bucket_chi.pvalue))} | {'유의' if bucket_chi.pvalue < .05 else '유의하지 않음'} |")
    add(f"| 연속 회차 lag-1 상관 | r={observed['lag1_seed_autocorrelation']:.3f} | permutation q={fmt_p(temporal_tests[0]['q'])} | {'유의' if temporal_tests[0]['q'] < .05 else '유의하지 않음'} |")
    add(f"| 회차 번호와 seed | ρ={observed['seed_round_spearman']:.3f} | permutation q={fmt_p(temporal_tests[-1]['q'])} | {'유의' if temporal_tests[-1]['q'] < .05 else '유의하지 않음'} |")
    add(f"| 연속 평균 \\|Δseed\\| | {observed['mean_adjacent_abs_delta']:,.1f} | 섞은 순서 평균 {np.mean(null['mean_adjacent_abs_delta']):,.1f} | q={fmt_p(temporal_tests[1]['q'])} |")
    add(f"| 생성번호 중복 조합 | {generated_duplicates} | 192개 대표 출력 | {'있음' if generated_duplicates else '없음'} |")
    add(f"| exact 6 대표 seed | {int(exact_mask.sum())} | 5-hit {int((~exact_mask).sum())} | Stage A 재구성 결과 |")
    add(f"| 대표값 vs Top-100 중앙값 | {representative_below_top_k_median} / {n}회에서 더 낮음 | paired Wilcoxon p={fmt_p(float(rank1_vs_median.pvalue))} | 선택 규칙 영향 |")
    add("")
    add("## 1. seed 분포")
    add("")
    add("| 통계 | 값 |")
    add("|---|---:|")
    for label, key in [("최소", "min"), ("1사분위", "q1"), ("중앙값", "median"), ("평균", "mean"), ("3사분위", "q3"), ("최대", "max"), ("표준편차", "sd")]:
        add(f"| {label} | {desc[key]:,.1f} |")
    add("")
    add("10만 단위 분포:")
    add("")
    add("```text")
    maximum_bucket = int(bucket_counts.max())
    for index, count in enumerate(bucket_counts):
        add(f"{index*100_000:06d}–{(index+1)*100_000-1:06d}  {int(count):3d}  {bar(int(count), maximum_bucket)}")
    add("```")
    add("")
    add(f"Pearson χ² 검정은 χ²={bucket_chi.statistic:.3f}, p={fmt_p(float(bucket_chi.pvalue))}; KS 검정은 D={ks.statistic:.3f}, p={fmt_p(float(ks.pvalue))}다. 다만 대표값 tie-break가 작은 seed를 선택할 수 있으므로, 이 균등 기준 비교는 물리 난수 검정이 아닌 진단값이다.")
    add("")
    add("### 대표값 선택 효과 대조")
    add("")
    add("| 집합 | 개수 | 평균 seed | 중앙값 | 10만 bucket 최대/최소 |")
    add("|---|---:|---:|---:|---:|")
    add(f"| 회차별 대표 rank 1 | {n:,} | {np.mean(seeds):,.1f} | {np.median(seeds):,.0f} | {bucket_counts.max()/bucket_counts.min():.4f} |")
    add(f"| Top-100 중 rank 2~100 | {len(top_k_other_seeds):,} | {np.mean(top_k_other_seeds):,.1f} | {np.median(top_k_other_seeds):,.0f} | — |")
    add(f"| 모든 4+/5+/6-hit 후보 | {len(hit_candidate_seeds):,} | {np.mean(hit_candidate_seeds):,.1f} | {np.median(hit_candidate_seeds):,.0f} | {hit_candidate_bucket_ratio:.4f} |")
    add("")
    add(f"대표 seed는 같은 회차 Top-100 중앙값보다 {representative_below_top_k_median}/{n}회 낮았고 paired Wilcoxon p={fmt_p(float(rank1_vs_median.pvalue))}였다. 다만 대표 seed가 Top-100 전체의 단순 최솟값인 회차는 {representative_equals_top_k_min}회뿐이다. 즉 낮은 쏠림은 `seed ↑` tie-break 하나만이 아니라, 먼저 적용되는 hit·MAE 그룹 안의 결정적 정렬과 결합된 결과다.")
    add("")
    add("## 2. 회차 순서 패턴")
    add("")
    add("| 검정 | 관측값 | 섞은 순서 평균 | raw p | BH q |")
    add("|---|---:|---:|---:|---:|")
    for test in temporal_tests:
        display_name = test["name"].replace("|", "\\|")
        add(f"| {display_name} | {test['observed']:.4f} | {test['null_mean']:.4f} | {fmt_p(test['p'])} | {fmt_p(test['q'])} |")
    add("")
    add(f"중앙값 위/아래 runs는 {runs}회, 기대 {expected_runs:.2f}, z={runs_z:.3f}, p={fmt_p(runs_p)}였다. 상승 {int(np.sum(signed_delta>0))}회, 하락 {int(np.sum(signed_delta<0))}회, 동일 0회이며 signed Δ의 중앙값은 {np.median(signed_delta):,.0f}, |Δ| 중앙값은 {np.median(abs_delta):,.0f}다.")
    add("")
    add("lag 1~20 자기상관:")
    add("")
    add("| lag | r | lag | r | lag | r | lag | r |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for start in range(0, 5):
        cells = []
        for offset in [0, 5, 10, 15]:
            item = lag_rows[start + offset]
            cells.extend([str(item["lag"]), f"{item['r']:.3f}"])
        add("| " + " | ".join(cells) + " |")
    add("")
    add("연속 회차 중 가장 가까운 10쌍:")
    add("")
    add("| 회차 | seed 변화 | Δ | \\|Δ\\| | bit 거리 | 생성번호 교집합 |")
    add("|---|---|---:|---:|---:|---:|")
    for item in adjacent_pairs[:10]:
        add(f"| {item['round_a']}→{item['round_b']} | {item['seed_a']:,}→{item['seed_b']:,} | {item['delta']:+,} | {item['abs_delta']:,} | {item['hamming']} | {item['overlap']} |")
    add("")
    add("## 3. 전체 seed 근접쌍과 군집")
    add("")
    add(f"seed를 수치순으로 정렬했을 때 최소 간격은 {int(gaps.min()):,}, 중앙 간격 {np.median(gaps):,.0f}, 평균 간격 {np.mean(gaps):,.1f}, 최대 간격 {int(gaps.max()):,}, 간격 변동계수 {gap_cv:.3f}다.")
    add("")
    add("| 검정 | 관측값 | 균등 Monte Carlo 평균 | raw p | BH q |")
    add("|---|---:|---:|---:|---:|")
    for test in gap_tests:
        add(f"| {test['name']} | {test['observed']:.4f} | {test['null_mean']:.4f} | {fmt_p(test['p'])} | {fmt_p(test['q'])} |")
    add("")
    add("가장 가까운 12쌍:")
    add("")
    add("| 회차 A | seed A | 회차 B | seed B | 간격 | bit 거리 | 생성번호 교집합 | 당첨번호 교집합 |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in nearest_pairs:
        add(f"| {item['round_a']} | {item['seed_a']:,} | {item['round_b']} | {item['seed_b']:,} | {item['gap']:,} | {item['hamming']} | {item['generated_overlap']} | {item['winner_overlap']} |")
    add("")
    add("가까운 seed가 비슷한 로또 조합을 만든다는 규칙은 보이지 않는다. SplitMix64의 mixing 때문에 작은 수치 차이도 내부 상태에서는 비선형적으로 확산된다.")
    add("")
    add("## 4. modulo·십진수·bit 특징")
    add("")
    add("### modulo")
    add("")
    add("| 검정 | χ² | raw p | BH q |")
    add("|---|---:|---:|---:|")
    for test in modulo_tests:
        add(f"| {test['name']} | {test['chi2']:.3f} | {fmt_p(test['p'])} | {fmt_p(test['q'])} |")
    add("")
    last_digits = modulo_counts[10]
    add("마지막 자리(0~9) 빈도: " + ", ".join(f"{digit}:{count}" for digit, count in enumerate(last_digits)) + ".")
    add("")
    add("### 0-padding 6자리 십진수")
    add("")
    add("| 자리 | χ² | raw p | BH q | 0~9 빈도 |")
    add("|---|---:|---:|---:|---|")
    for test in digit_tests:
        counts = "/".join(str(value) for value in test["counts"])
        add(f"| {test['name']} | {test['chi2']:.3f} | {fmt_p(test['p'])} | {fmt_p(test['q'])} | {counts} |")
    add("")
    add("### 20-bit 표현")
    add("")
    add(f"seed popcount 평균은 {popcount_mean:.3f}, 균등 기준 평균은 {np.mean(mc_popcount):.3f}, Monte Carlo p={fmt_p(popcount_p)}다. 개별 bit 중 BH q<0.05인 위치는 " + (", ".join(str(test["bit"]) for test in significant_bits) if significant_bits else "없다") + ". bit 19는 524,288 이상 여부이므로, 별개의 새로운 패턴이 아니라 앞서 본 낮은 seed 쏠림을 이진수로 다시 표현한 것이다.")
    add("")
    add("| bit | 1 관측 | 균등 기대 | raw p | BH q |")
    add("|---:|---:|---:|---:|---:|")
    for test in bit_tests:
        add(f"| {test['bit']} | {test['ones']} | {test['expected']:.2f} | {fmt_p(test['p'])} | {fmt_p(test['q'])} |")
    add("")
    add("## 5. 당첨번호·생성번호와의 관계")
    add("")
    add("| 당첨번호 특성 | seed와 Spearman ρ | permutation p | BH q |")
    add("|---|---:|---:|---:|")
    for test in feature_tests:
        add(f"| {test['name']} | {test['rho']:.3f} | {fmt_p(test['p'])} | {fmt_p(test['q'])} |")
    add("")
    add(f"당첨번호 범위 관계(ρ={feature_tests[1]['rho']:.3f})는 앞 96회에서 ρ={range_rho_first:.3f}, 뒤 96회에서 ρ={range_rho_last:.3f}로 강도가 일정하지 않았다. 대표 seed 대신 회차별 Top-100 seed 중앙값을 쓰면 ρ={range_rho_top_k_median:.3f}다. 따라서 관측 관계는 약한 정답 종속 선택 효과로 기록하되, 일반화된 seed 법칙으로 채택하지 않는다.")
    add("")
    add(f"전체 {len(output_overlaps):,}개 회차쌍에서 `|seed 차이|`와 생성번호 교집합의 Spearman ρ는 {pair_numeric_rho:.3f}, 20-bit Hamming 거리와 생성번호 교집합의 ρ는 {pair_bit_rho:.3f}였다. 반면 당첨번호 교집합과 생성번호 교집합의 ρ는 {pair_target_rho:.3f}다. 대표 출력이 정답과 5개 또는 6개를 공유하도록 선택됐기 때문에 마지막 관계는 구조적으로 생긴다.")
    add("")
    add(f"exact 6-hit 18개 seed의 중앙값은 {np.median(seeds[exact_mask]):,.0f}, 5-hit 174개의 중앙값은 {np.median(seeds[~exact_mask]):,.0f}였다. 두 그룹 seed 수준의 Mann–Whitney 검정은 U={mw.statistic:.1f}, p={fmt_p(float(mw.pvalue))}로, seed 크기 차이를 뒷받침하지 않는다.")
    add("")
    add("## 6. 해석 가능한 특징과 해석하면 안 되는 것")
    add("")
    add("확인된 특징:")
    add("")
    add("- 각 seed는 회차 정답을 5개 또는 6개 재구성하는 대표값이며 192개가 모두 고유하다.")
    add("- 수치 공간 전체에 넓게 퍼져 있고, 연속 회차의 증감·근접·bit 유사성이 안정적인 시계열 규칙으로 확인되지 않았다.")
    add("- 대표 seed에는 낮은 값 쏠림이 있지만, 모든 4+ 후보는 10만 bucket에서 거의 균일하다. 대표값 순위의 결정적 tie-break가 만든 선택 특징이다.")
    add("- 수치가 매우 가까운 seed끼리도 생성번호가 특별히 더 닮지 않는다.")
    add("- exact 6 여부가 seed 크기나 단순 bit 특징으로 분리되지 않는다.")
    add("")
    add("해석하면 안 되는 것:")
    add("")
    add("- 이 목록에서 다음 회차 seed를 외삽하는 것. 정답이 있어야 각 대표 seed를 고를 수 있었다.")
    add("- 낮은 seed나 특정 끝자리가 많아 보이는 부분을 물리적 로또 추첨의 원인으로 보는 것.")
    add("- 여러 lag·modulo 중 가장 눈에 띄는 하나만 사후 선택하는 것. 그래서 본 보고서는 BH 다중검정 보정을 사용했다.")
    add("- 1236회 정답에 맞는 seed가 과거 대표 seed 패턴에서 자동으로 선택된다고 가정하는 것. 별도의 정답 비사용 forward 선택 규칙이 필요하다.")
    add("")
    add("## 7. 회차별 대표 seed 전체 목록")
    add("")
    add("| 회차 | 당첨번호 | 대표 seed | 생성번호 | hit | MAE | 판정 |")
    add("|---:|---|---:|---|---:|---:|---|")
    for row in rows:
        label = "exact" if row["best_hits"] == 6 else "5-hit"
        add(f"| {row['round']} | {row['winner']} | {row['best_seed']:,} | {row['best_numbers']} | {row['best_hits']} | {row['best_positional_mae']:.6f} | {label} |")
    add("")
    add("## 재현 방법")
    add("")
    add("저장소 루트에서 다음 명령을 실행한다.")
    add("")
    add("```bash")
    add("python scripts/analyze_reverse_seed_patterns.py \\")
    add("  --input outputs/reverse-dataset/20260813-132436-reverse-batch/reverse-rounds.csv \\")
    add("  --top-k outputs/reverse-dataset/20260813-132436-reverse-batch/reverse-top-k.csv \\")
    add("  --hit-seeds outputs/reverse-dataset/20260813-132436-reverse-batch/reverse-hit-seeds.csv \\")
    add("  --output reports/uriel-v2-reverse-seed-pattern-analysis-2026-08-13.md")
    add("```")
    add("")
    add("이어서 흐름·불규칙성·비선형 확장 절을 추가한다.")
    add("")
    add("```bash")
    add("python scripts/analyze_reverse_seed_dynamics.py \\")
    add("  --input outputs/reverse-dataset/20260813-132436-reverse-batch/reverse-rounds.csv \\")
    add("  --report reports/uriel-v2-reverse-seed-pattern-analysis-2026-08-13.md")
    add("```")
    add("")
    add("핵심 시계열 검정은 seed 값 자체를 새로 생성하지 않고 동일한 192개 대표 seed의 회차 순서만 섞는다. 따라서 대표 seed 선택이 만든 주변 분포를 보존한 채 회차 순서 의존성만 검사한다. seed 공간 군집·popcount 검정은 `[0, 1,000,000)` 균등 표본을 별도로 사용한다.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"significant temporal={len(significant_temporal)}, modulo={len(significant_modulo)}, digits={len(significant_digits)}, bits={len(significant_bits)}, features={len(significant_features)}")


if __name__ == "__main__":
    main()
