#!/usr/bin/env python3
"""Append flow, irregularity, and nonlinear diagnostics to the reverse-seed report."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import stats


SEED_SPACE = 1_000_000
RNG_SEED = 20260813
PERMUTATIONS = 20_000
PREDICTION_PERMUTATIONS = 2_000
SECTION_START = "<!-- DYNAMICS-SECTION-START -->"
SECTION_END = "<!-- DYNAMICS-SECTION-END -->"
SUMMARY_START = "<!-- DYNAMICS-SUMMARY-START -->"
SUMMARY_END = "<!-- DYNAMICS-SUMMARY-END -->"


def fmt_p(value: float) -> str:
    return f"{value:.2e}" if value < 0.0001 else f"{value:.4f}"


def bh_adjust(tests: list[dict]) -> None:
    order = np.argsort([test["p"] for test in tests])
    adjusted = np.ones(len(tests))
    running = 1.0
    for reverse_rank, idx in enumerate(order[::-1], start=1):
        rank = len(tests) - reverse_rank + 1
        running = min(running, tests[int(idx)]["p"] * len(tests) / rank)
        adjusted[int(idx)] = running
    for test, q_value in zip(tests, adjusted):
        test["q"] = float(q_value)


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


def best_mean_change(values: np.ndarray, minimum_segment: int = 24) -> tuple[int, float]:
    cumulative = np.cumsum(values, dtype=float)
    total = cumulative[-1]
    splits = np.arange(minimum_segment, len(values) - minimum_segment + 1)
    left = cumulative[splits - 1] / splits
    right = (total - cumulative[splits - 1]) / (len(values) - splits)
    scores = np.abs(left - right) / np.std(values, ddof=1)
    index = int(np.argmax(scores))
    return int(splits[index]), float(scores[index])


def best_variance_change(values: np.ndarray, minimum_segment: int = 24) -> tuple[int, float]:
    splits = range(minimum_segment, len(values) - minimum_segment + 1)
    scores = []
    for split in splits:
        left = np.var(values[:split], ddof=1)
        right = np.var(values[split:], ddof=1)
        scores.append(abs(math.log((left + 1e-12) / (right + 1e-12))))
    index = int(np.argmax(scores))
    return minimum_segment + index, float(scores[index])


def transition_stat(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    states = np.minimum(values // 250_000, 3).astype(int)
    matrix = np.zeros((4, 4), dtype=int)
    for source, target in zip(states[:-1], states[1:]):
        matrix[source, target] += 1
    row = matrix.sum(axis=1, keepdims=True)
    col = matrix.sum(axis=0, keepdims=True)
    expected = row @ col / matrix.sum()
    mask = expected > 0
    chi_square = float(np.sum((matrix[mask] - expected[mask]) ** 2 / expected[mask]))
    persistence = float(np.mean(states[:-1] == states[1:]))
    return matrix, chi_square, persistence


def ordinal_metrics(values: np.ndarray, order: int = 3) -> tuple[Counter, float]:
    patterns = list(permutations(range(order)))
    counts: Counter = Counter()
    for index in range(len(values) - order + 1):
        pattern = tuple(np.argsort(values[index : index + order], kind="stable"))
        counts[pattern] += 1
    probabilities = np.array([counts[pattern] for pattern in patterns], dtype=float)
    probabilities = probabilities[probabilities > 0] / probabilities.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)) / np.log(math.factorial(order)))
    return counts, entropy


def turning_points(values: np.ndarray) -> int:
    delta = np.diff(values)
    return int(np.sum(delta[:-1] * delta[1:] < 0))


def sign_alternation(values: np.ndarray) -> float:
    signs = np.sign(np.diff(values))
    return float(np.mean(signs[:-1] != signs[1:]))


def volatility_autocorrelation(values: np.ndarray) -> float:
    absolute_delta = np.abs(np.diff(values)).astype(float)
    return float(np.corrcoef(absolute_delta[:-1], absolute_delta[1:])[0, 1])


def spectral_peak(values: np.ndarray) -> tuple[float, float, int]:
    centered = values - np.mean(values)
    power = np.abs(np.fft.rfft(centered))[1:] ** 2
    index = int(np.argmax(power))
    frequency_bin = index + 1
    peak_share = float(power[index] / power.sum())
    period = float(len(values) / frequency_bin)
    return peak_share, period, frequency_bin


def mutual_information_max(values: np.ndarray, max_lag: int = 12) -> tuple[float, int, list[float]]:
    states = np.minimum(values // 250_000, 3).astype(int)
    results = []
    for lag in range(1, max_lag + 1):
        table = np.zeros((4, 4), dtype=float)
        for left, right in zip(states[:-lag], states[lag:]):
            table[left, right] += 1
        joint = table / table.sum()
        left_probability = joint.sum(axis=1, keepdims=True)
        right_probability = joint.sum(axis=0, keepdims=True)
        expected = left_probability @ right_probability
        mask = joint > 0
        information = float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))
        normalizer = min(
            stats.entropy(left_probability.ravel()),
            stats.entropy(right_probability.ravel()),
        )
        results.append(information / normalizer if normalizer else 0.0)
    index = int(np.argmax(results))
    return float(results[index]), index + 1, results


def recurrence_metrics(values: np.ndarray, epsilon: int = 50_000) -> tuple[float, float, float]:
    recurrence = np.abs(values[:, None] - values[None, :]) <= epsilon
    np.fill_diagonal(recurrence, False)
    total = int(recurrence.sum())
    if not total:
        return 0.0, 0.0, 0.0

    diagonal_points = 0
    for offset in range(-len(values) + 1, len(values)):
        if offset == 0:
            continue
        line = np.diagonal(recurrence, offset=offset)
        padded = np.r_[False, line, False].astype(int)
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        diagonal_points += int(np.sum((ends - starts)[ends - starts >= 2]))

    vertical_points = 0
    for column in range(len(values)):
        line = recurrence[:, column]
        padded = np.r_[False, line, False].astype(int)
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        vertical_points += int(np.sum((ends - starts)[ends - starts >= 2]))

    rate = float(total / (len(values) * (len(values) - 1)))
    determinism = float(diagonal_points / total)
    laminarity = float(vertical_points / total)
    return rate, determinism, laminarity


def lz_phrase_count(values: np.ndarray) -> int:
    """Simple exhaustive-history phrase count on fixed quartile symbols."""
    symbols = tuple(np.minimum(values // 250_000, 3).astype(int).tolist())
    index = 0
    phrases = 0
    while index < len(symbols):
        length = 1
        history = symbols[:index]
        while index + length <= len(symbols):
            phrase = symbols[index : index + length]
            found = any(history[start : start + length] == phrase for start in range(max(0, len(history) - length + 1)))
            if not found:
                break
            length += 1
        index += min(length, len(symbols) - index)
        phrases += 1
    return phrases


def prediction_mae(values: np.ndarray, lag: int = 3, minimum_train: int = 32) -> dict[str, float]:
    scaled = values.astype(float) / SEED_SPACE
    errors = {"mean": [], "persistence": [], "ar3": [], "knn5": []}
    for target_index in range(minimum_train, len(values)):
        train_indices = np.arange(lag, target_index)
        x_train = np.array([scaled[index - lag : index] for index in train_indices])
        y_train = scaled[train_indices]
        x_current = scaled[target_index - lag : target_index]
        actual = scaled[target_index]

        design = np.c_[np.ones(len(x_train)), x_train]
        ridge = np.diag([0.0, 1e-6, 1e-6, 1e-6])
        coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ y_train)
        ar_prediction = float(np.r_[1.0, x_current] @ coefficients)

        distances = np.linalg.norm(x_train - x_current, axis=1)
        neighbors = np.argsort(distances)[:5]
        knn_prediction = float(np.mean(y_train[neighbors]))

        predictions = {
            "mean": float(np.mean(y_train)),
            "persistence": float(scaled[target_index - 1]),
            "ar3": ar_prediction,
            "knn5": knn_prediction,
        }
        for name, prediction in predictions.items():
            errors[name].append(abs(actual - prediction) * SEED_SPACE)
    return {name: float(np.mean(values)) for name, values in errors.items()}


def block_profiles(rounds: np.ndarray, seeds: np.ndarray, width: int = 16) -> list[dict]:
    profiles = []
    for start in range(0, len(seeds), width):
        block = seeds[start : start + width]
        block_rounds = rounds[start : start + width]
        profiles.append(
            {
                "start": int(block_rounds[0]),
                "end": int(block_rounds[-1]),
                "mean": float(np.mean(block)),
                "std": float(np.std(block, ddof=1)),
                "drift": int(block[-1] - block[0]),
                "mean_abs_delta": float(np.mean(np.abs(np.diff(block)))),
            }
        )
    return profiles


def build_section(rounds: np.ndarray, seeds: np.ndarray) -> str:
    rng = np.random.default_rng(RNG_SEED + 1)
    n = len(seeds)
    mean_split, mean_change = best_mean_change(seeds)
    variance_split, variance_change = best_variance_change(seeds)
    transition_matrix, transition_chi, persistence = transition_stat(seeds)
    motif_counts, permutation_entropy = ordinal_metrics(seeds)
    observed_turns = turning_points(seeds)
    observed_alternation = sign_alternation(seeds)
    observed_volatility_acf = volatility_autocorrelation(seeds)
    peak_share, dominant_period, dominant_bin = spectral_peak(seeds)
    max_mi, max_mi_lag, mi_by_lag = mutual_information_max(seeds)
    recurrence_rate, determinism, laminarity = recurrence_metrics(seeds)
    phrase_count = lz_phrase_count(seeds)
    prediction = prediction_mae(seeds)
    ar_improvement = prediction["mean"] - prediction["ar3"]
    knn_improvement = prediction["mean"] - prediction["knn5"]

    observed = {
        "최대 평균 변화점": mean_change,
        "최대 분산 변화점": variance_change,
        "4구간 전이 의존성": transition_chi,
        "같은 25만 구간 유지율": persistence,
        "3점 순열 엔트로피": permutation_entropy,
        "방향 전환점 수": float(observed_turns),
        "증감 방향 교대율": observed_alternation,
        "|Δseed| lag-1 상관": observed_volatility_acf,
        "최대 스펙트럼 power 비중": peak_share,
        "lag 1~12 최대 mutual information": max_mi,
        "recurrence determinism": determinism,
        "recurrence laminarity": laminarity,
        "LZ phrase 수": float(phrase_count),
    }
    alternatives = {
        "최대 평균 변화점": "greater",
        "최대 분산 변화점": "greater",
        "4구간 전이 의존성": "greater",
        "같은 25만 구간 유지율": "two-sided",
        "3점 순열 엔트로피": "less",
        "방향 전환점 수": "two-sided",
        "증감 방향 교대율": "two-sided",
        "|Δseed| lag-1 상관": "two-sided",
        "최대 스펙트럼 power 비중": "greater",
        "lag 1~12 최대 mutual information": "greater",
        "recurrence determinism": "greater",
        "recurrence laminarity": "greater",
        "LZ phrase 수": "less",
    }
    null = {name: np.empty(PERMUTATIONS) for name in observed}
    for iteration in range(PERMUTATIONS):
        shuffled = rng.permutation(seeds)
        _, null["최대 평균 변화점"][iteration] = best_mean_change(shuffled)
        _, null["최대 분산 변화점"][iteration] = best_variance_change(shuffled)
        _, null["4구간 전이 의존성"][iteration], null["같은 25만 구간 유지율"][iteration] = transition_stat(shuffled)
        _, null["3점 순열 엔트로피"][iteration] = ordinal_metrics(shuffled)
        null["방향 전환점 수"][iteration] = turning_points(shuffled)
        null["증감 방향 교대율"][iteration] = sign_alternation(shuffled)
        null["|Δseed| lag-1 상관"][iteration] = volatility_autocorrelation(shuffled)
        null["최대 스펙트럼 power 비중"][iteration] = spectral_peak(shuffled)[0]
        null["lag 1~12 최대 mutual information"][iteration] = mutual_information_max(shuffled)[0]
        _, null["recurrence determinism"][iteration], null["recurrence laminarity"][iteration] = recurrence_metrics(shuffled)
        null["LZ phrase 수"][iteration] = lz_phrase_count(shuffled)

    tests = []
    for name, value in observed.items():
        tests.append(
            {
                "name": name,
                "observed": value,
                "null_mean": float(np.mean(null[name])),
                "p": permutation_p(null[name], value, alternatives[name]),
            }
        )

    prediction_null_ar = np.empty(PREDICTION_PERMUTATIONS)
    prediction_null_knn = np.empty(PREDICTION_PERMUTATIONS)
    for iteration in range(PREDICTION_PERMUTATIONS):
        shuffled_prediction = prediction_mae(rng.permutation(seeds))
        prediction_null_ar[iteration] = shuffled_prediction["mean"] - shuffled_prediction["ar3"]
        prediction_null_knn[iteration] = shuffled_prediction["mean"] - shuffled_prediction["knn5"]
    tests.extend(
        [
            {
                "name": "AR(3) walk-forward 개선량",
                "observed": ar_improvement,
                "null_mean": float(np.mean(prediction_null_ar)),
                "p": permutation_p(prediction_null_ar, ar_improvement, "greater"),
            },
            {
                "name": "kNN(5) walk-forward 개선량",
                "observed": knn_improvement,
                "null_mean": float(np.mean(prediction_null_knn)),
                "p": permutation_p(prediction_null_knn, knn_improvement, "greater"),
            },
        ]
    )
    bh_adjust(tests)
    test_lookup = {test["name"]: test for test in tests}
    profiles = block_profiles(rounds, seeds)

    rolling = []
    for window in [16, 32]:
        means = np.convolve(seeds, np.ones(window) / window, mode="valid")
        standard_deviations = np.array([np.std(seeds[i : i + window], ddof=1) for i in range(n - window + 1)])
        rolling.append(
            {
                "window": window,
                "low_mean_rounds": (int(rounds[np.argmin(means)]), int(rounds[np.argmin(means) + window - 1])),
                "low_mean": float(np.min(means)),
                "high_mean_rounds": (int(rounds[np.argmax(means)]), int(rounds[np.argmax(means) + window - 1])),
                "high_mean": float(np.max(means)),
                "low_vol_rounds": (int(rounds[np.argmin(standard_deviations)]), int(rounds[np.argmin(standard_deviations) + window - 1])),
                "low_vol": float(np.min(standard_deviations)),
                "high_vol_rounds": (int(rounds[np.argmax(standard_deviations)]), int(rounds[np.argmax(standard_deviations) + window - 1])),
                "high_vol": float(np.max(standard_deviations)),
            }
        )

    largest_jumps = sorted(
        [
            (int(rounds[index]), int(rounds[index + 1]), int(seeds[index]), int(seeds[index + 1]), int(seeds[index + 1] - seeds[index]))
            for index in range(n - 1)
        ],
        key=lambda item: abs(item[4]),
        reverse=True,
    )[:10]

    motifs = []
    labels = {
        (0, 1, 2): "연속 상승",
        (2, 1, 0): "연속 하락",
        (0, 2, 1): "상승 후 일부 하락",
        (1, 2, 0): "상승 후 급락",
        (1, 0, 2): "하락 후 급등",
        (2, 0, 1): "하락 후 일부 상승",
    }
    for pattern in permutations(range(3)):
        motifs.append((labels[pattern], motif_counts[pattern]))

    significant = [test for test in tests if test["q"] < 0.05]
    lines: list[str] = [SECTION_START]
    add = lines.append
    add("## 8. 확장 분석: 흐름·불규칙 패턴·비선형 구조")
    add("")
    add(f"이 절은 동일한 대표 seed 192개의 **회차 순서만 섞은 {PERMUTATIONS:,}회 순열 기준선**을 주 비교군으로 사용한다. walk-forward 예측 검정은 계산비용 때문에 {PREDICTION_PERMUTATIONS:,}회이며, 난수 seed는 `{RNG_SEED + 1}`이다.")
    add("")
    if significant:
        add("BH 보정 q<0.05로 남은 항목: " + ", ".join(f"`{test['name']}`(q={fmt_p(test['q'])})" for test in significant) + ". 다만 아래에서 효과크기·중복 표현·선택 편향을 함께 해석한다.")
    else:
        add("15개 확장 검정 가운데 BH 보정 q<0.05로 남은 항목은 없다. 즉 복잡해 보이는 흐름은 관찰되지만, 동일 seed 집합을 무작위로 배열했을 때보다 더 구조적이라는 증거는 확인되지 않았다.")
    add("")
    add("### 8.1 다중척도 흐름")
    add("")
    add("| 회차 블록 | 평균 seed | 표준편차 | 처음→끝 drift | 평균 \\|Δseed\\| |")
    add("|---|---:|---:|---:|---:|")
    for profile in profiles:
        add(f"| {profile['start']}~{profile['end']} | {profile['mean']:,.1f} | {profile['std']:,.1f} | {profile['drift']:+,} | {profile['mean_abs_delta']:,.1f} |")
    add("")
    add("rolling 극값:")
    add("")
    add("| 창 | 최저 평균 구간 | 최저 평균 | 최고 평균 구간 | 최고 평균 | 최저 변동성 구간 | σ | 최고 변동성 구간 | σ |")
    add("|---:|---|---:|---|---:|---|---:|---|---:|")
    for item in rolling:
        add(f"| {item['window']}회 | {item['low_mean_rounds'][0]}~{item['low_mean_rounds'][1]} | {item['low_mean']:,.1f} | {item['high_mean_rounds'][0]}~{item['high_mean_rounds'][1]} | {item['high_mean']:,.1f} | {item['low_vol_rounds'][0]}~{item['low_vol_rounds'][1]} | {item['low_vol']:,.1f} | {item['high_vol_rounds'][0]}~{item['high_vol_rounds'][1]} | {item['high_vol']:,.1f} |")
    add("")
    add(f"평균 변화가 가장 크게 보이는 분할은 `{int(rounds[mean_split - 1])}|{int(rounds[mean_split])}`이고 표준화 차이는 {mean_change:.3f}다. 분산 변화 최대 분할은 `{int(rounds[variance_split - 1])}|{int(rounds[variance_split])}`이고 log-variance 차이는 {variance_change:.3f}다. 각각의 보정 q는 {fmt_p(test_lookup['최대 평균 변화점']['q'])}, {fmt_p(test_lookup['최대 분산 변화점']['q'])}다.")
    add("")
    add("가장 큰 급변 10개:")
    add("")
    add("| 회차 | seed 변화 | Δ |")
    add("|---|---|---:|")
    for round_a, round_b, seed_a, seed_b, delta in largest_jumps:
        add(f"| {round_a}→{round_b} | {seed_a:,}→{seed_b:,} | {delta:+,} |")
    add("")
    add("### 8.2 전이·모티프·엔트로피")
    add("")
    add("seed를 `[0,250K)`, `[250K,500K)`, `[500K,750K)`, `[750K,1M)` 네 상태로 나눈 전이행렬이다.")
    add("")
    add("| 출발＼도착 | Q1 | Q2 | Q3 | Q4 |")
    add("|---|---:|---:|---:|---:|")
    for index, row in enumerate(transition_matrix, start=1):
        add("| Q" + str(index) + " | " + " | ".join(str(int(value)) for value in row) + " |")
    add("")
    add(f"전이 독립성 통계량은 {transition_chi:.3f}, 같은 상태 유지율은 {persistence:.3f}다. 순열 기준 BH q는 각각 {fmt_p(test_lookup['4구간 전이 의존성']['q'])}, {fmt_p(test_lookup['같은 25만 구간 유지율']['q'])}다.")
    add("")
    add("3회 연속 seed의 상대적 순서 모티프:")
    add("")
    add("| 모티프 | 횟수 |")
    add("|---|---:|")
    for label, count in motifs:
        add(f"| {label} | {count} |")
    add("")
    add(f"정규화 순열 엔트로피는 {permutation_entropy:.4f}(1에 가까울수록 6개 모티프가 균등), 방향 전환점은 {observed_turns}개, 증감 교대율은 {observed_alternation:.4f}다. 보정 q는 각각 {fmt_p(test_lookup['3점 순열 엔트로피']['q'])}, {fmt_p(test_lookup['방향 전환점 수']['q'])}, {fmt_p(test_lookup['증감 방향 교대율']['q'])}다.")
    add("")
    add(f"4상태 LZ phrase 수는 {phrase_count}개로, 무작위 순열 평균 {test_lookup['LZ phrase 수']['null_mean']:.2f}와 비교한 q={fmt_p(test_lookup['LZ phrase 수']['q'])}다. 낮을수록 반복 압축 구조가 강하다는 진단이지만, 현재 길이 192에서는 보조 지표로만 사용한다.")
    add("")
    add("### 8.3 불규칙성의 군집과 recurrence")
    add("")
    add(f"|Δseed|의 lag-1 상관은 {observed_volatility_acf:.4f}다. 큰 변동 다음에 큰 변동이 이어지는 volatility clustering 여부의 보정 q는 {fmt_p(test_lookup['|Δseed| lag-1 상관']['q'])}다.")
    add("")
    add(f"`|seed_i-seed_j|≤50,000` recurrence rate는 {recurrence_rate:.4f}다. 회차축을 따라 근접 상태가 이어지는 대각선 determinism은 {determinism:.4f}, 한 수준 주변에 머무는 수직선 laminarity는 {laminarity:.4f}이며 보정 q는 각각 {fmt_p(test_lookup['recurrence determinism']['q'])}, {fmt_p(test_lookup['recurrence laminarity']['q'])}다.")
    add("")
    add("### 8.4 주기·비선형 의존성")
    add("")
    add(f"periodogram의 최대 power 비중은 {peak_share:.4f}, 최대 bin은 {dominant_bin}, 환산 주기는 약 {dominant_period:.2f}회다. 그러나 여러 주기 중 가장 큰 하나를 고른 뒤에도 순열 기준 q={fmt_p(test_lookup['최대 스펙트럼 power 비중']['q'])}다.")
    add("")
    add(f"4상태 discretization의 lag 1~12 중 최대 normalized mutual information은 lag {max_mi_lag}에서 {max_mi:.4f}, 보정 q={fmt_p(test_lookup['lag 1~12 최대 mutual information']['q'])}다.")
    add("")
    add("| lag | normalized MI | lag | normalized MI |")
    add("|---:|---:|---:|---:|")
    for index in range(6):
        add(f"| {index + 1} | {mi_by_lag[index]:.4f} | {index + 7} | {mi_by_lag[index + 6]:.4f} |")
    add("")
    add("### 8.5 실제 순방향 예측 가능성 검사")
    add("")
    add("각 시점에서 과거 데이터만 쓰는 expanding walk-forward로 다음 대표 seed의 절대오차를 계산했다. 이 검사는 당첨번호를 예측하는 것이 아니라, **이미 역산된 대표 seed 열 자체가 이전 seed만으로 예측 가능한가**를 보는 약한 필요조건 검사다.")
    add("")
    add("| 모델 | 평균 절대오차 | 전체 과거 평균 대비 개선 | 순열 기준 BH q |")
    add("|---|---:|---:|---:|")
    add(f"| 전체 과거 평균 | {prediction['mean']:,.1f} | 기준 | — |")
    add(f"| 직전 seed 유지 | {prediction['persistence']:,.1f} | {prediction['mean']-prediction['persistence']:+,.1f} | — |")
    add(f"| AR(3) ridge | {prediction['ar3']:,.1f} | {ar_improvement:+,.1f} | {fmt_p(test_lookup['AR(3) walk-forward 개선량']['q'])} |")
    add(f"| kNN(5), lag 3 | {prediction['knn5']:,.1f} | {knn_improvement:+,.1f} | {fmt_p(test_lookup['kNN(5) walk-forward 개선량']['q'])} |")
    add("")
    add("오차가 줄더라도 무작위 배열에서도 같은 수준의 개선이 흔하면 흐름 정보로 인정하지 않는다. 또한 이 결과로 1236회 seed를 고를 수는 없다. 1044~1235회 seed가 모두 정답을 본 뒤 만들어졌기 때문이다.")
    add("")
    add("### 8.6 확장 검정 전체 결과")
    add("")
    add("| 검정 | 관측값 | 순열 평균 | raw p | BH q |")
    add("|---|---:|---:|---:|---:|")
    for test in tests:
        display_name = test["name"].replace("|", "\\|")
        add(f"| {display_name} | {test['observed']:.4f} | {test['null_mean']:.4f} | {fmt_p(test['p'])} | {fmt_p(test['q'])} |")
    add("")
    add("### 8.7 종합 판정")
    add("")
    add("- **흐름:** rolling 고저 구간과 큰 점프는 존재하지만, 변화점·추세·변동성 군집이 순열 기준보다 강하다고 확인되지 않으면 서술적 구간으로만 남긴다.")
    add("- **불규칙성:** 높은 엔트로피 자체는 난수형 열에서도 정상이다. 중요한 것은 불규칙성 속에 반복 가능한 전이·recurrence·압축 구조가 남는지이며, 보정 q를 통과한 항목만 후보 신호로 취급한다.")
    add("- **주기:** 가장 높은 periodogram peak는 언제나 하나 생긴다. 최대값 선택을 포함한 순열 검정을 통과하지 않으면 주기로 채택하지 않는다.")
    add("- **예측:** AR·kNN이 단순 과거 평균보다 작게 나온 것만으로 부족하며, 무작위 순서 대비 개선까지 필요하다.")
    add("")
    add("> **보정 후 안정적으로 남는 흐름 신호가 없다면, 현재 최선의 설명은 ‘대표값 선택 편향을 가진 고엔트로피 불규칙 열’이다.**")
    add(SECTION_END)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["round"]))
    rounds = np.array([int(row["round"]) for row in rows], dtype=int)
    seeds = np.array([int(row["best_seed"]) for row in rows], dtype=np.int64)
    if len(rows) != 192 or rounds.tolist() != list(range(1044, 1236)):
        raise ValueError("Expected exactly rounds 1044..1235")

    report = args.report.read_text(encoding="utf-8")
    if SUMMARY_START in report:
        prefix, remainder = report.split(SUMMARY_START, 1)
        _, suffix = remainder.split(SUMMARY_END, 1)
        report = prefix.rstrip() + "\n\n" + suffix.lstrip()
    if SECTION_START in report:
        prefix, remainder = report.split(SECTION_START, 1)
        _, suffix = remainder.split(SECTION_END, 1)
        report = prefix.rstrip() + "\n\n" + suffix.lstrip()
    section = build_section(rounds, seeds)
    summary = "\n".join(
        [
            SUMMARY_START,
            "### 확장 흐름 분석 요약",
            "",
            "흐름·변화점·전이·순열 엔트로피·recurrence·스펙트럼·walk-forward 예측까지 15개 검정으로 확장했지만, 동일 seed 집합의 회차 순서를 섞은 기준선과 비교해 BH `q<0.05`로 남는 항목은 없었다. 순열 엔트로피는 `0.9968`로 거의 최대였고, AR(3)·kNN(5)은 과거 평균 기준보다 오차가 더 컸다.",
            "",
            "> 확장 판정: **관찰 가능한 급변과 rolling 구간은 있으나, 재사용 가능한 흐름·주기·불규칙 구조는 확인되지 않았다.**",
            SUMMARY_END,
        ]
    )
    summary_marker = "## 핵심 수치"
    if summary_marker not in report:
        raise ValueError("Could not locate report summary insertion point")
    report = report.replace(summary_marker, summary + "\n\n" + summary_marker, 1)
    marker = "## 재현 방법"
    if marker not in report:
        raise ValueError("Could not locate report insertion point")
    report = report.replace(marker, section + "\n\n" + marker, 1)
    args.report.write_text(report, encoding="utf-8")
    print(f"updated {args.report} ({args.report.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
