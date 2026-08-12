from __future__ import annotations

from collections.abc import Sequence


def hit_count(prediction: Sequence[int], winner: Sequence[int]) -> int:
    return len(set(prediction).intersection(winner))


def positional_deviations(prediction: Sequence[int], winner: Sequence[int]) -> tuple[int, ...]:
    if len(prediction) != len(winner):
        raise ValueError("비교할 번호 개수가 같아야 합니다")
    return tuple(predicted - actual for predicted, actual in zip(sorted(prediction), sorted(winner), strict=True))


def positional_mae(prediction: Sequence[int], winner: Sequence[int]) -> float:
    deviations = positional_deviations(prediction, winner)
    return sum(abs(value) for value in deviations) / len(deviations)
