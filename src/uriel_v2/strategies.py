from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from uriel_v2.models import Draw, Prediction
from uriel_v2.rng import generate_numbers, stable_seed


STRATEGIES = ("round", "history-digest", "rolling-mix")


def derive_seed(
    strategy: str,
    history: Sequence[Draw],
    target_round: int,
    *,
    variant: int = 0,
    history_window: int = 64,
) -> int:
    if strategy not in STRATEGIES:
        raise ValueError(f"알 수 없는 전략입니다: {strategy}")
    if target_round <= 0:
        raise ValueError("target_round는 양수여야 합니다")
    if variant < 0:
        raise ValueError("variant는 0 이상이어야 합니다")

    window = tuple(history[-history_window:]) if history_window > 0 else tuple(history)
    if strategy == "round":
        return stable_seed(strategy, target_round, variant)
    if not window:
        raise ValueError(f"{strategy} 전략에는 과거 회차가 필요합니다")

    canonical_history = tuple((draw.round_no, draw.numbers) for draw in window)
    if strategy == "history-digest":
        return stable_seed(strategy, target_round, canonical_history, variant)

    frequencies = Counter(number for draw in window for number in draw.numbers)
    frequency_signature = tuple(frequencies[number] for number in range(1, 46))
    weighted_sums = tuple((index + 1) * sum(draw.numbers) for index, draw in enumerate(window))
    gap_signature = tuple(
        number_b - number_a
        for draw in window[-16:]
        for number_a, number_b in zip(draw.numbers, draw.numbers[1:])
    )
    return stable_seed(
        strategy,
        target_round,
        window[-1].round_no,
        frequency_signature,
        weighted_sums,
        gap_signature,
        variant,
    )


def create_predictions(
    strategy: str,
    history: Sequence[Draw],
    target_round: int,
    *,
    candidates: int = 1,
    history_window: int = 64,
) -> list[Prediction]:
    if candidates <= 0:
        raise ValueError("candidates는 양수여야 합니다")
    return [
        Prediction(
            strategy=strategy,
            seed=(seed := derive_seed(
                strategy,
                history,
                target_round,
                variant=variant,
                history_window=history_window,
            )),
            numbers=generate_numbers(seed),
            variant=variant,
        )
        for variant in range(candidates)
    ]
