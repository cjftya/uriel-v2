from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Draw:
    round_no: int
    numbers: tuple[int, int, int, int, int, int]
    bonus: int | None = None


@dataclass(frozen=True, slots=True)
class Prediction:
    strategy: str
    seed: int
    numbers: tuple[int, int, int, int, int, int]
    variant: int


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    round_no: int
    strategy: str
    variant: int
    seed: int
    prediction: tuple[int, ...]
    winner: tuple[int, ...]
    hits: int
    set_distance: int
    positional_mae: float
    signed_bias: float
    deviations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReverseMatch:
    seed: int
    numbers: tuple[int, ...]
    hits: int
    positional_mae: float
    set_distance: int
    signed_bias: float
    deviations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SearchChunkTask:
    start: int
    end: int
    target: tuple[int, ...]
    min_hits: int
    result_limit: int
    collect_all_matches: bool = False


@dataclass(frozen=True, slots=True)
class SearchChunkResult:
    start: int
    end: int
    elapsed_seconds: float
    hit_distribution: tuple[int, ...]
    matches: tuple[ReverseMatch, ...]
    best: tuple[ReverseMatch, ...]


@dataclass(frozen=True, slots=True)
class ReverseSearchResult:
    start: int
    end: int
    elapsed_seconds: float
    hit_distribution: tuple[int, ...]
    matches: tuple[ReverseMatch, ...]
    best: tuple[ReverseMatch, ...]
    chunks: int
    workers: int
