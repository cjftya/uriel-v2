from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from uriel_v2.probabilistic_lab.schema import ExperimentBundle, JobSpec


AlgorithmRunner = Callable[["JobSpec", "np.random.Generator"], "ExperimentBundle"]


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._runners: dict[str, tuple[frozenset[str], AlgorithmRunner]] = {}

    def register(self, name: str, domains: set[str] | frozenset[str], runner: AlgorithmRunner) -> None:
        if name in self._runners:
            raise ValueError(f"algorithm already registered: {name}")
        self._runners[name] = (frozenset(domains), runner)

    def resolve(self, name: str, domain: str) -> AlgorithmRunner:
        try:
            domains, runner = self._runners[name]
        except KeyError as exc:
            raise KeyError(f"unknown algorithm: {name}") from exc
        if domain not in domains:
            raise ValueError(f"algorithm {name} does not support domain {domain}")
        return runner

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._runners))
