"""General stochastic-algorithm experiment infrastructure.

This package is intentionally isolated from the Lotto-specific experiment code.
It provides stable schemas, reproducible random streams, worker execution, trace
capture, and Parquet datasets for algorithm-performance modelling research.
"""

from uriel_v2.probabilistic_lab.schema import (
    AlgorithmSpec,
    BudgetSpec,
    ExperimentBundle,
    JobSpec,
    ProblemSpec,
    RunResult,
    TracePoint,
)

__all__ = [
    "AlgorithmSpec",
    "BudgetSpec",
    "ExperimentBundle",
    "JobSpec",
    "ProblemSpec",
    "RunResult",
    "TracePoint",
]
