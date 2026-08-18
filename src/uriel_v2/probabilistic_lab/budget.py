from __future__ import annotations

import math

from uriel_v2.probabilistic_lab.schema import BudgetSpec


def checkpoint_steps(budget: BudgetSpec) -> tuple[int, ...]:
    """Map fractional checkpoints to monotone, unique integer steps."""

    steps = {max(1, min(budget.total, math.ceil(budget.total * fraction))) for fraction in budget.checkpoints}
    steps.add(budget.total)
    return tuple(sorted(steps))


def budget_fraction(step: int, budget: BudgetSpec) -> float:
    return min(1.0, float(step) / float(budget.total))
