from __future__ import annotations

import math

import numpy as np
import pandas as pd


EARLY_CUTOFFS = (0.05, 0.10, 0.20)


def _safe_autocorrelation(values: np.ndarray) -> float:
    if values.size < 3 or float(np.std(values[:-1])) == 0.0 or float(np.std(values[1:])) == 0.0:
        return 0.0
    value = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    return value if math.isfinite(value) else 0.0


def extract_trajectory_features(traces: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe 5%, 10%, and 20% early-trajectory feature rows."""

    rows: list[dict[str, float | int | str]] = []
    if traces.empty:
        return pd.DataFrame(rows)
    for run_id, group in traces.groupby("run_id", sort=True):
        ordered = group.sort_values("step")
        for cutoff in EARLY_CUTOFFS:
            prefix = ordered[ordered["budget_fraction"] <= cutoff + 1e-12]
            if prefix.empty:
                prefix = ordered.iloc[[0]]
            steps = prefix["step"].to_numpy(dtype=float)
            best = prefix["best_so_far"].to_numpy(dtype=float)
            objectives = prefix["objective"].to_numpy(dtype=float)
            slope = 0.0
            if len(prefix) >= 2 and float(np.ptp(steps)) > 0.0:
                slope = float(np.polyfit(steps / steps.max(), best, 1)[0])
            transitions = np.abs(np.diff(objectives))
            rows.append(
                {
                    "run_id": str(run_id),
                    "cutoff": cutoff,
                    "observed_fraction": float(prefix["budget_fraction"].iloc[-1]),
                    "observed_steps": int(prefix["step"].iloc[-1]),
                    "objective_last": float(objectives[-1]),
                    "best_so_far": float(best[-1]),
                    "improvement_sum": float(prefix["improvement"].sum()),
                    "improvement_slope": slope,
                    "variance_mean": float(prefix["variance"].mean()),
                    "entropy_mean": float(prefix["entropy"].mean()),
                    "diversity_mean": float(prefix["diversity"].mean()),
                    "autocorrelation_lag1": _safe_autocorrelation(objectives),
                    "transition_magnitude_mean": float(np.mean(transitions)) if transitions.size else 0.0,
                    "stagnation_fraction": float(np.mean(prefix["improvement"].to_numpy(dtype=float) <= 0.0)),
                    "failure_signal_max": float(prefix["failure_signal"].max()),
                }
            )
    return pd.DataFrame(rows)
