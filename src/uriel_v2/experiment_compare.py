from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _judgment(cohorts: dict[str, Any]) -> str:
    if not cohorts:
        return "NO SIGNAL"
    primary = []
    for values in cohorts.values():
        budget = values.get("budgets", {}).get("1000", {})
        primary.append(
            (
                float(budget.get("mean_max_hit_effect", 0.0)),
                int(budget.get("algorithm_5_plus", 0)) - int(budget.get("random_5_plus", 0)),
                float(budget.get("paired_permutation_p", 1.0)),
            )
        )
    if primary and all(effect > 0 and lift_5 > 0 and p <= 0.05 for effect, lift_5, p in primary):
        return "SUCCESS"
    if primary and all(effect >= 0 and lift_5 >= 0 for effect, lift_5, _ in primary) and any(
        effect > 0 or lift_5 > 0 for effect, lift_5, _ in primary
    ):
        return "WEAK SIGNAL"
    return "NO SIGNAL"


def compare_experiments(
    *,
    combinadic_metrics: str | Path,
    seed_basin_metrics: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    combinadic = _load(combinadic_metrics)
    seed_basin = _load(seed_basin_metrics)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cohorts = sorted(set(combinadic["cohorts"]).intersection(seed_basin["cohorts"]))
    for cohort in cohorts:
        for budget in (10, 100, 1_000, 10_000):
            combinadic_budget = combinadic["cohorts"][cohort]["budgets"][str(budget)]
            basin_budget = seed_basin["cohorts"][cohort]["budgets"][str(budget)]
            rows.extend(
                [
                    {
                        "cohort": cohort,
                        "metric": f"Mean Max Hit @{budget}",
                        "combinadic_random": combinadic_budget["random_mean_max_hit"],
                        "combinadic": combinadic_budget["algorithm_mean_max_hit"],
                        "seed_basin_random": basin_budget["random_mean_max_hit"],
                        "seed_basin": basin_budget["algorithm_mean_max_hit"],
                    },
                    {
                        "cohort": cohort,
                        "metric": f"4+ @{budget}",
                        "combinadic_random": combinadic_budget["random_4_plus"],
                        "combinadic": combinadic_budget["algorithm_4_plus"],
                        "seed_basin_random": basin_budget["random_4_plus"],
                        "seed_basin": basin_budget["algorithm_4_plus"],
                    },
                    {
                        "cohort": cohort,
                        "metric": f"5+ @{budget}",
                        "combinadic_random": combinadic_budget["random_5_plus"],
                        "combinadic": combinadic_budget["algorithm_5_plus"],
                        "seed_basin_random": basin_budget["random_5_plus"],
                        "seed_basin": basin_budget["algorithm_5_plus"],
                    },
                    {
                        "cohort": cohort,
                        "metric": f"6-hit @{budget}",
                        "combinadic_random": combinadic_budget["random_6"],
                        "combinadic": combinadic_budget["algorithm_6"],
                        "seed_basin_random": basin_budget["random_6"],
                        "seed_basin": basin_budget["algorithm_6"],
                    },
                ]
            )
    with (destination / "algorithm_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cohort", "metric", "combinadic_random", "combinadic",
                "seed_basin_random", "seed_basin",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    combinadic_judgment = _judgment(combinadic["cohorts"])
    seed_basin_judgment = _judgment(seed_basin["cohorts"])
    if combinadic_judgment == "SUCCESS" and seed_basin_judgment == "SUCCESS":
        decision = "C. 둘을 Hybrid"
    elif combinadic_judgment == "SUCCESS":
        decision = "A. Combinadic 계속"
    elif seed_basin_judgment == "SUCCESS":
        decision = "B. Seed Basin 계속"
    else:
        decision = "D. 둘 다 종료"
    summary = {
        "combinadic": combinadic_judgment,
        "seed_basin": seed_basin_judgment,
        "decision": decision,
        "hybrid_allowed": combinadic_judgment == "SUCCESS" or seed_basin_judgment == "SUCCESS",
        "comparison_rows": len(rows),
    }
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
