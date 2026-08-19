from __future__ import annotations

import hashlib
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from uriel_v2.logging_config import _timezone
from uriel_v2.probabilistic_lab.problems import build_pilot_problems
from uriel_v2.probabilistic_lab.schema import AlgorithmSpec, BudgetSpec, JobSpec, canonical_json
from uriel_v2.probabilistic_lab.storage import write_dataset
from uriel_v2.probabilistic_lab.validation import validate_dataset
from uriel_v2.probabilistic_lab.worker import run_jobs
from uriel_v2.provenance import current_git_commit


COMPARISONS = {
    "sampling_rqmc_vs_iid": ("monte_carlo_iid", "rqmc_sobol"),
    "optimization_cmaes_vs_random": ("random_search", "cma_es"),
}


def core_algorithms() -> dict[str, AlgorithmSpec]:
    return {
        "monte_carlo_iid": AlgorithmSpec(
            algorithm="monte_carlo_iid",
            algorithm_family="independent_sampling",
            random_mechanism="Independent Sampling",
            configuration={"estimator": "sample_mean"},
        ),
        "rqmc_sobol": AlgorithmSpec(
            algorithm="rqmc_sobol",
            algorithm_family="structured_sampling",
            random_mechanism="Structured Sampling",
            configuration={"sequence": "Sobol", "scramble": "LMS+shift", "estimator": "sample_mean"},
        ),
        "random_search": AlgorithmSpec(
            algorithm="random_search",
            algorithm_family="random_search",
            random_mechanism="Independent Sampling",
            configuration={"proposal": "uniform_within_bounds"},
        ),
        "cma_es": AlgorithmSpec(
            algorithm="cma_es",
            algorithm_family="adaptive_distribution",
            random_mechanism="Adaptive Distribution",
            configuration={"sigma_fraction": 0.30, "boundary": "clip"},
        ),
    }


def build_phase2_jobs(
    *,
    instances_per_family: int,
    seed_replicates: int,
    master_seed: int,
    sampling_budget: int,
    optimization_budget: int,
) -> list[JobSpec]:
    if seed_replicates <= 0:
        raise ValueError("seed_replicates must be positive")
    algorithms = core_algorithms()
    problems = build_pilot_problems(instances_per_family, master_seed)
    jobs: list[JobSpec] = []
    for problem_index, problem in enumerate(problems):
        algorithm_names = (
            ("monte_carlo_iid", "rqmc_sobol")
            if problem.domain == "sampling"
            else ("random_search", "cma_es")
        )
        budget = BudgetSpec(
            budget_type="samples" if problem.domain == "sampling" else "evaluations",
            total=sampling_budget if problem.domain == "sampling" else optimization_budget,
        )
        for replicate in range(seed_replicates):
            sequence = np.random.SeedSequence([master_seed, problem_index, replicate])
            paired_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
            for algorithm_name in algorithm_names:
                jobs.append(
                    JobSpec(
                        problem=problem,
                        algorithm=algorithms[algorithm_name],
                        seed=paired_seed,
                        budget=budget,
                        rng_version=f"NumPy {np.__version__}",
                    )
                )
    return jobs


def _pair_id(row: pd.Series, comparison: str) -> str:
    payload = {
        "comparison": comparison,
        "problem_id": row["problem_id"],
        "seed": int(row["seed"]),
        "budget_type": row["budget_type"],
        "budget": int(row["budget"]),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:24]


def _bootstrap_mean_ci(values: np.ndarray, seed: int, iterations: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _preregistration(parameters: dict[str, Any], bootstrap_iterations: int) -> dict[str, Any]:
    return {
        "phase": 2,
        "comparisons": COMPARISONS,
        "job_parameters": parameters,
        "bootstrap_iterations": bootstrap_iterations,
        "inference_unit": "problem-instance mean across paired seeds",
        "pair_key": ["problem_id", "seed", "budget_type", "budget"],
        "primary_metric": "challenger quality - baseline quality, quality=1/(1+objective)",
        "mechanism_signal_criteria": {
            "problem_count_at_least_8": True,
            "mean_quality_difference_positive": True,
            "bootstrap_95_ci_lower_above_zero": True,
            "wilcoxon_one_sided_p_le_0_05": True,
            "challenger_seed_pair_win_rate_at_least_0_60": True,
            "all_problem_family_mean_differences_positive": True,
        },
        "phase2_completion_criteria": {
            "data_validation_pass": True,
            "four_algorithms_present": True,
            "three_random_mechanisms_present": True,
            "complete_pair_coverage": True,
            "all_pair_numeric_values_finite": True,
            "all_problem_seed_replicates_complete": True,
            "no_execution_failures": True,
        },
    }


def build_paired_comparisons(
    runs: pd.DataFrame,
    problems: pd.DataFrame,
    *,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pair_frames: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    key_columns = ["problem_id", "seed", "budget_type", "budget"]
    problem_columns = ["problem_id", "problem_family", "domain", "dimension", "condition_number", "multimodality"]
    for comparison_index, (comparison, (baseline_name, challenger_name)) in enumerate(COMPARISONS.items()):
        baseline = runs[runs["algorithm"] == baseline_name]
        challenger = runs[runs["algorithm"] == challenger_name]
        if baseline.duplicated(key_columns).any() or challenger.duplicated(key_columns).any():
            raise ValueError(f"duplicate paired key in {comparison}")
        merged = baseline.merge(challenger, on=key_columns, suffixes=("_baseline", "_challenger"), how="outer", indicator=True)
        if not (merged["_merge"] == "both").all():
            missing = merged.loc[merged["_merge"] != "both", key_columns + ["_merge"]]
            raise ValueError(f"incomplete pairing in {comparison}: {missing.head().to_dict('records')}")
        merged = merged.merge(problems[problem_columns], on="problem_id", how="left", validate="many_to_one")
        merged["comparison"] = comparison
        merged["baseline_algorithm"] = baseline_name
        merged["challenger_algorithm"] = challenger_name
        merged["pair_id"] = merged.apply(lambda row: _pair_id(row, comparison), axis=1)
        merged["quality_difference"] = merged["quality_final_challenger"] - merged["quality_final_baseline"]
        baseline_objective = 1.0 / merged["quality_final_baseline"].clip(lower=1e-300) - 1.0
        challenger_objective = 1.0 / merged["quality_final_challenger"].clip(lower=1e-300) - 1.0
        merged["objective_final_baseline"] = baseline_objective
        merged["objective_final_challenger"] = challenger_objective
        merged["objective_ratio"] = challenger_objective / baseline_objective.clip(lower=1e-15)
        merged["log10_objective_gain"] = np.log10(
            (baseline_objective + 1e-15) / (challenger_objective + 1e-15)
        )
        merged["target_difference"] = (
            merged["target_reached_challenger"].astype(int) - merged["target_reached_baseline"].astype(int)
        )
        merged["runtime_ratio"] = merged["runtime_challenger"] / merged["runtime_baseline"].clip(lower=1e-12)
        merged["winner"] = np.where(
            merged["quality_difference"] > 1e-12,
            "challenger",
            np.where(merged["quality_difference"] < -1e-12, "baseline", "tie"),
        )
        merged["oracle_quality"] = merged[["quality_final_baseline", "quality_final_challenger"]].max(axis=1)
        merged["baseline_oracle_regret"] = merged["oracle_quality"] - merged["quality_final_baseline"]
        merged["challenger_oracle_regret"] = merged["oracle_quality"] - merged["quality_final_challenger"]
        selected_columns = [
            "pair_id",
            "comparison",
            *key_columns,
            "problem_family",
            "domain",
            "dimension",
            "condition_number",
            "multimodality",
            "baseline_algorithm",
            "challenger_algorithm",
            "run_id_baseline",
            "run_id_challenger",
            "quality_final_baseline",
            "quality_final_challenger",
            "quality_difference",
            "objective_final_baseline",
            "objective_final_challenger",
            "objective_ratio",
            "log10_objective_gain",
            "target_reached_baseline",
            "target_reached_challenger",
            "target_difference",
            "runtime_baseline",
            "runtime_challenger",
            "runtime_ratio",
            "oracle_quality",
            "baseline_oracle_regret",
            "challenger_oracle_regret",
            "winner",
        ]
        pair_frames.append(merged[selected_columns])

    pairs = pd.concat(pair_frames, ignore_index=True).sort_values(["comparison", "problem_id", "seed"])
    problem_summary = (
        pairs.groupby(
            [
                "comparison",
                "problem_id",
                "problem_family",
                "domain",
                "dimension",
                "baseline_algorithm",
                "challenger_algorithm",
            ],
            as_index=False,
        )
        .agg(
            seed_pairs=("pair_id", "size"),
            quality_difference_mean=("quality_difference", "mean"),
            quality_difference_median=("quality_difference", "median"),
            challenger_win_rate=("winner", lambda values: float(np.mean(values == "challenger"))),
            target_difference_mean=("target_difference", "mean"),
            runtime_ratio_median=("runtime_ratio", "median"),
            log10_objective_gain_mean=("log10_objective_gain", "mean"),
            baseline_oracle_regret_mean=("baseline_oracle_regret", "mean"),
            challenger_oracle_regret_mean=("challenger_oracle_regret", "mean"),
        )
        .sort_values(["comparison", "problem_id"])
    )

    for comparison_index, (comparison, (baseline_name, challenger_name)) in enumerate(COMPARISONS.items()):
        pair_group = pairs[pairs["comparison"] == comparison]
        problem_group = problem_summary[problem_summary["comparison"] == comparison]
        differences = problem_group["quality_difference_mean"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(
            differences,
            bootstrap_seed + comparison_index,
            bootstrap_iterations,
        )
        if differences.size and np.any(np.abs(differences) > 1e-15):
            wilcoxon = stats.wilcoxon(differences, alternative="greater", zero_method="wilcox")
            statistic, p_value = float(wilcoxon.statistic), float(wilcoxon.pvalue)
        else:
            statistic, p_value = 0.0, 1.0
        family_means = {
            family: float(group["quality_difference_mean"].mean())
            for family, group in problem_group.groupby("problem_family", sort=True)
        }
        win_rate = float(np.mean(pair_group["winner"] == "challenger"))
        criteria = {
            "problem_count_at_least_8": len(problem_group) >= 8,
            "mean_quality_difference_positive": float(np.mean(differences)) > 0.0,
            "bootstrap_lower_above_zero": ci_low > 0.0,
            "wilcoxon_one_sided_p_le_0_05": p_value <= 0.05,
            "challenger_win_rate_at_least_0_60": win_rate >= 0.60,
            "all_family_means_positive": all(value > 0.0 for value in family_means.values()),
        }
        if all(criteria.values()):
            decision = "MECHANISM_SIGNAL"
        elif criteria["mean_quality_difference_positive"] and win_rate > 0.50:
            decision = "MIXED"
        else:
            decision = "NO_SIGNAL"
        summaries[comparison] = {
            "baseline_algorithm": baseline_name,
            "challenger_algorithm": challenger_name,
            "pair_count": int(len(pair_group)),
            "problem_count": int(len(problem_group)),
            "mean_quality_difference_problem_weighted": float(np.mean(differences)),
            "median_quality_difference_problem_weighted": float(np.median(differences)),
            "bootstrap_95_ci": [ci_low, ci_high],
            "wilcoxon_statistic": statistic,
            "wilcoxon_one_sided_p": p_value,
            "challenger_win_rate_seed_pair": win_rate,
            "tie_rate_seed_pair": float(np.mean(pair_group["winner"] == "tie")),
            "target_rate_difference_seed_pair": float(pair_group["target_difference"].mean()),
            "runtime_ratio_median_seed_pair": float(pair_group["runtime_ratio"].median()),
            "objective_ratio_median_seed_pair": float(pair_group["objective_ratio"].median()),
            "log10_objective_gain_mean_problem_weighted": float(
                problem_group["log10_objective_gain_mean"].mean()
            ),
            "baseline_oracle_regret_mean": float(pair_group["baseline_oracle_regret"].mean()),
            "challenger_oracle_regret_mean": float(pair_group["challenger_oracle_regret"].mean()),
            "family_mean_quality_difference": family_means,
            "criteria": criteria,
            "decision": decision,
        }
    return pairs, problem_summary, summaries


def _configuration(
    jobs: list[JobSpec], parameters: dict[str, Any], preregistration: dict[str, Any]
) -> dict[str, Any]:
    job_ids = sorted(job.run_id for job in jobs)
    return {
        "phase": 2,
        "parameters": parameters,
        "comparisons": COMPARISONS,
        "paired_seed_policy": "same problem_id, seed, budget_type, and budget within each comparison",
        "preregistration_sha256": hashlib.sha256(
            canonical_json(preregistration).encode("utf-8")
        ).hexdigest(),
        "job_count": len(jobs),
        "job_ids_sha256": hashlib.sha256("\n".join(job_ids).encode("utf-8")).hexdigest(),
        "git_commit": current_git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def run_phase2(
    run_directory: str | Path,
    *,
    instances_per_family: int = 8,
    seed_replicates: int = 10,
    master_seed: int = 20_260_820,
    sampling_budget: int = 4_096,
    optimization_budget: int = 4_096,
    bootstrap_iterations: int = 10_000,
    workers: str | int = "auto",
    resume: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    run_path = Path(run_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    parameters = {
        "instances_per_family": instances_per_family,
        "seed_replicates": seed_replicates,
        "master_seed": master_seed,
        "sampling_budget": sampling_budget,
        "optimization_budget": optimization_budget,
    }
    jobs = build_phase2_jobs(**parameters)
    preregistration = _preregistration(parameters, bootstrap_iterations)
    configuration = _configuration(jobs, parameters, preregistration)
    config_path = run_path / "config.json"
    preregistration_path = run_path / "preregistration.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing["job_ids_sha256"] != configuration["job_ids_sha256"]:
            raise ValueError("resume configuration mismatch: job_ids_sha256")
        if existing["preregistration_sha256"] != configuration["preregistration_sha256"]:
            raise ValueError("resume configuration mismatch: preregistration_sha256")
    else:
        config_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
        preregistration_path.write_text(
            json.dumps(preregistration, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    records = run_jobs(jobs, run_path, workers=workers, resume=resume, logger=logger)
    paths = write_dataset(run_path, jobs, records)
    validation = validate_dataset(run_path)
    runs = pd.read_parquet(paths["runs"])
    problems = pd.read_parquet(paths["problems"])
    pairs, problem_summary, comparisons = build_paired_comparisons(
        runs,
        problems,
        bootstrap_seed=master_seed,
        bootstrap_iterations=bootstrap_iterations,
    )
    comparison_directory = run_path / "data/comparisons"
    comparison_directory.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(comparison_directory / "paired_runs.parquet", index=False)
    problem_summary.to_parquet(comparison_directory / "paired_problem_summary.parquet", index=False)

    expected_pairs = len(jobs) // 2
    numeric_pair_columns = [
        "quality_final_baseline",
        "quality_final_challenger",
        "quality_difference",
        "objective_ratio",
        "log10_objective_gain",
        "runtime_ratio",
    ]
    phase2_checks = {
        "data_validation_pass": validation["status"] == "PASS",
        "four_algorithms_present": set(runs["algorithm"]) == {
            "monte_carlo_iid",
            "rqmc_sobol",
            "random_search",
            "cma_es",
        },
        "three_random_mechanisms_present": len(set(runs["random_mechanism"])) == 3,
        "complete_pair_coverage": len(pairs) == expected_pairs and pairs["pair_id"].is_unique,
        "all_pair_numeric_values_finite": bool(
            np.isfinite(pairs[numeric_pair_columns].to_numpy(dtype=float)).all()
        ),
        "all_problem_seed_replicates_complete": bool(
            (problem_summary["seed_pairs"] == seed_replicates).all()
            and len(problem_summary) == validation["problem_count"]
        ),
        "no_execution_failures": int(runs["failure"].sum()) == 0,
    }
    phase2_status = "PHASE_2_PASS" if all(phase2_checks.values()) else "PHASE_2_FAIL"
    summary = {
        "status": phase2_status,
        "scope": "Phase 2 paired random-mechanism comparison",
        "executed_at": datetime.now(_timezone()).isoformat(timespec="seconds"),
        "elapsed_seconds": perf_counter() - started_at,
        "problem_count": validation["problem_count"],
        "run_count": validation["run_count"],
        "pair_count": int(len(pairs)),
        "trace_count": validation["trace_count"],
        "trajectory_feature_count": validation["trajectory_feature_count"],
        "failure_count": validation["failure_count"],
        "algorithms": sorted(runs["algorithm"].unique().tolist()),
        "random_mechanisms": sorted(runs["random_mechanism"].unique().tolist()),
        "phase2_checks": phase2_checks,
        "utility_definition": {
            "primary": "paired quality difference on the same problem/seed/budget",
            "quality": "1 / (1 + objective)",
            "selection_diagnostic": "two-algorithm oracle regret within each pair",
            "inference_unit": "problem-instance mean across seeds",
        },
        "comparison_results": comparisons,
        "configuration": configuration,
    }
    (run_path / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_path / "comparison_summary.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
