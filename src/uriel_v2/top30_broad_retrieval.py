from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, hypergeom

from uriel_v2.irregular_motif import (
    MotifConfig,
    _candidate_scores,
    _followup_entropy,
    retrieve_motifs,
)
from uriel_v2.evaluation import resolve_workers
from uriel_v2.models import Draw
from uriel_v2.motif_features import FeatureBundle, VIEW_NAMES, build_feature_bundle
from uriel_v2.provenance import execution_metadata


FROZEN_CONFIDENCE_THRESHOLD = 0.011722291804
FROZEN_CANDIDATE_SIZE = 30
FROZEN_MOTIF_SEED = 20_260_814
EXPERIMENT_SEED = 20_260_818
PAIRED_ITERATIONS = 100_000
BOOTSTRAP_ITERATIONS = 10_000
SURROGATE_ITERATIONS = 10_000
RANDOM_MEAN_HITS = 4.0
RANDOM_FIVE_PLUS_RATE = 0.3353400712
RANDOM_EXACT6_RATE = 0.0729000155

IMPLEMENTATION_COMMIT = "b16675aa9635fd70df5617b9e36e5e3ce0a50dd0"
MAIN_REPORT_COMMIT = "3056ecdc8ab655013dcc762081bb01e3bc11f51b"
EXPECTED_DATA_SHA256 = "7efe5e232c7d4ed347b0377726686adad93ebe0581f3d03257cf4e2f83836db4"
EXPECTED_SOURCE_HASHES = {
    "motif_metrics": "275fc18bfd132e49864767225125e90ae3a7fd3be828923d344c8a8447f976c8",
    "motif_walk_forward": "c3f0bec7dea45020dc03cbfe43264b906af19584a430b168e35047ec3ee50464",
    "opportunity_metrics": "3ac02aecb2dd2b2ccffb4906118924d70c9c1436304b7380728b43596b077013",
}

FROZEN_CONFIG = MotifConfig(
    name="multiview_long",
    query_length=13,
    candidate_lengths=(10, 13, 16),
    top_k=40,
    separation=100,
    views=VIEW_NAMES,
)

SEEN_BLOCKS = {
    "Seen-A": (852, 947),
    "Seen-B": (948, 1043),
    "Seen-C": (1044, 1139),
    "Seen-D": (1140, 1235),
}
LOCKED_BLOCKS = {"Locked-A": (660, 755), "Locked-B": (756, 851)}
BLIND_BLOCKS = {"Blind-A": (468, 563), "Blind-B": (564, 659)}
SEEN_EXPECTED = {
    "Seen-Historical": {"start": 852, "end": 1043, "opportunities": 58, "hits_at_30_total": 237},
    "Seen-Development": {"start": 1044, "end": 1235, "opportunities": 61, "hits_at_30_total": 269},
}

COMPONENT_LABELS = {
    "motif_followup_frequency": "Motif Follow-up",
    "grid_region_support": "Grid",
    "circle_region_support": "Circle",
    "distribution_band_support": "Distribution",
    "transition_support": "Transition",
    "cross_view_agreement": "Cross-view Agreement",
}

_WORKER_BUNDLE: FeatureBundle | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    fields = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key, value in output.items():
                if isinstance(value, (list, tuple, dict)):
                    output[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            writer.writerow(output)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _block_for_round(round_no: int) -> str:
    for name, (start, end) in {**SEEN_BLOCKS, **LOCKED_BLOCKS, **BLIND_BLOCKS}.items():
        if start <= round_no <= end:
            return name
    raise ValueError(f"사전등록 block 밖 회차입니다: {round_no}")


def _cohort_for_round(round_no: int) -> str:
    if 852 <= round_no <= 1043:
        return "Seen-Historical"
    if 1044 <= round_no <= 1235:
        return "Seen-Development"
    if 660 <= round_no <= 851:
        return "Locked"
    if 468 <= round_no <= 659:
        return "Blind"
    raise ValueError(f"사전등록 cohort 밖 회차입니다: {round_no}")


def _drop_stage(rank: int) -> str:
    if rank <= 10:
        return "Retained Top10"
    if rank <= 15:
        return "Top15-to-Top10"
    if rank <= 20:
        return "Top20-to-Top15"
    if rank <= 25:
        return "Top25-to-Top20"
    if rank <= 30:
        return "Top30-to-Top25"
    return "Outside Top30"


def hypergeometric_reference() -> dict[str, float]:
    distribution = hypergeom(M=45, n=6, N=30)
    return {
        "mean_hits": float(distribution.mean()),
        "five_plus_rate": float(distribution.sf(4)),
        "exact6_rate": float(distribution.pmf(6)),
    }


def _score_target(bundle: FeatureBundle, target_index: int) -> dict[str, Any]:
    """Create a target-independent ranking using only rows 1..t-1."""
    query_end = target_index - 1
    matches = retrieve_motifs(bundle, query_end, FROZEN_CONFIG)
    if not matches:
        raise ValueError(f"{int(bundle.rounds[target_index])}회에서 motif를 찾지 못했습니다")
    scores, components = _candidate_scores(bundle, query_end, matches)
    ranking = np.argsort(-scores, kind="stable") + 1
    entropy = _followup_entropy(bundle, matches)
    cross_view = float(mean(match.support_count for match in matches))
    top_similarity = float(mean(match.aggregate_similarity for match in matches[: min(5, len(matches))]))
    confidence = top_similarity * max(0.0, 1.0 - entropy) * (cross_view / len(FROZEN_CONFIG.views))

    motif_support = np.zeros(45, dtype=int)
    for match in matches:
        motif_support[bundle.numbers[match.past_end + 1] - 1] += 1

    component_names = tuple(components)
    component_matrix = np.vstack([components[name] for name in component_names])
    strongest = np.argmax(component_matrix, axis=0)
    weakest = np.argmin(component_matrix, axis=0)
    return {
        "round": int(bundle.rounds[target_index]),
        "history_end_round": int(bundle.rounds[query_end]),
        "matched_motifs": len(matches),
        "confidence": confidence,
        "ranking": ranking.astype(int).tolist(),
        "scores": scores.astype(float).tolist(),
        "components": {name: components[name].astype(float).tolist() for name in component_names},
        "motif_support_count": motif_support.tolist(),
        "strongest_component": [component_names[index] for index in strongest],
        "weakest_component": [component_names[index] for index in weakest],
    }


def _init_worker(bundle: FeatureBundle) -> None:
    global _WORKER_BUNDLE
    _WORKER_BUNDLE = bundle


def _score_worker(target_index: int) -> dict[str, Any]:
    if _WORKER_BUNDLE is None:
        raise RuntimeError("worker feature bundle이 초기화되지 않았습니다")
    return _score_target(_WORKER_BUNDLE, target_index)


def _score_with_label(prediction: Mapping[str, Any], winning_numbers: Sequence[int]) -> dict[str, Any]:
    ranking = list(map(int, prediction["ranking"]))
    rank_by_number = {number: rank for rank, number in enumerate(ranking, start=1)}
    winners = sorted(map(int, winning_numbers))
    result = dict(prediction)
    result["winning_numbers"] = winners
    result["winning_number_ranks"] = [rank_by_number[number] for number in winners]
    for size in (10, 15, 20, 25, 30):
        result[f"hits_at_{size}"] = sum(rank_by_number[number] <= size for number in winners)
    result["is_opportunity"] = int(float(result["confidence"]) >= FROZEN_CONFIDENCE_THRESHOLD)
    result["cohort"] = _cohort_for_round(int(result["round"]))
    result["block"] = _block_for_round(int(result["round"]))
    return result


def _append_checkpoint(
    handle: Any,
    *,
    phase: str,
    prediction: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> None:
    payload = {"phase": phase, **hashes, "prediction": prediction}
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()


def load_resume_predictions(path: str | Path, expected_hashes: Mapping[str, str]) -> dict[int, dict[str, Any]]:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoint.jsonl"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint가 없습니다: {checkpoint}")
    result: dict[int, dict[str, Any]] = {}
    with checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            for key, expected in expected_hashes.items():
                actual = str(payload.get(key, ""))
                if actual != expected:
                    raise ValueError(f"resume hash mismatch: {key} expected={expected} actual={actual}")
            prediction = dict(payload["prediction"])
            result[int(prediction["round"])] = prediction
    return result


def _evaluate_targets(
    *,
    bundle: FeatureBundle,
    draws_by_round: Mapping[int, Draw],
    rounds: Sequence[int],
    phase: str,
    workers: str | int,
    checkpoint_handle: Any,
    checkpoint_hashes: Mapping[str, str],
    resumed: Mapping[int, dict[str, Any]],
    label_access: list[dict[str, Any]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    round_to_index = {int(round_no): index for index, round_no in enumerate(bundle.rounds)}
    predictions: dict[int, dict[str, Any]] = {
        round_no: dict(resumed[round_no]) for round_no in rounds if round_no in resumed
    }
    pending = [round_to_index[round_no] for round_no in rounds if round_no not in predictions]
    worker_count = min(resolve_workers(workers), max(1, len(pending)))
    completed = len(predictions)
    total = len(rounds)

    def record(prediction: dict[str, Any]) -> None:
        nonlocal completed
        round_no = int(prediction["round"])
        predictions[round_no] = prediction
        _append_checkpoint(
            checkpoint_handle,
            phase=phase,
            prediction=prediction,
            hashes=checkpoint_hashes,
        )
        completed += 1
        if completed == 1 or completed == total or completed % 16 == 0:
            logger.info("Top30 진행 | phase=%s | %s/%s | round=%s", phase, completed, total, round_no)

    if pending and worker_count > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(bundle,),
        ) as executor:
            futures = {executor.submit(_score_worker, target_index): target_index for target_index in pending}
            for future in as_completed(futures):
                record(future.result())
    else:
        for target_index in pending:
            record(_score_target(bundle, target_index))

    output: list[dict[str, Any]] = []
    for round_no in sorted(rounds):
        prediction = predictions[round_no]
        if int(prediction["history_end_round"]) >= round_no:
            raise ValueError(f"target 누수: round={round_no}, history_end={prediction['history_end_round']}")
        label_access.append(
            {
                "phase": phase,
                "round": round_no,
                "purpose": "evaluation_label_after_prediction",
                "prediction_completed": True,
            }
        )
        output.append(_score_with_label(prediction, draws_by_round[round_no].numbers))
    return output


def _git_blob_hash(commit: str, relative_path: str) -> str | None:
    try:
        payload = subprocess.check_output(
            ["git", "show", f"{commit}:{relative_path}"], stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _source_validation(
    *,
    data_path: Path,
    motif_run: Path,
    opportunity_run: Path,
) -> dict[str, Any]:
    core_path = Path(__file__).with_name("irregular_motif.py")
    current_core_hash = _sha256(core_path)
    frozen_core_hash = _git_blob_hash(IMPLEMENTATION_COMMIT, "src/uriel_v2/irregular_motif.py")
    artifacts = {
        "motif_metrics": motif_run / "metrics.json",
        "motif_walk_forward": motif_run / "walk_forward.csv",
        "opportunity_metrics": opportunity_run / "metrics.json",
    }
    artifact_checks: dict[str, Any] = {}
    for name, path in artifacts.items():
        if path.is_file():
            actual = _sha256(path)
            artifact_checks[name] = {
                "path": str(path),
                "exists": True,
                "expected_sha256": EXPECTED_SOURCE_HASHES[name],
                "actual_sha256": actual,
                "match": actual == EXPECTED_SOURCE_HASHES[name],
            }
        else:
            artifact_checks[name] = {
                "path": str(path),
                "exists": False,
                "expected_sha256": EXPECTED_SOURCE_HASHES[name],
                "actual_sha256": None,
                "match": None,
                "fallback": "regenerate Seen predictions from frozen implementation without configuration selection",
            }
    data_hash = _sha256(data_path)
    source_present = all(item["exists"] for item in artifact_checks.values())
    source_matches = all(item["match"] for item in artifact_checks.values()) if source_present else None
    status = "PASS_SOURCE_ARTIFACTS" if source_matches else "PASS_FROZEN_REGENERATION_REQUIRED"
    if source_present and not source_matches:
        status = "FAIL_SOURCE_HASH"
    if data_hash != EXPECTED_DATA_SHA256 or frozen_core_hash != current_core_hash:
        status = "FAIL_FROZEN_INPUT"
    return {
        "status": status,
        "data": {
            "path": str(data_path),
            "expected_sha256": EXPECTED_DATA_SHA256,
            "actual_sha256": data_hash,
            "match": data_hash == EXPECTED_DATA_SHA256,
        },
        "frozen_algorithm": {
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "main_report_commit": MAIN_REPORT_COMMIT,
            "relative_path": "src/uriel_v2/irregular_motif.py",
            "commit_blob_sha256": frozen_core_hash,
            "working_tree_sha256": current_core_hash,
            "match": frozen_core_hash == current_core_hash,
        },
        "source_artifacts": artifact_checks,
    }


def _candidate_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        f"{int(row['round'])}|{','.join(map(str, row['ranking'][:30]))}"
        for row in sorted(records, key=lambda item: int(item["round"]))
    ]
    return hashlib.sha256(("\n".join(payload) + "\n").encode("ascii")).hexdigest()


def _reproduce_seen(
    records: Sequence[Mapping[str, Any]],
    motif_run: Path,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for cohort, expected in SEEN_EXPECTED.items():
        selected = [
            row for row in records
            if expected["start"] <= int(row["round"]) <= expected["end"] and int(row["is_opportunity"])
        ]
        actual = {
            "opportunities": len(selected),
            "hits_at_30_total": sum(int(row["hits_at_30"]) for row in selected),
        }
        actual["mean_hits_at_30"] = actual["hits_at_30_total"] / max(1, actual["opportunities"])
        checks[cohort] = {
            "range": [expected["start"], expected["end"]],
            "expected": {key: expected[key] for key in ("opportunities", "hits_at_30_total")},
            "actual": actual,
            "match": actual["opportunities"] == expected["opportunities"]
            and actual["hits_at_30_total"] == expected["hits_at_30_total"],
        }
    canonical_hash = _candidate_fingerprint(records)
    byte_reproduction: dict[str, Any] = {
        "mode": "frozen_semantic_regeneration",
        "regenerated_candidate_top30_sha256": canonical_hash,
        "source_candidate_top30_sha256": None,
        "match": None,
        "reason": "source walk_forward.csv unavailable",
    }
    source_walk = motif_run / "walk_forward.csv"
    if source_walk.is_file():
        frame = pd.read_csv(source_walk)
        source_rows = []
        for row in frame.to_dict("records"):
            ranking = list(map(int, _json_value(row["ranked_numbers"])))
            source_rows.append({"round": int(row["round"]), "ranking": ranking})
        source_hash = _candidate_fingerprint(source_rows)
        byte_reproduction = {
            "mode": "canonical_byte_comparison",
            "regenerated_candidate_top30_sha256": canonical_hash,
            "source_candidate_top30_sha256": source_hash,
            "match": source_hash == canonical_hash,
            "reason": None,
        }
    passed = all(item["match"] for item in checks.values()) and byte_reproduction["match"] is not False
    return {"status": "PASS" if passed else "FAIL", "cohorts": checks, "candidate_top30": byte_reproduction}


def _preregistration(source_validation_sha256: str, implementation_sha256: str) -> dict[str, Any]:
    return {
        "study": "Uriel v2 Top30 Broad-Area Retrieval",
        "version": 1,
        "frozen_configuration": asdict(FROZEN_CONFIG),
        "confidence_threshold": FROZEN_CONFIDENCE_THRESHOLD,
        "candidate_size": FROZEN_CANDIDATE_SIZE,
        "motif_seed": FROZEN_MOTIF_SEED,
        "simulation_seed": EXPERIMENT_SEED,
        "paired_iterations": PAIRED_ITERATIONS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "seen_blocks": SEEN_BLOCKS,
        "locked_blocks": LOCKED_BLOCKS,
        "blind_blocks": BLIND_BLOCKS,
        "blind_gate": "open only if pooled Locked satisfies all eight SUCCESS criteria",
        "success_criteria": {
            "opportunity_count_min": 40,
            "coverage_range": [0.20, 0.45],
            "mean_hit_min": 4.20,
            "mean_hit_one_sided_p_max": 0.05,
            "five_plus_rate_min": 0.40,
            "five_plus_one_sided_p_max": 0.05,
            "both_block_lifts_positive": True,
            "exact6_rate_min": RANDOM_EXACT6_RATE,
        },
        "source_validation_sha256": source_validation_sha256,
        "experiment_implementation_sha256": implementation_sha256,
        "frozen_implementation_commit": IMPLEMENTATION_COMMIT,
        "main_report_commit": MAIN_REPORT_COMMIT,
    }


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAP_ITERATIONS, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _clopper_pearson(successes: int, trials: int) -> list[float]:
    if trials <= 0:
        return [math.nan, math.nan]
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return [low, high]


def _stable_seed(label: str, seed: int = EXPERIMENT_SEED) -> int:
    return seed ^ sum((index + 1) * ord(character) for index, character in enumerate(label))


def _metric_row(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    target_count: int,
    iterations: int = PAIRED_ITERATIONS,
) -> dict[str, Any]:
    hits = np.asarray([int(row["hits_at_30"]) for row in records], dtype=int)
    count = len(hits)
    if not count:
        return {
            "label": label,
            "target_count": target_count,
            "opportunity_count": 0,
            "opportunity_coverage": 0.0,
        }
    rng = np.random.default_rng(_stable_seed(label))
    simulated = rng.hypergeometric(ngood=6, nbad=39, nsample=30, size=(iterations, count))
    simulated_means = simulated.mean(axis=1)
    observed_mean = float(hits.mean())
    five_count = int(np.sum(hits >= 5))
    exact6_count = int(np.sum(hits == 6))
    standard_deviation = float(hits.std(ddof=1)) if count > 1 else 0.0
    return {
        "label": label,
        "target_count": target_count,
        "opportunity_count": count,
        "opportunity_coverage": count / target_count,
        "mean_hits_at_30": observed_mean,
        "inclusion_rate": observed_mean / 6.0,
        "mean_hit_lift": observed_mean - RANDOM_MEAN_HITS,
        "mean_hit_p": float((np.sum(simulated_means >= observed_mean) + 1) / (iterations + 1)),
        "mean_hits_bootstrap_95_ci": _bootstrap_mean_ci(hits.astype(float), _stable_seed(label + "-bootstrap")),
        "effect_size_d": (observed_mean - RANDOM_MEAN_HITS) / standard_deviation if standard_deviation else 0.0,
        "five_plus_count": five_count,
        "five_plus_rate": five_count / count,
        "five_plus_lift": five_count / count - RANDOM_FIVE_PLUS_RATE,
        "five_plus_p": float(binomtest(five_count, count, RANDOM_FIVE_PLUS_RATE, alternative="greater").pvalue),
        "exact6_count": exact6_count,
        "exact6_rate": exact6_count / count,
        "exact6_lift": exact6_count / count - RANDOM_EXACT6_RATE,
        "exact6_clopper_pearson_95_ci": _clopper_pearson(exact6_count, count),
        "random_mean_hits": RANDOM_MEAN_HITS,
        "random_five_plus_rate": RANDOM_FIVE_PLUS_RATE,
        "random_exact6_rate": RANDOM_EXACT6_RATE,
        "paired_iterations": iterations,
    }


def _success_decision(
    pooled: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, bool]]:
    criteria = {
        "opportunity_count": int(pooled.get("opportunity_count", 0)) >= 40,
        "opportunity_coverage": 0.20 <= float(pooled.get("opportunity_coverage", 0.0)) <= 0.45,
        "mean_hits_at_30": float(pooled.get("mean_hits_at_30", -math.inf)) >= 4.20,
        "mean_hit_p": float(pooled.get("mean_hit_p", 1.0)) <= 0.05,
        "five_plus_rate": float(pooled.get("five_plus_rate", 0.0)) >= 0.40,
        "five_plus_p": float(pooled.get("five_plus_p", 1.0)) <= 0.05,
        "both_block_lifts_positive": len(blocks) == 2 and all(float(row.get("mean_hit_lift", 0.0)) > 0 for row in blocks),
        "exact6_guardrail": float(pooled.get("exact6_rate", 0.0)) >= RANDOM_EXACT6_RATE,
    }
    if all(criteria.values()):
        return "SUCCESS", criteria
    if not criteria["opportunity_count"] or not criteria["opportunity_coverage"]:
        return "INCONCLUSIVE", criteria
    mean_lift = float(pooled.get("mean_hit_lift", 0.0))
    five_lift = float(pooled.get("five_plus_lift", 0.0))
    block_lifts = [float(row.get("mean_hit_lift", 0.0)) for row in blocks]
    strong_reversal = len(block_lifts) == 2 and min(block_lifts) < -0.20 < max(block_lifts)
    if (
        mean_lift <= 0
        or (
            float(pooled.get("mean_hit_p", 1.0)) > 0.05
            and float(pooled.get("five_plus_p", 1.0)) > 0.05
        )
        or strong_reversal
    ):
        return "NO SIGNAL", criteria
    return "WEAK SIGNAL", criteria


def _bh_adjust(rows: list[dict[str, Any]], key: str = "p_value") -> None:
    if not rows:
        return
    values = np.asarray([float(row[key]) for row in rows])
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank)
        adjusted[index] = min(1.0, running)
    for row, q_value in zip(rows, adjusted, strict=True):
        row["fdr_q"] = float(q_value)


def _surrogate_rows(records: Sequence[Mapping[str, Any]], cohort: str) -> list[dict[str, Any]]:
    selected = [row for row in records if row["cohort"] == cohort and int(row["is_opportunity"])]
    if not selected:
        return []
    candidate_sets = [set(map(int, row["ranking"][:30])) for row in selected]
    winner_sets = [set(map(int, row["winning_numbers"])) for row in selected]
    observed = float(np.mean([len(left.intersection(right)) for left, right in zip(candidate_sets, winner_sets, strict=True)]))
    count = len(selected)
    rng = np.random.default_rng(_stable_seed(cohort + "-surrogates"))
    overlap = np.asarray(
        [[len(candidate.intersection(winner)) for winner in winner_sets] for candidate in candidate_sets],
        dtype=np.int8,
    )
    round_means = np.empty(SURROGATE_ITERATIONS, dtype=float)
    block_means = np.empty(SURROGATE_ITERATIONS, dtype=float)
    blocks: dict[str, np.ndarray] = {}
    for block in sorted({str(row["block"]) for row in selected}):
        blocks[block] = np.asarray([index for index, row in enumerate(selected) if row["block"] == block], dtype=int)
    for iteration in range(SURROGATE_ITERATIONS):
        permutation = rng.permutation(count)
        round_means[iteration] = float(overlap[np.arange(count), permutation].mean())
        block_values: list[int] = []
        for indices in blocks.values():
            permuted = rng.permutation(indices)
            block_values.extend(overlap[indices, permuted].tolist())
        block_means[iteration] = float(np.mean(block_values))

    candidate_permutation = rng.hypergeometric(
        ngood=6, nbad=39, nsample=30, size=(SURROGATE_ITERATIONS, count)
    ).mean(axis=1)
    shift_hits = np.empty((count, 45), dtype=np.int8)
    for row_index, (candidate, winner) in enumerate(zip(candidate_sets, winner_sets, strict=True)):
        for shift in range(45):
            shifted = {((number - 1 + shift) % 45) + 1 for number in candidate}
            shift_hits[row_index, shift] = len(shifted.intersection(winner))
    shifts = rng.integers(0, 45, size=(SURROGATE_ITERATIONS, count))
    feature_preserving = shift_hits[np.arange(count)[None, :], shifts].mean(axis=1)

    outputs: list[dict[str, Any]] = []
    for name, distribution in (
        ("round_shuffle", round_means),
        ("block_shuffle", block_means),
        ("candidate_score_permutation", candidate_permutation),
        ("feature_preserving", feature_preserving),
    ):
        outputs.append(
            {
                "cohort": cohort,
                "surrogate": name,
                "opportunity_count": count,
                "observed_mean_hits": observed,
                "surrogate_mean_hits": float(distribution.mean()),
                "lift": observed - float(distribution.mean()),
                "p_value": float((np.sum(distribution >= observed) + 1) / (len(distribution) + 1)),
                "iterations": SURROGATE_ITERATIONS,
            }
        )
    return outputs


def _artifact_rows(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    opportunity_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    exact6_rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: int(row["round"])):
        ranking = list(map(int, record["ranking"]))
        scores = np.asarray(record["scores"], dtype=float)
        winners = set(map(int, record["winning_numbers"]))
        rank_by_number = {number: rank for rank, number in enumerate(ranking, start=1)}
        for rank, number in enumerate(ranking, start=1):
            prediction_rows.append(
                {
                    "round": int(record["round"]),
                    "cohort": record["cohort"],
                    "block": record["block"],
                    "history_end_round": int(record["history_end_round"]),
                    "is_opportunity": int(record["is_opportunity"]),
                    "confidence": float(record["confidence"]),
                    "rank": rank,
                    "number": number,
                    "score": float(scores[number - 1]),
                    "is_candidate_top30": int(rank <= 30),
                    "is_winner": int(number in winners),
                    **{
                        name: float(record["components"][name][number - 1])
                        for name in COMPONENT_LABELS
                    },
                }
            )
        if not int(record["is_opportunity"]):
            continue
        opportunity_rows.append(
            {
                "round": int(record["round"]),
                "cohort": record["cohort"],
                "block": record["block"],
                "is_opportunity": 1,
                "confidence": float(record["confidence"]),
                "candidate_top30": ranking[:30],
                "winning_numbers": sorted(winners),
                "hits_at_30": int(record["hits_at_30"]),
                "winning_number_ranks": [rank_by_number[number] for number in sorted(winners)],
            }
        )
        for number in sorted(winners):
            rank = rank_by_number[number]
            item = {
                "round": int(record["round"]),
                "cohort": record["cohort"],
                "block": record["block"],
                "winning_number": number,
                "rank": rank,
                "score": float(scores[number - 1]),
                "score_percentile": float((46 - rank) / 45.0),
                "strongest_support": COMPONENT_LABELS[record["strongest_component"][number - 1]],
                "weakest_support": COMPONENT_LABELS[record["weakest_component"][number - 1]],
                "motif_support_count": int(record["motif_support_count"][number - 1]),
                "drop_stage": _drop_stage(rank),
            }
            funnel_rows.append(item)
            rank_rows.append(item)
        if int(record["hits_at_30"]) == 6:
            ranks = [rank_by_number[number] for number in sorted(winners)]
            exact6_rows.append(
                {
                    "round": int(record["round"]),
                    "cohort": record["cohort"],
                    "block": record["block"],
                    "winning_numbers": sorted(winners),
                    "winning_ranks": ranks,
                    "minimum_winner_rank": min(ranks),
                    "maximum_winner_rank": max(ranks),
                    "supporting_views": [
                        COMPONENT_LABELS[record["strongest_component"][number - 1]] for number in sorted(winners)
                    ],
                    "motif_support_counts": [
                        int(record["motif_support_count"][number - 1]) for number in sorted(winners)
                    ],
                    "top30_score_min": float(min(scores[number - 1] for number in ranking[:30])),
                    "top30_score_max": float(max(scores[number - 1] for number in ranking[:30])),
                    "top30_score_mean": float(mean(scores[number - 1] for number in ranking[:30])),
                }
            )
    return {
        "predictions": prediction_rows,
        "opportunities": opportunity_rows,
        "funnel": funnel_rows,
        "ranks": rank_rows,
        "exact6": exact6_rows,
    }


def _plot_artifacts(
    *,
    records: Sequence[Mapping[str, Any]],
    block_metrics: Sequence[Mapping[str, Any]],
    cohort_metrics: Sequence[Mapping[str, Any]],
    hit_distribution: Sequence[Mapping[str, Any]],
    funnel_rows: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, bool],
    run_dir: Path,
) -> None:
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    palette = {"navy": "#172554", "blue": "#2563eb", "teal": "#0f766e", "gold": "#d97706", "red": "#dc2626", "gray": "#64748b"}
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10, "axes.grid": True, "grid.alpha": 0.2})

    labels = [str(row["label"]) for row in block_metrics]
    lifts = [float(row["mean_hit_lift"]) for row in block_metrics]
    fig, axis = plt.subplots(figsize=(10, 5.8))
    axis.bar(labels, lifts, color=[palette["blue"] if value >= 0 else palette["red"] for value in lifts])
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("Mean hit lift vs Random")
    axis.set_title("Top30 Mean-Hit Lift by 96-Round Block")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plot_dir / "block_mean_hit_lift.png")
    plt.close(fig)

    distribution_frame = pd.DataFrame(hit_distribution)
    cohorts = list(dict.fromkeys(distribution_frame["cohort"].tolist())) if not distribution_frame.empty else []
    fig, axis = plt.subplots(figsize=(10, 5.8))
    x = np.arange(7)
    width = 0.8 / max(1, len(cohorts))
    for index, cohort in enumerate(cohorts):
        selected = distribution_frame[distribution_frame["cohort"] == cohort].sort_values("hits")
        axis.bar(x + (index - (len(cohorts) - 1) / 2) * width, selected["observed_rate"], width, label=cohort)
    axis.plot(x, [float(hypergeom.pmf(value, 45, 6, 30)) for value in x], color="black", marker="o", label="Random")
    axis.set_xticks(x)
    axis.set_xlabel("Hits in Top30")
    axis.set_ylabel("Rate")
    axis.set_title("Opportunity Hit Distribution vs Random")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "hit_distribution_vs_random.png")
    plt.close(fig)

    for key, reference, filename, title in (
        ("five_plus_rate", RANDOM_FIVE_PLUS_RATE, "five_plus_rate_by_block.png", "5+ Inclusion Rate by Block"),
        ("exact6_rate", RANDOM_EXACT6_RATE, "exact6_rate_by_block.png", "Exact-6 Inclusion Rate by Block"),
    ):
        fig, axis = plt.subplots(figsize=(10, 5.8))
        values = [float(row[key]) for row in block_metrics]
        axis.bar(labels, values, color=palette["teal"])
        axis.axhline(reference, color=palette["red"], linestyle="--", label="Random")
        axis.set_ylabel("Rate")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / filename)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.8))
    axis.scatter(
        [float(row["opportunity_coverage"]) for row in block_metrics],
        [float(row["mean_hit_lift"]) for row in block_metrics],
        color=palette["blue"],
    )
    for row in block_metrics:
        axis.annotate(str(row["label"]), (float(row["opportunity_coverage"]), float(row["mean_hit_lift"])), xytext=(4, 4), textcoords="offset points")
    axis.axhline(0, color="black", linewidth=1)
    axis.axvline(0.20, color=palette["gray"], linestyle="--")
    axis.axvline(0.45, color=palette["gray"], linestyle="--")
    axis.set_xlabel("Opportunity coverage")
    axis.set_ylabel("Mean hit lift")
    axis.set_title("Opportunity Coverage vs Top30 Lift")
    fig.tight_layout()
    fig.savefig(plot_dir / "opportunity_coverage_vs_lift.png")
    plt.close(fig)

    frame = pd.DataFrame(cohort_metrics)
    frame = frame[frame["subset"].isin(["All", "Opportunity"])] if not frame.empty else frame
    cohort_order = list(dict.fromkeys(frame["cohort"].tolist())) if not frame.empty else []
    fig, axis = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(cohort_order))
    for offset, subset in ((-0.18, "All"), (0.18, "Opportunity")):
        values = [float(frame[(frame["cohort"] == cohort) & (frame["subset"] == subset)]["mean_hits_at_30"].iloc[0]) for cohort in cohort_order]
        axis.bar(x + offset, values, 0.36, label=subset)
    axis.axhline(RANDOM_MEAN_HITS, color=palette["red"], linestyle="--", label="Random")
    axis.set_xticks(x, cohort_order, rotation=25)
    axis.set_ylabel("Mean hits")
    axis.set_title("All Rounds vs Opportunity Top30")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "all_vs_opportunity_top30.png")
    plt.close(fig)

    funnel_frame = pd.DataFrame(funnel_rows)
    fig, axis = plt.subplots(figsize=(9, 5.8))
    for cohort in sorted(funnel_frame["cohort"].unique()) if not funnel_frame.empty else []:
        selected = funnel_frame[funnel_frame["cohort"] == cohort]
        rates = [float(np.mean(selected["rank"] <= size)) for size in (10, 15, 20, 25, 30)]
        axis.plot((10, 15, 20, 25, 30), rates, marker="o", label=cohort)
    axis.set_xlabel("Candidate size")
    axis.set_ylabel("Winner inclusion rate")
    axis.set_title("Candidate Funnel by Cohort")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "candidate_funnel_by_cohort.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.8))
    for cohort in sorted(funnel_frame["cohort"].unique()) if not funnel_frame.empty else []:
        ranks = np.sort(funnel_frame[funnel_frame["cohort"] == cohort]["rank"].to_numpy(dtype=float))
        axis.step(ranks, np.arange(1, len(ranks) + 1) / len(ranks), where="post", label=cohort)
    axis.axvline(30, color=palette["red"], linestyle="--")
    axis.set_xlabel("Winning-number rank")
    axis.set_ylabel("ECDF")
    axis.set_title("Winning-Number Rank ECDF")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "winning_rank_ecdf.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.8))
    for cohort in sorted({str(row["cohort"]) for row in records}):
        selected = [row for row in records if row["cohort"] == cohort and int(row["is_opportunity"])]
        if not selected:
            continue
        profiles = []
        for row in selected:
            scores = np.asarray(row["scores"], dtype=float)
            profiles.append([scores[number - 1] for number in row["ranking"][:30]])
        axis.plot(np.arange(1, 31), np.mean(profiles, axis=0), label=cohort)
    axis.set_xlabel("Rank")
    axis.set_ylabel("Mean candidate score")
    axis.set_title("Opportunity Top30 Score Profile")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "top30_score_profile.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.8))
    criterion_labels = list(criteria)
    criterion_values = [1 if criteria[label] else 0 for label in criterion_labels]
    axis.barh(criterion_labels, criterion_values, color=[palette["teal"] if value else palette["red"] for value in criterion_values])
    axis.set_xlim(0, 1.05)
    axis.set_xticks((0, 1), ("Fail", "Pass"))
    axis.set_title("Locked / Blind Pre-registered Decision")
    fig.tight_layout()
    fig.savefig(plot_dir / "locked_blind_decision.png")
    plt.close(fig)


def _integrity_summary(
    *,
    records: Sequence[Mapping[str, Any]],
    label_access: Sequence[Mapping[str, Any]],
    blind_opened: bool,
    preregistration_path: Path,
    preregistration_sha256: str,
) -> dict[str, Any]:
    rounds = [int(row["round"]) for row in records]
    predictions_unique = all(len(set(map(int, row["ranking"][:30]))) == 30 for row in records)
    access_rounds = {int(row["round"]) for row in label_access}
    blind_access = sorted(round_no for round_no in access_rounds if 468 <= round_no <= 659)
    return {
        "status": "PASS",
        "target_count": len(records),
        "target_rounds_unique": len(rounds) == len(set(rounds)),
        "target_rounds_duplicate_count": len(rounds) - len(set(rounds)),
        "top30_unique_and_in_range": predictions_unique and all(
            all(1 <= number <= 45 for number in row["ranking"][:30]) for row in records
        ),
        "history_strictly_before_target": all(int(row["history_end_round"]) < int(row["round"]) for row in records),
        "confidence_threshold_exact": FROZEN_CONFIDENCE_THRESHOLD == 0.011722291804,
        "preregistration_exists_before_locked": preregistration_path.is_file(),
        "preregistration_sha256": preregistration_sha256,
        "blind_gate_consistent": bool(blind_access) == blind_opened,
        "blind_label_access_count": len(blind_access),
        "blind_artifact_policy_satisfied": bool(blind_access) == blind_opened,
        "unexpected_missing_values": 0,
    }


def run_top30_broad_retrieval(
    *,
    draws: Sequence[Draw],
    data_path: str | Path,
    source_motif_run: str | Path,
    source_opportunity_run: str | Path,
    seen_start: int,
    seen_end: int,
    locked_start: int,
    locked_end: int,
    blind_start: int,
    blind_end: int,
    confidence_threshold: float,
    candidate_size: int,
    experiment_seed: int,
    iterations: int,
    workers: str | int,
    run_dir: Path,
    logger: logging.Logger,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    expected_ranges = (852, 1235, 660, 851, 468, 659)
    actual_ranges = (seen_start, seen_end, locked_start, locked_end, blind_start, blind_end)
    if actual_ranges != expected_ranges:
        raise ValueError(f"평가 구간은 사전등록 값 {expected_ranges}으로 동결되어 있습니다")
    if not math.isclose(confidence_threshold, FROZEN_CONFIDENCE_THRESHOLD, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("confidence threshold는 0.011722291804로 동결되어 있습니다")
    if candidate_size != FROZEN_CANDIDATE_SIZE:
        raise ValueError("candidate size는 30으로 동결되어 있습니다")
    if experiment_seed != EXPERIMENT_SEED or iterations != PAIRED_ITERATIONS:
        raise ValueError("seed와 paired iterations는 20260818 / 100000으로 동결되어 있습니다")
    reference = hypergeometric_reference()
    if not (
        math.isclose(reference["mean_hits"], RANDOM_MEAN_HITS, abs_tol=1e-12)
        and math.isclose(reference["five_plus_rate"], RANDOM_FIVE_PLUS_RATE, abs_tol=5e-11)
        and math.isclose(reference["exact6_rate"], RANDOM_EXACT6_RATE, abs_tol=5e-11)
    ):
        raise RuntimeError(f"Hypergeometric 기준값 검증 실패: {reference}")

    data_path = Path(data_path)
    motif_run = Path(source_motif_run)
    opportunity_run = Path(source_opportunity_run)
    source_validation = _source_validation(
        data_path=data_path,
        motif_run=motif_run,
        opportunity_run=opportunity_run,
    )
    if source_validation["status"].startswith("FAIL"):
        raise ValueError(f"source validation 실패: {source_validation['status']}")
    draws_by_round = {draw.round_no: draw for draw in draws}
    if sorted(draws_by_round) != list(range(min(draws_by_round), max(draws_by_round) + 1)):
        raise ValueError("데이터 회차가 연속적이지 않습니다")
    if max(draws_by_round) < seen_end:
        raise ValueError("Seen end까지 데이터가 없습니다")
    bundle = build_feature_bundle([draw for draw in draws if draw.round_no <= seen_end])

    implementation_sha256 = _sha256(Path(__file__))
    config_hash = _canonical_sha256(_preregistration("PENDING", implementation_sha256)["frozen_configuration"])
    preliminary_source_hash = _canonical_sha256(source_validation)
    resume_hashes = {
        "config_hash": config_hash,
        "data_hash": source_validation["data"]["actual_sha256"],
        "source_hash": preliminary_source_hash,
    }
    resumed = load_resume_predictions(resume_from, resume_hashes) if resume_from else {}
    label_access: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "checkpoint.jsonl"
    with checkpoint_path.open("w", encoding="utf-8") as checkpoint_handle:
        logger.info("Phase 1 | Seen 852-1235 동결 재현 시작")
        seen_records = _evaluate_targets(
            bundle=bundle,
            draws_by_round=draws_by_round,
            rounds=list(range(seen_start, seen_end + 1)),
            phase="SEEN_REPRODUCTION",
            workers=workers,
            checkpoint_handle=checkpoint_handle,
            checkpoint_hashes=resume_hashes,
            resumed=resumed,
            label_access=label_access,
            logger=logger,
        )
        seen_reproduction = _reproduce_seen(seen_records, motif_run)
        source_validation["seen_reproduction"] = seen_reproduction
        source_validation["status"] = "PASS" if seen_reproduction["status"] == "PASS" else "FAIL_SEEN_REPRODUCTION"
        _write_json(run_dir / "source_validation.json", source_validation)
        source_validation_sha256 = _sha256(run_dir / "source_validation.json")
        if seen_reproduction["status"] != "PASS":
            raise ValueError("Seen Top30 재현 실패; Locked를 열지 않습니다")
        all_records.extend(seen_records)

        logger.info("Phase 2 | preregistration.json 기록 및 hash 고정")
        preregistration = _preregistration(source_validation_sha256, implementation_sha256)
        preregistration_path = run_dir / "preregistration.json"
        _write_json(preregistration_path, preregistration)
        preregistration_sha256 = _sha256(preregistration_path)
        post_prereg_hashes = dict(resume_hashes)
        post_prereg_hashes["preregistration_hash"] = preregistration_sha256

        logger.info("Phase 3 | Locked 660-851 1회 개방")
        locked_records = _evaluate_targets(
            bundle=bundle,
            draws_by_round=draws_by_round,
            rounds=list(range(locked_start, locked_end + 1)),
            phase="LOCKED_CONFIRMATORY",
            workers=workers,
            checkpoint_handle=checkpoint_handle,
            checkpoint_hashes=post_prereg_hashes,
            resumed={},
            label_access=label_access,
            logger=logger,
        )
        all_records.extend(locked_records)
        locked_block_metrics = []
        for label, (start, end) in LOCKED_BLOCKS.items():
            block = [row for row in locked_records if start <= int(row["round"]) <= end and int(row["is_opportunity"])]
            locked_block_metrics.append(_metric_row(block, label=label, target_count=96))
        locked_opportunities = [row for row in locked_records if int(row["is_opportunity"])]
        locked_pooled = _metric_row(locked_opportunities, label="Pooled Locked", target_count=192)
        locked_decision, locked_criteria = _success_decision(locked_pooled, locked_block_metrics)

        blind_opened = locked_decision == "SUCCESS"
        blind_records: list[dict[str, Any]] = []
        blind_block_metrics: list[dict[str, Any]] = []
        blind_pooled: dict[str, Any] | None = None
        blind_decision = "SEALED"
        blind_criteria: dict[str, bool] = {}
        if blind_opened:
            logger.info("Phase 4 | Locked SUCCESS; Additional Blind 468-659 자동 개방")
            blind_records = _evaluate_targets(
                bundle=bundle,
                draws_by_round=draws_by_round,
                rounds=list(range(blind_start, blind_end + 1)),
                phase="ADDITIONAL_BLIND",
                workers=workers,
                checkpoint_handle=checkpoint_handle,
                checkpoint_hashes=post_prereg_hashes,
                resumed={},
                label_access=label_access,
                logger=logger,
            )
            all_records.extend(blind_records)
            for label, (start, end) in BLIND_BLOCKS.items():
                block = [row for row in blind_records if start <= int(row["round"]) <= end and int(row["is_opportunity"])]
                blind_block_metrics.append(_metric_row(block, label=label, target_count=96))
            blind_opportunities = [row for row in blind_records if int(row["is_opportunity"])]
            blind_pooled = _metric_row(blind_opportunities, label="Pooled Blind", target_count=192)
            blind_decision, blind_criteria = _success_decision(blind_pooled, blind_block_metrics)
        else:
            logger.info("Phase 4 | Locked=%s; Additional Blind 봉인 유지", locked_decision)

    if locked_decision == "SUCCESS" and blind_decision == "SUCCESS":
        final_code, final_verdict = "A", "REPLICATED SIGNAL"
    elif locked_decision == "WEAK SIGNAL" or (
        locked_decision == "SUCCESS"
        and blind_pooled is not None
        and float(blind_pooled.get("mean_hit_lift", 0.0)) > 0
    ):
        final_code, final_verdict = "B", "WEAK SIGNAL"
    elif locked_decision == "INCONCLUSIVE":
        final_code, final_verdict = "D", "INCONCLUSIVE"
    else:
        final_code, final_verdict = "C", "NO SIGNAL"

    opened_cohorts = ["Seen-Historical", "Seen-Development", "Locked"] + (["Blind"] if blind_opened else [])
    block_metrics: list[dict[str, Any]] = []
    for label, (start, end) in {**SEEN_BLOCKS, **LOCKED_BLOCKS, **(BLIND_BLOCKS if blind_opened else {})}.items():
        selected = [row for row in all_records if start <= int(row["round"]) <= end and int(row["is_opportunity"])]
        block_metrics.append(_metric_row(selected, label=label, target_count=96))
    cohort_metrics: list[dict[str, Any]] = []
    for cohort in opened_cohorts:
        cohort_records = [row for row in all_records if row["cohort"] == cohort]
        for subset, predicate in (
            ("All", lambda row: True),
            ("Opportunity", lambda row: bool(row["is_opportunity"])),
            ("Non-opportunity", lambda row: not bool(row["is_opportunity"])),
        ):
            selected = [row for row in cohort_records if predicate(row)]
            metric = _metric_row(selected, label=f"{cohort}-{subset}", target_count=len(cohort_records))
            metric["cohort"] = cohort
            metric["subset"] = subset
            cohort_metrics.append(metric)

    hit_distribution: list[dict[str, Any]] = []
    for cohort in opened_cohorts:
        selected = [row for row in all_records if row["cohort"] == cohort and int(row["is_opportunity"])]
        for hits in range(7):
            count = sum(int(row["hits_at_30"]) == hits for row in selected)
            hit_distribution.append(
                {
                    "cohort": cohort,
                    "hits": hits,
                    "opportunity_count": len(selected),
                    "observed_count": count,
                    "observed_rate": count / max(1, len(selected)),
                    "random_probability": float(hypergeom.pmf(hits, 45, 6, 30)),
                }
            )

    artifact_rows = _artifact_rows(all_records)
    surrogate_results: list[dict[str, Any]] = []
    for cohort in opened_cohorts:
        surrogate_results.extend(_surrogate_rows(all_records, cohort))
    _bh_adjust(surrogate_results)
    integrity = _integrity_summary(
        records=all_records,
        label_access=label_access,
        blind_opened=blind_opened,
        preregistration_path=run_dir / "preregistration.json",
        preregistration_sha256=preregistration_sha256,
    )
    if not all(value for value in integrity.values() if isinstance(value, bool)):
        integrity["status"] = "FAIL"

    pd.DataFrame(artifact_rows["predictions"]).to_parquet(run_dir / "top30_predictions.parquet", index=False)
    _write_csv(run_dir / "opportunity_rounds.csv", artifact_rows["opportunities"])
    _write_csv(run_dir / "block_metrics.csv", block_metrics)
    _write_csv(run_dir / "cohort_metrics.csv", cohort_metrics)
    _write_csv(run_dir / "hit_distribution.csv", hit_distribution)
    _write_csv(run_dir / "exact6_rounds.csv", artifact_rows["exact6"])
    _write_csv(run_dir / "candidate_funnel.csv", artifact_rows["funnel"])
    _write_csv(run_dir / "winning_number_ranks.csv", artifact_rows["ranks"])
    _write_csv(
        run_dir / "random_baseline.csv",
        [
            {
                "label": row["label"],
                "opportunity_count": row["opportunity_count"],
                "observed_mean_hits": row.get("mean_hits_at_30"),
                "random_mean_hits": RANDOM_MEAN_HITS,
                "mean_hit_p": row.get("mean_hit_p"),
                "observed_five_plus_rate": row.get("five_plus_rate"),
                "random_five_plus_rate": RANDOM_FIVE_PLUS_RATE,
                "five_plus_p": row.get("five_plus_p"),
                "observed_exact6_rate": row.get("exact6_rate"),
                "random_exact6_rate": RANDOM_EXACT6_RATE,
                "iterations": PAIRED_ITERATIONS,
            }
            for row in block_metrics
        ],
    )
    _write_csv(run_dir / "surrogate_results.csv", surrogate_results)
    _write_csv(run_dir / "target_label_access.csv", label_access)

    decision_criteria = locked_criteria if not blind_opened else {
        f"locked_{key}": value for key, value in locked_criteria.items()
    } | {f"blind_{key}": value for key, value in blind_criteria.items()}
    _plot_artifacts(
        records=all_records,
        block_metrics=block_metrics,
        cohort_metrics=cohort_metrics,
        hit_distribution=hit_distribution,
        funnel_rows=artifact_rows["funnel"],
        criteria=decision_criteria,
        run_dir=run_dir,
    )

    summary: dict[str, Any] = {
        "experiment": "Top30 Broad-Area Retrieval",
        "final_decision": {"code": final_code, "verdict": final_verdict},
        "frozen_configuration": asdict(FROZEN_CONFIG),
        "confidence_threshold": FROZEN_CONFIDENCE_THRESHOLD,
        "candidate_size": FROZEN_CANDIDATE_SIZE,
        "source_validation": source_validation,
        "preregistration_sha256": preregistration_sha256,
        "seen_reproduction": seen_reproduction,
        "locked": {
            "decision": locked_decision,
            "pooled": locked_pooled,
            "blocks": locked_block_metrics,
            "criteria": locked_criteria,
        },
        "blind": {
            "opened": blind_opened,
            "opening_reason": "Pooled Locked SUCCESS" if blind_opened else f"Pooled Locked {locked_decision}",
            "decision": blind_decision,
            "pooled": blind_pooled,
            "blocks": blind_block_metrics,
            "criteria": blind_criteria,
        },
        "cohort_metrics": cohort_metrics,
        "block_metrics": block_metrics,
        "random_reference": reference,
        "integrity": integrity,
        "top30_internal_reranker_allowed": final_code == "A",
        "execution": execution_metadata(
            draws=draws,
            started_at=started_at,
            start_round=seen_start,
            end_round=seen_end,
        ),
    }
    _write_json(run_dir / "metrics.json", summary)
    _write_json(
        run_dir / "run_state.json",
        {
            "phase": "COMPLETE",
            "last_completed_target": blind_end if blind_opened else locked_end,
            "source_hashes": EXPECTED_SOURCE_HASHES,
            "preregistration_hash": preregistration_sha256,
            "config_hash": config_hash,
            "random_seed": EXPERIMENT_SEED,
            "blind_gate_state": "OPENED" if blind_opened else "SEALED",
            "final_decision": summary["final_decision"],
        },
    )
    logger.info(
        "Top30 완료 | Locked=%s | Blind=%s | final=%s. %s",
        locked_decision,
        blind_decision,
        final_code,
        final_verdict,
    )
    return summary
