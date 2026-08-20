from __future__ import annotations

import argparse
import json
from pathlib import Path

from uriel_v2.logging_config import create_run_directory, setup_logging
from uriel_v2.probabilistic_lab.phase2 import run_phase2
from uriel_v2.probabilistic_lab.phase3 import run_phase3, validate_phase3_dataset
from uriel_v2.probabilistic_lab.phase4 import run_phase4
from uriel_v2.probabilistic_lab.phase5 import run_phase5
from uriel_v2.probabilistic_lab.phase6 import run_phase6
from uriel_v2.probabilistic_lab.pilot import run_pilot
from uriel_v2.probabilistic_lab.validation import validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uriel-probabilistic",
        description="문제 구조와 랜덤 메커니즘의 결과 분포를 수집하는 확률 실험 엔진",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("pilot", help="Phase 1 Monte Carlo/Random Search smoke pilot")
    pilot.add_argument("--output", default="artifacts/probabilistic", help="실행 결과 상위 디렉터리")
    pilot.add_argument("--instances-per-family", type=int, default=4)
    pilot.add_argument("--seeds", type=int, default=3, help="문제별 반복 seed 수")
    pilot.add_argument("--master-seed", type=int, default=20_260_819)
    pilot.add_argument("--monte-carlo-budget", type=int, default=4_096)
    pilot.add_argument("--random-search-budget", type=int, default=2_048)
    pilot.add_argument("--workers", default="auto")
    pilot.add_argument("--resume-from", default=None, help="이전 pilot 실행 디렉터리")
    pilot.add_argument("--verbose", action="store_true")

    phase2 = commands.add_parser("phase2", help="RQMC/CMA-ES paired random-mechanism comparison")
    phase2.add_argument("--output", default="artifacts/probabilistic", help="실행 결과 상위 디렉터리")
    phase2.add_argument("--instances-per-family", type=int, default=8)
    phase2.add_argument("--seeds", type=int, default=10, help="problem별 paired seed 수")
    phase2.add_argument("--master-seed", type=int, default=20_260_820)
    phase2.add_argument("--sampling-budget", type=int, default=4_096)
    phase2.add_argument("--optimization-budget", type=int, default=4_096)
    phase2.add_argument("--bootstrap-iterations", type=int, default=10_000)
    phase2.add_argument("--workers", default="auto")
    phase2.add_argument("--resume-from", default=None, help="이전 Phase 2 실행 디렉터리")
    phase2.add_argument("--verbose", action="store_true")

    phase3 = commands.add_parser("phase3", help="balanced multi-domain synthetic problem benchmark")
    phase3.add_argument("--output", default="artifacts/probabilistic", help="실행 결과 상위 디렉터리")
    phase3.add_argument("--instances-per-family", type=int, default=128)
    phase3.add_argument("--master-seed", type=int, default=20_260_821)
    phase3.add_argument("--folds", type=int, default=5)
    phase3.add_argument("--minimum-problems", type=int, default=1_000)
    phase3.add_argument("--verbose", action="store_true")

    phase4 = commands.add_parser("phase4", help="large-scale paired repeated execution")
    phase4.add_argument("--benchmark", required=True, help="Phase 3 실행 디렉터리")
    phase4.add_argument("--output", default="artifacts/probabilistic", help="실행 결과 상위 디렉터리")
    phase4.add_argument("--seeds", type=int, default=10)
    phase4.add_argument("--master-seed", type=int, default=20_260_822)
    phase4.add_argument("--sampling-budget", type=int, default=1_024)
    phase4.add_argument("--optimization-budget", type=int, default=1_024)
    phase4.add_argument("--bootstrap-iterations", type=int, default=10_000)
    phase4.add_argument("--workers", default="auto")
    phase4.add_argument("--resume-from", default=None)
    phase4.add_argument("--verbose", action="store_true")

    phase5 = commands.add_parser("phase5", help="Phase 4 dataset quality and reproducibility audit")
    phase5.add_argument("--phase4", required=True, help="Phase 4 실행 디렉터리")
    phase5.add_argument("--output", default="artifacts/probabilistic", help="감사 결과 상위 디렉터리")
    phase5.add_argument("--reproducibility-samples", type=int, default=1)
    phase5.add_argument("--verbose", action="store_true")

    phase6 = commands.add_parser("phase6", help="leakage-safe feature engineering and fold preprocessing")
    phase6.add_argument("--phase4", required=True, help="동결된 Phase 4 실행 디렉터리")
    phase6.add_argument("--phase5", required=True, help="통과한 Phase 5 감사 디렉터리")
    phase6.add_argument("--output", default="artifacts/probabilistic", help="Phase 6 결과 상위 디렉터리")
    phase6.add_argument("--resume-from", default=None, help="중단된 Phase 6 실행 디렉터리")
    phase6.add_argument("--verbose", action="store_true")

    validate = commands.add_parser("validate", help="생성된 Parquet 데이터셋 품질 검사")
    validate.add_argument("run_directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        run_directory = Path(args.run_directory)
        if (run_directory / "manifest.json").exists() and not (run_directory / "data/runs/runs.parquet").exists():
            result = validate_phase3_dataset(run_directory)
        else:
            result = validate_dataset(run_directory)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 1
    if args.command == "phase3":
        run_directory = create_run_directory(args.output, "probabilistic-phase3")
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase3(
            run_directory,
            instances_per_family=args.instances_per_family,
            master_seed=args.master_seed,
            folds=args.folds,
            minimum_problem_count=args.minimum_problems,
            logger=logger,
        )
        logger.info(
            "[SUMMARY] status=%s problems=%s families=%s domains=%s directory=%s",
            summary["status"],
            summary["problem_count"],
            summary["family_count"],
            summary["domain_count"],
            run_directory,
        )
        return 0 if summary["status"] == "PHASE_3_PASS" else 1
    if args.command == "phase4":
        run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(
            args.output, "probabilistic-phase4"
        )
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase4(
            run_directory,
            args.benchmark,
            seed_replicates=args.seeds,
            master_seed=args.master_seed,
            sampling_budget=args.sampling_budget,
            optimization_budget=args.optimization_budget,
            bootstrap_iterations=args.bootstrap_iterations,
            workers=args.workers,
            resume=True,
            logger=logger,
        )
        logger.info(
            "[SUMMARY] status=%s runs=%s pairs=%s directory=%s",
            summary["status"],
            summary["run_count"],
            summary["pair_count"],
            run_directory,
        )
        return 0 if summary["status"] == "PHASE_4_PASS" else 1
    if args.command == "phase5":
        run_directory = create_run_directory(args.output, "probabilistic-phase5")
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase5(
            run_directory,
            args.phase4,
            reproducibility_samples_per_algorithm_family=args.reproducibility_samples,
        )
        logger.info(
            "[SUMMARY] status=%s critical=%s warnings=%s directory=%s",
            summary["status"],
            summary["critical_issue_count"],
            summary["warning_count"],
            run_directory,
        )
        return 0 if summary["status"] == "PHASE_5_PASS" else 1
    if args.command == "phase6":
        run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(
            args.output, "probabilistic-phase6"
        )
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase6(
            run_directory,
            args.phase4,
            args.phase5,
            logger=logger,
        )
        return 0 if summary["status"] == "PHASE_6_PASS" else 1
    if args.command == "phase2":
        run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(
            args.output, "probabilistic-phase2"
        )
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase2(
            run_directory,
            instances_per_family=args.instances_per_family,
            seed_replicates=args.seeds,
            master_seed=args.master_seed,
            sampling_budget=args.sampling_budget,
            optimization_budget=args.optimization_budget,
            bootstrap_iterations=args.bootstrap_iterations,
            workers=args.workers,
            resume=True,
            logger=logger,
        )
        logger.info(
            "[SUMMARY] status=%s runs=%s pairs=%s directory=%s",
            summary["status"],
            summary["run_count"],
            summary["pair_count"],
            run_directory,
        )
        return 0 if summary["status"] == "PHASE_2_PASS" else 1

    run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(args.output, "probabilistic-pilot")
    logger = setup_logging(run_directory, args.verbose)
    summary = run_pilot(
        run_directory,
        instances_per_family=args.instances_per_family,
        seed_replicates=args.seeds,
        master_seed=args.master_seed,
        monte_carlo_budget=args.monte_carlo_budget,
        random_search_budget=args.random_search_budget,
        workers=args.workers,
        resume=True,
        logger=logger,
    )
    logger.info("[SUMMARY] status=%s runs=%s directory=%s", summary["status"], summary["run_count"], run_directory)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
