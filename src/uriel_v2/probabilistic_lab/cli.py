from __future__ import annotations

import argparse
import json
from pathlib import Path

from uriel_v2.logging_config import create_run_directory, setup_logging
from uriel_v2.probabilistic_lab.phase2 import run_phase2
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

    validate = commands.add_parser("validate", help="생성된 Parquet 데이터셋 품질 검사")
    validate.add_argument("run_directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        result = validate_dataset(args.run_directory)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 1
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
