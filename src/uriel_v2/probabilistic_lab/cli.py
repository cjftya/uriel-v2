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
from uriel_v2.probabilistic_lab.phase7 import run_phase7
from uriel_v2.probabilistic_lab.phase8 import run_phase8
from uriel_v2.probabilistic_lab.phase9 import run_phase9
from uriel_v2.probabilistic_lab.phase10 import run_phase10
from uriel_v2.probabilistic_lab.phase11 import run_phase11
from uriel_v2.probabilistic_lab.phase12 import run_phase12
from uriel_v2.probabilistic_lab.phase13 import run_phase13
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

    phase7 = commands.add_parser("phase7", help="leakage-safe LR/RF/GBM point-prediction baselines")
    phase7.add_argument("--phase6", required=True, help="통과한 Phase 6 feature 디렉터리")
    phase7.add_argument("--output", default="artifacts/probabilistic", help="Phase 7 결과 상위 디렉터리")
    phase7.add_argument("--resume-from", default=None, help="중단된 Phase 7 실행 디렉터리")
    phase7.add_argument("--master-seed", type=int, default=20_260_823)
    phase7.add_argument("--rf-estimators", type=int, default=48)
    phase7.add_argument("--gb-iterations", type=int, default=100)
    phase7.add_argument("--verbose", action="store_true")

    phase8 = commands.add_parser("phase8", help="conditional quality-distribution modelling")
    phase8.add_argument("--phase6", required=True, help="통과한 Phase 6 feature 디렉터리")
    phase8.add_argument("--phase7", required=True, help="통과한 Phase 7 baseline 디렉터리")
    phase8.add_argument("--output", default="artifacts/probabilistic", help="Phase 8 결과 상위 디렉터리")
    phase8.add_argument("--resume-from", default=None, help="중단된 Phase 8 실행 디렉터리")
    phase8.add_argument("--master-seed", type=int, default=20_260_824)
    phase8.add_argument("--gb-iterations", type=int, default=80)
    phase8.add_argument("--verbose", action="store_true")

    phase9 = commands.add_parser("phase9", help="failure probability and failure-type modelling")
    phase9.add_argument("--phase6", required=True, help="통과한 Phase 6 feature 디렉터리")
    phase9.add_argument("--phase7", required=True, help="통과한 Phase 7 baseline 디렉터리")
    phase9.add_argument("--phase8", required=True, help="통과한 Phase 8 distribution 디렉터리")
    phase9.add_argument("--output", default="artifacts/probabilistic", help="Phase 9 결과 상위 디렉터리")
    phase9.add_argument("--resume-from", default=None, help="중단된 Phase 9 실행 디렉터리")
    phase9.add_argument("--master-seed", type=int, default=20_260_825)
    phase9.add_argument("--gb-iterations", type=int, default=80)
    phase9.add_argument("--calibration-bins", type=int, default=10)
    phase9.add_argument("--verbose", action="store_true")

    phase10 = commands.add_parser("phase10", help="runtime distribution and first-passage survival")
    phase10.add_argument("--phase6", required=True, help="통과한 Phase 6 feature 디렉터리")
    phase10.add_argument("--phase7", required=True, help="통과한 Phase 7 baseline 디렉터리")
    phase10.add_argument("--phase8", required=True, help="통과한 Phase 8 quality distribution 디렉터리")
    phase10.add_argument("--phase9", required=True, help="통과한 Phase 9 failure distribution 디렉터리")
    phase10.add_argument("--output", default="artifacts/probabilistic", help="Phase 10 결과 상위 디렉터리")
    phase10.add_argument("--resume-from", default=None, help="중단된 Phase 10 실행 디렉터리")
    phase10.add_argument("--master-seed", type=int, default=20_260_826)
    phase10.add_argument("--gb-iterations", type=int, default=60)
    phase10.add_argument("--verbose", action="store_true")

    phase11 = commands.add_parser("phase11", help="hierarchical Bayesian partial pooling")
    phase11.add_argument("--phase6", required=True, help="통과한 Phase 6 feature 디렉터리")
    phase11.add_argument("--phase7", required=True, help="통과한 Phase 7 baseline 디렉터리")
    phase11.add_argument("--phase8", required=True, help="통과한 Phase 8 quality distribution 디렉터리")
    phase11.add_argument("--phase9", required=True, help="통과한 Phase 9 failure distribution 디렉터리")
    phase11.add_argument("--phase10", required=True, help="통과한 Phase 10 runtime-survival 디렉터리")
    phase11.add_argument("--output", default="artifacts/probabilistic", help="Phase 11 결과 상위 디렉터리")
    phase11.add_argument("--resume-from", default=None, help="중단된 Phase 11 실행 디렉터리")
    phase11.add_argument("--ridge-alpha", type=float, default=10.0)
    phase11.add_argument("--prior-strength", type=float, default=20.0)
    phase11.add_argument("--verbose", action="store_true")

    phase12 = commands.add_parser("phase12", help="cross-fitted Mixture-of-Experts routing")
    phase12.add_argument("--phase6", required=True, help="통과한 Phase 6 feature 디렉터리")
    phase12.add_argument("--phase7", required=True, help="통과한 Phase 7 baseline 디렉터리")
    phase12.add_argument("--phase8", required=True, help="통과한 Phase 8 quality distribution 디렉터리")
    phase12.add_argument("--phase9", required=True, help="통과한 Phase 9 failure distribution 디렉터리")
    phase12.add_argument("--phase10", required=True, help="통과한 Phase 10 runtime-survival 디렉터리")
    phase12.add_argument("--phase11", required=True, help="통과한 Phase 11 hierarchical 디렉터리")
    phase12.add_argument("--output", default="artifacts/probabilistic", help="Phase 12 결과 상위 디렉터리")
    phase12.add_argument("--resume-from", default=None, help="중단된 Phase 12 실행 디렉터리")
    phase12.add_argument("--master-seed", type=int, default=20_260_827)
    phase12.add_argument("--gate-iterations", type=int, default=40)
    phase12.add_argument("--minimum-gate-rows", type=int, default=100)
    phase12.add_argument("--verbose", action="store_true")

    phase13 = commands.add_parser("phase13", help="cross-fitted joint probability calibration")
    phase13.add_argument("--phase6", required=True, help="통과한 Phase 6 feature 디렉터리")
    phase13.add_argument("--phase7", required=True, help="통과한 Phase 7 baseline 디렉터리")
    phase13.add_argument("--phase8", required=True, help="통과한 Phase 8 quality distribution 디렉터리")
    phase13.add_argument("--phase9", required=True, help="통과한 Phase 9 failure distribution 디렉터리")
    phase13.add_argument("--phase10", required=True, help="통과한 Phase 10 runtime-survival 디렉터리")
    phase13.add_argument("--phase11", required=True, help="통과한 Phase 11 hierarchical 디렉터리")
    phase13.add_argument("--phase12", required=True, help="통과한 Phase 12 mixture 디렉터리")
    phase13.add_argument("--output", default="artifacts/probabilistic", help="Phase 13 결과 상위 디렉터리")
    phase13.add_argument("--resume-from", default=None, help="중단된 Phase 13 실행 디렉터리")
    phase13.add_argument("--master-seed", type=int, default=20_260_828)
    phase13.add_argument("--calibration-strength", type=float, default=200.0)
    phase13.add_argument("--minimum-class-rows", type=int, default=20)
    phase13.add_argument("--copula-shrinkage", type=float, default=200.0)
    phase13.add_argument("--verbose", action="store_true")

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
    if args.command == "phase7":
        run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(
            args.output, "probabilistic-phase7"
        )
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase7(
            run_directory,
            args.phase6,
            master_seed=args.master_seed,
            random_forest_estimators=args.rf_estimators,
            gradient_boosting_iterations=args.gb_iterations,
            logger=logger,
        )
        return 0 if summary["status"] == "PHASE_7_PASS" else 1
    if args.command == "phase8":
        run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(
            args.output, "probabilistic-phase8"
        )
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase8(
            run_directory,
            args.phase6,
            args.phase7,
            master_seed=args.master_seed,
            gradient_boosting_iterations=args.gb_iterations,
            logger=logger,
        )
        return 0 if summary["status"] == "PHASE_8_PASS" else 1
    if args.command == "phase9":
        run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(
            args.output, "probabilistic-phase9"
        )
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase9(
            run_directory,
            args.phase6,
            args.phase7,
            args.phase8,
            master_seed=args.master_seed,
            gradient_boosting_iterations=args.gb_iterations,
            calibration_bins=args.calibration_bins,
            logger=logger,
        )
        return 0 if summary["status"] == "PHASE_9_PASS" else 1
    if args.command == "phase10":
        run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(
            args.output, "probabilistic-phase10"
        )
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase10(
            run_directory,
            args.phase6,
            args.phase7,
            args.phase8,
            args.phase9,
            master_seed=args.master_seed,
            gradient_boosting_iterations=args.gb_iterations,
            logger=logger,
        )
        return 0 if summary["status"] == "PHASE_10_PASS" else 1
    if args.command == "phase11":
        run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(
            args.output, "probabilistic-phase11"
        )
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase11(
            run_directory,
            args.phase6,
            args.phase7,
            args.phase8,
            args.phase9,
            args.phase10,
            ridge_alpha=args.ridge_alpha,
            prior_strength=args.prior_strength,
            logger=logger,
        )
        return 0 if summary["status"] == "PHASE_11_PASS" else 1
    if args.command == "phase12":
        run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(
            args.output, "probabilistic-phase12"
        )
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase12(
            run_directory,
            args.phase6,
            args.phase7,
            args.phase8,
            args.phase9,
            args.phase10,
            args.phase11,
            master_seed=args.master_seed,
            gate_iterations=args.gate_iterations,
            minimum_gate_rows=args.minimum_gate_rows,
            logger=logger,
        )
        return 0 if summary["status"] == "PHASE_12_PASS" else 1
    if args.command == "phase13":
        run_directory = Path(args.resume_from) if args.resume_from else create_run_directory(
            args.output, "probabilistic-phase13"
        )
        logger = setup_logging(run_directory, args.verbose)
        summary = run_phase13(
            run_directory,
            args.phase6,
            args.phase7,
            args.phase8,
            args.phase9,
            args.phase10,
            args.phase11,
            args.phase12,
            master_seed=args.master_seed,
            calibration_strength=args.calibration_strength,
            minimum_class_rows=args.minimum_class_rows,
            copula_shrinkage=args.copula_shrinkage,
            logger=logger,
        )
        return 0 if summary["status"] == "PHASE_13_PASS" else 1
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
