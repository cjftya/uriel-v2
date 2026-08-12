from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from uriel_v2 import __version__
from uriel_v2.data import find_draw, load_draws
from uriel_v2.evaluation import evaluate_walk_forward
from uriel_v2.logging_config import create_run_directory, setup_logging
from uriel_v2.models import Draw, EvaluationRow, ReverseMatch
from uriel_v2.reverse import reverse_search
from uriel_v2.strategies import STRATEGIES, create_predictions


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default="lotto.xlsx", help="로또 엑셀 파일 (기본: lotto.xlsx)")
    parser.add_argument("--sheet", default=None, help="시트 이름 (기본: 첫 시트)")
    parser.add_argument("--output", default="outputs", help="실행 결과 디렉터리")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 표시")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uriel", description="Uriel v2: 재현 가능한 로또 시드 생성·검증·역산 실험")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="엑셀 데이터 구조와 최근 회차를 확인")
    _add_common_arguments(inspect)
    inspect.add_argument("--rows", type=int, default=10, help="표시할 최근 회차 수, 0이면 전체")

    predict = commands.add_parser("predict", help="지정 회차에 사용할 시드와 번호 생성")
    _add_common_arguments(predict)
    predict.add_argument("--round", type=int, default=None, help="대상 회차 (기본: 최신+1)")
    predict.add_argument("--strategies", default=",".join(STRATEGIES), help="쉼표로 구분한 전략")
    predict.add_argument("--candidates", type=int, default=1, help="전략별 시드 variant 수")
    predict.add_argument("--history-window", type=int, default=64, help="과거 참조 회차 수")

    evaluate = commands.add_parser("evaluate", help="과거 데이터 walk-forward 평가")
    _add_common_arguments(evaluate)
    evaluate.add_argument("--start-round", type=int, default=None, help="시작 회차 (기본: 최근 192회)")
    evaluate.add_argument("--end-round", type=int, default=None, help="종료 회차 (기본: 최신 회차)")
    evaluate.add_argument("--strategies", default=",".join(STRATEGIES), help="쉼표로 구분한 전략")
    evaluate.add_argument("--candidates", type=int, default=1, help="전략별 시드 variant 수")
    evaluate.add_argument("--history-window", type=int, default=64, help="과거 참조 회차 수")
    evaluate.add_argument("--minimum-history", type=int, default=64, help="평가 전 필요한 최소 과거 회차")
    evaluate.add_argument("--workers", default="auto", help="프로세스 워커 수 또는 auto")

    reverse = commands.add_parser("reverse", help="정답 번호에서 일치/근접 시드 구간 탐색")
    _add_common_arguments(reverse)
    target = reverse.add_mutually_exclusive_group(required=True)
    target.add_argument("--round", type=int, help="정답으로 사용할 과거 회차")
    target.add_argument("--numbers", help="정답 번호 6개, 예: 6,7,11,15,39,43")
    reverse.add_argument("--seed-start", type=int, default=0, help="시작 시드 (포함)")
    reverse.add_argument("--seed-end", type=int, default=1_000_000, help="종료 시드 (미포함)")
    reverse.add_argument("--min-hits", type=int, default=5, help="저장할 최소 적중 수")
    reverse.add_argument("--chunk-size", type=int, default=50_000, help="워커 작업 청크 크기")
    reverse.add_argument("--result-limit", type=int, default=100, help="저장할 상위 결과 수")
    reverse.add_argument("--workers", default="auto", help="프로세스 워커 수 또는 auto")
    return parser


def _parse_strategies(value: str) -> tuple[str, ...]:
    strategies = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = sorted(set(strategies).difference(STRATEGIES))
    if unknown:
        raise ValueError(f"알 수 없는 전략: {', '.join(unknown)} (가능: {', '.join(STRATEGIES)})")
    if not strategies:
        raise ValueError("전략을 하나 이상 지정해야 합니다")
    return strategies


def _parse_numbers(value: str) -> tuple[int, ...]:
    try:
        numbers = tuple(sorted(int(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise ValueError("numbers는 쉼표로 구분한 정수 6개여야 합니다") from exc
    if len(numbers) != 6 or len(set(numbers)) != 6 or any(number < 1 or number > 45 for number in numbers):
        raise ValueError("numbers는 서로 다른 1~45 정수 6개여야 합니다")
    return numbers


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_and_log(args: argparse.Namespace, logger: logging.Logger) -> list[Draw]:
    draws = load_draws(args.data, args.sheet)
    logger.info(
        "데이터 로드 완료 | 파일=%s | 회차=%s | 범위=%s~%s | 최신=%s",
        Path(args.data).resolve(), len(draws), draws[0].round_no, draws[-1].round_no, draws[-1].numbers,
    )
    return draws


def _run_inspect(args: argparse.Namespace, run_dir: Path, logger: logging.Logger) -> None:
    draws = _load_and_log(args, logger)
    shown = list(reversed(draws if args.rows == 0 else draws[-max(args.rows, 0):]))
    logger.info("회차 데이터 표시 | 행=%s", len(shown))
    for draw in shown:
        logger.info("회차=%4s | 당첨=%s | 보너스=%s", draw.round_no, " ".join(f"{n:02d}" for n in draw.numbers), draw.bonus)
    _write_json(run_dir / "data-summary.json", {
        "file": str(Path(args.data).resolve()), "sheet": args.sheet, "draw_count": len(draws),
        "first_round": draws[0].round_no, "last_round": draws[-1].round_no,
        "displayed": [asdict(draw) for draw in shown],
    })


def _run_predict(args: argparse.Namespace, run_dir: Path, logger: logging.Logger) -> None:
    draws = _load_and_log(args, logger)
    strategies = _parse_strategies(args.strategies)
    target_round = args.round or draws[-1].round_no + 1
    history = [draw for draw in draws if draw.round_no < target_round]
    if not history:
        raise ValueError("대상 회차보다 앞선 데이터가 없습니다")
    logger.info(
        "시드 생성 | 대상=%s회 | 마지막 과거=%s회 | 전략=%s | 후보/전략=%s | window=%s",
        target_round, history[-1].round_no, ",".join(strategies), args.candidates, args.history_window,
    )
    predictions = [
        prediction
        for strategy in strategies
        for prediction in create_predictions(
            strategy, history, target_round, candidates=args.candidates, history_window=args.history_window,
        )
    ]
    for prediction in predictions:
        logger.info(
            "전략=%-14s | variant=%3s | seed=%20s | 번호=%s",
            prediction.strategy, prediction.variant, prediction.seed,
            " ".join(f"{number:02d}" for number in prediction.numbers),
        )
    _write_json(run_dir / "predictions.json", {
        "target_round": target_round, "history_last_round": history[-1].round_no,
        "history_window": args.history_window, "predictions": [asdict(prediction) for prediction in predictions],
    })


def _evaluation_summary(rows: list[EvaluationRow], candidates: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "selection": "single" if candidates == 1 else "oracle-best", "candidates": candidates, "strategies": {},
    }
    for strategy in sorted({row.strategy for row in rows}):
        selected = [row for row in rows if row.strategy == strategy]
        hit_distribution = Counter(row.hits for row in selected)
        result["strategies"][strategy] = {
            "rounds": len(selected),
            "average_hits": sum(row.hits for row in selected) / len(selected),
            "max_hits": max(row.hits for row in selected),
            "hit_distribution": {str(hit): hit_distribution.get(hit, 0) for hit in range(7)},
            "hit_4_plus": sum(row.hits >= 4 for row in selected),
            "hit_5_plus": sum(row.hits >= 5 for row in selected),
            "average_positional_mae": sum(row.positional_mae for row in selected) / len(selected),
            "average_signed_bias": sum(row.signed_bias for row in selected) / len(selected),
        }
    return result


def _write_evaluation_csv(path: Path, rows: list[EvaluationRow]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "round", "strategy", "variant", "seed", "prediction", "winner", "hits", "set_distance",
            "positional_mae", "signed_bias", "delta_1", "delta_2", "delta_3", "delta_4", "delta_5", "delta_6",
        ])
        for row in rows:
            writer.writerow([
                row.round_no, row.strategy, row.variant, row.seed, "-".join(map(str, row.prediction)),
                "-".join(map(str, row.winner)), row.hits, row.set_distance, f"{row.positional_mae:.6f}",
                f"{row.signed_bias:.6f}", *row.deviations,
            ])


def _run_evaluate(args: argparse.Namespace, run_dir: Path, logger: logging.Logger) -> None:
    draws = _load_and_log(args, logger)
    strategies = _parse_strategies(args.strategies)
    end_round = args.end_round or draws[-1].round_no
    start_round = args.start_round or max(draws[0].round_no, end_round - 191)
    rows = evaluate_walk_forward(
        draws, start_round=start_round, end_round=end_round, strategies=strategies,
        candidates=args.candidates, history_window=args.history_window, minimum_history=args.minimum_history,
        workers=args.workers, logger=logger,
    )
    summary = _evaluation_summary(rows, args.candidates)
    for strategy, values in summary["strategies"].items():
        logger.info(
            "평가 요약 | 전략=%-14s | 회차=%s | 평균적중=%.4f | 최대=%s | 4+=%s | 5+=%s | 위치MAE=%.4f",
            strategy, values["rounds"], values["average_hits"], values["max_hits"],
            values["hit_4_plus"], values["hit_5_plus"], values["average_positional_mae"],
        )
    _write_evaluation_csv(run_dir / "evaluation.csv", rows)
    _write_json(run_dir / "summary.json", summary)


def _run_reverse(args: argparse.Namespace, run_dir: Path, logger: logging.Logger) -> None:
    draws = _load_and_log(args, logger)
    if args.round is not None:
        draw = find_draw(draws, args.round)
        target, target_source = draw.numbers, {"round": draw.round_no}
    else:
        target, target_source = _parse_numbers(args.numbers), {"numbers": _parse_numbers(args.numbers)}
    result = reverse_search(
        target=target, start=args.seed_start, end=args.seed_end, min_hits=args.min_hits,
        chunk_size=args.chunk_size, result_limit=args.result_limit, workers=args.workers, logger=logger,
    )
    logger.info(
        "역산 완료 | 탐색=%s | 시간=%.3fs | 속도=%s seeds/s | 분포=%s | 저장결과=%s",
        f"{result.end - result.start:,}", result.elapsed_seconds,
        f"{(result.end - result.start) / result.elapsed_seconds:,.0f}",
        dict(enumerate(result.hit_distribution)), len(result.matches),
    )
    displayed = result.matches if result.matches else result.best[:10]
    if not result.matches:
        logger.info("최소 적중 조건을 만족한 시드가 없어 탐색 구간의 최고 결과를 표시합니다")
    for match in displayed[:20]:
        logger.info(
            "적중=%s | MAE=%.3f | seed=%s | 번호=%s", match.hits, match.positional_mae,
            match.seed, " ".join(f"{number:02d}" for number in match.numbers),
        )
    _write_json(run_dir / "reverse-search.json", {
        "target": target, "target_source": target_source, "seed_start": result.start, "seed_end": result.end,
        "scanned": result.end - result.start, "elapsed_seconds": result.elapsed_seconds,
        "workers": result.workers, "chunks": result.chunks,
        "hit_distribution": {str(hit): count for hit, count in enumerate(result.hit_distribution)},
        "matches": [asdict(match) for match in result.matches], "best": [asdict(match) for match in result.best],
        "warning": "정답을 사용한 역산 진단 결과이며 미래 회차 예측 성능이 아닙니다.",
    })


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = create_run_directory(args.output, args.command)
    logger = setup_logging(run_dir, args.verbose)
    logger.info("Uriel v%s | command=%s | 결과=%s", __version__, args.command, run_dir.resolve())
    try:
        if args.command == "inspect":
            _run_inspect(args, run_dir, logger)
        elif args.command == "predict":
            _run_predict(args, run_dir, logger)
        elif args.command == "evaluate":
            _run_evaluate(args, run_dir, logger)
        elif args.command == "reverse":
            _run_reverse(args, run_dir, logger)
        else:
            parser.error(f"지원하지 않는 명령: {args.command}")
    except KeyboardInterrupt:
        logger.warning("사용자 요청으로 중단했습니다. 완료된 로그는 보존됩니다.")
        return 130
    except Exception as exc:
        logger.exception("실행 실패: %s", exc)
        return 2
    logger.info("실행 완료 | 결과=%s", run_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
