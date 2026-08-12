from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_TIMEZONE = "Asia/Seoul"


def _timezone() -> tzinfo:
    name = os.environ.get("URIEL_TIMEZONE", DEFAULT_TIMEZONE)
    if name in {"Asia/Seoul", "KST"}:
        return timezone(timedelta(hours=9), name="KST")
    try:
        return ZoneInfo(name)
    except KeyError as exc:
        raise ValueError(f"알 수 없는 URIEL_TIMEZONE입니다: {name}") from exc


class ZonedFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        current = datetime.fromtimestamp(record.created, tz=_timezone())
        return current.strftime(datefmt) if datefmt else current.isoformat(timespec="milliseconds")


def create_run_directory(base: str | Path, command: str) -> Path:
    base_path = Path(base)
    timestamp = datetime.now(_timezone()).strftime("%Y%m%d-%H%M%S")
    candidate = base_path / f"{timestamp}-{command}"
    suffix = 1
    while candidate.exists():
        candidate = base_path / f"{timestamp}-{command}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def setup_logging(run_directory: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("uriel")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = ZonedFormatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(run_directory / "uriel.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
