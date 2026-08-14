from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from time import perf_counter
from typing import Any, Sequence

from uriel_v2.logging_config import _timezone
from uriel_v2.models import Draw


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def execution_metadata(
    *,
    draws: Sequence[Draw],
    started_at: float,
    start_round: int,
    end_round: int,
) -> dict[str, Any]:
    return {
        "git_commit": current_git_commit(),
        "executed_at": datetime.now(_timezone()).isoformat(timespec="seconds"),
        "elapsed_seconds": perf_counter() - started_at,
        "python_version": platform.python_version(),
        "data_first_round": draws[0].round_no,
        "data_last_round": draws[-1].round_no,
        "evaluation_start_round": start_round,
        "evaluation_end_round": end_round,
    }
