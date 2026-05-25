"""Analysis placeholder for future pass/fail rules and plots."""

from __future__ import annotations

from pathlib import Path


def analyze_result(result_dir: Path) -> dict[str, str]:
    return {
        "result_dir": str(result_dir),
        "analysis_status": "not_implemented",
    }

