"""Result directory creation and summary writing."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from runner.case_schema import CaseSpec, case_to_dict
from runner.telemetry import TelemetrySample
from runner.workload_launcher import ResolvedWorkload


def create_result_dir(results_root: Path, case_id: str) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
    result_dir = results_root / f"{timestamp}_{safe_case_id}"
    suffix = 1
    while result_dir.exists():
        result_dir = results_root / f"{timestamp}_{safe_case_id}_{suffix}"
        suffix += 1
    result_dir.mkdir(parents=True)
    return result_dir


def copy_case_file(case_path: Path, result_dir: Path) -> None:
    shutil.copy2(case_path, result_dir / "case.yaml")


def write_summary(
    result_dir: Path,
    case: CaseSpec,
    resolved: ResolvedWorkload,
    status: str,
    sample_count: int,
    samples: list[TelemetrySample] | None = None,
    telemetry_sampler: str | None = None,
    return_code: int | None = None,
    sweep_plan: list[dict[str, Any]] | None = None,
) -> Path:
    power_stats = compute_power_stats(samples or [])
    workload_results = parse_workload_results(result_dir / "stdout.log")
    sweep_power_stats = compute_sweep_power_stats(case, samples or [], workload_results)
    summary: dict[str, Any] = {
        "case": case_to_dict(case),
        "status": status,
        "dry_run": status == "dry_run",
        "return_code": return_code,
        "workload_command": resolved.command,
        "workload_cwd": str(resolved.cwd),
        "workload_env": resolved.env,
        "workload_results": workload_results,
        "workload_summary": _last_steady_result(workload_results),
        "sweep_plan": sweep_plan or [],
        "sweep_power_stats": sweep_power_stats,
        "telemetry": {
            "sampler": telemetry_sampler or case.telemetry.sampler,
            "interval_ms": case.telemetry.interval_ms,
            "metrics": case.telemetry.metrics,
            "sample_count": sample_count,
            "power": power_stats,
        },
        "artifacts": {
            "case": "case.yaml",
            "summary": "summary.json",
            "telemetry": "telemetry.csv",
            "stdout": "stdout.log",
            "stderr": "stderr.log",
        },
    }
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def compute_sweep_power_stats(
    case: CaseSpec,
    samples: list[TelemetrySample],
    workload_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pending_starts: list[dict[str, Any]] = []
    sweep_stats: list[dict[str, Any]] = []
    warmup_sec = _warmup_seconds(case)

    for result in workload_results:
        phase = result.get("phase")
        if phase == "runner_sweep_point_start":
            pending_starts.append(result)
            continue
        if phase != "runner_sweep_point_end":
            continue

        start = _pop_matching_start(pending_starts, result)
        if start is None:
            continue

        start_timestamp_ns = _optional_int(start.get("timestamp_ns"))
        end_timestamp_ns = _optional_int(result.get("timestamp_ns"))
        if start_timestamp_ns is None or end_timestamp_ns is None:
            continue

        steady_window_fallback_used = warmup_sec is None
        if warmup_sec is None:
            steady_start_timestamp_ns = start_timestamp_ns
        else:
            steady_start_timestamp_ns = start_timestamp_ns + int(warmup_sec * 1_000_000_000)
        steady_end_timestamp_ns = end_timestamp_ns

        selected_samples = _filter_samples(
            samples,
            steady_start_timestamp_ns,
            steady_end_timestamp_ns,
        )
        if not selected_samples:
            selected_samples = _filter_samples(samples, start_timestamp_ns, end_timestamp_ns)
            steady_start_timestamp_ns = start_timestamp_ns
            steady_end_timestamp_ns = end_timestamp_ns
            steady_window_fallback_used = True

        sweep_stats.append(
            {
                "label": result.get("label", start.get("label", "")),
                "point": result.get("point", start.get("point", {})),
                "start_timestamp_ns": start_timestamp_ns,
                "end_timestamp_ns": end_timestamp_ns,
                "steady_start_timestamp_ns": steady_start_timestamp_ns,
                "steady_end_timestamp_ns": steady_end_timestamp_ns,
                "steady_window_fallback_used": steady_window_fallback_used,
                "sample_count": len(selected_samples),
                "power": compute_power_stats(selected_samples),
            }
        )

    return sweep_stats


def compute_power_stats(samples: list[TelemetrySample]) -> dict[str, float | None]:
    values = sorted(sample.power_w for sample in samples if sample.power_w is not None)
    if not values:
        return {
            "avg_w": None,
            "p95_w": None,
            "max_w": None,
        }
    return {
        "avg_w": sum(values) / len(values),
        "p95_w": _percentile(values, 95),
        "max_w": max(values),
    }


def _percentile(sorted_values: list[float], percentile: int) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _warmup_seconds(case: CaseSpec) -> float | None:
    value = case.parameters.get("warmup_sec")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _pop_matching_start(
    pending_starts: list[dict[str, Any]],
    end_marker: dict[str, Any],
) -> dict[str, Any] | None:
    end_label = end_marker.get("label")
    end_point = end_marker.get("point")
    for index, start in enumerate(pending_starts):
        if start.get("label") == end_label and start.get("point") == end_point:
            return pending_starts.pop(index)
    return pending_starts.pop(0) if pending_starts else None


def _filter_samples(
    samples: list[TelemetrySample],
    start_timestamp_ns: int,
    end_timestamp_ns: int,
) -> list[TelemetrySample]:
    return [
        sample
        for sample in samples
        if start_timestamp_ns <= sample.timestamp_ns <= end_timestamp_ns
    ]


def parse_workload_results(stdout_path: Path) -> list[dict[str, Any]]:
    if not stdout_path.exists():
        return []

    results: list[dict[str, Any]] = []
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            results.append(parsed)
    return results


def _last_steady_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    for result in reversed(results):
        if result.get("phase") == "steady":
            return result
    return results[-1] if results else None
