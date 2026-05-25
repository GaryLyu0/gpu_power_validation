"""Workload command resolution and dry-run launch scaffolding."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from string import Formatter
from typing import Any

from runner.case_schema import CaseSpec


@dataclass(frozen=True)
class ResolvedWorkload:
    command: list[str]
    cwd: Path
    env: dict[str, str]
    timeout_s: int | None


@dataclass(frozen=True)
class ResolvedSweepWorkload:
    label: str
    point: dict[str, Any]
    workload: ResolvedWorkload


THIRD_PARTY_WRAPPERS = {
    "nvbandwidth": ["python", "workloads/third_party_wrappers/nvbandwidth.py"],
    "nccl_tests": ["python", "workloads/third_party_wrappers/nccl_tests.py"],
    "cutlass": ["python", "workloads/third_party_wrappers/cutlass.py"],
    "cudnn_frontend": ["python", "workloads/third_party_wrappers/cudnn_frontend.py"],
}


def resolve_workload(case: CaseSpec, repo_root: Path) -> ResolvedWorkload:
    workload = case.workload
    cwd = (repo_root / workload.cwd).resolve() if workload.cwd else repo_root

    if workload.type == "none":
        command = ["python", "workloads/python/idle_sleep.py", "--seconds", str(case.duration_s)]
    elif workload.type == "command":
        command = workload.command or [workload.executable or "", *workload.args]
    elif workload.type == "python_module":
        command = [sys.executable, "-m", workload.module or "", *workload.args]
    elif workload.type == "third_party_wrapper":
        command = [
            *THIRD_PARTY_WRAPPERS[workload.wrapper or ""],
            *workload.args,
        ]
    else:  # Validation should catch this before resolution.
        raise ValueError(f"Unsupported workload type: {workload.type}")

    return ResolvedWorkload(
        command=command,
        cwd=cwd,
        env=workload.env,
        timeout_s=workload.timeout_s or case.duration_s + 30,
    )


def resolve_sweep_workloads(case: CaseSpec, repo_root: Path) -> list[ResolvedSweepWorkload]:
    resolved = resolve_workload(case, repo_root)
    fields = _template_fields(resolved.command)
    if not fields:
        return [ResolvedSweepWorkload(label="single", point={}, workload=resolved)]

    points = _expand_sweep_points(case.parameters.get("sweep", {}), fields)
    return [
        ResolvedSweepWorkload(
            label=_sweep_label(index, point),
            point=point,
            workload=ResolvedWorkload(
                command=[_format_arg(arg, point) for arg in resolved.command],
                cwd=resolved.cwd,
                env=resolved.env,
                timeout_s=resolved.timeout_s,
            ),
        )
        for index, point in enumerate(points)
    ]


def write_dry_run_logs(result_dir: Path, resolved: ResolvedWorkload) -> None:
    result_dir.joinpath("stdout.log").write_text(
        "DRY RUN: workload was not executed.\n"
        f"Resolved command: {' '.join(resolved.command)}\n",
        encoding="utf-8",
    )
    result_dir.joinpath("stderr.log").write_text("", encoding="utf-8")


def write_dry_run_sweep_logs(
    result_dir: Path, sweep_workloads: list[ResolvedSweepWorkload]
) -> None:
    lines = ["DRY RUN: sweep workloads were not executed."]
    for sweep in sweep_workloads:
        lines.append(f"[{sweep.label}] {' '.join(sweep.workload.command)}")
    result_dir.joinpath("stdout.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result_dir.joinpath("stderr.log").write_text("", encoding="utf-8")


def sweep_plan_to_dict(
    sweep_workloads: list[ResolvedSweepWorkload],
) -> list[dict[str, Any]]:
    return [
        {
            "label": sweep.label,
            "point": sweep.point,
            "command": sweep.workload.command,
        }
        for sweep in sweep_workloads
    ]


def _template_fields(command: list[str]) -> set[str]:
    fields: set[str] = set()
    formatter = Formatter()
    for arg in command:
        for _, field_name, _, _ in formatter.parse(arg):
            if field_name:
                fields.add(field_name)
    return fields


def _expand_sweep_points(sweep: Any, required_fields: set[str]) -> list[dict[str, Any]]:
    if not isinstance(sweep, dict):
        raise ValueError("parameters.sweep must be a mapping when workload args use templates")

    dimensions: list[tuple[str, list[dict[str, Any]]]] = []
    for field in sorted(required_fields):
        values = sweep.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"parameters.sweep.{field} must be a non-empty list")
        dimensions.append((field, [_normalize_sweep_value(item) for item in values]))

    points: list[dict[str, Any]] = []
    for combination in product(*[items for _, items in dimensions]):
        point: dict[str, Any] = {}
        for (field, _), item in zip(dimensions, combination):
            point[field] = item["value"]
            if item.get("label"):
                point[f"{field}_label"] = item["label"]
        points.append(point)
    return points


def _normalize_sweep_value(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        if "value" not in item:
            raise ValueError("sweep item mappings must contain a value field")
        normalized = {"value": str(item["value"])}
        if "label" in item:
            normalized["label"] = str(item["label"])
        elif "level" in item:
            normalized["label"] = str(item["level"])
        return normalized
    return {"value": str(item)}


def _format_arg(arg: str, point: dict[str, Any]) -> str:
    return arg.format(**point)


def _sweep_label(index: int, point: dict[str, Any]) -> str:
    parts = []
    for key, value in point.items():
        if key.endswith("_label"):
            continue
        label = point.get(f"{key}_label", value)
        parts.append(f"{key}-{_safe_label(str(label))}")
    return f"{index:03d}_" + "_".join(parts)


def _safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)

