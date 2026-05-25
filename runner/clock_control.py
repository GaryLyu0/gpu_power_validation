"""GPU clock control using nvidia-smi with dry-run safe planning."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence

from runner.case_schema import CaseSpec
from runner.workload_launcher import ResolvedSweepWorkload

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ClockPlan:
    enabled: bool
    mode: str
    sm_clock_mhz: list[int | str]
    mem_clock_mhz: list[int | str]
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "sm_clock_mhz": self.sm_clock_mhz,
            "mem_clock_mhz": self.mem_clock_mhz,
            "dry_run": self.dry_run,
        }


class ClockController:
    def __init__(
        self,
        nvidia_smi: str = "nvidia-smi",
        command_runner: CommandRunner = subprocess.run,
    ) -> None:
        self.nvidia_smi = nvidia_smi
        self.command_runner = command_runner

    def describe(self, case: CaseSpec, dry_run: bool = True) -> dict[str, object]:
        plan = self.plan(case, dry_run=dry_run)
        return plan.to_dict() if plan.enabled else {}

    def plan(self, case: CaseSpec, dry_run: bool) -> ClockPlan:
        if not case.clocks or not case.clocks.enabled:
            return ClockPlan(False, "none", [], [], dry_run)

        sm_values = self.resolve_sm_clocks(case, dry_run=dry_run)
        mem_values = _as_list(case.clocks.mem_clock_mhz)
        return ClockPlan(
            enabled=True,
            mode=case.clocks.mode,
            sm_clock_mhz=sm_values,
            mem_clock_mhz=mem_values,
            dry_run=dry_run,
        )

    def resolve_sm_clocks(self, case: CaseSpec, dry_run: bool) -> list[int | str]:
        if not case.clocks:
            return []
        raw_values = _as_list(case.clocks.sm_clock_mhz)
        if not raw_values or raw_values == ["auto"]:
            if dry_run:
                return ["auto_representative"]
            return self.select_representative_clocks(
                self.query_supported_graphics_clocks(case.gpus[0])
            )

        if not any(_is_auto_token(value) for value in raw_values):
            return raw_values

        if dry_run:
            return raw_values

        representative = self.select_representative_clocks(
            self.query_supported_graphics_clocks(case.gpus[0])
        )
        return [_resolve_auto_token(value, representative) for value in raw_values]

    def query_supported_graphics_clocks(self, gpu_index: int) -> list[int]:
        commands = [
            [
                self.nvidia_smi,
                "-i",
                str(gpu_index),
                "--query-supported-clocks=mem,gr",
                "--format=csv,noheader,nounits",
            ],
            [
                self.nvidia_smi,
                "-i",
                str(gpu_index),
                "--query-supported-clocks=graphics",
                "--format=csv,noheader,nounits",
            ],
        ]
        errors: list[str] = []
        for command in commands:
            completed = self.command_runner(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                clocks = _parse_clock_query(completed.stdout)
                if clocks:
                    return clocks
            errors.append((completed.stderr or completed.stdout or "").strip())
        raise RuntimeError(
            "Unable to query supported GPU clocks with nvidia-smi. "
            f"Last errors: {' | '.join(error for error in errors if error)}"
        )

    def select_representative_clocks(
        self, supported_clocks: Sequence[int], count: int = 5
    ) -> list[int]:
        unique = sorted(set(supported_clocks))
        if not unique:
            raise RuntimeError("No supported graphics clocks were reported")
        if len(unique) <= count:
            return unique
        return [unique[round(index * (len(unique) - 1) / (count - 1))] for index in range(count)]

    def apply_sm_clock(self, gpu_indices: Sequence[int], sm_clock_mhz: int | str | None) -> None:
        if sm_clock_mhz is None or isinstance(sm_clock_mhz, str):
            return
        for gpu_index in gpu_indices:
            command = self._privileged_command(
                [self.nvidia_smi, "-i", str(gpu_index), "-lgc", f"{sm_clock_mhz},{sm_clock_mhz}"]
            )
            completed = self.command_runner(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                _warn(
                    "Failed to lock GPU clock with nvidia-smi. "
                    "Clock control may require root/admin privileges or sudo. "
                    f"Command: {' '.join(command)}. "
                    f"stderr: {(completed.stderr or completed.stdout).strip()}"
                )

    def reset(self, gpu_indices: Sequence[int]) -> None:
        for gpu_index in gpu_indices:
            command = self._privileged_command([self.nvidia_smi, "-i", str(gpu_index), "-rgc"])
            completed = self.command_runner(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                _warn(
                    "Failed to reset GPU clocks with nvidia-smi. "
                    "Manual reset may be required on the target server. "
                    f"Command: {' '.join(command)}. "
                    f"stderr: {(completed.stderr or completed.stdout).strip()}"
                )

    def _privileged_command(self, command: list[str]) -> list[str]:
        if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0:
            return command
        sudo = shutil.which("sudo")
        if os.name != "nt" and sudo:
            return [sudo, "-n", *command]
        return command


def expand_sweep_with_clock_plan(
    case: CaseSpec,
    sweep_workloads: list[ResolvedSweepWorkload],
    controller: ClockController,
    dry_run: bool,
    no_clock_control: bool,
) -> tuple[list[ResolvedSweepWorkload], ClockPlan]:
    if no_clock_control:
        return sweep_workloads, ClockPlan(False, "disabled_by_cli", [], [], dry_run)

    plan = controller.plan(case, dry_run=dry_run)
    if not plan.enabled or case.clocks is None:
        return sweep_workloads, plan

    if case.clocks.mode != "sweep":
        clock_value = plan.sm_clock_mhz[0] if plan.sm_clock_mhz else None
        if clock_value is None:
            return sweep_workloads, plan
        return [
            ResolvedSweepWorkload(
                label=sweep.label,
                point={**sweep.point, "sm_clock_mhz": clock_value},
                workload=sweep.workload,
            )
            for sweep in sweep_workloads
        ], plan

    clock_values = plan.sm_clock_mhz or [None]
    expanded: list[ResolvedSweepWorkload] = []
    for clock_value, sweep in product(clock_values, sweep_workloads):
        point = dict(sweep.point)
        point["sm_clock_mhz"] = clock_value
        label = f"sm_clock-{_safe_label(str(clock_value))}_{sweep.label}"
        expanded.append(
            ResolvedSweepWorkload(
                label=label,
                point=point,
                workload=sweep.workload,
            )
        )
    return expanded, plan


def _as_list(value: object) -> list[int | str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (int, str)):
        return [value]
    return []


def _is_auto_token(value: int | str) -> bool:
    return isinstance(value, str) and value.startswith("auto")


def _resolve_auto_token(value: int | str, representative: list[int]) -> int:
    if isinstance(value, int):
        return value
    if value == "auto_low":
        return representative[0]
    if value == "auto_mid":
        return representative[len(representative) // 2]
    if value == "auto_high":
        return representative[-1]
    if value in {"auto", "auto_representative"}:
        return representative[len(representative) // 2]
    raise RuntimeError(f"Unknown clock auto token: {value}")


def _parse_clock_query(output: str) -> list[int]:
    clocks: list[int] = []
    for line in output.splitlines():
        values = [int(match) for match in re.findall(r"\d+", line)]
        if values:
            clocks.append(values[-1])
    return sorted(set(clocks))


def _safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)
