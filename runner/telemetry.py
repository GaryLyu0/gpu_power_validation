"""Telemetry sampler interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from runner.case_schema import CaseSpec


@dataclass(frozen=True)
class TelemetrySample:
    timestamp_ns: int
    gpu_index: int
    power_w: float | None
    temperature_c: float
    graphics_clock_mhz: int
    memory_clock_mhz: int
    utilization_gpu_percent: int
    utilization_memory_percent: int
    memory_used_mb: float
    memory_total_mb: float


class TelemetrySampler(Protocol):
    def start(self) -> None:
        """Start sampling and writing telemetry."""

    def stop(self) -> list[TelemetrySample]:
        """Stop sampling and return collected samples."""
