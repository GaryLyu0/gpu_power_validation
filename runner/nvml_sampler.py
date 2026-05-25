"""NVML and mock telemetry samplers."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any

from runner.case_schema import CaseSpec
from runner.telemetry import TelemetrySample


TELEMETRY_FIELDS = [
    "timestamp_ns",
    "gpu_index",
    "power_w",
    "temperature_c",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "memory_used_mb",
    "memory_total_mb",
]


class NvmlUnavailableError(RuntimeError):
    """Raised when real NVML telemetry cannot be started."""


class _CsvSampler:
    def __init__(self, case: CaseSpec, output_csv: Path) -> None:
        self.case = case
        self.output_csv = output_csv
        self.interval_s = case.telemetry.interval_ms / 1000.0
        self.samples: list[TelemetrySample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: Any | None = None
        self._writer: csv.DictWriter[str] | None = None

    def start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.output_csv.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=TELEMETRY_FIELDS)
        self._writer.writeheader()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[TelemetrySample]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_s * 2.0))
        self._close_csv()
        return self.samples

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval_s)

    def _sample_once(self) -> None:
        for sample in self._read_samples():
            self.samples.append(sample)
            if self._writer:
                self._writer.writerow(sample.__dict__)
        if self._stream:
            self._stream.flush()

    def _read_samples(self) -> list[TelemetrySample]:
        raise NotImplementedError

    def _close_csv(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "_CsvSampler":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()


class MockNvmlSampler(_CsvSampler):
    """Deterministic sampler for dry-run and laptop-safe tests."""

    def collect_mock(self, case: CaseSpec, output_csv: Path) -> list[TelemetrySample]:
        sampler = MockNvmlSampler(case, output_csv)
        sampler._open_for_collect()
        sample_count = max(2, min(5, case.duration_s))
        for step in range(sample_count):
            for sample in sampler._mock_samples(step):
                sampler.samples.append(sample)
                if sampler._writer:
                    sampler._writer.writerow(sample.__dict__)
        sampler._close_csv()
        return sampler.samples

    def _open_for_collect(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.output_csv.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=TELEMETRY_FIELDS)
        self._writer.writeheader()

    def _read_samples(self) -> list[TelemetrySample]:
        return self._mock_samples(len(self.samples))

    def _mock_samples(self, step: int) -> list[TelemetrySample]:
        timestamp_ns = time.time_ns()
        return [
            TelemetrySample(
                timestamp_ns=timestamp_ns,
                gpu_index=gpu_index,
                power_w=70.0 + step,
                temperature_c=35.0 + step,
                graphics_clock_mhz=1410,
                memory_clock_mhz=1593,
                utilization_gpu_percent=0 if self.case.category == "idle" else 10 + step,
                utilization_memory_percent=0 if self.case.category == "idle" else 5 + step,
                memory_used_mb=512.0,
                memory_total_mb=81920.0,
            )
            for gpu_index in self.case.gpus
        ]


class NvmlSampler(_CsvSampler):
    """Real NVML sampler using pynvml, initialized only at runtime."""

    def __init__(self, case: CaseSpec, output_csv: Path) -> None:
        super().__init__(case, output_csv)
        self._nvml: Any | None = None
        self._handles: list[tuple[int, Any]] = []

    def start(self) -> None:
        self._initialize_nvml()
        super().start()

    def stop(self) -> list[TelemetrySample]:
        try:
            return super().stop()
        finally:
            if self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except Exception:
                    pass

    def _initialize_nvml(self) -> None:
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise NvmlUnavailableError(
                "NVML telemetry requires the pynvml package. Install pynvml on "
                "the H100/B200 server or rerun with --mock-telemetry."
            ) from exc

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            self._handles = []
            for gpu_index in self.case.gpus:
                if gpu_index >= device_count:
                    raise NvmlUnavailableError(
                        f"GPU index {gpu_index} is out of range for NVML device "
                        f"count {device_count}. Rerun with --mock-telemetry for "
                        "local validation."
                    )
                self._handles.append((gpu_index, pynvml.nvmlDeviceGetHandleByIndex(gpu_index)))
        except NvmlUnavailableError:
            raise
        except Exception as exc:
            raise NvmlUnavailableError(
                "NVML telemetry could not be initialized. Ensure NVIDIA driver "
                "and NVML are available on the H100/B200 server, or rerun with "
                "--mock-telemetry."
            ) from exc
        self._nvml = pynvml

    def _read_samples(self) -> list[TelemetrySample]:
        assert self._nvml is not None
        timestamp_ns = time.time_ns()
        samples: list[TelemetrySample] = []
        for gpu_index, handle in self._handles:
            util = self._read_or_none(self._nvml.nvmlDeviceGetUtilizationRates, handle)
            memory = self._read_or_none(self._nvml.nvmlDeviceGetMemoryInfo, handle)
            power_mw = self._read_or_none(self._nvml.nvmlDeviceGetPowerUsage, handle)
            temperature = self._read_or_none(
                self._nvml.nvmlDeviceGetTemperature,
                handle,
                self._nvml.NVML_TEMPERATURE_GPU,
            )
            graphics_clock = self._read_or_none(
                self._nvml.nvmlDeviceGetClockInfo,
                handle,
                self._nvml.NVML_CLOCK_GRAPHICS,
            )
            memory_clock = self._read_or_none(
                self._nvml.nvmlDeviceGetClockInfo,
                handle,
                self._nvml.NVML_CLOCK_MEM,
            )
            samples.append(
                TelemetrySample(
                    timestamp_ns=timestamp_ns,
                    gpu_index=gpu_index,
                    power_w=(power_mw / 1000.0) if power_mw is not None else None,
                    temperature_c=float(temperature or 0),
                    graphics_clock_mhz=int(graphics_clock or 0),
                    memory_clock_mhz=int(memory_clock or 0),
                    utilization_gpu_percent=int(getattr(util, "gpu", 0) if util else 0),
                    utilization_memory_percent=int(
                        getattr(util, "memory", 0) if util else 0
                    ),
                    memory_used_mb=(memory.used / 1024 / 1024) if memory else 0.0,
                    memory_total_mb=(memory.total / 1024 / 1024) if memory else 0.0,
                )
            )
        return samples

    @staticmethod
    def _read_or_none(func: Any, *args: Any) -> Any | None:
        try:
            return func(*args)
        except Exception:
            return None
