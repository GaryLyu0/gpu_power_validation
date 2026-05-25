"""Dataclass-based case schema and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise SystemExit(
        "PyYAML is required to read case files. Install with: pip install -e ."
    ) from exc


VALID_CATEGORIES = {"idle", "io", "mem", "core", "real_workload"}
VALID_WORKLOAD_TYPES = {"command", "python_module", "third_party_wrapper", "none"}
VALID_TELEMETRY_SAMPLERS = {"mock_nvml", "nvml"}
VALID_THIRD_PARTY_WRAPPERS = {
    "nvbandwidth",
    "nccl_tests",
    "cutlass",
    "cudnn_frontend",
}


ClockValue = Union[int, str, list[Union[int, str]], None]


@dataclass(frozen=True)
class WorkloadSpec:
    type: str
    executable: str | None = None
    command: list[str] = field(default_factory=list)
    module: str | None = None
    args: list[str] = field(default_factory=list)
    wrapper: str | None = None
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: int | None = None


@dataclass(frozen=True)
class TelemetrySpec:
    sampler: str = "mock_nvml"
    interval_ms: int = 1000
    metrics: list[str] = field(
        default_factory=lambda: [
            "timestamp_s",
            "gpu_index",
            "power_watts",
            "temperature_c",
            "sm_clock_mhz",
            "memory_clock_mhz",
            "utilization_gpu_pct",
            "utilization_memory_pct",
        ]
    )


@dataclass(frozen=True)
class ClockSpec:
    enabled: bool = False
    mode: str = "none"
    sm_clock_mhz: ClockValue = None
    mem_clock_mhz: ClockValue = None
    power_limit_watts: int | None = None


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    category: str
    duration_s: int
    gpus: list[int]
    workload: WorkloadSpec
    telemetry: TelemetrySpec = field(default_factory=TelemetrySpec)
    clocks: ClockSpec | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    feature_name: str = ""
    case_name: str = ""
    platform: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class CaseValidationError(ValueError):
    """Raised when a case YAML does not match the expected schema."""


def load_case(path: Path) -> CaseSpec:
    """Load and validate a YAML case file."""
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise CaseValidationError(f"{path} must contain a YAML mapping")
    return parse_case(raw, source=path)


def parse_case(raw: dict[str, Any], source: Path | None = None) -> CaseSpec:
    required = ["case_id", "category", "workload"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise CaseValidationError(f"Missing required field(s): {', '.join(missing)}")

    workload = _parse_workload(_expect_mapping(raw["workload"], "workload"))
    telemetry_raw = _expect_mapping(raw.get("telemetry", {}), "telemetry")
    clocks_raw = raw.get("clock_control", raw.get("clocks"))
    parameters_raw = _expect_mapping(raw.get("parameters", {}), "parameters")
    platform_raw = _expect_mapping(raw.get("platform", {}), "platform")

    case = CaseSpec(
        case_id=_expect_str(raw["case_id"], "case_id"),
        title=_expect_str(raw.get("title", raw.get("case_name")), "title"),
        category=_expect_str(raw["category"], "category"),
        description=_expect_text(raw.get("description", ""), "description"),
        duration_s=_case_duration(raw, parameters_raw),
        gpus=_case_gpus(raw, platform_raw),
        workload=workload,
        telemetry=_parse_telemetry(telemetry_raw),
        clocks=_parse_clocks(clocks_raw) if clocks_raw is not None else None,
        tags=_expect_str_list(raw.get("tags", []), "tags"),
        feature_name=_expect_text(raw.get("feature_name", ""), "feature_name"),
        case_name=_expect_text(raw.get("case_name", raw.get("title", "")), "case_name"),
        platform=platform_raw,
        preconditions=_expect_str_list(raw.get("preconditions", []), "preconditions"),
        parameters=parameters_raw,
        expected=_expect_mapping(raw.get("expected", {}), "expected"),
        notes=_expect_str_list(raw.get("notes", []), "notes"),
    )
    _validate_case(case, source=source)
    return case


def case_to_dict(case: CaseSpec) -> dict[str, Any]:
    data: dict[str, Any] = {
        "case_id": case.case_id,
        "title": case.title,
        "category": case.category,
        "description": case.description,
        "duration_s": case.duration_s,
        "gpus": case.gpus,
        "workload": {
            "type": case.workload.type,
            "executable": case.workload.executable,
            "command": case.workload.command,
            "module": case.workload.module,
            "args": case.workload.args,
            "wrapper": case.workload.wrapper,
            "cwd": case.workload.cwd,
            "env": case.workload.env,
            "timeout_s": case.workload.timeout_s,
        },
        "telemetry": {
            "sampler": case.telemetry.sampler,
            "interval_ms": case.telemetry.interval_ms,
            "metrics": case.telemetry.metrics,
        },
        "tags": case.tags,
        "feature_name": case.feature_name,
        "case_name": case.case_name,
        "platform": case.platform,
        "preconditions": case.preconditions,
        "parameters": case.parameters,
        "expected": case.expected,
        "notes": case.notes,
    }
    if case.clocks:
        data["clock_control"] = {
            "enabled": case.clocks.enabled,
            "mode": case.clocks.mode,
            "sm_clock_mhz": case.clocks.sm_clock_mhz,
            "mem_clock_mhz": case.clocks.mem_clock_mhz,
            "power_limit_watts": case.clocks.power_limit_watts,
        }
    return _drop_none(data)


def _parse_workload(raw: dict[str, Any]) -> WorkloadSpec:
    workload_type = _expect_str(raw.get("type", "command"), "workload.type")
    return WorkloadSpec(
        type=workload_type,
        executable=_optional_str(raw.get("executable"), "workload.executable"),
        command=_expect_str_list(raw.get("command", []), "workload.command"),
        module=_optional_str(raw.get("module"), "workload.module"),
        args=_expect_str_list(raw.get("args", []), "workload.args"),
        wrapper=_optional_str(raw.get("wrapper"), "workload.wrapper"),
        cwd=_optional_str(raw.get("cwd"), "workload.cwd"),
        env=_expect_str_dict(raw.get("env", {}), "workload.env"),
        timeout_s=_optional_positive_int(raw.get("timeout_s"), "workload.timeout_s"),
    )


def _parse_telemetry(raw: dict[str, Any]) -> TelemetrySpec:
    return TelemetrySpec(
        sampler=_expect_str(raw.get("sampler", "mock_nvml"), "telemetry.sampler"),
        interval_ms=_expect_positive_int(
            raw.get("interval_ms", 1000), "telemetry.interval_ms"
        ),
        metrics=_expect_str_list(
            raw.get("metrics", TelemetrySpec().metrics), "telemetry.metrics"
        ),
    )


def _parse_clocks(raw: Any) -> ClockSpec:
    mapping = _expect_mapping(raw, "clocks")
    return ClockSpec(
        enabled=_expect_bool(mapping.get("enabled", False), "clock_control.enabled"),
        mode=_expect_text(mapping.get("mode", "none"), "clock_control.mode"),
        sm_clock_mhz=_optional_clock_value(
            mapping.get("lock_graphics_mhz", mapping.get("sm_clock_mhz")),
            "clock_control.sm_clock_mhz",
        ),
        mem_clock_mhz=_optional_clock_value(
            mapping.get("lock_memory_mhz", mapping.get("mem_clock_mhz")),
            "clock_control.mem_clock_mhz",
        ),
        power_limit_watts=_optional_positive_int(
            mapping.get("power_limit_watts"), "clock_control.power_limit_watts"
        ),
    )


def _validate_case(case: CaseSpec, source: Path | None) -> None:
    prefix = f"{source}: " if source else ""
    if case.category not in VALID_CATEGORIES:
        raise CaseValidationError(
            f"{prefix}category must be one of {sorted(VALID_CATEGORIES)}"
        )
    if case.workload.type not in VALID_WORKLOAD_TYPES:
        raise CaseValidationError(
            f"{prefix}workload.type must be one of {sorted(VALID_WORKLOAD_TYPES)}"
        )
    if case.telemetry.sampler not in VALID_TELEMETRY_SAMPLERS:
        raise CaseValidationError(
            f"{prefix}telemetry.sampler must be one of "
            f"{sorted(VALID_TELEMETRY_SAMPLERS)}"
        )
    if (
        case.workload.type == "command"
        and not case.workload.command
        and not case.workload.executable
    ):
        raise CaseValidationError(
            f"{prefix}command workloads require workload.command or workload.executable"
        )
    if case.workload.type == "python_module" and not case.workload.module:
        raise CaseValidationError(
            f"{prefix}python_module workloads require workload.module"
        )
    if case.workload.type == "third_party_wrapper" and not case.workload.wrapper:
        raise CaseValidationError(
            f"{prefix}third_party_wrapper workloads require workload.wrapper"
        )
    if (
        case.workload.type == "third_party_wrapper"
        and case.workload.wrapper not in VALID_THIRD_PARTY_WRAPPERS
    ):
        raise CaseValidationError(
            f"{prefix}workload.wrapper must be one of "
            f"{sorted(VALID_THIRD_PARTY_WRAPPERS)}"
        )


def _expect_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CaseValidationError(f"{field_name} must be a mapping")
    return value


def _expect_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CaseValidationError(f"{field_name} must be a non-empty string")
    return value


def _expect_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise CaseValidationError(f"{field_name} must be a boolean")
    return value


def _expect_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise CaseValidationError(f"{field_name} must be a string")
    return value


def _case_duration(raw: dict[str, Any], parameters: dict[str, Any]) -> int:
    if "duration_s" in raw:
        return _expect_positive_int(raw["duration_s"], "duration_s")
    warmup_sec = parameters.get("warmup_sec", 0)
    steady_sec = parameters.get("steady_sec")
    if not isinstance(warmup_sec, int) or warmup_sec < 0:
        raise CaseValidationError("parameters.warmup_sec must be a non-negative integer")
    if not isinstance(steady_sec, int) or steady_sec <= 0:
        raise CaseValidationError("parameters.steady_sec must be a positive integer")
    return max(1, warmup_sec + steady_sec)


def _case_gpus(raw: dict[str, Any], platform: dict[str, Any]) -> list[int]:
    if "gpus" in raw:
        return _expect_int_list(raw["gpus"], "gpus")
    if "gpus" in platform:
        return _expect_int_list(platform["gpus"], "platform.gpus")
    return [0]


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _expect_str(value, field_name)


def _expect_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise CaseValidationError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _expect_positive_int(value, field_name)


def _optional_clock_value(value: Any, field_name: str) -> ClockValue:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise CaseValidationError(f"{field_name} must be positive")
        return value
    if isinstance(value, str):
        if not value:
            raise CaseValidationError(f"{field_name} must not be empty")
        return value
    if isinstance(value, list):
        if not value:
            raise CaseValidationError(f"{field_name} must not be empty")
        normalized: list[int | str] = []
        for item in value:
            if isinstance(item, int):
                if item <= 0:
                    raise CaseValidationError(f"{field_name} values must be positive")
                normalized.append(item)
            elif isinstance(item, str) and item:
                normalized.append(item)
            else:
                raise CaseValidationError(
                    f"{field_name} must contain only positive integers or strings"
                )
        return normalized
    raise CaseValidationError(
        f"{field_name} must be a positive integer, string, or list"
    )


def _expect_int_list(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise CaseValidationError(f"{field_name} must be a list of integers")
    if not value:
        raise CaseValidationError(f"{field_name} must not be empty")
    return value


def _expect_str_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CaseValidationError(f"{field_name} must be a list of strings")
    return value


def _expect_str_dict(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise CaseValidationError(f"{field_name} must be a string-to-string mapping")
    return value


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item)
            for key, item in value.items()
            if item is not None and item != {}
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value
