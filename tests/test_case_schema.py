from pathlib import Path

import pytest

from runner.case_schema import CaseValidationError, load_case, parse_case


def test_sample_cases_validate() -> None:
    for case_path in Path("cases").rglob("*.yaml"):
        case = load_case(case_path)
        assert case.case_id
        assert case.gpus
        assert case.telemetry.sampler == "mock_nvml"


def test_rejects_unknown_third_party_wrapper() -> None:
    raw = {
        "case_id": "PWR-BAD-001",
        "title": "Bad wrapper",
        "category": "io",
        "duration_s": 1,
        "gpus": [0],
        "workload": {
            "type": "third_party_wrapper",
            "wrapper": "unknown_tool",
        },
    }

    with pytest.raises(CaseValidationError, match="workload.wrapper"):
        parse_case(raw)


def test_accepts_real_nvml_sampler_in_schema() -> None:
    raw = {
        "case_id": "PWR-NVML-001",
        "title": "Real NVML schema",
        "category": "core",
        "duration_s": 1,
        "gpus": [0],
        "workload": {
            "type": "none",
        },
        "telemetry": {
            "sampler": "nvml",
        },
    }

    assert parse_case(raw).telemetry.sampler == "nvml"
