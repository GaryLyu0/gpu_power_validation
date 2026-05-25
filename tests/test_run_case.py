import json
from pathlib import Path

from runner.run_case import run_case


def test_dry_run_writes_mock_result_artifacts(tmp_path: Path) -> None:
    result_dir = run_case(
        case_path=Path("cases/io/PWR-MEM-001.yaml"),
        dry_run=True,
        results_root=tmp_path,
    )

    assert result_dir.exists()
    assert result_dir.joinpath("case.yaml").exists()
    assert result_dir.joinpath("summary.json").exists()
    assert result_dir.joinpath("telemetry.csv").exists()
    assert result_dir.joinpath("stdout.log").exists()
    assert result_dir.joinpath("stderr.log").exists()

    summary = json.loads(result_dir.joinpath("summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run"
    assert summary["dry_run"] is True
    assert summary["case"]["case_id"] == "PWR-MEM-001"
    assert summary["telemetry"]["sampler"] == "mock_nvml"
    assert summary["workload_command"] == [
        "./build/workloads/h2d_d2d_copy",
        "--device",
        "0",
        "--mode",
        "h2d",
        "--bytes",
        "268435456",
        "--warmup-sec",
        "10",
        "--steady-sec",
        "60",
        "--presweep-bytes",
        "4096,65536,1048576,16777216,268435456",
    ]
    assert summary["telemetry"]["power"]["avg_w"] is not None


def test_mock_workload_with_mock_telemetry_runs_without_gpu(tmp_path: Path) -> None:
    result_dir = run_case(
        case_path=Path("cases/idle/PWR-IDLE-001.yaml"),
        dry_run=False,
        results_root=tmp_path,
        mock_workload=True,
        mock_telemetry=True,
    )

    summary = json.loads(result_dir.joinpath("summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["return_code"] == 0
    assert summary["telemetry"]["sampler"] == "mock_nvml"
    assert summary["telemetry"]["power"]["avg_w"] is not None
    assert summary["workload_command"][1:3] == ["-m", "workloads.python.sleep"]
