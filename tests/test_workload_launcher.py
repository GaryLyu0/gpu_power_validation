from pathlib import Path

from runner.case_schema import CaseSpec, TelemetrySpec, WorkloadSpec
from runner.case_schema import load_case
from runner.workload_launcher import resolve_sweep_workloads, resolve_workload


def test_resolves_idle_to_placeholder_without_gpu_runtime() -> None:
    case = CaseSpec(
        case_id="PWR-IDLE-TEST",
        title="Idle",
        category="idle",
        duration_s=7,
        gpus=[0],
        workload=WorkloadSpec(type="none"),
        telemetry=TelemetrySpec(sampler="mock_nvml"),
    )

    resolved = resolve_workload(case, Path.cwd())

    assert resolved.command == [
        "python",
        "workloads/python/idle_sleep.py",
        "--seconds",
        "7",
    ]
    assert resolved.timeout_s == 37


def test_resolves_third_party_wrapper_command_only() -> None:
    case = CaseSpec(
        case_id="PWR-IO-TEST",
        title="IO",
        category="io",
        duration_s=10,
        gpus=[0],
        workload=WorkloadSpec(
            type="third_party_wrapper",
            wrapper="nvbandwidth",
            args=["--dry-run-placeholder"],
            timeout_s=60,
        ),
    )

    resolved = resolve_workload(case, Path.cwd())

    assert resolved.command == [
        "python",
        "workloads/third_party_wrappers/nvbandwidth.py",
        "--dry-run-placeholder",
    ]
    assert resolved.timeout_s == 60


def test_resolves_read_write_level_sweep_without_gpu_runtime() -> None:
    case = load_case(Path("cases/mem/PWR-MEM-003.yaml"))

    sweep = resolve_sweep_workloads(case, Path.cwd())

    assert len(sweep) == 16
    assert sweep[0].point == {
        "load_factor": "0.125",
        "load_factor_label": "L1",
        "mode": "read",
    }
    assert sweep[0].workload.command == [
        "./build/workloads/read_write_levels",
        "--device",
        "0",
        "--mode",
        "read",
        "--load-factor",
        "0.125",
        "--buffer-mb",
        "4096",
        "--warmup-sec",
        "10",
        "--steady-sec",
        "60",
    ]
    assert sweep[-1].workload.command[4:7] == ["write", "--load-factor", "1.0"]


def test_resolves_base_ai_core_sweeps_without_gpu_runtime() -> None:
    tc_duty = resolve_sweep_workloads(
        load_case(Path("cases/core/power_gpu_op_tc_000.yaml")), Path.cwd()
    )
    tc_spatial = resolve_sweep_workloads(
        load_case(Path("cases/core/power_gpu_op_tc_001.yaml")), Path.cwd()
    )
    cc_fp = resolve_sweep_workloads(
        load_case(Path("cases/core/power_gpu_op_cc_000.yaml")), Path.cwd()
    )
    cc_int = resolve_sweep_workloads(
        load_case(Path("cases/core/power_gpu_op_cc_001.yaml")), Path.cwd()
    )
    tc_clock_duty = resolve_sweep_workloads(
        load_case(Path("cases/core/power_gpu_op_tc_002.yaml")), Path.cwd()
    )

    assert len(tc_duty) == 11
    assert tc_duty[0].workload.command[:2] == [
        "./build/workloads/tensor_core_burn",
        "--device",
    ]
    assert "--duty-cycle" in tc_duty[0].workload.command
    assert tc_duty[-1].workload.command[
        tc_duty[-1].workload.command.index("--duty-cycle") + 1
    ] == "1.0"

    assert len(tc_spatial) == 10
    assert tc_spatial[0].workload.command[
        tc_spatial[0].workload.command.index("--active-sm-fraction") + 1
    ] == "0.1"
    assert tc_spatial[-1].workload.command[
        tc_spatial[-1].workload.command.index("--active-sm-fraction") + 1
    ] == "1.0"

    assert len(cc_fp) == 11
    assert cc_fp[0].workload.command[:5] == [
        "./build/workloads/cuda_core_burn",
        "--device",
        "0",
        "--mode",
        "fp32_fma",
    ]
    assert cc_int[0].workload.command[:5] == [
        "./build/workloads/cuda_core_burn",
        "--device",
        "0",
        "--mode",
        "int32_logic",
    ]
    assert len(tc_clock_duty) == 11
    assert tc_clock_duty[0].workload.command[:2] == [
        "./build/workloads/tensor_core_burn",
        "--device",
    ]


def test_resolves_advanced_memory_sweeps_without_gpu_runtime() -> None:
    tma_direction = resolve_sweep_workloads(
        load_case(Path("cases/mem/PWR-MEM-004.yaml")), Path.cwd()
    )
    tma_compare = resolve_sweep_workloads(
        load_case(Path("cases/mem/PWR-MEM-005.yaml")), Path.cwd()
    )
    l2_sweep = resolve_sweep_workloads(
        load_case(Path("cases/mem/PWR-MEM-006.yaml")), Path.cwd()
    )
    sm_coverage = resolve_sweep_workloads(
        load_case(Path("cases/mem/PWR-MEM-007.yaml")), Path.cwd()
    )

    assert len(tma_direction) == 2
    assert tma_direction[0].workload.command[:5] == [
        "./build/workloads/tma_copy",
        "--device",
        "0",
        "--direction",
        "gm_to_smem",
    ]
    assert len(tma_compare) == 2
    assert tma_compare[-1].workload.command[
        tma_compare[-1].workload.command.index("--mechanism") + 1
    ] == "tma"
    assert [point.point["l2_target"] for point in l2_sweep] == ["low", "medium", "high"]
    assert sm_coverage[0].workload.command[
        sm_coverage[0].workload.command.index("--active-sm-fraction") + 1
    ] == "0.5"
    assert sm_coverage[-1].workload.command[
        sm_coverage[-1].workload.command.index("--active-sm-fraction") + 1
    ] == "1.0"


def test_resolves_real_workload_stubs_without_gpu_runtime() -> None:
    tdp = resolve_sweep_workloads(
        load_case(Path("cases/real_workload/power_tdp_000.yaml")), Path.cwd()
    )
    idle = resolve_sweep_workloads(
        load_case(Path("cases/real_workload/power_idle_000.yaml")), Path.cwd()
    )

    assert len(tdp) == 1
    assert tdp[0].workload.command[:5] == [
        "python",
        "workloads/python/real_workload_stub.py",
        "--mode",
        "tdp_matmul",
        "--device",
    ]
    assert "--matrix-size" in tdp[0].workload.command
    assert len(idle) == 1
    assert idle[0].workload.command[:4] == [
        "python",
        "workloads/python/real_workload_stub.py",
        "--mode",
        "idle",
    ]
