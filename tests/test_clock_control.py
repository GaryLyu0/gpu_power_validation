import subprocess
from pathlib import Path

from runner.case_schema import load_case
from runner.clock_control import ClockController, expand_sweep_with_clock_plan
from runner.workload_launcher import resolve_sweep_workloads


def test_select_representative_clocks() -> None:
    controller = ClockController()

    selected = controller.select_representative_clocks(
        [300, 600, 900, 1200, 1500, 1800, 2100], count=5
    )

    assert selected == [300, 900, 1200, 1500, 2100]


def test_query_supported_graphics_clocks_from_nvidia_smi_csv() -> None:
    def fake_runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="1593, 900\n1593, 1200\n1593, 1410\n1593, 1590\n",
            stderr="",
        )

    controller = ClockController(command_runner=fake_runner)

    assert controller.query_supported_graphics_clocks(0) == [900, 1200, 1410, 1590]


def test_dry_run_clock_sweep_expands_without_nvidia_smi() -> None:
    case = load_case(Path("cases/core/power_gpu_op_tc_002.yaml"))
    workloads = resolve_sweep_workloads(case, Path.cwd())
    controller = ClockController()

    expanded, plan = expand_sweep_with_clock_plan(
        case=case,
        sweep_workloads=workloads,
        controller=controller,
        dry_run=True,
        no_clock_control=False,
    )

    assert plan.sm_clock_mhz == ["auto_low", "auto_mid", "auto_high"]
    assert len(expanded) == 33
    assert expanded[0].point["sm_clock_mhz"] == "auto_low"
    assert expanded[-1].point["sm_clock_mhz"] == "auto_high"


def test_no_clock_control_override_leaves_workload_sweep_only() -> None:
    case = load_case(Path("cases/core/power_gpu_op_tc_002.yaml"))
    workloads = resolve_sweep_workloads(case, Path.cwd())

    expanded, plan = expand_sweep_with_clock_plan(
        case=case,
        sweep_workloads=workloads,
        controller=ClockController(),
        dry_run=True,
        no_clock_control=True,
    )

    assert not plan.enabled
    assert len(expanded) == 11
