"""CLI entrypoint for running a single GPU power validation case."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from runner.case_schema import CaseValidationError, load_case
from runner.clock_control import ClockController, expand_sweep_with_clock_plan
from runner.nvml_sampler import MockNvmlSampler, NvmlSampler, NvmlUnavailableError
from runner.result_writer import copy_case_file, create_result_dir, write_summary
from runner.telemetry import TelemetrySampler
from runner.workload_launcher import ResolvedWorkload
from runner.workload_launcher import ResolvedSweepWorkload
from runner.workload_launcher import resolve_sweep_workloads, write_dry_run_logs
from runner.workload_launcher import sweep_plan_to_dict, write_dry_run_sweep_logs


def run_case(
    case_path: Path,
    dry_run: bool,
    results_root: Path,
    mock_workload: bool = False,
    mock_telemetry: bool = False,
    no_clock_control: bool = False,
) -> Path:
    repo_root = Path.cwd()
    case = load_case(case_path)
    sweep_workloads = resolve_sweep_workloads(case, repo_root)
    clock_controller = ClockController()
    sweep_workloads, clock_plan = expand_sweep_with_clock_plan(
        case=case,
        sweep_workloads=sweep_workloads,
        controller=clock_controller,
        dry_run=dry_run,
        no_clock_control=no_clock_control,
    )
    if mock_workload:
        sweep_workloads = [
            ResolvedSweepWorkload(
                label="single",
                point={},
                workload=_mock_workload(case.duration_s, repo_root),
            )
        ]
    resolved = sweep_workloads[0].workload
    telemetry_sampler = _telemetry_sampler_name(dry_run, mock_telemetry)

    print(f"Case: {case.case_id} - {case.title}")
    print(f"Resolved workload command: {' '.join(resolved.command)}")
    if len(sweep_workloads) > 1:
        print(f"Sweep points: {len(sweep_workloads)}")
    print(
        "Telemetry: "
        + json.dumps(
            {
                "sampler": telemetry_sampler,
                "interval_ms": case.telemetry.interval_ms,
                "metrics": case.telemetry.metrics,
            }
        )
    )
    if clock_plan.enabled:
        print(f"Clock plan: {json.dumps(clock_plan.to_dict())}")

    result_dir = create_result_dir(results_root, case.case_id)
    copy_case_file(case_path, result_dir)
    telemetry_csv = result_dir / "telemetry.csv"

    if dry_run:
        samples = MockNvmlSampler(case, telemetry_csv).collect_mock(case, telemetry_csv)
        if len(sweep_workloads) > 1:
            write_dry_run_sweep_logs(result_dir, sweep_workloads)
        else:
            write_dry_run_logs(result_dir, resolved)
        status = "dry_run"
        return_code = None
    else:
        sampler = _build_sampler(case, telemetry_csv, mock_telemetry)
        try:
            sampler.start()
        except NvmlUnavailableError as exc:
            raise SystemExit(str(exc)) from exc

        try:
            return_code = _run_sweep_workloads(
                sweep_workloads=sweep_workloads,
                result_dir=result_dir,
                case=case,
                clock_controller=clock_controller,
                clock_control_enabled=clock_plan.enabled and not no_clock_control,
            )
        finally:
            samples = sampler.stop()
            if clock_plan.enabled and not no_clock_control:
                clock_controller.reset(case.gpus)
        status = "passed" if return_code == 0 else "failed"

    write_summary(
        result_dir=result_dir,
        case=case,
        resolved=resolved,
        status=status,
        sample_count=len(samples),
        samples=samples,
        telemetry_sampler=telemetry_sampler,
        return_code=return_code,
        sweep_plan=sweep_plan_to_dict(sweep_workloads),
    )
    print(f"Result directory: {result_dir}")
    return result_dir


def _build_sampler(
    case, telemetry_csv: Path, mock_telemetry: bool
) -> TelemetrySampler:
    if mock_telemetry:
        return MockNvmlSampler(case, telemetry_csv)
    return NvmlSampler(case, telemetry_csv)


def _telemetry_sampler_name(dry_run: bool, mock_telemetry: bool) -> str:
    return "mock_nvml" if dry_run or mock_telemetry else "nvml"


def _mock_workload(duration_s: int, repo_root: Path) -> ResolvedWorkload:
    seconds = str(min(duration_s, 5))
    return ResolvedWorkload(
        command=[sys.executable, "-m", "workloads.python.sleep", "--seconds", seconds],
        cwd=repo_root,
        env={},
        timeout_s=int(float(seconds)) + 30,
    )


def _run_workload(resolved: ResolvedWorkload, result_dir: Path) -> int:
    return _run_sweep_workloads(
        sweep_workloads=[ResolvedSweepWorkload(label="single", point={}, workload=resolved)],
        result_dir=result_dir,
        case=None,
        clock_controller=None,
        clock_control_enabled=False,
    )


def _run_sweep_workloads(
    sweep_workloads: list[ResolvedSweepWorkload],
    result_dir: Path,
    case,
    clock_controller: ClockController | None,
    clock_control_enabled: bool,
) -> int:
    env = os.environ.copy()
    with result_dir.joinpath("stdout.log").open("w", encoding="utf-8") as stdout:
        with result_dir.joinpath("stderr.log").open("w", encoding="utf-8") as stderr:
            return_codes = []
            for sweep in sweep_workloads:
                if clock_control_enabled and case and clock_controller:
                    clock_controller.apply_sm_clock(case.gpus, sweep.point.get("sm_clock_mhz"))
                workload_env = env.copy()
                workload_env.update(sweep.workload.env)
                stdout.write(
                    json.dumps(
                        {
                            "phase": "runner_sweep_point",
                            "label": sweep.label,
                            "point": sweep.point,
                        }
                    )
                    + "\n"
                )
                stdout.flush()
                start_timestamp_ns = time.time_ns()
                stdout.write(
                    json.dumps(
                        {
                            "phase": "runner_sweep_point_start",
                            "label": sweep.label,
                            "point": sweep.point,
                            "timestamp_ns": start_timestamp_ns,
                        }
                    )
                    + "\n"
                )
                stdout.flush()
                return_code = -1
                try:
                    completed = subprocess.run(
                        sweep.workload.command,
                        cwd=sweep.workload.cwd,
                        env=workload_env,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=sweep.workload.timeout_s,
                        check=False,
                    )
                    return_code = completed.returncode
                    return_codes.append(return_code)
                except subprocess.TimeoutExpired:
                    stderr.write(
                        f"Workload {sweep.label} timed out after "
                        f"{sweep.workload.timeout_s} seconds\n"
                    )
                    return_codes.append(return_code)
                finally:
                    end_timestamp_ns = time.time_ns()
                    stdout.write(
                        json.dumps(
                            {
                                "phase": "runner_sweep_point_end",
                                "label": sweep.label,
                                "point": sweep.point,
                                "timestamp_ns": end_timestamp_ns,
                                "return_code": return_code,
                            }
                        )
                        + "\n"
                    )
                    stdout.flush()
            return 0 if all(code == 0 for code in return_codes) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path, help="Path to case YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate without running")
    parser.add_argument(
        "--mock-workload",
        action="store_true",
        help="Run a sleep placeholder instead of the case workload",
    )
    parser.add_argument(
        "--mock-telemetry",
        action="store_true",
        help="Use mock telemetry instead of requiring pynvml/NVML",
    )
    parser.add_argument(
        "--no-clock-control",
        action="store_true",
        help="Disable nvidia-smi clock control even when the case requests it",
    )
    parser.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="Directory where result folders are created",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_case(
            args.case,
            args.dry_run,
            args.results_root,
            mock_workload=args.mock_workload,
            mock_telemetry=args.mock_telemetry,
            no_clock_control=args.no_clock_control,
        )
    except CaseValidationError as exc:
        raise SystemExit(f"Invalid case: {exc}") from exc


if __name__ == "__main__":
    main()
