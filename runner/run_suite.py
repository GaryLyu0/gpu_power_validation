"""CLI entrypoint for running all case YAML files under a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from runner.run_case import run_case


def find_cases(case_dir: Path) -> list[Path]:
    return sorted(path for path in case_dir.rglob("*.yaml") if path.is_file())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path, help="Root case directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate without running")
    parser.add_argument(
        "--mock-workload",
        action="store_true",
        help="Run sleep placeholders instead of case workloads",
    )
    parser.add_argument(
        "--mock-telemetry",
        action="store_true",
        help="Use mock telemetry instead of requiring pynvml/NVML",
    )
    parser.add_argument(
        "--no-clock-control",
        action="store_true",
        help="Disable nvidia-smi clock control even when cases request it",
    )
    parser.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="Directory where result folders are created",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = find_cases(args.case_dir)
    if not cases:
        raise SystemExit(f"No case YAML files found under {args.case_dir}")

    print(f"Discovered {len(cases)} case(s)")
    result_dirs = [
        run_case(
            case_path=case_path,
            dry_run=args.dry_run,
            results_root=args.results_root,
            mock_workload=args.mock_workload,
            mock_telemetry=args.mock_telemetry,
            no_clock_control=args.no_clock_control,
        )
        for case_path in cases
    ]
    print("Suite complete")
    for result_dir in result_dirs:
        print(f"  {result_dir}")


if __name__ == "__main__":
    main()
