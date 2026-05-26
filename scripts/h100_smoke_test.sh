#!/usr/bin/env bash
set -euo pipefail

python3 -B -m pytest
python3 -m runner.run_suite --case-dir cases --dry-run
CUDA_ARCHITECTURES=90 bash scripts/build_workloads.sh
python3 -m runner.run_case --case cases/idle/PWR-IDLE-001.yaml --mock-workload --mock-telemetry
python3 -m runner.run_case --case cases/io/PWR-MEM-001.yaml --dry-run
python3 -m runner.run_case --case cases/core/power_gpu_op_tc_000.yaml --dry-run
