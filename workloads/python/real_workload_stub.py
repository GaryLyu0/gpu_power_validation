"""Pluggable real-workload stub for telemetry and runner integration.

The idle mode has no GPU dependency. The tdp_matmul mode uses PyTorch CUDA when
available and exits with a clear JSON error when it is not available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["idle", "tdp_matmul"], required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-sec", type=float, default=30.0)
    parser.add_argument("--steady-sec", type=float, default=60.0)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    if args.mode == "idle":
        run_idle(args)
        return
    run_tdp_matmul(args)


def run_idle(args: argparse.Namespace) -> None:
    total_sec = max(0.0, args.warmup_sec) + max(0.0, args.steady_sec)
    time.sleep(total_sec)
    _print_json(
        {
            "workload": "real_workload_stub",
            "mode": "idle",
            "phase": "steady",
            "warmup_sec": args.warmup_sec,
            "steady_sec": args.steady_sec,
            "status": "completed",
            "throughput_iter_per_sec": 0.0,
        }
    )


def run_tdp_matmul(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError:
        _print_json(
            {
                "workload": "real_workload_stub",
                "mode": "tdp_matmul",
                "status": "unavailable",
                "error": "PyTorch is not installed. Install torch on the H100/B200 server or replace the workload command.",
            },
            stream=sys.stderr,
        )
        raise SystemExit(2)

    if not torch.cuda.is_available():
        _print_json(
            {
                "workload": "real_workload_stub",
                "mode": "tdp_matmul",
                "status": "unavailable",
                "error": "PyTorch CUDA is not available. Run on the H100/B200 server or replace the workload command.",
            },
            stream=sys.stderr,
        )
        raise SystemExit(2)

    torch.cuda.set_device(args.device)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    size = args.matrix_size
    a = torch.randn((size, size), device=f"cuda:{args.device}", dtype=dtype)
    b = torch.randn((size, size), device=f"cuda:{args.device}", dtype=dtype)
    c = torch.empty((size, size), device=f"cuda:{args.device}", dtype=dtype)

    _run_matmul_window(torch, a, b, c, max(0.0, args.warmup_sec))
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    iterations = _run_matmul_window(torch, a, b, c, args.steady_sec)
    stop.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(stop)
    elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
    ops_per_iter = 2.0 * size * size * size
    tflops = (ops_per_iter * iterations) / elapsed_s / 1.0e12

    _print_json(
        {
            "workload": "real_workload_stub",
            "mode": "tdp_matmul",
            "phase": "steady",
            "device": args.device,
            "dtype": args.dtype,
            "matrix_size": size,
            "warmup_sec": args.warmup_sec,
            "steady_sec": args.steady_sec,
            "elapsed_ms": elapsed_ms,
            "iterations": iterations,
            "throughput_tflops": tflops,
            "status": "completed",
        }
    )


def _run_matmul_window(torch: Any, a: Any, b: Any, c: Any, seconds: float) -> int:
    deadline = time.monotonic() + max(0.0, seconds)
    iterations = 0
    while time.monotonic() < deadline:
        torch.matmul(a, b, out=c)
        iterations += 1
    return iterations


def _print_json(payload: dict[str, Any], stream: Any = sys.stdout) -> None:
    print(json.dumps(payload, sort_keys=True), file=stream, flush=True)


if __name__ == "__main__":
    main()
