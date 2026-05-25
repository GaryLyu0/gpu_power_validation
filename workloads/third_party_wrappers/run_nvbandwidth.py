"""Optional nvbandwidth cross-check helper.

This wrapper does not participate in the primary IO cases yet. It exists only to
record the intended command boundary for future cross-checks against the
independent CUDA microbenchmark.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        default="third_party/nvbandwidth/build/nvbandwidth",
        help="Path to a separately built nvbandwidth executable",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    binary = Path(parsed.binary)
    if not binary.exists():
        raise SystemExit(
            f"nvbandwidth binary not found at {binary}. Build the submodule "
            "separately; this framework does not modify third_party."
        )
    raise SystemExit(subprocess.call([str(binary), *parsed.args]))


if __name__ == "__main__":
    main()
