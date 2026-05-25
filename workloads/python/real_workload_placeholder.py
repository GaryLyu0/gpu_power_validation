"""Real-workload placeholder."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    print(f"placeholder real workload for {args.seconds} seconds")


if __name__ == "__main__":
    main()

