"""Materialize the frozen synthetic tuning and holdout artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.simulation import build_suite, write_suite  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/stage1"))
    parser.add_argument(
        "--suite",
        choices=("tuning", "synthetic_holdout", "all"),
        default="all",
    )
    args = parser.parse_args()
    names = ("tuning", "synthetic_holdout") if args.suite == "all" else (args.suite,)
    for name in names:
        suite = build_suite(name)
        events_path, manifest_path = write_suite(suite, args.output)
        print(f"{name}: {len(suite.events)} events -> {events_path}")
        print(f"{name}: {len(suite.truth)} scenarios -> {manifest_path}")


if __name__ == "__main__":
    main()
