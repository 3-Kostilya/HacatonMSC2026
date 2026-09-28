"""Build and validate B2 tuning/holdout artifacts; 2021 is always excluded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.b2_validation import validate_b2_suites  # noqa: E402
from stage1.simulation import build_b2_suite, write_suite  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/b2")
    args = parser.parse_args()
    paths = [
        args.output / f"{suite}_{suffix}"
        for suite in ("tuning", "synthetic_holdout")
        for suffix in ("events.jsonl", "truth_manifest.json")
    ]
    report_path = args.output / "b2-validation.json"
    if report_path.exists() or any(path.exists() for path in paths):
        parser.error("B2 output already exists; choose a new directory")
    tuning = build_b2_suite("tuning")
    holdout = build_b2_suite("synthetic_holdout")
    report = validate_b2_suites(tuning, holdout)
    if not report["passed"]:
        raise RuntimeError(f"B2 validation failed: {report}")
    write_suite(tuning, args.output)
    write_suite(holdout, args.output)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": True,
                "tuning_events": len(tuning.events),
                "holdout_events": len(holdout.events),
                "scenarios_per_suite": len(tuning.truth),
                "validation_channels_per_suite": len(tuning.manifest()["validation_channels"]),
                "excluded_source_years": [2021],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
