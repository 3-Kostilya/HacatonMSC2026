"""Reproduce B's bounded M1 audit. Never reads the excluded year 2021."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.audit_m1_handoff import YEARS, audit_handoff  # noqa: E402
from stage1.ingestion.pipeline import run_ingestion  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=100000)
    args = parser.parse_args()
    if args.max_rows < 1:
        parser.error("max-rows must be positive")
    if args.output.exists():
        parser.error("choose a new output directory")
    data = args.data_root.resolve()
    config = {
        "channels": str(data / "datasets/справочник_каналов_датчиков.csv"),
        "objects": str(data / "datasets/справочник_объектов_диспетчер.csv"),
        "output": str(args.output.resolve() / "handoff"),
        "memory_limit": "512MB",
        "batch_size": 25000,
        "sources": [
            {"path": str(data / "datasets/журнал_событий_пример.csv"), "max_rows": args.max_rows},
            *[
                {"path": str(data / f"zip_files/ext-journal-{year}.7z"), "max_rows": args.max_rows}
                for year in YEARS
            ],
        ],
    }
    run_ingestion(
        config, progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True)
    )
    report = audit_handoff(config["output"])
    report["limitations"] = [
        "Bounded prefixes only; this is B1, not full-scale acceptance of A1",
        "2021 excluded by user instruction; no source for 2021 opened",
        "CSV/7z transport reused; audit does not reuse normalization or reporting",
        "No detector tuning, pseudo-labeling or forecasting performed",
    ]
    destination = args.output / "b-audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
