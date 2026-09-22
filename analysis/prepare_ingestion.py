"""Run only milestone 1; config paths are relative to the project root."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.ingestion.pipeline import run_ingestion  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "stage1/config/ingestion_pilot.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Read all supplied sources; ignores per-source pilot limits",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("channels", "objects", "output"):
        config[name] = str((ROOT / config[name]).resolve())
    for item in config["sources"]:
        item["path"] = str((ROOT / item["path"]).resolve())
        if args.full:
            item["max_rows"] = None
    if args.output:
        config["output"] = str(args.output.resolve())
    result = run_ingestion(
        config, progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True)
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "scope",
                    "input_rows",
                    "dispositions",
                    "parquet_bytes",
                    "sanity_checks",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
