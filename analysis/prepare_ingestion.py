"""Run only milestone 1; config paths are relative to the project root."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.ingestion.pipeline import recover_derived_ingestion, run_ingestion  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "stage1/config/ingestion_pilot.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume", action="store_true", help="Resume a failed .inprogress raw load"
    )
    parser.add_argument(
        "--recover-derived",
        action="store_true",
        help="Finalize a validated interrupted run from committed derived tables",
    )
    parser.add_argument(
        "--partitioned-classification",
        action="store_true",
        help="Bound deduplication memory with hash-partitioned scratch Parquet",
    )
    parser.add_argument("--memory-limit", help="DuckDB memory limit; may increase on resume")
    parser.add_argument(
        "--resume-source-sha256",
        help="Pre-failure SHA-256 for a legacy partial source lacking active_source metadata",
    )
    parser.add_argument(
        "--recovery-classification-strategy",
        choices=("global_window_v1", "partitioned_hash_v1"),
        help="Trusted strategy for recovering a legacy manifest without a checkpoint",
    )
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
    if args.memory_limit:
        config["memory_limit"] = args.memory_limit
    if args.recover_derived:
        if args.resume or args.partitioned_classification or args.resume_source_sha256:
            parser.error(
                "--recover-derived cannot be combined with raw resume/classification options"
            )
        result = recover_derived_ingestion(
            config, trusted_classification_strategy=args.recovery_classification_strategy
        )
    else:
        if args.recovery_classification_strategy:
            parser.error("--recovery-classification-strategy requires --recover-derived")
        result = run_ingestion(
            config,
            progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True),
            resume=args.resume,
            resume_source_sha256=args.resume_source_sha256,
            partitioned_classification=args.partitioned_classification,
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
