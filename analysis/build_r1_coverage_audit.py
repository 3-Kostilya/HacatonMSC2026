"""Audit observed channel bounds without inventing continuous coverage.

The result describes where events are present in the published M1 history.
It cannot establish channel validity periods or prove that silent intervals
contain no missing journal records.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_r1_state_mapping import _sha256, _validated_clean_files  # noqa: E402
from stage1.ingestion.dictionaries import load_dictionaries  # noqa: E402


COVERAGE_VERSION = "r1-observation-bounds-v1"
BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("channel_id", pa.string()),
        pa.field("sensor_type", pa.string()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("year", pa.int16(), nullable=False),
        pa.field("first_observed_at", pa.timestamp("us"), nullable=False),
        pa.field("last_observed_at", pa.timestamp("us"), nullable=False),
        pa.field("event_rows", pa.int64(), nullable=False),
        pa.field("active_days", pa.int32(), nullable=False),
        pa.field("fault_text_rows", pa.int64(), nullable=False),
        pa.field("normal_text_rows", pa.int64(), nullable=False),
        pa.field("channel_in_current_dictionary", pa.bool_(), nullable=False),
        pa.field("historical_applicability_status", pa.string(), nullable=False),
        pa.field("continuous_observation_status", pa.string(), nullable=False),
    ]
)


def _missing_months(quality: dict[str, Any]) -> list[str]:
    present = {(int(row["year"]), int(row["month"])) for row in quality["by_partition"]}
    if not present:
        raise ValueError("M1 quality report has no calendar partitions")
    year, month = min(present)
    last = max(present)
    missing = []
    while (year, month) <= last:
        if (year, month) not in present:
            missing.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return missing


def _observed_bounds(files: list[Path], spill_dir: Path) -> pa.Table:
    spill_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='4GB'")
        connection.execute("SET threads=2")
        connection.execute("SET temp_directory=?", [str(spill_dir)])
        return connection.execute(
            """
            SELECT channel_id, sensor_type, source,
                   CAST(year(timestamp) AS SMALLINT) AS year,
                   min(timestamp) AS first_observed_at,
                   max(timestamp) AS last_observed_at,
                   count(*)::BIGINT AS event_rows,
                   count(DISTINCT CAST(timestamp AS DATE))::INTEGER AS active_days,
                   count(*) FILTER (WHERE value_state = 'Неисправен')::BIGINT
                       AS fault_text_rows,
                   count(*) FILTER (WHERE value_state = 'Норма')::BIGINT
                       AS normal_text_rows
            FROM read_parquet(?, hive_partitioning=false)
            GROUP BY 1, 2, 3, 4
            ORDER BY 1, 2, 3, 4
            """,
            [[str(path) for path in files]],
        ).to_arrow_table()
    finally:
        connection.close()
        if not any(spill_dir.iterdir()):
            spill_dir.rmdir()


def build(
    *,
    input_manifest: Path,
    output: Path,
    channels_dictionary: Path | None = None,
    objects_dictionary: Path | None = None,
) -> dict[str, Any]:
    """Publish observed bounds and an explicit no-continuity conclusion."""

    started = time.monotonic()
    input_manifest = input_manifest.resolve()
    output = output.resolve()
    pending = output.with_name(output.name + ".inprogress")
    if output.exists() or pending.exists():
        raise FileExistsError("coverage output or .inprogress directory already exists")
    m1 = json.loads(input_manifest.read_text(encoding="utf-8"))
    if (
        m1.get("schema_version") != "ingestion-v1"
        or m1.get("status") != "complete"
        or m1.get("scope") != "full_supplied_sources"
    ):
        raise ValueError("coverage audit requires a complete full M1 artifact")
    m1_dir = input_manifest.parent
    quality_path = m1_dir / "data_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("scope") != m1["scope"] or quality.get("input_rows") != m1.get("input_rows"):
        raise ValueError("M1 manifest and quality report disagree")
    clean_files = _validated_clean_files(m1_dir, quality)
    dictionaries = m1.get("dictionaries", [])
    if len(dictionaries) != 2:
        raise ValueError("M1 requires channel and object dictionaries")
    paths = [
        (override or Path(item["path"])).resolve()
        for item, override in zip(
            dictionaries, (channels_dictionary, objects_dictionary), strict=True
        )
    ]
    if any(_sha256(path) != item["sha256"] for path, item in zip(paths, dictionaries, strict=True)):
        raise ValueError("M1 dictionary changed since ingestion")
    channels, _, dictionary_audit = load_dictionaries(*paths)
    known_channels = {row["channel_id"] for row in channels}
    input_manifest_sha256 = _sha256(input_manifest)
    quality_sha256 = _sha256(quality_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    aggregate = _observed_bounds(clean_files, pending / "spill")
    rows = []
    for row in aggregate.to_pylist():
        present = row["channel_id"] in known_channels
        rows.append(
            {
                **row,
                "channel_in_current_dictionary": present,
                "historical_applicability_status": (
                    "unverified_no_effective_dates" if present else "unknown_channel"
                ),
                "continuous_observation_status": "unverified_events_only",
            }
        )
    table = pa.Table.from_pylist(rows, schema=BOUNDS_SCHEMA)
    accepted_rows = int(quality["dispositions"]["accepted"])
    if sum(row["event_rows"] for row in rows) != accepted_rows:
        raise ValueError("observed bounds do not conserve M1 accepted rows")
    bounds_path = pending / "channel_observation_bounds.parquet"
    pq.write_table(table, bounds_path, compression="zstd")
    source_counts = Counter(row["source"] for row in rows)
    source_bounds: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = source_bounds.setdefault(
            row["source"],
            {
                "source": row["source"],
                "first_observed_at": row["first_observed_at"],
                "last_observed_at": row["last_observed_at"],
                "event_rows": 0,
                "channel_year_groups": 0,
            },
        )
        item["first_observed_at"] = min(item["first_observed_at"], row["first_observed_at"])
        item["last_observed_at"] = max(item["last_observed_at"], row["last_observed_at"])
        item["event_rows"] += row["event_rows"]
        item["channel_year_groups"] += 1
    report = {
        "schema_version": COVERAGE_VERSION,
        "status": "observed_bounds_only",
        "input_manifest_sha256": input_manifest_sha256,
        "data_quality_sha256": quality_sha256,
        "channel_dictionary_sha256": dictionaries[0]["sha256"],
        "clean_file_count": len(clean_files),
        "accepted_rows": accepted_rows,
        "observation_bound_rows": table.num_rows,
        "distinct_channels": len({row["channel_id"] for row in rows}),
        "observed_unknown_channel_rows": sum(
            row["event_rows"] for row in rows if not row["channel_in_current_dictionary"]
        ),
        "exact_fault_text_rows": sum(row["fault_text_rows"] for row in rows),
        "known_type_fault_text_rows": sum(
            row["fault_text_rows"] for row in rows if row["sensor_type"] is not None
        ),
        "unknown_type_fault_text_rows": sum(
            row["fault_text_rows"] for row in rows if row["sensor_type"] is None
        ),
        "source_count": len(source_counts),
        "source_observed_bounds": [
            {
                **item,
                "first_observed_at": item["first_observed_at"].isoformat(),
                "last_observed_at": item["last_observed_at"].isoformat(),
            }
            for _, item in sorted(source_bounds.items())
        ],
        "missing_global_months_between_first_and_last_partition": _missing_months(quality),
        "confirmed_continuous_channel_intervals": 0,
        "confirmed_historical_applicability_intervals": 0,
        "future_negative_labels_authorized": False,
        "evidence_limitations": [
            "First/last event timestamps and active days describe observed records only.",
            "The current channel dictionary has no effective-from/effective-to dates.",
            "M1 archives and event gaps do not prove continuous channel observation or complete exports.",
            "No target=0 or confident new onset may be inferred from these bounds alone.",
        ],
        "object_mapping_available": dictionary_audit["object_mapping_available"],
    }
    report_path = pending / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": COVERAGE_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": input_manifest_sha256,
        "data_quality_sha256": quality_sha256,
        "channel_dictionary": str(paths[0]),
        "channel_dictionary_sha256": dictionaries[0]["sha256"],
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (bounds_path, report_path)
        },
        "row_count": table.num_rows,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (pending / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels-dictionary", type=Path)
    parser.add_argument("--objects-dictionary", type=Path)
    args = parser.parse_args()
    result = build(
        input_manifest=args.input_manifest,
        output=args.output,
        channels_dictionary=args.channels_dictionary,
        objects_dictionary=args.objects_dictionary,
    )
    print(json.dumps({"output": str(args.output.resolve()), **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
