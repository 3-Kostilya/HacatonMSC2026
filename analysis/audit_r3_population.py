"""Estimate full R3 hourly-grid size without creating labels or features.

Two causal population policies are counted: all hours after the channel's
first observation, and the smaller superset of label-admissible hours within
168 hours of an explicit same-channel normal message. Neither uses outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.state_labeling.forecast import SPLITS  # noqa: E402
from stage1.state_labeling.operational import (  # noqa: E402
    ARCHIVE_SEGMENTS,
    RECENT_NORMAL,
    segment_at,
    source_is_full_archive,
)
from stage1.state_labeling.rules import KNOWN_SENSOR_TYPES  # noqa: E402


HOUR = timedelta(hours=1)
POPULATION_VERSION = "r3-a-recent-normal-grid-v1"
INTERVAL_SCHEMA = pa.schema([
    pa.field("channel_id", pa.string(), nullable=False),
    pa.field("archive_segment", pa.int8(), nullable=False),
    pa.field("start_at", pa.timestamp("us"), nullable=False),
    pa.field("end_exclusive", pa.timestamp("us"), nullable=False),
])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ceil_hour(at: datetime) -> datetime:
    rounded = at.replace(minute=0, second=0, microsecond=0)
    return rounded if at == rounded else rounded + HOUR


def _count_interval(start: datetime, end_exclusive: datetime, counts: Counter[str]) -> None:
    for split in SPLITS:
        left, right = max(start, split.start), min(end_exclusive, split.end)
        if left < right:
            counts[split.name] += int((right - left) / HOUR)


def audit(m1_dir: Path, *, output: Path | None = None) -> dict:
    m1_dir = m1_dir.resolve()
    manifest = json.loads((m1_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("scope") != "full_supplied_sources":
        raise ValueError("R3 population audit requires full completed M1")
    all_grid = Counter()
    known_channels = 0
    for row in pq.read_table(m1_dir / "sensor_statistics.parquet",
                             columns=["sensor_type", "first_at"]).to_pylist():
        if row["sensor_type"] not in KNOWN_SENSOR_TYPES or row["first_at"] is None:
            continue
        known_channels += 1
        start = _ceil_hour(row["first_at"])
        for segment_start, segment_end in ARCHIVE_SEGMENTS:
            if max(start, segment_start) < segment_end:
                _count_interval(max(start, segment_start), segment_end, all_grid)

    files = sorted((m1_dir / "clean").glob("year=*/month=*/*.parquet"))
    if not files:
        raise FileNotFoundError("M1 clean Parquet files are missing")
    pending = None
    writer = None
    interval_path = None
    buffer: list[dict] = []
    if output is not None:
        output = output.resolve()
        pending = output.with_name(output.name + ".inprogress")
        if output.exists() or pending.exists():
            raise FileExistsError("R3 population output or .inprogress directory exists")
        if m1_dir == output or m1_dir in output.parents:
            raise ValueError("R3 population output must be outside M1")
        output.parent.mkdir(parents=True, exist_ok=True)
        pending.mkdir()
        interval_path = pending / "candidate_intervals.parquet"
        writer = pq.ParquetWriter(interval_path, INTERVAL_SCHEMA, compression="zstd")
    db = duckdb.connect(":memory:")
    db.execute("SET memory_limit='4GB'")
    db.execute("SET threads=2")
    normal_grid: Counter[str] = Counter()
    normal_rows = 0
    normal_channels: set[str] = set()
    interval_count = 0
    key: tuple[str, int] | None = None
    start: datetime | None = None
    end: datetime | None = None

    def flush() -> None:
        nonlocal interval_count
        if start is not None and end is not None:
            _count_interval(start, end + HOUR, normal_grid)
            interval_count += 1
            if writer is not None:
                assert key is not None
                buffer.append({
                    "channel_id": key[0],
                    "archive_segment": key[1],
                    "start_at": start,
                    "end_exclusive": end + HOUR,
                })
                if len(buffer) >= 20_000:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=INTERVAL_SCHEMA))
                    buffer.clear()

    try:
        reader = db.execute(
            """SELECT channel_id, sensor_type, timestamp, source
               FROM read_parquet(?, hive_partitioning=false)
               WHERE value_state = 'Норма' AND channel_id IS NOT NULL
                 AND sensor_type IS NOT NULL
               ORDER BY channel_id, timestamp""",
            [[str(path) for path in files]],
        ).to_arrow_reader(batch_size=100_000)
        for batch in reader:
            for row in batch.to_pylist():
                at = row["timestamp"]
                sensor_type = row["sensor_type"]
                if sensor_type not in KNOWN_SENSOR_TYPES or not source_is_full_archive(row["source"], at):
                    continue
                segment = segment_at(at)
                assert segment is not None
                interval_start = _ceil_hour(at)
                interval_end = (at + RECENT_NORMAL).replace(minute=0, second=0, microsecond=0)
                interval_end = min(interval_end, ARCHIVE_SEGMENTS[segment][1] - HOUR)
                if interval_start > interval_end:
                    continue
                normal_rows += 1
                normal_channels.add(row["channel_id"])
                row_key = row["channel_id"], segment
                if row_key != key or start is None or end is None or interval_start > end + HOUR:
                    flush()
                    key, start, end = row_key, interval_start, interval_end
                else:
                    end = max(end, interval_end)
        flush()
    finally:
        db.close()
        if writer is not None:
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=INTERVAL_SCHEMA))
            writer.close()
    report = {
        "schema_version": POPULATION_VERSION,
        "source_m1_manifest": str(m1_dir / "manifest.json"),
        "source_m1_manifest_sha256": _sha256(m1_dir / "manifest.json"),
        "source_clean_files": len(files),
        "known_type_channels": known_channels,
        "all_hours_after_first_observation": dict(sorted(all_grid.items())),
        "all_hours_total": sum(all_grid.values()),
        "explicit_normal_rows": normal_rows,
        "channels_with_explicit_normal": len(normal_channels),
        "merged_normal_intervals": interval_count,
        "hours_with_prior_normal_at_most_168h": dict(sorted(normal_grid.items())),
        "normal_hours_total": sum(normal_grid.values()),
        "outside_normal_window_hours": {
            name: all_grid[name] - normal_grid[name] for name in sorted(all_grid)
        },
    }
    if pending is not None and interval_path is not None and output is not None:
        if pq.ParquetFile(interval_path).metadata.num_rows != interval_count:
            raise ValueError("R3 population physical interval count differs")
        report_path = pending / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        artifact_manifest = {
            "schema_version": POPULATION_VERSION,
            "status": "complete",
            "source_m1_manifest_sha256": report["source_m1_manifest_sha256"],
            "selection_rule": "hourly t with same-channel exact Norma at t or within previous 168h",
            "selection_uses_future_events": False,
            "interval_count": interval_count,
            "candidate_hour_count": report["normal_hours_total"],
            "files": {
                path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in (interval_path, report_path)
            },
        }
        (pending / "manifest.json").write_text(
            json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        pending.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.m1_dir, output=args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
