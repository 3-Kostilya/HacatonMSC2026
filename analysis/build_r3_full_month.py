"""Build one full-population R3 A feature month from causal Norma intervals.

The output is a resumable month shard, not a labeled training dataset. Every
source month is read from M1; only accepted archive events enter features.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterator

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_a2_hourly import _monthly_files  # noqa: E402
from analysis.r2_b2_handoff import load_b2_for_a2  # noqa: E402
from stage1.features.hourly import FeatureEvent, HourlyConfig  # noqa: E402
from stage1.features.r2 import build_state_history_rows  # noqa: E402
from stage1.features.r3 import (  # noqa: E402
    A2_DIAGNOSTIC_FIELDS,
    A2_FEATURE_FIELDS,
    FEATURE_PACK_SCHEMA,
    KEY_FIELDS,
    MODEL_FEATURE_ALLOWLIST,
    R2_DIAGNOSTIC_FIELDS,
    R2_FEATURE_FIELDS,
    ROW_STATUS_SCHEMA,
)
from stage1.features.r3_full import build_selected_hourly_rows  # noqa: E402
from stage1.features.schema import A2_SCHEMA, FEATURE_VERSION, validate_a2_table  # noqa: E402
from stage1.state_labeling.operational import (  # noqa: E402
    ARCHIVE_SEGMENTS,
    segment_at,
    source_is_full_archive,
)


FULL_PACK_VERSION = "r3-a-feature-pack-full-v2"
POPULATION_VERSION = "r3-a-recent-normal-grid-v1"
HOUR = timedelta(hours=1)
CONTEXT = timedelta(days=28, hours=168 + 24)
BUFFER_ROWS = 40_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    try:
        start = datetime.strptime(month, "%Y-%m")
    except ValueError as exc:
        raise ValueError("month must be YYYY-MM") from exc
    end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1)
    if segment_at(start) is None or segment_at(end - HOUR) != segment_at(start):
        raise ValueError("month is outside one accepted R1 archive segment")
    return start, end


def _intervals_for_month(
    path: Path, start: datetime, end: datetime
) -> tuple[dict[str, list[tuple[datetime, datetime]]], int]:
    intervals: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    hours = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=50_000):
        for row in batch.to_pylist():
            left = max(row["start_at"], start)
            right = min(row["end_exclusive"], end)
            if left >= right:
                continue
            if row["archive_segment"] != segment_at(left):
                raise ValueError("population interval crosses an archive segment")
            intervals[row["channel_id"]].append((left, right))
            hours += int((right - left) / HOUR)
    for channel, channel_intervals in intervals.items():
        channel_intervals.sort()
        if any(current[0] < previous[1] for previous, current in zip(
            channel_intervals, channel_intervals[1:]
        )):
            raise ValueError(f"overlapping population intervals for {channel}")
    return dict(intervals), hours


def _prediction_times(intervals: list[tuple[datetime, datetime]]) -> list[datetime]:
    return [
        left + i * HOUR
        for left, right in intervals
        for i in range(int((right - left) / HOUR))
    ]


@contextmanager
def _duckdb_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Keep concurrent month processes from sharing DuckDB spill files."""
    temp_root = ROOT / "output" / "r3-duckdb-temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    if not temp_root.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("DuckDB temporary root must be inside the workspace")
    with tempfile.TemporaryDirectory(
        prefix="month-", dir=temp_root, ignore_cleanup_errors=True
    ) as temporary:
        database = duckdb.connect(":memory:", config={"temp_directory": temporary})
        try:
            yield database
        finally:
            database.close()


def _channel_events(
    files: list[Path], channels: list[str], start: datetime, end: datetime
) -> Iterator[tuple[str, list[FeatureEvent]]]:
    with _duckdb_connection() as database:
        database.execute("SET memory_limit='4GB'")
        database.execute("SET threads=2")
        # A hash semi-join is essential here: channel_id = ANY(?) becomes extremely
        # slow for thousands of selected channels on the full M1 monthly Parquet.
        database.execute(
            "CREATE TEMP TABLE selected_channels AS SELECT UNNEST(?::VARCHAR[]) AS channel_id",
            [channels],
        )
        current: str | None = None
        events: list[FeatureEvent] = []
        reader = database.execute(
            """SELECT row_id, channel_id, timestamp, alarm, value_numeric,
                      value_state, sensor_type, object_id, join_status, quality_flags,
                      source
               FROM read_parquet(?, hive_partitioning=false) AS events
               SEMI JOIN selected_channels USING (channel_id)
               WHERE timestamp >= ? AND timestamp < ?
               ORDER BY channel_id, timestamp, row_id""",
            [[str(path) for path in files], start, end],
        ).to_arrow_reader(batch_size=40_000)
        for batch in reader:
            for row in batch.to_pylist():
                channel = row["channel_id"]
                if current is not None and channel != current:
                    yield current, events
                    events = []
                current = channel
                if source_is_full_archive(row["source"], row["timestamp"]):
                    events.append(FeatureEvent.from_clean_record(row))
        if current is not None:
            yield current, events


def _project(a2_rows: list[dict], r2_rows: list[dict]) -> tuple[pa.Table, pa.Table]:
    if len(a2_rows) != len(r2_rows):
        raise ValueError("A2 and R2 rows differ in one full-grid channel")
    features = []
    statuses = []
    for a, r in zip(a2_rows, r2_rows):
        if (a["channel_id"], a["prediction_time"]) != (
            r["channel_id"], r["prediction_time"]
        ) or a["run_id"] != r["source_a2_run_id"]:
            raise ValueError("A2/R2 full-grid key or run mismatch")
        features.append({
            **{name: a[name] for name in (*KEY_FIELDS, *A2_FEATURE_FIELDS)},
            **{name: r[name] for name in R2_FEATURE_FIELDS},
        })
        statuses.append({
            **{name: a[name] for name in (*KEY_FIELDS, *A2_DIAGNOSTIC_FIELDS)},
            **{name: r[name] for name in R2_DIAGNOSTIC_FIELDS},
        })
    return (
        pa.Table.from_pylist(features, schema=FEATURE_PACK_SCHEMA),
        pa.Table.from_pylist(statuses, schema=ROW_STATUS_SCHEMA),
    )


def build_month(
    *, m1_manifest: Path, population_dir: Path, b2_dir: Path,
    month: str, output_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    m1_manifest = m1_manifest.resolve()
    population_dir = population_dir.resolve()
    b2_dir = b2_dir.resolve()
    output_root = output_root.resolve()
    start, end = _month_bounds(month)
    month_dir = output_root / f"year={start.year}" / f"month={start.month:02d}"
    pending = month_dir.with_name(month_dir.name + ".inprogress")
    if pending.exists():
        raise FileExistsError("inspect existing full R3 .inprogress directory before retry")
    if month_dir.exists():
        path = month_dir / "manifest.json"
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("status") != "complete_month"
            or existing.get("schema_version") != FULL_PACK_VERSION
            or existing.get("month") != month
            or existing.get("source_m1_manifest_sha256") != _sha256(m1_manifest)
            or existing.get("source_population_manifest_sha256") != _sha256(
                population_dir / "manifest.json"
            )
            or existing.get("source_b2_catalog_manifest_sha256") != _sha256(
                b2_dir / "manifest.json"
            )
        ):
            raise ValueError("existing full R3 month has different lineage")
        for name, schema in (
            ("features.parquet", FEATURE_PACK_SCHEMA),
            ("row_status.parquet", ROW_STATUS_SCHEMA),
        ):
            file = month_dir / name
            expected = existing["files"][name]
            parquet = pq.ParquetFile(file)
            if (
                _sha256(file) != expected["sha256"]
                or parquet.metadata.num_rows != existing["row_count"]
                or not parquet.schema_arrow.equals(schema, check_metadata=False)
            ):
                raise ValueError(f"existing full R3 month file differs: {file}")
        print(f"{month}: reuse verified complete month", flush=True)
        return existing
    m1 = json.loads(m1_manifest.read_text(encoding="utf-8"))
    if m1.get("status") != "complete" or m1.get("scope") != "full_supplied_sources":
        raise ValueError("full R3 month requires completed full M1")
    population_manifest_path = population_dir / "manifest.json"
    m1_sha = _sha256(m1_manifest)
    population_sha = _sha256(population_manifest_path)
    population = json.loads(population_manifest_path.read_text(encoding="utf-8"))
    interval_path = population_dir / "candidate_intervals.parquet"
    if (
        population.get("schema_version") != POPULATION_VERSION
        or population.get("status") != "complete"
        or population.get("source_m1_manifest_sha256") != m1_sha
        or population.get("selection_uses_future_events") is not False
        or _sha256(interval_path) != population["files"][interval_path.name]["sha256"]
    ):
        raise ValueError("population intervals have wrong provenance or SHA-256")
    intervals, expected_hours = _intervals_for_month(interval_path, start, end)
    if not intervals:
        raise ValueError("month contains no recent-Norma prediction points")
    channels = sorted(intervals)
    print(f"{month}: {expected_hours} hours across {len(channels)} channels", flush=True)
    catalog = load_b2_for_a2(b2_dir, local_m1_manifest=m1_manifest, channels=channels)
    print(f"{month}: B2 catalog verified in {time.monotonic() - started:.1f}s", flush=True)
    by_episode: dict[str, list] = defaultdict(list)
    for episode in catalog.episodes:
        by_episode[episode.channel_id].append(episode)
    segment = segment_at(start)
    assert segment is not None
    context_start = max(start - CONTEXT, ARCHIVE_SEGMENTS[segment][0])
    files, missing = _monthly_files(m1_manifest.parent, context_start, end)
    if missing or not files:
        raise ValueError(f"M1 source months are missing for {month}: {missing}")
    config = HourlyConfig()
    config_sha = hashlib.sha256(json.dumps({
        "schema_version": FULL_PACK_VERSION,
        "month": month,
        "m1_manifest_sha256": m1_sha,
        "population_manifest_sha256": population_sha,
        "b2_manifest_sha256": catalog.audit["catalog_manifest_sha256"],
        "baseline_lookback_seconds": config.baseline_lookback.total_seconds(),
        "baseline_embargo_seconds": config.baseline_embargo.total_seconds(),
        "window_hours": [1, 6, 24, 168],
        "selected_hours": expected_hours,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    provenance = {
        "schema_version": FEATURE_VERSION,
        "run_id": "r3full-" + config_sha[:16],
        "config_sha256": config_sha,
        "input_manifest_sha256": m1_sha,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    pending.mkdir(parents=True)
    feature_path = pending / "features.parquet"
    status_path = pending / "row_status.parquet"
    feature_writer = pq.ParquetWriter(feature_path, FEATURE_PACK_SCHEMA, compression="zstd")
    status_writer = pq.ParquetWriter(status_path, ROW_STATUS_SCHEMA, compression="zstd")
    feature_buffer: list[pa.Table] = []
    status_buffer: list[pa.Table] = []
    buffered_rows = 0
    row_count = 0
    channels_processed = 0
    source_events = 0
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)

    def flush() -> None:
        nonlocal buffered_rows
        if not feature_buffer:
            return
        feature_batch = pa.concat_tables(feature_buffer)
        status_batch = pa.concat_tables(status_buffer)
        if feature_batch.num_rows != status_batch.num_rows:
            raise ValueError("full R3 buffered feature/status row count differs")
        for table in (feature_batch, status_batch):
            if any(table.column(field.name).null_count for field in table.schema if not field.nullable):
                raise ValueError("full R3 required output field is null")
        feature_writer.write_table(feature_batch)
        status_writer.write_table(status_batch)
        feature_buffer.clear()
        status_buffer.clear()
        buffered_rows = 0

    try:
        for channel, events in _channel_events(files, channels, context_start, end):
            if channel not in intervals:
                continue
            if not events:
                raise ValueError(f"{channel}: no accepted M1 context for selected hours")
            requested = _prediction_times(intervals[channel])
            a2_rows = build_selected_hourly_rows(events, channel, requested, config=config)
            for row in a2_rows:
                row.update(provenance)
            strict_audit = channels_processed < 3
            if strict_audit:
                validate_a2_table(pa.Table.from_pylist(a2_rows, schema=A2_SCHEMA))
            # This intermediate field is not published: the full-grid source is
            # the population manifest, not a bounded A2 artifact.
            r2_table = build_state_history_rows(
                a2_rows, events,
                source_a2_manifest_sha256=population_sha,
                completed_episodes=by_episode.get(channel, []),
                validate=strict_audit,
            )
            feature_table, status_table = _project(a2_rows, r2_table.to_pylist())
            feature_buffer.append(feature_table)
            status_buffer.append(status_table)
            buffered_rows += feature_table.num_rows
            row_count += feature_table.num_rows
            channels_processed += 1
            source_events += len(events)
            if channels_processed == 1 or channels_processed % 100 == 0:
                print(
                    f"{month}: {channels_processed}/{len(channels)} channels, "
                    f"{row_count}/{expected_hours} hours in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
            for name in (
                "availability_status", "numeric_data_status", "discrete_data_status"
            ):
                for item in pc.value_counts(status_table.column(name)).to_pylist():
                    status_counts[name][item["values"]] += item["counts"]
            if buffered_rows >= BUFFER_ROWS:
                flush()
        flush()
    finally:
        feature_writer.close()
        status_writer.close()
    if channels_processed != len(channels) or row_count != expected_hours:
        raise ValueError(
            f"full R3 {month} conserved {row_count}/{expected_hours} hours and "
            f"{channels_processed}/{len(channels)} channels"
        )
    if pq.ParquetFile(feature_path).metadata.num_rows != row_count or (
        pq.ParquetFile(status_path).metadata.num_rows != row_count
    ):
        raise ValueError("full R3 month physical Parquet row count mismatch")
    month_manifest = {
        "schema_version": FULL_PACK_VERSION,
        "status": "complete_month",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "month": month,
        "start_at": start.isoformat(),
        "end_at": end.isoformat(),
        "source_m1_manifest_sha256": m1_sha,
        "source_population_manifest_sha256": population_sha,
        "source_b2_catalog_manifest_sha256": catalog.audit["catalog_manifest_sha256"],
        "selection_policy": POPULATION_VERSION,
        "feature_count": len(MODEL_FEATURE_ALLOWLIST),
        "row_count": row_count,
        "channel_count": channels_processed,
        "source_events_in_context": source_events,
        "status_counts": {
            name: dict(sorted(counts.items())) for name, counts in sorted(status_counts.items())
        },
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (feature_path, status_path)
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (pending / "manifest.json").write_text(
        json.dumps(month_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.rename(month_dir)
    return month_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-manifest", required=True, type=Path)
    parser.add_argument("--population-dir", required=True, type=Path)
    parser.add_argument("--b2-dir", required=True, type=Path)
    parser.add_argument("--month", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = build_month(
        m1_manifest=args.m1_manifest,
        population_dir=args.population_dir,
        b2_dir=args.b2_dir,
        month=args.month,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
