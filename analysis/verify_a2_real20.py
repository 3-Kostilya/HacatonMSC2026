"""Independently check published A2 rows against accepted M1 event rows.

This verifier deliberately does not call the A2 feature builder. It samples
three decision hours per selected channel and recomputes exact window counts
and a few descriptive values directly from the clean Parquet source.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from stage1.features import A2_SCHEMA, FEATURE_VERSION


WINDOW_HOURS = (1, 6, 24, 168)
EXCLUDED_FLAGS = frozenset({"channel_time_conflict", "nonfinite_numeric", "invalid"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _month_bounds(path: Path) -> tuple[datetime, datetime]:
    year = next(int(part[5:]) for part in path.parts if part.startswith("year="))
    month = next(int(part[6:]) for part in path.parts if part.startswith("month="))
    start = datetime(year, month, 1)
    end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return start, end


def _relevant_files(clean_dir: Path, earliest: datetime, latest: datetime) -> list[Path]:
    files = []
    for path in sorted(clean_dir.rglob("*.parquet")):
        month_start, month_end = _month_bounds(path)
        if month_start <= latest and month_end > earliest:
            files.append(path)
    if not files:
        raise ValueError("no M1 clean partitions overlap selected A2 checks")
    return files


def _sample_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["channel_id"]].append(row)
    selected = []
    for channel_id in sorted(grouped):
        ordered = sorted(grouped[channel_id], key=lambda row: row["prediction_time"])
        for index in sorted({0, len(ordered) // 2, len(ordered) - 1}):
            selected.append(ordered[index])
    return selected


def _load_source_events(
    clean_dir: Path, channels: list[str], earliest: datetime, latest: datetime
) -> tuple[dict[str, list[dict]], list[str]]:
    files = _relevant_files(clean_dir, earliest, latest)
    dataset = ds.dataset([str(path) for path in files], format="parquet")
    predicate = (
        ds.field("channel_id").isin(channels)
        & (ds.field("timestamp") > earliest)
        & (ds.field("timestamp") <= latest)
    )
    scanner = dataset.scanner(
        columns=[
            "channel_id",
            "timestamp",
            "alarm",
            "value_numeric",
            "value_state",
            "quality_flags",
        ],
        filter=predicate,
        batch_size=32_768,
    )
    events: dict[str, list[dict]] = defaultdict(list)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            events[row["channel_id"]].append(row)
    for values in events.values():
        values.sort(key=lambda row: row["timestamp"])
    return events, [str(path) for path in files]


def _assert_equal(actual: object, expected: object, key: str) -> None:
    if actual != expected:
        raise AssertionError(f"{key}: published={actual!r}, independent={expected!r}")


def _assert_float(actual: object, expected: float | None, key: str) -> None:
    if expected is None:
        _assert_equal(actual, None, key)
    elif actual is None or not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{key}: published={actual!r}, independent={expected!r}")


def verify(
    m1_dir: Path,
    a2_dir: Path,
    *,
    expected_channels: int = 20,
    max_feature_rows: int = 100_000,
) -> dict:
    m1_manifest = m1_dir / "manifest.json"
    a2_manifest = a2_dir / "manifest.json"
    feature_path = a2_dir / "features.parquet"
    source = json.loads(m1_manifest.read_text(encoding="utf-8"))
    published = json.loads(a2_manifest.read_text(encoding="utf-8"))
    if source.get("status") != "complete" or published.get("status") != "complete":
        raise ValueError("both M1 and A2 manifests must be complete")
    if published.get("schema_version") != FEATURE_VERSION:
        raise ValueError("unexpected A2 schema version")
    metadata = pq.ParquetFile(feature_path).metadata
    if metadata.num_rows > max_feature_rows:
        raise ValueError("A2 output exceeds bounded verifier row limit")
    table = pq.read_table(feature_path)
    if not table.schema.equals(A2_SCHEMA, check_metadata=False):
        raise ValueError("A2 Parquet schema differs from declared version")
    rows = table.to_pylist()
    if not rows:
        raise ValueError("A2 output has no rows")
    if len({(row["channel_id"], row["prediction_time"]) for row in rows}) != len(rows):
        raise ValueError("duplicate A2 hourly key")
    manifest_hash = _sha256(m1_manifest)
    for row in rows:
        if row["schema_version"] != FEATURE_VERSION:
            raise ValueError("row has wrong A2 version")
        if row["input_manifest_sha256"] != manifest_hash:
            raise ValueError("row has wrong M1 manifest provenance")
        if row["availability_status"] != "eligible" and not row["availability_reasons"]:
            raise ValueError("unavailable row has no reason")
    samples = _sample_rows(rows)
    earliest = min(row["prediction_time"] for row in samples) - timedelta(hours=168)
    latest = max(row["prediction_time"] for row in samples)
    channels = sorted({row["channel_id"] for row in samples})
    if len(channels) != expected_channels:
        raise ValueError(f"expected {expected_channels} channels, found {len(channels)}")
    events, partitions = _load_source_events(m1_dir / "clean", channels, earliest, latest)
    checked_fields = 0
    for row in samples:
        channel = row["channel_id"]
        decision_at = row["prediction_time"]
        for hours in WINDOW_HOURS:
            window_start = decision_at - timedelta(hours=hours)
            window = [
                item
                for item in events.get(channel, ())
                if window_start < item["timestamp"] <= decision_at
            ]
            prefix = f"{channel}@{decision_at.isoformat()}:{hours}h"
            _assert_equal(row[f"event_count_{hours}h"], len(window), prefix + ":event_count")
            _assert_equal(
                row[f"alarm_count_{hours}h"],
                sum(bool(item["alarm"]) for item in window),
                prefix + ":alarm_count",
            )
            checked_fields += 2
            excluded = [
                item for item in window if EXCLUDED_FLAGS.intersection(item["quality_flags"] or ())
            ]
            _assert_equal(
                row[f"excluded_quality_count_{hours}h"],
                len(excluded),
                prefix + ":excluded_quality_count",
            )
            usable = [
                item
                for item in window
                if not EXCLUDED_FLAGS.intersection(item["quality_flags"] or ())
            ]
            numeric = [
                float(item["value_numeric"])
                for item in usable
                if item["value_numeric"] is not None and math.isfinite(item["value_numeric"])
            ]
            _assert_equal(row[f"numeric_count_{hours}h"], len(numeric), prefix + ":numeric_count")
            _assert_float(
                row[f"numeric_median_{hours}h"],
                float(statistics.median(numeric)) if numeric else None,
                prefix + ":numeric_median",
            )
            checked_fields += 3
    return {
        "status": "verified",
        "schema_version": FEATURE_VERSION,
        "m1_manifest_sha256": manifest_hash,
        "a2_features_sha256": _sha256(feature_path),
        "feature_rows": len(rows),
        "channels": channels,
        "sampled_hours": len(samples),
        "checked_fields": checked_fields,
        "input_partitions": partitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m1_dir", type=Path)
    parser.add_argument("a2_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-channels", type=int, default=20)
    args = parser.parse_args()
    report = verify(args.m1_dir, args.a2_dir, expected_channels=args.expected_channels)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "input_partitions"}))


if __name__ == "__main__":
    main()
