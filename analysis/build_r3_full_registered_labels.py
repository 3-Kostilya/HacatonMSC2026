"""Apply B3 registered-event labels to one immutable A3 full-population month.

The output is a resumable month shard. It is not a training-ready model table:
model admission and channel-continuity evidence remain separate decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_a2_hourly import _monthly_files  # noqa: E402
from analysis.build_r1_state_mapping import _sha256  # noqa: E402
from analysis.build_r3_full_month import FULL_PACK_VERSION, POPULATION_VERSION, _duckdb_connection  # noqa: E402
from analysis.build_r3_registered_labels import LABEL_SCHEMA  # noqa: E402
from stage1.features.r3 import FEATURE_PACK_SCHEMA, MODEL_FEATURE_ALLOWLIST, ROW_STATUS_SCHEMA  # noqa: E402
from stage1.state_labeling.forecast import (  # noqa: E402
    HORIZON, LABEL_VERSION, PredictionPoint, RegisteredTargetIndex,
)
from stage1.state_labeling.operational import (  # noqa: E402
    ARCHIVE_SEGMENTS, RECENT_NORMAL, segment_at, source_is_full_archive,
)
from stage1.state_labeling.registered_episodes import (  # noqa: E402
    EPISODE_VERSION, RegisteredEpisode, StateEvent,
)
from stage1.state_labeling.rules import RULESET_VERSION  # noqa: E402


def _verify_inputs(a3_dir: Path, m1_manifest: Path, b2_dir: Path, month: str) -> tuple[dict, dict]:
    pack = json.loads((a3_dir / "manifest.json").read_text(encoding="utf-8"))
    b2_manifest = b2_dir / "manifest.json"
    b2 = json.loads(b2_manifest.read_text(encoding="utf-8"))
    if (
        pack.get("schema_version") != FULL_PACK_VERSION
        or pack.get("status") != "complete"
        or pack.get("selection_policy") != POPULATION_VERSION
        or pack.get("ruleset_version") != RULESET_VERSION
        or pack.get("source_m1_manifest_sha256") != _sha256(m1_manifest)
        or pack.get("source_b2_catalog_manifest_sha256") != _sha256(b2_manifest)
        or b2.get("status") != "complete"
        or b2.get("schema_version") != EPISODE_VERSION
        or b2.get("ruleset_version") != RULESET_VERSION
        or b2.get("input_manifest_sha256") != _sha256(m1_manifest)
    ):
        raise ValueError("A3/B2/M1 source lineage or version differs")
    allowlist_path = a3_dir / pack["allowlist_file"]
    if _sha256(allowlist_path) != pack["allowlist_sha256"]:
        raise ValueError("A3 allowlist SHA-256 mismatch")
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    if (
        allowlist.get("schema_version") != FULL_PACK_VERSION
        or [item["name"] for item in allowlist["feature_columns"]]
        != list(MODEL_FEATURE_ALLOWLIST)
    ):
        raise ValueError("A3 model feature allowlist differs")
    members = [member for member in pack["chunks"] if member["month"] == month]
    if len(members) != 1:
        raise ValueError(f"A3 full package has no unique {month} shard")
    member = members[0]
    shard_path = a3_dir / member["manifest_file"]
    if _sha256(shard_path) != member["manifest_sha256"]:
        raise ValueError("A3 month manifest SHA-256 mismatch")
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    if (
        shard.get("status") != "complete_month"
        or shard.get("month") != month
        or shard.get("source_m1_manifest_sha256") != pack["source_m1_manifest_sha256"]
        or shard.get("source_b2_catalog_manifest_sha256") != pack["source_b2_catalog_manifest_sha256"]
        or shard.get("row_count") != member["rows"]
    ):
        raise ValueError("A3 month manifest differs from full package")
    for name, schema, key in (
        ("features.parquet", FEATURE_PACK_SCHEMA, "features_file"),
        ("row_status.parquet", ROW_STATUS_SCHEMA, "row_status_file"),
    ):
        path = a3_dir / member[key]
        if _sha256(path) != shard["files"][name]["sha256"]:
            raise ValueError(f"A3 month file SHA-256 mismatch: {name}")
        parquet = pq.ParquetFile(path)
        if (
            not parquet.schema_arrow.equals(schema, check_metadata=False)
            or parquet.metadata.num_rows != member["rows"]
        ):
            raise ValueError(f"A3 month file schema or rows differ: {name}")
    catalog = b2_dir / "registered_state_episodes.parquet"
    if _sha256(catalog) != b2["files"][catalog.name]["sha256"]:
        raise ValueError("B2 catalog SHA-256 mismatch")
    return pack, member


def _prediction_groups(features: Path, statuses: Path) -> Iterator[tuple[str, list[tuple[PredictionPoint, dict]]]]:
    feature_rows = (
        row for batch in pq.ParquetFile(features).iter_batches(
            batch_size=40_000, columns=["channel_id", "prediction_time", "sensor_type"]
        ) for row in batch.to_pylist()
    )
    status_rows = (
        row for batch in pq.ParquetFile(statuses).iter_batches(
            batch_size=40_000, columns=[
                "channel_id", "prediction_time", "availability_status",
                "numeric_data_status", "discrete_data_status",
            ]
        ) for row in batch.to_pylist()
    )
    current: str | None = None
    grouped: list[tuple[PredictionPoint, dict]] = []
    prior_time: datetime | None = None
    from itertools import zip_longest

    for feature, status in zip_longest(feature_rows, status_rows):
        if feature is None or status is None:
            raise ValueError("A3 feature and status row counts differ")
        key = (feature["channel_id"], feature["prediction_time"])
        if key != (status["channel_id"], status["prediction_time"]):
            raise ValueError("A3 feature and status keys differ")
        channel, at = key
        if current is not None and channel != current:
            if channel < current:
                raise ValueError("A3 month keys are not sorted by channel")
            yield current, grouped
            grouped = []
            prior_time = None
        if prior_time is not None and at <= prior_time:
            raise ValueError("A3 channel prediction times are not strictly increasing")
        current, prior_time = channel, at
        grouped.append((PredictionPoint(channel, feature["sensor_type"], at), status))
    if current is not None:
        yield current, grouped


def _state_groups(files: list[Path], start: datetime, end: datetime) -> Iterator[tuple[str, list[StateEvent]]]:
    with _duckdb_connection() as database:
        database.execute("SET memory_limit='4GB'")
        database.execute("SET threads=2")
        reader = database.execute(
            """SELECT row_id, channel_id, sensor_type, timestamp, value_state, alarm, source
               FROM read_parquet(?, hive_partitioning=false)
               WHERE timestamp >= ? AND timestamp < ? AND value_state IS NOT NULL
               ORDER BY channel_id, timestamp, row_id""",
            [[str(path) for path in files], start, end],
        ).to_arrow_reader(batch_size=40_000)
        current: str | None = None
        grouped: list[StateEvent] = []
        for batch in reader:
            for row in batch.to_pylist():
                channel = row["channel_id"]
                if current is not None and channel != current:
                    yield current, grouped
                    grouped = []
                current = channel
                if source_is_full_archive(row["source"], row["timestamp"]):
                    grouped.append(StateEvent(
                        row["row_id"], channel, row["sensor_type"],
                        row["timestamp"], row["value_state"], row["alarm"],
                    ))
        if current is not None:
            yield current, grouped


def _episode_groups(b2_dir: Path) -> dict[str, list[RegisteredEpisode]]:
    episodes: dict[str, list[RegisteredEpisode]] = defaultdict(list)
    path = b2_dir / "registered_state_episodes.parquet"
    for batch in pq.ParquetFile(path).iter_batches(batch_size=20_000):
        for row in batch.to_pylist():
            episodes[row["channel_id"]].append(RegisteredEpisode(**row))
    return episodes


def build_month(*, a3_dir: Path, m1_manifest: Path, b2_dir: Path,
                month: str, output_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    a3_dir, m1_manifest, b2_dir, output_root = (
        path.resolve() for path in (a3_dir, m1_manifest, b2_dir, output_root)
    )
    pack, member = _verify_inputs(a3_dir, m1_manifest, b2_dir, month)
    month_start = datetime.fromisoformat(member["start_at"])
    month_end = datetime.fromisoformat(member["end_at"])
    segment = segment_at(month_start)
    if segment is None or segment_at(month_end - timedelta(microseconds=1)) != segment:
        raise ValueError("A3 month crosses an excluded archive segment")
    segment_start, segment_end = ARCHIVE_SEGMENTS[segment]
    context_start = max(segment_start, month_start - RECENT_NORMAL)
    observed_until = min(segment_end, month_end + HORIZON + timedelta(microseconds=1))
    files, missing = _monthly_files(m1_manifest.parent, context_start, observed_until)
    if missing or not files:
        raise ValueError(f"M1 months needed for {month} labels are missing: {missing}")
    output = output_root / f"year={month_start.year}" / f"month={month_start.month:02d}"
    pending = output.with_name(output.name + ".inprogress")
    if output.exists() or pending.exists():
        raise FileExistsError(f"B3 month output or .inprogress exists: {output}")
    pending.mkdir(parents=True)
    writer = pq.ParquetWriter(pending / "registered_forecast_labels.parquet", LABEL_SCHEMA,
                              compression="zstd")
    counts: Counter[str] = Counter()
    assigned: Counter[str] = Counter()
    split_statuses: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    branches: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    positive_episodes: set[str] = set()
    assigned_positive_episodes: set[str] = set()
    positive_channels: set[str] = set()
    status_channels: dict[str, set[str]] = defaultdict(set)
    channel_days: dict[str, dict[str, set[tuple[str, date]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    channels: set[str] = set()
    row_count = 0
    buffer: list[dict] = []
    events = iter(_state_groups(files, context_start, observed_until))
    current_events = next(events, None)
    episodes = _episode_groups(b2_dir)
    try:
        for channel, points in _prediction_groups(
            a3_dir / member["features_file"], a3_dir / member["row_status_file"]
        ):
            while current_events is not None and current_events[0] < channel:
                current_events = next(events, None)
            channel_events = current_events[1] if current_events and current_events[0] == channel else []
            index = RegisteredTargetIndex(
                channel, channel_events, episodes.get(channel, []),
                observed_until=observed_until,
            )
            channels.add(channel)
            for point, status in points:
                label = index.label(point)
                row_count += 1
                counts[label.label_status] += 1
                split_statuses[label.split_status] += 1
                by_type[label.sensor_type or "<unknown>"][label.label_status] += 1
                status_channels[label.label_status].add(channel)
                channel_days[label.label_status][label.sensor_type or "<unknown>"].add(
                    (channel, point.at.date())
                )
                if label.split_status == "assigned":
                    assigned[label.label_status] += 1
                for field in ("availability_status", "numeric_data_status", "discrete_data_status"):
                    branches[label.label_status][field][status[field]] += 1
                if label.target_episode_id is not None:
                    positive_episodes.add(label.target_episode_id)
                    positive_channels.add(channel)
                    if label.split_status == "assigned":
                        assigned_positive_episodes.add(label.target_episode_id)
                buffer.append({name: getattr(label, name) for name in LABEL_SCHEMA.names})
                if len(buffer) >= 40_000:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=LABEL_SCHEMA))
                    buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=LABEL_SCHEMA))
        writer.close()
    except BaseException:
        writer.close()
        raise
    if row_count != member["rows"] or len(channels) != member["channels"]:
        raise ValueError("B3 month did not conserve all A3 keys/channels")
    report = {
        "schema_version": LABEL_VERSION,
        "status": "complete_full_month_labels_not_training_ready",
        "month": month,
        "row_count": row_count,
        "channel_count": len(channels),
        "label_status": dict(sorted(counts.items())),
        "assigned_label_status": dict(sorted(assigned.items())),
        "split_status": dict(sorted(split_statuses.items())),
        "by_type": {name: dict(sorted(items.items())) for name, items in sorted(by_type.items())},
        "row_status_by_label": {
            name: {field: dict(sorted(items.items())) for field, items in sorted(fields.items())}
            for name, fields in sorted(branches.items())
        },
        "unique_positive_episode_ids": sorted(positive_episodes),
        "assigned_positive_episode_ids": sorted(assigned_positive_episodes),
        "positive_channel_count": len(positive_channels),
        "positive_channel_ids": sorted(positive_channels),
        "channel_ids_by_label": {
            label: sorted(ids) for label, ids in sorted(status_channels.items())
        },
        "channel_days_by_type_and_label": {
            label: {name: len(days) for name, days in sorted(types.items())}
            for label, types in sorted(channel_days.items())
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report_path = pending / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    label_path = pending / "registered_forecast_labels.parquet"
    manifest = {
        "schema_version": LABEL_VERSION,
        "status": "complete_month",
        "month": month,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_a3_full_manifest_sha256": _sha256(a3_dir / "manifest.json"),
        "source_a3_month_manifest_sha256": member["manifest_sha256"],
        "source_m1_manifest_sha256": pack["source_m1_manifest_sha256"],
        "source_b2_catalog_manifest_sha256": pack["source_b2_catalog_manifest_sha256"],
        "row_count": row_count,
        "channel_count": len(channels),
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (label_path, report_path)
        },
    }
    (pending / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--b2-dir", type=Path, required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_month(
        a3_dir=args.a3_dir, m1_manifest=args.m1_manifest, b2_dir=args.b2_dir,
        month=args.month, output_root=args.output_root,
    )
    print(json.dumps({key: result[key] for key in (
        "month", "row_count", "label_status", "assigned_label_status",
        "elapsed_seconds",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
