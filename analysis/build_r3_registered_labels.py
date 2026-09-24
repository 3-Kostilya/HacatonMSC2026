"""Audit R3 registered-state labels on an existing bounded A2/R2 QA grid.

This is a contract/causality check, not a training set. The full R3 population
must be joined to A3's future feature pack on channel and prediction time.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
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

from analysis.build_a2_hourly import _monthly_files  # noqa: E402
from analysis.build_r1_state_mapping import _sha256  # noqa: E402
from analysis.build_registered_state_episodes import _validated_m1, EPISODE_SCHEMA  # noqa: E402
from stage1.state_labeling.forecast import (  # noqa: E402
    HORIZON,
    LABEL_VERSION,
    SPLITS,
    PredictionPoint,
    RegisteredTargetIndex,
)
from stage1.state_labeling.operational import RECENT_NORMAL, source_is_full_archive  # noqa: E402
from stage1.state_labeling.registered_episodes import (  # noqa: E402
    EPISODE_VERSION,
    RegisteredEpisode,
    StateEvent,
)
from stage1.state_labeling.rules import RULESET_VERSION  # noqa: E402


LABEL_SCHEMA = pa.schema(
    [
        pa.field("channel_id", pa.string(), nullable=False),
        pa.field("sensor_type", pa.string()),
        pa.field("prediction_time", pa.timestamp("us"), nullable=False),
        pa.field("horizon_end", pa.timestamp("us"), nullable=False),
        pa.field("target", pa.int8()),
        pa.field("label_status", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("label_available_at", pa.timestamp("us")),
        pa.field("target_episode_id", pa.string()),
        pa.field("prior_normal_at", pa.timestamp("us")),
        pa.field("split", pa.string()),
        pa.field("split_status", pa.string(), nullable=False),
        pa.field("ruleset_version", pa.string(), nullable=False),
        pa.field("label_version", pa.string(), nullable=False),
    ]
)


def _load_inputs(a2_dir: Path, m1_manifest: Path, b2_dir: Path) -> tuple[dict, dict, list[dict]]:
    _validated_m1(m1_manifest)
    a2 = json.loads((a2_dir / "manifest.json").read_text(encoding="utf-8"))
    b2 = json.loads((b2_dir / "manifest.json").read_text(encoding="utf-8"))
    m1_hash = _sha256(m1_manifest)
    quality_hash = _sha256(m1_manifest.parent / "data_quality.json")
    if a2.get("status") != "complete" or a2.get("input_manifest_sha256") != m1_hash:
        raise ValueError("A2 is incomplete or built from another M1")
    if (
        b2.get("status") != "complete"
        or b2.get("schema_version") != EPISODE_VERSION
        or b2.get("ruleset_version") != RULESET_VERSION
        or b2.get("input_manifest_sha256") != m1_hash
        or b2.get("input_data_quality_sha256") != quality_hash
    ):
        raise ValueError("B2 version or M1 provenance differs from R3")
    for name in ("registered_state_episodes.parquet", "report.json"):
        path = b2_dir / name
        expected = b2["files"][name]
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise ValueError(f"B2 file differs from published manifest: {name}")
    parquet = pq.ParquetFile(b2_dir / "registered_state_episodes.parquet")
    if not parquet.schema_arrow.equals(EPISODE_SCHEMA, check_metadata=False):
        raise ValueError("B2 episode schema differs")
    if parquet.metadata.num_rows != b2["episode_count"]:
        raise ValueError("B2 episode row count differs")
    features_path = a2_dir / "features.parquet"
    if _sha256(features_path) != a2["features_sha256"]:
        raise ValueError("A2 feature file differs from its manifest")
    rows = pq.read_table(
        features_path, columns=["channel_id", "sensor_type", "prediction_time"]
    ).to_pylist()
    if len(rows) != a2["feature_rows"]:
        raise ValueError("A2 physical row count differs from its manifest")
    keys = {(row["channel_id"], row["prediction_time"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("A2 prediction keys are not unique")
    return a2, b2, rows


def _events(
    files: list[Path], channels: list[str], start: datetime, end: datetime
) -> dict[str, list[StateEvent]]:
    database = duckdb.connect(":memory:")
    database.execute("SET memory_limit='4GB'")
    database.execute("SET threads=2")
    result: dict[str, list[StateEvent]] = defaultdict(list)
    try:
        reader = database.execute(
            """
            SELECT row_id, channel_id, sensor_type, timestamp, value_state, alarm, source
            FROM read_parquet(?, hive_partitioning=false)
            WHERE channel_id = ANY(?) AND timestamp >= ? AND timestamp < ?
              AND value_state IS NOT NULL
            ORDER BY channel_id, timestamp, row_id
            """,
            [[str(path) for path in files], channels, start, end],
        ).to_arrow_reader(batch_size=50_000)
        for batch in reader:
            for row in batch.to_pylist():
                if source_is_full_archive(row["source"], row["timestamp"]):
                    result[row["channel_id"]].append(
                        StateEvent(
                            row["row_id"], row["channel_id"], row["sensor_type"],
                            row["timestamp"], row["value_state"], row["alarm"],
                        )
                    )
    finally:
        database.close()
    return result


def _global_episode_split_counts(path: Path) -> dict[str, Any]:
    table = pq.read_table(
        path, columns=["episode_id", "channel_id", "sensor_type", "start_at", "onset_status"]
    )
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    channels: dict[str, set[str]] = defaultdict(set)
    for row in table.to_pylist():
        if row["onset_status"] != "candidate_new_onset":
            continue
        for split in SPLITS:
            if split.start <= row["start_at"] < split.end:
                counts[split.name][row["sensor_type"]] += 1
                channels[split.name].add(row["channel_id"])
                break
    return {
        split.name: {
            "candidate_episodes": sum(counts[split.name].values()),
            "channels_with_candidate": len(channels[split.name]),
            "by_type": dict(sorted(counts[split.name].items())),
        }
        for split in SPLITS
    }


def _r2_statuses(
    r2_dir: Path, *, a2_manifest: Path, m1_manifest: Path, expected_keys: set[tuple[str, datetime]]
) -> tuple[dict[tuple[str, datetime], str], str]:
    manifest_path = r2_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("source_a2_manifest_sha256") != _sha256(a2_manifest)
        or manifest.get("source_m1_manifest_sha256") != _sha256(m1_manifest)
        or manifest.get("ruleset_version") != RULESET_VERSION
    ):
        raise ValueError("R2 A status table differs from A2/M1/R1")
    path = r2_dir / "state_history.parquet"
    expected = manifest["files"][path.name]
    if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
        raise ValueError("R2 A status file differs from its manifest")
    table = pq.read_table(
        path, columns=["channel_id", "prediction_time", "discrete_data_status"]
    )
    rows = {
        (row["channel_id"], row["prediction_time"]): row["discrete_data_status"]
        for row in table.to_pylist()
    }
    if table.num_rows != len(rows) or set(rows) != expected_keys:
        raise ValueError("R2 A and A2 prediction keys differ")
    return rows, _sha256(manifest_path)


def build(
    *, a2_dir: Path, m1_manifest: Path, b2_dir: Path, output: Path,
    r2_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    a2_dir, m1_manifest, b2_dir, output = (
        a2_dir.resolve(), m1_manifest.resolve(), b2_dir.resolve(), output.resolve()
    )
    pending = output.with_name(output.name + ".inprogress")
    if output.exists() or pending.exists():
        raise FileExistsError("R3 QA output or .inprogress directory exists")
    a2, b2, point_rows = _load_inputs(a2_dir, m1_manifest, b2_dir)
    channels = sorted(a2["config"]["channels"])
    if set(row["channel_id"] for row in point_rows) != set(channels):
        raise ValueError("A2 prediction channels disagree with its manifest")
    first = min(row["prediction_time"] for row in point_rows)
    last = max(row["prediction_time"] for row in point_rows)
    context_start = first - RECENT_NORMAL
    context_end = last + HORIZON + timedelta(microseconds=1)
    files, missing = _monthly_files(m1_manifest.parent, context_start, context_end)
    if missing or not files:
        raise ValueError(f"M1 months needed for QA labels are missing: {missing}")
    events = _events(files, channels, context_start, context_end)
    episodes_table = pq.read_table(
        b2_dir / "registered_state_episodes.parquet",
        filters=[("channel_id", "in", channels)],
    )
    by_episode: dict[str, list[RegisteredEpisode]] = defaultdict(list)
    for row in episodes_table.to_pylist():
        by_episode[row["channel_id"]].append(RegisteredEpisode(**row))
    by_points: dict[str, list[PredictionPoint]] = defaultdict(list)
    for row in point_rows:
        by_points[row["channel_id"]].append(
            PredictionPoint(row["channel_id"], row["sensor_type"], row["prediction_time"])
        )
    labels = []
    for channel in channels:
        index = RegisteredTargetIndex(
            channel, events.get(channel, []), by_episode.get(channel, []),
            observed_until=context_end,
        )
        labels.extend(index.label(point) for point in by_points[channel])
    if len(labels) != len(point_rows):
        raise ValueError("R3 did not conserve all prediction keys")
    if any(
        label.target == 1 and (
            label.target_episode_id is None
            or label.label_available_at is None
            or label.label_available_at <= label.prediction_time
        )
        for label in labels
    ):
        raise ValueError("R3 positive has missing or non-future onset evidence")
    status_counts = Counter(label.label_status for label in labels)
    reason_counts = Counter(label.reason for label in labels)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    positive_snapshots: Counter[str] = Counter()
    assigned_positive_by_split: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        by_type[label.sensor_type or "<unknown>"][label.label_status] += 1
        by_split[label.split or "<outside>"][label.label_status] += 1
        if label.target_episode_id is not None:
            positive_snapshots[label.target_episode_id] += 1
            if label.split_status == "assigned" and label.split is not None:
                assigned_positive_by_split[label.split].add(label.target_episode_id)
    if sum(map(len, assigned_positive_by_split.values())) != len(
        set().union(*assigned_positive_by_split.values())
    ):
        raise ValueError("one positive episode appears in multiple assigned splits")
    r2_status_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    r2_manifest_hash = None
    if r2_dir is not None:
        expected_keys = {(label.channel_id, label.prediction_time) for label in labels}
        r2_status, r2_manifest_hash = _r2_statuses(
            r2_dir.resolve(),
            a2_manifest=a2_dir / "manifest.json",
            m1_manifest=m1_manifest,
            expected_keys=expected_keys,
        )
        for label in labels:
            r2_status_by_label[label.label_status][
                r2_status[(label.channel_id, label.prediction_time)]
            ] += 1
    report = {
        "schema_version": LABEL_VERSION,
        "status": "bounded_qa_only_not_train_test",
        "prediction_rows": len(labels),
        "channel_count": len(channels),
        "horizon_hours": 24,
        "split_boundaries": [
            {"name": item.name, "start": item.start.isoformat(), "end": item.end.isoformat()}
            for item in SPLITS
        ],
        "split_policy": "purge prediction rows whose 24h horizon reaches the next split",
        "source_event_rows": sum(len(rows) for rows in events.values()),
        "selected_channel_episode_rows": episodes_table.num_rows,
        "label_status": dict(sorted(status_counts.items())),
        "label_reason": dict(sorted(reason_counts.items())),
        "by_type": {name: dict(counts) for name, counts in sorted(by_type.items())},
        "by_split": {name: dict(counts) for name, counts in sorted(by_split.items())},
        "split_status": dict(Counter(label.split_status for label in labels)),
        "positive_snapshot_count": sum(positive_snapshots.values()),
        "unique_positive_episode_count": len(positive_snapshots),
        "max_snapshots_per_positive_episode": max(positive_snapshots.values(), default=0),
        "assigned_unique_positive_episodes_by_split": {
            split.name: len(assigned_positive_by_split[split.name]) for split in SPLITS
        },
        "r2_discrete_data_status_by_label": {
            status: dict(sorted(counts.items()))
            for status, counts in sorted(r2_status_by_label.items())
        },
        "global_candidate_episode_split_counts": _global_episode_split_counts(
            b2_dir / "registered_state_episodes.parquet"
        ),
        "limitations": [
            "The 20-channel A2 grid was selected for structural QA and cannot be used as train/test.",
            "Negatives assume the seven annual archive exports are complete; channel continuity is unverified.",
            "A3 full feature keys and admissibility are required before final R3 class counts and training.",
        ],
    }
    table = pa.Table.from_pylist([asdict(label) for label in labels], schema=LABEL_SCHEMA)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    labels_path = pending / "registered_forecast_labels.parquet"
    pq.write_table(table, labels_path, compression="zstd")
    report_path = pending / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": LABEL_VERSION,
        "status": "complete_bounded_qa",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_a2_manifest_sha256": _sha256(a2_dir / "manifest.json"),
        "source_m1_manifest_sha256": _sha256(m1_manifest),
        "source_b2_manifest_sha256": _sha256(b2_dir / "manifest.json"),
        "source_r2_manifest_sha256": r2_manifest_hash,
        "episode_version": EPISODE_VERSION,
        "ruleset_version": RULESET_VERSION,
        "row_count": table.num_rows,
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (labels_path, report_path)
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (pending / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--b2-dir", type=Path, required=True)
    parser.add_argument("--r2-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        a2_dir=args.a2_dir,
        m1_manifest=args.m1_manifest,
        b2_dir=args.b2_dir,
        output=args.output,
        r2_dir=args.r2_dir,
    )
    print(json.dumps({"output": str(args.output.resolve()), **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
