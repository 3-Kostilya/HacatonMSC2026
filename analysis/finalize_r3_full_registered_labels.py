"""Audit all full A3/B3 month shards and publish split-level label counts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_r1_state_mapping import _sha256  # noqa: E402
from analysis.build_r3_full_month import FULL_PACK_VERSION  # noqa: E402
from analysis.build_r3_registered_labels import LABEL_SCHEMA  # noqa: E402
from stage1.state_labeling.forecast import LABEL_VERSION  # noqa: E402


def finalize(*, a3_dir: Path, labels_dir: Path,
             refresh_summary: bool = False) -> dict[str, Any]:
    a3_dir, labels_dir = a3_dir.resolve(), labels_dir.resolve()
    a3_manifest_path = a3_dir / "manifest.json"
    a3 = json.loads(a3_manifest_path.read_text(encoding="utf-8"))
    if a3.get("schema_version") != FULL_PACK_VERSION or a3.get("status") != "complete":
        raise ValueError("full A3 pack is incomplete or has wrong version")
    expected_a3 = _sha256(a3_manifest_path)
    report_path = labels_dir / "report.json"
    manifest_path = labels_dir / "manifest.json"
    if (report_path.exists() or manifest_path.exists()) and not refresh_summary:
        raise FileExistsError("full B3 report or manifest already exists")
    if refresh_summary:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("source_a3_full_manifest_sha256") != expected_a3:
            raise ValueError("existing B3 summary derives from another A3 pack")
    totals: Counter[str] = Counter()
    assigned: dict[str, Counter[str]] = defaultdict(Counter)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    split_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    all_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_split_type: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    assigned_by_split_type: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    row_status: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    positive_episodes: dict[str, set[str]] = defaultdict(set)
    assigned_episodes: dict[str, set[str]] = defaultdict(set)
    positive_channels: dict[str, set[str]] = defaultdict(set)
    channels_by_label: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    channel_days_by_type: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    chunks = []
    label_paths: list[str] = []
    row_count = 0
    for chunk in a3["chunks"]:
        month = chunk["month"]
        year = int(month[:4])
        split = "train" if year < 2025 else "validation" if year == 2025 else "test"
        directory = labels_dir / f"year={year}" / f"month={month[5:]}"
        shard_path = directory / "manifest.json"
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        if (
            shard.get("schema_version") != LABEL_VERSION
            or shard.get("status") != "complete_month"
            or shard.get("month") != month
            or shard.get("source_a3_full_manifest_sha256") != expected_a3
            or shard.get("source_a3_month_manifest_sha256") != chunk["manifest_sha256"]
            or shard.get("row_count") != chunk["rows"]
        ):
            raise ValueError(f"B3/A3 month provenance or row count differs: {month}")
        label_path = directory / "registered_forecast_labels.parquet"
        label_paths.append(str(label_path))
        local_report_path = directory / "report.json"
        for path in (label_path, local_report_path):
            if _sha256(path) != shard["files"][path.name]["sha256"]:
                raise ValueError(f"B3 month SHA-256 mismatch: {path}")
        parquet = pq.ParquetFile(label_path)
        if (
            not parquet.schema_arrow.equals(LABEL_SCHEMA, check_metadata=False)
            or parquet.metadata.num_rows != chunk["rows"]
        ):
            raise ValueError(f"B3 month label schema/rows differ: {month}")
        local = json.loads(local_report_path.read_text(encoding="utf-8"))
        if sum(local["label_status"].values()) != chunk["rows"]:
            raise ValueError(f"B3 report does not conserve all rows: {month}")
        if (
            sum(local["split_status"].values()) != chunk["rows"]
            or sum(local["assigned_label_status"].values())
            != local["split_status"].get("assigned", 0)
            or any(
                sum(counts.values()) != local["label_status"][label]
                for label, fields in local["row_status_by_label"].items()
                for counts in fields.values()
            )
        ):
            raise ValueError(f"B3 split or A3 status cross-counts differ: {month}")
        totals.update(local["label_status"])
        by_split[split].update(local["label_status"])
        assigned[split].update(local["assigned_label_status"])
        split_statuses[split].update(local["split_status"])
        for sensor_type, counts in local["by_type"].items():
            all_by_type[sensor_type].update(counts)
            by_split_type[split][sensor_type].update(counts)
            assigned_by_split_type[split][sensor_type].update(counts)
        if local["split_status"].get("purged_boundary", 0):
            with duckdb.connect(":memory:") as database:
                purged = database.execute(
                    """SELECT sensor_type, label_status, count(*)
                       FROM read_parquet(?) WHERE split_status = 'purged_boundary'
                       GROUP BY sensor_type, label_status""",
                    [str(label_path)],
                ).fetchall()
            if sum(count for _, _, count in purged) != local["split_status"]["purged_boundary"]:
                raise ValueError(f"B3 purged type counts differ: {month}")
            for sensor_type, label, count in purged:
                assigned_by_split_type[split][sensor_type or "<unknown>"][label] -= count
        for label, fields in local["row_status_by_label"].items():
            for field, counts in fields.items():
                row_status[label][field].update(counts)
        positive_episodes[split].update(local["unique_positive_episode_ids"])
        assigned_episodes[split].update(local["assigned_positive_episode_ids"])
        positive_channels[split].update(local["positive_channel_ids"])
        for label, ids in local["channel_ids_by_label"].items():
            channels_by_label[split][label].update(ids)
        for label, types in local["channel_days_by_type_and_label"].items():
            channel_days_by_type[split][label].update(types)
        row_count += chunk["rows"]
        chunks.append({
            "month": month,
            "manifest_file": shard_path.relative_to(labels_dir).as_posix(),
            "manifest_sha256": _sha256(shard_path),
            "rows": chunk["rows"],
        })
    if row_count != a3["row_count"] or len(chunks) != a3["chunk_count"]:
        raise ValueError("B3 full package does not conserve A3 rows/months")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if assigned_episodes[left] & assigned_episodes[right]:
            raise ValueError("one positive episode appears in two assigned splits")
    if any(
        sum(counts[label] for counts in assigned_by_split_type[split].values())
        != assigned[split][label]
        for split in ("train", "validation", "test")
        for label in ("positive", "negative", "unknown", "excluded")
    ):
        raise ValueError("B3 assigned class counts differ by type")
    assigned_positive_channels: dict[str, dict[str, int]] = defaultdict(dict)
    assigned_positive_channel_days: dict[str, dict[str, int]] = defaultdict(dict)
    assigned_positive_episodes_by_type: dict[str, dict[str, int]] = defaultdict(dict)
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=4")
        groups = database.execute(
            """SELECT split, sensor_type, count(DISTINCT target_episode_id),
                      count(DISTINCT channel_id),
                      count(DISTINCT (channel_id, CAST(prediction_time AS DATE)))
               FROM read_parquet(?)
               WHERE label_status = 'positive' AND split_status = 'assigned'
               GROUP BY split, sensor_type""",
            [label_paths],
        ).fetchall()
    for split, sensor_type, episode_count, channels, days in groups:
        name = sensor_type or "<unknown>"
        assigned_positive_episodes_by_type[split][name] = episode_count
        assigned_positive_channels[split][name] = channels
        assigned_positive_channel_days[split][name] = days
    if any(
        sum(assigned_positive_episodes_by_type[split].values()) != len(assigned_episodes[split])
        for split in ("train", "validation", "test")
    ):
        raise ValueError("B3 assigned episode counts differ by sensor type")
    report = {
        "schema_version": LABEL_VERSION,
        "status": "full_registered_label_audit_not_training_ready",
        "row_count": row_count,
        "month_count": len(chunks),
        "label_status": dict(sorted(totals.items())),
        "label_status_by_split": {
            split: dict(sorted(by_split[split].items())) for split in ("train", "validation", "test")
        },
        "split_status_by_split": {
            split: dict(sorted(split_statuses[split].items()))
            for split in ("train", "validation", "test")
        },
        "assigned_label_status_by_split": {
            split: dict(sorted(assigned[split].items())) for split in ("train", "validation", "test")
        },
        "label_status_by_type": {
            name: dict(sorted(counts.items())) for name, counts in sorted(all_by_type.items())
        },
        "label_status_by_split_and_type": {
            split: {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(by_split_type[split].items())
            } for split in ("train", "validation", "test")
        },
        "assigned_label_status_by_split_and_type": {
            split: {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(assigned_by_split_type[split].items())
            } for split in ("train", "validation", "test")
        },
        "types_without_both_assigned_classes_by_split": {
            split: [
                name for name, counts in sorted(assigned_by_split_type[split].items())
                if counts["positive"] == 0 or counts["negative"] == 0
            ] for split in ("train", "validation", "test")
        },
        "row_status_by_label": {
            label: {field: dict(sorted(counts.items())) for field, counts in sorted(fields.items())}
            for label, fields in sorted(row_status.items())
        },
        "unique_positive_episodes_by_split": {
            split: len(positive_episodes[split]) for split in ("train", "validation", "test")
        },
        "unique_positive_episodes_total": len(set().union(*positive_episodes.values())),
        "assigned_unique_positive_episodes_by_split": {
            split: len(assigned_episodes[split]) for split in ("train", "validation", "test")
        },
        "assigned_unique_positive_episodes_by_split_and_type": {
            split: dict(sorted(assigned_positive_episodes_by_type[split].items()))
            for split in ("train", "validation", "test")
        },
        "positive_channels_by_split": {
            split: len(positive_channels[split]) for split in ("train", "validation", "test")
        },
        "positive_channels_total": len(set().union(*positive_channels.values())),
        "assigned_positive_channels_by_split_and_type": {
            split: dict(sorted(assigned_positive_channels[split].items()))
            for split in ("train", "validation", "test")
        },
        "assigned_positive_channel_days_by_split_and_type": {
            split: dict(sorted(assigned_positive_channel_days[split].items()))
            for split in ("train", "validation", "test")
        },
        "channels_by_split_and_label": {
            split: {
                label: len(ids) for label, ids in sorted(channels_by_label[split].items())
            } for split in ("train", "validation", "test")
        },
        "channel_days_by_split_type_and_label": {
            split: {
                label: dict(sorted(types.items()))
                for label, types in sorted(channel_days_by_type[split].items())
            } for split in ("train", "validation", "test")
        },
        "limitations": [
            "Labels identify future registered journal messages, not physical sensor failures.",
            "Archive completeness is an operational assumption; channel continuity is unverified.",
            "Feature eligibility and final model admission remain separate from this label audit.",
        ],
    }
    labels_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    manifest = {
        "schema_version": LABEL_VERSION,
        "status": "complete_full_labels",
        "purpose": "full_registered_event_targets",
        "not_training_ready": True,
        "reason_not_training_ready": "channel observation continuity and model admission are not established",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_a3_full_manifest_sha256": expected_a3,
        "source_m1_manifest_sha256": a3["source_m1_manifest_sha256"],
        "source_b2_catalog_manifest_sha256": a3["source_b2_catalog_manifest_sha256"],
        "row_count": row_count,
        "chunk_count": len(chunks),
        "report_sha256": _sha256(report_path),
        "chunks": chunks,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--refresh-summary", action="store_true")
    args = parser.parse_args()
    report = finalize(
        a3_dir=args.a3_dir, labels_dir=args.labels_dir,
        refresh_summary=args.refresh_summary,
    )
    print(json.dumps({key: report[key] for key in (
        "row_count", "month_count", "label_status", "assigned_label_status_by_split",
        "unique_positive_episodes_by_split", "positive_channels_by_split",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
