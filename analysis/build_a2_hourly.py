"""Build a bounded, reproducible A2 hourly feature slice from published M1 Parquet.

This command deliberately requires explicit time bounds and at most 20 channels.
It reads only monthly partitions intersecting the required causal lookback and
never creates a channel-by-all-history Cartesian grid.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.features import (  # noqa: E402
    A2_SCHEMA,
    FEATURE_VERSION,
    FeatureEvent,
    HourlyConfig,
    build_hourly_rows,
    validate_a2_table,
)
from stage1.registry import load_registry  # noqa: E402


MAX_CHANNELS = 20
MAX_HOURS = 31 * 24
MAX_EVENTS_PER_CHANNEL = 500_000
# A conservative bound: 28-day baseline + 168-hour rolling window + 24-hour
# baseline embargo. It is deliberately larger than any one feature's need.
CAUSAL_LOOKBACK = timedelta(days=28, hours=168 + 24)
CLEAN_COLUMNS = (
    "channel_id",
    "timestamp",
    "alarm",
    "value_numeric",
    "value_state",
    "sensor_type",
    "object_id",
    "join_status",
    "quality_flags",
)


def _hour(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid local timestamp: {value!r}") from exc
    if result.tzinfo is not None or (result.minute, result.second, result.microsecond) != (0, 0, 0):
        raise ValueError("start/end must be local naive timestamps at the top of an hour")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_channel_stats(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"M1 channel statistics are missing: {path}")
    table = pq.read_table(path)
    required = {"channel_id", "sensor_type", "rows", "first_at", "last_at"}
    if not required.issubset(table.schema.names):
        raise ValueError(f"channel statistics lack {sorted(required - set(table.schema.names))}")
    return table.to_pylist()


def select_real_20(
    stats: list[dict[str, Any]],
    start_at: datetime,
    end_at: datetime,
    *,
    seed: int,
    target_counts: Counter[str] | None = None,
) -> dict[str, str | None]:
    """Select one active channel per registered type plus one unknown channel.

    Selection is stratified and SHA-based so it is stable across Python hash
    seeds. When target counts are supplied, only channels actually observed in
    the requested interval qualify. The top quartile by interval activity keeps
    the sample useful; the rare hatch type is retained even when sparse.
    """
    type_names = sorted(policy.sensor_type for policy in load_registry())
    selected: dict[str, str | None] = {}
    for sensor_type in [*type_names, None]:
        candidates = [
            row
            for row in stats
            if row["sensor_type"] == sensor_type
            and row["first_at"] is not None
            and row["last_at"] is not None
            and row["first_at"] < end_at
            and row["last_at"] >= start_at
            and int(row["rows"]) > 0
            and (target_counts is None or target_counts[str(row["channel_id"])] > 0)
        ]
        if not candidates:
            name = sensor_type if sensor_type is not None else "<unknown>"
            raise ValueError(f"no active M1 channel for type {name!r} in requested period")
        candidates.sort(
            key=lambda row: (
                -(
                    target_counts[str(row["channel_id"])]
                    if target_counts is not None
                    else int(row["rows"])
                ),
                str(row["channel_id"]),
            )
        )
        top_quartile = candidates[: max(1, (len(candidates) + 3) // 4)]

        def rank(row: dict[str, Any]) -> bytes:
            key = f"{seed}\0{sensor_type}\0{row['channel_id']}".encode("utf-8")
            return hashlib.sha256(key).digest()

        chosen = min(top_quartile, key=lambda row: (rank(row), str(row["channel_id"])))
        channel_id = str(chosen["channel_id"])
        if channel_id in selected:
            raise ValueError(f"channel appears in several sensor types: {channel_id}")
        selected[channel_id] = sensor_type
    if len(selected) != MAX_CHANNELS:
        raise ValueError("real-20 selection did not produce exactly 20 unique channels")
    return dict(sorted(selected.items()))


def _channels_from_file(path: Path, stats: list[dict[str, Any]]) -> dict[str, str | None]:
    content = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        requested = json.loads(content)
        if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
            raise ValueError("channels JSON must be an array of channel ID strings")
    else:
        requested = [line.strip() for line in content.splitlines() if line.strip()]
    if not 1 <= len(requested) <= MAX_CHANNELS or len(set(requested)) != len(requested):
        raise ValueError("channels file must contain 1-20 unique channel IDs")
    stats_by_channel = {str(row["channel_id"]): row["sensor_type"] for row in stats}
    missing = sorted(set(requested) - set(stats_by_channel))
    if missing:
        raise ValueError(f"channels absent from M1 statistics: {missing}")
    return {channel_id: stats_by_channel[channel_id] for channel_id in sorted(requested)}


def _monthly_files(
    artifact: Path, from_at: datetime, end_at: datetime
) -> tuple[list[Path], list[str]]:
    month = datetime(from_at.year, from_at.month, 1)
    files: list[Path] = []
    missing: list[str] = []
    while month < end_at:
        directory = artifact / "clean" / f"year={month.year}" / f"month={month.month}"
        members = sorted(directory.glob("*.parquet"))
        if members:
            files.extend(members)
        else:
            missing.append(month.strftime("%Y-%m"))
        month = datetime(month.year + (month.month == 12), month.month % 12 + 1, 1)
    return files, missing


def _target_channel_counts(files: list[Path], start_at: datetime, end_at: datetime) -> Counter[str]:
    """Count target-interval presence with one bounded, column-pruned scan."""
    counts: Counter[str] = Counter()
    if not files:
        return counts
    parquet = ds.dataset(files, format="parquet")
    predicate = (pc.field("timestamp") >= pa.scalar(start_at)) & (
        pc.field("timestamp") < pa.scalar(end_at)
    )
    for batch in parquet.to_batches(
        columns=["channel_id"], filter=predicate, batch_size=131_072, use_threads=False
    ):
        for item in pc.value_counts(batch.column("channel_id")).to_pylist():
            if item["values"] is not None:
                counts[str(item["values"])] += int(item["counts"])
    return counts


def _events_for_channel(
    parquet: ds.Dataset, channel_id: str, from_at: datetime, end_at: datetime
) -> list[FeatureEvent]:
    predicate = (
        (pc.field("channel_id") == channel_id)
        & (pc.field("timestamp") >= pa.scalar(from_at))
        & (pc.field("timestamp") < pa.scalar(end_at))
    )
    events: list[FeatureEvent] = []
    for batch in parquet.to_batches(
        columns=list(CLEAN_COLUMNS), filter=predicate, batch_size=16_384, use_threads=False
    ):
        if len(events) + batch.num_rows > MAX_EVENTS_PER_CHANNEL:
            raise ValueError(
                f"{channel_id}: more than {MAX_EVENTS_PER_CHANNEL} events in bounded context"
            )
        events.extend(FeatureEvent.from_clean_record(row) for row in batch.to_pylist())
    events.sort(key=lambda item: item.timestamp)
    return events


def build_slice(
    *,
    input_manifest: Path,
    start_at: datetime,
    end_at: datetime,
    output: Path,
    channels_file: Path | None = None,
    select_real_20_channels: bool = False,
    seed: int = 20260923,
) -> dict[str, Any]:
    """Build and atomically publish one short A2 slice; never overwrite outputs.

    On failure the ``.inprogress`` directory is retained for inspection, but
    there is no complete manifest and a retry must use a new output path.
    """
    if (channels_file is None) == (not select_real_20_channels):
        raise ValueError("choose exactly one of channels_file or select_real_20_channels")
    if start_at.tzinfo is not None or end_at.tzinfo is not None:
        raise ValueError("A2 timestamps must be local naive")
    if any((start_at.minute, start_at.second, start_at.microsecond)) or any(
        (end_at.minute, end_at.second, end_at.microsecond)
    ):
        raise ValueError("A2 start/end must be at the top of an hour")
    if not start_at < end_at <= start_at + timedelta(hours=MAX_HOURS):
        raise ValueError(f"A2 interval must be positive and at most {MAX_HOURS} hours")
    if start_at < datetime(2022, 1, 1) and end_at > datetime(2021, 1, 1):
        raise ValueError("2021 is an explicitly excluded source gap; choose another A2 interval")

    input_manifest = input_manifest.resolve()
    artifact = input_manifest.parent
    output = output.resolve()
    pending = output.with_name(output.name + ".inprogress")
    if output.exists() or pending.exists():
        raise FileExistsError(
            "A2 output or its .inprogress directory exists; inspect it and choose a new path"
        )
    if artifact == output or artifact in output.parents:
        raise ValueError("A2 output must be outside the published M1 artifact")
    manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "ingestion-v1" or manifest.get("status") != "complete":
        raise ValueError("A2 requires a complete ingestion-v1 manifest")
    if manifest.get("scope") != "full_supplied_sources":
        raise ValueError("A2 real-data validation requires a full-history M1 artifact")
    manifest_sha = _sha256(input_manifest)
    stats = _read_channel_stats(artifact / "sensor_statistics.parquet")
    context_start = start_at - CAUSAL_LOOKBACK
    files, missing_months = _monthly_files(artifact, context_start, end_at)
    if not files:
        raise ValueError("M1 has no clean Parquet in the requested interval and causal context")
    target_counts: Counter[str] | None = None
    if select_real_20_channels:
        target_files, _ = _monthly_files(artifact, start_at, end_at)
        target_counts = _target_channel_counts(target_files, start_at, end_at)
        channels = select_real_20(stats, start_at, end_at, seed=seed, target_counts=target_counts)
        selection_mode = "seeded_type_stratified_real_20_v1"
    else:
        assert channels_file is not None
        channels = _channels_from_file(channels_file, stats)
        selection_mode = "explicit_channels_file_v1"
    parquet = ds.dataset(files, format="parquet")
    missing_columns = set(CLEAN_COLUMNS) - set(parquet.schema.names)
    if missing_columns:
        raise ValueError(f"M1 clean Parquet lacks A2 fields: {sorted(missing_columns)}")
    feature_config = HourlyConfig()
    config = {
        "feature_version": FEATURE_VERSION,
        "input_manifest_sha256": manifest_sha,
        "start_at": start_at,
        "end_at": end_at,
        "causal_context_start_at": context_start,
        "channels": channels,
        "selection_mode": selection_mode,
        "selection_seed": seed if select_real_20_channels else None,
        "feature_config": feature_config,
        "max_events_per_channel": MAX_EVENTS_PER_CHANNEL,
    }
    config_sha = _canonical_sha256(config)
    provenance = {
        "schema_version": FEATURE_VERSION,
        "run_id": f"a2-{config_sha[:16]}",
        "config_sha256": config_sha,
        "input_manifest_sha256": manifest_sha,
    }

    quality_path = artifact / "data_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    object_mapping_available = quality.get("dictionary_audit", {}).get("object_mapping_available")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    feature_path = pending / "features.parquet"
    by_channel: list[dict[str, Any]] = []
    availability_counts: Counter[str] = Counter()
    numeric_branch_rows = state_branch_rows = mixed_branch_rows = 0
    total_rows = 0
    expected_fields = set(A2_SCHEMA.names) - set(provenance)
    writer = pq.ParquetWriter(feature_path, A2_SCHEMA, compression="zstd")
    try:
        for channel_id, sensor_type in channels.items():
            events = _events_for_channel(parquet, channel_id, context_start, end_at)
            rows = build_hourly_rows(events, channel_id, start_at, end_at, config=feature_config)
            if any(set(row) != expected_fields for row in rows):
                raise ValueError(f"{channel_id}: core row fields differ from A2_SCHEMA")
            for row in rows:
                row.update(provenance)
                availability_counts[str(row["availability_status"])] += 1
                numeric_count = row.get("numeric_count_24h") or 0
                state_count = row.get("state_count_24h") or 0
                numeric_branch_rows += int(numeric_count > 0)
                state_branch_rows += int(state_count > 0)
                mixed_branch_rows += int(numeric_count > 0 and state_count > 0)
            table = pa.Table.from_pylist(rows, schema=A2_SCHEMA)
            validate_a2_table(table)
            writer.write_table(table)
            total_rows += table.num_rows
            by_channel.append(
                {
                    "channel_id": channel_id,
                    "sensor_type": sensor_type,
                    "input_events_in_context": len(events),
                    "input_events_in_target": (
                        target_counts[channel_id]
                        if target_counts is not None
                        else sum(start_at <= event.timestamp < end_at for event in events)
                    ),
                    "output_hours": table.num_rows,
                    "first_input_at": events[0].timestamp.isoformat() if events else None,
                    "last_input_at": events[-1].timestamp.isoformat() if events else None,
                }
            )
    finally:
        writer.close()

    expected_rows = len(channels) * int((end_at - start_at).total_seconds() // 3600)
    if total_rows != expected_rows:
        raise ValueError(f"A2 row count mismatch: {total_rows} != {expected_rows}")
    physical_rows = pq.ParquetFile(feature_path).metadata.num_rows
    if physical_rows != total_rows:
        raise ValueError("A2 physical Parquet row count mismatch")
    numeric_gap = {
        "hours_without_numeric_24h": total_rows - numeric_branch_rows,
        "channels_without_any_context_event": [
            item["channel_id"] for item in by_channel if item["input_events_in_context"] == 0
        ],
        "numeric_branch_24h_rows": numeric_branch_rows,
        "state_branch_24h_rows": state_branch_rows,
        "mixed_branch_24h_rows": mixed_branch_rows,
        "pure_numeric_lifetime_channels": sum(
            int(row["numeric_count"]) > 0 and int(row["state_count"]) == 0 for row in stats
        ),
        "pure_numeric_lifetime_channels_with_28d_span": sum(
            int(row["numeric_count"]) > 0
            and int(row["state_count"]) == 0
            and row["first_at"] is not None
            and row["last_at"] is not None
            and row["last_at"] - row["first_at"] >= timedelta(days=28)
            for row in stats
        ),
    }
    validation = {
        "schema_version": FEATURE_VERSION,
        "input_manifest_sha256": manifest_sha,
        "selection_mode": selection_mode,
        "real_20_channels_selected": len(channels) == 20 and select_real_20_channels,
        "known_types_represented": sorted({value for value in channels.values() if value}),
        "unknown_channel_represented": None in channels.values(),
        "selection_uses_target_presence_for_QA_only": select_real_20_channels,
        "channel_count": len(channels),
        "hour_count": total_rows,
        "expected_hour_count": expected_rows,
        "availability_counts": dict(sorted(availability_counts.items())),
        "numeric_coverage": numeric_gap,
        "object_mapping_available": object_mapping_available,
        "peer_features_status": "unavailable"
        if object_mapping_available is not True
        else "not_built",
        "missing_context_or_target_months": missing_months,
        "channels": by_channel,
    }
    published_manifest = {
        "schema_version": FEATURE_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": manifest_sha,
        "config": _jsonable(config),
        "config_sha256": config_sha,
        "run_id": provenance["run_id"],
        "features_file": "features.parquet",
        "features_sha256": _sha256(feature_path),
        "feature_rows": total_rows,
        "feature_bytes": feature_path.stat().st_size,
        "validation_report": "validation_report.json",
    }
    (pending / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (pending / "manifest.json").write_text(
        json.dumps(published_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending.rename(output)
    return {"manifest": published_manifest, "validation": validation, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument(
        "--start", required=True, help="Inclusive local hour, e.g. 2025-06-20T00:00:00"
    )
    parser.add_argument(
        "--end", required=True, help="Exclusive local hour, e.g. 2025-06-27T00:00:00"
    )
    parser.add_argument("--output", required=True, type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--channels-file", type=Path)
    choice.add_argument("--select-real-20", action="store_true")
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    result = build_slice(
        input_manifest=args.input_manifest,
        start_at=_hour(args.start),
        end_at=_hour(args.end),
        output=args.output,
        channels_file=args.channels_file,
        select_real_20_channels=args.select_real_20,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": result["output"],
                "run_id": result["manifest"]["run_id"],
                "feature_rows": result["manifest"]["feature_rows"],
                "channel_count": result["validation"]["channel_count"],
                "availability_counts": result["validation"]["availability_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
