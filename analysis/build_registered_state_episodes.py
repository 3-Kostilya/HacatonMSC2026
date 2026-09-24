"""Build the B2 registered-state episode catalog from the complete M1 artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from stage1.state_labeling.operational import source_is_full_archive  # noqa: E402
from stage1.state_labeling.registered_episodes import (  # noqa: E402
    EPISODE_VERSION,
    EpisodeBuilder,
    RegisteredEpisode,
    StateEvent,
)
from stage1.state_labeling.rules import RULESET_VERSION, TARGET_DEFINITION  # noqa: E402


EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("channel_id", pa.string(), nullable=False),
        pa.field("sensor_type", pa.string(), nullable=False),
        pa.field("target_kind", pa.string(), nullable=False),
        pa.field("start_at", pa.timestamp("us"), nullable=False),
        pa.field("confirmed_at", pa.timestamp("us"), nullable=False),
        pa.field("end_at", pa.timestamp("us")),
        pa.field("onset_status", pa.string(), nullable=False),
        pa.field("end_status", pa.string(), nullable=False),
        pa.field("prior_normal_at", pa.timestamp("us")),
        pa.field("first_row_id", pa.int64(), nullable=False),
        pa.field("last_fault_at", pa.timestamp("us"), nullable=False),
        pa.field("fault_message_count", pa.int64(), nullable=False),
        pa.field("uncertain_intervening_state", pa.bool_(), nullable=False),
        pa.field("evidence", pa.list_(pa.string()), nullable=False),
        pa.field("ruleset_version", pa.string(), nullable=False),
        pa.field("episode_version", pa.string(), nullable=False),
    ]
)


def _validated_m1(input_manifest: Path) -> tuple[list[Path], dict[str, Any]]:
    manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "ingestion-v1"
        or manifest.get("status") != "complete"
        or manifest.get("scope") != "full_supplied_sources"
    ):
        raise ValueError("B2 requires a complete full M1 artifact")
    quality = json.loads((input_manifest.parent / "data_quality.json").read_text(encoding="utf-8"))
    if quality.get("scope") != manifest["scope"] or quality.get("input_rows") != manifest.get("input_rows"):
        raise ValueError("M1 manifest and quality report disagree")
    files = _validated_clean_files(input_manifest.parent, quality)
    files.sort(key=lambda path: (
        int(path.parent.parent.name.removeprefix("year=")),
        int(path.parent.name.removeprefix("month=")),
        path.name,
    ))
    years = {int(path.parent.parent.name.removeprefix("year=")) for path in files}
    if years != set(TARGET_DEFINITION["included_years"]):
        raise ValueError(f"M1 clean years differ from accepted R1: {sorted(years)}")
    return files, quality


def _scan(files: list[Path], spill: Path) -> tuple[EpisodeBuilder, Counter[str]]:
    spill.mkdir()
    database = duckdb.connect(":memory:")
    database.execute("SET memory_limit='4GB'")
    database.execute("SET threads=2")
    database.execute("SET temp_directory=?", [str(spill)])
    builder = EpisodeBuilder()
    source_counts: Counter[str] = Counter()
    try:
        for index, path in enumerate(files, start=1):
            reader = database.execute(
                """
                SELECT row_id, channel_id, sensor_type, timestamp, value_state, alarm, source
                FROM read_parquet(?, hive_partitioning=false)
                WHERE value_state IS NOT NULL AND channel_id IS NOT NULL
                ORDER BY timestamp, channel_id, row_id
                """,
                [str(path)],
            ).to_arrow_reader(batch_size=50_000)
            for batch in reader:
                for row in batch.to_pylist():
                    at = row["timestamp"]
                    if not source_is_full_archive(row["source"], at):
                        source_counts["excluded_example_or_mismatched_source"] += 1
                        continue
                    source_counts["accepted_text_rows"] += 1
                    builder.add(
                        StateEvent(
                            row_id=row["row_id"],
                            channel_id=row["channel_id"],
                            sensor_type=row["sensor_type"],
                            at=at,
                            value_state=row["value_state"],
                            alarm=row["alarm"],
                        )
                    )
            print(f"B2 scan {index}/{len(files)}: {path.parent.parent.name}/{path.parent.name}", flush=True)
        builder.finish()
    finally:
        database.close()
        if not any(spill.iterdir()):
            spill.rmdir()
    return builder, source_counts


def _report(builder: EpisodeBuilder, source_counts: Counter[str]) -> dict[str, Any]:
    episodes = builder.episodes
    if sum(item.fault_message_count for item in episodes) != builder.message_counts["known_fault_rows"]:
        raise ValueError("known-type fault messages were lost or duplicated across episodes")
    if len({item.episode_id for item in episodes}) != len(episodes):
        raise ValueError("episode IDs are not unique")
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_year: dict[int, Counter[str]] = defaultdict(Counter)
    by_channel: dict[str, Counter[str]] = defaultdict(Counter)
    onset_counts: Counter[str] = Counter()
    end_counts: Counter[str] = Counter()
    short_returns: Counter[str] = Counter()
    sensitivity: Counter[str] = Counter()
    previous: dict[tuple[str, str], RegisteredEpisode] = {}
    for item in episodes:
        year = item.start_at.year
        for group in (by_type[item.sensor_type], by_year[year], by_channel[item.channel_id]):
            group["episodes"] += 1
            group["fault_messages"] += item.fault_message_count
            group[item.onset_status] += 1
            if item.end_at is None:
                group["open"] += 1
            if item.uncertain_intervening_state or item.onset_status != "candidate_new_onset":
                group["doubtful"] += 1
        onset_counts[item.onset_status] += 1
        end_counts[item.end_status] += 1
        if item.prior_normal_at is not None and item.onset_status in {
            "candidate_new_onset", "stale_normal"
        }:
            age = (item.start_at - item.prior_normal_at).total_seconds() / 3600
            for hours in (24, 72, 168, 336):
                if 0 < age <= hours:
                    sensitivity[f"prior_normal_within_{hours}h"] += 1
        key = (item.channel_id, item.sensor_type)
        old = previous.get(key)
        if old is not None and old.end_at is not None and old.end_at < item.start_at:
            delta = (item.start_at - old.end_at).total_seconds() / 3600
            bucket = "le_1h" if delta <= 1 else "le_6h" if delta <= 6 else "le_24h" if delta <= 24 else "gt_24h"
            short_returns[bucket] += 1
            if delta <= 1:
                by_type[item.sensor_type]["return_within_1h"] += 1
        previous[key] = item
    return {
        "schema_version": EPISODE_VERSION,
        "ruleset_version": RULESET_VERSION,
        "target_kind": TARGET_DEFINITION["target_kind"],
        "source_counts": dict(source_counts),
        "message_counts": builder.message_counts,
        "episode_count": len(episodes),
        "channel_count": len(by_channel),
        "onset_status": dict(sorted(onset_counts.items())),
        "end_status": dict(sorted(end_counts.items())),
        "short_returns_after_explicit_norma": dict(short_returns),
        "prior_normal_sensitivity": dict(sensitivity),
        "by_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "by_year": {str(key): dict(value) for key, value in sorted(by_year.items())},
        "by_channel": {key: dict(value) for key, value in sorted(by_channel.items())},
        "meaning": "registered journal episodes, not verified physical failures",
        "coverage": "full archive assumed; channel continuity not verified",
    }


def build(input_manifest: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    input_manifest = input_manifest.resolve()
    output = output.resolve()
    pending = output.with_name(output.name + ".inprogress")
    if output.exists() or pending.exists():
        raise FileExistsError("B2 output or .inprogress directory already exists")
    files, quality = _validated_m1(input_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    builder, source_counts = _scan(files, pending / "spill")
    report = _report(builder, source_counts)
    table = pa.Table.from_pylist([item.as_dict() for item in builder.episodes], schema=EPISODE_SCHEMA)
    episodes_path = pending / "registered_state_episodes.parquet"
    pq.write_table(table, episodes_path, compression="zstd")
    report_path = pending / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": EPISODE_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": _sha256(input_manifest),
        "input_data_quality_sha256": _sha256(input_manifest.parent / "data_quality.json"),
        "m1_accepted_rows": quality["dispositions"]["accepted"],
        "ruleset_version": RULESET_VERSION,
        "episode_count": table.num_rows,
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (episodes_path, report_path)
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
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input_manifest, args.output)
    print(json.dumps({"output": str(args.output.resolve()), **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
