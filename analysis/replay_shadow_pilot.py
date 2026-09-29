"""Bounded historical shadow replay using M1 only, never future labels or B2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import groupby
import json
from pathlib import Path
import time
from typing import Iterable, Iterator

import duckdb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_a2_hourly import _monthly_files
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.r6_rule import TERMS
from stage1.shadow.stream import Observation, PILOT_VERSION, ShadowStream
from stage1.state_labeling.operational import ARCHIVE_SEGMENTS, segment_at


HOUR = timedelta(hours=1)
COLUMNS = (
    "row_id",
    "channel_id",
    "timestamp",
    "alarm",
    "value_numeric",
    "value_state",
    "sensor_type",
    "object_id",
    "join_status",
    "quality_flags",
    "source",
)
OUTPUT_SCHEMA = pa.schema(
    [
        ("channel_id", pa.string()),
        ("prediction_time", pa.timestamp("us")),
        ("sensor_type", pa.string()),
        *[(name, pa.int64()) for name in TERMS],
        ("admission_status", pa.string()),
        ("admission_reasons", pa.list_(pa.string())),
        ("availability_status", pa.string()),
        ("last_explicit_normal_at", pa.timestamp("us")),
        ("baseline_fit_end_at", pa.timestamp("us")),
        ("history_through", pa.timestamp("us")),
        ("admission_through", pa.timestamp("us")),
        ("pilot_version", pa.string()),
        ("ruleset_version", pa.string()),
        ("rule_score", pa.float64()),
        ("above_frozen_threshold", pa.bool_()),
        ("warning_emitted", pa.bool_()),
        ("warning_status", pa.string()),
    ]
)

B_INPUT_SCHEMA = pa.schema(
    [
        ("channel_id", pa.string()),
        ("sensor_type", pa.string()),
        ("prediction_time", pa.timestamp("us")),
        ("admission_status", pa.string()),
        ("admission_reason", pa.string()),
        ("history_through", pa.timestamp("us")),
        ("admission_through", pa.timestamp("us")),
        *[(name, pa.int64()) for name in TERMS],
    ]
)


def to_b_input(row: dict) -> dict:
    """Explicit unavailable-type sentinel is not an inferred sensor type."""
    result = {name: row[name] for name in B_INPUT_SCHEMA.names if name != "admission_reason"}
    result["sensor_type"] = row["sensor_type"] or "<unknown_or_conflicting>"
    result["admission_reason"] = "|".join(row["admission_reasons"]) or None
    return result


def select_channels(
    database: duckdb.DuckDBPyConnection,
    files: list[Path],
    history_start: datetime,
    start: datetime,
    maximum: int,
) -> list[dict]:
    """One hash-ranked channel per observed type; no replay-period event is read."""
    return (
        database.execute(
            """WITH past AS (
            SELECT channel_id,
                CASE WHEN COUNT(DISTINCT COALESCE(sensor_type,'<unknown>'))=1
                     THEN MIN(COALESCE(sensor_type,'<unknown>'))
                     ELSE '<conflicting>' END AS observed_type
            FROM read_parquet(?,hive_partitioning=false)
            WHERE timestamp>=? AND timestamp<?
              AND split_part(replace(source,chr(92),'/'),'/',-1)=
                  'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z'
            GROUP BY channel_id
        ), ranked AS (
            SELECT *, row_number() OVER (PARTITION BY observed_type ORDER BY sha256(channel_id)) AS rank
            FROM past
        ) SELECT channel_id,observed_type FROM ranked WHERE rank=1
          ORDER BY observed_type LIMIT ?""",
            [[str(path) for path in files], history_start, start, maximum],
        )
        .to_arrow_table()
        .to_pylist()
    )


def observation_groups(rows: Iterable[dict]) -> Iterator[list[Observation]]:
    # A group may straddle source record batches. Never publish a partial second.
    for _, group in groupby(rows, key=lambda row: (row["timestamp"], row["channel_id"])):
        yield [Observation.from_record(row) for row in group]


def replay(
    stream: ShadowStream,
    groups: Iterable[list[Observation]],
    channels: list[str],
    start: datetime,
    end: datetime,
) -> Iterator[tuple[list[dict], float]]:
    if (
        end <= start
        or end - start > timedelta(days=31)
        or any(
            at.tzinfo is not None or (at.minute, at.second, at.microsecond) != (0, 0, 0)
            for at in (start, end)
        )
    ):
        raise ValueError("replay needs whole local hours and at most 31 days")
    pending = iter(groups)
    group = next(pending, None)
    while group is not None and group[0].event.timestamp < start:
        stream.observe_group(group)
        group = next(pending, None)
    at = start
    while at < end:
        started = time.perf_counter()
        while group is not None and group[0].event.timestamp <= at:
            stream.observe_group(group)
            group = next(pending, None)
        result = stream.predict(at, channels)
        yield result, time.perf_counter() - started
        at += HOUR


def run(
    *,
    m1_dir: Path,
    freeze_path: Path,
    contract_path: Path,
    output_dir: Path,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict:
    started = time.perf_counter()
    contract, freeze = read_json(contract_path), read_json(freeze_path)
    if (
        contract["schema_version"] != PILOT_VERSION
        or contract["source_r6_freeze_lf_sha256"] != frozen_rule_sha256(freeze_path)
        or contract["source_m1_manifest_sha256"] != sha256(m1_dir / "manifest.json")
        or freeze["feature_terms"] != TERMS
        or freeze["frozen_threshold"] != 7.1
        or contract["automatic_actions_enabled"] is not False
    ):
        raise ValueError("pilot input or frozen rule lineage differs")
    settings = {
        "baseline_lookback_days": 28,
        "baseline_embargo_hours": 24,
        "maximum_feature_window_hours": 168,
        "event_retention_days": 37,
        "recent_explicit_normal_hours": 168,
    }
    if any(contract.get(key) != value for key, value in settings.items()):
        raise ValueError("pilot admission settings differ from implemented version")
    if read_json(m1_dir / "manifest.json")["status"] != "complete":
        raise ValueError("M1 must be complete")
    start = start or datetime.fromisoformat(contract["first_replay_start"])
    end = end or datetime.fromisoformat(contract["first_replay_end_exclusive"])
    segment = segment_at(start)
    if (
        segment is None
        or end <= start
        or end - start > timedelta(days=31)
        or any((at.minute, at.second, at.microsecond) != (0, 0, 0) for at in (start, end))
        or segment_at(end - HOUR) != segment
    ):
        raise ValueError("replay must stay in one accepted archive segment")
    history_start = ARCHIVE_SEGMENTS[segment][0]
    files, missing = _monthly_files(m1_dir, history_start, end)
    if missing or not files:
        raise ValueError(f"M1 months absent: {missing}")
    maximum = contract["first_replay_max_channels"]
    if not isinstance(maximum, int) or not 1 <= maximum <= 20:
        raise ValueError("diagnostic pilot supports at most 20 channels")
    pending_output = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending_output.exists():
        raise FileExistsError(output_dir)
    process = psutil.Process()
    prediction_counts, reason_counts = Counter(), Counter()
    by_type = defaultdict(Counter)
    scored_days, configured_days = set(), set()
    latency, row_count, max_retained = [], 0, 0
    pending_output.mkdir(parents=True)
    output_file = pending_output / "shadow_predictions.parquet"
    b_input_file = pending_output / "hours.parquet"
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        database.execute("SET memory_limit='2GB'")
        selected = select_channels(database, files, history_start, start, maximum)
        channels = [row["channel_id"] for row in selected]
        if not channels:
            raise ValueError("no channels observed before replay start")
        print(f"selected {len(channels)} channels using pre-start observations", flush=True)
        database.execute(
            "CREATE TEMP TABLE selected_channels AS SELECT UNNEST(?::VARCHAR[]) AS channel_id",
            [channels],
        )
        reader = database.execute(
            "SELECT "
            + ",".join(COLUMNS)
            + """
            FROM read_parquet(?,hive_partitioning=false) e
            SEMI JOIN selected_channels USING(channel_id)
            WHERE timestamp>=? AND timestamp<?
              AND split_part(replace(source,chr(92),'/'),'/',-1)=
                  'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z'
            ORDER BY timestamp,channel_id,row_id""",
            [[str(path) for path in files], history_start, end],
        ).to_arrow_reader(batch_size=25_000)
        rows = (row for batch in reader for row in batch.to_pylist())
        stream = ShadowStream(
            threshold=freeze["frozen_threshold"],
            cooldown_hours=freeze["per_channel_warning_cooldown_hours"],
        )
        with (
            pq.ParquetWriter(output_file, OUTPUT_SCHEMA, compression="zstd") as writer,
            pq.ParquetWriter(b_input_file, B_INPUT_SCHEMA, compression="zstd") as b_writer,
        ):
            for result, seconds in replay(stream, observation_groups(rows), channels, start, end):
                writer.write_table(pa.Table.from_pylist(result, schema=OUTPUT_SCHEMA))
                b_writer.write_table(
                    pa.Table.from_pylist([to_b_input(row) for row in result], schema=B_INPUT_SCHEMA)
                )
                latency.append(seconds)
                row_count += len(result)
                max_retained = max(max_retained, stream.retained_event_count)
                for row in result:
                    available = row["rule_score"] is not None
                    prediction_counts[row["admission_status"]] += 1
                    reason_counts.update(row["admission_reasons"])
                    kind = row["sensor_type"] or "<unknown>"
                    by_type[kind]["hours"] += 1
                    by_type[kind]["scored_hours"] += int(available)
                    by_type[kind]["emitted_research_warnings"] += int(row["warning_emitted"])
                    key = row["channel_id"], row["prediction_time"].date()
                    configured_days.add(key)
                    if available:
                        scored_days.add(key)
                if result[0]["prediction_time"].hour == 0:
                    print(
                        f"replayed through {result[0]['prediction_time']}; "
                        f"accepted observations={stream.accepted_rows}",
                        flush=True,
                    )
    expected = len(channels) * int((end - start) / HOUR)
    if row_count != expected:
        raise ValueError("requested channel-hour grid is incomplete")
    scored = prediction_counts["eligible"]
    warnings = sum(item["emitted_research_warnings"] for item in by_type.values())
    sources = [
        {
            "file": path.relative_to(m1_dir).as_posix(),
            "sha256": sha256(path),
            "metadata_rows": pq.ParquetFile(path).metadata.num_rows,
        }
        for path in files
    ]
    memory = process.memory_info()
    report = {
        "schema_version": PILOT_VERSION,
        "status": "a_historical_shadow_replay_complete_b_joint_review_pending",
        "purpose": "technical_process_check_not_new_model_quality",
        "source_m1_manifest_sha256": sha256(m1_dir / "manifest.json"),
        "source_freeze_lf_sha256": frozen_rule_sha256(freeze_path),
        "source_pilot_contract_lf_sha256": frozen_rule_sha256(contract_path),
        "code_lf_sha256": {
            name: frozen_rule_sha256(Path(__file__).resolve().parents[1] / name)
            for name in (
                "analysis/replay_shadow_pilot.py",
                "stage1/shadow/stream.py",
                "stage1/features/hourly.py",
                "stage1/features/r2.py",
                "stage1/state_labeling/registered_episodes.py",
                "stage1/state_labeling/operational.py",
                "stage1/state_labeling/rules.py",
                "ml/forecast/r6_rule.py",
            )
        },
        "sources": sources,
        "selection_uses_only_pre_start_observations": True,
        "selected_channels": selected,
        "history_start": history_start.isoformat(),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "configured_channel_hours": row_count,
        "conditionally_scored_hours": scored,
        "coverage": scored / row_count,
        "availability_status": "unknown_for_every_row",
        "admission_status_counts": dict(prediction_counts),
        "reason_counts": dict(reason_counts),
        "by_type": {
            kind: {**dict(counts), "coverage": counts["scored_hours"] / counts["hours"]}
            for kind, counts in sorted(by_type.items())
        },
        "emitted_research_warnings": warnings,
        "configured_channel_days": len(configured_days),
        "scored_channel_days": len(scored_days),
        "warnings_per_1000_configured_channel_days": warnings * 1000 / len(configured_days),
        "accepted_past_observations": stream.accepted_rows,
        "maximum_retained_events_at_prediction": max_retained,
        "resources": {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
            "hour_processing_p95_seconds": float(np.quantile(latency, 0.95)),
            "hour_processing_max_seconds": max(latency),
            "hour_latency_scope": "event delivery in local replay and scoring; initial historical warmup excluded",
            "memory_measurement": "Windows OS process peak; current RSS fallback elsewhere",
            "duckdb_threads": 2,
            "duckdb_memory_limit": "2GB",
        },
        "frozen_threshold": 7.1,
        "automatic_actions_enabled": False,
        "deployment_approved": False,
        "physical_failure_claim": False,
        "quality_metrics_computed": False,
        "limitations": [
            "Only selected channels and one fixed diagnostic week, not full M1 population.",
            "No B2 catalog, A3 features, R3 index, target or future label read by inference.",
            "Complete event-time groups are assumed; actual arrival latency is unavailable.",
            "Conditional admission requires joint B review; channel continuity remains unknown.",
            "Restart checkpointing and a real future-source contract are not implemented.",
            "No precision/recall or customer-approved coverage, warning or resource limits.",
        ],
    }
    report_file = pending_output / "report.json"
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": PILOT_VERSION,
        "status": report["status"],
        "report_sha256": sha256(report_file),
        "prediction_rows": row_count,
        "prediction_file": output_file.name,
        "prediction_sha256": sha256(output_file),
        "b_input_file": b_input_file.name,
        "b_input_sha256": sha256(b_input_file),
    }
    (pending_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    pending_output.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=Path("ml/r6_frozen_rule_v1.json"))
    parser.add_argument("--contract", type=Path, default=Path("ml/shadow_pilot_contract_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=datetime.fromisoformat)
    parser.add_argument("--end", type=datetime.fromisoformat)
    args = parser.parse_args()
    result = run(
        m1_dir=args.m1_dir,
        freeze_path=args.freeze,
        contract_path=args.contract,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "configured_channel_hours",
                    "conditionally_scored_hours",
                    "coverage",
                    "emitted_research_warnings",
                    "resources",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
