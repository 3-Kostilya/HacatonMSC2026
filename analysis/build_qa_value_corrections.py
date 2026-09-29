"""Build sparse, QA-adjusted corrections for every affected published A3 hour.

Only observations not already excluded by the established quality policy can
change existing numeric or state-transition features. The original R3 pack is
immutable; each correction row keeps its unchanged R2 fields and provenance.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_a2_hourly import _monthly_files
from analysis.recompute_qa_value_features import COMPARE_FIELDS, _same, _sha256
from stage1.features.hourly import FeatureEvent, HourlyConfig
from stage1.features.qa_values import qa_window_counts
from stage1.features.r3 import FEATURE_PACK_SCHEMA
from stage1.features.r3_full import build_selected_hourly_rows
from stage1.state_labeling.operational import source_is_full_archive
from stage1.value_quality import (
    EPOCH_VALUE_ARTIFACTS,
    QA_VALUE_RULESET_VERSION,
    TEMPERATURE_SERVICE_CODE_CANDIDATES,
    assess_value,
)


def _affected_events(con: duckdb.DuckDBPyConnection, m1_dir: Path) -> list[dict]:
    paths = sorted((m1_dir / "clean").rglob("*.parquet"))
    if not paths:
        raise ValueError("M1 clean Parquet is missing")
    numeric_placeholders = ",".join("?" for _ in TEMPERATURE_SERVICE_CODE_CANDIDATES)
    epoch_placeholders = ",".join("?" for _ in EPOCH_VALUE_ARTIFACTS)
    excluded = HourlyConfig().excluded_quality_flags
    flag_condition = " OR ".join("list_contains(quality_flags, ?)" for _ in excluded)
    query = f"""
        SELECT channel_id,timestamp,sensor_type,value_raw,value_numeric,value_state,
               quality_flags,source
        FROM read_parquet(?,hive_partitioning=false)
        WHERE ((sensor_type='Датчик температуры' AND value_numeric IN ({numeric_placeholders}))
            OR (sensor_type='Газовый датчик' AND value_numeric>100)
            OR value_state IN ({epoch_placeholders}))
          AND NOT ({flag_condition})
        ORDER BY channel_id,timestamp
    """
    params = [
        [str(path) for path in paths],
        *sorted(TEMPERATURE_SERVICE_CODE_CANDIDATES),
        *sorted(EPOCH_VALUE_ARTIFACTS),
        *sorted(excluded),
    ]
    result = con.execute(query, params)
    fields = [item[0] for item in result.description]
    rows = [dict(zip(fields, row)) for row in result.fetchall()]
    output = []
    for row in rows:
        assessment = assess_value(row["sensor_type"], row["value_raw"], row["value_numeric"])
        if not source_is_full_archive(row["source"], row["timestamp"]):
            continue
        if not assessment.numeric_measurement_usable or not assessment.state_transition_usable:
            output.append({**row, "qa_category": assessment.category})
    return output


def _a3_files(a3_dir: Path, start: datetime, end: datetime) -> list[Path]:
    month = datetime(start.year, start.month, 1)
    files = []
    while month < end:
        path = a3_dir / f"year={month.year}" / f"month={month.month:02d}" / "features.parquet"
        if path.is_file():
            files.append(path)
        month = datetime(month.year + (month.month == 12), month.month % 12 + 1, 1)
    return files


def _records(con: duckdb.DuckDBPyConnection, query: str, params: list) -> list[dict]:
    result = con.execute(query, params)
    fields = [item[0] for item in result.description]
    return [dict(zip(fields, row)) for row in result.fetchall()]


def build(m1_dir: Path, a3_dir: Path, output_dir: Path) -> dict:
    m1_dir, a3_dir, output_dir = (path.resolve() for path in (m1_dir, a3_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    con = duckdb.connect()
    try:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='2GB'")
        events = _affected_events(con, m1_dir)
        corrected: dict[tuple[str, datetime], dict] = {}
        quality: dict[tuple[str, datetime], dict] = {}
        changed_fields: Counter[str] = Counter()
        candidate_keys = 0
        for event in events:
            channel_id, event_time = event["channel_id"], event["timestamp"]
            # Seven-day windows and the delayed 28-day baseline are both covered.
            candidate_end = event_time + timedelta(days=37)
            a3_paths = _a3_files(a3_dir, event_time, candidate_end)
            if not a3_paths:
                continue
            saved_rows = _records(
                con,
                "SELECT * FROM read_parquet(?,hive_partitioning=false) "
                "WHERE channel_id=? AND prediction_time>=? AND prediction_time<? "
                "ORDER BY prediction_time",
                [[str(path) for path in a3_paths], channel_id, event_time, candidate_end],
            )
            if not saved_rows:
                continue
            candidate_keys += len(saved_rows)
            times = [row["prediction_time"] for row in saved_rows]
            baseline_start = min(t.replace(hour=0) for t in times) - timedelta(hours=192, days=28)
            context_end = max(times) + timedelta(microseconds=1)
            m1_paths, missing = _monthly_files(m1_dir, baseline_start, context_end)
            if missing:
                raise ValueError(f"missing M1 context for {channel_id}: {missing}")
            source_rows = _records(
                con,
                "SELECT channel_id,timestamp,alarm,value_numeric,value_state,sensor_type,"
                "object_id,join_status,quality_flags,source,value_raw "
                "FROM read_parquet(?,hive_partitioning=false) "
                "WHERE channel_id=? AND timestamp>=? AND timestamp<? "
                "ORDER BY timestamp,row_id",
                [[str(path) for path in m1_paths], channel_id, baseline_start, context_end],
            )
            source_rows = [
                row
                for row in source_rows
                if source_is_full_archive(row["source"], row["timestamp"])
            ]
            legacy_events = [FeatureEvent.from_clean_record(row) for row in source_rows]
            qa_events = [
                FeatureEvent.from_clean_record(row, apply_qa_value_policy=True)
                for row in source_rows
            ]
            old_rows = build_selected_hourly_rows(legacy_events, channel_id, times)
            new_rows = build_selected_hourly_rows(qa_events, channel_id, times)
            for saved, old, new in zip(saved_rows, old_rows, new_rows):
                mismatch = [name for name in COMPARE_FIELDS if not _same(saved[name], old[name])]
                if mismatch:
                    raise ValueError(
                        f"published A3 parity failed for {channel_id} "
                        f"{saved['prediction_time']}: {mismatch}"
                    )
                changes = {
                    name: new[name] for name in COMPARE_FIELDS if not _same(old[name], new[name])
                }
                if not changes:
                    continue
                key = channel_id, saved["prediction_time"]
                patched = {**saved, **changes}
                if key in corrected:
                    if any(
                        not _same(corrected[key][name], patched[name])
                        for name in FEATURE_PACK_SCHEMA.names
                    ):
                        raise ValueError(f"overlapping event corrections disagree for {key}")
                else:
                    corrected[key] = patched
                    quality[key] = {
                        "channel_id": channel_id,
                        "prediction_time": saved["prediction_time"],
                        **qa_window_counts(qa_events, saved["prediction_time"]),
                    }
                    changed_fields.update(changes.keys())
        corrected_rows = [corrected[key] for key in sorted(corrected)]
        qa_rows = [quality[key] for key in sorted(quality)]
    finally:
        con.close()

    output_dir.mkdir(parents=True)
    feature_path = output_dir / "feature_corrections.parquet"
    quality_path = output_dir / "qa_quality_counts.parquet"
    pq.write_table(pa.Table.from_pylist(corrected_rows, schema=FEATURE_PACK_SCHEMA), feature_path)
    pq.write_table(pa.Table.from_pylist(qa_rows), quality_path)
    report = {
        "schema_version": "qa-value-corrections-v1",
        "status": "complete",
        "qa_value_ruleset_version": QA_VALUE_RULESET_VERSION,
        "source_m1_manifest_sha256": _sha256(m1_dir / "manifest.json"),
        "source_a3_manifest_sha256": _sha256(a3_dir / "manifest.json"),
        "unexcluded_affected_events": len(events),
        "affected_event_categories": dict(Counter(event["qa_category"] for event in events)),
        "candidate_a3_keys_examined_including_overlap": candidate_keys,
        "corrected_a3_rows": len(corrected_rows),
        "changed_fields": dict(changed_fields),
        "feature_corrections_file_sha256": _sha256(feature_path),
        "qa_quality_counts_file_sha256": _sha256(quality_path),
        "scope": "sparse corrections to pre-existing A3 features only; QA counts on corrected keys",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", required=True, type=Path)
    parser.add_argument("--a3-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.m1_dir, args.a3_dir, args.output_dir)
    print(
        json.dumps(
            {
                "unexcluded_affected_events": report["unexcluded_affected_events"],
                "corrected_a3_rows": report["corrected_a3_rows"],
                "changed_fields": len(report["changed_fields"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
