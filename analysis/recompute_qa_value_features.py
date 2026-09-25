"""Recompute selected published R3 hours with opt-in QA value guards.

This is an impact audit. It leaves the accepted R3 feature pack untouched and
checks that the legacy calculation still reproduces its saved numeric/state
features before reporting the QA-adjusted values.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import duckdb

from analysis.build_a2_hourly import _monthly_files
from stage1.features.hourly import FeatureEvent, feature_at
from stage1.features.qa_values import qa_window_counts
from stage1.features.r3 import A2_FEATURE_FIELDS
from stage1.features.schema import WINDOW_HOURS, _WINDOW_NUMERIC_FIELDS
from stage1.state_labeling.operational import source_is_full_archive
from stage1.value_quality import QA_VALUE_RULESET_VERSION


COMPARE_FIELDS = (
    "baseline_numeric_count",
    "baseline_numeric_median",
    "baseline_numeric_mad",
    "baseline_state_count",
    "baseline_dominant_state",
    *(
        f"{name}_{hours}h"
        for hours in WINDOW_HOURS
        for name in (
            "event_count",
            "alarm_count",
            "numeric_count",
            "state_count",
            "state_transitions",
            "state_distinct_count",
            *_WINDOW_NUMERIC_FIELDS,
        )
    ),
)
assert set(COMPARE_FIELDS).issubset(A2_FEATURE_FIELDS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right) <= 1e-9 * max(1.0, abs(left), abs(right))
    return left == right


def _case_parts(value: str) -> tuple[str, datetime]:
    channel_id, separator, text = value.partition("@")
    if not separator or not channel_id:
        raise ValueError("case must be CHANNEL_ID@YYYY-MM-DDTHH:MM:SS")
    at = datetime.fromisoformat(text)
    if at.tzinfo is not None or (at.minute, at.second, at.microsecond) != (0, 0, 0):
        raise ValueError("case timestamp must be a whole local hour")
    return channel_id, at


def recompute_case(m1_dir: Path, a3_dir: Path, value: str) -> dict:
    channel_id, start = _case_parts(value)
    end = start + timedelta(hours=24)
    if (start.year, start.month) != (
        (end - timedelta(hours=1)).year,
        (end - timedelta(hours=1)).month,
    ):
        raise ValueError("one case must remain inside one A3 month")
    a3_path = a3_dir / f"year={start.year}" / f"month={start.month:02d}" / "features.parquet"
    if not a3_path.is_file():
        raise FileNotFoundError(a3_path)
    con = duckdb.connect()
    try:
        result = con.execute(
            "SELECT * FROM read_parquet(?) WHERE channel_id=? "
            "AND prediction_time>=? AND prediction_time<? ORDER BY prediction_time",
            [str(a3_path), channel_id, start, end],
        )
        fields = [description[0] for description in result.description]
        published = [dict(zip(fields, row)) for row in result.fetchall()]
        if not published:
            raise ValueError(f"case has no published A3 hours: {value}")
        earliest_fit_end = start.replace(hour=0) - timedelta(hours=192)
        context_start = earliest_fit_end - timedelta(days=28)
        paths, missing = _monthly_files(m1_dir, context_start, end)
        if missing:
            raise ValueError(f"case context has missing M1 months: {missing}")
        result = con.execute(
            "SELECT channel_id,timestamp,alarm,value_numeric,value_state,sensor_type,"
            "object_id,join_status,quality_flags,source,value_raw "
            "FROM read_parquet(?,hive_partitioning=false) "
            "WHERE channel_id=? AND timestamp>=? AND timestamp<? "
            "ORDER BY timestamp,row_id",
            [[str(path) for path in paths], channel_id, context_start, end],
        )
        fields = [description[0] for description in result.description]
        records = [
            record
            for record in (dict(zip(fields, row)) for row in result.fetchall())
            if source_is_full_archive(record["source"], record["timestamp"])
        ]
    finally:
        con.close()
    legacy_events = [FeatureEvent.from_clean_record(record) for record in records]
    qa_events = [
        FeatureEvent.from_clean_record(record, apply_qa_value_policy=True) for record in records
    ]
    categories = dict(
        Counter(
            event.qa_value_category
            for event in qa_events
            if event.qa_value_category not in (None, "ordinary_numeric", "ordinary_text")
            and event.timestamp >= start - timedelta(hours=168)
        )
    )
    changed_rows = 0
    changed_fields: Counter[str] = Counter()
    examples = []
    legacy_mismatches = []
    for saved in published:
        t = saved["prediction_time"]
        fit_end = t.replace(hour=0) - timedelta(hours=192)
        legacy = feature_at(legacy_events, channel_id, t, baseline_fit_end_at=fit_end)
        qa = feature_at(qa_events, channel_id, t, baseline_fit_end_at=fit_end)
        mismatches = [name for name in COMPARE_FIELDS if not _same(legacy[name], saved[name])]
        if mismatches:
            legacy_mismatches.append({"prediction_time": t.isoformat(), "fields": mismatches})
        changed = {
            name: {"previous": legacy[name], "qa": qa[name]}
            for name in COMPARE_FIELDS
            if not _same(legacy[name], qa[name])
        }
        if changed:
            changed_rows += 1
            changed_fields.update(changed.keys())
        if len(examples) < 8 and (
            changed
            or any(
                value
                for name, value in qa_window_counts(qa_events, t).items()
                if name.endswith("_24h")
            )
        ):
            examples.append(
                {
                    "prediction_time": t.isoformat(),
                    "changed": changed,
                    "qa_counts": {
                        key: count
                        for key, count in qa_window_counts(qa_events, t).items()
                        if key.endswith("_24h") and count
                    },
                }
            )
    return {
        "case": value,
        "source_events": len(records),
        "qa_categories_in_context": categories,
        "published_hours": len(published),
        "legacy_mismatches": legacy_mismatches[:8],
        "legacy_mismatch_count": len(legacy_mismatches),
        "qa_changed_hours": changed_rows,
        "qa_changed_fields": dict(changed_fields),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", required=True, type=Path)
    parser.add_argument("--a3-dir", required=True, type=Path)
    parser.add_argument("--case", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {
        "schema_version": "qa-value-feature-impact-v1",
        "qa_value_ruleset_version": QA_VALUE_RULESET_VERSION,
        "source_m1_manifest_sha256": _sha256(args.m1_dir / "manifest.json"),
        "source_a3_manifest_sha256": _sha256(args.a3_dir / "manifest.json"),
        "cases": [recompute_case(args.m1_dir, args.a3_dir, value) for value in args.case],
    }
    if any(case["legacy_mismatch_count"] for case in report["cases"]):
        raise ValueError("legacy calculation did not reproduce the saved A3 feature rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(report["cases"]),
                "hours": sum(case["published_hours"] for case in report["cases"]),
                "changed_hours": sum(case["qa_changed_hours"] for case in report["cases"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
