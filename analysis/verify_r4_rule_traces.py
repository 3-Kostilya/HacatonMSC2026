"""Independently re-count sampled R4 rule history from clean M1 events."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.train_r4_discrete_baselines import read_json, sha256  # noqa: E402
from stage1.features.hourly import HourlyConfig  # noqa: E402
from stage1.state_labeling.rules import classify_message, in_scope_year  # noqa: E402


def _source_files(root: Path, points: pd.DataFrame) -> list[str]:
    months = set()
    for at in points.prediction_time:
        for point in (at - timedelta(hours=24), at + timedelta(hours=24)):
            if not in_scope_year(point.year):
                raise ValueError("trace window crosses an excluded source year")
            months.add((point.year, point.month))
    files = []
    for year, month in sorted(months):
        matches = sorted((root / "clean" / f"year={year}" /
                          f"month={month}").glob("*.parquet"))
        if not matches:
            raise FileNotFoundError(f"M1 trace source is missing: {year}-{month:02d}")
        files.extend(str(path) for path in matches)
    return files


def run(*, audit_dir: Path, m1_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"trace verification output exists: {output_dir}")
    audit_manifest = read_json(audit_dir / "manifest.json")
    m1_manifest = read_json(m1_dir / "manifest.json")
    trace_path = audit_dir / "trace_review.json"
    if (audit_manifest["status"] != "validation_only_b_review"
            or sha256(trace_path) != audit_manifest["trace_review_sha256"]
            or m1_manifest["status"] != "complete"):
        raise ValueError("trace or M1 manifest differs")
    traces = read_json(trace_path)
    if len(traces) != 50:
        raise ValueError("expected 50 deterministic rule traces")
    points = pd.DataFrame([{  # source context only, never fitted to a model
        "sample_key": item["sample_key"],
        "channel_id": item["channel_id"],
        "prediction_time": pd.Timestamp(item["prediction_time"]),
        "target": item["target"],
        "target_episode_id": item["target_episode_id"],
        "label_available_at": pd.Timestamp(item["label_available_at"]),
        "reported_fault_count_24h": item["registered_fault_text_count_24h"],
        "reported_normal_count_24h": item["normal_message_count_24h"],
        "outcome": item["outcome"],
    } for item in traces])
    files = _source_files(m1_dir, points)
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=4")
        database.register("points", points[["sample_key", "channel_id", "prediction_time"]])
        raw = database.execute(
            """SELECT p.sample_key, p.prediction_time, m.timestamp,
                      m.value_state, m.alarm, m.sensor_type, m.quality_flags
               FROM read_parquet(?) AS m JOIN points AS p
                 ON m.channel_id = p.channel_id
                AND m.timestamp > p.prediction_time - INTERVAL '24 hours'
                AND m.timestamp <= p.prediction_time + INTERVAL '24 hours'
               WHERE m.value_state IS NOT NULL""",
            [files],
        ).fetch_df()
    excluded = HourlyConfig().excluded_quality_flags
    by_key = {key: [] for key in points.sample_key}
    for item in raw.itertuples(index=False):
        flags = list(item.quality_flags) if item.quality_flags is not None else []
        if excluded.intersection(flags):
            continue
        alarm = None if pd.isna(item.alarm) else bool(item.alarm)
        meaning = classify_message(item.sensor_type, item.value_state, alarm)
        by_key[item.sample_key].append({
            "timestamp": item.timestamp,
            "value_state": item.value_state,
            "registered_fault": bool(meaning.target_message_candidate),
            "normal": meaning.category == "normal",
        })
    for events in by_key.values():
        events.sort(key=lambda item: item["timestamp"])
    results = []
    fault_mismatch = 0
    normal_mismatch = 0
    missing_target_message = 0
    for point in points.itertuples(index=False):
        events = by_key[point.sample_key]
        prior = [item for item in events if item["timestamp"] <= point.prediction_time]
        future = [item for item in events if item["timestamp"] > point.prediction_time]
        faults = [item["timestamp"] for item in prior if item["registered_fault"]]
        normals = [item["timestamp"] for item in prior if item["normal"]]
        target_at = point.label_available_at
        target_exists = any(
            item["registered_fault"] and item["timestamp"] == target_at
            for item in future
        ) if point.target == 1 else None
        fault_mismatch += len(faults) != point.reported_fault_count_24h
        normal_mismatch += len(normals) != point.reported_normal_count_24h
        missing_target_message += point.target == 1 and not target_exists
        results.append({
            "sample_key": point.sample_key,
            "channel_id": point.channel_id,
            "prediction_time": point.prediction_time.isoformat(),
            "outcome": point.outcome,
            "target_episode_id": point.target_episode_id,
            "target_message_present_in_future_m1": target_exists,
            "reported_fault_count_24h": int(point.reported_fault_count_24h),
            "recounted_fault_count_24h": len(faults),
            "reported_normal_count_24h": int(point.reported_normal_count_24h),
            "recounted_normal_count_24h": len(normals),
            "latest_prior_fault_at": faults[-1].isoformat() if faults else None,
            "latest_prior_normal_at": normals[-1].isoformat() if normals else None,
            "future_exact_fault_count_24h": sum(
                item["registered_fault"] for item in future
            ),
        })
    report = {
        "schema_version": "r4-b-rule-raw-trace-verification-v1",
        "status": "complete" if not (fault_mismatch or normal_mismatch or missing_target_message)
                  else "mismatch_requires_review",
        "source_validation_audit_manifest_sha256": sha256(audit_dir / "manifest.json"),
        "source_m1_manifest_sha256": sha256(m1_dir / "manifest.json"),
        "traces_checked": len(points),
        "matched_episode_traces": int(points.outcome.eq("matched_episode").sum()),
        "unmatched_warning_traces": int(points.outcome.ne("matched_episode").sum()),
        "fault_24h_count_mismatches": int(fault_mismatch),
        "normal_24h_count_mismatches": int(normal_mismatch),
        "future_target_message_missing": int(missing_target_message),
        "m1_month_files_scanned": len(files),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "traces.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "schema_version": report["schema_version"],
            "status": report["status"],
            "source_validation_audit_manifest_sha256": (
                report["source_validation_audit_manifest_sha256"]
            ),
            "source_m1_manifest_sha256": report["source_m1_manifest_sha256"],
            "traces_sha256": sha256(output_dir / "traces.json"),
            "report_sha256": sha256(output_dir / "report.json"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "complete":
        raise ValueError("raw M1 trace verification found mismatches; inspect report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--m1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(audit_dir=args.audit_dir, m1_dir=args.m1_dir,
                 output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
