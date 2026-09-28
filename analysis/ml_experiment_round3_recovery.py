"""Research-only transfer of ML's causal recovery-reset to a frozen Q2 score.

The decision stream receives only score-threshold booleans and past admission.
Retrospective B3 labels enter only after both warning streams are committed.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_q2_recovery_reset_a import assess_emitted, stream_groups
from analysis.ml_experiment_metric_audit import replay
from analysis.prepare_ml_experiment import sha256
from analysis.q2_recovery_reset_a import RecoveryResetPolicy, VERSION


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(source: Path, column: str, threshold: float, output: Path, year: int = 2025):
    if output.exists():
        raise FileExistsError(output)
    if year not in (2024, 2025):
        raise ValueError("only frozen tuning and open validation years are in scope")
    q2 = Path("output/q2-a-full-sparse-20260926-v5")
    m1 = Path("output/milestone1/full_20260922")
    package = read(q2 / "manifest.json")
    if sha256(m1 / "manifest.json") != package["source_manifests"]["m1"]:
        raise AssertionError("accepted M1 source changed")
    if sha256(q2 / "episode_diagnostics.parquet") != package["files"]["episode_diagnostics.parquet"]["sha256"]:
        raise AssertionError("accepted full episode catalog changed")
    if source.parent.name == "routing":
        routing = read(source.parent / "report.json")
        if (sha256(source) != routing["artifact_hashes"][source.name]
                or sha256(source.parent / "selection.json") != routing["selection_sha256"]
                or routing["selected_policy"] != "flexible"):
            raise AssertionError("frozen routing source or 2024 policy changed")
    admissions = []
    for month in (m for m in package["months"] if m["month"].startswith(f"{year}-")):
        path = q2 / f"year={year}/month={month['month'][5:]}/admission.parquet"
        if sha256(path) != month["files"]["admission.parquet"]["sha256"]:
            raise AssertionError("accepted admission changed")
        admissions.append(str(path))
    raw = []
    report = read(q2 / "report.json")
    for item in report["source_m1_files"]:
        path = m1 / item["file"]
        if sha256(path) != item["sha256"] or "year=2021" in str(path) or "year=2026" in str(path):
            raise AssertionError("accepted raw source changed")
        raw.append(str(path))
    if len(raw) != 72 or len(admissions) != 12:
        raise AssertionError("source coverage incomplete")
    full = pq.read_table(q2 / "episode_diagnostics.parquet").to_pandas()
    full = full.loc[full.target_episode_id.str.contains(f":{year}-", regex=False)]
    expected_full = 1204 if year == 2024 else 2142
    if len(full) != expected_full or full.target_episode_id.nunique() != expected_full:
        raise AssertionError("full denominator changed")
    output.mkdir(parents=True)
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory=?", [str(output / "duckdb-temp")])
        schema = pq.ParquetFile(source).schema_arrow
        if column not in schema.names or not pa.types.is_float32(schema.field(column).type):
            raise ValueError("frozen score column must be float32")
        source_years = db.execute("SELECT MIN(year(prediction_time)),MAX(year(prediction_time)) FROM read_parquet(?)", [str(source)]).fetchone()
        if source_years != (year, year):
            raise AssertionError("frozen score source belongs to another year")
        original = replay(db, source, column, threshold, expected_full)
        db.execute(f'''CREATE TEMP TABLE scores AS SELECT channel_id,prediction_time,sensor_type,
            target,target_episode_id,label_available_at,"{column}" AS catboost_score
            FROM read_parquet(?) WHERE "{column}">=?''', [str(source), threshold])
        labels = db.execute("SELECT * FROM scores ORDER BY prediction_time,channel_id").fetch_df()
        if len(labels) == 0:
            raise AssertionError("no threshold decisions")
        db.execute("CREATE TEMP TABLE selected_channels AS SELECT DISTINCT channel_id FROM scores UNION SELECT '228571'")
        past = db.execute('''SELECT s.channel_id,s.prediction_time,s.sensor_type,
            a.admission_status,a.admission_evidence_through,a.last_explicit_normal_at,
            a.blocking_qa_count_24h,a.availability_status FROM scores s
            LEFT JOIN read_parquet(?,hive_partitioning=false) a
            USING(channel_id,prediction_time) WHERE a.sensor_type=s.sensor_type
            ORDER BY s.prediction_time,s.channel_id''', [admissions]).fetch_df()
        if (len(past) != len(labels) or past.duplicated(["channel_id", "prediction_time"]).any()
                or not past.admission_status.eq("eligible").all()):
            raise AssertionError("score/admission key or eligibility mismatch")
        db.execute('''CREATE TEMP TABLE events AS SELECT row_id,channel_id,timestamp,
            sensor_type,value_state,alarm FROM read_parquet(?,hive_partitioning=false) e
            SEMI JOIN selected_channels USING(channel_id) WHERE value_state IS NOT NULL
            AND timestamp>=TIMESTAMP '2019-01-01' AND timestamp<CAST(? AS TIMESTAMP)
            AND year(timestamp)<>2021 AND split_part(replace(source,chr(92),'/'),'/',-1)=
            'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z' ''', [raw, f"{year+1}-01-01"])
        raw_count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        reader = db.execute("SELECT * FROM events ORDER BY timestamp,channel_id,row_id").to_arrow_reader(batch_size=100_000)
        groups = iter(stream_groups(reader))
        following = next(groups, None)
        old, new = RecoveryResetPolicy(allow_reset=False), RecoveryResetPolicy(allow_reset=True)
        old_rows, new_rows = [], []
        for at, frame in past.groupby("prediction_time", sort=True):
            when = at.to_pydatetime()
            while following is not None and following[0][0] <= when:
                _, events = following
                old.observe_group(events)
                new.observe_group(events)
                following = next(groups, None)
            decisions = frame.drop(columns="prediction_time").to_dict("records")
            for row in decisions:
                row["above_threshold"] = True
                row["blocking_qa_count_24h"] = int(row["blocking_qa_count_24h"])
                for field in ["admission_evidence_through", "last_explicit_normal_at"]:
                    row[field] = row[field].to_pydatetime() if pd.notna(row[field]) else None
            old_rows.extend(old.decide(when, decisions))
            new_rows.extend(new.decide(when, decisions))
        reader.close()
        days = db.execute("SELECT COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE))) FROM read_parquet(?)", [str(source)]).fetchone()[0]
        old_metric, old_alerts, old_ids = assess_emitted(old_rows, labels, full, days)
        new_metric, new_alerts, new_ids = assess_emitted(new_rows, labels, full, days)
        if (old_metric["matched_episodes"] != original["matched_episodes"]
                or old_metric["emitted_warnings"] != original["emitted_warnings"]
                or abs(old_metric["precision"] - original["episode_precision"]) > 1e-14
                or abs(old_metric["full_episode_recall"] - original["full_episode_recall"]) > 1e-14):
            raise AssertionError("causal baseline differs from frozen production replay")
        resets = new_alerts.loc[new_alerts.reason.eq("recovered_episode_reset")]
        if not (resets.previous_warning_at.lt(resets.observed_onset_at)
                & resets.observed_onset_at.lt(resets.observed_recovery_at)
                & resets.observed_recovery_at.le(resets.prediction_time)
                & resets.observed_onset_at.le(resets.previous_warning_at + timedelta(hours=24))).all():
            raise AssertionError("reset witness is not strictly past")
        if not all(row["past_state_agrees_with_admission"] for row in new_rows):
            raise AssertionError("past state/admission mismatch")
    old_alerts.to_parquet(output / "control_warnings.parquet", index=False)
    new_alerts.to_parquet(output / "candidate_warnings.parquet", index=False)
    result = {"status": "RESEARCH_CAUSAL_RESET_FROZEN_Q2_SCORE",
              "source_sha256": sha256(source), "source": str(source), "column": column,
              "threshold": threshold, "year": year, "causal_policy_version": VERSION,
              "model_and_threshold_frozen": True, "source_raw_text_events": raw_count,
              "above_threshold_decisions": len(labels), "full_episode_count": expected_full,
              "control": old_metric, "candidate": new_metric,
              "reset_warnings": len(resets), "new_episode_ids": len(new_ids - old_ids),
              "lost_episode_ids": len(old_ids - new_ids),
              "production_baseline_warning_parity": True,
              "research_only": True, "test_2026_read": False, "data_2021_read": False,
              "caveats": ["Q2 score stream excludes unknown-label hours as in the published offline evaluator.",
                          "Already-open 2025 outcomes are exploratory, not blind confirmation.",
                          "Short registered episodes may not equal actionable physical failures."]}
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["control", "candidate", "reset_warnings"]},
                     ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/ml-experiment-round2/routing/validation_scores.parquet"))
    parser.add_argument("--column", default="score_flexible")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--year", type=int, choices=[2024, 2025], default=2025)
    parser.add_argument("--output", type=Path, default=Path("output/ml-experiment-round3/recovery-q2-routing"))
    args = parser.parse_args()
    run(args.source, args.column, args.threshold, args.output, args.year)
