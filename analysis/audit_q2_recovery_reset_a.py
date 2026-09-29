"""Audit B's recovery-reset proposal on past M1 and unchanged saved Q2 scores."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta
from itertools import groupby
import json
from pathlib import Path
import time

import duckdb
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_q2_oracle_a import B_SHA, Q2_SHA
from analysis.build_quality_improvement_a import safe_path
from analysis.build_sparse_population_a import write_json
from analysis.q2_recovery_reset_a import RecoveryResetPolicy, VERSION
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256
from analysis.verify_q2_b_metrics_a import DECISION_SHA
from ml.forecast.alert_eval import evaluate_alerts
from stage1.state_labeling.registered_episodes import StateEvent


MODEL = "linear121"
THRESHOLD = 0.9993761875927455


def assess_emitted(decisions, labels: pd.DataFrame, full_episodes: pd.DataFrame, days: int):
    """Retrospective labels enter only here, after causal decisions are committed."""
    emitted = pd.DataFrame(decisions)
    emitted = emitted.loc[emitted.warning_emitted].merge(
        labels, on=["channel_id", "prediction_time", "sensor_type"], validate="one_to_one"
    )
    if len(emitted) != sum(row["warning_emitted"] for row in decisions):
        raise ValueError("emitted warning is missing an unchanged evaluation label")
    matched, outcomes = set(), []
    for row in emitted.itertuples(index=False):
        if row.target == 1:
            if pd.isna(row.target_episode_id) or not timedelta(
                0
            ) < row.label_available_at - row.prediction_time <= timedelta(hours=24):
                raise ValueError("invalid fixed positive label")
            outcome = (
                "duplicate_episode_warning"
                if row.target_episode_id in matched
                else "matched_episode"
            )
            matched.add(row.target_episode_id)
        elif row.target == 0:
            outcome = "no_target_in_horizon"
        else:
            raise ValueError("evaluation population must stay binary")
        outcomes.append(outcome)
    emitted["outcome"] = outcomes
    precision, recall = len(matched) / len(emitted), len(matched) / len(full_episodes)
    metrics = {
        "emitted_warnings": len(emitted),
        "matched_episodes": len(matched),
        "unmatched_warnings": len(emitted) - len(matched),
        "precision": precision,
        "full_episode_recall": recall,
        "warnings_per_1000_channel_days": len(emitted) * 1000 / days,
        "unmatched_per_1000_channel_days": (len(emitted) - len(matched)) * 1000 / days,
        "outcomes": dict(Counter(outcomes)),
    }
    return metrics, emitted, matched


def stream_groups(reader):
    def rows():
        for batch in reader:
            yield from batch.to_pylist()

    for key, group in groupby(rows(), key=lambda row: (row["timestamp"], row["channel_id"])):
        yield (
            key,
            [
                StateEvent(
                    row["row_id"],
                    row["channel_id"],
                    row["sensor_type"],
                    row["timestamp"],
                    row["value_state"],
                    row["alarm"],
                )
                for row in group
            ],
        )


def audit(*, q2_dir: Path, experiment: Path, decision: Path, m1_dir: Path, output_dir: Path):
    begun = time.perf_counter()
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    for path, expected in (
        (q2_dir / "manifest.json", Q2_SHA),
        (experiment / "manifest.json", B_SHA),
        (decision, DECISION_SHA),
    ):
        if sha256(path) != expected:
            raise ValueError("fixed Q2/B input differs")
    q2, b = read_json(q2_dir / "manifest.json"), read_json(experiment / "manifest.json")
    if sha256(m1_dir / "manifest.json") != q2["source_manifests"]["m1"]:
        raise ValueError("accepted M1 differs")
    for name in ("report.json", "episode_diagnostics.parquet"):
        if sha256(q2_dir / name) != q2["files"][name]["sha256"]:
            raise ValueError("Q2 source report/episodes changed")
    if sha256(experiment / "report.json") != b["report_sha256"]:
        raise ValueError("B source report changed")
    trained = read_json(experiment / "report.json")
    choice = read_json(decision)["models"][MODEL]["goal_check"]["diagnostic_best_full_f1"]
    if choice["threshold"] != THRESHOLD:
        raise ValueError("published diagnostic threshold changed")
    scores = []
    for item in b["score_files"]:
        path = safe_path(experiment, item["name"])
        if sha256(path) != item["sha256"]:
            raise ValueError("saved validation score changed")
        scores.append(str(path))
    if [item["name"] for item in b["score_files"]] != [
        f"validation_2025-{i:02d}.parquet" for i in range(1, 13)
    ]:
        raise ValueError("score scope differs from all twelve validation months")
    admissions = []
    for month in (row for row in q2["months"] if row["split"] == "validation"):
        path = q2_dir / f"year=2025/month={month['month'][5:]}/admission.parquet"
        if sha256(path) != month["files"]["admission.parquet"]["sha256"]:
            raise ValueError("causal Q2 admission changed")
        admissions.append(str(path))
    source_files = read_json(q2_dir / "report.json")["source_m1_files"]
    raw_files = []
    for row in source_files:
        path = safe_path(m1_dir, row["file"])
        if (
            any(part in {"year=2021", "year=2026"} for part in path.parts)
            or sha256(path) != row["sha256"]
        ):
            raise ValueError("excluded/test source or changed M1 file")
        raw_files.append(str(path))
    if len(raw_files) != 72:
        raise ValueError("requires all accepted 72 train/validation M1 files")
    print("verified 72 M1 source files, all twelve score/admission files", flush=True)
    full = pq.ParquetFile(q2_dir / "episode_diagnostics.parquet").read().to_pandas()
    full = full.loc[full.split.eq("validation")]
    if len(full) != 2142 or full.target_episode_id.nunique() != 2142:
        raise ValueError("full Recall denominator changed")
    pending.mkdir(parents=True)
    baseline, candidate, resumed = [
        RecoveryResetPolicy(allow_reset=mode) for mode in (False, True, True)
    ]
    old_rows, new_rows, checkpoint_checks = [], [], []
    completed = Counter()
    with duckdb.connect(config={"temp_directory": str(pending / "db-spill")}) as db:
        db.execute("SET threads=1")
        db.execute("SET memory_limit='1500MB'")
        db.execute(
            "CREATE TEMP TABLE scores AS SELECT channel_id,prediction_time,sensor_type,"
            "target,target_episode_id,label_available_at,score_linear121 AS catboost_score "
            "FROM read_parquet(?,hive_partitioning=false) WHERE score_linear121>=?",
            [scores, THRESHOLD],
        )
        labels = db.execute("SELECT * FROM scores ORDER BY prediction_time,channel_id").fetch_df()
        db.execute(
            "CREATE TEMP TABLE selected_channels AS SELECT DISTINCT channel_id FROM scores "
            "UNION SELECT '228571'"
        )  # Protected temperature sentinel, not newly admitted.
        past = db.execute(
            "SELECT s.channel_id,s.prediction_time,s.sensor_type,a.admission_status,"
            "a.admission_evidence_through,a.last_explicit_normal_at,a.blocking_qa_count_24h,"
            "a.availability_status FROM scores s LEFT JOIN read_parquet(?,hive_partitioning=false) a "
            "USING(channel_id,prediction_time) WHERE a.sensor_type=s.sensor_type "
            "ORDER BY s.prediction_time,s.channel_id",
            [admissions],
        ).fetch_df()
        if (
            len(past) != len(labels)
            or len(labels) != 6973
            or past.duplicated(["channel_id", "prediction_time"]).any()
            or not past.admission_status.eq("eligible").all()
        ):
            raise ValueError("same score keys do not exactly match causal eligible admission")
        db.execute(
            "CREATE TEMP TABLE events AS SELECT row_id,channel_id,timestamp,sensor_type,"
            "value_state,alarm FROM read_parquet(?,hive_partitioning=false) e "
            "SEMI JOIN selected_channels USING(channel_id) WHERE value_state IS NOT NULL "
            "AND timestamp>=TIMESTAMP '2019-01-01' AND timestamp<TIMESTAMP '2026-01-01' "
            "AND year(timestamp)<>2021 AND split_part(replace(source,chr(92),'/'),'/',-1)="
            "'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z'",
            [raw_files],
        )
        raw_count, raw_channels = db.execute(
            "SELECT COUNT(*),COUNT(DISTINCT channel_id) FROM events"
        ).fetchone()
        print(f"replaying {raw_count} text observations on {raw_channels} channels", flush=True)
        reader = db.execute(
            "SELECT * FROM events ORDER BY timestamp,channel_id,row_id"
        ).to_arrow_reader(batch_size=100_000)
        groups = iter(stream_groups(reader))
        following = next(groups, None)
        hours = list(past.groupby("prediction_time", sort=True))
        for index, (at, frame) in enumerate(hours):
            when = at.to_pydatetime()
            while following is not None and following[0][0] <= when:
                (_, channel), events = following
                registered = candidate.builder.states.get(channel)
                live = registered.open_episode if registered else None
                for policy in (baseline, candidate, resumed):
                    policy.observe_group(events)
                if (
                    live is not None
                    and live.end_at == events[0].at
                    and live.onset_status == "candidate_new_onset"
                    and live.end_status == "exact_norma"
                    and not live.uncertain_intervening_state
                ):
                    completed[(str(events[0].at.year), live.sensor_type)] += 1
                following = next(groups, None)
            records = frame.drop(columns="prediction_time").to_dict("records")
            for row in records:
                row["above_threshold"] = True
                row["blocking_qa_count_24h"] = int(row["blocking_qa_count_24h"])
                for name in ("admission_evidence_through", "last_explicit_normal_at"):
                    row[name] = row[name].to_pydatetime() if pd.notna(row[name]) else None
            old_rows.extend(baseline.decide(when, records))
            fresh = candidate.decide(when, records)
            repeated = resumed.decide(when, records)
            if fresh != repeated:
                raise ValueError("checkpoint-resumed decision differs from continuous replay")
            new_rows.extend(fresh)
            if index == len(hours) - 1 or hours[index + 1][0].month != at.month:
                uninterrupted = candidate.checkpoint()
                if resumed.checkpoint() != uninterrupted:
                    raise ValueError("continuous/resumed state differs at month boundary")
                path = pending / f"checkpoint_2025-{at.month:02d}.json"
                write_json(path, uninterrupted)
                resumed = RecoveryResetPolicy.restore(read_json(path))
                if resumed.checkpoint() != uninterrupted:
                    raise ValueError("checkpoint JSON roundtrip changed causal state")
                checkpoint_checks.append(
                    {
                        "month": f"2025-{at.month:02d}",
                        "decisions_equal": True,
                        "state_equal": True,
                        "checkpoint_sha256": sha256(path),
                    }
                )
                print(
                    f"checked 2025-{at.month:02d}: {len(new_rows)} decisions, checkpoint equal",
                    flush=True,
                )
        reader.close()
        days = trained["validation"]["eligible_channel_days_with_binary_label"]
        old_metrics, old_alerts, old_matched = assess_emitted(old_rows, labels, full, days)
        new_metrics, new_alerts, new_matched = assess_emitted(new_rows, labels, full, days)
        canonical, canonical_alerts = evaluate_alerts(
            labels, "catboost_score", THRESHOLD, channel_days=days
        )
        if (
            old_metrics["matched_episodes"] != 346
            or old_metrics["emitted_warnings"] != 1305
            or set(zip(old_alerts.channel_id, old_alerts.prediction_time))
            != set(zip(canonical_alerts.channel_id, canonical_alerts.prediction_time))
            or canonical["matched_episodes"] != 346
        ):
            raise ValueError("control changed published Q2 warning rule")
        if not all(row["past_state_agrees_with_admission"] for row in new_rows):
            raise ValueError("raw R1 past state disagrees with Q2 admission")
        resets = new_alerts.loc[new_alerts.reason.eq("recovered_episode_reset")]
        if not (
            resets.previous_warning_at.lt(resets.observed_onset_at)
            & resets.observed_onset_at.lt(resets.observed_recovery_at)
            & resets.observed_recovery_at.le(resets.prediction_time)
            & resets.observed_onset_at.le(resets.previous_warning_at + timedelta(hours=24))
        ).all():
            raise ValueError("reset witness is not known in the warning horizon/past")
        traces = []
        # Up to three actual witnesses per type, chosen chronologically, not by outcome.
        sample = (
            resets.sort_values(["prediction_time", "channel_id"]).groupby("sensor_type").head(3)
        )
        for i, row in enumerate(sample.itertuples(index=False)):
            messages = db.execute(
                "SELECT * FROM events WHERE channel_id=? AND timestamp>? "
                "AND timestamp<=? ORDER BY timestamp,row_id",
                [
                    row.channel_id,
                    row.previous_warning_at - timedelta(hours=168),
                    row.prediction_time,
                ],
            ).fetch_df()
            messages["trace_id"] = i
            messages["warning_time"] = row.prediction_time
            traces.append(messages)
        day_counts = db.execute(
            "SELECT sensor_type,COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE))) "
            "channel_days FROM read_parquet(?) GROUP BY sensor_type",
            [scores],
        ).fetch_df()
        by_type = []
        for kind in sorted(set(full.sensor_type) | set(day_counts.sensor_type)):
            group = full.loc[full.sensor_type.eq(kind)]
            day_row = day_counts.loc[day_counts.sensor_type.eq(kind), "channel_days"]
            type_days = int(day_row.iloc[0]) if len(day_row) else 0
            row = {
                "sensor_type": kind,
                "full_episodes": len(group),
                "binary_eligible_channel_days": type_days,
            }
            for name, alerts in (("control", old_alerts), ("candidate", new_alerts)):
                subset = alerts.loc[alerts.sensor_type.eq(kind)]
                matches = int(subset.outcome.eq("matched_episode").sum())
                row[name] = {
                    "warnings": len(subset),
                    "matched": matches,
                    "precision": matches / len(subset) if len(subset) else 0.0,
                    "full_type_recall": matches / len(group) if len(group) else None,
                    "unmatched_per_1000_channel_days": (len(subset) - matches) * 1000 / type_days
                    if type_days
                    else None,
                }
            by_type.append(row)
        for name, metrics in (("control", old_metrics), ("candidate", new_metrics)):
            if (
                sum(row[name]["warnings"] for row in by_type) != metrics["emitted_warnings"]
                or sum(row[name]["matched"] for row in by_type) != metrics["matched_episodes"]
            ):
                raise ValueError(
                    "by-type table does not cover every warning, including zero-episode types"
                )
        db.register("warning_channels", new_alerts[["channel_id"]].drop_duplicates())
        channel_days = (
            db.execute(
                "SELECT channel_id,COUNT(DISTINCT CAST(prediction_time AS DATE)) channel_days "
                "FROM read_parquet(?) SEMI JOIN warning_channels USING(channel_id) GROUP BY channel_id",
                [scores],
            )
            .fetch_df()
            .set_index("channel_id")
            .channel_days.to_dict()
        )
        channel_load = []
        for channel, group in new_alerts.groupby("channel_id"):
            previous = old_alerts.loc[old_alerts.channel_id.eq(channel)]
            ordered = group.sort_values("prediction_time")
            gaps = ordered.prediction_time.diff().dt.total_seconds() / 3600
            matches = int(group.outcome.eq("matched_episode").sum())
            per_day = group.groupby(group.prediction_time.dt.date).size()
            channel_load.append(
                {
                    "channel_id": channel,
                    "sensor_type": group.sensor_type.iloc[0],
                    "binary_eligible_channel_days": int(channel_days[channel]),
                    "control_warnings": len(previous),
                    "control_matched": int(previous.outcome.eq("matched_episode").sum()),
                    "candidate_warnings": len(group),
                    "candidate_matched": matches,
                    "candidate_unmatched": len(group) - matches,
                    "candidate_unmatched_per_1000_channel_days": (len(group) - matches)
                    * 1000
                    / channel_days[channel],
                    "recovery_reset_warnings": int(
                        group.reason.eq("recovered_episode_reset").sum()
                    ),
                    "intervals_under_24h": int(gaps.lt(24).sum()),
                    "minimum_warning_interval_hours": float(gaps.min())
                    if gaps.notna().any()
                    else None,
                    "max_warnings_per_calendar_day": int(per_day.max()),
                }
            )
        sentinel = db.execute(
            "SELECT COUNT(*),COUNT(DISTINCT timestamp) FROM events WHERE channel_id='228571'"
        ).fetchone()
    month_rows = []
    for month in range(1, 13):
        row = {"month": f"2025-{month:02d}"}
        for name, alerts in (("control", old_alerts), ("candidate", new_alerts)):
            subset = alerts.loc[alerts.prediction_time.dt.month.eq(month)]
            row[name] = {
                "warnings": len(subset),
                "matched": int(subset.outcome.eq("matched_episode").sum()),
            }
        month_rows.append(row)
    intervals = (
        new_alerts.sort_values(["channel_id", "prediction_time"])
        .groupby("channel_id")
        .prediction_time.diff()
        .dt.total_seconds()
        / 3600
    )
    channel_counts = new_alerts.groupby("channel_id").size()
    for name, frame in (
        ("control_warnings.parquet", old_alerts),
        ("candidate_warnings.parquet", new_alerts),
        ("past_decisions.parquet", pd.DataFrame(new_rows)),
        ("load_by_channel.parquet", pd.DataFrame(channel_load)),
    ):
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False), pending / name, compression="zstd"
        )
    if traces:
        pq.write_table(
            pa.Table.from_pandas(pd.concat(traces, ignore_index=True), preserve_index=False),
            pending / "trace_samples.parquet",
            compression="zstd",
        )
    report = {
        "schema_version": VERSION,
        "status": "research_diagnostic_not_rule_approval",
        "source_q2_manifest_sha256": Q2_SHA,
        "source_b_manifest_sha256": B_SHA,
        "source_decision_sha256": DECISION_SHA,
        "source_m1_manifest_sha256": q2["source_manifests"]["m1"],
        "model": MODEL,
        "threshold": THRESHOLD,
        "full_positive_episodes": 2142,
        "binary_eligible_rows": trained["validation"]["rows"],
        "binary_channel_days": days,
        "above_threshold_decisions": len(labels),
        "scored_channels": labels.channel_id.nunique(),
        "raw_text_rows": raw_count,
        "text_rows_consumed_through_last_decision": candidate.builder.message_counts["text_rows"],
        "audited_channels_including_temperature_sentinel": raw_channels,
        "source_m1_files": source_files,
        "control": old_metrics,
        "candidate": new_metrics,
        "gained_episodes": sorted(new_matched - old_matched),
        "lost_episodes": sorted(old_matched - new_matched),
        "reset_emitted_warnings": len(resets),
        "reset_warning_outcomes": dict(Counter(resets.outcome)),
        "warnings_separated_by_less_than_24h": int(intervals.lt(24).sum()),
        "max_warnings_on_one_channel": int(channel_counts.max()),
        "max_warnings_on_one_channel_per_calendar_day": max(
            row["max_warnings_per_calendar_day"] for row in channel_load
        ),
        "reset_episode_duration_seconds": {
            "median": float(
                (resets.observed_recovery_at - resets.observed_onset_at).dt.total_seconds().median()
            ),
            "minimum": float(
                (resets.observed_recovery_at - resets.observed_onset_at).dt.total_seconds().min()
            ),
        },
        "reset_to_warning_seconds": {
            "median": float(
                (resets.prediction_time - resets.observed_recovery_at).dt.total_seconds().median()
            ),
            "minimum": float(
                (resets.prediction_time - resets.observed_recovery_at).dt.total_seconds().min()
            ),
            "within_one_hour": int(
                (resets.prediction_time - resets.observed_recovery_at).le(timedelta(hours=1)).sum()
            ),
        },
        "trace_samples": [
            {
                "trace_id": i,
                "channel_id": row.channel_id,
                "sensor_type": row.sensor_type,
                "previous_warning_at": row.previous_warning_at.isoformat(),
                "onset_at": row.observed_onset_at.isoformat(),
                "normal_at": row.observed_recovery_at.isoformat(),
                "new_warning_at": row.prediction_time.isoformat(),
            }
            for i, row in enumerate(sample.itertuples(index=False))
        ],
        "by_type": by_type,
        "by_month": month_rows,
        "confident_completed_transitions_in_processed_past": [
            {"year": year, "sensor_type": kind, "count": n}
            for (year, kind), n in sorted(completed.items())
        ],
        "temperature_sentinel": {
            "channel_id": "228571",
            "text_rows_in_source": sentinel[0],
            "seconds_in_source": sentinel[1],
            "candidate_decisions": int(labels.channel_id.eq("228571").sum()),
            "protective_admission_unchanged": True,
        },
        "checkpoint_checks": checkpoint_checks,
        "causal_reset_witness_checks": True,
        "past_state_matches_all_score_admissions": True,
        "no_model_fit_or_threshold_choice": True,
        "labels_admission_or_frozen_rules_changed": False,
        "test_data_read": False,
        "customer_semantics_or_live_pilot_approved": False,
        "limitations": [
            "Already seen validation, fixed diagnostic threshold: not independent quality confirmation or rule selection.",
            "The past-only policy receives separate verified Q2 admission; it does not recompute the full gate here.",
            "Unknown-label hours are absent from this retrospective score population, not changed into negatives.",
            "All 72 M1 files are verified, but raw state replay is restricted to score-triggering channels and a protected sentinel.",
            "Registered transitions do not prove physically separate failures or operator usefulness.",
            "Event-time replay assumes complete closed groups; real delivery time/late-event source contract is not available.",
        ],
        "code_lf_sha256": {
            name: frozen_rule_sha256(Path(name))
            for name in (
                "analysis/q2_recovery_reset_a.py",
                "analysis/audit_q2_recovery_reset_a.py",
                "stage1/state_labeling/registered_episodes.py",
                "stage1/state_labeling/operational.py",
                "stage1/state_labeling/rules.py",
                "stage1/shadow/checkpoint.py",
            )
        },
        "resources": {
            "elapsed_seconds": round(time.perf_counter() - begun, 3),
            "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", 0),
            "duckdb_threads": 1,
            "duckdb_memory_limit": "1500MB",
        },
    }
    write_json(pending / "report.json", report)
    write_json(
        pending / "manifest.json",
        {
            "schema_version": VERSION,
            "months": [],
            "files": {
                path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                for path in sorted(pending.iterdir())
                if path.is_file()
            },
        },
    )
    pending.rename(output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("q2-dir", "experiment", "decision", "m1-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    report = audit(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("control", "candidate", "reset_emitted_warnings", "resources")
            }
        )
    )


if __name__ == "__main__":
    main()
