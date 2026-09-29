"""Independently recount all frozen R6 test inputs from M1 and verified B2.

This is post-freeze reproduction, never threshold selection. Future labels are
used only for evaluation after past-only inputs and scores have been rebuilt.
Published M1/A3/B2 and B's frozen decision are immutable.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import platform
import threading
import time

import duckdb
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from analysis.build_a2_hourly import _monthly_files
from analysis.build_r3_full_month import _duckdb_connection
from analysis.r2_b2_handoff import load_b2_for_a2
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.alert_eval import evaluate_alerts
from ml.forecast.r6_rule import TERMS, predict_rule
from stage1.features.hourly import FeatureEvent, HourlyConfig
from stage1.features.r6_history import HISTORY_VERSION, iter_rule_history
from stage1.state_labeling.rules import DICTIONARY_CANDIDATES, KNOWN_SENSOR_TYPES, classify_message


KEYS = ["channel_id", "prediction_time"]
MONTHS = [f"2026-{month:02d}" for month in range(1, 7)]


class Resources:
    def __init__(self):
        self.phases = {}
        self.peak_rss = 0
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            info = self.process.memory_info()
            self.peak_rss = max(self.peak_rss, info.rss, getattr(info, "peak_wset", 0))
            self.stop.wait(0.05)

    @contextmanager
    def phase(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = round(time.perf_counter() - start, 3)
            print(f"{name}: {self.phases[name]} s", flush=True)


def semantic_mapping() -> pa.Table:
    pairs = {(kind, "Неисправен") for kind in KNOWN_SENSOR_TYPES} | set(DICTIONARY_CANDIDATES)
    rows = []
    for kind, state in sorted(pairs):
        meaning = classify_message(kind, state, False)
        alarm_meaning = classify_message(kind, state, True)
        if (meaning.category, meaning.target_message_candidate) != (
            alarm_meaning.category,
            alarm_meaning.target_message_candidate,
        ):
            raise ValueError("the frozen semantic interpretation unexpectedly depends on alarm")
        if meaning.category == "technical_fault":
            rows.append(
                {
                    "sensor_type": kind,
                    "value_state": state,
                    "is_registered_fault": meaning.target_message_candidate,
                }
            )
    return pa.Table.from_pylist(rows)


def recount_inputs(database, paths, completed):
    """Independent SQL range-count path; prediction_keys contains no labels."""
    names = [row[0] for row in database.execute("DESCRIBE prediction_keys").fetchall()]
    if names != KEYS:
        raise ValueError("raw inference accepts only channel/time keys, never target fields")
    start, end = database.execute(
        "SELECT min(prediction_time)-INTERVAL '168 hours', "
        "max(prediction_time)+INTERVAL '1 microsecond' FROM prediction_keys"
    ).fetchone()
    database.register("semantic_mapping", semantic_mapping())
    database.execute(
        "CREATE TEMP TABLE channels AS SELECT DISTINCT channel_id FROM prediction_keys"
    )
    database.execute(
        """CREATE TEMP TABLE raw_state_events AS
        SELECT e.row_id,e.channel_id,e.timestamp,e.sensor_type,e.value_state,
               e.alarm,e.quality_flags,m.is_registered_fault
        FROM read_parquet(?,hive_partitioning=false) e
        JOIN semantic_mapping m USING(sensor_type,value_state)
        SEMI JOIN channels USING(channel_id)
        WHERE e.timestamp>=? AND e.timestamp<?
          AND split_part(replace(e.source,chr(92),'/'),'/',-1)=
              'ext-journal-' || CAST(year(e.timestamp) AS VARCHAR) || '.7z'
          AND year(e.timestamp)<>2021""",
        [[str(path) for path in paths], start, end],
    )
    exclusions = " OR ".join(
        "list_contains(r.quality_flags,?)" for _ in HourlyConfig().excluded_quality_flags
    )
    database.execute(
        f"""CREATE TEMP TABLE raw_state_counts AS
        SELECT p.channel_id,p.prediction_time,
            CAST(count(*) FILTER(WHERE r.is_registered_fault AND
                r.timestamp>p.prediction_time-INTERVAL '24 hours') AS BIGINT)
                AS registered_fault_text_count_24h,
            CAST(count(*) FILTER(WHERE r.is_registered_fault) AS BIGINT)
                AS registered_fault_text_count_168h,
            CAST(count(r.timestamp) FILTER(WHERE
                r.timestamp>p.prediction_time-INTERVAL '24 hours') AS BIGINT)
                AS technical_message_count_24h
        FROM prediction_keys p LEFT JOIN raw_state_events r
          ON p.channel_id=r.channel_id AND r.timestamp<=p.prediction_time
         AND r.timestamp>p.prediction_time-INTERVAL '168 hours'
         AND NOT ({exclusions})
        GROUP BY p.channel_id,p.prediction_time""",
        sorted(HourlyConfig().excluded_quality_flags),
    )
    schema = pa.schema(
        [pa.field("channel_id", pa.string()), pa.field("end_at", pa.timestamp("us"))]
    )
    database.register(
        "completed_episodes",
        pa.Table.from_pylist(
            [{"channel_id": episode.channel_id, "end_at": episode.end_at} for episode in completed],
            schema=schema,
        ),
    )
    database.execute(
        """CREATE TEMP TABLE raw_episode_counts AS
        SELECT p.channel_id,p.prediction_time,CAST(count(e.end_at) AS BIGINT)
            AS completed_episode_count_168h
        FROM prediction_keys p LEFT JOIN completed_episodes e
          ON p.channel_id=e.channel_id AND e.end_at<=p.prediction_time
         AND e.end_at>p.prediction_time-INTERVAL '168 hours'
        GROUP BY p.channel_id,p.prediction_time"""
    )
    database.execute(
        "CREATE TEMP TABLE raw_rule_inputs AS SELECT s.*,e.completed_episode_count_168h "
        "FROM raw_state_counts s JOIN raw_episode_counts e USING(channel_id,prediction_time)"
    )
    return {
        "semantic_source_rows": database.execute(
            "SELECT count(*) FROM raw_state_events"
        ).fetchone()[0],
        "source_start": start.isoformat(),
        "source_end_exclusive": end.isoformat(),
    }


def warning_replay(frame, expected_alerts, threshold):
    """Second chronological warning/cooldown implementation, with no tuning."""
    previous, matched, rows = {}, set(), []
    ordered = frame.sort_values(KEYS)
    for row in ordered.itertuples(index=False):
        if row.rule_score < threshold:
            continue
        if row.channel_id in previous and row.prediction_time < previous[
            row.channel_id
        ] + timedelta(hours=24):
            continue
        previous[row.channel_id] = row.prediction_time
        outcome = "no_target_in_horizon"
        if row.target == 1:
            outcome = (
                "duplicate_episode_warning"
                if row.target_episode_id in matched
                else "matched_episode"
            )
            matched.add(row.target_episode_id)
        rows.append((row.channel_id, row.prediction_time, outcome))
    expected = list(
        expected_alerts.sort_values(KEYS)[[*KEYS, "outcome"]].itertuples(index=False, name=None)
    )
    if rows != expected:
        raise ValueError("chronological warning replay differs from the batch evaluator")
    return {"warnings_checked": len(rows), "mismatches": 0, "cooldown_carries_across_months": True}


def sample_history_replay(database, completed, threshold):
    sample = database.execute(
        """WITH ranked AS (
        SELECT p.channel_id,p.prediction_time,p.sensor_type,
               2.0*r.registered_fault_text_count_24h+0.5*r.registered_fault_text_count_168h
                 +r.completed_episode_count_168h+0.1*r.technical_message_count_24h AS score,
               row_number() OVER w_first AS first_hour,
               row_number() OVER w_last AS last_hour,
               row_number() OVER w_hash AS hashed_hour,
               row_number() OVER w_score AS high_score
        FROM source_points p JOIN raw_rule_inputs r USING(channel_id,prediction_time)
        WINDOW w_first AS (PARTITION BY date_trunc('month',p.prediction_time),p.sensor_type
                           ORDER BY p.prediction_time,p.channel_id),
               w_last AS (PARTITION BY date_trunc('month',p.prediction_time),p.sensor_type
                          ORDER BY p.prediction_time DESC,p.channel_id),
               w_hash AS (PARTITION BY date_trunc('month',p.prediction_time),p.sensor_type
                          ORDER BY hash(p.channel_id,p.prediction_time),p.channel_id,p.prediction_time),
               w_score AS (PARTITION BY date_trunc('month',p.prediction_time),p.sensor_type
                           ORDER BY score DESC,p.prediction_time,p.channel_id))
        SELECT r.* FROM ranked p JOIN raw_rule_inputs r USING(channel_id,prediction_time)
        WHERE first_hour=1 OR last_hour=1 OR high_score=1 OR hashed_hour<=3
        ORDER BY channel_id,prediction_time"""
    ).fetch_df()
    database.register(
        "sample_channels", pa.Table.from_pandas(sample[["channel_id"]].drop_duplicates())
    )
    records = (
        database.execute(
            "SELECT r.* FROM raw_state_events r SEMI JOIN sample_channels USING(channel_id) "
            "ORDER BY channel_id,timestamp,row_id"
        )
        .to_arrow_table()
        .to_pylist()
    )
    by_event, by_episode = defaultdict(list), defaultdict(list)
    for row in records:
        by_event[row["channel_id"]].append(FeatureEvent.from_clean_record(row))
    for episode in completed:
        by_episode[episode.channel_id].append(episode)
    checked = 0
    for channel, group in sample.groupby("channel_id", sort=True):
        times = group.prediction_time.dt.to_pydatetime().tolist()
        events, episodes = by_event[channel], by_episode[channel]
        batch = list(iter_rule_history(events, episodes, channel, times))
        for saved, row in zip(group.to_dict("records"), batch):
            if any(row[name] != saved[name] for name in TERMS):
                raise ValueError(f"SQL/Python history mismatch: {channel} {row['prediction_time']}")
            at = row["prediction_time"]
            prefix = list(
                iter_rule_history(
                    [event for event in events if event.timestamp <= at],
                    [episode for episode in episodes if episode.end_at <= at],
                    channel,
                    [at],
                )
            )[0]
            if prefix != row:
                raise ValueError("future-prefix truncation changed a prior prediction")
            future = FeatureEvent(
                channel,
                at + timedelta(microseconds=1),
                False,
                value_state="Неисправен",
                sensor_type="Датчик дыма",
            )
            if list(iter_rule_history([*events, future], episodes, channel, [at]))[0] != row:
                raise ValueError("adversarial future event changed a prior prediction")
            checked += 1
        batch_score = predict_rule(
            pd.DataFrame(batch), eligibility_status="eligible", threshold=threshold
        )
        for index, row in enumerate(batch):
            single = predict_rule(
                pd.DataFrame([row]), eligibility_status="eligible", threshold=threshold
            )
            if (
                single.rule_score.iloc[0] != batch_score.rule_score.iloc[index]
                or single.alert.iloc[0] != batch_score.alert.iloc[index]
            ):
                raise ValueError("batch/single-clock scoring mismatch")
    return {
        "sample_hours": checked,
        "sample_channels": int(sample.channel_id.nunique()),
        "mismatches": 0,
        "future_event_mutation_checks": checked,
        "future_prefix_truncation_checks": checked,
        "selection": "per month/type first,last,highest past score,three hashed keys; no target used",
    }


def run(*, m1_dir, b2_dir, a3_dir, index_dir, freeze_path, decision_path, output_dir):
    output_dir = output_dir.resolve()
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError("choose a new audit directory; preserve existing results")
    start = time.perf_counter()
    resources = Resources()
    resources.thread.start()
    try:
        freeze, decision = read_json(freeze_path), read_json(decision_path)
        a3, index, m1 = (read_json(path / "manifest.json") for path in (a3_dir, index_dir, m1_dir))
        if (
            freeze["feature_terms"] != TERMS
            or freeze["frozen_threshold"] != 7.1
            or decision["source_freeze_sha256"] != frozen_rule_sha256(freeze_path)
            or sha256(a3_dir / "manifest.json") != freeze["source_a3_manifest_sha256"]
            or sha256(index_dir / "manifest.json") != freeze["source_r3_admission_manifest_sha256"]
            or a3["source_m1_manifest_sha256"] != sha256(m1_dir / "manifest.json")
            or m1["status"] != "complete"
            or freeze["physical_failure_claim"]
        ):
            raise ValueError("frozen rule or source lineage differs")
        a_chunks, i_chunks = (
            {row["month"]: row for row in manifest["chunks"]} for manifest in (a3, index)
        )
        feature_paths, candidate_paths = [], []
        with resources.phase("verify_sources"):
            for month in MONTHS:
                a, i = a_chunks[month], i_chunks[month]
                local = read_json(index_dir / i["manifest_file"])
                feature = a3_dir / a["features_file"]
                candidate = (
                    index_dir / i["manifest_file"]
                ).parent / "conditional_discrete_keys.parquet"
                if (
                    sha256(feature) != a["features_sha256"]
                    or sha256(candidate) != local["candidate_sha256"]
                    or sha256(index_dir / i["manifest_file"]) != i["manifest_sha256"]
                    or sha256(a3_dir / a["manifest_file"]) != a["manifest_sha256"]
                ):
                    raise ValueError(f"test source file differs: {month}")
                feature_paths.append(str(feature))
                candidate_paths.append(str(candidate))
        pending.mkdir(parents=True)
        with _duckdb_connection() as database:
            database.execute("SET threads=2")
            database.execute("SET memory_limit='2GB'")
            projection = ",".join(f'f."{name}" AS "saved_{name}"' for name in TERMS)
            database.execute(
                f"""CREATE TEMP TABLE source_points AS SELECT c.*,f.sensor_type AS feature_type,{projection}
                FROM read_parquet(?,hive_partitioning=false) c
                LEFT JOIN read_parquet(?,hive_partitioning=false) f USING(channel_id,prediction_time)
                WHERE c.split='test'""",
                [candidate_paths, feature_paths],
            )
            total, unique, invalid = database.execute(
                """SELECT count(*),count(DISTINCT(channel_id,prediction_time)),
                count(*) FILTER(WHERE sensor_type IS DISTINCT FROM feature_type OR
                    target NOT IN (0,1) OR admission_status<>'conditional_archive_assumption'
                    OR availability_status<>'unknown' OR year(prediction_time)<>2026)
                FROM source_points"""
            ).fetchone()
            if total != unique or invalid or total != decision["test_conditionally_admitted_hours"]:
                raise ValueError("test keys, population or type lineage differ")
            database.execute(
                "CREATE TEMP TABLE prediction_keys AS SELECT channel_id,prediction_time FROM source_points"
            )
            channels = [
                row[0]
                for row in database.execute(
                    "SELECT DISTINCT channel_id FROM prediction_keys"
                ).fetchall()
            ]
            with resources.phase("verify_b2_catalog"):
                catalog = load_b2_for_a2(
                    b2_dir, local_m1_manifest=m1_dir / "manifest.json", channels=channels
                )
                if (
                    catalog.audit["catalog_manifest_sha256"]
                    != a3["source_b2_catalog_manifest_sha256"]
                ):
                    raise ValueError("A3 used a different B2 catalog")
            first, last = database.execute(
                "SELECT min(prediction_time),max(prediction_time) FROM prediction_keys"
            ).fetchone()
            paths, missing = _monthly_files(
                m1_dir, first - timedelta(hours=168), last + timedelta(microseconds=1)
            )
            if missing or not paths or any("year=2021" in path.parts for path in paths):
                raise ValueError("M1 past context is incomplete or includes excluded 2021")
            with resources.phase("raw_history_to_four_rule_inputs"):
                raw_audit = recount_inputs(database, paths, catalog.episodes)
            differences = database.execute(
                "SELECT count(*),"
                + ",".join(
                    f'count(*) FILTER(WHERE r."{name}" IS DISTINCT FROM p."saved_{name}")'
                    for name in TERMS
                )
                + " FROM raw_rule_inputs r FULL JOIN source_points p USING(channel_id,prediction_time)"
            ).fetchone()
            if differences[0] != total or any(differences[1:]):
                raise ValueError(f"raw M1/B2 inputs differ from saved A3: {differences}")
            with resources.phase("sample_python_batch_clock_and_causality"):
                sample = sample_history_replay(
                    database, catalog.episodes, freeze["frozen_threshold"]
                )
            frames, files = [], []
            with resources.phase("all_test_scoring_and_prediction_write"):
                for month in MONTHS:
                    frame = database.execute(
                        "SELECT p.channel_id,p.prediction_time,p.sensor_type,p.target,p.target_episode_id,"
                        "p.label_available_at,"
                        + ",".join(f'r."{name}"' for name in TERMS)
                        + " FROM source_points p JOIN raw_rule_inputs r USING(channel_id,prediction_time) "
                        "WHERE strftime(p.prediction_time,'%Y-%m')=? ORDER BY channel_id,prediction_time",
                        [month],
                    ).fetch_df()
                    score = sum(
                        weight * frame[name].to_numpy(dtype=float) for name, weight in TERMS.items()
                    )
                    scored = predict_rule(
                        frame, eligibility_status="eligible", threshold=freeze["frozen_threshold"]
                    )
                    if not np.array_equal(score, scored.rule_score.to_numpy()):
                        raise ValueError("independent arithmetic differs from frozen scorer")
                    history_path = pending / f"history_inputs_{month}.parquet"
                    frame[[*KEYS, *TERMS]].to_parquet(history_path, index=False, compression="zstd")
                    frame["rule_score"] = score
                    frame["above_frozen_threshold"] = score >= freeze["frozen_threshold"]
                    frame = frame.drop(columns=list(TERMS))
                    path = pending / f"predictions_{month}.parquet"
                    frame.to_parquet(path, index=False, compression="zstd")
                    files.append(
                        {
                            "month": month,
                            "file": path.name,
                            "rows": len(frame),
                            "sha256": sha256(path),
                            "history_inputs_sha256": sha256(history_path),
                            "history_inputs_file": history_path.name,
                        }
                    )
                    frames.append(frame)
            full = pd.concat(frames, ignore_index=True)
            del frames
            with resources.phase("frozen_warning_metrics_and_chronological_replay"):
                days = len(set(zip(full.channel_id, full.prediction_time.dt.date)))
                metrics, alerts = evaluate_alerts(
                    full, "rule_score", freeze["frozen_threshold"], channel_days=days
                )
                warning_audit = warning_replay(full, alerts, freeze["frozen_threshold"])
                comparisons = {
                    "test_conditionally_admitted_episodes": metrics["eligible_positive_episodes"],
                    "test_matched_episodes": metrics["matched_episodes"],
                    "test_unmatched_warnings": metrics["unmatched_warnings"],
                    "test_unmatched_warnings_per_1000_channel_days": metrics[
                        "unmatched_warnings_per_1000_channel_days"
                    ],
                    "test_median_lead_hours": metrics["median_lead_hours"],
                }
                if any(
                    not np.isclose(value, decision[name], rtol=0, atol=1e-12)
                    for name, value in comparisons.items()
                ):
                    raise ValueError(
                        "reproduced frozen test metrics differ from B's published decision"
                    )
                alerts.to_parquet(
                    pending / "emitted_alerts.parquet", index=False, compression="zstd"
                )
            with resources.phase("fingerprint_used_m1_parquet"):
                source_files = [
                    {
                        "path": path.relative_to(m1_dir).as_posix(),
                        "rows": pq.ParquetFile(path).metadata.num_rows,
                        "sha256": sha256(path),
                    }
                    for path in paths
                ]
        report = {
            "schema_version": "r6-a-independent-history-audit-v1",
            "status": "passed_numerical_reproduction_b_original_files_not_provided",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "post_freeze_independent_reproduction_not_model_selection",
            "history_version": HISTORY_VERSION,
            "physical_failure_claim": False,
            "deployment_approved": False,
            "source_freeze_sha256": frozen_rule_sha256(freeze_path),
            "source_b_decision_sha256": sha256(decision_path),
            "source_m1_manifest_sha256": sha256(m1_dir / "manifest.json"),
            "source_a3_manifest_sha256": sha256(a3_dir / "manifest.json"),
            "source_admission_manifest_sha256": sha256(index_dir / "manifest.json"),
            "b2_selection": catalog.audit,
            "source_m1_files": source_files,
            "raw_history": raw_audit,
            "test_hours_checked": total,
            "rule_input_mismatches": dict(zip(TERMS, differences[1:])),
            "scorer_arithmetic_mismatches": 0,
            "sample_history_replay": sample,
            "warning_replay": warning_audit,
            "alerts": metrics,
            "hourly_pr_auc": float(average_precision_score(full.target, full.rule_score)),
            "channel_days": days,
            "raw_a3_test_hours": sum(a_chunks[month]["rows"] for month in MONTHS),
            "resources": {
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "phase_seconds": resources.phases,
                "peak_working_set_bytes": resources.peak_rss,
                "sampling_interval_seconds": 0.05,
                "duckdb_threads": 2,
                "duckdb_memory_limit": "2GB",
                "python_version": platform.python_version(),
                "duckdb_version": duckdb.__version__,
                "psutil_version": psutil.__version__,
                "platform": platform.platform(),
            },
            "limitations": [
                "Only the four inputs of the chosen rule were fully recounted, not all 111 A3 fields.",
                "Scores are causal, but the retrospective R3 evaluation index uses future label eligibility; it is not live admission.",
                "B's original R6 output was unavailable: numerical metrics agree, byte/key identity of its predictions is not asserted.",
                "B2 completed-episode provenance and accepted subset are verified; episode extraction is not reimplemented here.",
                "Timing is one local run including input checks and diagnostics, not a production SLA.",
                "The opened test is used only for reproduction; no rule, threshold, target or population was tuned.",
            ],
        }
        report_path = pending / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (pending / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": report["schema_version"],
                    "status": report["status"],
                    "report_sha256": sha256(report_path),
                    "monthly_predictions": files,
                    "emitted_alerts_sha256": sha256(pending / "emitted_alerts.parquet"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        pending.rename(output_dir)
        return report
    finally:
        resources.stop.set()
        resources.thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("m1-dir", "b2-dir", "a3-dir", "index-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--freeze", default=Path("ml/r6_frozen_rule_v1.json"), type=Path)
    parser.add_argument("--decision", default=Path("ml/r6_b_final_decision_v1.json"), type=Path)
    args = parser.parse_args()
    report = run(
        m1_dir=args.m1_dir,
        b2_dir=args.b2_dir,
        a3_dir=args.a3_dir,
        index_dir=args.index_dir,
        freeze_path=args.freeze,
        decision_path=args.decision,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "test_hours_checked", "alerts", "resources")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
