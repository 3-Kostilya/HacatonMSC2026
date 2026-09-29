"""Trace the December 2025 gas-warning cluster back to past M1 events."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.train_r4_discrete_baselines import read_json, sha256


def _numeric(frame: pd.DataFrame, name: str) -> dict:
    values = pd.to_numeric(frame[name], errors="coerce")
    return {"median": float(values.median()) if values.notna().any() else None,
            "minimum": float(values.min()) if values.notna().any() else None,
            "maximum": float(values.max()) if values.notna().any() else None,
            "missing": int(values.isna().sum())}


def run(*, error_dir: Path, experiment: Path, decision: Path,
        q2_dir: Path, m1_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    error = read_json(error_dir / "report.json")
    error_manifest = read_json(error_dir / "manifest.json")
    q2 = read_json(q2_dir / "report.json")
    trained = read_json(experiment / "report.json")
    selected = read_json(decision)
    if (error_manifest["report_sha256"] != sha256(error_dir / "report.json")
            or error_manifest["warning_cases_sha256"]
            != sha256(error_dir / "warning_cases.parquet")
            or error["source_q2_manifest_sha256"] != sha256(q2_dir / "manifest.json")
            or error["source_experiment_manifest_sha256"]
            != sha256(experiment / "manifest.json")
            or selected["source_experiment_manifest_sha256"]
            != sha256(experiment / "manifest.json")
            or q2["source_manifests"]["m1"] != sha256(m1_dir / "manifest.json")):
        raise ValueError("Q2/B error audit or M1 lineage differs")
    cases = pq.read_table(error_dir / "warning_cases.parquet").to_pandas()
    gas = cases.loc[(cases.sensor_type == "Газовый датчик")
                    & (cases.outcome == "no_target_in_horizon")].copy()
    dates = gas.prediction_time.dt.strftime("%Y-%m-%d")
    top = dates.value_counts().head(2)
    if list(top.index) != ["2025-12-09", "2025-12-10"] or list(top) != [43, 41]:
        raise ValueError("expected gas warning cluster changed")
    burst = gas.loc[dates.isin(top.index)].copy()
    december = next(item for item in trained["score_files"]
                    if item["name"] == "validation_2025-12.parquet")
    december_scores = experiment / december["name"]
    if sha256(december_scores) != december["sha256"]:
        raise ValueError("December validation scores differ")
    channels = sorted(burst.channel_id.unique().tolist())
    day_sets = {day: set(burst.loc[burst.prediction_time.dt.strftime("%Y-%m-%d") == day,
                                     "channel_id"]) for day in top.index}
    m1_file = m1_dir / "clean/year=2025/month=12/data_0.parquet"
    source_info = {item["file"]: item["sha256"] for item in q2["source_m1_files"]}
    if source_info.get("clean/year=2025/month=12/data_0.parquet") != sha256(m1_file):
        raise ValueError("December M1 source differs")
    first = burst.prediction_time.min()
    last = burst.prediction_time.max()
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.register("burst_channels", pd.DataFrame({"channel_id": channels}))
        db.register("burst_keys", burst[["channel_id", "prediction_time"]])
        model_scores = db.execute("""SELECT s.channel_id,s.prediction_time,
            s.score_base51,s.score_full121,s.score_linear121
            FROM read_parquet(?) s SEMI JOIN burst_keys
            USING(channel_id,prediction_time)""", [str(december_scores)]).fetch_df()
        feature_file = q2_dir / "year=2025/month=12/model_features.parquet"
        q2_manifest = read_json(q2_dir / "manifest.json")
        december_q2 = next(x for x in q2_manifest["months"] if x["month"] == "2025-12")
        if (sha256(feature_file)
                != december_q2["files"]["model_features.parquet"]["sha256"]):
            raise ValueError("December Q2 features differ")
        feature_rows = db.execute("""SELECT f.* FROM read_parquet(?) f
            SEMI JOIN burst_keys USING(channel_id,prediction_time)""",
            [str(feature_file)]).fetch_df()
        past = db.execute("""SELECT m.channel_id,m.timestamp,m.value_state,m.value_numeric,
            m.value_raw,m.alarm,m.quality_flags
            FROM read_parquet(?,hive_partitioning=false) m
            SEMI JOIN burst_channels USING(channel_id)
            WHERE m.timestamp>? AND m.timestamp<=?
              AND split_part(replace(m.source,chr(92),'/'),'/',-1)
                  ='ext-journal-2025.7z'""",
            [str(m1_file), first - pd.Timedelta(hours=168), last]).fetch_df()
    burst = burst.merge(model_scores, on=["channel_id", "prediction_time"],
                        validate="one_to_one")
    if len(burst) != 84:
        raise ValueError("gas burst score keys differ")
    feature_rows = burst[["channel_id", "prediction_time", "score_linear121"]].merge(
        feature_rows, on=["channel_id", "prediction_time"], validate="one_to_one")
    model_file = experiment / "linear121.joblib"
    model_manifest = read_json(experiment / "manifest.json")
    if sha256(model_file) != model_manifest["model_sha256"]["linear121"]:
        raise ValueError("saved linear model differs")
    linear = joblib.load(model_file)
    names = trained["feature_sets"]["linear121"]
    model_data = feature_rows[names].assign(
        sensor_type=feature_rows.sensor_type.fillna("<unknown>"))
    repeated = linear.predict_proba(model_data)[:, 1]
    if not np.allclose(repeated, feature_rows.score_linear121, atol=1e-6):
        raise ValueError("burst score is not reproducible from saved model/features")
    preprocess = linear.steps[0][1]
    classifier = linear.steps[1][1]
    transformed = preprocess.transform(model_data)
    matrix = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    contributions = np.asarray(matrix) * classifier.coef_[0]
    median_contributions = np.median(contributions, axis=0)
    ranked = sorted(zip(preprocess.get_feature_names_out(), median_contributions,
                        strict=True), key=lambda item: item[1], reverse=True)
    # Restrict each warning's diagnostic past window; do not read any future event.
    past.sort_values(["channel_id", "timestamp"], inplace=True)
    by_channel = {channel: group for channel, group in past.groupby("channel_id")}
    window_counts = []
    fault_168 = []
    normal_168 = []
    recent_state = Counter()
    recent_quality = Counter()
    for channel, at in burst[["channel_id", "prediction_time"]].itertuples(
        index=False, name=None
    ):
        events = by_channel.get(channel)
        if events is None:
            window_counts.append(0)
            fault_168.append(0)
            normal_168.append(0)
            continue
        recent = events.loc[(events.timestamp > at - pd.Timedelta(hours=24))
                            & (events.timestamp <= at)]
        week = events.loc[(events.timestamp > at - pd.Timedelta(hours=168))
                          & (events.timestamp <= at)]
        window_counts.append(len(recent))
        fault_168.append(int(week.value_state.eq("Неисправен").sum()))
        normal_168.append(int(week.value_state.eq("Норма").sum()))
        recent_state.update(str(x) for x in recent.value_state.fillna("<numeric>"))
        for flags in recent.quality_flags:
            if flags is not None:
                recent_quality.update(flags)
    burst["raw_m1_rows_24h"] = window_counts
    burst["raw_m1_fault_rows_168h"] = fault_168
    burst["raw_m1_normal_rows_168h"] = normal_168
    fault_events = past.loc[past.value_state == "Неисправен"].copy()
    fault_by_day = []
    for day, group in fault_events.groupby(fault_events.timestamp.dt.strftime("%Y-%m-%d")):
        fault_by_day.append({"day": day, "rows": len(group),
                             "channels": group.channel_id.nunique()})
    fault_by_second = fault_events.groupby("timestamp").agg(
        rows=("channel_id", "size"), channels=("channel_id", "nunique"))
    top_fault_seconds = [
        {"timestamp": at.isoformat(), "rows": int(row.rows),
         "channels": int(row.channels)}
        for at, row in fault_by_second.sort_values("rows", ascending=False).head(10).iterrows()
    ]
    by_day = {}
    base_threshold = selected["models"]["base51"]["goal_check"][
        "diagnostic_best_full_f1"]["threshold"]
    full_threshold = selected["models"]["full121"]["goal_check"][
        "diagnostic_best_full_f1"]["threshold"]
    for day, group in burst.groupby(burst.prediction_time.dt.strftime("%Y-%m-%d")):
        by_day[day] = {
            "warnings": len(group), "channels": group.channel_id.nunique(),
            "prediction_hour_counts": {str(k): int(v) for k, v in
                                       group.prediction_time.dt.hour.value_counts().sort_index().items()},
            "score": _numeric(group, "score"),
            "base51_score": _numeric(group, "score_base51"),
            "full121_score": _numeric(group, "score_full121"),
            "base51_rows_above_own_diagnostic_threshold": int(
                group.score_base51.ge(base_threshold).sum()),
            "full121_rows_above_own_diagnostic_threshold": int(
                group.score_full121.ge(full_threshold).sum()),
            "event_count_24h": _numeric(group, "event_count_24h"),
            "technical_message_count_168h": _numeric(
                group, "technical_message_count_168h"),
            "registered_fault_text_count_168h": _numeric(
                group, "registered_fault_text_count_168h"),
            "baseline_event_count": _numeric(group, "baseline_event_count"),
            "last_observation_age_seconds": _numeric(group,
                                                      "last_observation_age_seconds"),
            "completed_episode_count_168h": _numeric(group,
                                                       "completed_episode_count_168h"),
            "raw_m1_rows_24h": _numeric(group, "raw_m1_rows_24h"),
            "raw_m1_fault_rows_168h": _numeric(group, "raw_m1_fault_rows_168h"),
            "raw_m1_normal_rows_168h": _numeric(group, "raw_m1_normal_rows_168h"),
            "missing_last_completed_episode_age": int(
                group.last_completed_episode_end_age_seconds.isna().sum()),
            "qa_gas_negative_count_nonzero": int(
                group.qa_gas_negative_reading_count_24h.gt(0).sum()),
            "qa_gas_alarm_count_nonzero": int(
                group.qa_gas_alarm_level_candidate_count_24h.gt(0).sum()),
        }
    result = {
        "schema_version": "q2-b-gas-burst-retrospective-audit-v1",
        "source_error_manifest_sha256": sha256(error_dir / "manifest.json"),
        "source_experiment_manifest_sha256": sha256(experiment / "manifest.json"),
        "source_decision_sha256": sha256(decision),
        "source_m1_file_sha256": sha256(m1_file),
        "days": by_day,
        "unique_channels_across_both_days": len(channels),
        "channels_on_both_days": len(day_sets["2025-12-09"] & day_sets["2025-12-10"]),
        "raw_m1_rows_in_168h_union": len(past),
        "recent_24h_value_states": dict(recent_state),
        "recent_24h_quality_flags": dict(recent_quality),
        "raw_m1_fault_rows_by_day_in_union": fault_by_day,
        "raw_m1_top_fault_seconds_in_union": top_fault_seconds,
        "linear_model_median_contributions": {
            "largest_positive": [{"field": str(name), "logit_contribution": float(value)}
                                 for name, value in ranked[:10]],
            "largest_negative": [{"field": str(name), "logit_contribution": float(value)}
                                 for name, value in ranked[-10:]],
            "intercept": float(classifier.intercept_[0]),
            "interpretation": "Standardized linear terms are diagnostic associations, not causal effects.",
        },
        "test_data_read": False,
        "diagnostic_only": True,
        "limitation": "Only the past 168 hours of these flagged channels were read; no causal interpretation is asserted.",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("error-dir", "experiment", "decision", "q2-dir", "m1-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"unique_channels": result["unique_channels_across_both_days"],
                      "repeat_channels": result["channels_on_both_days"]}), flush=True)


if __name__ == "__main__":
    main()
