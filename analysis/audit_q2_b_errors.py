"""Diagnose fixed Q2/B validation warnings without fitting or retuning.

Retrospective onset distances are diagnostic outputs, never model inputs.
Only accepted 2025 validation scores and Q2/B3 diagnostics are read.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.alert_eval import evaluate_alerts


FEATURES = (
    "last_observation_age_seconds",
    "baseline_event_count",
    "event_count_24h",
    "registered_fault_text_count_24h",
    "technical_message_count_168h",
    "registered_fault_text_count_168h",
    "normal_message_count_24h",
    "completed_episode_count_168h",
    "last_completed_episode_end_age_seconds",
    "qa_gas_negative_reading_count_24h",
    "qa_gas_alarm_level_candidate_count_24h",
)
MODEL = "linear121"


def _paths(experiment: Path, q2_dir: Path) -> tuple[list[Path], list[Path], dict]:
    trained = read_json(experiment / "report.json")
    manifest = read_json(experiment / "manifest.json")
    if (trained["schema_version"] != "q2-b-expanded-validation-v1"
            or manifest["report_sha256"] != sha256(experiment / "report.json")
            or trained["q2_manifest_sha256"] != sha256(q2_dir / "manifest.json")
            or len(trained["score_files"]) != 12):
        raise ValueError("Q2/B validation lineage differs")
    scores = [experiment / item["name"] for item in trained["score_files"]]
    if any(sha256(path) != item["sha256"] for path, item in
           zip(scores, trained["score_files"], strict=True)):
        raise ValueError("saved validation score file differs")
    q2 = read_json(q2_dir / "manifest.json")
    features = [q2_dir / f"year=2025/month={i:02d}/model_features.parquet"
                for i in range(1, 13)]
    for month, path in zip((m for m in q2["months"] if m["split"] == "validation"),
                           features, strict=True):
        if month["files"]["model_features.parquet"]["sha256"] != sha256(path):
            raise ValueError(f"Q2 validation feature file differs: {path}")
    return scores, features, trained


def _onset_distances(alerts: pd.DataFrame, q2_dir: Path) -> pd.DataFrame:
    positives = pq.read_table(q2_dir / "positive_hour_diagnostics.parquet",
                              columns=["channel_id", "split", "target_episode_id",
                                       "label_available_at"]).to_pandas()
    positives = positives.loc[positives.split == "validation"]
    onset = positives.drop_duplicates("target_episode_id")
    if (onset.target_episode_id.nunique() != 2142
            or positives.groupby("target_episode_id").label_available_at.nunique().max() != 1):
        raise ValueError("assigned validation episode onsets differ")
    by_channel: dict[str, list[pd.Timestamp]] = defaultdict(list)
    for channel, at in onset[["channel_id", "label_available_at"]].itertuples(
        index=False, name=None
    ):
        by_channel[channel].append(at)
    for values in by_channel.values():
        values.sort()
    before = []
    after = []
    for channel, at in alerts[["channel_id", "prediction_time"]].itertuples(
        index=False, name=None
    ):
        values = by_channel.get(channel, [])
        prev = bisect_right(values, at) - 1
        future = bisect_right(values, at)
        before.append((at - values[prev]).total_seconds() / 3600 if prev >= 0 else np.nan)
        after.append((values[future] - at).total_seconds() / 3600
                     if future < len(values) else np.nan)
    result = alerts.copy()
    result["hours_since_previous_assigned_onset"] = before
    result["hours_to_next_assigned_onset"] = after
    return result


def _episodes(db: duckdb.DuckDBPyConnection, scores: list[Path],
              q2_dir: Path, threshold: float, matched: set[str]) -> pd.DataFrame:
    scored = db.execute("""SELECT target_episode_id, ANY_VALUE(channel_id) channel_id,
        ANY_VALUE(sensor_type) sensor_type, COUNT(*) positive_hours,
        MAX(score_linear121) max_score,
        COUNT(*) FILTER(WHERE score_linear121>=?) above_threshold_hours
        FROM read_parquet(?) WHERE target=1 GROUP BY target_episode_id""",
        [threshold, [str(path) for path in scores]]).fetch_df()
    full = pq.read_table(q2_dir / "episode_diagnostics.parquet").to_pandas()
    full = full.loc[full.split == "validation", [
        "target_episode_id", "channel_id", "sensor_type", "positive_hours",
        "candidate_hours", "reasons_every_positive_hour",
    ]]
    result = full.merge(scored, on="target_episode_id", how="left", validate="one_to_one",
                        suffixes=("", "_scored"))
    if (len(result) != 2142 or result.target_episode_id.nunique() != 2142
            or not result.loc[result.candidate_hours > 0, "positive_hours_scored"].eq(
                result.loc[result.candidate_hours > 0, "candidate_hours"]).all()
            or result.loc[result.candidate_hours == 0, "positive_hours_scored"].notna().any()
            or not result.loc[result.candidate_hours > 0, "channel_id_scored"].eq(
                result.loc[result.candidate_hours > 0, "channel_id"]).all()):
        raise ValueError("Q2 positive episodes/available hours disagree with saved scores")
    result["error_class"] = np.select(
        [result.candidate_hours.eq(0), result.target_episode_id.isin(matched),
         result.above_threshold_hours.fillna(0).eq(0)],
        ["unavailable", "matched", "below_threshold"],
        default="cooldown_suppressed",
    )
    return result


def _concentration(alerts: pd.DataFrame) -> dict:
    false = alerts.loc[alerts.outcome != "matched_episode"]
    result = {}
    for kind, group in false.groupby("sensor_type", dropna=False):
        counts = group.channel_id.value_counts()
        result[str(kind)] = {
            "unmatched_warnings": len(group),
            "channels": len(counts),
            "top_1_channel_share": float(counts.head(1).sum() / len(group)),
            "top_5_channel_share": float(counts.head(5).sum() / len(group)),
            "top_10_channel_share": float(counts.head(10).sum() / len(group)),
            "top_5_channels": [{"channel_id": str(channel), "warnings": int(count)}
                               for channel, count in counts.head(5).items()],
        }
    return result


def _feature_summary(alerts: pd.DataFrame) -> dict:
    result = {}
    for (kind, outcome), group in alerts.groupby(["sensor_type", "outcome"], dropna=False):
        fields = {}
        for name in FEATURES:
            values = pd.to_numeric(group[name], errors="coerce")
            fields[name] = {"median": float(values.median()) if values.notna().any() else None,
                            "nonzero_share": float(values.gt(0).mean()),
                            "missing_share": float(values.isna().mean())}
        result.setdefault(str(kind), {})[outcome] = {"warnings": len(group),
                                                      "features": fields}
    return result


def _false_proximity_by_type(false: pd.DataFrame) -> dict:
    result = {}
    for kind, group in false.groupby("sensor_type", dropna=False):
        before = group.hours_since_previous_assigned_onset
        after = group.hours_to_next_assigned_onset
        result[str(kind)] = {
            "unmatched_warnings": len(group),
            "previous_onset_within_24h": int(before.le(24).sum()),
            "previous_onset_within_168h": int(before.le(168).sum()),
            "next_onset_24_to_48h": int((after.gt(24) & after.le(48)).sum()),
            "next_onset_48_to_168h": int((after.gt(48) & after.le(168)).sum()),
            "no_next_assigned_onset_in_2025": int(after.isna().sum()),
        }
    return result


def _cooldown_sources(db: duckdb.DuckDBPyConnection, scores: list[Path],
                      episodes: pd.DataFrame, alerts: pd.DataFrame,
                      threshold: float) -> tuple[pd.DataFrame, dict]:
    missed = episodes.loc[episodes.error_class == "cooldown_suppressed",
                          ["target_episode_id"]]
    db.register("cooldown_missed", missed)
    selected = db.execute("""SELECT target_episode_id,channel_id,prediction_time
        FROM read_parquet(?) s SEMI JOIN cooldown_missed USING(target_episode_id)
        WHERE s.target=1 AND s.score_linear121>=?
        ORDER BY target_episode_id,prediction_time""",
        [[str(path) for path in scores], threshold]).fetch_df()
    first = selected.drop_duplicates("target_episode_id")
    by_channel = {}
    for channel, group in alerts.groupby("channel_id"):
        ordered = group.sort_values("prediction_time")
        by_channel[channel] = (ordered.prediction_time.tolist(), ordered.outcome.tolist())
    causes = []
    for episode, channel, at in first.itertuples(index=False, name=None):
        times, outcomes = by_channel[channel]
        previous = bisect_right(times, at) - 1
        if previous < 0:
            raise ValueError("above-threshold missed episode has no prior warning")
        delta_hours = (at - times[previous]).total_seconds() / 3600
        if not 0 <= delta_hours < 24:
            raise ValueError("above-threshold missed episode is outside cooldown")
        causes.append({"target_episode_id": episode,
                       "first_high_score_at": at,
                       "blocking_warning_outcome": outcomes[previous],
                       "hours_after_blocking_warning": delta_hours})
    result = pd.DataFrame(causes)
    if len(result) != len(missed):
        raise ValueError("cooldown suppression sources do not cover missed episodes")
    counts = {str(k): int(v) for k, v in result.blocking_warning_outcome.value_counts().items()}
    return result, counts


def run(*, experiment: Path, decision: Path, q2_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    scores, features, trained = _paths(experiment, q2_dir)
    choice_file = read_json(decision)
    if (choice_file["source_experiment_manifest_sha256"]
            != sha256(experiment / "manifest.json")
            or choice_file["any_model_meets_both_strict_goals"]):
        raise ValueError("Q2/B fixed research decision differs")
    choice = choice_file["models"][MODEL]["goal_check"]["diagnostic_best_full_f1"]
    threshold = float(choice["threshold"])
    days = trained["validation"]["eligible_channel_days_with_binary_label"]
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        selected = db.execute("""SELECT channel_id,prediction_time,sensor_type,target,
            target_episode_id,label_available_at,score_linear121 AS catboost_score
            FROM read_parquet(?) WHERE score_linear121>=?""",
            [[str(path) for path in scores], threshold]).fetch_df()
        metrics, alerts = evaluate_alerts(selected, "catboost_score", threshold,
                                          channel_days=days)
        if (metrics["matched_episodes"] != choice["matched_episodes"]
                or metrics["emitted_warnings"] != choice["emitted_warnings"]):
            raise ValueError("emitted warning replay differs from frozen validation choice")
        matched = set(alerts.loc[alerts.outcome == "matched_episode", "target_episode_id"])
        episodes = _episodes(db, scores, q2_dir, threshold, matched)
        cooldown, cooldown_causes = _cooldown_sources(db, scores, episodes, alerts,
                                                       threshold)
        db.register("warning_keys", alerts[["channel_id", "prediction_time"]])
        projection = ",".join(f'f."{name}"' for name in FEATURES)
        past = db.execute(
            "SELECT f.channel_id,f.prediction_time,f.sensor_type," + projection
            + " FROM read_parquet(?) f SEMI JOIN warning_keys USING(channel_id,prediction_time)",
            [[str(path) for path in features]],
        ).fetch_df()
    alerts = alerts.merge(past.drop(columns="sensor_type"),
                          on=["channel_id", "prediction_time"], validate="one_to_one")
    if len(alerts) != metrics["emitted_warnings"]:
        raise ValueError("past feature join did not cover every emitted warning")
    alerts = _onset_distances(alerts, q2_dir)
    episodes = episodes.merge(cooldown, on="target_episode_id", how="left",
                              validate="one_to_one")
    false = alerts.loc[alerts.outcome != "matched_episode"]
    by_type = {
        str(kind): {str(status): int(n) for status, n in group.error_class.value_counts().items()}
        for kind, group in episodes.groupby("sensor_type", dropna=False)
    }
    classes = {str(status): int(n) for status, n in episodes.error_class.value_counts().items()}
    if (classes.get("unavailable") != 783 or classes.get("matched") != 346
            or sum(classes.values()) != 2142):
        raise ValueError("episode error decomposition differs from accepted Q2/B counts")
    near = {
        "false_with_previous_onset_within_24h": int(false.hours_since_previous_assigned_onset.le(24).sum()),
        "false_with_previous_onset_within_168h": int(false.hours_since_previous_assigned_onset.le(168).sum()),
        "false_with_next_onset_within_24h": int(false.hours_to_next_assigned_onset.le(24).sum()),
        "false_with_next_onset_24_to_48h": int((
            false.hours_to_next_assigned_onset.gt(24)
            & false.hours_to_next_assigned_onset.le(48)).sum()),
        "false_with_next_onset_48_to_168h": int((
            false.hours_to_next_assigned_onset.gt(48)
            & false.hours_to_next_assigned_onset.le(168)).sum()),
        "false_without_next_assigned_onset_in_2025": int(false.hours_to_next_assigned_onset.isna().sum()),
    }
    month_type = []
    for (month, kind), group in alerts.groupby(
        [alerts.prediction_time.dt.strftime("%Y-%m"), "sensor_type"], dropna=False
    ):
        month_type.append({"month": str(month), "sensor_type": str(kind),
                           "warnings": len(group),
                           "matched": int((group.outcome == "matched_episode").sum()),
                           "unmatched": int((group.outcome != "matched_episode").sum())})
    day_type = []
    for (day, kind), group in alerts.groupby(
        [alerts.prediction_time.dt.strftime("%Y-%m-%d"), "sensor_type"], dropna=False
    ):
        day_type.append({"day": str(day), "sensor_type": str(kind),
                         "warnings": len(group),
                         "channels": group.channel_id.nunique(),
                         "matched": int((group.outcome == "matched_episode").sum())})
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if pending.exists():
        raise FileExistsError(pending)
    pending.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(alerts, preserve_index=False),
                   pending / "warning_cases.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(episodes, preserve_index=False),
                   pending / "episode_cases.parquet", compression="zstd")
    report = {
        "schema_version": "q2-b-fixed-warning-error-audit-v1",
        "source_experiment_manifest_sha256": sha256(experiment / "manifest.json"),
        "source_decision_sha256": sha256(decision),
        "source_q2_manifest_sha256": sha256(q2_dir / "manifest.json"),
        "model": MODEL, "threshold": threshold,
        "warning_counts": {"emitted": len(alerts), "matched": len(matched),
                           "unmatched": len(false),
                           "outcomes": {str(k): int(v) for k, v in alerts.outcome.value_counts().items()}},
        "episode_classes": classes,
        "cooldown_suppression_first_high_score_blockers": cooldown_causes,
        "episode_classes_by_sensor_type": by_type,
        "false_warning_onset_proximity": near,
        "false_warning_onset_proximity_by_sensor_type": _false_proximity_by_type(false),
        "warnings_by_month_and_sensor_type": month_type,
        "warnings_by_day_and_sensor_type": day_type,
        "false_warning_channel_concentration": _concentration(alerts),
        "past_feature_summary_by_type_and_outcome": _feature_summary(alerts),
        "diagnostic_only": True,
        "no_fit_or_threshold_change": True,
        "test_data_read": False,
        "limitations": [
            "Unknown-label hours are absent from warning/cooldown evaluation.",
            "Next-onset proximity is retrospective and cannot be used as a live feature.",
            "No onset after the 2025 validation boundary was read; late-year distances are censored.",
            "Journal fault entries are not confirmed physical failures.",
        ],
    }
    (pending / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    manifest = {"schema_version": report["schema_version"],
                "report_sha256": sha256(pending / "report.json"),
                "warning_cases_sha256": sha256(pending / "warning_cases.parquet"),
                "episode_cases_sha256": sha256(pending / "episode_cases.parquet")}
    (pending / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                                           + "\n", encoding="utf-8")
    pending.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "decision", "q2-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"episode_classes": result["episode_classes"],
                      "warning_counts": result["warning_counts"]}), flush=True)


if __name__ == "__main__":
    main()
