"""Exact retrospective Q2 Recall ceiling; oracle labels are never model inputs."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import timedelta
from pathlib import Path
import time

import duckdb
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_quality_improvement_a import safe_path
from analysis.build_sparse_population_a import write_json
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.alert_eval import COOLDOWN, evaluate_alerts


Q2_SHA = "c9775a94bfcff09d1c641b010b93d62e9927209017ccbc7346749f78525de619"
B_SHA = "4d78bb063ce2759a3b00bbe1a469e6679dc4aed479f08936db1aec437783c04a"
COLUMNS = (
    "channel_id",
    "prediction_time",
    "sensor_type",
    "target",
    "target_episode_id",
    "label_available_at",
)


def optimal_schedule(points: pd.DataFrame, cooldown: timedelta = COOLDOWN):
    """Greedy witness and independent DP optimum for fixed-spacing positive points.

    Episode windows MUST span less than cooldown. Then any feasible selection
    automatically has at most one point per episode, so ordinary point scheduling
    is exact. Without that invariant this method refuses to claim an optimum.
    """
    if cooldown <= timedelta(0) or not set(COLUMNS) <= set(points.columns):
        raise ValueError("invalid cooldown or incomplete positive points")
    if points.empty or points[list(COLUMNS)].isna().any().any():
        raise ValueError("positive points must be nonempty and fully attributed")
    if not points.target.eq(1).all() or points.duplicated(["channel_id", "prediction_time"]).any():
        raise ValueError("nonpositive or duplicate oracle point")
    if points.prediction_time.dt.tz is not None or points.label_available_at.dt.tz is not None:
        raise ValueError("oracle uses naive journal times, not delivery timestamps")
    lead = points.label_available_at - points.prediction_time
    if not (lead.gt(timedelta(0)) & lead.le(timedelta(hours=24))).all():
        raise ValueError("positive point lies outside the accepted horizon")
    grouped = points.groupby("target_episode_id")
    if not grouped[["channel_id", "sensor_type", "label_available_at"]].nunique().eq(1).all().all():
        raise ValueError("episode attribution changes")
    spans = grouped.prediction_time.agg(["min", "max"])
    if not (spans["max"] - spans["min"]).lt(cooldown).all():
        raise ValueError("episode span invalidates one-warning point-scheduling proof")
    if not points.groupby("channel_id").sensor_type.nunique().eq(1).all():
        raise ValueError("channel type changes; by-type capacity cannot be attributed")
    selected, channels = [], []
    for channel, group in points.groupby("channel_id", sort=True):
        ordered = group.sort_values("prediction_time", kind="mergesort")
        times = ordered.prediction_time.tolist()
        best = [0] * (len(times) + 1)
        for i, at in enumerate(times):
            compatible = bisect_right(times, at - cooldown, 0, i)
            best[i + 1] = max(best[i], 1 + best[compatible])
        last, witness = None, []
        for row in ordered[list(COLUMNS)].to_dict("records"):
            if last is None or row["prediction_time"] - last >= cooldown:
                witness.append(row)
                last = row["prediction_time"]
        if len(witness) != best[-1] or len({r["target_episode_id"] for r in witness}) != len(
            witness
        ):
            raise ValueError("greedy witness disagrees with DP or reuses an episode")
        selected.extend(witness)
        channels.append(
            {
                "channel_id": channel,
                "sensor_type": ordered.sensor_type.iloc[0],
                "positive_hours": len(ordered),
                "available_episodes": ordered.target_episode_id.nunique(),
                "greedy_matches": len(witness),
                "dynamic_program_matches": best[-1],
            }
        )
    return selected, channels


def audit(*, q2_dir: Path, experiment: Path, output_dir: Path):
    begun = time.perf_counter()
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    if sha256(q2_dir / "manifest.json") != Q2_SHA or sha256(experiment / "manifest.json") != B_SHA:
        raise ValueError("accepted Q2/B package differs")
    manifest = read_json(q2_dir / "manifest.json")
    for name in ("positive_hour_diagnostics.parquet", "episode_diagnostics.parquet"):
        if sha256(q2_dir / name) != manifest["files"][name]["sha256"]:
            raise ValueError("Q2 positive diagnostics differ")
    full = pq.ParquetFile(q2_dir / "positive_hour_diagnostics.parquet").read().to_pandas()
    val = full.loc[full.split == "validation"]
    if (
        len(val) != 19588
        or val.target_episode_id.nunique() != 2142
        or not val.target.eq(1).all()
        or not val.split_status.eq("assigned").all()
    ):
        raise ValueError("assigned full validation denominator differs")
    points = val.loc[val.admission_status == "eligible", list(COLUMNS)].copy()
    if len(points) != 16202 or points.target_episode_id.nunique() != 1359:
        raise ValueError("accepted eligible positive population differs")
    b_manifest = read_json(experiment / "manifest.json")
    expected_names = [f"validation_2025-{i:02d}.parquet" for i in range(1, 13)]
    if [r["name"] for r in b_manifest["score_files"]] != expected_names:
        raise ValueError("B validation monthly scope differs")
    paths = []
    for item in b_manifest["score_files"]:
        file = safe_path(experiment, item["name"])
        if sha256(file) != item["sha256"]:
            raise ValueError("B scores differ before the independent oracle comparison")
        paths.append(str(file))
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.register("a_positive", pa.Table.from_pandas(points, preserve_index=False))
        projection = ",".join(COLUMNS)
        db.execute(
            "CREATE TEMP TABLE b_positive AS SELECT "
            + projection
            + " FROM read_parquet(?,hive_partitioning=false) WHERE target=1",
            [paths],
        )
        for left, right in (("a_positive", "b_positive"), ("b_positive", "a_positive")):
            if db.execute(
                f"SELECT COUNT(*) FROM (SELECT * FROM {left} EXCEPT ALL SELECT * FROM {right})"
            ).fetchone()[0]:
                raise ValueError("B saved positive points differ from A causal admission")
        independent_points = db.execute("SELECT * FROM b_positive").fetch_df()
    witness, channels = optimal_schedule(points)
    b_witness, _ = optimal_schedule(independent_points)
    if len(witness) != len(b_witness):
        raise ValueError("independent B-score input yields another oracle capacity")
    oracle = points.assign(rule_score=1.0)
    canonical, alerts = evaluate_alerts(oracle, "rule_score", 0.5, channel_days=1)
    if canonical["matched_episodes"] != len(witness) or canonical["emitted_warnings"] != len(
        witness
    ):
        raise ValueError("canonical cooldown/matching disagrees with optimum")
    full_by_type = val.groupby("sensor_type").target_episode_id.nunique().to_dict()
    available = points.groupby("sensor_type").target_episode_id.nunique().to_dict()
    matched = Counter(r["sensor_type"] for r in witness)
    needed = len(set(val.target_episode_id)) // 2 + 1
    if len(witness) != 1116:
        raise ValueError("published preliminary oracle result was not reproduced")
    pending.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pandas(alerts, preserve_index=False),
        pending / "oracle_schedule.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(channels), pending / "channel_capacity.parquet", compression="zstd"
    )
    report = {
        "schema_version": "q2-a-exact-oracle-capacity-v1",
        "status": "diagnostic_verified_not_model_quality",
        "source_q2_manifest_sha256": Q2_SHA,
        "source_b_manifest_sha256": B_SHA,
        "full_positive_episodes": 2142,
        "eligible_positive_hours": len(points),
        "available_positive_episodes": 1359,
        "admission_only_recall_ceiling": 1359 / 2142,
        "cooldown_hours": 24,
        "initial_cooldown_state": "empty_at_validation_start_as_in_B",
        "one_warning_per_episode": True,
        "oracle_matched_episodes": len(witness),
        "recall_ceiling_with_cooldown": len(witness) / 2142,
        "unavailable_episodes": 783,
        "capacity_loss_within_available_population": 1359 - len(witness),
        "minimum_matches_for_strict_recall_above_half": needed,
        "minimum_fraction_of_oracle_capacity": needed / len(witness),
        "maximum_additional_misses_at_goal_floor": len(witness) - needed,
        "by_type": [
            {
                "sensor_type": kind,
                "all_episodes": n,
                "available_episodes": available.get(kind, 0),
                "oracle_matches": matched[kind],
                "oracle_recall_ceiling": matched[kind] / n,
            }
            for kind, n in sorted(full_by_type.items())
        ],
        "checks": {
            "greedy_equals_independent_dynamic_program": True,
            "B_saved_positive_keys_equal_A": True,
            "B_input_capacity_equals_A": True,
            "canonical_alert_evaluator_agrees": True,
            "all_episode_positive_spans_lt_cooldown": True,
        },
        "proof": [
            "Removing every false warning can only relax the feasible schedules.",
            "Every episode's positive points span less than 24h, so a 24h-separated schedule cannot reuse an episode.",
            "The earliest feasible point leaves at least as much remaining time as any later first choice; greedy is optimal by exchange induction.",
            "A separate prefix DP evaluates taking/skipping every point and agrees channel by channel.",
        ],
        "diagnostic_uses_future_labels": True,
        "oracle_is_not_a_causal_model": True,
        "prediction_rules_changed": False,
        "labels_changed": False,
        "test_data_read": False,
        "limitations": [
            "This bound is conditional on the fixed hourly grid, Q2 admission, B3 labels and current matching/cooldown.",
            "Unknown-label hours are absent, as in the retrospective B evaluation; live-stream quality is not measured.",
            "A selected witness is not a uniquely optimal episode set; do not declare particular unselected episodes universally impossible.",
            "The oracle knows future labels and emits no false warnings; it is not achievable-quality evidence.",
        ],
        "code_lf_sha256": frozen_rule_sha256(Path(__file__)),
        "dependencies_lf_sha256": {
            name: frozen_rule_sha256(Path(name)) for name in ("ml/forecast/alert_eval.py",)
        },
        "resources": {
            "elapsed_seconds": round(time.perf_counter() - begun, 3),
            "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", 0),
        },
    }
    write_json(pending / "report.json", report)
    write_json(
        pending / "manifest.json",
        {
            "schema_version": report["schema_version"],
            "files": {
                name: {"sha256": sha256(pending / name), "bytes": (pending / name).stat().st_size}
                for name in ("report.json", "oracle_schedule.parquet", "channel_capacity.parquet")
            },
        },
    )
    pending.rename(output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("q2-dir", "experiment", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(**vars(args))
    print(
        {
            k: report[k]
            for k in (
                "oracle_matched_episodes",
                "recall_ceiling_with_cooldown",
                "checks",
                "resources",
            )
        }
    )


if __name__ == "__main__":
    main()
