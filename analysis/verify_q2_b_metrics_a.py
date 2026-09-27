"""Independent replay of published Q2 warnings; no fit, new grid or rule change."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_q2_oracle_a import B_SHA, Q2_SHA
from analysis.build_sparse_population_a import write_json
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256


DECISION_SHA = "055cf87cdcf722f79d253a74811358dc72ea3932aa2387cd838b5a0c2867ce88"
ERROR_SHA = "a70c34d0ce731c3bb0baa5f5935e72934c2694715f968b9175872af1886fecba"
GAS_SHA = "2ee64d9adb593f4e6c48a008aced24d2ad1a2bada350eded60ecd0687fc2d9ad"
WARNING_COLUMNS = (
    "channel_id",
    "prediction_time",
    "sensor_type",
    "target",
    "target_episode_id",
    "label_available_at",
    "score",
    "outcome",
)


def chronological_warnings(selected: pd.DataFrame) -> pd.DataFrame:
    """Jump to the next 24h-compatible point, independently of alert_eval's loop.

    Input is already thresholded binary validation rows. Initial state is empty,
    just as in the published retrospective evaluation, not a live-stream claim.
    """
    required = set(WARNING_COLUMNS) - {"outcome"}
    if (
        not required <= set(selected)
        or selected.duplicated(["channel_id", "prediction_time"]).any()
    ):
        raise ValueError("incomplete or duplicate warning candidate")
    if (
        selected[["channel_id", "prediction_time", "sensor_type", "target", "score"]]
        .isna()
        .any()
        .any()
        or not selected.target.isin([0, 1]).all()
    ):
        raise ValueError("warning candidates must be attributed binary rows")
    if not np.isfinite(selected.score).all() or not selected.score.between(0, 1).all():
        raise ValueError("invalid warning score")
    positives = selected.loc[selected.target.eq(1)]
    lead = positives.label_available_at - positives.prediction_time
    if (
        positives[["target_episode_id", "label_available_at"]].isna().any().any()
        or not (lead.gt(pd.Timedelta(0)) & lead.le(pd.Timedelta(hours=24))).all()
    ):
        raise ValueError("positive candidate lacks a valid episode/horizon")
    records, matched = [], set()
    for _, group in selected.groupby("channel_id", sort=True):
        ordered = group.sort_values("prediction_time", kind="mergesort")
        times = ordered.prediction_time.to_numpy(dtype="datetime64[ns]")
        rows = ordered[[name for name in WARNING_COLUMNS if name != "outcome"]].to_dict("records")
        index = 0
        while index < len(rows):
            row = rows[index]
            episode = row["target_episode_id"]
            if row["target"] == 0:
                outcome = "no_target_in_horizon"
            elif episode in matched:
                outcome = "duplicate_episode_warning"
            else:
                outcome = "matched_episode"
                matched.add(episode)
            records.append({**row, "outcome": outcome})
            index = int(np.searchsorted(times, times[index] + np.timedelta64(24, "h"), side="left"))
    return pd.DataFrame(records, columns=WARNING_COLUMNS)


def episode_classes(
    episodes: pd.DataFrame, available: set, above: set, matched: set
) -> pd.DataFrame:
    if (
        episodes.target_episode_id.isna().any()
        or episodes.target_episode_id.duplicated().any()
        or not matched <= above <= available <= set(episodes.target_episode_id)
    ):
        raise ValueError("episode sets or attribution disagree")
    result = episodes[["target_episode_id", "channel_id", "sensor_type"]].copy()
    result["error_class"] = [
        "unavailable"
        if episode not in available
        else "matched"
        if episode in matched
        else "below_threshold"
        if episode not in above
        else "cooldown_suppressed"
        for episode in result.target_episode_id
    ]
    return result


def equal_relations(db, left: str, right: str, columns: tuple[str, ...]) -> None:
    projection = ",".join(columns)
    for first, second in ((left, right), (right, left)):
        count = db.execute(
            f"SELECT COUNT(*) FROM (SELECT {projection} FROM {first} "
            f"EXCEPT ALL SELECT {projection} FROM {second})"
        ).fetchone()[0]
        if count:
            raise ValueError(f"independent {left}/{right} replay differs: {count}")


def verify(
    *,
    experiment: Path,
    q2_dir: Path,
    decision: Path,
    error_dir: Path,
    gas_report: Path,
    score_verification: Path,
    output_dir: Path,
) -> dict:
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    for path, expected in (
        (experiment / "manifest.json", B_SHA),
        (q2_dir / "manifest.json", Q2_SHA),
        (decision, DECISION_SHA),
        (error_dir / "manifest.json", ERROR_SHA),
        (gas_report, GAS_SHA),
    ):
        if sha256(path) != expected:
            raise ValueError(f"recreated file differs from B's published hash: {path}")
    verified_manifest = read_json(score_verification / "manifest.json")
    verified_path = score_verification / "report.json"
    if sha256(verified_path) != verified_manifest["files"]["report.json"]["sha256"]:
        raise ValueError("A all-score verification report changed")
    verified = read_json(verified_path)
    if (
        verified["source_b_manifest_sha256"] != B_SHA
        or verified["source_q2_manifest_sha256"] != Q2_SHA
        or verified["rows_checked"] != 13945520
        or verified["score_float32_mismatches"] != 0
        or verified["lineage_mismatches"] != 0
    ):
        raise ValueError("full saved-score verification is incomplete")
    manifest = read_json(experiment / "manifest.json")
    paths = []
    for item in manifest["score_files"]:
        path = experiment / item["name"]
        if sha256(path) != item["sha256"]:
            raise ValueError("source score file changed")
        paths.append(str(path))
    q2 = read_json(q2_dir / "manifest.json")
    episode_path = q2_dir / "episode_diagnostics.parquet"
    if sha256(episode_path) != q2["files"][episode_path.name]["sha256"]:
        raise ValueError("A full episode denominator changed")
    full = pq.ParquetFile(episode_path).read().to_pandas()
    episodes = full.loc[full.split.eq("validation")]
    if len(episodes) != 2142:
        raise ValueError("full validation denominator differs")
    error_manifest = read_json(error_dir / "manifest.json")
    for name in ("report.json", "warning_cases.parquet", "episode_cases.parquet"):
        key = "report_sha256" if name == "report.json" else name.split(".")[0] + "_sha256"
        if sha256(error_dir / name) != error_manifest[key]:
            raise ValueError("recreated B error-audit member changed")
    choices = read_json(decision)
    if choices["any_model_meets_both_strict_goals"]:
        raise ValueError("unexpected published goal decision")
    results, independently_emitted = {}, None
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='3GB'")
        for model, data in choices["models"].items():
            choice = data["goal_check"]["diagnostic_best_full_f1"]
            threshold = choice["threshold"]
            selected = db.execute(
                "SELECT channel_id,prediction_time,sensor_type,target,target_episode_id,"
                f"label_available_at,score_{model}::DOUBLE score FROM read_parquet(?) "
                f"WHERE score_{model}>=?",
                [paths, threshold],
            ).fetch_df()
            warnings = chronological_warnings(selected)
            matched = set(warnings.loc[warnings.outcome.eq("matched_episode"), "target_episode_id"])
            if (
                len(warnings) != choice["emitted_warnings"]
                or len(matched) != choice["matched_episodes"]
            ):
                raise ValueError(f"independent warning metrics differ for {model}")
            results[model] = {
                "threshold": threshold,
                "emitted_warnings": len(warnings),
                "matched_episodes": len(matched),
                "precision": len(matched) / len(warnings),
                "full_episode_recall": len(matched) / 2142,
                "not_a_frozen_deployment_threshold": True,
            }
            if model != "linear121":
                continue
            independently_emitted = warnings
            db.register("independent_warnings", warnings)
            db.read_parquet(str(error_dir / "warning_cases.parquet")).create_view("b_warnings")
            equal_relations(db, "independent_warnings", "b_warnings", WARNING_COLUMNS)
            positive = db.execute(
                "SELECT target_episode_id FROM read_parquet(?) WHERE target=1", [paths]
            ).fetch_df()
            available = set(positive.target_episode_id)
            above = set(selected.loc[selected.target.eq(1), "target_episode_id"])
            classes = episode_classes(episodes, available, above, matched)
            db.register("independent_classes", classes)
            db.read_parquet(str(error_dir / "episode_cases.parquet")).create_view("b_classes")
            equal_relations(
                db,
                "independent_classes",
                "b_classes",
                ("target_episode_id", "channel_id", "sensor_type", "error_class"),
            )
            decomposition = dict(Counter(classes.error_class))
            if decomposition != {
                "unavailable": 783,
                "below_threshold": 844,
                "cooldown_suppressed": 169,
                "matched": 346,
            }:
                raise ValueError("independent episode decomposition differs")
            missed = set(
                classes.loc[classes.error_class.eq("cooldown_suppressed"), "target_episode_id"]
            )
            first = (
                selected.loc[selected.target_episode_id.isin(missed)]
                .sort_values("prediction_time")
                .drop_duplicates("target_episode_id")
            )
            blockers = []
            for row in first.itertuples(index=False):
                before = warnings.loc[
                    warnings.channel_id.eq(row.channel_id)
                    & warnings.prediction_time.le(row.prediction_time)
                ]
                if before.empty:
                    raise ValueError("missed above-threshold episode lacks a blocking warning")
                previous = before.iloc[-1]
                if (
                    not pd.Timedelta(0)
                    <= row.prediction_time - previous.prediction_time
                    < pd.Timedelta(hours=24)
                ):
                    raise ValueError("blocked episode is outside cooldown")
                blockers.append(previous.outcome)
            blocker_counts = dict(Counter(blockers))
    if independently_emitted is None or len(first) != 169:
        raise ValueError("linear replay is incomplete")
    gas = independently_emitted.loc[
        independently_emitted.sensor_type.eq("Газовый датчик")
        & independently_emitted.outcome.ne("matched_episode")
        & independently_emitted.prediction_time.dt.strftime("%Y-%m-%d").isin(
            ["2025-12-09", "2025-12-10"]
        )
    ]
    if len(gas) != 84 or gas.channel_id.nunique() != 49:
        raise ValueError("published gas cluster differs")
    report = {
        "schema_version": "q2-a-independent-B-metrics-acceptance-v1",
        "status": "technical_negative_experiment_accepted_by_A",
        "source_b_manifest_sha256": B_SHA,
        "source_q2_manifest_sha256": Q2_SHA,
        "all_score_verification_manifest_sha256": sha256(score_verification / "manifest.json"),
        "recreated_B_files_equal_published_SHA256": {
            "final_threshold_choice": DECISION_SHA,
            "error_audit_manifest": ERROR_SHA,
            "gas_burst_report": GAS_SHA,
        },
        "independent_evaluator": "per-channel searchsorted to next time >= t+24h; no alert_eval call",
        "full_positive_episodes": 2142,
        "available_positive_episodes": len(available),
        "models_at_published_diagnostic_thresholds": results,
        "linear_warning_keys_and_attribution_equal_B": True,
        "linear_all_episode_classes_equal_B": True,
        "linear_episode_classes": decomposition,
        "first_high_score_blocker_outcomes": blocker_counts,
        "gas_cluster": {
            "unmatched_warnings": len(gas),
            "channels": gas.channel_id.nunique(),
            "precision_if_all_84_removed_counterfactually": 346 / (1305 - 84),
        },
        "no_fit_or_threshold_change": True,
        "labels_or_admission_changed": False,
        "test_data_read": False,
        "customer_semantics_approved": False,
        "requirements_met": False,
        "limitations": [
            "Recreating the published grids is reproduction, not an additional tuning experiment.",
            "Validation is already seen; technical agreement is not independent quality confirmation.",
            "Unknown-label hours do not participate in the retrospective warning/cooldown population.",
            "Journal episodes, not proven physical failures, are the target.",
        ],
        "code_lf_sha256": frozen_rule_sha256(Path(__file__)),
    }
    pending.mkdir(parents=True)
    write_json(pending / "report.json", report)
    pq.write_table(
        pa.Table.from_pandas(independently_emitted, preserve_index=False),
        pending / "independent_warning_keys.parquet",
        compression="zstd",
    )
    write_json(
        pending / "manifest.json",
        {
            "schema_version": report["schema_version"],
            "files": {
                name: {"sha256": sha256(pending / name), "bytes": (pending / name).stat().st_size}
                for name in ("report.json", "independent_warning_keys.parquet")
            },
        },
    )
    pending.rename(output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "experiment",
        "q2-dir",
        "decision",
        "error-dir",
        "gas-report",
        "score-verification",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    result = verify(**vars(parser.parse_args()))
    print({"status": result["status"], "linear_episode_classes": result["linear_episode_classes"]})


if __name__ == "__main__":
    main()
