"""Run the pre-frozen rule once on the sealed conditional test population."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from analysis.train_r4_discrete_baselines import read_json, sha256, verify_inputs
from ml.forecast.alert_eval import evaluate_alerts
from ml.forecast.r6_rule import RULE_VERSION, TERMS, predict_rule


KEYS = ["channel_id", "prediction_time"]


def score_month(database: duckdb.DuckDBPyConnection, candidate: Path,
                features: Path, threshold: float) -> pd.DataFrame:
    projection = ", ".join(f'f."{name}"' for name in TERMS)
    frame = database.execute(
        f"""SELECT c.channel_id,c.prediction_time,c.sensor_type,
                   f.sensor_type AS a3_sensor_type,c.target,
                   c.target_episode_id,c.label_available_at,c.split,{projection}
            FROM read_parquet(?) c
            JOIN read_parquet(?) f USING (channel_id,prediction_time)
            WHERE c.split='test'""",
        [str(candidate), str(features)],
    ).fetch_df()
    if (frame.duplicated(KEYS).any() or not frame.split.eq("test").all()
            or not frame.sensor_type.eq(frame.a3_sensor_type).all()
            or not frame.target.isin([0, 1]).all()):
        raise ValueError("test keys, types, split or target differ")
    scored = predict_rule(frame, eligibility_status="eligible",
                          threshold=threshold)
    if scored.rule_score.isna().any() or scored.alert.isna().any():
        raise ValueError("admitted test row did not receive a score")
    frame["rule_score"] = scored.rule_score.to_numpy()
    frame["above_frozen_threshold"] = scored.alert.to_numpy(dtype=bool)
    positives = frame.loc[frame.target.eq(1)]
    lead = (positives.label_available_at - positives.prediction_time
            ).dt.total_seconds() / 3600
    if (positives.target_episode_id.isna().any() or lead.isna().any()
            or not lead.gt(0).all() or not lead.le(24).all()):
        raise ValueError("test positive episode has invalid horizon or lineage")
    return frame.drop(columns=["a3_sensor_type", "split", *TERMS])


def _by_type(frame: pd.DataFrame, alerts: pd.DataFrame) -> dict:
    emitted = {str(name): group for name, group in
               alerts.groupby("sensor_type", dropna=False)}
    result = {}
    for name, group in frame.groupby("sensor_type", dropna=False):
        key = str(name)
        y = group.target.to_numpy(dtype=np.int8)
        found = emitted.get(key)
        matched = (int(found.outcome.eq("matched_episode").sum())
                   if found is not None else 0)
        unmatched = len(found) - matched if found is not None else 0
        days = len(set(zip(group.channel_id, group.prediction_time.dt.date)))
        result[key] = {
            "admitted_hours": len(group),
            "positive_hours": int(y.sum()),
            "positive_episodes": int(group.loc[group.target.eq(1),
                                           "target_episode_id"].nunique()),
            "channel_days": days,
            "hourly_pr_auc": (float(average_precision_score(y, group.rule_score))
                              if len(set(y)) == 2 else None),
            "matched_episodes": matched,
            "unmatched_warnings": unmatched,
            "unmatched_per_1000_channel_days": (
                unmatched * 1000 / days if days else None),
        }
    return result


def run(freeze_path: Path, validation_path: Path, qa_impact_path: Path,
        a3_dir: Path, b3_dir: Path, index_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"sealed test output already exists: {output_dir}")
    started = time.perf_counter()
    freeze = read_json(freeze_path)
    validation = read_json(validation_path)
    if (freeze["schema_version"] != "r6-frozen-conditional-journal-rule-v1"
            or freeze["status"] != "frozen_before_single_sealed_test"
            or freeze["model_version"] != RULE_VERSION
            or freeze["feature_terms"] != TERMS
            or freeze["excluded_year"] != 2021
            or freeze["test_scores_or_labels_seen_before_freeze"]
            or validation["status"] != "passed_before_sealed_test"
            or validation["source_freeze_sha256"] != sha256(freeze_path)
            or freeze["source_r6_qa_impact_report_sha256"]
            != sha256(qa_impact_path)
            or read_json(qa_impact_path)["r4_rule_score_changed_rows_all_corrected"]):
        raise ValueError("test attempted without an accepted frozen decision")
    contract, _, pairs = verify_inputs(
        Path("ml/r3_conditional_training_contract_v1.json"),
        Path("ml/r3_discrete_feature_allowlist_v1.json"),
        a3_dir, b3_dir, index_dir)
    if (sha256(a3_dir / "manifest.json") != freeze["source_a3_manifest_sha256"]
            or sha256(index_dir / "manifest.json")
            != freeze["source_r3_admission_manifest_sha256"]
            or sha256(Path("ml/r3_conditional_training_contract_v1.json"))
            != freeze["source_r3_contract_sha256"]):
        raise ValueError("frozen source package differs")
    months = [(a, i) for a, i in pairs if a["month"].startswith("2026-")]
    if [a["month"] for a, _ in months] != [f"2026-{month:02d}" for month in range(1, 7)]:
        raise ValueError("sealed test months differ from the freeze")
    b3_manifest = read_json(b3_dir / "manifest.json")
    b3_chunks = {item["month"]: item for item in b3_manifest["chunks"]}
    raw_test_hours = sum(b3_chunks[a["month"]]["rows"] for a, _ in months)
    full_status: dict[str, Counter[str]] = defaultdict(Counter)
    for a, _ in months:
        item = b3_chunks[a["month"]]
        detail = read_json(b3_dir / item["manifest_file"])
        for name, counts in detail["status_counts"].items():
            full_status[name].update(counts)
    output_dir.mkdir(parents=True)
    parts = []
    monthly_files = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        for a, i in months:
            month = a["month"]
            candidate = ((index_dir / i["manifest_file"]).parent /
                         "conditional_discrete_keys.parquet")
            features = a3_dir / a["features_file"]
            frame = score_month(database, candidate, features,
                                freeze["frozen_threshold"])
            if len(frame) != i["rows"]:
                raise ValueError(f"sealed test row count differs: {month}")
            path = output_dir / f"predictions_{month}.parquet"
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False),
                           path, compression="zstd")
            parts.append(frame)
            monthly_files.append({"month": month, "file": path.name,
                                  "rows": len(frame), "sha256": sha256(path)})
            print(f"scored {month}: {len(frame):,} admitted hours", flush=True)
    full = pd.concat(parts, ignore_index=True)
    del parts
    channel_days = len(set(zip(full.channel_id, full.prediction_time.dt.date)))
    metrics, alerts = evaluate_alerts(
        full, "rule_score", freeze["frozen_threshold"], channel_days=channel_days)
    alert_path = output_dir / "emitted_alerts.parquet"
    pq.write_table(pa.Table.from_pandas(alerts, preserve_index=False),
                   alert_path, compression="zstd")
    y = full.target.to_numpy(dtype=np.int8)
    by_month = {}
    for month, group in full.groupby(full.prediction_time.dt.strftime("%Y-%m")):
        positive = int(group.target.sum())
        matched = alerts.loc[
            alerts.prediction_time.dt.strftime("%Y-%m").eq(month)
            & alerts.outcome.eq("matched_episode")]
        unmatched = alerts.loc[
            alerts.prediction_time.dt.strftime("%Y-%m").eq(month)
            & ~alerts.outcome.eq("matched_episode")]
        by_month[month] = {"admitted_hours": len(group),
                           "positive_hours": positive,
                           "positive_episodes": int(group.loc[
                               group.target.eq(1), "target_episode_id"].nunique()),
                           "matched_warnings_emitted": len(matched),
                           "unmatched_warnings_emitted": len(unmatched)}
    report = {
        "schema_version": "r6-b-single-sealed-test-v1",
        "status": "single_frozen_test_evaluated",
        "source_freeze_sha256": sha256(freeze_path),
        "source_validation_freeze_report_sha256": sha256(validation_path),
        "source_r3_contract_sha256": sha256(
            Path("ml/r3_conditional_training_contract_v1.json")),
        "source_b3_manifest_sha256": sha256(b3_dir / "manifest.json"),
        "target": contract["prediction_target"],
        "physical_failure_claim": False,
        "score_is_calibrated_probability": False,
        "threshold_is_product_approved": False,
        "test_period": "2026-01_to_2026-06",
        "raw_hourly_population": raw_test_hours,
        "full_population_status_counts": {
            name: dict(sorted(counts.items())) for name, counts in full_status.items()},
        "conditionally_admitted_hours": len(full),
        "conditional_admission_rate": len(full) / raw_test_hours,
        "conditionally_admitted_positive_hours": int(y.sum()),
        "conditionally_admitted_positive_episodes": int(full.loc[
            full.target.eq(1), "target_episode_id"].nunique()),
        "conditionally_admitted_channel_days": channel_days,
        "conditionally_admitted_channels": int(full.channel_id.nunique()),
        "hourly_pr_auc": (float(average_precision_score(y, full.rule_score))
                          if len(set(y)) == 2 else None),
        "alerts": metrics,
        "by_month": by_month,
        "by_sensor_type": _by_type(full, alerts),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "This is one fixed-rule evaluation on a conditionally admitted test subset.",
            "The 2-unmatched-warning validation budget is exploratory, not product-approved.",
            "The score is not a calibrated probability of physical sensor failure.",
            "Unknown and excluded rows are unavailable, not zero-risk predictions.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": report["schema_version"],
        "status": report["status"],
        "source_freeze_sha256": report["source_freeze_sha256"],
        "report_sha256": sha256(report_path),
        "emitted_alerts_sha256": sha256(alert_path),
        "monthly_predictions": monthly_files,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--qa-impact", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.freeze, args.validation_audit, args.qa_impact,
                 args.a3_dir, args.b3_dir, args.index_dir, args.output_dir)
    print(json.dumps({"alerts": report["alerts"],
                      "hourly_pr_auc": report["hourly_pr_auc"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
