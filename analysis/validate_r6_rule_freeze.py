"""Reproduce the frozen R4 rule on every validation hour before opening test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from analysis.train_r4_discrete_baselines import read_json, sha256
from analysis.r6_provenance import frozen_rule_sha256
from ml.forecast.alert_eval import evaluate_alerts
from ml.forecast.r6_rule import RULE_VERSION, TERMS, predict_rule


KEYS = ["channel_id", "prediction_time"]
COMPARE = ["sensor_type", "target", "target_episode_id", "label_available_at"]


def _different(left: pd.Series, right: pd.Series) -> bool:
    return bool((~(left.eq(right).fillna(False) |
                    (left.isna() & right.isna()))).any())


def validate(freeze_path: Path, a3_dir: Path, index_dir: Path,
             r4_audit_dir: Path, qa_impact_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    freeze = read_json(freeze_path)
    source = (
        (a3_dir / "manifest.json", "source_a3_manifest_sha256"),
        (index_dir / "manifest.json", "source_r3_admission_manifest_sha256"),
        (r4_audit_dir / "manifest.json", "source_r4_validation_audit_manifest_sha256"),
        (qa_impact_path, "source_r6_qa_impact_report_sha256"),
    )
    if (freeze["schema_version"] != "r6-frozen-conditional-journal-rule-v1"
            or freeze["model_version"] != RULE_VERSION
            or freeze["feature_terms"] != TERMS
            or freeze["excluded_year"] != 2021
            or freeze["final_product_threshold_approved"]):
        raise ValueError("R6 freeze differs from accepted rule")
    for path, field in source:
        if sha256(path) != freeze[field]:
            raise ValueError(f"frozen source hash differs: {path}")
    qa = read_json(qa_impact_path)
    if (qa["r4_rule_changed_fields"]
            or qa["r4_rule_score_changed_rows_all_corrected"]
            or qa["qa_count_features_training_ready"]):
        raise ValueError("QA decision does not preserve the accepted rule")
    a_chunks = {item["month"]: item for item in read_json(
        a3_dir / "manifest.json")["chunks"]}
    i_chunks = {item["month"]: item for item in read_json(
        index_dir / "manifest.json")["chunks"]}
    r4_manifest = read_json(r4_audit_dir / "manifest.json")
    months = sorted((item for item in r4_manifest["files"]
                     if item["file"].startswith("validation_")
                     and item["month"].startswith("2025-")),
                    key=lambda item: item["month"])
    if len(months) != 12:
        raise ValueError("expected 12 validation months")
    frames = []
    monthly = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        for item in months:
            month = item["month"]
            candidate = ((index_dir / i_chunks[month]["manifest_file"]).parent /
                         "conditional_discrete_keys.parquet")
            features = a3_dir / a_chunks[month]["features_file"]
            projection = ", ".join(f'f."{name}"' for name in TERMS)
            joined = database.execute(
                f"""SELECT c.channel_id,c.prediction_time,c.sensor_type,
                           c.target,c.target_episode_id,c.label_available_at,
                           {projection}
                    FROM read_parquet(?) c
                    JOIN read_parquet(?) f USING (channel_id,prediction_time)
                    WHERE c.split='validation'""",
                [str(candidate), str(features)],
            ).fetch_df()
            path = r4_audit_dir / item["file"]
            if sha256(path) != item["sha256"]:
                raise ValueError(f"R4 validation prediction hash differs: {month}")
            accepted = pd.read_parquet(
                path, columns=[*KEYS, *COMPARE, "rule_score"])
            if (len(joined) != len(accepted) or len(joined) != item["rows"]
                    or joined.duplicated(KEYS).any() or accepted.duplicated(KEYS).any()):
                raise ValueError(f"validation keys differ: {month}")
            replay = predict_rule(
                joined, eligibility_status="eligible",
                threshold=freeze["frozen_threshold"])
            joined["rule_score"] = replay.rule_score.to_numpy()
            comparison = joined[[*KEYS, *COMPARE, "rule_score"]].merge(
                accepted, on=KEYS, how="outer", validate="one_to_one",
                suffixes=("_new", "_r4"), indicator=True)
            if not comparison._merge.eq("both").all():
                raise ValueError(f"validation keys differ: {month}")
            for field in COMPARE:
                if _different(comparison[f"{field}_new"],
                              comparison[f"{field}_r4"]):
                    raise ValueError(f"validation {field} differs: {month}")
            delta = np.abs(comparison.rule_score_new - comparison.rule_score_r4)
            if not np.isfinite(delta).all() or float(delta.max()) != 0:
                raise ValueError(f"frozen score differs from R4: {month}")
            frames.append(joined[[*KEYS, *COMPARE, "rule_score"]])
            monthly.append({"month": month, "rows": len(joined),
                            "positive_hours": int(joined.target.sum()),
                            "maximum_score_difference": 0.0})
    full = pd.concat(frames, ignore_index=True)
    channel_days = len(set(zip(full.channel_id, full.prediction_time.dt.date)))
    if (len(full) != freeze["validation_rows"]
            or int(full.target.sum()) != freeze["validation_positive_hours"]
            or channel_days != 117_266):
        raise ValueError("validation population differs from R4")
    metrics, alerts = evaluate_alerts(
        full, "rule_score", freeze["frozen_threshold"], channel_days=channel_days)
    if (metrics["matched_episodes"] != freeze["validation_matched_episodes"]
            or metrics["unmatched_warnings"] != freeze["validation_unmatched_warnings"]
            or metrics["eligible_positive_episodes"]
            != freeze["validation_positive_episodes"]):
        raise ValueError("frozen validation warning metrics differ")
    by_type = {}
    for sensor_type, group in alerts.groupby("sensor_type", dropna=False):
        matched = int(group.outcome.eq("matched_episode").sum())
        by_type[str(sensor_type)] = {
            "matched_episodes": matched,
            "unmatched_warnings": len(group) - matched,
        }
    report = {
        "schema_version": "r6-b-validation-freeze-audit-v1",
        "status": "passed_before_sealed_test",
        "source_freeze_sha256": frozen_rule_sha256(freeze_path),
        "source_r4_audit_manifest_sha256": sha256(r4_audit_dir / "manifest.json"),
        "validation_rows": len(full),
        "validation_positive_hours": int(full.target.sum()),
        "validation_channel_days": channel_days,
        "maximum_score_difference_from_r4": 0.0,
        "monthly": monthly,
        "alerts": metrics,
        "alerts_by_sensor_type": by_type,
        "test_scores_or_labels_read": False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--r4-audit-dir", type=Path, required=True)
    parser.add_argument("--qa-impact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.freeze, args.a3_dir, args.index_dir,
                      args.r4_audit_dir, args.qa_impact, args.output_dir)
    print(json.dumps({"validation_rows": report["validation_rows"],
                      "alerts": report["alerts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
