"""Audit the diagnostic Q2 threshold across validation months."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
from sklearn.metrics import average_precision_score

from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.alert_eval import evaluate_alerts


def run(*, experiment: Path, decision: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source = read_json(experiment / "report.json")
    final = read_json(decision)
    if (source["schema_version"] != "q2-b-expanded-validation-v1"
            or final["source_experiment_manifest_sha256"]
            != sha256(experiment / "manifest.json")):
        raise ValueError("Q2 source or threshold decision differs")
    choice = final["models"]["linear121"]["goal_check"]["diagnostic_best_full_f1"]
    threshold = choice["threshold"]
    files = [experiment / item["name"] for item in source["score_files"]]
    monthly_ap = {}
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        for item, path in zip(source["score_files"], files, strict=True):
            if sha256(path) != item["sha256"]:
                raise ValueError(f"saved score content differs: {path}")
            frame = db.execute("SELECT target, score_linear121 FROM read_parquet(?)",
                               [str(path)]).fetch_df()
            monthly_ap[item["name"][11:18]] = {
                "rows": len(frame),
                "positive_hours": int(frame.target.sum()),
                "hourly_average_precision": float(average_precision_score(
                    frame.target, frame.score_linear121)),
            }
        selected = db.execute("""SELECT channel_id,prediction_time,sensor_type,target,
            target_episode_id,label_available_at,score_linear121 AS catboost_score
            FROM read_parquet(?) WHERE score_linear121>=?""",
            [[str(path) for path in files], threshold]).fetch_df()
    metrics, alerts = evaluate_alerts(
        selected, "catboost_score", threshold,
        channel_days=source["validation"]["eligible_channel_days_with_binary_label"])
    if (metrics["matched_episodes"] != choice["matched_episodes"]
            or metrics["emitted_warnings"] != choice["emitted_warnings"]):
        raise ValueError("monthly replay differs from selected validation decision")
    month = alerts.prediction_time.dt.strftime("%Y-%m")
    grouped = alerts.groupby(month)
    for label, group in grouped:
        monthly_ap[label]["warnings"] = len(group)
        monthly_ap[label]["matched"] = int((group.outcome == "matched_episode").sum())
        monthly_ap[label]["precision"] = (monthly_ap[label]["matched"] / len(group))
    for entry in monthly_ap.values():
        entry.setdefault("warnings", 0)
        entry.setdefault("matched", 0)
        entry.setdefault("precision", None)
    report = {
        "schema_version": "q2-b-monthly-validation-audit-v1",
        "source_experiment_manifest_sha256": sha256(experiment / "manifest.json"),
        "source_threshold_decision_sha256": sha256(decision),
        "model": "linear121",
        "threshold": threshold,
        "monthly": monthly_ap,
        "global_matched": metrics["matched_episodes"],
        "global_warnings": metrics["emitted_warnings"],
        "note": "Month is the warning month; cooldown and episode matching run globally.",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"global_matched": result["global_matched"],
                      "global_warnings": result["global_warnings"]}), flush=True)


if __name__ == "__main__":
    main()
