"""Independent production replay of frozen research policies; no tuning here."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.ml_experiment_eval import EVALUATION_VERSION
from ml.forecast.alert_eval import evaluate_alerts


def replay(db, source: Path, column: str, threshold: float, total: int) -> dict:
    schema = pq.ParquetFile(source).schema_arrow
    # Reproduce native scalar pandas precision BEFORE filtering in SQL.
    native_threshold = float(np.float32(threshold)) if pa.types.is_float32(schema.field(column).type) else threshold
    count, days = db.execute("""SELECT COUNT(DISTINCT CASE WHEN target=1
        THEN target_episode_id END),COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE)))
        FROM read_parquet(?)""", [str(source)]).fetchone()
    columns = ["channel_id", "prediction_time", "sensor_type", "target",
               "target_episode_id", "label_available_at"]
    projection = ",".join(f'"{name}"' for name in columns)
    frame = db.execute(f'SELECT {projection},"{column}" AS catboost_score '
                       f'FROM read_parquet(?) WHERE "{column}">=?',
                       [str(source), native_threshold]).fetch_df()
    metric, alerts = evaluate_alerts(frame, "catboost_score", threshold,
                                     channel_days=max(days, 1))
    p = metric["episode_precision"]
    r = metric["matched_episodes"] / total
    metric.update({"eligible_positive_episodes": count,
                   "available_episode_recall": metric["matched_episodes"] / count if count else 0,
                   "episode_recall": r, "full_episode_recall": r,
                   "full_episode_f1": 2*p*r/(p+r) if p+r else 0,
                   "episode_f1": 2*p*r/(p+r) if p+r else 0,
                   "matched_episode_ids": sorted(alerts.loc[
                       alerts.outcome.eq("matched_episode"), "target_episode_id"].tolist()),
                   "full_episode_count": total})
    return metric


def run(directory: Path, data: Path, mode: str) -> dict:
    destination = directory / "metric_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if mode == "pooled":
        cases = [(name, fold, directory / f"scores_{fold}.parquet", f"score_{name}",
                  entry["tune"]["selected"]["threshold"],
                  entry["tune"]["selected"] if fold == "tune" else entry["validation_frozen"])
                 for name, entry in report["experiments"].items()
                 for fold in ["tune", "validation"]]
    elif mode == "linear":
        cases = [(name, fold, directory / f"{fold}_scores.parquet", f"score_{name}",
                  choice["threshold"], choice if fold == "tune" else report["validation"][name])
                 for name, choice in report["tune"].items()
                 for fold in ["tune", "validation"]]
    elif mode == "specialists":
        cases = [(name, "validation", directory / "validation_scores.parquet", f"score_{name}",
                  0.0, metric) for name, metric in report["validation"].items()]
    else:
        raise ValueError(mode)
    results = []
    keys = ["emitted_warnings", "matched_episodes", "unmatched_warnings",
            "suppressed_positive_score_rows", "duplicate_episode_warnings",
            "episode_precision", "full_episode_recall", "full_episode_f1", "median_lead_hours"]
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        for name, fold, source, column, threshold, expected in cases:
            actual = replay(db, source, column, threshold, manifest["full_episode_count"][fold])
            for key in keys:
                if key in expected and actual[key] != expected[key]:
                    raise AssertionError(f"{mode}/{name}/{fold}: {key} actual={actual[key]} expected={expected[key]}")
            if "matched_episode_ids" in expected and set(actual["matched_episode_ids"]) != set(expected["matched_episode_ids"]):
                raise AssertionError(f"{mode}/{name}/{fold}: matched IDs differ")
            results.append({"variant": name, "fold": fold, "threshold": threshold,
                            "matched_episodes": actual["matched_episodes"],
                            "emitted_warnings": actual["emitted_warnings"],
                            "full_episode_count": actual["full_episode_count"],
                            "expected_evaluation_version": expected.get("evaluation_version")})
            print(f"production parity passed {mode}/{name}/{fold}", flush=True)
    result = {"status": "canonical_production_warning_parity_passed",
              "evaluation_version": EVALUATION_VERSION, "mode": mode,
              "evaluation_source_sha256": hashlib.sha256(Path(
                  "analysis/ml_experiment_eval.py").read_bytes()).hexdigest(),
              "production_source_sha256": hashlib.sha256(Path(
                  "ml/forecast/alert_eval.py").read_bytes()).hexdigest(),
              "policies_checked": results, "selection_changed": False,
              "test_2026_read": False, "data_2021_read": False}
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("output/ml-experiment/data"))
    parser.add_argument("--mode", choices=["pooled", "linear", "specialists"], required=True)
    args = parser.parse_args()
    run(args.directory, args.data, args.mode)
