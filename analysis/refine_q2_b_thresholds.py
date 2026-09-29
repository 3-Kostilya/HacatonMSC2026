"""Refine score-only warning thresholds on the saved Q2 validation scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from analysis.run_q2_b_expanded import (ALL_VALIDATION_EPISODES, SCORE_NAMES,
                                        selected_metrics)
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.v2_threshold import choose_threshold


def score_quantiles() -> np.ndarray:
    # Declared before labels are accessed; equally applied to each model.
    return np.unique(np.concatenate((
        np.array([0.80, 0.90]),
        np.linspace(0.95, 0.995, 16),
        np.linspace(0.995, 0.9999, 32),
        np.linspace(0.9999, 0.999999, 20),
    )))


def run(*, experiment: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    original = read_json(experiment / "report.json")
    manifest = read_json(experiment / "manifest.json")
    if (original["schema_version"] != "q2-b-expanded-validation-v1"
            or manifest["report_sha256"] != sha256(experiment / "report.json")
            or original["validation"]["all_assigned_episodes"] != ALL_VALIDATION_EPISODES):
        raise ValueError("Q2 score source differs")
    files = [experiment / item["name"] for item in original["score_files"]]
    if any(sha256(path) != item["sha256"] for path, item in
           zip(files, original["score_files"], strict=True)):
        raise ValueError("saved validation scores differ")
    columns = ["channel_id", "prediction_time", "sensor_type", "target",
               "target_episode_id", "label_available_at",
               *[f"score_{name}" for name in SCORE_NAMES]]
    frame = pq.read_table(files, columns=columns).to_pandas()
    if (len(frame) != original["validation"]["rows"]
            or int(frame.target.sum()) != original["validation"]["positive_hours"]):
        raise ValueError("saved validation population differs")
    days = original["validation"]["eligible_channel_days_with_binary_label"]
    curves = {}
    goals = {}
    by_type = {}
    for name in SCORE_NAMES:
        column = f"score_{name}"
        thresholds = sorted({float(x) for x in np.quantile(
            frame[column].to_numpy(), score_quantiles())})
        curve = []
        for threshold in thresholds:
            metric, _ = selected_metrics(frame, column, threshold, days)
            curve.append(metric)
        curves[name] = curve
        goals[name] = choose_threshold(curve,
                                       full_positive_episodes=ALL_VALIDATION_EPISODES)
        choice = goals[name]["selected"] or goals[name]["diagnostic_best_full_f1"]
        _, alerts = selected_metrics(frame, column, choice["threshold"], days)
        by_type[name] = {
            str(kind): {"warnings": len(group),
                        "matched": int((group.outcome == "matched_episode").sum())}
            for kind, group in alerts.groupby("sensor_type", dropna=False)
        }
        print(f"{name}: {len(curve)} thresholds, diagnostic={choice}", flush=True)
    output.mkdir(parents=True)
    report = {
        "schema_version": "q2-b-refined-validation-thresholds-v1",
        "source_experiment_manifest_sha256": sha256(experiment / "manifest.json"),
        "quantiles": score_quantiles().tolist(),
        "full_positive_episodes": ALL_VALIDATION_EPISODES,
        "goal_checks": goals,
        "by_sensor_type_at_selected_or_diagnostic": by_type,
        "test_data_read": False,
        "limitation": "Validation selection; no independent quality confirmation.",
    }
    (output / "curves.json").write_text(json.dumps(curves, ensure_ascii=False, indent=2)
                                        + "\n", encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)
                                        + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": report["schema_version"],
        "source_experiment_manifest_sha256": report["source_experiment_manifest_sha256"],
        "report_sha256": sha256(output / "report.json"),
        "curves_sha256": sha256(output / "curves.json"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
