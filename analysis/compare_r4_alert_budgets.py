"""Compare R4 models at common validation alert budgets after cooldown.

Threshold choices are exploratory validation decisions, never test estimates.
The fixed budget grid is a comparison device, not a product alarm policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.audit_r4_b_validation import MODEL_SCORES  # noqa: E402
from analysis.train_r4_discrete_baselines import read_json, sha256  # noqa: E402
from ml.forecast.alert_eval import evaluate_alerts  # noqa: E402


BUDGETS = (1.0, 2.0, 4.0, 6.0)
SOURCE_COLUMNS = (
    "channel_id", "prediction_time", "sensor_type", "target",
    "target_episode_id", "label_available_at", *MODEL_SCORES.values(),
)


def thresholds(scores: np.ndarray, model: str, initial_threshold: float) -> list[float]:
    tail = float(np.quantile(scores, 0.99))
    if model == "rule":
        choices = np.unique(scores[scores >= tail])
    else:
        q = np.concatenate([
            np.linspace(0.99, 0.99995, 160),
            1 - np.geomspace(5e-5, 1e-6, 20),
        ])
        choices = np.quantile(scores, q)
    return sorted({float(value) for value in choices} | {float(initial_threshold)})


def select_under_budget(curve: list[dict], budget: float) -> dict | None:
    eligible = [item for item in curve
                if item["unmatched_warnings_per_1000_channel_days"] <= budget]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (
        item["matched_episodes"], -item["unmatched_warnings"],
        item["episode_precision"], item["threshold"],
    ))


def run(*, audit_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"comparison output already exists: {output_dir}")
    audit_manifest = read_json(audit_dir / "manifest.json")
    audit_report = read_json(audit_dir / "report.json")
    if (audit_manifest["schema_version"] != "r4-b-validation-audit-v1"
            or sha256(audit_dir / "report.json") != audit_manifest["report_sha256"]
            or audit_report["status"] != "validation_only_b_review"):
        raise ValueError("R4 validation audit version or hash differs")
    months = [item for item in audit_manifest["files"]
              if item["file"].startswith("validation_")]
    if (len(months) != 12 or any(item["month"][:4] != "2025" for item in months)):
        raise ValueError("expected 12 sealed validation months")
    frames = []
    for item in months:
        path = audit_dir / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"R4 prediction month hash differs: {item['month']}")
        frame = pd.read_parquet(path, columns=list(SOURCE_COLUMNS))
        if len(frame) != item["rows"]:
            raise ValueError(f"R4 prediction month row count differs: {item['month']}")
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True)
    if (len(full) != audit_report["validation_rows"]
            or int(full.target.sum()) != audit_report["validation_positive_hours"]):
        raise ValueError("validation row population differs")
    channel_days = audit_report["validation_channel_days"]
    curves = {}
    for model, score in MODEL_SCORES.items():
        initial = audit_report["models_at_hour_f1_threshold"][model]["threshold"]
        candidates = thresholds(full[score].to_numpy(), model, initial)
        curve = []
        for index, threshold in enumerate(candidates):
            result, _ = evaluate_alerts(full, score, threshold, channel_days=channel_days)
            curve.append(result)
            if index % 20 == 0:
                print(json.dumps({"model": model, "thresholds_done": index + 1,
                                  "thresholds_total": len(candidates)}), flush=True)
        curves[model] = curve
    comparison = {}
    for budget in BUDGETS:
        comparison[str(budget)] = {
            model: select_under_budget(curve, budget)
            for model, curve in curves.items()
        }
    best_event_f1 = {
        model: max(curve, key=lambda item: (
            item["episode_f1"], item["episode_recall"],
            -item["unmatched_warnings"],
        )) for model, curve in curves.items()
    }
    report = {
        "schema_version": "r4-b-alert-budget-comparison-v1",
        "status": "validation_only_model_selection_evidence",
        "source_validation_audit_manifest_sha256": sha256(audit_dir / "manifest.json"),
        "cooldown_hours": 24,
        "validation_rows": len(full),
        "validation_channel_days": channel_days,
        "budget_grid_unmatched_warnings_per_1000_channel_days": list(BUDGETS),
        "threshold_candidates_by_model": {model: len(curve) for model, curve in curves.items()},
        "best_event_f1": best_event_f1,
        "at_common_budgets": comparison,
        "limitations": [
            "All thresholds and method comparisons were selected on validation.",
            "The budget grid is exploratory and is not an approved operating constraint.",
            "No test labels or scores were opened.",
        ],
    }
    output_dir.mkdir(parents=True)
    (output_dir / "curves.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "schema_version": report["schema_version"],
            "status": report["status"],
            "source_validation_audit_manifest_sha256": (
                report["source_validation_audit_manifest_sha256"]
            ),
            "curves_sha256": sha256(output_dir / "curves.json"),
            "report_sha256": sha256(output_dir / "report.json"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(audit_dir=args.audit_dir, output_dir=args.output_dir)
    print(json.dumps({"best_event_f1": report["best_event_f1"],
                      "at_common_budgets": report["at_common_budgets"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
