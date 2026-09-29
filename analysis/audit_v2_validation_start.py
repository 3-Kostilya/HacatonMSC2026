"""Audit the accepted train/validation starting point for a new experiment.

This reads only saved validation curves and aggregate validation label counts.
It never opens the previously evaluated test predictions or labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.train_r4_discrete_baselines import sha256
from ml.forecast.v2_threshold import choose_threshold


PRECISION_GOAL = 0.7
RECALL_GOAL = 0.5
EXPECTED_VALIDATION_ROWS = 1_365_077
EXPECTED_CONDITIONAL_EPISODES = 261


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def curve_package(directory: Path, pinned_manifest_hash: str) -> dict:
    manifest_path = directory / "manifest.json"
    if sha256(manifest_path) != pinned_manifest_hash:
        raise ValueError(f"validation curve package is not the accepted version: {directory}")
    manifest = read_json(manifest_path)
    if (sha256(directory / "curves.json") != manifest["curves_sha256"]
            or sha256(directory / "report.json") != manifest["report_sha256"]):
        raise ValueError(f"validation curve package integrity differs: {directory}")
    return read_json(directory / "curves.json")


def summarize(curve: list[dict], all_episodes: int) -> dict:
    if not curve or any(item["eligible_positive_episodes"]
                        != EXPECTED_CONDITIONAL_EPISODES for item in curve):
        raise ValueError("validation episode denominator differs")
    selected = choose_threshold(curve, full_positive_episodes=all_episodes,
                                precision_goal=PRECISION_GOAL, recall_goal=RECALL_GOAL)
    return {
        "thresholds_examined": selected["thresholds_examined"],
        "requirements_feasible_on_checked_grid": (
            selected["requirements_feasible_on_checked_grid"]),
        "selected_threshold_if_feasible": (
            selected["selected"]["threshold"] if selected["selected"] else None),
        "diagnostic_best_full_f1_on_checked_grid": selected["diagnostic_best_full_f1"],
    }


def run(*, r3_dir: Path, r4_audit_dir: Path, r4_dir: Path, r5_dir: Path,
        decision_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    decision = read_json(decision_path)
    r4 = curve_package(r4_dir, decision["source_r4_budget_manifest_sha256"])
    r5 = curve_package(r5_dir, decision["source_r5_b_ablation_manifest_sha256"])
    r4_curve_manifest = read_json(r4_dir / "manifest.json")
    r4_audit_manifest = read_json(r4_audit_dir / "manifest.json")
    r4_audit = read_json(r4_audit_dir / "report.json")
    if (sha256(r4_audit_dir / "manifest.json")
            != r4_curve_manifest["source_validation_audit_manifest_sha256"]
            or sha256(r4_audit_dir / "report.json")
            != r4_audit_manifest["report_sha256"]
            or r4_audit["validation_rows"] != EXPECTED_VALIDATION_ROWS):
        raise ValueError("accepted R4 validation audit differs")
    r3_manifest_path = r3_dir / "manifest.json"
    r3 = read_json(r3_dir / "report.json")
    r3_manifest = read_json(r3_manifest_path)
    if (sha256(r3_dir / "report.json") != r3_manifest["report_sha256"]
            or r3["status"] != "full_registered_label_audit_not_training_ready"):
        raise ValueError("R3 full-population validation counts are not pinned")
    all_episodes = r3["assigned_unique_positive_episodes_by_split"]["validation"]
    if (all_episodes < EXPECTED_CONDITIONAL_EPISODES
            or decision["validation_rows"] != EXPECTED_VALIDATION_ROWS
            or decision["validation_positive_episodes"]
            != EXPECTED_CONDITIONAL_EPISODES):
        raise ValueError("accepted validation population differs")
    models = {f"r4_{name}": summarize(curve, all_episodes)
              for name, curve in r4.items()}
    models.update({f"r5_{name}": summarize(curve, all_episodes)
                   for name, curve in r5.items()})
    full_by_type = r3["assigned_unique_positive_episodes_by_split_and_type"]["validation"]
    eligible_by_type = {name: item["positive_episodes"]
                        for name, item in r4_audit["by_sensor_type"].items()}
    if (sum(full_by_type.values()) != all_episodes
            or sum(eligible_by_type.values()) != EXPECTED_CONDITIONAL_EPISODES
            or any(eligible_by_type.get(name, 0) > count
                   for name, count in full_by_type.items())):
        raise ValueError("validation type episode counts differ")
    by_type = {name: {
        "full_assigned_positive_episodes": full_by_type.get(name, 0),
        "conditionally_eligible_positive_episodes": eligible_by_type.get(name, 0),
        "maximum_full_recall_at_fixed_admission": (
            eligible_by_type.get(name, 0) / full_by_type[name]
            if full_by_type.get(name, 0) else None),
    } for name in sorted(set(full_by_type) | set(eligible_by_type))}
    report = {
        "schema_version": "ml-v2-validation-start-audit-v1",
        "status": "baseline_validation_evidence_not_new_model_selection",
        "r3_full_manifest_sha256": sha256(r3_manifest_path),
        "r4_curves_manifest_sha256": sha256(r4_dir / "manifest.json"),
        "r4_validation_audit_manifest_sha256": sha256(r4_audit_dir / "manifest.json"),
        "r5_curves_manifest_sha256": sha256(r5_dir / "manifest.json"),
        "validation_rows": EXPECTED_VALIDATION_ROWS,
        "conditionally_eligible_positive_episodes": EXPECTED_CONDITIONAL_EPISODES,
        "full_assigned_positive_episodes": all_episodes,
        "maximum_full_recall_at_fixed_admission": (
            EXPECTED_CONDITIONAL_EPISODES / all_episodes),
        "precision_goal_strictly_greater_than": PRECISION_GOAL,
        "recall_goal_strictly_greater_than": RECALL_GOAL,
        "models": models,
        "by_sensor_type": by_type,
        "any_checked_threshold_meets_both": any(
            value["requirements_feasible_on_checked_grid"] for value in models.values()),
        "test_scores_or_labels_read": False,
        "limitations": [
            "Previously saved quantile grids do not prove a global optimum over all scores.",
            "Validation has been used for prior R4/R5 development and is not independent confirmation.",
            "New features and admission changes require a new version and common-population rerun.",
            "Labels are future registered journal messages, not verified physical failures.",
        ],
    }
    output_dir.mkdir(parents=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": report["schema_version"],
        "report_sha256": sha256(report_path),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-dir", type=Path, required=True)
    parser.add_argument("--r4-audit-dir", type=Path, required=True)
    parser.add_argument("--r4-dir", type=Path, required=True)
    parser.add_argument("--r5-dir", type=Path, required=True)
    parser.add_argument("--decision", type=Path,
                        default=Path("ml/r5_b_ablation_decision_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(r3_dir=args.r3_dir, r4_audit_dir=args.r4_audit_dir,
                 r4_dir=args.r4_dir, r5_dir=args.r5_dir,
                 decision_path=args.decision, output_dir=args.output_dir)
    print(json.dumps({
        "validation_rows": report["validation_rows"],
        "conditional_episodes": report["conditionally_eligible_positive_episodes"],
        "full_assigned_episodes": report["full_assigned_positive_episodes"],
        "any_checked_threshold_meets_both": report["any_checked_threshold_meets_both"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
