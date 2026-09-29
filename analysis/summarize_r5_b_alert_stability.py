"""Compare R5 B and accepted R4 rule warnings by month and sensor type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.alert_eval import evaluate_alerts


BASE = ["channel_id", "prediction_time", "sensor_type", "target",
        "target_episode_id", "label_available_at"]


def _counts(alerts: pd.DataFrame, column: str) -> dict:
    result = {}
    for key, part in alerts.groupby(column, dropna=False):
        matched = int(part.outcome.eq("matched_episode").sum())
        result[str(key)] = {
            "matched_episodes": matched,
            "unmatched_warnings": len(part) - matched,
        }
    return result


def _summarize(frame: pd.DataFrame, score: str, threshold: float,
               channel_days: int) -> tuple[dict, set[str]]:
    metrics, emitted = evaluate_alerts(frame, score, threshold,
                                        channel_days=channel_days)
    emitted["warning_month"] = emitted.prediction_time.dt.strftime("%Y-%m")
    matched = emitted.loc[emitted.outcome.eq("matched_episode"),
                          "target_episode_id"]
    return {
        "threshold": threshold,
        "matched_episodes": metrics["matched_episodes"],
        "unmatched_warnings": metrics["unmatched_warnings"],
        "by_sensor_type": _counts(emitted, "sensor_type"),
        "by_warning_month": _counts(emitted, "warning_month"),
    }, set(matched)


def run(r5_dir: Path, r4_audit_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = read_json(r5_dir / "manifest.json")
    report = read_json(r5_dir / "report.json")
    if (manifest["schema_version"] != "r5-b-fixed-population-ablation-v1"
            or manifest["report_sha256"] != sha256(r5_dir / "report.json")
            or report["validation_rows"] != 1_365_077):
        raise ValueError("R5 B report lineage differs")
    months = manifest["monthly_predictions"]
    if len(months) != 12 or any(not item["month"].startswith("2025-")
                                for item in months):
        raise ValueError("R5 validation months differ")
    for item in months:
        if sha256(r5_dir / item["file"]) != item["sha256"]:
            raise ValueError("R5 validation prediction hash differs")
    variants = report["variant_features"]
    score_names = [f"score_{name}" for name in variants]
    r5 = pd.concat((pd.read_parquet(r5_dir / item["file"],
                                  columns=[*BASE, *score_names])
                    for item in months), ignore_index=True)
    if (len(r5) != report["validation_rows"]
            or int(r5.target.sum()) != report["validation_positive_hours"]):
        raise ValueError("R5 validation population differs")
    stability = {}
    matched_sets = {}
    for name in variants:
        score = f"score_{name}"
        threshold = report["alerts_at_budget"][name]["threshold"]
        summary, matches = _summarize(
            r5.rename(columns={score: "catboost_score"}),
            "catboost_score", threshold, report["validation_channel_days"])
        if (summary["matched_episodes"]
                != report["alerts_at_budget"][name]["matched_episodes"]
                or summary["unmatched_warnings"]
                != report["alerts_at_budget"][name]["unmatched_warnings"]):
            raise ValueError(f"R5 warning count differs: {name}")
        stability[name] = summary
        matched_sets[name] = matches
    del r5
    r4_manifest = read_json(r4_audit_dir / "manifest.json")
    if (sha256(r4_audit_dir / "manifest.json")
            != report["source_r4_audit_manifest_sha256"]):
        raise ValueError("R4 audit lineage differs")
    r4_months = [item for item in r4_manifest["files"]
                 if item["file"].startswith("validation_")]
    if len(r4_months) != 12:
        raise ValueError("R4 validation months differ")
    for item in r4_months:
        if sha256(r4_audit_dir / item["file"]) != item["sha256"]:
            raise ValueError("R4 validation prediction hash differs")
    rule = pd.concat((pd.read_parquet(r4_audit_dir / item["file"],
                                    columns=[*BASE, "rule_score"])
                      for item in r4_months), ignore_index=True)
    if (len(rule) != report["validation_rows"]
            or int(rule.target.sum()) != report["validation_positive_hours"]):
        raise ValueError("R4 rule validation population differs")
    threshold = report["r4_rule_reference_at_budget"]["threshold"]
    rule_summary, rule_matches = _summarize(
        rule, "rule_score", threshold, report["validation_channel_days"])
    if (rule_summary["matched_episodes"]
            != report["r4_rule_reference_at_budget"]["matched_episodes"]
            or rule_summary["unmatched_warnings"]
            != report["r4_rule_reference_at_budget"]["unmatched_warnings"]):
        raise ValueError("accepted R4 rule warning count differs")
    stability["r4_rule"] = rule_summary
    result = {
        "schema_version": "r5-b-alert-stability-v1",
        "status": "validation_only",
        "source_r5_b_manifest_sha256": sha256(r5_dir / "manifest.json"),
        "source_r4_audit_manifest_sha256": sha256(r4_audit_dir / "manifest.json"),
        "warning_cooldown_hours": 24,
        "methods": stability,
        "matched_episode_overlap_with_r4_rule": {
            name: {"shared": len(matches & rule_matches),
                   "new": len(matches - rule_matches),
                   "lost": len(rule_matches - matches)}
            for name, matches in matched_sets.items()
        },
    }
    output_dir.mkdir(parents=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": result["schema_version"],
        "status": result["status"],
        "source_r5_b_manifest_sha256": result["source_r5_b_manifest_sha256"],
        "report_sha256": sha256(report_path),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5-dir", type=Path, required=True)
    parser.add_argument("--r4-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.r5_dir, args.r4_audit_dir, args.output_dir)
    print(json.dumps({"overlap": result["matched_episode_overlap_with_r4_rule"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
