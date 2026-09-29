"""Summarize R4 validation warning stability by month and sensor type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.audit_r4_b_validation import MODEL_SCORES  # noqa: E402
from analysis.train_r4_discrete_baselines import read_json, sha256  # noqa: E402


def _group(full: pd.DataFrame, truth: pd.DataFrame, alerts: dict[str, pd.DataFrame],
           key: str, value: str) -> dict:
    source = full.loc[full[key] == value]
    group_truth = truth.loc[truth[key] == value]
    days = len(source[["channel_id", "prediction_time"]].assign(
        day=source.prediction_time.dt.date
    )[["channel_id", "day"]].drop_duplicates())
    result = {
        "rows": len(source),
        "positive_hours": int(source.target.sum()),
        "positive_episodes": len(group_truth),
        "channel_days": days,
        "models": {},
    }
    for model, table in alerts.items():
        group_alerts = table.loc[table[key] == value]
        matched = group_alerts.loc[group_alerts.outcome == "matched_episode"]
        unmatched = len(group_alerts) - len(matched)
        matched_truth = table.loc[
            (table.outcome == "matched_episode")
            & table.target_episode_id.isin(group_truth.target_episode_id)
        ]
        result["models"][model] = {
            "emitted_warnings": len(group_alerts),
            "matched_warnings_emitted_in_group": len(matched),
            "unmatched_warnings": unmatched,
            "matched_episodes_with_onset_in_group": len(matched_truth),
            "episode_recall": (
                len(matched_truth) / len(group_truth) if len(group_truth) else None
            ),
            "unmatched_warnings_per_1000_channel_days": (
                unmatched * 1000 / days if days else None
            ),
        }
    return result


def run(*, audit_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"stability output exists: {output_dir}")
    manifest = read_json(audit_dir / "manifest.json")
    report = read_json(audit_dir / "report.json")
    if (manifest["schema_version"] != "r4-b-validation-audit-v1"
            or sha256(audit_dir / "report.json") != manifest["report_sha256"]):
        raise ValueError("R4 validation audit differs")
    files = {item["file"]: item for item in manifest["files"]}
    months = [item for item in manifest["files"]
              if item["file"].startswith("validation_")]
    full_frames = []
    for item in months:
        path = audit_dir / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"validation month hash differs: {item['file']}")
        full_frames.append(pd.read_parquet(path, columns=[
            "channel_id", "prediction_time", "sensor_type", "target",
            "target_episode_id", "label_available_at",
        ]))
    full = pd.concat(full_frames, ignore_index=True)
    full["month"] = full.prediction_time.dt.strftime("%Y-%m")
    if len(full) != report["validation_rows"]:
        raise ValueError("validation row count differs")
    truth = full.loc[full.target == 1, [
        "target_episode_id", "sensor_type", "label_available_at",
    ]].drop_duplicates("target_episode_id")
    truth = truth.assign(month=truth.label_available_at.dt.strftime("%Y-%m"))
    alerts = {}
    for model in MODEL_SCORES:
        filename = f"alerts_{model}.parquet"
        path = audit_dir / filename
        if sha256(path) != files[filename]["sha256"]:
            raise ValueError(f"R4 alert table hash differs: {model}")
        table = pd.read_parquet(path)
        table["month"] = table.prediction_time.dt.strftime("%Y-%m")
        alerts[model] = table
    by_month = {
        month: _group(full, truth, alerts, "month", month)
        for month in sorted(full.month.unique())
    }
    by_type = {
        sensor_type: _group(full, truth, alerts, "sensor_type", sensor_type)
        for sensor_type in sorted(full.sensor_type.unique())
    }
    result = {
        "schema_version": "r4-b-validation-stability-v1",
        "status": "validation_only",
        "source_validation_audit_manifest_sha256": sha256(audit_dir / "manifest.json"),
        "month_count": len(by_month),
        "sensor_type_count": len(by_type),
        "by_month": by_month,
        "by_sensor_type": by_type,
        "attribution": {
            "warning_load_month": "warning_prediction_time",
            "episode_recall_month": "registered_onset_time",
            "warning_load_type": "warning_sensor_type",
            "episode_recall_type": "registered_episode_sensor_type",
        },
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": result["schema_version"],
        "status": result["status"],
        "source_validation_audit_manifest_sha256": (
            result["source_validation_audit_manifest_sha256"]
        ),
        "report_sha256": sha256(output_dir / "report.json"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(audit_dir=args.audit_dir, output_dir=args.output_dir)
    print(json.dumps({"month_count": report["month_count"],
                      "sensor_type_count": report["sensor_type_count"]}))


if __name__ == "__main__":
    main()
