"""Compare B's frozen R4 validation output with an A-side batch replay.

Parquet byte hashes may differ because DuckDB can return joined rows in a
different order. Compare values by the prediction key, while retaining the
source manifests and the strict lineage checks done by the scoring script.
Sealed test rows are never scored or used for selection; the upstream
integrity check still hashes all source files and inspects their schemas.
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

from analysis.train_r4_discrete_baselines import read_json, sha256  # noqa: E402
from ml.forecast.alert_eval import evaluate_alerts  # noqa: E402


KEY = ["channel_id", "prediction_time"]
SCORES = ["rule_score", "logistic_score", "catboost_score"]
FLOAT_FIELDS = SCORES + ["score"]
TOLERANCE = 1e-12


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate R4 prediction key")
    return frame.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def compare_frames(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Compare identical populations by key, allowing only roundoff in scores."""
    if set(left.columns) != set(right.columns) or len(left) != len(right):
        raise ValueError("R4 replay schema or row count differs")
    row_order_equal = left[KEY].equals(right[KEY])
    left, right = _sorted(left), _sorted(right)
    if not left[KEY].equals(right[KEY]):
        raise ValueError("R4 replay prediction keys differ")
    for column in left.columns:
        if column not in KEY + FLOAT_FIELDS and not left[column].equals(right[column]):
            raise ValueError(f"R4 replay values differ: {column}")
    score_differences = {}
    for column in FLOAT_FIELDS:
        if column not in left:
            continue
        old = left[column].to_numpy(dtype=float)
        new = right[column].to_numpy(dtype=float)
        if not np.isclose(old, new, rtol=0, atol=TOLERANCE,
                          equal_nan=True).all():
            raise ValueError(f"R4 replay scores differ beyond tolerance: {column}")
        score_differences[column] = {
            "different_float_rows": int(np.count_nonzero(old != new)),
            "max_abs_difference": float(np.nanmax(np.abs(old - new))) if len(old) else 0.0,
        }
    return {
        "rows": len(left),
        "source_row_order_equal": row_order_equal,
        "score_differences": score_differences,
    }


def run(*, b_dir: Path, a_dir: Path, budget_dir: Path,
        decision_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"R4 A audit output already exists: {output_dir}")
    decision = read_json(decision_path)
    b_manifest, a_manifest = (read_json(root / "manifest.json")
                              for root in (b_dir, a_dir))
    b_report, a_report = (read_json(root / "report.json")
                          for root in (b_dir, a_dir))
    budget_manifest = read_json(budget_dir / "manifest.json")
    budget = read_json(budget_dir / "report.json")
    if (sha256(b_dir / "manifest.json")
            != decision["source_r4_validation_audit_manifest_sha256"]
            or sha256(budget_dir / "manifest.json")
            != decision["source_r4_budget_comparison_manifest_sha256"]
            or sha256(budget_dir / "report.json") != budget_manifest["report_sha256"]
            or any(manifest["schema_version"] != "r4-b-validation-audit-v1"
                   for manifest in (b_manifest, a_manifest))
            or any(sha256(root / "report.json") != manifest["report_sha256"]
                   for root, manifest in ((b_dir, b_manifest), (a_dir, a_manifest)))
            or any(sha256(root / "trace_review.json")
                   != manifest["trace_review_sha256"]
                   for root, manifest in ((b_dir, b_manifest), (a_dir, a_manifest)))):
        raise ValueError("R4 audit provenance differs")
    if (b_report["source_r3_contract_sha256"] != a_report["source_r3_contract_sha256"]
            or b_report["source_r3_contract_sha256"]
            != decision["source_r3_contract_sha256"]
            or b_report["source_r4_model_manifest_sha256"]
            != a_report["source_r4_model_manifest_sha256"]
            or b_report["source_r4_model_manifest_sha256"]
            != decision["source_r4_baselines_manifest_sha256"]
            or b_report["validation_rows"] != a_report["validation_rows"]
            or b_report["episode_lineage"] != a_report["episode_lineage"]
            or b_report["models_at_hour_f1_threshold"]
            != a_report["models_at_hour_f1_threshold"]):
        raise ValueError("R4 validation metrics or lineage differ")
    b_files = {item["file"]: item for item in b_manifest["files"]}
    a_files = {item["file"]: item for item in a_manifest["files"]}
    months = sorted(name for name in b_files if name.startswith("validation_"))
    if (set(b_files) != set(a_files) or len(months) != 12
            or any(not name.startswith("validation_2025-") for name in months)):
        raise ValueError("R4 replay lacks the twelve validation months")
    comparisons = {}
    validation_parts = []
    for name in months:
        if any(sha256(root / name) != files[name]["sha256"]
               for root, files in ((b_dir, b_files), (a_dir, a_files))):
            raise ValueError(f"R4 validation file hash differs from manifest: {name}")
        old = pd.read_parquet(b_dir / name)
        new = pd.read_parquet(a_dir / name)
        comparisons[name] = compare_frames(old, new)
        validation_parts.append(new[[
            "channel_id", "prediction_time", "sensor_type", "target",
            "target_episode_id", "label_available_at", "rule_score",
        ]])
    for name in sorted(set(b_files) - set(months)):
        if any(sha256(root / name) != files[name]["sha256"]
               for root, files in ((b_dir, b_files), (a_dir, a_files))):
            raise ValueError(f"R4 warning file hash differs from manifest: {name}")
        comparisons[name] = compare_frames(
            pd.read_parquet(b_dir / name), pd.read_parquet(a_dir / name))
    selected = decision["validation_only_candidate_threshold"]
    candidate, _ = evaluate_alerts(
        pd.concat(validation_parts, ignore_index=True), "rule_score", selected,
        channel_days=a_report["validation_channel_days"],
    )
    budget_candidate = budget["at_common_budgets"]["2.0"]["rule"]
    if (candidate["matched_episodes"] != decision["validation_candidate_matched_episodes"]
            or candidate["unmatched_warnings"]
            != decision["validation_candidate_unmatched_warnings"]
            or candidate["matched_episodes"] != budget_candidate["matched_episodes"]
            or candidate["unmatched_warnings"] != budget_candidate["unmatched_warnings"]
            or abs(selected - budget_candidate["threshold"]) > TOLERANCE):
        raise ValueError("R4 candidate threshold result differs")
    report = {
        "schema_version": "r4-a-batch-replay-audit-v1",
        "status": "passed",
        "source_b_audit_manifest_sha256": sha256(b_dir / "manifest.json"),
        "source_a_replay_manifest_sha256": sha256(a_dir / "manifest.json"),
        "source_b_budget_manifest_sha256": sha256(budget_dir / "manifest.json"),
        "validation_rows": sum(item["rows"] for name, item in comparisons.items()
                               if name in months),
        "validation_months": len(months),
        "candidate_threshold": selected,
        "candidate_matched_episodes": candidate["matched_episodes"],
        "candidate_unmatched_warnings": candidate["unmatched_warnings"],
        "max_score_difference_by_model": {
            score: max(item["score_differences"][score]["max_abs_difference"]
                       for name, item in comparisons.items() if name in months)
            for score in SCORES
        },
        "months_with_different_row_order": sum(
            not comparisons[name]["source_row_order_equal"] for name in months
        ),
        "comparisons": comparisons,
        "limitations": [
            "The replay uses the same frozen A3 features and B scoring implementation.",
            "This audit does not re-read M1 raw events or score sealed test rows.",
            "The 2-per-1000-channel-days budget is exploratory, not product-approved.",
        ],
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--budget-dir", type=Path, required=True)
    parser.add_argument("--decision", type=Path,
                        default=Path("ml/r4_b_baseline_decision_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(b_dir=args.b_dir, a_dir=args.a_dir, budget_dir=args.budget_dir,
                 decision_path=args.decision, output_dir=args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "status", "validation_rows", "validation_months", "candidate_threshold",
        "candidate_matched_episodes", "candidate_unmatched_warnings",
        "max_score_difference_by_model", "months_with_different_row_order",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
