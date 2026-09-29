"""Audit sparse QA corrections against the frozen R4 rule without opening test labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from analysis.train_r4_discrete_baselines import read_json, sha256, rule_score


RULE_FIELDS = (
    "registered_fault_text_count_24h",
    "registered_fault_text_count_168h",
    "completed_episode_count_168h",
    "technical_message_count_24h",
)
KEYS = ["channel_id", "prediction_time"]


def changed(old: pd.Series, new: pd.Series) -> pd.Series:
    """Compare values while treating two missing values as the same."""
    return ~(old.eq(new).fillna(False) | (old.isna() & new.isna()))


def audit(a3_dir: Path, index_dir: Path, overlay_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    a3_manifest = read_json(a3_dir / "manifest.json")
    index_manifest = read_json(index_dir / "manifest.json")
    overlay = read_json(overlay_dir / "report.json")
    if (overlay["schema_version"] != "qa-value-corrections-v1"
            or overlay["status"] != "complete"
            or overlay["source_a3_manifest_sha256"] != sha256(a3_dir / "manifest.json")
            or overlay["feature_corrections_file_sha256"]
            != sha256(overlay_dir / "feature_corrections.parquet")
            or overlay["qa_quality_counts_file_sha256"]
            != sha256(overlay_dir / "qa_quality_counts.parquet")
            or index_manifest["source_a3_manifest_sha256"]
            != overlay["source_a3_manifest_sha256"]):
        raise ValueError("QA overlay or R3 source lineage differs")
    corrections = pq.read_table(overlay_dir / "feature_corrections.parquet").to_pandas()
    if (len(corrections) != overlay["corrected_a3_rows"]
            or corrections.duplicated(KEYS).any()
            or corrections.prediction_time.dt.year.eq(2021).any()):
        raise ValueError("QA correction keys differ")
    allowlist = read_json(Path("ml/r3_discrete_feature_allowlist_v1.json"))
    fields = allowlist["feature_names"]
    a3_chunks = {item["month"]: item for item in a3_manifest["chunks"]}
    index_chunks = {item["month"]: item for item in index_manifest["chunks"]}
    by_split: Counter[str] = Counter()
    by_year: Counter[str] = Counter()
    changed_r4_fields: Counter[str] = Counter()
    changed_rule_fields: Counter[str] = Counter()
    admitted_positive_hours: Counter[str] = Counter()
    rule_score_changed = 0
    threshold_crossings = 0
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        for month, block in corrections.groupby(
                corrections.prediction_time.dt.strftime("%Y-%m"), sort=True):
            if month not in a3_chunks:
                raise ValueError(f"corrected month missing from A3: {month}")
            database.register("correction_month", block)
            a3_path = a3_dir / a3_chunks[month]["features_file"]
            columns = ", ".join(f'f."{name}" AS "old_{name}"' for name in fields)
            old = database.execute(
                f"""SELECT q.channel_id, q.prediction_time, {columns}
                    FROM correction_month q
                    JOIN read_parquet(?) f USING (channel_id, prediction_time)""",
                [str(a3_path)],
            ).fetch_df()
            joined = block.merge(old, on=KEYS, validate="one_to_one")
            if len(joined) != len(block):
                raise ValueError(f"QA/A3 key mismatch in {month}")
            for name in fields:
                count = int(changed(joined[f"old_{name}"], joined[name]).sum())
                if count:
                    changed_r4_fields[name] += count
                    if name in RULE_FIELDS:
                        changed_rule_fields[name] += count
            old_rule = pd.DataFrame({name: joined[f"old_{name}"] for name in RULE_FIELDS})
            previous = rule_score(old_rule)
            updated = rule_score(joined)
            rule_score_changed += int((previous != updated).sum())
            threshold_crossings += int(((previous >= 7.1) != (updated >= 7.1)).sum())
            if month < "2026-01":
                item = index_chunks[month]
                candidate = ((index_dir / item["manifest_file"]).parent /
                             "conditional_discrete_keys.parquet")
                admitted = database.execute(
                    """SELECT c.split, c.target FROM correction_month q
                        JOIN read_parquet(?) c USING (channel_id, prediction_time)""",
                    [str(candidate)],
                ).fetch_df()
                if not admitted.split.isin(["train", "validation"]).all():
                    raise ValueError(f"unexpected split in {month}")
                by_split.update(admitted.split.astype(str))
                by_year[month[:4]] += len(admitted)
                for split, subset in admitted.groupby("split"):
                    admitted_positive_hours[str(split)] += int(subset.target.sum())
            database.unregister("correction_month")
    report = {
        "schema_version": "r6-b-qa-baseline-impact-v1",
        "status": "validated_train_validation_only",
        "source_a3_manifest_sha256": sha256(a3_dir / "manifest.json"),
        "source_r3_index_manifest_sha256": sha256(index_dir / "manifest.json"),
        "source_qa_corrections_report_sha256": sha256(overlay_dir / "report.json"),
        "corrected_a3_rows": len(corrections),
        "admitted_corrected_rows_by_split": dict(by_split),
        "admitted_corrected_rows_by_year": dict(sorted(by_year.items())),
        "admitted_positive_hours_by_split": dict(admitted_positive_hours),
        "r4_allowlist_changed_fields": dict(changed_r4_fields),
        "r4_rule_changed_fields": dict(changed_rule_fields),
        "r4_rule_score_changed_rows_all_corrected": rule_score_changed,
        "r4_rule_threshold_7_1_crossings_all_corrected": threshold_crossings,
        "qa_count_features_training_ready": False,
        "test_labels_read": False,
    }
    if changed_rule_fields or rule_score_changed or threshold_crossings:
        raise ValueError("QA changes the frozen R4 rule; repeat validation before R6")
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.a3_dir, args.index_dir, args.overlay_dir,
                           args.output_dir), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
