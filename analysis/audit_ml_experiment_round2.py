"""Independent full-key and production warning audit of round-two research."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from analysis.audit_ml_experiment import metadata_parity
from analysis.ml_experiment_metric_audit import replay
from analysis.prepare_ml_experiment import sha256


KEYS = ["matched_episodes","emitted_warnings","unmatched_warnings",
        "duplicate_episode_warnings","suppressed_positive_score_rows","episode_precision",
        "full_episode_recall","full_episode_f1","median_lead_hours"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cases(root: Path):
    original = Path("output/ml-experiment/data")
    q3 = root/"coverage/data"
    for family in ["models","coverage/retrained","coverage/moderate"]:
        directory = root/family
        report = read(directory/"report.json")
        for name,item in report["experiments"].items():
            threshold = item["tune"]["selected"]["threshold"]
            for fold in ["tune","validation"]:
                expected = item["tune"]["selected"] if fold=="tune" else item["validation_frozen"]
                data = original if family == "models" else q3
                yield family,name,fold,directory/f"scores_{fold}.parquet",f"score_{name}",threshold,expected,data
    directory = root/"coverage/frozen-models"
    report = read(directory/"report.json")
    for name,choice in report["selection"]["tune"].items():
        for fold in ["tune","validation"]:
            expected = choice if fold=="tune" else report["validation"][name]
            yield "coverage/frozen-models",name,fold,directory/f"scores_{fold}.parquet",name,choice["threshold"],expected,q3
    for family,data in [("routing",original),("coverage-routing",q3)]:
        directory = root/family
        report = read(directory/"report.json")
        if sha256(directory/"selection.json") != report["selection_sha256"]:
            raise AssertionError(f"policy changed after scoring: {family}")
        for name in report["validation"]:
            for fold in ["tune","validation"]:
                expected = report[fold][name]
                yield family,name,fold,directory/f"{fold}_scores.parquet",f"score_{name}",0.0,expected,data
    for temporal_family in ["temporal-full-context", "temporal-fallback"]:
        directory = root/temporal_family
        if not (directory/"report.json").exists():
            continue
        report = read(directory/"report.json")
        if temporal_family == "temporal-full-context":
            if not report["aggregation_precedes_label_join"] or report["source_label_mask_used"]:
                raise AssertionError("temporal policy used the future label mask")
        elif report["selection"]["source_context"] != "all eligible Q2 hours before original target join":
            raise AssertionError("temporal fallback is not derived from causal context")
        for name,choice in report["tune"].items():
            for fold in ["tune","validation"]:
                expected = choice if fold=="tune" else report["validation"][name]
                yield temporal_family,name,fold,directory/f"{fold}_scores.parquet",name,choice["threshold"],expected,original


def run(root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    checked, parity = [], {}
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory=?", [str(output/"duckdb-temp")])
        for family,name,fold,source,column,threshold,expected,data in cases(root):
            dataset = read(data/"manifest.json")
            full = dataset["full_episode_count"][fold]
            if full != (1204 if fold=="tune" else 2142):
                raise AssertionError("full denominator differs")
            key = str(source)
            if key not in parity:
                parity[key] = metadata_parity(db,source,data/f"{fold}.parquet",
                    dataset["row_stats"][fold]["rows"],2024 if fold=="tune" else 2025)
            actual = replay(db,source,column,threshold,full)
            for field in KEYS:
                if field in expected and actual[field] != expected[field]:
                    if field not in ["episode_precision","full_episode_recall","full_episode_f1"] or abs(
                            actual[field]-expected[field])>1e-14:
                        raise AssertionError(f"{family}/{name}/{fold} canonical {field} differs")
            if "matched_episode_ids" in expected and set(actual["matched_episode_ids"]) != set(expected["matched_episode_ids"]):
                raise AssertionError(f"{family}/{name}/{fold} matched episode identities differ")
            checked.append({"family":family,"name":name,"fold":fold,
                            "matched_episodes":actual["matched_episodes"],
                            "emitted_warnings":actual["emitted_warnings"],
                            "full_episode_f1":actual["full_episode_f1"],
                            "exact_matched_episode_identity": "matched_episode_ids" in expected,
                            "production_warning_and_full_metadata_parity":True})
            print("round2 independent parity passed",family,name,fold,flush=True)
    report = {"status":"round2_full_metadata_and_production_warning_parity_passed",
              "checks":checked,"score_metadata":parity,"test_2026_read":False,
              "data_2021_read":False,"production_policy_changed":False}
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("output/ml-experiment-round2"))
    parser.add_argument("--output",type=Path,default=Path("output/ml-experiment-round2/independent-audit"))
    args = parser.parse_args()
    run(args.root,args.output)
