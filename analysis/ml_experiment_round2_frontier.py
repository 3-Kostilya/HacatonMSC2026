"""Open-2025 diagnostic threshold headroom on the unchanged complete B3 target.

This must not alter any 2024-frozen operating policy or be called a holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from analysis.audit_ml_experiment import metadata_parity
from analysis.ml_experiment_round2_temporal import canonical_metric, compact, threshold_curve
from analysis.prepare_ml_experiment import sha256


SOURCES = {
    "frozen_model": ("coverage/frozen-models/scores_validation.parquet", ["score_tree","score_linear"]),
    "retrained": ("coverage/retrained/scores_validation.parquet", ["score_engineered_episode"]),
    "moderate": ("coverage/moderate/scores_validation.parquet", ["score_q3_episode_sqrt"]),
    "type_router": ("coverage-routing/validation_scores.parquet", ["score_flexible","score_regularized"]),
}


def run(root: Path, output: Path, families: list[str] | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    data = root / "coverage/data"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["full_episode_count"]["validation"] == 2142
    expected = manifest["row_stats"]["validation"]
    results, curves = {}, {}
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory=?", [str(output/"duckdb-temp")])
        for family in families or SOURCES:
            relative,columns = SOURCES[family]
            source = root/relative
            parity = metadata_parity(db,source,data/"validation.parquet",expected["rows"],2025)
            available = db.execute("""SELECT COUNT(DISTINCT CASE WHEN target=1
                THEN target_episode_id END) FROM read_parquet(?)""",[str(source)]).fetchone()[0]
            if available != expected["available_episodes"] or available != 2123:
                raise AssertionError("research extension episode availability changed")
            for column in columns:
                curve = threshold_curve(db,source,column,2142,points=45)
                optimistic = max(curve,key=lambda row:row["full_episode_f1"])
                canonical,_ = canonical_metric(db,source,column,optimistic["threshold"],2142)
                for key in ["matched_episodes","emitted_warnings","episode_precision",
                            "full_episode_recall","full_episode_f1"]:
                    if abs(canonical[key]-optimistic[key])>1e-14:
                        raise AssertionError(f"production warning parity differs: {family}/{column}/{key}")
                constrained = [row for row in curve if row["episode_precision"]>.265]
                key = family+"/"+column
                results[key] = {"optimistic_best_full_f1":optimistic,
                    "best_recall_at_diagnostic_precision_gt_0_265":
                        max(constrained,key=lambda row:row["full_episode_recall"]) if constrained else None,
                    "joint_P_gt_0_7_R_gt_0_5":any(row["episode_precision"]>.7 and
                        row["full_episode_recall"]>.5 for row in curve),
                    "complete_metadata":parity,"available_episodes":available,
                    "independent_production_warning_parity":True,"source_sha256":sha256(source)}
                curves[key] = curve
                print("OPEN2025 Q3 DIAGNOSTIC",key,compact(optimistic),flush=True)
    winner = max(results,key=lambda key:results[key]["optimistic_best_full_f1"]["full_episode_f1"])
    result = {"status":"OPEN2025_OPTIMISTIC_DIAGNOSTIC_NO_POLICY_PROMOTION",
              "winner_selected_using_2025_labels":winner,"by_score":results,
              "points":sum(map(len,curves.values())),"full_episode_count":2142,
              "original_2024_frozen_policies_changed":False,
              "physical_availability_approved":False,"production_admission_changed":False,
              "test_2026_read":False,"data_2021_read":False,
              "limitations":["Every threshold here was chosen using previously opened 2025 outcomes.",
                "This is an optimistic upper-bound exploration, not a new validated winner.",
                "Q3 research admission still needs independent B approval before any production change."]}
    (output/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"curves.json").write_text(json.dumps(curves,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("output/ml-experiment-round2"))
    parser.add_argument("--output",type=Path,default=Path("output/ml-experiment-round2/coverage-frontier"))
    parser.add_argument("--families",nargs="+",choices=list(SOURCES))
    args = parser.parse_args()
    run(args.root,args.output,args.families)
