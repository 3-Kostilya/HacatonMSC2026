"""Causal score smoothing with raw fallback across genuinely missing hours.

The full-context source calculated past scores before any target join. This
post-aggregation rowwise transform never reads target values or future times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from analysis.audit_ml_experiment import metadata_parity
from analysis.ml_experiment_round2_temporal import META, canonical_metric, threshold_curve
from analysis.prepare_ml_experiment import sha256


def literal(path):
    return "'"+str(path).replace("'","''")+"'"


def columns():
    values = []
    for kind in ["pooled","linear"]:
        raw = f"score_{kind}_raw"
        for hours in [2,3]:
            mean = f"score_{kind}_meanlogit_{hours}h"
            probability = f"1/(1+exp(-CAST({mean} AS DOUBLE)))"
            values.append(f"CAST(coalesce({probability},{raw}) AS FLOAT) AS score_{kind}_fallback_mean_{hours}h")
            values.append(f"CAST(coalesce(score_{kind}_min_{hours}h,{raw}) AS FLOAT) "
                          f"AS score_{kind}_fallback_min_{hours}h")
        rising = f"score_{kind}_rising1h"
        values.append(f"CAST(coalesce(1/(1+exp(-CAST({rising} AS DOUBLE))),{raw}) AS FLOAT) "
                      f"AS score_{kind}_fallback_rising1h")
    return values


def derive(db, source: Path, destination: Path, data: Path, year: int):
    projection = ",".join(f'"{name}"' for name in META)
    db.execute(f"COPY (SELECT {projection}, {','.join(columns())} FROM read_parquet({literal(source)})) "
               f"TO {literal(destination)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)")
    n = db.execute("SELECT COUNT(*) FROM read_parquet(?)",[str(data)]).fetchone()[0]
    metadata_parity(db,destination,data,n,year)
    return {"source_sha256":sha256(source),"scores_sha256":sha256(destination),
            "full_rows":n,"complete_metadata_parity":True,"causal_aggregation_source":True}


def run(root: Path,output: Path):
    if output.exists():
        raise FileExistsError(output)
    causal = root/"temporal-full-context"
    prior = json.loads((causal/"report.json").read_text(encoding="utf-8"))
    if (not prior["aggregation_precedes_label_join"] or prior["source_label_mask_used"]
            or prior["status"]!="CAUSAL_ALL_ELIGIBLE_CONTEXT_2024_SELECTION_FROZEN_OPEN2025_TRANSFER"):
        raise AssertionError("fallback requires completed label-free full-context source")
    output.mkdir(parents=True)
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory=?",[str(output/"duckdb-temp")])
        tune_source=causal/"tune_scores.parquet"
        tune=output/"tune_scores.parquet"
        tune_provenance=derive(db,tune_source,tune,Path("output/ml-experiment/data/tune.parquet"),2024)
        names=[name for name in pq.ParquetFile(tune).schema_arrow.names if name.startswith("score_")]
        curves={name:threshold_curve(db,tune,name,1204,45) for name in names}
        choices={name:max(curve,key=lambda row:(row["full_episode_f1"],row["episode_precision"]))
                 for name,curve in curves.items()}
        winner=max(choices,key=lambda name:choices[name]["full_episode_f1"])
        selection={"selection_year":2024,"selected_column":winner,"tune":choices,
                   "causal_full_eligible_source_sha256":sha256(causal/"selection.json"),
                   "source_context":"all eligible Q2 hours before original target join",
                   "tune_source":tune_provenance,"cooldown_hours":24}
        (output/"selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
        (output/"curves.json").write_text(json.dumps(curves,ensure_ascii=False,indent=2),encoding="utf-8")
        print("2024 selected causal fallback",winner,choices[winner]["full_episode_f1"],flush=True)
        # Nothing from 2025 has been read before the selection file exists.
        validation=output/"validation_scores.parquet"
        val_provenance=derive(db,causal/"validation_scores.parquet",validation,
                              Path("output/ml-experiment/data/validation.parquet"),2025)
        transfers={name:canonical_metric(db,validation,name,choice["threshold"],2142)[0]
                   for name,choice in choices.items()}
        reported_tune,_=canonical_metric(db,tune,winner,choices[winner]["threshold"],1204)
        for key in ["matched_episodes","emitted_warnings","episode_precision",
                    "full_episode_recall","full_episode_f1"]:
            if abs(reported_tune[key]-choices[winner][key])>1e-14:
                raise AssertionError(f"causal fallback2024 production parity differs: {key}")
    report={"status":"CAUSAL_FALLBACK_SELECTED_2024_OPEN2025_TRANSFER",
            "selection":selection,"tune":choices,"validation":transfers,
            "selected_validation":transfers[winner],"validation_source":val_provenance,
            "model_threshold_chosen_only_2024":True,"data_2021_read":False,"test_2026_read":False,
            "physical_availability_approved":False}
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({name:{key:metric[key] for key in ["matched_episodes","emitted_warnings",
         "episode_precision","full_episode_recall","full_episode_f1"]} for name,metric in transfers.items()}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("output/ml-experiment-round2"))
    parser.add_argument("--output",type=Path,default=Path("output/ml-experiment-round2/temporal-fallback"))
    args=parser.parse_args()
    run(args.root,args.output)
