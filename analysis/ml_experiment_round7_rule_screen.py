"""Screen single causal feature vetoes on 2024 standard warnings; transfer to open 2025."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.ml_experiment_round6_standard_specialist import extract


KINDS=("Датчик дыма","Состояние фазы")
FORBIDDEN={"channel_id","prediction_time","sensor_type","outcome","target_episode_id",
           "last_explicit_normal_at"}


def candidates(frame: pd.DataFrame) -> list[dict]:
    found=[]
    for kind in KINDS:
        subset=frame[frame.sensor_type.eq(kind)]
        positive=subset.outcome.eq("matched_known_episode")
        negative=subset.outcome.eq("known_no_target")
        for name in subset.columns:
            if name in FORBIDDEN or not pd.api.types.is_numeric_dtype(subset[name]):
                continue
            for direction,bound in (("below",subset.loc[positive,name].min()),
                                    ("above",subset.loc[positive,name].max())):
                if pd.isna(bound):
                    continue
                removed=(subset[name]<bound if direction=="below" else subset[name]>bound)
                n=int((removed & negative).sum())
                if n>=5:
                    found.append({"sensor_type":kind,"feature":name,"direction":direction,
                                  "bound":float(bound),"removed_negative_2024":n,
                                  "removed_unknown_2024":int((removed & subset.outcome.eq(
                                      "unknown_target")).sum()),
                                  "removed_positive_2024":int((removed & positive).sum())})
    return sorted(found,key=lambda item:-item["removed_negative_2024"])


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tune=extract(2024)
    tune["normal_age_hours"]=(tune.prediction_time-tune.last_explicit_normal_at).dt.total_seconds()/3600
    rules=candidates(tune)
    # Selection is frozen before any 2025 warning outcomes are opened.
    selection={"status":"2024_only_single_feature_no_known_hit_loss_screen",
               "top_rules":rules[:25],"all_rule_count":len(rules),
               "test_2026_read":False,"data_2021_read":False}
    (output/"frozen_2024_rules.json").write_text(
        json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"top_rules":rules[:10]},ensure_ascii=False),flush=True)
    validation=extract(2025)
    validation["normal_age_hours"]=(validation.prediction_time-
                                    validation.last_explicit_normal_at).dt.total_seconds()/3600
    for rule in rules[:25]:
        subset=validation[validation.sensor_type.eq(rule["sensor_type"])]
        removed=(subset[rule["feature"]]<rule["bound"] if rule["direction"]=="below"
                 else subset[rule["feature"]]>rule["bound"])
        rule["removed_positive_2025"]=int((removed & subset.outcome.eq(
            "matched_known_episode")).sum())
        rule["removed_negative_2025"]=int((removed & subset.outcome.eq(
            "known_no_target")).sum())
        rule["removed_unknown_2025"]=int((removed & subset.outcome.eq(
            "unknown_target")).sum())
    result={"status":"open_2025_single_feature_rule_transfer",
            "top_rules":rules[:25],"test_2026_read":False,"data_2021_read":False}
    (output/"open_2025_report.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"top_rules":rules[:10]},ensure_ascii=False),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,
                        default=Path("output/ml-experiment-round7/rule-screen"))
    run(parser.parse_args().output)
