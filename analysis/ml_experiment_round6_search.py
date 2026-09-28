"""Small 2024-only specialist ablation with strict known-episode preservation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from analysis.ml_experiment_round6_standard_specialist import extract, feature_names, matrix
from analysis.prepare_ml_experiment import sha256


VARIANTS = (
    {"name":"depth3_reference","depth":3,"iterations":320,"positive_weight":1,
     "feature_set":"full"},
    {"name":"depth2_positive2","depth":2,"iterations":420,"positive_weight":2,
     "feature_set":"full"},
    {"name":"depth4_positive2","depth":4,"iterations":420,"positive_weight":2,
     "feature_set":"full"},
    {"name":"depth3_positive4","depth":3,"iterations":420,"positive_weight":4,
     "feature_set":"full"},
    {"name":"depth3_compact","depth":3,"iterations":420,"positive_weight":2,
     "feature_set":"compact"},
)
COMPACT = ["sensor_type","score_linear","score_tree","normal_age_hours",
           "last_observation_age_seconds","unknown_state_count_168h",
           "normal_message_count_168h","alarm_count_168h","event_count_24h"]


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frame = extract(2024)
    frame["normal_age_hours"] = ((frame.prediction_time-frame.last_explicit_normal_at)
                                 .dt.total_seconds()/3600).astype("float32")
    train_mask = frame.prediction_time.lt(pd.Timestamp("2024-09-01")) & frame.outcome.ne("unknown_target")
    hold_mask = frame.prediction_time.ge(pd.Timestamp("2024-09-01"))
    y = frame.loc[train_mask].outcome.eq("matched_known_episode").astype("int8")
    positive = frame.outcome.eq("matched_known_episode")
    full_names = feature_names()
    summaries = []
    for option in VARIANTS:
        names = full_names if option["feature_set"]=="full" else COMPACT
        model = CatBoostClassifier(iterations=option["iterations"],depth=option["depth"],
                                   learning_rate=.035,l2_leaf_reg=15,loss_function="Logloss",
                                   random_seed=42,thread_count=2,verbose=False,
                                   allow_writing_files=False,cat_features=["sensor_type"],
                                   class_weights=[1,option["positive_weight"]])
        model.fit(matrix(frame.loc[train_mask],names),y)
        path = output/f"{option['name']}.cbm"
        model.save_model(str(path))
        score = model.predict_proba(matrix(frame,names),thread_count=2)[:,1].astype("float32")
        floor = float(score[positive.to_numpy()].min())
        threshold = float(max(0,np.floor(floor*1000)/1000-.001))
        removed = score < threshold
        hold_negative = hold_mask & frame.outcome.eq("known_no_target")
        summary = {**option,"model_sha256":sha256(path),"feature_names":names,
                   "min_2024_positive_score":floor,"threshold":threshold,
                   "all_removed":int(removed.sum()),
                   "all_removed_known_negative":int((removed & frame.outcome.eq("known_no_target")).sum()),
                   "all_removed_known_positive":int((removed & positive).sum()),
                   "holdout_removed_known_negative":int((removed & hold_negative).sum()),
                   "holdout_removed_known_positive":int((removed & hold_mask & positive).sum())}
        frame[["channel_id","prediction_time","sensor_type","outcome","target_episode_id"]].assign(
            specialist_score=score).to_parquet(output/f"{option['name']}_2024_scores.parquet",index=False)
        summaries.append(summary)
        print(option["name"],summary["holdout_removed_known_negative"],
              summary["all_removed_known_negative"],flush=True)
    selected = max(summaries,key=lambda item:(item["holdout_removed_known_negative"],
                                             item["all_removed_known_negative"]))
    result = {"status":"2024_only_standard_specialist_ablation",
              "fit_before":"2024-09-01","selection_year":2024,
              "holdout_from":"2024-09-01","selected":selected["name"],
              "variants":summaries,"test_2026_read":False,"data_2021_read":False}
    (output/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),
                                      encoding="utf-8")
    print(json.dumps({"selected":selected["name"],"holdout_removed_known_negative":
                      selected["holdout_removed_known_negative"]},ensure_ascii=False),flush=True)
    return result


def evaluate_open_2025(output: Path) -> dict:
    report = json.loads((output/"report.json").read_text(encoding="utf-8"))
    frame = extract(2025)
    frame["normal_age_hours"] = ((frame.prediction_time-frame.last_explicit_normal_at)
                                 .dt.total_seconds()/3600).astype("float32")
    summaries = []
    for variant in report["variants"]:
        path = output/f"{variant['name']}.cbm"
        if sha256(path)!=variant["model_sha256"]:
            raise AssertionError("2024 ablation model changed")
        model = CatBoostClassifier(thread_count=2)
        model.load_model(str(path))
        scores = model.predict_proba(
            matrix(frame,variant["feature_names"]),thread_count=2)[:,1].astype("float32")
        removed = scores<variant["threshold"]
        summaries.append({"name":variant["name"],"threshold":variant["threshold"],
                          "removed_warnings":int(removed.sum()),
                          "removed_known_positive":int((removed & frame.outcome.eq(
                              "matched_known_episode")).sum()),
                          "removed_known_negative":int((removed & frame.outcome.eq(
                              "known_no_target")).sum()),
                          "removed_unknown":int((removed & frame.outcome.eq(
                              "unknown_target")).sum())})
    result={"status":"open_2025_static_ablation_sensitivity_not_stream_replay",
            "variants":summaries,"test_2026_read":False,"data_2021_read":False}
    (output/"open_2025_static_report.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,
                        default=Path("output/ml-experiment-round6/2024-ablation"))
    parser.add_argument("--evaluate-2025",action="store_true")
    args=parser.parse_args()
    if args.evaluate_2025:
        evaluate_open_2025(args.output)
    else:
        run(args.output)
