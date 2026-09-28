"""Test pre-2024 type specialists as no-known-recall-loss warning vetoes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from analysis.ml_experiment_round6_standard_specialist import extract
from analysis.ml_experiment_specialists import engineered_input
from analysis.prepare_ml_experiment import sha256


MODEL_ROOT = Path("output/ml-experiment/specialists")
NAMES = json.loads(Path("output/ml-experiment/data/manifest.json").read_text(
    encoding="utf-8"))["all_features"]
MODELS = {"Датчик дыма":("type0_episode","type0_moderate","type0_hardnegative"),
          "Состояние фазы":("type1_episode","type1_moderate","type1_hardnegative")}


def score_year(year: int, output: Path) -> pd.DataFrame:
    frame = extract(year)
    x = engineered_input(frame,NAMES)
    fit = json.loads((MODEL_ROOT/"fit_manifest.json").read_text(encoding="utf-8"))
    for kind,model_names in MODELS.items():
        mask = frame.sensor_type.eq(kind)
        for name in model_names:
            path = MODEL_ROOT/f"{name}.cbm"
            model = CatBoostClassifier(thread_count=2)
            model.load_model(str(path))
            if model.feature_names_!=list(x.columns) or fit[name]["feature_names"]!=list(x.columns):
                raise AssertionError("pretrained specialist feature contract differs")
            scores = np.full(len(frame),np.nan,dtype="float32")
            scores[mask.to_numpy()] = model.predict_proba(
                x.loc[mask],thread_count=2)[:,1].astype("float32")
            frame[f"score_{name}"]=scores
    columns=["channel_id","prediction_time","sensor_type","outcome","target_episode_id"]
    frame[columns+[f"score_{name}" for models in MODELS.values() for name in models]].to_parquet(
        output/f"warning_scores_{year}.parquet",index=False)
    return frame


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tune = score_year(2024,output)
    options = []
    for kind,models in MODELS.items():
        subset = tune[tune.sensor_type.eq(kind)]
        positive = subset.outcome.eq("matched_known_episode")
        negative = subset.outcome.eq("known_no_target")
        for name in models:
            score=subset[f"score_{name}"]
            low=float(score[positive].min())
            threshold=float(np.nextafter(np.float32(low),np.float32(0)))
            removed=score.lt(threshold)
            options.append({"sensor_type":kind,"model":name,
                            "model_sha256":sha256(MODEL_ROOT/f"{name}.cbm"),
                            "threshold":threshold,"min_positive_score":low,
                            "matched_known_episodes":int(positive.sum()),
                            "removed_known_positive":int((positive & removed).sum()),
                            "removed_known_negative":int((negative & removed).sum()),
                            "removed_unknown":int((subset.outcome.eq("unknown_target") & removed).sum())})
    chosen={kind:max([o for o in options if o["sensor_type"]==kind],
                     key=lambda o:o["removed_known_negative"])
            for kind in MODELS}
    report={"status":"pre2024_type_specialist_veto_2024_selection",
            "training_latest_year":2023,"selection_year":2024,
            "options":options,"selected":chosen,
            "test_2026_read":False,"data_2021_read":False}
    (output/"frozen_2024_selection.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"selected":chosen},ensure_ascii=False),flush=True)
    return report


def evaluate_open_2025(output: Path) -> dict:
    selected=json.loads((output/"frozen_2024_selection.json").read_text(encoding="utf-8"))["selected"]
    frame=score_year(2025,output)
    metrics=[]
    for kind,models in MODELS.items():
        subset=frame[frame.sensor_type.eq(kind)]
        for name in models:
            option=selected[kind]
            # The selected threshold is displayed for the chosen model only.
            if name!=option["model"]:
                continue
            removed=subset[f"score_{name}"].lt(option["threshold"])
            metrics.append({"sensor_type":kind,"model":name,"threshold":option["threshold"],
                            "removed_warnings":int(removed.sum()),
                            "removed_known_positive":int((removed & subset.outcome.eq(
                                "matched_known_episode")).sum()),
                            "removed_known_negative":int((removed & subset.outcome.eq(
                                "known_no_target")).sum()),
                            "removed_unknown":int((removed & subset.outcome.eq(
                                "unknown_target")).sum())})
    report={"status":"open_2025_static_pretrained_veto_transfer",
            "selected":metrics,"test_2026_read":False,"data_2021_read":False}
    (output/"open_2025_static_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,
                        default=Path("output/ml-experiment-round7/pretrained-veto"))
    parser.add_argument("--evaluate-2025",action="store_true")
    args=parser.parse_args()
    if args.evaluate_2025:
        evaluate_open_2025(args.output)
    else:
        run(args.output)
