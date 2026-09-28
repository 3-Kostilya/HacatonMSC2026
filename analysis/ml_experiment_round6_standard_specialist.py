"""Research-only second-stage model for standard smoke and phase warnings.

Train on early 2024, freeze a no-known-episode-loss threshold on all 2024,
then inspect the already-open 2025. Unknown outcomes never become negatives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from catboost import CatBoostClassifier
import duckdb
import numpy as np
import pandas as pd

from analysis.prepare_ml_experiment import sha256


KINDS = ("Датчик дыма", "Состояние фазы")
ROOT = Path("output/ml-experiment-round6")
Q2 = Path("output/q2-a-full-sparse-20260926-v5")
Q3 = Path("output/q3-a-coverage-reentry-20260927-v4")
BASE = Path("output/ml-experiment-round5")
SOURCES = {
    2024: (BASE / "q3-full-context-linear-2024-phase-age5-tree95-gas",
           BASE / "q3-full-context-linear-2024-base/full_context_scores.parquet",
           BASE / "q3-full-context-tree-2024-base/full_context_scores.parquet"),
    2025: (BASE / "q3-full-context-linear-2025-phase-age5-tree95-gas",
           Path("output/ml-experiment-round4/q3-full-context-linear-th99544/full_context_scores.parquet"),
           Path("output/ml-experiment-round4/q3-full-context-tree-th992-v2/full_context_scores.parquet")),
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_names() -> list[str]:
    names = read(Path("output/ml-experiment/data/manifest.json"))["all_features"]
    prefixes = ("event_count_", "alarm_count_", "normal_message_count_",
                "registered_fault_text_count_", "technical_message_count_",
                "unknown_state_count_", "state_transitions_")
    chosen = [name for name in names if name == "sensor_type" or name.startswith(prefixes)]
    chosen += ["last_observation_age_seconds", "last_completed_episode_end_age_seconds",
               "score_linear", "score_tree", "normal_age_hours"]
    if len(chosen) != len(set(chosen)):
        raise AssertionError("duplicate specialist feature")
    return chosen


def extract(year: int) -> pd.DataFrame:
    warning_dir, linear, tree = SOURCES[year]
    warning = warning_dir / "candidate_warnings.parquet"
    q2_manifest = read(Q2 / "manifest.json")
    frames = []
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        for month in range(1, 13):
            folder = f"year={year}/month={month:02d}"
            q2_features = Q2 / folder / "model_features.parquet"
            q3_features = Q3 / folder / "new_model_features.parquet"
            q2_row = next(item for item in q2_manifest["months"]
                          if item["month"] == f"{year}-{month:02d}")
            q3_row = read(Q3 / folder / "manifest.json")
            if (sha256(q2_features) != q2_row["files"]["model_features.parquet"]["sha256"]
                    or sha256(q3_features) != q3_row["files"]["new_model_features.parquet"]["sha256"]):
                raise AssertionError("causal feature source changed")
            part = db.execute('''WITH candidates AS (
                SELECT channel_id,prediction_time,sensor_type,outcome,target_episode_id,
                    last_explicit_normal_at
                FROM read_parquet(?)
                WHERE year(prediction_time)=? AND month(prediction_time)=?
                  AND reason='standard_24h_warning'
                  AND sensor_type IN ('Датчик дыма','Состояние фазы')
            ), features AS (
                SELECT * FROM read_parquet(?,hive_partitioning=false)
                UNION ALL SELECT * FROM read_parquet(?,hive_partitioning=false)
            )
            SELECT c.*,f.* EXCLUDE(channel_id,prediction_time,sensor_type),
                   l.score AS score_linear,t.score AS score_tree
            FROM candidates c JOIN features f USING(channel_id,prediction_time,sensor_type)
            JOIN read_parquet(?) l USING(channel_id,prediction_time,sensor_type)
            JOIN read_parquet(?) t USING(channel_id,prediction_time,sensor_type)''',
                [str(warning),year,month,str(q2_features),str(q3_features),
                 str(linear),str(tree)]).fetch_df()
            frames.append(part)
            print("extracted",year,month,len(part),flush=True)
    result = (pd.concat(frames,ignore_index=True)
              .sort_values(["prediction_time","channel_id"],kind="stable")
              .reset_index(drop=True))
    expected = pd.read_parquet(warning,columns=["channel_id","prediction_time","sensor_type","reason"])
    expected = expected[expected.reason.eq("standard_24h_warning") &
                        expected.sensor_type.isin(KINDS)]
    if len(result) != len(expected) or result.duplicated(["channel_id","prediction_time"]).any():
        raise AssertionError("standard warning feature join differs")
    result["normal_age_hours"] = ((result.prediction_time-result.last_explicit_normal_at)
                                  .dt.total_seconds()/3600).astype("float32")
    return result


def matrix(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    output = frame[names].copy()
    output["sensor_type"] = output.sensor_type.astype(str)
    for name in names:
        if name != "sensor_type":
            output[name] = pd.to_numeric(output[name],errors="coerce").replace(
                [np.inf,-np.inf],np.nan).fillna(-1).astype("float32")
    return output


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    names = feature_names()
    tune = extract(2024)
    eligible = tune.outcome.isin(["matched_known_episode","known_no_target"])
    train = tune.loc[eligible & tune.prediction_time.lt(pd.Timestamp("2024-09-01"))].copy()
    hold = tune.loc[eligible & tune.prediction_time.ge(pd.Timestamp("2024-09-01"))].copy()
    y = train.outcome.eq("matched_known_episode").astype("int8")
    if y.sum()<100 or hold.outcome.eq("matched_known_episode").sum()<50:
        raise AssertionError("insufficient chronological specialist support")
    model = CatBoostClassifier(iterations=320,depth=3,learning_rate=.035,
                               l2_leaf_reg=15,loss_function="Logloss",random_seed=42,
                               thread_count=2,verbose=False,allow_writing_files=False,
                               cat_features=["sensor_type"])
    model.fit(matrix(train,names),y)
    model_path = output / "standard_smoke_phase_2024_jan_aug.cbm"
    model.save_model(str(model_path))
    tune["specialist_score"] = model.predict_proba(matrix(tune,names),thread_count=2)[:,1].astype("float32")
    positives = tune.outcome.eq("matched_known_episode")
    # Deliberately round down and leave 0.001 margin below every known 2024 hit.
    threshold = float(np.floor(float(tune.loc[positives,"specialist_score"].min())*1000)/1000-.001)
    if not 0 <= threshold <= 1:
        raise AssertionError("invalid specialist threshold")
    validation = tune.prediction_time.ge(pd.Timestamp("2024-09-01"))
    def summary(frame: pd.DataFrame, selected: pd.Series) -> dict:
        return {"warnings":int(len(frame)),
                "known_positive":int(frame.outcome.eq("matched_known_episode").sum()),
                "known_negative":int(frame.outcome.eq("known_no_target").sum()),
                "unknown":int(frame.outcome.eq("unknown_target").sum()),
                "removed_at_threshold":int((~selected).sum()),
                "removed_known_positive":int((~selected & frame.outcome.eq("matched_known_episode")).sum()),
                "removed_known_negative":int((~selected & frame.outcome.eq("known_no_target")).sum()),
                "removed_unknown":int((~selected & frame.outcome.eq("unknown_target")).sum())}
    tune_selected = tune.specialist_score.ge(threshold)
    tune_summary = summary(tune,tune_selected)
    hold_summary = summary(tune.loc[validation],tune_selected.loc[validation])
    tune[["channel_id","prediction_time","sensor_type","outcome","target_episode_id",
          "specialist_score"]].to_parquet(output/"2024_standard_warning_scores.parquet",index=False)
    report = {"status":"research_standard_warning_specialist_2024_only",
              "fit_before":"2024-09-01","threshold_selection_year":2024,
              "model_sha256":sha256(model_path),"feature_names":names,
              "threshold":threshold,"train_rows":len(train),"train_positives":int(y.sum()),
              "holdout_rows":len(hold),"holdout_positives":int(hold.outcome.eq("matched_known_episode").sum()),
              "all_2024":tune_summary,"holdout_2024":hold_summary,
              "test_2026_read":False,"data_2021_read":False}
    (output/"frozen_2024_selection.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"threshold":threshold,"all_2024":tune_summary,
                      "holdout_2024":hold_summary},ensure_ascii=False),flush=True)
    return report


def evaluate_open_2025(output: Path) -> dict:
    selection = read(output / "frozen_2024_selection.json")
    model_path = output / "standard_smoke_phase_2024_jan_aug.cbm"
    if selection["model_sha256"] != sha256(model_path):
        raise AssertionError("frozen 2024 specialist changed")
    model = CatBoostClassifier(thread_count=2)
    model.load_model(str(model_path))
    frame = extract(2025)
    frame["specialist_score"] = model.predict_proba(
        matrix(frame,selection["feature_names"]),thread_count=2)[:,1].astype("float32")
    selected = frame.specialist_score.ge(selection["threshold"])
    report = {"status":"open_2025_static_warning_sensitivity_not_stream_replay",
              "threshold_source":"frozen_2024_selection.json",
              "warnings":len(frame),"known_positive":int(frame.outcome.eq("matched_known_episode").sum()),
              "known_negative":int(frame.outcome.eq("known_no_target").sum()),
              "unknown":int(frame.outcome.eq("unknown_target").sum()),
              "removed_warnings":int((~selected).sum()),
              "removed_known_positive":int((~selected & frame.outcome.eq("matched_known_episode")).sum()),
              "removed_known_negative":int((~selected & frame.outcome.eq("known_no_target")).sum()),
              "removed_unknown":int((~selected & frame.outcome.eq("unknown_target")).sum()),
              "test_2026_read":False,"data_2021_read":False}
    frame[["channel_id","prediction_time","sensor_type","outcome","target_episode_id",
           "specialist_score"]].to_parquet(output/"2025_standard_warning_scores.parquet",index=False)
    (output/"open_2025_static_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def score_full_shortlist(output: Path, year: int) -> dict:
    selection = read(output / "frozen_2024_selection.json")
    model_path = output / "standard_smoke_phase_2024_jan_aug.cbm"
    if selection["model_sha256"] != sha256(model_path):
        raise AssertionError("frozen specialist changed")
    model = CatBoostClassifier(thread_count=2)
    model.load_model(str(model_path))
    _, linear, tree = SOURCES[year]
    parts = []
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute('''CREATE TEMP TABLE shortlist_linear AS
            SELECT channel_id,prediction_time,sensor_type,score AS score_linear
            FROM read_parquet(?) WHERE
              (sensor_type='Датчик дыма' AND score>=?) OR
              (sensor_type='Состояние фазы' AND score>=?)''',
            [str(linear),0.995442509341302,0.9958])
        db.execute('''CREATE TEMP TABLE shortlist AS
            SELECT l.*,t.score AS score_tree FROM shortlist_linear l
            JOIN read_parquet(?) t USING(channel_id,prediction_time,sensor_type)
            WHERE t.score>=0.95''',[str(tree)])
        total = db.execute("SELECT COUNT(*) FROM shortlist").fetchone()[0]
        for month in range(1,13):
            folder = f"year={year}/month={month:02d}"
            frame = db.execute('''WITH features AS (
                SELECT * FROM read_parquet(?,hive_partitioning=false)
                UNION ALL SELECT * FROM read_parquet(?,hive_partitioning=false)
            ), admission AS (
                SELECT channel_id,prediction_time,sensor_type,last_explicit_normal_at
                FROM read_parquet(?,hive_partitioning=false) WHERE admission_status='eligible'
                UNION ALL SELECT channel_id,prediction_time,sensor_type,last_explicit_normal_at
                FROM read_parquet(?,hive_partitioning=false) WHERE combined_status='eligible'
            )
            SELECT k.*, f.* EXCLUDE(channel_id,prediction_time,sensor_type),
                   a.last_explicit_normal_at FROM shortlist k
            JOIN features f USING(channel_id,prediction_time,sensor_type)
            JOIN admission a USING(channel_id,prediction_time,sensor_type)
            WHERE year(k.prediction_time)=? AND month(k.prediction_time)=?''',
                [str(Q2/folder/"model_features.parquet"),
                 str(Q3/folder/"new_model_features.parquet"),
                 str(Q2/folder/"admission.parquet"),
                 str(Q3/folder/"new_admission.parquet"),year,month]).fetch_df()
            frame["normal_age_hours"] = ((frame.prediction_time-frame.last_explicit_normal_at)
                                         .dt.total_seconds()/3600).astype("float32")
            frame["specialist_score"] = model.predict_proba(
                matrix(frame,selection["feature_names"]),thread_count=2)[:,1].astype("float32")
            parts.append(frame[["channel_id","prediction_time","sensor_type","specialist_score"]])
            print("scored shortlist",year,month,len(frame),flush=True)
    scores = pd.concat(parts,ignore_index=True)
    if len(scores)!=total or scores.duplicated(["channel_id","prediction_time"]).any():
        raise AssertionError("specialist shortlist has missing or duplicate feature rows")
    static_path = output/f"{year}_standard_warning_scores.parquet"
    if static_path.exists():
        static = pd.read_parquet(static_path)
        parity = static.merge(scores,on=["channel_id","prediction_time","sensor_type"],
                              how="left",validate="one_to_one",suffixes=("_static","_all"))
        if (len(parity)!=len(static) or parity.specialist_score_all.isna().any()
                or not np.array_equal(parity.specialist_score_static.to_numpy(),
                                      parity.specialist_score_all.to_numpy())):
            raise AssertionError("all-hour and emitted-warning specialist scores differ")
    destination = output/f"all_shortlist_scores_{year}.parquet"
    scores.to_parquet(destination,index=False)
    report = {"status":"label_free_all_shortlist_specialist_scores",
              "year":year,"rows":len(scores),"score_sha256":sha256(destination),
              "model_sha256":selection["model_sha256"],"threshold":selection["threshold"],
              "linear_source_sha256":sha256(linear),"tree_source_sha256":sha256(tree),
              "test_2026_read":False,"data_2021_read":False}
    (output/f"all_shortlist_scores_{year}_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def adopt_2024_ablation(source: Path, output: Path) -> dict:
    """Freeze the variant selected without opening the 2025 outcomes."""
    if output.exists():
        raise FileExistsError(output)
    ablation = read(source/"report.json")
    if (ablation["selection_year"]!=2024 or ablation["fit_before"]!="2024-09-01"
            or ablation["test_2026_read"] or ablation["data_2021_read"]):
        raise AssertionError("ablation chronology differs")
    selected = next(item for item in ablation["variants"]
                    if item["name"]==ablation["selected"])
    if selected["all_removed_known_positive"] or selected["holdout_removed_known_positive"]:
        raise AssertionError("selected specialist loses 2024 episodes")
    original = source/f"{selected['name']}.cbm"
    if sha256(original)!=selected["model_sha256"]:
        raise AssertionError("selected 2024 model changed")
    output.mkdir(parents=True)
    model_path = output/"standard_smoke_phase_2024_jan_aug.cbm"
    shutil.copyfile(original,model_path)
    shutil.copyfile(source/f"{selected['name']}_2024_scores.parquet",
                    output/"2024_standard_warning_scores.parquet")
    selection = {"status":"frozen_2024_ablation_selected_specialist",
                 "fit_before":"2024-09-01","threshold_selection_year":2024,
                 "model_sha256":sha256(model_path),"feature_names":selected["feature_names"],
                 "threshold":selected["threshold"],"selected_variant":selected["name"],
                 "selection_report_sha256":sha256(source/"report.json"),
                 "all_2024_removed_known_negative":selected["all_removed_known_negative"],
                 "holdout_2024_removed_known_negative":selected["holdout_removed_known_negative"],
                 "test_2026_read":False,"data_2021_read":False}
    (output/"frozen_2024_selection.json").write_text(
        json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(selection,ensure_ascii=False),flush=True)
    return selection


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"static-2024")
    parser.add_argument("--evaluate-2025",action="store_true")
    parser.add_argument("--score-shortlist",type=int,choices=[2024,2025])
    parser.add_argument("--adopt-ablation",type=Path)
    args=parser.parse_args()
    if args.adopt_ablation:
        adopt_2024_ablation(args.adopt_ablation,args.output)
    elif args.score_shortlist:
        score_full_shortlist(args.output,args.score_shortlist)
    elif args.evaluate_2025:
        evaluate_open_2025(args.output)
    else:
        run(args.output)
