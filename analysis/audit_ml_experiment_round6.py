"""Independent label, score and known-episode audit of the standard-warning specialist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from analysis.prepare_ml_experiment import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_year(year: int, root: Path) -> dict:
    if year == 2024:
        base = Path("output/ml-experiment-round5/q3-full-context-linear-2024-phase-age5-tree95-gas")
        candidate = root/"fullstream-2024-v2"
        linear = Path("output/ml-experiment-round5/q3-full-context-linear-2024-base/full_context_scores.parquet")
        tree = Path("output/ml-experiment-round5/q3-full-context-tree-2024-base/full_context_scores.parquet")
        full = 1204
    else:
        base = Path("output/ml-experiment-round5/q3-full-context-linear-2025-phase-age5-tree95-gas")
        candidate = root/"fullstream-2025-v2"
        linear = Path("output/ml-experiment-round4/q3-full-context-linear-th99544/full_context_scores.parquet")
        tree = Path("output/ml-experiment-round4/q3-full-context-tree-th992-v2/full_context_scores.parquet")
        full = 2142
    specialist_dir = root/"static-2024-v4"
    specialist = specialist_dir/f"all_shortlist_scores_{year}.parquet"
    score_report = read(specialist_dir/f"all_shortlist_scores_{year}_report.json")
    selection = read(specialist_dir/"frozen_2024_selection.json")
    report = read(candidate/"report.json")
    if (report["year"]!=year or report["model_name"]!="linear"
            or report["standard_specialist_threshold"]!=selection["threshold"]
            or report["standard_specialist_score_sha256"]!=sha256(specialist)
            or score_report["score_sha256"]!=sha256(specialist)
            or score_report["model_sha256"]!=selection["model_sha256"]
            or report["full_context_scores_sha256"]!=sha256(linear)
            or report["veto_tree_score_sha256"]!=sha256(tree)
            or report["standard_specialist_gated_decisions"]<1):
        raise AssertionError(f"{year}: frozen specialist provenance differs")
    old = pd.read_parquet(base/"candidate_warnings.parquet")
    new = pd.read_parquet(candidate/"candidate_warnings.parquet")
    old_hits = set(old.loc[old.outcome.eq("matched_known_episode"),"target_episode_id"])
    new_hits = set(new.loc[new.outcome.eq("matched_known_episode"),"target_episode_id"])
    metric = report["candidate"]
    if (not old_hits <= new_hits or len(new)>=len(old)
            or len(new)!=metric["warnings"] or len(new_hits)!=metric["matched_known_episodes"]
            or metric["full_recall_lower_bound"]!=len(new_hits)/full
            or metric["precision_lower_bound"]!=len(new_hits)/len(new)
            or new.duplicated(["channel_id","prediction_time"]).any()):
        raise AssertionError(f"{year}: no-loss Precision improvement differs")
    q3 = read(Path("output/q3-a-coverage-reentry-20260927-v4/report.json"))
    labels=[]
    for month in range(1,13):
        path=Path(f"output/r3-b-full-months-20260925-v2/"
                  f"year={year}/month={month:02d}/registered_forecast_labels.parquet")
        expected=next(item["sha256"] for item in q3["source_b3_labels"]
                      if item["month"]==f"{year}-{month:02d}")
        if sha256(path)!=expected:
            raise AssertionError("B3 source labels changed")
        labels.append(str(path))
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        rows,bad_linear,bad_tree,bad_specialist,bad_age,bad_reset,bad_label = db.execute('''
            SELECT COUNT(*),
            COUNT(*) FILTER(WHERE l.channel_id IS NULL OR l.sensor_type!=w.sensor_type
                OR l.score<? OR (w.sensor_type='Состояние фазы' AND l.score<?)),
            COUNT(*) FILTER(WHERE t.channel_id IS NULL OR t.sensor_type!=w.sensor_type
                OR (w.sensor_type!='Газовый датчик' AND t.score<?)),
            COUNT(*) FILTER(WHERE w.reason='standard_24h_warning'
                AND w.sensor_type IN ('Датчик дыма','Состояние фазы')
                AND (s.channel_id IS NULL OR s.sensor_type!=w.sensor_type
                     OR s.specialist_score<?)),
            COUNT(*) FILTER(WHERE w.sensor_type='Датчик дыма' AND
                (w.last_explicit_normal_at IS NULL OR
                 w.prediction_time-w.last_explicit_normal_at>INTERVAL '5 hours')),
            COUNT(*) FILTER(WHERE w.reason='recovered_episode_reset' AND NOT(
                w.previous_warning_at<w.observed_onset_at
                AND w.observed_onset_at<w.observed_recovery_at
                AND w.observed_recovery_at<=w.prediction_time)),
            COUNT(*) FILTER(WHERE b.channel_id IS NULL OR b.sensor_type!=w.sensor_type
                OR b.target IS DISTINCT FROM w.target
                OR b.target_episode_id IS DISTINCT FROM w.target_episode_id
                OR b.label_available_at IS DISTINCT FROM w.label_available_at)
            FROM read_parquet(?) w LEFT JOIN read_parquet(?) l USING(channel_id,prediction_time)
            LEFT JOIN read_parquet(?) t USING(channel_id,prediction_time)
            LEFT JOIN read_parquet(?) s USING(channel_id,prediction_time)
            LEFT JOIN read_parquet(?,hive_partitioning=false) b
            USING(channel_id,prediction_time)''',
            [report["threshold"],report["type_thresholds"]["Состояние фазы"],
             report["veto_tree_threshold"],selection["threshold"],
             str(candidate/"candidate_warnings.parquet"),str(linear),str(tree),
             str(specialist),labels]).fetchone()
    if rows!=len(new) or any((bad_linear,bad_tree,bad_specialist,bad_age,bad_reset,bad_label)):
        raise AssertionError(f"{year}: independent warning/source parity differs")
    return {"year":year,"baseline_warnings":len(old),"specialist_warnings":len(new),
            "old_known_episodes":len(old_hits),"new_known_episodes":len(new_hits),
            "lost_known_episodes":len(old_hits-new_hits),
            "precision_lower_bound":metric["precision_lower_bound"],
            "full_recall_lower_bound":metric["full_recall_lower_bound"],
            "unknown_outcome_warnings":metric["unknown_outcome_warnings"]}


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    root=Path("output/ml-experiment-round6")
    checks=[audit_year(year,root) for year in (2024,2025)]
    result={"status":"independent_standard_specialist_fullstream_parity_passed",
            "checks":checks,"test_2026_read":False,"data_2021_read":False}
    output.mkdir(parents=True)
    (output/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),
                                      encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,
                        default=Path("output/ml-experiment-round6/independent-audit-v1"))
    run(parser.parse_args().output)
