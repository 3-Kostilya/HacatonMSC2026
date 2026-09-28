"""Causal persistence experiments on fixed pre-2024 models; selection on 2024.

The current score is available at prediction_time. All other scores in each
aggregate are earlier, and complete consecutive hourly history is required.
Missing hours never carry a stale score forward. Targets/cooldown are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.ml_experiment_eval import EVALUATION_VERSION, PreparedEvaluation
from ml.forecast.alert_eval import evaluate_alerts


META = ["channel_id", "prediction_time", "sensor_type", "target",
        "target_episode_id", "label_available_at"]
SOURCE_SPECS = {
    "pooled": {"directory": "pooled", "column": "score_engineered_episode",
               "tune": "scores_tune.parquet", "validation": "scores_validation.parquet",
               "model": "engineered_episode.cbm"},
    "linear": {"directory": "linear", "column": "score_log_episode_sqrt",
               "tune": "tune_scores.parquet", "validation": "validation_scores.parquet",
               "model": "log_episode_sqrt.joblib"},
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def literal(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def compact(metric):
    return {key: value for key, value in metric.items()
            if key not in {"matched_episode_ids", "by_type"}}


def temporal_projection_sql(source_relation: str, metadata: list[str] | None = None) -> str:
    """Project causal aggregates from a relation containing two raw probabilities."""
    logits = ",".join(
        f"CASE WHEN isfinite(score_{kind}_raw) THEN "
        f"ln(greatest(least(CAST(score_{kind}_raw AS DOUBLE),1-1e-6),1e-6)/"
        f"(1-greatest(least(CAST(score_{kind}_raw AS DOUBLE),1-1e-6),1e-6))) END AS {kind}_logit"
        for kind in SOURCE_SPECS)
    aggregates = []
    projection = ",".join((metadata or META) + [f"score_{kind}_raw" for kind in SOURCE_SPECS])
    for kind in SOURCE_SPECS:
        for hours in [2, 3, 6]:
            aggregates.append(
                f"CASE WHEN count({kind}_logit) OVER w{hours}={hours} "
                f"THEN CAST(avg({kind}_logit) OVER w{hours} AS FLOAT) END "
                f"AS score_{kind}_meanlogit_{hours}h")
        for hours in [2, 3]:
            aggregates.append(
                f"CASE WHEN count({kind}_logit) OVER w{hours}={hours} "
                f"THEN CAST(min(score_{kind}_raw) OVER w{hours} AS FLOAT) END "
                f"AS score_{kind}_min_{hours}h")
        aggregates.append(
            f"CASE WHEN prediction_time-lag(prediction_time) OVER ordered=INTERVAL '1 HOUR' "
            f"THEN CAST({kind}_logit+0.5*({kind}_logit-lag({kind}_logit) OVER ordered) AS FLOAT) "
            f"END AS score_{kind}_rising1h")
    windows = ["ordered AS (PARTITION BY channel_id ORDER BY prediction_time)"]
    windows.extend(f"w{hours} AS (PARTITION BY channel_id ORDER BY prediction_time "
                   f"RANGE BETWEEN INTERVAL '{hours-1} HOUR' PRECEDING AND CURRENT ROW)"
                   for hours in [2, 3, 6])
    return (f"WITH raw AS ({source_relation}),logits AS (SELECT *,{logits} FROM raw) "
            f"SELECT {projection},{','.join(aggregates)} FROM logits WINDOW {','.join(windows)}")


def verify_metadata(db, left: Path, right: Path, expected_rows: int) -> None:
    differences = " OR ".join(f'a."{name}" IS DISTINCT FROM b."{name}"' for name in META[2:])
    left_n, right_n = [db.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
                       for path in [left, right]]
    if left_n != expected_rows or right_n != expected_rows:
        raise AssertionError(f"source/output row count differs: {left_n}/{right_n}/{expected_rows}")
    bad = db.execute(f"""SELECT COUNT(*) FROM read_parquet(?) a
        FULL OUTER JOIN read_parquet(?) b USING(channel_id,prediction_time)
        WHERE a.channel_id IS NULL OR b.channel_id IS NULL OR {differences}""",
                     [str(left), str(right)]).fetchone()[0]
    if bad:
        raise AssertionError(f"score metadata differs in {bad} rows")


def derive_fold(db, root: Path, fold: str, output: Path, expected_rows: int) -> dict:
    sources = {kind: root / spec["directory"] / spec[fold] for kind, spec in SOURCE_SPECS.items()}
    verify_metadata(db, sources["pooled"], sources["linear"], expected_rows)
    # Explicit, unique hourly keys make a window count of h prove h consecutive
    # observations in an h-hour range ending at the current available score.
    invalid = db.execute("""SELECT COUNT(*) FROM (SELECT channel_id,prediction_time,COUNT(*) n
        FROM read_parquet(?) GROUP BY channel_id,prediction_time)
        WHERE n<>1 OR prediction_time<>date_trunc('hour',prediction_time)""",
                         [str(sources["pooled"])]).fetchone()[0]
    if invalid:
        raise AssertionError("causal persistence requires unique hourly prediction keys")
    select = ",".join(f'a."{name}"' for name in META)
    select += f',a."{SOURCE_SPECS["pooled"]["column"]}" AS score_pooled_raw'
    select += f',b."{SOURCE_SPECS["linear"]["column"]}" AS score_linear_raw'
    relation = (f"SELECT {select} FROM read_parquet({literal(sources['pooled'])}) a "
                f"JOIN read_parquet({literal(sources['linear'])}) b USING(channel_id,prediction_time)")
    projection = temporal_projection_sql(relation)
    db.execute(f"COPY ({projection}) TO {literal(output)} "
               "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)")
    verify_metadata(db, sources["pooled"], output, expected_rows)
    expected_year = 2024 if fold == "tune" else 2025
    forbidden = db.execute("SELECT COUNT(*) FROM read_parquet(?) WHERE year(prediction_time)<>? "
                           "OR target NOT IN(0,1)", [str(output), expected_year]).fetchone()[0]
    if forbidden:
        raise AssertionError("output fold contains forbidden year or unknown label")
    return {"rows": expected_rows, "metadata_mismatches": 0,
            "sources_sha256": {kind: digest(path) for kind, path in sources.items()},
            "output_sha256": digest(output)}


def native_threshold(source: Path, column: str, threshold: float) -> float:
    kind = pq.ParquetFile(source).schema_arrow.field(column).type
    return float(np.float32(threshold)) if pa.types.is_float32(kind) else threshold


def threshold_curve(db, source: Path, column: str, full_count: int, points: int = 45):
    quantiles = (1-np.geomspace(.20,.00001,points-1)).tolist()
    thresholds, maximum = db.execute(f'SELECT quantile_cont("{column}",?),max("{column}") '
                                     f'FROM read_parquet(?) WHERE isfinite("{column}")',
                                     [quantiles, str(source)]).fetchone()
    if maximum is None:
        raise ValueError(f"no available scores for {column}")
    thresholds = sorted(set(float(value) for value in thresholds))
    thresholds.append(float(maximum+max(1.0,abs(maximum))*1e-6))
    available, days = db.execute("SELECT COUNT(DISTINCT CASE WHEN target=1 THEN target_episode_id END),"
                                "COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE))) "
                                "FROM read_parquet(?)", [str(source)]).fetchone()
    columns = ",".join(f'"{name}"' for name in META)
    frame = db.execute(f'SELECT {columns},"{column}" FROM read_parquet(?) '
                       f'WHERE isfinite("{column}") AND "{column}">=?',
                       [str(source), native_threshold(source, column, thresholds[0])]).fetch_df()
    prepared = PreparedEvaluation(frame, full_count, days)
    curve = []
    for threshold in thresholds:
        metric = compact(prepared.evaluate(column, threshold))
        metric["eligible_positive_episodes"] = available
        metric["available_episode_recall"] = metric["matched_episodes"]/available if available else 0
        curve.append(metric)
    return curve


def canonical_metric(db, source: Path, column: str, threshold: float, full_count: int):
    columns = ",".join(f'"{name}"' for name in META)
    available, days = db.execute("SELECT COUNT(DISTINCT CASE WHEN target=1 THEN target_episode_id END),"
                                "COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE))) "
                                "FROM read_parquet(?)", [str(source)]).fetchone()
    frame = db.execute(f'SELECT {columns},"{column}" AS catboost_score FROM read_parquet(?) '
                       f'WHERE isfinite("{column}") AND "{column}">=?',
                       [str(source), native_threshold(source,column,threshold)]).fetch_df()
    metric, alerts = evaluate_alerts(frame,"catboost_score",threshold,channel_days=days)
    fast = PreparedEvaluation(frame,full_count,days).evaluate("catboost_score",threshold)
    for key in ["emitted_warnings","matched_episodes","unmatched_warnings",
                "suppressed_positive_score_rows","duplicate_episode_warnings","median_lead_hours"]:
        if fast[key]!=metric[key]:
            raise AssertionError(f"independent production warning replay differs: {column}/{key}")
    p = metric["episode_precision"]
    r = metric["matched_episodes"]/full_count
    metric.update({"score_column": column,"full_episode_count": full_count,
                   "eligible_positive_episodes": available,"available_episode_recall":
                       metric["matched_episodes"]/available if available else 0,
                   "episode_recall": r,"full_episode_recall": r,
                   "full_episode_f1": 2*p*r/(p+r) if p+r else 0,
                   "episode_f1": 2*p*r/(p+r) if p+r else 0,
                   "evaluation_version": EVALUATION_VERSION})
    return metric, alerts


def warning_audit(alerts):
    by_type = {}
    for kind, group in alerts.groupby("sensor_type",dropna=False):
        tp = group.outcome.eq("matched_episode")
        by_type[str(kind)] = {"warnings": len(group),"TP": int(tp.sum()),"FP": int((~tp).sum()),
                             "median_TP_lead_hours": float(group.loc[tp,"lead_hours"].median())
                                 if tp.any() else None}
    return {"by_type": by_type,
            "TP_lead_hours": {str(q): float(alerts.loc[alerts.outcome.eq("matched_episode"),
                                                      "lead_hours"].quantile(q))
                               for q in [.1,.5,.9]} if alerts.outcome.eq("matched_episode").any() else {},
            "interpretation": "Persistence uses only currently available and prior contiguous hourly scores. "
                "Requiring persistence can remove transient false warnings but delays the first warning "
                "and can miss short pre-event windows; type/lead comparisons are descriptive only."}


def run_masked_rejected(root: Path, output: Path, points: int = 45):
    if output.exists():
        raise FileExistsError(output)
    if not 40 <= points <= 80:
        raise ValueError("40 to 80 quantile points are required")
    manifest = json.loads((root/"data/manifest.json").read_text(encoding="utf-8"))
    if manifest["full_episode_count"]["tune"]!=1204 or manifest["full_episode_count"]["validation"]!=2142:
        raise AssertionError("fixed full B3 denominators differ")
    output.mkdir(parents=True)
    started = time.monotonic()
    source_models = {kind: {"model": spec["model"],"sha256": digest(root/spec["directory"]/spec["model"]),
                            "fit_end_exclusive": "2024-01-01"} for kind,spec in SOURCE_SPECS.items()}
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory=?",[str(output/"duckdb-temp")])
        tune_file = output/"tune_scores.parquet"
        tune_provenance = derive_fold(db,root,"tune",tune_file,manifest["row_stats"]["tune"]["rows"])
        names = [name for name in pq.ParquetFile(tune_file).schema_arrow.names if name.startswith("score_")]
        curves,selections = {},{}
        for column in names:
            curve = threshold_curve(db,tune_file,column,1204,points)
            curves[column] = curve
            selections[column] = max(curve,key=lambda row:(row["full_episode_f1"],row["episode_precision"]))
            best = selections[column]
            print(f"2024 {column}: P={best['episode_precision']:.4f} R={best['full_episode_recall']:.4f} "
                  f"F1={best['full_episode_f1']:.4f}",flush=True)
        winner = max(selections,key=lambda column:selections[column]["full_episode_f1"])
        selection = {"selection_year": 2024,"selected_column": winner,"selections": selections,
                     "threshold_points_requested": points,"source_models": source_models,
                     "tune_provenance": tune_provenance,"evaluation_version": EVALUATION_VERSION,
                     "policy": "24h canonical channel cooldown; one TP per distinct assigned episode",
                     "contiguity": "Each h-hour aggregate requires exactly h hourly scores including the current score.",
                     "score_aggregation_uses_targets": False}
        (output/"selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
        (output/"curves.json").write_text(json.dumps(curves,ensure_ascii=False,indent=2),encoding="utf-8")
        # No 2025 source predictions or labels have been read before this point.
        validation_file = output/"validation_scores.parquet"
        validation_provenance = derive_fold(db,root,"validation",validation_file,
                                            manifest["row_stats"]["validation"]["rows"])
        transfers,audits = {},{}
        for column,chosen in selections.items():
            metric,alerts = canonical_metric(db,validation_file,column,chosen["threshold"],2142)
            transfers[column] = metric
            audits[column] = warning_audit(alerts)
            print(f"2025 FROZEN {column}: P={metric['episode_precision']:.4f} "
                  f"R={metric['full_episode_recall']:.4f} F1={metric['full_episode_f1']:.4f}",flush=True)
        canonical_tune,_ = canonical_metric(db,tune_file,winner,selections[winner]["threshold"],1204)
        for key in ["emitted_warnings","matched_episodes","suppressed_positive_score_rows",
                    "episode_precision","full_episode_recall","full_episode_f1","median_lead_hours"]:
            if canonical_tune[key]!=selections[winner][key]:
                raise AssertionError(f"independent canonical tune replay differs: {key}")
        report = {"status": "REJECTED_RETROSPECTIVE_LABEL_MASK_CONTEXT",
                  "promote_aggregates": False,
                  "selected_column": winner,"tune": selections,"validation": transfers,
                  "selected_validation": transfers[winner],"source_models": source_models,
                  "tune_provenance": tune_provenance,"validation_provenance": validation_provenance,
                  "independent_canonical_tune_replay": "passed","warnings_audit": audits,
                  "elapsed_seconds": time.monotonic()-started,"test_2026_read": False,"data_2021_read": False,
                  "limitations": ["2025 was opened in earlier experiments and is exploratory temporal transfer.",
                                  "Temporal scoring starts with empty history at each fold boundary.",
                                  "A missing source hour makes a strict persistence score unavailable."]}
        (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def score_full_context(root: Path, package: Path, year: int, destination: Path,
                       models: dict, base: list[str], linear_spec: dict) -> dict:
    """Score ALL eligible feature hours. No labels or retrospective masks enter."""
    import pandas as pd

    from analysis.ml_experiment_features import engineered_input
    from analysis.ml_experiment_linear import transform

    writer = None
    count = 0
    months = []
    started = time.monotonic()
    try:
        for month in range(1,13):
            source = package/f"year={year}/month={month:02d}/model_features.parquet"
            parquet = pq.ParquetFile(source)
            columns = list(dict.fromkeys(["channel_id","prediction_time",*base]))
            if {"target","target_episode_id","label_available_at"}&set(columns):
                raise AssertionError("target columns entered full-context scoring")
            monthly = 0
            for batch in parquet.iter_batches(batch_size=100000,columns=columns):
                frame = batch.to_pandas()
                if not frame.prediction_time.dt.year.eq(year).all():
                    raise AssertionError("full context contains an excluded year")
                out = frame[["channel_id","prediction_time","sensor_type"]].copy()
                matrix = engineered_input(frame,base,engineered=True)
                if list(matrix.columns)!=models["pooled"].feature_names_:
                    raise AssertionError("original pooled feature order changed")
                out["score_pooled_raw"] = models["pooled"].predict_proba(matrix,thread_count=2)[:,1].astype("float32")
                del matrix
                matrix = transform(frame,linear_spec["names"],linear_spec["name"],linear_spec["limits"])
                out["score_linear_raw"] = models["linear"].predict_proba(matrix)[:,1].astype("float32")
                if not np.isfinite(out[["score_pooled_raw","score_linear_raw"]].to_numpy()).all():
                    raise AssertionError("original model produced unavailable probability")
                table = pa.Table.from_pandas(out,preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(destination,table.schema,compression="zstd")
                writer.write_table(table)
                count += len(frame)
                monthly += len(frame)
                del frame,out,matrix,table
            if monthly!=parquet.metadata.num_rows:
                raise AssertionError("eligible source feature rows were filtered")
            months.append({"month": f"{year}-{month:02d}","eligible_rows": monthly})
            print(f"FULL CONTEXT {year}-{month:02d}: {monthly} rows; "
                  f"cumulative {count},elapsed {time.monotonic()-started:.1f}s",flush=True)
        del pd  # No label frame is materialized in the scoring path.
    finally:
        if writer is not None:
            writer.close()
    return {"eligible_context_rows": count,"feature_months": months,
            "target_columns_read_during_scoring": False,"source_label_mask_used": False,
            "elapsed_seconds": time.monotonic()-started,"raw_context_sha256": digest(destination)}


def derive_full_context_fold(db,root: Path,package: Path,fold: str,output: Path,
                             models: dict,base: list[str],linear_spec: dict) -> dict:
    year = 2024 if fold=="tune" else 2025
    raw_file = output/f"{fold}_all_eligible_raw.parquet"
    stats = score_full_context(root,package,year,raw_file,models,base,linear_spec)
    # Windows are computed BEFORE any evaluation-label join. The context has
    # three metadata fields and raw probabilities, and no targets whatsoever.
    aggregate_file = output/f"{fold}_all_eligible_aggregates.parquet"
    projection = temporal_projection_sql(f"SELECT * FROM read_parquet({literal(raw_file)})",
                                         metadata=["channel_id","prediction_time","sensor_type"])
    db.execute(f"COPY ({projection}) TO {literal(aggregate_file)} "
               "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)")
    raw_n,agg_n = [db.execute("SELECT COUNT(*) FROM read_parquet(?)",[str(path)]).fetchone()[0]
                   for path in [raw_file,aggregate_file]]
    if raw_n!=agg_n or raw_n!=stats["eligible_context_rows"]:
        raise AssertionError("context aggregation altered the eligible feature population")
    label_file = root/"data"/f"{fold}.parquet"
    destination = output/f"{fold}_scores.parquet"
    names = [name for name in pq.ParquetFile(aggregate_file).schema_arrow.names if name.startswith("score_")]
    columns = ",".join([f'd."{name}"' for name in META]+[f'a."{name}"' for name in names])
    db.execute(f"COPY (SELECT {columns} FROM read_parquet({literal(label_file)}) d "
               f"JOIN read_parquet({literal(aggregate_file)}) a USING(channel_id,prediction_time) "
               "WHERE d.sensor_type IS NOT DISTINCT FROM a.sensor_type) "
               f"TO {literal(destination)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)")
    expected = db.execute("SELECT COUNT(*) FROM read_parquet(?)",[str(label_file)]).fetchone()[0]
    verify_metadata(db,label_file,destination,expected)
    # Reproduce the historical raw controls exactly on their accepted keys.
    for kind,spec in SOURCE_SPECS.items():
        original = root/spec["directory"]/spec[fold]
        bad = db.execute(f'SELECT COUNT(*) FROM read_parquet(?) a JOIN read_parquet(?) b '
                         f'USING(channel_id,prediction_time) WHERE a."score_{kind}_raw" '
                         f'IS DISTINCT FROM b."{spec["column"]}"',
                         [str(destination),str(original)]).fetchone()[0]
        if bad:
            raise AssertionError(f"target-free scoring changed original {kind} controls in {bad} rows")
    stats.update({"evaluation_rows": expected,"evaluation_metadata_mismatches": 0,
                  "raw_control_score_mismatches": 0,"aggregation_precedes_label_join": True,
                  "evaluation_scores_sha256": digest(destination),
                  "aggregate_context_sha256": digest(aggregate_file)})
    return stats


def run(root: Path,output: Path,points: int=45):
    """Deployable causal score context, then fixed-population retrospective evaluation."""
    import joblib
    from catboost import CatBoostClassifier

    if output.exists():
        raise FileExistsError(output)
    if not 40<=points<=80:
        raise ValueError("40 to 80 quantile points are required")
    manifest = json.loads((root/"data/manifest.json").read_text(encoding="utf-8"))
    if manifest["full_episode_count"]["tune"]!=1204 or manifest["full_episode_count"]["validation"]!=2142:
        raise AssertionError("full B3 denominators differ")
    package = Path("output/q2-a-full-sparse-20260926-v5")
    if digest(package/"manifest.json")!=manifest["source_hashes"]["q2"]:
        raise AssertionError("Q2 source manifest changed")
    base = manifest["all_features"]
    original = json.loads((root/"linear/frozen_selection_canonical_v2.json").read_text(encoding="utf-8"))
    linear_spec = next(item for item in original["variants"] if item["name"]=="log_episode_sqrt")
    models = {"pooled": CatBoostClassifier(thread_count=2),
              "linear": joblib.load(root/"linear/log_episode_sqrt.joblib")}
    models["pooled"].load_model(str(root/"pooled/engineered_episode.cbm"))
    source_models = {kind:{"model": spec["model"],"sha256": digest(root/spec["directory"]/spec["model"]),
                           "fit_end_exclusive": "2024-01-01"} for kind,spec in SOURCE_SPECS.items()}
    output.mkdir(parents=True)
    started = time.monotonic()
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory=?",[str(output/"duckdb-temp")])
        tune_provenance = derive_full_context_fold(db,root,package,"tune",output,models,base,linear_spec)
        tune_file = output/"tune_scores.parquet"
        names = [name for name in pq.ParquetFile(tune_file).schema_arrow.names if name.startswith("score_")]
        curves,selections = {},{}
        for column in names:
            curve = threshold_curve(db,tune_file,column,1204,points)
            curves[column] = curve
            selections[column] = max(curve,key=lambda row:(row["full_episode_f1"],row["episode_precision"]))
            best = selections[column]
            print(f"CAUSAL2024 {column}: P={best['episode_precision']:.4f} "
                  f"R={best['full_episode_recall']:.4f} F1={best['full_episode_f1']:.4f}",flush=True)
        winner = max(selections,key=lambda column:selections[column]["full_episode_f1"])
        selection = {"selection_year":2024,"selected_column":winner,"selections":selections,
                     "source_models":source_models,"tune_provenance":tune_provenance,
                     "evaluation_version":EVALUATION_VERSION,"threshold_points_requested":points,
                     "score_context":"ALL eligible Q2 feature hours, without target or future-label filtering",
                     "aggregation_precedes_label_join":True,"cooldown_hours":24,
                     "score_aggregation_uses_targets":False}
        (output/"selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
        (output/"curves.json").write_text(json.dumps(curves,ensure_ascii=False,indent=2),encoding="utf-8")
        # The 2025 feature stream and its label join are first opened only now.
        validation_provenance = derive_full_context_fold(db,root,package,"validation",output,models,base,linear_spec)
        validation_file = output/"validation_scores.parquet"
        transfers,audits = {},{}
        for column,chosen in selections.items():
            metric,alerts = canonical_metric(db,validation_file,column,chosen["threshold"],2142)
            transfers[column] = metric
            audits[column] = warning_audit(alerts)
            print(f"CAUSAL2025 FROZEN {column}: P={metric['episode_precision']:.4f} "
                  f"R={metric['full_episode_recall']:.4f} F1={metric['full_episode_f1']:.4f}",flush=True)
        canonical_tune,_ = canonical_metric(db,tune_file,winner,selections[winner]["threshold"],1204)
        for key in ["emitted_warnings","matched_episodes","suppressed_positive_score_rows",
                    "episode_precision","full_episode_recall","full_episode_f1","median_lead_hours"]:
            if canonical_tune[key]!=selections[winner][key]:
                raise AssertionError(f"independent canonical2024 replay differs: {key}")
        report = {"status":"CAUSAL_ALL_ELIGIBLE_CONTEXT_2024_SELECTION_FROZEN_OPEN2025_TRANSFER",
                  "selected_column":winner,"tune":selections,"validation":transfers,
                  "selected_validation":transfers[winner],"source_models":source_models,
                  "tune_provenance":tune_provenance,"validation_provenance":validation_provenance,
                  "independent_canonical_replay":"all frozen2025 policies plus2024 winner passed",
                  "warnings_audit":audits,"elapsed_seconds":time.monotonic()-started,
                  "test_2026_read":False,"data_2021_read":False,
                  "aggregation_precedes_label_join":True,"source_label_mask_used":False,
                  "limits":["2025 is previously opened exploratory temporal transfer.",
                            "Context starts empty at each fold boundary; no synthetic missing-hour scores.",
                            "The earlier masked-context experiment is rejected and not a candidate."]}
        (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("output/ml-experiment"))
    parser.add_argument("--output",type=Path,default=Path("output/ml-experiment-round2/temporal-full-context"))
    parser.add_argument("--points",type=int,default=45)
    args = parser.parse_args()
    run(args.root,args.output,args.points)
