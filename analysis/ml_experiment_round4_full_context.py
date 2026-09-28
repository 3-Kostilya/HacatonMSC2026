"""Q3 + recovery-reset sensitivity with every eligible score hour, including unknown labels.

This is not a replacement for the binary-label retrospective experiment: unknown
warning outcomes stay unknown, but their causal cooldown effect is included.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

from catboost import CatBoostClassifier
import duckdb
import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_q2_recovery_reset_a import stream_groups
from analysis.ml_experiment_features import engineered_input
from analysis.ml_experiment_linear import transform
from analysis.prepare_ml_experiment import sha256
from analysis.q2_recovery_reset_a import COOLDOWN, RecoveryResetPolicy
from stage1.state_labeling.operational import segment_at


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def literal(path: Path):
    return "'" + str(path).replace("'", "''") + "'"


def score_delta(package: Path, destination: Path, predict, year: int):
    writer = None
    count = 0
    for month in [f"{i:02d}" for i in range(1, 13)]:
        folder = package / f"year={year}/month={month}"
        manifest = read(folder / "manifest.json")
        source = folder / "new_model_features.parquet"
        if sha256(source) != manifest["files"]["new_model_features.parquet"]["sha256"]:
            raise AssertionError("Q3 source features changed")
        for batch in pq.ParquetFile(source).iter_batches(batch_size=70000):
            frame = batch.to_pandas()
            meta = frame[["channel_id", "prediction_time", "sensor_type"]].copy()
            meta["score"] = predict(frame).astype("float32")
            table = pa.Table.from_pandas(meta, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(frame)
        print("scored all Q3 candidate hours",month,flush=True)
    if writer is None:
        raise AssertionError("Q3 delta is empty")
    writer.close()
    return count


def assess(db, rows: list[dict], label_files: list[str], full: int):
    emitted = pd.DataFrame(rows)
    emitted = emitted.loc[emitted.warning_emitted].copy()
    db.register("emitted", emitted)
    joined = db.execute('''SELECT e.*,l.target,l.target_episode_id,l.label_available_at,
        l.sensor_type AS label_sensor_type FROM emitted e
        LEFT JOIN read_parquet(?,hive_partitioning=false) l
        USING(channel_id,prediction_time)
        ORDER BY e.prediction_time,e.channel_id''', [label_files]).fetch_df()
    db.unregister("emitted")
    if (len(joined) != len(emitted) or joined.duplicated(["channel_id", "prediction_time"]).any()
            or joined.label_sensor_type.isna().any()
            or not joined.sensor_type.eq(joined.label_sensor_type).all()):
        raise AssertionError("complete B3 label key/type join changed")
    matched = set()
    outcomes = []
    for row in joined.itertuples(index=False):
        if pd.isna(row.target):
            outcomes.append("unknown_target")
        elif int(row.target) == 1:
            if pd.isna(row.target_episode_id) or not timedelta(0) < (
                    row.label_available_at - row.prediction_time) <= timedelta(hours=24):
                raise AssertionError("positive label horizon differs")
            if row.target_episode_id in matched:
                outcomes.append("duplicate_known_episode")
            else:
                outcomes.append("matched_known_episode")
                matched.add(row.target_episode_id)
        elif int(row.target) == 0:
            outcomes.append("known_no_target")
        else:
            raise AssertionError("invalid assigned target value")
    joined["outcome"] = outcomes
    unknown = outcomes.count("unknown_target")
    precision_lower = len(matched) / len(joined) if len(joined) else 0.0
    metric = {"warnings": len(joined), "matched_known_episodes": len(matched),
              "full_recall_lower_bound": len(matched)/full,
              "full_recall_loose_upper_bound": min(full,len(matched)+unknown)/full,
              "unknown_outcome_warnings": unknown,
              "known_no_target_warnings": outcomes.count("known_no_target"),
              "duplicate_known_episode_warnings": outcomes.count("duplicate_known_episode"),
              "precision_lower_bound": precision_lower,
              "precision_known_evaluable": len(matched)/(len(joined)-unknown)
                if len(joined)>unknown else 0.0,
              "precision_loose_upper_bound": min(full,len(matched)+unknown)/len(joined)
                if len(joined) else 0.0}
    return metric,joined


def standard_warning_available(policy: RecoveryResetPolicy, channel: str, at) -> bool:
    """Only previously emitted warnings can determine a standard 24h opportunity."""
    state = policy.warnings.get(channel)
    return bool(state is None or state.segment != segment_at(at)
                or state.last_warning_at is None
                or at >= state.last_warning_at + COOLDOWN)


def run(output: Path, threshold: float = 0.992, model_name: str = "tree",
        year: int = 2025, type_thresholds: dict[str, float] | None = None,
        score_source: Path | None = None, smoke_max_normal_age_hours: float | None = None,
        veto_score_source: Path | None = None, veto_threshold: float | None = None,
        veto_exempt_gas: bool = False,
        standard_specialist_scores: Path | None = None):
    if output.exists():
        raise FileExistsError(output)
    if model_name not in {"tree", "linear"}:
        raise ValueError("only frozen tree and linear controls are supported")
    if year not in {2024, 2025}:
        raise ValueError("only 2024 tuning and open 2025 evaluation are supported")
    type_thresholds = type_thresholds or {}
    if not all(0 <= value <= 1 for value in type_thresholds.values()):
        raise ValueError("type thresholds must be probabilities")
    if smoke_max_normal_age_hours is not None and smoke_max_normal_age_hours <= 0:
        raise ValueError("normal age limit must be positive")
    if (veto_score_source is None) != (veto_threshold is None):
        raise ValueError("veto score source and threshold must be provided together")
    if veto_threshold is not None and not 0 <= veto_threshold <= 1:
        raise ValueError("veto threshold must be a probability")
    fold = "tune" if year == 2024 else "validation"
    full_episodes = 1204 if year == 2024 else 2142
    q2 = Path("output/q2-a-full-sparse-20260926-v5")
    q3 = Path("output/q3-a-coverage-reentry-20260927-v4")
    m1 = Path("output/milestone1/full_20260922")
    b3 = Path("output/r3-b-full-months-20260925-v2")
    context = Path(f"output/ml-experiment-round2/temporal-full-context/{fold}_all_eligible_raw.parquet")
    temporal = read(context.parent / "report.json")
    provenance = temporal[f"{fold}_provenance"]
    if (sha256(context) != provenance["raw_context_sha256"]
            or temporal["source_label_mask_used"] or not temporal["aggregation_precedes_label_join"]):
        raise AssertionError("Q2 full score context is not verified label-free")
    q3_report = read(q3 / "report.json")
    verification = read(Path("output/q3-a-coverage-reentry-verification-20260927-v4.json"))
    if (verification["status"] != "independent_delta_invariants_features_and_oracle_verified"
            or sha256(q3 / "manifest.json") != verification["source_manifest_sha256"]):
        raise AssertionError("Q3 admission source verification changed")
    if model_name == "tree":
        model_path = Path("output/ml-experiment/pooled/engineered_episode.cbm")
        if sha256(model_path) != temporal["source_models"]["pooled"]["sha256"]:
            raise AssertionError("frozen tree changed")
        features = read(Path("output/ml-experiment/data/manifest.json"))["all_features"]
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        def tree_predict(frame):
            return model.predict_proba(engineered_input(frame,features),thread_count=2)[:,1]

        predict = tree_predict
        context_column, frozen_column = "score_pooled_raw", "score_tree"
    else:
        model_path = Path("output/ml-experiment/linear/log_episode_sqrt.joblib")
        if sha256(model_path) != temporal["source_models"]["linear"]["sha256"]:
            raise AssertionError("frozen linear model changed")
        selection = read(Path("output/ml-experiment/linear/frozen_selection_canonical_v2.json"))
        if selection["selected_variant"] != "log_episode_sqrt":
            raise AssertionError("frozen linear preprocessing changed")
        spec = next(item for item in selection["variants"] if item["name"] == "log_episode_sqrt")
        model = joblib.load(model_path)
        def linear_predict(frame):
            return model.predict_proba(transform(
                frame,spec["names"],"log_episode_sqrt",spec["limits"]))[:,1]

        predict = linear_predict
        context_column, frozen_column = "score_linear_raw", "score_linear"
    output.mkdir(parents=True)
    if score_source is None:
        delta = output / "q3_all_eligible_scores.parquet"
        delta_rows = score_delta(q3,delta,predict,year)
    else:
        prior = read(score_source.parent / "report.json")
        if (prior["model_name"] != model_name or prior.get("year",2025) != year
                or prior["model_sha256"] != sha256(model_path)
                or prior["source_q2_raw_score_sha256"] != sha256(context)
                or prior["full_context_scores_sha256"] != sha256(score_source)):
            raise AssertionError("reused full score stream differs")
        delta = score_source.parent / "q3_all_eligible_scores.parquet"
        delta_rows = prior["q3_delta_rows"]
    q3_admissions,q2_admissions,labels = [],[],[]
    q2_manifest = read(q2 / "manifest.json")
    if sha256(m1 / "manifest.json") != q2_manifest["source_manifests"]["m1"]:
        raise AssertionError("M1 source changed")
    for month in range(1,13):
        folder = f"year={year}/month={month:02d}"
        qm = read(q3 / folder / "manifest.json")
        qa = q3 / folder / "new_admission.parquet"
        if sha256(qa) != qm["files"]["new_admission.parquet"]["sha256"]:
            raise AssertionError("Q3 admission changed")
        q3_admissions.append(str(qa))
        q2row = next(item for item in q2_manifest["months"] if item["month"]==f"{year}-{month:02d}")
        old = q2 / folder / "admission.parquet"
        if sha256(old) != q2row["files"]["admission.parquet"]["sha256"]:
            raise AssertionError("Q2 admission changed")
        q2_admissions.append(str(old))
        label = b3 / folder / "registered_forecast_labels.parquet"
        expected = next(item["sha256"] for item in q3_report["source_b3_labels"]
                        if item["month"]==f"{year}-{month:02d}")
        if sha256(label) != expected:
            raise AssertionError("B3 source labels changed")
        labels.append(str(label))
    raw = []
    for item in read(q2 / "report.json")["source_m1_files"]:
        source = m1 / item["file"]
        if sha256(source) != item["sha256"]:
            raise AssertionError("raw M1 source changed")
        raw.append(str(source))
    if len(raw)!=72:
        raise AssertionError("raw source months differ")
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory=?",[str(output/"duckdb-temp")])
        frozen = Path(f"output/ml-experiment-round2/coverage/frozen-models/delta_{fold}_scores.parquet")
        known_rows,bad_scores = db.execute(f'''SELECT COUNT(*),COUNT(*) FILTER(WHERE
            b.channel_id IS NULL OR a.sensor_type IS DISTINCT FROM b.sensor_type
            OR a.score IS DISTINCT FROM b.{frozen_column})
            FROM read_parquet(?) b LEFT JOIN read_parquet(?) a
            USING(channel_id,prediction_time)''',[str(frozen),str(delta)]).fetchone()
        if bad_scores or known_rows == 0:
            raise AssertionError("full-context Q3 scores differ from saved binary control")
        source = score_source or output / "full_context_scores.parquet"
        if score_source is None:
            db.execute(f'''COPY (SELECT channel_id,prediction_time,sensor_type,
                {context_column} AS score FROM read_parquet({literal(context)})
                UNION ALL SELECT * FROM read_parquet({literal(delta)}))
                TO {literal(source)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
        total,dupes = db.execute('''SELECT COUNT(*),COUNT(*)-COUNT(DISTINCT(channel_id,prediction_time))
            FROM read_parquet(?)''',[str(source)]).fetchone()
        if dupes or total!=provenance["eligible_context_rows"]+delta_rows:
            raise AssertionError("Q2+Q3 full-context keys differ")
        db.execute('''CREATE TEMP TABLE candidate_scores AS SELECT * FROM read_parquet(?)
            WHERE score >= CASE sensor_type
              WHEN 'Датчик дыма' THEN ? WHEN 'Состояние фазы' THEN ?
              WHEN 'Датчик температуры' THEN ? ELSE ? END''',
            [str(source),type_thresholds.get("Датчик дыма",threshold),
             type_thresholds.get("Состояние фазы",threshold),
             type_thresholds.get("Датчик температуры",threshold),threshold])
        if veto_score_source is not None:
            veto = read(veto_score_source.parent / "report.json")
            if (veto["model_name"] != "tree" or veto.get("year",2025) != year
                    or veto["model_sha256"] != temporal["source_models"]["pooled"]["sha256"]
                    or veto["full_context_scores_sha256"] != sha256(veto_score_source)
                    or veto["full_context_rows"] != total):
                raise AssertionError("auxiliary frozen tree score source differs")
            db.execute('''CREATE TEMP TABLE scores AS SELECT a.* FROM candidate_scores a
                JOIN read_parquet(?) v USING(channel_id,prediction_time)
                WHERE a.sensor_type=v.sensor_type
                AND (v.score>=? OR (? AND a.sensor_type='Газовый датчик'))''',
                [str(veto_score_source),veto_threshold,veto_exempt_gas])
        else:
            db.execute("CREATE TEMP TABLE scores AS SELECT * FROM candidate_scores")
        decisions = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        db.execute("CREATE TEMP TABLE selected_channels AS SELECT DISTINCT channel_id FROM scores")
        admission_query = '''SELECT channel_id,prediction_time,sensor_type,
            admission_status,admission_evidence_through,last_explicit_normal_at,
            blocking_qa_count_24h,availability_status FROM read_parquet(?,hive_partitioning=false)
            WHERE admission_status='eligible'
            UNION ALL SELECT channel_id,prediction_time,sensor_type,
            combined_status,admission_evidence_through,last_explicit_normal_at,
            blocking_qa_count_24h,availability_status FROM read_parquet(?,hive_partitioning=false)'''
        specialist_select = ",x.specialist_score" if standard_specialist_scores else ""
        specialist_join = ("LEFT JOIN read_parquet(?) x ON x.channel_id=s.channel_id "
                           "AND x.prediction_time=s.prediction_time AND x.sensor_type=s.sensor_type"
                           if standard_specialist_scores else "")
        specialist_args = []
        specialist_threshold = None
        if standard_specialist_scores is not None:
            specialist_report = read(standard_specialist_scores.parent /
                                     f"all_shortlist_scores_{year}_report.json")
            frozen_specialist = read(standard_specialist_scores.parent /
                                     "frozen_2024_selection.json")
            if (specialist_report["year"] != year
                    or specialist_report["score_sha256"] != sha256(standard_specialist_scores)
                    or specialist_report["model_sha256"] != frozen_specialist["model_sha256"]
                    or specialist_report["linear_source_sha256"] != sha256(source)
                    or veto_score_source is None
                    or specialist_report["tree_source_sha256"] != sha256(veto_score_source)
                    or specialist_report["threshold"] != frozen_specialist["threshold"]):
                raise AssertionError("standard specialist score provenance differs")
            specialist_threshold = float(specialist_report["threshold"])
            specialist_args.append(str(standard_specialist_scores))
        past = db.execute(f'''WITH a AS ({admission_query})
            SELECT s.channel_id,s.prediction_time,s.sensor_type,a.admission_status,
            a.admission_evidence_through,a.last_explicit_normal_at,
            a.blocking_qa_count_24h,a.availability_status{specialist_select} FROM scores s
            LEFT JOIN a USING(channel_id,prediction_time)
            {specialist_join}
            WHERE a.sensor_type=s.sensor_type ORDER BY s.prediction_time,s.channel_id''',
            [q2_admissions,q3_admissions,*specialist_args]).fetch_df()
        if (len(past)!=decisions or past.duplicated(["channel_id","prediction_time"]).any()
                or not past.admission_status.eq("eligible").all()):
            raise AssertionError("all score decisions lack exact protected admission")
        if standard_specialist_scores is not None and past.loc[
                past.sensor_type.isin(["Датчик дыма","Состояние фазы"]),
                "specialist_score"].isna().any():
            raise AssertionError("targeted shortlist decision lacks specialist score")
        db.execute(f'''CREATE TEMP TABLE events AS SELECT row_id,channel_id,timestamp,
            sensor_type,value_state,alarm FROM read_parquet(?,hive_partitioning=false) e
            SEMI JOIN selected_channels USING(channel_id) WHERE value_state IS NOT NULL
            AND timestamp>=TIMESTAMP '2019-01-01' AND timestamp<TIMESTAMP '{year+1}-01-01'
            AND year(timestamp)<>2021 AND split_part(replace(source,chr(92),'/'),'/',-1)=
            'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z' ''',[raw])
        reader = db.execute("SELECT * FROM events ORDER BY timestamp,channel_id,row_id").to_arrow_reader(batch_size=100000)
        groups = iter(stream_groups(reader))
        following = next(groups,None)
        old,new = RecoveryResetPolicy(allow_reset=False),RecoveryResetPolicy(allow_reset=True)
        old_rows,new_rows=[],[]
        specialist_gated_decisions = 0
        for at,frame in past.groupby("prediction_time",sort=True):
            when=at.to_pydatetime()
            while following is not None and following[0][0]<=when:
                _,events=following
                old.observe_group(events)
                new.observe_group(events)
                following=next(groups,None)
            candidates=frame.drop(columns="prediction_time").to_dict("records")
            old_candidates,new_candidates=[],[]
            for row in candidates:
                specialist_score = row.pop("specialist_score",None)
                row["blocking_qa_count_24h"]=int(row["blocking_qa_count_24h"])
                for name in ["admission_evidence_through","last_explicit_normal_at"]:
                    row[name]=row[name].to_pydatetime() if pd.notna(row[name]) else None
                row["above_threshold"] = not (
                    smoke_max_normal_age_hours is not None
                    and row["sensor_type"] == "Датчик дыма"
                    and (row["last_explicit_normal_at"] is None or
                         when-row["last_explicit_normal_at"] > timedelta(
                             hours=smoke_max_normal_age_hours))
                )
                old_row,new_row = row.copy(),row.copy()
                if (specialist_threshold is not None
                        and row["sensor_type"] in {"Датчик дыма","Состояние фазы"}
                        and specialist_score < specialist_threshold):
                    if standard_warning_available(old,row["channel_id"],when):
                        old_row["above_threshold"] = False
                    if standard_warning_available(new,row["channel_id"],when):
                        specialist_gated_decisions += int(new_row["above_threshold"])
                        new_row["above_threshold"] = False
                old_candidates.append(old_row)
                new_candidates.append(new_row)
            old_rows.extend(old.decide(when,old_candidates))
            new_rows.extend(new.decide(when,new_candidates))
        reader.close()
        if not all(row["past_state_agrees_with_admission"] for row in new_rows):
            raise AssertionError("raw state disagrees with full-context admission")
        old_metric,old_alerts=assess(db,old_rows,labels,full_episodes)
        new_metric,new_alerts=assess(db,new_rows,labels,full_episodes)
    old_alerts.to_parquet(output/"control_warnings.parquet",index=False)
    new_alerts.to_parquet(output/"candidate_warnings.parquet",index=False)
    report={"status":"ALL_ELIGIBLE_CONTEXT_UNKNOWN_OUTCOMES_PRESERVED",
            "threshold":threshold,"type_thresholds":type_thresholds,
            "smoke_max_normal_age_hours":smoke_max_normal_age_hours,
            "veto_tree_threshold":veto_threshold,
            "veto_exempt_gas":veto_exempt_gas,
            "veto_tree_score_sha256":sha256(veto_score_source) if veto_score_source else None,
            "standard_specialist_score_sha256":sha256(standard_specialist_scores)
                if standard_specialist_scores else None,
            "standard_specialist_threshold":specialist_threshold,
            "standard_specialist_gated_decisions":specialist_gated_decisions,
            "year":year,"model_name":model_name,
            "model_sha256":sha256(model_path),
            "full_context_rows":total,"q3_delta_rows":delta_rows,
            "above_threshold_decisions":decisions,"q3_known_score_parity_rows":known_rows,
            "q3_known_score_mismatches":bad_scores,"source_q2_raw_score_sha256":sha256(context),
            "source_q3_manifest_sha256":sha256(q3/"manifest.json"),
            "full_context_scores_sha256":sha256(source),
            "control":old_metric,"candidate":new_metric,
            "test_2026_read":False,"data_2021_read":False,
            "limits":["Unknown-outcome warnings participate in cooldown, but are not labeled negative.",
                      "Known episode recall is a lower bound under unknown warning outcomes.",
                      "Q3 admission and recovery-reset both require joint B/operations approval.",
                      "Open 2025 is not blind confirmation."]}
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"control":old_metric,"candidate":new_metric},ensure_ascii=False),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/ml-experiment-round4/q3-full-context-th992"))
    parser.add_argument("--threshold",type=float,default=0.992)
    parser.add_argument("--model",choices=["tree","linear"],default="tree")
    parser.add_argument("--year",type=int,choices=[2024,2025],default=2025)
    parser.add_argument("--smoke-threshold",type=float)
    parser.add_argument("--phase-threshold",type=float)
    parser.add_argument("--temp-threshold",type=float)
    parser.add_argument("--score-source",type=Path)
    parser.add_argument("--smoke-max-normal-age-hours",type=float)
    parser.add_argument("--veto-score-source",type=Path)
    parser.add_argument("--veto-threshold",type=float)
    parser.add_argument("--veto-exempt-gas",action="store_true")
    parser.add_argument("--standard-specialist-scores",type=Path)
    args=parser.parse_args()
    types={name:value for name,value in [
        ("Датчик дыма",args.smoke_threshold),
        ("Состояние фазы",args.phase_threshold),
        ("Датчик температуры",args.temp_threshold)] if value is not None}
    run(args.output,args.threshold,args.model,args.year,types,args.score_source,
        args.smoke_max_normal_age_hours,args.veto_score_source,args.veto_threshold,
        args.veto_exempt_gas,args.standard_specialist_scores)
