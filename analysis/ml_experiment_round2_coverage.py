"""Independent Q3 research intake, then evaluate unchanged models on added hours.

Not approval of physical availability or production admission. Labels, the
complete B3 denominator and 24h cooldown stay unchanged. Policy2024 is frozen
before scoring2025. Old source packages are read-only.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from catboost import CatBoostClassifier
import duckdb
import joblib
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.ml_experiment_eval import PreparedEvaluation, search_thresholds
from analysis.ml_experiment_features import engineered_input
from analysis.ml_experiment_linear import transform
from analysis.ml_experiment_metric_audit import replay
from analysis.ml_experiment_pooled import META
from analysis.prepare_ml_experiment import sha256
from stage1.state_labeling.rules import KNOWN_SENSOR_TYPES


ALLOWED_YEARS = {2019, 2020, 2022, 2023, 2024, 2025}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def literal(path):
    return "'" + str(path).replace("'", "''") + "'"


def guard_violations(db, source: Path) -> int:
    """Recompute both relaxations independently, entirely without target fields."""
    known = ",".join(literal(kind) for kind in sorted(KNOWN_SENSOR_TYPES))
    return db.execute(f"""WITH raw AS (SELECT *,
        admission_status<>'excluded' AND last_explicit_normal_at IS NOT NULL
        AND last_explicit_normal_at<=prediction_time
        AND last_explicit_normal_at>=prediction_time-INTERVAL '168 hours'
        AND admission_evidence_through IS NOT NULL
        AND admission_evidence_through<=prediction_time AND blocking_qa_count_24h=0
        AND ambiguous_seconds_24h=0 AS independently_protected,
        quality_rows_24h>0 AND last_conflict_at>prediction_time-INTERVAL '24 hours'
        AND last_conflict_at<last_explicit_normal_at
        AND (last_hard_quality_at IS NULL OR last_hard_quality_at<=prediction_time-INTERVAL '24 hours')
        AS independently_recovered FROM read_parquet(?)),
        checked AS (SELECT *,
        list_filter(admission_reasons,r->NOT(r='insufficient_history'
          AND independently_protected AND second_usable_at IS NOT NULL
          OR r='quality_exclusions_24h' AND independently_protected AND independently_recovered))
          AS remaining FROM raw)
        SELECT COUNT(*) FROM checked WHERE NOT COALESCE(independently_protected,false)
        OR combined_status<>'eligible' OR len(remaining)<>0
        OR first_usable_at IS NULL OR first_usable_at>prediction_time
        OR second_usable_at>prediction_time
        OR second_usable_at<=first_usable_at
        OR admission_status='eligible'
        OR excluded_quality_count_24h<>quality_rows_24h
        OR availability_status<>'unknown'
        OR NOT protected_ok
        OR sensor_type NOT IN ({known})
        OR list_has_any(list_transform(list_filter([first_usable_at,second_usable_at,
             last_explicit_normal_at,admission_evidence_through,last_conflict_at,last_hard_quality_at],
             t->t IS NOT NULL), t->CASE WHEN year(t)<2021 THEN 0 WHEN year(t)>=2022 THEN 1 ELSE 2 END),
             [CASE WHEN year(prediction_time)<2021 THEN 1 ELSE 0 END,2])
        OR last_conflict_at>prediction_time OR last_hard_quality_at>prediction_time
        OR year(prediction_time) NOT IN (2019,2020,2022,2023,2024,2025)""", [str(source)]).fetchone()[0]


def prepare(root: Path, package: Path) -> Path:
    data = root / "data"
    if data.exists():
        raise FileExistsError(data)
    data.mkdir(parents=True)
    base = Path("output/ml-experiment/data")
    q2 = Path("output/q2-a-full-sparse-20260926-v5")
    b3 = Path("output/r3-b-full-months-20260925-v2")
    manifest = read(base / "manifest.json")
    source = read(package / "manifest.json")
    verification = read("output/q3-a-coverage-reentry-verification-20260927-v4.json")
    if (sha256(package / "manifest.json") != verification["source_manifest_sha256"]
            or source["source_q2_manifest_sha256"] != sha256(q2 / "manifest.json")
            or manifest["source_hashes"]["b3"] != sha256(b3 / "manifest.json")):
        raise AssertionError("Q3/Q2/B3 provenance differs")
    if verification["status"] != "independent_delta_invariants_features_and_oracle_verified":
        raise AssertionError("Q3 independent source verification is incomplete")
    for name, spec in source["files"].items():
        if sha256(package/name) != spec["sha256"]:
            raise AssertionError(f"Q3 published file hash differs: {name}")
    columns = pq.ParquetFile(base / "train.parquet").schema_arrow.names
    names = manifest["all_features"]
    writers, audits = {}, []
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='1500MB'")
        try:
            for record in source["months"]:
                month = record["month"]
                year = int(month[:4])
                if year not in ALLOWED_YEARS:
                    raise AssertionError(f"unapproved month: {month}")
                folder = Path(f"year={year}/month={month[5:]}")
                m = read(package / folder / "manifest.json")
                for filename, spec in m["files"].items():
                    if sha256(package / folder / filename) != spec["sha256"]:
                        raise AssertionError(f"Q3 monthly file hash differs: {month}/{filename}")
                admission = package / folder / "new_admission.parquet"
                feature = package / folder / "new_model_features.parquet"
                bad = guard_violations(db, admission)
                if bad:
                    raise AssertionError(f"Q3 independent guarded admission violations: {month}/{bad}")
                duplicate = db.execute("""SELECT COUNT(*)-COUNT(DISTINCT(channel_id,prediction_time))
                    FROM read_parquet(?)""", [str(feature)]).fetchone()[0]
                overlap = db.execute("""SELECT COUNT(*) FROM read_parquet(?) a JOIN read_parquet(?) b
                    USING(channel_id,prediction_time)""",
                    [str(feature), str(q2 / folder / "model_features.parquet")]).fetchone()[0]
                metadata_bad = db.execute("""SELECT COUNT(*) FROM read_parquet(?) a FULL JOIN
                    read_parquet(?) f USING(channel_id,prediction_time)
                    WHERE a.channel_id IS NULL OR f.channel_id IS NULL
                    OR a.sensor_type IS DISTINCT FROM f.sensor_type""",
                    [str(admission), str(feature)]).fetchone()[0]
                if duplicate or overlap or metadata_bad:
                    raise AssertionError(f"Q3 delta keys are not a disjoint one-to-one extension: {month}")
                fold = "validation" if year == 2025 else "tune" if year == 2024 else "train"
                boundary = {"train": "2024-01-01", "tune": "2025-01-01", "validation": "2026-01-01"}[fold]
                label = b3 / folder / "registered_forecast_labels.parquet"
                expected_hash = next(x["sha256"] for x in read(package / "report.json")["source_b3_labels"]
                                     if x["month"] == month)
                if sha256(label) != expected_hash:
                    raise AssertionError(f"original B3 labels differ: {month}")
                projection = ",".join(f'f."{name}"' if name in names or name in ["channel_id", "prediction_time"]
                                      else f'l."{name}"' for name in columns)
                sample = "AND (l.target=1 OR hash(f.channel_id,f.prediction_time)%10000<200)" if fold == "train" else ""
                reader = db.execute(f"""SELECT {projection} FROM read_parquet(?) f
                    JOIN read_parquet(?) l USING(channel_id,prediction_time)
                    WHERE f.sensor_type IS NOT DISTINCT FROM l.sensor_type
                    AND l.split_status='assigned' AND l.target IN (0,1)
                    AND l.label_available_at<CAST(? AS TIMESTAMP) {sample}""",
                    [str(feature), str(label), boundary]).to_arrow_reader(batch_size=100_000)
                rows = 0
                for batch in reader:
                    if fold not in writers:
                        writers[fold] = pq.ParquetWriter(data / f"delta_{fold}.parquet", batch.schema, compression="zstd")
                    writers[fold].write_batch(batch)
                    rows += batch.num_rows
                audits.append({"month": month, "added_binary_sampled_rows": rows, "guard_violations": bad,
                               "base_overlap": overlap, "metadata_mismatches": metadata_bad})
                print("Q3 independent intake", month, "added", rows, flush=True)
        finally:
            for writer in writers.values():
                writer.close()
        stats = {}
        for fold in ["train", "tune", "validation"]:
            destination = data / f"{fold}.parquet"
            db.execute(f"""COPY (SELECT * FROM read_parquet({literal(base / f'{fold}.parquet')})
                UNION ALL SELECT * FROM read_parquet({literal(data / f'delta_{fold}.parquet')}))
                TO {literal(destination)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)""")
            n, pos, episodes, duplicates = db.execute("""SELECT COUNT(*),SUM(target),
                COUNT(DISTINCT target_episode_id) FILTER(WHERE target=1),
                COUNT(*)-COUNT(DISTINCT(channel_id,prediction_time)) FROM read_parquet(?)""",
                [str(destination)]).fetchone()
            if duplicates:
                raise AssertionError("extended fold repeats keys")
            stats[fold] = {"rows": n, "positive_hours": pos, "available_episodes": episodes}
    result = {**manifest, "schema_version": "round2-q3-research-intake-v1", "row_stats": stats,
              "source_q3_manifest_sha256": sha256(package / "manifest.json"),
              "independent_source_verification_sha256": sha256(Path("output/q3-a-coverage-reentry-verification-20260927-v4.json")),
              "independent_intake_checks": audits, "admission_policy": "combined",
              "physical_availability_approved": False, "production_admission_changed": False,
              "scope": "separate research extension, unknown outcomes remain unknown",
              "files": {fold: {"path": f"{fold}.parquet", "sha256": sha256(data / f"{fold}.parquet")}
                        for fold in ["train", "tune", "validation"]}}
    (data / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Q3 admitted research folds", json.dumps(stats), flush=True)
    return data


def frozen_scores(root: Path, data: Path):
    """Cheap ablation: only added hours are scored; original predictions reused."""
    output = root / "frozen-models"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    original = Path("output/ml-experiment")
    manifest = read(data / "manifest.json")
    names = manifest["all_features"]
    tree = CatBoostClassifier()
    tree.load_model(str(original / "pooled/engineered_episode.cbm"))
    linear = read(original / "linear/frozen_selection_canonical_v2.json")
    name = linear["selected_variant"]
    spec = next(x for x in linear["variants"] if x["name"] == name)
    model = joblib.load(original / "linear" / f"{name}.joblib")
    selections, metrics = {}, {}
    for fold in ["tune", "validation"]:
        writer = None
        for batch in pq.ParquetFile(data / f"delta_{fold}.parquet").iter_batches(batch_size=70_000):
            frame = batch.to_pandas()
            out = frame[META].copy()
            out["score_tree"] = tree.predict_proba(engineered_input(frame, names), thread_count=2)[:, 1].astype("float32")
            out["score_linear"] = model.predict_proba(transform(frame, spec["names"], name, spec["limits"]))[:, 1].astype("float32")
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output / f"delta_{fold}_scores.parquet", table.schema, compression="zstd")
            writer.write_table(table)
        if writer is None:
            raise AssertionError("Q3 fold unexpectedly empty")
        writer.close()
        with duckdb.connect() as db:
            db.execute("SET threads=2")
            db.execute("SET memory_limit='1500MB'")
            projection = ",".join(f'a."{key}"' for key in META)
            db.execute(f"""COPY (SELECT {projection},a.score_engineered_episode AS score_tree,
                b.score_log_episode_sqrt AS score_linear
                FROM read_parquet({literal(original / f'pooled/scores_{fold}.parquet')}) a
                JOIN read_parquet({literal(original / f'linear/{fold}_scores.parquet')}) b USING(channel_id,prediction_time)
                UNION ALL SELECT * FROM read_parquet({literal(output / f'delta_{fold}_scores.parquet')}))
                TO {literal(output / f'scores_{fold}.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)""")
        frame = pq.read_table(output / f"scores_{fold}.parquet").to_pandas()
        evaluation = PreparedEvaluation(frame, manifest["full_episode_count"][fold])
        if fold == "tune":
            for column in ["score_tree", "score_linear"]:
                curve = search_thresholds(evaluation, column, points=45)
                selections[column] = max(curve, key=lambda x: x["full_episode_f1"])
            choice = max(selections, key=lambda x: selections[x]["full_episode_f1"])
            (output / "selection.json").write_text(json.dumps({"selection_year": 2024,
                "selected_column": choice, "tune": selections, "model_training_through": 2023}, indent=2), encoding="utf-8")
        else:
            metrics = {column: evaluation.evaluate(column, s["threshold"]) for column, s in selections.items()}
        print("Q3 frozen model scoring complete", fold, flush=True)
        del frame, evaluation
        gc.collect()
    audits = []
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='1500MB'")
        for fold in ["tune", "validation"]:
            for column, s in selections.items():
                actual = replay(db, output / f"scores_{fold}.parquet", column, s["threshold"], manifest["full_episode_count"][fold])
                expected = s if fold == "tune" else metrics[column]
                for key in ["matched_episodes", "emitted_warnings", "full_episode_f1", "episode_precision", "full_episode_recall"]:
                    if actual[key] != expected[key]:
                        raise AssertionError(f"canonical warning parity differs {fold}/{column}/{key}")
                audits.append({"fold": fold, "column": column, "production_parity": True})
    report = {"selection": read(output / "selection.json"), "validation": metrics,
              "data_manifest_sha256": sha256(data / "manifest.json"), "audits": audits,
              "scope": "Q3 research admission extension with unchanged original models, 2024-only thresholds",
              "model_sha256": {"tree": sha256(original / "pooled/engineered_episode.cbm"),
                               "linear": sha256(original / "linear" / f"{name}.joblib")},
              "production_changed": False}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({column: {key: value for key, value in m.items() if key in
        ["matched_episodes", "emitted_warnings", "episode_precision", "full_episode_recall", "full_episode_f1"]}
        for column, m in metrics.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/ml-experiment-round2/coverage"))
    parser.add_argument("--package", type=Path, default=Path("output/q3-a-coverage-reentry-20260927-v4"))
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    data = args.root / "data" if args.score_only else prepare(args.root, args.package)
    frozen_scores(args.root, data)
