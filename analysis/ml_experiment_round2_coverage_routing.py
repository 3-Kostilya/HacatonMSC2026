"""Type routing on separately audited Q3 research admission; not production."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from analysis.audit_ml_experiment import metadata_parity
from analysis.ml_experiment_round2_routing import (META, MINIMUM_SUPPORT, N,
    aggregate, compact, evaluate_saved, grid, group_curve, optimize, regularized_grid,
    write_routes)
from analysis.prepare_ml_experiment import sha256


MODELS = ["score_tree", "score_linear", "score_retrained"]


def literal(path: Path):
    return "'" + str(path).replace("'", "''") + "'"


def align_sources(db, root: Path, fold: str, destination: Path, expected_rows: int) -> None:
    old = root / "frozen-models" / f"scores_{fold}.parquet"
    new = root / "retrained" / f"scores_{fold}.parquet"
    differences = " OR ".join(f'a."{name}" IS DISTINCT FROM b."{name}"' for name in META[2:])
    different, matched = db.execute(f"""SELECT COUNT(*) FILTER(WHERE
        a.channel_id IS NULL OR b.channel_id IS NULL OR {differences}),
        COUNT(*) FROM read_parquet(?) a FULL OUTER JOIN read_parquet(?) b
        USING(channel_id,prediction_time)""", [str(old), str(new)]).fetchone()
    if different or matched != expected_rows:
        raise AssertionError("Q3 frozen/retrained scores have different keys or metadata")
    projection = ",".join(f'a."{name}"' for name in META)
    db.execute(f"""COPY (SELECT {projection},a.score_tree,a.score_linear,
        b.score_engineered_episode AS score_retrained FROM read_parquet({literal(old)}) a
        JOIN read_parquet({literal(new)}) b USING(channel_id,prediction_time))
        TO {literal(destination)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)""")
    metadata_parity(db, destination, root / "data" / f"{fold}.parquet", expected_rows,
                    2024 if fold == "tune" else 2025)


def run(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    if manifest["full_episode_count"]["tune"] != 1204 or manifest["full_episode_count"]["validation"] != 2142:
        raise AssertionError("complete B3 episode denominator changed")
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory=?", [str(output / "duckdb-temp")])
        tune = output / "joined_tune.parquet"
        align_sources(db, root, "tune", tune, manifest["row_stats"]["tune"]["rows"])
        stats = db.execute("""SELECT sensor_type,
            COUNT(DISTINCT CASE WHEN target=1 THEN target_episode_id END),
            COUNT(DISTINCT (channel_id,CAST(prediction_time AS DATE)))
            FROM read_parquet(?) GROUP BY sensor_type""", [str(tune)]).fetchall()
        supports = {kind: int(count) for kind, count, _ in stats}
        supported = [kind for kind, count in supports.items() if count >= MINIMUM_SUPPORT]
        groups = {kind: [kind] for kind in supported}
        groups["__default__"] = [kind for kind in supports if kind not in supported]
        days = {kind: int(value) for kind, _, value in stats}
        if db.execute("""SELECT COUNT(*) FROM (SELECT channel_id
            FROM read_parquet(?) GROUP BY channel_id
            HAVING COUNT(DISTINCT sensor_type)>1)""", [str(tune)]).fetchone()[0]:
            raise AssertionError("Q3 type routes are not channel-additive")
        curves, references = {}, {}
        flexible, regularized = {kind: [] for kind in supported}, {kind: [] for kind in supported}
        for model in MODELS:
            global_grid = grid(db, tune, model)
            options = {group: group_curve(db, tune, model,
                global_grid+(grid(db,tune,model,kinds) if group != "__default__" else []),
                kinds, sum(days[kind] for kind in kinds)) for group,kinds in groups.items()}
            global_curves = [{**aggregate({group: rows[threshold] for group,rows in options.items()}),
                              "threshold": threshold, "model": model} for threshold in global_grid]
            reference = max(global_curves, key=lambda row: (row["full_episode_f1"],row["episode_precision"]))
            references[model] = reference
            curves[model] = {group: list(rows.values()) for group,rows in options.items()}
            for group in supported:
                flexible[group].extend(options[group].values())
                support = supports[group]
                local = max(options[group].values(), key=lambda row:
                    2*row["matched_episodes"]/(support+row["emitted_warnings"]))
                bounded = regularized_grid(reference["threshold"], local["threshold"], support)
                regularized[group].extend(group_curve(db,tune,model,bounded,groups[group],days[group]).values())
            print("Q3 2024 routes", model, compact(reference), flush=True)
        # The predeclared safe fallback is the original pooled tree, not the
        # fresh winner or a rare-type threshold chosen from tiny samples.
        fallback = next(row for row in curves["score_tree"]["__default__"]
                        if row["threshold"] == references["score_tree"]["threshold"])
        choices, metrics = {}, {}
        for name, options in [("flexible",flexible),("regularized",regularized)]:
            choices[name], metrics[name] = optimize(options, fallback)
        winner = max(metrics, key=lambda name: metrics[name]["full_episode_f1"])
        policies = {name: {kind: {"model": row["model"],"threshold": row["threshold"]}
                           for kind,row in mapping.items()} for name,mapping in choices.items()}
        selection = {"selection_year":2024,"selected_policy":winner,"supported_types":supports,
                     "minimum_type_support":MINIMUM_SUPPORT,"policies":policies,
                     "tune_metrics":metrics,"global_references":references,
                     "research_admission_approval":False,"fallback":policies[winner]["__default__"],
                     "source_q3_manifest_sha256":manifest["source_q3_manifest_sha256"],
                     "tune_score_sha256":sha256(tune),"full_episode_count":N}
        (output/"selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
        (output/"curves.json").write_text(json.dumps(curves,ensure_ascii=False,indent=2),encoding="utf-8")
        # Only after the 2024 policy is immutable do we read validation scores.
        routed_tune = output / "tune_scores.parquet"
        write_routes((batch.to_pandas() for batch in pq.ParquetFile(tune).iter_batches(batch_size=100_000)),
                     policies,routed_tune)
        available = sum(supports.values())
        tune_checks = {name: evaluate_saved(db,routed_tune,f"score_{name}","tune",
            sum(days.values()),available) for name in policies}
        for name in policies:
            if (any(tune_checks[name][key] != metrics[name][key]
                    for key in ["matched_episodes","emitted_warnings"])
                    or abs(tune_checks[name]["full_episode_f1"]-metrics[name]["full_episode_f1"])>1e-14):
                raise AssertionError("Q3 additive optimization differs from complete warning replay")
        validation = output / "joined_validation.parquet"
        align_sources(db,root,"validation",validation,manifest["row_stats"]["validation"]["rows"])
        routed_val = output / "validation_scores.parquet"
        write_routes((batch.to_pandas() for batch in pq.ParquetFile(validation).iter_batches(batch_size=100_000)),
                     policies,routed_val)
        available, channel_days = db.execute("""SELECT COUNT(DISTINCT CASE WHEN target=1
            THEN target_episode_id END), COUNT(DISTINCT(channel_id,CAST(prediction_time AS DATE)))
            FROM read_parquet(?)""", [str(routed_val)]).fetchone()
        transfers = {name: evaluate_saved(db,routed_val,f"score_{name}","validation",
            channel_days,available) for name in policies}
    result = {"selection":selection,"tune":tune_checks,"validation":transfers,
              "selected_policy":winner,"selected_validation":transfers[winner],
              "source_metadata_parity":"full exact Q3 key/type/target/episode/time joins",
              "independent_production_warning_parity":True,
              "policy_frozen_before_2025":True,"selection_sha256":sha256(output/"selection.json"),
              "model_hashes": {"original_tree":sha256(Path("output/ml-experiment/pooled/engineered_episode.cbm")),
                               "original_linear":sha256(Path("output/ml-experiment/linear/log_episode_sqrt.joblib")),
                               "q3_retrained_tree":sha256(root/"retrained/engineered_episode.cbm")},
              "research_only":True,"physical_availability_approved":False}
    (output/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({name:compact(metric) for name,metric in transfers.items()}),flush=True)
    return result


def resume_frozen_transfer(root: Path, output: Path) -> dict:
    """Resume only after independently rechecking an already frozen 2024 policy."""
    destination = output / "report.json"
    if destination.exists():
        raise FileExistsError(destination)
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    if (selection["selection_year"] != 2024
            or selection["source_q3_manifest_sha256"] != manifest["source_q3_manifest_sha256"]
            or selection["tune_score_sha256"] != sha256(output / "joined_tune.parquet")
            or (output / "joined_validation.parquet").exists()
            or (output / "validation_scores.parquet").exists()):
        raise AssertionError("frozen selection or safe resume state differs")
    policies = selection["policies"]
    winner = selection["selected_policy"]
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory=?", [str(output / "duckdb-temp")])
        metadata_parity(db, output / "joined_tune.parquet", root / "data/tune.parquet",
                        manifest["row_stats"]["tune"]["rows"], 2024)
        metadata_parity(db, output / "tune_scores.parquet", root / "data/tune.parquet",
                        manifest["row_stats"]["tune"]["rows"], 2024)
        available, channel_days = db.execute("""SELECT COUNT(DISTINCT CASE WHEN target=1
            THEN target_episode_id END),COUNT(DISTINCT(channel_id,CAST(prediction_time AS DATE)))
            FROM read_parquet(?)""", [str(output / "tune_scores.parquet")]).fetchone()
        tune_checks = {name: evaluate_saved(db,output / "tune_scores.parquet",f"score_{name}",
            "tune",channel_days,available) for name in policies}
        for name in policies:
            expected = selection["tune_metrics"][name]
            if (any(tune_checks[name][key] != expected[key]
                    for key in ["matched_episodes","emitted_warnings"])
                    or abs(tune_checks[name]["full_episode_f1"]-expected["full_episode_f1"])>1e-14):
                raise AssertionError("resume 2024 canonical production warnings differ")
        validation = output / "joined_validation.parquet"
        align_sources(db,root,"validation",validation,manifest["row_stats"]["validation"]["rows"])
        routed_val = output / "validation_scores.parquet"
        write_routes((batch.to_pandas() for batch in pq.ParquetFile(validation).iter_batches(batch_size=100_000)),
                     policies,routed_val)
        available, channel_days = db.execute("""SELECT COUNT(DISTINCT CASE WHEN target=1
            THEN target_episode_id END),COUNT(DISTINCT(channel_id,CAST(prediction_time AS DATE)))
            FROM read_parquet(?)""", [str(routed_val)]).fetchone()
        transfers = {name: evaluate_saved(db,routed_val,f"score_{name}","validation",
            channel_days,available) for name in policies}
    result = {"selection":selection,"tune":tune_checks,"validation":transfers,
              "selected_policy":winner,"selected_validation":transfers[winner],
              "source_metadata_parity":"full exact Q3 key/type/target/episode/time joins",
              "independent_production_warning_parity":True,
              "policy_frozen_before_2025":True,"selection_sha256":sha256(selection_path),
              "model_hashes": {"original_tree":sha256(Path("output/ml-experiment/pooled/engineered_episode.cbm")),
                               "original_linear":sha256(Path("output/ml-experiment/linear/log_episode_sqrt.joblib")),
                               "q3_retrained_tree":sha256(root/"retrained/engineered_episode.cbm")},
              "research_only":True,"physical_availability_approved":False,
              "resume_verified_2024_before_2025":True}
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({name:compact(metric) for name,metric in transfers.items()}),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("output/ml-experiment-round2/coverage"))
    parser.add_argument("--output",type=Path,default=Path("output/ml-experiment-round2/coverage-routing"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    resume_frozen_transfer(args.root,args.output) if args.resume else run(args.root,args.output)
