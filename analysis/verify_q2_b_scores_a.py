"""Replay every received Q2/B score from pinned A features and saved B models."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from catboost import CatBoostClassifier
import duckdb
import joblib
import numpy as np
import psutil
import pyarrow.parquet as pq

from analysis.audit_q2_oracle_a import B_SHA, Q2_SHA
from analysis.build_quality_improvement_a import quoted, safe_path
from analysis.build_sparse_population_a import write_json
from analysis.r6_provenance import frozen_rule_sha256
from analysis.run_r5_b_ablation import model_input
from analysis.train_r4_discrete_baselines import read_json, sha256


NAMES = ("base51", "full121", "linear121")


def check_lineage(database):
    """Full key/label equality, no silent inner-join dropping or duplicate inflation."""
    labels = " OR ".join(
        f"e.{name} IS DISTINCT FROM s.{name}"
        for name in ("sensor_type", "target", "target_episode_id", "label_available_at")
    )
    scores = " OR ".join(
        f"s.score_{name} IS NULL OR NOT isfinite(s.score_{name}) "
        f"OR s.score_{name}<0 OR s.score_{name}>1"
        for name in NAMES
    )
    values = database.execute(
        "SELECT COUNT(*),COUNT(DISTINCT (COALESCE(e.channel_id,s.channel_id),"
        "COALESCE(e.prediction_time,s.prediction_time))),"
        "COUNT(*) FILTER(WHERE e.channel_id IS NULL OR s.channel_id IS NULL OR "
        "e.sensor_type IS DISTINCT FROM e.label_sensor_type OR " + labels + "),"
        "COUNT(*) FILTER(WHERE " + scores + ") FROM expected e FULL JOIN scores s "
        "USING(channel_id,prediction_time)"
    ).fetchone()
    result = dict(
        zip(("joined_rows", "distinct_keys", "lineage_mismatches", "invalid_scores"), values)
    )
    if (
        result["joined_rows"] != result["distinct_keys"]
        or result["lineage_mismatches"]
        or result["invalid_scores"]
    ):
        raise ValueError(f"Q2/B source key/label/score violation: {result}")
    return result


def compare_scores(repeated, stored):
    """B persisted float32. Compare exactly after the identical serialization cast."""
    repeated = np.asarray(repeated, dtype=np.float32)
    stored = np.asarray(stored, dtype=np.float32)
    if (
        repeated.shape != stored.shape
        or not np.isfinite(repeated).all()
        or not np.isfinite(stored).all()
    ):
        raise ValueError("replayed scores have a different shape or nonfinite values")
    if ((repeated < 0) | (repeated > 1) | (stored < 0) | (stored > 1)).any():
        raise ValueError("replayed scores lie outside [0,1]")
    return {
        "rows": len(stored),
        "different_float32_values": int((repeated != stored).sum()),
        "max_absolute_difference": float(
            np.max(np.abs(repeated.astype(float) - stored.astype(float)), initial=0)
        ),
    }


def verify(*, experiment: Path, q2_dir: Path, b3_dir: Path, output_dir: Path):
    begun = time.perf_counter()
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    if sha256(experiment / "manifest.json") != B_SHA or sha256(q2_dir / "manifest.json") != Q2_SHA:
        raise ValueError("published B/Q2 manifest differs")
    b = read_json(experiment / "manifest.json")
    trained = read_json(experiment / "report.json")
    q = read_json(q2_dir / "manifest.json")
    truth = read_json(b3_dir / "manifest.json")
    b3_sha = sha256(b3_dir / "manifest.json")
    if b3_sha != q["source_manifests"]["b3"] or b3_sha != trained["b3_manifest_sha256"]:
        raise ValueError("fixed B3 label source differs")
    for name, digest in (("report.json", b["report_sha256"]), ("curves.json", b["curves_sha256"])):
        if sha256(experiment / name) != digest:
            raise ValueError("B report/curves differ")
    allowlist_file = q2_dir / "model_feature_allowlist.json"
    if sha256(allowlist_file) != q["files"]["model_feature_allowlist.json"]["sha256"]:
        raise ValueError("A allowlist differs")
    allowlist = read_json(allowlist_file)
    for model_name in NAMES:
        wanted = (
            allowlist["base_feature_names"]
            if model_name == "base51"
            else allowlist["feature_names"]
        )
        if trained["feature_sets"][model_name] != wanted:
            raise ValueError("model feature order/allowlist differs")
    model_files = {
        name: experiment / (name + (".joblib" if name == "linear121" else ".cbm")) for name in NAMES
    }
    for name, path in model_files.items():
        if sha256(path) != b["model_sha256"][name]:
            raise ValueError("received model differs")
    models = {
        name: CatBoostClassifier().load_model(str(model_files[name]))
        for name in ("base51", "full121")
    }
    # Load only the project-produced model after its published manifest and member hashes match.
    models["linear121"] = joblib.load(model_files["linear121"])
    for name in NAMES:
        actual = (
            models[name].feature_names_
            if name != "linear121"
            else list(models[name].feature_names_in_)
        )
        if actual != trained["feature_sets"][name]:
            raise ValueError("serialized model feature names differ")
    months = {m["month"]: m for m in q["months"]}
    labels = {m["month"]: m for m in truth["chunks"]}
    expected_names = [f"validation_2025-{i:02d}.parquet" for i in range(1, 13)]
    if [item["name"] for item in b["score_files"]] != expected_names:
        raise ValueError("B score scope is not all twelve validation months")
    pending.mkdir(parents=True)
    results = []
    with duckdb.connect(config={"temp_directory": str(pending / "db-spill")}) as db:
        db.execute("SET memory_limit='3GB'")
        db.execute("SET threads=2")
        db.execute("SET preserve_insertion_order=false")
        for item in b["score_files"]:
            month = item["name"][11:18]
            score = safe_path(experiment, item["name"])
            if sha256(score) != item["sha256"]:
                raise ValueError("B monthly score hash differs")
            source_month = months[month]
            features = q2_dir / f"year=2025/month={month[5:]}/model_features.parquet"
            feature_sha = source_month["files"]["model_features.parquet"]["sha256"]
            if sha256(features) != feature_sha:
                raise ValueError("A monthly features differ")
            label_item = labels[month]
            label_manifest = safe_path(b3_dir, label_item["manifest_file"])
            label_meta = read_json(label_manifest)
            if sha256(label_manifest) != label_item["manifest_sha256"]:
                raise ValueError("B3 monthly manifest differs")
            label_file = label_manifest.parent / "registered_forecast_labels.parquet"
            label_sha = label_meta["files"]["registered_forecast_labels.parquet"]["sha256"]
            if sha256(label_file) != label_sha:
                raise ValueError("B3 monthly labels differ")
            for view, path in (("features", features), ("labels", label_file), ("scores", score)):
                db.execute(
                    f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT * FROM "
                    f"read_parquet({quoted(str(path))},hive_partitioning=false)"
                )
            db.execute(
                "CREATE OR REPLACE TEMP VIEW expected AS SELECT f.*,l.target,l.target_episode_id,"
                "l.label_available_at,l.sensor_type AS label_sensor_type FROM features f JOIN labels l "
                "USING(channel_id,prediction_time) WHERE l.split='validation' "
                "AND l.split_status='assigned' AND l.target IN (0,1)"
            )
            lineage = check_lineage(db)
            if lineage["joined_rows"] != pq.ParquetFile(score).metadata.num_rows:
                raise ValueError("score file row count differs from source join")
            replay = {
                name: {"rows": 0, "different_float32_values": 0, "max_absolute_difference": 0.0}
                for name in NAMES
            }
            reader = db.execute(
                "SELECT e.*,s.score_base51,s.score_full121,s.score_linear121 "
                "FROM expected e JOIN scores s USING(channel_id,prediction_time)"
            ).to_arrow_reader(batch_size=100_000)
            for batch in reader:
                frame = batch.to_pandas()
                for name in NAMES:
                    names = trained["feature_sets"][name]
                    if name == "linear121":
                        data = frame[names].assign(
                            sensor_type=frame.sensor_type.fillna("<unknown>")
                        )
                        repeated = models[name].predict_proba(data)[:, 1]
                    else:
                        repeated = models[name].predict_proba(
                            model_input(frame, names), thread_count=2
                        )[:, 1]
                    delta = compare_scores(repeated, frame["score_" + name].to_numpy())
                    replay[name]["rows"] += delta["rows"]
                    replay[name]["different_float32_values"] += delta["different_float32_values"]
                    replay[name]["max_absolute_difference"] = max(
                        replay[name]["max_absolute_difference"], delta["max_absolute_difference"]
                    )
            if any(
                r["rows"] != lineage["joined_rows"] or r["different_float32_values"]
                for r in replay.values()
            ):
                raise ValueError(
                    f"saved B models do not reproduce all scores for {month}: {replay}"
                )
            positive_hours = db.execute("SELECT COUNT(*) FROM scores WHERE target=1").fetchone()[0]
            row = {
                "month": month,
                **lineage,
                "positive_hours": positive_hours,
                "model_replay": replay,
                "score_sha256": item["sha256"],
                "feature_sha256": feature_sha,
                "label_sha256": label_sha,
            }
            results.append(row)
            write_json(pending / f"verified_{month}.json", row)
            print(
                f"verified {month}: {lineage['joined_rows']} rows, all three float32 score streams exact",
                flush=True,
            )
    total = sum(r["joined_rows"] for r in results)
    if (
        total != trained["validation"]["rows"]
        or sum(r["positive_hours"] for r in results) != trained["validation"]["positive_hours"]
    ):
        raise ValueError("full validation totals differ")
    report = {
        "schema_version": "q2-a-b-all-saved-scores-verification-v1",
        "status": "verified",
        "source_b_manifest_sha256": B_SHA,
        "source_q2_manifest_sha256": Q2_SHA,
        "source_b3_manifest_sha256": b3_sha,
        "rows_checked": total,
        "months": results,
        "score_streams_checked": list(NAMES),
        "prediction_values_replayed": total * len(NAMES),
        "lineage_mismatches": 0,
        "score_float32_mismatches": 0,
        "no_fit_or_threshold_change": True,
        "test_events_read": False,
        "limitations": [
            "Scores are replayed from pinned A feature files, not by recomputing every M1 event.",
            "Model training is not repeated; verification is not independent statistical quality confirmation.",
        ],
        "code_lf_sha256": frozen_rule_sha256(Path(__file__)),
        "dependencies_lf_sha256": {
            name: frozen_rule_sha256(Path(name))
            for name in ("analysis/run_r5_b_ablation.py", "analysis/train_r4_discrete_baselines.py")
        },
        "resources": {
            "elapsed_seconds": round(time.perf_counter() - begun, 3),
            "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", 0),
            "duckdb_memory_limit": "3GB",
            "duckdb_threads": 2,
            "prediction_batch_rows": 100000,
        },
    }
    write_json(pending / "report.json", report)
    write_json(
        pending / "manifest.json",
        {
            "schema_version": report["schema_version"],
            "files": {
                f.name: {"sha256": sha256(f), "bytes": f.stat().st_size}
                for f in [*(pending.glob("verified_*.json")), pending / "report.json"]
            },
        },
    )
    pending.rename(output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "q2-dir", "b3-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    report = verify(**vars(parser.parse_args()))
    print(
        {
            k: report[k]
            for k in (
                "rows_checked",
                "prediction_values_replayed",
                "score_float32_mismatches",
                "resources",
            )
        }
    )


if __name__ == "__main__":
    main()
