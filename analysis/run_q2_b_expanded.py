"""Compare fixed model families on the verified Q2 train/validation population.

The opened 2026 test is never read. Unknown labels and purged boundaries are
never converted into negatives. This is a research validation experiment.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.run_r5_b_ablation import fit_model, model_input
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.alert_eval import evaluate_alerts
from ml.forecast.v2_threshold import choose_threshold


Q2_MANIFEST_SHA256 = "c9775a94bfcff09d1c641b010b93d62e9927209017ccbc7346749f78525de619"
ALL_VALIDATION_EPISODES = 2142
AVAILABLE_VALIDATION_EPISODES = 1359
NEGATIVE_SAMPLE_PER_10000 = 200
SCORE_NAMES = ("base51", "full121", "linear121")


def parts(package: Path, b3_dir: Path) -> tuple[list[dict], list[str], list[str]]:
    manifest = read_json(package / "manifest.json")
    allowlist = read_json(package / "model_feature_allowlist.json")
    old = read_json(Path("ml/r3_discrete_feature_allowlist_v1.json"))
    b3 = read_json(b3_dir / "manifest.json")
    if (sha256(package / "manifest.json") != Q2_MANIFEST_SHA256
            or manifest["source_manifests"]["b3"] != sha256(b3_dir / "manifest.json")
            or allowlist["base_feature_names"] != old["feature_names"]
            or len(allowlist["feature_names"]) != 121
            or allowlist["feature_names"] != [
                *allowlist["base_feature_names"], *allowlist["qa_feature_names"],
                *allowlist["missingness_feature_names"]]
            or len(manifest["months"]) != 72):
        raise ValueError("Q2 source, allowlist, or monthly scope differs")
    bchunks = {x["month"]: x for x in b3["chunks"]}
    result = []
    for month in manifest["months"]:
        label = month["month"]
        if label.startswith(("2021-", "2026-")) or label not in bchunks:
            raise ValueError(f"forbidden or missing month: {label}")
        split = "validation" if label.startswith("2025-") else "train"
        if month["split"] != split:
            raise ValueError(f"split mismatch: {label}")
        root = package / f"year={label[:4]}" / f"month={label[5:]}"
        b3root = b3_dir / Path(bchunks[label]["manifest_file"]).parent
        result.append({"month": label, "split": split,
                       "features": root / "model_features.parquet",
                       "labels": b3root / "registered_forecast_labels.parquet"})
    if len({x["month"] for x in result}) != 72:
        raise ValueError("duplicate Q2 month")
    return result, allowlist["base_feature_names"], allowlist["feature_names"]


def rows(db: duckdb.DuckDBPyConnection, part: dict, *, sample: bool):
    split = part["split"]
    sample_sql = (f"AND (l.target=1 OR hash(f.channel_id,f.prediction_time) % 10000 "
                  f"< {NEGATIVE_SAMPLE_PER_10000})" if sample else "")
    query = f"""SELECT f.*, l.target, l.target_episode_id, l.label_available_at
        FROM read_parquet(?) f JOIN read_parquet(?) l USING(channel_id,prediction_time)
        WHERE l.split=? AND l.split_status='assigned' AND l.target IN (0,1)
          AND f.sensor_type IS NOT DISTINCT FROM l.sensor_type {sample_sql}"""
    return db.execute(query, [str(part["features"]), str(part["labels"]), split])


def check_join(db: duckdb.DuckDBPyConnection, part: dict) -> None:
    bad = db.execute("""SELECT COUNT(*) FROM read_parquet(?) f JOIN read_parquet(?) l
        USING(channel_id,prediction_time) WHERE f.sensor_type IS DISTINCT FROM l.sensor_type""",
                     [str(part["features"]), str(part["labels"])]).fetchone()[0]
    if bad:
        raise ValueError(f"feature/label sensor type differs: {part['month']}")


def fit_linear(train: pd.DataFrame, names: list[str]):
    numeric = [x for x in names if x != "sensor_type"]
    pre = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="constant", fill_value=-1),
                                  StandardScaler()), numeric),
        ("type", OneHotEncoder(handle_unknown="ignore"), ["sensor_type"]),
    ], sparse_threshold=1.0)
    model = make_pipeline(pre, SGDClassifier(
        loss="log_loss", alpha=1e-4, class_weight="balanced", random_state=42,
        max_iter=300, tol=1e-3))
    model.fit(train[names].assign(sensor_type=train.sensor_type.fillna("<unknown>")),
              train.target.to_numpy(dtype=np.int8))
    return model


def selected_metrics(frame: pd.DataFrame, column: str, threshold: float,
                     channel_days: int) -> tuple[dict, pd.DataFrame]:
    # Filtering first avoids repeatedly copying millions of subthreshold rows.
    selected = frame.loc[frame[column] >= threshold,
                         ["channel_id", "prediction_time", "sensor_type", "target",
                          "target_episode_id", "label_available_at", column]].rename(
                              columns={column: "catboost_score"})
    result, alerts = evaluate_alerts(selected, "catboost_score", threshold,
                                     channel_days=channel_days)
    result["eligible_positive_episodes"] = AVAILABLE_VALIDATION_EPISODES
    result["episode_recall"] = result["matched_episodes"] / AVAILABLE_VALIDATION_EPISODES
    p = result["episode_precision"]
    r = result["episode_recall"]
    result["episode_f1"] = 2 * p * r / (p + r) if p + r else 0.0
    return result, alerts


def threshold_grid(scores: np.ndarray) -> list[float]:
    # Predeclared score-only quantiles, identical for every model.
    quantiles = [0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.9975,
                 0.999, 0.9995, 0.9999, 0.99995]
    return sorted({float(x) for x in np.quantile(scores, quantiles)})


def run(*, package: Path, b3_dir: Path, verification: Path,
        output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    verified = read_json(verification)
    if (verified.get("source_manifest_sha256") != Q2_MANIFEST_SHA256
            or verified.get("status") != "full_export_and_independent_M1_stream_checks_passed"
            or verified.get("mismatches") != 0):
        raise ValueError("independent full Q2 verification has not passed")
    items, base, full = parts(package, b3_dir)
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if pending.exists():
        raise FileExistsError(pending)
    pending.mkdir(parents=True)
    train_parts = []
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='4GB'")
        for item in items:
            check_join(db, item)
            if item["split"] != "train":
                continue
            frame = rows(db, item, sample=True).fetch_df()
            train_parts.append(frame)
            print(f"train {item['month']}: {len(frame)} sampled rows", flush=True)
    train = pd.concat(train_parts, ignore_index=True)
    del train_parts
    if train.loc[train.target == 1, "target_episode_id"].isna().any():
        raise ValueError("positive train rows have no episode ID")
    train_stats = {"rows": len(train), "positive_hours": int(train.target.sum()),
                   "positive_episodes": int(train.loc[train.target == 1,
                                                       "target_episode_id"].nunique())}
    models = {"base51": fit_model(train, base),
              "full121": fit_model(train, full),
              "linear121": fit_linear(train, full)}
    for name, model in models.items():
        path = pending / (name + (".cbm" if name != "linear121" else ".joblib"))
        model.save_model(str(path)) if name != "linear121" else joblib.dump(model, path)
        print(f"fitted {name}", flush=True)
    del train

    scored_parts = []
    val_stats = Counter()
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='4GB'")
        for item in items:
            if item["split"] != "validation":
                continue
            reader = rows(db, item, sample=False).to_arrow_reader(batch_size=100_000)
            writer = None
            path = pending / f"validation_{item['month']}.parquet"
            month_rows = month_positive = 0
            for batch in reader:
                frame = batch.to_pandas()
                out = frame[["channel_id", "prediction_time", "sensor_type", "target",
                             "target_episode_id", "label_available_at"]].copy()
                out["score_base51"] = models["base51"].predict_proba(
                    model_input(frame, base))[:, 1].astype("float32")
                out["score_full121"] = models["full121"].predict_proba(
                    model_input(frame, full))[:, 1].astype("float32")
                linear_x = frame[full].assign(sensor_type=frame.sensor_type.fillna("<unknown>"))
                out["score_linear121"] = models["linear121"].predict_proba(
                    linear_x)[:, 1].astype("float32")
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                month_rows += len(out)
                month_positive += int(out.target.sum())
            if writer is None:
                raise ValueError(f"empty validation month: {item['month']}")
            writer.close()
            scored_parts.append(path)
            val_stats["rows"] += month_rows
            val_stats["positive_hours"] += month_positive
            print(f"validation {item['month']}: {month_rows} / {month_positive}", flush=True)

    columns = ["channel_id", "prediction_time", "sensor_type", "target",
               "target_episode_id", "label_available_at",
               *[f"score_{name}" for name in SCORE_NAMES]]
    validation = pq.read_table(scored_parts, columns=columns).to_pandas()
    positive_episodes = validation.loc[validation.target == 1, "target_episode_id"].nunique()
    if positive_episodes != AVAILABLE_VALIDATION_EPISODES:
        raise ValueError(f"Q2/B episode count differs: {positive_episodes}")
    channel_days = len(set(zip(validation.channel_id, validation.prediction_time.dt.date)))
    curves = {}
    goals = {}
    ap = {}
    alert_types = {}
    for name in SCORE_NAMES:
        column = f"score_{name}"
        ap[name] = float(average_precision_score(validation.target, validation[column]))
        curve = []
        for threshold in threshold_grid(validation[column].to_numpy()):
            metric, _ = selected_metrics(validation, column, threshold, channel_days)
            curve.append(metric)
        curves[name] = curve
        goals[name] = choose_threshold(curve, full_positive_episodes=ALL_VALIDATION_EPISODES)
        choice = goals[name]["selected"] or goals[name]["diagnostic_best_full_f1"]
        _, alerts = selected_metrics(validation, column, choice["threshold"], channel_days)
        alert_types[name] = {
            str(kind): {"warnings": len(group),
                        "matched": int((group.outcome == "matched_episode").sum())}
            for kind, group in alerts.groupby("sensor_type", dropna=False)
        }
        print(f"{name}: AP={ap[name]:.5f}, P={choice['episode_precision']:.4f}, "
              f"R={choice['full_episode_recall']:.4f}", flush=True)
    episodes = pq.read_table(package / "episode_diagnostics.parquet").to_pandas()
    truth = episodes.loc[episodes.split == "validation"].groupby("sensor_type").agg(
        all_episodes=("target_episode_id", "nunique"),
        available_episodes=("candidate_hours", lambda x: int((x > 0).sum())))
    by_type = {str(kind): {"all_episodes": int(row.all_episodes),
                           "available_episodes": int(row.available_episodes),
                           "alerts": {name: alert_types[name].get(str(kind),
                                        {"warnings": 0, "matched": 0}) for name in SCORE_NAMES}}
               for kind, row in truth.iterrows()}
    report = {
        "schema_version": "q2-b-expanded-validation-v1",
        "status": "research_validation_only_no_new_independent_test",
        "q2_manifest_sha256": Q2_MANIFEST_SHA256,
        "b3_manifest_sha256": sha256(b3_dir / "manifest.json"),
        "independent_verification_sha256": sha256(verification),
        "training": train_stats,
        "train_negative_sample_per_10000": NEGATIVE_SAMPLE_PER_10000,
        "validation": {**dict(val_stats), "available_episodes": int(positive_episodes),
                       "all_assigned_episodes": ALL_VALIDATION_EPISODES,
                       "eligible_channel_days_with_binary_label": channel_days},
        "feature_sets": {"base51": base, "full121": full, "linear121": full},
        "hourly_average_precision": ap,
        "threshold_goals": goals,
        "by_type": by_type,
        "score_files": [{"name": path.name, "sha256": sha256(path)} for path in scored_parts],
        "test_data_read": False,
        "limitations": [
            "The target is a future registered journal entry, not physical failure.",
            "Admission assumes archive completeness; physical continuity is unproven.",
            "Thresholds were selected on open 2025 validation, not a new independent test.",
            "Scores from negative-sampled training are not calibrated probabilities.",
        ],
    }
    (pending / "curves.json").write_text(json.dumps(curves, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    (pending / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    (pending / "manifest.json").write_text(json.dumps({
        "schema_version": report["schema_version"],
        "q2_manifest_sha256": Q2_MANIFEST_SHA256,
        "report_sha256": sha256(pending / "report.json"),
        "curves_sha256": sha256(pending / "curves.json"),
        "model_sha256": {name: sha256(pending / (name + (".cbm" if name != "linear121"
                                                       else ".joblib"))) for name in SCORE_NAMES},
        "score_files": report["score_files"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "b3-dir", "verification", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(**{k.replace("-", "_"): v for k, v in vars(parser.parse_args()).items()})


if __name__ == "__main__":
    main()
