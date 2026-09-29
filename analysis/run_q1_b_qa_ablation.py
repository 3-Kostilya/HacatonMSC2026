"""Compare frozen R4 CatBoost with train-only QA additions on fixed R3 keys.

Only the accepted train and validation index parts are opened. The previously
opened test is not used for fitting, scoring, or selecting a threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from analysis.run_r5_b_ablation import alert_grid, fit_model, model_input
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.alert_eval import evaluate_alerts
from ml.forecast.v2_threshold import choose_threshold


Q1_MANIFEST_SHA256 = "0f273047bef8678a9ef57dde38b33c938ffa2ea8345e6840de10fbab3940735d"
FULL_VALIDATION_EPISODES = 2142
TRAIN_ROWS = 136_992
TRAIN_POSITIVES = 6_871
VALIDATION_ROWS = 1_365_077
VALIDATION_POSITIVES = 1_943
VALIDATION_EPISODES = 261
VALIDATION_CHANNEL_DAYS = 117_266


def verify_sources(q1_dir: Path, index_dir: Path, r4_dir: Path) -> tuple[list[dict], list[str], list[str], dict]:
    package = read_json(q1_dir / "manifest.json")
    package_report = read_json(q1_dir / "report.json")
    allowlist = read_json(q1_dir / "model_feature_allowlist.json")
    old_allowlist = read_json(Path("ml/r3_discrete_feature_allowlist_v1.json"))
    index = read_json(index_dir / "manifest.json")
    old_model = read_json(r4_dir / "manifest.json")
    if (
        sha256(q1_dir / "manifest.json") != Q1_MANIFEST_SHA256
        or package["status"] != "a_train_validation_features_ready_b_review_pending"
        or package["row_count"] != 7_870_121
        or package["chunk_count"] != 72
        or package["report_sha256"] != sha256(q1_dir / "report.json")
        or package_report["source_admission_manifest_sha256"] != sha256(index_dir / "manifest.json")
        or package["allowlist_sha256"] != sha256(q1_dir / "model_feature_allowlist.json")
        or allowlist["base_feature_names"] != old_allowlist["feature_names"]
        or allowlist["feature_names"] != allowlist["base_feature_names"] + allowlist["qa_feature_names"]
        or len(allowlist["base_feature_names"]) != 51
        or len(allowlist["qa_feature_names"]) != 20
        or old_model["schema_version"] != "r4-conditional-discrete-baselines-v1"
        or sha256(r4_dir / "catboost.cbm") != old_model["catboost_sha256"]
    ):
        raise ValueError("Q1, accepted R3 index, or frozen R4 lineage differs")
    index_chunks = {item["month"]: item for item in index["chunks"]}
    result = []
    for item in package["chunks"]:
        month = item["month"]
        if month.startswith("2021-") or not (
            month < "2025-01" or "2025-01" <= month <= "2025-12"
        ):
            raise ValueError("Q1 experiment may only read train and validation months")
        if item["split"] != ("validation" if month.startswith("2025-") else "train"):
            raise ValueError(f"Q1 split differs: {month}")
        part = q1_dir / f"year={month[:4]}" / f"month={month[5:]}"
        features = part / "model_features.parquet"
        gate = part / "qa_gate.parquet"
        index_part = index_dir / index_chunks[month]["manifest_file"]
        candidate = index_part.parent / "conditional_discrete_keys.parquet"
        index_local = read_json(index_part)
        if (
            item["rows"] != index_chunks[month]["rows"]
            or index_local["candidate_sha256"] != sha256(candidate)
            or item["files"]["model_features.parquet"]["sha256"] != sha256(features)
            or item["files"]["qa_gate.parquet"]["sha256"] != sha256(gate)
            or item["qa_gate_unknown"] != 0
            or pq.ParquetFile(features).metadata.num_rows != item["rows"]
            or set(pq.ParquetFile(features).schema_arrow.names)
            != {"channel_id", "prediction_time", *allowlist["feature_names"]}
        ):
            raise ValueError(f"Q1 feature/index/gate differs: {month}")
        result.append({"month": month, "split": item["split"], "rows": item["rows"],
                       "features": features, "candidate": candidate})
    if len(result) != 72 or len({item["month"] for item in result}) != 72:
        raise ValueError("Q1 months incomplete or duplicated")
    return result, allowlist["base_feature_names"], allowlist["qa_feature_names"], old_model


def joined_month(database: duckdb.DuckDBPyConnection, item: dict,
                 names: list[str]) -> pd.DataFrame:
    split = item["split"]
    projection = ", ".join(f'f."{name}"' for name in names if name != "sensor_type")
    sample = ("AND (c.target = 1 OR hash(c.channel_id, c.prediction_time) % 10000 < 200)"
              if split == "train" else "")
    frame = database.execute(
        f"""SELECT c.channel_id, c.prediction_time, c.sensor_type AS label_sensor_type,
                   f.sensor_type, c.target, c.target_episode_id, c.label_available_at,
                   {projection}
            FROM read_parquet(?) AS c
            JOIN read_parquet(?) AS f USING(channel_id, prediction_time)
            WHERE c.split = ? {sample}""",
        [str(item["candidate"]), str(item["features"]), split],
    ).fetch_df()
    if (
        frame.duplicated(["channel_id", "prediction_time"]).any()
        or not frame.sensor_type.fillna("<unknown>").eq(
            frame.label_sensor_type.fillna("<unknown>")).all()
        or (split == "validation" and len(frame) != item["rows"])
        or frame.target.isna().any()
    ):
        raise ValueError(f"Q1 keys, type or binary labels differ: {item['month']}")
    return frame.drop(columns="label_sensor_type")


def active_qa_features(train: pd.DataFrame, qa: list[str]) -> list[str]:
    """Select only nonconstant QA fields without looking at validation."""
    return [name for name in qa if train[name].nunique(dropna=False) > 1]


def compact_metrics(frame: pd.DataFrame, columns: list[str]) -> dict:
    y = frame.target.to_numpy(dtype=np.int8)
    return {name: (float(average_precision_score(y, frame[name]))
                   if len(set(y)) == 2 else None) for name in columns}


def run(*, q1_dir: Path, index_dir: Path, r4_dir: Path, output_dir: Path) -> dict:
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    items, base, qa, old_model_manifest = verify_sources(q1_dir, index_dir, r4_dir)
    pending.mkdir(parents=True)
    train_parts = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        for item in items:
            if item["split"] == "train":
                train_parts.append(joined_month(database, item, base + qa))
    train = pd.concat(train_parts, ignore_index=True).sort_values(
        ["channel_id", "prediction_time"], kind="mergesort").reset_index(drop=True)
    if (len(train) != TRAIN_ROWS or int(train.target.sum()) != TRAIN_POSITIVES
            or train.loc[train.target == 1, "target_episode_id"].isna().any()):
        raise ValueError("Q1 train sample differs from frozen R4")
    selected_qa = active_qa_features(train, qa)
    baseline = CatBoostClassifier()
    baseline.load_model(str(r4_dir / "catboost.cbm"))
    extended = fit_model(train, base + selected_qa)
    extended_file = pending / "catboost_plus_qa.cbm"
    extended.save_model(str(extended_file))
    print(f"train rows={len(train)} positive={TRAIN_POSITIVES} active_QA={len(selected_qa)}", flush=True)
    del train, train_parts

    parts = []
    months = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        for item in items:
            if item["split"] != "validation":
                continue
            frame = joined_month(database, item, base + qa)
            scored = frame[["channel_id", "prediction_time", "sensor_type", "target",
                            "target_episode_id", "label_available_at"]].copy()
            scored["score_baseline"] = baseline.predict_proba(model_input(frame, base))[:, 1]
            scored["score_plus_qa"] = extended.predict_proba(
                model_input(frame, base + selected_qa))[:, 1]
            path = pending / f"validation_{item['month']}.parquet"
            pq.write_table(pa.Table.from_pandas(scored, preserve_index=False), path,
                           compression="zstd")
            months.append({"month": item["month"], "rows": len(scored),
                           "file": path.name, "sha256": sha256(path)})
            parts.append(scored)
            print(f"validation {item['month']} rows={len(scored)}", flush=True)
    validation = pd.concat(parts, ignore_index=True)
    channel_days = len(set(zip(validation.channel_id, validation.prediction_time.dt.date)))
    episodes = validation.loc[validation.target == 1, "target_episode_id"]
    if (len(validation) != VALIDATION_ROWS or int(validation.target.sum()) != VALIDATION_POSITIVES
            or episodes.nunique() != VALIDATION_EPISODES or episodes.isna().any()
            or channel_days != VALIDATION_CHANNEL_DAYS):
        raise ValueError("Q1 validation population differs from accepted R4")
    score_columns = ["score_baseline", "score_plus_qa"]
    ranking = compact_metrics(validation, score_columns)
    old_pr_auc = read_json(r4_dir / "report.json")["models"]["catboost"]["average_precision"]
    if abs(ranking["score_baseline"] - old_pr_auc) > 1e-12:
        raise ValueError("Q1 base matrix or frozen R4 scores differ")
    curves = {}
    alerts = {}
    goals = {}
    type_alerts = {}
    for name in ("baseline", "plus_qa"):
        column = f"score_{name}"
        curve, chosen = alert_grid(validation, column, channel_days)
        curves[name] = curve
        alerts[name] = chosen
        goals[name] = choose_threshold(curve, full_positive_episodes=FULL_VALIDATION_EPISODES)
        work = validation[["channel_id", "prediction_time", "sensor_type", "target",
                           "target_episode_id", "label_available_at", column]].rename(
                               columns={column: "catboost_score"})
        _, emitted = evaluate_alerts(work, "catboost_score", chosen["threshold"],
                                     channel_days=channel_days)
        type_alerts[name] = {
            kind: {"emitted": len(group),
                   "matched": int((group.outcome == "matched_episode").sum())}
            for kind, group in emitted.groupby("sensor_type")
        }
        print(f"{name}: AP={ranking[column]:.6f}, matched={chosen['matched_episodes']}, "
              f"unmatched={chosen['unmatched_warnings']}", flush=True)
    monthly = {str(month): compact_metrics(part, score_columns) for month, part in
               validation.groupby(validation.prediction_time.dt.strftime("%Y-%m"))}
    by_type = {}
    for kind, part in validation.groupby("sensor_type", dropna=False):
        key = str(kind)
        by_type[key] = {"rows": len(part), "positive_hours": int(part.target.sum()),
                        "available_positive_episodes": part.loc[
                            part.target == 1, "target_episode_id"].nunique(),
                        "pr_auc": compact_metrics(part, score_columns),
                        "warnings_at_budget": {name: type_alerts[name].get(
                            kind, {"emitted": 0, "matched": 0}) for name in alerts}}
    report = {
        "schema_version": "q1-b-fixed-population-qa-ablation-v1",
        "status": "validation_only_no_independent_test",
        "source_q1_manifest_sha256": sha256(q1_dir / "manifest.json"),
        "source_r3_index_manifest_sha256": sha256(index_dir / "manifest.json"),
        "source_r4_model_manifest_sha256": sha256(r4_dir / "manifest.json"),
        "train_negative_sample_per_10000": 200,
        "train_rows": TRAIN_ROWS, "train_positive_hours": TRAIN_POSITIVES,
        "validation_rows": VALIDATION_ROWS, "validation_positive_hours": VALIDATION_POSITIVES,
        "validation_available_episodes": VALIDATION_EPISODES,
        "validation_all_assigned_positive_episodes": FULL_VALIDATION_EPISODES,
        "validation_channel_days": channel_days,
        "base_feature_count": len(base), "qa_feature_count": len(qa),
        "active_qa_selected_on_train": selected_qa,
        "hourly_pr_auc": ranking,
        "alerts_at_exploratory_budget_2_per_1000_channel_days": alerts,
        "goal_checks": goals,
        "by_month_hourly_pr_auc": monthly, "by_sensor_type": by_type,
        "test_scores_or_labels_read": False,
        "limitations": [
            "Q1 uses the unchanged retrospective R3 population; no live admission is approved.",
            "The full-population recall ceiling remains 261/2142 regardless of QA features.",
            "The warning budget and validation thresholds are exploratory, not an approved operating limit.",
            "Negative train rows were deterministically sampled; scores are not calibrated probabilities.",
            "The target is a future registered journal entry, not confirmed physical device failure.",
        ],
    }
    (pending / "curves.json").write_text(json.dumps(curves, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    (pending / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    manifest = {
        "schema_version": report["schema_version"], "status": report["status"],
        "source_q1_manifest_sha256": report["source_q1_manifest_sha256"],
        "report_sha256": sha256(pending / "report.json"),
        "curves_sha256": sha256(pending / "curves.json"),
        "r4_baseline_model_sha256": old_model_manifest["catboost_sha256"],
        "plus_qa_model_sha256": sha256(extended_file), "validation_parts": months,
    }
    (pending / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    pending.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--r4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"hourly_pr_auc": result["hourly_pr_auc"],
                      "alerts": result["alerts_at_exploratory_budget_2_per_1000_channel_days"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
