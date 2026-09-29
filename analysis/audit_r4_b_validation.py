"""Score and audit the complete R4 validation set without opening test.

The accepted R3 contract pins A3 features, B3 labels and B's admission index.
Saved R4 models are compared on identical validation rows. Alert-level scoring
uses a 24-hour per-channel cooldown and one match per registered episode.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.train_r4_discrete_baselines import (  # noqa: E402
    clean_numeric, read_json, rule_score, sha256, sha256_pinned_text, verify_inputs,
)
from ml.forecast.alert_eval import evaluate_alerts  # noqa: E402
from stage1.state_labeling.rules import TARGET_DEFINITION  # noqa: E402


MODEL_SCORES = {
    "rule": "rule_score",
    "logistic_regression": "logistic_score",
    "catboost": "catboost_score",
}
RULE_FIELDS = (
    "registered_fault_text_count_24h",
    "registered_fault_text_count_168h",
    "completed_episode_count_168h",
    "technical_message_count_24h",
    "normal_message_count_24h",
    "last_completed_episode_end_age_seconds",
)


def _validation_month(database: duckdb.DuckDBPyConnection, a3_dir: Path,
                      index_dir: Path, a: dict, i: dict,
                      names: list[str]) -> pd.DataFrame:
    candidate = (index_dir / i["manifest_file"]).parent / "conditional_discrete_keys.parquet"
    features = a3_dir / a["features_file"]
    feature_sql = ", ".join(f'f."{name}"' for name in names if name != "sensor_type")
    return database.execute(
        f"""SELECT c.channel_id, c.prediction_time, c.sensor_type,
                   f.sensor_type AS feature_sensor_type,
                   c.target, c.target_episode_id, c.label_available_at,
                   c.availability_status, {feature_sql}
            FROM read_parquet(?) AS c
            JOIN read_parquet(?) AS f USING (channel_id, prediction_time)
            WHERE c.split = 'validation'""",
        [str(candidate), str(features)],
    ).fetch_df()


def _check_episodes(full: pd.DataFrame, b2_dir: Path) -> dict:
    b2_manifest = read_json(b2_dir / "manifest.json")
    path = b2_dir / "registered_state_episodes.parquet"
    if (b2_manifest["status"] != "complete"
            or sha256(path) != b2_manifest["files"][path.name]["sha256"]):
        raise ValueError("B2 episode catalog hash differs")
    positive = full.loc[full.target == 1, [
        "target_episode_id", "channel_id", "sensor_type", "label_available_at",
        "prediction_time",
    ]]
    if positive.target_episode_id.isna().any():
        raise ValueError("positive validation row has no episode ID")
    lead = (positive.label_available_at - positive.prediction_time).dt.total_seconds() / 3600
    if not (lead.gt(0) & lead.le(24)).all():
        raise ValueError("positive R3 label is not strictly in the next 24 hours")
    consistency = positive.groupby("target_episode_id")[[
        "channel_id", "sensor_type", "label_available_at",
    ]].nunique(dropna=False)
    if not consistency.eq(1).all().all():
        raise ValueError("one target episode has conflicting channel/type/onset")
    truth = positive.drop_duplicates("target_episode_id").drop(columns="prediction_time")
    with duckdb.connect(":memory:") as database:
        database.register("truth", truth)
        matched = database.execute(
            """SELECT t.target_episode_id, t.channel_id, t.sensor_type,
                      t.label_available_at, e.episode_id, e.channel_id AS catalog_channel,
                      e.sensor_type AS catalog_type, e.start_at, e.confirmed_at,
                      e.onset_status, e.prior_normal_at
               FROM truth AS t LEFT JOIN read_parquet(?) AS e
                 ON t.target_episode_id = e.episode_id""",
            [str(path)],
        ).fetch_df()
    if (len(matched) != len(truth)
            or matched.episode_id.isna().any()
            or not matched.channel_id.eq(matched.catalog_channel).all()
            or not matched.sensor_type.eq(matched.catalog_type).all()
            or not matched.label_available_at.eq(matched.start_at).all()
            or not matched.start_at.eq(matched.confirmed_at).all()
            or not matched.onset_status.eq("candidate_new_onset").all()
            or matched.prior_normal_at.isna().any()
            or not matched.prior_normal_at.lt(matched.start_at).all()):
        raise ValueError("R3 positive episodes disagree with B2 registered catalog")
    return {
        "catalog_sha256": sha256(path),
        "positive_hours_checked": len(positive),
        "independent_episodes_checked": len(truth),
        "minimum_lead_hours": float(lead.min()),
        "maximum_lead_hours": float(lead.max()),
    }


def _group_ranking(full: pd.DataFrame, column: str) -> dict:
    positives = int(full.target.sum())
    negatives = len(full) - positives
    return {
        "rows": len(full), "positive_hours": positives,
        "negative_hours": negatives,
        "positive_episodes": full.loc[full.target == 1, "target_episode_id"].nunique(),
        "average_precision": {
            name: (
                float(average_precision_score(full.target, full[score]))
                if positives and negatives else None
            ) for name, score in MODEL_SCORES.items()
        },
        "group": column,
    }


def _trace_key(row: pd.Series) -> str:
    value = f'{row["channel_id"]}|{row["prediction_time"].isoformat()}'
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def run(*, contract_path: Path, allowlist_path: Path, a3_dir: Path,
        b3_dir: Path, index_dir: Path, b2_dir: Path, model_dir: Path,
        output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"audit output already exists: {output_dir}")
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if pending.exists():
        raise FileExistsError(f"audit output is in progress: {pending}")
    contract, allowlist, pairs = verify_inputs(
        contract_path, allowlist_path, a3_dir, b3_dir, index_dir)
    r4_manifest = read_json(model_dir / "manifest.json")
    r4_report = read_json(model_dir / "report.json")
    expected = (
        ("report.json", "report_sha256"),
        ("logistic_regression.joblib", "logistic_sha256"),
        ("catboost.cbm", "catboost_sha256"),
    )
    if (r4_report["status"] != "validation_only"
            or r4_report["r3_contract_sha256"] != sha256_pinned_text(contract_path)
            or r4_manifest["r3_contract_sha256"] != sha256_pinned_text(contract_path)
            or any(sha256(model_dir / file) != r4_manifest[field] for file, field in expected)):
        raise ValueError("R4 model artifact lineage differs")
    logistic = joblib.load(model_dir / "logistic_regression.joblib")
    catboost = CatBoostClassifier()
    catboost.load_model(str(model_dir / "catboost.cbm"))
    names = allowlist["feature_names"]
    numeric = [name for name in names if name != "sensor_type"]
    pending.mkdir(parents=True)
    prediction_frames = []
    chunks = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=4")
        for a, i in pairs:
            frame = _validation_month(database, a3_dir, index_dir, a, i, names)
            if frame.empty:
                continue
            if (a["month"][:4] != "2025"
                    or not frame.sensor_type.eq(frame.feature_sensor_type).all()
                    or not frame.availability_status.eq("unknown").all()
                    or not frame.target.isin([0, 1]).all()):
                raise ValueError(f"invalid validation population: {a['month']}")
            x = clean_numeric(frame, names)
            cat_x = x.copy()
            cat_x[numeric] = cat_x[numeric].fillna(-1)
            result = frame[[
                "channel_id", "prediction_time", "sensor_type", "target",
                "target_episode_id", "label_available_at", *RULE_FIELDS,
            ]].copy()
            result["rule_score"] = rule_score(frame)
            result["logistic_score"] = logistic.predict_proba(x)[:, 1]
            result["catboost_score"] = catboost.predict_proba(cat_x)[:, 1]
            prediction_frames.append(result)
            path = pending / f"validation_{a['month']}.parquet"
            pq.write_table(pa.Table.from_pandas(result, preserve_index=False),
                           path, compression="zstd")
            chunks.append({"month": a["month"], "file": path.name,
                           "rows": len(result), "sha256": sha256(path)})
            print(json.dumps({"month": a["month"], "rows": len(result)}), flush=True)
    full = pd.concat(prediction_frames, ignore_index=True)
    if (len(full) != sum(item["rows"] for item in chunks)
            or len(full) != r4_report["validation_rows"]):
        raise ValueError("validation row count differs from initial R4 run")
    if int(full.target.sum()) != r4_report["validation_positive_rows"]:
        raise ValueError("validation positive count differs")
    episode_check = _check_episodes(full, b2_dir)
    channel_days = len(full[["channel_id", "prediction_time"]].assign(
        day=full.prediction_time.dt.date
    )[["channel_id", "day"]].drop_duplicates())
    if channel_days != r4_report["validation_channel_days"]:
        raise ValueError("validation channel-day count differs")
    evaluations = {}
    alert_tables = {}
    for name, score in MODEL_SCORES.items():
        threshold = r4_report["models"][name]["threshold_selected_on_validation"]
        evaluation, alerts = evaluate_alerts(full, score, threshold,
                                             channel_days=channel_days)
        evaluations[name] = evaluation
        alert_tables[name] = alerts
        path = pending / f"alerts_{name}.parquet"
        pq.write_table(pa.Table.from_pandas(alerts, preserve_index=False),
                       path, compression="zstd")
        chunks.append({"file": path.name, "rows": len(alerts), "sha256": sha256(path)})
    by_month = {
        str(month): _group_ranking(group, "month")
        for month, group in full.groupby(full.prediction_time.dt.strftime("%Y-%m"), sort=True)
    }
    by_type = {
        str(sensor_type): _group_ranking(group, "sensor_type")
        for sensor_type, group in full.groupby("sensor_type", sort=True)
    }
    rule_alerts = alert_tables["rule"]
    rule_detail = rule_alerts.merge(full[[
        "channel_id", "prediction_time", *RULE_FIELDS,
    ]], on=["channel_id", "prediction_time"], validate="one_to_one")
    rule_feature_audit = {}
    for outcome, group in rule_detail.groupby("outcome"):
        rule_feature_audit[outcome] = {
            "warnings": len(group),
            "prior_exact_fault_24h": int(group.registered_fault_text_count_24h.gt(0).sum()),
            "prior_completed_episode_168h": int(group.completed_episode_count_168h.gt(0).sum()),
            "prior_normal_24h": int(group.normal_message_count_24h.gt(0).sum()),
        }
    trace_groups = []
    for matched in (True, False):
        group = rule_detail.loc[
            rule_detail.outcome.eq("matched_episode") == matched
        ].copy()
        group["sample_key"] = group.apply(_trace_key, axis=1)
        trace_groups.append(group.sort_values("sample_key").head(25))
    trace = pd.concat(trace_groups, ignore_index=True)
    b2_path = b2_dir / "registered_state_episodes.parquet"
    first_at, last_at = full.prediction_time.min(), full.prediction_time.max()
    segments = [
        (datetime.fromisoformat(start), datetime.fromisoformat(end))
        for start, end in TARGET_DEFINITION["archive_segments"]
        if datetime.fromisoformat(start) <= first_at < datetime.fromisoformat(end)
        and last_at < datetime.fromisoformat(end)
    ]
    if len(segments) != 1:
        raise ValueError("validation spans multiple archive segments")
    segment_start = segments[0][0]
    with duckdb.connect(":memory:") as database:
        database.register("rule_alerts", rule_alerts)
        active_across_all_archives = database.execute(
            """SELECT count(*) FROM rule_alerts AS a JOIN read_parquet(?) AS e
                ON a.channel_id = e.channel_id AND a.sensor_type = e.sensor_type
               AND e.start_at <= a.prediction_time
               AND (e.end_at IS NULL OR e.end_at > a.prediction_time)""",
            [str(b2_path)],
        ).fetchone()[0]
        active = database.execute(
            """SELECT count(*) FROM rule_alerts AS a JOIN read_parquet(?) AS e
                ON a.channel_id = e.channel_id AND a.sensor_type = e.sensor_type
               AND e.start_at >= ? AND e.start_at <= a.prediction_time
               AND (e.end_at IS NULL OR e.end_at > a.prediction_time)""",
            [str(b2_path), segment_start],
        ).fetchone()[0]
        database.register("trace", trace)
        prior = database.execute(
            """SELECT t.channel_id, t.prediction_time, e.episode_id,
                      e.start_at, e.end_at, e.onset_status
               FROM trace AS t LEFT JOIN read_parquet(?) AS e
                 ON t.channel_id = e.channel_id AND t.sensor_type = e.sensor_type
                AND e.end_at <= t.prediction_time
                AND e.end_at > t.prediction_time - INTERVAL '168 hours'
               ORDER BY t.channel_id, t.prediction_time, e.end_at DESC""",
            [str(b2_path)],
        ).fetch_df()
    if active:
        raise ValueError(f"{active} rule alerts occur in active registered episodes")
    prior_by_key = {}
    for _, item in prior.dropna(subset=["episode_id"]).iterrows():
        key = (item.channel_id, item.prediction_time)
        prior_by_key.setdefault(key, []).append({
            "episode_id": item.episode_id,
            "start_at": item.start_at.isoformat(),
            "end_at": item.end_at.isoformat(),
            "onset_status": item.onset_status,
        })
    trace_rows = []
    for _, item in trace.iterrows():
        data = item.to_dict()
        data["prediction_time"] = item.prediction_time.isoformat()
        data["label_available_at"] = item.label_available_at.isoformat()
        data["prior_registered_episodes_168h"] = prior_by_key.get(
            (item.channel_id, item.prediction_time), []
        )[:5]
        for key, value in data.items():
            if isinstance(value, (np.integer, np.floating)):
                data[key] = value.item()
            elif pd.isna(value) if not isinstance(value, list) else False:
                data[key] = None
        trace_rows.append(data)
    (pending / "trace_review.json").write_text(
        json.dumps(trace_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "r4-b-validation-audit-v1",
        "status": "validation_only_b_review",
        "source_r3_contract_sha256": sha256_pinned_text(contract_path),
        "source_r4_model_manifest_sha256": sha256(model_dir / "manifest.json"),
        "source_b2_catalog_manifest_sha256": sha256(b2_dir / "manifest.json"),
        "validation_rows": len(full),
        "validation_positive_hours": int(full.target.sum()),
        "validation_channel_days": channel_days,
        "episode_lineage": episode_check,
        "active_episode_overlap_for_rule_warnings": active,
        "cross_segment_open_episode_overlaps_ignored": (
            active_across_all_archives - active
        ),
        "models_at_hour_f1_threshold": evaluations,
        "rule_feature_audit": rule_feature_audit,
        "by_month": by_month,
        "by_sensor_type": by_type,
        "trace_rows": len(trace_rows),
        "limitations": [
            "All thresholds came from the same validation set; no final test estimate is claimed.",
            "A3 feature causality is checked by its independent tests; these traces use derived data.",
            "The target is a registered journal episode under conditional archive completeness.",
        ],
    }
    (pending / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": report["schema_version"],
        "status": report["status"],
        "report_sha256": sha256(pending / "report.json"),
        "trace_review_sha256": sha256(pending / "trace_review.json"),
        "files": chunks,
    }
    (pending / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path,
                        default=Path("ml/r3_conditional_training_contract_v1.json"))
    parser.add_argument("--allowlist", type=Path,
                        default=Path("ml/r3_discrete_feature_allowlist_v1.json"))
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--b2-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(contract_path=args.contract, allowlist_path=args.allowlist,
                 a3_dir=args.a3_dir, b3_dir=args.b3_dir, index_dir=args.index_dir,
                 b2_dir=args.b2_dir, model_dir=args.model_dir, output_dir=args.output_dir)
    print(json.dumps({"models": report["models_at_hour_f1_threshold"],
                      "episode_lineage": report["episode_lineage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
