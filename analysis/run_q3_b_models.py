"""Fixed Q3/B model comparison: choose on 2023, confirm unchanged on 2024.

Only pre-2025 train data is read. Unknowns and purged horizons are not negatives.
Completed artifacts can be resumed only with identical source and code pins.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier
import duckdb
import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from analysis.run_q2_b_expanded import fit_linear, threshold_grid
from analysis.run_r5_b_ablation import fit_model, model_input
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.alert_eval import evaluate_alerts
from ml.forecast.v2_threshold import choose_threshold


POLICIES = ("base", "cold_start", "after_normal", "combined")
MODELS = ("base51", "full121", "linear121")
NEGATIVE_SAMPLE = 200
Q2_SHA = "c9775a94bfcff09d1c641b010b93d62e9927209017ccbc7346749f78525de619"
Q3_SHA = "294d8d3a4cadb640737ec39f7adef2c8d344fcd8d4398724fcb779ebabcdf4db"
KEYS = ("channel_id", "prediction_time")


def write_json(path: Path, value: dict) -> None:
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def lf_hash(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()


def fold_bounds(year: int, *, training: bool) -> tuple[pd.Timestamp, pd.Timestamp]:
    if year not in (2023, 2024):
        raise ValueError("only predefined internal years 2023/2024 are allowed")
    return (pd.Timestamp("2019-01-01") if training else pd.Timestamp(f"{year}-01-01"),
            pd.Timestamp(f"{year if training else year + 1}-01-01"))


def source_parts(q2_dir: Path, package: Path, b3_dir: Path) -> tuple[list[dict], dict]:
    if sha256(q2_dir / "manifest.json") != Q2_SHA or sha256(package / "manifest.json") != Q3_SHA:
        raise ValueError("accepted source package changed")
    q2, q3, b3 = (read_json(root / "manifest.json") for root in (q2_dir, package, b3_dir))
    if sha256(b3_dir / "manifest.json") != q2["source_manifests"]["b3"]:
        raise ValueError("B3 source changed")
    chunks = {item["month"]: item for item in b3["chunks"]}
    delta = {item["month"]: item for item in q3["months"]}
    parts = []
    for item in q2["months"]:
        month = item["month"]
        if int(month[:4]) > 2024:
            continue
        if int(month[:4]) not in (2019, 2020, 2022, 2023, 2024):
            raise ValueError("forbidden source year")
        folder = q2_dir / Path(item["manifest_file"]).parent
        extra = package / Path(delta[month]["manifest_file"]).parent
        label_folder = b3_dir / Path(chunks[month]["manifest_file"]).parent
        label_manifest = read_json(label_folder / "manifest.json")
        paths = {
            "features": (folder / "model_features.parquet", item["files"]["model_features.parquet"]["sha256"]),
            "delta_features": (extra / "new_model_features.parquet", delta[month]["files"]["new_model_features.parquet"]["sha256"]),
            "delta_admission": (extra / "new_admission.parquet", delta[month]["files"]["new_admission.parquet"]["sha256"]),
            "labels": (label_folder / "registered_forecast_labels.parquet", label_manifest["files"]["registered_forecast_labels.parquet"]["sha256"]),
        }
        if sha256(label_folder / "manifest.json") != chunks[month]["manifest_sha256"]:
            raise ValueError(f"label month manifest changed: {month}")
        for path, expected in paths.values():
            if sha256(path) != expected:
                raise ValueError(f"source payload changed: {path}")
        parts.append({"month": month, **{key: path for key, (path, _) in paths.items()}})
    if len(parts) != 60:
        raise ValueError("incomplete internal train years")
    allowlist = read_json(q2_dir / "model_feature_allowlist.json")
    return parts, {"base51": allowlist["base_feature_names"],
                   "full121": allowlist["feature_names"], "linear121": allowlist["feature_names"]}


def row_query(part: dict, policy: str, year: int, *, training: bool) -> str:
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    start, end = fold_bounds(year, training=training)
    features = "SELECT f.*,true AS source_base FROM read_parquet(" + quoted(part["features"]) + ",hive_partitioning=false) f"
    if policy != "base":
        features += (" UNION ALL SELECT f.*,false AS source_base FROM read_parquet("
                     + quoted(part["delta_features"]) + ",hive_partitioning=false) f JOIN read_parquet("
                     + quoted(part["delta_admission"]) + ",hive_partitioning=false) d USING(channel_id,prediction_time) "
                     + f"WHERE d.{policy}_status='eligible'")
    sampled = (f"AND (l.target=1 OR hash(f.channel_id,f.prediction_time)%10000<{NEGATIVE_SAMPLE})"
               if training else "")
    return ("SELECT f.*,l.target,l.target_episode_id,l.label_available_at,l.horizon_end "
            + "FROM (" + features + ") f JOIN read_parquet(" + quoted(part["labels"])
            + ",hive_partitioning=false) l USING(channel_id,prediction_time) "
            + "WHERE l.target IN (0,1) AND l.split_status='assigned' "
            + "AND f.sensor_type IS NOT DISTINCT FROM l.sensor_type "
            + f"AND l.prediction_time>=TIMESTAMP '{start}' AND l.prediction_time<TIMESTAMP '{end}' "
            + f"AND l.horizon_end<TIMESTAMP '{end}' {sampled} ORDER BY f.channel_id,f.prediction_time")


def truth(package: Path, year: int, policy: str) -> tuple[pd.DataFrame, int]:
    start, end = fold_bounds(year, training=False)
    points = pq.ParquetFile(package / "positive_hour_audit.parquet").read().to_pandas()
    points = points.loc[(points.prediction_time >= start) & (points.prediction_time < end)
                        & (points.horizon_end < end) & points.split_status.eq("assigned")]
    full = points.drop_duplicates("target_episode_id")
    status = "admission_status" if policy == "base" else policy + "_status"
    return full, int(points.loc[points[status].eq("eligible"), "target_episode_id"].nunique())


def select_candidate(candidates: list[dict]) -> dict:
    feasible = [item for item in candidates if item["episode_precision"] > 0.7
                and item["full_episode_recall"] > 0.5]
    choices = feasible or candidates
    if not choices:
        raise ValueError("no candidates")
    def rank(item):
        primary = item["full_episode_recall"] if feasible else item["full_episode_f1"]
        return (-primary, -item["episode_precision"], item["unmatched_warnings_per_1000_channel_days"],
                item["policy"], item["model"], -item["threshold"])
    chosen = sorted(choices, key=rank)[0]
    return {"requirements_met_on_2023": bool(feasible),
            "working_selection": chosen if feasible else None,
            "candidate_for_2024": chosen,
            "candidate_status": "selected_for_confirmation" if feasible else "diagnostic_only"}


def _save_model(model, path: Path, family: str) -> None:
    model.save_model(str(path)) if family != "linear121" else joblib.dump(model, path)


def _load_model(path: Path, family: str):
    if family == "linear121":
        return joblib.load(path)
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def fit_and_score(parts: list[dict], feature_sets: dict, policy: str, year: int,
                  families: tuple[str, ...], folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    sample_file, sample_meta = folder / "sampled_train.parquet", folder / "sampled_train.json"
    if sample_meta.exists():
        stats = read_json(sample_meta)
        if stats["sample_sha256"] != sha256(sample_file):
            raise ValueError("training cache changed")
        train = pq.ParquetFile(sample_file).read().to_pandas()
    else:
        frames = []
        with duckdb.connect(":memory:") as db:
            db.execute("SET threads=2")
            db.execute("SET memory_limit='3GB'")
            for part in parts:
                if int(part["month"][:4]) >= year:
                    continue
                frame = db.execute(row_query(part, policy, year, training=True)).fetch_df()
                frames.append(frame)
                print(f"{policy}/{year} train {part['month']}: {len(frame)}", flush=True)
        train = pd.concat(frames, ignore_index=True)
        del frames
        if train.duplicated(list(KEYS)).any() or train.target.nunique() != 2:
            raise ValueError("training keys/classes are invalid")
        if train.horizon_end.max() >= pd.Timestamp(f"{year}-01-01"):
            raise ValueError("training horizon touches evaluation year")
        pq.write_table(pa.Table.from_pandas(train, preserve_index=False), sample_file, compression="zstd")
        stats = {"rows": len(train), "positive_hours": int(train.target.sum()),
                 "positive_episodes": int(train.loc[train.target.eq(1), "target_episode_id"].nunique()),
                 "min_prediction_time": str(train.prediction_time.min()),
                 "max_prediction_time": str(train.prediction_time.max()),
                 "max_label_available_at": str(train.label_available_at.max()),
                 "sample_sha256": sha256(sample_file), "negative_sample_per_10000": NEGATIVE_SAMPLE,
                 "positive_episodes_by_type": {str(k): int(v) for k, v in train.loc[train.target.eq(1)].groupby("sensor_type").target_episode_id.nunique().items()}}
        write_json(sample_meta, stats)
    models = {}
    hashes = {}
    for family in families:
        path = folder / (family + (".joblib" if family == "linear121" else ".cbm"))
        metadata = folder / (family + "_fit.json")
        if metadata.exists():
            cached = read_json(metadata)
            if cached["sample_sha256"] != stats["sample_sha256"] or cached["model_sha256"] != sha256(path):
                raise ValueError("model cache changed")
            model = _load_model(path, family)
        else:
            begun = time.perf_counter()
            print(f"fitting {policy}/{year}/{family}", flush=True)
            model = (fit_linear if family == "linear121" else fit_model)(train, feature_sets[family])
            _save_model(model, path, family)
            write_json(metadata, {"sample_sha256": stats["sample_sha256"],
                                  "model_sha256": sha256(path), "elapsed_seconds": time.perf_counter() - begun})
        models[family] = model
        hashes[family] = sha256(path)
        print(f"fitted {policy}/{year}/{family}", flush=True)
    del train
    gc.collect()
    score_files = []
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='3GB'")
        for part in parts:
            if int(part["month"][:4]) != year:
                continue
            path = folder / f"scores_{part['month']}.parquet"
            metadata = path.with_suffix(".json")
            if metadata.exists():
                cached = read_json(metadata)
                if cached["model_sha256"] != hashes or cached["score_sha256"] != sha256(path):
                    raise ValueError("score cache changed")
            else:
                pending = path.with_name(path.name + ".inprogress")
                reader = db.execute(row_query(part, policy, year, training=False)).to_arrow_reader(batch_size=100_000)
                writer = None
                count = positive = 0
                for batch in reader:
                    frame = batch.to_pandas()
                    out = frame[[*KEYS, "sensor_type", "target", "target_episode_id",
                                 "label_available_at", "source_base"]].copy()
                    for family, model in models.items():
                        names = feature_sets[family]
                        x = (frame[names].assign(sensor_type=frame.sensor_type.fillna("<unknown>"))
                             if family == "linear121" else model_input(frame, names))
                        out[f"score_{family}"] = model.predict_proba(x)[:, 1].astype("float32")
                    schema = pa.schema([
                        ("channel_id", pa.string()), ("prediction_time", pa.timestamp("us")),
                        ("sensor_type", pa.string()), ("target", pa.int64()),
                        ("target_episode_id", pa.string()), ("label_available_at", pa.timestamp("us")),
                        ("source_base", pa.bool_()),
                        *[(f"score_{family}", pa.float32()) for family in families],
                    ])
                    table = pa.Table.from_pandas(out, schema=schema, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(pending, table.schema, compression="zstd")
                    writer.write_table(table)
                    count += len(out)
                    positive += int(out.target.sum())
                if writer is None:
                    raise ValueError("empty evaluation month")
                writer.close()
                pending.replace(path)
                write_json(metadata, {"rows": count, "positive_hours": positive,
                                      "model_sha256": hashes, "score_sha256": sha256(path)})
            score_files.append(path)
            print(f"scored {policy}/{year}/{part['month']}", flush=True)
    del models
    gc.collect()
    return {"training": stats, "model_sha256": hashes, "score_files": score_files}


def warning_metrics(db, column: str, threshold: float, days: int, available: int,
                    *, common: bool) -> tuple[dict, pd.DataFrame]:
    scope = " AND source_base" if common else ""
    frame = db.execute(f"""SELECT channel_id,prediction_time,sensor_type,target,
        target_episode_id,label_available_at,{column} AS catboost_score
        FROM scores WHERE {column}>=? {scope}""", [threshold]).fetch_df()
    metric, alerts = evaluate_alerts(frame, "catboost_score", threshold, channel_days=days)
    metric["eligible_positive_episodes"] = available
    metric["episode_recall"] = metric["matched_episodes"] / available if available else 0.0
    p, r = metric["episode_precision"], metric["episode_recall"]
    metric["episode_f1"] = 2 * p * r / (p + r) if p + r else 0.0
    return metric, alerts


def breakdowns(alerts: pd.DataFrame, full: pd.DataFrame) -> dict:
    by_type = {}
    for kind in sorted(set(full.sensor_type) | set(alerts.sensor_type)):
        group = alerts.loc[alerts.sensor_type.eq(kind)]
        matched = int(group.outcome.eq("matched_episode").sum())
        total = int(full.loc[full.sensor_type.eq(kind), "target_episode_id"].nunique())
        by_type[str(kind)] = {"all_episodes": total, "warnings": len(group), "matched": matched,
                              "precision": matched / len(group) if len(group) else None,
                              "full_recall": matched / total if total else None}
    by_month = []
    for month, group in alerts.groupby(alerts.prediction_time.dt.strftime("%Y-%m")):
        by_month.append({"warning_month": month, "warnings": len(group),
                         "matched": int(group.outcome.eq("matched_episode").sum())})
    matched_ids = set(alerts.loc[alerts.outcome.eq("matched_episode"), "target_episode_id"])
    by_onset_month = []
    for month, group in full.groupby(full.label_available_at.dt.strftime("%Y-%m")):
        by_onset_month.append({"onset_month": month, "all_episodes": len(group),
                               "matched": int(group.target_episode_id.isin(matched_ids).sum())})
    loads = alerts.groupby(["channel_id", alerts.prediction_time.dt.date]).size()
    return {"by_type": by_type, "by_warning_month": by_month,
            "by_episode_onset_month": by_onset_month,
            "max_warnings_per_channel_calendar_day": int(loads.max()) if len(loads) else 0}


def evaluate_stage(stage: dict, package: Path, policy: str, year: int,
                   families: tuple[str, ...], folder: Path, fixed: float | None = None) -> dict:
    full, available = truth(package, year, policy)
    _, common_available = truth(package, year, "base")
    total = len(full)
    result = {"policy": policy, "year": year, "all_assigned_episodes": total,
              "available_episodes": available, "training": stage["training"], "models": {}}
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        db.read_parquet([str(path) for path in stage["score_files"]], hive_partitioning=False).create_view("scores")
        n, positive, actual, days, common_days = db.execute("""SELECT count(*),sum(target),
            count(DISTINCT target_episode_id) FILTER(WHERE target=1),
            count(DISTINCT(channel_id,cast(prediction_time AS DATE))),
            count(DISTINCT(channel_id,cast(prediction_time AS DATE))) FILTER(WHERE source_base)
            FROM scores""").fetchone()
        if actual != available:
            raise ValueError("scored available episodes disagree with full diagnostic")
        result["evaluation"] = {"rows": n, "positive_hours": positive,
                                "channel_days": days, "common_channel_days": common_days}
        result["available_episodes_by_type"] = {
            str(kind): int(count) for kind, count in db.execute(
                "SELECT sensor_type,count(DISTINCT target_episode_id) FROM scores "
                "WHERE target=1 GROUP BY sensor_type"
            ).fetchall()
        }
        for family in families:
            column = f"score_{family}"
            values = db.execute(f"SELECT target,{column} FROM scores").fetch_df()
            ap = float(average_precision_score(values.target, values[column]))
            grid = threshold_grid(values[column].to_numpy()) if fixed is None else [fixed]
            del values
            gc.collect()
            curve = []
            for threshold in grid:
                metric, _ = warning_metrics(db, column, threshold, days, available, common=False)
                curve.append(metric)
            goals = choose_threshold(curve, full_positive_episodes=total)
            choice = goals["selected"] or goals["diagnostic_best_full_f1"]
            metric, alerts = warning_metrics(db, column, choice["threshold"], days, available, common=False)
            common, _ = warning_metrics(db, column, choice["threshold"], common_days, common_available, common=True)
            common["full_episode_recall"] = common["matched_episodes"] / total
            selected = {**choice, "policy": policy, "model": family,
                        "unmatched_warnings_per_1000_channel_days": metric["unmatched_warnings_per_1000_channel_days"]}
            pq.write_table(pa.Table.from_pandas(alerts, preserve_index=False),
                           folder / f"alerts_{family}.parquet", compression="zstd")
            result["models"][family] = {"hourly_average_precision": ap, "goals": goals,
                                          "candidate": selected, "common_keys_metrics": common,
                                          "diagnostics": {**metric, **breakdowns(alerts, full)},
                                          "curve": curve}
            print(f"evaluated {policy}/{year}/{family}: P={choice['episode_precision']:.4f} "
                  f"R={choice['full_episode_recall']:.4f}", flush=True)
    write_json(folder / "report.json", result)
    return result


def run(*, q2_dir: Path, package: Path, b3_dir: Path, verification: Path,
        output_dir: Path) -> dict:
    begun = time.perf_counter()
    reviewed = read_json(verification)
    if (reviewed["source_package_manifest_sha256"] != Q3_SHA
            or reviewed["months_verified"] != 72 or reviewed["new_rows_verified"] != 2554406):
        raise ValueError("B source acceptance differs")
    parts, feature_sets = source_parts(q2_dir, package, b3_dir)
    code_paths = (Path(__file__), Path("analysis/run_q2_b_expanded.py"),
                  Path("analysis/run_r5_b_ablation.py"), Path("ml/forecast/alert_eval.py"),
                  Path("ml/forecast/v2_threshold.py"), Path("docs/ml-q3-b-model-comparison-protocol.md"))
    pins = {"schema_version": "q3-b-model-comparison-v1", "q2_manifest_sha256": Q2_SHA,
            "q3_manifest_sha256": Q3_SHA, "b3_manifest_sha256": sha256(b3_dir / "manifest.json"),
            "B_verification_sha256": sha256(verification),
            "code_and_protocol_lf_sha256": {str(path): lf_hash(path) for path in code_paths},
            "negative_sample_per_10000": NEGATIVE_SAMPLE, "feature_sets": feature_sets,
            "selection_year": 2023, "confirmation_year": 2024, "test_data_read": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = output_dir / "inputs.json"
    if inputs.exists() and read_json(inputs) != pins:
        raise ValueError("existing experiment has different source/code pins")
    if not inputs.exists():
        write_json(inputs, pins)
    comparisons = {}
    candidates = []
    for policy in POLICIES:
        folder = output_dir / f"selection_2023/{policy}"
        stage = fit_and_score(parts, feature_sets, policy, 2023, MODELS, folder)
        result = evaluate_stage(stage, package, policy, 2023, MODELS, folder)
        comparisons[policy] = result
        candidates.extend(value["candidate"] for value in result["models"].values())
    decision_file = output_dir / "decision_2023.json"
    decision = select_candidate(candidates)
    if decision_file.exists() and read_json(decision_file) != decision:
        raise ValueError("frozen 2023 decision changed")
    write_json(decision_file, decision)
    chosen = decision["candidate_for_2024"]
    print(f"frozen 2023 candidate: {chosen['policy']}/{chosen['model']} threshold={chosen['threshold']}", flush=True)
    folder = output_dir / "confirmation_2024"
    families = (chosen["model"],)
    stage = fit_and_score(parts, feature_sets, chosen["policy"], 2024, families, folder)
    confirmation = evaluate_stage(stage, package, chosen["policy"], 2024, families, folder,
                                  fixed=chosen["threshold"])
    checked = confirmation["models"][chosen["model"]]["candidate"]
    confirmed = checked["episode_precision"] > 0.7 and checked["full_episode_recall"] > 0.5
    report = {"schema_version": pins["schema_version"], "source_pins": pins,
              "comparison_2023": comparisons, "decision_2023": decision,
              "confirmation_2024": confirmation,
              "working_threshold": chosen["threshold"] if decision["requirements_met_on_2023"] and confirmed else None,
              "requirements_met_on_2024": confirmed,
              "elapsed_seconds": time.perf_counter() - begun,
              "limitations": ["Internal retrospective research, not new independent quality evidence.",
                              "Future journal fault entries are not confirmed physical failures.",
                              "Unknown-label hours are absent from retrospective warning cooldown.",
                              "2025 scores and opened 2026 test were not read for model selection."]}
    write_json(output_dir / "report.json", report)
    write_json(output_dir / "manifest.json", {"schema_version": pins["schema_version"],
                                               "report_sha256": sha256(output_dir / "report.json"),
                                               "decision_sha256": sha256(decision_file),
                                               "inputs_sha256": sha256(inputs)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("q2-dir", "package", "b3-dir", "verification", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
