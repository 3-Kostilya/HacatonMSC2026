"""Compare R5/A anomaly blocks on the fixed conditional R3 population.

Only train rows fit supervised models; only validation rows choose thresholds.
The sealed test split is never loaded for fitting, scoring, or model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from catboost import CatBoostClassifier
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.compare_r4_alert_budgets import select_under_budget  # noqa: E402
from analysis.train_r4_discrete_baselines import (  # noqa: E402
    clean_numeric, read_json, sha256, sha256_pinned_text, verify_inputs,
)
from ml.forecast.alert_eval import evaluate_alerts  # noqa: E402


KEYS = ("channel_id", "prediction_time")
NEGATIVE_SAMPLE_PER_10000 = 200
BUDGET = 2.0
R4_TRAIN_ROWS = 136_992
R4_TRAIN_POSITIVES = 6_871
R4_VALIDATION_ROWS = 1_365_077
R4_VALIDATION_POSITIVES = 1_943
R4_VALIDATION_EPISODES = 261
R4_CHANNEL_DAYS = 117_266


def variants(contract: dict) -> dict[str, list[str]]:
    groups = contract["feature_groups"]
    if (contract["schema_version"] != "r5-a-anomaly-features-v1"
            or set(groups) != {"stat", "isolation_forest", "clusters", "transitions"}
            or len(groups["stat"]) != 4):
        raise ValueError("unapproved R5 feature contract")
    stat = groups["stat"]
    result = {
        "baseline": [],
        "stat": stat[:3],
        "isolation_forest": groups["isolation_forest"],
        "clusters": groups["clusters"],
        "clusters_transitions": groups["clusters"] + groups["transitions"],
        "change_point": stat[3:],
        "full": stat + groups["isolation_forest"] + groups["clusters"]
                + groups["transitions"],
    }
    flat = result["full"]
    if len(flat) != 9 or len(flat) != len(set(flat)):
        raise ValueError("R5 model features are duplicated or missing")
    return result


def verify_r5(root: Path, a3_dir: Path, index_dir: Path,
              pairs: list[tuple[dict, dict]], contract: dict) -> dict[str, dict]:
    manifest = read_json(root / "manifest.json")
    if (manifest["schema_version"] != contract["schema_version"]
            or manifest["source_a3_manifest_sha256"] != sha256(a3_dir / "manifest.json")
            or manifest["source_candidate_manifest_sha256"]
            != sha256(index_dir / "manifest.json")
            or manifest["target_columns_read"]):
        raise ValueError("R5 source lineage or population differs")
    chunks = {item["month"]: item for item in manifest["chunks"]}
    if len(chunks) != len(manifest["chunks"]):
        raise ValueError("duplicate R5 month in manifest")
    expected = {a["month"] for a, _ in pairs if a["month"] < "2026-01"}
    all_months = {a["month"] for a, _ in pairs}
    rows_by_month = {a["month"]: i["rows"] for a, i in pairs}
    if (set(chunks) not in (expected, all_months)
            or manifest["rows"] != sum(rows_by_month[m] for m in chunks)):
        raise ValueError("R5 monthly population differs")
    allowed = set(variants(contract)["full"])
    for month in expected:
        item = chunks[month]
        path = root / item["features_file"]
        if sha256(path) != item["features_sha256"]:
            raise ValueError(f"R5 feature hash differs: {month}")
        parquet = pq.ParquetFile(path)
        fields = set(parquet.schema_arrow.names)
        if (item["rows"] != rows_by_month[month]
                or parquet.metadata.num_rows != item["rows"]
                or not allowed <= fields
                or {"target", "target_episode_id", "split"} & fields):
            raise ValueError(f"R5 schema differs or target leaked: {month}")
    audit = root / "audit.json"
    if not audit.exists():
        raise ValueError("R5 independent audit is missing")
    audited = read_json(audit)
    if (audited["status"] != "passed"
            or audited["months"] != len(chunks)):
        raise ValueError("R5 independent audit did not pass")
    return chunks


def joined_month(database: duckdb.DuckDBPyConnection, a3_dir: Path,
                 index_dir: Path, r5_dir: Path, a: dict, i: dict,
                 r: dict, baseline: list[str], extra: list[str],
                 split: str) -> pd.DataFrame:
    if split not in {"train", "validation"}:
        raise ValueError("R5 B may only read train or validation")
    candidate = index_dir / i["manifest_file"]
    candidate = candidate.parent / "conditional_discrete_keys.parquet"
    features = a3_dir / a["features_file"]
    new_features = r5_dir / r["features_file"]
    numeric = [name for name in baseline if name != "sensor_type"]
    projection = ", ".join([f'f."{name}"' for name in numeric]
                           + [f'r."{name}"' for name in extra])
    condition = ("AND (c.target = 1 OR "
                 "hash(c.channel_id, c.prediction_time) % 10000 < ?)"
                 if split == "train" else "")
    args = [str(candidate), str(features), str(new_features), split]
    if split == "train":
        args.append(NEGATIVE_SAMPLE_PER_10000)
    frame = database.execute(
        f"""SELECT c.channel_id, c.prediction_time, c.sensor_type,
                   f.sensor_type AS a3_sensor_type,
                   r.sensor_type AS r5_sensor_type,
                   c.target, c.target_episode_id, c.label_available_at,
                   c.split, {projection}
            FROM read_parquet(?) AS c
            JOIN read_parquet(?) AS f USING (channel_id, prediction_time)
            JOIN read_parquet(?) AS r USING (channel_id, prediction_time)
            WHERE c.split = ? {condition}""",
        args,
    ).fetch_df()
    if (frame.duplicated(list(KEYS)).any()
            or not frame.sensor_type.eq(frame.a3_sensor_type).all()
            or not frame.sensor_type.eq(frame.r5_sensor_type).all()
            or not frame.split.eq(split).all()):
        raise ValueError(f"R5/R3 join differs in {a['month']}")
    return frame.drop(columns=["a3_sensor_type", "r5_sensor_type", "split"])


def model_input(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    result = clean_numeric(frame, names)
    numeric = [name for name in names if name != "sensor_type"]
    result[numeric] = result[numeric].fillna(-1)
    return result


def fit_model(train: pd.DataFrame, names: list[str]) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=250, depth=6, learning_rate=0.05, loss_function="Logloss",
        auto_class_weights="Balanced", cat_features=["sensor_type"],
        thread_count=2, random_seed=42, verbose=False,
        allow_writing_files=False,
    )
    model.fit(model_input(train, names), train.target.to_numpy(dtype=np.int8))
    return model


def score_metrics(frame: pd.DataFrame, columns: list[str]) -> dict:
    y = frame.target.to_numpy(dtype=np.int8)
    result = {}
    for name in columns:
        result[name] = (float(average_precision_score(y, frame[name]))
                        if len(set(y)) == 2 else None)
    return result


def alert_grid(frame: pd.DataFrame, column: str, channel_days: int,
               fixed_thresholds: tuple[float, ...] = ()) -> tuple[list[dict], dict]:
    scores = frame[column].to_numpy(dtype=np.float64)
    candidates = set(np.quantile(scores, np.linspace(0.98, 0.99995, 81)).tolist())
    candidates.update(float(value) for value in fixed_thresholds)
    work = frame[[*KEYS, "sensor_type", "target", "target_episode_id",
                  "label_available_at", column]].rename(
        columns={column: "catboost_score"})
    curve = []
    for threshold in sorted(candidates):
        metrics, _ = evaluate_alerts(
            work, "catboost_score", threshold, channel_days=channel_days)
        curve.append(metrics)
    chosen = select_under_budget(curve, BUDGET)
    if chosen is None:
        raise ValueError(f"no threshold at common warning budget: {column}")
    return curve, chosen


def run(*, a3_dir: Path, b3_dir: Path, index_dir: Path, r5_dir: Path,
        r4_dir: Path, r4_audit_dir: Path, r4_budget_dir: Path,
        output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"R5 B output already exists: {output_dir}")
    contract, allowlist, pairs = verify_inputs(
        Path("ml/r3_conditional_training_contract_v1.json"),
        Path("ml/r3_discrete_feature_allowlist_v1.json"),
        a3_dir, b3_dir, index_dir)
    r5_contract = read_json(Path("ml/r5_a_feature_contract_v1.json"))
    choices = variants(r5_contract)
    r5_chunks = verify_r5(r5_dir, a3_dir, index_dir, pairs, r5_contract)
    r4_manifest = read_json(r4_dir / "manifest.json")
    r4_audit = read_json(r4_audit_dir / "report.json")
    budget = read_json(r4_budget_dir / "report.json")
    if (r4_manifest["schema_version"] != "r4-conditional-discrete-baselines-v1"
            or r4_manifest["r3_contract_sha256"] != sha256_pinned_text(
                Path("ml/r3_conditional_training_contract_v1.json"))
            or sha256(r4_dir / "catboost.cbm") != r4_manifest["catboost_sha256"]
            or r4_audit["validation_rows"] != R4_VALIDATION_ROWS
            or budget["validation_rows"] != R4_VALIDATION_ROWS
            or budget["validation_channel_days"] != R4_CHANNEL_DAYS):
        raise ValueError("R4 comparison lineage differs")
    baseline = allowlist["feature_names"]
    extras = choices["full"]
    train_parts = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        for a, i in pairs:
            if a["month"] >= "2025-01":
                continue
            part = joined_month(database, a3_dir, index_dir, r5_dir, a, i,
                                r5_chunks[a["month"]], baseline, extras, "train")
            if not part.empty:
                train_parts.append(part)
    train = pd.concat(train_parts, ignore_index=True).sort_values(
        list(KEYS), kind="mergesort").reset_index(drop=True)
    if (len(train) != R4_TRAIN_ROWS
            or int(train.target.sum()) != R4_TRAIN_POSITIVES):
        raise ValueError("R5 train sample differs from R4")
    if train.loc[train.target == 1, "target_episode_id"].isna().any():
        raise ValueError("R5 positive train row lacks episode")
    output_dir.mkdir(parents=True)
    baseline_model = CatBoostClassifier()
    baseline_model.load_model(str(r4_dir / "catboost.cbm"))
    models = {"baseline": baseline_model}
    model_hashes = {"baseline": r4_manifest["catboost_sha256"]}
    for name, extra in choices.items():
        if name == "baseline":
            continue
        model = fit_model(train, baseline + extra)
        path = output_dir / f"catboost_{name}.cbm"
        model.save_model(str(path))
        models[name] = model
        model_hashes[name] = sha256(path)
        print(f"fitted {name}: {len(train):,} rows", flush=True)
    train_coverage = {
        name: int(train[fields].notna().all(axis=1).sum()) if fields else len(train)
        for name, fields in choices.items()
    }
    del train, train_parts
    validation_parts = []
    month_files = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        for a, i in pairs:
            month = a["month"]
            if not month.startswith("2025-"):
                continue
            part = joined_month(database, a3_dir, index_dir, r5_dir, a, i,
                                r5_chunks[month], baseline, extras, "validation")
            scored = part[[*KEYS, "sensor_type", "target",
                           "target_episode_id", "label_available_at"]].copy()
            for name, extra in choices.items():
                scored[f"score_{name}"] = models[name].predict_proba(
                    model_input(part, baseline + extra))[:, 1]
            for name, fields in choices.items():
                if fields:
                    scored[f"available_{name}"] = part[fields].notna().all(axis=1)
            file = output_dir / f"validation_{month}.parquet"
            pq.write_table(pa.Table.from_pandas(scored, preserve_index=False),
                           file, compression="zstd")
            month_files.append({"month": month, "file": file.name,
                                "rows": len(scored), "sha256": sha256(file)})
            validation_parts.append(scored)
            print(f"scored {month}: {len(scored):,} rows", flush=True)
    full = pd.concat(validation_parts, ignore_index=True)
    del validation_parts
    channel_days = len(set(zip(full.channel_id, full.prediction_time.dt.date)))
    positives = full.loc[full.target == 1, "target_episode_id"]
    if (len(full) != R4_VALIDATION_ROWS
            or int(full.target.sum()) != R4_VALIDATION_POSITIVES
            or positives.nunique() != R4_VALIDATION_EPISODES
            or positives.isna().any()
            or channel_days != R4_CHANNEL_DAYS):
        raise ValueError("R5 validation population differs from accepted R4")
    score_columns = [f"score_{name}" for name in choices]
    ranking = score_metrics(full, score_columns)
    r4_pr_auc = read_json(r4_dir / "report.json")["models"]["catboost"][
        "average_precision"]
    if abs(ranking["score_baseline"] - r4_pr_auc) > 1e-12:
        raise ValueError("R5 baseline score differs from accepted R4 CatBoost")
    curves = {}
    alerts = {}
    initial = read_json(r4_dir / "report.json")["models"]["catboost"][
        "threshold_selected_on_validation"]
    accepted_r4_threshold = budget["at_common_budgets"][str(BUDGET)][
        "catboost"]["threshold"]
    for name in choices:
        column = f"score_{name}"
        curve, chosen = alert_grid(
            full, column, channel_days,
            fixed_thresholds=(initial, accepted_r4_threshold)
            if name == "baseline" else ())
        curves[name] = curve
        alerts[name] = chosen
        print(f"evaluated {name}: {chosen['matched_episodes']} matched, "
              f"{chosen['unmatched_warnings']} unmatched", flush=True)
    monthly = {
        month: score_metrics(frame, score_columns)
        for month, frame in full.groupby(full.prediction_time.dt.strftime("%Y-%m"))
    }
    by_type = {}
    for sensor_type, part in full.groupby("sensor_type", dropna=False):
        by_type[str(sensor_type)] = {
            "rows": len(part), "positive_hours": int(part.target.sum()),
            "positive_episodes": part.loc[part.target == 1, "target_episode_id"].nunique(),
            "pr_auc": score_metrics(part, score_columns),
            "coverage": {
                name: int(part[f"available_{name}"].sum())
                for name in choices if name != "baseline"
            },
        }
    coverage = {
        "train_sample": train_coverage,
        "validation": {
            name: int(full[f"available_{name}"].sum())
            for name in choices if name != "baseline"
        },
    }
    common = full.loc[full["available_full"]]
    common_coverage_sensitivity = {
        "rows": len(common),
        "positive_hours": int(common.target.sum()),
        "positive_episodes": common.loc[
            common.target == 1, "target_episode_id"].nunique(),
        "pr_auc": score_metrics(common, score_columns),
    }
    r4_rule = budget["at_common_budgets"][str(BUDGET)]["rule"]
    r4_catboost = budget["at_common_budgets"][str(BUDGET)]["catboost"]
    if (r4_rule["matched_episodes"] != 78
            or r4_rule["unmatched_warnings"] != 177):
        raise ValueError("R4 rule reference differs")
    report = {
        "schema_version": "r5-b-fixed-population-ablation-v1",
        "status": "validation_only",
        "target": contract["prediction_target"],
        "physical_failure_claim": False,
        "source_r3_contract_sha256": sha256_pinned_text(
            Path("ml/r3_conditional_training_contract_v1.json")),
        "source_r5_manifest_sha256": sha256(r5_dir / "manifest.json"),
        "source_r4_model_manifest_sha256": sha256(r4_dir / "manifest.json"),
        "source_r4_audit_manifest_sha256": sha256(r4_audit_dir / "manifest.json"),
        "source_r4_budget_manifest_sha256": sha256(r4_budget_dir / "manifest.json"),
        "train_rows": R4_TRAIN_ROWS,
        "train_positives": R4_TRAIN_POSITIVES,
        "validation_rows": len(full),
        "validation_positive_hours": int(full.target.sum()),
        "validation_positive_episodes": positives.nunique(),
        "validation_channel_days": channel_days,
        "warning_cooldown_hours": 24,
        "exploratory_unmatched_budget_per_1000_channel_days": BUDGET,
        "variant_features": choices,
        "pr_auc": ranking,
        "alerts_at_budget": alerts,
        "r4_rule_reference_at_budget": r4_rule,
        "r4_catboost_reference_at_budget": r4_catboost,
        "coverage": coverage,
        "common_complete_r5_coverage_sensitivity": common_coverage_sensitivity,
        "by_month_pr_auc": monthly,
        "by_sensor_type": by_type,
        "limitations": [
            "All variants use identical R3 train and validation rows; missing R5 values remain missing and are encoded for CatBoost, not dropped.",
            "Common complete-R5 coverage is a sensitivity ranking only; alert burdens are evaluated on the full fixed population.",
            "R5 A scores on train are temporal out-of-fold; validation models were frozen before 2025.",
            "The warning budget and thresholds are exploratory validation choices, not product-approved operating settings.",
            "The target is a future registered journal event under conditional archive completeness, not physical failure.",
            "Sealed test rows and labels were not scored or used for selection.",
        ],
    }
    (output_dir / "curves.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": report["schema_version"],
        "status": report["status"],
        "source_r5_manifest_sha256": report["source_r5_manifest_sha256"],
        "report_sha256": sha256(output_dir / "report.json"),
        "curves_sha256": sha256(output_dir / "curves.json"),
        "models": model_hashes,
        "monthly_predictions": month_files,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--r5-dir", type=Path, required=True)
    parser.add_argument("--r4-dir", type=Path, required=True)
    parser.add_argument("--r4-audit-dir", type=Path, required=True)
    parser.add_argument("--r4-budget-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({"pr_auc": result["pr_auc"],
                      "alerts_at_budget": result["alerts_at_budget"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
