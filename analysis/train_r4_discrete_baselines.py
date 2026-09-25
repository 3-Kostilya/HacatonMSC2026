"""Train conditional journal-event baselines without opening the test split.

The input contract pins the R3 admission index and A's 51-feature allowlist.
Only train rows fit preprocessing/models; validation is used for comparison.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_inputs(contract_path: Path, allowlist_path: Path, a3_dir: Path,
                  b3_dir: Path, index_dir: Path) -> tuple[dict, dict, list[tuple[dict, dict]]]:
    contract = read_json(contract_path)
    allowlist = read_json(allowlist_path)
    a3 = read_json(a3_dir / "manifest.json")
    index = read_json(index_dir / "manifest.json")
    if (contract["schema_version"] != "r3-conditional-training-contract-v1"
            or contract["status"] != "jointly_accepted_conditional"
            or not contract["accepted_for_training"]
            or not contract["sealed_test_for_model_selection"]):
        raise ValueError("R3 conditional training contract has not been accepted")
    expected = (
        (allowlist_path, "feature_allowlist_sha256"),
        (a3_dir / "manifest.json", "source_a3_manifest_sha256"),
        (b3_dir / "manifest.json", "source_b3_manifest_sha256"),
        (index_dir / "manifest.json", "source_admission_manifest_sha256"),
    )
    for path, field in expected:
        if sha256(path) != contract[field]:
            raise ValueError(f"R3 source hash differs: {path}")
    names = allowlist["feature_names"]
    if (allowlist["schema_version"] != contract["feature_allowlist_version"]
            or allowlist["admission_rule_version"] != contract["admission_rule_version"]
            or len(names) != len(set(names))):
        raise ValueError("approved feature list differs")
    if (len(names) != 51 or allowlist["categorical_feature_names"] != ["sensor_type"]
            or index["schema_version"] != contract["admission_rule_version"]
            or index["row_count"] != contract["candidate_rows"]
            or index["chunk_count"] != contract["candidate_months"]):
        raise ValueError("R3 admission/feature version differs")
    a_chunks = {item["month"]: item for item in a3["chunks"]}
    i_chunks = {item["month"]: item for item in index["chunks"]}
    if len(a_chunks) != len(i_chunks) or set(a_chunks) != set(i_chunks):
        raise ValueError("A3 and R3 admission months differ")
    if any(month.startswith("2021-") for month in a_chunks):
        raise ValueError("intentionally excluded year appears in R3 inputs")
    pairs = []
    for month in sorted(a_chunks):
        a, i = a_chunks[month], i_chunks[month]
        for root, item in ((a3_dir, a), (index_dir, i)):
            if sha256(root / item["manifest_file"]) != item["manifest_sha256"]:
                raise ValueError(f"R3 month manifest differs: {month}")
        local = read_json(index_dir / i["manifest_file"])
        candidate = (index_dir / i["manifest_file"]).parent / "conditional_discrete_keys.parquet"
        features = a3_dir / a["features_file"]
        if (sha256(candidate) != local["candidate_sha256"]
                or sha256(features) != a["features_sha256"]
                or local["row_count"] != i["rows"]):
            raise ValueError(f"R3 month data differs: {month}")
        fields = pq.ParquetFile(features).schema_arrow.names
        if not set(names) <= set(fields):
            raise ValueError(f"approved feature missing from A3: {month}")
        pairs.append((a, i))
    return contract, allowlist, pairs


def join_month(database: duckdb.DuckDBPyConnection, a3_dir: Path, index_dir: Path,
               a: dict, i: dict, names: list[str], split: str,
               negative_sample_per_10000: int) -> pd.DataFrame:
    candidate = (index_dir / i["manifest_file"]).parent / "conditional_discrete_keys.parquet"
    features = a3_dir / a["features_file"]
    feature_sql = ", ".join(f'f."{name}"' for name in names)
    condition = "c.target = 1 OR hash(c.channel_id, c.prediction_time) % 10000 < ?"
    where = f"c.split = ? AND ({condition})" if split == "train" else "c.split = ?"
    args = [str(candidate), str(features), split]
    if split == "train":
        args.append(negative_sample_per_10000)
    result = database.execute(
        f"""SELECT c.channel_id, c.prediction_time, c.target,
                   c.target_episode_id, c.split, {feature_sql}
            FROM read_parquet(?) AS c
            JOIN read_parquet(?) AS f
              USING (channel_id, prediction_time)
            WHERE {where}""",
        args,
    )
    return result.fetch_df()


def rule_score(frame: pd.DataFrame) -> np.ndarray:
    def count(name: str) -> np.ndarray:
        return pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy(dtype=float)
    return (2.0 * count("registered_fault_text_count_24h")
            + 0.5 * count("registered_fault_text_count_168h")
            + count("completed_episode_count_168h")
            + 0.1 * count("technical_message_count_24h"))


def clean_numeric(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    result = frame[names].copy()
    for name in names:
        if name == "sensor_type":
            result[name] = result[name].fillna("<unknown>").astype(str)
        else:
            result[name] = pd.to_numeric(result[name], errors="coerce").astype("float32")
            result[name] = result[name].replace([np.inf, -np.inf], np.nan)
    return result


def choose_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-15)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def evaluate(y: np.ndarray, score: np.ndarray,
             positive_best: dict[str, float], channel_days: int) -> dict:
    threshold, best_f1 = choose_threshold(y, score)
    predicted = score >= threshold
    tp = int(np.count_nonzero(predicted & (y == 1)))
    fp = int(np.count_nonzero(predicted & (y == 0)))
    return {
        "average_precision": float(average_precision_score(y, score)),
        "threshold_selected_on_validation": threshold,
        "hour_precision": tp / (tp + fp) if tp + fp else 0.0,
        "hour_recall": tp / int(np.count_nonzero(y == 1)),
        "hour_f1": best_f1,
        "true_positive_hours": tp,
        "false_positive_hours": fp,
        "false_positive_hours_per_1000_channel_days": fp * 1000 / channel_days,
        "positive_episode_recall": (
            sum(value >= threshold for value in positive_best.values()) / len(positive_best)
            if positive_best else 0.0
        ),
        "positive_episodes": len(positive_best),
    }


def run(*, contract_path: Path, allowlist_path: Path, a3_dir: Path,
        b3_dir: Path, index_dir: Path, output_dir: Path,
        negative_sample_per_10000: int = 200) -> dict:
    if not 1 <= negative_sample_per_10000 <= 10000:
        raise ValueError("negative sampling rate must be in 1..10000")
    if output_dir.exists():
        raise FileExistsError(f"R4 output already exists: {output_dir}")
    contract, allowlist, pairs = verify_inputs(
        contract_path, allowlist_path, a3_dir, b3_dir, index_dir)
    names = allowlist["feature_names"]
    numeric = [name for name in names if name != "sensor_type"]
    train_frames = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=4")
        for a, i in pairs:
            frame = join_month(database, a3_dir, index_dir, a, i, names,
                               "train", negative_sample_per_10000)
            if not frame.empty:
                train_frames.append(frame)
    train = pd.concat(train_frames, ignore_index=True)
    train = train.sort_values(["channel_id", "prediction_time"], kind="mergesort")
    y_train = train["target"].to_numpy(dtype=np.int8)
    if set(y_train) != {0, 1}:
        raise ValueError("sampled train lacks both classes")
    x_train = clean_numeric(train, names)
    transform = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("type", OneHotEncoder(handle_unknown="ignore"), ["sensor_type"]),
    ], sparse_threshold=0)
    logistic = Pipeline([
        ("transform", transform),
        ("model", LogisticRegression(max_iter=300, class_weight="balanced")),
    ])
    logistic.fit(x_train, y_train)
    cat_train = x_train.copy()
    cat_train[numeric] = cat_train[numeric].fillna(-1)
    cat = CatBoostClassifier(
        iterations=250, depth=6, learning_rate=0.05, loss_function="Logloss",
        auto_class_weights="Balanced", cat_features=["sensor_type"],
        thread_count=4, random_seed=42, verbose=False,
        allow_writing_files=False,
    )
    cat.fit(cat_train, y_train)
    del train, train_frames, x_train, cat_train
    scores = defaultdict(list)
    labels = []
    types = []
    episode_best = defaultdict(dict)
    episode_type = {}
    channel_days = set()
    channel_days_by_type = set()
    validation_rows = 0
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=4")
        for a, i in pairs:
            frame = join_month(database, a3_dir, index_dir, a, i, names,
                               "validation", negative_sample_per_10000)
            if frame.empty:
                continue
            validation_rows += len(frame)
            y = frame["target"].to_numpy(dtype=np.int8)
            x = clean_numeric(frame, names)
            cat_x = x.copy()
            cat_x[numeric] = cat_x[numeric].fillna(-1)
            monthly = {
                "rule": rule_score(frame),
                "logistic_regression": logistic.predict_proba(x)[:, 1],
                "catboost": cat.predict_proba(cat_x)[:, 1],
            }
            labels.append(y)
            sensor_type = frame["sensor_type"].fillna("<unknown>").astype(str)
            types.append(sensor_type.to_numpy())
            for name, score in monthly.items():
                scores[name].append(np.asarray(score))
                for episode, row_type, value in zip(
                    frame.loc[y == 1, "target_episode_id"], sensor_type[y == 1], score[y == 1]
                ):
                    if pd.isna(episode):
                        raise ValueError("positive validation row lacks episode ID")
                    if episode in episode_type and episode_type[episode] != row_type:
                        raise ValueError("validation episode crosses sensor types")
                    episode_type[episode] = row_type
                    previous = episode_best[name].get(episode, -np.inf)
                    episode_best[name][episode] = max(previous, float(value))
            channel_days.update(zip(
                frame["channel_id"], frame["prediction_time"].dt.date,
            ))
            channel_days_by_type.update(zip(
                sensor_type, frame["channel_id"], frame["prediction_time"].dt.date,
            ))
            print(json.dumps({"validation_month": a["month"], "rows": len(frame)}), flush=True)
    y_validation = np.concatenate(labels)
    if set(y_validation) != {0, 1}:
        raise ValueError("validation lacks both classes")
    validation_types = np.concatenate(types)
    score_arrays = {name: np.concatenate(parts) for name, parts in scores.items()}
    global_metrics = {
        name: evaluate(y_validation, score, episode_best[name], len(channel_days))
        for name, score in score_arrays.items()
    }
    by_type = {}
    for sensor_type in sorted(set(validation_types)):
        mask = validation_types == sensor_type
        local_y = y_validation[mask]
        positives = int(np.count_nonzero(local_y == 1))
        negatives = int(np.count_nonzero(local_y == 0))
        days = sum(item[0] == sensor_type for item in channel_days_by_type)
        episodes = sum(item == sensor_type for item in episode_type.values())
        by_type[sensor_type] = {
            "rows": int(np.count_nonzero(mask)), "positive_hours": positives,
            "negative_hours": negatives, "positive_episodes": episodes,
            "channel_days": days, "models": {},
        }
        for name, score in score_arrays.items():
            threshold = global_metrics[name]["threshold_selected_on_validation"]
            local_score = score[mask]
            selected = local_score >= threshold
            tp = int(np.count_nonzero(selected & (local_y == 1)))
            fp = int(np.count_nonzero(selected & (local_y == 0)))
            positive_episode_scores = [
                value for episode, value in episode_best[name].items()
                if episode_type[episode] == sensor_type
            ]
            by_type[sensor_type]["models"][name] = {
                "average_precision": (
                    float(average_precision_score(local_y, local_score))
                    if positives and negatives else None
                ),
                "true_positive_hours": tp,
                "false_positive_hours": fp,
                "positive_episode_recall": (
                    sum(value >= threshold for value in positive_episode_scores) / episodes
                    if episodes else None
                ),
                "false_positive_hours_per_1000_channel_days": fp * 1000 / days,
            }
    report = {
        "schema_version": "r4-conditional-discrete-baselines-v1",
        "status": "validation_only",
        "r3_contract_sha256": sha256(contract_path),
        "r3_admission_manifest_sha256": contract["source_admission_manifest_sha256"],
        "feature_allowlist_sha256": contract["feature_allowlist_sha256"],
        "train_negative_sample_per_10000": negative_sample_per_10000,
        "training_rows": int(len(y_train)),
        "training_positive_rows": int(np.count_nonzero(y_train == 1)),
        "validation_rows": validation_rows,
        "validation_positive_rows": int(np.count_nonzero(y_validation == 1)),
        "validation_channel_days": len(channel_days),
        "models": global_metrics,
        "by_sensor_type": by_type,
        "limitations": [
            "Target is a future registered journal message under an archive-completeness assumption.",
            "Training uses deterministic negative subsampling; probabilities are not calibrated.",
            "Thresholds are selected on validation and are not final test metrics.",
            "The sealed test split was not loaded for model fitting, selection or thresholding.",
        ],
    }
    output_dir.mkdir(parents=True)
    joblib.dump(logistic, output_dir / "logistic_regression.joblib")
    cat.save_model(str(output_dir / "catboost.cbm"))
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": report["schema_version"],
        "status": report["status"],
        "report_sha256": sha256(output_dir / "report.json"),
        "logistic_sha256": sha256(output_dir / "logistic_regression.joblib"),
        "catboost_sha256": sha256(output_dir / "catboost.cbm"),
        "r3_contract_sha256": report["r3_contract_sha256"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--negative-sample-per-10000", type=int, default=200)
    args = parser.parse_args()
    report = run(contract_path=args.contract, allowlist_path=args.allowlist,
                 a3_dir=args.a3_dir, b3_dir=args.b3_dir, index_dir=args.index_dir,
                 output_dir=args.output_dir,
                 negative_sample_per_10000=args.negative_sample_per_10000)
    print(json.dumps({"models": report["models"],
                      "validation_rows": report["validation_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
