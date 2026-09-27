"""Bounded pooled CatBoost comparison with model selection confined to 2024.

Run from project root: python -m analysis.ml_experiment_pooled.
2025 is previously opened exploratory transfer data, never a selection source.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from analysis.ml_experiment_features import engineered_input

META = ["channel_id", "prediction_time", "sensor_type", "target", "target_episode_id", "label_available_at"]
CONFIGS = {
    "base121_balanced": {"engineered": False, "weights": "balanced", "iterations": 250, "depth": 6},
    "engineered_balanced": {"engineered": True, "weights": "balanced", "iterations": 400, "depth": 6},
    "engineered_moderate": {"engineered": True, "weights": "sqrt", "iterations": 400, "depth": 6},
    "engineered_episode": {"engineered": True, "weights": "episode", "iterations": 250, "depth": 5},
    "engineered_unweighted": {"engineered": True, "weights": "none", "iterations": 250, "depth": 5},
}


def training_weights(train: pd.DataFrame, strategy: str) -> tuple[np.ndarray, list[float] | None]:
    positive = train.target.to_numpy() == 1
    ratio = float((~positive).sum() / positive.sum())
    sample_weight = np.ones(len(train), dtype=np.float32)
    if strategy == "episode":
        if train.loc[positive, "target_episode_id"].isna().any():
            raise ValueError("positive train row lacks target episode")
        counts = train.loc[positive, "target_episode_id"].value_counts()
        sample_weight[positive] = 1 / train.loc[positive, "target_episode_id"].map(counts).to_numpy()
        sample_weight[positive] *= positive.sum() / sample_weight[positive].sum()
    return sample_weight, ([1.0, ratio] if strategy in {"balanced", "episode"}
                           else [1.0, np.sqrt(ratio)] if strategy == "sqrt" else None)


def score_files(models: dict, configs: dict, base: list[str], input_path: Path, output_path: Path,
                *, feature_builder=engineered_input) -> dict:
    writer = None
    stats = {"rows": 0, "positives": 0}
    try:
        for batch in pq.ParquetFile(input_path).iter_batches(batch_size=70_000, columns=list(dict.fromkeys(META + base))):
            frame = batch.to_pandas()
            out = frame[META].copy()
            raw = feature_builder(frame, base, engineered=False)
            engineered = feature_builder(frame, base, engineered=True)
            for name, model in models.items():
                matrix = engineered if configs[name]["engineered"] else raw
                out[f"score_{name}"] = model.predict_proba(matrix, thread_count=2)[:, 1].astype("float32")
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            stats["rows"] += len(frame)
            stats["positives"] += int(frame.target.sum())
            del frame, raw, engineered, out, table
    finally:
        if writer is not None:
            writer.close()
    return stats


def run(data: Path, output: Path, selected_configs: list[str] | None = None,
        *, feature_builder=engineered_input, configs_override: dict | None = None,
        resume: bool = False) -> dict:
    from analysis.ml_experiment_eval import PreparedEvaluation, search_thresholds
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    base = manifest["all_features"]
    train = pq.read_table(data / "train.parquet", columns=list(dict.fromkeys(META + base))).to_pandas()
    years = set(train.prediction_time.dt.year.unique())
    if not years <= {2019, 2020, 2022, 2023}:
        raise ValueError(f"train years differ: {years}")
    raw = feature_builder(train, base, engineered=False)
    engineered = feature_builder(train, base, engineered=True)
    models, report = {}, {"train_rows": len(train), "train_positive_hours": int(train.target.sum()),
                          "train_years": sorted(int(year) for year in years), "experiments": {}}
    saved_report = (json.loads((output / "report.json").read_text(encoding="utf-8"))
                    if resume and (output / "report.json").exists() else {})
    available_configs = CONFIGS if configs_override is None else configs_override
    configs = {name: available_configs[name] for name in
               (selected_configs if selected_configs is not None else available_configs)}
    report["feature_builder"] = getattr(feature_builder, "__qualname__", type(feature_builder).__name__)
    for name, config in configs.items():
        started = time.monotonic()
        weight, class_weights = training_weights(train, config["weights"])
        matrix = engineered if config["engineered"] else raw
        path = output / f"{name}.cbm"
        saved = saved_report.get("experiments", {}).get(name)
        if resume and path.exists() and saved is not None:
            if saved["config"] != config or saved["features"] != list(matrix.columns):
                raise ValueError(f"resume config or feature list differs: {name}")
            model = CatBoostClassifier()
            model.load_model(str(path))
            if model.feature_names_ != list(matrix.columns) or model.tree_count_ != config["iterations"]:
                raise ValueError(f"resume model feature names or iterations differ: {name}")
            models[name] = model
            report["experiments"][name] = saved
            print(f"resumed {name}: {model.tree_count_} trees", flush=True)
            continue
        model = CatBoostClassifier(iterations=config["iterations"], depth=config["depth"], learning_rate=0.05,
                                   loss_function="Logloss", cat_features=["sensor_type"], class_weights=class_weights,
                                   thread_count=2, random_seed=42, verbose=False, allow_writing_files=False,
                                   l2_leaf_reg=5, border_count=64)
        model.fit(matrix, train.target.to_numpy(dtype=np.int8), sample_weight=weight)
        model.save_model(str(path))
        importance = sorted(zip(matrix.columns, model.feature_importances_.tolist()), key=lambda item: -item[1])
        report["experiments"][name] = {"config": config, "features": list(matrix.columns),
                                        "top_feature_importances": importance[:35], "train_seconds": time.monotonic() - started}
        models[name] = model
        print(f"fitted {name}: {len(matrix.columns)} features, {time.monotonic()-started:.1f}s", flush=True)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    del train, raw, engineered, matrix
    gc.collect()
    report["score_stats"] = {}
    # Thresholds and variant choice are frozen before any 2025 scores are inspected.
    report["score_stats"]["tune"] = score_files(models, configs, base, data / "tune.parquet",
                                                 output / "scores_tune.parquet", feature_builder=feature_builder)
    tune = pq.read_table(output / "scores_tune.parquet").to_pandas()
    evaluation = PreparedEvaluation(tune, full_episode_count=manifest["full_episode_count"]["tune"])
    for name in configs:
        column = f"score_{name}"
        curve = search_thresholds(evaluation, column, points=45)
        best = max(curve, key=lambda item: (item["full_episode_f1"], item["episode_precision"]))
        report["experiments"][name]["tune"] = {"hour_ap": float(average_precision_score(tune.target, tune[column])),
                                                 "selected": best, "threshold_curve": curve}
        print(f"tune {name}: P={best['episode_precision']:.4f} R={best['full_episode_recall']:.4f}", flush=True)
    report["selected_variant"] = max(configs, key=lambda name: report["experiments"][name]["tune"]["selected"]["full_episode_f1"])
    (output / "selection.json").write_text(json.dumps({"selected_variant": report["selected_variant"],
                                                        "frozen_thresholds": {name: report["experiments"][name]["tune"]["selected"]["threshold"] for name in configs}}, indent=2), encoding="utf-8")
    del tune, evaluation
    gc.collect()
    report["score_stats"]["validation"] = score_files(models, configs, base, data / "validation.parquet",
                                                       output / "scores_validation.parquet", feature_builder=feature_builder)
    validation = pq.read_table(output / "scores_validation.parquet").to_pandas()
    evaluation = PreparedEvaluation(validation, full_episode_count=manifest["full_episode_count"]["validation"])
    for name in configs:
        column = f"score_{name}"
        threshold = report["experiments"][name]["tune"]["selected"]["threshold"]
        metric = evaluation.evaluate(column, threshold)
        report["experiments"][name]["validation_frozen"] = metric
        report["experiments"][name]["validation_hour_ap"] = float(average_precision_score(validation.target, validation[column]))
        print(f"2025 frozen {name}: P={metric['episode_precision']:.4f} R={metric['full_episode_recall']:.4f}", flush=True)
    report["selection_scope"] = "2024 only; 2025 previously opened exploratory temporal transfer"
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_saved_scores(data: Path, output: Path) -> dict:
    """Re-evaluate saved predictions without fitting or reading feature tables."""
    from analysis.ml_experiment_eval import PreparedEvaluation, search_thresholds
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    tune = pq.read_table(output / "scores_tune.parquet").to_pandas()
    prepared = PreparedEvaluation(tune, full_episode_count=manifest["full_episode_count"]["tune"])
    for name, experiment in report["experiments"].items():
        column = f"score_{name}"
        curve = search_thresholds(prepared, column, points=45)
        choice = max(curve, key=lambda metric: (metric["full_episode_f1"], metric["episode_precision"]))
        experiment["tune"] = {"hour_ap": float(average_precision_score(tune.target, tune[column])),
                              "selected": choice, "threshold_curve": curve}
    report["selected_variant"] = max(report["experiments"], key=lambda name:
                                       report["experiments"][name]["tune"]["selected"]["full_episode_f1"])
    thresholds = {name: experiment["tune"]["selected"]["threshold"]
                  for name, experiment in report["experiments"].items()}
    (output / "selection.json").write_text(json.dumps({"selected_variant": report["selected_variant"],
                                                        "frozen_thresholds": thresholds}, indent=2), encoding="utf-8")
    del tune, prepared
    gc.collect()
    validation = pq.read_table(output / "scores_validation.parquet").to_pandas()
    prepared = PreparedEvaluation(validation, full_episode_count=manifest["full_episode_count"]["validation"])
    for name, experiment in report["experiments"].items():
        column = f"score_{name}"
        metric = prepared.evaluate(column, thresholds[name])
        experiment["validation_frozen"] = metric
        experiment["validation_hour_ap"] = float(average_precision_score(validation.target, validation[column]))
        print(f"audited2025 {name}: P={metric['episode_precision']:.4f} R={metric['full_episode_recall']:.4f}", flush=True)
    report["evaluator_replay"] = "fresh process; canonical float32 threshold comparisons"
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("output/ml-experiment/data"))
    parser.add_argument("--output", type=Path, default=Path("output/ml-experiment/pooled"))
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evaluate_saved_scores(args.data, args.output) if args.evaluate_only else run(args.data, args.output, args.configs, resume=args.resume)
