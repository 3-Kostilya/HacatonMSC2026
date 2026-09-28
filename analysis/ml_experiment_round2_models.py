"""Three train-only weighting/capacity hypotheses on the accepted temporal folds.

All sampled negative rows remain present. 2024 alone freezes model/threshold
choices before any 2025 scores are computed. The opened 2025 year is transfer
diagnostics, not a sealed holdout. Neither 2021 nor 2026 is accessed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from analysis.ml_experiment_features import engineered_input
from analysis.ml_experiment_pooled import META, score_files
from analysis.ml_experiment_eval import PreparedEvaluation, search_thresholds


CONFIGS = {
    "episode_capacity": {"engineered": True, "weighting": "episode", "iterations": 600, "depth": 6},
    "episode_recent": {"engineered": True, "weighting": "episode_recent", "iterations": 400, "depth": 6},
    "episode_lead": {"engineered": True, "weighting": "episode_lead", "iterations": 400, "depth": 6},
}
YEAR_FACTORS = {2019: 0.25, 2020: 0.25, 2022: 0.7, 2023: 1.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_for(train: pd.DataFrame, strategy: str) -> tuple[np.ndarray, list[float], dict]:
    """Keep targets fixed; positive episode mass is independent of hour count.

    Lead weighting normalizes the kernel *within* each episode, preserving equal
    episode totals. Recent weighting deliberately tilts both classes by year.
    """
    if strategy not in {"episode", "episode_recent", "episode_lead"}:
        raise ValueError(strategy)
    positive = train.target.eq(1)
    if not train.target.isin([0, 1]).all() or not positive.any() or positive.all():
        raise ValueError("weights require binary labels and both classes")
    ids = train.loc[positive, "target_episode_id"]
    if ids.isna().any():
        raise ValueError("positive train row lacks episode ID")
    kernel = pd.Series(1.0, index=ids.index)
    emphasized = 0
    if strategy == "episode_lead":
        lead = (pd.to_datetime(train.loc[positive, "label_available_at"])
                - pd.to_datetime(train.loc[positive, "prediction_time"])).dt.total_seconds() / 3600
        if lead.isna().any() or not ((lead > 0) & (lead <= 24)).all():
            raise ValueError("positive leadtime must lie in (0,24] hours")
        emphasis = lead.between(1, 6, inclusive="both")
        kernel.loc[emphasis] = 3.0
        emphasized = int(emphasis.sum())
    totals = kernel.groupby(ids).transform("sum")
    weight = np.ones(len(train), dtype=np.float32)
    weight[positive.to_numpy()] = (kernel / totals * (int(positive.sum()) / ids.nunique())).to_numpy(dtype=np.float32)
    if strategy == "episode_recent":
        years = pd.to_datetime(train.prediction_time).dt.year
        factors = years.map(YEAR_FACTORS)
        if factors.isna().any():
            raise ValueError("unexpected train year in recency weighting")
        weight *= factors.to_numpy(dtype=np.float32)
    positive_weight = float(weight[positive.to_numpy()].sum(dtype=np.float64))
    negative_weight = float(weight[~positive.to_numpy()].sum(dtype=np.float64))
    classes = [1.0, negative_weight / positive_weight]
    detail = {"positive_weight_sum": positive_weight, "negative_weight_sum": negative_weight,
              "class_weights": classes, "positive_episodes": int(ids.nunique()),
              "emphasized_1_to_6h_rows": emphasized, "minimum_sample_weight": float(weight.min()),
              "maximum_sample_weight": float(weight.max()), "negative_rows_retained": int((~positive).sum()),
              "positive_rows_retained": int(positive.sum()), "targets_changed": False,
              "year_factors": YEAR_FACTORS if strategy == "episode_recent" else None}
    return weight, classes, detail


def write_report(output: Path, report: dict) -> None:
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def run(data: Path, output: Path, configs: dict, *, resume: bool = False) -> dict:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["full_episode_count"]["validation"] != 2142:
        raise ValueError("fixed validation episode denominator differs")
    prior = (json.loads((output / "report.json").read_text(encoding="utf-8"))
             if resume and (output / "report.json").exists() else None)
    if not resume and any(output.iterdir()):
        raise FileExistsError("output is nonempty; use --resume for matching checkpoints")
    source_hashes = {split: sha256(data / f"{split}.parquet") for split in ("train", "tune", "validation")}
    if any(source_hashes[split] != manifest["files"][split]["sha256"] for split in source_hashes):
        raise ValueError("accepted data parquet hash differs")
    if prior is not None and prior["provenance"]["data_hashes"] != source_hashes:
        raise ValueError("resume data hash differs")
    base = manifest["all_features"]
    train = pq.read_table(data / "train.parquet", columns=list(dict.fromkeys(META + base))).to_pandas()
    years = sorted(int(year) for year in train.prediction_time.dt.year.unique())
    if not set(years) <= set(YEAR_FACTORS):
        raise ValueError(f"forbidden train year: {years}")
    matrix = engineered_input(train, base)
    report = {"schema_version": "ml-experiment-round2-models-v1", "experiments": {},
              "train_rows": len(train), "train_positive_hours": int(train.target.sum()),
              "train_years": years, "feature_count": len(matrix.columns), "feature_names": list(matrix.columns),
              "provenance": {"data_manifest_sha256": sha256(data / "manifest.json"), "data_hashes": source_hashes,
                             "source_hashes": manifest["source_hashes"], "feature_builder_sha256": sha256(Path("analysis/ml_experiment_features.py")),
                             "runner_sha256": sha256(Path(__file__)), "base_allowlist": base},
              "selection_scope": "full 2024 only; freeze selection.json before scoring opened 2025",
              "full_episode_count": manifest["full_episode_count"], "cooldown_hours": 24,
              "test_2026_read": False, "year_2021_read": False}
    models = {}
    for name, config in configs.items():
        model_path = output / f"{name}.cbm"
        checkpoint = prior.get("experiments", {}).get(name) if prior is not None else None
        if checkpoint is not None and model_path.exists():
            if checkpoint["config"] != config or sha256(model_path) != checkpoint["model_sha256"]:
                raise ValueError(f"resume checkpoint differs: {name}")
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            if model.feature_names_ != list(matrix.columns) or model.tree_count_ != config["iterations"]:
                raise ValueError(f"resume model structure differs: {name}")
            models[name], report["experiments"][name] = model, checkpoint
            print(f"resumed {name} ({model.tree_count_} trees)", flush=True)
            continue
        if model_path.exists():
            raise FileExistsError(f"untracked completed model would be overwritten: {model_path}")
        fit_started = time.monotonic()
        weights, class_weights, weight_details = weights_for(train, config["weighting"])
        model = CatBoostClassifier(iterations=config["iterations"], depth=config["depth"], learning_rate=.05,
                                   loss_function="Logloss", cat_features=["sensor_type"], class_weights=class_weights,
                                   thread_count=2, random_seed=42, verbose=100, allow_writing_files=False,
                                   border_count=64, l2_leaf_reg=5)
        model.fit(matrix, train.target.to_numpy(dtype=np.int8), sample_weight=weights)
        model.save_model(str(model_path))
        report["experiments"][name] = {"config": config, "weight_details": weight_details,
                                      "train_seconds": time.monotonic() - fit_started,
                                      "model_sha256": sha256(model_path),
                                      "top_feature_importances": sorted(zip(matrix.columns, model.feature_importances_.tolist()), key=lambda item: -item[1])[:35]}
        models[name] = model
        write_report(output, report)
        print(f"fitted {name}: {report['experiments'][name]['train_seconds']:.1f}s", flush=True)
    del train, matrix
    gc.collect()
    report["score_stats"] = {}
    report["score_stats"]["tune"] = score_files(models, configs, base, data / "tune.parquet", output / "scores_tune.parquet")
    tune = pq.read_table(output / "scores_tune.parquet").to_pandas()
    prepared = PreparedEvaluation(tune, full_episode_count=manifest["full_episode_count"]["tune"])
    for name in configs:
        column = f"score_{name}"
        curve = search_thresholds(prepared, column, points=61)
        choice = max(curve, key=lambda metric: (metric["full_episode_f1"], metric["episode_precision"]))
        report["experiments"][name]["tune"] = {"selected": choice, "threshold_curve": curve,
                                                 "hour_ap": float(average_precision_score(tune.target, tune[column]))}
        print(f"tune {name}: P={choice['episode_precision']:.4f} R={choice['full_episode_recall']:.4f}", flush=True)
    selected_name = max(configs, key=lambda name: report["experiments"][name]["tune"]["selected"]["full_episode_f1"])
    report["selected_variant"] = selected_name
    selection = {"selected_variant": selected_name,
                 "frozen_thresholds": {name: report["experiments"][name]["tune"]["selected"]["threshold"] for name in configs},
                 "selection_year": 2024, "source_scores_sha256": sha256(output / "scores_tune.parquet")}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    report["selection_sha256"] = sha256(output / "selection.json")
    write_report(output, report)
    del tune, prepared
    gc.collect()
    report["score_stats"]["validation"] = score_files(models, configs, base, data / "validation.parquet", output / "scores_validation.parquet")
    validation = pq.read_table(output / "scores_validation.parquet").to_pandas()
    prepared = PreparedEvaluation(validation, full_episode_count=2142)
    for name in configs:
        column = f"score_{name}"
        metric = prepared.evaluate(column, selection["frozen_thresholds"][name])
        report["experiments"][name]["validation_frozen"] = metric
        report["experiments"][name]["validation_hour_ap"] = float(average_precision_score(validation.target, validation[column]))
        print(f"2025 frozen {name}: P={metric['episode_precision']:.4f} R={metric['full_episode_recall']:.4f}", flush=True)
    report["score_hashes"] = {split: sha256(output / f"scores_{split}.parquet") for split in ("tune", "validation")}
    report["elapsed_seconds_this_process"] = time.monotonic() - started
    write_report(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("output/ml-experiment/data"))
    parser.add_argument("--output", type=Path, default=Path("output/ml-experiment-round2/models"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--capacity-iterations", type=int, choices=[400, 600], default=600)
    args = parser.parse_args()
    configs = {name: dict(config) for name, config in CONFIGS.items()}
    configs["episode_capacity"]["iterations"] = args.capacity_iterations
    run(args.data, args.output, configs, resume=args.resume)
