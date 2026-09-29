"""Train one pinned research CatBoost from Q3 causal feature folds.

The configuration was selected on 2024 in the neighboring experiment. This
script does not search 2025, 2026, thresholds, sensor cohorts or labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ml.service_candidate.features import engineered_input
from ml.service_candidate.loader import ResearchRiskModel, sha256, source_sha256


Q2_SHA = "c9775a94bfcff09d1c641b010b93d62e9927209017ccbc7346749f78525de619"
Q3_SHA = "294d8d3a4cadb640737ec39f7adef2c8d344fcd8d4398724fcb779ebabcdf4db"
B3_SHA = "f5bf90b3f5878a749e12d1cfd7f36b843c6e97d38f2c033c1af1f6672d9332f9"
MODEL_CONFIG = {
    "iterations": 250,
    "depth": 5,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5,
    "border_count": 64,
    "random_seed": 42,
    "thread_count": 2,
    "loss_function": "Logloss",
}
META = (
    "channel_id",
    "prediction_time",
    "target",
    "target_episode_id",
    "label_available_at",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def episode_weights(train: pd.DataFrame) -> tuple[np.ndarray, list[float], dict]:
    positive = train.target.eq(1)
    if not train.target.isin([0, 1]).all() or positive.all() or not positive.any():
        raise ValueError("both approved binary classes are required")
    if train.loc[positive, "target_episode_id"].isna().any():
        raise ValueError("positive training hour lacks an episode ID")
    counts = train.loc[positive, "target_episode_id"].value_counts()
    weight = np.ones(len(train), dtype=np.float32)
    weight[positive.to_numpy()] = (
        1 / train.loc[positive, "target_episode_id"].map(counts).to_numpy()
    )
    weight[positive.to_numpy()] *= positive.sum() / weight[positive.to_numpy()].sum()
    negative = int((~positive).sum())
    positive_hours = int(positive.sum())
    return (
        weight,
        [1.0, negative / positive_hours],
        {
            "rows": len(train),
            "positive_hours": positive_hours,
            "sampled_negative_hours": negative,
            "positive_episodes": int(counts.size),
            "all_positive_hours_retained": True,
            "negative_sampling_fraction": 0.02,
            "sample_weights_for_equal_positive_episode_mass": True,
        },
    )


def run(
    *, data: Path, q2: Path, q3: Path, b3: Path, verification: Path, reference: Path, output: Path
) -> dict:
    begun = time.perf_counter()
    pending = output.with_name(output.name + ".inprogress")
    if output.exists() or pending.exists():
        raise FileExistsError(output)
    if (
        sha256(q2 / "manifest.json"),
        sha256(q3 / "manifest.json"),
        sha256(b3 / "manifest.json"),
    ) != (Q2_SHA, Q3_SHA, B3_SHA):
        raise ValueError("accepted Q2/Q3/B3 source manifests differ")
    audit = read_json(verification)
    if (
        audit["status"] != "independent_delta_invariants_features_and_oracle_verified"
        or audit["source_manifest_sha256"] != Q3_SHA
        or audit["new_rows_verified"] != 2554406
    ):
        raise ValueError("Q3 independent verification differs")
    manifest = read_json(data / "manifest.json")
    if (
        manifest["schema_version"] != "round2-q3-research-intake-v1"
        or manifest["source_q3_manifest_sha256"] != Q3_SHA
        or manifest["source_hashes"] != {"q2": Q2_SHA, "b3": B3_SHA}
        or manifest["admission_policy"] != "combined"
        or manifest["physical_availability_approved"]
        or manifest["train_years"] != [2019, 2020, 2022, 2023]
        or manifest["tune_year"] != 2024
        or manifest["validation_year"] != 2025
        or manifest["negative_sampling_fraction"] != 0.02
        or manifest["row_stats"]["train"]["rows"] != 1011240
    ):
        raise ValueError("unexpected neighboring research fold contract")
    if (
        sha256(data / "train.parquet") != manifest["files"]["train"]["sha256"]
        or sha256(data / "tune.parquet") != manifest["files"]["tune"]["sha256"]
    ):
        raise ValueError("training/2024 fold hash differs")
    names = manifest["all_features"]
    frame = pq.read_table(
        data / "train.parquet", columns=list(dict.fromkeys((*META, *names)))
    ).to_pandas()
    if (
        len(frame) != manifest["row_stats"]["train"]["rows"]
        or frame.target.sum() != manifest["row_stats"]["train"]["positive_hours"]
        or set(frame.prediction_time.dt.year) != set(manifest["train_years"])
        or frame.label_available_at.ge(pd.Timestamp("2024-01-01")).any()
        or frame.duplicated(["channel_id", "prediction_time"]).any()
    ):
        raise ValueError("training rows, years, label boundary or keys differ")
    matrix = engineered_input(frame, names)
    weights, classes, train_stats = episode_weights(frame)
    if matrix.shape[1] != 304 or matrix.columns.duplicated().any():
        raise ValueError("the neighboring 304-feature transformation differs")
    reference_model = reference / "engineered_episode.cbm"
    selection = read_json(reference / "selection.json")
    if selection["selected_variant"] != "engineered_episode":
        raise ValueError("reference model was not selected on 2024")
    threshold = selection["frozen_thresholds"]["engineered_episode"]
    if not 0 < threshold < 1:
        raise ValueError("invalid frozen 2024 research threshold")
    pending.mkdir(parents=True)
    model = CatBoostClassifier(
        **MODEL_CONFIG,
        cat_features=["sensor_type"],
        class_weights=classes,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(matrix, frame.target.to_numpy(dtype=np.int8), sample_weight=weights)
    model_path = pending / "model.cbm"
    model.save_model(str(model_path))
    if model.feature_names_ != matrix.columns.tolist():
        raise ValueError("trained model feature contract differs")
    reference_loaded = CatBoostClassifier()
    reference_loaded.load_model(str(reference_model))
    if reference_loaded.feature_names_ != matrix.columns.tolist():
        raise ValueError("neighbor model has a different feature order")
    sample = matrix.iloc[:4096]
    score = model.predict_proba(sample, thread_count=2)[:, 1]
    reference_score = reference_loaded.predict_proba(sample, thread_count=2)[:, 1]
    maximum_gap = float(np.max(np.abs(score - reference_score)))
    identical_model = sha256(model_path) == sha256(reference_model)
    if not identical_model and maximum_gap > 1e-7:
        raise ValueError("new training differs from the independently scored neighbor model")
    contract_sha = source_sha256(Path(__file__).with_name("features.py"))
    metadata = {
        "schema_version": "registered-journal-fault-service-research-v1",
        "model_sha256": sha256(model_path),
        "base_feature_names": names,
        "engineered_feature_names": matrix.columns.tolist(),
        "feature_transform_source_sha256": contract_sha,
        "source_train_manifest_sha256": sha256(data / "manifest.json"),
        "source_q2_manifest_sha256": Q2_SHA,
        "source_q3_manifest_sha256": Q3_SHA,
        "source_b3_manifest_sha256": B3_SHA,
        "source_q3_independent_verification_sha256": sha256(verification),
        "reference_model_sha256": sha256(reference_model),
        "exact_reference_model_file": identical_model,
        "maximum_prediction_gap_on_4096_training_hours": maximum_gap,
        "training_years": manifest["train_years"],
        "research_tuning_year": 2024,
        "open_research_diagnostic_year": 2025,
        "excluded_year": 2021,
        "old_open_test_year_used": False,
        "target": "new registered journal 'Неисправен' episode in (t,t+24h]",
        "physical_failure_prediction_claim": False,
        "score_is_calibrated_probability": False,
        "research_threshold_from_2024": threshold if identical_model else None,
        "research_threshold_is_operational": False,
        "production_approved": False,
        "automatic_actions_allowed": False,
        "full_episode_recall_denominator_in_open_2025": 2142,
        "training": train_stats,
        "model_config": MODEL_CONFIG,
        "admission_policy": "combined_Q3_research_not_jointly_approved",
        "service_input": "precomputed causal Q2/Q3 121 features plus explicit admission_status",
        "service_output": "uncalibrated risk_score; warning remains null",
    }
    write_json(pending / "model_metadata.json", metadata)
    loaded = ResearchRiskModel(pending)
    service_sample = frame.iloc[:128][names].copy()
    service_sample["admission_status"] = "eligible"
    loaded_scores = loaded.score(service_sample).risk_score.to_numpy()
    if not np.allclose(loaded_scores, score[:128], rtol=0, atol=1e-12):
        raise ValueError("serialized service loader differs from the trained model")
    report = {
        "status": "trained_research_model_not_production_approved",
        "model_identical_to_independently_scored_neighbor": identical_model,
        "sample_score_maximum_gap": maximum_gap,
        "serialization_loader_parity": True,
        "model_sha256": sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "train_rows": len(frame),
        "train_positive_hours": train_stats["positive_hours"],
        "train_positive_episodes": train_stats["positive_episodes"],
        "research_threshold_2024": metadata["research_threshold_from_2024"],
        "opened_2025_not_used_for_selection": True,
        "test_2026_not_read": True,
        "elapsed_seconds": round(time.perf_counter() - begun, 3),
    }
    write_json(pending / "training_report.json", report)
    pending.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "q2", "q3", "b3", "verification", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
