"""Build an outcome-blind, channel/month-diverse audit sample of ML warnings.

This is a diagnostic reweighting, not an estimate of production-stream Precision
or a new alert policy. The 2026 test and excluded 2021 year are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from analysis.prepare_ml_experiment import sha256


SOURCES = {
    2024: Path("output/ml-experiment-round7/fullstream-2024-unknown21-v1"),
    2025: Path("output/ml-experiment-round7/fullstream-2025-unknown21-v2"),
}
SEED = "diversity-audit-v1"
CHANNEL_MONTH_CAP = 1
TYPE_MONTH_CAP = 30
SENSITIVITY_RUNS = 30


def select_keys(metadata: pd.DataFrame, seed: str = SEED) -> pd.DataFrame:
    """Use only keys and sensor type; do not accept labels or model outcomes."""
    required = {"channel_id", "prediction_time", "sensor_type"}
    if set(metadata.columns) != required:
        raise ValueError("selection accepts only label-free warning metadata")
    frame = metadata.copy()
    if frame.duplicated(["channel_id", "prediction_time"]).any():
        raise AssertionError("warning keys are not unique")
    frame["month"] = frame.prediction_time.dt.month
    frame["selection_hash"] = [
        hashlib.sha256(f"{seed}|{row.sensor_type}|{row.channel_id}|"
                       f"{row.prediction_time.isoformat()}".encode()).hexdigest()
        for row in frame.itertuples(index=False)
    ]
    frame = frame.sort_values(["selection_hash", "channel_id", "prediction_time"],
                              kind="stable")
    frame = frame.groupby(["sensor_type", "month", "channel_id"],
                          sort=False, group_keys=False).head(CHANNEL_MONTH_CAP)
    frame = frame.groupby(["sensor_type", "month"],
                          sort=False, group_keys=False).head(TYPE_MONTH_CAP)
    return frame.sort_values(["prediction_time", "channel_id"],
                             kind="stable").reset_index(drop=True)


def counts(frame: pd.DataFrame) -> dict:
    outcomes = frame.outcome.value_counts().to_dict()
    n = len(frame)
    positive = int(outcomes.get("matched_known_episode", 0))
    assigned = int((frame.outcome.eq("matched_known_episode") &
                    frame.split_status.eq("assigned")).sum())
    return {
        "warnings": n,
        "matched_known_episode": positive,
        "matched_assigned_episode": assigned,
        "known_no_target": int(outcomes.get("known_no_target", 0)),
        "unknown_target": int(outcomes.get("unknown_target", 0)),
        "precision_lower_bound": positive / n if n else None,
        "assigned_only_precision_lower_bound": assigned / n if n else None,
        "channels": int(frame.channel_id.nunique()),
        "sensor_types": int(frame.sensor_type.nunique()),
        "months": int(frame.prediction_time.dt.month.nunique()),
        "temperature_warnings": int(frame.sensor_type.eq("Датчик температуры").sum()),
        "by_sensor_type": {
            kind: int(value) for kind, value in frame.sensor_type.value_counts().items()
        },
    }


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    # Metadata are read and selected before outcome-bearing warning files are opened.
    selections = {}
    source_hashes = {}
    for year, root in SOURCES.items():
        source = root / "candidate_warnings.parquet"
        source_hashes[year] = sha256(source)
        metadata = pd.read_parquet(
            source, columns=["channel_id", "prediction_time", "sensor_type"])
        selections[year] = select_keys(metadata)
    output.mkdir(parents=True)
    for year, keys in selections.items():
        keys.to_parquet(output / f"selection_keys_{year}.parquet", index=False)
    results = {}
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        for year, root in SOURCES.items():
            source = root / "candidate_warnings.parquet"
            label_source = ("output/r3-b-full-months-20260925-v2/"
                            f"year={year}/month=*/registered_forecast_labels.parquet")
            full = pd.read_parquet(source)
            keys = selections[year][["channel_id", "prediction_time", "sensor_type"]]
            selected = full.merge(keys, on=["channel_id", "prediction_time", "sensor_type"],
                                  how="inner", validate="one_to_one")
            if len(selected) != len(keys):
                raise AssertionError("frozen selection lost warning keys")
            def with_status(warnings: pd.DataFrame) -> pd.DataFrame:
                db.register("warnings", warnings)
                joined = db.execute('''SELECT w.*,l.split_status,l.sensor_type AS label_sensor_type
                    FROM warnings w JOIN read_parquet(?,hive_partitioning=false) l
                    USING(channel_id,prediction_time)''', [label_source]).fetch_df()
                db.unregister("warnings")
                if (len(joined) != len(warnings) or
                        not joined.sensor_type.eq(joined.label_sensor_type).all()):
                    raise AssertionError("selected warning/B3 label parity differs")
                return joined.drop(columns="label_sensor_type")
            full = with_status(full)
            selected = with_status(selected)
            selected.to_parquet(output / f"diverse_warnings_{year}.parquet", index=False)
            results[year] = {
                "source_warning_sha256": source_hashes[year],
                "selection_keys_sha256": sha256(output / f"selection_keys_{year}.parquet"),
                "selected_warning_sha256": sha256(output / f"diverse_warnings_{year}.parquet"),
                "full_stream": counts(full),
                "diverse_audit_sample": counts(selected),
            }
            metadata = full[["channel_id", "prediction_time", "sensor_type"]]
            sensitivity = []
            for index in range(SENSITIVITY_RUNS):
                other_keys = select_keys(metadata, f"{SEED}-sensitivity-{index:02d}")
                other = full.merge(
                    other_keys[["channel_id", "prediction_time", "sensor_type"]],
                    on=["channel_id", "prediction_time", "sensor_type"],
                    how="inner", validate="one_to_one")
                sensitivity.append(counts(other))
            results[year]["seed_sensitivity"] = {
                "runs": SENSITIVITY_RUNS,
                "precision_lower_bound_min_median_max": [
                    float(value) for value in np.quantile(
                        [item["precision_lower_bound"] for item in sensitivity],
                        [0, .5, 1])],
                "assigned_only_precision_lower_bound_min_median_max": [
                    float(value) for value in np.quantile(
                        [item["assigned_only_precision_lower_bound"]
                         for item in sensitivity], [0, .5, 1])],
                "warnings_min_max": [min(item["warnings"] for item in sensitivity),
                                      max(item["warnings"] for item in sensitivity)],
            }
    report = {
        "status": "outcome_blind_diversity_audit_not_production_precision",
        "selection_fields": ["channel_id", "prediction_time", "sensor_type"],
        "seed": SEED,
        "max_warning_per_channel_month": CHANNEL_MONTH_CAP,
        "max_warning_per_sensor_type_month": TYPE_MONTH_CAP,
        "years": results,
        "test_2026_read": False,
        "data_2021_read": False,
        "limits": [
            "The audit changes warning composition; it is not a production-stream estimate.",
            "Unknown outcomes are not negatives, and purged-boundary hits are excluded from assigned-only Precision.",
            "Selected warnings are sampled after the sequential stream and are not a new cooldown policy.",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({year: item["diverse_audit_sample"] for year, item in results.items()},
                     ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("output/ml-experiment-diversity-v1"))
    run(parser.parse_args().output)
