"""Produce smoke-sensor risk scores from historical events available at --as-of."""

import argparse
import hashlib
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from dataset import FAULT, NONFAULT, canonicalize, features_at
from extract import OUT, ROOT


def load_events(path):
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        raise ValueError("Events must be a CSV or Parquet file")
    frame = frame.rename(
        columns={
            "ид_канала_данных": "channel_id",
            "значение_датчика": "state",
            "тревожное": "alarm",
        }
    )
    if "timestamp" not in frame and {"дата", "время"}.issubset(frame):
        frame["timestamp"] = pd.to_datetime(frame["дата"] + " " + frame["время"])
    return frame


def predict(events, as_of, model_dir=OUT, dictionary_path=None):
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is not None:
        raise ValueError("Use the same timezone-naive local timestamp format as the source journal")
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    builder_hash = hashlib.sha256(Path(__file__).with_name("dataset.py").read_bytes()).hexdigest()
    if builder_hash != metadata["feature_builder_sha256"]:
        raise ValueError("Feature builder changed; rebuild the model before inference")
    model_path = model_dir / "catboost_smoke_failure.cbm"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model and metadata do not match")
    dictionary_path = dictionary_path or ROOT / "dataset" / "справочник_каналов_датчиков.csv"
    dictionary = pd.read_csv(dictionary_path, dtype=str, keep_default_na=False)
    if dictionary["ид_канала_данных"].duplicated().any():
        raise ValueError("Duplicate channel IDs in dictionary")
    smoke_ids = set(dictionary.loc[dictionary["тип_датчика"].eq("Датчик дыма"), "ид_канала_данных"])
    events = events.copy()
    events["channel_id"] = events.channel_id.astype(str)
    events["timestamp"] = pd.to_datetime(events.timestamp, errors="raise")
    # Filtering precedes both feature computation and eligibility decisions.
    events = events.loc[events.channel_id.isin(smoke_ids) & events.timestamp.le(as_of)]
    if events.empty:
        raise ValueError("No smoke-sensor history at or before the requested time")
    canonical = canonicalize(events)
    rows, parts = [], []
    for channel, group in canonical.groupby("channel_id", sort=True):
        latest = group.iloc[-1]
        age = (as_of - latest.timestamp).total_seconds() / 3600
        if latest.state == FAULT:
            status = "already_faulty"
        elif latest.state not in NONFAULT:
            status = "unknown_state"
        elif age > metadata["max_gap_hours"]:
            status = "stale_observation"
        elif as_of - group.timestamp.iloc[0] < pd.Timedelta(
            hours=metadata["minimum_history_hours"]
        ):
            status = "insufficient_history"
        else:
            status = "scored"
            parts.append(features_at(group, [as_of], metadata["max_gap_hours"]))
        rows.append(
            {
                "channel_id": channel,
                "timestamp": as_of,
                "status": status,
                "last_state": latest.state,
                "hours_since_event": age,
                "risk_score": np.nan,
                "warning": pd.NA,
            }
        )
    output = pd.DataFrame(rows).set_index("channel_id")
    if parts:
        features = pd.concat(parts, ignore_index=True)
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        score = model.predict_proba(features[metadata["features"]])[:, 1]
        output.loc[features.channel_id, "risk_score"] = score
        output.loc[features.channel_id, "warning"] = score >= metadata["threshold"]
    return output.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, help="Defaults to events.parquet inside --model-dir")
    parser.add_argument(
        "--as-of",
        required=True,
        help="Timestamp in the journal's local time, e.g. 2023-09-01T12:00:00",
    )
    parser.add_argument("--model-dir", type=Path, default=OUT)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Defaults to risk_scores.csv inside --model-dir"
    )
    args = parser.parse_args()
    event_path = args.events or args.model_dir / "events.parquet"
    output_path = args.output or args.model_dir / "risk_scores.csv"
    result = predict(load_events(event_path), args.as_of, args.model_dir, args.dictionary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.status.value_counts().to_string())
    print(f"Saved {len(result)} channels to {output_path}")


if __name__ == "__main__":
    main()
