"""Versioned M0 interchange envelopes for the two-person ML pipeline.

Feature/model-specific columns require a later schema version. These envelopes
define the stable keys, provenance, nullability and local-time convention.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import math

import pyarrow as pa


CONTRACT_VERSION = "ml-m0-v1"
AVAILABILITY = frozenset({"eligible", "unknown", "excluded"})
SPLITS = frozenset({"train", "validation", "test"})
WINDOW_HOURS = (1, 6, 24, 168)
LOCAL_TIME = pa.timestamp("us")
PROVENANCE = (
    pa.field("schema_version", pa.string(), nullable=False),
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("config_sha256", pa.string(), nullable=False),
    pa.field("input_manifest_sha256", pa.string(), nullable=False),
)


SCHEMAS = {
    "clean": pa.schema(
        [
            *PROVENANCE,
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_row", pa.int64(), nullable=False),
            pa.field("event_id_raw", pa.string(), nullable=False),
            pa.field("channel_id_raw", pa.string(), nullable=False),
            pa.field("date_raw", pa.string(), nullable=False),
            pa.field("time_raw", pa.string(), nullable=False),
            pa.field("alarm_raw", pa.string(), nullable=False),
            pa.field("value_raw", pa.string(), nullable=False),
            pa.field("channel_id", pa.string(), nullable=False),
            pa.field("timestamp", LOCAL_TIME, nullable=False),
            pa.field("alarm", pa.bool_(), nullable=False),
            pa.field("value_numeric", pa.float64()),
            pa.field("value_state", pa.string()),
            pa.field("is_numeric", pa.bool_(), nullable=False),
            pa.field("sensor_type", pa.string()),
            pa.field("object_id", pa.string()),
            pa.field("join_status", pa.string(), nullable=False),
            pa.field("quality_flags", pa.list_(pa.string()), nullable=False),
        ]
    ),
    "hourly_features": pa.schema(
        [
            *PROVENANCE,
            pa.field("channel_id", pa.string(), nullable=False),
            pa.field("prediction_time", LOCAL_TIME, nullable=False),
            pa.field("sensor_type", pa.string()),
            pa.field("availability_status", pa.string(), nullable=False),
            pa.field("availability_reasons", pa.list_(pa.string()), nullable=False),
            pa.field("last_observation_age_seconds", pa.float64()),
            *[pa.field(f"event_count_{hours}h", pa.int64()) for hours in WINDOW_HOURS],
            *[pa.field(f"alarm_count_{hours}h", pa.int64()) for hours in WINDOW_HOURS],
            pa.field("numeric_median_24h", pa.float64()),
            pa.field("state_transitions_24h", pa.int64()),
        ]
    ),
    "anomaly_scores": pa.schema(
        [
            *PROVENANCE,
            pa.field("channel_id", pa.string(), nullable=False),
            pa.field("as_of", LOCAL_TIME, nullable=False),
            pa.field("method", pa.string(), nullable=False),
            pa.field("method_version", pa.string(), nullable=False),
            pa.field("score", pa.float64()),
            pa.field("score_measure", pa.string()),
            pa.field("score_status", pa.string(), nullable=False),
            pa.field("score_reasons", pa.list_(pa.string()), nullable=False),
            pa.field("evidence", pa.list_(pa.string()), nullable=False),
            pa.field("fit_end_at", LOCAL_TIME),
            pa.field("cluster_id", pa.string()),
            pa.field("cluster_distance", pa.float64()),
            pa.field("distance_measure", pa.string()),
            pa.field("distance_status", pa.string(), nullable=False),
        ]
    ),
    "pseudo_failure_episodes": pa.schema(
        [
            *PROVENANCE,
            pa.field("episode_id", pa.string(), nullable=False),
            pa.field("channel_id", pa.string(), nullable=False),
            pa.field("kind", pa.string(), nullable=False),
            pa.field("start_at", LOCAL_TIME, nullable=False),
            pa.field("confirmed_at", LOCAL_TIME, nullable=False),
            pa.field("end_at", LOCAL_TIME),
            pa.field("evidence", pa.list_(pa.string()), nullable=False),
            pa.field("labeler_version", pa.string(), nullable=False),
        ]
    ),
    "training_dataset": pa.schema(
        [
            *PROVENANCE,
            pa.field("channel_id", pa.string(), nullable=False),
            pa.field("prediction_time", LOCAL_TIME, nullable=False),
            pa.field("horizon_hours", pa.int16(), nullable=False),
            pa.field("target", pa.int8()),
            pa.field("eligibility", pa.string(), nullable=False),
            pa.field("eligibility_reasons", pa.list_(pa.string()), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("episode_id", pa.string()),
        ]
    ),
    "predictions": pa.schema(
        [
            *PROVENANCE,
            pa.field("channel_id", pa.string(), nullable=False),
            pa.field("prediction_time", LOCAL_TIME, nullable=False),
            pa.field("horizon_hours", pa.int16(), nullable=False),
            pa.field("score", pa.float64()),
            pa.field("probability", pa.float64()),
            pa.field("risk_level", pa.string()),
            pa.field("status", pa.string(), nullable=False),
            pa.field("status_reasons", pa.list_(pa.string()), nullable=False),
            pa.field("model_version", pa.string(), nullable=False),
        ]
    ),
}

KEYS = {
    "clean": ("source", "source_row"),
    "hourly_features": ("channel_id", "prediction_time"),
    "anomaly_scores": ("channel_id", "as_of", "method", "method_version"),
    "pseudo_failure_episodes": ("episode_id",),
    "training_dataset": ("channel_id", "prediction_time", "horizon_hours"),
    "predictions": ("channel_id", "prediction_time", "horizon_hours", "model_version"),
}

FORBIDDEN_ML_FEATURES = frozenset(
    {"source", "source_row", "event_id_raw", "split", "episode_id", "target", "run_id"}
)


def in_feature_window(timestamp: datetime, as_of: datetime, hours: int) -> bool:
    """A feature window is left-open and right-closed: (t-W, t]."""
    _require_local_times(timestamp, as_of)
    if hours <= 0:
        raise ValueError("hours must be positive")
    return as_of - timedelta(hours=hours) < timestamp <= as_of


def in_future_window(start_at: datetime, as_of: datetime, hours: int) -> bool:
    """A target onset window is left-open and right-closed: (t, t+H]."""
    _require_local_times(start_at, as_of)
    if hours <= 0:
        raise ValueError("hours must be positive")
    return as_of < start_at <= as_of + timedelta(hours=hours)


def _require_local_times(*values: datetime) -> None:
    if any(not isinstance(value, datetime) or value.tzinfo is not None for value in values):
        raise ValueError("timestamps must use the journal's local naive time")


def validate_table(name: str, table: pa.Table) -> None:
    """Validate the M0 envelope, keys, and unavailable-result semantics."""
    schema = SCHEMAS[name]
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"{name}: schema differs from {CONTRACT_VERSION}")
    seen = set()
    for row in table.to_pylist():
        if any(row[field.name] is None for field in schema if not field.nullable):
            raise ValueError(f"{name}: required field is null")
        key = tuple(row[field] for field in KEYS[name])
        if any(value is None for value in key) or key in seen:
            raise ValueError(f"{name}: missing or duplicate key {key!r}")
        seen.add(key)
        if row["schema_version"] != CONTRACT_VERSION:
            raise ValueError(f"{name}: unexpected schema version")
        for field in schema:
            value = row[field.name]
            if pa.types.is_floating(field.type) and value is not None and not math.isfinite(value):
                raise ValueError(f"{name}: {field.name} must be finite")
        for status, reasons in (
            ("availability_status", "availability_reasons"),
            ("score_status", "score_reasons"),
            ("eligibility", "eligibility_reasons"),
            ("status", "status_reasons"),
        ):
            if status in row and row[status] != "eligible" and not row[reasons]:
                raise ValueError(f"{name}: unavailable result requires reasons")
        if (
            not row["run_id"]
            or len(row["config_sha256"]) != 64
            or len(row["input_manifest_sha256"]) != 64
        ):
            raise ValueError(f"{name}: missing run/config provenance")
        if name == "training_dataset":
            if row["target"] not in (None, 0, 1):
                raise ValueError("training_dataset: target must be 0, 1 or null")
            if row["eligibility"] not in AVAILABILITY or row["split"] not in SPLITS:
                raise ValueError("training_dataset: invalid eligibility or split")
            if (row["target"] is None) != (row["eligibility"] != "eligible"):
                raise ValueError("training_dataset: unavailable target must be null")
            if row["horizon_hours"] <= 0:
                raise ValueError("training_dataset: horizon must be positive")
        if name == "pseudo_failure_episodes":
            if row["confirmed_at"] < row["start_at"]:
                raise ValueError("pseudo_failure_episodes: confirmation precedes onset")
            if row["end_at"] is not None and row["end_at"] < row["confirmed_at"]:
                raise ValueError("pseudo_failure_episodes: recovery precedes confirmation")
        if name == "predictions" and row["horizon_hours"] <= 0:
            raise ValueError("predictions: horizon must be positive")
        if name == "predictions":
            if row["status"] != "eligible" and (
                row["probability"] is not None or row["risk_level"] is not None
            ):
                raise ValueError("predictions: unavailable probability/risk must be null")
            if row["probability"] is not None and not 0 <= row["probability"] <= 1:
                raise ValueError("predictions: probability must be between 0 and 1")
            if row["risk_level"] not in (None, "LOW", "MEDIUM", "HIGH"):
                raise ValueError("predictions: invalid risk level")
        for status_field, value_field in (
            ("score_status", "score"),
            ("status", "score"),
        ):
            if status_field in row:
                if row[status_field] not in AVAILABILITY:
                    raise ValueError(f"{name}: invalid {status_field}")
                if row[status_field] != "eligible" and row[value_field] is not None:
                    raise ValueError(f"{name}: unavailable result must be null")
                if row[status_field] == "eligible" and row[value_field] is None:
                    raise ValueError(f"{name}: eligible result must have a score")
        if name == "hourly_features" and row["availability_status"] not in AVAILABILITY:
            raise ValueError("hourly_features: invalid availability status")
        if name == "anomaly_scores":
            if row["fit_end_at"] is not None and row["fit_end_at"] > row["as_of"]:
                raise ValueError("anomaly_scores: fit uses future data")
            if row["distance_status"] not in AVAILABILITY:
                raise ValueError("anomaly_scores: invalid distance status")
            if row["distance_status"] != "eligible" and row["cluster_distance"] is not None:
                raise ValueError("anomaly_scores: unavailable distance must be null")
            if row["distance_status"] == "eligible" and (
                row["cluster_distance"] is None or not row["distance_measure"]
            ):
                raise ValueError("anomaly_scores: available distance requires value and measure")
