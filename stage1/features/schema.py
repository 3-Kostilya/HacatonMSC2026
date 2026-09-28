"""Versioned hourly feature schema and the exact M0 interchange projection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import math
from typing import Any

import pyarrow as pa

from stage1.ml_m0_contracts import (
    CONTRACT_VERSION,
    PROVENANCE,
    SCHEMAS,
    validate_table,
)


FEATURE_VERSION = "ml-m2-features-v1"
WINDOW_HOURS = (1, 6, 24, 168)
STATUSES = frozenset({"eligible", "unknown", "excluded"})

_WINDOW_NUMERIC_FIELDS = (
    "numeric_median",
    "numeric_mad",
    "numeric_iqr",
    "numeric_std",
    "numeric_q10",
    "numeric_q90",
    "numeric_min",
    "numeric_max",
    "numeric_range",
    "numeric_delta",
    "numeric_slope_per_hour",
)


def _window_fields(hours: int) -> list[pa.Field]:
    suffix = f"_{hours}h"
    return [
        *[
            pa.field(f"{name}{suffix}", pa.int64(), nullable=False)
            for name in (
                "event_count",
                "alarm_count",
                "excluded_quality_count",
                "numeric_count",
                "state_count",
            )
        ],
        *[pa.field(f"{name}{suffix}", pa.float64()) for name in _WINDOW_NUMERIC_FIELDS],
        pa.field(f"state_transitions{suffix}", pa.int64()),
        pa.field(f"state_distinct_count{suffix}", pa.int64()),
        pa.field(f"maximum_gap_seconds{suffix}", pa.float64()),
        pa.field(f"coverage{suffix}", pa.float64()),
        pa.field(f"window_status{suffix}", pa.string(), nullable=False),
        pa.field(f"window_reasons{suffix}", pa.list_(pa.string()), nullable=False),
    ]


A2_SCHEMA = pa.schema(
    [
        *PROVENANCE,
        pa.field("channel_id", pa.string(), nullable=False),
        pa.field("prediction_time", pa.timestamp("us"), nullable=False),
        pa.field("sensor_type", pa.string()),
        pa.field("object_id", pa.string()),
        pa.field("availability_status", pa.string(), nullable=False),
        pa.field("availability_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("last_observation_age_seconds", pa.float64()),
        pa.field("baseline_fit_end_at", pa.timestamp("us"), nullable=False),
        pa.field("baseline_status", pa.string(), nullable=False),
        pa.field("baseline_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("baseline_event_count", pa.int64(), nullable=False),
        pa.field("baseline_numeric_count", pa.int64(), nullable=False),
        pa.field("baseline_state_count", pa.int64(), nullable=False),
        pa.field("baseline_numeric_median", pa.float64()),
        pa.field("baseline_numeric_mad", pa.float64()),
        pa.field("baseline_dominant_state", pa.string()),
        pa.field("peer_status", pa.string(), nullable=False),
        pa.field("peer_reasons", pa.list_(pa.string()), nullable=False),
        *[field for hours in WINDOW_HOURS for field in _window_fields(hours)],
    ]
)


def validate_a2_table(table: pa.Table) -> None:
    """Validate the physical A2 schema, keys, null statuses and finite features."""

    if not table.schema.equals(A2_SCHEMA, check_metadata=False):
        raise ValueError("hourly_features: schema differs from ml-m2-features-v1")
    seen: set[tuple[str, datetime]] = set()
    for row in table.to_pylist():
        if any(row[field.name] is None for field in A2_SCHEMA if not field.nullable):
            raise ValueError("hourly_features: required field is null")
        key = row["channel_id"], row["prediction_time"]
        if key in seen:
            raise ValueError(f"hourly_features: duplicate key {key!r}")
        seen.add(key)
        if row["schema_version"] != FEATURE_VERSION:
            raise ValueError("hourly_features: unexpected schema version")
        if (
            not row["run_id"]
            or len(row["config_sha256"]) != 64
            or len(row["input_manifest_sha256"]) != 64
        ):
            raise ValueError("hourly_features: missing run/config provenance")
        for field in A2_SCHEMA:
            value = row[field.name]
            if pa.types.is_floating(field.type) and value is not None and not math.isfinite(value):
                raise ValueError(f"hourly_features: {field.name} must be finite")
            if pa.types.is_integer(field.type) and value is not None and value < 0:
                raise ValueError(f"hourly_features: {field.name} cannot be negative")
        for status_name, reasons_name in (
            ("availability_status", "availability_reasons"),
            ("baseline_status", "baseline_reasons"),
            ("peer_status", "peer_reasons"),
            *((f"window_status_{hours}h", f"window_reasons_{hours}h") for hours in WINDOW_HOURS),
        ):
            if row[status_name] not in STATUSES:
                raise ValueError(f"hourly_features: invalid {status_name}")
            if row[status_name] != "eligible" and not row[reasons_name]:
                raise ValueError(f"hourly_features: {status_name} requires reasons")
        if row["baseline_fit_end_at"] > row["prediction_time"]:
            raise ValueError("hourly_features: baseline uses future data")
        for hours in WINDOW_HOURS:
            suffix = f"_{hours}h"
            if row[f"alarm_count{suffix}"] > row[f"event_count{suffix}"]:
                raise ValueError("hourly_features: alarm count exceeds event count")
            usable_count = row[f"event_count{suffix}"] - row[f"excluded_quality_count{suffix}"]
            if (
                usable_count < 0
                or row[f"numeric_count{suffix}"] > usable_count
                or row[f"state_count{suffix}"] > usable_count
            ):
                raise ValueError("hourly_features: branch count exceeds usable event count")
            if row[f"numeric_count{suffix}"] == 0 and any(
                row[f"{name}{suffix}"] is not None for name in _WINDOW_NUMERIC_FIELDS
            ):
                raise ValueError("hourly_features: empty numeric window has non-null statistics")
            if row[f"state_count{suffix}"] == 0 and (
                row[f"state_transitions{suffix}"] is not None
                or row[f"state_distinct_count{suffix}"] is not None
            ):
                raise ValueError("hourly_features: empty state window has non-null statistics")
            coverage = row[f"coverage{suffix}"]
            if coverage is not None and not 0 <= coverage <= 1:
                raise ValueError("hourly_features: coverage outside [0, 1]")


def project_m0_hourly(
    rows: pa.Table | Iterable[Mapping[str, Any]],
    *,
    provenance: Mapping[str, str] | None = None,
) -> pa.Table:
    """Project A2 features onto the frozen, exact M0 hourly interchange envelope."""

    records = rows.to_pylist() if isinstance(rows, pa.Table) else list(rows)
    projected = []
    for row in records:
        source = {**row, **(provenance or {})}
        projected.append(
            {
                "schema_version": CONTRACT_VERSION,
                "run_id": source["run_id"],
                "config_sha256": source["config_sha256"],
                "input_manifest_sha256": source["input_manifest_sha256"],
                "channel_id": source["channel_id"],
                "prediction_time": source["prediction_time"],
                "sensor_type": source["sensor_type"],
                "availability_status": source["availability_status"],
                "availability_reasons": source["availability_reasons"],
                "last_observation_age_seconds": source["last_observation_age_seconds"],
                **{
                    f"event_count_{hours}h": source[f"event_count_{hours}h"]
                    for hours in WINDOW_HOURS
                },
                **{
                    f"alarm_count_{hours}h": source[f"alarm_count_{hours}h"]
                    for hours in WINDOW_HOURS
                },
                "numeric_median_24h": source["numeric_median_24h"],
                "state_transitions_24h": source["state_transitions_24h"],
            }
        )
    result = pa.Table.from_pylist(projected, schema=SCHEMAS["hourly_features"])
    validate_table("hourly_features", result)
    return result
