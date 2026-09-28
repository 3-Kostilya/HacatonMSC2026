"""R3 A feature-only handoff; target and temporal split belong to B.

The model allowlist is deliberately narrower than the physical source schemas.
Keys and diagnostic statuses travel alongside features, never inside the list.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import pyarrow as pa

from stage1.features.r2 import CATEGORY_FIELDS, R2_STATE_SCHEMA, validate_r2_table
from stage1.features.schema import (
    A2_SCHEMA,
    WINDOW_HOURS,
    _WINDOW_NUMERIC_FIELDS,
    validate_a2_table,
)


R3_PACK_VERSION = "r3-a-feature-pack-v1"
KEY_FIELDS = ("channel_id", "prediction_time")
A2_FEATURE_FIELDS = (
    "sensor_type",
    "last_observation_age_seconds",
    "baseline_event_count",
    "baseline_numeric_count",
    "baseline_state_count",
    "baseline_numeric_median",
    "baseline_numeric_mad",
    "baseline_dominant_state",
    *(
        f"{name}_{hours}h"
        for hours in WINDOW_HOURS
        for name in (
            "event_count",
            "alarm_count",
            "excluded_quality_count",
            "numeric_count",
            "state_count",
            *_WINDOW_NUMERIC_FIELDS,
            "state_transitions",
            "state_distinct_count",
            "maximum_gap_seconds",
            "coverage",
        )
    ),
)
R2_FEATURE_FIELDS = (
    *(f"{name}_{hours}h" for hours in WINDOW_HOURS for name in CATEGORY_FIELDS),
    "last_completed_episode_end_age_seconds",
    "completed_episode_count_168h",
    "completed_episode_mean_duration_seconds_168h",
)
MODEL_FEATURE_ALLOWLIST = (*A2_FEATURE_FIELDS, *R2_FEATURE_FIELDS)
A2_DIAGNOSTIC_FIELDS = (
    "availability_status",
    "availability_reasons",
    "baseline_fit_end_at",
    "baseline_status",
    "baseline_reasons",
    "peer_status",
    "peer_reasons",
    *(name for hours in WINDOW_HOURS for name in (f"window_status_{hours}h", f"window_reasons_{hours}h")),
)
R2_DIAGNOSTIC_FIELDS = (
    "episode_history_status",
    "numeric_data_status",
    "numeric_data_reasons",
    "discrete_data_status",
    "discrete_data_reasons",
)


def _fields(schema: pa.Schema, names: tuple[str, ...]) -> list[pa.Field]:
    return [schema.field(name) for name in names]


FEATURE_PACK_SCHEMA = pa.schema(
    [*_fields(A2_SCHEMA, KEY_FIELDS), *_fields(A2_SCHEMA, A2_FEATURE_FIELDS),
     *_fields(R2_STATE_SCHEMA, R2_FEATURE_FIELDS)]
)
ROW_STATUS_SCHEMA = pa.schema(
    [*_fields(A2_SCHEMA, KEY_FIELDS), *_fields(A2_SCHEMA, A2_DIAGNOSTIC_FIELDS),
     *_fields(R2_STATE_SCHEMA, R2_DIAGNOSTIC_FIELDS)]
)


def build_pack_tables(a2: pa.Table, r2: pa.Table) -> tuple[pa.Table, pa.Table]:
    """Join an exact A2/R2 key set without admitting target-derived columns."""

    validate_a2_table(a2)
    validate_r2_table(r2)
    a2_rows = {(row["channel_id"], row["prediction_time"]): row for row in a2.to_pylist()}
    r2_rows = {(row["channel_id"], row["prediction_time"]): row for row in r2.to_pylist()}
    if not a2_rows or a2_rows.keys() != r2_rows.keys():
        raise ValueError("R3 requires nonempty, exactly aligned A2 and R2 keys")
    features: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for key in sorted(a2_rows):
        a, r = a2_rows[key], r2_rows[key]
        if r["source_a2_run_id"] != a["run_id"]:
            raise ValueError(f"R2/A2 run mismatch for {key!r}")
        features.append(
            {**{name: a[name] for name in (*KEY_FIELDS, *A2_FEATURE_FIELDS)},
             **{name: r[name] for name in R2_FEATURE_FIELDS}}
        )
        statuses.append(
            {**{name: a[name] for name in (*KEY_FIELDS, *A2_DIAGNOSTIC_FIELDS)},
             **{name: r[name] for name in R2_DIAGNOSTIC_FIELDS}}
        )
    return (
        pa.Table.from_pylist(features, schema=FEATURE_PACK_SCHEMA),
        pa.Table.from_pylist(statuses, schema=ROW_STATUS_SCHEMA),
    )


def summarize_partitions(features: pa.Table, statuses: pa.Table) -> dict[str, Any]:
    """Small audit for one time chunk; no target labels or split assignment."""

    times = features.column("prediction_time").to_pylist()
    if not times or any(not isinstance(t, datetime) for t in times):
        raise ValueError("R3 chunk must have prediction timestamps")
    return {
        "rows": features.num_rows,
        "channels": len(set(features.column("channel_id").to_pylist())),
        "min_prediction_time": min(times).isoformat(),
        "max_prediction_time": max(times).isoformat(),
        "availability_status": dict(sorted(Counter(statuses.column("availability_status").to_pylist()).items())),
        "numeric_data_status": dict(sorted(Counter(statuses.column("numeric_data_status").to_pylist()).items())),
        "discrete_data_status": dict(sorted(Counter(statuses.column("discrete_data_status").to_pylist()).items())),
    }
