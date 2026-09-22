"""Explicit raw/normalized schemas; event IDs are provenance, never model features."""

import pyarrow as pa

RAW_COLUMNS = ("event_id_raw", "channel_id_raw", "date_raw", "time_raw", "alarm_raw", "value_raw")
STAGING_SCHEMA = pa.schema(
    [
        ("row_id", pa.int64()),
        ("source", pa.string()),
        ("source_row", pa.int64()),
        *[(name, pa.string()) for name in RAW_COLUMNS],
        ("event_id", pa.string()),
        ("channel_id", pa.string()),
        ("timestamp", pa.timestamp("us")),
        ("alarm", pa.bool_()),
        ("value_numeric", pa.float64()),
        ("value_state", pa.string()),
        ("is_numeric", pa.bool_()),
        ("quality_flags", pa.list_(pa.string())),
        ("invalid", pa.bool_()),
        ("repeated_header", pa.bool_()),
    ]
)
CHANNEL_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "channel_id",
            "sensor_type",
            "engineering_system_type",
            "engineering_system_tag",
            "sensor_name",
            "object_id",
        )
    ]
)
OBJECT_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "object_id",
            "hierarchy_level",
            "parent_object_id",
            "object_kind",
            "object_name",
        )
    ]
)
