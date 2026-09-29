"""Column contracts for the local Parquet operational store.

The event schema is intentionally compatible with the *current* Stage 1 / M1
Parquet output (`stage1/ingestion/sql.py`) and also accepts the older
`NormalizedEvent.to_record()` naming through aliases in `storage.py`.

The forecast schema mirrors dictionaries returned by
`ml.forecast.shadow_pilot.run_shadow_batch` plus a few optional input/audit
fields. The episode schema mirrors `stage1.contracts.Episode.to_record()`.
"""

SENSOR_COLUMNS = [
    "channel_id",
    "name",
    "sensor_type",
    "engineering_system_type",
    "engineering_system_tag",
    "sensor_name",
    "object_id",
    "hierarchy_level",
    "parent_object_id",
    "object_kind",
    "object_name",
    "unit",
    "updated_at",
]

# Current M1 clean Parquet columns (plus optional `unit`).
EVENT_COLUMNS = [
    "row_id",
    "source",
    "source_row",
    "event_id_raw",
    "channel_id_raw",
    "date_raw",
    "time_raw",
    "alarm_raw",
    "value_raw",
    "event_id",
    "channel_id",
    "timestamp",
    "alarm",
    "value_numeric",
    "value_state",
    "is_numeric",
    "sensor_type",
    "engineering_system_type",
    "engineering_system_tag",
    "sensor_name",
    "object_id",
    "hierarchy_level",
    "parent_object_id",
    "object_kind",
    "object_name",
    "join_status",
    "quality_flags",
    "unit",
]

FORECAST_COLUMNS = [
    "policy_version",
    "freeze_sha256",
    "channel_id",
    "prediction_time",
    "sensor_type",
    "admission_status",
    "admission_reason",
    "prediction_status",
    "unavailable_reason",
    "rule_score",
    "threshold",
    "threshold_crossed",
    "shadow_warning",
    "warning_reason",
    "score_contributions",
    "delivery_mode",
    "automatic_action_taken",
    # Optional evidence cutoffs / R6 inputs. Current shadow decisions do not
    # necessarily emit these, but storing them when present is useful for audit.
    "history_through",
    "admission_through",
    "registered_fault_text_count_24h",
    "registered_fault_text_count_168h",
    "completed_episode_count_168h",
    "technical_message_count_24h",
    # Optional research-model fields. These are deliberately separate from R6.
    "research_score",
    "research_prediction_status",
    "research_model_version",
    "research_score_kind",
    "round7_score_linear",
    "round7_score_tree",
    "round7_score_specialist",
    "round7_passes_common_gates",
    "round7_passes_standard_gates",
]

EPISODE_COLUMNS = [
    "episode_id",
    "channel_id",
    "sensor_type",
    "sensor_group",
    "anomaly_type",
    "decision",
    "start_at",
    "confirmed_at",
    "ruleset_version",
    "evidence",
    "observation_quality",
    "cause_hypothesis",
    "object_id",
    "end_at",
    "score",
    "origin",
    "metadata",
]

IMPORT_COLUMNS = [
    "batch_id",
    "filename",
    "stored_filename",
    "file_hash",
    "dataset_kind",
    "rows_count",
    "status",
    "error_message",
    "imported_at",
    "finished_at",
]
