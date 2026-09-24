"""Causal R2 state history and operation-specific data availability.

This supplements the frozen A2 hourly table. It never creates a forecasting
label or treats a message count as a completed episode.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Any, Iterable, Mapping

import pyarrow as pa

from stage1.features.hourly import FeatureEvent, HourlyConfig
from stage1.features.schema import WINDOW_HOURS
from stage1.state_labeling.rules import RULESET_VERSION, classify_message


R2_VERSION = "r2-a-state-history-v1"
CATEGORY_FIELDS = (
    "technical_message_count",
    "registered_fault_text_count",
    "normal_message_count",
    "environmental_alarm_count",
    "unknown_state_count",
)
R2_STATE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("source_a2_run_id", pa.string(), nullable=False),
        pa.field("source_a2_manifest_sha256", pa.string(), nullable=False),
        pa.field("ruleset_version", pa.string(), nullable=False),
        pa.field("channel_id", pa.string(), nullable=False),
        pa.field("prediction_time", pa.timestamp("us"), nullable=False),
        *[
            pa.field(f"{name}_{hours}h", pa.int64(), nullable=False)
            for hours in WINDOW_HOURS
            for name in CATEGORY_FIELDS
        ],
        pa.field("last_completed_episode_end_age_seconds", pa.float64()),
        pa.field("completed_episode_count_168h", pa.int64()),
        pa.field("completed_episode_mean_duration_seconds_168h", pa.float64()),
        pa.field("episode_history_status", pa.string(), nullable=False),
        pa.field("numeric_data_status", pa.string(), nullable=False),
        pa.field("numeric_data_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("discrete_data_status", pa.string(), nullable=False),
        pa.field("discrete_data_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("model_admission_status", pa.string(), nullable=False),
        pa.field("model_admission_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("future_label_status", pa.string(), nullable=False),
        pa.field("future_label_reasons", pa.list_(pa.string()), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class CompletedEpisode:
    """Minimal future B-catalog adapter; only a known end enters past features."""

    channel_id: str
    start_at: datetime
    end_at: datetime

    def __post_init__(self) -> None:
        if (
            not self.channel_id
            or self.start_at.tzinfo is not None
            or self.end_at.tzinfo is not None
        ):
            raise ValueError("episode needs a channel and local naive timestamps")
        if self.end_at <= self.start_at:
            raise ValueError("completed episode must end after it starts")


def _branch_status(row: Mapping[str, Any], *, discrete: bool) -> tuple[str, list[str]]:
    if row["availability_status"] == "excluded":
        return "excluded", ["all_causal_observations_excluded"]
    reasons = []
    if row["sensor_type"] is None:
        reasons.append("sensor_type_unknown_or_conflicting")
    if row["baseline_status"] != "eligible":
        reasons.append("baseline_unusable")
    if "insufficient_history" in row["availability_reasons"]:
        reasons.append("insufficient_history")
    if row["excluded_quality_count_24h"]:
        reasons.append("quality_exclusions_24h")
    if discrete:
        if row["baseline_state_count"] == 0 or row["state_count_24h"] == 0:
            reasons.append("state_history_missing")
        if row["state_transitions_24h"] is None:
            reasons.append("state_transitions_unavailable")
        if "same_time_state_ambiguity" in row["window_reasons_24h"]:
            reasons.append("same_time_state_ambiguity")
    else:
        if row["baseline_numeric_count"] == 0 or row["numeric_count_24h"] == 0:
            reasons.append("numeric_history_missing")
        if (
            row["baseline_numeric_median"] is None
            or row["baseline_numeric_mad"] is None
            or row["numeric_median_24h"] is None
        ):
            reasons.append("numeric_summary_unavailable")
    return ("unknown", reasons) if reasons else ("eligible", [])


def _semantic_prefix(events: list[FeatureEvent]) -> tuple[list[datetime], dict[str, list[int]]]:
    excluded_flags = HourlyConfig().excluded_quality_flags
    ordered = sorted(
        (
            event
            for event in events
            if event.value_state is not None
            and not excluded_flags.intersection(event.quality_flags)
        ),
        key=lambda event: event.timestamp,
    )
    times = [event.timestamp for event in ordered]
    prefixes = {name: [0] for name in CATEGORY_FIELDS}
    for event in ordered:
        meaning = classify_message(event.sensor_type, event.value_state, event.alarm)
        increments = {
            "technical_message_count": meaning.category == "technical_fault",
            "registered_fault_text_count": meaning.target_message_candidate,
            "normal_message_count": meaning.category == "normal",
            "environmental_alarm_count": meaning.category == "environmental_alarm",
            "unknown_state_count": meaning.category == "unknown",
        }
        for name, prefix in prefixes.items():
            prefix.append(prefix[-1] + int(increments[name]))
    return times, prefixes


def _episode_prefix(
    episodes: list[CompletedEpisode],
) -> tuple[list[datetime], list[float]]:
    ordered = sorted(episodes, key=lambda episode: episode.end_at)
    ends = [episode.end_at for episode in ordered]
    duration_prefix = [0.0]
    for episode in ordered:
        duration_prefix.append(
            duration_prefix[-1] + (episode.end_at - episode.start_at).total_seconds()
        )
    return ends, duration_prefix


def build_state_history_rows(
    a2_rows: Iterable[Mapping[str, Any]],
    events: Iterable[FeatureEvent],
    *,
    source_a2_manifest_sha256: str,
    completed_episodes: Iterable[CompletedEpisode] | None = None,
) -> pa.Table:
    """Create one R2 supplement per A2 row using only event/episode history <= t."""

    by_channel_events: dict[str, list[FeatureEvent]] = defaultdict(list)
    for event in events:
        by_channel_events[event.channel_id].append(event)
    by_channel_episodes: dict[str, list[CompletedEpisode]] = defaultdict(list)
    if completed_episodes is not None:
        for episode in completed_episodes:
            by_channel_episodes[episode.channel_id].append(episode)
    event_prefixes = {
        channel_id: _semantic_prefix(channel_events)
        for channel_id, channel_events in by_channel_events.items()
    }
    episode_prefixes = {
        channel_id: _episode_prefix(channel_episodes)
        for channel_id, channel_episodes in by_channel_episodes.items()
    }
    output = []
    for row in a2_rows:
        channel_id = row["channel_id"]
        t = row["prediction_time"]
        times, prefixes = event_prefixes.get(
            channel_id, ([], {name: [0] for name in CATEGORY_FIELDS})
        )
        upper = bisect_right(times, t)
        result: dict[str, Any] = {
            "schema_version": R2_VERSION,
            "source_a2_run_id": row["run_id"],
            "source_a2_manifest_sha256": source_a2_manifest_sha256,
            "ruleset_version": RULESET_VERSION,
            "channel_id": channel_id,
            "prediction_time": t,
        }
        for hours in WINDOW_HOURS:
            lower = bisect_right(times, t - timedelta(hours=hours))
            for name, prefix in prefixes.items():
                result[f"{name}_{hours}h"] = prefix[upper] - prefix[lower]
        if completed_episodes is None:
            result.update(
                last_completed_episode_end_age_seconds=None,
                completed_episode_count_168h=None,
                completed_episode_mean_duration_seconds_168h=None,
                episode_history_status="catalog_unavailable",
            )
        else:
            ends, durations = episode_prefixes.get(channel_id, ([], [0.0]))
            closed = bisect_right(ends, t)
            recent = bisect_right(ends, t - timedelta(hours=168))
            count = closed - recent
            result.update(
                last_completed_episode_end_age_seconds=(
                    (t - ends[closed - 1]).total_seconds() if closed else None
                ),
                completed_episode_count_168h=count,
                completed_episode_mean_duration_seconds_168h=(
                    (durations[closed] - durations[recent]) / count if count else None
                ),
                episode_history_status="completed_only",
            )
        numeric_status, numeric_reasons = _branch_status(row, discrete=False)
        discrete_status, discrete_reasons = _branch_status(row, discrete=True)
        result.update(
            numeric_data_status=numeric_status,
            numeric_data_reasons=numeric_reasons,
            discrete_data_status=discrete_status,
            discrete_data_reasons=discrete_reasons,
            model_admission_status="unknown",
            model_admission_reasons=["model_contract_pending"],
            future_label_status="unknown",
            future_label_reasons=["computed_by_b_after_episode_and_coverage_contract"],
        )
        output.append(result)
    table = pa.Table.from_pylist(output, schema=R2_STATE_SCHEMA)
    validate_r2_table(table)
    return table


def validate_r2_table(table: pa.Table) -> None:
    if not table.schema.equals(R2_STATE_SCHEMA, check_metadata=False):
        raise ValueError("R2 state-history schema differs from declared version")
    seen = set()
    for row in table.to_pylist():
        key = row["channel_id"], row["prediction_time"]
        if key in seen:
            raise ValueError("duplicate channel/prediction_time in R2 state history")
        seen.add(key)
        if row["schema_version"] != R2_VERSION or row["ruleset_version"] != RULESET_VERSION:
            raise ValueError("R2 state history has unexpected version")
        if row["model_admission_status"] != "unknown" or row["future_label_status"] != "unknown":
            raise ValueError("R2 A may not publish model or future-label eligibility")
        for name in CATEGORY_FIELDS:
            values = [row[f"{name}_{hours}h"] for hours in WINDOW_HOURS]
            if any(value is None or value < 0 for value in values) or values != sorted(values):
                raise ValueError("R2 causal state counts must be nonnegative and nested")
        for field in (
            "last_completed_episode_end_age_seconds",
            "completed_episode_mean_duration_seconds_168h",
        ):
            value = row[field]
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("R2 completed-episode feature must be finite and nonnegative")
