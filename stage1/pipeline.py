"""Integration layer from normalized events to auditable baseline outcomes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Iterable, Mapping

from stage1.contracts import Episode, NormalizedEvent
from stage1.detectors import (
    DiscreteDetectorConfig,
    NumericDetectorConfig,
    detect_discrete_patterns,
    detect_numeric_level_shifts,
)
from stage1.episodes import consolidate_episodes
from stage1.observability import ChannelAudit, ObservabilityPolicy, audit_channel
from stage1.registry import TypePolicy, TypeRegistry, load_registry


@dataclass(frozen=True, slots=True)
class ChannelOutcome:
    channel_id: str
    sensor_type: str
    processing_mode: str
    detector_result: Episode
    detector_results: tuple[Episode, ...]
    observability: ChannelAudit
    context_status: str
    context_reason: str


def _primary_mode(policy: TypePolicy) -> str:
    if policy.modes["numeric"] == "primary":
        return "numeric"
    if policy.modes["discrete"] in {"primary", "conditional"}:
        return "discrete"
    raise ValueError(f"{policy.sensor_type}: no supported primary baseline mode")


def evaluate_channel(
    events: Iterable[NormalizedEvent],
    *,
    registry: TypeRegistry | None = None,
    expected_cadence: timedelta | None = None,
    numeric_config: NumericDetectorConfig | None = None,
    discrete_config: DiscreteDetectorConfig | None = None,
) -> ChannelOutcome:
    ordered = sorted(events, key=lambda event: event.timestamp)
    if not ordered:
        raise ValueError("events must not be empty")
    if len({event.channel_id for event in ordered}) != 1:
        raise ValueError("events must belong to one channel")
    if len({event.sensor_type for event in ordered}) != 1:
        raise ValueError("events must belong to one sensor type")
    type_registry = registry or load_registry()
    policy = type_registry.get(ordered[0].sensor_type)
    mode = _primary_mode(policy)
    group = policy.family
    if mode == "numeric":
        config = (
            replace(numeric_config, sensor_group=group)
            if numeric_config
            else NumericDetectorConfig(sensor_group=group)
        )
        detections = detect_numeric_level_shifts(ordered, config)
    else:
        config = (
            replace(discrete_config, sensor_group=group)
            if discrete_config
            else DiscreteDetectorConfig(sensor_group=group)
        )
        detections = detect_discrete_patterns(ordered, config)
    consolidated = tuple(consolidate_episodes(detections))
    result = next(
        (episode for episode in consolidated if episode.decision.value == "candidate"),
        consolidated[0],
    )

    observability = audit_channel(
        ordered[0].channel_id,
        ordered,
        ordered[-1].timestamp,
        ObservabilityPolicy(
            expected_cadence=expected_cadence,
            excluded_quality_flags=frozenset(
                {"channel_time_conflict", "invalid_timestamp", "nonfinite_numeric"}
            ),
        ),
    )
    has_object_link = bool(ordered[0].object_id) and all(
        event.object_id == ordered[0].object_id for event in ordered
    )
    context_status = "available" if has_object_link else "unknown"
    context_reason = "confirmed_object_id" if has_object_link else "unconfirmed_context_link"
    return ChannelOutcome(
        channel_id=ordered[0].channel_id,
        sensor_type=ordered[0].sensor_type,
        processing_mode=mode,
        detector_result=result,
        detector_results=consolidated,
        observability=observability,
        context_status=context_status,
        context_reason=context_reason,
    )


def evaluate_catalog(
    events: Iterable[NormalizedEvent],
    *,
    registry: TypeRegistry | None = None,
    cadence_by_type: Mapping[str, timedelta] | None = None,
) -> list[ChannelOutcome]:
    type_registry = registry or load_registry()
    grouped: dict[tuple[str, str], list[NormalizedEvent]] = {}
    for event in events:
        grouped.setdefault((event.sensor_type, event.channel_id), []).append(event)
    cadence = cadence_by_type or {}
    return [
        evaluate_channel(
            channel_events,
            registry=type_registry,
            expected_cadence=cadence.get(sensor_type),
        )
        for (sensor_type, _), channel_events in sorted(grouped.items())
    ]
