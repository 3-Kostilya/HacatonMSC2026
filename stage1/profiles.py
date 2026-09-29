"""Frozen, causal per-channel behavior profiles for baseline comparison."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import statistics
from typing import Iterable

from stage1.contracts import NormalizedEvent


@dataclass(frozen=True, slots=True)
class BehaviorProfile:
    channel_id: str
    sensor_type: str
    trained_until: datetime
    first_observation: datetime
    last_observation: datetime
    event_count: int
    numeric_count: int
    numeric_median: float | None
    numeric_mad: float | None
    known_states: tuple[str, ...]
    dominant_state: str | None
    quality_flags: tuple[str, ...]


def _median_absolute_deviation(values: list[float], center: float) -> float:
    return float(statistics.median(abs(value - center) for value in values))


def fit_behavior_profiles(
    events: Iterable[NormalizedEvent],
    trained_until: datetime,
    min_events: int = 10,
    min_numeric: int = 5,
) -> dict[str, BehaviorProfile]:
    """Fit profiles using events strictly before an explicit temporal boundary."""
    if trained_until.tzinfo is not None and trained_until.utcoffset() is not None:
        raise ValueError("trained_until must use local naive time")
    grouped: dict[str, list[NormalizedEvent]] = defaultdict(list)
    for event in events:
        if event.timestamp < trained_until:
            grouped[event.channel_id].append(event)

    profiles = {}
    for channel_id, channel_events in grouped.items():
        channel_events.sort(key=lambda event: event.timestamp)
        sensor_types = {event.sensor_type for event in channel_events}
        if len(sensor_types) != 1:
            raise ValueError(f"channel {channel_id} has multiple sensor types: {sensor_types}")
        numeric = [
            event.numeric_value for event in channel_events if event.numeric_value is not None
        ]
        states = Counter(event.raw_value for event in channel_events if event.numeric_value is None)
        median = float(statistics.median(numeric)) if numeric else None
        mad = _median_absolute_deviation(numeric, median) if median is not None else None
        flags = set(flag for event in channel_events for flag in event.quality_flags)
        if len(channel_events) < min_events:
            flags.add("insufficient_history")
        if numeric and len(numeric) < min_numeric:
            flags.add("insufficient_numeric_history")
        if numeric and mad == 0:
            flags.add("zero_numeric_mad")
        profiles[channel_id] = BehaviorProfile(
            channel_id=channel_id,
            sensor_type=next(iter(sensor_types)),
            trained_until=trained_until,
            first_observation=channel_events[0].timestamp,
            last_observation=channel_events[-1].timestamp,
            event_count=len(channel_events),
            numeric_count=len(numeric),
            numeric_median=median,
            numeric_mad=mad,
            known_states=tuple(sorted(states)),
            dominant_state=states.most_common(1)[0][0] if states else None,
            quality_flags=tuple(sorted(flags)),
        )
    return profiles


def numeric_deviation_mad(event: NormalizedEvent, profile: BehaviorProfile) -> float | None:
    """Return signed robust deviation, or unknown for a degenerate/inapplicable profile."""
    if event.channel_id != profile.channel_id or event.sensor_type != profile.sensor_type:
        raise ValueError("event and profile refer to different channels or types")
    if event.timestamp < profile.trained_until:
        raise ValueError("scored event precedes the frozen profile boundary")
    if event.numeric_value is None or profile.numeric_median is None:
        return None
    if profile.numeric_mad is None or profile.numeric_mad == 0:
        return None
    return (event.numeric_value - profile.numeric_median) / profile.numeric_mad


def state_is_known(event: NormalizedEvent, profile: BehaviorProfile) -> bool | None:
    if event.channel_id != profile.channel_id or event.sensor_type != profile.sensor_type:
        raise ValueError("event and profile refer to different channels or types")
    if event.timestamp < profile.trained_until:
        raise ValueError("scored event precedes the frozen profile boundary")
    if event.numeric_value is not None or not profile.known_states:
        return None
    return event.raw_value in profile.known_states
