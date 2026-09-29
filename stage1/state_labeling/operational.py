"""Explicit R1 archive assumption for predicting registered journal messages.

The complete export of seven annual archives is an experimental assumption,
not evidence that any individual channel remained online throughout a gap.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import ntpath
from typing import Iterable

from .rules import TARGET_DEFINITION, classify_message


ARCHIVE_SEGMENTS = tuple(
    (datetime.fromisoformat(start), datetime.fromisoformat(end))
    for start, end in TARGET_DEFINITION["archive_segments"]
)
RECENT_NORMAL = timedelta(hours=TARGET_DEFINITION["recent_normal_hours"])
HORIZON = timedelta(hours=TARGET_DEFINITION["horizon_hours"])


def segment_at(at: datetime) -> int | None:
    """Return an assumed archive segment, excluding every unarchived period."""
    if at.tzinfo is not None:
        raise ValueError("local journal timestamps must be timezone-naive")
    return next(
        (index for index, (start, end) in enumerate(ARCHIVE_SEGMENTS) if start <= at < end),
        None,
    )


def source_is_full_archive(source: str, at: datetime) -> bool:
    """Exclude the isolated example and archives attributed to a wrong year."""
    if segment_at(at) is None:
        return False
    return ntpath.basename(source.replace("/", "\\")) == f"ext-journal-{at.year}.7z"


def future_window_in_archive(at: datetime, horizon: timedelta = HORIZON) -> bool:
    """Both ends of (t,t+H] must stay strictly inside one archive segment."""
    if horizon <= timedelta(0):
        raise ValueError("horizon must be positive")
    segment = segment_at(at)
    return segment is not None and at + horizon < ARCHIVE_SEGMENTS[segment][1]


def recent_explicit_normal(normal_at: datetime | None, at: datetime) -> bool:
    """A prior normal for a new onset; same-second states are not ordered."""
    return (
        normal_at is not None
        and segment_at(normal_at) is not None
        and segment_at(normal_at) == segment_at(at)
        and timedelta(0) < at - normal_at <= RECENT_NORMAL
    )


def recent_normal_for_prediction(normal_at: datetime | None, at: datetime) -> bool:
    """A normal at prediction time is already known; this is not physical uptime."""
    return (
        normal_at is not None
        and segment_at(normal_at) is not None
        and segment_at(normal_at) == segment_at(at)
        and timedelta(0) <= at - normal_at <= RECENT_NORMAL
    )


def same_timestamp_state_conflict(values: Iterable[str | None]) -> bool:
    """Different state texts in one channel-second cannot be causally ordered."""
    return len({value for value in values if value is not None}) > 1


def registered_state_effect(
    sensor_type: str | None, value_state: str | None, alarm: bool | None
) -> str:
    """R1 effect on explicit-normal history, independent of the alarm bit.

    Other operational/technical/unknown texts create uncertainty; the exact
    target and recovery texts alone advance the registered-health state.
    Environmental observations and numeric records are neutral.
    """
    interpretation = classify_message(sensor_type, value_state, alarm)
    if interpretation.status == "unknown_type":
        return "uncertain"
    if value_state is None:
        return "neutral"
    if interpretation.target_message_candidate:
        return "fault"
    if value_state == "Норма":
        return "normal"
    if interpretation.category == "environmental_alarm" or (
        interpretation.reason == "observed_clear_state"
    ):
        return "neutral"
    return "uncertain"
