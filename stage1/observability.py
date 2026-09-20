"""Causal observability audit for channel histories.

The module deliberately does not resample or forward-fill events.  It describes
only timestamps that were actually observed at or before the decision time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import math
from typing import Iterable, Protocol


DEFAULT_WINDOWS = (
    ("1h", timedelta(hours=1)),
    ("6h", timedelta(hours=6)),
    ("24h", timedelta(hours=24)),
    ("7d", timedelta(days=7)),
)


class AuditStatus(StrEnum):
    """Whether an interval may be used by downstream feature calculations."""

    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNKNOWN = "unknown"


class EventLike(Protocol):
    channel_id: str
    timestamp: datetime
    quality_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservabilityPolicy:
    """Explicit assumptions required to interpret a channel's event cadence."""

    expected_cadence: timedelta | None
    minimum_history: timedelta = timedelta(days=7)
    minimum_history_events: int = 2
    minimum_window_events: int = 2
    minimum_coverage: float = 0.5
    maximum_gap_factor: float = 3.0
    excluded_quality_flags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.expected_cadence is not None and self.expected_cadence <= timedelta(0):
            raise ValueError("expected_cadence must be positive when present")
        if self.minimum_history < timedelta(0):
            raise ValueError("minimum_history cannot be negative")
        if self.minimum_history_events < 1 or self.minimum_window_events < 1:
            raise ValueError("minimum event counts must be positive")
        if not 0 <= self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be between 0 and 1")
        if not math.isfinite(self.maximum_gap_factor) or self.maximum_gap_factor < 1:
            raise ValueError("maximum_gap_factor must be finite and at least 1")


@dataclass(frozen=True, slots=True)
class ObservationInterval:
    """A run bounded by real observations, never by imputed endpoints."""

    start_at: datetime
    end_at: datetime
    event_count: int


@dataclass(frozen=True, slots=True)
class WindowAudit:
    name: str
    start_at: datetime
    end_at: datetime
    status: AuditStatus
    reasons: tuple[str, ...]
    flags: tuple[str, ...]
    event_count: int
    excluded_event_count: int
    first_observed_at: datetime | None
    last_observed_at: datetime | None
    age_seconds: float | None
    coverage: float | None
    maximum_gap_seconds: float | None
    observed_intervals: tuple[ObservationInterval, ...]


@dataclass(frozen=True, slots=True)
class ChannelAudit:
    channel_id: str
    decision_at: datetime
    status: AuditStatus
    reasons: tuple[str, ...]
    history_event_count: int
    history_start_at: datetime | None
    history_span_seconds: float | None
    history_sufficient: bool
    windows: tuple[WindowAudit, ...]

    def window(self, name: str) -> WindowAudit:
        """Return a named audit window."""

        for item in self.windows:
            if item.name == name:
                return item
        raise KeyError(name)


def _validate_naive(name: str, timestamp: datetime) -> None:
    if not isinstance(timestamp, datetime):
        raise ValueError(f"{name} must be a datetime")
    if timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
        raise ValueError(f"{name} must use local naive time")


def _deduplicate(timestamps: Iterable[datetime]) -> tuple[datetime, ...]:
    return tuple(sorted(set(timestamps)))


def _observed_runs(
    timestamps: tuple[datetime, ...], maximum_gap: timedelta
) -> tuple[ObservationInterval, ...]:
    if not timestamps:
        return ()
    runs: list[ObservationInterval] = []
    start = previous = timestamps[0]
    count = 1
    for timestamp in timestamps[1:]:
        if timestamp - previous > maximum_gap:
            runs.append(ObservationInterval(start, previous, count))
            start = timestamp
            count = 1
        else:
            count += 1
        previous = timestamp
    runs.append(ObservationInterval(start, previous, count))
    return tuple(runs)


def _audit_window(
    name: str,
    duration: timedelta,
    decision_at: datetime,
    usable_timestamps: tuple[datetime, ...],
    excluded_timestamps: tuple[datetime, ...],
    policy: ObservabilityPolicy,
    exclusion_reasons: tuple[str, ...],
) -> WindowAudit:
    start_at = decision_at - duration
    timestamps = tuple(ts for ts in usable_timestamps if start_at <= ts <= decision_at)
    excluded_count = sum(start_at <= ts <= decision_at for ts in excluded_timestamps)
    first_at = timestamps[0] if timestamps else None
    last_at = timestamps[-1] if timestamps else None
    age = (decision_at - last_at).total_seconds() if last_at is not None else None

    if exclusion_reasons:
        return WindowAudit(
            name,
            start_at,
            decision_at,
            AuditStatus.EXCLUDE,
            exclusion_reasons,
            ("explicit_exclusion",),
            len(timestamps),
            excluded_count,
            first_at,
            last_at,
            age,
            None,
            None,
            tuple(ObservationInterval(ts, ts, 1) for ts in timestamps),
        )

    if policy.expected_cadence is None:
        status = AuditStatus.UNKNOWN
        reasons = ("cadence_unknown",)
        if not timestamps:
            reasons += ("no_usable_observations",)
        return WindowAudit(
            name,
            start_at,
            decision_at,
            status,
            reasons,
            ("coverage_not_computable", "gap_not_computable"),
            len(timestamps),
            excluded_count,
            first_at,
            last_at,
            age,
            None,
            None,
            tuple(ObservationInterval(ts, ts, 1) for ts in timestamps),
        )

    cadence = policy.expected_cadence
    maximum_gap = cadence * policy.maximum_gap_factor
    expected_count = math.floor(duration / cadence) + 1
    coverage = min(1.0, len(timestamps) / expected_count)
    gaps: list[float] = []
    if timestamps:
        gaps.append((timestamps[0] - start_at).total_seconds())
        gaps.extend(
            (right - left).total_seconds() for left, right in zip(timestamps, timestamps[1:])
        )
        gaps.append((decision_at - timestamps[-1]).total_seconds())
    maximum_gap_seconds = max(gaps) if gaps else duration.total_seconds()
    flags: list[str] = []
    reasons: list[str] = []
    if not timestamps:
        flags.append("no_observations")
        reasons.append("no_usable_observations")
    if len(timestamps) < policy.minimum_window_events:
        flags.append("too_few_observations")
        reasons.append("insufficient_window_events")
    if coverage < policy.minimum_coverage:
        flags.append("low_coverage")
        reasons.append("coverage_below_threshold")
    if maximum_gap_seconds > maximum_gap.total_seconds():
        flags.append("large_gap")
        reasons.append("gap_above_threshold")
    if last_at is not None and decision_at - last_at > maximum_gap:
        flags.append("stale_last_observation")
    if excluded_count:
        flags.append("quality_exclusions_present")
    if not timestamps and excluded_count:
        status = AuditStatus.EXCLUDE
        reasons = ["all_interval_observations_excluded"]
    elif reasons:
        status = AuditStatus.UNKNOWN
    else:
        status = AuditStatus.INCLUDE
    return WindowAudit(
        name,
        start_at,
        decision_at,
        status,
        tuple(reasons),
        tuple(flags),
        len(timestamps),
        excluded_count,
        first_at,
        last_at,
        age,
        coverage,
        maximum_gap_seconds,
        _observed_runs(timestamps, maximum_gap),
    )


def audit_channel(
    channel_id: str,
    events: Iterable[EventLike],
    decision_at: datetime,
    policy: ObservabilityPolicy,
    *,
    windows: tuple[tuple[str, timedelta], ...] = DEFAULT_WINDOWS,
    exclusion_reasons: tuple[str, ...] = (),
) -> ChannelAudit:
    """Audit one channel using only events available at ``decision_at``.

    ``exclusion_reasons`` is for explicit external exclusions such as an invalid
    channel mapping.  Missing cadence or history is uncertainty, not exclusion.
    """

    if not isinstance(channel_id, str) or not channel_id.strip():
        raise ValueError("channel_id must be a non-empty string")
    _validate_naive("decision_at", decision_at)
    if len({name for name, _ in windows}) != len(windows):
        raise ValueError("window names must be unique")
    if any(duration <= timedelta(0) for _, duration in windows):
        raise ValueError("window durations must be positive")

    usable: list[datetime] = []
    excluded: list[datetime] = []
    for event in events:
        if event.channel_id != channel_id:
            continue
        _validate_naive("event.timestamp", event.timestamp)
        if event.timestamp > decision_at:
            continue
        if policy.excluded_quality_flags.intersection(event.quality_flags):
            excluded.append(event.timestamp)
        else:
            usable.append(event.timestamp)
    usable_timestamps = _deduplicate(usable)
    excluded_timestamps = _deduplicate(excluded)

    history_start = usable_timestamps[0] if usable_timestamps else None
    history_span = (
        (decision_at - history_start).total_seconds() if history_start is not None else None
    )
    history_sufficient = bool(
        history_start is not None
        and decision_at - history_start >= policy.minimum_history
        and len(usable_timestamps) >= policy.minimum_history_events
    )
    window_audits = tuple(
        _audit_window(
            name,
            duration,
            decision_at,
            usable_timestamps,
            excluded_timestamps,
            policy,
            exclusion_reasons,
        )
        for name, duration in windows
    )

    if exclusion_reasons:
        status = AuditStatus.EXCLUDE
        reasons = tuple(exclusion_reasons)
    elif not usable_timestamps and excluded_timestamps:
        status = AuditStatus.EXCLUDE
        reasons = ("all_causal_observations_excluded",)
    else:
        reasons_list: list[str] = []
        if not history_sufficient:
            reasons_list.append("insufficient_history")
        if policy.expected_cadence is None:
            reasons_list.append("cadence_unknown")
        if any(item.status is not AuditStatus.INCLUDE for item in window_audits):
            reasons_list.append("one_or_more_windows_unusable")
        reasons = tuple(reasons_list)
        status = AuditStatus.UNKNOWN if reasons else AuditStatus.INCLUDE

    return ChannelAudit(
        channel_id,
        decision_at,
        status,
        reasons,
        len(usable_timestamps),
        history_start,
        history_span,
        history_sufficient,
        window_audits,
    )
