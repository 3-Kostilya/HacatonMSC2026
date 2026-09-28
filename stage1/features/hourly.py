"""Causal per-channel hourly features; silence is never imputed as a reading."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
import statistics
from typing import Any

from stage1.contracts import NormalizedEvent
from stage1.features.schema import WINDOW_HOURS
from stage1.value_quality import assess_value


@dataclass(frozen=True, slots=True)
class FeatureEvent:
    """Only the observed fields needed for A2; source IDs are not features."""

    channel_id: str
    timestamp: datetime
    alarm: bool
    value_numeric: float | None = None
    value_state: str | None = None
    sensor_type: str | None = None
    object_id: str | None = None
    join_status: str | None = None
    quality_flags: tuple[str, ...] = ()
    qa_value_category: str | None = None
    numeric_measurement_usable: bool = True
    state_transition_usable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.channel_id, str) or not self.channel_id.strip():
            raise ValueError("channel_id must be nonempty")
        _require_local_time("timestamp", self.timestamp)
        if not isinstance(self.alarm, bool):
            raise ValueError("alarm must be boolean")
        if self.value_numeric is not None and not math.isfinite(self.value_numeric):
            raise ValueError("value_numeric must be finite")
        if self.value_numeric is not None and self.value_state is not None:
            raise ValueError("an event cannot have both numeric and state values")
        if not isinstance(self.quality_flags, tuple):
            object.__setattr__(self, "quality_flags", tuple(self.quality_flags))

    @classmethod
    def from_normalized(
        cls,
        event: NormalizedEvent,
        *,
        apply_qa_value_policy: bool = False,
    ) -> FeatureEvent:
        assessment = (
            assess_value(event.sensor_type, event.raw_value, event.numeric_value)
            if apply_qa_value_policy
            else None
        )
        return cls(
            channel_id=event.channel_id,
            timestamp=event.timestamp,
            alarm=event.alarm,
            value_numeric=event.numeric_value,
            value_state=event.raw_value if event.numeric_value is None else None,
            sensor_type=event.sensor_type,
            object_id=event.object_id,
            quality_flags=tuple(event.quality_flags),
            qa_value_category=assessment.category if assessment else None,
            numeric_measurement_usable=(
                assessment.numeric_measurement_usable if assessment else True
            ),
            state_transition_usable=(assessment.state_transition_usable if assessment else True),
        )

    @classmethod
    def from_clean_record(
        cls,
        record: Mapping[str, Any],
        *,
        apply_qa_value_policy: bool = False,
    ) -> FeatureEvent:
        numeric = record.get("value_numeric")
        state = record.get("value_state")
        if numeric is None and state is None:
            state = record.get("value_raw")
        assessment = (
            assess_value(
                record.get("sensor_type"), str(record.get("value_raw") or state or ""), numeric
            )
            if apply_qa_value_policy
            else None
        )
        return cls(
            channel_id=record["channel_id"],
            timestamp=record["timestamp"],
            alarm=record["alarm"],
            value_numeric=numeric,
            value_state=state,
            sensor_type=record.get("sensor_type"),
            object_id=record.get("object_id"),
            join_status=record.get("join_status"),
            quality_flags=tuple(record.get("quality_flags") or ()),
            qa_value_category=assessment.category if assessment else None,
            numeric_measurement_usable=(
                assessment.numeric_measurement_usable if assessment else True
            ),
            state_transition_usable=(assessment.state_transition_usable if assessment else True),
        )


@dataclass(frozen=True, slots=True)
class HourlyConfig:
    """Explicit assumptions for availability; unknown cadence is not dropout."""

    expected_cadence: timedelta | None = None
    minimum_history: timedelta = timedelta(days=7)
    minimum_history_events: int = 2
    minimum_window_events: int = 2
    minimum_coverage: float = 0.5
    maximum_gap_factor: float = 3.0
    baseline_lookback: timedelta = timedelta(days=28)
    baseline_embargo: timedelta = timedelta(hours=24)
    minimum_baseline_events: int = 10
    excluded_quality_flags: frozenset[str] = frozenset(
        {"channel_time_conflict", "nonfinite_numeric", "invalid_timestamp"}
    )

    def __post_init__(self) -> None:
        if self.expected_cadence is not None and self.expected_cadence <= timedelta(0):
            raise ValueError("expected_cadence must be positive")
        if self.minimum_history < timedelta(0):
            raise ValueError("minimum_history cannot be negative")
        if self.minimum_history_events < 1 or self.minimum_window_events < 1:
            raise ValueError("minimum event counts must be positive")
        if self.baseline_lookback <= timedelta(0) or self.baseline_embargo < timedelta(0):
            raise ValueError("invalid baseline lookback or embargo")
        if self.minimum_baseline_events < 1:
            raise ValueError("minimum_baseline_events must be positive")
        if not 0 <= self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be within [0, 1]")
        if not math.isfinite(self.maximum_gap_factor) or self.maximum_gap_factor < 1:
            raise ValueError("maximum_gap_factor must be finite and at least one")


def _require_local_time(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise ValueError(f"{name} must use the journal's local naive time")


def _quantile(values: list[float], fraction: float) -> float:
    """Linear-interpolated empirical quantile, defined even for one reading."""

    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def _numeric_features(events: list[FeatureEvent]) -> dict[str, float | None]:
    names = (
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
    numeric = [
        (item.timestamp, item.value_numeric)
        for item in events
        if item.value_numeric is not None and item.numeric_measurement_usable
    ]
    if not numeric:
        return dict.fromkeys(names)
    values = [value for _, value in numeric]
    median = float(statistics.median(values))
    by_time: dict[datetime, list[float]] = defaultdict(list)
    for timestamp, value in numeric:
        by_time[timestamp].append(value)
    ordered_times = sorted(by_time)
    chronological = [float(statistics.median(by_time[ts])) for ts in ordered_times]
    delta: float | None = None
    slope: float | None = None
    if len(ordered_times) >= 2:
        delta = chronological[-1] - chronological[0]
        x = [(ts - ordered_times[0]).total_seconds() / 3600 for ts in ordered_times]
        center_x = statistics.fmean(x)
        center_y = statistics.fmean(chronological)
        denominator = sum((value - center_x) ** 2 for value in x)
        if denominator > 0:
            slope = (
                sum((xx - center_x) * (yy - center_y) for xx, yy in zip(x, chronological))
                / denominator
            )
    return {
        "numeric_median": median,
        "numeric_mad": float(statistics.median(abs(value - median) for value in values)),
        "numeric_iqr": _quantile(values, 0.75) - _quantile(values, 0.25),
        "numeric_std": float(statistics.pstdev(values)) if len(values) >= 2 else None,
        "numeric_q10": _quantile(values, 0.10),
        "numeric_q90": _quantile(values, 0.90),
        "numeric_min": float(min(values)),
        "numeric_max": float(max(values)),
        "numeric_range": float(max(values) - min(values)),
        "numeric_delta": delta,
        "numeric_slope_per_hour": slope,
    }


def _state_features(events: list[FeatureEvent]) -> tuple[int | None, int | None, bool]:
    states = [
        (item.timestamp, item.value_state)
        for item in events
        if item.value_state is not None and item.state_transition_usable
    ]
    if not states:
        return None, None, False
    by_time: dict[datetime, set[str]] = defaultdict(set)
    for timestamp, state in states:
        by_time[timestamp].add(state)
    ambiguity = False
    transitions = 0
    previous: str | None = None
    for timestamp in sorted(by_time):
        group = by_time[timestamp]
        if len(group) != 1:
            ambiguity = True
            previous = None  # Do not invent a transition across an unordered conflict.
            continue
        current = next(iter(group))
        if previous is not None and current != previous:
            transitions += 1
        previous = current
    return transitions, len({state for _, state in states}), ambiguity


def _window_features(
    events: list[FeatureEvent], prediction_time: datetime, hours: int, config: HourlyConfig
) -> dict[str, Any]:
    duration = timedelta(hours=hours)
    usable = [
        item
        for item in events
        if not config.excluded_quality_flags.intersection(item.quality_flags)
    ]
    excluded_count = len(events) - len(usable)
    unique_times = sorted({item.timestamp for item in usable})
    numeric = [
        item
        for item in usable
        if item.value_numeric is not None and item.numeric_measurement_usable
    ]
    states = [
        item for item in usable if item.value_state is not None and item.state_transition_usable
    ]
    transitions, distinct_count, ambiguity = _state_features(states)
    maximum_gap = None
    if unique_times:
        edges = [prediction_time - duration, *unique_times, prediction_time]
        maximum_gap = max((right - left).total_seconds() for left, right in zip(edges, edges[1:]))
    coverage = None
    if config.expected_cadence is not None:
        expected = max(1, math.ceil(duration / config.expected_cadence))
        coverage = min(1.0, len(unique_times) / expected)
    reasons: list[str] = []
    if not usable:
        reasons.append("no_usable_observations")
    if excluded_count:
        reasons.append("quality_exclusions_present")
    if ambiguity:
        reasons.append("same_time_state_ambiguity")
    if config.expected_cadence is None:
        reasons.append("cadence_unknown")
    else:
        required_events = min(config.minimum_window_events, expected)
        if len(unique_times) < required_events:
            reasons.append("insufficient_window_events")
        if coverage is not None and coverage < config.minimum_coverage:
            reasons.append("coverage_below_threshold")
        if (
            maximum_gap is not None
            and maximum_gap > config.expected_cadence.total_seconds() * config.maximum_gap_factor
        ):
            reasons.append("gap_above_threshold")
    if excluded_count and not usable:
        status = "excluded"
        reasons.append("all_interval_observations_excluded")
    else:
        status = "unknown" if reasons else "eligible"
    suffix = f"_{hours}h"
    output: dict[str, Any] = {
        f"event_count{suffix}": len(events),
        f"alarm_count{suffix}": sum(item.alarm for item in events),
        f"excluded_quality_count{suffix}": excluded_count,
        f"numeric_count{suffix}": len(numeric),
        f"state_count{suffix}": len(states),
        f"state_transitions{suffix}": transitions,
        f"state_distinct_count{suffix}": distinct_count,
        f"maximum_gap_seconds{suffix}": maximum_gap,
        f"coverage{suffix}": coverage,
        f"window_status{suffix}": status,
        f"window_reasons{suffix}": reasons,
    }
    output.update({f"{name}{suffix}": value for name, value in _numeric_features(numeric).items()})
    return output


def _baseline_features(
    events: list[FeatureEvent], fit_end_at: datetime, config: HourlyConfig
) -> dict[str, Any]:
    start = fit_end_at - config.baseline_lookback
    baseline = [
        item
        for item in events
        if start < item.timestamp < fit_end_at
        and not config.excluded_quality_flags.intersection(item.quality_flags)
    ]
    numeric = [
        item.value_numeric
        for item in baseline
        if item.value_numeric is not None and item.numeric_measurement_usable
    ]
    states = [
        item.value_state
        for item in baseline
        if item.value_state is not None and item.state_transition_usable
    ]
    reasons: list[str] = []
    if len(baseline) < config.minimum_baseline_events:
        reasons.append("insufficient_baseline_events")
    if not baseline or baseline[-1].timestamp - baseline[0].timestamp < config.minimum_history:
        reasons.append("insufficient_baseline_span")
    state_counts = Counter(states)
    dominant = (
        min(state_counts, key=lambda state: (-state_counts[state], state)) if state_counts else None
    )
    median = float(statistics.median(numeric)) if numeric else None
    return {
        "baseline_fit_end_at": fit_end_at,
        "baseline_status": "unknown" if reasons else "eligible",
        "baseline_reasons": reasons,
        "baseline_event_count": len(baseline),
        "baseline_numeric_count": len(numeric),
        "baseline_state_count": len(states),
        "baseline_numeric_median": median,
        "baseline_numeric_mad": (
            float(statistics.median(abs(value - median) for value in numeric))
            if median is not None
            else None
        ),
        "baseline_dominant_state": dominant,
    }


@dataclass(slots=True)
class _PrefixState:
    """Incremental causal metadata; avoids rescanning all history at each hour."""

    past_count: int = 0
    usable_count: int = 0
    first_usable_at: datetime | None = None
    last_usable_at: datetime | None = None
    unique_usable_count: int = 0
    types: set[str] = field(default_factory=set)
    linked_objects: set[str] = field(default_factory=set)

    def add(self, item: FeatureEvent, config: HourlyConfig) -> None:
        self.past_count += 1
        if item.sensor_type:
            self.types.add(item.sensor_type)
        if item.join_status == "linked" and item.object_id is not None:
            self.linked_objects.add(item.object_id)
        if config.excluded_quality_flags.intersection(item.quality_flags):
            return
        self.usable_count += 1
        if self.first_usable_at is None:
            self.first_usable_at = item.timestamp
        if self.last_usable_at != item.timestamp:
            self.unique_usable_count += 1
        self.last_usable_at = item.timestamp


def _feature_at_sorted(
    events: list[FeatureEvent],
    times: list[datetime],
    channel_id: str,
    prediction_time: datetime,
    config: HourlyConfig,
    baseline_fit_end_at: datetime,
    *,
    prefix: _PrefixState | None = None,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    end = bisect_right(times, prediction_time)
    past = None
    if prefix is None:
        past = events[:end]
        prefix = _PrefixState()
        for item in past:
            prefix.add(item, config)
    if baseline is None:
        if past is None:
            past = events[:end]
        baseline = _baseline_features(past, baseline_fit_end_at, config)
    types = prefix.types
    sensor_type = next(iter(types)) if len(types) == 1 else None
    linked_objects = prefix.linked_objects
    object_id = next(iter(linked_objects)) if len(linked_objects) == 1 else None
    peer_reasons = ["peer_input_unavailable"] if object_id else ["object_mapping_unavailable"]
    last_age = (
        (prediction_time - prefix.last_usable_at).total_seconds()
        if prefix.last_usable_at is not None
        else None
    )
    result: dict[str, Any] = {
        "channel_id": channel_id,
        "prediction_time": prediction_time,
        "sensor_type": sensor_type,
        "object_id": object_id,
        "last_observation_age_seconds": last_age,
        "peer_status": "unknown",
        "peer_reasons": peer_reasons,
    }
    result.update(baseline)
    window_statuses = []
    for hours in WINDOW_HOURS:
        start = prediction_time - timedelta(hours=hours)
        lower = bisect_right(times, start)
        window = events[lower:end]
        result.update(_window_features(window, prediction_time, hours, config))
        window_statuses.append(result[f"window_status_{hours}h"])
    reasons: list[str] = []
    if prefix.past_count == 0:
        reasons.append("no_observations_yet")
    elif prefix.usable_count == 0:
        reasons.append("all_causal_observations_excluded")
    if len(types) == 0:
        reasons.append("sensor_type_unknown")
    elif len(types) > 1:
        reasons.append("sensor_type_conflict")
    if len(linked_objects) > 1:
        reasons.append("object_mapping_conflict")
    if prefix.first_usable_at is None or (
        prediction_time - prefix.first_usable_at < config.minimum_history
        or prefix.unique_usable_count < config.minimum_history_events
    ):
        reasons.append("insufficient_history")
    if result["baseline_status"] != "eligible":
        reasons.extend(result["baseline_reasons"])
    if any(status != "eligible" for status in window_statuses):
        reasons.append("one_or_more_windows_unusable")
    if any(result[f"excluded_quality_count_{hours}h"] for hours in WINDOW_HOURS):
        reasons.append("quality_exclusions_present")
    if config.expected_cadence is None:
        reasons.append("cadence_unknown")
    result["availability_reasons"] = list(dict.fromkeys(reasons))
    if prefix.past_count and not prefix.usable_count:
        result["availability_status"] = "excluded"
    else:
        result["availability_status"] = "unknown" if reasons else "eligible"
    return result


def feature_at(
    events: Iterable[FeatureEvent],
    channel_id: str,
    prediction_time: datetime,
    *,
    config: HourlyConfig | None = None,
    baseline_fit_end_at: datetime | None = None,
) -> dict[str, Any]:
    """Calculate one snapshot from observed events at or before ``prediction_time``."""

    _require_local_time("prediction_time", prediction_time)
    if not isinstance(channel_id, str) or not channel_id.strip():
        raise ValueError("channel_id must be nonempty")
    config = config or HourlyConfig()
    latest_fit_end = prediction_time - timedelta(hours=max(WINDOW_HOURS)) - config.baseline_embargo
    if baseline_fit_end_at is None:
        baseline_fit_end_at = latest_fit_end
    _require_local_time("baseline_fit_end_at", baseline_fit_end_at)
    if baseline_fit_end_at > latest_fit_end:
        raise ValueError("baseline_fit_end_at overlaps the checked window or admission embargo")
    ordered = sorted(
        (item for item in events if item.channel_id == channel_id), key=lambda e: e.timestamp
    )
    times = [item.timestamp for item in ordered]
    return _feature_at_sorted(
        ordered, times, channel_id, prediction_time, config, baseline_fit_end_at
    )


def build_hourly_rows(
    events: Iterable[FeatureEvent],
    channel_id: str,
    start_at: datetime,
    end_at: datetime,
    *,
    config: HourlyConfig | None = None,
) -> list[dict[str, Any]]:
    """Build the bounded hourly grid ``[start_at, end_at)`` for one channel.

    A single baseline cutoff is frozen before the first checked 168-hour interval,
    with an additional admission delay. Monthly callers supply overlap/history;
    this routine never builds a channel×all-history Cartesian product.
    """

    _require_local_time("start_at", start_at)
    _require_local_time("end_at", end_at)
    if not isinstance(channel_id, str) or not channel_id.strip():
        raise ValueError("channel_id must be nonempty")
    if end_at < start_at:
        raise ValueError("end_at precedes start_at")
    if any(
        (timestamp.minute, timestamp.second, timestamp.microsecond) != (0, 0, 0)
        for timestamp in (start_at, end_at)
    ):
        raise ValueError("hourly grid bounds must be whole local hours")
    config = config or HourlyConfig()
    ordered = sorted(
        (item for item in events if item.channel_id == channel_id), key=lambda e: e.timestamp
    )
    times = [item.timestamp for item in ordered]
    fit_end = start_at - timedelta(hours=max(WINDOW_HOURS)) - config.baseline_embargo
    baseline = _baseline_features(ordered, fit_end, config)
    prefix = _PrefixState()
    cursor = 0
    output: list[dict[str, Any]] = []
    prediction_time = start_at
    while prediction_time < end_at:
        while cursor < len(ordered) and ordered[cursor].timestamp <= prediction_time:
            prefix.add(ordered[cursor], config)
            cursor += 1
        output.append(
            _feature_at_sorted(
                ordered,
                times,
                channel_id,
                prediction_time,
                config,
                fit_end,
                prefix=prefix,
                baseline=baseline,
            )
        )
        prediction_time += timedelta(hours=1)
    return output
