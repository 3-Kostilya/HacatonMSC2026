"""Explainable, causal baseline detectors for normalized in-memory events.

The rules in this module are deliberately hypotheses, not learned production
thresholds.  Every detector uses only a channel's earlier observations as its
baseline and returns the public :class:`stage1.contracts.Episode` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import hashlib
import statistics
from typing import Iterable, Sequence

from stage1.contracts import Decision, Episode, NormalizedEvent
from stage1.normalization import CHANNEL_TIME_CONFLICT, INVALID_TIMESTAMP, NONFINITE_NUMERIC


RULESET_VERSION = "stage1-baseline-v2"


@dataclass(frozen=True, slots=True)
class NumericDetectorConfig:
    """Configurable hypotheses for a sustained within-channel level shift."""

    baseline_size: int = 12
    min_sustained: int = 3
    mad_multiplier: float = 6.0
    zero_mad_absolute_delta: float = 0.5
    sensor_group: str = "numeric"
    blocking_quality_flags: tuple[str, ...] = (
        CHANNEL_TIME_CONFLICT,
        INVALID_TIMESTAMP,
        NONFINITE_NUMERIC,
    )

    def __post_init__(self) -> None:
        if self.baseline_size < 3:
            raise ValueError("baseline_size must be at least 3")
        if self.min_sustained < 2:
            raise ValueError("min_sustained must be at least 2")
        if self.mad_multiplier <= 0 or self.zero_mad_absolute_delta <= 0:
            raise ValueError("numeric thresholds must be positive")


@dataclass(frozen=True, slots=True)
class DiscreteDetectorConfig:
    """Configurable hypotheses for repeated states and rapid switching."""

    baseline_size: int = 6
    repeated_state_count: int = 3
    repeated_state_window: timedelta = timedelta(minutes=10)
    transition_count: int = 4
    transition_window: timedelta = timedelta(minutes=10)
    repeat_acceleration_factor: float = 3.0
    sensor_group: str = "discrete"
    known_states: frozenset[str] | None = None
    blocking_quality_flags: tuple[str, ...] = (
        CHANNEL_TIME_CONFLICT,
        INVALID_TIMESTAMP,
    )

    def __post_init__(self) -> None:
        if self.baseline_size < 2:
            raise ValueError("baseline_size must be at least 2")
        if self.repeated_state_count < 2 or self.transition_count < 2:
            raise ValueError("sustained discrete thresholds must be at least 2")
        if self.repeated_state_window <= timedelta(0) or self.transition_window <= timedelta(0):
            raise ValueError("discrete windows must be positive")
        if self.repeat_acceleration_factor <= 1:
            raise ValueError("repeat_acceleration_factor must be greater than 1")


@dataclass(frozen=True, slots=True)
class ContextDetectorConfig:
    """Guarded coincidence rule; object identity is the required relation."""

    coincidence_window: timedelta = timedelta(minutes=2)
    min_coincidences: int = 2
    sensor_group: str = "context"

    def __post_init__(self) -> None:
        if self.coincidence_window <= timedelta(0):
            raise ValueError("coincidence_window must be positive")
        if self.min_coincidences < 2:
            raise ValueError("min_coincidences must be at least 2")


def _episode_id(channel_id: str, anomaly_type: str, start_at: datetime) -> str:
    key = f"{channel_id}|{anomaly_type}|{start_at.isoformat()}".encode()
    return "ep-" + hashlib.sha1(key).hexdigest()[:16]


def _ordered_single_channel(events: Iterable[NormalizedEvent]) -> list[NormalizedEvent]:
    ordered = sorted(events, key=lambda event: event.timestamp)
    if not ordered:
        raise ValueError("events must not be empty")
    channel_ids = {event.channel_id for event in ordered}
    sensor_types = {event.sensor_type for event in ordered}
    if len(channel_ids) != 1 or len(sensor_types) != 1:
        raise ValueError("a detector call must contain one channel and sensor type")
    return ordered


def _make_episode(
    events: Sequence[NormalizedEvent],
    *,
    sensor_group: str,
    anomaly_type: str,
    decision: Decision,
    start_at: datetime,
    confirmed_at: datetime,
    evidence: tuple[str, ...] = (),
    quality: tuple[str, ...] = (),
    cause: str = "unknown",
    score: float | None = None,
    metadata: dict[str, object] | None = None,
) -> Episode:
    first = events[0]
    return Episode(
        episode_id=_episode_id(first.channel_id, anomaly_type, start_at),
        channel_id=first.channel_id,
        sensor_type=first.sensor_type,
        sensor_group=sensor_group,
        anomaly_type=anomaly_type,
        decision=decision,
        start_at=start_at,
        confirmed_at=confirmed_at,
        ruleset_version=RULESET_VERSION,
        evidence=evidence,
        observation_quality=quality,
        cause_hypothesis=cause,
        object_id=first.object_id,
        score=score,
        metadata=metadata or {},
    )


def _blocking_flags(events: Sequence[NormalizedEvent], allowed: tuple[str, ...]) -> tuple[str, ...]:
    present = {flag for event in events for flag in event.quality_flags}
    return tuple(sorted(present.intersection(allowed)))


def _numeric_history(
    ordered: Sequence[NormalizedEvent], cfg: NumericDetectorConfig
) -> tuple[list[NormalizedEvent], list[NormalizedEvent], Episode | None]:
    """Collect the numeric baseline without skipping blocking raw events."""

    baseline: list[NormalizedEvent] = []
    evaluation_start = len(ordered)
    for index, event in enumerate(ordered):
        blocked = _blocking_flags((event,), cfg.blocking_quality_flags)
        if blocked:
            return (
                baseline,
                [],
                _make_episode(
                    ordered,
                    sensor_group=cfg.sensor_group,
                    anomaly_type="numeric_level_shift",
                    decision=Decision.UNKNOWN,
                    start_at=event.timestamp,
                    confirmed_at=event.timestamp,
                    quality=tuple(f"blocking_quality:{flag}" for flag in blocked),
                ),
            )
        if event.numeric_value is not None:
            baseline.append(event)
            if len(baseline) == cfg.baseline_size:
                evaluation_start = index + 1
                break

    required = cfg.baseline_size + cfg.min_sustained
    numeric_count = sum(event.numeric_value is not None for event in ordered)
    if len(baseline) < cfg.baseline_size or numeric_count < required:
        return (
            baseline,
            [],
            _make_episode(
                ordered,
                sensor_group=cfg.sensor_group,
                anomaly_type="numeric_level_shift",
                decision=Decision.UNKNOWN,
                start_at=ordered[0].timestamp,
                confirmed_at=ordered[-1].timestamp,
                quality=(f"insufficient_numeric_history:{numeric_count}<{required}",),
            ),
        )
    return baseline, list(ordered[evaluation_start:]), None


def detect_numeric_level_shift(
    events: Iterable[NormalizedEvent], config: NumericDetectorConfig | None = None
) -> Episode:
    """Detect a sustained robust level shift after a strictly earlier baseline."""

    cfg = config or NumericDetectorConfig()
    ordered = _ordered_single_channel(events)
    baseline, evaluation, invalid = _numeric_history(ordered, cfg)
    if invalid is not None:
        return invalid
    baseline_values = [event.numeric_value for event in baseline]
    median = statistics.median(baseline_values)
    mad = statistics.median(abs(value - median) for value in baseline_values)
    threshold = cfg.zero_mad_absolute_delta if mad == 0 else cfg.mad_multiplier * mad

    run: list[NormalizedEvent] = []
    direction = 0
    for event in evaluation:
        blocked = _blocking_flags((event,), cfg.blocking_quality_flags)
        if blocked:
            return _make_episode(
                ordered,
                sensor_group=cfg.sensor_group,
                anomaly_type="numeric_level_shift",
                decision=Decision.UNKNOWN,
                start_at=event.timestamp,
                confirmed_at=event.timestamp,
                quality=tuple(f"blocking_quality:{flag}" for flag in blocked),
                metadata={"baseline_end_at": baseline[-1].timestamp.isoformat(sep=" ")},
            )
        if event.numeric_value is None:
            run = []
            direction = 0
            continue
        delta = event.numeric_value - median
        current_direction = 1 if delta > threshold else -1 if delta < -threshold else 0
        if current_direction == 0:
            run = []
            direction = 0
            continue
        if current_direction != direction:
            run = [event]
            direction = current_direction
        else:
            run.append(event)
        if len(run) >= cfg.min_sustained:
            start = run[0]
            confirmed = run[cfg.min_sustained - 1]
            scale_note = "absolute_fallback" if mad == 0 else "mad"
            return _make_episode(
                ordered,
                sensor_group=cfg.sensor_group,
                anomaly_type="numeric_level_shift",
                decision=Decision.CANDIDATE,
                start_at=start.timestamp,
                confirmed_at=confirmed.timestamp,
                evidence=(
                    f"sustained_points={cfg.min_sustained}",
                    f"baseline_median={median:g}",
                    f"baseline_mad={mad:g}",
                    f"threshold={threshold:g} ({scale_note})",
                    f"direction={'up' if direction > 0 else 'down'}",
                ),
                cause="local_or_environmental",
                score=min(1.0, 0.5 + 0.1 * len(run)),
                metadata={
                    "baseline_end_at": baseline[-1].timestamp.isoformat(sep=" "),
                    "causal_baseline_points": len(baseline),
                },
            )

    return _make_episode(
        ordered,
        sensor_group=cfg.sensor_group,
        anomaly_type="numeric_level_shift",
        decision=Decision.NO_CANDIDATE,
        start_at=evaluation[0].timestamp,
        confirmed_at=evaluation[-1].timestamp,
        evidence=(f"no_sustained_run_of_{cfg.min_sustained}",),
        metadata={"baseline_end_at": baseline[-1].timestamp.isoformat(sep=" ")},
    )


def detect_discrete_pattern(
    events: Iterable[NormalizedEvent], config: DiscreteDetectorConfig | None = None
) -> Episode:
    """Detect sustained duplicate bursts or rapid alternating transitions."""

    cfg = config or DiscreteDetectorConfig()
    ordered = _ordered_single_channel(events)
    baseline = ordered[: cfg.baseline_size]
    blocked = _blocking_flags(baseline, cfg.blocking_quality_flags)
    if blocked:
        return _make_episode(
            ordered,
            sensor_group=cfg.sensor_group,
            anomaly_type="discrete_pattern",
            decision=Decision.UNKNOWN,
            start_at=ordered[0].timestamp,
            confirmed_at=baseline[-1].timestamp,
            quality=tuple(f"blocking_quality:{flag}" for flag in blocked),
        )
    required = cfg.baseline_size + 1
    if len(ordered) < required:
        return _make_episode(
            ordered,
            sensor_group=cfg.sensor_group,
            anomaly_type="discrete_pattern",
            decision=Decision.UNKNOWN,
            start_at=ordered[0].timestamp,
            confirmed_at=ordered[-1].timestamp,
            quality=(f"insufficient_discrete_history:{len(ordered)}<{required}",),
        )
    if cfg.known_states is not None:
        unknown = sorted(
            {event.raw_value for event in baseline if event.raw_value not in cfg.known_states}
        )
        if unknown:
            return _make_episode(
                ordered,
                sensor_group=cfg.sensor_group,
                anomaly_type="discrete_pattern",
                decision=Decision.UNKNOWN,
                start_at=ordered[0].timestamp,
                confirmed_at=baseline[-1].timestamp,
                quality=("unknown_states:" + ",".join(unknown),),
            )

    evaluation = ordered[cfg.baseline_size :]
    baseline_max_run: dict[str, int] = {}
    baseline_repeat_gaps: dict[str, list[float]] = {}
    current_value = None
    current_run = 0
    for event in baseline:
        current_run = current_run + 1 if event.raw_value == current_value else 1
        current_value = event.raw_value
        baseline_max_run[current_value] = max(baseline_max_run.get(current_value, 0), current_run)
    for previous, current in zip(baseline, baseline[1:]):
        if previous.raw_value == current.raw_value:
            baseline_repeat_gaps.setdefault(current.raw_value, []).append(
                (current.timestamp - previous.timestamp).total_seconds()
            )
    baseline_transitions: list[datetime] = []
    for previous, current in zip(baseline, baseline[1:]):
        if previous.raw_value != current.raw_value:
            baseline_transitions.append(current.timestamp)
    baseline_transition_peak = 0
    for index, timestamp in enumerate(baseline_transitions):
        inside = [
            item
            for item in baseline_transitions[: index + 1]
            if timestamp - item <= cfg.transition_window
        ]
        baseline_transition_peak = max(baseline_transition_peak, len(inside))

    repeat_run: list[NormalizedEvent] = []
    transitions: list[NormalizedEvent] = []
    previous = baseline[-1]
    for event in evaluation:
        blocked = _blocking_flags((event,), cfg.blocking_quality_flags)
        if blocked:
            return _make_episode(
                ordered,
                sensor_group=cfg.sensor_group,
                anomaly_type="discrete_pattern",
                decision=Decision.UNKNOWN,
                start_at=event.timestamp,
                confirmed_at=event.timestamp,
                quality=tuple(f"blocking_quality:{flag}" for flag in blocked),
                metadata={"baseline_end_at": baseline[-1].timestamp.isoformat(sep=" ")},
            )
        if cfg.known_states is not None and event.raw_value not in cfg.known_states:
            return _make_episode(
                ordered,
                sensor_group=cfg.sensor_group,
                anomaly_type="discrete_pattern",
                decision=Decision.UNKNOWN,
                start_at=event.timestamp,
                confirmed_at=event.timestamp,
                quality=(f"unknown_states:{event.raw_value}",),
                metadata={"baseline_end_at": baseline[-1].timestamp.isoformat(sep=" ")},
            )
        if repeat_run and (
            repeat_run[-1].raw_value != event.raw_value
            or event.timestamp - repeat_run[0].timestamp > cfg.repeated_state_window
        ):
            repeat_run = []
        repeat_run.append(event)
        repeat_is_novel = baseline_max_run.get(event.raw_value, 0) < cfg.repeated_state_count
        baseline_gaps = baseline_repeat_gaps.get(event.raw_value, [])
        current_gap = (
            (repeat_run[-1].timestamp - repeat_run[0].timestamp).total_seconds()
            / (len(repeat_run) - 1)
            if len(repeat_run) > 1
            else float("inf")
        )
        repeat_is_accelerated = bool(baseline_gaps) and current_gap <= (
            statistics.median(baseline_gaps) / cfg.repeat_acceleration_factor
        )
        if len(repeat_run) >= cfg.repeated_state_count and (
            repeat_is_novel or repeat_is_accelerated
        ):
            confirmed = repeat_run[cfg.repeated_state_count - 1]
            return _make_episode(
                ordered,
                sensor_group=cfg.sensor_group,
                anomaly_type="repeated_state_burst",
                decision=Decision.CANDIDATE,
                start_at=repeat_run[0].timestamp,
                confirmed_at=confirmed.timestamp,
                evidence=(
                    f"state={event.raw_value}",
                    f"consecutive_messages={cfg.repeated_state_count}",
                    f"window_seconds={int(cfg.repeated_state_window.total_seconds())}",
                ),
                cause="local_or_reporting",
                metadata={"baseline_end_at": baseline[-1].timestamp.isoformat(sep=" ")},
            )

        if event.raw_value != previous.raw_value:
            transitions.append(event)
        previous = event
        while transitions and event.timestamp - transitions[0].timestamp > cfg.transition_window:
            transitions.pop(0)
        if (
            len(transitions) >= cfg.transition_count
            and baseline_transition_peak < cfg.transition_count
        ):
            confirmed = transitions[cfg.transition_count - 1]
            return _make_episode(
                ordered,
                sensor_group=cfg.sensor_group,
                anomaly_type="rapid_switching",
                decision=Decision.CANDIDATE,
                start_at=transitions[0].timestamp,
                confirmed_at=confirmed.timestamp,
                evidence=(
                    f"transitions={cfg.transition_count}",
                    f"window_seconds={int(cfg.transition_window.total_seconds())}",
                ),
                cause="local_or_operational",
                metadata={"baseline_end_at": baseline[-1].timestamp.isoformat(sep=" ")},
            )

    return _make_episode(
        ordered,
        sensor_group=cfg.sensor_group,
        anomaly_type="discrete_pattern",
        decision=Decision.NO_CANDIDATE,
        start_at=evaluation[0].timestamp,
        confirmed_at=evaluation[-1].timestamp,
        evidence=("no_sustained_discrete_pattern",),
        metadata={"baseline_end_at": baseline[-1].timestamp.isoformat(sep=" ")},
    )


def detect_context_coincidence(
    target_events: Iterable[NormalizedEvent],
    context_events: Iterable[NormalizedEvent],
    config: ContextDetectorConfig | None = None,
) -> Episode:
    """Evaluate coincidences only when a shared object relation is explicit."""

    cfg = config or ContextDetectorConfig()
    target = _ordered_single_channel(target_events)
    context = sorted(context_events, key=lambda event: event.timestamp)
    object_id = target[0].object_id
    if not object_id or any(event.object_id != object_id for event in target):
        return _make_episode(
            target,
            sensor_group=cfg.sensor_group,
            anomaly_type="cross_channel_coincidence",
            decision=Decision.UNKNOWN,
            start_at=target[0].timestamp,
            confirmed_at=target[-1].timestamp,
            quality=("unconfirmed_context_link",),
        )
    linked = [
        event
        for event in context
        if event.object_id == object_id and event.channel_id != target[0].channel_id
    ]
    if not linked:
        return _make_episode(
            target,
            sensor_group=cfg.sensor_group,
            anomaly_type="cross_channel_coincidence",
            decision=Decision.UNKNOWN,
            start_at=target[0].timestamp,
            confirmed_at=target[-1].timestamp,
            quality=("no_confirmed_linked_context_events",),
        )

    matched: list[tuple[NormalizedEvent, NormalizedEvent]] = []
    for event in target:
        earlier = [
            other
            for other in linked
            if other.timestamp <= event.timestamp
            and event.timestamp - other.timestamp <= cfg.coincidence_window
        ]
        if earlier:
            matched.append((event, earlier[-1]))
        if len(matched) >= cfg.min_coincidences:
            start, _ = matched[0]
            confirmed, _ = matched[cfg.min_coincidences - 1]
            channels = sorted({other.channel_id for _, other in matched})
            return _make_episode(
                target,
                sensor_group=cfg.sensor_group,
                anomaly_type="cross_channel_coincidence",
                decision=Decision.CANDIDATE,
                start_at=start.timestamp,
                confirmed_at=confirmed.timestamp,
                evidence=(
                    f"confirmed_object_id={object_id}",
                    f"causal_coincidences={cfg.min_coincidences}",
                    "context_channels=" + ",".join(channels),
                ),
                cause="shared_context_hypothesis",
            )

    return _make_episode(
        target,
        sensor_group=cfg.sensor_group,
        anomaly_type="cross_channel_coincidence",
        decision=Decision.NO_CANDIDATE,
        start_at=target[0].timestamp,
        confirmed_at=target[-1].timestamp,
        evidence=(f"coincidences={len(matched)}<{cfg.min_coincidences}",),
    )


def detect_numeric_level_shifts(
    events: Iterable[NormalizedEvent], config: NumericDetectorConfig | None = None
) -> list[Episode]:
    """Emit sequential level-shift episodes while preserving recovery boundaries."""

    cfg = config or NumericDetectorConfig()
    ordered = _ordered_single_channel(events)
    first = detect_numeric_level_shift(ordered, cfg)
    if first.decision is not Decision.CANDIDATE:
        return [first]

    baseline, evaluation, invalid = _numeric_history(ordered, cfg)
    if invalid is not None:
        return [invalid]
    median = statistics.median(event.numeric_value for event in baseline)
    mad = statistics.median(abs(event.numeric_value - median) for event in baseline)
    threshold = cfg.zero_mad_absolute_delta if mad == 0 else cfg.mad_multiplier * mad
    run: list[NormalizedEvent] = []
    direction = 0
    active: Episode | None = None
    emitted: list[Episode] = []
    for event in evaluation:
        if _blocking_flags((event,), cfg.blocking_quality_flags):
            if active is not None:
                emitted.append(active)
            break
        if event.numeric_value is None:
            if active is not None:
                emitted.append(replace(active, end_at=event.timestamp))
                active = None
            run = []
            direction = 0
            continue
        delta = event.numeric_value - median
        current = 1 if delta > threshold else -1 if delta < -threshold else 0
        if current == 0:
            if active is not None:
                emitted.append(replace(active, end_at=event.timestamp))
                active = None
            run = []
            direction = 0
            continue
        if current != direction:
            if active is not None:
                emitted.append(replace(active, end_at=event.timestamp))
                active = None
            run = [event]
            direction = current
        else:
            run.append(event)
        if active is None and len(run) >= cfg.min_sustained:
            confirmed = run[cfg.min_sustained - 1]
            active = _make_episode(
                ordered,
                sensor_group=cfg.sensor_group,
                anomaly_type="numeric_level_shift",
                decision=Decision.CANDIDATE,
                start_at=run[0].timestamp,
                confirmed_at=confirmed.timestamp,
                evidence=(
                    f"sustained_points={cfg.min_sustained}",
                    f"baseline_median={median:g}",
                    f"baseline_mad={mad:g}",
                    f"threshold={threshold:g}",
                    f"direction={'up' if direction > 0 else 'down'}",
                ),
                cause="local_or_environmental",
                metadata={"baseline_end_at": baseline[-1].timestamp.isoformat(sep=" ")},
            )
    if active is not None:
        emitted.append(active)
    return emitted or [first]


def detect_discrete_patterns(
    events: Iterable[NormalizedEvent], config: DiscreteDetectorConfig | None = None
) -> list[Episode]:
    """Emit chronological discrete detections instead of replacing an earlier one."""

    cfg = config or DiscreteDetectorConfig()
    ordered = _ordered_single_channel(events)
    initial = detect_discrete_pattern(ordered, cfg)
    if initial.decision is not Decision.CANDIDATE:
        return [initial]
    baseline = ordered[: cfg.baseline_size]
    remaining = ordered[cfg.baseline_size :]
    emitted: list[Episode] = []
    while remaining:
        result = detect_discrete_pattern(baseline + remaining, cfg)
        if result.decision is not Decision.CANDIDATE:
            break
        later = [event for event in remaining if event.timestamp > result.confirmed_at]
        recovery = next(
            (
                event
                for event in later
                if result.anomaly_type == "repeated_state_burst"
                and not any(item == f"state={event.raw_value}" for item in result.evidence)
            ),
            None,
        )
        if result.anomaly_type == "rapid_switching":
            previous_value = next(
                event.raw_value for event in ordered if event.timestamp == result.confirmed_at
            )
            last_transition = result.confirmed_at
            for event in later:
                if event.raw_value != previous_value:
                    last_transition = event.timestamp
                previous_value = event.raw_value
                if event.timestamp - last_transition > cfg.transition_window:
                    recovery = event
                    break
        emitted.append(replace(result, end_at=recovery.timestamp) if recovery else result)
        if recovery is None:
            break
        remaining = [event for event in later if event.timestamp >= recovery.timestamp]
    return emitted
