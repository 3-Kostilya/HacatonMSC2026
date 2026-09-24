"""R3 labels for future registered journal episodes, never physical failures.

The state timeline is built from text events in accepted full archives. A
prediction at ``t`` sees only groups at or before ``t``. Future groups are used
only to decide the label and its availability time, never as model features.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from .operational import (
    HORIZON,
    future_window_in_archive,
    recent_normal_for_prediction,
    registered_state_effect,
    same_timestamp_state_conflict,
    segment_at,
)
from .registered_episodes import EPISODE_VERSION, RegisteredEpisode, StateEvent
from .rules import KNOWN_SENSOR_TYPES, RULESET_VERSION, TARGET_DEFINITION


LABEL_VERSION = "registered-state-r3-labels-v1"


@dataclass(frozen=True, slots=True)
class PredictionPoint:
    channel_id: str
    sensor_type: str | None
    at: datetime


@dataclass(frozen=True, slots=True)
class RegisteredForecastLabel:
    channel_id: str
    sensor_type: str | None
    prediction_time: datetime
    horizon_end: datetime
    target: int | None
    label_status: str
    reason: str
    label_available_at: datetime | None
    target_episode_id: str | None
    prior_normal_at: datetime | None
    split: str | None
    split_status: str
    ruleset_version: str = RULESET_VERSION
    label_version: str = LABEL_VERSION


@dataclass(frozen=True, slots=True)
class _Snapshot:
    at: datetime
    sensor_type: str | None
    last_normal_at: datetime | None
    active: bool
    uncertain: bool


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    name: str
    start: datetime
    end: datetime


# Frozen before model selection. The archive gap has no input points and no
# history is carried across it; all forecast horizons end before a split edge.
SPLITS = (
    TemporalSplit("train", datetime(2019, 1, 1), datetime(2025, 1, 1)),
    TemporalSplit("validation", datetime(2025, 1, 1), datetime(2026, 1, 1)),
    TemporalSplit("test", datetime(2026, 1, 1), datetime(2026, 7, 1)),
)


def split_at(at: datetime, horizon: timedelta = HORIZON) -> tuple[str | None, str]:
    if at.tzinfo is not None or horizon <= timedelta(0):
        raise ValueError("split requires naive local time and a positive horizon")
    if segment_at(at) is None:
        return None, "outside_accepted_archive"
    for split in SPLITS:
        if split.start <= at < split.end:
            status = "assigned" if at + horizon < split.end else "purged_boundary"
            return split.name, status
    return None, "outside_splits"


class ChannelTimeline:
    """Compact causal state changes and future ambiguity times for one channel."""

    def __init__(self, channel_id: str, events: Iterable[StateEvent]) -> None:
        self.channel_id = channel_id
        self.snapshots: list[_Snapshot] = []
        self.times: list[datetime] = []
        self.adverse_times: list[datetime] = []
        grouped: list[StateEvent] = []
        previous: datetime | None = None
        for event in events:
            if event.channel_id != channel_id or event.at.tzinfo is not None:
                raise ValueError("timeline event has wrong channel or timezone")
            if previous is not None and event.at < previous:
                raise ValueError("timeline events must be sorted by timestamp")
            if previous is not None and event.at != previous:
                self._consume(grouped)
                grouped = []
            grouped.append(event)
            previous = event.at
        if grouped:
            self._consume(grouped)

    def _consume(self, group: list[StateEvent]) -> None:
        at = group[0].at
        segment = segment_at(at)
        if segment is None:
            raise ValueError("timeline event outside accepted archive segments")
        previous = self.snapshots[-1] if self.snapshots else None
        prev_segment = segment_at(previous.at) if previous else None
        types = {event.sensor_type for event in group}
        effects = {
            registered_state_effect(event.sensor_type, event.value_state, event.alarm)
            for event in group
        }
        fault_types = {event.sensor_type for event in group if
                       registered_state_effect(event.sensor_type, event.value_state, event.alarm)
                       == "fault"}
        sensor_type = (
            next(iter(types)) if len(types) == 1
            else next(iter(fault_types)) if len(fault_types) == 1 else None
        )
        if sensor_type is None and previous is not None and prev_segment == segment:
            sensor_type = previous.sensor_type
        changed_type = previous is not None and previous.sensor_type != sensor_type
        if previous is None or prev_segment != segment or changed_type:
            normal_at, active, uncertain = None, False, False
        else:
            normal_at, active, uncertain = (
                previous.last_normal_at, previous.active, previous.uncertain
            )
        conflict = same_timestamp_state_conflict(e.value_state for e in group) or len(types) > 1
        adverse = conflict or "fault" in effects or "uncertain" in effects or changed_type
        if conflict:
            normal_at, uncertain = None, True
            if "fault" in effects:
                active = True
        elif "fault" in effects:
            normal_at, active, uncertain = None, True, False
        elif effects == {"normal"}:
            normal_at, active, uncertain = at, False, False
        elif "uncertain" in effects:
            normal_at, uncertain = None, True
        if adverse:
            self.adverse_times.append(at)
        if (
            previous is None or prev_segment != segment or changed_type
            or (sensor_type, normal_at, active, uncertain) != (
                previous.sensor_type, previous.last_normal_at, previous.active, previous.uncertain
            )
        ):
            self.snapshots.append(_Snapshot(at, sensor_type, normal_at, active, uncertain))
            self.times.append(at)

    def at(self, when: datetime) -> _Snapshot | None:
        index = bisect_right(self.times, when) - 1
        snapshot = self.snapshots[index] if index >= 0 else None
        return snapshot if snapshot and segment_at(snapshot.at) == segment_at(when) else None

    def first_adverse_after(self, when: datetime) -> datetime | None:
        index = bisect_right(self.adverse_times, when)
        return self.adverse_times[index] if index < len(self.adverse_times) else None


class RegisteredTargetIndex:
    """Label arbitrary prediction points for one channel against full B2 history."""

    def __init__(
        self,
        channel_id: str,
        events: Iterable[StateEvent],
        episodes: Iterable[RegisteredEpisode],
        *,
        observed_until: datetime,
    ) -> None:
        if observed_until.tzinfo is not None:
            raise ValueError("observed_until must be naive local time")
        self.channel_id = channel_id
        self.observed_until = observed_until
        self.timeline = ChannelTimeline(channel_id, events)
        self.episodes = sorted(
            (episode for episode in episodes if episode.channel_id == channel_id),
            key=lambda episode: episode.start_at,
        )
        if any(
            episode.ruleset_version != RULESET_VERSION
            or episode.episode_version != EPISODE_VERSION
            or episode.target_kind != TARGET_DEFINITION["target_kind"]
            or episode.confirmed_at != episode.start_at
            for episode in self.episodes
        ):
            raise ValueError("B2 episode version or first-message confirmation differs from R1")
        self.episode_times = [episode.start_at for episode in self.episodes]
        self.candidates_by_type: dict[str, list[RegisteredEpisode]] = {}
        for episode in self.episodes:
            if episode.onset_status == "candidate_new_onset":
                self.candidates_by_type.setdefault(episode.sensor_type, []).append(episode)
        for members in self.candidates_by_type.values():
            members.sort(key=lambda episode: episode.start_at)
        self.candidate_times_by_type = {
            sensor_type: [episode.start_at for episode in members]
            for sensor_type, members in self.candidates_by_type.items()
        }

    def label(self, point: PredictionPoint) -> RegisteredForecastLabel:
        if point.channel_id != self.channel_id or point.at.tzinfo is not None:
            raise ValueError("prediction point has wrong channel or timezone")
        horizon_end = point.at + HORIZON
        split, split_status = split_at(point.at)
        normal_at: datetime | None = None

        def result(
            target: int | None, status: str, reason: str,
            available_at: datetime | None = None,
            episode_id: str | None = None,
        ) -> RegisteredForecastLabel:
            return RegisteredForecastLabel(
                point.channel_id, point.sensor_type, point.at, horizon_end,
                target, status, reason, available_at, episode_id, normal_at,
                split, split_status,
            )

        if point.sensor_type not in KNOWN_SENSOR_TYPES or segment_at(point.at) is None:
            return result(None, "excluded", "unsupported_type_or_archive")
        prior_index = bisect_right(self.episode_times, point.at) - 1
        if prior_index >= 0:
            prior_episode = self.episodes[prior_index]
            if segment_at(prior_episode.start_at) == segment_at(point.at):
                if prior_episode.end_at is None or prior_episode.end_at > point.at:
                    if prior_episode.sensor_type == point.sensor_type:
                        if prior_episode.end_status == "open_type_or_archive_boundary":
                            return result(None, "unknown", "unresolved_type_or_archive_boundary")
                        return result(None, "excluded", "registered_episode_active_at_t")
        state = self.timeline.at(point.at)
        if state is None or state.sensor_type != point.sensor_type:
            return result(None, "unknown", "no_verified_same_type_history")
        if state.active:
            return result(None, "excluded", "registered_episode_active_at_t")
        if state.uncertain:
            return result(None, "unknown", "uncertain_state_at_t")
        normal_at = state.last_normal_at
        if not recent_normal_for_prediction(normal_at, point.at):
            return result(None, "unknown", "no_recent_explicit_normal_at_t")

        first_adverse = self.timeline.first_adverse_after(point.at)
        candidates = self.candidates_by_type.get(point.sensor_type, [])
        candidate_index = bisect_right(self.candidate_times_by_type.get(point.sensor_type, []), point.at)
        candidate = (
            candidates[candidate_index]
            if candidate_index < len(candidates) else None
        )
        if candidate is not None and (
            candidate.start_at > horizon_end
            or segment_at(candidate.start_at) != segment_at(point.at)
            or candidate.confirmed_at >= self.observed_until
        ):
            candidate = None
        if candidate is not None and (
            first_adverse is None or first_adverse >= candidate.start_at
        ):
            return result(
                1, "positive", "new_confident_registered_onset",
                candidate.confirmed_at, candidate.episode_id,
            )
        if first_adverse is not None and first_adverse <= horizon_end:
            return result(None, "unknown", "future_state_or_onset_uncertain")
        if horizon_end >= self.observed_until:
            return result(None, "unknown", "future_events_not_loaded")
        if not future_window_in_archive(point.at):
            return result(None, "unknown", "future_archive_window_incomplete")
        return result(0, "negative", "no_registered_onset_in_assumed_complete_archive",
                      horizon_end)


def label_points(
    points: Iterable[PredictionPoint],
    events: Iterable[StateEvent],
    episodes: Iterable[RegisteredEpisode],
    *,
    observed_until: datetime,
) -> list[RegisteredForecastLabel]:
    points_by_channel: dict[str, list[PredictionPoint]] = {}
    events_by_channel: dict[str, list[StateEvent]] = {}
    episodes_by_channel: dict[str, list[RegisteredEpisode]] = {}
    for point in points:
        points_by_channel.setdefault(point.channel_id, []).append(point)
    for event in events:
        events_by_channel.setdefault(event.channel_id, []).append(event)
    for episode in episodes:
        episodes_by_channel.setdefault(episode.channel_id, []).append(episode)
    result = []
    for channel_id, selected in sorted(points_by_channel.items()):
        index = RegisteredTargetIndex(
            channel_id,
            sorted(events_by_channel.get(channel_id, []), key=lambda item: (item.at, item.row_id)),
            episodes_by_channel.get(channel_id, []),
            observed_until=observed_until,
        )
        result.extend(index.label(point) for point in selected)
    return result
