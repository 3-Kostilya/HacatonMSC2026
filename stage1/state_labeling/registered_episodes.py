"""Causal, per-channel catalog of registered ``Неисправен`` episodes.

This records journal transitions, not verified physical failures or channel uptime.
Rows sharing a timestamp are evaluated together; their row IDs provide identity,
never a causal order within that timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from .operational import (
    recent_explicit_normal,
    registered_state_effect,
    same_timestamp_state_conflict,
    segment_at,
)
from .rules import RULESET_VERSION, TARGET_DEFINITION


EPISODE_VERSION = "registered-state-episodes-b2-v1"


@dataclass(frozen=True, slots=True)
class StateEvent:
    row_id: int
    channel_id: str
    sensor_type: str | None
    at: datetime
    value_state: str
    alarm: bool | None = None


@dataclass(slots=True)
class RegisteredEpisode:
    episode_id: str
    channel_id: str
    sensor_type: str
    target_kind: str
    start_at: datetime
    confirmed_at: datetime
    end_at: datetime | None
    onset_status: str
    end_status: str
    prior_normal_at: datetime | None
    first_row_id: int
    last_fault_at: datetime
    fault_message_count: int = 1
    uncertain_intervening_state: bool = False
    evidence: list[str] = field(default_factory=list)
    ruleset_version: str = RULESET_VERSION
    episode_version: str = EPISODE_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(slots=True)
class _ChannelState:
    segment: int
    sensor_type: str | None
    last_normal_at: datetime | None = None
    uncertain_since_normal: bool = False
    open_episode: RegisteredEpisode | None = None


class EpisodeBuilder:
    """Consume globally time-ordered text events, retaining only channel state."""

    def __init__(self) -> None:
        self.episodes: list[RegisteredEpisode] = []
        self.states: dict[str, _ChannelState] = {}
        self.message_counts = {
            "text_rows": 0,
            "known_fault_rows": 0,
            "unknown_type_fault_rows": 0,
            "known_normal_rows": 0,
            "conflicting_channel_timestamps": 0,
            "conflicting_target_timestamps": 0,
        }
        self._group: list[StateEvent] = []
        self._last_key: tuple[datetime, str] | None = None

    def add(self, event: StateEvent) -> None:
        if not event.channel_id or event.at.tzinfo is not None:
            raise ValueError("events require a channel and naive local timestamp")
        key = (event.at, event.channel_id)
        if self._last_key is not None and key < self._last_key:
            raise ValueError("events must be sorted by timestamp and channel")
        if self._last_key is not None and key != self._last_key:
            self._flush()
        self._group.append(event)
        self._last_key = key

    def finish(self) -> list[RegisteredEpisode]:
        self._flush()
        return self.episodes

    def _flush(self) -> None:
        if not self._group:
            return
        group, self._group = self._group, []
        at, channel_id = group[0].at, group[0].channel_id
        segment = segment_at(at)
        if segment is None:
            raise ValueError("event falls outside accepted archive segments")

        self.message_counts["text_rows"] += len(group)
        known_faults = [
            row for row in group
            if registered_state_effect(row.sensor_type, row.value_state, row.alarm) == "fault"
        ]
        unknown_faults = [
            row for row in group if row.value_state == "Неисправен" and row.sensor_type is None
        ]
        self.message_counts["known_fault_rows"] += len(known_faults)
        self.message_counts["unknown_type_fault_rows"] += len(unknown_faults)
        self.message_counts["known_normal_rows"] += sum(
            registered_state_effect(row.sensor_type, row.value_state, row.alarm) == "normal"
            for row in group
        )
        types = {row.sensor_type for row in group}
        conflict = same_timestamp_state_conflict(row.value_state for row in group) or len(types) > 1
        if conflict:
            self.message_counts["conflicting_channel_timestamps"] += 1
            if known_faults:
                self.message_counts["conflicting_target_timestamps"] += 1

        # A changed or unknown type cannot inherit the prior type's healthy state.
        fault_types = {row.sensor_type for row in known_faults}
        sensor_type = (
            next(iter(types)) if len(types) == 1
            else next(iter(fault_types)) if len(fault_types) == 1 else None
        )
        state = self.states.get(channel_id)
        if sensor_type is None and state is not None and state.segment == segment:
            # Missing type metadata is uncertainty, not an observed type change.
            sensor_type = state.sensor_type
        if state is None or state.segment != segment or state.sensor_type != sensor_type:
            if state is not None and state.open_episode is not None:
                state.open_episode.end_status = "open_type_or_archive_boundary"
                state.open_episode.evidence.append("type_or_archive_boundary")
            state = _ChannelState(segment=segment, sensor_type=sensor_type)
            self.states[channel_id] = state

        if conflict:
            state.last_normal_at = None
            state.uncertain_since_normal = True
            if state.open_episode is not None:
                state.open_episode.uncertain_intervening_state = True
                state.open_episode.evidence.append("same_timestamp_conflict")

        if known_faults:
            first = min(known_faults, key=lambda row: row.row_id)
            if state.open_episode is None:
                if conflict:
                    onset_status = "conflicted_same_timestamp"
                elif state.uncertain_since_normal:
                    onset_status = "uncertain_prior_state"
                elif recent_explicit_normal(state.last_normal_at, at):
                    onset_status = "candidate_new_onset"
                elif state.last_normal_at is not None:
                    onset_status = "stale_normal"
                else:
                    onset_status = "left_censored"
                episode = RegisteredEpisode(
                    episode_id=f"{channel_id}:{at.isoformat()}:{first.row_id}",
                    channel_id=channel_id,
                    sensor_type=first.sensor_type,  # type: ignore[arg-type]
                    target_kind=TARGET_DEFINITION["target_kind"],
                    start_at=at,
                    confirmed_at=at,
                    end_at=None,
                    onset_status=onset_status,
                    end_status="open_unknown",
                    prior_normal_at=state.last_normal_at,
                    first_row_id=first.row_id,
                    last_fault_at=at,
                    fault_message_count=len(known_faults),
                    uncertain_intervening_state=conflict,
                    evidence=["exact_neispraven", onset_status],
                )
                self.episodes.append(episode)
                state.open_episode = episode
            else:
                episode = state.open_episode
                episode.fault_message_count += len(known_faults)
                episode.last_fault_at = at
            state.last_normal_at = None
            return

        if conflict:
            return
        effects = {
            registered_state_effect(row.sensor_type, row.value_state, row.alarm)
            for row in group
        }
        if effects == {"normal"}:
            if state.open_episode is not None and at > state.open_episode.start_at:
                episode = state.open_episode
                episode.end_at = at
                episode.end_status = (
                    "exact_norma_after_uncertainty"
                    if episode.uncertain_intervening_state else "exact_norma"
                )
                episode.evidence.append("later_exact_norma")
                state.open_episode = None
            state.last_normal_at = at
            state.uncertain_since_normal = False
        elif "uncertain" in effects:
            state.last_normal_at = None
            state.uncertain_since_normal = True
            if state.open_episode is not None:
                state.open_episode.uncertain_intervening_state = True
                state.open_episode.evidence.append("intervening_uncertain_text")


def build_episodes(events: Iterable[StateEvent]) -> EpisodeBuilder:
    builder = EpisodeBuilder()
    for event in events:
        builder.add(event)
    builder.finish()
    return builder
