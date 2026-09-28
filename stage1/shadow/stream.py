"""Watermarked past-only admission and the unchanged R6 score.

The caller delivers a complete channel-timestamp group before closing the
prediction watermark. This API never receives future labels or B2 catalogs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Mapping, Sequence

import pandas as pd

from ml.forecast.r6_rule import TERMS, predict_rule
from stage1.features.hourly import FeatureEvent, HourlyConfig, _PrefixState, _baseline_features
from stage1.features.r2 import _branch_status
from stage1.state_labeling.operational import recent_normal_for_prediction, source_is_full_archive
from stage1.state_labeling.operational import segment_at
from stage1.state_labeling.registered_episodes import EpisodeBuilder, StateEvent
from stage1.state_labeling.rules import KNOWN_SENSOR_TYPES, RULESET_VERSION, classify_message


PILOT_VERSION = "shadow-pilot-a-causal-admission-v1"
RETENTION = timedelta(days=37)
FORBIDDEN_FIELDS = frozenset(
    {
        "target",
        "label_status",
        "label_available_at",
        "target_episode_id",
        "split",
        "split_status",
        "future_label_status",
        "admission_status",
    }
)


@dataclass(frozen=True, slots=True)
class Observation:
    row_id: int
    event: FeatureEvent
    source: str

    @classmethod
    def from_record(cls, row: Mapping) -> Observation:
        if FORBIDDEN_FIELDS.intersection(row):
            raise ValueError("future labels and retrospective admission are forbidden inputs")
        if not isinstance(row["row_id"], int) or not isinstance(row["source"], str):
            raise ValueError("observation needs a row identity and source")
        return cls(row["row_id"], FeatureEvent.from_clean_record(row), row["source"])


@dataclass(slots=True)
class _Channel:
    segment: int
    events: deque[FeatureEvent] = field(default_factory=deque)
    prefix: _PrefixState = field(default_factory=_PrefixState)
    episodes: EpisodeBuilder = field(default_factory=EpisodeBuilder)
    faults: deque[datetime] = field(default_factory=deque)
    technical: deque[datetime] = field(default_factory=deque)
    completed: deque[datetime] = field(default_factory=deque)
    baseline_day: datetime | None = None
    baseline: dict | None = None
    metadata_uncertain: bool = False
    last_warning_at: datetime | None = None
    last_observation_at: datetime | None = None


class ShadowStream:
    """Bounded event history, explicit unavailability, and causal warning cooldown.

    The admission proposal preserves legacy *past* discrete data requirements;
    it is not the jointly approved retrospective R3 evaluation population.
    Availability remains unknown, even for a conditionally scored hour.
    """

    def __init__(self, *, threshold: float, cooldown_hours: int = 24) -> None:
        if threshold != 7.1 or cooldown_hours != 24:
            raise ValueError("shadow replay must retain the frozen R6 threshold and cooldown")
        self.threshold = threshold
        self.cooldown = timedelta(hours=cooldown_hours)
        self.config = HourlyConfig()
        self.channels: dict[str, _Channel] = {}
        self.watermark: datetime | None = None
        self._last_group: tuple[datetime, str] | None = None
        self.accepted_rows = self.ignored_source_rows = 0

    def observe_group(self, group: Sequence[Observation]) -> None:
        """Consume one CLOSED channel-second; never use row IDs as a time order."""
        if not group:
            raise ValueError("observation group cannot be empty")
        at, channel = group[0].event.timestamp, group[0].event.channel_id
        if any((row.event.timestamp, row.event.channel_id) != (at, channel) for row in group):
            raise ValueError("group must contain one complete channel timestamp")
        key = at, channel
        if self.watermark is not None and at <= self.watermark:
            raise ValueError("late event at or before the published prediction watermark")
        if self._last_group is not None and key <= self._last_group:
            raise ValueError("closed groups must be unique and sorted by time and channel")
        self._last_group = key
        accepted = [row for row in group if source_is_full_archive(row.source, at)]
        self.ignored_source_rows += len(group) - len(accepted)
        if not accepted:
            return
        self.accepted_rows += len(accepted)
        segment = segment_at(at)
        assert segment is not None
        state = self.channels.get(channel)
        if state is None or state.segment != segment:
            state = _Channel(segment)
            self.channels[channel] = state
        for row in accepted:
            event = row.event
            state.last_observation_at = at
            state.events.append(event)
            state.prefix.add(event, self.config)
            # Only uniqueness/conflict is used; do not retain an unbounded metadata set.
            state.prefix.types = set(sorted(state.prefix.types)[:2])
            state.prefix.linked_objects = set(sorted(state.prefix.linked_objects)[:2])
            if event.sensor_type not in KNOWN_SENSOR_TYPES:
                state.metadata_uncertain = True
            if not self.config.excluded_quality_flags.intersection(event.quality_flags):
                meaning = classify_message(event.sensor_type, event.value_state, event.alarm)
                if meaning.target_message_candidate:
                    state.faults.append(at)
                if meaning.category == "technical_fault":
                    state.technical.append(at)
            if event.value_state is not None:
                # Reuse accepted B2 transitions, but only on events already received.
                state.episodes.add(
                    StateEvent(
                        row.row_id, channel, event.sensor_type, at, event.value_state, event.alarm
                    )
                )
        state.episodes.finish()
        registered = state.episodes.states.get(channel)
        for episode in state.episodes.episodes:
            if (
                episode.end_at is not None
                and episode.onset_status == "candidate_new_onset"
                and episode.end_status == "exact_norma"
                and not episode.uncertain_intervening_state
            ):
                state.completed.append(episode.end_at)
        # Retain only the live episode object; completed history is a rolling queue.
        state.episodes.episodes[:] = (
            [registered.open_episode] if registered and registered.open_episode else []
        )
        if registered and registered.open_episode:
            # This is an inference state, not a full B2 evidence catalog.
            registered.open_episode.evidence.clear()
        if (
            registered
            and registered.last_normal_at == at
            and all(row.event.sensor_type in KNOWN_SENSOR_TYPES for row in accepted)
        ):
            state.metadata_uncertain = False
        self._prune(state, at)

    @staticmethod
    def _prune(state: _Channel, at: datetime) -> None:
        for queue, horizon in (
            (state.events, RETENTION),
            (state.faults, timedelta(hours=168)),
            (state.technical, timedelta(hours=24)),
            (state.completed, timedelta(hours=168)),
        ):
            while (
                queue
                and (queue[0].timestamp if queue is state.events else queue[0]) <= at - horizon
            ):
                queue.popleft()

    def _snapshot(self, channel: str, at: datetime) -> dict:
        state = self.channels.get(channel)
        segment = segment_at(at)
        row = {
            "channel_id": channel,
            "prediction_time": at,
            "sensor_type": None,
            **dict.fromkeys(TERMS, 0),
            "admission_status": "unknown",
            "admission_reasons": [],
            "availability_status": "unknown",
            "last_explicit_normal_at": None,
            "history_through": None,
            "admission_through": None,
            "baseline_fit_end_at": at.replace(hour=0) - timedelta(hours=192),
        }
        if segment is None:
            row.update(admission_status="excluded", admission_reasons=["outside_accepted_archive"])
            return row
        if state is None or state.segment != segment:
            row["admission_reasons"] = ["no_observations_in_current_archive_segment"]
            return row
        self._prune(state, at)
        types = state.prefix.types
        sensor_type = next(iter(types)) if len(types) == 1 else None
        row.update(
            sensor_type=sensor_type,
            history_through=state.last_observation_at,
            admission_through=state.last_observation_at,
            registered_fault_text_count_24h=sum(t > at - timedelta(hours=24) for t in state.faults),
            registered_fault_text_count_168h=len(state.faults),
            technical_message_count_24h=len(state.technical),
            completed_episode_count_168h=len(state.completed),
        )
        day = at.replace(hour=0)
        if state.baseline_day != day:
            state.baseline_day = day
            state.baseline = _baseline_features(
                list(state.events), row["baseline_fit_end_at"], self.config
            )
        recent = [
            event for event in reversed(state.events) if event.timestamp > at - timedelta(hours=24)
        ]
        usable = [
            event
            for event in recent
            if not self.config.excluded_quality_flags.intersection(event.quality_flags)
        ]
        text_by_time: dict[datetime, set[str]] = {}
        for event in usable:
            if event.value_state is not None:
                text_by_time.setdefault(event.timestamp, set()).add(event.value_state)
        ambiguity = any(len(values) > 1 for values in text_by_time.values())
        insufficient = (
            state.prefix.first_usable_at is None
            or at - state.prefix.first_usable_at < self.config.minimum_history
            or state.prefix.unique_usable_count < self.config.minimum_history_events
        )
        assert state.baseline is not None
        past_status, reasons = _branch_status(
            {
                **state.baseline,
                "sensor_type": sensor_type,
                "availability_status": "excluded" if not state.prefix.usable_count else "unknown",
                "availability_reasons": ["insufficient_history"] if insufficient else [],
                "excluded_quality_count_24h": len(recent) - len(usable),
                "state_count_24h": sum(event.value_state is not None for event in usable),
                "state_transitions_24h": 0 if text_by_time else None,
                "window_reasons_24h": ["same_time_state_ambiguity"] if ambiguity else [],
            },
            discrete=True,
        )
        registered = state.episodes.states.get(channel)
        normal_at = registered.last_normal_at if registered else None
        row["last_explicit_normal_at"] = normal_at
        if sensor_type not in KNOWN_SENSOR_TYPES:
            reasons.append("unknown_or_conflicting_sensor_type")
        if registered and registered.open_episode is not None:
            reasons.append("registered_episode_active_at_t")
            past_status = "excluded"
        if registered and registered.uncertain_since_normal:
            reasons.append("uncertain_past_registered_state")
        if state.metadata_uncertain:
            reasons.append("unknown_type_observed_since_explicit_normal")
        if not recent_normal_for_prediction(normal_at, at):
            reasons.append("no_recent_explicit_normal_at_t")
        row["admission_reasons"] = sorted(set(reasons))
        row["admission_status"] = (
            "eligible"
            if not reasons and past_status == "eligible"
            else "excluded"
            if past_status == "excluded"
            else "unknown"
        )
        return row

    def predict(self, at: datetime, channels: Sequence[str]) -> list[dict]:
        """Close an event-time watermark and predict every configured channel."""
        if at.tzinfo is not None or (at.minute, at.second, at.microsecond) != (0, 0, 0):
            raise ValueError("prediction time must be a naive local whole hour")
        if self.watermark is not None and at <= self.watermark:
            raise ValueError("prediction watermarks must be strictly increasing")
        if self._last_group is not None and self._last_group[0] > at:
            raise ValueError("future observations already consumed before prediction")
        if (
            not channels
            or len(set(channels)) != len(channels)
            or any(not channel for channel in channels)
        ):
            raise ValueError("configured channels must be nonempty and unique")
        rows = [self._snapshot(channel, at) for channel in channels]
        frame = pd.DataFrame(rows)
        scores = predict_rule(
            frame, eligibility_status=frame["admission_status"], threshold=self.threshold
        )
        for index, row in enumerate(rows):
            score = scores.loc[index, "rule_score"]
            alert = scores.loc[index, "alert"]
            row.update(
                pilot_version=PILOT_VERSION,
                ruleset_version=RULESET_VERSION,
                rule_score=None if pd.isna(score) else float(score),
                above_frozen_threshold=None if pd.isna(alert) else bool(alert),
                warning_emitted=False,
                warning_status="not_scored",
            )
            if row["rule_score"] is not None:
                state = self.channels[row["channel_id"]]
                if not row["above_frozen_threshold"]:
                    row["warning_status"] = "below_threshold"
                elif state.last_warning_at and at < state.last_warning_at + self.cooldown:
                    row["warning_status"] = "suppressed_cooldown"
                else:
                    state.last_warning_at = at
                    row.update(warning_status="emitted_research_warning", warning_emitted=True)
        self.watermark = at
        return rows

    @property
    def retained_event_count(self) -> int:
        return sum(len(state.events) for state in self.channels.values())
