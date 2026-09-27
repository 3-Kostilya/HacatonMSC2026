"""Research-only past-event recovery reset; frozen R6/Q2 are not modified."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Mapping, Sequence

from stage1.shadow.checkpoint import CheckpointError, decode, digest, encode
from stage1.state_labeling.operational import ARCHIVE_SEGMENTS, segment_at
from stage1.state_labeling.registered_episodes import EpisodeBuilder, RegisteredEpisode
from stage1.state_labeling.registered_episodes import StateEvent, _ChannelState
from stage1.state_labeling.rules import RULESET_VERSION


VERSION = "q2-a-causal-recovery-reset-research-v1"
COOLDOWN = timedelta(hours=24)
DECISION_FIELDS = frozenset(
    {
        "channel_id",
        "sensor_type",
        "admission_status",
        "admission_evidence_through",
        "last_explicit_normal_at",
        "blocking_qa_count_24h",
        "availability_status",
        "above_threshold",
    }
)


@dataclass
class WarningState:
    segment: int
    last_warning_at: datetime | None = None
    warning_type: str | None = None
    recovered_onset_at: datetime | None = None
    recovered_at: datetime | None = None


class RecoveryResetPolicy:
    """Only closed source groups and past admission can influence a warning.

    Past admission is supplied by the separately verified Q2/A causal gate.
    This lightweight policy does not recompute that gate or prove physical uptime.
    No future label, complete B2 catalog, model fitting or threshold API exists.
    """

    def __init__(self, *, allow_reset: bool):
        if type(allow_reset) is not bool:
            raise ValueError("reset mode must be boolean")
        self.allow_reset = allow_reset
        self.builder = EpisodeBuilder()
        self.warnings: dict[str, WarningState] = {}
        self.last_group: tuple[datetime, str] | None = None
        self.watermark: datetime | None = None

    def _warning_state(self, channel: str, at: datetime) -> WarningState:
        segment = segment_at(at)
        if segment is None:
            raise ValueError("outside accepted archive segments")
        state = self.warnings.get(channel)
        if state is None or state.segment != segment:
            state = WarningState(segment)
            self.warnings[channel] = state
            self.builder.states.pop(channel, None)
        return state

    @staticmethod
    def _clear_recovery(state: WarningState) -> None:
        state.recovered_onset_at = state.recovered_at = None

    def observe_group(self, events: Sequence[StateEvent]) -> None:
        if not events:
            raise ValueError("a complete channel-second group is required")
        at, channel = events[0].at, events[0].channel_id
        if (
            any((event.at, event.channel_id) != (at, channel) for event in events)
            or (self.watermark is not None and at <= self.watermark)
            or (self.last_group is not None and (at, channel) <= self.last_group)
        ):
            raise ValueError("mixed, duplicate, late or unordered closed group")
        warning = self._warning_state(channel, at)
        previous = self.builder.states.get(channel)
        live = previous.open_episode if previous else None
        for event in events:
            self.builder.add(event)
        self.builder.finish()
        registered = self.builder.states[channel]
        if (
            registered.uncertain_since_normal
            or registered.sensor_type != warning.warning_type
            or (live is not None and live.uncertain_intervening_state)
        ):
            self._clear_recovery(warning)
        if (
            live is not None
            and live.end_at == at
            and live.end_status == "exact_norma"
            and live.onset_status == "candidate_new_onset"
            and not live.uncertain_intervening_state
            and warning.last_warning_at is not None
            and warning.warning_type == live.sensor_type == registered.sensor_type
            and warning.last_warning_at < live.start_at <= warning.last_warning_at + COOLDOWN
            and live.start_at < at < warning.last_warning_at + COOLDOWN
            and segment_at(live.start_at) == warning.segment
        ):
            warning.recovered_onset_at, warning.recovered_at = live.start_at, at
        # State is causal and bounded; never retain a completed full-history catalog.
        if registered.open_episode is not None:
            registered.open_episode.evidence.clear()
        self.builder.episodes.clear()
        self.last_group = at, channel

    def decide(self, at: datetime, decisions: Sequence[Mapping]) -> list[dict]:
        if (
            at.tzinfo is not None
            or (at.minute, at.second, at.microsecond) != (0, 0, 0)
            or (self.watermark is not None and at <= self.watermark)
            or (self.last_group is not None and self.last_group[0] > at)
        ):
            raise ValueError("decision requires a new closed hour with no future observations")
        if not decisions or len({row["channel_id"] for row in decisions}) != len(decisions):
            raise ValueError("decision channels must be nonempty and unique")
        result = []
        for row in decisions:
            if set(row) != DECISION_FIELDS:
                raise ValueError(
                    "only declared past admission fields are accepted; labels forbidden"
                )
            if (
                row["admission_status"] not in {"eligible", "unknown", "excluded"}
                or row["availability_status"] != "unknown"
                or type(row["above_threshold"]) is not bool
                or type(row["blocking_qa_count_24h"]) is not int
                or row["blocking_qa_count_24h"] < 0
            ):
                raise ValueError("invalid past admission/score decision")
            for name in ("admission_evidence_through", "last_explicit_normal_at"):
                value = row[name]
                if value is not None and (value.tzinfo is not None or value > at):
                    raise ValueError("future admission evidence is forbidden")
            eligible = row["admission_status"] == "eligible"
            if eligible and (
                row["admission_evidence_through"] is None
                or row["last_explicit_normal_at"] is None
                or row["blocking_qa_count_24h"]
            ):
                raise ValueError("eligible admission lacks past normal/evidence or has blocking QA")
            channel = row["channel_id"]
            state = self._warning_state(channel, at)
            registered = self.builder.states.get(channel)
            healthy = (
                registered is not None
                and registered.segment == state.segment
                and registered.sensor_type == row["sensor_type"]
                and registered.open_episode is None
                and not registered.uncertain_since_normal
                and registered.last_normal_at == row["last_explicit_normal_at"]
            )
            in_cooldown = (
                state.last_warning_at is not None and at < state.last_warning_at + COOLDOWN
            )
            reset = bool(
                self.allow_reset
                and in_cooldown
                and eligible
                and healthy
                and state.warning_type == row["sensor_type"]
                and state.recovered_at is not None
                and state.recovered_at <= at
            )
            emitted = eligible and row["above_threshold"] and (not in_cooldown or reset)
            previous_warning = state.last_warning_at
            record = {
                "channel_id": channel,
                "prediction_time": at,
                "sensor_type": row["sensor_type"],
                "warning_emitted": emitted,
                "reason": "not_eligible"
                if not eligible
                else "below_threshold"
                if not row["above_threshold"]
                else "recovered_episode_reset"
                if emitted and reset
                else "standard_24h_warning"
                if emitted
                else "suppressed_24h",
                "previous_warning_at": previous_warning,
                "observed_onset_at": state.recovered_onset_at if emitted and reset else None,
                "observed_recovery_at": state.recovered_at if emitted and reset else None,
                "admission_evidence_through": row["admission_evidence_through"],
                "last_explicit_normal_at": row["last_explicit_normal_at"],
                "past_state_agrees_with_admission": bool(healthy),
            }
            if emitted:
                state.last_warning_at, state.warning_type = at, row["sensor_type"]
                self._clear_recovery(state)
            result.append(record)
        self.watermark = at
        return result

    def checkpoint(self) -> dict:
        if self.watermark is None or (self.last_group and self.last_group[0] > self.watermark):
            raise CheckpointError("checkpoint needs a closed decision hour")
        payload = encode(
            {
                "version": VERSION,
                "ruleset": RULESET_VERSION,
                "allow_reset": self.allow_reset,
                "watermark": self.watermark,
                "last_group": self.last_group,
                "warning_states": {
                    cid: asdict(state) for cid, state in sorted(self.warnings.items())
                },
                "registered_states": {
                    cid: asdict(state) for cid, state in sorted(self.builder.states.items())
                },
                "message_counts": self.builder.message_counts,
            }
        )
        return {"payload": payload, "sha256": digest(payload)}

    @classmethod
    def restore(cls, envelope: Mapping) -> RecoveryResetPolicy:
        if (
            set(envelope) != {"payload", "sha256"}
            or digest(envelope["payload"]) != envelope["sha256"]
        ):
            raise CheckpointError("checkpoint content hash differs")
        data = decode(envelope["payload"])
        expected = {
            "version",
            "ruleset",
            "allow_reset",
            "watermark",
            "last_group",
            "warning_states",
            "registered_states",
            "message_counts",
        }
        if (
            set(data) != expected
            or data["version"] != VERSION
            or data["ruleset"] != RULESET_VERSION
        ):
            raise CheckpointError("checkpoint policy version differs")
        stream = cls(allow_reset=data["allow_reset"])
        watermark = data["watermark"]
        if (
            not isinstance(watermark, datetime)
            or watermark.tzinfo is not None
            or (watermark.minute, watermark.second, watermark.microsecond) != (0, 0, 0)
        ):
            raise CheckpointError("invalid checkpoint watermark")

        def past(value):
            if value is not None and (
                not isinstance(value, datetime) or value.tzinfo is not None or value > watermark
            ):
                raise CheckpointError("checkpoint contains future or invalid evidence")

        last = data["last_group"]
        if last is not None:
            if not isinstance(last, tuple) or len(last) != 2 or not isinstance(last[1], str):
                raise CheckpointError("invalid source cursor")
            past(last[0])
        stream.watermark, stream.last_group = watermark, last
        stream.builder._last_key = last
        if set(data["message_counts"]) != set(stream.builder.message_counts) or any(
            type(v) is not int or v < 0 for v in data["message_counts"].values()
        ):
            raise CheckpointError("invalid checkpoint counters")
        stream.builder.message_counts = data["message_counts"]
        for cid, raw in data["warning_states"].items():
            state = WarningState(**raw)
            if (
                not cid
                or type(state.segment) is not int
                or not 0 <= state.segment < len(ARCHIVE_SEGMENTS)
            ):
                raise CheckpointError("invalid warning channel/segment")
            for value in (state.last_warning_at, state.recovered_at, state.recovered_onset_at):
                past(value)
                if value is not None and segment_at(value) != state.segment:
                    raise CheckpointError("warning state crosses excluded archive gap")
            if (state.recovered_at is None) != (state.recovered_onset_at is None) or (
                state.recovered_at is not None
                and (
                    state.last_warning_at is None
                    or not state.last_warning_at
                    < state.recovered_onset_at
                    < state.recovered_at
                    < state.last_warning_at + COOLDOWN
                )
            ):
                raise CheckpointError("invalid recovery token chronology")
            stream.warnings[cid] = state
        for cid, raw in data["registered_states"].items():
            values = dict(raw)
            episode = values.pop("open_episode")
            state = _ChannelState(**values)
            if cid not in stream.warnings or state.segment != stream.warnings[cid].segment:
                raise CheckpointError("registered state channel/segment differs")
            past(state.last_normal_at)
            if (
                state.last_normal_at is not None
                and segment_at(state.last_normal_at) != state.segment
            ):
                raise CheckpointError("normal history crosses archive gap")
            if episode is not None:
                state.open_episode = RegisteredEpisode(**episode)
                for value in (
                    state.open_episode.start_at,
                    state.open_episode.confirmed_at,
                    state.open_episode.prior_normal_at,
                    state.open_episode.last_fault_at,
                ):
                    past(value)
                if (
                    state.open_episode.channel_id != cid
                    or state.open_episode.end_at is not None
                    or state.open_episode.start_at != state.open_episode.confirmed_at
                    or segment_at(state.open_episode.start_at) != state.segment
                ):
                    raise CheckpointError("invalid open episode in checkpoint")
            stream.builder.states[cid] = state
        return stream
