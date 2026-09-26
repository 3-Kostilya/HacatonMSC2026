"""Versioned JSON state at a closed prediction hour; never deserialize executable objects."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import json
from typing import Mapping

from stage1.features.hourly import FeatureEvent, HourlyConfig, _PrefixState
from stage1.shadow.stream import PILOT_VERSION, ShadowStream, _Channel
from stage1.state_labeling.operational import ARCHIVE_SEGMENTS
from stage1.state_labeling.registered_episodes import RegisteredEpisode, _ChannelState
from stage1.state_labeling.rules import RULESET_VERSION


CHECKPOINT_VERSION = "shadow-a-closed-hour-state-v1"


class CheckpointError(ValueError):
    """Fail closed; a failed restore must never silently create an empty history."""


def encode(value):
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            raise CheckpointError("checkpoint requires local naive timestamps")
        return {"$datetime": value.isoformat()}
    if isinstance(value, timedelta):
        return {"$seconds": value.total_seconds()}
    if isinstance(value, (set, frozenset)):
        return {"$set": sorted(value)}
    if isinstance(value, tuple):
        return {"$tuple": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    return value


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if any(key.startswith("$") for key in value):
        if len(value) != 1:
            raise CheckpointError("ambiguous checkpoint value")
        if "$datetime" in value:
            at = datetime.fromisoformat(value["$datetime"])
            if at.tzinfo is not None:
                raise CheckpointError("checkpoint requires local naive timestamps")
            return at
        if "$seconds" in value:
            return timedelta(seconds=value["$seconds"])
        if "$set" in value:
            return set(value["$set"])
        if "$tuple" in value:
            return tuple(decode(item) for item in value["$tuple"])
        raise CheckpointError("unknown checkpoint value tag")
    return {key: decode(item) for key, item in value.items()}


def canonical(value) -> bytes:
    return json.dumps(
        encode(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def parse_json(data: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CheckpointError("duplicate checkpoint JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise CheckpointError(f"nonfinite checkpoint number: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid_constant)


def snapshot(stream: ShadowStream) -> dict:
    """Only a complete prediction hour is a resumable publication boundary."""
    at = stream.watermark
    if at is None or (stream._last_group and stream._last_group[0] > at):
        raise CheckpointError("checkpoint requires a closed prediction hour")
    channels = {}
    for channel, state in sorted(stream.channels.items()):
        if state.episodes._group:
            raise CheckpointError("cannot checkpoint a partial channel-second")
        registered = state.episodes.states.get(channel)
        if set(state.episodes.states) - {channel}:
            raise CheckpointError("episode state contains another channel")
        channels[channel] = {
            "segment": state.segment,
            "events": [asdict(item) for item in state.events],
            "prefix": asdict(state.prefix),
            "registered": asdict(registered) if registered else None,
            "episode_counts": dict(state.episodes.message_counts),
            "episode_last_key": state.episodes._last_key,
            "faults": list(state.faults),
            "technical": list(state.technical),
            "completed": list(state.completed),
            "baseline_day": state.baseline_day,
            "baseline": state.baseline,
            "metadata_uncertain": state.metadata_uncertain,
            "last_warning_at": state.last_warning_at,
            "last_observation_at": state.last_observation_at,
        }
    return encode(
        {
            "schema_version": CHECKPOINT_VERSION,
            "pilot_version": PILOT_VERSION,
            "ruleset_version": RULESET_VERSION,
            "config": asdict(stream.config),
            "threshold": stream.threshold,
            "cooldown": stream.cooldown,
            "watermark": at,
            "last_group": stream._last_group,
            "accepted_rows": stream.accepted_rows,
            "ignored_source_rows": stream.ignored_source_rows,
            "channels": channels,
        }
    )


def _count(value) -> bool:
    return type(value) is int and value >= 0


def _past(at, watermark, *, optional=True):
    if at is None and optional:
        return
    if not isinstance(at, datetime) or at.tzinfo is not None or at > watermark:
        raise CheckpointError("checkpoint contains invalid or future state time")


def restore(payload: Mapping) -> ShadowStream:
    """Validate the whole state before making a resumed stream available."""
    try:
        data = decode(dict(payload))
        required = {
            "schema_version",
            "pilot_version",
            "ruleset_version",
            "config",
            "threshold",
            "cooldown",
            "watermark",
            "last_group",
            "accepted_rows",
            "ignored_source_rows",
            "channels",
        }
        if (
            set(data) != required
            or data["schema_version"] != CHECKPOINT_VERSION
            or data["pilot_version"] != PILOT_VERSION
            or data["ruleset_version"] != RULESET_VERSION
            or data["config"] != asdict(HourlyConfig())
            or data["threshold"] != 7.1
            or data["cooldown"] != timedelta(hours=24)
        ):
            raise CheckpointError("checkpoint version or admission policy differs")
        watermark = data["watermark"]
        _past(watermark, watermark, optional=False)
        if (watermark.minute, watermark.second, watermark.microsecond) != (0, 0, 0):
            raise CheckpointError("checkpoint watermark must be a whole hour")
        for name in ("accepted_rows", "ignored_source_rows"):
            if not _count(data[name]):
                raise CheckpointError("invalid checkpoint observation count")
        last_group = data["last_group"]
        if last_group is not None:
            _past(last_group[0], watermark, optional=False)
            if len(last_group) != 2 or not isinstance(last_group[1], str) or not last_group[1]:
                raise CheckpointError("invalid source group cursor")
        stream = ShadowStream(threshold=7.1)
        stream.watermark, stream._last_group = watermark, last_group
        stream.accepted_rows, stream.ignored_source_rows = (
            data["accepted_rows"],
            data["ignored_source_rows"],
        )
        for channel, saved in data["channels"].items():
            if not isinstance(channel, str) or not channel:
                raise CheckpointError("invalid checkpoint channel")
            state = _Channel(saved["segment"])
            if type(state.segment) is not int or not 0 <= state.segment < len(ARCHIVE_SEGMENTS):
                raise CheckpointError("invalid archive segment")
            start, end = ARCHIVE_SEGMENTS[state.segment]
            state.events = deque(FeatureEvent(**item) for item in saved["events"])
            state.prefix = _PrefixState(**saved["prefix"])
            prefix = state.prefix
            if (
                not all(
                    _count(getattr(prefix, name))
                    for name in ("past_count", "usable_count", "unique_usable_count")
                )
                or not 0 <= prefix.unique_usable_count <= prefix.usable_count <= prefix.past_count
                or prefix.past_count < len(state.events)
                or not all(
                    isinstance(values, set)
                    and len(values) <= 2
                    and all(isinstance(item, str) for item in values)
                    for values in (prefix.types, prefix.linked_objects)
                )
            ):
                raise CheckpointError("invalid accumulated checkpoint metadata")
            for at in (prefix.first_usable_at, prefix.last_usable_at):
                _past(at, watermark)
            if (
                (prefix.usable_count == 0) != (prefix.first_usable_at is None)
                or (prefix.first_usable_at is None) != (prefix.last_usable_at is None)
                or (
                    prefix.first_usable_at is not None
                    and prefix.first_usable_at > prefix.last_usable_at
                )
            ):
                raise CheckpointError("invalid usable history chronology")
            for name in ("faults", "technical", "completed"):
                setattr(state, name, deque(saved[name]))
            for queue in (state.events, state.faults, state.technical, state.completed):
                times = [
                    item.timestamp if isinstance(item, FeatureEvent) else item for item in queue
                ]
                for at in times:
                    _past(at, watermark, optional=False)
                    if not start <= at < end:
                        raise CheckpointError("checkpoint history crosses archive segment")
                if times != sorted(times) or any(
                    item.channel_id != channel for item in state.events
                ):
                    raise CheckpointError("checkpoint history order or channel differs")
            counts = saved["episode_counts"]
            if set(counts) != set(state.episodes.message_counts) or not all(
                _count(value) for value in counts.values()
            ):
                raise CheckpointError("invalid episode counters")
            state.episodes.message_counts = counts
            state.episodes._last_key = saved["episode_last_key"]
            if state.episodes._last_key is not None:
                key = state.episodes._last_key
                _past(key[0], watermark, optional=False)
                if len(key) != 2 or key[1] != channel:
                    raise CheckpointError("invalid episode group cursor")
            registered = saved["registered"]
            if registered is not None:
                episode = registered["open_episode"]
                if episode is not None:
                    episode = RegisteredEpisode(**episode)
                    if episode.channel_id != channel or episode.end_at is not None:
                        raise CheckpointError("invalid open episode identity")
                    for at in (episode.start_at, episode.confirmed_at, episode.last_fault_at):
                        _past(at, watermark, optional=False)
                    _past(episode.prior_normal_at, watermark)
                    if (
                        not start <= episode.start_at <= episode.last_fault_at < end
                        or episode.confirmed_at != episode.start_at
                        or not _count(episode.fault_message_count)
                        or episode.fault_message_count == 0
                        or episode.ruleset_version != RULESET_VERSION
                    ):
                        raise CheckpointError("invalid open episode chronology")
                registered = _ChannelState(**{**registered, "open_episode": episode})
                if registered.segment != state.segment:
                    raise CheckpointError("episode segment differs")
                _past(registered.last_normal_at, watermark)
                if episode is not None and registered.last_normal_at is not None:
                    raise CheckpointError("active fault cannot inherit normal state")
                state.episodes.states[channel] = registered
                state.episodes.episodes = [episode] if episode is not None else []
            for name in (
                "baseline_day",
                "baseline",
                "metadata_uncertain",
                "last_warning_at",
                "last_observation_at",
            ):
                setattr(state, name, saved[name])
            if type(state.metadata_uncertain) is not bool:
                raise CheckpointError("invalid metadata uncertainty")
            for at in (state.baseline_day, state.last_warning_at, state.last_observation_at):
                _past(at, watermark)
            if state.baseline is not None:
                _past(state.baseline["baseline_fit_end_at"], watermark, optional=False)
            stream.channels[channel] = state
        if (
            sum(state.prefix.past_count for state in stream.channels.values())
            > stream.accepted_rows
        ):
            raise CheckpointError("checkpoint observation counts differ")
        if snapshot(stream) != payload:
            raise CheckpointError("checkpoint state fields or canonical representation differ")
        return stream
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"invalid checkpoint state: {error}") from error
