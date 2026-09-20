"""Aggregate confirmed shared context without multiplying local diagnoses."""

from __future__ import annotations

from datetime import timedelta
import hashlib
from typing import Iterable

from stage1.contracts import Decision, Episode, NormalizedEvent, Origin
from stage1.detectors import RULESET_VERSION


def detect_shared_state(
    events: Iterable[NormalizedEvent],
    *,
    state: str,
    minimum_channels: int = 2,
    coincidence_window: timedelta = timedelta(minutes=2),
) -> Episode:
    ordered = sorted(events, key=lambda event: (event.timestamp, event.channel_id))
    if not ordered:
        raise ValueError("events must not be empty")
    if minimum_channels < 2 or coincidence_window <= timedelta(0):
        raise ValueError("context thresholds must be positive and use at least two channels")
    object_ids = {event.object_id for event in ordered if event.object_id}
    synthetic = all(event.source.startswith("synthetic:") for event in ordered)
    origin = Origin.SYNTHETIC if synthetic else Origin.OBSERVED
    if len(object_ids) != 1 or any(not event.object_id for event in ordered):
        first = ordered[0]
        return Episode(
            episode_id="ctx-unknown-" + hashlib.sha1(first.channel_id.encode()).hexdigest()[:12],
            channel_id=f"context:{first.channel_id}",
            sensor_type="shared_context",
            sensor_group="context",
            anomaly_type="shared_state",
            decision=Decision.UNKNOWN,
            start_at=first.timestamp,
            confirmed_at=ordered[-1].timestamp,
            ruleset_version=RULESET_VERSION,
            evidence=(),
            observation_quality=("unconfirmed_context_link",),
            origin=origin,
        )
    object_id = next(iter(object_ids))
    matching = [event for event in ordered if event.raw_value == state]
    for index, current in enumerate(matching):
        start = current.timestamp - coincidence_window
        window = [event for event in matching[: index + 1] if event.timestamp >= start]
        by_channel = {}
        for event in window:
            by_channel[event.channel_id] = event
        if len(by_channel) >= minimum_channels:
            selected = sorted(by_channel.values(), key=lambda event: event.timestamp)
            start_at = selected[0].timestamp
            confirmed_at = selected[-1].timestamp
            key = f"{object_id}|{state}|{start_at.isoformat()}".encode()
            return Episode(
                episode_id="ctx-" + hashlib.sha1(key).hexdigest()[:16],
                channel_id=f"context:{object_id}",
                sensor_type="shared_context",
                sensor_group="context",
                anomaly_type="shared_state",
                decision=Decision.CANDIDATE,
                start_at=start_at,
                confirmed_at=confirmed_at,
                ruleset_version=RULESET_VERSION,
                evidence=(
                    f"state={state}",
                    f"distinct_channels={len(by_channel)}",
                    "channels=" + ",".join(sorted(by_channel)),
                ),
                observation_quality=(),
                cause_hypothesis="shared_context_hypothesis",
                object_id=object_id,
                origin=origin,
            )
    first = ordered[0]
    return Episode(
        episode_id="ctx-none-" + hashlib.sha1(object_id.encode()).hexdigest()[:12],
        channel_id=f"context:{object_id}",
        sensor_type="shared_context",
        sensor_group="context",
        anomaly_type="shared_state",
        decision=Decision.NO_CANDIDATE,
        start_at=first.timestamp,
        confirmed_at=ordered[-1].timestamp,
        ruleset_version=RULESET_VERSION,
        evidence=(f"distinct_channels_below_{minimum_channels}",),
        observation_quality=(),
        cause_hypothesis="shared_context_hypothesis",
        object_id=object_id,
        origin=origin,
    )
