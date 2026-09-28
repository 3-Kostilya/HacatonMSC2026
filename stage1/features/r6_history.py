"""Past-only inputs of the frozen R6 journal rule, with no target or admission.

This bounded-window replay accepts already normalized events and the verified
unambiguous-completed B2 adapter. The caller still owns conditional admission;
empty history must never be used to assert that a channel is available.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Iterable, Iterator

from stage1.features.hourly import FeatureEvent, HourlyConfig
from stage1.features.r2 import CompletedEpisode
from stage1.state_labeling.operational import segment_at
from stage1.state_labeling.rules import classify_message


HISTORY_VERSION = "r6-a-causal-rule-history-v1"


def iter_rule_history(
    events: Iterable[FeatureEvent],
    completed_episodes: Iterable[CompletedEpisode],
    channel_id: str,
    prediction_times: Iterable[datetime],
) -> Iterator[dict]:
    """Advance clocks in order, adding only observations/ends already known.

    Windows are (t-H,t]. Future events and unfinished episodes cannot enter a
    snapshot. Lists may contain future data for offline replay, but the cursors
    do not consume it until its observation time. Input retention belongs to
    the caller; the rolling count queues themselves retain at most 168 hours.
    """
    ordered = sorted(
        (event for event in events if event.channel_id == channel_id),
        key=lambda event: event.timestamp,
    )
    episodes = sorted(
        (episode for episode in completed_episodes if episode.channel_id == channel_id),
        key=lambda episode: episode.end_at,
    )
    for episode in episodes:
        if segment_at(episode.start_at) is None or (
            segment_at(episode.start_at) != segment_at(episode.end_at)
        ):
            raise ValueError("completed episode crosses an excluded archive boundary")
    faults: deque[datetime] = deque()
    technical: deque[datetime] = deque()
    ends: deque[datetime] = deque()
    event_cursor = episode_cursor = 0
    previous: datetime | None = None
    excluded = HourlyConfig().excluded_quality_flags
    for at in prediction_times:
        segment = segment_at(at)
        if segment is None or (at.minute, at.second, at.microsecond) != (0, 0, 0):
            raise ValueError("prediction must be a whole local hour in an accepted archive")
        if previous is not None and at <= previous:
            raise ValueError("prediction times must be strictly increasing")
        previous = at
        while event_cursor < len(ordered) and ordered[event_cursor].timestamp <= at:
            event = ordered[event_cursor]
            event_cursor += 1
            if segment_at(event.timestamp) != segment or excluded.intersection(event.quality_flags):
                continue
            meaning = classify_message(event.sensor_type, event.value_state, event.alarm)
            if meaning.target_message_candidate:
                faults.append(event.timestamp)
            if meaning.category == "technical_fault":
                technical.append(event.timestamp)
        while episode_cursor < len(episodes) and episodes[episode_cursor].end_at <= at:
            episode = episodes[episode_cursor]
            episode_cursor += 1
            if segment_at(episode.end_at) == segment:
                ends.append(episode.end_at)
        lower24, lower168 = at - timedelta(hours=24), at - timedelta(hours=168)
        for queue, lower in ((faults, lower168), (technical, lower24), (ends, lower168)):
            while queue and queue[0] <= lower:
                queue.popleft()
        yield {
            "channel_id": channel_id,
            "prediction_time": at,
            "registered_fault_text_count_24h": sum(time > lower24 for time in faults),
            "registered_fault_text_count_168h": len(faults),
            "technical_message_count_24h": len(technical),
            "completed_episode_count_168h": len(ends),
        }
