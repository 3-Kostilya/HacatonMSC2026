"""Deterministic consolidation of detector outputs into operational episodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from stage1.contracts import Decision, Episode


@dataclass(frozen=True, slots=True)
class EpisodeAggregationConfig:
    merge_gap: timedelta = timedelta(minutes=15)
    suppression_window: timedelta = timedelta(minutes=10)

    def __post_init__(self) -> None:
        if self.merge_gap < timedelta(0) or self.suppression_window < timedelta(0):
            raise ValueError("episode windows cannot be negative")


def _cause_key(episode: Episode) -> tuple[str, str, str, str]:
    return (
        episode.channel_id,
        episode.sensor_type,
        episode.anomaly_type,
        episode.cause_hypothesis,
    )


def _merged_end_at(previous: Episode, current: Episode) -> datetime | None:
    """Combine known ends without inventing a recovery for an open component."""

    if previous.end_at is None or current.end_at is None:
        # An open component means that the merged episode's end is unknown.
        return None
    return max(previous.end_at, current.end_at)


def _last_confirmed_at(previous: Episode, current: Episode) -> datetime:
    """Keep the greatest confirmation timestamp from both input components."""

    values = [previous.confirmed_at, current.confirmed_at]
    for component in (previous, current):
        recorded = component.metadata.get("last_confirmed_at")
        if isinstance(recorded, datetime):
            if recorded.tzinfo is None:
                values.append(recorded)
        elif isinstance(recorded, str):
            try:
                parsed = datetime.fromisoformat(recorded)
            except ValueError:
                parsed = None
            if parsed is not None and parsed.tzinfo is None:
                values.append(parsed)
    return max(values)


def consolidate_episodes(
    detections: Iterable[Episode],
    *,
    recovery_by_channel: Mapping[str, datetime] | None = None,
    config: EpisodeAggregationConfig | None = None,
) -> list[Episode]:
    """Merge close detections, close them at recovery, then suppress cooldown noise.

    Different channels, anomaly types, sensor types, or cause hypotheses always
    remain separate.  UNKNOWN and NO_CANDIDATE decisions pass through unchanged.
    """

    cfg = config or EpisodeAggregationConfig()
    recoveries = recovery_by_channel or {}
    passthrough: list[Episode] = []
    candidates: list[Episode] = []
    for episode in detections:
        if episode.decision is Decision.CANDIDATE:
            candidates.append(episode)
        else:
            passthrough.append(episode)

    candidates.sort(key=lambda episode: (_cause_key(episode), episode.start_at))
    merged: list[Episode] = []
    for episode in candidates:
        if merged and _cause_key(merged[-1]) == _cause_key(episode):
            previous = merged[-1]
            previous_edge = previous.end_at or previous.confirmed_at
            if episode.start_at - previous_edge <= cfg.merge_gap:
                evidence = tuple(dict.fromkeys(previous.evidence + episode.evidence))
                quality = tuple(
                    dict.fromkeys(previous.observation_quality + episode.observation_quality)
                )
                metadata = dict(previous.metadata)
                metadata["merged_detection_count"] = int(
                    metadata.get("merged_detection_count", 1)
                ) + int(episode.metadata.get("merged_detection_count", 1))
                metadata["last_confirmed_at"] = _last_confirmed_at(previous, episode).isoformat(
                    sep=" "
                )
                merged[-1] = replace(
                    previous,
                    confirmed_at=min(previous.confirmed_at, episode.confirmed_at),
                    # Any open component keeps the merged episode's end unknown;
                    # a known end cannot establish recovery for another component.
                    end_at=_merged_end_at(previous, episode),
                    evidence=evidence,
                    observation_quality=quality,
                    score=max(
                        (value for value in (previous.score, episode.score) if value is not None),
                        default=None,
                    ),
                    metadata=metadata,
                )
                continue
        merged.append(episode)

    closed: list[Episode] = []
    for episode in merged:
        recovery = recoveries.get(episode.channel_id)
        if recovery is not None and recovery >= episode.confirmed_at:
            episode = replace(
                episode,
                end_at=recovery,
                evidence=tuple(dict.fromkeys(episode.evidence + ("recovery_observed",))),
            )
        closed.append(episode)

    emitted: list[Episode] = []
    last_by_key: dict[tuple[str, str, str, str], Episode] = {}
    for episode in sorted(closed, key=lambda item: item.start_at):
        key = _cause_key(episode)
        previous = last_by_key.get(key)
        if previous is not None and previous.end_at is not None:
            if episode.start_at <= previous.end_at + cfg.suppression_window:
                continue
        emitted.append(episode)
        last_by_key[key] = episode

    return sorted(passthrough + emitted, key=lambda episode: episode.start_at)
