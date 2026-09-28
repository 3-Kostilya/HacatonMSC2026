"""Leakage-safe handoff from detected episodes to a future forecasting target."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from stage1.contracts import Decision, Episode


@dataclass(frozen=True, slots=True)
class ForecastLabel:
    channel_id: str
    prediction_at: datetime
    horizon_end: datetime
    value: int
    reason: str
    next_episode_start: datetime | None = None


def _overlaps_future_window(
    interval: tuple[datetime, datetime], start: datetime, end: datetime
) -> bool:
    left, right = interval
    return left < end and right > start


def label_future_onsets(
    channel_id: str,
    prediction_times: Iterable[datetime],
    episodes: Iterable[Episode],
    *,
    observed_until: datetime,
    unknown_intervals: Iterable[tuple[datetime, datetime]] = (),
    horizon: timedelta = timedelta(hours=24),
) -> list[ForecastLabel]:
    """Label new candidate onsets; never turn an ongoing episode into a success."""
    if horizon <= timedelta(0):
        raise ValueError("horizon must be positive")
    relevant = sorted(
        (
            episode
            for episode in episodes
            if episode.channel_id == channel_id and episode.decision is Decision.CANDIDATE
        ),
        key=lambda episode: episode.start_at,
    )
    unknown = tuple(unknown_intervals)
    labels = []
    for prediction_at in prediction_times:
        horizon_end = prediction_at + horizon
        if horizon_end > observed_until:
            labels.append(
                ForecastLabel(
                    channel_id,
                    prediction_at,
                    horizon_end,
                    -1,
                    "future_window_not_observed",
                )
            )
            continue
        if any(
            _overlaps_future_window(interval, prediction_at, horizon_end) for interval in unknown
        ):
            labels.append(
                ForecastLabel(
                    channel_id,
                    prediction_at,
                    horizon_end,
                    -1,
                    "future_window_unknown",
                )
            )
            continue
        ongoing = any(
            episode.start_at <= prediction_at
            and (episode.end_at is None or episode.end_at > prediction_at)
            for episode in relevant
        )
        if ongoing:
            labels.append(
                ForecastLabel(
                    channel_id,
                    prediction_at,
                    horizon_end,
                    -1,
                    "episode_already_ongoing",
                )
            )
            continue
        future = next(
            (episode for episode in relevant if prediction_at < episode.start_at <= horizon_end),
            None,
        )
        if future is not None and future.confirmed_at > observed_until:
            labels.append(
                ForecastLabel(
                    channel_id,
                    prediction_at,
                    horizon_end,
                    -1,
                    "episode_confirmation_not_observed",
                    future.start_at,
                )
            )
            continue
        labels.append(
            ForecastLabel(
                channel_id,
                prediction_at,
                horizon_end,
                1 if future is not None else 0,
                "new_episode_onset" if future is not None else "observed_no_new_onset",
                future.start_at if future is not None else None,
            )
        )
    return labels
