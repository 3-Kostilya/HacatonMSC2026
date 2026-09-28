"""Sparse causal hourly calculations for the full R3 A prediction grid."""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timedelta
from typing import Iterable

from stage1.features.hourly import (
    FeatureEvent,
    HourlyConfig,
    _PrefixState,
    _baseline_features,
    _feature_at_sorted,
)
from stage1.features.schema import WINDOW_HOURS


def build_selected_hourly_rows(
    events: Iterable[FeatureEvent],
    channel_id: str,
    prediction_times: Iterable[datetime],
    *,
    config: HourlyConfig | None = None,
) -> list[dict]:
    """Calculate only requested past-selected hours, with a daily frozen baseline.

    A baseline for each local day ends before that day's entire 168-hour window
    and embargo. Event history at or before each t is accumulated once.
    """

    config = config or HourlyConfig()
    ordered = sorted((event for event in events if event.channel_id == channel_id),
                     key=lambda event: event.timestamp)
    times = [event.timestamp for event in ordered]
    requested = list(prediction_times)
    if requested != sorted(set(requested)):
        raise ValueError("prediction times must be unique and sorted")
    if any(
        at.tzinfo is not None or (at.minute, at.second, at.microsecond) != (0, 0, 0)
        for at in requested
    ):
        raise ValueError("prediction times must be naive local whole hours")
    prefix = _PrefixState()
    cursor = 0
    current_day: datetime | None = None
    baseline = None
    fit_end = None
    rows = []
    for at in requested:
        while cursor < len(ordered) and ordered[cursor].timestamp <= at:
            prefix.add(ordered[cursor], config)
            cursor += 1
        day = at.replace(hour=0)
        if day != current_day:
            current_day = day
            fit_end = day - timedelta(hours=max(WINDOW_HOURS)) - config.baseline_embargo
            lower = bisect_left(times, fit_end - config.baseline_lookback)
            upper = bisect_left(times, fit_end)
            baseline = _baseline_features(ordered[lower:upper], fit_end, config)
        assert fit_end is not None and baseline is not None
        rows.append(_feature_at_sorted(
            ordered, times, channel_id, at, config, fit_end,
            prefix=prefix, baseline=baseline,
        ))
    return rows
