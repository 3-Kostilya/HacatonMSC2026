"""Causal counts of QA value categories for a separate feature sidecar."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable

from stage1.features.hourly import FeatureEvent
from stage1.features.schema import WINDOW_HOURS
from stage1.value_quality import QA_VALUE_RULESET_VERSION


QA_CATEGORIES = (
    "temperature_service_code_candidate",
    "epoch_value_artifact",
    "gas_negative_reading",
    "gas_above_physical_percent",
    "gas_alarm_level_candidate",
)


def qa_window_counts(
    events: Iterable[FeatureEvent],
    prediction_time: datetime,
) -> dict[str, int | str]:
    """Count only already-observed events in left-open, right-closed windows."""

    if prediction_time.tzinfo is not None:
        raise ValueError("prediction_time must use local naive journal time")
    observed = list(events)
    result: dict[str, int | str] = {"qa_value_ruleset_version": QA_VALUE_RULESET_VERSION}
    for hours in WINDOW_HOURS:
        lower = prediction_time - timedelta(hours=hours)
        counts = Counter(
            event.qa_value_category
            for event in observed
            if lower < event.timestamp <= prediction_time
        )
        for category in QA_CATEGORIES:
            result[f"qa_{category}_count_{hours}h"] = counts[category]
    return result
