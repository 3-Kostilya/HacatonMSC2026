from __future__ import annotations

import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from app.config import PROJECT_ROOT
from app.storage import ParquetStore

# Backend is started from backend/, while the frozen ML and Stage-1 packages live
# at the repository root. Make those packages importable without requiring the
# user to modify PYTHONPATH manually.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.experimental_round7 import Round7ResearchModels
from stage1.features.hourly import FeatureEvent, feature_at
from stage1.features.qa_values import qa_window_counts
from stage1.state_labeling.operational import registered_state_effect
from stage1.state_labeling.rules import classify_message
from stage1.value_quality import assess_value

WINDOW_HOURS = (1, 6, 24, 168)
HISTORY_LOOKBACK = timedelta(days=36)
RESEARCH_SCORE_KIND = "round7_linear_score"
LIVE_POLICY_VERSION = "round7-raw-upload-v1"


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _to_float(value: Any) -> float | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _to_bool(value: Any) -> bool:
    value = _clean(value)
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "t", "yes", "y", "да"}


def _quality_flags(store: ParquetStore, value: Any) -> tuple[str, ...]:
    decoded = store.decode_json(value, [])
    if isinstance(decoded, list):
        return tuple(str(item) for item in decoded)
    return ()


def _feature_events(store: ParquetStore, frame: pd.DataFrame) -> list[FeatureEvent]:
    output: list[FeatureEvent] = []
    if frame.empty:
        return output

    for _, row in frame.iterrows():
        channel_id = str(_clean(row.get("channel_id")) or "").strip()
        stamp = pd.to_datetime(row.get("timestamp"), errors="coerce")
        if not channel_id or pd.isna(stamp):
            continue
        timestamp = stamp.to_pydatetime().replace(tzinfo=None)
        raw = _clean(row.get("value_raw"))
        state = _clean(row.get("value_state"))
        numeric = _to_float(row.get("value_numeric"))
        sensor_type = str(_clean(row.get("sensor_type")) or "unknown")
        raw_text = "" if raw is None else str(raw)
        if not raw_text and state is not None:
            raw_text = str(state)
        assessment = assess_value(sensor_type, raw_text, numeric)
        output.append(
            FeatureEvent(
                channel_id=channel_id,
                timestamp=timestamp,
                alarm=_to_bool(row.get("alarm")),
                value_numeric=numeric,
                value_state=(str(state) if state is not None and numeric is None else None),
                sensor_type=sensor_type,
                object_id=(str(_clean(row.get("object_id"))) if _clean(row.get("object_id")) is not None else None),
                join_status=(str(_clean(row.get("join_status"))) if _clean(row.get("join_status")) is not None else None),
                quality_flags=_quality_flags(store, row.get("quality_flags")),
                qa_value_category=assessment.category,
                numeric_measurement_usable=assessment.numeric_measurement_usable,
                state_transition_usable=assessment.state_transition_usable,
            )
        )
    output.sort(key=lambda event: (event.channel_id, event.timestamp))
    return output


def _semantic_counts(events: list[FeatureEvent], prediction_time: datetime) -> dict[str, int]:
    result: dict[str, int] = {}
    for hours in WINDOW_HOURS:
        lower = prediction_time - timedelta(hours=hours)
        counters = {
            "technical_message_count": 0,
            "registered_fault_text_count": 0,
            "normal_message_count": 0,
            "environmental_alarm_count": 0,
            "unknown_state_count": 0,
        }
        for event in events:
            if not (lower < event.timestamp <= prediction_time) or event.value_state is None:
                continue
            meaning = classify_message(event.sensor_type, event.value_state, event.alarm)
            counters["technical_message_count"] += int(meaning.category == "technical_fault")
            counters["registered_fault_text_count"] += int(meaning.target_message_candidate)
            counters["normal_message_count"] += int(meaning.category == "normal")
            counters["environmental_alarm_count"] += int(meaning.category == "environmental_alarm")
            counters["unknown_state_count"] += int(meaning.category == "unknown")
        for name, count in counters.items():
            result[f"{name}_{hours}h"] = count
    return result


def _episode_features(events: list[FeatureEvent], prediction_time: datetime) -> dict[str, float | int | None]:
    """Small causal adapter for the same registered state semantics used by Stage 1.

    An episode opens on exact registered ``Неисправен`` and closes on later exact
    ``Норма``. This is journal state history, not a claim of verified physical failure.
    """

    open_start: datetime | None = None
    completed: list[tuple[datetime, datetime]] = []
    for event in sorted(events, key=lambda item: item.timestamp):
        if event.timestamp > prediction_time or event.value_state is None:
            continue
        effect = registered_state_effect(event.sensor_type, event.value_state, event.alarm)
        if effect == "fault" and open_start is None:
            open_start = event.timestamp
        elif effect == "normal" and open_start is not None and event.timestamp > open_start:
            completed.append((open_start, event.timestamp))
            open_start = None

    lower = prediction_time - timedelta(hours=168)
    recent = [(start, end) for start, end in completed if lower < end <= prediction_time]
    last_end = completed[-1][1] if completed else None
    return {
        "last_completed_episode_end_age_seconds": (
            (prediction_time - last_end).total_seconds() if last_end is not None else None
        ),
        "completed_episode_count_168h": len(recent),
        "completed_episode_mean_duration_seconds_168h": (
            sum((end - start).total_seconds() for start, end in recent) / len(recent)
            if recent
            else None
        ),
    }


def _registered_state(events: list[FeatureEvent], prediction_time: datetime) -> str | None:
    state: str | None = None
    for event in sorted(events, key=lambda item: item.timestamp):
        if event.timestamp > prediction_time or event.value_state is None:
            continue
        effect = registered_state_effect(event.sensor_type, event.value_state, event.alarm)
        if effect == "fault":
            state = "Неисправен"
        elif effect == "normal":
            state = "Норма"
        elif effect == "uncertain":
            state = "Неизвестно"
    return state


def _last_explicit_normal(events: list[FeatureEvent], prediction_time: datetime) -> datetime | None:
    normal: datetime | None = None
    for event in sorted(events, key=lambda item: item.timestamp):
        if event.timestamp > prediction_time or event.value_state is None:
            continue
        if registered_state_effect(event.sensor_type, event.value_state, event.alarm) == "normal":
            normal = event.timestamp
    return normal


def _admission_status(
    events: list[FeatureEvent],
    prediction_time: datetime,
    current_state: str | None,
    last_normal: datetime | None,
) -> tuple[str, str | None]:
    if current_state == "Неисправен":
        return "already_faulty", "already_registered_fault"
    usable = [event for event in events if event.timestamp <= prediction_time]
    if not usable:
        return "not_available", "no_history"
    sensor_types = {event.sensor_type for event in usable if event.sensor_type and event.sensor_type != "unknown"}
    if len(sensor_types) != 1:
        return "not_available", "unknown_or_conflicting_sensor_type"
    unique_times = sorted({event.timestamp for event in usable})
    if not unique_times:
        return "not_available", "no_history"
    if last_normal is None:
        return "not_available", "no_explicit_normal"
    return "scored", None


def _base_row(
    store: ParquetStore,
    model: Round7ResearchModels,
    channel_id: str,
    events: list[FeatureEvent],
    prediction_time: datetime,
) -> tuple[dict[str, Any], str | None]:
    a2 = feature_at(events, channel_id, prediction_time)
    semantic = _semantic_counts(events, prediction_time)
    episode = _episode_features(events, prediction_time)
    qa = qa_window_counts(events, prediction_time)
    source: dict[str, Any] = {**a2, **semantic, **episode, **qa}

    base_names = model.base_names
    row: dict[str, Any] = {}
    for name in base_names:
        if name.startswith("missing__"):
            continue
        row[name] = source.get(name)

    for name in base_names:
        if name.startswith("missing__"):
            original = name.removeprefix("missing__")
            row[name] = int(_clean(row.get(original)) is None)

    # The frozen model needs every field explicitly present. Unknown optional
    # values remain None and are handled by its pinned transform.
    for name in base_names:
        row.setdefault(name, None)

    return row, _registered_state(events, prediction_time)


def _round7_in_cooldown(store: ParquetStore, channel_id: str, prediction_time: datetime) -> bool:
    """Apply the frozen 24-hour warning suppression to newly uploaded hours."""

    previous = store.forecasts(channel_id)
    if previous.empty:
        return False
    previous["prediction_time"] = pd.to_datetime(previous["prediction_time"], errors="coerce")
    previous = previous[
        previous["prediction_time"].notna()
        & previous["policy_version"].astype(str).str.startswith("round7-")
        & previous["warning_reason"].astype(str).eq("round7_gates_passed_record_only")
    ]
    if previous.empty:
        return False
    last = previous["prediction_time"].max().to_pydatetime().replace(tzinfo=None)
    return timedelta(0) <= prediction_time - last < timedelta(hours=24)


def score_channels(
    store: ParquetStore,
    channel_ids: Iterable[str],
) -> int:
    ids = sorted({str(value) for value in channel_ids if str(value).strip()})
    if not ids:
        return 0

    try:
        model = Round7ResearchModels()
    except (ValueError, KeyError, OSError):
        return 0

    # First find each channel's latest event, then read only the history needed
    # for 168h windows + the frozen baseline lookback/embargo.
    latest = store.latest_event_times(ids)
    if not latest:
        return 0
    min_time = min(latest.values()) - HISTORY_LOOKBACK
    # A periodic upload closes the preceding whole hour.  The current hour is
    # deliberately left open so a partial upload cannot become a prediction.
    prediction_times = {
        channel_id: stamp.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        for channel_id, stamp in latest.items()
    }
    max_time = max(latest.values())
    history = store.events_for_channels(ids, start_at=min_time, end_at=max_time)
    feature_events = _feature_events(store, history)
    by_channel: dict[str, list[FeatureEvent]] = defaultdict(list)
    for event in feature_events:
        by_channel[event.channel_id].append(event)

    records: list[dict[str, Any]] = []
    for channel_id in ids:
        prediction_time = prediction_times.get(channel_id)
        events = by_channel.get(channel_id, [])
        if prediction_time is None or not events:
            continue

        base, current_state = _base_row(store, model, channel_id, events, prediction_time)
        last_normal = _last_explicit_normal(events, prediction_time)
        prediction_status, unavailable_reason = _admission_status(
            events, prediction_time, current_state, last_normal
        )
        score_linear = None
        score_tree = None
        score_specialist = None
        passes_common = None
        passes_standard = None
        research_status = "not_available"
        if prediction_status == "scored":
            service = pd.DataFrame([
                {
                    "channel_id": channel_id,
                    "prediction_time": prediction_time,
                    "admission_status": "eligible",
                    "last_explicit_normal_at": last_normal,
                    **base,
                }
            ])
            scored = model.score(service).iloc[0]
            score_linear = _to_float(scored.get("score_linear"))
            score_tree = _to_float(scored.get("score_tree"))
            score_specialist = _to_float(scored.get("score_specialist"))
            passes_common = _clean(scored.get("passes_common_gates"))
            passes_standard = _clean(scored.get("passes_standard_gates"))
            research_status = str(scored.get("prediction_status") or "not_available")

        model_passed = bool(passes_standard) if prediction_status == "scored" else None
        suppressed = bool(
            model_passed and _round7_in_cooldown(store, channel_id, prediction_time)
        )
        crossed = False if suppressed else model_passed
        warning_reason = (
            "suppressed_24h"
            if suppressed
            else "round7_gates_passed_record_only"
            if crossed is True
            else "round7_gates_not_passed"
            if crossed is False
            else "no_prediction"
        )

        sensor_type = str(base.get("sensor_type") or events[-1].sensor_type or "unknown")
        records.append(
            {
                "policy_version": LIVE_POLICY_VERSION,
                "freeze_sha256": None,
                "channel_id": channel_id,
                "prediction_time": prediction_time,
                "sensor_type": sensor_type,
                "admission_status": "eligible" if prediction_status == "scored" else "unknown",
                "admission_reason": unavailable_reason,
                "prediction_status": prediction_status,
                "unavailable_reason": unavailable_reason,
                "rule_score": score_linear if prediction_status == "scored" else None,
                "threshold": None,
                "threshold_crossed": crossed,
                "shadow_warning": False,
                "warning_reason": warning_reason,
                "score_contributions": {
                    "linear": score_linear,
                    "tree": score_tree,
                    "specialist": score_specialist,
                    "passes_common_gates": bool(passes_common)
                    if passes_common is not None
                    else None,
                    "passes_standard_gates": bool(passes_standard)
                    if passes_standard is not None
                    else None,
                }
                if prediction_status == "scored"
                else None,
                "delivery_mode": "record_only",
                "automatic_action_taken": False,
                "history_through": prediction_time,
                "admission_through": prediction_time,
                "registered_fault_text_count_24h": base.get("registered_fault_text_count_24h"),
                "registered_fault_text_count_168h": base.get("registered_fault_text_count_168h"),
                "completed_episode_count_168h": base.get("completed_episode_count_168h"),
                "technical_message_count_24h": base.get("technical_message_count_24h"),
                "research_score": score_linear,
                "research_prediction_status": research_status,
                "research_model_version": str(model.manifest["schema_version"]),
                "research_score_kind": RESEARCH_SCORE_KIND,
                "round7_score_linear": score_linear,
                "round7_score_tree": score_tree,
                "round7_score_specialist": score_specialist,
                "round7_passes_common_gates": passes_common,
                "round7_passes_standard_gates": passes_standard,
            }
        )

    return store.append_forecasts(pd.DataFrame(records)) if records else 0
