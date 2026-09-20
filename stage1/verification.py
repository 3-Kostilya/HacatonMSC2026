"""Executable causal invariants used by the Stage-1 readiness report."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from stage1.contracts import Decision, NormalizedEvent
from stage1.detectors import (
    detect_discrete_pattern,
    detect_discrete_patterns,
    detect_numeric_level_shift,
)
from stage1.pipeline import evaluate_channel


BASE = datetime(2026, 1, 1)


def _event(minute: int, value: object, *, numeric: bool = True, flags=()):
    return NormalizedEvent(
        channel_id="verification",
        timestamp=BASE + timedelta(minutes=minute),
        raw_value=str(value),
        numeric_value=float(value) if numeric else None,
        sensor_type="Датчик температуры" if numeric else "Датчик дыма",
        alarm=False,
        source="causal-verification",
        quality_flags=tuple(flags),
    )


def verify_causal_invariants() -> dict[str, object]:
    checks: dict[str, bool] = {}

    numeric = [_event(index, value) for index, value in enumerate([10] * 12 + [20] * 3)]
    prefix = detect_numeric_level_shift(numeric)
    future = detect_numeric_level_shift(
        numeric
        + [
            _event(
                1000,
                20,
                flags=("channel_time_conflict",),
            )
        ]
    )
    checks["future_quality_preserves_confirmed_past"] = (
        prefix.decision is Decision.CANDIDATE
        and future.decision is Decision.CANDIDATE
        and (prefix.start_at, prefix.confirmed_at) == (future.start_at, future.confirmed_at)
    )

    invalid_stream = numeric[:12] + [
        _event(12, 20),
        replace(
            _event(13, "NaN", numeric=False, flags=("nonfinite_numeric",)),
            sensor_type="Датчик температуры",
        ),
        _event(14, 20),
        _event(15, 20),
    ]
    invalid_result = detect_numeric_level_shift(invalid_stream)
    checks["blocking_raw_event_breaks_numeric_confirmation"] = (
        invalid_result.decision is Decision.UNKNOWN
    )

    recurrence = [
        _event(index, value)
        for index, value in enumerate([10] * 12 + [20] * 3 + [10] * 5 + [20] * 3)
    ]
    recurrence_result = evaluate_channel(recurrence)
    checks["merged_active_recurrence_stays_open"] = (
        recurrence_result.detector_result.decision is Decision.CANDIDATE
        and recurrence_result.detector_result.end_at is None
    )

    accelerated = [_event(index * 60, "normal", numeric=False) for index in range(6)]
    accelerated.extend(_event(minute, "normal", numeric=False) for minute in (360, 361, 362))
    checks["same_state_acceleration_detected"] = (
        detect_discrete_pattern(accelerated).decision is Decision.CANDIDATE
    )

    switching = [_event(index, "normal", numeric=False) for index in range(6)]
    switching.extend(
        _event(6 + index, value, numeric=False)
        for index, value in enumerate(("alarm", "normal", "alarm", "normal"))
    )
    switching.extend(_event(index, "normal", numeric=False) for index in range(10, 60))
    switching.extend(
        _event(60 + index, value, numeric=False)
        for index, value in enumerate(("alarm", "normal", "alarm", "normal"))
    )
    switching_results = detect_discrete_patterns(switching)
    checks["rapid_switching_recovers_and_recurs"] = (
        len(switching_results) == 2
        and switching_results[0].end_at is not None
        and switching_results[1].end_at is None
    )

    return {
        "all_passed": all(checks.values()),
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
    }
