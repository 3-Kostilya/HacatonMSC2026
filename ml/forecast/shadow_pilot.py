"""Frozen R6 warning decisions for a past-only, record-only shadow pilot.

Admission is supplied by A and must be justified using information available
at the prediction hour. This module never reads target labels or sends alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from analysis.r6_provenance import frozen_rule_sha256
from ml.forecast.r6_rule import RULE_VERSION, TERMS


POLICY_VERSION = "r6-b-shadow-warning-policy-v1"
PINNED_FREEZE_SHA256 = "02d397b8a8d734449ec852504836e759e1a605a28f059befb4738b213a13d9d9"
FORBIDDEN_FIELDS = frozenset({
    "target", "target_episode_id", "label_available_at", "split",
    "future_label", "future_episode_id", "outcome",
})


def forbidden_input_fields(names: Iterable[str]) -> set[str]:
    """Fail closed when retrospective target lineage enters the pilot input."""
    return {name for name in names if name in FORBIDDEN_FIELDS
            or name.startswith(("target_", "future_", "label_"))}


def _time(value: object, name: str) -> datetime:
    if isinstance(value, datetime):
        at = value
    elif isinstance(value, str):
        at = datetime.fromisoformat(value)
    else:
        raise ValueError(f"{name} must be a timestamp")
    if at.tzinfo is not None:
        raise ValueError(f"{name} must use local timezone-naive journal time")
    return at


@dataclass(frozen=True)
class ShadowPolicy:
    threshold: float
    cooldown: timedelta
    freeze_sha256: str

    @classmethod
    def from_freeze(cls, path: Path) -> ShadowPolicy:
        digest = frozen_rule_sha256(path)
        config = json.loads(path.read_text(encoding="utf-8"))
        if (digest != PINNED_FREEZE_SHA256
                or config["schema_version"] != "r6-frozen-conditional-journal-rule-v1"
                or config["model_version"] != RULE_VERSION
                or config["feature_terms"] != TERMS
                or config["frozen_threshold"] != 7.1
                or config["per_channel_warning_cooldown_hours"] != 24
                or config["physical_failure_claim"]
                or config["final_product_threshold_approved"]):
            raise ValueError("shadow pilot requires the accepted frozen R6 rule")
        return cls(threshold=7.1, cooldown=timedelta(hours=24), freeze_sha256=digest)


@dataclass
class ShadowState:
    """Persisted between batches so cooldown survives month and process boundaries."""

    last_prediction_at: dict[str, datetime] = field(default_factory=dict)
    last_warning_at: dict[str, datetime] = field(default_factory=dict)

    def checkpoint(self, policy: ShadowPolicy) -> dict:
        return {
            "schema_version": POLICY_VERSION,
            "freeze_sha256": policy.freeze_sha256,
            "last_prediction_at": {key: value.isoformat() for key, value in
                                   sorted(self.last_prediction_at.items())},
            "last_warning_at": {key: value.isoformat() for key, value in
                                sorted(self.last_warning_at.items())},
        }

    @classmethod
    def restore(cls, checkpoint: Mapping, policy: ShadowPolicy) -> ShadowState:
        if (checkpoint.get("schema_version") != POLICY_VERSION
                or checkpoint.get("freeze_sha256") != policy.freeze_sha256):
            raise ValueError("shadow checkpoint belongs to another policy")
        state = cls(
            last_prediction_at={key: _time(value, "last_prediction_at") for key, value
                                in checkpoint["last_prediction_at"].items()},
            last_warning_at={key: _time(value, "last_warning_at") for key, value
                             in checkpoint["last_warning_at"].items()},
        )
        if any(channel not in state.last_prediction_at
               or at > state.last_prediction_at[channel]
               for channel, at in state.last_warning_at.items()):
            raise ValueError("shadow checkpoint warning chronology differs")
        return state


def _count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def decide_shadow_hour(row: Mapping, state: ShadowState, policy: ShadowPolicy) -> dict:
    """Return one explainable decision; never emit to a customer or actuator."""
    leaked = forbidden_input_fields(row)
    if leaked:
        raise ValueError(f"future-label fields are forbidden in pilot input: {sorted(leaked)}")
    channel = row.get("channel_id")
    sensor_type = row.get("sensor_type")
    if not isinstance(channel, str) or not channel or not isinstance(sensor_type, str) or not sensor_type:
        raise ValueError("channel_id and sensor_type must be explicit")
    at = _time(row.get("prediction_time"), "prediction_time")
    if (at.minute, at.second, at.microsecond) != (0, 0, 0) or at.year == 2021:
        raise ValueError("prediction must be a whole eligible local hour")
    previous = state.last_prediction_at.get(channel)
    if previous is not None and at <= previous:
        raise ValueError("channel predictions must be strictly chronological")
    admission = row.get("admission_status")
    if admission not in {"eligible", "unknown", "excluded"}:
        raise ValueError("admission_status must be eligible, unknown or excluded")
    reason = row.get("admission_reason")
    if admission != "eligible" and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("unavailable admission must include a reason")
    if admission == "eligible" and reason not in (None, ""):
        raise ValueError("eligible admission cannot have an unavailable reason")

    result = {
        "policy_version": POLICY_VERSION,
        "freeze_sha256": policy.freeze_sha256,
        "channel_id": channel,
        "prediction_time": at.isoformat(),
        "sensor_type": sensor_type,
        "admission_status": admission,
        "prediction_status": "unavailable",
        "unavailable_reason": reason if admission != "eligible" else None,
        "rule_score": None,
        "threshold_crossed": None,
        "shadow_warning": False,
        "warning_reason": "no_prediction",
        "score_contributions": None,
        "delivery_mode": "record_only",
        "automatic_action_taken": False,
    }
    if admission == "eligible":
        history_through = row.get("history_through")
        admission_through = row.get("admission_through")
        if history_through is None or admission_through is None:
            result["unavailable_reason"] = "missing_evidence_cutoff"
        elif (_time(history_through, "history_through") > at
              or _time(admission_through, "admission_through") > at):
            result["unavailable_reason"] = "future_evidence"
        else:
            counts = {name: _count(row.get(name)) for name in TERMS}
            if any(value is None for value in counts.values()):
                result["unavailable_reason"] = "missing_or_invalid_rule_count"
            else:
                contributions = {name: TERMS[name] * value for name, value in counts.items()}
                score = sum(contributions.values())
                crossed = score >= policy.threshold
                result.update(prediction_status="scored", unavailable_reason=None,
                              rule_score=score, threshold_crossed=crossed,
                              score_contributions=contributions)
                if not crossed:
                    result["warning_reason"] = "below_frozen_threshold"
                elif (channel in state.last_warning_at
                      and at < state.last_warning_at[channel] + policy.cooldown):
                    result["warning_reason"] = "channel_cooldown"
                else:
                    result["shadow_warning"] = True
                    result["warning_reason"] = "recorded_shadow_warning"
                    state.last_warning_at[channel] = at
    state.last_prediction_at[channel] = at
    return result


def run_shadow_batch(rows: Iterable[Mapping], state: ShadowState,
                     policy: ShadowPolicy) -> list[dict]:
    """Process rows in arrival order, retaining state across repeated calls."""
    return [decide_shadow_hour(row, state, policy) for row in rows]


def summarize_shadow_batch(decisions: Iterable[Mapping], *,
                           expected_channel_hours: int | None = None) -> dict:
    """Summarize observable load only; future outcomes are deliberately absent."""
    if expected_channel_hours is not None and expected_channel_hours < 0:
        raise ValueError("expected hours cannot be negative")
    by_type: dict[str, Counter] = defaultdict(Counter)
    observed_days: dict[str, set] = defaultdict(set)
    scored_days: dict[str, set] = defaultdict(set)
    reasons = Counter()
    count = 0
    for row in decisions:
        count += 1
        sensor_type = row["sensor_type"]
        if row["automatic_action_taken"] or row["delivery_mode"] != "record_only":
            raise ValueError("shadow report contains an external action")
        group = by_type[sensor_type]
        group["decisions"] += 1
        day = (row["channel_id"], _time(row["prediction_time"], "prediction_time").date())
        observed_days[sensor_type].add(day)
        if row["prediction_status"] == "scored":
            group["scored_hours"] += 1
            scored_days[sensor_type].add(day)
        else:
            group["unavailable_hours"] += 1
            reasons[row["unavailable_reason"]] += 1
        if row["shadow_warning"]:
            group["shadow_warnings"] += 1
        if row["warning_reason"] == "channel_cooldown":
            group["cooldown_suppressions"] += 1
    types = {}
    for name, counts in sorted(by_type.items()):
        observed = len(observed_days[name])
        scored = len(scored_days[name])
        types[name] = {
            "decisions": counts["decisions"],
            "scored_hours": counts["scored_hours"],
            "unavailable_hours": counts["unavailable_hours"],
            "shadow_warnings": counts["shadow_warnings"],
            "cooldown_suppressions": counts["cooldown_suppressions"],
            "observed_channel_days": observed,
            "scored_channel_days": scored,
            "warnings_per_1000_observed_channel_days": (
                counts["shadow_warnings"] * 1000 / observed if observed else None),
            "warnings_per_1000_scored_channel_days": (
                counts["shadow_warnings"] * 1000 / scored if scored else None),
        }
    total_scored = sum(item["scored_hours"] for item in types.values())
    if expected_channel_hours is not None and expected_channel_hours < count:
        raise ValueError("expected hours cannot be fewer than recorded decisions")
    return {
        "schema_version": "r6-b-shadow-observable-summary-v1",
        "decisions": count,
        "scored_hours": total_scored,
        "unavailable_hours": count - total_scored,
        "shadow_warnings": sum(item["shadow_warnings"] for item in types.values()),
        "coverage_of_submitted_hours": total_scored / count if count else None,
        "expected_channel_hours": expected_channel_hours,
        "coverage_of_expected_hours": (total_scored / expected_channel_hours
                                       if expected_channel_hours else None),
        "unavailable_reasons": dict(sorted(reasons.items())),
        "by_sensor_type": types,
        "future_label_metrics_available": False,
        "budget_is_product_approved": False,
    }
