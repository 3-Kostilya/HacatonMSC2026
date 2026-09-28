"""Stable, serializable contracts shared by stage-one processing steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
import math
from typing import Any


class Decision(StrEnum):
    """Outcome of evaluating an observed interval."""

    CANDIDATE = "candidate"
    NO_CANDIDATE = "no_candidate"
    UNKNOWN = "unknown"


class Origin(StrEnum):
    """How an episode or observation entered the evaluation set."""

    OBSERVED = "observed"
    SYNTHETIC = "synthetic"
    OPERATOR = "operator"


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_naive_timestamp(name: str, value: datetime) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is not None and value.utcoffset() is not None:
        raise ValueError(f"{name} must use the journal's local naive time")


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    """One source event without silently discarding its original representation."""

    channel_id: str
    timestamp: datetime
    raw_value: str
    alarm: bool
    sensor_type: str
    source: str
    event_id: str | None = None
    object_id: str | None = None
    numeric_value: float | None = None
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("channel_id", "raw_value", "sensor_type", "source"):
            _require_text(name, getattr(self, name))
        _require_naive_timestamp("timestamp", self.timestamp)
        if not isinstance(self.alarm, bool):
            raise ValueError("alarm must be bool")
        if self.numeric_value is not None and not math.isfinite(self.numeric_value):
            raise ValueError("numeric_value must be finite when present")
        if len(set(self.quality_flags)) != len(self.quality_flags):
            raise ValueError("quality_flags must not contain duplicates")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["timestamp"] = self.timestamp.isoformat(sep=" ")
        record["quality_flags"] = list(self.quality_flags)
        return record


@dataclass(frozen=True, slots=True)
class Episode:
    """Explainable result for a suspicious, ordinary, or undecidable interval."""

    episode_id: str
    channel_id: str
    sensor_type: str
    sensor_group: str
    anomaly_type: str
    decision: Decision
    start_at: datetime
    confirmed_at: datetime
    ruleset_version: str
    evidence: tuple[str, ...]
    observation_quality: tuple[str, ...]
    cause_hypothesis: str = "unknown"
    object_id: str | None = None
    end_at: datetime | None = None
    score: float | None = None
    origin: Origin = Origin.OBSERVED
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "channel_id",
            "sensor_type",
            "sensor_group",
            "anomaly_type",
            "ruleset_version",
            "cause_hypothesis",
        ):
            _require_text(name, getattr(self, name))
        for name in ("start_at", "confirmed_at"):
            _require_naive_timestamp(name, getattr(self, name))
        if self.end_at is not None:
            _require_naive_timestamp("end_at", self.end_at)
        if self.confirmed_at < self.start_at:
            raise ValueError("confirmed_at cannot precede start_at")
        if self.end_at is not None and self.end_at < self.start_at:
            raise ValueError("end_at cannot precede start_at")
        if self.score is not None and (not math.isfinite(self.score) or not 0 <= self.score <= 1):
            raise ValueError("score must be finite and between 0 and 1")
        if self.decision is Decision.CANDIDATE and not self.evidence:
            raise ValueError("candidate episodes require at least one evidence item")
        if self.decision is Decision.UNKNOWN and not self.observation_quality:
            raise ValueError("unknown episodes require an observation-quality reason")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        for name in ("start_at", "confirmed_at", "end_at"):
            value = getattr(self, name)
            record[name] = value.isoformat(sep=" ") if value is not None else None
        record["decision"] = self.decision.value
        record["origin"] = self.origin.value
        record["evidence"] = list(self.evidence)
        record["observation_quality"] = list(self.observation_quality)
        return record
