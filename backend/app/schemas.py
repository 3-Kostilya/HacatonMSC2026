from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PredictionStatus = Literal[
    "scored",
    "already_faulty",
    "unknown_state",
    "stale_observation",
    "insufficient_history",
    "not_available",
]

SensorGroup = Literal["healthy", "warning", "unavailable", "failed", "anomaly"]


class HealthResponse(BaseModel):
    status: str


class DashboardSummary(BaseModel):
    totalSensors: int
    withoutWarnings: int
    warnings: int
    predictionUnavailable: int

    # Legacy/internal counters are kept so older clients do not break.
    registeredFaults: int = 0
    anomalyCandidates: int = 0


class SensorListItem(BaseModel):
    id: str
    name: str
    type: str
    objectName: str | None = None
    currentState: str

    # Backward-compatible frontend field. It is a normalized threshold index,
    # NOT a calibrated probability of physical failure.
    riskScore: float | None = None

    warning: bool | None = None
    predictionStatus: PredictionStatus
    anomalyCandidate: bool

    # Current ML-native fields.
    ruleScore: float | None = None
    threshold: float | None = None
    thresholdCrossed: bool | None = None
    mlPredictionStatus: str | None = None

    # Separate research score. It is not used as the operational warning.
    researchScore: float | None = None
    researchPredictionStatus: str | None = None


class SensorDetails(SensorListItem):
    lastEventAt: str | None = None
    riskFactors: list[str] = Field(default_factory=list)
    objectId: str | None = None
    sensorType: str | None = None


class SensorEvent(BaseModel):
    timestamp: str
    state: str
    alarm: bool
    value: str | float | int | None
    unit: str | None = None


class SensorAssessment(BaseModel):
    id: str
    currentState: str
    riskScore: float | None = None
    warning: bool | None = None
    predictionStatus: PredictionStatus
    anomalyCandidate: bool
    riskFactors: list[str] = Field(default_factory=list)

    ruleScore: float | None = None
    threshold: float | None = None
    thresholdCrossed: bool | None = None
    mlPredictionStatus: str | None = None
    predictionTime: str | None = None
    admissionStatus: str | None = None
    admissionReason: str | None = None
    unavailableReason: str | None = None
    warningReason: str | None = None
    policyVersion: str | None = None
    scoreContributions: dict[str, float] | None = None

    researchScore: float | None = None
    researchPredictionStatus: str | None = None
    researchModelVersion: str | None = None
    researchScoreKind: str | None = None


class NormalizedEventInput(BaseModel):
    channel_id: str
    timestamp: str
    # JSON ingest keeps the older, simple API names. storage.py converts them
    # to current M1 Parquet names: raw_value -> value_raw, etc.
    raw_value: str
    alarm: bool
    sensor_type: str
    source: str = "api"
    event_id: str | None = None
    object_id: str | None = None
    object_name: str | None = None
    numeric_value: float | None = None
    unit: str | None = None
    quality_flags: list[str] = Field(default_factory=list)


class ForecastDecisionInput(BaseModel):
    policy_version: str
    freeze_sha256: str | None = None
    channel_id: str
    prediction_time: str
    sensor_type: str
    admission_status: str
    admission_reason: str | None = None
    prediction_status: str
    unavailable_reason: str | None = None
    rule_score: float | None = None
    threshold: float | None = None
    threshold_crossed: bool | None = None
    shadow_warning: bool = False
    warning_reason: str | None = None
    score_contributions: dict[str, float] | None = None
    delivery_mode: str = "record_only"
    automatic_action_taken: bool = False
    history_through: str | None = None
    admission_through: str | None = None
    registered_fault_text_count_24h: int | None = None
    registered_fault_text_count_168h: int | None = None
    completed_episode_count_168h: int | None = None
    technical_message_count_24h: int | None = None

    # Optional CatBoost research output. Kept separate from rule_score/threshold.
    research_score: float | None = None
    research_prediction_status: str | None = None
    research_model_version: str | None = None
    research_score_kind: str | None = None


class EpisodeInput(BaseModel):
    episode_id: str
    channel_id: str
    sensor_type: str
    sensor_group: str
    anomaly_type: str
    decision: str
    start_at: str
    confirmed_at: str
    ruleset_version: str
    evidence: list[str] = Field(default_factory=list)
    observation_quality: list[str] = Field(default_factory=list)
    cause_hypothesis: str = "unknown"
    object_id: str | None = None
    end_at: str | None = None
    score: float | None = None
    origin: str = "observed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredResponse(BaseModel):
    status: str
    rows: int


class ImportResponse(BaseModel):
    batchId: str
    filename: str
    datasetKind: str
    rowsCount: int
    status: str
    errorMessage: str | None = None
