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


SensorGroup = Literal[
    "failed",
    "warning",
    "anomaly",
]


class HealthResponse(BaseModel):
    status: str


class DashboardSummary(BaseModel):
    totalSensors: int
    registeredFaults: int
    warnings: int
    anomalyCandidates: int
    predictionUnavailable: int


class SensorListItem(BaseModel):
    id: str

    name: str
    type: str

    objectName: str | None = None

    currentState: str

    # Старое поле frontend.
    # Это НЕ вероятность поломки.
    riskScore: float | None = None

    warning: bool | None = None

    predictionStatus: PredictionStatus

    anomalyCandidate: bool

    # Настоящие поля текущего ML.
    ruleScore: float | None = None

    threshold: float | None = None

    thresholdCrossed: bool | None = None

    mlPredictionStatus: str | None = None


class SensorDetails(
    SensorListItem
):
    lastEventAt: str | None = None

    riskFactors: list[str] = Field(
        default_factory=list
    )

    objectId: str | None = None

    sensorType: str | None = None


class SensorEvent(BaseModel):
    timestamp: str

    state: str

    alarm: bool

    value: (
        str
        | float
        | int
        | None
    )

    unit: str | None = None


class SensorAssessment(BaseModel):
    id: str

    currentState: str

    riskScore: float | None = None

    warning: bool | None = None

    predictionStatus: PredictionStatus

    anomalyCandidate: bool

    riskFactors: list[str] = Field(
        default_factory=list
    )

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

    scoreContributions: (
        dict[str, float]
        | None
    ) = None


class NormalizedEventInput(BaseModel):
    channel_id: str

    timestamp: str

    raw_value: str

    alarm: bool

    sensor_type: str

    source: str = "api"

    event_id: str | None = None

    object_id: str | None = None

    object_name: str | None = None

    numeric_value: float | None = None

    unit: str | None = None

    quality_flags: list[str] = Field(
        default_factory=list
    )


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

    score_contributions: (
        dict[str, float]
        | None
    ) = None

    delivery_mode: str = "record_only"

    automatic_action_taken: bool = False

    history_through: str | None = None

    admission_through: str | None = None

    registered_fault_text_count_24h: (
        int | None
    ) = None

    registered_fault_text_count_168h: (
        int | None
    ) = None

    completed_episode_count_168h: (
        int | None
    ) = None

    technical_message_count_24h: (
        int | None
    ) = None


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

    evidence: list[str] = Field(
        default_factory=list
    )

    observation_quality: list[str] = Field(
        default_factory=list
    )

    cause_hypothesis: str = "unknown"

    object_id: str | None = None

    end_at: str | None = None

    score: float | None = None

    origin: str = "observed"

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


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