from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

PredictionStatus = Literal[
    "scored",
    "already_faulty",
    "unknown_state",
    "stale_observation",
    "insufficient_history",
    "not_available"
]


class SensorListItem(BaseModel):
    id: str
    name: str
    type: str
    objectName: str | None = None

    currentState: str

    riskScore: float | None = Field(
        default=None,
        ge=0,
        le=1
    )

    warning: bool | None = None

    predictionStatus: PredictionStatus

    anomalyCandidate: bool = False


class SensorDetails(SensorListItem):
    lastEventAt: datetime | None = None

    riskFactors: list[str] = []


class SensorEvent(BaseModel):
    timestamp: datetime
    state: str

    alarm: bool

    value: str | float | int | None = None
    unit: str | None = None


class SensorAssessment(BaseModel):
    id: str

    currentState: str

    riskScore: float | None = Field(
        default=None,
        ge=0,
        le=1
    )

    warning: bool | None = None

    predictionStatus: PredictionStatus

    anomalyCandidate: bool

    riskFactors: list[str] = []


class DashboardSummary(BaseModel):
    totalSensors: int

    registeredFaults: int

    warnings: int

    anomalyCandidates: int

    predictionUnavailable: int