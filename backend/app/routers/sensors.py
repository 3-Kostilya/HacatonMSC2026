from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from ..schemas import SensorAssessment, SensorDetails, SensorEvent, SensorListItem
from ..services import (
    get_sensor,
    get_sensor_assessment,
    get_sensor_history,
    get_sensors,
    search_sensors,
)

router = APIRouter(
    prefix="/api/sensors",
    tags=["Sensors"]
)


@router.get(
    "",
    response_model=list[SensorListItem]
)
def sensors(
    group: Literal[
        "failed",
        "warning",
        "anomaly"
    ] | None = None,

    limit: int = Query(
        default=100,
        ge=1,
        le=1000
    )
):

    return get_sensors(
        group=group,
        limit=limit
    )


@router.get(
    "/search",
    response_model=list[SensorListItem]
)
def sensors_search(
    q: str = Query(
        min_length=1
    ),

    limit: int = Query(
        default=50,
        ge=1,
        le=100
    )
):

    return search_sensors(
        query=q,
        limit=limit
    )


@router.get(
    "/{sensor_id}/history",
    response_model=list[SensorEvent]
)
def sensor_history(
    sensor_id: str
):

    sensor = get_sensor(
        sensor_id
    )

    if sensor is None:

        raise HTTPException(
            status_code=404,
            detail="Sensor not found"
        )

    return get_sensor_history(
        sensor_id
    )


@router.get(
    "/{sensor_id}/assessment",
    response_model=SensorAssessment
)
def sensor_assessment(
    sensor_id: str
):

    assessment = get_sensor_assessment(
        sensor_id
    )

    if assessment is None:

        raise HTTPException(
            status_code=404,
            detail="Sensor not found"
        )

    return assessment


@router.get(
    "/{sensor_id}",
    response_model=SensorDetails
)
def sensor_details(
    sensor_id: str
):

    sensor = get_sensor(
        sensor_id
    )

    if sensor is None:

        raise HTTPException(
            status_code=404,
            detail="Sensor not found"
        )

    return sensor