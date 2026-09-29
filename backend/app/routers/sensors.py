from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from app.config import (
    MAX_HISTORY_ITEMS,
)
from app.dependencies import (
    get_store,
)
from app.schemas import (
    SensorAssessment,
    SensorDetails,
    SensorEvent,
    SensorListItem,
)
from app.services import (
    search_sensors,
    sensor_assessment,
    sensor_details,
    sensor_history,
    sensor_list,
)
from app.storage import (
    ParquetStore,
)

router = APIRouter(
    prefix="/api/sensors",
    tags=[
        "Sensors",
    ],
)


@router.get(
    "",
    response_model=list[
        SensorListItem
    ],
)
def get_sensors(
    group: str | None = Query(
        default=None,
        pattern=(
            "^(failed|warning|anomaly)$"
        ),
    ),
    store: ParquetStore = Depends(
        get_store
    ),
):
    return sensor_list(
        store,
        group,
    )


@router.get(
    "/search",
    response_model=list[
        SensorListItem
    ],
)
def search(
    q: str = Query(
        default=""
    ),
    store: ParquetStore = Depends(
        get_store
    ),
):
    return search_sensors(
        store,
        q,
    )


@router.get(
    "/{sensor_id}",
    response_model=SensorDetails,
)
def get_sensor(
    sensor_id: str,
    store: ParquetStore = Depends(
        get_store
    ),
):

    result = sensor_details(
        store,
        sensor_id,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Sensor not found"
            ),
        )

    return result


@router.get(
    "/{sensor_id}/history",
    response_model=list[
        SensorEvent
    ],
)
def get_sensor_history(
    sensor_id: str,

    limit: int = Query(
        default=50,
        ge=1,
        le=MAX_HISTORY_ITEMS,
    ),

    store: ParquetStore = Depends(
        get_store
    ),
):

    if (
        sensor_details(
            store,
            sensor_id,
        )
        is None
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "Sensor not found"
            ),
        )

    return sensor_history(
        store,
        sensor_id,
        limit,
    )


@router.get(
    "/{sensor_id}/assessment",
    response_model=SensorAssessment,
)
def get_sensor_assessment(
    sensor_id: str,

    store: ParquetStore = Depends(
        get_store
    ),
):

    result = sensor_assessment(
        store,
        sensor_id,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Sensor not found"
            ),
        )

    return result