import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
)

from app.dependencies import (
    get_store,
)
from app.schemas import (
    EpisodeInput,
    ForecastDecisionInput,
    NormalizedEventInput,
    StoredResponse,
)
from app.storage import (
    ParquetStore,
)

router = APIRouter(
    prefix="/api",
    tags=[
        "Ingestion",
    ],
)


@router.post(
    "/events",
    response_model=StoredResponse,
)
def ingest_events(
    events: list[
        NormalizedEventInput
    ],

    store: ParquetStore = Depends(
        get_store
    ),
):

    frame = pd.DataFrame(
        [
            item.model_dump()
            for item in events
        ]
    )

    rows = store.append_events(
        frame
    )

    return {
        "status":
            "stored",

        "rows":
            rows,
    }


@router.post(
    "/ml/forecasts",
    response_model=StoredResponse,
)
def ingest_forecasts(
    forecasts: list[
        ForecastDecisionInput
    ],

    store: ParquetStore = Depends(
        get_store
    ),
):

    frame = pd.DataFrame(
        [
            item.model_dump()
            for item in forecasts
        ]
    )

    rows = (
        store.append_forecasts(
            frame
        )
    )

    return {
        "status":
            "stored",

        "rows":
            rows,
    }


@router.post(
    "/ml/episodes",
    response_model=StoredResponse,
)
def ingest_episodes(
    episodes: list[
        EpisodeInput
    ],

    store: ParquetStore = Depends(
        get_store
    ),
):

    frame = pd.DataFrame(
        [
            item.model_dump()
            for item in episodes
        ]
    )

    rows = (
        store.append_episodes(
            frame
        )
    )

    return {
        "status":
            "stored",

        "rows":
            rows,
    }