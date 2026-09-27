from __future__ import annotations

import pandas as pd

from app.storage import (
    ParquetStore,
)


def get_sensors(
    store: ParquetStore,
) -> pd.DataFrame:

    return store.sensors()


def get_events(
    store: ParquetStore,
    sensor_id: str | None = None,
) -> pd.DataFrame:

    return store.events(
        sensor_id
    )


def get_forecasts(
    store: ParquetStore,
    sensor_id: str | None = None,
) -> pd.DataFrame:

    return store.forecasts(
        sensor_id
    )


def get_episodes(
    store: ParquetStore,
    sensor_id: str | None = None,
) -> pd.DataFrame:

    return store.episodes(
        sensor_id
    )