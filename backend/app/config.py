from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(BASE_DIR / ".env")


DATA_DIR = BASE_DIR / os.getenv(
    "DATA_DIR",
    "data",
)

STORE_DIR = DATA_DIR / "store"

INCOMING_DIR = DATA_DIR / "incoming"
PROCESSED_DIR = DATA_DIR / "processed"
FAILED_DIR = DATA_DIR / "failed"
REFERENCE_DIR = DATA_DIR / "reference"
RAW_RUNS_DIR = DATA_DIR / "runs"
MODEL_DIR = DATA_DIR / "model"
RUNTIME_MODEL_DIR = MODEL_DIR / "current"
CHANNELS_REFERENCE_FILE = REFERENCE_DIR / "channels.csv"
OBJECTS_REFERENCE_FILE = REFERENCE_DIR / "objects.csv"

EVENTS_DIR = STORE_DIR / "events"
FORECASTS_DIR = STORE_DIR / "forecasts"
EPISODES_DIR = STORE_DIR / "episodes"

SENSORS_FILE = (
    STORE_DIR / "sensors.parquet"
)

IMPORTS_FILE = (
    STORE_DIR / "imports.parquet"
)


R6_THRESHOLD = float(
    os.getenv(
        "R6_THRESHOLD",
        "7.1",
    )
)


MAX_HISTORY_ITEMS = int(
    os.getenv(
        "MAX_HISTORY_ITEMS",
        "200",
    )
)


CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        (
            "http://localhost:5173,"
            "http://127.0.0.1:5173"
        ),
    ).split(",")
    if origin.strip()
]


for directory in (
    DATA_DIR,
    STORE_DIR,
    INCOMING_DIR,
    PROCESSED_DIR,
    FAILED_DIR,
    REFERENCE_DIR,
    RAW_RUNS_DIR,
    MODEL_DIR,
    RUNTIME_MODEL_DIR,
    EVENTS_DIR,
    FORECASTS_DIR,
    EPISODES_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )