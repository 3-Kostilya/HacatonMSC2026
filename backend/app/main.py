from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from app.config import (
    CORS_ORIGINS,
)
from app.routers import (
    dashboard,
    imports,
    ingest,
    sensors,
)
from app.schemas import (
    HealthResponse,
)

app = FastAPI(
    title=(
        "Engineering Infrastructure "
        "Monitoring API"
    ),

    version="2.0.0",

    description=(
        "FastAPI backend with "
        "Parquet operational storage "
        "and ML-compatible contracts."
    ),
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=(
        CORS_ORIGINS
    ),

    allow_credentials=True,

    allow_methods=[
        "*",
    ],

    allow_headers=[
        "*",
    ],
)


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=[
        "Health",
    ],
)
def health():

    return {
        "status":
            "ok"
    }


app.include_router(
    dashboard.router
)

app.include_router(
    sensors.router
)

app.include_router(
    imports.router
)

app.include_router(
    ingest.router
)