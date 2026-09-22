from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import APP_TITLE, APP_VERSION, FRONTEND_ORIGINS
from .routers.dashboard import router as dashboard_router
from .routers.sensors import router as sensors_router

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=FRONTEND_ORIGINS,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


@app.get(
    "/api/health",
    tags=["System"]
)
def health():

    return {
        "status": "ok"
    }


app.include_router(
    dashboard_router
)

app.include_router(
    sensors_router
)