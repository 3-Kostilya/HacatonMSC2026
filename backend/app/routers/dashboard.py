from fastapi import (
    APIRouter,
    Depends,
)

from app.dependencies import (
    get_store,
)
from app.schemas import (
    DashboardSummary,
)
from app.services import (
    dashboard_summary,
)
from app.storage import (
    ParquetStore,
)

router = APIRouter(
    prefix="/api/dashboard",
    tags=[
        "Dashboard",
    ],
)


@router.get(
    "/summary",
    response_model=DashboardSummary,
)
def get_dashboard_summary(
    store: ParquetStore = Depends(
        get_store
    ),
):
    return dashboard_summary(
        store
    )