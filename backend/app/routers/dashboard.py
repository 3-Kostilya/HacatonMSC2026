from fastapi import APIRouter

from ..schemas import DashboardSummary
from ..services import get_dashboard_summary

router = APIRouter(
    prefix="/api/dashboard",
    tags=["Dashboard"]
)


@router.get(
    "/summary",
    response_model=DashboardSummary
)
def dashboard_summary():

    return get_dashboard_summary()