"""Health check endpoint."""

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from voyageai.config import settings


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str
    version: str
    timestamp: datetime


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health status."""
    return HealthResponse(
        status="UP",
        service=settings.service_name,
        version=settings.service_version,
        timestamp=datetime.now(),
    )


