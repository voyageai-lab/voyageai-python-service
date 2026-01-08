"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from voyageai.config import settings
from voyageai.routers import health, planning, rag, tools

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    yield
    logger.info(f"Shutting down {settings.service_name}")


app = FastAPI(
    title="VoyageAI Python Service",
    description="AI-powered travel itinerary generation service",
    version=settings.service_version,
    lifespan=lifespan,
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(planning.router, prefix="/api/v1", tags=["Planning"])
app.include_router(tools.router, prefix="/api/v1", tags=["Tools"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])

