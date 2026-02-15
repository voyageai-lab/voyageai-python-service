"""Health check endpoints with dependency checks.

Module 13 enhancements:
- /health: Simple liveness probe (always UP if the process is running)
- /health/ready: Readiness probe checking Redis, MongoDB, Kafka
  Used by K8s liveness/readiness probes and load balancers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from voyageai.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str
    version: str
    timestamp: datetime


class ReadinessResponse(BaseModel):
    """Readiness check response with dependency statuses."""

    status: str
    service: str
    version: str
    timestamp: datetime
    dependencies: dict[str, dict[str, Any]]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe: check that the service process is running.

    This should be fast and never depend on external services.
    K8s uses this for the liveness probe - if it fails, the pod is restarted.
    """
    return HealthResponse(
        status="UP",
        service=settings.service_name,
        version=settings.service_version,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Readiness probe: check all downstream dependencies.

    K8s uses this for the readiness probe - if it fails, the pod
    is removed from the service load balancer (no traffic sent).

    Checks:
    - Redis: PING command
    - MongoDB: server_info() command
    - Kafka: metadata() call
    """
    dependencies: dict[str, dict[str, Any]] = {}
    overall_status = "UP"

    # Check Redis
    try:
        import redis as redis_lib

        r = redis_lib.from_url(settings.redis_url, socket_timeout=2)
        r.ping()
        dependencies["redis"] = {"status": "UP"}
        r.close()
    except Exception as e:
        dependencies["redis"] = {"status": "DOWN", "error": str(e)[:200]}
        overall_status = "DEGRADED"

    # Check MongoDB
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient(
            settings.mongodb_uri, serverSelectionTimeoutMS=2000
        )
        await client.server_info()
        dependencies["mongodb"] = {"status": "UP"}
        client.close()
    except Exception as e:
        dependencies["mongodb"] = {"status": "DOWN", "error": str(e)[:200]}
        overall_status = "DEGRADED"

    # Check Kafka
    try:
        from confluent_kafka.admin import AdminClient

        admin = AdminClient(
            {"bootstrap.servers": settings.kafka_bootstrap_servers}
        )
        metadata = admin.list_topics(timeout=3)
        topic_count = len(metadata.topics)
        dependencies["kafka"] = {
            "status": "UP",
            "topics": topic_count,
        }
    except Exception as e:
        dependencies["kafka"] = {"status": "DOWN", "error": str(e)[:200]}
        overall_status = "DEGRADED"

    return ReadinessResponse(
        status=overall_status,
        service=settings.service_name,
        version=settings.service_version,
        timestamp=datetime.now(timezone.utc),
        dependencies=dependencies,
    )


