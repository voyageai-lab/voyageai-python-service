"""Planning API endpoints for itinerary generation."""

import logging
import time

from fastapi import APIRouter, HTTPException

from voyageai.schemas.task import GenerateRequest, GenerateResponse, TaskStatus
from voyageai.services.ai_service import ai_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate_itinerary(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a travel itinerary from user requirements.

    This endpoint is designed to be called by the Java backend as an internal API.
    It accepts travel requirements and returns a structured itinerary.

    Args:
        request: Generation request containing task_id, user_id, project_id, and requirements

    Returns:
        GenerateResponse with status and generated itinerary or error
    """
    start_time = time.time()

    logger.info(
        f"Processing task {request.task_id} for user {request.user_id}, "
        f"project {request.project_id}"
    )

    try:
        itinerary = await ai_service.generate_itinerary(request.requirements)

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Task {request.task_id} completed in {processing_time_ms}ms - "
            f"Generated {itinerary.metadata.total_days}-day itinerary for "
            f"{itinerary.metadata.destination}"
        )

        return GenerateResponse(
            task_id=request.task_id,
            status=TaskStatus.COMPLETED,
            itinerary=itinerary,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.error(f"Task {request.task_id} failed after {processing_time_ms}ms: {e}")

        return GenerateResponse(
            task_id=request.task_id,
            status=TaskStatus.FAILED,
            error=str(e),
            processing_time_ms=processing_time_ms,
        )


