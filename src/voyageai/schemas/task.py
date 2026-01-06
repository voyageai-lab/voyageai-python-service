"""Pydantic models for task request/response."""

from enum import Enum

from pydantic import BaseModel, Field

from voyageai.schemas.itinerary import StructuredItinerary


class TaskStatus(str, Enum):
    """Status of a generation task."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GenerateRequest(BaseModel):
    """Request from Java backend to generate itinerary."""

    task_id: str = Field(..., description="Unique task identifier")
    user_id: str = Field(..., description="User who requested the generation")
    project_id: str = Field(..., description="Project this task belongs to")
    requirements: str = Field(..., max_length=2000, description="User's travel requirements")


class GenerateResponse(BaseModel):
    """Response to Java backend with generation result."""

    task_id: str
    status: TaskStatus
    itinerary: StructuredItinerary | None = None
    error: str | None = None
    processing_time_ms: int = 0

