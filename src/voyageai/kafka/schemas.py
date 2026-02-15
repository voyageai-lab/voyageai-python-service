"""Kafka event schemas as Pydantic models.

These schemas mirror the Java DTOs in the voyageai-backend project:
- PlanningRequestEvent  (Java produces -> Python consumes)
- PlanningProgressEvent (Python produces -> Java consumes)
- PlanningResultEvent   (Python produces -> Java consumes)

Field naming uses camelCase to match Java Jackson serialization.
Pydantic's model_config with populate_by_name=True allows both
camelCase (from Kafka) and snake_case (in Python code) access.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlanningRequestEvent(BaseModel):
    """Event consumed from planning.request topic.

    Published by Java backend when a user submits a travel planning request.
    The Python worker consumes this event to start AI-powered itinerary generation.

    Attributes:
        task_id: Unique identifier for this planning task (also Kafka message key).
        user_id: Authenticated user who submitted the request.
        project_id: Project this task belongs to (for conversation context).
        requirements: User's natural language travel requirements.
        task_type: INITIAL_PLANNING or CONVERSATION_UPDATE.
        timestamp: When the request was created.
    """

    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId")
    user_id: str = Field(alias="userId")
    project_id: str = Field(alias="projectId")
    requirements: str
    task_type: str = Field(alias="taskType")
    conversation_context: str | None = Field(default=None, alias="conversationContext")
    timestamp: datetime


class ClarificationReplyEvent(BaseModel):
    """Event consumed when a user replies to clarification questions.

    Published by Java backend when user answers inline questions.
    The Python worker consumes this to resume the planning pipeline.

    Attributes:
        task_id: Task this reply belongs to.
        user_id: User who replied.
        project_id: Project context.
        answers: User's answers (list of {question_id, answer} dicts).
        original_requirements: The original planning requirements.
        timestamp: When the reply was sent.
    """

    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId")
    user_id: str = Field(alias="userId")
    project_id: str = Field(alias="projectId")
    answers: list[dict[str, Any]] = Field(default_factory=list)
    original_requirements: str = Field(default="", alias="originalRequirements")
    conversation_context: str | None = Field(default=None, alias="conversationContext")
    timestamp: datetime


class PlanningProgressEvent(BaseModel):
    """Event published to planning.progress topic.

    Sent by the Python worker to report processing progress.
    The Java backend consumes these to update Redis and push SSE events.

    Progress stages:
        PROCESSING(10%)  -> RAG_SEARCH(30%) -> TOOL_CALLING(50%)
        -> GENERATING(70%) -> SAVING(90%) -> COMPLETED(100%)

    Event types (rich streaming, Phase 1):
        stage_change  - Pipeline stage transition (existing behaviour, enhanced)
        thinking      - Agent reasoning text
        tool_start    - Tool call initiated (tool name, arguments)
        tool_result   - Tool call completed (result summary, latency)
        plan_outline  - Plan summary before full generation
        clarification_needed - Agent asks clarification questions (Phase 2)
        clarification_answer - User replies to clarification questions (Phase 2)

    Attributes:
        task_id: Task this progress update belongs to.
        stage: Current processing stage name.
        percent: Progress percentage (0-100).
        message: Human-readable status message.
        event_type: Granular event subtype (e.g. thinking, tool_start).
        event_data: Structured payload for the event (JSON-serializable dict).
        timestamp: When this progress event was created.
    """

    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId")
    stage: str
    percent: int
    message: str
    event_type: str | None = Field(default=None, alias="eventType")
    event_data: dict[str, Any] | None = Field(default=None, alias="eventData")
    timestamp: datetime


class PlanningResultEvent(BaseModel):
    """Event published to planning.result topic.

    Sent by the Python worker when processing is complete (success or failure).
    The Java backend consumes this to save results and finalize the task.

    Attributes:
        task_id: Task this result belongs to.
        user_id: User who submitted the original request.
        project_id: Project this task belongs to.
        status: COMPLETED or FAILED.
        itinerary_json: Serialized StructuredItinerary JSON (null on failure).
        tool_trace: List of tool calls with timing info.
        error: Error message (null on success).
        processing_time_ms: Total processing time in milliseconds.
        total_tokens: Total OpenAI tokens consumed.
        timestamp: When processing completed.
    """

    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(alias="taskId")
    user_id: str = Field(alias="userId")
    project_id: str = Field(alias="projectId")
    status: str
    itinerary_json: str | None = Field(default=None, alias="itineraryJson")
    tool_trace: list[dict[str, Any]] | None = Field(default=None, alias="toolTrace")
    error: str | None = None
    processing_time_ms: int | None = Field(default=None, alias="processingTimeMs")
    total_tokens: int | None = Field(default=None, alias="totalTokens")
    # Observability: cost tracking
    total_cost_usd: float | None = Field(default=None, alias="totalCostUsd")
    cost_breakdown: list[dict[str, Any]] | None = Field(default=None, alias="costBreakdown")
    timestamp: datetime
