"""Kafka worker for processing planning requests.

The worker ties together the full pipeline:
1. Consume PlanningRequestEvent from Kafka
2. Check idempotency guard (Redis SETNX)
3. Send progress events through processing stages
4. Run the AI agent pipeline via ResilientAgentPipeline (retry/timeout/fallback)
5. Save results to MongoDB
6. Publish PlanningResultEvent back to Kafka
7. On permanent failure, send to DLQ

Module 13 enhancements:
- ResilientAgentPipeline wraps agent with retry, timeout, and fallback
- Dead Letter Queue for permanently failed messages
- Structured logging with trace_id correlation
- Cost tracking per task
- Graceful shutdown with drain support

Progress stages:
    PROCESSING(10%) → RAG_SEARCH(30%) → TOOL_CALLING(50%)
    → GENERATING(70%) → SAVING(90%) → COMPLETED(100%)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from voyageai.config import settings
from voyageai.kafka.dlq import DeadLetterProducer
from voyageai.kafka.idempotency import IdempotencyGuard
from voyageai.kafka.producer import KafkaProgressProducer
from voyageai.kafka.schemas import ClarificationReplyEvent, PlanningRequestEvent
from voyageai.logging_config import clear_trace_context, set_trace_context
from voyageai.resilience import CostTracker, ResilientAgentPipeline
from voyageai.config import settings
from voyageai.services.agent_types import AgentResponse, LLMCallRecord, ProgressCallback
from voyageai.services.responses_agent_service import ResponsesAgentService
from voyageai.storage.mongodb import MongoDBResultStore

logger = logging.getLogger(__name__)


class PlanningWorker:
    """Processes planning requests from Kafka using the AI agent pipeline.

    This worker is the core of the async processing system. It:
    - Ensures each task is processed exactly once (idempotency)
    - Reports real-time progress via Kafka events
    - Stores results in MongoDB for persistence
    - Handles errors gracefully with proper status reporting

    The worker runs in a synchronous context (Kafka consumer thread)
    but uses asyncio.run() to execute the async agent pipeline.
    """

    def __init__(
        self,
        producer: KafkaProgressProducer | None = None,
        idempotency_guard: IdempotencyGuard | None = None,
        result_store: MongoDBResultStore | None = None,
        agent_service: ResponsesAgentService | None = None,
        dlq_producer: DeadLetterProducer | None = None,
    ) -> None:
        self._producer = producer or KafkaProgressProducer()
        self._guard = idempotency_guard or IdempotencyGuard()
        self._store = result_store or MongoDBResultStore()
        if agent_service:
            self._agent = agent_service
        elif settings.use_responses_api:
            self._agent = ResponsesAgentService()
        else:
            from voyageai.services.agent_service import AgentService
            self._agent = AgentService()
        self._dlq = dlq_producer or DeadLetterProducer()

    def handle_request(self, event: PlanningRequestEvent) -> None:
        """Handle a planning request event from Kafka.

        This is the main entry point called by the KafkaRequestConsumer.
        It runs synchronously (in the consumer thread) and uses
        asyncio.run() to execute the async pipeline.

        Module 13 enhancements:
        - Structured logging with trace context
        - ResilientAgentPipeline (retry/timeout/fallback)
        - DLQ for permanent failures
        - Cost tracking

        Args:
            event: The planning request event to process.
        """
        task_id = event.task_id

        # Set trace context for structured logging
        set_trace_context(task_id=task_id, user_id=event.user_id)

        logger.info(
            "Processing planning request: task_id=%s, user_id=%s",
            task_id,
            event.user_id,
        )

        # Step 1: Idempotency check
        if not self._guard.acquire(task_id):
            logger.info("Task already processed, skipping: %s", task_id)
            clear_trace_context()
            return

        start_time = time.time()

        try:
            # Step 2: Send initial progress
            self._producer.send_progress(
                task_id=task_id,
                stage="PROCESSING",
                percent=10,
                message="Received planning request, starting pipeline...",
            )

            # Step 3: Run the entire async pipeline in a single event loop
            # This prevents "Event loop is closed" errors from Motor/MongoDB
            # by keeping all async operations in one asyncio.run() call.
            asyncio.run(
                self._run_and_save(event, start_time)
            )

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Worker error: {type(e).__name__}: {e}"
            logger.exception("Worker failed for task: %s", task_id)
            self._handle_failure(event, error_msg, processing_time_ms)
        finally:
            clear_trace_context()

    async def _run_and_save(
        self, event: PlanningRequestEvent, start_time: float
    ) -> None:
        """Run pipeline and save result in a single async context.

        Phase 2 enhancement: Pre-flight analysis with clarification questions.
        If the request is vague, the agent emits a clarification_needed event
        and returns without generating an itinerary. The worker will be resumed
        when the user replies.

        This avoids the 'Event loop is closed' error by keeping the
        agent pipeline and MongoDB save in the same event loop.
        """
        task_id = event.task_id

        # Phase 2: Pre-flight analysis — check if clarification is needed
        progress_callback = self._make_progress_callback(task_id)
        try:
            analysis = await self._agent.analyze_request(
                requirements=event.requirements,
                conversation_context=event.conversation_context,
                progress_callback=progress_callback,
            )
            if not analysis.get("ready", True):
                # Clarification needed — send event and stop processing
                # The task stays in PROCESSING state; it will be resumed
                # when the user replies via ClarificationReplyEvent.
                self._producer.send_agent_event(
                    task_id=task_id,
                    stage="CLARIFICATION",
                    percent=15,
                    message="I have some questions before planning your trip...",
                    event_type="clarification_needed",
                    event_data={"questions": analysis.get("questions", [])},
                )
                logger.info("Clarification needed for task %s, waiting for user reply", task_id)
                return
        except Exception as e:
            logger.warning("Pre-flight analysis failed, proceeding with planning: %s", e)

        response = await self._run_pipeline(event)

        processing_time_ms = int((time.time() - start_time) * 1000)

        if response.success and response.itinerary:
            # Save to MongoDB
            self._producer.send_progress(
                task_id=task_id,
                stage="SAVING",
                percent=90,
                message="Saving itinerary to database...",
            )

            itinerary_json = response.itinerary.model_dump_json()
            tool_trace_list = [
                {
                    "tool": t.tool_name,
                    "arguments": t.arguments,
                    "latency_ms": t.latency_ms,
                    "success": t.success,
                }
                for t in response.tool_trace
            ]
            cost_breakdown = self._build_cost_breakdown(response)

            await self._store.save_result(
                task_id=task_id,
                user_id=event.user_id,
                project_id=event.project_id,
                status="COMPLETED",
                itinerary_json=itinerary_json,
                tool_trace=tool_trace_list,
                processing_time_ms=processing_time_ms,
                total_tokens=response.total_tokens,
                total_cost_usd=response.total_cost_usd,
                cost_breakdown=cost_breakdown,
            )

            # Send result event
            self._producer.send_result(
                task_id=task_id,
                user_id=event.user_id,
                project_id=event.project_id,
                status="COMPLETED",
                itinerary_json=itinerary_json,
                tool_trace=tool_trace_list,
                processing_time_ms=processing_time_ms,
                total_tokens=response.total_tokens,
                total_cost_usd=response.total_cost_usd,
                cost_breakdown=cost_breakdown,
            )

            # Send completion progress
            self._producer.send_progress(
                task_id=task_id,
                stage="COMPLETED",
                percent=100,
                message="Itinerary generated successfully!",
            )

            self._guard.mark_completed(task_id)
            logger.info(
                "Task completed: task_id=%s, time=%dms, tokens=%d, cost=$%.6f",
                task_id,
                processing_time_ms,
                response.total_tokens,
                response.total_cost_usd,
            )

        else:
            # Agent failed (after retries and fallback)
            error_msg = response.error or "Unknown error during generation"
            await self._async_handle_failure(
                event, error_msg, processing_time_ms
            )

    async def _async_handle_failure(
        self,
        event: PlanningRequestEvent,
        error: str,
        processing_time_ms: int,
    ) -> None:
        """Async version of failure handler (used inside _run_and_save)."""
        task_id = event.task_id

        try:
            await self._store.save_result(
                task_id=task_id,
                user_id=event.user_id,
                project_id=event.project_id,
                status="FAILED",
                error=error,
                processing_time_ms=processing_time_ms,
            )
        except Exception as e:
            logger.error("Failed to save error to MongoDB: %s", e)

        self._send_failure_events(event, error, processing_time_ms)

    def _handle_failure(
        self,
        event: PlanningRequestEvent,
        error: str,
        processing_time_ms: int,
    ) -> None:
        """Handle task failure: save error, send result, DLQ, release lock.

        Called from sync context (exception handler in handle_request).
        Uses a fresh event loop for MongoDB operations.

        Args:
            event: Original request event.
            error: Error message.
            processing_time_ms: Time spent before failure.
        """
        task_id = event.task_id

        try:
            asyncio.run(
                self._store.save_result(
                    task_id=task_id,
                    user_id=event.user_id,
                    project_id=event.project_id,
                    status="FAILED",
                    error=error,
                    processing_time_ms=processing_time_ms,
                )
            )
        except Exception as e:
            logger.error("Failed to save error to MongoDB: %s", e)

        self._send_failure_events(event, error, processing_time_ms)

    def _send_failure_events(
        self,
        event: PlanningRequestEvent,
        error: str,
        processing_time_ms: int,
    ) -> None:
        """Send failure result/progress events and DLQ (sync, no MongoDB)."""
        task_id = event.task_id

        try:
            self._producer.send_result(
                task_id=task_id,
                user_id=event.user_id,
                project_id=event.project_id,
                status="FAILED",
                error=error,
                processing_time_ms=processing_time_ms,
            )

            self._producer.send_progress(
                task_id=task_id,
                stage="FAILED",
                percent=0,
                message=error,
            )
        except Exception as e:
            logger.error("Failed to send failure events: %s", e)

        # Send to Dead Letter Queue for manual inspection
        try:
            original_event_dict = json.loads(
                event.model_dump_json(by_alias=True)
            )
            self._dlq.send_to_dlq(
                original_event=original_event_dict,
                error=error,
                attempts=1,
            )
            self._dlq.flush(timeout=2.0)
        except Exception as e:
            logger.error("Failed to send to DLQ: %s", e)

        # Release lock to allow retry
        self._guard.release(task_id)

    def handle_clarification_reply(self, event: ClarificationReplyEvent) -> None:
        """Handle a user's reply to clarification questions.

        Enriches the original requirements with the user's answers and
        re-runs the full planning pipeline.

        Args:
            event: Clarification reply event with user answers.
        """
        task_id = event.task_id
        set_trace_context(task_id=task_id, user_id=event.user_id)

        logger.info(
            "Processing clarification reply: task_id=%s, answers=%d",
            task_id,
            len(event.answers),
        )

        start_time = time.time()

        try:
            # Build enriched requirements from original + answers
            answer_lines = []
            for ans in event.answers:
                q = ans.get("question", "")
                a = ans.get("answer", "")
                answer_lines.append(f"- {q}: {a}")
            answers_text = "\n".join(answer_lines) if answer_lines else ""

            enriched_requirements = (
                f"{event.original_requirements}\n\n"
                f"Additional details from user:\n{answers_text}"
            )

            # Create a synthetic PlanningRequestEvent with enriched requirements.
            # Preserve conversation_context so the agent keeps project continuity.
            enriched_event = PlanningRequestEvent(
                task_id=task_id,
                user_id=event.user_id,
                project_id=event.project_id,
                requirements=enriched_requirements,
                task_type="CONVERSATION_UPDATE" if event.conversation_context else "INITIAL_PLANNING",
                conversation_context=event.conversation_context,
                timestamp=event.timestamp,
            )

            # Send progress update
            self._producer.send_progress(
                task_id=task_id,
                stage="PROCESSING",
                percent=20,
                message="Great, I have all the information I need! Starting your trip plan...",
            )

            # Run the pipeline (skip pre-flight analysis on retry)
            asyncio.run(self._run_pipeline_and_save(enriched_event, start_time))

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Worker error on clarification reply: {type(e).__name__}: {e}"
            logger.exception("Worker failed for clarification reply: %s", task_id)
            self._handle_failure(enriched_event if 'enriched_event' in dir() else
                                 PlanningRequestEvent(
                                     task_id=task_id, user_id=event.user_id,
                                     project_id=event.project_id,
                                     requirements=event.original_requirements,
                                     task_type="INITIAL_PLANNING",
                                     timestamp=event.timestamp,
                                 ),
                                 error_msg, processing_time_ms)
        finally:
            clear_trace_context()

    async def _run_pipeline_and_save(
        self, event: PlanningRequestEvent, start_time: float
    ) -> None:
        """Run pipeline and save — used by clarification reply handler.

        Unlike _run_and_save, this skips the pre-flight analysis.
        """
        task_id = event.task_id
        response = await self._run_pipeline(event)
        processing_time_ms = int((time.time() - start_time) * 1000)

        if response.success and response.itinerary:
            self._producer.send_progress(
                task_id=task_id, stage="SAVING", percent=90,
                message="Saving itinerary to database...",
            )
            itinerary_json = response.itinerary.model_dump_json()
            tool_trace_list = [
                {"tool": t.tool_name, "arguments": t.arguments,
                 "latency_ms": t.latency_ms, "success": t.success}
                for t in response.tool_trace
            ]
            cost_breakdown = self._build_cost_breakdown(response)
            await self._store.save_result(
                task_id=task_id, user_id=event.user_id, project_id=event.project_id,
                status="COMPLETED", itinerary_json=itinerary_json,
                tool_trace=tool_trace_list, processing_time_ms=processing_time_ms,
                total_tokens=response.total_tokens,
                total_cost_usd=response.total_cost_usd,
                cost_breakdown=cost_breakdown,
            )
            self._producer.send_result(
                task_id=task_id, user_id=event.user_id, project_id=event.project_id,
                status="COMPLETED", itinerary_json=itinerary_json,
                tool_trace=tool_trace_list, processing_time_ms=processing_time_ms,
                total_tokens=response.total_tokens,
                total_cost_usd=response.total_cost_usd,
                cost_breakdown=cost_breakdown,
            )
            self._producer.send_progress(
                task_id=task_id, stage="COMPLETED", percent=100,
                message="Itinerary generated successfully!",
            )
            self._guard.mark_completed(task_id)
        else:
            error_msg = response.error or "Unknown error during generation"
            await self._async_handle_failure(event, error_msg, processing_time_ms)

    @staticmethod
    def _build_cost_breakdown(response: AgentResponse) -> list[dict[str, Any]]:
        """Extract serializable cost breakdown from agent response."""
        return [
            {
                "label": c.label,
                "model": c.model,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "cost_usd": c.cost_usd,
            }
            for c in response.llm_calls
        ]

    def _make_progress_callback(self, task_id: str) -> ProgressCallback:
        """Create an async progress callback that bridges agent events to Kafka.

        The agent service calls this callback at every decision point (tool calls,
        thinking, stage changes). The callback maps event types to Kafka messages
        with an interpolated progress percentage.

        Args:
            task_id: Task identifier for Kafka message key.

        Returns:
            Async callback: (event_type, data) -> None
        """
        # Mutable progress counter — updated as events flow through
        progress_state = {"percent": 20}

        _PERCENT_MAP = {
            "stage_change": 5,   # bump +5
            "thinking": 3,       # bump +3
            "tool_start": 2,     # bump +2
            "tool_result": 4,    # bump +4
            "plan_outline": 5,   # bump +5
            "cost_summary": 0,   # no bump — informational only
            "clarification_needed": 0,
        }

        async def _callback(event_type: str, data: dict[str, Any]) -> None:
            bump = _PERCENT_MAP.get(event_type, 2)
            # Cap at 85 to leave room for SAVING and COMPLETED stages
            progress_state["percent"] = min(85, progress_state["percent"] + bump)

            message = data.get("message") or data.get("text") or event_type
            if isinstance(message, str) and len(message) > 200:
                message = message[:200] + "..."

            try:
                self._producer.send_agent_event(
                    task_id=task_id,
                    stage="PROCESSING",
                    percent=progress_state["percent"],
                    message=str(message),
                    event_type=event_type,
                    event_data=data,
                )
            except Exception:
                logger.warning("Failed to send agent event: %s", event_type, exc_info=True)

        return _callback

    async def _run_pipeline(
        self, event: PlanningRequestEvent
    ) -> AgentResponse:
        """Run the async AI agent pipeline with progress updates.

        Module 13: Uses ResilientAgentPipeline for:
        - Retry with exponential backoff on transient failures
        - Timeout to prevent hung tasks
        - Fallback response when all retries exhausted
        - Cost tracking per task

        Phase 1 enhancement: Rich progress callback for real-time SSE streaming.

        Args:
            event: Planning request event with user requirements.

        Returns:
            AgentResponse with itinerary and tool trace.
        """
        task_id = event.task_id

        # Create rich progress callback for real-time agent events
        progress_callback = self._make_progress_callback(task_id)

        # Progress: Starting the pipeline
        self._producer.send_agent_event(
            task_id=task_id,
            stage="PROCESSING",
            percent=15,
            message="Analyzing your travel request...",
            event_type="stage_change",
            event_data={"stage": "PROCESSING", "message": "Analyzing your travel request..."},
        )

        # Run agent through resilient pipeline (retry/timeout/fallback)
        pipeline = ResilientAgentPipeline(
            agent=self._agent,
            max_retries=3,
            timeout_seconds=settings.worker_pipeline_timeout_seconds,
            budget_limit_usd=1.0,
        )

        response = await pipeline.execute(
            requirements=event.requirements,
            user_id=event.user_id,
            conversation_context=event.conversation_context,
            progress_callback=progress_callback,
        )

        # Log cost tracking summary
        cost_summary = pipeline.cost_tracker.summary()
        logger.info(
            "Pipeline cost: tokens=%d, cost=$%.6f, requests=%d",
            cost_summary["total_tokens"],
            cost_summary["total_cost_usd"],
            cost_summary["requests"],
        )

        return response

    def shutdown(self) -> None:
        """Gracefully shut down worker resources.

        Flushes all producers (including DLQ) before closing
        to ensure no messages are lost during shutdown.
        """
        self._producer.flush()
        self._producer.close()
        self._dlq.close()
        self._guard.close()
        asyncio.run(self._store.close())
        logger.info("Worker shut down")
