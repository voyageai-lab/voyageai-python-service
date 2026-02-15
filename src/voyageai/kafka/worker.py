"""Kafka worker for processing planning requests.

The worker ties together the full pipeline:
1. Consume PlanningRequestEvent from Kafka
2. Check idempotency guard (Redis SETNX)
3. Send progress events through processing stages
4. Run the AI agent pipeline (Tool-RAG → tools → structured output)
5. Save results to MongoDB
6. Publish PlanningResultEvent back to Kafka

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
from voyageai.kafka.idempotency import IdempotencyGuard
from voyageai.kafka.producer import KafkaProgressProducer
from voyageai.kafka.schemas import PlanningRequestEvent
from voyageai.services.agent_service import AgentResponse, AgentService
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
        agent_service: AgentService | None = None,
    ) -> None:
        self._producer = producer or KafkaProgressProducer()
        self._guard = idempotency_guard or IdempotencyGuard()
        self._store = result_store or MongoDBResultStore()
        self._agent = agent_service or AgentService()

    def handle_request(self, event: PlanningRequestEvent) -> None:
        """Handle a planning request event from Kafka.

        This is the main entry point called by the KafkaRequestConsumer.
        It runs synchronously (in the consumer thread) and uses
        asyncio.run() to execute the async pipeline.

        Args:
            event: The planning request event to process.
        """
        task_id = event.task_id
        logger.info(
            "Processing planning request: task_id=%s, user_id=%s",
            task_id,
            event.user_id,
        )

        # Step 1: Idempotency check
        if not self._guard.acquire(task_id):
            logger.info("Task already processed, skipping: %s", task_id)
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

            # Step 3: Run the async agent pipeline
            response = asyncio.run(
                self._run_pipeline(event)
            )

            # Step 4: Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            if response.success and response.itinerary:
                # Step 5a: Save to MongoDB
                self._producer.send_progress(
                    task_id=task_id,
                    stage="SAVING",
                    percent=90,
                    message="Saving itinerary to database...",
                )

                itinerary_json = response.itinerary.model_dump_json()
                tool_trace_list = [
                    {
                        "tool": t.name,
                        "arguments": t.arguments,
                        "latency_ms": t.latency_ms,
                        "success": t.success,
                    }
                    for t in response.tool_trace
                ]

                asyncio.run(
                    self._store.save_result(
                        task_id=task_id,
                        user_id=event.user_id,
                        project_id=event.project_id,
                        status="COMPLETED",
                        itinerary_json=itinerary_json,
                        tool_trace=tool_trace_list,
                        processing_time_ms=processing_time_ms,
                        total_tokens=response.total_tokens,
                    )
                )

                # Step 5b: Send result event
                self._producer.send_result(
                    task_id=task_id,
                    user_id=event.user_id,
                    project_id=event.project_id,
                    status="COMPLETED",
                    itinerary_json=itinerary_json,
                    tool_trace=tool_trace_list,
                    processing_time_ms=processing_time_ms,
                    total_tokens=response.total_tokens,
                )

                # Step 6: Send completion progress
                self._producer.send_progress(
                    task_id=task_id,
                    stage="COMPLETED",
                    percent=100,
                    message="Itinerary generated successfully!",
                )

                self._guard.mark_completed(task_id)
                logger.info(
                    "Task completed: task_id=%s, time=%dms, tokens=%d",
                    task_id,
                    processing_time_ms,
                    response.total_tokens,
                )

            else:
                # Agent failed
                error_msg = response.error or "Unknown error during generation"
                self._handle_failure(
                    event, error_msg, processing_time_ms
                )

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Worker error: {type(e).__name__}: {e}"
            logger.exception("Worker failed for task: %s", task_id)
            self._handle_failure(event, error_msg, processing_time_ms)

    def _handle_failure(
        self,
        event: PlanningRequestEvent,
        error: str,
        processing_time_ms: int,
    ) -> None:
        """Handle task failure: save error, send result, release lock.

        Args:
            event: Original request event.
            error: Error message.
            processing_time_ms: Time spent before failure.
        """
        task_id = event.task_id

        try:
            # Save error to MongoDB
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

        try:
            # Send failure result event
            self._producer.send_result(
                task_id=task_id,
                user_id=event.user_id,
                project_id=event.project_id,
                status="FAILED",
                error=error,
                processing_time_ms=processing_time_ms,
            )

            # Send failure progress
            self._producer.send_progress(
                task_id=task_id,
                stage="FAILED",
                percent=0,
                message=error,
            )
        except Exception as e:
            logger.error("Failed to send failure events: %s", e)

        # Release lock to allow retry
        self._guard.release(task_id)

    async def _run_pipeline(
        self, event: PlanningRequestEvent
    ) -> AgentResponse:
        """Run the async AI agent pipeline with progress updates.

        This is the core async method that orchestrates:
        1. Tool-RAG for relevant tool selection
        2. Tool calling with real-time data
        3. Structured itinerary generation

        Args:
            event: Planning request event with user requirements.

        Returns:
            AgentResponse with itinerary and tool trace.
        """
        task_id = event.task_id

        # Progress: RAG search phase
        self._producer.send_progress(
            task_id=task_id,
            stage="RAG_SEARCH",
            percent=30,
            message="Selecting relevant tools and searching knowledge base...",
        )

        # Progress: Tool calling phase
        self._producer.send_progress(
            task_id=task_id,
            stage="TOOL_CALLING",
            percent=50,
            message="Calling tools to gather real-time data...",
        )

        # Run the agent with tool calling
        response = await self._agent.generate_with_tools(
            requirements=event.requirements,
            use_tool_rag=True,
            tool_rag_top_k=4,
            user_id=event.user_id,
        )

        # Progress: Generating phase
        self._producer.send_progress(
            task_id=task_id,
            stage="GENERATING",
            percent=70,
            message="Generating structured itinerary...",
        )

        return response

    def shutdown(self) -> None:
        """Gracefully shut down worker resources."""
        self._producer.flush()
        self._producer.close()
        self._guard.close()
        asyncio.run(self._store.close())
        logger.info("Worker shut down")
