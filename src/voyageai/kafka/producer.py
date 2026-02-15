"""Kafka producer for sending progress and result events.

Uses confluent-kafka (librdkafka C wrapper) for production-grade
performance and reliability. Key design decisions:

1. JSON serialization with camelCase field names for Java compatibility
2. task_id as message key for partition ordering guarantee
3. Delivery callback for async error handling
4. Flush on shutdown to avoid message loss
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import KafkaException, Producer

from voyageai.config import settings
from voyageai.kafka.schemas import PlanningProgressEvent, PlanningResultEvent

logger = logging.getLogger(__name__)


class KafkaProgressProducer:
    """Produces planning progress and result events to Kafka.

    This producer is used by the Python worker to report back to the
    Java backend through Kafka topics:
    - planning.progress: incremental status updates
    - planning.result: final success/failure with itinerary

    Thread-safe: confluent-kafka Producer is thread-safe internally.
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        progress_topic: str | None = None,
        result_topic: str | None = None,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._progress_topic = progress_topic or settings.kafka_topic_planning_progress
        self._result_topic = result_topic or settings.kafka_topic_planning_result
        self._producer: Producer | None = None

    def _ensure_producer(self) -> Producer:
        """Lazy-initialize the Kafka producer.

        Lazy init avoids connection errors during import/testing.
        The producer is reused across all sends for connection pooling.
        """
        if self._producer is None:
            config = {
                "bootstrap.servers": self._bootstrap_servers,
                # Idempotent producer: exactly-once delivery semantics
                "enable.idempotence": True,
                # Wait for all replicas to acknowledge
                "acks": "all",
                # Retry up to 3 times on transient failures
                "retries": 3,
                # Linger briefly to batch small messages for throughput
                "linger.ms": 10,
            }
            self._producer = Producer(config)
            logger.info(
                "Kafka producer initialized: servers=%s", self._bootstrap_servers
            )
        return self._producer

    @staticmethod
    def _delivery_callback(err: Any, msg: Any) -> None:
        """Called once per message to indicate delivery result.

        This callback runs in the producer's background thread.
        """
        if err is not None:
            logger.error(
                "Message delivery failed: topic=%s, key=%s, error=%s",
                msg.topic(),
                msg.key(),
                err,
            )
        else:
            logger.debug(
                "Message delivered: topic=%s, partition=%d, offset=%d",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )

    def _serialize_event(self, event: PlanningProgressEvent | PlanningResultEvent) -> bytes:
        """Serialize event to JSON bytes with camelCase field names.

        Uses by_alias=True to produce camelCase keys matching Java DTOs.
        """
        return event.model_dump_json(by_alias=True).encode("utf-8")

    def send_progress(
        self,
        task_id: str,
        stage: str,
        percent: int,
        message: str,
    ) -> None:
        """Send a progress update event to the planning.progress topic.

        Args:
            task_id: Task identifier (used as partition key).
            stage: Current processing stage (e.g., RAG_SEARCH).
            percent: Progress percentage (0-100).
            message: Human-readable status message.
        """
        event = PlanningProgressEvent(
            task_id=task_id,
            stage=stage,
            percent=percent,
            message=message,
            timestamp=datetime.now(timezone.utc),
        )

        producer = self._ensure_producer()
        try:
            producer.produce(
                topic=self._progress_topic,
                key=task_id.encode("utf-8"),
                value=self._serialize_event(event),
                callback=self._delivery_callback,
            )
            # Trigger delivery callbacks without blocking
            producer.poll(0)
            logger.info(
                "Progress event queued: task_id=%s, stage=%s, percent=%d",
                task_id,
                stage,
                percent,
            )
        except KafkaException as e:
            logger.error("Failed to send progress event: task_id=%s, error=%s", task_id, e)
            raise

    def send_result(
        self,
        task_id: str,
        user_id: str,
        project_id: str,
        status: str,
        itinerary_json: str | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
        error: str | None = None,
        processing_time_ms: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Send a result event to the planning.result topic.

        Args:
            task_id: Task identifier (used as partition key).
            user_id: User who submitted the request.
            project_id: Project this task belongs to.
            status: COMPLETED or FAILED.
            itinerary_json: Serialized itinerary (on success).
            tool_trace: List of tool execution records.
            error: Error message (on failure).
            processing_time_ms: Total processing duration.
            total_tokens: Total LLM tokens consumed.
        """
        event = PlanningResultEvent(
            task_id=task_id,
            user_id=user_id,
            project_id=project_id,
            status=status,
            itinerary_json=itinerary_json,
            tool_trace=tool_trace,
            error=error,
            processing_time_ms=processing_time_ms,
            total_tokens=total_tokens,
            timestamp=datetime.now(timezone.utc),
        )

        producer = self._ensure_producer()
        try:
            producer.produce(
                topic=self._result_topic,
                key=task_id.encode("utf-8"),
                value=self._serialize_event(event),
                callback=self._delivery_callback,
            )
            # Trigger delivery callbacks
            producer.poll(0)
            logger.info(
                "Result event queued: task_id=%s, status=%s",
                task_id,
                status,
            )
        except KafkaException as e:
            logger.error("Failed to send result event: task_id=%s, error=%s", task_id, e)
            raise

    def flush(self, timeout: float = 10.0) -> int:
        """Flush all queued messages, blocking until delivered or timeout.

        Should be called during graceful shutdown to avoid message loss.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Number of messages still in queue (0 = all delivered).
        """
        if self._producer is not None:
            remaining = self._producer.flush(timeout)
            if remaining > 0:
                logger.warning(
                    "Kafka producer flush timeout: %d messages remaining", remaining
                )
            return remaining
        return 0

    def close(self) -> None:
        """Flush and close the producer."""
        self.flush()
        self._producer = None
        logger.info("Kafka producer closed")
