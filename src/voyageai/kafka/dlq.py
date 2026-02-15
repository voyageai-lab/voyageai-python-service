"""Dead Letter Queue (DLQ) producer for failed messages.

When a message fails processing after all retries, it is sent to
the planning.dlq topic for manual inspection and reprocessing.

DLQ messages include the original message plus error metadata:
- Original event JSON
- Error message and stack trace
- Number of processing attempts
- Timestamp of failure
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Producer

from voyageai.config import settings

logger = logging.getLogger(__name__)


class DeadLetterProducer:
    """Sends failed messages to the Dead Letter Queue topic.

    The DLQ topic allows operators to:
    1. Monitor failure rates
    2. Debug why messages failed
    3. Replay messages after fixing the root cause

    DLQ message format:
    {
        "original_event": { ... },
        "error": "error message",
        "stack_trace": "...",
        "attempts": 3,
        "failed_at": "2025-06-01T12:00:00Z"
    }
    """

    DLQ_TOPIC = "planning.dlq"

    def __init__(
        self,
        bootstrap_servers: str | None = None,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._producer: Producer | None = None

    def _ensure_producer(self) -> Producer:
        if self._producer is None:
            self._producer = Producer({
                "bootstrap.servers": self._bootstrap_servers,
                "acks": "all",
            })
        return self._producer

    def send_to_dlq(
        self,
        original_event: dict[str, Any],
        error: Exception | str,
        attempts: int = 1,
    ) -> None:
        """Send a failed message to the DLQ topic.

        Args:
            original_event: The original event that failed processing.
            error: The exception or error message.
            attempts: Number of times processing was attempted.
        """
        error_msg = str(error)
        stack = traceback.format_exc() if isinstance(error, Exception) else ""

        dlq_message = {
            "original_event": original_event,
            "error": error_msg,
            "stack_trace": stack,
            "attempts": attempts,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }

        producer = self._ensure_producer()
        try:
            task_id = original_event.get("taskId", "unknown")
            producer.produce(
                topic=self.DLQ_TOPIC,
                key=task_id.encode("utf-8") if isinstance(task_id, str) else b"unknown",
                value=json.dumps(dlq_message).encode("utf-8"),
            )
            producer.poll(0)
            logger.warning(
                "Message sent to DLQ: task_id=%s, error=%s, attempts=%d",
                task_id,
                error_msg[:200],
                attempts,
            )
        except Exception as e:
            logger.error("Failed to send to DLQ: %s", e)

    def flush(self, timeout: float = 5.0) -> None:
        if self._producer:
            self._producer.flush(timeout)

    def close(self) -> None:
        self.flush()
        self._producer = None
