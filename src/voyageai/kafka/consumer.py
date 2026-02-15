"""Kafka consumer for receiving planning request events.

Uses confluent-kafka Consumer with JSON deserialization.
The consumer runs in a dedicated thread, polling the planning.request
topic and dispatching events to a handler callback.

Key design decisions:
1. Manual poll loop (not auto-assign) for explicit control
2. JSON deserialization with Pydantic validation
3. Configurable handler callback for testability
4. Graceful shutdown via threading.Event
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException

from voyageai.config import settings
from voyageai.kafka.schemas import ClarificationReplyEvent, PlanningRequestEvent

logger = logging.getLogger(__name__)


class KafkaRequestConsumer:
    """Consumes planning request events from the planning.request topic.

    The consumer subscribes to the topic, deserializes JSON messages
    into PlanningRequestEvent, and dispatches them to the registered handler.

    Usage:
        consumer = KafkaRequestConsumer(handler=my_handler_fn)
        consumer.start()  # blocks until shutdown
        # or
        consumer.start_background()  # runs in a thread

    The handler function signature:
        def handler(event: PlanningRequestEvent) -> None: ...
    """

    def __init__(
        self,
        handler: Callable[[PlanningRequestEvent], None],
        clarification_handler: Callable[[ClarificationReplyEvent], None] | None = None,
        bootstrap_servers: str | None = None,
        group_id: str | None = None,
        topic: str | None = None,
    ) -> None:
        self._handler = handler
        self._clarification_handler = clarification_handler
        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._group_id = group_id or settings.kafka_group_id
        self._topic = topic or settings.kafka_topic_planning_request
        self._clarification_topic = "planning.clarification.reply"
        self._shutdown_event = threading.Event()
        self._consumer: Consumer | None = None
        self._thread: threading.Thread | None = None

    def _create_consumer(self) -> Consumer:
        """Create and configure a Kafka consumer instance."""
        config = {
            "bootstrap.servers": self._bootstrap_servers,
            "group.id": self._group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": settings.kafka_auto_commit,
            "auto.commit.interval.ms": settings.kafka_auto_commit_interval_ms,
            "session.timeout.ms": settings.kafka_session_timeout_ms,
        }
        consumer = Consumer(config)
        topics = [self._topic]
        if self._clarification_handler:
            topics.append(self._clarification_topic)
        consumer.subscribe(topics)
        logger.info(
            "Kafka consumer created: servers=%s, group=%s, topic=%s",
            self._bootstrap_servers,
            self._group_id,
            self._topic,
        )
        return consumer

    def _deserialize_message(self, raw_value: bytes) -> PlanningRequestEvent | None:
        """Deserialize JSON bytes into a PlanningRequestEvent.

        Returns None if deserialization fails (message is logged and skipped).
        """
        try:
            data = json.loads(raw_value)
            return PlanningRequestEvent.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("Failed to deserialize message: %s", e)
            return None

    def _deserialize_clarification(self, raw_value: bytes) -> ClarificationReplyEvent | None:
        """Deserialize JSON bytes into a ClarificationReplyEvent."""
        try:
            data = json.loads(raw_value)
            return ClarificationReplyEvent.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("Failed to deserialize clarification reply: %s", e)
            return None

    def start(self) -> None:
        """Start consuming messages (blocking).

        Runs until shutdown() is called. Each message is deserialized
        and dispatched to the handler. Errors in the handler are caught
        and logged to prevent consumer crash.
        """
        self._consumer = self._create_consumer()
        logger.info("Consumer started, polling topic: %s", self._topic)

        try:
            while not self._shutdown_event.is_set():
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(
                            "Reached end of partition: %s [%d] @ %d",
                            msg.topic(),
                            msg.partition(),
                            msg.offset(),
                        )
                    else:
                        logger.error("Consumer error: %s", msg.error())
                    continue

                # Route based on topic
                topic = msg.topic()
                if topic == self._clarification_topic and self._clarification_handler:
                    # Clarification reply event
                    reply_event = self._deserialize_clarification(msg.value())
                    if reply_event is None:
                        logger.warning(
                            "Skipping undeserializable clarification at offset %d",
                            msg.offset(),
                        )
                        continue
                    logger.info(
                        "Received clarification reply: task_id=%s",
                        reply_event.task_id,
                    )
                    try:
                        self._clarification_handler(reply_event)
                    except Exception:
                        logger.exception(
                            "Clarification handler failed for task_id=%s",
                            reply_event.task_id,
                        )
                else:
                    # Planning request event
                    event = self._deserialize_message(msg.value())
                    if event is None:
                        logger.warning(
                            "Skipping undeserializable message at offset %d",
                            msg.offset(),
                        )
                        continue

                    logger.info(
                        "Received planning request: task_id=%s, user_id=%s",
                        event.task_id,
                        event.user_id,
                    )

                    try:
                        self._handler(event)
                    except Exception:
                        logger.exception(
                            "Handler failed for task_id=%s", event.task_id
                        )
        except KafkaException as e:
            logger.error("Kafka consumer error: %s", e)
            raise
        finally:
            self._consumer.close()
            logger.info("Consumer closed")

    def start_background(self) -> threading.Thread:
        """Start consuming in a background daemon thread.

        Returns:
            The daemon thread running the consumer loop.
        """
        self._thread = threading.Thread(
            target=self.start,
            name="kafka-consumer",
            daemon=True,
        )
        self._thread.start()
        logger.info("Consumer started in background thread")
        return self._thread

    def shutdown(self) -> None:
        """Signal the consumer to stop and wait for thread completion."""
        logger.info("Shutting down Kafka consumer...")
        self._shutdown_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
            logger.info("Consumer thread joined")
