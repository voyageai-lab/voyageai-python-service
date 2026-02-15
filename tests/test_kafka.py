"""Tests for Kafka event schemas, producer, and consumer.

Tests schema serialization/deserialization (no Kafka broker needed),
and producer/consumer logic with mocked confluent-kafka clients.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from voyageai.kafka.schemas import (
    PlanningProgressEvent,
    PlanningRequestEvent,
    PlanningResultEvent,
)
from voyageai.kafka.producer import KafkaProgressProducer
from voyageai.kafka.consumer import KafkaRequestConsumer


# =========================================================================
# Schema Serialization Tests
# =========================================================================


class TestPlanningRequestEvent:
    """Test PlanningRequestEvent serialization/deserialization."""

    def test_deserialize_from_java_json(self):
        """Verify Python can parse JSON produced by Java (camelCase fields)."""
        java_json = {
            "taskId": "task-001",
            "userId": "user-123",
            "projectId": "project-456",
            "requirements": "Visit Tokyo for 3 days",
            "taskType": "INITIAL_PLANNING",
            "timestamp": "2025-06-01T12:00:00Z",
        }
        event = PlanningRequestEvent.model_validate(java_json)

        assert event.task_id == "task-001"
        assert event.user_id == "user-123"
        assert event.project_id == "project-456"
        assert event.requirements == "Visit Tokyo for 3 days"
        assert event.task_type == "INITIAL_PLANNING"
        assert event.timestamp.year == 2025

    def test_roundtrip_serialization(self):
        """Verify event can be serialized and deserialized without data loss."""
        event = PlanningRequestEvent(
            task_id="rt-001",
            user_id="user-x",
            project_id="proj-y",
            requirements="Test roundtrip",
            task_type="CONVERSATION_UPDATE",
            timestamp=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        json_str = event.model_dump_json(by_alias=True)
        restored = PlanningRequestEvent.model_validate_json(json_str)

        assert restored.task_id == event.task_id
        assert restored.user_id == event.user_id
        assert restored.requirements == event.requirements


class TestPlanningProgressEvent:
    """Test PlanningProgressEvent serialization/deserialization."""

    def test_serialize_to_camel_case(self):
        """Verify serialization produces camelCase keys for Java consumer."""
        event = PlanningProgressEvent(
            task_id="task-002",
            stage="RAG_SEARCH",
            percent=30,
            message="Searching knowledge base...",
            timestamp=datetime.now(timezone.utc),
        )
        data = json.loads(event.model_dump_json(by_alias=True))

        assert "taskId" in data
        assert data["taskId"] == "task-002"
        assert data["stage"] == "RAG_SEARCH"
        assert data["percent"] == 30

    def test_all_stages(self):
        """Verify all planning stages can be represented."""
        stages = [
            ("PROCESSING", 10),
            ("RAG_SEARCH", 30),
            ("TOOL_CALLING", 50),
            ("GENERATING", 70),
            ("SAVING", 90),
            ("COMPLETED", 100),
        ]
        for stage, percent in stages:
            event = PlanningProgressEvent(
                task_id="stage-test",
                stage=stage,
                percent=percent,
                message=f"Stage: {stage}",
                timestamp=datetime.now(timezone.utc),
            )
            assert event.stage == stage
            assert event.percent == percent


class TestPlanningResultEvent:
    """Test PlanningResultEvent serialization/deserialization."""

    def test_success_result_serialization(self):
        """Verify successful result event serializes correctly."""
        event = PlanningResultEvent(
            task_id="task-003",
            user_id="user-789",
            project_id="proj-abc",
            status="COMPLETED",
            itinerary_json='{"days": []}',
            tool_trace=[
                {"tool": "weather", "latency_ms": 150},
                {"tool": "currency", "latency_ms": 80},
            ],
            processing_time_ms=5200,
            total_tokens=3500,
            timestamp=datetime.now(timezone.utc),
        )
        data = json.loads(event.model_dump_json(by_alias=True))

        assert data["taskId"] == "task-003"
        assert data["status"] == "COMPLETED"
        assert data["itineraryJson"] == '{"days": []}'
        assert len(data["toolTrace"]) == 2
        assert data["processingTimeMs"] == 5200

    def test_failed_result_serialization(self):
        """Verify failed result event correctly includes error."""
        event = PlanningResultEvent(
            task_id="task-004",
            user_id="user-111",
            project_id="proj-222",
            status="FAILED",
            error="OpenAI API rate limit exceeded",
            processing_time_ms=1200,
            timestamp=datetime.now(timezone.utc),
        )
        data = json.loads(event.model_dump_json(by_alias=True))

        assert data["status"] == "FAILED"
        assert data["error"] == "OpenAI API rate limit exceeded"
        assert data["itineraryJson"] is None
        assert data["toolTrace"] is None


# =========================================================================
# Producer Tests (mocked confluent-kafka)
# =========================================================================


class TestKafkaProgressProducer:
    """Test KafkaProgressProducer with mocked Kafka client."""

    @patch("voyageai.kafka.producer.Producer")
    def test_send_progress_calls_produce(self, mock_producer_cls):
        """Verify send_progress() calls confluent-kafka produce()."""
        mock_producer = MagicMock()
        mock_producer_cls.return_value = mock_producer

        producer = KafkaProgressProducer(
            bootstrap_servers="localhost:9092",
            progress_topic="test.progress",
            result_topic="test.result",
        )

        producer.send_progress(
            task_id="task-p1",
            stage="RAG_SEARCH",
            percent=30,
            message="Searching...",
        )

        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args
        assert call_kwargs.kwargs["topic"] == "test.progress"
        assert call_kwargs.kwargs["key"] == b"task-p1"
        # Verify JSON payload
        payload = json.loads(call_kwargs.kwargs["value"])
        assert payload["taskId"] == "task-p1"
        assert payload["stage"] == "RAG_SEARCH"
        assert payload["percent"] == 30

    @patch("voyageai.kafka.producer.Producer")
    def test_send_result_calls_produce(self, mock_producer_cls):
        """Verify send_result() calls confluent-kafka produce()."""
        mock_producer = MagicMock()
        mock_producer_cls.return_value = mock_producer

        producer = KafkaProgressProducer(
            bootstrap_servers="localhost:9092",
            progress_topic="test.progress",
            result_topic="test.result",
        )

        producer.send_result(
            task_id="task-r1",
            user_id="user-1",
            project_id="proj-1",
            status="COMPLETED",
            itinerary_json='{"days": []}',
            tool_trace=[{"tool": "weather"}],
            processing_time_ms=3000,
            total_tokens=2000,
        )

        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args
        assert call_kwargs.kwargs["topic"] == "test.result"
        payload = json.loads(call_kwargs.kwargs["value"])
        assert payload["taskId"] == "task-r1"
        assert payload["status"] == "COMPLETED"
        assert payload["processingTimeMs"] == 3000

    @patch("voyageai.kafka.producer.Producer")
    def test_flush_delegates_to_producer(self, mock_producer_cls):
        """Verify flush() calls the underlying producer flush."""
        mock_producer = MagicMock()
        mock_producer.flush.return_value = 0
        mock_producer_cls.return_value = mock_producer

        producer = KafkaProgressProducer(bootstrap_servers="localhost:9092")
        # Force producer initialization
        producer.send_progress("t1", "PROCESSING", 10, "Test")

        remaining = producer.flush(timeout=5.0)
        mock_producer.flush.assert_called_once_with(5.0)
        assert remaining == 0

    @patch("voyageai.kafka.producer.Producer")
    def test_lazy_initialization(self, mock_producer_cls):
        """Verify producer is not created until first send."""
        producer = KafkaProgressProducer(bootstrap_servers="localhost:9092")
        mock_producer_cls.assert_not_called()

        mock_producer_cls.return_value = MagicMock()
        producer.send_progress("t1", "PROCESSING", 10, "Test")
        mock_producer_cls.assert_called_once()


# =========================================================================
# Consumer Tests (mocked confluent-kafka)
# =========================================================================


class TestKafkaRequestConsumer:
    """Test KafkaRequestConsumer with mocked Kafka client."""

    @patch("voyageai.kafka.consumer.Consumer")
    def test_message_dispatched_to_handler(self, mock_consumer_cls):
        """Verify received message is deserialized and sent to handler."""
        # Set up mock consumer that returns one message then shuts down
        mock_consumer = MagicMock()
        mock_consumer_cls.return_value = mock_consumer

        java_json = json.dumps({
            "taskId": "task-c1",
            "userId": "user-c1",
            "projectId": "proj-c1",
            "requirements": "Consumer test",
            "taskType": "INITIAL_PLANNING",
            "timestamp": "2025-06-01T12:00:00Z",
        }).encode("utf-8")

        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.value.return_value = java_json

        received_events = []

        def handler(event):
            received_events.append(event)

        consumer = KafkaRequestConsumer(
            handler=handler,
            bootstrap_servers="localhost:9092",
            group_id="test-group",
            topic="test.request",
        )

        # First poll returns message, second poll triggers shutdown
        call_count = 0

        def poll_side_effect(timeout=1.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_msg
            consumer.shutdown()
            return None

        mock_consumer.poll.side_effect = poll_side_effect

        consumer.start()

        assert len(received_events) == 1
        assert received_events[0].task_id == "task-c1"
        assert received_events[0].user_id == "user-c1"
        assert received_events[0].requirements == "Consumer test"
        mock_consumer.close.assert_called_once()

    @patch("voyageai.kafka.consumer.Consumer")
    def test_invalid_message_is_skipped(self, mock_consumer_cls):
        """Verify malformed messages are logged and skipped."""
        mock_consumer = MagicMock()
        mock_consumer_cls.return_value = mock_consumer

        # Invalid JSON
        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.value.return_value = b"not valid json"
        mock_msg.offset.return_value = 42

        received_events = []
        handler = lambda event: received_events.append(event)

        consumer = KafkaRequestConsumer(
            handler=handler,
            bootstrap_servers="localhost:9092",
        )

        call_count = 0

        def poll_side_effect(timeout=1.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_msg
            consumer.shutdown()
            return None

        mock_consumer.poll.side_effect = poll_side_effect

        consumer.start()

        # Handler should not have been called
        assert len(received_events) == 0
        mock_consumer.close.assert_called_once()

    @patch("voyageai.kafka.consumer.Consumer")
    def test_handler_exception_does_not_crash_consumer(self, mock_consumer_cls):
        """Verify consumer continues even if handler raises."""
        mock_consumer = MagicMock()
        mock_consumer_cls.return_value = mock_consumer

        java_json = json.dumps({
            "taskId": "task-err",
            "userId": "user-err",
            "projectId": "proj-err",
            "requirements": "Error test",
            "taskType": "INITIAL_PLANNING",
            "timestamp": "2025-06-01T12:00:00Z",
        }).encode("utf-8")

        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.value.return_value = java_json

        def failing_handler(event):
            raise RuntimeError("Handler crashed!")

        consumer = KafkaRequestConsumer(
            handler=failing_handler,
            bootstrap_servers="localhost:9092",
        )

        call_count = 0

        def poll_side_effect(timeout=1.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_msg
            consumer.shutdown()
            return None

        mock_consumer.poll.side_effect = poll_side_effect

        # Should not raise
        consumer.start()
        mock_consumer.close.assert_called_once()
