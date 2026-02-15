"""Tests for Module 13: Production hardening - resilience, DLQ, structured logging.

Tests cover:
1. CostTracker: token/cost accumulation and budget limits
2. ResilientAgentPipeline: retry, timeout, fallback
3. DeadLetterProducer: DLQ message format and delivery
4. StructuredJsonFormatter: JSON log output with trace context
5. Health check endpoints: liveness and readiness
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voyageai.kafka.dlq import DeadLetterProducer
from voyageai.logging_config import (
    StructuredJsonFormatter,
    clear_trace_context,
    get_trace_context,
    set_trace_context,
)
from voyageai.resilience import CostTracker, ResilientAgentPipeline
from voyageai.services.agent_service import AgentResponse


# =====================================================================
# CostTracker Tests
# =====================================================================


class TestCostTracker:
    """Test token usage and cost tracking."""

    def test_initial_state(self):
        tracker = CostTracker()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.requests == 0
        assert not tracker.is_over_budget()

    def test_record_tokens(self):
        tracker = CostTracker()
        tracker.record(input_tokens=1000, output_tokens=500)
        assert tracker.total_tokens == 1500
        assert tracker.requests == 1
        assert tracker.total_cost_usd > 0

    def test_multiple_records_accumulate(self):
        tracker = CostTracker()
        tracker.record(input_tokens=500, output_tokens=500)
        tracker.record(input_tokens=500, output_tokens=500)
        assert tracker.total_tokens == 2000
        assert tracker.requests == 2

    def test_budget_limit(self):
        tracker = CostTracker(budget_limit_usd=0.001)
        # Record enough tokens to exceed $0.001 budget
        tracker.record(input_tokens=50000, output_tokens=50000)
        assert tracker.is_over_budget()

    def test_under_budget(self):
        tracker = CostTracker(budget_limit_usd=100.0)
        tracker.record(input_tokens=1000, output_tokens=1000)
        assert not tracker.is_over_budget()

    def test_summary(self):
        tracker = CostTracker(budget_limit_usd=10.0)
        tracker.record(input_tokens=1000, output_tokens=500)
        summary = tracker.summary()
        assert summary["total_tokens"] == 1500
        assert summary["requests"] == 1
        assert summary["budget_limit_usd"] == 10.0
        assert not summary["is_over_budget"]
        assert isinstance(summary["total_cost_usd"], float)


# =====================================================================
# ResilientAgentPipeline Tests
# =====================================================================


class TestResilientAgentPipeline:
    """Test retry, timeout, and fallback behaviors."""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.generate_with_tools = AsyncMock()
        return agent

    @pytest.fixture
    def success_response(self):
        return AgentResponse(
            itinerary=None,  # simplified for test
            tool_trace=[],
            raw_response="test",
            success=True,
            error=None,
            total_tokens=100,
            processing_time_ms=50,
        )

    @pytest.fixture
    def failure_response(self):
        return AgentResponse(
            itinerary=None,
            tool_trace=[],
            raw_response="",
            success=False,
            error="LLM error",
            total_tokens=10,
            processing_time_ms=10,
        )

    @pytest.mark.asyncio
    async def test_successful_execution(self, mock_agent, success_response):
        """Pipeline returns agent response on first success."""
        mock_agent.generate_with_tools.return_value = success_response
        pipeline = ResilientAgentPipeline(
            agent=mock_agent, max_retries=3, timeout_seconds=30
        )

        result = await pipeline.execute("Plan trip to Tokyo")

        assert result.success is True
        assert result.total_tokens == 100
        mock_agent.generate_with_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, mock_agent, success_response):
        """Pipeline retries on RuntimeError then succeeds."""
        mock_agent.generate_with_tools.side_effect = [
            RuntimeError("Transient error"),
            success_response,
        ]
        pipeline = ResilientAgentPipeline(
            agent=mock_agent, max_retries=3, timeout_seconds=30
        )

        result = await pipeline.execute("Plan trip to Paris")

        assert result.success is True
        assert mock_agent.generate_with_tools.call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_after_all_retries(self, mock_agent):
        """Pipeline uses fallback when all retries are exhausted."""
        mock_agent.generate_with_tools.side_effect = RuntimeError("Persistent error")
        pipeline = ResilientAgentPipeline(
            agent=mock_agent, max_retries=2, timeout_seconds=30
        )

        result = await pipeline.execute("Plan trip to London")

        assert result.success is False
        assert "failed after retries" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_triggers_fallback(self, mock_agent):
        """Pipeline uses fallback when timeout is exceeded."""

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(10)

        mock_agent.generate_with_tools.side_effect = slow_call
        pipeline = ResilientAgentPipeline(
            agent=mock_agent, max_retries=1, timeout_seconds=1
        )

        result = await pipeline.execute("Plan trip to Berlin")

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_cost_tracking_after_execution(self, mock_agent, success_response):
        """Cost tracker accumulates token usage after execution."""
        mock_agent.generate_with_tools.return_value = success_response
        pipeline = ResilientAgentPipeline(
            agent=mock_agent, max_retries=3, timeout_seconds=30
        )

        await pipeline.execute("Plan trip")

        assert pipeline.cost_tracker.total_tokens > 0
        assert pipeline.cost_tracker.requests == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_value_error(self, mock_agent):
        """Pipeline does NOT retry on ValueError (bad input)."""
        mock_agent.generate_with_tools.side_effect = ValueError("Bad input")
        pipeline = ResilientAgentPipeline(
            agent=mock_agent, max_retries=3, timeout_seconds=30
        )

        result = await pipeline.execute("Bad request")

        assert result.success is False
        assert mock_agent.generate_with_tools.call_count == 1

    @pytest.mark.asyncio
    async def test_agent_failure_triggers_retry(self, mock_agent, failure_response, success_response):
        """Agent returning success=False raises RuntimeError which triggers retry."""
        mock_agent.generate_with_tools.side_effect = [
            failure_response,
            success_response,
        ]
        pipeline = ResilientAgentPipeline(
            agent=mock_agent, max_retries=3, timeout_seconds=30
        )

        result = await pipeline.execute("Plan trip")

        # First call: failure_response → RuntimeError → retry
        # Second call: success_response → return
        assert result.success is True
        assert mock_agent.generate_with_tools.call_count == 2


# =====================================================================
# DeadLetterProducer Tests
# =====================================================================


class TestDeadLetterProducer:
    """Test DLQ message formatting and sending."""

    @patch("voyageai.kafka.dlq.Producer")
    def test_send_to_dlq_format(self, mock_producer_cls):
        """DLQ message contains original event, error, and metadata."""
        mock_producer = MagicMock()
        mock_producer_cls.return_value = mock_producer

        dlq = DeadLetterProducer()
        original = {"taskId": "task-001", "requirements": "Tokyo trip"}

        dlq.send_to_dlq(
            original_event=original,
            error="Processing failed",
            attempts=3,
        )

        # Verify produce was called
        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args
        assert call_kwargs.kwargs["topic"] == "planning.dlq"
        assert call_kwargs.kwargs["key"] == b"task-001"

        # Parse the message value
        message = json.loads(call_kwargs.kwargs["value"])
        assert message["original_event"]["taskId"] == "task-001"
        assert message["error"] == "Processing failed"
        assert message["attempts"] == 3
        assert "failed_at" in message

    @patch("voyageai.kafka.dlq.Producer")
    def test_send_to_dlq_with_exception(self, mock_producer_cls):
        """DLQ message includes stack trace when given an Exception."""
        mock_producer = MagicMock()
        mock_producer_cls.return_value = mock_producer

        dlq = DeadLetterProducer()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            dlq.send_to_dlq(
                original_event={"taskId": "task-002"},
                error=e,
                attempts=1,
            )

        call_kwargs = mock_producer.produce.call_args
        message = json.loads(call_kwargs.kwargs["value"])
        assert message["error"] == "Test error"
        assert message["stack_trace"] != ""

    @patch("voyageai.kafka.dlq.Producer")
    def test_dlq_handles_produce_error(self, mock_producer_cls):
        """DLQ gracefully handles producer errors without raising."""
        mock_producer = MagicMock()
        mock_producer.produce.side_effect = Exception("Kafka down")
        mock_producer_cls.return_value = mock_producer

        dlq = DeadLetterProducer()
        # Should not raise
        dlq.send_to_dlq(
            original_event={"taskId": "task-003"},
            error="Some error",
        )

    @patch("voyageai.kafka.dlq.Producer")
    def test_close_flushes_producer(self, mock_producer_cls):
        """Close flushes pending messages then sets producer to None."""
        mock_producer = MagicMock()
        mock_producer_cls.return_value = mock_producer

        dlq = DeadLetterProducer()
        dlq._ensure_producer()
        dlq.close()

        mock_producer.flush.assert_called_once()
        assert dlq._producer is None


# =====================================================================
# Structured Logging Tests
# =====================================================================


class TestStructuredLogging:
    """Test JSON log formatter and trace context."""

    def setup_method(self):
        clear_trace_context()

    def teardown_method(self):
        clear_trace_context()

    def test_set_and_get_trace_context(self):
        set_trace_context(task_id="task-001", user_id="user-123")
        ctx = get_trace_context()
        assert ctx["task_id"] == "task-001"
        assert ctx["user_id"] == "user-123"

    def test_clear_trace_context(self):
        set_trace_context(task_id="task-001")
        clear_trace_context()
        ctx = get_trace_context()
        assert ctx["task_id"] is None

    def test_json_formatter_output(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed
        assert "service" in parsed

    def test_json_formatter_with_trace_context(self):
        set_trace_context(task_id="task-999", user_id="user-456")
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="With context", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["task_id"] == "task-999"
        assert parsed["user_id"] == "user-456"

    def test_json_formatter_with_exception(self):
        formatter = StructuredJsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="Error occurred", args=(), exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "test error"


# =====================================================================
# Health Endpoint Tests
# =====================================================================


class TestHealthEndpoints:
    """Test health check API endpoints."""

    @pytest.mark.asyncio
    async def test_liveness_probe(self):
        """Liveness probe always returns UP."""
        from voyageai.routers.health import health_check

        result = await health_check()
        assert result.status == "UP"
        assert result.service == "voyageai-python-service"

    @pytest.mark.asyncio
    async def test_readiness_checks_dependencies(self):
        """Readiness probe reports dependency statuses."""
        from voyageai.routers.health import readiness_check

        # With no real services running, dependencies will be DOWN
        result = await readiness_check()
        assert result.status in ("UP", "DEGRADED")
        assert "redis" in result.dependencies
        assert "mongodb" in result.dependencies
        assert "kafka" in result.dependencies

        # Each dependency has at least a status field
        for dep_name, dep_info in result.dependencies.items():
            assert "status" in dep_info
