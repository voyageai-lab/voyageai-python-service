"""Structured JSON logging configuration with trace_id correlation.

Production systems need structured logs for:
1. Machine-parseable format (ELK, Datadog, CloudWatch)
2. Trace correlation across services (trace_id from Kafka headers)
3. Consistent field names for dashboards and alerts
4. Performance metrics embedded in log entries

Usage:
    from voyageai.logging_config import setup_logging, set_trace_context
    setup_logging()

    # In worker handler:
    set_trace_context(task_id="task-001", user_id="user-123")
    logger.info("Processing started")
    # → {"timestamp": "...", "level": "INFO", "message": "Processing started",
    #    "task_id": "task-001", "user_id": "user-123", "service": "voyageai-python-service"}
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from voyageai.config import settings

# Thread-local storage for trace context
_trace_context = threading.local()


def set_trace_context(
    task_id: str | None = None,
    user_id: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Set trace context for the current thread.

    Called at the start of each request/message handling to
    attach correlation IDs to all subsequent log entries.
    """
    if task_id is not None:
        _trace_context.task_id = task_id
    if user_id is not None:
        _trace_context.user_id = user_id
    if trace_id is not None:
        _trace_context.trace_id = trace_id


def clear_trace_context() -> None:
    """Clear trace context after processing completes."""
    _trace_context.task_id = None
    _trace_context.user_id = None
    _trace_context.trace_id = None


def get_trace_context() -> dict[str, str | None]:
    """Get the current trace context."""
    return {
        "task_id": getattr(_trace_context, "task_id", None),
        "user_id": getattr(_trace_context, "user_id", None),
        "trace_id": getattr(_trace_context, "trace_id", None),
    }


class StructuredJsonFormatter(logging.Formatter):
    """JSON log formatter that includes trace context and service metadata.

    Output format:
    {
        "timestamp": "2025-06-01T12:00:00.000Z",
        "level": "INFO",
        "logger": "voyageai.kafka.worker",
        "message": "Task completed",
        "service": "voyageai-python-service",
        "task_id": "task-001",
        "user_id": "user-123",
        "extra": { ... }
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": settings.service_name,
        }

        # Add trace context from thread-local storage
        ctx = get_trace_context()
        if ctx["task_id"]:
            log_entry["task_id"] = ctx["task_id"]
        if ctx["user_id"]:
            log_entry["user_id"] = ctx["user_id"]
        if ctx["trace_id"]:
            log_entry["trace_id"] = ctx["trace_id"]

        # Add exception info if present
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        # Add any extra fields passed via logger.info("msg", extra={...})
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "module",
            "funcName", "levelno", "lineno", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread",
            "threadName", "exc_info", "exc_text", "msecs", "message",
            "levelname", "taskName",
        }
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith("_")
        }
        if extra:
            log_entry["extra"] = extra

        return json.dumps(log_entry, default=str)


def setup_logging(json_format: bool = True, level: str = "INFO") -> None:
    """Configure application-wide logging.

    Args:
        json_format: Use structured JSON format (True for production,
                     False for development with human-readable output).
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if json_format and not settings.debug:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )

    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("confluent_kafka").setLevel(logging.WARNING)
