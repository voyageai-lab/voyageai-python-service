"""Idempotency guard using Redis SETNX for exactly-once processing.

In an at-least-once Kafka setup, the same message may be delivered
multiple times (e.g., consumer crash before offset commit). The
idempotency guard ensures each task is processed only once.

Algorithm:
1. On receiving a task, attempt Redis SETNX on key "task:{taskId}:lock"
2. If SETNX returns True → first time seeing this task → proceed
3. If SETNX returns False → duplicate delivery → skip processing
4. Lock has TTL to auto-expire if worker crashes mid-processing

This is simpler and faster than distributed transactions, and
sufficient for our use case where reprocessing a task is safe
but wasteful (wasted OpenAI API calls and compute).
"""

from __future__ import annotations

import logging

import redis

from voyageai.config import settings

logger = logging.getLogger(__name__)


class IdempotencyGuard:
    """Redis-based idempotency guard for Kafka message deduplication.

    Uses Redis SETNX (SET if Not eXists) with TTL for distributed locking.
    This prevents the same planning task from being processed multiple times
    when Kafka delivers duplicate messages.

    Usage:
        guard = IdempotencyGuard()
        if guard.acquire(task_id):
            try:
                process_task(task_id)
                guard.mark_completed(task_id)
            except Exception:
                guard.release(task_id)
    """

    def __init__(
        self,
        redis_url: str | None = None,
        lock_ttl_seconds: int | None = None,
    ) -> None:
        self._redis_url = redis_url or settings.redis_url
        self._lock_ttl = lock_ttl_seconds or settings.redis_lock_ttl_seconds
        self._client: redis.Redis | None = None

    def _ensure_client(self) -> redis.Redis:
        """Lazy-initialize Redis client."""
        if self._client is None:
            self._client = redis.Redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            logger.info("Redis client initialized: %s", self._redis_url)
        return self._client

    def _lock_key(self, task_id: str) -> str:
        """Generate the Redis key for a task lock."""
        return f"task:{task_id}:lock"

    def _status_key(self, task_id: str) -> str:
        """Generate the Redis key for task processing status."""
        return f"task:{task_id}:status"

    def acquire(self, task_id: str) -> bool:
        """Attempt to acquire processing lock for a task.

        Uses SETNX (SET if Not eXists) which is atomic:
        - If key doesn't exist → set key + return True (lock acquired)
        - If key exists → return False (already processing/processed)

        The lock has a TTL to auto-expire if the worker crashes.

        Args:
            task_id: Unique task identifier.

        Returns:
            True if lock acquired (first time), False if duplicate.
        """
        client = self._ensure_client()
        key = self._lock_key(task_id)

        acquired = client.set(key, "PROCESSING", nx=True, ex=self._lock_ttl)
        if acquired:
            logger.info("Lock acquired for task: %s", task_id)
            return True
        else:
            current = client.get(key)
            logger.warning(
                "Duplicate task detected: %s (status=%s)", task_id, current
            )
            return False

    def mark_completed(self, task_id: str) -> None:
        """Mark a task as completed (updates lock value, extends TTL).

        After successful processing, update the lock value to COMPLETED
        and extend TTL to keep the dedup record longer.
        """
        client = self._ensure_client()
        key = self._lock_key(task_id)
        # Keep completed records for 24 hours
        client.set(key, "COMPLETED", ex=86400)
        logger.info("Task marked completed: %s", task_id)

    def release(self, task_id: str) -> None:
        """Release the lock (delete key) on failure.

        Called when processing fails and should be retried on the
        next Kafka delivery. Deleting the key allows reprocessing.
        """
        client = self._ensure_client()
        key = self._lock_key(task_id)
        client.delete(key)
        logger.info("Lock released for task: %s (will allow retry)", task_id)

    def get_status(self, task_id: str) -> str | None:
        """Get the current processing status of a task.

        Returns:
            'PROCESSING', 'COMPLETED', or None if no record.
        """
        client = self._ensure_client()
        return client.get(self._lock_key(task_id))

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Redis client closed")
