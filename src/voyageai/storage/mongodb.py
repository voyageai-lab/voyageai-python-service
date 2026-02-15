"""MongoDB storage for planning results using Motor (async driver).

Motor wraps PyMongo with asyncio support, allowing non-blocking
database operations in the async Python worker pipeline.

The planning_results collection stores:
- Full itinerary JSON
- Tool execution trace
- Metadata (tokens, timing, user/project IDs)

A unique index on task_id provides an additional idempotency layer
at the database level (beyond the Redis SETNX guard).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from voyageai.config import settings

logger = logging.getLogger(__name__)


class MongoDBResultStore:
    """Async MongoDB storage for planning task results.

    Provides save and retrieval operations for completed planning tasks.
    Uses Motor's AsyncIOMotorClient for non-blocking I/O.

    The unique index on task_id ensures that even if two workers
    somehow process the same task, only one result is stored.
    """

    def __init__(
        self,
        uri: str | None = None,
        database: str | None = None,
        collection: str | None = None,
    ) -> None:
        self._uri = uri or settings.mongodb_uri
        self._database_name = database or settings.mongodb_database
        self._collection_name = collection or settings.mongodb_collection_results
        self._client: AsyncIOMotorClient | None = None
        self._db: AsyncIOMotorDatabase | None = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy-initialize MongoDB connection and create indexes.

        Motor's AsyncIOMotorClient is bound to the event loop that was
        running when it was created. If the event loop changes (e.g.
        because asyncio.run() was called again), we must re-create the
        client to avoid 'Event loop is closed' errors.
        """
        import asyncio

        current_loop = asyncio.get_running_loop()

        if self._initialized and self._client is not None:
            # Check if the client's event loop is still the current one
            try:
                client_loop = self._client.get_io_loop()
                if client_loop is current_loop and not client_loop.is_closed():
                    return  # Still valid
            except Exception:
                pass
            # Stale client — close and re-create
            logger.info("MongoDB client bound to stale event loop, re-initializing...")
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._initialized = False

        if self._initialized:
            return

        self._client = AsyncIOMotorClient(self._uri)
        self._db = self._client[self._database_name]
        collection = self._db[self._collection_name]

        # Create unique index on task_id for idempotency
        await collection.create_index("task_id", unique=True)
        logger.info(
            "MongoDB initialized: %s/%s.%s",
            self._uri,
            self._database_name,
            self._collection_name,
        )
        self._initialized = True

    @property
    def _collection(self):
        """Get the planning_results collection."""
        return self._db[self._collection_name]

    async def save_result(
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
        total_cost_usd: float | None = None,
        cost_breakdown: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Save a planning result to MongoDB.

        Uses upsert with task_id as the filter to ensure idempotency.
        If a result already exists for this task_id, it is updated
        (not duplicated).

        Args:
            task_id: Unique task identifier.
            user_id: User who submitted the request.
            project_id: Project this task belongs to.
            status: COMPLETED or FAILED.
            itinerary_json: Serialized itinerary JSON.
            tool_trace: List of tool execution records.
            error: Error message (on failure).
            processing_time_ms: Total processing duration.
            total_tokens: Total LLM tokens consumed.
            total_cost_usd: Total estimated cost in USD.
            cost_breakdown: Per-LLM-call cost breakdown.

        Returns:
            True if document was inserted/updated, False on error.
        """
        await self._ensure_initialized()

        document = {
            "task_id": task_id,
            "user_id": user_id,
            "project_id": project_id,
            "status": status,
            "itinerary_json": itinerary_json,
            "tool_trace": tool_trace,
            "error": error,
            "processing_time_ms": processing_time_ms,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
            "cost_breakdown": cost_breakdown,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        try:
            result = await self._collection.update_one(
                {"task_id": task_id},
                {"$set": document},
                upsert=True,
            )
            logger.info(
                "Saved result to MongoDB: task_id=%s, matched=%d, modified=%d, upserted=%s",
                task_id,
                result.matched_count,
                result.modified_count,
                result.upserted_id,
            )
            return True
        except Exception as e:
            logger.error("Failed to save result: task_id=%s, error=%s", task_id, e)
            return False

    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve a planning result by task_id.

        Args:
            task_id: Unique task identifier.

        Returns:
            Result document or None if not found.
        """
        await self._ensure_initialized()
        doc = await self._collection.find_one({"task_id": task_id})
        if doc:
            doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        return doc

    async def get_results_by_project(
        self, project_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Retrieve planning results for a project.

        Args:
            project_id: Project identifier.
            limit: Maximum number of results to return.

        Returns:
            List of result documents, newest first.
        """
        await self._ensure_initialized()
        cursor = (
            self._collection.find({"project_id": project_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._initialized = False
            logger.info("MongoDB client closed")
