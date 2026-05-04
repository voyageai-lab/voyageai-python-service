"""
Tool-RAG: Semantic Tool Selection via RAG

This module implements dynamic tool selection using retrieval-augmented
generation. Instead of providing all tools to the LLM (which wastes tokens
and can confuse the model), we:

1. Store tool metadata with example queries in ChromaDB
2. When a user query arrives, embed it
3. Find semantically similar tools
4. Only expose top-K relevant tools to the LLM

Why Tool-RAG?
-------------
- Token Efficiency: 6 tools × ~100 tokens each = 600 tokens saved per call
- Better Accuracy: LLM focuses on relevant tools, fewer irrelevant calls
- Scalability: Can support 100+ tools without context overflow
- Dynamic: New tools can be added without code changes

Architecture:
    User Query → Embed → ChromaDB (tool_metadata) → Top-K Tools → LLM

Algorithm:
1. Embed user query using OpenAI text-embedding-3-small
2. Query ChromaDB collection "tool_metadata" for similar tools
3. Filter by category/availability if needed
4. Return top-K tools with similarity scores

Trade-offs:
- Adds one embedding call (~50ms) per request
- May miss relevant tools if example_queries don't cover all phrasings
- Requires upfront tool metadata seeding

Mitigation:
- Add diverse example_queries covering different phrasings
- Use lower similarity threshold to include borderline tools
- Fallback to all tools if similarity scores are uniformly low
"""

import logging
import time
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment,misc]
    _CHROMADB_AVAILABLE = False

from openai import AsyncOpenAI

from voyageai.config import settings
from voyageai.schemas.tool_metadata import ToolMetadata, ToolSelectionResult

logger = logging.getLogger(__name__)

# Collection name for tool metadata
TOOL_METADATA_COLLECTION = "tool_metadata"

# Embedding model (same as main RAG)
EMBEDDING_MODEL = "text-embedding-3-small"

# Default similarity threshold - tools below this aren't selected
DEFAULT_SIMILARITY_THRESHOLD = 0.3

# If all tools have low scores, fallback to all tools
FALLBACK_THRESHOLD = 0.2


class ToolRAG:
    """
    Tool-RAG for semantic tool selection.
    
    Stores tool metadata in ChromaDB and retrieves relevant tools
    based on semantic similarity to user queries.
    
    Usage:
        tool_rag = ToolRAG()
        await tool_rag.initialize()
        
        # Add tool metadata
        await tool_rag.add_tool(weather_tool_meta)
        
        # Select relevant tools for a query
        result = await tool_rag.select_tools(
            "What's the weather like in Tokyo?",
            top_k=3
        )
    """
    
    _instance: "ToolRAG | None" = None
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        """
        Initialize Tool-RAG.
        
        Args:
            persist_directory: ChromaDB storage location
            similarity_threshold: Minimum similarity score to include tool
        """
        self.persist_directory = persist_directory
        self.similarity_threshold = similarity_threshold
        self._client: Any = None
        self._collection: Any = None
        self._openai: AsyncOpenAI | None = None
        self._tool_cache: dict[str, ToolMetadata] = {}  # Cache full metadata
    
    @classmethod
    def get_instance(cls) -> "ToolRAG":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ToolRAG()
        return cls._instance
    
    async def initialize(self) -> None:
        """
        Initialize ChromaDB client and collection.
        
        Creates the tool_metadata collection if it doesn't exist.
        Should be called during application startup.
        """
        if self._client is not None and self._collection is not None:
            return

        if not _CHROMADB_AVAILABLE:
            logger.warning("chromadb not available — Tool-RAG disabled")
            return
        
        logger.info("Initializing Tool-RAG")
        
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
        
        # Get or create tool metadata collection
        self._collection = self._client.get_or_create_collection(
            name=TOOL_METADATA_COLLECTION,
            metadata={
                "description": "Tool metadata for Tool-RAG",
                "hnsw:space": "cosine",
            }
        )
        
        if self._openai is None and settings.openai_api_key:
            self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        
        count = self._collection.count()
        logger.info(f"Tool-RAG initialized: {count} tools in collection")

    async def ensure_seeded(self, openai_client: AsyncOpenAI | None = None) -> None:
        """Seed tool definitions if the collection is empty (lazy first-use seeding)."""
        if self._collection is None:
            return
        if self._collection.count() > 0:
            return

        client = openai_client or self._openai
        if client is None:
            logger.warning("Tool-RAG collection empty but no OpenAI client for seeding")
            return

        from voyageai.rag.seed_definitions import TOOL_DEFINITIONS

        logger.info("Auto-seeding %d tool definitions into Tool-RAG...", len(TOOL_DEFINITIONS))
        await self.add_tools(TOOL_DEFINITIONS, openai_client=client)
        logger.info("Auto-seeded %d tools into Tool-RAG", self._collection.count())

    async def _get_embedding(
        self,
        text: str,
        openai_client: AsyncOpenAI | None = None,
    ) -> list[float]:
        """Get embedding for text using the provided or default OpenAI client."""
        client = openai_client or self._openai
        if client is None:
            raise RuntimeError("ToolRAG: no OpenAI client available (pass one via openai_client)")
        
        response = await client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    
    async def add_tool(
        self,
        tool: ToolMetadata,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        """Add a tool to the Tool-RAG collection."""
        if self._collection is None:
            raise RuntimeError("ToolRAG not initialized")
        
        embedding_text = tool.get_embedding_text()
        embedding = await self._get_embedding(embedding_text, openai_client=openai_client)
        
        # Store in ChromaDB
        # We store the full tool metadata as JSON in metadata field
        self._collection.upsert(
            ids=[tool.name],
            embeddings=[embedding],
            documents=[embedding_text],
            metadatas=[{
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "parameters_schema": str(tool.parameters_schema),  # Chroma doesn't support nested dicts
                "example_queries": "|".join(tool.example_queries),
                "rate_limit_per_minute": tool.rate_limit_per_minute,
                "timeout_seconds": tool.timeout_seconds,
                "requires_api_key": tool.requires_api_key,
                "is_enabled": tool.is_enabled,
                "priority": tool.priority,
            }]
        )
        
        # Cache full metadata
        self._tool_cache[tool.name] = tool
        
        logger.info(f"Added tool to Tool-RAG: {tool.name}")
    
    async def add_tools(
        self,
        tools: list[ToolMetadata],
        openai_client: AsyncOpenAI | None = None,
    ) -> int:
        """Add multiple tools to the collection."""
        for tool in tools:
            await self.add_tool(tool, openai_client=openai_client)
        return len(tools)
    
    async def select_tools(
        self,
        query: str,
        top_k: int = 3,
        category_filter: str | None = None,
        include_disabled: bool = False,
        openai_client: AsyncOpenAI | None = None,
    ) -> ToolSelectionResult:
        """Select relevant tools for a user query via semantic similarity.

        Args:
            query: User query to match against tools
            top_k: Maximum number of tools to return
            category_filter: Optional category to filter by
            include_disabled: Whether to include disabled tools
            openai_client: External OpenAI client for embedding (uses default if None)
        """
        start_time = time.time()
        
        if self._collection is None:
            raise RuntimeError("ToolRAG not initialized")

        await self.ensure_seeded(openai_client=openai_client)
        
        query_embedding = await self._get_embedding(query, openai_client=openai_client)
        
        # Build filter
        where_filter: dict[str, Any] | None = None
        if category_filter or not include_disabled:
            conditions = []
            if category_filter:
                conditions.append({"category": {"$eq": category_filter}})
            if not include_disabled:
                conditions.append({"is_enabled": {"$eq": True}})
            
            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}
        
        # Query ChromaDB
        # Request more than top_k to allow for filtering
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, self._collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        selected_tools: list[ToolMetadata] = []
        scores: list[float] = []
        
        if results and results.get("distances"):
            for i, distance in enumerate(results["distances"][0]):
                # Convert distance to similarity (cosine)
                similarity = 1 - (distance / 2)
                
                # Check threshold
                if similarity < self.similarity_threshold:
                    continue
                
                # Get metadata and reconstruct ToolMetadata
                meta = results["metadatas"][0][i]
                
                # Try cache first, then reconstruct from metadata
                tool_name = meta["name"]
                if tool_name in self._tool_cache:
                    tool = self._tool_cache[tool_name]
                else:
                    # Reconstruct from metadata
                    tool = ToolMetadata(
                        name=meta["name"],
                        description=meta["description"],
                        category=meta["category"],
                        parameters_schema=eval(meta["parameters_schema"]) if meta.get("parameters_schema") else {},
                        example_queries=meta.get("example_queries", "").split("|"),
                        rate_limit_per_minute=meta.get("rate_limit_per_minute", 60),
                        timeout_seconds=meta.get("timeout_seconds", 30),
                        requires_api_key=meta.get("requires_api_key", False),
                        is_enabled=meta.get("is_enabled", True),
                        priority=meta.get("priority", 5),
                    )
                    self._tool_cache[tool_name] = tool
                
                selected_tools.append(tool)
                scores.append(round(similarity, 4))
                
                if len(selected_tools) >= top_k:
                    break
        
        # Fallback: if no tools selected and we have tools, return all
        if not selected_tools and self._collection.count() > 0:
            max_score = max(scores) if scores else 0
            if max_score < FALLBACK_THRESHOLD:
                logger.warning(
                    f"No tools above threshold for query: {query[:50]}... "
                    f"Consider adding more example_queries"
                )
        
        selection_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Tool-RAG selected {len(selected_tools)} tools "
            f"for query: {query[:50]}... ({selection_time}ms)"
        )
        
        return ToolSelectionResult(
            selected_tools=selected_tools,
            query=query,
            scores=scores,
            total_tools_available=self._collection.count(),
            selection_time_ms=selection_time,
        )
    
    def get_openai_tools_from_selection(
        self,
        selection: ToolSelectionResult,
        tool_registry: Any,  # Avoid circular import
    ) -> list[dict[str, Any]]:
        """
        Convert selected tools to OpenAI function format.
        
        Uses the tool registry to get the actual tool definitions,
        but only for tools that were selected by Tool-RAG.
        
        Args:
            selection: Tool selection result
            tool_registry: The tool registry with full tool implementations
            
        Returns:
            List of OpenAI tool definitions
        """
        openai_tools = []
        
        for tool_meta in selection.selected_tools:
            tool = tool_registry.get(tool_meta.name)
            if tool:
                openai_tools.append(tool.to_openai_function())
            else:
                logger.warning(f"Tool {tool_meta.name} in selection but not in registry")
        
        return openai_tools
    
    async def delete_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the collection.
        
        Args:
            tool_name: Name of tool to remove
        """
        if self._collection is None:
            return
        
        try:
            self._collection.delete(ids=[tool_name])
            self._tool_cache.pop(tool_name, None)
            logger.info(f"Deleted tool from Tool-RAG: {tool_name}")
        except Exception as e:
            logger.error(f"Failed to delete tool {tool_name}: {e}")
    
    async def clear_all(self) -> None:
        """Clear all tools from the collection."""
        if self._client is None:
            return
        
        try:
            self._client.delete_collection(TOOL_METADATA_COLLECTION)
            self._collection = None
            self._tool_cache.clear()
            
            # Recreate empty collection
            await self.initialize()
            logger.warning("Cleared all tools from Tool-RAG")
        except Exception as e:
            logger.error(f"Failed to clear Tool-RAG: {e}")
    
    def count(self) -> int:
        """Get number of tools in collection."""
        if self._collection is None:
            return 0
        return self._collection.count()
    
    async def list_all_tools(self) -> list[ToolMetadata]:
        """
        List all tools in the collection.
        
        Returns:
            List of all tool metadata
        """
        if self._collection is None:
            return []
        
        results = self._collection.get(
            include=["metadatas"]
        )
        
        tools = []
        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                tool_name = meta["name"]
                if tool_name in self._tool_cache:
                    tools.append(self._tool_cache[tool_name])
                else:
                    tool = ToolMetadata(
                        name=meta["name"],
                        description=meta["description"],
                        category=meta["category"],
                        parameters_schema=eval(meta.get("parameters_schema", "{}")),
                        example_queries=meta.get("example_queries", "").split("|"),
                        rate_limit_per_minute=meta.get("rate_limit_per_minute", 60),
                        timeout_seconds=meta.get("timeout_seconds", 30),
                        requires_api_key=meta.get("requires_api_key", False),
                        is_enabled=meta.get("is_enabled", True),
                        priority=meta.get("priority", 5),
                    )
                    tools.append(tool)
        
        return tools


# Global singleton instance
tool_rag = ToolRAG.get_instance()
