"""
Tests for Module 10: Tool-RAG and Rate Limiting

These tests cover:
1. ToolMetadata schema validation
2. ToolRAG tool selection
3. Token bucket rate limiter
4. Integration with AgentService
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voyageai.schemas.tool_metadata import (
    RateLimitConfig,
    ToolMetadata,
    ToolSelectionResult,
)
from voyageai.tools.rate_limiter import (
    RateLimitExceeded,
    RateLimiter,
    TokenBucket,
)


# ============================================================================
# ToolMetadata Schema Tests
# ============================================================================

class TestToolMetadataSchema:
    """Test ToolMetadata Pydantic model."""
    
    def test_minimal_valid_metadata(self):
        """Test creating metadata with minimal required fields."""
        meta = ToolMetadata(
            name="test_tool",
            description="A test tool",
            category="info",
            example_queries=["How do I use this?"],
        )
        
        assert meta.name == "test_tool"
        assert meta.description == "A test tool"
        assert meta.category == "info"
        assert len(meta.example_queries) == 1
        # Check defaults
        assert meta.rate_limit_per_minute == 60
        assert meta.timeout_seconds == 30
        assert meta.requires_api_key is False
        assert meta.is_enabled is True
        assert meta.priority == 5
    
    def test_full_metadata(self):
        """Test creating metadata with all fields."""
        meta = ToolMetadata(
            name="weather_tool",
            description="Get weather forecast",
            category="external_api",
            parameters_schema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
            example_queries=[
                "What's the weather?",
                "Will it rain?",
                "Temperature forecast",
            ],
            rate_limit_per_minute=30,
            timeout_seconds=10,
            requires_api_key=False,
            is_enabled=True,
            priority=8,
        )
        
        assert meta.rate_limit_per_minute == 30
        assert meta.timeout_seconds == 10
        assert meta.priority == 8
        assert len(meta.example_queries) == 3
    
    def test_get_embedding_text(self):
        """Test generating embedding text."""
        meta = ToolMetadata(
            name="geocode",
            description="Convert location to coordinates",
            category="info",
            example_queries=["Where is Tokyo?", "Find coordinates"],
        )
        
        text = meta.get_embedding_text()
        
        assert "geocode" in text
        assert "Convert location to coordinates" in text
        assert "Where is Tokyo?" in text
        assert "Find coordinates" in text
    
    def test_rate_limit_bounds(self):
        """Test rate limit validation bounds."""
        # Valid lower bound
        meta = ToolMetadata(
            name="test",
            description="test",
            category="info",
            example_queries=["test"],
            rate_limit_per_minute=1,
        )
        assert meta.rate_limit_per_minute == 1
        
        # Valid upper bound
        meta = ToolMetadata(
            name="test",
            description="test",
            category="info",
            example_queries=["test"],
            rate_limit_per_minute=1000,
        )
        assert meta.rate_limit_per_minute == 1000
    
    def test_timeout_bounds(self):
        """Test timeout validation bounds."""
        # Valid lower bound
        meta = ToolMetadata(
            name="test",
            description="test",
            category="info",
            example_queries=["test"],
            timeout_seconds=1,
        )
        assert meta.timeout_seconds == 1
        
        # Valid upper bound
        meta = ToolMetadata(
            name="test",
            description="test",
            category="info",
            example_queries=["test"],
            timeout_seconds=120,
        )
        assert meta.timeout_seconds == 120


class TestToolSelectionResult:
    """Test ToolSelectionResult model."""
    
    def test_empty_selection(self):
        """Test creating empty selection result."""
        result = ToolSelectionResult(
            query="test query",
            selected_tools=[],
            scores=[],
            total_tools_available=6,
            selection_time_ms=50,
        )
        
        assert result.query == "test query"
        assert len(result.selected_tools) == 0
        assert result.total_tools_available == 6
    
    def test_selection_with_tools(self):
        """Test creating selection with tools."""
        tool = ToolMetadata(
            name="weather",
            description="Get weather",
            category="api",
            example_queries=["weather"],
        )
        
        result = ToolSelectionResult(
            query="What's the weather?",
            selected_tools=[tool],
            scores=[0.85],
            total_tools_available=6,
            selection_time_ms=100,
        )
        
        assert len(result.selected_tools) == 1
        assert result.scores[0] == 0.85


# ============================================================================
# Token Bucket Rate Limiter Tests
# ============================================================================

class TestTokenBucket:
    """Test TokenBucket implementation."""
    
    def test_initial_state(self):
        """Test bucket starts full."""
        bucket = TokenBucket(
            tokens=60,
            max_tokens=60,
            refill_rate=1.0,
        )
        
        assert bucket.tokens == 60
        assert bucket.max_tokens == 60
    
    def test_consume_tokens(self):
        """Test consuming tokens."""
        bucket = TokenBucket(
            tokens=10,
            max_tokens=10,
            refill_rate=1.0,
        )
        
        # Consume one token
        assert bucket.consume(1) is True
        assert bucket.tokens < 10
        
        # Consume multiple tokens
        assert bucket.consume(5) is True
    
    def test_consume_exceeds_available(self):
        """Test consuming more tokens than available."""
        bucket = TokenBucket(
            tokens=5,
            max_tokens=10,
            refill_rate=1.0,
        )
        
        # Can't consume 10 when only 5 available
        assert bucket.consume(10) is False
        # Tokens should not have been consumed
        assert bucket.tokens >= 4.9  # Allow for small refill
    
    def test_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(
            tokens=0,
            max_tokens=10,
            refill_rate=10.0,  # 10 tokens per second
            last_refill=time.time() - 1.0,  # 1 second ago
        )
        
        # Refill should add tokens
        bucket.refill()
        
        # Should have ~10 tokens now (10 tokens/sec * 1 sec)
        assert bucket.tokens >= 9.0
    
    def test_refill_caps_at_max(self):
        """Test that refill doesn't exceed max_tokens."""
        bucket = TokenBucket(
            tokens=5,
            max_tokens=10,
            refill_rate=100.0,  # Very high rate
            last_refill=time.time() - 10.0,  # 10 seconds ago
        )
        
        bucket.refill()
        
        # Should cap at max_tokens
        assert bucket.tokens == 10
    
    def test_time_until_available(self):
        """Test calculating time until tokens available."""
        bucket = TokenBucket(
            tokens=0,
            max_tokens=10,
            refill_rate=1.0,  # 1 token per second
        )
        
        # Need to wait for tokens
        wait_time = bucket.time_until_available(5)
        assert 4.0 <= wait_time <= 5.0


class TestRateLimiter:
    """Test RateLimiter class."""
    
    @pytest.fixture
    def limiter(self):
        """Create a fresh rate limiter for each test."""
        return RateLimiter(
            default_max_tokens=10,
            default_refill_rate=1.0,
        )
    
    @pytest.mark.asyncio
    async def test_is_allowed_initially(self, limiter):
        """Test that requests are allowed initially."""
        allowed = await limiter.is_allowed("user1", "tool1")
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, limiter):
        """Test that rate limit is enforced."""
        # Exhaust all tokens
        for _ in range(10):
            await limiter.is_allowed("user1", "tool1")
        
        # Next request should be denied
        allowed = await limiter.is_allowed("user1", "tool1")
        assert allowed is False
    
    @pytest.mark.asyncio
    async def test_per_user_isolation(self, limiter):
        """Test that different users have separate buckets."""
        # Exhaust user1's tokens
        for _ in range(10):
            await limiter.is_allowed("user1", "tool1")
        
        # user2 should still be allowed
        allowed = await limiter.is_allowed("user2", "tool1")
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_per_tool_isolation(self, limiter):
        """Test that different tools have separate buckets."""
        # Exhaust tokens for tool1
        for _ in range(10):
            await limiter.is_allowed("user1", "tool1")
        
        # tool2 should still be allowed
        allowed = await limiter.is_allowed("user1", "tool2")
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_check_and_consume_raises(self, limiter):
        """Test that check_and_consume raises on limit exceeded."""
        # Exhaust tokens
        for _ in range(10):
            await limiter.check_and_consume("user1", "tool1")
        
        # Should raise
        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.check_and_consume("user1", "tool1")
        
        assert exc_info.value.tool_name == "tool1"
        assert exc_info.value.user_id == "user1"
        assert exc_info.value.retry_after > 0
    
    @pytest.mark.asyncio
    async def test_get_status(self, limiter):
        """Test getting rate limit status."""
        # Consume some tokens
        await limiter.is_allowed("user1", "tool1")
        await limiter.is_allowed("user1", "tool1")
        
        status = await limiter.get_status("user1", "tool1")
        
        assert status["user_id"] == "user1"
        assert status["tool_name"] == "tool1"
        assert status["tokens_remaining"] <= 8  # Consumed 2
        assert status["max_tokens"] == 10
    
    @pytest.mark.asyncio
    async def test_reset(self, limiter):
        """Test resetting rate limit."""
        # Exhaust tokens
        for _ in range(10):
            await limiter.is_allowed("user1", "tool1")
        
        # Reset
        await limiter.reset("user1", "tool1")
        
        # Should be allowed again
        allowed = await limiter.is_allowed("user1", "tool1")
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_configure_tool(self, limiter):
        """Test configuring per-tool rate limits."""
        limiter.configure_tool("expensive_tool", max_tokens=5, refill_rate=0.5)
        
        # Should only allow 5 requests
        for _ in range(5):
            allowed = await limiter.is_allowed("user1", "expensive_tool")
            assert allowed is True
        
        # 6th request should be denied
        allowed = await limiter.is_allowed("user1", "expensive_tool")
        assert allowed is False


# ============================================================================
# ToolRAG Tests (Mocked)
# ============================================================================

class TestToolRAGMocked:
    """Test ToolRAG with mocked dependencies."""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client."""
        with patch("voyageai.rag.tool_rag.AsyncOpenAI") as mock:
            client = MagicMock()
            # Create a proper async mock for embeddings
            async def mock_create(*args, **kwargs):
                return MagicMock(
                    data=[MagicMock(embedding=[0.1] * 1536)]
                )
            client.embeddings.create = mock_create
            mock.return_value = client
            yield mock
    
    @pytest.fixture
    def mock_chroma(self):
        """Mock ChromaDB client."""
        with patch("voyageai.rag.tool_rag.chromadb") as mock:
            client = MagicMock()
            collection = MagicMock()
            
            # Mock collection methods
            collection.count.return_value = 6
            collection.query.return_value = {
                "distances": [[0.2, 0.4, 0.6]],  # Cosine distances
                "metadatas": [[
                    {
                        "name": "get_weather_forecast",
                        "description": "Get weather",
                        "category": "external_api",
                        "parameters_schema": "{}",
                        "example_queries": "weather|forecast",
                        "rate_limit_per_minute": 60,
                        "timeout_seconds": 10,
                        "requires_api_key": False,
                        "is_enabled": True,
                        "priority": 8,
                    },
                    {
                        "name": "geocode_location",
                        "description": "Geocode",
                        "category": "info",
                        "parameters_schema": "{}",
                        "example_queries": "location|coordinates",
                        "rate_limit_per_minute": 60,
                        "timeout_seconds": 10,
                        "requires_api_key": False,
                        "is_enabled": True,
                        "priority": 9,
                    },
                    {
                        "name": "convert_currency",
                        "description": "Currency",
                        "category": "external_api",
                        "parameters_schema": "{}",
                        "example_queries": "convert|exchange",
                        "rate_limit_per_minute": 60,
                        "timeout_seconds": 10,
                        "requires_api_key": False,
                        "is_enabled": True,
                        "priority": 7,
                    },
                ]],
                "documents": [["weather doc", "geocode doc", "currency doc"]],
            }
            
            client.get_or_create_collection.return_value = collection
            mock.PersistentClient.return_value = client
            
            yield mock, collection
    
    @pytest.mark.asyncio
    async def test_select_tools_returns_results(self, mock_openai, mock_chroma):
        """Test that select_tools returns relevant tools."""
        from voyageai.rag.tool_rag import ToolRAG
        
        rag = ToolRAG(persist_directory="./test_data")
        await rag.initialize()
        
        result = await rag.select_tools("What's the weather in Tokyo?", top_k=3)
        
        assert result.query == "What's the weather in Tokyo?"
        assert len(result.selected_tools) <= 3
        assert result.total_tools_available == 6
    
    @pytest.mark.asyncio
    async def test_select_tools_scores(self, mock_openai, mock_chroma):
        """Test that scores are calculated correctly."""
        from voyageai.rag.tool_rag import ToolRAG
        
        rag = ToolRAG(persist_directory="./test_data")
        await rag.initialize()
        
        result = await rag.select_tools("test query", top_k=3)
        
        # Scores should be between 0 and 1
        for score in result.scores:
            assert 0 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_add_tool(self, mock_openai, mock_chroma):
        """Test adding a tool to the collection."""
        from voyageai.rag.tool_rag import ToolRAG
        
        _, collection = mock_chroma
        
        rag = ToolRAG(persist_directory="./test_data")
        await rag.initialize()
        
        tool = ToolMetadata(
            name="test_tool",
            description="Test description",
            category="test",
            example_queries=["test query"],
        )
        
        await rag.add_tool(tool)
        
        # Verify upsert was called
        collection.upsert.assert_called_once()


# ============================================================================
# Integration Tests (Require Real Services)
# ============================================================================

@pytest.mark.integration
class TestToolRAGIntegration:
    """
    Integration tests for Tool-RAG.
    
    These tests require:
    - OPENAI_API_KEY environment variable
    - ChromaDB (in-memory for tests)
    
    Run with: pytest -m integration
    """
    
    @pytest.fixture
    async def tool_rag(self, tmp_path):
        """Create Tool-RAG instance with temp storage."""
        from voyageai.rag.tool_rag import ToolRAG
        
        rag = ToolRAG(persist_directory=str(tmp_path / "chroma"))
        await rag.initialize()
        
        # Add sample tools
        tools = [
            ToolMetadata(
                name="get_weather_forecast",
                description="Get weather forecast for a location",
                category="external_api",
                example_queries=[
                    "What's the weather like?",
                    "Will it rain?",
                    "Temperature forecast",
                ],
            ),
            ToolMetadata(
                name="geocode_location",
                description="Convert location name to coordinates",
                category="info",
                example_queries=[
                    "Where is Tokyo?",
                    "Find coordinates",
                    "Location lookup",
                ],
            ),
        ]
        
        for tool in tools:
            await rag.add_tool(tool)
        
        yield rag
        
        # Cleanup
        await rag.clear_all()
    
    @pytest.mark.asyncio
    async def test_weather_query_selects_weather_tool(self, tool_rag):
        """Test that weather-related query selects weather tool."""
        result = await tool_rag.select_tools(
            "What will the weather be like in Paris next week?",
            top_k=2
        )
        
        tool_names = [t.name for t in result.selected_tools]
        assert "get_weather_forecast" in tool_names
    
    @pytest.mark.asyncio
    async def test_location_query_selects_geocode_tool(self, tool_rag):
        """Test that location query selects geocode tool."""
        result = await tool_rag.select_tools(
            "Where exactly is the Eiffel Tower located?",
            top_k=2
        )
        
        tool_names = [t.name for t in result.selected_tools]
        assert "geocode_location" in tool_names
