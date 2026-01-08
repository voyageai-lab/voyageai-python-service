"""
Unit tests for RAG (Retrieval-Augmented Generation) components.

Tests cover:
1. RRF fusion algorithm
2. ChromaDB vector search (with real API calls)
3. Query optimization
4. Reranking strategies
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from voyageai.rag.fusion import rrf_merge, rrf_score, simple_interleave
from voyageai.rag.chroma import ChromaClient
from voyageai.rag.reranker import RerankStrategy, rerank_by_diversity


# ============================================================================
# RRF Fusion Tests
# ============================================================================

class TestRRFScore:
    """Tests for individual RRF score calculation."""
    
    def test_rrf_score_rank_0(self):
        """First rank should have highest score."""
        score = rrf_score(0, k=60)
        assert score == 1 / 61  # 1/(60+0+1)
    
    def test_rrf_score_rank_decreases(self):
        """Higher ranks should have lower scores."""
        score_0 = rrf_score(0)
        score_1 = rrf_score(1)
        score_5 = rrf_score(5)
        
        assert score_0 > score_1 > score_5
    
    def test_rrf_score_different_k(self):
        """Different k values affect score magnitude."""
        score_k30 = rrf_score(0, k=30)
        score_k60 = rrf_score(0, k=60)
        
        assert score_k30 > score_k60  # Smaller k = higher scores


class TestRRFMerge:
    """Tests for RRF merge algorithm."""
    
    def test_merge_single_list(self):
        """Single list should return as-is with RRF scores."""
        results = [
            {"content": "A", "score": 0.9},
            {"content": "B", "score": 0.8},
        ]
        
        merged = rrf_merge([results])
        
        assert len(merged) == 2
        assert merged[0]["content"] == "A"
        assert "rrf_score" in merged[0]
    
    def test_merge_overlapping_results(self):
        """Documents in both lists should have higher scores."""
        list1 = [
            {"content": "Both lists", "score": 0.9},
            {"content": "Only list 1", "score": 0.8},
        ]
        list2 = [
            {"content": "Both lists", "score": 0.9},  # Same doc
            {"content": "Only list 2", "score": 0.8},
        ]
        
        merged = rrf_merge([list1, list2])
        
        # "Both lists" should be first (appears in both)
        assert merged[0]["content"] == "Both lists"
        assert merged[0]["rrf_score"] > merged[1]["rrf_score"]
    
    def test_merge_with_weights(self):
        """Weights should affect ranking."""
        list1 = [{"content": "List 1 top", "score": 0.9}]
        list2 = [{"content": "List 2 top", "score": 0.9}]
        
        # Weight list1 higher
        merged = rrf_merge([list1, list2], weights=[0.8, 0.2])
        
        # List1's result should be first due to higher weight
        assert merged[0]["content"] == "List 1 top"
    
    def test_merge_empty_lists(self):
        """Empty input should return empty output."""
        assert rrf_merge([]) == []
        assert rrf_merge([[], []]) == []
    
    def test_merge_deduplication(self):
        """Duplicate content should be merged."""
        list1 = [
            {"content": "Same content here", "search_type": "vector"},
        ]
        list2 = [
            {"content": "Same content here", "search_type": "bm25"},
        ]
        
        merged = rrf_merge([list1, list2])
        
        assert len(merged) == 1
        assert "vector" in merged[0]["fusion_sources"]
        assert "bm25" in merged[0]["fusion_sources"]


class TestSimpleInterleave:
    """Tests for simple interleaving algorithm."""
    
    def test_interleave_basic(self):
        """Basic interleaving should alternate results."""
        list1 = [{"content": "A"}, {"content": "C"}]
        list2 = [{"content": "B"}, {"content": "D"}]
        
        result = simple_interleave([list1, list2], max_results=4)
        
        # Should interleave: A, B, C, D
        contents = [r["content"] for r in result]
        assert contents == ["A", "B", "C", "D"]
    
    def test_interleave_deduplication(self):
        """Duplicates should be skipped."""
        list1 = [{"content": "Same"}, {"content": "A"}]
        list2 = [{"content": "Same"}, {"content": "B"}]
        
        result = simple_interleave([list1, list2], max_results=3)
        
        # "Same" should only appear once
        contents = [r["content"] for r in result]
        assert contents.count("Same") == 1


# ============================================================================
# ChromaDB Tests
# ============================================================================

class TestChromaClient:
    """Tests for ChromaDB client."""
    
    @pytest.fixture
    def client(self):
        """Create a client with test collection."""
        return ChromaClient(
            persist_directory="./data/test_chroma",
            collection_name="test_collection"
        )
    
    @pytest.mark.asyncio
    async def test_initialize(self, client):
        """Test client initialization."""
        await client.initialize()
        assert client._collection is not None
    
    @pytest.mark.asyncio
    async def test_add_and_search(self, client):
        """Test adding documents and searching."""
        await client.initialize()
        
        # Clean up first
        await client.delete_collection()
        await client.initialize()
        
        # Add test documents
        docs = [
            "Cherry blossom season in Tokyo is beautiful",
            "The Eiffel Tower is in Paris",
            "New York has amazing pizza"
        ]
        metadatas = [
            {"destination": "Tokyo"},
            {"destination": "Paris"},
            {"destination": "New York"}
        ]
        
        count = await client.add_documents(docs, metadatas)
        assert count == 3
        
        # Search
        results = await client.search("Tokyo cherry blossoms", top_k=2)
        
        assert len(results) > 0
        assert results[0]["metadata"]["destination"] == "Tokyo"
        
        # Cleanup
        await client.delete_collection()


# ============================================================================
# Reranker Tests
# ============================================================================

class TestReranker:
    """Tests for reranking strategies."""
    
    @pytest.mark.asyncio
    async def test_diversity_reranking(self):
        """Test MMR diversity reranking."""
        results = [
            {"content": "Tokyo guide 1", "metadata": {"destination": "Tokyo", "source": "guide1"}},
            {"content": "Tokyo guide 2", "metadata": {"destination": "Tokyo", "source": "guide1"}},
            {"content": "Paris guide", "metadata": {"destination": "Paris", "source": "guide2"}},
            {"content": "Tokyo guide 3", "metadata": {"destination": "Tokyo", "source": "guide1"}},
        ]
        
        # With pure diversity (lambda=0), Paris should be selected early
        diverse = await rerank_by_diversity(results, top_k=2, lambda_param=0.3)
        
        # Should get at least one result
        assert len(diverse) >= 1
    
    @pytest.mark.asyncio
    async def test_diversity_empty_input(self):
        """Empty input should return empty output."""
        result = await rerank_by_diversity([], top_k=5)
        assert result == []
    
    @pytest.mark.asyncio
    async def test_diversity_fewer_than_topk(self):
        """When results < top_k, return all."""
        results = [{"content": "A"}, {"content": "B"}]
        diverse = await rerank_by_diversity(results, top_k=5)
        assert len(diverse) == 2


# ============================================================================
# Query Optimizer Tests (Mocked)
# ============================================================================

class TestQueryOptimizer:
    """Tests for query optimization (with mocked LLM)."""
    
    @pytest.mark.asyncio
    async def test_expand_query_mocked(self):
        """Test query expansion with mocked LLM."""
        from voyageai.rag.query_optimizer import expand_query
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """best time to visit Tokyo Japan
when to travel to Tokyo
ideal Tokyo trip season"""
        
        with patch("voyageai.rag.query_optimizer.AsyncOpenAI") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await expand_query("best time tokyo")
        
        # Original query should be included
        assert "best time tokyo" in result
        assert len(result) >= 2
    
    @pytest.mark.asyncio
    async def test_generate_hyde_mocked(self):
        """Test HyDE generation with mocked LLM."""
        from voyageai.rag.query_optimizer import generate_hyde
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """The best time to visit Tokyo 
is during spring (March-May) when cherry blossoms bloom, or autumn (October-November) 
for beautiful fall foliage."""
        
        with patch("voyageai.rag.query_optimizer.AsyncOpenAI") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await generate_hyde("best time tokyo")
        
        assert "cherry" in result.lower() or "spring" in result.lower()


# ============================================================================
# Integration Tests
# ============================================================================

class TestHybridSearchIntegration:
    """Integration tests for hybrid search (requires initialized DBs)."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_vector_only(self):
        """Test hybrid search with only vector results."""
        from voyageai.rag.hybrid_search import hybrid_search
        from voyageai.rag.chroma import chroma_client
        
        # Initialize and add test data
        await chroma_client.initialize()
        
        # Clean and add fresh data
        await chroma_client.delete_collection()
        await chroma_client.initialize()
        
        await chroma_client.add_documents(
            documents=["Test document about Tokyo travel"],
            metadatas=[{"destination": "Tokyo"}]
        )
        
        # Search (ES not available, so vector only)
        result = await hybrid_search(
            query="Tokyo travel",
            top_k=1,
            rerank=False
        )
        
        assert result.stats.vector_results >= 1
        
        # Cleanup
        await chroma_client.delete_collection()

