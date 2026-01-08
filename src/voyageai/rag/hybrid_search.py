"""
Hybrid Search Orchestrator

Combines vector search (ChromaDB) and keyword search (Elasticsearch)
using Reciprocal Rank Fusion (RRF) for optimal retrieval.

Pipeline:
    1. Query Optimization (optional):
       - Query expansion: Generate alternative phrasings
       - HyDE: Generate hypothetical answer for embedding
    
    2. Parallel Search:
       - Vector search: ChromaDB with OpenAI embeddings
       - BM25 search: Elasticsearch keyword matching
    
    3. Fusion:
       - RRF: Merge results using reciprocal rank fusion
       - Deduplication: Remove duplicate documents
    
    4. Reranking (optional):
       - LLM-based: Use LLM to score relevance
       - Diversity: MMR for diverse results

Why Hybrid Search?
    - Vector search: Captures semantic meaning ("good weather" ≈ "pleasant climate")
    - BM25 search: Exact keyword matches ("Tokyo" = "Tokyo", not "Kyoto")
    - Combined: Best of both worlds

Usage:
    results = await hybrid_search(
        query="best time to visit Tokyo",
        top_k=5,
        use_query_expansion=True,
        rerank=True
    )
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from voyageai.rag.chroma import chroma_client
from voyageai.rag.elasticsearch_client import es_client
from voyageai.rag.fusion import rrf_merge
from voyageai.rag.query_optimizer import expand_query, generate_hyde
from voyageai.rag.reranker import RerankStrategy, rerank_results

logger = logging.getLogger(__name__)


@dataclass
class SearchPipelineStats:
    """Statistics from the search pipeline for explainability."""
    
    original_query: str = ""
    expanded_queries: list[str] = field(default_factory=list)
    hyde_text: str | None = None
    
    vector_results: int = 0
    bm25_results: int = 0
    after_fusion: int = 0
    after_rerank: int = 0
    
    vector_latency_ms: int = 0
    bm25_latency_ms: int = 0
    fusion_latency_ms: int = 0
    rerank_latency_ms: int = 0
    total_latency_ms: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": {
                "original": self.original_query,
                "expanded": self.expanded_queries,
                "hyde": self.hyde_text[:100] + "..." if self.hyde_text and len(self.hyde_text) > 100 else self.hyde_text,
            },
            "results": {
                "vector_hits": self.vector_results,
                "bm25_hits": self.bm25_results,
                "after_fusion": self.after_fusion,
                "after_rerank": self.after_rerank,
            },
            "latency_ms": {
                "vector": self.vector_latency_ms,
                "bm25": self.bm25_latency_ms,
                "fusion": self.fusion_latency_ms,
                "rerank": self.rerank_latency_ms,
                "total": self.total_latency_ms,
            }
        }


@dataclass
class HybridSearchResult:
    """Result from hybrid search including stats and documents."""
    
    results: list[dict[str, Any]]
    stats: SearchPipelineStats


async def hybrid_search(
    query: str,
    top_k: int = 5,
    use_query_expansion: bool = False,
    use_hyde: bool = False,
    rerank: bool = True,
    rerank_strategy: RerankStrategy = RerankStrategy.LLM_BASED,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    fetch_k: int = 20,
) -> HybridSearchResult:
    """
    Perform hybrid search combining vector and BM25 search.
    
    Args:
        query: User's search query
        top_k: Number of final results to return
        use_query_expansion: Whether to expand query with variations
        use_hyde: Whether to use HyDE (Hypothetical Document Embeddings)
        rerank: Whether to rerank results with LLM
        rerank_strategy: Reranking strategy to use
        vector_weight: Weight for vector search in RRF (0-1)
        bm25_weight: Weight for BM25 search in RRF (0-1)
        fetch_k: Number of results to fetch from each search method
        
    Returns:
        HybridSearchResult with results and pipeline statistics
    """
    start_time = time.time()
    stats = SearchPipelineStats(original_query=query)
    
    # Initialize clients if needed
    await chroma_client.initialize()
    await es_client.initialize()
    
    # ==================== Query Optimization ====================
    
    search_queries = [query]
    
    if use_query_expansion:
        expanded = await expand_query(query)
        search_queries = expanded
        stats.expanded_queries = expanded
    
    if use_hyde:
        hyde_text = await generate_hyde(query)
        stats.hyde_text = hyde_text
        # Use HyDE for vector search
        vector_query = hyde_text
    else:
        vector_query = query
    
    # ==================== Parallel Search ====================
    
    async def vector_search_task() -> tuple[list[dict], int]:
        """Execute vector search and track timing."""
        t0 = time.time()
        results = await chroma_client.search(vector_query, top_k=fetch_k)
        latency = int((time.time() - t0) * 1000)
        return results, latency
    
    async def bm25_search_task() -> tuple[list[dict], int]:
        """Execute BM25 search and track timing."""
        t0 = time.time()
        # Search with all query variations
        all_results: list[dict] = []
        seen_content: set[str] = set()
        
        for q in search_queries[:3]:  # Limit to 3 queries
            results = await es_client.search(q, top_k=fetch_k)
            for r in results:
                content_hash = r.get("content", "")[:100]
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_results.append(r)
        
        latency = int((time.time() - t0) * 1000)
        return all_results[:fetch_k], latency
    
    # Run searches in parallel
    (vector_results, vector_latency), (bm25_results, bm25_latency) = await asyncio.gather(
        vector_search_task(),
        bm25_search_task()
    )
    
    stats.vector_results = len(vector_results)
    stats.bm25_results = len(bm25_results)
    stats.vector_latency_ms = vector_latency
    stats.bm25_latency_ms = bm25_latency
    
    logger.info(
        f"Search returned: {len(vector_results)} vector, {len(bm25_results)} BM25 "
        f"(vector: {vector_latency}ms, bm25: {bm25_latency}ms)"
    )
    
    # ==================== Fusion ====================
    
    fusion_start = time.time()
    
    if vector_results or bm25_results:
        merged = rrf_merge(
            [vector_results, bm25_results],
            weights=[vector_weight, bm25_weight]
        )
    else:
        merged = []
    
    stats.after_fusion = len(merged)
    stats.fusion_latency_ms = int((time.time() - fusion_start) * 1000)
    
    # ==================== Reranking ====================
    
    rerank_start = time.time()
    
    if rerank and merged and rerank_strategy != RerankStrategy.NONE:
        final_results = await rerank_results(
            query=query,
            results=merged,
            strategy=rerank_strategy,
            top_k=top_k
        )
    else:
        final_results = merged[:top_k]
    
    stats.after_rerank = len(final_results)
    stats.rerank_latency_ms = int((time.time() - rerank_start) * 1000)
    
    # ==================== Finalize ====================
    
    stats.total_latency_ms = int((time.time() - start_time) * 1000)
    
    logger.info(
        f"Hybrid search complete: {len(final_results)} results, "
        f"total {stats.total_latency_ms}ms"
    )
    
    return HybridSearchResult(
        results=final_results,
        stats=stats
    )


async def simple_vector_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Simple vector-only search (no hybrid, no reranking).
    
    Useful for:
    - Quick searches where speed is critical
    - Testing vector search in isolation
    - When Elasticsearch is not available
    """
    await chroma_client.initialize()
    return await chroma_client.search(query, top_k=top_k)


async def simple_bm25_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Simple BM25-only search (no hybrid, no reranking).
    
    Useful for:
    - Exact keyword matching
    - Testing BM25 search in isolation
    - When ChromaDB is not available
    """
    await es_client.initialize()
    return await es_client.search(query, top_k=top_k)

