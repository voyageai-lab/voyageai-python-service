"""
RAG Router - Endpoints for Retrieval-Augmented Generation

Provides endpoints for:
1. Hybrid search (vector + BM25)
2. Document indexing
3. Search diagnostics and stats

These endpoints are used by the agent service and can also
be called directly for testing.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from voyageai.rag.chroma import chroma_client
from voyageai.rag.elasticsearch_client import es_client
from voyageai.rag.hybrid_search import (
    HybridSearchResult,
    hybrid_search,
    simple_bm25_search,
    simple_vector_search,
)
from voyageai.rag.reranker import RerankStrategy

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    """Request for hybrid search."""
    
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    use_query_expansion: bool = Field(default=False, description="Expand query with variations")
    use_hyde: bool = Field(default=False, description="Use HyDE for vector search")
    rerank: bool = Field(default=True, description="Rerank results with LLM")
    rerank_strategy: str = Field(default="llm_based", description="Reranking strategy")
    vector_weight: float = Field(default=0.6, ge=0, le=1, description="Weight for vector search")
    bm25_weight: float = Field(default=0.4, ge=0, le=1, description="Weight for BM25 search")


class SearchResult(BaseModel):
    """Individual search result."""
    
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    search_type: str | None = None
    fusion_sources: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response from hybrid search."""
    
    results: list[SearchResult]
    stats: dict[str, Any]
    total_results: int


class IndexRequest(BaseModel):
    """Request to index documents."""
    
    documents: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Documents to index with content, title, metadata"
    )


class IndexResponse(BaseModel):
    """Response from indexing."""
    
    indexed_vector: int
    indexed_bm25: int
    total: int


class StatsResponse(BaseModel):
    """RAG system statistics."""
    
    vector_documents: int
    bm25_available: bool
    bm25_documents: int | None


# ============================================================================
# Search Endpoints
# ============================================================================

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Perform hybrid search combining vector and BM25 search.
    
    Pipeline:
    1. Query optimization (optional expansion, HyDE)
    2. Parallel search (ChromaDB + Elasticsearch)
    3. RRF fusion
    4. LLM reranking (optional)
    
    Example:
        POST /api/v1/rag/search
        {
            "query": "best time to visit Tokyo",
            "top_k": 5,
            "rerank": true
        }
    """
    try:
        # Convert string strategy to enum
        try:
            strategy = RerankStrategy(request.rerank_strategy)
        except ValueError:
            strategy = RerankStrategy.LLM_BASED
        
        result: HybridSearchResult = await hybrid_search(
            query=request.query,
            top_k=request.top_k,
            use_query_expansion=request.use_query_expansion,
            use_hyde=request.use_hyde,
            rerank=request.rerank,
            rerank_strategy=strategy,
            vector_weight=request.vector_weight,
            bm25_weight=request.bm25_weight,
        )
        
        # Convert to response model
        search_results = []
        for r in result.results:
            search_results.append(SearchResult(
                content=r.get("content", ""),
                metadata=r.get("metadata", {}),
                score=r.get("score"),
                rrf_score=r.get("rrf_score"),
                rerank_score=r.get("rerank_score"),
                search_type=r.get("search_type"),
                fusion_sources=r.get("fusion_sources", []),
            ))
        
        return SearchResponse(
            results=search_results,
            stats=result.stats.to_dict(),
            total_results=len(search_results),
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/vector")
async def vector_search(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
) -> dict[str, Any]:
    """
    Simple vector-only search (no hybrid, no reranking).
    
    Example: /api/v1/rag/search/vector?query=tokyo%20weather&top_k=5
    """
    try:
        results = await simple_vector_search(query, top_k)
        return {
            "query": query,
            "results": results,
            "total": len(results),
            "search_type": "vector"
        }
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/bm25")
async def bm25_search(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
) -> dict[str, Any]:
    """
    Simple BM25-only search (no hybrid, no reranking).
    
    Example: /api/v1/rag/search/bm25?query=cherry%20blossom&top_k=5
    """
    try:
        results = await simple_bm25_search(query, top_k)
        return {
            "query": query,
            "results": results,
            "total": len(results),
            "search_type": "bm25"
        }
    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Indexing Endpoints
# ============================================================================

@router.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest) -> IndexResponse:
    """
    Index documents into both vector and BM25 indexes.
    
    Document format:
    {
        "content": "The best time to visit Tokyo is...",
        "title": "Tokyo Travel Guide",
        "destination": "Tokyo",
        "category": "seasonal",
        "source": "travel_guide.txt"
    }
    
    Example:
        POST /api/v1/rag/index
        {
            "documents": [
                {"content": "...", "title": "...", ...}
            ]
        }
    """
    try:
        # Initialize clients
        await chroma_client.initialize()
        await es_client.initialize()
        
        # Extract content and metadata
        contents = []
        metadatas = []
        for doc in request.documents:
            contents.append(doc.get("content", ""))
            metadatas.append({
                "title": doc.get("title", ""),
                "destination": doc.get("destination", ""),
                "category": doc.get("category", "general"),
                "source": doc.get("source", "unknown"),
            })
        
        # Index to ChromaDB
        vector_count = await chroma_client.add_documents(
            documents=contents,
            metadatas=metadatas,
        )
        
        # Index to Elasticsearch
        bm25_count = await es_client.add_documents(request.documents)
        
        return IndexResponse(
            indexed_vector=vector_count,
            indexed_bm25=bm25_count,
            total=len(request.documents),
        )
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# System Endpoints
# ============================================================================

@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Get RAG system statistics.
    
    Returns document counts and availability status.
    """
    try:
        await chroma_client.initialize()
        await es_client.initialize()
        
        vector_count = chroma_client.count()
        bm25_available = es_client.is_available
        
        bm25_count = None
        if bm25_available:
            try:
                response = await es_client._client.count(index=es_client.index_name)
                bm25_count = response["count"]
            except Exception:
                pass
        
        return StatsResponse(
            vector_documents=vector_count,
            bm25_available=bm25_available,
            bm25_documents=bm25_count,
        )
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset")
async def reset_indexes() -> dict[str, str]:
    """
    Reset both indexes (for testing/development).
    
    WARNING: This deletes all indexed documents!
    """
    try:
        await chroma_client.initialize()
        await chroma_client.delete_collection()
        
        await es_client.initialize()
        await es_client.delete_index()
        
        return {"status": "ok", "message": "Indexes reset successfully"}
        
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

