"""
VoyageAI RAG (Retrieval-Augmented Generation) Package

This package implements hybrid search combining:
- Vector search (ChromaDB with OpenAI embeddings)
- BM25 keyword search (Elasticsearch)
- RRF (Reciprocal Rank Fusion) for result merging
- LLM-based reranking for final relevance scoring

Module 10 Additions (Tool-RAG):
- tool_rag.py: Semantic tool selection via RAG

Key Components:
- chroma.py: Vector search using ChromaDB
- elasticsearch_client.py: BM25 search using Elasticsearch
- hybrid_search.py: Combines both search methods with RRF
- query_optimizer.py: Query expansion and HyDE
- reranker.py: LLM-based result reranking
- tool_rag.py: Tool-RAG for dynamic tool selection
"""

from voyageai.rag.hybrid_search import hybrid_search
from voyageai.rag.reranker import RerankStrategy, rerank_results
from voyageai.rag.tool_rag import ToolRAG, tool_rag

__all__ = [
    "hybrid_search",
    "RerankStrategy",
    "rerank_results",
    # Module 10: Tool-RAG
    "ToolRAG",
    "tool_rag",
]

