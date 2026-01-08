"""
VoyageAI RAG (Retrieval-Augmented Generation) Package

This package implements hybrid search combining:
- Vector search (ChromaDB with OpenAI embeddings)
- BM25 keyword search (Elasticsearch)
- RRF (Reciprocal Rank Fusion) for result merging
- LLM-based reranking for final relevance scoring

Key Components:
- chroma.py: Vector search using ChromaDB
- elasticsearch.py: BM25 search using Elasticsearch
- hybrid_search.py: Combines both search methods with RRF
- query_optimizer.py: Query expansion and HyDE
- reranker.py: LLM-based result reranking
"""

from voyageai.rag.hybrid_search import hybrid_search
from voyageai.rag.reranker import RerankStrategy, rerank_results

__all__ = [
    "hybrid_search",
    "RerankStrategy",
    "rerank_results",
]

