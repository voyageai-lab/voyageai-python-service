"""
Reranking Module for Advanced RAG

Implements multiple reranking strategies to improve retrieval quality
after initial search (vector + BM25).

Reranking Process:
    1. Get top-N candidates from hybrid search
    2. Score each candidate's relevance to query
    3. Reorder by new relevance scores
    4. Return top-K most relevant

Strategies:
    - LLM_BASED: Use LLM to score relevance (most accurate, slower)
    - CROSS_ENCODER: Use cross-encoder model (balanced)
    - DIVERSITY: MMR for diverse results (reduces redundancy)
    - NONE: No reranking (fastest)

Why Reranking?
    - Initial retrieval is recall-focused (find all relevant docs)
    - Reranking is precision-focused (find most relevant)
    - LLM understands nuanced relevance better than embeddings
"""

import json
import logging
from enum import Enum
from typing import Any

from openai import AsyncOpenAI

from voyageai.config import settings

logger = logging.getLogger(__name__)


class RerankStrategy(str, Enum):
    """Available reranking strategies."""
    
    LLM_BASED = "llm_based"
    DIVERSITY = "diversity"
    NONE = "none"


async def rerank_with_llm(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Use LLM to score and rerank results based on relevance.
    
    The LLM evaluates each document against the query and assigns
    a relevance score from 0-10.
    
    Args:
        query: User query
        results: List of search results
        top_k: Number of top results to return
        
    Returns:
        Reranked results with rerank_score
    """
    if not results:
        return []
    
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Format documents for evaluation (limit to top 10)
    docs_to_rank = results[:10]
    docs_text = ""
    for i, result in enumerate(docs_to_rank):
        content = result.get("content", "")[:400]  # Truncate long content
        title = result.get("metadata", {}).get("title", "")
        docs_text += f"\nDocument {i+1}:"
        if title:
            docs_text += f"\nTitle: {title}"
        docs_text += f"\nContent: {content}...\n"
    
    prompt = f"""Query: "{query}"

Rate the relevance of each document to the travel query on a scale of 0-10.

Scoring criteria:
- 9-10: Directly answers the query with specific details
- 7-8: Highly relevant with useful information
- 5-6: Somewhat relevant, partial answer
- 3-4: Tangentially related
- 0-2: Not relevant to the query

{docs_text}

Return ONLY a JSON object with scores. Format: {{"scores": [score1, score2, ...]}}"""

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a travel information relevance scorer. Be strict but fair."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        scores = parsed.get("scores", [])
        
        # Add scores to results
        for i, score in enumerate(scores):
            if i < len(docs_to_rank):
                docs_to_rank[i]["rerank_score"] = float(score)
        
        # Sort by rerank score
        reranked = sorted(
            docs_to_rank,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True
        )
        
        # Filter out very low scores (< 3.0)
        MIN_RELEVANCE = 3.0
        filtered = [r for r in reranked if r.get("rerank_score", 0) >= MIN_RELEVANCE]
        
        # Keep at least 2 results if we have them
        if len(filtered) < 2 and len(reranked) >= 2:
            filtered = reranked[:2]
        elif len(filtered) == 0 and len(reranked) > 0:
            filtered = reranked[:1]
        
        logger.info(
            f"LLM reranked {len(docs_to_rank)} docs -> "
            f"{len(filtered)} passed threshold ({MIN_RELEVANCE})"
        )
        
        return filtered[:top_k]
        
    except Exception as e:
        logger.error(f"LLM reranking failed: {e}")
        return results[:top_k]


async def rerank_by_diversity(
    results: list[dict[str, Any]],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Rerank using Maximal Marginal Relevance (MMR) for diversity.
    
    MMR balances:
    - Relevance: How relevant is this document?
    - Diversity: How different is it from already selected docs?
    
    Formula:
        MMR = λ * Sim(d, q) - (1-λ) * max(Sim(d, d_i))
        
    Where:
    - λ = 1: Pure relevance (no diversity)
    - λ = 0: Pure diversity (ignore relevance)
    - λ = 0.5: Balanced (default)
    
    Args:
        results: List of search results
        top_k: Number of results to return
        lambda_param: Balance between relevance and diversity
        
    Returns:
        Diversified results
    """
    if len(results) <= top_k:
        return results
    
    # Start with most relevant (first result)
    selected = [results[0]]
    remaining = results[1:]
    
    while len(selected) < top_k and remaining:
        best_score = -float('inf')
        best_idx = 0
        
        for i, candidate in enumerate(remaining):
            # Simple diversity: check metadata differences
            candidate_dest = candidate.get("metadata", {}).get("destination", "")
            candidate_source = candidate.get("metadata", {}).get("source", "")
            
            # Calculate max similarity to already selected
            max_sim = 0.0
            for sel in selected:
                sel_dest = sel.get("metadata", {}).get("destination", "")
                sel_source = sel.get("metadata", {}).get("source", "")
                
                # Simple similarity: same destination or source = similar
                sim = 0.0
                if candidate_dest == sel_dest and candidate_dest:
                    sim += 0.5
                if candidate_source == sel_source and candidate_source:
                    sim += 0.5
                
                max_sim = max(max_sim, sim)
            
            # Get relevance score (from RRF or original score)
            relevance = candidate.get("rrf_score", candidate.get("score", 0))
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        
        selected.append(remaining.pop(best_idx))
    
    logger.info(f"Diversity reranking: selected {len(selected)} diverse results")
    return selected


async def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    strategy: RerankStrategy = RerankStrategy.LLM_BASED,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Rerank search results using the specified strategy.
    
    Args:
        query: User query
        results: Initial search results
        strategy: Reranking strategy to use
        top_k: Number of results to return
        
    Returns:
        Reranked results
    """
    if not results:
        return []
    
    if strategy == RerankStrategy.LLM_BASED:
        return await rerank_with_llm(query, results, top_k)
    elif strategy == RerankStrategy.DIVERSITY:
        return await rerank_by_diversity(results, top_k)
    elif strategy == RerankStrategy.NONE:
        return results[:top_k]
    else:
        return results[:top_k]

