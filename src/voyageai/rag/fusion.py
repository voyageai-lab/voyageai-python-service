"""
Result Fusion Algorithms for Hybrid Search

Implements Reciprocal Rank Fusion (RRF) and other fusion strategies
to combine results from multiple search methods (vector + BM25).

RRF (Reciprocal Rank Fusion):
    score(d) = Σ 1 / (k + rank(d))
    
    Where:
    - k is a constant (typically 60)
    - rank(d) is the document's rank in each result list
    - Final score is sum of reciprocal ranks across all lists

Why RRF?
    1. Score-agnostic: Works without normalizing different score scales
    2. Handles missing documents: Only uses ranks where document appears
    3. Robust: Proven effective in many IR benchmarks
    4. Simple: Easy to implement and understand

Reference:
    Cormack, Clarke, Buettcher (2009). "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods"
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# RRF constant - higher values give more weight to lower-ranked documents
# 60 is the standard value from the original paper
RRF_K = 60


def rrf_score(rank: int, k: int = RRF_K) -> float:
    """
    Calculate RRF score for a single rank.
    
    Args:
        rank: 0-indexed position in result list
        k: RRF constant (default 60)
        
    Returns:
        RRF score (higher is better)
    """
    return 1.0 / (k + rank + 1)


def rrf_merge(
    result_lists: list[list[dict[str, Any]]],
    weights: list[float] | None = None,
    content_key: str = "content",
) -> list[dict[str, Any]]:
    """
    Merge multiple result lists using Reciprocal Rank Fusion.
    
    Args:
        result_lists: List of result lists from different search methods
        weights: Optional weights for each result list (e.g., [0.7, 0.3])
        content_key: Key to use for identifying duplicate documents
        
    Returns:
        Merged and ranked list of results
    
    Example:
        vector_results = [{"content": "A", ...}, {"content": "B", ...}]
        bm25_results = [{"content": "B", ...}, {"content": "C", ...}]
        
        merged = rrf_merge([vector_results, bm25_results], weights=[0.7, 0.3])
        # "B" gets highest score (appears in both lists)
    """
    if not result_lists:
        return []
    
    # Default to equal weights
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Score tracking: content_hash -> {score, result, ranks}
    scores: dict[str, dict[str, Any]] = {}
    
    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx]
        
        for rank, result in enumerate(results):
            # Use first 200 chars of content as hash key
            content = result.get(content_key, "")
            content_hash = hash(content[:200]) if content else hash(str(result))
            
            # Calculate weighted RRF score
            score = weight * rrf_score(rank)
            
            if content_hash not in scores:
                scores[content_hash] = {
                    "rrf_score": 0.0,
                    "result": result.copy(),
                    "ranks": {},
                    "sources": [],
                }
            
            scores[content_hash]["rrf_score"] += score
            scores[content_hash]["ranks"][f"list_{list_idx}"] = rank + 1
            scores[content_hash]["sources"].append(result.get("search_type", f"list_{list_idx}"))
    
    # Sort by RRF score
    sorted_results = sorted(
        scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    
    # Format output
    merged = []
    for item in sorted_results:
        result = item["result"]
        result["rrf_score"] = round(item["rrf_score"], 6)
        result["fusion_ranks"] = item["ranks"]
        result["fusion_sources"] = list(set(item["sources"]))
        merged.append(result)
    
    logger.info(
        f"RRF merged {sum(len(r) for r in result_lists)} results into {len(merged)} unique"
    )
    
    return merged


def simple_interleave(
    result_lists: list[list[dict[str, Any]]],
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """
    Simple round-robin interleaving of results.
    
    Less sophisticated than RRF but useful for comparison.
    
    Args:
        result_lists: List of result lists
        max_results: Maximum number of results to return
        
    Returns:
        Interleaved results
    """
    if not result_lists:
        return []
    
    merged = []
    seen = set()
    
    max_len = max(len(r) for r in result_lists)
    
    for i in range(max_len):
        for results in result_lists:
            if i < len(results):
                content = results[i].get("content", "")
                content_hash = hash(content[:200])
                
                if content_hash not in seen:
                    seen.add(content_hash)
                    merged.append(results[i])
                    
                    if len(merged) >= max_results:
                        return merged
    
    return merged

