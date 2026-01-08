"""
Query Optimization Module

Implements query expansion and HyDE (Hypothetical Document Embeddings)
to improve retrieval quality.

Techniques:
1. Query Expansion: Generate semantically similar queries
2. Query Rewriting: Make queries more specific and search-friendly
3. HyDE: Generate hypothetical answer to use as query

Why Query Optimization?
    - User queries are often ambiguous or incomplete
    - Multiple phrasings can find different relevant documents
    - HyDE bridges the gap between query and document language
"""

import logging

from openai import AsyncOpenAI

from voyageai.config import settings

logger = logging.getLogger(__name__)


async def expand_query(query: str, num_expansions: int = 3) -> list[str]:
    """
    Expand the user query with alternative phrasings.
    
    Uses LLM to generate semantically similar queries that might
    match different documents in the corpus.
    
    Args:
        query: Original user query
        num_expansions: Number of alternative queries to generate
        
    Returns:
        List of queries (original + expansions)
        
    Example:
        query = "best time visit tokyo"
        expand_query(query) -> [
            "best time visit tokyo",
            "when to visit Tokyo Japan",
            "Tokyo travel season recommendations",
            "ideal month for Tokyo trip"
        ]
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    prompt = f"""Given the travel-related search query: "{query}"

Generate {num_expansions} alternative phrasings that capture the same intent.

Rules:
1. Keep the destination/topic the same
2. Vary the wording naturally
3. Include both formal and casual phrasings
4. Add relevant context words (travel, trip, visit, etc.)

Return only the alternative queries, one per line, no numbering."""

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a travel query expansion specialist. Generate natural alternative phrasings."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=200
        )
        
        expansions = response.choices[0].message.content.strip().split('\n')
        expansions = [e.strip() for e in expansions if e.strip()]
        
        # Include original query at the start
        queries = [query] + expansions[:num_expansions]
        
        logger.info(f"Expanded query '{query[:30]}...' into {len(queries)} variations")
        return queries
        
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return [query]


async def rewrite_query(query: str) -> str:
    """
    Rewrite the query to be more specific and search-friendly.
    
    Improves query quality by:
    - Expanding abbreviations
    - Adding context
    - Fixing grammar
    - Making intent explicit
    
    Args:
        query: Original user query
        
    Returns:
        Optimized query string
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    prompt = f"""Rewrite this travel search query to be more effective for document retrieval:
"{query}"

Make it:
- More specific and detailed
- Include relevant travel context
- Expand any abbreviations
- Use clearer terminology

Return ONLY the rewritten query, no explanation."""

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a travel query optimization expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"Rewrote query: '{query[:30]}...' -> '{rewritten[:50]}...'")
        return rewritten
        
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return query


async def generate_hyde(query: str) -> str:
    """
    Generate a Hypothetical Document Embedding (HyDE) for the query.
    
    HyDE works by:
    1. Generate a hypothetical answer to the query
    2. Use the answer's embedding instead of query embedding
    3. The answer is more similar to actual documents in style/content
    
    This bridges the semantic gap between:
    - Query language ("best time tokyo")
    - Document language ("The optimal period for visiting Tokyo is...")
    
    Args:
        query: User's question
        
    Returns:
        Hypothetical answer text for embedding
        
    Reference:
        Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    prompt = f"""Generate a hypothetical travel guide paragraph that would answer this question:
"{query}"

Write as if you're a travel guidebook author. Include:
- Specific details (dates, seasons, temperatures)
- Practical recommendations
- Local context and tips

Write 2-3 sentences of content that would appear in a travel guide."""

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are writing content for a professional travel guidebook."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        hyde_text = response.choices[0].message.content.strip()
        logger.info(f"Generated HyDE for: '{query[:30]}...'")
        return hyde_text
        
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return query

