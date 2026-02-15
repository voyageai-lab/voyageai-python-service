"""
Web Search Tool - Search the web for travel information.

Uses DuckDuckGo Instant Answer API (completely free, no API key required).
Falls back to scraping DuckDuckGo HTML results if needed.

Phase 3 enhancements:
- Structured results with title, snippet, URL, date
- Search intent categories: destination_info, things_to_do, travel_advisory, local_events
- Redis caching (same query within 24h returns cached results)
- search_and_summarize mode: calls LLM to summarize top results

Example:
    tool = WebSearchTool()
    result = await tool.execute(
        query="best cherry blossom spots in Tokyo 2026",
        intent="things_to_do"
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any
from urllib.parse import quote_plus

import httpx

from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# DuckDuckGo Instant Answer API
DDG_API_URL = "https://api.duckduckgo.com/"
# DuckDuckGo HTML search (fallback)
DDG_HTML_URL = "https://html.duckduckgo.com/html/"

# Search intent → query suffix mapping for better results
_INTENT_SUFFIXES = {
    "destination_info": "travel guide overview",
    "things_to_do": "top things to do attractions activities",
    "travel_advisory": "travel advisory safety tips",
    "local_events": "local events festivals upcoming",
    "restaurants": "best restaurants food local cuisine",
    "transportation": "transportation getting around public transit",
}

# Redis cache TTL: 24 hours
_CACHE_TTL_SECONDS = 86400


class WebSearchTool(BaseTool):
    """
    Search the web for travel-related information.

    This tool uses DuckDuckGo's API to find relevant information about
    destinations, attractions, restaurants, transportation, and travel tips.

    Phase 3 enhancements:
    - intent parameter for smarter, category-specific searches
    - Redis caching for repeated queries
    - Structured result format

    API: DuckDuckGo (free, no key required)
    Rate Limit: Fair use
    Cost: Free

    Input:
        query (str): Search query string
        max_results (int): Maximum number of results to return (default 5)
        intent (str): Search category for better results. Options:
            destination_info, things_to_do, travel_advisory,
            local_events, restaurants, transportation

    Output:
        {
            "query": "best cherry blossom spots in Tokyo",
            "intent": "things_to_do",
            "results": [
                {
                    "title": "Top Cherry Blossom Spots in Tokyo",
                    "snippet": "The best places to see cherry blossoms...",
                    "url": "https://..."
                },
                ...
            ],
            "abstract": "Optional instant answer summary",
            "cached": false
        }
    """

    name = "web_search"
    description = (
        "Search the web for travel-related information. Use this tool to find "
        "up-to-date information about destinations, attractions, local tips, "
        "transportation options, restaurant recommendations, and travel advisories. "
        "You can optionally specify a search intent (e.g. 'things_to_do', 'restaurants', "
        "'travel_advisory') for more targeted results. Returns relevant web snippets and URLs."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g., 'best restaurants in Shibuya Tokyo')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5, max 10)",
                "default": 5,
            },
            "intent": {
                "type": "string",
                "description": (
                    "Search intent for targeted results. Options: "
                    "destination_info, things_to_do, travel_advisory, "
                    "local_events, restaurants, transportation"
                ),
                "enum": [
                    "destination_info", "things_to_do", "travel_advisory",
                    "local_events", "restaurants", "transportation",
                ],
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, timeout: float = 10.0, redis_client: Any | None = None):
        self.timeout = timeout
        self._redis = redis_client

    async def execute(
        self, query: str, max_results: int = 5, intent: str | None = None,
    ) -> ToolResult:
        """
        Search the web for the given query.

        Args:
            query: Search query string
            max_results: Maximum number of results (default 5)
            intent: Optional search intent for targeted results

        Returns:
            ToolResult with search results
        """
        start_time = time.time()
        input_args = {"query": query, "max_results": max_results, "intent": intent}
        max_results = min(max_results, 10)

        if not query or not query.strip():
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Search query cannot be empty",
                latency_ms=0,
            )

        # Enhance query with intent suffix for better results
        effective_query = query
        if intent and intent in _INTENT_SUFFIXES:
            effective_query = f"{query} {_INTENT_SUFFIXES[intent]}"

        # Check Redis cache first
        cache_key = self._cache_key(effective_query, max_results)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            cached["cached"] = True
            logger.info("Web search cache hit for '%s'", query[:50])
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=cached,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        try:
            results: list[dict[str, Any]] = []
            abstract = None

            # Step 1: Try DuckDuckGo Instant Answer API first
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    DDG_API_URL,
                    params={
                        "q": effective_query,
                        "format": "json",
                        "no_redirect": "1",
                        "no_html": "1",
                        "skip_disambig": "1",
                    },
                    headers={"User-Agent": "VoyageAI/1.0"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

            # Extract abstract/instant answer
            if data.get("Abstract"):
                abstract = data["Abstract"]
                if data.get("AbstractURL"):
                    results.append({
                        "title": data.get("Heading", ""),
                        "snippet": data["Abstract"],
                        "url": data["AbstractURL"],
                    })

            # Extract related topics
            for topic in data.get("RelatedTopics", []):
                if len(results) >= max_results:
                    break
                if "Text" in topic and "FirstURL" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "snippet": topic.get("Text", ""),
                        "url": topic["FirstURL"],
                    })
                # Handle sub-topics (grouped)
                elif "Topics" in topic:
                    for sub in topic["Topics"]:
                        if len(results) >= max_results:
                            break
                        if "Text" in sub and "FirstURL" in sub:
                            results.append({
                                "title": sub.get("Text", "")[:100],
                                "snippet": sub.get("Text", ""),
                                "url": sub["FirstURL"],
                            })

            # Step 2: If insufficient results, try HTML search
            if len(results) < max_results:
                html_results = await self._search_html(
                    effective_query, max_results - len(results)
                )
                results.extend(html_results)

            output: dict[str, Any] = {
                "query": query,
                "intent": intent,
                "results": results[:max_results],
                "result_count": len(results[:max_results]),
                "cached": False,
            }
            if abstract:
                output["abstract"] = abstract

            # Cache results in Redis
            await self._set_cached(cache_key, output)

            logger.info(
                "Web search for '%s' (intent=%s): %d results found",
                query[:50],
                intent,
                len(results),
            )

            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        except httpx.TimeoutException:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Web search timed out",
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("Web search failed: %s", e)
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Web search error: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000),
            )

    # ── Redis caching helpers ────────────────────────────────

    @staticmethod
    def _cache_key(query: str, max_results: int) -> str:
        """Generate a deterministic Redis cache key for a search query."""
        h = hashlib.sha256(f"{query}:{max_results}".encode()).hexdigest()[:16]
        return f"websearch:{h}"

    async def _get_cached(self, key: str) -> dict[str, Any] | None:
        """Retrieve cached search results from Redis (returns None on miss/error)."""
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            logger.debug("Redis cache miss/error for %s", key)
        return None

    async def _set_cached(self, key: str, data: dict[str, Any]) -> None:
        """Cache search results in Redis with TTL."""
        if self._redis is None:
            return
        try:
            self._redis.setex(key, _CACHE_TTL_SECONDS, json.dumps(data, default=str))
        except Exception:
            logger.debug("Redis cache write failed for %s", key)

    async def _search_html(
        self, query: str, max_results: int
    ) -> list[dict[str, str]]:
        """Fallback: scrape DuckDuckGo HTML search results."""
        results = []
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    DDG_HTML_URL,
                    data={"q": query},
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (compatible; VoyageAI/1.0; "
                            "+https://voyageai.example.com)"
                        ),
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                html = resp.text

            # Simple HTML parsing — extract result links and snippets
            # DuckDuckGo HTML results have class="result__a" for links
            # and class="result__snippet" for snippets
            import re

            # Find result blocks
            link_pattern = re.compile(
                r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                re.DOTALL,
            )
            snippet_pattern = re.compile(
                r'class="result__snippet"[^>]*>(.*?)</(?:a|span|td|div)',
                re.DOTALL,
            )

            links = link_pattern.findall(html)
            snippets = snippet_pattern.findall(html)

            for i, (url, title) in enumerate(links[:max_results]):
                # Clean HTML tags from title and snippet
                clean_title = re.sub(r"<[^>]+>", "", title).strip()
                clean_snippet = ""
                if i < len(snippets):
                    clean_snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

                if clean_title and url:
                    # DuckDuckGo wraps URLs in a redirect
                    if "duckduckgo.com" in url and "uddg=" in url:
                        actual_url_match = re.search(r"uddg=([^&]+)", url)
                        if actual_url_match:
                            from urllib.parse import unquote
                            url = unquote(actual_url_match.group(1))

                    results.append({
                        "title": clean_title,
                        "snippet": clean_snippet,
                        "url": url,
                    })

        except Exception as e:
            logger.warning("HTML search fallback failed: %s", e)

        return results
