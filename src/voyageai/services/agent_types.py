"""Shared types, constants, and helpers for the agent service.

Contains AgentResponse, prompt templates, cost tracking, and
shared helper functions used by ResponsesAgentService and the
worker/resilience pipeline.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from voyageai.schemas.itinerary import StructuredItinerary
from voyageai.schemas.tool import ToolCallTrace
from voyageai.schemas.tool_metadata import ToolSelectionResult

logger = logging.getLogger(__name__)

# Type alias for the progress callback function injected by the worker.
# Signature: async callback(event_type: str, data: dict) -> None
ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call for cost tracking."""
    label: str  # e.g. "tool_calling_iter_1", "final_generation", "analysis"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0


# Model pricing (USD per 1M tokens, as of 2025-2026)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "o4-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "o1-mini": (3.00, 12.00),
    "o1": (15.00, 60.00),
    "text-embedding-3-small": (0.02, 0.0),
}


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a single LLM call."""
    key = model.lower()
    # Find best matching pricing key
    pricing = _MODEL_PRICING.get(key)
    if not pricing:
        for k, v in _MODEL_PRICING.items():
            if k in key or key.startswith(k.split("-")[0]):
                pricing = v
                break
    if not pricing:
        pricing = (0.15, 0.60)  # default to gpt-4o-mini
    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return round(input_cost + output_cost, 6)


@dataclass
class AgentResponse:
    """Response from the agent including tool trace and selection info."""

    itinerary: StructuredItinerary | None = None
    tool_trace: list[ToolCallTrace] = field(default_factory=list)
    raw_response: str = ""
    success: bool = True
    error: str | None = None
    total_tokens: int = 0
    processing_time_ms: int = 0
    # Module 10: Tool-RAG information
    tool_selection: ToolSelectionResult | None = None
    selected_tool_names: list[str] = field(default_factory=list)
    # Observability: per-call cost breakdown
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    total_cost_usd: float = 0.0


# Core tools that are always included when Tool-RAG is active.
# These tools are fundamental to virtually every travel planning request.
# RAG dynamically selects additional tools on top of these.
CORE_TOOLS: set[str] = {
    "geocode_location",   # almost every tool needs coordinates first
    "get_weather_forecast",  # weather is relevant for nearly all travel queries
}

# When the total number of registered tools is below this threshold,
# skip Tool-RAG entirely and give all tools to the LLM.  The token
# overhead of ~15 tool schemas is negligible in a 128K context window,
# and eliminating RAG avoids the risk of missing critical tools.
TOOL_RAG_SKIP_THRESHOLD: int = 20


_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are an expert travel planner assistant with access to real-time tools.

Today's date is {today}. Use this as the reference date when the user says "next week", "this month", etc.
When calling weather, holiday, or flight tools, always use dates in or near {current_year} unless the user specifies otherwise.

Your goal is to create detailed, practical travel itineraries. You have access to the following tools:
{tool_list}

IMPORTANT WORKFLOW:
1. First, geocode the destination to get coordinates
2. Then use tools that need coordinates (weather, distance, attractions, restaurants)
3. Use place-search tools for high-quality restaurant and hotel recommendations with ratings
4. Use googlemaps__get_place_details to get official website URLs, phone numbers, and reviews for key places
5. Use flight search when the user mentions flying or needs flight info between cities
6. Use web search for destination guides, travel tips, and up-to-date info
7. Check holidays for the destination country
8. If Xiaohongshu (小红书) travel content is provided in the user message, USE it to enrich the itinerary:
   - Extract specific restaurant names, ticket prices, local tips, and hidden gems from the content.
   - Include the Xiaohongshu post URLs in source_links for relevant activities.
   - Do NOT call xiaohongshu tools yourself — the content is already pre-fetched for you.
9. Consider currency conversion for budget
10. Finally, generate a comprehensive itinerary

QUALITY FILTERING:
- Prioritize highly-rated places (4.0+ stars on Google Maps, 7.0+ on Foursquare).
- Do NOT recommend places with Google Maps ratings below 3.5 unless they are the only option for a specific category.
- When presenting places, mention the rating so users can make informed decisions.
- Sort recommendations by rating when possible — best-rated first.

TOOL FALLBACK & ERROR HANDLING:
- If a tool returns an authentication error (invalid key, expired, 401, 403, blocked), do NOT retry it. Switch to an alternative tool immediately.
- If a place-search tool returns zero results, try a different one or use web search as a fallback.
- For island, rural, or remote destinations (e.g., Hawaii, Maldives, rural countryside), use LARGER search radii for place/restaurant/attraction tools (10000-50000 meters).
  The geocoded center of an island/region may be far from any populated area.
- NEVER retry the exact same tool call with the same parameters if it just failed.

When generating the final itinerary:
- Include specific times for each activity
- Consider weather conditions when planning outdoor activities
- Account for holidays (some attractions may be closed)
- Provide practical budget estimates in local currency
- Use real data from tool results (attractions, restaurants, flights, etc.)
- Include sunrise/sunset times for photography opportunities

TRAVEL TIPS & REMINDERS (IMPORTANT):
In addition to the simple "tips" array, populate "travel_tips" with structured tips:
Each travel_tip has: {{"category": "booking|closure|dress_code|safety|logistics|budget|cultural", "message": "...", "priority": "high|medium|low", "applies_to": "act-day1-001 or null", "advance_days": number or null}}

You MUST include structured tips for:
- Booking deadlines: If a place requires advance booking, specify how many days ahead (advance_days field). Category: "booking", priority: "high".
- Closure days: If a museum, temple, or attraction is closed on certain days (e.g., Mondays). Category: "closure", priority: "high".
- Dress codes: If a temple, church, or venue requires specific clothing (long sleeves, no shorts, head covering). Category: "dress_code", priority: "high".
- Safety reminders: Scam warnings, health advisories, altitude sickness, water safety. Category: "safety".
- Logistics: Early arrival tips, last entry times, peak hours to avoid. Category: "logistics".
- Budget tips: Free entry days, combo ticket deals, money-saving strategies. Category: "budget".
- Cultural etiquette: Tipping customs, shoe removal, photography rules. Category: "cultural".

SOURCE LINKS & WEBSITE URLS (IMPORTANT):
For each activity in the final itinerary, you MUST populate:
- "website_url": The official website URL for the attraction/restaurant if found in tool results (e.g., from Foursquare, OpenStreetMap, or web search). Set to null if not available.
- "source_links": An array of reference links where users can learn more. Each link has:
  {{"title": "display text", "url": "https://...", "source": "official|xiaohongshu|foursquare|google_maps|web_search", "snippet": "optional brief description"}}

  Populate source_links from:
  - Tool results that return website URLs (source: "official" or "foursquare")
  - Xiaohongshu posts (source: "xiaohongshu") — For each Xiaohongshu note you retrieved via get_feed_detail,
    construct the URL as https://www.xiaohongshu.com/explore/{{note_id}} where note_id is the "id"/"noteId" field.
    Use the note's "title" or "displayTitle" as the link title. Include a snippet summarizing the
    post's key recommendations (e.g., "798 likes · 门票26欧, 彩色玻璃窗美得像天堂"). You MUST include
    at least 1-2 Xiaohongshu links in source_links when xiaohongshu results are available.
    Distribute them across activities that the Xiaohongshu post discusses.
  - Web search results (source: "web_search")
  - Google Maps links (source: "google_maps")

  Even if a tool doesn't return a direct URL, you can construct useful links (e.g., Google Maps search URL for a location).

MULTI-OPTION ALTERNATIVES (IMPORTANT):
For each day, generate 1-2 alternative full-day schedules under the "alternatives" field.
Each alternative is an array of activities (same structure as the primary "activities" array) representing a complete day plan.
Guidelines:
- The primary "activities" array is the RECOMMENDED schedule.
- Each alternative should highlight a different aspect of the destination (e.g., foodie focus, art & culture, off-the-beaten-path, family-friendly, nightlife).
- Give each alternative a distinctive flavor — suggest unique and interesting places that differ from the primary schedule.
- Keep the same time structure (similar start/end times) so alternatives are easily swappable.
- Include 2-4 activities per alternative, just like the primary schedule.

Example "alternatives" field for a single day:
"alternatives": [
  [
    {{"activity_id": "alt1-day1-001", "time": "09:00-11:30", "title": "...", ...}},
    {{"activity_id": "alt1-day1-002", "time": "12:00-13:30", "title": "...", ...}}
  ],
  [
    {{"activity_id": "alt2-day1-001", "time": "09:00-11:00", "title": "...", ...}},
    {{"activity_id": "alt2-day1-002", "time": "11:30-13:00", "title": "...", ...}}
  ]
]

REFERENCE ITINERARIES (RAG):
When your RAG search results include documents with source="reference" or relevance_weight="high",
these are curated reference itineraries imported by the user (from travel blogs, articles, or personal notes).
- PRIORITIZE recommendations from reference documents over generic search results.
- Incorporate specific places, restaurants, and tips mentioned in reference content.
- Cite the reference source in source_links when using its recommendations.
- If reference content conflicts with other data, prefer the reference unless it is clearly outdated.

ITINERARY EDITING:
When the user asks to edit an existing itinerary (e.g., "Day 1: add a museum", "Day 2: remove shopping", "replace X with Y"):
1. Parse the edit operation: ADD, REMOVE, REPLACE, or REORDER
2. Identify the target: specific day number + activity (if mentioned)
3. If adding a new activity, use tools to search for suitable places
4. Generate a COMPLETE updated itinerary (all days, not just the changed day)
5. Keep unchanged days/activities exactly as they were
6. Adjust times and schedules when activities are added or removed

Always call relevant tools before generating the final itinerary to ensure accuracy."""


def _build_system_prompt(tool_descriptions: list[dict[str, str]]) -> str:
    """Build the agent system prompt with a dynamically generated tool list.

    The current date is injected so the LLM uses correct years for
    weather forecasts, holidays, flight searches, etc.

    Args:
        tool_descriptions: List of {"name": ..., "description": ...} dicts,
            typically from ``tool_registry.get_tool_descriptions()`` or
            from a Tool-RAG selection.
    """
    from datetime import date

    tool_names = [t["name"] for t in tool_descriptions]
    tool_lines = "\n".join(
        f"- {t['name']}: {t['description']}" for t in tool_descriptions
    )
    today = date.today()
    logger.info(
        "System prompt built with %d tools: %s",
        len(tool_names), ", ".join(tool_names),
    )
    return _AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        tool_list=tool_lines,
        today=today.isoformat(),
        current_year=today.year,
    )


# Patterns that indicate a permanent / non-retriable tool failure.
# If a tool error matches any of these, the agent should NOT retry it.
_PERMANENT_ERROR_PATTERNS: list[str] = [
    "invalid",
    "expired",
    "blocked",
    "unauthorized",
    "forbidden",
    "api key",
    "401",
    "403",
    "not configured",
    "quota exceeded",
]


def _is_permanent_error(error: str | None) -> bool:
    """Return True if the tool error indicates a permanent failure."""
    if not error:
        return False
    lower = error.lower()
    return any(pat in lower for pat in _PERMANENT_ERROR_PATTERNS)
