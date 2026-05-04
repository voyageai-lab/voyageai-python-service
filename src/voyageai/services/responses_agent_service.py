"""AI Agent Service using the OpenAI Responses API.

Orchestrator that coordinates:
- Tool-RAG (semantic tool selection)
- Xiaohongshu pre-fetch (delegated to ``xiaohongshu_prefetch``)
- Itinerary generation / editing (delegated to ``itinerary_generator``)
- Responses API tool-calling loop
- Failed tool tracking, heartbeat, clarification detection

Shared types (AgentResponse, prompt templates, cost tracking) live in
``agent_types.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any
from urllib.parse import urlparse

from openai import AsyncOpenAI

from voyageai.config import settings
from voyageai.rag.tool_rag import ToolRAG, tool_rag
from voyageai.schemas.tool import ToolCallTrace
from voyageai.schemas.tool_metadata import ToolSelectionResult
from voyageai.services.agent_types import (
    CORE_TOOLS,
    TOOL_RAG_SKIP_THRESHOLD,
    AgentResponse,
    LLMCallRecord,
    ProgressCallback,
    _build_system_prompt,
    _calc_cost,
    _is_permanent_error,
)
from voyageai.services.itinerary_generator import (
    generate_structured_final,
    parse_itinerary,
)
from voyageai.services.xiaohongshu_prefetch import (
    extract_destination,
    inject_xhs_source_links,
    prefetch_xiaohongshu,
)
from voyageai.tools.registry import tool_registry

logger = logging.getLogger(__name__)

_PRIVATE_HOST_PREFIXES = ("localhost", "127.", "10.", "172.16.", "192.168.", "0.0.0.0")


def _is_public_url(url: str) -> bool:
    """Return True if *url* is reachable from the public internet.

    OpenAI's Responses API calls native MCP servers directly, so the URL
    must be publicly accessible — Docker service names, localhost, and
    private-IP addresses will all fail with HTTP 424.
    """
    hostname = urlparse(url).hostname or ""
    if not hostname or "." not in hostname:
        return False
    return not any(hostname.startswith(p) for p in _PRIVATE_HOST_PREFIXES)


# ── Responses API tool format helpers ────────────────────────────────


def _build_responses_tools(
    selected_tool_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert registered tools to Responses API function tool format."""
    tools: list[dict[str, Any]] = []

    openai_tools = tool_registry.get_openai_tools()
    for t in openai_tools:
        fn = t.get("function", {})
        name = fn.get("name", "")
        if name.startswith("xiaohongshu__"):
            continue
        if selected_tool_names and name not in selected_tool_names:
            continue
        tools.append({
            "type": "function",
            "name": name,
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })

    return tools


_COUNTRY_HINTS = {
    "japan": "JP", "tokyo": "JP", "osaka": "JP", "kyoto": "JP",
    "china": "CN", "beijing": "CN", "shanghai": "CN", "北京": "CN", "上海": "CN",
    "france": "FR", "paris": "FR",
    "italy": "IT", "rome": "IT", "milan": "IT",
    "spain": "ES", "barcelona": "ES", "madrid": "ES",
    "germany": "DE", "berlin": "DE", "munich": "DE",
    "uk": "GB", "london": "GB", "england": "GB",
    "thailand": "TH", "bangkok": "TH",
    "korea": "KR", "seoul": "KR",
    "usa": "US", "new york": "US", "los angeles": "US", "hawaii": "US",
    "australia": "AU", "sydney": "AU", "melbourne": "AU",
    "singapore": "SG",
    "vietnam": "VN", "hanoi": "VN",
    "india": "IN", "delhi": "IN", "mumbai": "IN",
    "turkey": "TR", "istanbul": "TR",
    "greece": "GR", "athens": "GR",
    "portugal": "PT", "lisbon": "PT",
    "indonesia": "ID", "bali": "ID",
    "malaysia": "MY", "kuala lumpur": "MY",
    "mexico": "MX",
    "brazil": "BR",
    "canada": "CA", "toronto": "CA", "vancouver": "CA",
}


def _guess_country_code(destination: str) -> str:
    """Best-effort country code from destination string."""
    lower = destination.lower()
    for hint, code in _COUNTRY_HINTS.items():
        if hint in lower:
            return code
    return "US"


_COUNTRY_CURRENCY: dict[str, str] = {
    "JP": "JPY", "CN": "CNY", "KR": "KRW", "TH": "THB", "VN": "VND",
    "IN": "INR", "ID": "IDR", "MY": "MYR", "SG": "SGD", "AU": "AUD",
    "GB": "GBP", "FR": "EUR", "IT": "EUR", "ES": "EUR", "DE": "EUR",
    "PT": "EUR", "GR": "EUR", "TR": "TRY", "MX": "MXN", "BR": "BRL",
    "CA": "CAD", "US": "USD",
}


def _guess_currency(country_code: str) -> str | None:
    """Map a country code to its primary currency."""
    return _COUNTRY_CURRENCY.get(country_code)


class ResponsesAgentService:
    """AI Agent using the OpenAI Responses API."""

    def __init__(
        self,
        model: str | None = None,
        max_iterations: int = 8,
        temperature: float = 0.7,
        use_tool_rag: bool = False,
        tool_rag_top_k: int = 8,
    ):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.use_tool_rag = use_tool_rag
        self.tool_rag_top_k = tool_rag_top_k
        self._tool_rag: ToolRAG = tool_rag
        self._is_local_model: bool = False

    def with_api_key(self, api_key: str) -> "ResponsesAgentService":
        """Return a shallow copy that uses a different OpenAI API key."""
        clone = ResponsesAgentService(
            model=self.model,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            use_tool_rag=self.use_tool_rag,
            tool_rag_top_k=self.tool_rag_top_k,
        )
        clone.client = AsyncOpenAI(api_key=api_key)
        clone._tool_rag = self._tool_rag
        return clone

    def with_ollama(self) -> "ResponsesAgentService":
        """Return a clone that uses Ollama via its Responses API endpoint.

        Ollama supports /v1/responses with function calling.
        We skip OpenAI-only tool types (web_search, mcp) and flag the
        resulting itinerary with ``local_model=True``.
        """
        clone = ResponsesAgentService(
            model=settings.ollama_model,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            use_tool_rag=self.use_tool_rag,
            tool_rag_top_k=self.tool_rag_top_k,
        )
        clone.client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",
            timeout=600.0,
        )
        clone._tool_rag = self._tool_rag
        clone._is_local_model = True
        return clone

    def with_gemini(self, api_key: str) -> "ResponsesAgentService":
        """Return a clone that uses Google Gemini via its OpenAI-compatible endpoint."""
        clone = ResponsesAgentService(
            model=settings.gemini_model,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            use_tool_rag=self.use_tool_rag,
            tool_rag_top_k=self.tool_rag_top_k,
        )
        clone.client = AsyncOpenAI(
            base_url=settings.gemini_base_url,
            api_key=api_key,
        )
        clone._tool_rag = self._tool_rag
        return clone

    # ── Progress callback helper ───────────────────────────────────

    @staticmethod
    async def _emit(
        callback: ProgressCallback | None,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        if callback is None:
            return
        try:
            await callback(event_type, data)
        except Exception:
            logger.warning(
                "Progress callback error for event %s",
                event_type,
                exc_info=True,
            )

    # ── Heartbeat during long-running coroutines ───────────────────

    async def _with_heartbeat(
        self,
        coro: Any,
        callback: ProgressCallback | None,
        interval: float = 10,
        message: str = "Still working...",
    ) -> Any:
        """Run a coroutine with periodic heartbeat events."""
        if callback is None:
            return await coro

        async def heartbeat() -> None:
            count = 0
            while True:
                await asyncio.sleep(interval)
                count += 1
                elapsed = int(count * interval)
                await self._emit(callback, "thinking", {
                    "message": f"{message} ({elapsed}s elapsed)",
                    "elapsed_heartbeats": count,
                    "elapsed_seconds": elapsed,
                })

        task = asyncio.create_task(heartbeat())
        try:
            return await coro
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ── Clarification detection (pre-flight analysis) ──────────────

    async def analyze_request(
        self,
        requirements: str,
        conversation_context: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Analyze a user request to determine if clarification is needed.

        Uses a fast Chat Completions call (single LLM call, not an agent loop).
        Returns ``{"ready": True}`` or ``{"ready": False, "questions": [...]}``.
        """
        context_section = ""
        if conversation_context:
            context_section = (
                f"\nPrevious conversation history:\n---\n{conversation_context}\n---\n\n"
                "The user is now sending a follow-up message within the same project.\n"
                "Consider information from the conversation history when checking for missing details.\n\n"
            )

        analysis_prompt = (
            f"Analyze this travel planning request and determine if any critical information is missing.\n"
            f"{context_section}"
            f'User request: "{requirements}"\n\n'
            "Decide whether the request is DETAILED ENOUGH or needs a few quick questions.\n\n"
            "DETAILED ENOUGH (return ready=true) when the request includes:\n"
            "- A clear destination AND at least TWO of: duration, dates, budget, interests\n"
            '- Example: "5 days in Tokyo, food and culture" → ready=true\n'
            '- Example: "Paris next week, $3000 budget" → ready=true\n\n'
            "NEEDS QUESTIONS (return ready=false) when:\n"
            "- No destination at all → ask for destination\n"
            "- Only a destination with nothing else → ask 1-2 quick questions about "
            "duration, interests, or budget to personalize the trip\n"
            '- Example: "plan a trip to Japan" → ask about duration + interests\n'
            '- Example: "I want to visit Europe" → ask which country/city + duration\n\n'
            "Smart defaults (use these if you decide ready=true with partial info):\n"
            "- No duration → 5 days\n"
            "- No dates → 2 weeks from today\n"
            "- No budget → Medium ($100-200/day)\n"
            "- No interests → infer from destination\n\n"
            "Respond with JSON:\n"
            '  Ready: {"ready": true}\n'
            '  Need info: {"ready": false, "questions": [{"id": "unique_id", '
            '"question": "...", "type": "single_choice|multiple_choice|free_text", '
            '"options": ["A", "B", "C"]}]}\n\n'
            "Rules:\n"
            "- Ask at most 2-3 questions, keep them SHORT with practical options\n"
            "- Do NOT ask about things already mentioned in the request\n"
            '- For "type": "single_choice" for budget/duration, "multiple_choice" for interests, '
            '"free_text" for open questions\n'
            "- RESPOND WITH ONLY JSON, no other text."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You analyze travel requests and identify missing information. Respond with JSON only."},
                    {"role": "user", "content": analysis_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.3,
            )
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)

            if result.get("ready", True):
                logger.info("Request analysis: ready to plan")
                return {"ready": True}

            questions = result.get("questions", [])
            logger.info("Request analysis: needs clarification, %d questions", len(questions))

            await self._emit(progress_callback, "clarification_needed", {
                "questions": questions,
            })
            return {"ready": False, "questions": questions}

        except Exception as e:
            logger.warning("Request analysis failed, proceeding anyway: %s", e)
            return {"ready": True}

    # ── Tool-RAG (semantic tool selection) ─────────────────────────

    async def _select_tools_with_rag(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[str], ToolSelectionResult | None]:
        """Select relevant tools using Tool-RAG."""
        k = top_k or self.tool_rag_top_k

        try:
            await self._tool_rag.initialize()
            selection = await self._tool_rag.select_tools(
                query, top_k=k, openai_client=self.client,
            )

            if not selection.selected_tools:
                logger.warning("Tool-RAG returned no tools, falling back to all tools")
                all_names = [n for n in tool_registry.list_tools() if not n.startswith("xiaohongshu__")]
                return all_names, None

            selected_names = [t.name for t in selection.selected_tools if not t.name.startswith("xiaohongshu__")]

            for core_name in sorted(CORE_TOOLS - set(selected_names)):
                if tool_registry.get(core_name):
                    selected_names.append(core_name)
                    if core_name in self._tool_rag._tool_cache:
                        selection.selected_tools.append(self._tool_rag._tool_cache[core_name])
                    logger.info("Injected core tool: %s", core_name)

            selection.selected_tools = [
                t for t in selection.selected_tools if not t.name.startswith("xiaohongshu__")
            ]

            logger.info(
                "Tool-RAG selected %d tools: %s (%dms)",
                len(selected_names),
                selected_names,
                selection.selection_time_ms,
            )
            return selected_names, selection

        except Exception as e:
            logger.error("Tool-RAG failed: %s, falling back to all tools", e)
            all_names = [n for n in tool_registry.list_tools() if not n.startswith("xiaohongshu__")]
            return all_names, None

    # ── call_single_tool (debug / tools router) ────────────────────

    async def call_single_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallTrace:
        """Call a single tool directly (for testing/debugging)."""
        result = await tool_registry.execute(tool_name, arguments)
        return ToolCallTrace(
            call_id=f"direct-{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            arguments=arguments,
            result=result.output,
            success=result.success,
            error=result.error,
            latency_ms=result.latency_ms,
        )

    # ── Edit entry point (delegates to surgical editor) ─────────────

    async def edit_existing_itinerary(
        self,
        edit_instruction: str,
        existing_itinerary_json: str,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResponse:
        """Apply a surgical edit to an existing itinerary.

        Uses a single LLM call (no tool calling) to apply the targeted
        modification, then bumps the version number.
        """
        import json as _json

        from voyageai.services.itinerary_generator import edit_itinerary

        # Parse current version from existing itinerary
        current_version = 1
        try:
            existing_data = _json.loads(existing_itinerary_json)
            current_version = existing_data.get("version", 1)
        except Exception:
            pass

        response = await edit_itinerary(
            client=self.client,
            edit_instruction=edit_instruction,
            existing_itinerary_json=existing_itinerary_json,
            progress_callback=progress_callback,
            temperature=self.temperature,
        )

        if response.success and response.itinerary:
            response.itinerary.version = current_version + 1

        return response

    # ── Pre-flight parallel tool execution ──────────────────────────

    async def _preflight_tools(
        self,
        requirements: str,
        user_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        all_tool_traces: list[ToolCallTrace] | None = None,
    ) -> str | None:
        """Run common data-gathering tools in parallel before the agent loop.

        Phase 1: geocode the destination (~50ms).
        Phase 2: with coordinates, run weather / holidays / flights /
                 attractions / restaurants / places concurrently.

        Returns a formatted string of all results ready for injection
        into the agent context, or None if geocoding fails.
        """
        from datetime import date, timedelta

        destination = extract_destination(requirements)
        if not destination:
            logger.info("Pre-flight: could not extract destination, skipping")
            return None

        await self._emit(progress_callback, "stage_change", {
            "stage": "TOOL_CALLING",
            "message": f"Gathering data for {destination}...",
        })

        # Phase 1 — geocode
        await self._emit(progress_callback, "tool_start", {
            "tool": "geocode_location", "arguments": {"location": destination},
        })
        geo_result = await tool_registry.execute(
            "geocode_location", {"location": destination},
            user_id=user_id, enable_rate_limit=user_id is not None,
        )
        if all_tool_traces is not None:
            all_tool_traces.append(ToolCallTrace(
                call_id=f"preflight-geo-{uuid.uuid4().hex[:8]}",
                tool_name="geocode_location",
                arguments={"location": destination},
                result=geo_result.output,
                success=geo_result.success,
                error=geo_result.error,
                latency_ms=geo_result.latency_ms,
            ))
        await self._emit(progress_callback, "tool_result", {
            "tool": "geocode_location",
            "success": geo_result.success,
            "latency_ms": geo_result.latency_ms,
            "summary": json.dumps(geo_result.output, default=str)[:300] if geo_result.success else (geo_result.error or "")[:300],
        })

        if not geo_result.success or not geo_result.output:
            logger.warning("Pre-flight geocode failed, falling back to agent loop")
            return None

        lat = geo_result.output.get("latitude")
        lng = geo_result.output.get("longitude")
        if lat is None or lng is None:
            return None

        # Parse dates / country from requirements (best-effort)
        today = date.today()
        start_date = (today + timedelta(days=14)).isoformat()
        end_date = (today + timedelta(days=19)).isoformat()

        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", requirements)
        if date_match:
            start_date = date_match.group(1)

        country_code = _guess_country_code(destination)

        # Check which tools the user request implies
        wants_flights = any(
            kw in requirements.lower()
            for kw in ["flight", "fly", "airport", "budget", "$", "预算"]
        )

        # Phase 2 — run all data tools in parallel
        logger.info(
            "Pre-flight phase 2: running tools in parallel (lat=%.4f, lng=%.4f)",
            lat, lng,
        )

        async def _safe_execute(name: str, args: dict) -> tuple[str, Any]:
            """Execute a tool and return (name, result), never raising."""
            try:
                await self._emit(progress_callback, "tool_start", {
                    "tool": name, "arguments": args,
                })
                result = await tool_registry.execute(
                    name, args,
                    user_id=user_id, enable_rate_limit=user_id is not None,
                )
                if all_tool_traces is not None:
                    all_tool_traces.append(ToolCallTrace(
                        call_id=f"preflight-{name}-{uuid.uuid4().hex[:6]}",
                        tool_name=name,
                        arguments=args,
                        result=result.output,
                        success=result.success,
                        error=result.error,
                        latency_ms=result.latency_ms,
                    ))
                await self._emit(progress_callback, "tool_result", {
                    "tool": name,
                    "success": result.success,
                    "latency_ms": result.latency_ms,
                    "summary": (
                        json.dumps(result.output, default=str)[:300]
                        if result.success
                        else (result.error or "")[:200]
                    ),
                })
                return (name, result)
            except Exception as e:
                logger.warning("Pre-flight tool %s raised: %s", name, e)
                from voyageai.tools.base import ToolResult
                return (name, ToolResult(output=None, success=False, error=str(e), latency_ms=0))

        # Build task list dynamically — only call tools that are registered
        available = set(tool_registry.list_tools())

        preflight_plan: list[tuple[str, dict]] = [
            ("get_weather_forecast", {
                "latitude": lat, "longitude": lng,
                "start_date": start_date, "end_date": end_date,
            }),
            ("get_public_holidays", {
                "country_code": country_code,
                "year": today.year,
            }),
        ]

        # Prefer Google Maps over Overpass (faster, more reliable)
        if "googlemaps__search_places" in available:
            preflight_plan.append(("googlemaps__search_places", {
                "query": f"top attractions and things to do in {destination}",
                "latitude": lat, "longitude": lng,
                "radius": 10000, "limit": 10,
            }))
            preflight_plan.append(("googlemaps__search_places", {
                "query": f"best restaurants and food in {destination}",
                "latitude": lat, "longitude": lng,
                "radius": 10000, "limit": 5,
            }))
            preflight_plan.append(("googlemaps__search_places", {
                "query": f"best hotels in {destination}",
                "latitude": lat, "longitude": lng,
                "radius": 15000, "limit": 5,
            }))
        else:
            preflight_plan.append(("search_attractions", {
                "latitude": lat, "longitude": lng,
                "radius": 10000, "categories": "cultural,architecture,natural",
                "limit": 10,
            }))

        # Currency conversion (USD → destination currency)
        dest_currency = _guess_currency(country_code)
        if dest_currency and dest_currency != "USD" and "convert_currency" in available:
            preflight_plan.append(("convert_currency", {
                "from_currency": "USD",
                "to_currency": dest_currency,
                "amount": 100,
            }))

        # Web search for seasonal/events info
        if "web_search" in available:
            preflight_plan.append(("web_search", {
                "query": f"{destination} travel tips events {today.year}",
                "intent": "destination_info",
                "max_results": 5,
            }))

        if wants_flights and "search_flights" in available:
            preflight_plan.append(("search_flights", {
                "origin": "JFK",
                "destination": destination,
                "departure_date": start_date,
                "adults": 1,
                "max_results": 3,
            }))

        # Filter to only tools that actually exist in the registry
        tasks = [
            _safe_execute(name, args)
            for name, args in preflight_plan
            if name in available
        ]

        results = await asyncio.gather(*tasks)

        # Format results into text
        sections = [f"## Geocode\n{json.dumps(geo_result.output, default=str)}"]
        for name, result in results:
            if result.success and result.output:
                data = json.dumps(result.output, default=str)
                if len(data) > 3000:
                    data = data[:3000] + "... (truncated)"
                sections.append(f"## {name}\n{data}")
            else:
                sections.append(f"## {name}\nFailed: {result.error or 'no data'}")

        logger.info(
            "Pre-flight complete: %d tools executed, %d succeeded",
            len(results) + 1,
            sum(1 for _, r in results if r.success) + (1 if geo_result.success else 0),
        )
        return "\n\n".join(sections)

    # ── Main entry point ───────────────────────────────────────────

    async def generate_with_tools(
        self,
        requirements: str,
        max_iterations: int | None = None,
        use_tool_rag: bool | None = None,
        tool_rag_top_k: int | None = None,
        user_id: str | None = None,
        conversation_context: str | None = None,
        progress_callback: ProgressCallback | None = None,
        existing_itinerary_context: str | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """Generate an itinerary using the Responses API with tool calling.

        Full-featured implementation with Tool-RAG, Xiaohongshu pre-fetch,
        failed tool tracking, plan outline, heartbeat, and retry logic.
        """
        start_time = time.time()
        max_iter = max_iterations or self.max_iterations
        if self._is_local_model and max_iter > 2:
            max_iter = 2
        all_tool_traces: list[ToolCallTrace] = []
        llm_calls: list[LLMCallRecord] = []
        total_tokens = 0

        # ── Tool-RAG selection ──
        # Local models (Ollama) don't ship embedding models; skip Tool-RAG.
        if self._is_local_model:
            use_tool_rag = False
        should_use_tool_rag = use_tool_rag if use_tool_rag is not None else self.use_tool_rag
        rag_top_k = tool_rag_top_k or self.tool_rag_top_k

        total_registered = len(tool_registry.list_tools())
        if should_use_tool_rag and total_registered < TOOL_RAG_SKIP_THRESHOLD:
            logger.info(
                "Tool-RAG skipped: only %d tools registered (threshold=%d), using all",
                total_registered, TOOL_RAG_SKIP_THRESHOLD,
            )
            should_use_tool_rag = False

        tool_selection: ToolSelectionResult | None = None
        selected_tool_names: list[str] = []

        if should_use_tool_rag:
            await self._emit(progress_callback, "stage_change", {
                "stage": "RAG_SEARCH",
                "message": "Selecting relevant tools for your request...",
            })
            selected_tool_names, tool_selection = await self._select_tools_with_rag(
                requirements, top_k=rag_top_k,
            )
            if tool_selection:
                await self._emit(progress_callback, "thinking", {
                    "text": (
                        f"Selected {len(selected_tool_names)} tools: "
                        f"{', '.join(selected_tool_names)} "
                        f"(took {tool_selection.selection_time_ms}ms)"
                    ),
                })
        else:
            selected_tool_names = [
                n for n in tool_registry.list_tools()
                if not n.startswith("xiaohongshu__")
            ]

        function_tools = _build_responses_tools(selected_tool_names)

        # web_search and mcp are OpenAI-only; skip for local models
        if not self._is_local_model:
            if settings.enable_builtin_web_search:
                function_tools.append({
                    "type": "web_search",
                    "search_context_size": "medium",
                })

            if settings.google_maps_mcp_url and _is_public_url(settings.google_maps_mcp_url):
                function_tools.append({
                    "type": "mcp",
                    "server_label": "google_maps",
                    "server_url": settings.google_maps_mcp_url,
                    "require_approval": "never",
                })
        if not self._is_local_model and settings.google_maps_mcp_url and not _is_public_url(settings.google_maps_mcp_url):
            logger.info(
                "Skipping native MCP for Google Maps — URL %s is not "
                "publicly reachable (OpenAI servers cannot reach Docker/"
                "localhost addresses). Using local MCP adapter tools instead.",
                settings.google_maps_mcp_url,
            )

        if should_use_tool_rag and tool_selection:
            tool_desc_list = [
                {"name": t.name, "description": t.description}
                for t in tool_selection.selected_tools
            ]
        else:
            tool_desc_list = tool_registry.get_tool_descriptions()
        system_instructions = _build_system_prompt(tool_desc_list)

        # ── Xiaohongshu pre-fetch (parallel, skip for local models) ──
        xhs_task: asyncio.Task[str | None] | None = None
        xhs_injected = False
        if self._is_local_model:
            logger.info("Skipping XHS pre-fetch for local model (saves 2-3 min)")
            xhs_injected = True
        else:
            xhs_destination = extract_destination(requirements)
            xhs_task = asyncio.create_task(
                prefetch_xiaohongshu(
                    destination=xhs_destination,
                    progress_callback=progress_callback,
                )
            )

        # Build initial input
        input_items: list[dict[str, Any]] = []
        has_existing = bool(existing_itinerary_context)

        if has_existing:
            # Follow-up in a project that already has a generated itinerary.
            # Inject the actual structured JSON so the LLM updates it in place.
            itinerary_snippet = existing_itinerary_context
            if len(itinerary_snippet) > 30000:
                itinerary_snippet = itinerary_snippet[:30000] + "\n... (truncated)"

            context_parts = []
            if conversation_context:
                context_parts.append(
                    f"Previous conversation:\n{conversation_context}\n---\n"
                )
            context_parts.append(
                f"Here is the current itinerary (JSON) that the user already has:\n"
                f"```json\n{itinerary_snippet}\n```\n\n"
                f"The user wants to update this itinerary with the following request:\n\n"
                f"{requirements}\n\n"
                "IMPORTANT: You MUST use the existing itinerary as your starting point. "
                "Apply the user's requested changes while preserving everything else. "
                "Do NOT discard or regenerate the itinerary from scratch. "
                "Gather any additional information needed using tools, "
                "then output a complete updated itinerary in the same JSON structure."
            )
            input_items.append({
                "role": "user",
                "content": "\n".join(context_parts),
            })
        elif conversation_context:
            input_items.append({
                "role": "user",
                "content": (
                    f"Here is the previous conversation history for context:\n\n"
                    f"{conversation_context}\n\n---\n\n"
                    f"Now the user has a follow-up request. Please update or refine the "
                    f"travel plan based on this new request:\n\n{requirements}\n\n"
                    "Consider the previous conversation when making changes. "
                    "Gather any additional information needed using the available tools, "
                    "then generate an updated complete itinerary."
                ),
            })
        else:
            input_items.append({
                "role": "user",
                "content": (
                    f"Please create a travel itinerary based on these requirements:\n\n"
                    f"{requirements}\n\n"
                    "First, gather relevant information using the available tools, "
                    "then generate a complete itinerary."
                ),
            })

        # ── Pre-flight parallel tool execution ──────────────────────
        # Instead of relying on the LLM to call tools one-by-one,
        # we proactively run common data-gathering tools in parallel
        # and inject results before the agent loop starts.
        if not has_existing:
            preflight_results = await self._preflight_tools(
                requirements=requirements,
                user_id=user_id,
                progress_callback=progress_callback,
                all_tool_traces=all_tool_traces,
            )
            if preflight_results:
                input_items.append({
                    "role": "system",
                    "content": (
                        "The following data has already been gathered for this request. "
                        "Do NOT re-call these tools. Use this data directly. "
                        "Only call additional tools if you need information not "
                        "covered below.\n\n" + preflight_results
                    ),
                })
                logger.info(
                    "Injected pre-flight tool results (%d chars)",
                    len(preflight_results),
                )

        try:
            failed_tools: dict[str, str] = {}

            for iteration in range(max_iter):
                logger.info("Responses API iteration %d/%d", iteration + 1, max_iter)

                await self._emit(progress_callback, "stage_change", {
                    "stage": "TOOL_CALLING",
                    "message": f"Agent reasoning (Responses API), iteration {iteration + 1}...",
                })

                # ── Inject Xiaohongshu context when ready ──
                if not xhs_injected and xhs_task is not None and xhs_task.done():
                    try:
                        xhs_context = xhs_task.result()
                        if xhs_context:
                            input_items.append({"role": "system", "content": xhs_context})
                            logger.info("Injected Xiaohongshu context (%d chars) before iteration %d",
                                        len(xhs_context), iteration + 1)
                    except Exception as e:
                        logger.warning("Xiaohongshu pre-fetch result error: %s", e)
                    xhs_injected = True
                elif not xhs_injected and xhs_task is not None and iteration >= 1:
                    if not xhs_task.done():
                        try:
                            await asyncio.wait_for(asyncio.shield(xhs_task), timeout=10)
                        except asyncio.TimeoutError:
                            logger.info("Xiaohongshu pre-fetch still running at iteration %d", iteration + 1)
                        except Exception as e:
                            logger.warning("Xiaohongshu pre-fetch error at iteration %d: %s", iteration + 1, e)
                            xhs_injected = True
                    if xhs_task.done():
                        try:
                            xhs_context = xhs_task.result()
                            if xhs_context:
                                input_items.append({"role": "system", "content": xhs_context})
                                logger.info("Injected Xiaohongshu context (%d chars) at iteration %d",
                                            len(xhs_context), iteration + 1)
                        except Exception as e:
                            logger.warning("Xiaohongshu pre-fetch result error: %s", e)
                        xhs_injected = True

                # ── Inject failed tools advisory ──
                if failed_tools:
                    advisory_lines = [f"- {name}: {reason}" for name, reason in failed_tools.items()]
                    advisory = (
                        "IMPORTANT: The following tools have PERMANENT errors and "
                        "must NOT be retried. Use alternative tools instead:\n"
                        + "\n".join(advisory_lines)
                        + "\nPrefer web_search as a fallback for place/restaurant/hotel data."
                    )
                    input_items.append({"role": "system", "content": advisory})

                # ── Responses API call ──
                api_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "instructions": system_instructions,
                    "tools": function_tools,
                    "input": input_items,
                    "temperature": self.temperature,
                    "parallel_tool_calls": True,
                }
                # Nudge the model to output JSON directly on the last
                # allowed iteration so we can skip generate_structured_final.
                # Ollama may not support the text.format parameter, so only
                # apply it for OpenAI models.
                if iteration == max_iter - 1 and not self._is_local_model:
                    api_kwargs["text"] = {
                        "format": {"type": "json_object"},
                    }
                response = await self.client.responses.create(**api_kwargs)

                usage = response.usage
                in_tok = usage.input_tokens if usage else 0
                out_tok = usage.output_tokens if usage else 0
                call_cost = _calc_cost(self.model, in_tok, out_tok)
                llm_calls.append(LLMCallRecord(
                    label=f"responses_iter_{iteration + 1}",
                    model=self.model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=call_cost,
                ))
                total_tokens += (in_tok + out_tok)

                has_function_calls = False
                input_items.extend(response.output)

                # Collect all function calls for parallel execution
                pending_calls: list[tuple[Any, str, dict]] = []
                for item in response.output:
                    item_type = item.type
                    if item_type == "function_call":
                        has_function_calls = True
                        try:
                            arguments = json.loads(item.arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                        pending_calls.append((item, item.name, arguments))
                    elif item_type == "web_search_call":
                        await self._emit(progress_callback, "tool_result", {
                            "tool": "web_search (built-in)",
                            "success": True,
                            "latency_ms": 0,
                            "summary": "Built-in web search completed",
                        })
                    elif item_type == "message":
                        content_text = ""
                        for part in (item.content or []):
                            if hasattr(part, "text"):
                                content_text += part.text
                        if content_text:
                            await self._emit(progress_callback, "thinking", {
                                "text": content_text[:1000],
                            })

                # Execute tool calls in parallel
                if pending_calls:
                    if len(pending_calls) > 1:
                        logger.info(
                            "Executing %d tool calls in parallel: %s",
                            len(pending_calls),
                            [name for _, name, _ in pending_calls],
                        )

                    for _, name, args in pending_calls:
                        await self._emit(progress_callback, "tool_start", {
                            "tool": name, "arguments": args,
                        })

                    async def _run_tool(call_item: Any, name: str, args: dict):
                        return await tool_registry.execute(
                            name, args,
                            user_id=user_id,
                            enable_rate_limit=user_id is not None,
                        )

                    results = await asyncio.gather(
                        *[_run_tool(it, n, a) for it, n, a in pending_calls],
                        return_exceptions=True,
                    )

                    for (call_item, tool_name, arguments), result in zip(
                        pending_calls, results,
                    ):
                        if isinstance(result, Exception):
                            from voyageai.tools.base import ToolResult
                            result = ToolResult(
                                output=None, success=False,
                                error=str(result), latency_ms=0,
                            )

                        trace = ToolCallTrace(
                            call_id=call_item.call_id,
                            tool_name=tool_name,
                            arguments=arguments,
                            result=result.output,
                            success=result.success,
                            error=result.error,
                            latency_ms=result.latency_ms,
                        )
                        all_tool_traces.append(trace)

                        if not result.success and _is_permanent_error(result.error):
                            if tool_name not in failed_tools:
                                failed_tools[tool_name] = result.error or "unknown"
                                logger.warning(
                                    "Marking tool %s as permanently failed: %s",
                                    tool_name, result.error,
                                )

                        result_summary = ""
                        if result.success and result.output:
                            s = json.dumps(result.output, default=str)
                            result_summary = s[:500] + ("..." if len(s) > 500 else "")
                        elif result.error:
                            result_summary = result.error[:300]
                        await self._emit(progress_callback, "tool_result", {
                            "tool": tool_name,
                            "success": result.success,
                            "latency_ms": result.latency_ms,
                            "summary": result_summary,
                        })

                        output_content = (
                            json.dumps(result.output, default=str)
                            if result.success
                            else json.dumps({"error": result.error})
                        )
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": call_item.call_id,
                            "output": output_content,
                        })

                if has_function_calls:
                    # If next iteration is the last one, nudge the model to
                    # produce JSON directly instead of more tool calls.
                    if iteration + 1 == max_iter - 1:
                        input_items.append({
                            "role": "system",
                            "content": (
                                "You have gathered sufficient data. On your NEXT response, "
                                "output the COMPLETE travel itinerary as a single JSON object "
                                "following the OUTPUT FORMAT in your instructions. "
                                "Do NOT call any more tools."
                            ),
                        })
                    continue

                # ── No function calls — LLM is done reasoning ──

                # Last chance: inject Xiaohongshu context
                if not xhs_injected and xhs_task is not None and xhs_task.done():
                    try:
                        xhs_context = xhs_task.result()
                        if xhs_context:
                            input_items.append({"role": "system", "content": xhs_context})
                            logger.info("Injected Xiaohongshu context (%d chars) before final generation",
                                        len(xhs_context))
                    except Exception:
                        pass
                    xhs_injected = True

                # ── Try to parse JSON directly from agent output ──
                await self._emit(progress_callback, "stage_change", {
                    "stage": "GENERATING",
                    "message": "Generating full structured itinerary...",
                })

                final_text = response.output_text or ""
                itinerary = parse_itinerary(final_text)

                if itinerary:
                    logger.info(
                        "Direct JSON parse succeeded — skipped generate_structured_final "
                        "(%d days, %d chars)",
                        len(itinerary.days), len(final_text),
                    )
                else:
                    logger.info(
                        "Direct JSON parse failed (len=%d), falling back to "
                        "generate_structured_final",
                        len(final_text),
                    )
                    itinerary = await self._with_heartbeat(
                        generate_structured_final(
                            client=self.client,
                            requirements=requirements,
                            input_items=input_items,
                            progress_callback=progress_callback,
                            llm_calls=llm_calls,
                            temperature=self.temperature,
                            model_override=(
                                self.model if self._is_local_model else None
                            ),
                        ),
                        progress_callback,
                        interval=8,
                        message="Crafting your detailed itinerary — this takes a moment for quality results...",
                    )

                # Post-process: inject Xiaohongshu source_links
                if itinerary and xhs_task is not None and xhs_task.done():
                    try:
                        _xhs_ctx = xhs_task.result()
                        if _xhs_ctx:
                            inject_xhs_source_links(itinerary, _xhs_ctx)
                    except Exception:
                        pass

                # Flag local-model itineraries so the frontend shows a warning
                if itinerary and self._is_local_model:
                    from voyageai.schemas.itinerary import StructuredItinerary
                    data = itinerary.model_dump()
                    data["local_model"] = True
                    itinerary = StructuredItinerary.model_validate(data)

                processing_time = int((time.time() - start_time) * 1000)
                total_cost = round(sum(c.cost_usd for c in llm_calls), 6)

                await self._emit(progress_callback, "cost_summary", {
                    "total_tokens": total_tokens,
                    "total_cost_usd": total_cost,
                    "llm_calls": len(llm_calls),
                    "tool_calls": len(all_tool_traces),
                    "processing_time_ms": processing_time,
                    "api": "responses",
                    "breakdown": [
                        {
                            "label": c.label,
                            "model": c.model,
                            "input_tokens": c.input_tokens,
                            "output_tokens": c.output_tokens,
                            "cost_usd": c.cost_usd,
                        }
                        for c in llm_calls
                    ],
                })

                return AgentResponse(
                    itinerary=itinerary,
                    tool_trace=all_tool_traces,
                    raw_response=final_text,
                    success=True,
                    total_tokens=total_tokens,
                    processing_time_ms=processing_time,
                    tool_selection=tool_selection,
                    selected_tool_names=selected_tool_names,
                    llm_calls=llm_calls,
                    total_cost_usd=total_cost,
                )

            raise RuntimeError(f"Responses API agent exceeded max iterations ({max_iter})")

        except Exception as e:
            logger.error("Responses API agent failed: %s", e)
            total_cost = round(sum(c.cost_usd for c in llm_calls), 6)
            return AgentResponse(
                itinerary=None,
                tool_trace=all_tool_traces,
                success=False,
                error=str(e),
                total_tokens=total_tokens,
                processing_time_ms=int((time.time() - start_time) * 1000),
                tool_selection=tool_selection,
                selected_tool_names=selected_tool_names,
                llm_calls=llm_calls,
                total_cost_usd=total_cost,
            )

