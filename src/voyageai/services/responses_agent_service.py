"""
AI Agent Service using the OpenAI Responses API.

Uses ``client.responses.create()`` which supports:
- Built-in ``web_search`` tool (no API key, billed per-token)
- Native MCP server connections (Google Maps, Xiaohongshu)
- ``previous_response_id`` for server-managed conversation state
- ``FunctionTool`` definitions identical to existing tools
- Structured text output via ``text.format``
- Streaming with granular events

Features:
- Tool-RAG (semantic tool selection)
- Xiaohongshu pre-fetch (parallel search + auth recovery + context injection)
- XHS source link post-processing
- Failed tool tracking (permanent error detection)
- Clarification detection (analyze_request)
- Plan outline generation
- Heartbeat during long generation
- Retry for incomplete days in final generation
- call_single_tool for debug endpoints

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

from openai import AsyncOpenAI
from pydantic import ValidationError

from voyageai.config import settings
from voyageai.rag.tool_rag import ToolRAG, tool_rag
from voyageai.schemas.itinerary import SourceLink, StructuredItinerary
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
from voyageai.tools.registry import tool_registry

logger = logging.getLogger(__name__)


# ── Responses API tool format helpers ────────────────────────────────


def _build_responses_tools(
    selected_tool_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert registered tools to Responses API function tool format.

    The Responses API uses a slightly different schema:
    ``{"type": "function", "name": ..., "parameters": ..., "description": ...}``
    instead of the Chat Completions nested ``{"type": "function", "function": {...}}``
    format.
    """
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


class ResponsesAgentService:
    """AI Agent using the OpenAI Responses API."""

    # JSON structure example for prompt-guided final generation.
    _ITINERARY_JSON_EXAMPLE = """{
  "metadata": {
    "destination": "Tokyo, Japan",
    "start_date": "2024-04-01",
    "end_date": "2024-04-03",
    "total_days": 3,
    "budget": "Medium ($100-200/day)",
    "interests": ["culture", "food"],
    "best_season": "Spring (cherry blossom season)",
    "currency": "JPY",
    "language": "Japanese"
  },
  "days": [
    {
      "day_number": 1,
      "date": "2024-04-01",
      "theme": "Arrival and City Exploration",
      "summary": "Start the day at historic Asakusa, then head to Tokyo Station for lunch.",
      "weather_forecast": "Sunny, 18°C / 64°F",
      "total_walking_km": 4.2,
      "activities": [
        {
          "activity_id": "act-day1-001",
          "time": "09:00-11:00",
          "title": "Visit Senso-ji Temple",
          "description": "Explore Tokyo's oldest Buddhist temple in Asakusa.",
          "location": {
            "name": "Senso-ji Temple",
            "latitude": 35.7148,
            "longitude": 139.7967,
            "address": "2-3-1 Asakusa, Taito City, Tokyo",
            "place_type": "temple"
          },
          "estimated_cost": "Free",
          "duration_minutes": 120,
          "notes": ["Visit early morning to avoid crowds"],
          "rating": 4.7,
          "website_url": "https://www.senso-ji.jp/",
          "source_links": [
            {"title": "Official Website", "url": "https://www.senso-ji.jp/", "source": "official"}
          ]
        }
      ],
      "alternatives": [
        [
          {"activity_id": "alt1-day1-001", "time": "09:00-11:30", "title": "Tsukiji Outer Market", "description": "Explore the famous fish market area.", "location": {"name": "Tsukiji Market", "latitude": 35.6654, "longitude": 139.7707, "address": "Tsukiji, Chuo City, Tokyo"}, "estimated_cost": "$20-30"}
        ]
      ]
    }
  ],
  "tips": ["Get a Suica card for easy transit"],
  "travel_tips": [
    {"category": "booking", "message": "Book Senso-ji guided tour 3 days ahead", "priority": "high", "applies_to": "act-day1-001", "advance_days": 3}
  ]
}"""

    def __init__(
        self,
        model: str | None = None,
        max_iterations: int = 10,
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
            "Check for these essentials:\n"
            "1. Destination (where to go) - is it specific enough?\n"
            "2. Duration (how many days)\n"
            "3. Dates or time period (when)\n"
            "4. Budget level\n"
            "5. Interests or preferences\n\n"
            "If the request has enough information to plan a trip (at least destination + duration "
            "or enough context to infer them), respond with:\n"
            '{"ready": true}\n\n'
            "If critical information is missing, respond with:\n"
            '{\n  "ready": false,\n  "questions": [\n    {\n      "id": "unique_id",\n'
            '      "question": "What would you like to know?",\n'
            '      "type": "single_choice|multiple_choice|free_text",\n'
            '      "options": ["Option A", "Option B", "Option C"]\n    }\n  ]\n}\n\n'
            "Rules:\n"
            "- Ask at most 3 questions (focus on the most important missing info)\n"
            "- If destination and duration are clear, it's usually ready\n"
            '- For "type", use "single_choice" for budget/dates, "multiple_choice" for interests, '
            '"free_text" for open questions\n'
            "- Options should be practical and concise\n"
            "- RESPOND WITH ONLY JSON, no other text."
        )

        try:
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
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
        """Select relevant tools using Tool-RAG.

        Returns:
            Tuple of (selected tool names list, selection result for logging).
        """
        k = top_k or self.tool_rag_top_k

        try:
            await self._tool_rag.initialize()
            selection = await self._tool_rag.select_tools(query, top_k=k)

            if not selection.selected_tools:
                logger.warning("Tool-RAG returned no tools, falling back to all tools")
                all_names = [n for n in tool_registry.list_tools() if not n.startswith("xiaohongshu__")]
                return all_names, None

            selected_names = [t.name for t in selection.selected_tools if not t.name.startswith("xiaohongshu__")]

            # Ensure core tools are always included
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

    # ── Xiaohongshu pre-fetch (programmatic, not LLM-driven) ──────

    @staticmethod
    def _extract_destination(requirements: str) -> str:
        """Extract the destination name from the requirements string."""
        first_line = requirements.split("\n")[0].strip()
        m = re.search(r"trip to\s+(.+?)(?:\s+from\s|\s+in\s+\w+\s+\d|\s*$)", first_line, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,;:!?")
        m = re.search(
            r"(?:visit|travel(?:ing)?\s+to|going\s+to|heading\s+to)\s+(.+?)(?:\s+from\s|\s+in\s+\w+\s+\d|\s*$)",
            first_line, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().rstrip(".,;:!?")
        return first_line[:50]

    async def _recover_xhs_auth(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> bool:
        """Detect Xiaohongshu login expiry, push QR code via SSE, poll for login."""
        import base64

        status_tool = tool_registry.get("xiaohongshu__check_login_status")
        qr_tool = tool_registry.get("xiaohongshu__get_login_qrcode")
        if not status_tool or not qr_tool:
            logger.warning("XHS auth tools not available, cannot recover login")
            return False

        try:
            status_result = await asyncio.wait_for(status_tool.execute(), timeout=15)
            status_text = ""
            if status_result.success and status_result.output:
                status_text = str(status_result.output)
            elif status_result.success and not status_result.output:
                status_text = status_result.error or ""
            if "已登录" in status_text:
                logger.info("XHS auth check: already logged in")
                return True
            logger.info("XHS auth check: not logged in — starting QR recovery")
        except Exception as e:
            logger.warning("XHS check_login_status failed: %s", e)
            return False

        try:
            qr_result = await asyncio.wait_for(qr_tool.execute(), timeout=15)
        except Exception as e:
            logger.warning("XHS get_login_qrcode failed: %s", e)
            return False

        if not qr_result.success or not qr_result.output:
            logger.warning("XHS QR code retrieval failed: %s", qr_result.error)
            return False

        qr_data = qr_result.output if isinstance(qr_result.output, dict) else {}
        qr_image = qr_data.get("qrcode", qr_data.get("image", ""))
        if isinstance(qr_image, bytes):
            qr_base64 = base64.b64encode(qr_image).decode()
        elif isinstance(qr_image, str) and qr_image.startswith("data:"):
            qr_base64 = qr_image
        elif isinstance(qr_image, str):
            qr_base64 = qr_image
        else:
            logger.warning("XHS QR code: unexpected format")
            return False

        expiry_text = str(qr_data.get("expiry", qr_data.get("expiresAt", "")))
        expiry_iso = None
        ts_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", expiry_text)
        if ts_match:
            expiry_iso = ts_match.group(1).replace(" ", "T") + "Z"

        logger.info("XHS auth: pushing QR code to frontend (expires %s)", expiry_iso or "unknown")
        await self._emit(progress_callback, "auth_required", {
            "service": "xiaohongshu",
            "serviceName": "小红书",
            "qrCodeBase64": qr_base64,
            "expiresAt": expiry_iso,
            "message": "请用小红书 App 扫码登录",
        })

        poll_interval = 5
        max_polls = 24
        for attempt in range(max_polls):
            await asyncio.sleep(poll_interval)
            try:
                check = await asyncio.wait_for(status_tool.execute(), timeout=10)
                check_text = str(check.output) if check.success and check.output else (check.error or "")
                if "已登录" in check_text:
                    logger.info("XHS auth: login recovered after %ds", (attempt + 1) * poll_interval)
                    await self._emit(progress_callback, "auth_success", {
                        "service": "xiaohongshu",
                        "serviceName": "小红书",
                    })
                    return True
            except Exception as e:
                logger.debug("XHS auth poll %d failed: %s", attempt + 1, e)

        logger.warning("XHS auth: login recovery timed out after %ds", max_polls * poll_interval)
        await self._emit(progress_callback, "auth_expired", {
            "service": "xiaohongshu",
            "serviceName": "小红书",
            "message": "登录超时，小红书推荐内容将不可用",
        })
        return False

    async def _prefetch_xiaohongshu(
        self,
        destination: str,
        progress_callback: ProgressCallback | None = None,
    ) -> str | None:
        """Pre-fetch Xiaohongshu travel content for the destination.

        Runs search_feeds -> get_feed_detail (top 1-2 posts) programmatically.
        If the initial search fails, attempts automatic auth recovery via QR code.
        Returns a formatted text block to inject into the LLM context, or None.
        """
        try:
            search_tool = tool_registry.get("xiaohongshu__search_feeds")
            if not search_tool:
                return None

            logger.info("Xiaohongshu pre-fetch: searching for '%s'", destination)
            await self._emit(progress_callback, "thinking", {
                "text": f"Searching Xiaohongshu for '{destination}' travel tips...",
            })

            keyword = f"{destination}旅游攻略"
            search_result = await asyncio.wait_for(
                search_tool.execute(keyword=keyword), timeout=60,
            )

            if not search_result.success or not search_result.output:
                logger.warning("Xiaohongshu search failed: %s — attempting auth recovery", search_result.error)
                recovered = await self._recover_xhs_auth(progress_callback)
                if recovered:
                    logger.info("XHS auth recovered, retrying search")
                    await self._emit(progress_callback, "thinking", {
                        "text": f"Login recovered! Retrying Xiaohongshu search for '{destination}'...",
                    })
                    search_result = await asyncio.wait_for(
                        search_tool.execute(keyword=keyword), timeout=60,
                    )
                    if not search_result.success or not search_result.output:
                        logger.warning("Xiaohongshu search still failed after auth recovery: %s", search_result.error)
                        return None
                else:
                    return None

            return await self._build_xhs_context(search_result, destination, progress_callback)

        except asyncio.TimeoutError:
            logger.warning("Xiaohongshu pre-fetch timed out — attempting auth recovery")
            recovered = await self._recover_xhs_auth(progress_callback)
            if recovered:
                try:
                    logger.info("XHS auth recovered after timeout, retrying search")
                    await self._emit(progress_callback, "thinking", {
                        "text": f"Login recovered! Retrying Xiaohongshu search for '{destination}'...",
                    })
                    search_tool = tool_registry.get("xiaohongshu__search_feeds")
                    if search_tool:
                        keyword = f"{destination}旅游攻略"
                        retry_result = await asyncio.wait_for(
                            search_tool.execute(keyword=keyword), timeout=60,
                        )
                        if retry_result.success and retry_result.output:
                            return await self._build_xhs_context(retry_result, destination, progress_callback)
                except Exception as retry_err:
                    logger.warning("Xiaohongshu retry after auth recovery failed: %s", retry_err)
            return None
        except Exception as e:
            logger.warning("Xiaohongshu pre-fetch failed: %s", e)
            return None

    async def _build_xhs_context(
        self,
        search_result: Any,
        destination: str,
        progress_callback: ProgressCallback | None = None,
    ) -> str | None:
        """Build formatted context text from a successful XHS search result."""
        detail_tool = tool_registry.get("xiaohongshu__get_feed_detail")

        feeds = search_result.output if isinstance(search_result.output, dict) else {}
        feed_list = feeds.get("feeds", [])
        if not feed_list:
            return None

        sections: list[str] = []
        for feed in feed_list[:2]:
            feed_id = feed.get("id", "")
            xsec_token = feed.get("xsecToken", "")
            title = feed.get("noteCard", {}).get("displayTitle", "Unknown")
            interact = feed.get("noteCard", {}).get("interactInfo", {})
            likes = interact.get("likedCount", "?")
            collects = interact.get("collectedCount", "?")
            author = feed.get("noteCard", {}).get("user", {}).get("nickname", "Unknown")
            note_url = (
                f"https://www.xiaohongshu.com/explore/{feed_id}"
                f"?xsec_token={xsec_token}&xsec_source=pc_search"
                if xsec_token else
                f"https://www.xiaohongshu.com/explore/{feed_id}"
            )

            desc_text = ""
            if detail_tool and feed_id and xsec_token:
                try:
                    detail_result = await asyncio.wait_for(
                        detail_tool.execute(feed_id=feed_id, xsec_token=xsec_token),
                        timeout=30,
                    )
                    if detail_result.success and detail_result.output:
                        data = detail_result.output if isinstance(detail_result.output, dict) else {}
                        note = data.get("data", {}).get("note", data)
                        desc_text = note.get("desc", "")
                except Exception as e:
                    logger.warning("Xiaohongshu get_feed_detail failed for %s: %s", feed_id, e)

            section = (
                f"### {title}\n"
                f"Author: {author} | Likes: {likes} | Collects: {collects}\n"
                f"URL: {note_url}\n"
            )
            if desc_text:
                truncated = desc_text[:1500] + ("..." if len(desc_text) > 1500 else "")
                section += f"Content:\n{truncated}\n"
            sections.append(section)

        if not sections:
            return None

        result_text = (
            "=== XIAOHONGSHU (小红书) TRAVEL RECOMMENDATIONS ===\n"
            "The following are popular travel posts from Xiaohongshu. "
            "Use this information to enrich your itinerary with local tips, "
            "restaurant recommendations, and insider knowledge. "
            "Include the Xiaohongshu URLs in source_links (source: 'xiaohongshu').\n\n"
            + "\n---\n".join(sections)
        )

        logger.info(
            "Xiaohongshu pre-fetch: %d posts fetched, %d chars of content",
            len(sections), len(result_text),
        )
        await self._emit(progress_callback, "thinking", {
            "text": f"Found {len(sections)} Xiaohongshu travel posts with tips and recommendations.",
        })
        return result_text

    @staticmethod
    def _inject_xhs_source_links(
        itinerary: StructuredItinerary,
        xhs_context: str,
    ) -> None:
        """Post-process: add Xiaohongshu links to itinerary source_links."""
        xhs_posts: list[dict[str, str]] = []
        for block in xhs_context.split("---"):
            title_m = re.search(r"###\s*(.+)", block)
            url_m = re.search(r"URL:\s*(https://\S+)", block)
            author_m = re.search(r"Author:\s*(.+?)\s*\|", block)
            likes_m = re.search(r"Likes:\s*(\S+)", block)
            if title_m and url_m:
                xhs_posts.append({
                    "title": title_m.group(1).strip(),
                    "url": url_m.group(1).strip(),
                    "snippet": f"by {author_m.group(1).strip() if author_m else '?'}, {likes_m.group(1) if likes_m else '?'} likes",
                })

        if not xhs_posts:
            return

        added = 0
        for day in itinerary.days:
            if not day.activities:
                continue
            act = day.activities[0]
            if act.source_links is None:
                act.source_links = []
            existing_urls = {sl.url for sl in act.source_links}
            xhs_links = []
            for post in xhs_posts:
                if post["url"] not in existing_urls:
                    xhs_links.append(SourceLink(
                        title=post["title"],
                        url=post["url"],
                        source="xiaohongshu",
                        snippet=post["snippet"],
                    ))
            act.source_links = xhs_links + list(act.source_links)
            added += len(xhs_links)
        if added > 0:
            logger.info("Post-processed: added %d Xiaohongshu links to source_links", added)

    # ── Plan outline generation ────────────────────────────────────

    async def _generate_plan_outline(
        self,
        requirements: str,
        input_items: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Generate a quick plan outline from tool findings.

        Uses Chat Completions for a fast, cheap single call.
        """
        tool_summaries = []
        assistant_text = ""
        for item in input_items:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if item.get("type") == "function_call_output":
                    tool_summaries.append(str(item.get("output", ""))[:400])
                elif role == "user" and content:
                    assistant_text = content
            elif hasattr(item, "type") and item.type == "message":
                for part in (item.content or []):
                    if hasattr(part, "text"):
                        assistant_text = part.text

        findings = "\n".join(tool_summaries[-6:]) if tool_summaries else ""
        reasoning = assistant_text[:1500] if assistant_text else ""

        outline_prompt = (
            f"Based on this travel request and research findings, generate a brief plan outline.\n\n"
            f"User request: {requirements}\n\n"
            f"Research findings:\n{findings}\n\n"
            f"Agent analysis:\n{reasoning}\n\n"
            'Generate a JSON outline with:\n'
            '- "summary": One-sentence trip summary\n'
            '- "daily_themes": Array of {"day": N, "theme": "Theme text", "highlight": "Key attraction"}\n'
            '- "estimated_budget": Budget range string\n'
            '- "weather_summary": Brief weather note (if known)\n\n'
            "Keep it concise. Return ONLY JSON."
        )

        try:
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "Generate a brief JSON travel plan outline. Return only valid JSON."},
                    {"role": "user", "content": outline_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=600,
                temperature=0.4,
            )
            content = response.choices[0].message.content or "{}"
            outline = json.loads(content)
            logger.info("Plan outline generated: %d daily themes", len(outline.get("daily_themes", [])))
            return outline
        except Exception as e:
            logger.warning("Plan outline generation failed (non-blocking): %s", e)
            return None

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

    # ── JSON model call (with o-series support) ────────────────────

    @staticmethod
    def _is_o_series(model: str) -> bool:
        return model.startswith("o1") or model.startswith("o3") or model.startswith("o4")

    async def _call_json_model(
        self,
        model: str,
        system: str,
        user: str,
    ) -> tuple[str, str, int, int, int, str]:
        """Call the LLM with json_object response format.

        Returns (content, finish_reason, total_tokens, input_tokens,
                 output_tokens, reasoning_content).
        """
        is_reasoning = self._is_o_series(model)
        system_role = "developer" if is_reasoning else "system"

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": system_role, "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }

        if is_reasoning:
            kwargs["max_completion_tokens"] = settings.max_tokens
        else:
            kwargs["max_tokens"] = settings.max_tokens
            kwargs["temperature"] = self.temperature

        response = await self.client.chat.completions.create(**kwargs)

        _usage = response.usage
        tokens = _usage.total_tokens if _usage else 0
        input_tokens = _usage.prompt_tokens if _usage else 0
        output_tokens = _usage.completion_tokens if _usage else 0
        reasoning_tokens = 0
        if _usage and hasattr(_usage, "completion_tokens_details"):
            details = _usage.completion_tokens_details
            if details and hasattr(details, "reasoning_tokens"):
                reasoning_tokens = details.reasoning_tokens or 0

        finish_reason = response.choices[0].finish_reason or "unknown"
        content = response.choices[0].message.content or ""

        reasoning_content = ""
        msg = response.choices[0].message
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning_content = msg.reasoning_content

        logger.info(
            "JSON model call: model=%s, finish_reason=%s, tokens=%d "
            "(in=%d, out=%d, reasoning=%d), output_len=%d",
            model, finish_reason, tokens, input_tokens, output_tokens,
            reasoning_tokens, len(content),
        )
        return content, finish_reason, tokens, input_tokens, output_tokens, reasoning_content

    # ── Itinerary validation ───────────────────────────────────────

    @staticmethod
    def _validate_itinerary(
        content: str,
    ) -> tuple[StructuredItinerary | None, str | None]:
        if not content or not content.strip():
            return None, "Empty response"

        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e}"

        if not isinstance(data, dict):
            return None, f"Expected JSON object, got {type(data).__name__}"

        try:
            itinerary = StructuredItinerary.model_validate(data)
            return itinerary, None
        except ValidationError as e:
            return None, f"Pydantic validation: {e}"

    def _parse_itinerary(self, text: str) -> StructuredItinerary | None:
        """Try to extract a StructuredItinerary from text (JSON block)."""
        if not text:
            return None
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return StructuredItinerary.model_validate(data)
        except Exception:
            pass
        return None

    # ── Structured final generation (with retry for incomplete days) ──

    async def _generate_structured_final(
        self,
        requirements: str,
        input_items: list[dict[str, Any]],
        progress_callback: ProgressCallback | None,
        llm_calls: list[LLMCallRecord],
        max_retries: int = 2,
    ) -> StructuredItinerary | None:
        """Generate a structured itinerary with prompt-guided JSON generation.

        Uses the Chat Completions API with json_object format for reliable
        structured output, with retry logic for incomplete days.
        """
        final_model = settings.openai_final_model
        total_tokens = 0
        call_records: list[LLMCallRecord] = []

        # Condense tool findings from input_items
        tool_summary_parts: list[str] = []
        assistant_reasoning = ""
        for item in input_items:
            if isinstance(item, dict):
                if item.get("type") == "function_call_output":
                    out = str(item.get("output", ""))
                    tool_summary_parts.append(out[:800] if len(out) > 800 else out)
                elif item.get("role") == "user" and item.get("content"):
                    assistant_reasoning = item["content"]
            elif hasattr(item, "type"):
                if item.type == "message":
                    for part in (item.content or []):
                        if hasattr(part, "text"):
                            assistant_reasoning = part.text

        tool_findings = "\n".join(tool_summary_parts) if tool_summary_parts else "No tools were called."
        reasoning_snippet = assistant_reasoning[:2000] if assistant_reasoning else ""

        final_prompt = f"""Generate a COMPLETE travel itinerary as JSON.

## User Request
{requirements}

## Research Findings (from tools)
{tool_findings}

## Agent Analysis
{reasoning_snippet}

## JSON Structure
Follow this base structure (but with ALL days filled in). The required fields are:
metadata (destination, start_date, end_date, total_days, budget, interests),
days[] (day_number, date, theme, activities[]), activities (activity_id, time, title, description, location, estimated_cost, notes).

Example with both required and optional enrichment fields:

```json
{self._ITINERARY_JSON_EXAMPLE}
```

## CRITICAL RULES
1. The "days" array MUST contain one entry for EVERY day of the trip.
2. Each day must have 2-4 activities with realistic times, real GPS coordinates, and cost estimates.
3. activity_id format: "act-dayN-NNN" (e.g., "act-day3-002").
4. time format: "HH:MM-HH:MM" (e.g., "09:00-11:30").
5. date format: "YYYY-MM-DD".
6. location must include real latitude and longitude coordinates.
7. Do NOT skip, abbreviate, or combine any days into one.
8. Each day MUST include an "alternatives" array with 1-2 alternative full-day schedules.
9. IMPORTANT — add ANY additional fields useful for this trip (distance_from_previous, duration_minutes, etc.).

Return ONLY the JSON object, no other text."""

        system_msg = (
            "You are a travel itinerary generator. Output a single JSON object "
            "following the base structure shown in the example. Include ALL days requested. "
            "You MUST include distance_from_previous for every activity except the first each day. "
            "You MUST include website_url and source_links for EVERY activity. "
            "Add any additional fields valuable for this trip. "
            "Respond with ONLY valid JSON, no markdown, no commentary."
        )

        content, finish_reason, tokens, _in, _out, reasoning = await self._call_json_model(
            model=final_model, system=system_msg, user=final_prompt,
        )
        total_tokens += tokens
        call_records.append(LLMCallRecord(
            label="final_generation",
            model=final_model,
            input_tokens=_in,
            output_tokens=_out,
            cost_usd=_calc_cost(final_model, _in, _out),
        ))

        if reasoning and progress_callback:
            await self._emit(progress_callback, "thinking", {
                "message": reasoning,
                "source": "reasoning_model",
                "model": final_model,
            })

        itinerary, errors = self._validate_itinerary(content)

        if itinerary:
            expected_days = itinerary.metadata.total_days
            actual_days = len(itinerary.days)

            if actual_days >= expected_days:
                logger.info("Itinerary generated: %d days, model=%s", actual_days, final_model)
                llm_calls.extend(call_records)
                return itinerary

            logger.warning("Itinerary has %d/%d days. Retrying...", actual_days, expected_days)

            for retry in range(max_retries):
                retry_content, _, retry_tokens, r_in, r_out, retry_reasoning = await self._call_json_model(
                    model=final_model,
                    system=f"Generate a COMPLETE {expected_days}-day travel itinerary as JSON. You MUST include ALL {expected_days} days.",
                    user=(
                        f"The previous attempt only produced {actual_days} out of {expected_days} days.\n\n"
                        f"User request: {requirements}\n\n"
                        f"Please generate the COMPLETE itinerary with ALL {expected_days} days "
                        f"(day_number 1 through {expected_days}). Each day needs 2-4 activities "
                        "with real coordinates. Do NOT stop after day 1.\n\n"
                        f"JSON structure example:\n```json\n{self._ITINERARY_JSON_EXAMPLE}\n```\n\n"
                        "Return ONLY the JSON object."
                    ),
                )
                total_tokens += retry_tokens
                call_records.append(LLMCallRecord(
                    label=f"final_generation_retry_{retry + 1}",
                    model=final_model,
                    input_tokens=r_in,
                    output_tokens=r_out,
                    cost_usd=_calc_cost(final_model, r_in, r_out),
                ))

                if retry_reasoning and progress_callback:
                    await self._emit(progress_callback, "thinking", {
                        "message": retry_reasoning,
                        "source": "reasoning_model",
                        "model": final_model,
                    })

                retry_itinerary, _ = self._validate_itinerary(retry_content)
                if retry_itinerary and len(retry_itinerary.days) > actual_days:
                    logger.info("Retry %d succeeded: %d days", retry + 1, len(retry_itinerary.days))
                    llm_calls.extend(call_records)
                    return retry_itinerary

            logger.warning("All retries exhausted, returning %d-day itinerary", actual_days)
            llm_calls.extend(call_records)
            return itinerary

        logger.warning("Itinerary validation failed: %s", errors[:300] if errors else "unknown")

        for retry in range(max_retries):
            retry_content, _, retry_tokens, r_in, r_out, retry_reasoning = await self._call_json_model(
                model=final_model, system=system_msg,
                user=(
                    f"Your previous JSON was invalid: {errors[:500]}\n\n"
                    f"Please fix and regenerate the complete itinerary.\n\n"
                    f"User request: {requirements}\n\n"
                    f"JSON structure:\n```json\n{self._ITINERARY_JSON_EXAMPLE}\n```\n\n"
                    "Return ONLY valid JSON."
                ),
            )
            total_tokens += retry_tokens
            call_records.append(LLMCallRecord(
                label=f"validation_retry_{retry + 1}",
                model=final_model,
                input_tokens=r_in,
                output_tokens=r_out,
                cost_usd=_calc_cost(final_model, r_in, r_out),
            ))

            if retry_reasoning and progress_callback:
                await self._emit(progress_callback, "thinking", {
                    "message": retry_reasoning,
                    "source": "reasoning_model",
                    "model": final_model,
                })

            retry_itinerary, _ = self._validate_itinerary(retry_content)
            if retry_itinerary:
                logger.info("Validation retry %d succeeded: %d days", retry + 1, len(retry_itinerary.days))
                llm_calls.extend(call_records)
                return retry_itinerary

        llm_calls.extend(call_records)
        return None

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
        **kwargs: Any,
    ) -> AgentResponse:
        """Generate an itinerary using the Responses API with tool calling.

        Full-featured implementation with Tool-RAG, Xiaohongshu pre-fetch,
        failed tool tracking, plan outline, heartbeat, and retry logic.
        """
        start_time = time.time()
        max_iter = max_iterations or self.max_iterations
        all_tool_traces: list[ToolCallTrace] = []
        llm_calls: list[LLMCallRecord] = []
        total_tokens = 0

        # ── Tool-RAG selection ──
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

        # Build Responses API tools
        function_tools = _build_responses_tools(selected_tool_names)

        if settings.enable_builtin_web_search:
            function_tools.append({
                "type": "web_search",
                "search_context_size": "medium",
            })

        if settings.google_maps_mcp_url:
            function_tools.append({
                "type": "mcp",
                "server_label": "google_maps",
                "server_url": settings.google_maps_mcp_url,
                "require_approval": "never",
            })

        # Build system instructions from selected tools
        if should_use_tool_rag and tool_selection:
            tool_desc_list = [
                {"name": t.name, "description": t.description}
                for t in tool_selection.selected_tools
            ]
        else:
            tool_desc_list = tool_registry.get_tool_descriptions()
        system_instructions = _build_system_prompt(tool_desc_list)

        # ── Xiaohongshu pre-fetch (parallel with first Responses API call) ──
        xhs_destination = self._extract_destination(requirements)
        xhs_task: asyncio.Task[str | None] = asyncio.create_task(
            self._prefetch_xiaohongshu(
                destination=xhs_destination,
                progress_callback=progress_callback,
            )
        )
        xhs_injected = False

        # Build initial input
        input_items: list[dict[str, Any]] = []

        if conversation_context:
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

        try:
            failed_tools: dict[str, str] = {}

            for iteration in range(max_iter):
                logger.info("Responses API iteration %d/%d", iteration + 1, max_iter)

                await self._emit(progress_callback, "stage_change", {
                    "stage": "TOOL_CALLING",
                    "message": f"Agent reasoning (Responses API), iteration {iteration + 1}...",
                })

                # ── Inject Xiaohongshu context when ready ──
                if not xhs_injected and xhs_task.done():
                    try:
                        xhs_context = xhs_task.result()
                        if xhs_context:
                            input_items.append({"role": "system", "content": xhs_context})
                            logger.info("Injected Xiaohongshu context (%d chars) before iteration %d",
                                        len(xhs_context), iteration + 1)
                    except Exception as e:
                        logger.warning("Xiaohongshu pre-fetch result error: %s", e)
                    xhs_injected = True
                elif not xhs_injected and iteration >= 1:
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
                response = await self.client.responses.create(
                    model=self.model,
                    instructions=system_instructions,
                    tools=function_tools,
                    input=input_items,
                    temperature=self.temperature,
                )

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

                # Process output items
                has_function_calls = False
                input_items.extend(response.output)

                for item in response.output:
                    item_type = item.type

                    if item_type == "function_call":
                        has_function_calls = True
                        tool_name = item.name
                        try:
                            arguments = json.loads(item.arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                        await self._emit(progress_callback, "tool_start", {
                            "tool": tool_name,
                            "arguments": arguments,
                        })

                        result = await tool_registry.execute(
                            tool_name,
                            arguments,
                            user_id=user_id,
                            enable_rate_limit=user_id is not None,
                        )

                        trace = ToolCallTrace(
                            call_id=item.call_id,
                            tool_name=tool_name,
                            arguments=arguments,
                            result=result.output,
                            success=result.success,
                            error=result.error,
                            latency_ms=result.latency_ms,
                        )
                        all_tool_traces.append(trace)

                        # Track permanently failed tools
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
                            "call_id": item.call_id,
                            "output": output_content,
                        })

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

                if has_function_calls:
                    continue

                # ── No function calls — LLM is done reasoning ──

                # Last chance: inject Xiaohongshu context
                if not xhs_injected and xhs_task.done():
                    try:
                        xhs_context = xhs_task.result()
                        if xhs_context:
                            input_items.append({"role": "system", "content": xhs_context})
                            logger.info("Injected Xiaohongshu context (%d chars) before final generation",
                                        len(xhs_context))
                    except Exception:
                        pass
                    xhs_injected = True

                # ── Plan outline ──
                logger.info("Tool calling complete, generating plan outline...")
                await self._emit(progress_callback, "stage_change", {
                    "stage": "OUTLINE",
                    "message": "Creating plan outline...",
                })

                outline = await self._generate_plan_outline(
                    requirements=requirements, input_items=input_items,
                )
                if outline:
                    await self._emit(progress_callback, "plan_outline", outline)

                # ── Final generation ──
                await self._emit(progress_callback, "stage_change", {
                    "stage": "GENERATING",
                    "message": "Generating full structured itinerary (Responses API)...",
                })

                final_text = response.output_text or ""
                itinerary = self._parse_itinerary(final_text)

                if not itinerary:
                    itinerary = await self._with_heartbeat(
                        self._generate_structured_final(
                            requirements, input_items, progress_callback, llm_calls,
                        ),
                        progress_callback,
                        interval=8,
                        message="Crafting your detailed itinerary — this takes a moment for quality results...",
                    )

                # Post-process: inject Xiaohongshu links
                if itinerary and xhs_injected and xhs_task.done():
                    try:
                        xhs_ctx = xhs_task.result()
                        if xhs_ctx:
                            self._inject_xhs_source_links(itinerary, xhs_ctx)
                    except Exception:
                        pass

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
