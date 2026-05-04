"""
Agent Service - DEPRECATED legacy Chat Completions agent.

.. deprecated::
    This module is superseded by ``responses_agent_service.py`` which uses the
    OpenAI Responses API. All shared types (AgentResponse, ProgressCallback,
    LLMCallRecord, _build_system_prompt, etc.) have been extracted to
    ``agent_types.py``. This module re-exports them for backward compatibility.

    New code should import from ``agent_types`` or use ``ResponsesAgentService``
    directly. The ``AgentService`` class remains functional but is no longer
    on the default code path (``settings.use_responses_api`` defaults to True).
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from pydantic import ValidationError

from voyageai.config import settings
from voyageai.rag.tool_rag import ToolRAG, tool_rag
from voyageai.schemas.itinerary import StructuredItinerary
from voyageai.schemas.tool import ToolCallTrace
from voyageai.schemas.tool_metadata import ToolSelectionResult
from voyageai.tools.registry import tool_registry

# Re-export shared types from agent_types for backward compatibility.
# New code should import directly from agent_types.
from voyageai.services.agent_types import (  # noqa: F401
    CORE_TOOLS,
    TOOL_RAG_SKIP_THRESHOLD,
    AgentResponse,
    LLMCallRecord,
    ProgressCallback,
    _AGENT_SYSTEM_PROMPT_TEMPLATE,
    _build_system_prompt,
    _calc_cost,
    _is_permanent_error,
    _PERMANENT_ERROR_PATTERNS,
)

logger = logging.getLogger(__name__)

class AgentService:
    """
    AI Agent with tool calling capability.
    
    This service implements the ReAct (Reason + Act) pattern:
    1. LLM reasons about what tools to call
    2. Tools are executed and results returned
    3. LLM incorporates results into final response
    4. Loop until LLM decides no more tools needed
    
    Module 10 Enhancement (Tool-RAG):
    When use_tool_rag is enabled, the agent uses semantic search to
    select only the most relevant tools for each query, rather than
    providing all tools. This improves efficiency and accuracy.
    
    Key Design Decisions:
    - Max iterations to prevent infinite loops
    - All tool calls tracked for observability
    - Parallel tool execution when multiple tools called
    - Structured output for final itinerary
    - Tool-RAG for dynamic tool selection (Module 10)
    - Rate limiting for tool execution (Module 10)
    """
    
    def __init__(
        self,
        model: str | None = None,
        max_iterations: int = 10,
        temperature: float = 0.7,
        use_tool_rag: bool = False,
        tool_rag_top_k: int = 8,
    ):
        """
        Initialize the agent service.
        
        Args:
            model: OpenAI model to use (default from settings)
            max_iterations: Maximum tool calling iterations
            temperature: LLM temperature for generation
            use_tool_rag: Whether to use Tool-RAG for dynamic tool selection
            tool_rag_top_k: Number of tools to select when using Tool-RAG
        """
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_model
        self.max_iterations = max_iterations
        self.temperature = temperature
        # Module 10: Tool-RAG settings
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
        """Fire-and-forget helper to invoke the progress callback safely."""
        if callback is None:
            return
        try:
            await callback(event_type, data)
        except Exception:
            logger.warning("Progress callback error for event %s", event_type, exc_info=True)

    async def _with_heartbeat(
        self,
        coro: Any,
        callback: ProgressCallback | None,
        interval: float = 10,
        message: str = "Still working...",
    ) -> Any:
        """Run a coroutine with periodic heartbeat 'thinking' events.

        During long LLM calls (e.g., o4-mini final generation taking 30-60s),
        the user sees no progress. This sends periodic events so the frontend
        can show a "still working" indicator.
        """
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

    # ── Phase 2: Pre-flight analysis (clarification questions) ──

    async def analyze_request(
        self,
        requirements: str,
        conversation_context: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Analyze a user request to determine if clarification is needed.

        Makes a fast LLM call (gpt-4o-mini, ~1s) to check for missing info.
        Returns either ``{"ready": True}`` or ``{"ready": False, "questions": [...]}``.

        Detection rules:
        - No destination -> ask where
        - No duration -> ask how many days
        - No dates -> ask when
        - No budget -> ask budget range
        - No interests -> suggest categories
        - Ambiguous destination -> suggest options

        Args:
            requirements: Raw user requirements text.
            conversation_context: Previous conversation history for follow-up requests.
            progress_callback: Optional callback to emit events during analysis.

        Returns:
            Dict with ``ready`` bool and optional ``questions`` list.
        """
        # Include conversation context so follow-up messages are analyzed
        # with full history (e.g., user already mentioned "I'm in Seattle")
        context_section = ""
        if conversation_context:
            context_section = f"""
Previous conversation history:
---
{conversation_context}
---

The user is now sending a follow-up message within the same project.
Consider information from the conversation history when checking for missing details.
For example, if the user previously mentioned a location or duration, that still applies.

"""

        analysis_prompt = f"""Analyze this travel planning request and determine if any critical information is missing.
{context_section}
User request: "{requirements}"

Check for these essentials:
1. Destination (where to go) - is it specific enough?
2. Duration (how many days)
3. Dates or time period (when)
4. Budget level
5. Interests or preferences

If the request has enough information to plan a trip (at least destination + duration or enough context to infer them), respond with:
{{"ready": true}}

If critical information is missing, respond with:
{{
  "ready": false,
  "questions": [
    {{
      "id": "unique_id",
      "question": "What would you like to know?",
      "type": "single_choice|multiple_choice|free_text",
      "options": ["Option A", "Option B", "Option C"]
    }}
  ]
}}

Rules:
- Ask at most 3 questions (focus on the most important missing info)
- If destination and duration are clear, it's usually ready
- For "type", use "single_choice" for budget/dates, "multiple_choice" for interests, "free_text" for open questions
- Options should be practical and concise
- RESPOND WITH ONLY JSON, no other text."""

        try:
            kwargs: dict[str, Any] = {
                "model": settings.openai_model,  # fast model for analysis
                "messages": [
                    {"role": "system", "content": "You analyze travel requests and identify missing information. Respond with JSON only."},
                    {"role": "user", "content": analysis_prompt},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 500,
                "temperature": 0.3,
            }
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)

            if result.get("ready", True):
                logger.info("Request analysis: ready to plan")
                return {"ready": True}

            questions = result.get("questions", [])
            logger.info("Request analysis: needs clarification, %d questions", len(questions))

            # Emit clarification event via callback
            await self._emit(progress_callback, "clarification_needed", {
                "questions": questions,
            })

            return {"ready": False, "questions": questions}

        except Exception as e:
            logger.warning("Request analysis failed, proceeding anyway: %s", e)
            # If analysis fails, proceed with planning rather than blocking
            return {"ready": True}

    async def _execute_tool_calls(
        self,
        message: ChatCompletionMessage,
        user_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[dict[str, Any]], list[ToolCallTrace]]:
        """
        Execute all tool calls from an LLM message.
        
        Args:
            message: LLM message containing tool_calls
            user_id: Optional user ID for rate limiting
            
        Returns:
            Tuple of (tool_results for next LLM call, tool_traces for logging)
        """
        tool_results = []
        tool_traces = []
        
        if not message.tool_calls:
            return tool_results, tool_traces
        
        for tool_call in message.tool_calls:
            call_id = tool_call.id
            tool_name = tool_call.function.name
            
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
                logger.error(f"Failed to parse tool arguments: {tool_call.function.arguments}")
            
            # Emit tool_start event
            await self._emit(progress_callback, "tool_start", {
                "tool": tool_name,
                "arguments": arguments,
            })
            
            # Execute the tool (with rate limiting if user_id provided)
            result = await tool_registry.execute(
                tool_name,
                arguments,
                user_id=user_id,
                enable_rate_limit=user_id is not None
            )
            
            # Create trace record
            trace = ToolCallTrace(
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result.output,
                success=result.success,
                error=result.error,
                latency_ms=result.latency_ms,
            )
            tool_traces.append(trace)
            
            # Emit tool_result event (truncate output for SSE readability)
            result_summary = ""
            if result.success and result.output:
                summary_str = json.dumps(result.output, default=str)
                result_summary = summary_str[:500] + ("..." if len(summary_str) > 500 else "")
            elif result.error:
                result_summary = result.error[:300]
            await self._emit(progress_callback, "tool_result", {
                "tool": tool_name,
                "success": result.success,
                "latency_ms": result.latency_ms,
                "summary": result_summary,
            })
            
            # Format result for LLM
            if result.success:
                result_content = json.dumps(result.output, default=str)
            else:
                result_content = json.dumps({"error": result.error})
            
            tool_results.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_content,
            })
            
            logger.info(
                f"Tool {tool_name}: {'success' if result.success else 'failed'} "
                f"({result.latency_ms}ms)"
            )
        
        return tool_results, tool_traces
    
    # ── Xiaohongshu pre-fetch (programmatic, not LLM-driven) ────────

    @staticmethod
    def _extract_destination(requirements: str) -> str:
        """Extract the destination name from the requirements string.

        Handles formats like:
        - "Plan a 3 day trip to Barcelona"
        - "Plan a 3 day trip to Barcelona\\n\\nAdditional details..."
        - "Tokyo in March for 5 days"
        """
        import re
        # Take the first line only (before additional details)
        first_line = requirements.split("\n")[0].strip()
        # Try "trip to X" pattern
        m = re.search(r"trip to\s+(.+?)(?:\s+from\s|\s+in\s+\w+\s+\d|\s*$)", first_line, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,;:!?")
        # Try "visit X" or "travel to X" pattern
        m = re.search(r"(?:visit|travel(?:ing)?\s+to|going\s+to|heading\s+to)\s+(.+?)(?:\s+from\s|\s+in\s+\w+\s+\d|\s*$)", first_line, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,;:!?")
        # Fallback: use first line (capped at 50 chars)
        return first_line[:50]

    async def _recover_xhs_auth(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> bool:
        """Detect Xiaohongshu login expiry, push QR code via SSE, poll for login.

        Called when a Xiaohongshu search/detail call fails and we suspect
        the headless browser session has expired.

        Flow:
        1. check_login_status → if already logged in, return True
        2. get_login_qrcode → extract base64 image + expiry
        3. Emit ``auth_required`` event via SSE (frontend shows QR code)
        4. Poll check_login_status every 5 s for up to 120 s
        5. On success → emit ``auth_success``; on timeout → emit ``auth_expired``

        Returns:
            True if login was recovered (or was already valid), False otherwise.
        """
        import base64

        status_tool = tool_registry.get("xiaohongshu__check_login_status")
        qr_tool = tool_registry.get("xiaohongshu__get_login_qrcode")
        if not status_tool or not qr_tool:
            logger.warning("XHS auth tools not available, cannot recover login")
            return False

        # 1. Verify login is actually expired
        try:
            status_result = await asyncio.wait_for(
                status_tool.execute(), timeout=15,
            )
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

        # 2. Get QR code
        try:
            qr_result = await asyncio.wait_for(
                qr_tool.execute(), timeout=15,
            )
        except Exception as e:
            logger.warning("XHS get_login_qrcode failed: %s", e)
            return False

        # Extract base64 image and expiry text from MCP response.
        # The MCP tool returns ToolResult whose .output may contain the
        # raw text + image content.  The MCPToolAdapter joins text parts;
        # image data may be embedded as base64 in the output or available
        # as a separate content part.  We handle both cases.
        qr_base64: str | None = None
        expiry_text: str = ""

        if qr_result.success and qr_result.output:
            raw = qr_result.output
            if isinstance(raw, dict):
                qr_base64 = raw.get("qrcode_base64") or raw.get("image") or raw.get("data")
                expiry_text = raw.get("message", "") or raw.get("text", "")
            elif isinstance(raw, str):
                expiry_text = raw
                # Try to extract a base64-encoded PNG embedded in text
                import re
                b64_match = re.search(r"[A-Za-z0-9+/=]{100,}", raw)
                if b64_match:
                    qr_base64 = b64_match.group(0)

        if not qr_base64:
            logger.warning("XHS QR code: could not extract base64 image")
            return False

        # Parse expiry timestamp from text like "请在 2026-02-17 08:30:38 前扫码登录"
        import re
        expiry_iso = ""
        ts_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", expiry_text)
        if ts_match:
            expiry_iso = ts_match.group(1).replace(" ", "T") + "Z"

        # 3. Push auth_required event to frontend via SSE
        logger.info("XHS auth: pushing QR code to frontend (expires %s)", expiry_iso or "unknown")
        await self._emit(progress_callback, "auth_required", {
            "service": "xiaohongshu",
            "serviceName": "小红书",
            "qrCodeBase64": qr_base64,
            "expiresAt": expiry_iso,
            "message": "请用小红书 App 扫码登录",
        })

        # 4. Poll login status every 5 s, up to 120 s
        poll_interval = 5
        max_polls = 24  # 24 * 5s = 120s
        for attempt in range(max_polls):
            await asyncio.sleep(poll_interval)
            try:
                check = await asyncio.wait_for(
                    status_tool.execute(), timeout=10,
                )
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

        # 5. Timed out — user did not scan in time
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

        Runs search_feeds → get_feed_detail (top 1-2 posts) programmatically
        so the LLM doesn't have to call these tools itself (avoiding parameter
        format issues and wasted iterations).

        If the initial search fails (e.g. login expired), attempts automatic
        auth recovery via QR code pushed to the frontend.

        Returns a formatted text block to inject into the LLM context, or None
        if the search fails / returns nothing.
        """
        try:
            search_tool = tool_registry.get("xiaohongshu__search_feeds")
            detail_tool = tool_registry.get("xiaohongshu__get_feed_detail")
            if not search_tool:
                return None

            logger.info("Xiaohongshu pre-fetch: searching for '%s'", destination)
            await self._emit(progress_callback, "thinking", {
                "text": f"Searching Xiaohongshu for '{destination}' travel tips...",
            })

            # Step 1: Search (headless browser can be slow — allow 60s)
            keyword = f"{destination}旅游攻略"
            logger.info("Xiaohongshu search keyword: '%s'", keyword)
            search_result = await asyncio.wait_for(
                search_tool.execute(keyword=keyword),
                timeout=60,
            )

            # If search failed, attempt auth recovery and retry once
            if not search_result.success or not search_result.output:
                logger.warning("Xiaohongshu search failed: %s — attempting auth recovery", search_result.error)
                recovered = await self._recover_xhs_auth(progress_callback)
                if recovered:
                    logger.info("XHS auth recovered, retrying search")
                    await self._emit(progress_callback, "thinking", {
                        "text": f"Login recovered! Retrying Xiaohongshu search for '{destination}'...",
                    })
                    search_result = await asyncio.wait_for(
                        search_tool.execute(keyword=keyword),
                        timeout=60,
                    )
                    if not search_result.success or not search_result.output:
                        logger.warning("Xiaohongshu search still failed after auth recovery: %s", search_result.error)
                        return None
                else:
                    return None

            return await self._build_xhs_context(
                search_result, destination, progress_callback,
            )

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
                            search_tool.execute(keyword=keyword),
                            timeout=60,
                        )
                        if retry_result.success and retry_result.output:
                            # Re-enter the normal flow with the result
                            return await self._build_xhs_context(
                                retry_result, destination, progress_callback,
                            )
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
        """Build formatted context text from a successful XHS search result.

        Parses the feed list from *search_result*, fetches details for the
        top 1-2 posts, and returns a text block ready to inject into the
        LLM conversation.  Returns None if no usable content was found.
        """
        from voyageai.tools.base import ToolResult  # noqa: F811 — local import to avoid circular

        detail_tool = tool_registry.get("xiaohongshu__get_feed_detail")

        feeds = search_result.output if isinstance(search_result.output, dict) else {}
        feed_list = feeds.get("feeds", [])
        if not feed_list:
            return None

        sections: list[str] = []
        posts_to_fetch = feed_list[:2]
        for feed in posts_to_fetch:
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
        """Post-process: add Xiaohongshu links to itinerary source_links.

        The LLM often ignores instructions to include Xiaohongshu links.
        This method extracts post URLs/titles from the pre-fetch context
        and appends them to the first activity of each day as a reliable
        fallback.
        """
        import re

        # Parse post metadata from the context block
        xhs_posts: list[dict[str, str]] = []
        # Pattern: ### TITLE\nAuthor: XXX | Likes: N | Collects: N\nURL: https://...
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
            # Add to first activity of each day
            act = day.activities[0]
            if act.source_links is None:
                act.source_links = []
            # Check if xiaohongshu links already exist
            existing_urls = {sl.url for sl in act.source_links}
            xhs_links = []
            for post in xhs_posts:
                if post["url"] not in existing_urls:
                    from voyageai.schemas.itinerary import SourceLink
                    xhs_links.append(SourceLink(
                        title=post["title"],
                        url=post["url"],
                        source="xiaohongshu",
                        snippet=post["snippet"],
                    ))
            # Prepend xiaohongshu links so they show in the default view
            act.source_links = xhs_links + list(act.source_links)
            added += len(xhs_links)
        if added > 0:
            logger.info("Post-processed: added %d Xiaohongshu links to source_links", added)

    async def _select_tools_with_rag(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[dict[str, Any]], ToolSelectionResult | None]:
        """
        Select relevant tools using Tool-RAG.
        
        Args:
            query: User query for tool selection
            top_k: Number of tools to select (uses default if None)
            
        Returns:
            Tuple of (OpenAI tools list, selection result for logging)
        """
        k = top_k or self.tool_rag_top_k
        
        try:
            # Initialize Tool-RAG if needed
            await self._tool_rag.initialize()
            
            # Select tools
            selection = await self._tool_rag.select_tools(query, top_k=k)
            
            if not selection.selected_tools:
                # Fallback to all tools if Tool-RAG returns nothing
                logger.warning("Tool-RAG returned no tools, falling back to all tools")
                return tool_registry.get_openai_tools(), None
            
            # Convert to OpenAI format
            openai_tools = self._tool_rag.get_openai_tools_from_selection(
                selection, tool_registry
            )
            
            logger.info(
                f"Tool-RAG selected {len(openai_tools)} tools: "
                f"{[t.name for t in selection.selected_tools]} "
                f"({selection.selection_time_ms}ms)"
            )
            
            return openai_tools, selection
            
        except Exception as e:
            logger.error(f"Tool-RAG failed: {e}, falling back to all tools")
            return tool_registry.get_openai_tools(), None
    
    async def generate_with_tools(
        self,
        requirements: str,
        max_iterations: int | None = None,
        use_tool_rag: bool | None = None,
        tool_rag_top_k: int | None = None,
        user_id: str | None = None,
        conversation_context: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResponse:
        """
        Generate an itinerary using the agent with tool calling.
        
        This method:
        1. (Optional) Uses Tool-RAG to select relevant tools
        2. Sends the user requirements to the LLM with available tools
        3. Executes any tool calls the LLM makes
        4. Loops until LLM stops calling tools or max iterations reached
        5. Parses the final response into a StructuredItinerary
        
        Args:
            requirements: User's travel requirements
            max_iterations: Override default max iterations
            use_tool_rag: Override default Tool-RAG setting
            tool_rag_top_k: Override default Tool-RAG top-K
            user_id: User ID for rate limiting
            conversation_context: Previous conversation history for follow-up requests
            progress_callback: Optional async callback for granular progress events.
                Signature: async callback(event_type: str, data: dict) -> None
            
        Returns:
            AgentResponse with itinerary, tool trace, and selection info
        """
        start_time = time.time()
        max_iter = max_iterations or self.max_iterations
        all_tool_traces: list[ToolCallTrace] = []
        total_tokens = 0
        llm_calls: list[LLMCallRecord] = []
        
        # Determine whether to use Tool-RAG
        should_use_tool_rag = use_tool_rag if use_tool_rag is not None else self.use_tool_rag
        rag_top_k = tool_rag_top_k or self.tool_rag_top_k
        
        # Strategy 3: Skip Tool-RAG when total tools < TOOL_RAG_SKIP_THRESHOLD
        total_registered = len(tool_registry.list_tools())
        if should_use_tool_rag and total_registered < TOOL_RAG_SKIP_THRESHOLD:
            logger.info(
                "Tool-RAG skipped: only %d tools registered (threshold=%d), using all",
                total_registered, TOOL_RAG_SKIP_THRESHOLD,
            )
            should_use_tool_rag = False
        
        # Select tools (using Tool-RAG or all tools)
        tool_selection: ToolSelectionResult | None = None
        selected_tool_names: list[str] = []
        
        if should_use_tool_rag:
            await self._emit(progress_callback, "stage_change", {
                "stage": "RAG_SEARCH",
                "message": "Selecting relevant tools for your request...",
            })
            openai_tools, tool_selection = await self._select_tools_with_rag(
                requirements, top_k=rag_top_k
            )
            if tool_selection:
                # Strategy 2: Ensure core tools are always included
                rag_selected_names = {t.name for t in tool_selection.selected_tools}
                missing_core = CORE_TOOLS - rag_selected_names
                for core_name in sorted(missing_core):
                    core_tool = tool_registry.get(core_name)
                    if core_tool:
                        openai_tools.append(core_tool.to_openai_function())
                        # Also add to selection metadata for logging
                        if core_name in self._tool_rag._tool_cache:
                            tool_selection.selected_tools.append(
                                self._tool_rag._tool_cache[core_name]
                            )
                        logger.info("Injected core tool: %s", core_name)

                # Filter out xiaohongshu tools — they are pre-fetched
                # programmatically, so the LLM should not call them directly.
                tool_selection.selected_tools = [
                    t for t in tool_selection.selected_tools
                    if not t.name.startswith("xiaohongshu__")
                ]
                openai_tools = [
                    t for t in openai_tools
                    if not t.get("function", {}).get("name", "").startswith("xiaohongshu__")
                ]

                selected_tool_names = [t.name for t in tool_selection.selected_tools]
                await self._emit(progress_callback, "thinking", {
                    "text": (
                        f"Selected {len(selected_tool_names)} tools: "
                        f"{', '.join(selected_tool_names)} "
                        f"(took {tool_selection.selection_time_ms}ms)"
                        + (f" [+{len(missing_core)} core]" if missing_core else "")
                    ),
                })
        else:
            openai_tools = [
                t for t in tool_registry.get_openai_tools()
                if not t.get("function", {}).get("name", "").startswith("xiaohongshu__")
            ]
            selected_tool_names = [
                n for n in tool_registry.list_tools()
                if not n.startswith("xiaohongshu__")
            ]
        
        # Build system prompt dynamically from whichever tools were selected
        if should_use_tool_rag and tool_selection:
            tool_desc_list = [
                {"name": t.name, "description": t.description}
                for t in tool_selection.selected_tools
            ]
        else:
            tool_desc_list = tool_registry.get_tool_descriptions()
        system_prompt = _build_system_prompt(tool_desc_list)

        # ── Xiaohongshu pre-fetch (runs IN PARALLEL with first LLM iteration) ──
        # Fire-and-forget the pre-fetch task. We'll await it before iteration 2
        # so the LLM can use the results. This saves ~40s of serial waiting.
        xhs_destination = self._extract_destination(requirements)
        xhs_task: asyncio.Task[str | None] = asyncio.create_task(
            self._prefetch_xiaohongshu(
                destination=xhs_destination,
                progress_callback=progress_callback,
            )
        )
        xhs_injected = False  # True once we inject the results into messages
        
        # Initialize messages (without xiaohongshu context — injected later)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        
        # If there's conversation context (follow-up request), include it
        # so the agent understands the previous conversation and can update the plan
        if conversation_context:
            messages.append({
                "role": "user",
                "content": f"""Here is the previous conversation history for context:

{conversation_context}

---

Now the user has a follow-up request. Please update or refine the travel plan based on this new request:

{requirements}

Consider the previous conversation when making changes. Gather any additional information needed using the available tools, then generate an updated complete itinerary."""
            })
            logger.info(
                "Using conversation context (%d chars) for follow-up request",
                len(conversation_context),
            )
        else:
            messages.append({
                "role": "user",
                "content": f"""Please create a travel itinerary based on these requirements:

{requirements}

First, gather relevant information using the available tools, then generate a complete itinerary."""
            })
        
        try:
            # Track tools that have permanently failed (auth errors, etc.)
            # so the LLM is told not to retry them.
            failed_tools: dict[str, str] = {}  # tool_name -> error reason

            for iteration in range(max_iter):
                logger.info(f"Agent iteration {iteration + 1}/{max_iter}")
                
                await self._emit(progress_callback, "stage_change", {
                    "stage": "TOOL_CALLING",
                    "message": f"Agent reasoning, iteration {iteration + 1}...",
                })

                # Inject Xiaohongshu context once the pre-fetch completes.
                # On iteration >= 1, await the result (non-blocking if done).
                if not xhs_injected and xhs_task.done():
                    try:
                        xhs_context = xhs_task.result()
                        if xhs_context:
                            messages.append({
                                "role": "system",
                                "content": xhs_context,
                            })
                            logger.info(
                                "Injected Xiaohongshu context (%d chars) before iteration %d",
                                len(xhs_context), iteration + 1,
                            )
                    except Exception as e:
                        logger.warning("Xiaohongshu pre-fetch result error: %s", e)
                    xhs_injected = True
                elif not xhs_injected and iteration >= 1:
                    # If iteration 2+ and pre-fetch still running, wait briefly
                    if not xhs_task.done():
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(xhs_task), timeout=10,
                            )
                        except asyncio.TimeoutError:
                            logger.info("Xiaohongshu pre-fetch still running at iteration %d, will retry next", iteration + 1)
                        except Exception as e:
                            logger.warning("Xiaohongshu pre-fetch error at iteration %d: %s", iteration + 1, e)
                            xhs_injected = True  # Don't retry on real errors
                    # Check again if done now
                    if xhs_task.done():
                        try:
                            xhs_context = xhs_task.result()
                            if xhs_context:
                                messages.append({
                                    "role": "system",
                                    "content": xhs_context,
                                })
                                logger.info(
                                    "Injected Xiaohongshu context (%d chars) at iteration %d",
                                    len(xhs_context), iteration + 1,
                                )
                        except Exception as e:
                            logger.warning("Xiaohongshu pre-fetch result error: %s", e)
                        xhs_injected = True

                # If there are permanently failed tools, inject an advisory
                # system message so the LLM avoids retrying them.
                if failed_tools:
                    advisory_lines = [
                        f"- {name}: {reason}" for name, reason in failed_tools.items()
                    ]
                    advisory = (
                        "IMPORTANT: The following tools have PERMANENT errors and "
                        "must NOT be retried. Use alternative tools instead:\n"
                        + "\n".join(advisory_lines)
                        + "\nPrefer web_search as a fallback for place/restaurant/hotel data."
                    )
                    messages.append({"role": "system", "content": advisory})
                
                # Call LLM with tools
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
                
                # Track per-call token usage
                _usage = response.usage
                _in_tok = _usage.prompt_tokens if _usage else 0
                _out_tok = _usage.completion_tokens if _usage else 0
                _reason_tok = 0
                if _usage and hasattr(_usage, "completion_tokens_details"):
                    _det = _usage.completion_tokens_details
                    if _det and hasattr(_det, "reasoning_tokens"):
                        _reason_tok = _det.reasoning_tokens or 0
                _call_cost = _calc_cost(self.model, _in_tok, _out_tok)
                llm_calls.append(LLMCallRecord(
                    label=f"tool_calling_iter_{iteration + 1}",
                    model=self.model,
                    input_tokens=_in_tok,
                    output_tokens=_out_tok,
                    reasoning_tokens=_reason_tok,
                    cost_usd=_call_cost,
                ))
                total_tokens += _usage.total_tokens if _usage else 0
                
                choice = response.choices[0]
                assistant_message = choice.message
                
                # Emit thinking event with LLM's reasoning text
                if assistant_message.content:
                    await self._emit(progress_callback, "thinking", {
                        "text": assistant_message.content[:1000],
                    })
                
                # Add assistant message to conversation
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in (assistant_message.tool_calls or [])
                    ] if assistant_message.tool_calls else None,
                })
                
                # Check if LLM wants to call tools
                if assistant_message.tool_calls:
                    # Execute tools (with rate limiting if user_id provided)
                    tool_results, traces = await self._execute_tool_calls(
                        assistant_message,
                        user_id=user_id,
                        progress_callback=progress_callback,
                    )
                    all_tool_traces.extend(traces)

                    # Detect permanently failed tools
                    for trace in traces:
                        if not trace.success and _is_permanent_error(trace.error):
                            if trace.tool_name not in failed_tools:
                                failed_tools[trace.tool_name] = trace.error or "unknown"
                                logger.warning(
                                    "Marking tool %s as permanently failed: %s",
                                    trace.tool_name, trace.error,
                                )
                    
                    # Add tool results to messages
                    messages.extend(tool_results)
                    
                    continue  # Next iteration
                
                # No tool calls — LLM is done reasoning.
                # Last chance: inject Xiaohongshu context before final generation
                if not xhs_injected and xhs_task.done():
                    try:
                        xhs_context = xhs_task.result()
                        if xhs_context:
                            messages.append({"role": "system", "content": xhs_context})
                            logger.info("Injected Xiaohongshu context (%d chars) before final generation", len(xhs_context))
                    except Exception:
                        pass
                    xhs_injected = True

                # Generate a quick plan outline before full generation (Phase 4)
                logger.info("Tool calling complete, generating plan outline...")
                
                await self._emit(progress_callback, "stage_change", {
                    "stage": "OUTLINE",
                    "message": "Creating plan outline...",
                })

                outline = await self._generate_plan_outline(
                    requirements=requirements,
                    messages=messages,
                )
                if outline:
                    await self._emit(progress_callback, "plan_outline", outline)

                await self._emit(progress_callback, "stage_change", {
                    "stage": "GENERATING",
                    "message": "Generating full structured itinerary...",
                })
                
                # Run final generation with periodic heartbeat so the user
                # knows we're still working (o4-mini can take 30-60s).
                # The callback is also passed into _generate_structured_itinerary
                # so it can emit o-series reasoning_content ("thinking") events
                # directly to the frontend — users can watch the AI think in real time.
                itinerary, content, final_tokens, final_llm_calls = (
                    await self._with_heartbeat(
                        self._generate_structured_itinerary(
                            requirements=requirements,
                            messages=messages,
                            progress_callback=progress_callback,
                        ),
                        progress_callback,
                        interval=8,
                        message="Crafting your detailed itinerary — this takes a moment for quality results...",
                    )
                )
                total_tokens += final_tokens
                llm_calls.extend(final_llm_calls)

                # Post-process: inject Xiaohongshu links into source_links
                # if the LLM didn't include them (LLM often ignores this).
                if itinerary and xhs_injected and xhs_task.done():
                    try:
                        xhs_ctx = xhs_task.result()
                        if xhs_ctx:
                            self._inject_xhs_source_links(itinerary, xhs_ctx)
                    except Exception:
                        pass
                
                processing_time = int((time.time() - start_time) * 1000)
                total_cost = round(sum(c.cost_usd for c in llm_calls), 6)
                
                logger.info(
                    f"Agent completed: {len(all_tool_traces)} tool calls, "
                    f"{total_tokens} tokens, ${total_cost:.6f}, {processing_time}ms"
                    + (f", Tool-RAG selected {len(selected_tool_names)} tools" if should_use_tool_rag else "")
                )
                
                # Emit cost_summary event for real-time observability
                await self._emit(progress_callback, "cost_summary", {
                    "total_tokens": total_tokens,
                    "total_cost_usd": total_cost,
                    "llm_calls": len(llm_calls),
                    "tool_calls": len(all_tool_traces),
                    "processing_time_ms": processing_time,
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
                    raw_response=content,
                    success=True,
                    total_tokens=total_tokens,
                    processing_time_ms=processing_time,
                    tool_selection=tool_selection,
                    selected_tool_names=selected_tool_names,
                    llm_calls=llm_calls,
                    total_cost_usd=total_cost,
                )
            
            # Max iterations reached
            raise RuntimeError(f"Agent exceeded max iterations ({max_iter})")
            
        except Exception as e:
            # Cancel any in-flight pre-fetch task
            if not xhs_task.done():
                xhs_task.cancel()
            logger.error(f"Agent failed: {e}")
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
    
    # ── Prompt-guided JSON generation with validation ─────────────────
    
    # JSON structure example embedded in the prompt (replaces strict schema).
    # Kept minimal (2 activities in 1 day) so the model learns the shape
    # without wasting tokens on a full multi-day example.
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
          "description": "Explore Tokyo's oldest Buddhist temple in Asakusa. The Kaminarimon gate and Nakamise shopping street lead to the main hall.",
          "location": {
            "name": "Senso-ji Temple",
            "latitude": 35.7148,
            "longitude": 139.7967,
            "address": "2-3-1 Asakusa, Taito City, Tokyo",
            "place_type": "temple"
          },
          "estimated_cost": "Free",
          "duration_minutes": 120,
          "notes": ["Visit early morning to avoid crowds", "Don't miss the five-story pagoda"],
          "highlights": ["Kaminarimon Thunder Gate", "Nakamise-dori shopping street"],
          "rating": 4.7,
          "booking_required": false,
          "accessibility": "Wheelchair accessible main hall",
          "website_url": "https://www.senso-ji.jp/",
          "source_links": [
            {"title": "Official Website", "url": "https://www.senso-ji.jp/", "source": "official", "snippet": "Tokyo's oldest temple, founded in 645 AD"},
            {"title": "Senso-ji on Google Maps", "url": "https://maps.google.com/?q=Senso-ji+Temple+Tokyo", "source": "google_maps"},
            {"title": "浅草寺打卡攻略", "url": "https://www.xiaohongshu.com/explore/sensoji", "source": "xiaohongshu", "snippet": "小红书旅行达人推荐的浅草寺最佳拍照点和周边美食"}
          ]
        },
        {
          "activity_id": "act-day1-002",
          "time": "12:00-13:30",
          "title": "Lunch at Ramen Street",
          "description": "Sample authentic Tokyo ramen at Tokyo Station's underground ramen alley. Eight acclaimed ramen shops compete for your taste buds.",
          "location": {
            "name": "Tokyo Ramen Street",
            "latitude": 35.6812,
            "longitude": 139.7671,
            "address": "1-9-1 Marunouchi, Chiyoda City, Tokyo",
            "place_type": "restaurant"
          },
          "estimated_cost": "$12-15",
          "duration_minutes": 90,
          "notes": ["Try the tsukemen (dipping noodles)"],
          "distance_from_previous": {
            "km": 4.8,
            "transport_mode": "subway",
            "transport_detail": "Ginza Line from Asakusa to Kanda, then walk",
            "duration_minutes": 20,
            "transit_cost": "¥210 (~$1.50)"
          },
          "cuisine_type": "Japanese Ramen",
          "reservation_tip": "No reservation needed — queue during off-peak hours",
          "website_url": "https://www.tokyoeki-1bangai.co.jp/ramenstreet/",
          "source_links": [
            {"title": "Official Website", "url": "https://www.tokyoeki-1bangai.co.jp/ramenstreet/", "source": "official"},
            {"title": "东京拉面街必吃推荐", "url": "https://www.xiaohongshu.com/explore/ramen-street", "source": "xiaohongshu", "snippet": "8家拉面名店全测评，附排队时间"}
          ]
        }
      ]
    }
  ],
  "tips": ["Get a Suica card for easy transit", "Carry cash — many small shops don't accept cards"],
  "packing_suggestions": ["Comfortable walking shoes", "Portable WiFi or SIM card"],
  "emergency_info": {"police": "110", "ambulance": "119", "embassy_note": "Check your country's embassy in Tokyo"}
}"""

    async def _generate_structured_itinerary(
        self,
        requirements: str,
        messages: list[dict[str, Any]],
        max_retries: int = 2,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[StructuredItinerary, str, int, list[LLMCallRecord]]:
        """
        Generate a structured itinerary using prompt-guided JSON generation.
        
        Strategy:
        1. Condense tool results from the conversation into a summary
        2. Use gpt-4o with json_object mode (no strict schema constraints)
        3. Guide structure via prompt with an example
        4. Validate with Pydantic (lenient: coerce types, fill defaults)
        5. If days are missing, retry with targeted "complete the missing days" prompt
        6. Emit o-series reasoning_content as "thinking" events for frontend display
        
        Args:
            requirements: Original user requirements
            messages: Full conversation messages (used to extract tool summaries)
            max_retries: Max retries for incomplete/invalid output
            
        Returns:
            Tuple of (validated itinerary, raw JSON content, total tokens, llm_call_records)
        """
        total_tokens = 0
        call_records: list[LLMCallRecord] = []
        final_model = settings.openai_final_model
        
        # Condense tool findings from the conversation
        tool_summary_parts: list[str] = []
        assistant_reasoning = ""
        for msg in messages:
            if msg.get("role") == "tool":
                tool_content = msg.get("content", "")
                if len(tool_content) > 800:
                    tool_content = tool_content[:800] + "..."
                tool_summary_parts.append(tool_content)
            elif msg.get("role") == "assistant" and msg.get("content"):
                assistant_reasoning = msg["content"]

        tool_findings = "\n".join(tool_summary_parts) if tool_summary_parts else "No tools were called."
        reasoning_snippet = assistant_reasoning[:2000] if assistant_reasoning else ""
        
        # Build the prompt with structure example
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
1. The "days" array MUST contain one entry for EVERY day of the trip. If the user asks for 6 days, you must produce day_number 1 through 6.
2. Each day must have 2-4 activities with realistic times, real GPS coordinates, and cost estimates.
3. activity_id format: "act-dayN-NNN" (e.g., "act-day3-002").
4. time format: "HH:MM-HH:MM" (e.g., "09:00-11:30").
5. date format: "YYYY-MM-DD".
6. location must include real latitude and longitude coordinates.
7. Do NOT skip, abbreviate, or combine any days into one.
8. Each day MUST include an "alternatives" array with 1-2 alternative full-day schedules (each a list of activities). The primary "activities" is the recommended plan; alternatives highlight different themes.
9. IMPORTANT — you are free to add ANY additional fields that are useful for this specific trip. The schema is flexible. Add fields like:
   - distance_from_previous: {{ km, transport_mode, transport_detail, duration_minutes, transit_cost }} for EVERY activity after the first one each day
   - duration_minutes, highlights, rating, booking_required, booking_url, reservation_tip
   - cuisine_type (for restaurants), accommodation_class (for hotels)
   - weather_forecast, total_walking_km (on day level)
   - best_season, currency, language, packing_suggestions, emergency_info (on metadata/root level)
   Use your judgment — add what's most valuable for THIS specific destination and trip type.

Return ONLY the JSON object, no other text."""

        system_msg = (
            "You are a travel itinerary generator. Output a single JSON object "
            "following the base structure shown in the example. Include ALL days requested. "
            "You MUST include distance_from_previous (with km, transport_mode, duration_minutes) "
            "for every activity except the first one each day. "
            "You MUST include website_url (official website, null if unknown) and "
            "source_links (array of reference links with title, url, source, snippet) "
            "for EVERY activity. Populate source_links from tool results and known URLs. "
            "Add any additional fields you think are valuable for this specific trip — "
            "the schema is flexible and the frontend will render them. "
            "Respond with ONLY valid JSON, no markdown, no commentary."
        )
        
        # First attempt
        content, finish_reason, tokens, _in, _out, reasoning = await self._call_json_model(
            model=final_model,
            system=system_msg,
            user=final_prompt,
        )
        total_tokens += tokens
        call_records.append(LLMCallRecord(
            label="final_generation",
            model=final_model,
            input_tokens=_in,
            output_tokens=_out,
            cost_usd=_calc_cost(final_model, _in, _out),
        ))
        
        # Emit o-series reasoning/thinking content to frontend
        if reasoning and progress_callback:
            await self._emit(progress_callback, "thinking", {
                "message": reasoning,
                "source": "reasoning_model",
                "model": final_model,
            })
        
        # Validate
        itinerary, errors = self._validate_itinerary(content)
        
        if itinerary:
            expected_days = itinerary.metadata.total_days
            actual_days = len(itinerary.days)
            
            if actual_days >= expected_days:
                logger.info(
                    "Itinerary generated successfully: %d days, model=%s, "
                    "finish_reason=%s",
                    actual_days, final_model, finish_reason,
                )
                return itinerary, content, total_tokens, call_records
            
            # Days are missing — retry with targeted completion prompt
            logger.warning(
                "Itinerary has %d/%d days (model=%s, finish_reason=%s). Retrying...",
                actual_days, expected_days, final_model, finish_reason,
            )
            
            for retry in range(max_retries):
                retry_content, retry_finish, retry_tokens, r_in, r_out, retry_reasoning = await self._call_json_model(
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
                
                retry_itinerary, retry_errors = self._validate_itinerary(retry_content)
                if retry_itinerary and len(retry_itinerary.days) > actual_days:
                    logger.info(
                        "Retry %d succeeded: %d days (finish_reason=%s)",
                        retry + 1, len(retry_itinerary.days), retry_finish,
                    )
                    return retry_itinerary, retry_content, total_tokens, call_records
                
                logger.warning(
                    "Retry %d: got %d days (finish_reason=%s, errors=%s)",
                    retry + 1,
                    len(retry_itinerary.days) if retry_itinerary else 0,
                    retry_finish,
                    retry_errors[:200] if retry_errors else "none",
                )
            
            # All retries exhausted — return best result we have
            logger.warning(
                "All retries exhausted, returning %d-day itinerary", actual_days
            )
            return itinerary, content, total_tokens, call_records
        
        # Initial parse/validation failed entirely — retry
        logger.warning(
            "Initial itinerary validation failed (finish_reason=%s): %s",
            finish_reason, errors[:300] if errors else "unknown",
        )
        
        for retry in range(max_retries):
            retry_content, retry_finish, retry_tokens, r_in, r_out, retry_reasoning = await self._call_json_model(
                model=final_model,
                system=system_msg,
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
            
            retry_itinerary, retry_errors = self._validate_itinerary(retry_content)
            if retry_itinerary:
                logger.info(
                    "Validation retry %d succeeded: %d days",
                    retry + 1, len(retry_itinerary.days),
                )
                return retry_itinerary, retry_content, total_tokens, call_records
            
            logger.warning(
                "Validation retry %d failed: %s",
                retry + 1, retry_errors[:200] if retry_errors else "unknown",
            )
        
        raise ValueError(
            f"Failed to generate valid itinerary after {max_retries + 1} attempts. "
            f"Last error: {errors}"
        )

    @staticmethod
    def _is_o_series(model: str) -> bool:
        """Check if the model is an o-series reasoning model (o1, o3, o4, etc.)."""
        # o-series model names: o1, o1-mini, o3, o3-mini, o4-mini, etc.
        return model.startswith("o1") or model.startswith("o3") or model.startswith("o4")

    async def _call_json_model(
        self,
        model: str,
        system: str,
        user: str,
    ) -> tuple[str, str, int, int, int, str]:
        """
        Call the LLM with json_object response format (no strict schema).
        
        Handles differences between standard models (gpt-4o, gpt-4o-mini) and
        o-series reasoning models (o4-mini, o3, etc.):
        - o-series uses max_completion_tokens instead of max_tokens
        - o-series uses developer role instead of system role
        - o-series returns reasoning_content (chain-of-thought thinking)
        
        Returns:
            Tuple of (content, finish_reason, total_tokens, input_tokens,
                      output_tokens, reasoning_content)
        """
        is_reasoning = self._is_o_series(model)
        
        # o-series uses "developer" role instead of "system" role
        system_role = "developer" if is_reasoning else "system"
        
        # Build API kwargs — o-series requires different parameters
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": system_role, "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        
        if is_reasoning:
            # o-series: use max_completion_tokens, don't set temperature
            kwargs["max_completion_tokens"] = settings.max_tokens
        else:
            # Standard models: use max_tokens and temperature
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
        
        # Extract reasoning content (o-series chain-of-thought thinking)
        # This is the model's internal reasoning process that can be shown to users
        reasoning_content = ""
        msg = response.choices[0].message
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning_content = msg.reasoning_content
        
        logger.info(
            "JSON model call: model=%s, finish_reason=%s, tokens=%d "
            "(in=%d, out=%d, reasoning=%d), output_len=%d, reasoning_content_len=%d",
            model, finish_reason, tokens, input_tokens, output_tokens,
            reasoning_tokens, len(content), len(reasoning_content),
        )
        
        return content, finish_reason, tokens, input_tokens, output_tokens, reasoning_content

    @staticmethod
    def _validate_itinerary(
        content: str,
    ) -> tuple[StructuredItinerary | None, str | None]:
        """
        Parse and validate itinerary JSON with lenient Pydantic parsing.
        
        Handles common LLM output quirks:
        - Extra/missing fields (ignored by Pydantic)
        - Minor type mismatches (coerced)
        - Markdown code fences around JSON
        
        Returns:
            Tuple of (validated itinerary or None, error message or None)
        """
        if not content or not content.strip():
            return None, "Empty response"
        
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            # Remove ```json ... ``` wrapping
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
        
        # Lenient validation: allow extra fields, coerce types
        try:
            itinerary = StructuredItinerary.model_validate(data)
            return itinerary, None
        except ValidationError as e:
            return None, f"Pydantic validation: {e}"

    async def _generate_plan_outline(
        self,
        requirements: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Generate a quick plan outline from tool findings (Phase 4).

        Makes a fast LLM call to produce a structured outline that the
        frontend can display for user approval before full generation.

        Returns:
            Dict with summary, daily_themes, estimated_budget, weather_summary.
            None if outline generation fails (non-blocking).
        """
        # Extract tool results for context
        tool_summaries = []
        assistant_text = ""
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                tool_summaries.append(content[:400])
            elif msg.get("role") == "assistant" and msg.get("content"):
                assistant_text = msg["content"]

        findings = "\n".join(tool_summaries[-6:]) if tool_summaries else ""
        reasoning = assistant_text[:1500] if assistant_text else ""

        outline_prompt = f"""Based on this travel request and research findings, generate a brief plan outline.

User request: {requirements}

Research findings:
{findings}

Agent analysis:
{reasoning}

Generate a JSON outline with:
- "summary": One-sentence trip summary
- "daily_themes": Array of {{"day": N, "theme": "Theme text", "highlight": "Key attraction"}}
- "estimated_budget": Budget range string
- "weather_summary": Brief weather note (if known)

Keep it concise. Return ONLY JSON."""

        try:
            kwargs: dict[str, Any] = {
                "model": settings.openai_model,  # fast model
                "messages": [
                    {"role": "system", "content": "Generate a brief JSON travel plan outline. Return only valid JSON."},
                    {"role": "user", "content": outline_prompt},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 600,
                "temperature": 0.4,
            }
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or "{}"
            outline = json.loads(content)
            logger.info("Plan outline generated: %d daily themes", len(outline.get("daily_themes", [])))
            return outline

        except Exception as e:
            logger.warning("Plan outline generation failed (non-blocking): %s", e)
            return None

    async def call_single_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallTrace:
        """
        Call a single tool directly (for testing/debugging).
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            ToolCallTrace with result
        """
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


# Singleton instance (deprecated — use ResponsesAgentService instead)
agent_service = AgentService()

