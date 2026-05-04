"""
Responses API Agent Service — modern alternative to the Chat Completions agent.

Uses OpenAI's Responses API (``client.responses.create()``) which supports:
- Built-in ``web_search`` tool (no API key, billed per-token)
- Native MCP server connections (Google Maps, Xiaohongshu)
- ``previous_response_id`` for server-managed conversation state
- ``FunctionTool`` definitions identical to existing tools
- Structured text output via ``text.format``
- Streaming with granular events

The Responses API eliminates the need for manual message array management
and tool-call loop bookkeeping. The API itself manages the conversation
context and returns structured output items.

This service shares the same ``AgentResponse`` dataclass and
``ProgressCallback`` type as the original ``AgentService``, so the
worker and router layers need no changes.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

from voyageai.config import settings
from voyageai.schemas.itinerary import StructuredItinerary
from voyageai.schemas.tool import ToolCallTrace
from voyageai.services.agent_service import (
    AgentResponse,
    LLMCallRecord,
    ProgressCallback,
    _build_system_prompt,
    _calc_cost,
)
from voyageai.tools.registry import tool_registry

logger = logging.getLogger(__name__)


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
    """Agent using the OpenAI Responses API.

    Drop-in replacement for ``AgentService.generate_with_tools()`` — same
    interface, same return type, different underlying API.
    """

    def __init__(
        self,
        model: str | None = None,
        max_iterations: int = 10,
        temperature: float = 0.7,
    ):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_model
        self.max_iterations = max_iterations
        self.temperature = temperature

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
            logger.warning("Progress callback error for event %s", event_type, exc_info=True)

    async def generate_with_tools(
        self,
        requirements: str,
        max_iterations: int | None = None,
        user_id: str | None = None,
        conversation_context: str | None = None,
        progress_callback: ProgressCallback | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """Generate an itinerary using the Responses API with tool calling.

        Mirrors ``AgentService.generate_with_tools()`` but uses
        ``client.responses.create()`` instead of Chat Completions.
        """
        start_time = time.time()
        max_iter = max_iterations or self.max_iterations
        all_tool_traces: list[ToolCallTrace] = []
        llm_calls: list[LLMCallRecord] = []
        total_tokens = 0

        # Build tools list
        tool_names = [
            n for n in tool_registry.list_tools()
            if not n.startswith("xiaohongshu__")
        ]
        function_tools = _build_responses_tools(tool_names)

        # Add built-in web search
        if settings.enable_builtin_web_search:
            function_tools.append({
                "type": "web_search",
                "search_context_size": "medium",
            })

        # Add native MCP servers
        if settings.google_maps_mcp_url:
            function_tools.append({
                "type": "mcp",
                "server_label": "google_maps",
                "server_url": settings.google_maps_mcp_url,
                "require_approval": "never",
            })

        # Build system instructions
        tool_desc_list = tool_registry.get_tool_descriptions()
        system_instructions = _build_system_prompt(tool_desc_list)

        # Build initial input
        input_items: list[dict[str, Any]] = []

        if conversation_context:
            input_items.append({
                "role": "user",
                "content": (
                    f"Previous conversation:\n{conversation_context}\n\n---\n\n"
                    f"Follow-up request:\n{requirements}\n\n"
                    "Update the travel plan based on this new request."
                ),
            })
        else:
            input_items.append({
                "role": "user",
                "content": (
                    f"Create a travel itinerary:\n\n{requirements}\n\n"
                    "First gather information with tools, then generate a complete itinerary."
                ),
            })

        try:
            for iteration in range(max_iter):
                logger.info("Responses API iteration %d/%d", iteration + 1, max_iter)

                await self._emit(progress_callback, "stage_change", {
                    "stage": "TOOL_CALLING",
                    "message": f"Agent reasoning (Responses API), iteration {iteration + 1}...",
                })

                response = await self.client.responses.create(
                    model=self.model,
                    instructions=system_instructions,
                    tools=function_tools,
                    input=input_items,
                    temperature=self.temperature,
                )

                # Track token usage
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

                # No function calls — generate final itinerary
                await self._emit(progress_callback, "stage_change", {
                    "stage": "GENERATING",
                    "message": "Generating structured itinerary (Responses API)...",
                })

                final_text = response.output_text or ""

                itinerary = self._parse_itinerary(final_text)
                if not itinerary:
                    itinerary = await self._generate_structured_final(
                        requirements, input_items, progress_callback, llm_calls,
                    )

                processing_time = int((time.time() - start_time) * 1000)
                total_cost = round(sum(c.cost_usd for c in llm_calls), 6)

                await self._emit(progress_callback, "cost_summary", {
                    "total_tokens": total_tokens,
                    "total_cost_usd": total_cost,
                    "llm_calls": len(llm_calls),
                    "tool_calls": len(all_tool_traces),
                    "processing_time_ms": processing_time,
                    "api": "responses",
                })

                return AgentResponse(
                    itinerary=itinerary,
                    tool_trace=all_tool_traces,
                    raw_response=final_text,
                    success=True,
                    total_tokens=total_tokens,
                    processing_time_ms=processing_time,
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
                llm_calls=llm_calls,
                total_cost_usd=total_cost,
            )

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

    async def _generate_structured_final(
        self,
        requirements: str,
        input_items: list[dict[str, Any]],
        progress_callback: ProgressCallback | None,
        llm_calls: list[LLMCallRecord],
    ) -> StructuredItinerary | None:
        """Make a dedicated final call with the reasoning model for structured output."""
        final_model = settings.openai_final_model

        generation_prompt = (
            "Based on all the tool results above, generate a COMPLETE travel itinerary "
            "as a valid JSON object.\n\n"
            "The JSON MUST have: metadata (destination, start_date, end_date, total_days, "
            "budget, interests), days (array of day_number, date, theme, activities), and tips.\n\n"
            "Each activity MUST have: activity_id, time, title, description, location "
            "(name, latitude, longitude, address), estimated_cost, notes, website_url, source_links.\n\n"
            "Also include travel_tips with structured tips (category, message, priority, advance_days).\n\n"
            "RESPOND WITH ONLY THE JSON OBJECT. NO markdown, NO explanation."
        )

        input_items.append({"role": "user", "content": generation_prompt})

        try:
            response = await self.client.responses.create(
                model=final_model,
                input=input_items,
                temperature=1.0,
            )

            usage = response.usage
            in_tok = usage.input_tokens if usage else 0
            out_tok = usage.output_tokens if usage else 0
            llm_calls.append(LLMCallRecord(
                label="responses_final_generation",
                model=final_model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=_calc_cost(final_model, in_tok, out_tok),
            ))

            text = response.output_text or ""
            return self._parse_itinerary(text)
        except Exception as e:
            logger.error("Responses API final generation failed: %s", e)
            return None
