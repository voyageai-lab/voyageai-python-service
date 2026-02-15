"""
Agent Service - AI agent with tool calling capability.

This service orchestrates the LLM with tool calls to generate enhanced
travel itineraries. The agent can:
1. Call tools to gather real-time data (weather, currency, etc.)
2. Use tool results to inform itinerary generation
3. Track all tool calls for observability

Module 10 Enhancement (Tool-RAG):
Instead of providing all tools to the LLM (token-heavy, confusing),
we use semantic search to select only the top-K relevant tools based
on the user's query. This:
- Saves tokens (fewer tools in context)
- Improves accuracy (LLM focuses on relevant tools)
- Enables scaling to 100+ tools

Tool Calling Flow:
    User Input → Tool-RAG (select top-K) → LLM (with selected tools) →
    Tool Calls → Execute Tools → Results back to LLM → Final Response

Example:
    agent = AgentService()
    response = await agent.generate_with_tools(
        requirements="Plan a 3-day trip to Tokyo in March",
        max_iterations=5,
        use_tool_rag=True,  # Enable dynamic tool selection
        tool_rag_top_k=3    # Select top 3 relevant tools
    )
"""

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

# Type alias for the progress callback function injected by the worker.
# Signature: async callback(event_type: str, data: dict) -> None
ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

from voyageai.config import settings
from voyageai.rag.tool_rag import ToolRAG, tool_rag
from voyageai.schemas.itinerary import StructuredItinerary
from voyageai.schemas.tool import ToolCallTrace
from voyageai.schemas.tool_metadata import ToolSelectionResult
from voyageai.tools.registry import tool_registry

# Note: make_strict_schema (in ai_service.py) is still used by the standalone AIService
# for non-agent requests, but the agent now uses prompt-guided JSON instead.

logger = logging.getLogger(__name__)

# System prompt for the agent with tools
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

AGENT_SYSTEM_PROMPT = """You are an expert travel planner assistant with access to real-time tools.

Your goal is to create detailed, practical travel itineraries. You have access to the following tools:
- geocode_location: Convert city/place names to coordinates (USE THIS FIRST)
- get_weather_forecast: Get weather forecast (needs coordinates from geocode)
- convert_currency: Convert between currencies for budget planning
- convert_timezone: Convert times between timezones for flight planning
- calculate_distance: Calculate distance between locations
- get_public_holidays: Check for public holidays that might affect plans
- web_search: Search the web for up-to-date travel info, tips, and advisories
- search_attractions: Find tourist attractions and points of interest near a location (free, OSM data)
- search_restaurants: Find restaurants and cafes near a location (free, OSM data)
- search_places_foursquare: Search for places (restaurants, hotels, attractions) using Foursquare's 100M+ POI database with ratings
- search_flights: Search for flight offers between airports with prices and airlines (use IATA airport codes like SEA, NRT, JFK)
- googlemaps__search_places: Search for places using Google Maps (via MCP) with ratings, price levels, and opening hours (very accurate global data)
- googlemaps__get_directions: Get driving/walking/transit directions between two locations with distance, duration, and step-by-step route (via MCP)

IMPORTANT WORKFLOW:
1. First, use geocode_location to get coordinates for the destination
2. Then use other tools (weather, distance, attractions, restaurants) that need coordinates
3. Use search_places_foursquare for high-quality restaurant and hotel recommendations with ratings
4. Use search_flights when the user mentions flying or needs flight info between cities
5. Use googlemaps__search_places for accurate place search with ratings, price levels, and Google Maps URLs
6. Use googlemaps__get_directions when the user needs transit/driving/walking directions between locations
7. Use web_search for destination guides, travel tips, and up-to-date info
6. Check holidays for the destination country
7. Consider currency conversion for budget
8. Finally, generate a comprehensive itinerary

When generating the final itinerary:
- Include specific times for each activity
- Consider weather conditions when planning outdoor activities
- Account for holidays (some attractions may be closed)
- Provide practical budget estimates in local currency
- Use real attraction and restaurant data from tool results
- Include flight prices when available from search_flights
- Include sunrise/sunset times for photography opportunities

Always call relevant tools before generating the final itinerary to ensure accuracy."""


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
            openai_tools = tool_registry.get_openai_tools()
            selected_tool_names = tool_registry.list_tools()
        
        # Build system prompt that mentions the available tools
        system_prompt = AGENT_SYSTEM_PROMPT
        if should_use_tool_rag and tool_selection:
            # Customize system prompt to mention only selected tools
            tool_descriptions = "\n".join([
                f"- {t.name}: {t.description}"
                for t in tool_selection.selected_tools
            ])
            system_prompt = f"""You are an expert travel planner assistant with access to real-time tools.

Your goal is to create detailed, practical travel itineraries. You have access to the following tools:
{tool_descriptions}

IMPORTANT WORKFLOW:
1. First, use geocode_location (if available) to get coordinates for the destination
2. Then use other tools (weather, distance) that need coordinates
3. Check holidays for the destination country (if available)
4. Consider currency conversion for budget (if available)
5. Finally, generate a comprehensive itinerary

When generating the final itinerary:
- Include specific times for each activity
- Consider weather conditions when planning outdoor activities
- Account for holidays (some attractions may be closed)
- Provide practical budget estimates in local currency
- Include sunrise/sunset times for photography opportunities

Always call relevant tools before generating the final itinerary to ensure accuracy."""
        
        # Initialize messages
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
            for iteration in range(max_iter):
                logger.info(f"Agent iteration {iteration + 1}/{max_iter}")
                
                await self._emit(progress_callback, "stage_change", {
                    "stage": "TOOL_CALLING",
                    "message": f"Agent reasoning, iteration {iteration + 1}...",
                })
                
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
                    
                    # Add tool results to messages
                    messages.extend(tool_results)
                    
                    continue  # Next iteration
                
                # No tool calls — LLM is done reasoning.
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
                
                itinerary, content, final_tokens, final_llm_calls = (
                    await self._generate_structured_itinerary(
                        requirements=requirements,
                        messages=messages,
                    )
                )
                total_tokens += final_tokens
                llm_calls.extend(final_llm_calls)
                
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
    "interests": ["culture", "food"]
  },
  "days": [
    {
      "day_number": 1,
      "date": "2024-04-01",
      "theme": "Arrival and City Exploration",
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
          "notes": ["Visit early morning to avoid crowds"]
        },
        {
          "activity_id": "act-day1-002",
          "time": "12:00-13:30",
          "title": "Lunch at Ramen Street",
          "description": "Sample authentic Tokyo ramen at Tokyo Station's underground ramen alley.",
          "location": {
            "name": "Tokyo Ramen Street",
            "latitude": 35.6812,
            "longitude": 139.7671,
            "address": "1-9-1 Marunouchi, Chiyoda City, Tokyo",
            "place_type": "restaurant"
          },
          "estimated_cost": "$12-15",
          "notes": ["Try the tsukemen (dipping noodles)"]
        }
      ]
    }
  ],
  "tips": ["Get a Suica card for easy transit", "Carry cash — many small shops don't accept cards"]
}"""

    async def _generate_structured_itinerary(
        self,
        requirements: str,
        messages: list[dict[str, Any]],
        max_retries: int = 2,
    ) -> tuple[StructuredItinerary, str, int, list[LLMCallRecord]]:
        """
        Generate a structured itinerary using prompt-guided JSON generation.
        
        Strategy:
        1. Condense tool results from the conversation into a summary
        2. Use gpt-4o with json_object mode (no strict schema constraints)
        3. Guide structure via prompt with an example
        4. Validate with Pydantic (lenient: coerce types, fill defaults)
        5. If days are missing, retry with targeted "complete the missing days" prompt
        
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

## Required JSON Structure
Follow this exact structure (but with ALL days filled in):

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

Return ONLY the JSON object, no other text."""

        system_msg = (
            "You are a travel itinerary generator. Output a single JSON object "
            "matching the structure shown in the example. Include ALL days requested. "
            "Respond with ONLY valid JSON, no markdown, no commentary."
        )
        
        # First attempt
        content, finish_reason, tokens, _in, _out = await self._call_json_model(
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
                retry_content, retry_finish, retry_tokens, r_in, r_out = await self._call_json_model(
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
            retry_content, retry_finish, retry_tokens, r_in, r_out = await self._call_json_model(
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
    ) -> tuple[str, str, int, int, int]:
        """
        Call the LLM with json_object response format (no strict schema).
        
        Handles differences between standard models (gpt-4o, gpt-4o-mini) and
        o-series reasoning models (o4-mini, o3, etc.):
        - o-series uses max_completion_tokens instead of max_tokens
        - o-series uses developer role instead of system role
        
        Returns:
            Tuple of (content, finish_reason, total_tokens, input_tokens, output_tokens)
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
        
        logger.info(
            "JSON model call: model=%s, finish_reason=%s, tokens=%d "
            "(in=%d, out=%d, reasoning=%d), output_len=%d",
            model, finish_reason, tokens, input_tokens, output_tokens,
            reasoning_tokens, len(content),
        )
        
        return content, finish_reason, tokens, input_tokens, output_tokens

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


# Singleton instance
agent_service = AgentService()

