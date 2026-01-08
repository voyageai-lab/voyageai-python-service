"""
Agent Service - AI agent with tool calling capability.

This service orchestrates the LLM with tool calls to generate enhanced
travel itineraries. The agent can:
1. Call tools to gather real-time data (weather, currency, etc.)
2. Use tool results to inform itinerary generation
3. Track all tool calls for observability

Tool Calling Flow:
    User Input → LLM (with tools) → Tool Calls → Execute Tools →
    Results back to LLM → Final Response

Example:
    agent = AgentService()
    response = await agent.generate_with_tools(
        requirements="Plan a 3-day trip to Tokyo in March",
        max_iterations=5
    )
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from voyageai.config import settings
from voyageai.schemas.itinerary import StructuredItinerary
from voyageai.schemas.tool import ToolCallTrace
from voyageai.services.ai_service import make_strict_schema
from voyageai.tools.registry import tool_registry

logger = logging.getLogger(__name__)

# System prompt for the agent with tools
AGENT_SYSTEM_PROMPT = """You are an expert travel planner assistant with access to real-time tools.

Your goal is to create detailed, practical travel itineraries. You have access to the following tools:
- geocode_location: Convert city/place names to coordinates (USE THIS FIRST)
- get_weather_forecast: Get weather forecast (needs coordinates from geocode)
- convert_currency: Convert between currencies for budget planning
- convert_timezone: Convert times between timezones for flight planning
- calculate_distance: Calculate distance between locations
- get_public_holidays: Check for public holidays that might affect plans

IMPORTANT WORKFLOW:
1. First, use geocode_location to get coordinates for the destination
2. Then use other tools (weather, distance) that need coordinates
3. Check holidays for the destination country
4. Consider currency conversion for budget
5. Finally, generate a comprehensive itinerary

When generating the final itinerary:
- Include specific times for each activity
- Consider weather conditions when planning outdoor activities
- Account for holidays (some attractions may be closed)
- Provide practical budget estimates in local currency
- Include sunrise/sunset times for photography opportunities

Always call relevant tools before generating the final itinerary to ensure accuracy."""


@dataclass
class AgentResponse:
    """Response from the agent including tool trace."""
    
    itinerary: StructuredItinerary | None = None
    tool_trace: list[ToolCallTrace] = field(default_factory=list)
    raw_response: str = ""
    success: bool = True
    error: str | None = None
    total_tokens: int = 0
    processing_time_ms: int = 0


class AgentService:
    """
    AI Agent with tool calling capability.
    
    This service implements the ReAct (Reason + Act) pattern:
    1. LLM reasons about what tools to call
    2. Tools are executed and results returned
    3. LLM incorporates results into final response
    4. Loop until LLM decides no more tools needed
    
    Key Design Decisions:
    - Max iterations to prevent infinite loops
    - All tool calls tracked for observability
    - Parallel tool execution when multiple tools called
    - Structured output for final itinerary
    """
    
    def __init__(
        self,
        model: str | None = None,
        max_iterations: int = 10,
        temperature: float = 0.7,
    ):
        """
        Initialize the agent service.
        
        Args:
            model: OpenAI model to use (default from settings)
            max_iterations: Maximum tool calling iterations
            temperature: LLM temperature for generation
        """
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_model
        self.max_iterations = max_iterations
        self.temperature = temperature
    
    async def _execute_tool_calls(
        self,
        message: ChatCompletionMessage
    ) -> tuple[list[dict[str, Any]], list[ToolCallTrace]]:
        """
        Execute all tool calls from an LLM message.
        
        Args:
            message: LLM message containing tool_calls
            
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
            
            # Execute the tool
            result = await tool_registry.execute(tool_name, arguments)
            
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
    
    async def generate_with_tools(
        self,
        requirements: str,
        max_iterations: int | None = None,
    ) -> AgentResponse:
        """
        Generate an itinerary using the agent with tool calling.
        
        This method:
        1. Sends the user requirements to the LLM with available tools
        2. Executes any tool calls the LLM makes
        3. Loops until LLM stops calling tools or max iterations reached
        4. Parses the final response into a StructuredItinerary
        
        Args:
            requirements: User's travel requirements
            max_iterations: Override default max iterations
            
        Returns:
            AgentResponse with itinerary and tool trace
        """
        start_time = time.time()
        max_iter = max_iterations or self.max_iterations
        all_tool_traces: list[ToolCallTrace] = []
        total_tokens = 0
        
        # Initialize messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Please create a travel itinerary based on these requirements:

{requirements}

First, gather relevant information using the available tools, then generate a complete itinerary."""
            },
        ]
        
        try:
            for iteration in range(max_iter):
                logger.info(f"Agent iteration {iteration + 1}/{max_iter}")
                
                # Call LLM with tools
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_registry.get_openai_tools(),
                    tool_choice="auto",
                    temperature=self.temperature,
                )
                
                total_tokens += response.usage.total_tokens if response.usage else 0
                
                choice = response.choices[0]
                assistant_message = choice.message
                
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
                    # Execute tools
                    tool_results, traces = await self._execute_tool_calls(assistant_message)
                    all_tool_traces.extend(traces)
                    
                    # Add tool results to messages
                    messages.extend(tool_results)
                    
                    continue  # Next iteration
                
                # No tool calls - LLM is done reasoning
                # Now request structured output
                logger.info("Tool calling complete, generating structured itinerary...")
                
                # Final call for structured output
                final_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages + [{
                        "role": "user",
                        "content": "Now please provide the final itinerary in the required JSON format."
                    }],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "travel_itinerary",
                            "schema": make_strict_schema(
                                StructuredItinerary.model_json_schema()
                            ),
                            "strict": True,
                        },
                    },
                    temperature=self.temperature,
                )
                
                total_tokens += final_response.usage.total_tokens if final_response.usage else 0
                
                content = final_response.choices[0].message.content
                if not content:
                    raise ValueError("Empty final response from LLM")
                
                # Parse and validate itinerary
                itinerary_data = json.loads(content)
                itinerary = StructuredItinerary.model_validate(itinerary_data)
                
                processing_time = int((time.time() - start_time) * 1000)
                
                logger.info(
                    f"Agent completed: {len(all_tool_traces)} tool calls, "
                    f"{total_tokens} tokens, {processing_time}ms"
                )
                
                return AgentResponse(
                    itinerary=itinerary,
                    tool_trace=all_tool_traces,
                    raw_response=content,
                    success=True,
                    total_tokens=total_tokens,
                    processing_time_ms=processing_time,
                )
            
            # Max iterations reached
            raise RuntimeError(f"Agent exceeded max iterations ({max_iter})")
            
        except Exception as e:
            logger.error(f"Agent failed: {e}")
            return AgentResponse(
                itinerary=None,
                tool_trace=all_tool_traces,
                success=False,
                error=str(e),
                total_tokens=total_tokens,
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
    
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

