"""
Tools Router - Endpoints for testing and invoking tools directly.

These endpoints are primarily for:
1. Testing individual tools work correctly
2. Debugging tool behavior
3. Demo purposes
4. Module 10: Tool-RAG selection testing

In production, tools are typically called by the agent service,
not directly via these endpoints.

Security Note:
These endpoints should be protected or disabled in production
to prevent unauthorized tool invocation.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from voyageai.rag.tool_rag import tool_rag
from voyageai.schemas.tool import ToolCallRequest, ToolCallResponse
from voyageai.schemas.tool_metadata import ToolMetadata, ToolSelectionResult
from voyageai.services.responses_agent_service import ResponsesAgentService
from voyageai.tools.rate_limiter import rate_limiter
from voyageai.tools.registry import tool_registry

_agent_service = ResponsesAgentService()

logger = logging.getLogger(__name__)

router = APIRouter()


class ToolInfo(BaseModel):
    """Information about an available tool."""
    name: str
    description: str
    parameters: dict[str, Any]


class ToolListResponse(BaseModel):
    """Response containing all available tools."""
    tools: list[ToolInfo]
    total: int


# ============================================================================
# Tool Discovery Endpoints
# ============================================================================

@router.get("/tools", response_model=ToolListResponse)
async def list_tools() -> ToolListResponse:
    """
    List all available tools.
    
    Returns information about each tool including:
    - Name: Used to invoke the tool
    - Description: What the tool does
    - Parameters: Expected input schema
    """
    tools = []
    for tool in tool_registry._tools.values():
        tools.append(ToolInfo(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters_schema,
        ))
    
    return ToolListResponse(tools=tools, total=len(tools))


# ============================================================================
# Generic Tool Invocation
# ============================================================================

@router.post("/tools/invoke", response_model=ToolCallResponse)
async def invoke_tool(request: ToolCallRequest) -> ToolCallResponse:
    """
    Invoke a tool by name with the provided arguments.
    
    This is a generic endpoint that can call any registered tool.
    
    Example:
        POST /api/v1/tools/invoke
        {
            "tool_name": "geocode_location",
            "arguments": {"location": "Tokyo, Japan"}
        }
    """
    trace = await _agent_service.call_single_tool(
        request.tool_name,
        request.arguments,
    )
    
    return ToolCallResponse(
        tool_name=trace.tool_name,
        success=trace.success,
        result=trace.result,
        error=trace.error,
        latency_ms=trace.latency_ms,
    )


# ============================================================================
# Individual Tool Endpoints (for easier testing via GET)
# ============================================================================

@router.get("/tools/geocode")
async def geocode_location(
    location: str = Query(..., description="Location to geocode (e.g., 'Tokyo, Japan')")
) -> dict[str, Any]:
    """
    Geocode a location name to coordinates.
    
    Example: /api/v1/tools/geocode?location=Paris,%20France
    """
    result = await tool_registry.execute("geocode_location", {"location": location})
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "success": True,
        "data": result.output,
        "latency_ms": result.latency_ms,
    }


@router.get("/tools/weather")
async def get_weather(
    latitude: float = Query(..., ge=-90, le=90, description="Latitude"),
    longitude: float = Query(..., ge=-180, le=180, description="Longitude"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
) -> dict[str, Any]:
    """
    Get weather forecast for a location.
    
    Example: /api/v1/tools/weather?latitude=35.68&longitude=139.76&start_date=2026-01-10&end_date=2026-01-12
    """
    result = await tool_registry.execute("get_weather_forecast", {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
    })
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "success": True,
        "data": result.output,
        "latency_ms": result.latency_ms,
    }


@router.get("/tools/currency")
async def convert_currency(
    from_currency: str = Query(..., description="Source currency code (e.g., USD)"),
    to_currency: str = Query(..., description="Target currency code (e.g., JPY)"),
    amount: float = Query(..., gt=0, description="Amount to convert"),
) -> dict[str, Any]:
    """
    Convert currency.
    
    Example: /api/v1/tools/currency?from_currency=USD&to_currency=JPY&amount=100
    """
    result = await tool_registry.execute("convert_currency", {
        "from_currency": from_currency,
        "to_currency": to_currency,
        "amount": amount,
    })
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "success": True,
        "data": result.output,
        "latency_ms": result.latency_ms,
    }


@router.get("/tools/timezone")
async def convert_timezone(
    time: str = Query(..., description="Time to convert (ISO format or HH:MM)"),
    from_timezone: str = Query(..., description="Source timezone (e.g., America/New_York)"),
    to_timezone: str = Query(..., description="Target timezone (e.g., Asia/Tokyo)"),
    date: str | None = Query(None, description="Date if time is HH:MM (YYYY-MM-DD)"),
) -> dict[str, Any]:
    """
    Convert time between timezones.
    
    Example: /api/v1/tools/timezone?time=14:00&from_timezone=PST&to_timezone=JST&date=2026-01-10
    """
    result = await tool_registry.execute("convert_timezone", {
        "time": time,
        "from_timezone": from_timezone,
        "to_timezone": to_timezone,
        "date": date,
    })
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "success": True,
        "data": result.output,
        "latency_ms": result.latency_ms,
    }


@router.get("/tools/distance")
async def calculate_distance(
    from_latitude: float = Query(..., ge=-90, le=90, description="Starting latitude"),
    from_longitude: float = Query(..., ge=-180, le=180, description="Starting longitude"),
    to_latitude: float = Query(..., ge=-90, le=90, description="Destination latitude"),
    to_longitude: float = Query(..., ge=-180, le=180, description="Destination longitude"),
) -> dict[str, Any]:
    """
    Calculate distance between two points.
    
    Example: /api/v1/tools/distance?from_latitude=35.68&from_longitude=139.76&to_latitude=34.69&to_longitude=135.50
    """
    result = await tool_registry.execute("calculate_distance", {
        "from_latitude": from_latitude,
        "from_longitude": from_longitude,
        "to_latitude": to_latitude,
        "to_longitude": to_longitude,
    })
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "success": True,
        "data": result.output,
        "latency_ms": result.latency_ms,
    }


@router.get("/tools/holidays")
async def get_holidays(
    country_code: str = Query(..., description="ISO country code (e.g., JP, US)"),
    year: int = Query(..., description="Year (e.g., 2026)"),
) -> dict[str, Any]:
    """
    Get public holidays for a country.
    
    Example: /api/v1/tools/holidays?country_code=JP&year=2026
    """
    result = await tool_registry.execute("get_public_holidays", {
        "country_code": country_code,
        "year": year,
    })
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "success": True,
        "data": result.output,
        "latency_ms": result.latency_ms,
    }


# ============================================================================
# Module 10: Tool-RAG Endpoints
# ============================================================================

class ToolSelectRequest(BaseModel):
    """Request for tool selection via Tool-RAG."""
    query: str = Field(..., description="User query for tool selection")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of tools to select")


class ToolSelectResponse(BaseModel):
    """Response from Tool-RAG selection."""
    query: str
    selected_tools: list[dict[str, Any]]
    scores: list[float]
    total_available: int
    selection_time_ms: int


@router.post("/tools/select", response_model=ToolSelectResponse)
async def select_tools(request: ToolSelectRequest) -> ToolSelectResponse:
    """
    Select relevant tools using Tool-RAG.
    
    This endpoint demonstrates the Tool-RAG capability:
    - Takes a user query
    - Returns top-K semantically similar tools
    - Shows similarity scores for explainability
    
    Example:
        POST /api/v1/tools/select
        {
            "query": "What's the weather like in Tokyo?",
            "top_k": 3
        }
    """
    try:
        await tool_rag.initialize()
        result = await tool_rag.select_tools(request.query, top_k=request.top_k)
        
        return ToolSelectResponse(
            query=result.query,
            selected_tools=[
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                }
                for t in result.selected_tools
            ],
            scores=result.scores,
            total_available=result.total_tools_available,
            selection_time_ms=result.selection_time_ms,
        )
    except Exception as e:
        logger.error(f"Tool selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/rag/count")
async def get_tool_rag_count() -> dict[str, Any]:
    """
    Get the number of tools in the Tool-RAG collection.
    
    Useful for checking if tools have been seeded.
    """
    try:
        await tool_rag.initialize()
        count = tool_rag.count()
        return {
            "count": count,
            "message": "Run 'python scripts/seed_tools.py' if count is 0"
        }
    except Exception as e:
        logger.error(f"Failed to get Tool-RAG count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/rag/list")
async def list_tool_rag_tools() -> dict[str, Any]:
    """
    List all tools in the Tool-RAG collection.
    
    Returns metadata for all indexed tools.
    """
    try:
        await tool_rag.initialize()
        tools = await tool_rag.list_all_tools()
        
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "example_queries": t.example_queries[:3],  # Limit for readability
                    "rate_limit_per_minute": t.rate_limit_per_minute,
                }
                for t in tools
            ],
            "total": len(tools),
        }
    except Exception as e:
        logger.error(f"Failed to list Tool-RAG tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Module 10: Rate Limiting Endpoints
# ============================================================================

@router.get("/tools/ratelimit/status")
async def get_rate_limit_status(
    user_id: str = Query(..., description="User ID to check"),
    tool_name: str = Query(..., description="Tool name to check"),
) -> dict[str, Any]:
    """
    Get rate limit status for a user/tool combination.
    
    Returns:
    - Remaining tokens
    - Max tokens
    - Time until full refill
    
    Example: /api/v1/tools/ratelimit/status?user_id=user123&tool_name=geocode_location
    """
    try:
        status = await rate_limiter.get_status(user_id, tool_name)
        return {
            "user_id": status["user_id"],
            "tool_name": status["tool_name"],
            "tokens_remaining": status["tokens_remaining"],
            "max_tokens": status["max_tokens"],
            "refill_rate_per_second": status["refill_rate_per_second"],
            "is_rate_limited": status["tokens_remaining"] <= 0,
        }
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/ratelimit/reset")
async def reset_rate_limit(
    user_id: str = Query(..., description="User ID to reset"),
    tool_name: str | None = Query(None, description="Tool name (optional, resets all if not provided)"),
) -> dict[str, Any]:
    """
    Reset rate limit for a user.
    
    Admin endpoint for clearing rate limit buckets.
    
    Example: /api/v1/tools/ratelimit/reset?user_id=user123&tool_name=geocode_location
    """
    try:
        await rate_limiter.reset(user_id, tool_name)
        return {
            "success": True,
            "message": f"Rate limit reset for user {user_id}" + (f" tool {tool_name}" if tool_name else " (all tools)"),
        }
    except Exception as e:
        logger.error(f"Failed to reset rate limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

