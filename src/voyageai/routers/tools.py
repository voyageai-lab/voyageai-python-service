"""
Tools Router - Endpoints for testing and invoking tools directly.

These endpoints are primarily for:
1. Testing individual tools work correctly
2. Debugging tool behavior
3. Demo purposes

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

from voyageai.schemas.tool import ToolCallRequest, ToolCallResponse
from voyageai.services.agent_service import agent_service
from voyageai.tools.registry import tool_registry

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
    trace = await agent_service.call_single_tool(
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

