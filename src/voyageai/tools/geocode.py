"""
Geocode Tool - Convert location names to coordinates.

Uses OpenStreetMap Nominatim API (free, no API key required).
https://nominatim.openstreetmap.org/

Usage Policy:
- Max 1 request per second (we add delay)
- Must include User-Agent header
- No heavy usage (we cache results in practice)

Example:
    tool = GeocodeTool()
    result = await tool.execute(location="Tokyo, Japan")
    # result.output = {"latitude": 35.68, "longitude": 139.76, ...}
"""

import logging
import time

import httpx

from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Nominatim API endpoint (free, no key needed)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Required by Nominatim usage policy
USER_AGENT = "VoyageAI/1.0 (travel-planning-service)"


class GeocodeTool(BaseTool):
    """
    Convert location name to geographic coordinates.
    
    This tool uses OpenStreetMap's Nominatim service for geocoding.
    It's essential for other tools that require lat/lon coordinates.
    
    API: https://nominatim.openstreetmap.org/
    Rate Limit: 1 request/second (we handle this internally)
    Cost: Free
    
    Input:
        location (str): Name of the location (city, address, landmark)
        
    Output:
        {
            "location": "Tokyo, Japan",
            "latitude": 35.6764225,
            "longitude": 139.6500557,
            "display_name": "Tokyo, Japan",
            "type": "city",
            "country": "Japan",
            "country_code": "jp"
        }
    """
    
    name = "geocode_location"
    description = (
        "Convert a location name (city, address, or landmark) to geographic coordinates "
        "(latitude and longitude). Use this before calling weather or distance tools."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Name of the location to geocode (e.g., 'Tokyo, Japan', 'Eiffel Tower')"
            }
        },
        "required": ["location"],
        "additionalProperties": False
    }
    
    def __init__(self, timeout: float = 10.0):
        """
        Initialize the geocode tool.
        
        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout
        self._last_request_time = 0.0
    
    async def _respect_rate_limit(self) -> None:
        """
        Ensure we don't exceed Nominatim's rate limit (1 req/sec).
        
        This is important for compliance with OSM usage policy.
        In production, you'd want a distributed rate limiter.
        """
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 1.0:
            await self._async_sleep(1.0 - elapsed)
        self._last_request_time = time.time()
    
    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(seconds)
    
    async def execute(self, location: str) -> ToolResult:
        """
        Geocode a location name to coordinates.
        
        Args:
            location: Name of the location (e.g., "Paris, France")
            
        Returns:
            ToolResult with coordinates and location details
        """
        start_time = time.time()
        input_args = {"location": location}
        
        if not location or not location.strip():
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Location cannot be empty",
                latency_ms=0
            )
        
        try:
            # Respect rate limit
            await self._respect_rate_limit()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    NOMINATIM_URL,
                    params={
                        "q": location.strip(),
                        "format": "json",
                        "limit": 1,
                        "addressdetails": 1,  # Include address breakdown
                    },
                    headers={"User-Agent": USER_AGENT},
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
            
            if not data:
                return ToolResult(
                    tool_name=self.name,
                    input_args=input_args,
                    output=None,
                    success=False,
                    error=f"Location not found: {location}",
                    latency_ms=int((time.time() - start_time) * 1000)
                )
            
            # Extract the first (most relevant) result
            result = data[0]
            address = result.get("address", {})
            
            output = {
                "location": location,
                "latitude": float(result["lat"]),
                "longitude": float(result["lon"]),
                "display_name": result.get("display_name", location),
                "type": result.get("type", "unknown"),
                "country": address.get("country", ""),
                "country_code": address.get("country_code", "").upper(),
            }
            
            logger.info(f"Geocoded '{location}' -> ({output['latitude']}, {output['longitude']})")
            
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000)
            )
            
        except httpx.TimeoutException:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Geocoding timed out for: {location}",
                latency_ms=int((time.time() - start_time) * 1000)
            )
        except httpx.HTTPStatusError as e:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Geocoding HTTP error: {e.response.status_code}",
                latency_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            logger.error(f"Geocoding failed for '{location}': {e}")
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Geocoding error: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000)
            )

