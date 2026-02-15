"""
Restaurant Search Tool.

Searches for restaurants and food establishments using the Overpass API
(OpenStreetMap). This provides free, community-maintained restaurant data
worldwide.  Includes failover across multiple Overpass mirrors for
reliability when the primary endpoint times out or returns errors.

API: Overpass (OpenStreetMap) - free, no key required
Rate Limit: Fair use (~1 request per second)
Cost: Free

Example:
    tool = RestaurantSearchTool()
    result = await tool.execute(
        latitude=35.6762,
        longitude=139.6503,
        radius=1000,
        cuisine="japanese,ramen"
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Overpass API endpoints (failover order)
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]


class RestaurantSearchTool(BaseTool):
    """
    Search for restaurants near a location using OpenStreetMap data.

    Queries the Overpass API for restaurant, cafe, and food-related
    establishments within a given radius.

    Input:
        latitude (float): Center latitude
        longitude (float): Center longitude
        radius (int): Search radius in meters (default 1000, max 5000)
        cuisine (str): Comma-separated cuisine types (optional)
        limit (int): Max results (default 10, max 20)

    Output:
        List of restaurants with name, cuisine, coordinates, and details.
    """

    name = "search_restaurants"
    description = (
        "Search for restaurants, cafes, and food establishments near a location. "
        "Requires latitude and longitude. Can filter by cuisine type "
        "(e.g., japanese, italian, thai, seafood). Returns restaurant names, "
        "types, coordinates, and cuisine information."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "Latitude of the center point",
            },
            "longitude": {
                "type": "number",
                "description": "Longitude of the center point",
            },
            "radius": {
                "type": "integer",
                "description": "Search radius in meters (default 1000, max 5000)",
                "default": 1000,
            },
            "cuisine": {
                "type": "string",
                "description": "Comma-separated cuisine types (e.g., 'japanese,seafood')",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10, max 20)",
                "default": 10,
            },
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False,
    }

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    async def execute(
        self,
        latitude: float,
        longitude: float,
        radius: int = 1000,
        cuisine: str = "",
        limit: int = 10,
    ) -> ToolResult:
        """Search for restaurants near the given coordinates."""
        start_time = time.time()
        input_args = {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius,
            "cuisine": cuisine,
            "limit": limit,
        }

        radius = min(radius, 5000)
        limit = min(limit, 20)

        # Build Overpass QL query
        # Search for nodes and ways tagged as restaurant, cafe, or fast_food
        cuisine_filter = ""
        if cuisine:
            # Build regex filter for cuisine types
            cuisines = [c.strip() for c in cuisine.split(",")]
            cuisine_regex = "|".join(cuisines)
            cuisine_filter = f'["cuisine"~"{cuisine_regex}",i]'

        overpass_query = f"""
[out:json][timeout:25];
(
  node["amenity"="restaurant"]{cuisine_filter}(around:{radius},{latitude},{longitude});
  node["amenity"="cafe"]{cuisine_filter}(around:{radius},{latitude},{longitude});
  way["amenity"="restaurant"]{cuisine_filter}(around:{radius},{latitude},{longitude});
);
out center {limit};
"""

        try:
            data = await self._query_overpass(overpass_query)

            # Parse results
            restaurants = []
            elements = data.get("elements", [])

            for elem in elements[:limit]:
                tags = elem.get("tags", {})
                name = tags.get("name", "")
                if not name:
                    continue

                # Get coordinates (node has lat/lon, way has center)
                lat = elem.get("lat") or (elem.get("center", {}).get("lat"))
                lon = elem.get("lon") or (elem.get("center", {}).get("lon"))

                restaurants.append({
                    "name": name,
                    "cuisine": tags.get("cuisine", ""),
                    "latitude": lat,
                    "longitude": lon,
                    "amenity_type": tags.get("amenity", "restaurant"),
                    "opening_hours": tags.get("opening_hours", ""),
                    "phone": tags.get("phone", ""),
                    "website": tags.get("website", ""),
                    "address": _build_address(tags),
                })

            output = {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius_m": radius,
                "cuisine_filter": cuisine,
                "restaurants": restaurants,
                "count": len(restaurants),
            }

            logger.info(
                "Restaurant search at (%.4f, %.4f): %d results",
                latitude, longitude, len(restaurants),
            )

            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        except httpx.TimeoutException:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Restaurant search timed out (all Overpass endpoints)",
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("Restaurant search failed: %s", e)
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Restaurant search error: {e}",
                latency_ms=int((time.time() - start_time) * 1000),
            )

    async def _query_overpass(self, query: str) -> dict[str, Any]:
        """Query Overpass with failover across mirror endpoints."""
        last_exc: Exception | None = None

        for endpoint in _OVERPASS_ENDPOINTS:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        endpoint,
                        data={"data": query},
                        headers={"User-Agent": "VoyageAI/1.0 (travel-planner)"},
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                logger.warning(
                    "Overpass endpoint %s failed: %s — trying next mirror",
                    endpoint, exc,
                )
                last_exc = exc
                continue

        # All endpoints failed
        raise last_exc or RuntimeError("All Overpass endpoints failed")


def _build_address(tags: dict[str, str]) -> str:
    """Build an address string from OSM tags."""
    parts = []
    for key in ("addr:street", "addr:housenumber", "addr:city", "addr:postcode"):
        if key in tags:
            parts.append(tags[key])
    return ", ".join(parts) if parts else ""
