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
        radius (int): Search radius in meters (default 3000, max 15000)
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
        "types, coordinates, and cuisine information. "
        "Use a larger radius (5000-15000) for island or rural destinations."
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
                "description": "Search radius in meters (default 3000, max 15000). Use 5000-15000 for island or rural areas.",
                "default": 3000,
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

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    # Maximum allowed radius (meters)
    _MAX_RADIUS = 15000

    async def execute(
        self,
        latitude: float,
        longitude: float,
        radius: int = 3000,
        cuisine: str = "",
        limit: int = 10,
    ) -> ToolResult:
        """Search for restaurants near the given coordinates.

        If the initial search returns no results and the radius is below
        the maximum, automatically retries with an expanded radius (3x)
        to handle sparse/remote locations like islands or rural areas.
        """
        start_time = time.time()
        input_args = {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius,
            "cuisine": cuisine,
            "limit": limit,
        }

        radius = min(radius, self._MAX_RADIUS)
        limit = min(limit, 20)

        try:
            restaurants, final_radius = await self._search_with_auto_expand(
                latitude, longitude, radius, cuisine, limit,
            )

            output = {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius_m": final_radius,
                "cuisine_filter": cuisine,
                "restaurants": restaurants,
                "count": len(restaurants),
            }
            if final_radius != radius:
                output["auto_expanded"] = True
                output["original_radius_m"] = radius

            logger.info(
                "Restaurant search at (%.4f, %.4f): %d results (radius=%dm)",
                latitude, longitude, len(restaurants), final_radius,
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

    async def _search_with_auto_expand(
        self,
        latitude: float,
        longitude: float,
        radius: int,
        cuisine: str,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Run the Overpass query, auto-expanding radius on empty results.

        Returns:
            Tuple of (restaurant list, final radius used).
        """
        current_radius = radius

        while True:
            elements = await self._run_overpass_query(
                latitude, longitude, current_radius, cuisine, limit,
            )

            restaurants = self._parse_elements(elements, limit)

            if restaurants or current_radius >= self._MAX_RADIUS:
                return restaurants, current_radius

            # Auto-expand: triple the radius up to max
            expanded = min(current_radius * 3, self._MAX_RADIUS)
            logger.info(
                "No restaurants at %dm, expanding radius to %dm",
                current_radius, expanded,
            )
            current_radius = expanded

    async def _run_overpass_query(
        self,
        latitude: float,
        longitude: float,
        radius: int,
        cuisine: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Build and execute the Overpass QL query, return raw elements."""
        cuisine_filter = ""
        if cuisine:
            cuisines = [c.strip() for c in cuisine.split(",")]
            cuisine_regex = "|".join(cuisines)
            cuisine_filter = f'["cuisine"~"{cuisine_regex}",i]'

        overpass_query = f"""
[out:json][timeout:10];
(
  node["amenity"="restaurant"]{cuisine_filter}(around:{radius},{latitude},{longitude});
  node["amenity"="cafe"]{cuisine_filter}(around:{radius},{latitude},{longitude});
  way["amenity"="restaurant"]{cuisine_filter}(around:{radius},{latitude},{longitude});
);
out center {limit};
"""
        data = await self._query_overpass(overpass_query)
        return data.get("elements", [])

    @staticmethod
    def _parse_elements(
        elements: list[dict[str, Any]], limit: int,
    ) -> list[dict[str, Any]]:
        """Parse Overpass elements into restaurant dicts."""
        restaurants: list[dict[str, Any]] = []
        for elem in elements[:limit]:
            tags = elem.get("tags", {})
            name = tags.get("name", "")
            if not name:
                continue

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
        return restaurants

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
