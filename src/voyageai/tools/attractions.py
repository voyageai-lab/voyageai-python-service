"""
Attractions / Points of Interest Tool.

Searches for attractions and points of interest using the Overpass API
(OpenStreetMap). This API is free, requires no API key, and provides
rich data about tourist attractions, landmarks, cultural sites, and
natural features worldwide.

Strategy:
    1. Query Overpass API for tourism/historic/leisure nodes near coordinates.
    2. Failover across multiple Overpass mirrors for reliability.
    3. Return structured results with name, type, coordinates.

API: Overpass (OpenStreetMap) - free, no key required
Rate Limit: Fair use (~1 request per second)
Cost: Free

Example:
    tool = AttractionsTool()
    result = await tool.execute(
        latitude=35.6762,
        longitude=139.6503,
        radius=5000,
        categories="cultural,architecture"
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

# Map user-friendly category names to OSM tag filters
# Each entry is a list of (key, value) pairs that form an OR clause in Overpass QL
_CATEGORY_OSM_TAGS: dict[str, list[tuple[str, str]]] = {
    "cultural": [
        ("tourism", "museum"),
        ("tourism", "gallery"),
        ("amenity", "arts_centre"),
        ("amenity", "theatre"),
    ],
    "architecture": [
        ("tourism", "attraction"),
        ("building", "cathedral"),
        ("building", "church"),
        ("building", "temple"),
    ],
    "historic": [
        ("historic", "monument"),
        ("historic", "memorial"),
        ("historic", "castle"),
        ("historic", "ruins"),
        ("historic", "archaeological_site"),
        ("historic", "fort"),
    ],
    "natural": [
        ("natural", "peak"),
        ("natural", "waterfall"),
        ("natural", "glacier"),
        ("natural", "volcano"),
        ("natural", "hot_spring"),
        ("leisure", "nature_reserve"),
        ("boundary", "national_park"),
    ],
    "religion": [
        ("amenity", "place_of_worship"),
        ("building", "mosque"),
        ("building", "synagogue"),
    ],
    "museums": [
        ("tourism", "museum"),
    ],
    "amusements": [
        ("tourism", "theme_park"),
        ("leisure", "amusement_arcade"),
        ("leisure", "water_park"),
    ],
    "food": [
        ("amenity", "restaurant"),
        ("amenity", "cafe"),
        ("amenity", "food_court"),
    ],
    "beaches": [
        ("natural", "beach"),
        ("leisure", "beach_resort"),
    ],
}

# Default tags when no category filter is specified (broad tourism search)
_DEFAULT_TAGS: list[tuple[str, str]] = [
    ("tourism", "museum"),
    ("tourism", "attraction"),
    ("tourism", "viewpoint"),
    ("tourism", "gallery"),
    ("historic", "monument"),
    ("historic", "memorial"),
    ("historic", "castle"),
    ("historic", "ruins"),
    ("leisure", "nature_reserve"),
    ("amenity", "theatre"),
]


class AttractionsTool(BaseTool):
    """
    Search for attractions and points of interest near a location.

    Uses the Overpass API (OpenStreetMap) to find tourist attractions, landmarks,
    museums, natural features, and other points of interest.
    Free, no API key required. Failovers across multiple Overpass mirrors.

    Input:
        latitude (float): Latitude of the center point
        longitude (float): Longitude of the center point
        radius (int): Search radius in meters (default 5000, max 25000)
        categories (str): Comma-separated categories to filter
        limit (int): Max results (default 10, max 20)

    Output:
        List of attractions with name, kind, coordinates, and details.
    """

    name = "search_attractions"
    description = (
        "Search for tourist attractions, landmarks, museums, and points of interest "
        "near a specific location. Requires latitude and longitude coordinates. "
        "Can filter by categories: cultural, architecture, historic, natural, "
        "religion, museums, amusements, food, beaches."
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
                "description": "Search radius in meters (default 5000, max 25000)",
                "default": 5000,
            },
            "categories": {
                "type": "string",
                "description": (
                    "Comma-separated categories: cultural, architecture, historic, "
                    "natural, religion, museums, amusements, food, beaches"
                ),
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
        radius: int = 5000,
        categories: str = "",
        limit: int = 10,
    ) -> ToolResult:
        """Search for attractions near the given coordinates."""
        start_time = time.time()
        input_args = {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius,
            "categories": categories,
            "limit": limit,
        }

        radius = min(radius, 25000)
        limit = min(limit, 20)

        # Resolve OSM tag filters from categories
        tag_filters = self._resolve_tags(categories)

        # Build Overpass QL query
        query = self._build_overpass_query(latitude, longitude, radius, limit, tag_filters)

        try:
            data = await self._query_overpass(query)

            # Parse results
            attractions = []
            elements = data.get("elements", [])

            for elem in elements[:limit]:
                tags = elem.get("tags", {})
                name = tags.get("name", "")
                if not name:
                    continue  # Skip unnamed features

                lat = elem.get("lat") or (elem.get("center", {}).get("lat"))
                lon = elem.get("lon") or (elem.get("center", {}).get("lon"))

                # Build a human-readable kind string
                kind = self._infer_kind(tags)

                attractions.append({
                    "name": name,
                    "kind": kind,
                    "latitude": lat,
                    "longitude": lon,
                    "distance_m": None,
                    "description": tags.get("description", ""),
                    "wikipedia": tags.get("wikipedia", ""),
                    "website": tags.get("website", ""),
                    "opening_hours": tags.get("opening_hours", ""),
                })

            output = {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius_m": radius,
                "categories": categories,
                "attractions": attractions,
                "count": len(attractions),
            }

            logger.info(
                "Attractions search at (%.4f, %.4f): %d results",
                latitude, longitude, len(attractions),
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
                error="Attractions search timed out (all Overpass endpoints)",
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("Attractions search failed: %s", e)
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Attractions search error: {e}",
                latency_ms=int((time.time() - start_time) * 1000),
            )

    def _resolve_tags(self, categories: str) -> list[tuple[str, str]]:
        """Convert user-friendly category names to OSM tag pairs."""
        if not categories:
            return _DEFAULT_TAGS

        tags: list[tuple[str, str]] = []
        for cat in categories.split(","):
            cat = cat.strip().lower()
            if cat in _CATEGORY_OSM_TAGS:
                tags.extend(_CATEGORY_OSM_TAGS[cat])
            else:
                # Unknown category — try as a generic tourism tag
                tags.append(("tourism", cat))

        return tags if tags else _DEFAULT_TAGS

    @staticmethod
    def _build_overpass_query(
        lat: float, lon: float, radius: int, limit: int,
        tags: list[tuple[str, str]],
    ) -> str:
        """Build an Overpass QL query from tag filters."""
        # Each tag pair becomes a node + way query within the radius
        union_parts: list[str] = []
        for key, value in tags:
            union_parts.append(
                f'  node["{key}"="{value}"](around:{radius},{lat},{lon});'
            )
            union_parts.append(
                f'  way["{key}"="{value}"](around:{radius},{lat},{lon});'
            )

        union_body = "\n".join(union_parts)
        # out center resolves way centroids; out meta gives more tags
        return f"""[out:json][timeout:25];
(
{union_body}
);
out center {limit};
"""

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

    @staticmethod
    def _infer_kind(tags: dict[str, str]) -> str:
        """Build a human-readable kind string from OSM tags."""
        parts: list[str] = []
        for key in ("tourism", "historic", "natural", "leisure", "amenity"):
            val = tags.get(key)
            if val:
                parts.append(val)
        return ",".join(parts) if parts else "attraction"
