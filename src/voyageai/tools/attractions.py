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
        radius (int): Search radius in meters (default 5000, max 50000)
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
        "religion, museums, amusements, food, beaches. "
        "Use a larger radius (10000-50000) for island or rural destinations."
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
                "description": "Search radius in meters (default 5000, max 50000). Use 10000-50000 for island or rural areas.",
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

    # Maximum allowed radius (meters)
    _MAX_RADIUS = 50000

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    async def execute(
        self,
        latitude: float,
        longitude: float,
        radius: int = 5000,
        categories: str = "",
        limit: int = 10,
    ) -> ToolResult:
        """Search for attractions near the given coordinates.

        If the initial search returns no results and the radius is below
        the maximum, automatically retries with an expanded radius (3x)
        to handle sparse/remote locations like islands or rural areas.
        """
        start_time = time.time()
        input_args = {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius,
            "categories": categories,
            "limit": limit,
        }

        radius = min(radius, self._MAX_RADIUS)
        limit = min(limit, 20)

        # Resolve OSM tag filters from categories
        tag_filters = self._resolve_tags(categories)

        try:
            attractions, final_radius = await self._search_with_auto_expand(
                latitude, longitude, radius, limit, tag_filters,
            )

            output = {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius_m": final_radius,
                "categories": categories,
                "attractions": attractions,
                "count": len(attractions),
            }
            if final_radius != radius:
                output["auto_expanded"] = True
                output["original_radius_m"] = radius

            logger.info(
                "Attractions search at (%.4f, %.4f): %d results (radius=%dm)",
                latitude, longitude, len(attractions), final_radius,
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

    async def _search_with_auto_expand(
        self,
        latitude: float,
        longitude: float,
        radius: int,
        limit: int,
        tag_filters: list[tuple[str, str]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Run the Overpass query, auto-expanding radius on empty results.

        Returns:
            Tuple of (attraction list, final radius used).
        """
        current_radius = radius

        while True:
            query = self._build_overpass_query(
                latitude, longitude, current_radius, limit, tag_filters,
            )
            data = await self._query_overpass(query)
            elements = data.get("elements", [])

            attractions = self._parse_elements(elements, limit)

            if attractions or current_radius >= self._MAX_RADIUS:
                return attractions, current_radius

            # Auto-expand: triple the radius up to max
            expanded = min(current_radius * 3, self._MAX_RADIUS)
            logger.info(
                "No attractions at %dm, expanding radius to %dm",
                current_radius, expanded,
            )
            current_radius = expanded

    def _parse_elements(
        self, elements: list[dict[str, Any]], limit: int,
    ) -> list[dict[str, Any]]:
        """Parse Overpass elements into attraction dicts."""
        attractions: list[dict[str, Any]] = []
        for elem in elements[:limit]:
            tags = elem.get("tags", {})
            name = tags.get("name", "")
            if not name:
                continue

            lat = elem.get("lat") or (elem.get("center", {}).get("lat"))
            lon = elem.get("lon") or (elem.get("center", {}).get("lon"))

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
        return attractions

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
        return f"""[out:json][timeout:10];
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
