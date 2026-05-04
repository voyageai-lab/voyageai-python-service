"""
Foursquare Places Search Tool.

Searches for places (restaurants, attractions, hotels, etc.) using the
Foursquare Places API v3. Provides rich data including ratings, categories,
photos, and tips from 100M+ global POIs.

API: Foursquare Places API v3
Auth: Bearer token (Service API Key)
Free Tier: Starter credit (no credit card required)
Rate Limit: Varies by plan

Example:
    tool = FoursquareSearchTool(api_key="FSQ...")
    result = await tool.execute(
        query="best sushi restaurants",
        latitude=35.6762,
        longitude=139.6503,
        radius=2000,
        limit=5,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from voyageai.config import settings
from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Foursquare Places API v3 endpoint (new domain as of 2025)
FSQ_BASE_URL = "https://api.foursquare.com/v3/places/search"

# Category mapping: user-friendly names → Foursquare category IDs
# Full list: https://docs.foursquare.com/data-products/docs/categories
_CATEGORY_IDS = {
    "restaurant": "13065",
    "hotel": "19014",
    "museum": "10027",
    "park": "16032",
    "shopping": "17000",
    "nightlife": "10032",
    "cafe": "13032",
    "bar": "13003",
    "attraction": "16000",
    "landmark": "16026",
    "beach": "16003",
    "airport": "19040",
    "train_station": "19047",
}


class FoursquareSearchTool(BaseTool):
    """
    Search for places using the Foursquare Places API.

    Uses Foursquare's database of 100M+ global POIs to find restaurants,
    attractions, hotels, and other places with ratings and details.

    Input:
        query (str): Search query (e.g., "pizza", "museums", "hotel")
        latitude (float): Center latitude (optional if near provided)
        longitude (float): Center longitude (optional if near provided)
        near (str): Location name as alternative to lat/lon (e.g., "Tokyo, Japan")
        radius (int): Search radius in meters (default 2000, max 50000)
        categories (str): Comma-separated category names to filter
        limit (int): Max results (default 5, max 10)

    Output:
        List of places with name, address, rating, category, coordinates.
    """

    name = "search_places_foursquare"
    description = (
        "Search for places (restaurants, hotels, attractions, museums, cafes, bars, "
        "shops, landmarks) using Foursquare's 100M+ POI database. "
        "Returns names, addresses, ratings, and categories. "
        "Requires either latitude/longitude or a location name (near)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g., 'best pizza', 'art museum', 'budget hotel')",
            },
            "latitude": {
                "type": "number",
                "description": "Center latitude",
            },
            "longitude": {
                "type": "number",
                "description": "Center longitude",
            },
            "near": {
                "type": "string",
                "description": "Location name (e.g., 'Tokyo, Japan'). Alternative to lat/lon.",
            },
            "radius": {
                "type": "integer",
                "description": "Search radius in meters (default 2000, max 50000)",
                "default": 2000,
            },
            "categories": {
                "type": "string",
                "description": (
                    "Comma-separated categories to filter: restaurant, hotel, museum, "
                    "park, shopping, nightlife, cafe, bar, attraction, landmark, beach"
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 5, max 10)",
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, api_key: str = "", timeout: float = 10.0):
        self.api_key = api_key or settings.foursquare_api_key
        self.timeout = timeout

    async def execute(
        self,
        query: str,
        latitude: float | None = None,
        longitude: float | None = None,
        near: str | None = None,
        radius: int = 2000,
        categories: str = "",
        limit: int = 5,
    ) -> ToolResult:
        """Search Foursquare for places matching the query."""
        start_time = time.time()
        input_args = {
            "query": query,
            "latitude": latitude,
            "longitude": longitude,
            "near": near,
            "radius": radius,
            "categories": categories,
            "limit": limit,
        }

        if not self.api_key:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Foursquare API key not configured (set FOURSQUARE_API_KEY)",
                latency_ms=0,
            )

        radius = min(radius, 50000)
        limit = min(limit, 10)

        # Build request params
        # Request website, url, and rating fields for source link enrichment + quality filtering
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": "name,location,geocodes,categories,distance,fsq_id,website,link,rating",
        }

        if latitude is not None and longitude is not None:
            params["ll"] = f"{latitude},{longitude}"
            params["radius"] = radius
        elif near:
            params["near"] = near
        else:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Either latitude/longitude or 'near' location name is required",
                latency_ms=0,
            )

        # Resolve category IDs
        if categories:
            cat_ids = []
            for cat in categories.split(","):
                cat = cat.strip().lower()
                if cat in _CATEGORY_IDS:
                    cat_ids.append(_CATEGORY_IDS[cat])
            if cat_ids:
                params["categories"] = ",".join(cat_ids)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    FSQ_BASE_URL,
                    params=params,
                    headers={
                        "Authorization": self.api_key,
                        "Accept": "application/json",
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

            # Parse results, filtering by minimum rating if configured
            min_rating = settings.foursquare_min_rating
            places = []
            filtered_count = 0
            for result in data.get("results", []):
                rating = result.get("rating")
                # Foursquare uses 0-10 scale
                if min_rating > 0 and rating is not None and rating < min_rating:
                    filtered_count += 1
                    continue

                location = result.get("location", {})
                geocodes = result.get("geocodes", {}).get("main", {})
                categories_list = result.get("categories", [])

                fsq_id = result.get("fsq_id", "")
                website = result.get("website", "")
                fsq_link = result.get("link", "")

                places.append({
                    "name": result.get("name", ""),
                    "address": location.get("formatted_address", ""),
                    "latitude": geocodes.get("latitude"),
                    "longitude": geocodes.get("longitude"),
                    "category": categories_list[0].get("name", "") if categories_list else "",
                    "distance_m": result.get("distance"),
                    "rating": rating,
                    "fsq_id": fsq_id,
                    "website": website,
                    "foursquare_url": fsq_link or (f"https://foursquare.com/v/{fsq_id}" if fsq_id else ""),
                })

            places.sort(key=lambda x: (x.get("rating") or 0), reverse=True)
            places = places[:limit]

            output = {
                "query": query,
                "places": places,
                "count": len(places),
                "source": "foursquare",
            }

            logger.info(
                "Foursquare search '%s': %d results",
                query, len(places),
            )

            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"Foursquare API error: {e.response.status_code}"
            if e.response.status_code == 401:
                error_msg = "Foursquare API key is invalid or expired"
            elif e.response.status_code == 429:
                error_msg = "Foursquare API rate limit exceeded"
            logger.error("Foursquare search failed: %s", error_msg)
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=error_msg,
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except httpx.TimeoutException:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Foursquare search timed out",
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("Foursquare search failed: %s", e)
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Foursquare search error: {e}",
                latency_ms=int((time.time() - start_time) * 1000),
            )
