"""
Google Maps Tools — Places Search, Place Details, and Directions.

Provides three tools powered by the Google Maps Platform APIs:
1. google_maps_search_places: Text Search (Places API New) for finding places
2. google_maps_place_details: Get rich details (reviews, hours, photos) for a place
3. google_maps_directions: Get driving/walking/transit directions between two points

API: Google Maps Platform — Places API (New), Directions API
Auth: API key (query parameter)
Free Tier: $200/month free credit (covers ~11k text searches or ~40k directions)
Rate Limit: Varies by endpoint

Example:
    tool = GoogleMapsSearchPlacesTool(api_key="AIza...")
    result = await tool.execute(
        query="best ramen in Shinjuku",
        latitude=35.6938,
        longitude=139.7034,
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

# ---------------------------------------------------------------------------
# Google Maps Search Places Tool
# ---------------------------------------------------------------------------

class GoogleMapsSearchPlacesTool(BaseTool):
    """
    Search for places using Google Maps Text Search (New).

    Uses the Places API (New) textSearch endpoint for rich place results
    including ratings, price levels, and business status.

    Input:
        query (str): Natural-language search query
        latitude (float): Bias center latitude (optional)
        longitude (float): Bias center longitude (optional)
        radius (int): Bias radius in meters (default 5000)
        type (str): Place type filter (restaurant, hotel, museum, etc.)
        limit (int): Max results (default 5, max 10)

    Output:
        List of places with name, address, rating, type, coordinates, price level.
    """

    name = "google_maps_search_places"
    description = (
        "Search for places (restaurants, hotels, attractions, museums, cafes, "
        "bars, parks, landmarks) using Google Maps. Returns names, addresses, "
        "ratings, price levels, and opening hours. Very accurate global data."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language search query "
                    "(e.g., 'ramen near Shinjuku station', 'museums in Paris')"
                ),
            },
            "latitude": {
                "type": "number",
                "description": "Bias center latitude (optional)",
            },
            "longitude": {
                "type": "number",
                "description": "Bias center longitude (optional)",
            },
            "radius": {
                "type": "integer",
                "description": "Bias radius in meters (default 5000, max 50000)",
                "default": 5000,
            },
            "type": {
                "type": "string",
                "description": (
                    "Place type filter: restaurant, cafe, bar, hotel, lodging, "
                    "museum, park, tourist_attraction, shopping_mall, airport, "
                    "train_station, church, spa, night_club"
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 5, max 10)",
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    # New Places API text-search endpoint
    _URL = "https://places.googleapis.com/v1/places:searchText"

    def __init__(self, api_key: str = "", timeout: float = 10.0):
        self.api_key = api_key or settings.google_maps_api_key
        self.timeout = timeout

    async def execute(
        self,
        query: str,
        latitude: float | None = None,
        longitude: float | None = None,
        radius: int = 5000,
        type: str | None = None,
        limit: int = 5,
    ) -> ToolResult:
        start_time = time.time()
        input_args = {
            "query": query,
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius,
            "type": type,
            "limit": limit,
        }

        if not self.api_key:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Google Maps API key not configured (set GOOGLE_MAPS_API_KEY)",
                latency_ms=0,
            )

        limit = min(limit, 10)
        radius = min(radius, 50000)

        # Build request body (Places API New uses POST + JSON)
        body: dict[str, Any] = {
            "textQuery": query,
            "maxResultCount": limit,
            "languageCode": "en",
        }

        if latitude is not None and longitude is not None:
            body["locationBias"] = {
                "circle": {
                    "center": {"latitude": latitude, "longitude": longitude},
                    "radius": float(radius),
                }
            }

        if type:
            body["includedType"] = type

        # Field mask — controls which fields are returned (and billed)
        field_mask = (
            "places.displayName,places.formattedAddress,places.location,"
            "places.rating,places.userRatingCount,places.priceLevel,"
            "places.types,places.businessStatus,places.googleMapsUri,"
            "places.primaryType"
        )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self._URL,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": self.api_key,
                        "X-Goog-FieldMask": field_mask,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

            places = []
            for p in data.get("places", [])[:limit]:
                loc = p.get("location", {})
                display = p.get("displayName", {})
                places.append({
                    "name": display.get("text", ""),
                    "address": p.get("formattedAddress", ""),
                    "latitude": loc.get("latitude"),
                    "longitude": loc.get("longitude"),
                    "rating": p.get("rating"),
                    "user_ratings_total": p.get("userRatingCount"),
                    "price_level": p.get("priceLevel"),
                    "type": p.get("primaryType", ""),
                    "business_status": p.get("businessStatus", ""),
                    "google_maps_url": p.get("googleMapsUri", ""),
                })

            output = {
                "query": query,
                "places": places,
                "count": len(places),
                "source": "google_maps",
            }

            logger.info("Google Maps search '%s': %d results", query, len(places))

            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        except httpx.HTTPStatusError as e:
            error_body = e.response.text[:200]
            error_msg = f"Google Maps API error: {e.response.status_code} — {error_body}"
            if e.response.status_code == 403:
                error_msg = "Google Maps API key invalid or API not enabled"
            elif e.response.status_code == 429:
                error_msg = "Google Maps API rate limit / quota exceeded"
            logger.error("Google Maps search failed: %s", error_msg)
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
                error="Google Maps search timed out",
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("Google Maps search error: %s", e)
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Google Maps search error: {e}",
                latency_ms=int((time.time() - start_time) * 1000),
            )


# ---------------------------------------------------------------------------
# Google Maps Directions Tool
# ---------------------------------------------------------------------------

class GoogleMapsDirectionsTool(BaseTool):
    """
    Get directions between two locations using Google Maps Directions API.

    Returns route distance, duration, steps, and transit details.

    Input:
        origin (str): Starting point (address or "lat,lng")
        destination (str): Ending point (address or "lat,lng")
        mode (str): Travel mode — driving, walking, bicycling, transit
        departure_time (str): For transit — ISO datetime or "now"

    Output:
        Route with distance, duration, steps summary, and transit info.
    """

    name = "google_maps_directions"
    description = (
        "Get directions between two locations via Google Maps. "
        "Returns distance, duration, and route steps. "
        "Supports driving, walking, bicycling, and public transit with live schedules. "
        "Use addresses or 'lat,lng' format for origin/destination."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": (
                    "Starting location — address or 'lat,lng' "
                    "(e.g., 'Shinjuku Station, Tokyo' or '35.69,139.70')"
                ),
            },
            "destination": {
                "type": "string",
                "description": (
                    "Ending location — address or 'lat,lng' "
                    "(e.g., 'Tokyo Tower' or '35.6586,139.7454')"
                ),
            },
            "mode": {
                "type": "string",
                "description": "Travel mode: driving, walking, bicycling, transit (default: driving)",
                "default": "driving",
            },
            "departure_time": {
                "type": "string",
                "description": (
                    "Departure time for transit — 'now' or ISO datetime. "
                    "Only used when mode=transit."
                ),
            },
        },
        "required": ["origin", "destination"],
        "additionalProperties": False,
    }

    _URL = "https://maps.googleapis.com/maps/api/directions/json"

    def __init__(self, api_key: str = "", timeout: float = 10.0):
        self.api_key = api_key or settings.google_maps_api_key
        self.timeout = timeout

    async def execute(
        self,
        origin: str,
        destination: str,
        mode: str = "driving",
        departure_time: str | None = None,
    ) -> ToolResult:
        start_time = time.time()
        input_args = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "departure_time": departure_time,
        }

        if not self.api_key:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Google Maps API key not configured (set GOOGLE_MAPS_API_KEY)",
                latency_ms=0,
            )

        mode = mode if mode in ("driving", "walking", "bicycling", "transit") else "driving"

        params: dict[str, str] = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "key": self.api_key,
            "language": "en",
        }

        if mode == "transit" and departure_time:
            if departure_time == "now":
                params["departure_time"] = "now"
            else:
                # Parse ISO to epoch
                import datetime as _dt
                try:
                    dt = _dt.datetime.fromisoformat(departure_time)
                    params["departure_time"] = str(int(dt.timestamp()))
                except ValueError:
                    params["departure_time"] = "now"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    self._URL,
                    params=params,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

            status = data.get("status", "UNKNOWN")
            if status != "OK":
                return ToolResult(
                    tool_name=self.name,
                    input_args=input_args,
                    output=None,
                    success=False,
                    error=f"Directions API status: {status}",
                    latency_ms=int((time.time() - start_time) * 1000),
                )

            route = data["routes"][0]
            leg = route["legs"][0]

            # Build step summaries (first 10 steps max)
            steps = []
            for step in leg.get("steps", [])[:10]:
                s: dict[str, Any] = {
                    "instruction": step.get("html_instructions", "")
                        .replace("<b>", "").replace("</b>", "")
                        .replace("<div>", " ").replace("</div>", ""),
                    "distance": step["distance"]["text"],
                    "duration": step["duration"]["text"],
                    "travel_mode": step.get("travel_mode", mode),
                }
                # Transit details
                transit = step.get("transit_details")
                if transit:
                    line = transit.get("line", {})
                    s["transit_line"] = line.get("short_name") or line.get("name", "")
                    s["transit_type"] = line.get("vehicle", {}).get("type", "")
                    dep = transit.get("departure_stop", {})
                    arr = transit.get("arrival_stop", {})
                    s["from_stop"] = dep.get("name", "")
                    s["to_stop"] = arr.get("name", "")
                steps.append(s)

            output = {
                "origin": leg.get("start_address", origin),
                "destination": leg.get("end_address", destination),
                "distance": leg["distance"]["text"],
                "duration": leg["duration"]["text"],
                "mode": mode,
                "steps": steps,
                "summary": route.get("summary", ""),
                "source": "google_maps_directions",
            }

            logger.info(
                "Google Maps directions %s → %s: %s, %s",
                origin, destination, output["distance"], output["duration"],
            )

            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"Google Maps Directions error: {e.response.status_code}"
            logger.error("Directions failed: %s", error_msg)
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
                error="Google Maps Directions timed out",
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("Google Maps Directions error: %s", e)
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Directions error: {e}",
                latency_ms=int((time.time() - start_time) * 1000),
            )
