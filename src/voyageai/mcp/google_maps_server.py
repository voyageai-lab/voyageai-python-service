"""
Google Maps MCP Server.

A Model Context Protocol (MCP) server that exposes Google Maps APIs as tools.
This server runs as a standalone service and communicates with the VoyageAI agent
via the MCP Streamable HTTP transport.

Tools provided:
  - search_places: Text Search via Google Maps Places API (New)
  - get_place_details: Place Details via Google Maps Places API (New)
  - get_directions: Driving/walking/transit directions via Directions API

Usage:
    # Run directly (for development):
    python -m voyageai.mcp.google_maps_server

    # Or via mcp CLI:
    mcp run voyageai.mcp.google_maps_server:mcp

Environment Variables:
    GOOGLE_MAPS_API_KEY: Required. Your Google Maps Platform API key.
    MCP_PORT: Optional. Port to listen on (default 8080).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Create the MCP server
# ---------------------------------------------------------------------------

_port = int(os.environ.get("MCP_PORT", "8080"))

mcp = FastMCP(
    "VoyageAI Google Maps",
    instructions=(
        "Provides Google Maps tools for travel planning: "
        "place search with ratings/prices, and directions with real-time transit."
    ),
    host="0.0.0.0",
    port=_port,
)

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")


# ---------------------------------------------------------------------------
# Tool: search_places
# ---------------------------------------------------------------------------

MIN_RATING = float(os.environ.get("GOOGLE_MAPS_MIN_RATING", "3.5"))


@mcp.tool()
async def search_places(
    query: str,
    latitude: float | None = None,
    longitude: float | None = None,
    radius: int = 5000,
    place_type: str | None = None,
    limit: int = 5,
    min_rating: float | None = None,
) -> str:
    """Search for places (restaurants, hotels, attractions, museums, cafes,
    bars, parks, landmarks) using Google Maps.

    Returns names, addresses, ratings, price levels, and Google Maps URLs.
    Very accurate global data with 200M+ places.
    Low-rated places (below min_rating) are automatically filtered out.

    Args:
        query: Natural language search (e.g., 'ramen near Shinjuku station')
        latitude: Bias center latitude (optional)
        longitude: Bias center longitude (optional)
        radius: Bias radius in meters (default 5000, max 50000)
        place_type: Filter by type: restaurant, cafe, bar, hotel, lodging,
                    museum, park, tourist_attraction, shopping_mall, airport,
                    train_station, church, spa, night_club
        limit: Maximum results (default 5, max 10)
        min_rating: Minimum rating threshold (0-5). Defaults to 3.5.
    """
    if not GOOGLE_MAPS_API_KEY:
        return json.dumps({"error": "GOOGLE_MAPS_API_KEY not configured"})

    limit = min(limit, 10)
    radius = min(radius, 50000)

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

    if place_type:
        body["includedType"] = place_type

    field_mask = (
        "places.id,places.displayName,places.formattedAddress,places.location,"
        "places.rating,places.userRatingCount,places.priceLevel,"
        "places.types,places.businessStatus,places.googleMapsUri,"
        "places.primaryType,places.websiteUri"
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://places.googleapis.com/v1/places:searchText",
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
                    "X-Goog-FieldMask": field_mask,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

        rating_threshold = min_rating if min_rating is not None else MIN_RATING

        places = []
        filtered_count = 0
        for p in data.get("places", []):
            rating = p.get("rating")
            if rating_threshold > 0 and rating is not None and rating < rating_threshold:
                filtered_count += 1
                continue

            loc = p.get("location", {})
            display = p.get("displayName", {})
            places.append({
                "name": display.get("text", ""),
                "place_id": p.get("id", ""),
                "address": p.get("formattedAddress", ""),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
                "rating": rating,
                "user_ratings_total": p.get("userRatingCount"),
                "price_level": p.get("priceLevel"),
                "type": p.get("primaryType", ""),
                "business_status": p.get("businessStatus", ""),
                "google_maps_url": p.get("googleMapsUri", ""),
                "website": p.get("websiteUri", ""),
            })

        places.sort(key=lambda x: (x.get("rating") or 0), reverse=True)
        places = places[:limit]

        return json.dumps({
            "query": query,
            "places": places,
            "count": len(places),
            "filtered_below_rating": filtered_count,
            "min_rating_applied": rating_threshold,
            "source": "google_maps_mcp",
        }, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"Google Maps API error: {e.response.status_code}"})
    except Exception as e:
        return json.dumps({"error": f"Search failed: {e}"})


# ---------------------------------------------------------------------------
# Tool: get_place_details
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_place_details(
    place_id: str,
) -> str:
    """Get detailed information about a specific place from Google Maps.

    Returns website URL, phone number, opening hours, reviews,
    editorial summary, and Google Maps link. Use the place_id
    returned by search_places.

    Args:
        place_id: Google Maps place ID (from search_places results)
    """
    if not GOOGLE_MAPS_API_KEY:
        return json.dumps({"error": "GOOGLE_MAPS_API_KEY not configured"})

    field_mask = (
        "id,displayName,formattedAddress,location,"
        "rating,userRatingCount,priceLevel,primaryType,"
        "websiteUri,nationalPhoneNumber,internationalPhoneNumber,"
        "regularOpeningHours,googleMapsUri,"
        "editorialSummary,reviews"
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://places.googleapis.com/v1/places/{place_id}",
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
                    "X-Goog-FieldMask": field_mask,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

        loc = data.get("location", {})
        display = data.get("displayName", {})

        hours = data.get("regularOpeningHours", {})
        weekday_text = hours.get("weekdayDescriptions", [])

        reviews_raw = data.get("reviews", [])[:5]
        reviews = []
        for r in reviews_raw:
            reviews.append({
                "author": r.get("authorAttribution", {}).get(
                    "displayName", ""
                ),
                "rating": r.get("rating"),
                "text": (r.get("text", {}).get("text", ""))[:200],
                "time": r.get("relativePublishTimeDescription", ""),
            })

        editorial = data.get("editorialSummary", {})

        result = {
            "name": display.get("text", ""),
            "place_id": data.get("id", place_id),
            "address": data.get("formattedAddress", ""),
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "rating": data.get("rating"),
            "user_ratings_total": data.get("userRatingCount"),
            "price_level": data.get("priceLevel"),
            "type": data.get("primaryType", ""),
            "website": data.get("websiteUri", ""),
            "phone": data.get("internationalPhoneNumber", "")
            or data.get("nationalPhoneNumber", ""),
            "google_maps_url": data.get("googleMapsUri", ""),
            "opening_hours": weekday_text,
            "editorial_summary": editorial.get("text", ""),
            "reviews": reviews,
            "source": "google_maps_mcp",
        }

        return json.dumps(result, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps(
            {"error": f"Place Details API error: {e.response.status_code}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Place details failed: {e}"})


# ---------------------------------------------------------------------------
# Tool: get_directions
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_directions(
    origin: str,
    destination: str,
    mode: str = "driving",
    departure_time: str | None = None,
) -> str:
    """Get directions between two locations via Google Maps.

    Returns distance, duration, and step-by-step route.
    Supports driving, walking, bicycling, and public transit with live schedules.

    Args:
        origin: Starting location — address or 'lat,lng'
                (e.g., 'Shinjuku Station, Tokyo' or '35.69,139.70')
        destination: Ending location — address or 'lat,lng'
                     (e.g., 'Tokyo Tower' or '35.6586,139.7454')
        mode: Travel mode: driving, walking, bicycling, transit (default: driving)
        departure_time: For transit: 'now' or ISO datetime string
    """
    if not GOOGLE_MAPS_API_KEY:
        return json.dumps({"error": "GOOGLE_MAPS_API_KEY not configured"})

    mode = mode if mode in ("driving", "walking", "bicycling", "transit") else "driving"

    params: dict[str, str] = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "key": GOOGLE_MAPS_API_KEY,
        "language": "en",
    }

    if mode == "transit" and departure_time:
        if departure_time == "now":
            params["departure_time"] = "now"
        else:
            import datetime as _dt
            try:
                dt = _dt.datetime.fromisoformat(departure_time)
                params["departure_time"] = str(int(dt.timestamp()))
            except ValueError:
                params["departure_time"] = "now"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/directions/json",
                params=params,
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

        status = data.get("status", "UNKNOWN")
        if status != "OK":
            return json.dumps({"error": f"Directions API status: {status}"})

        route = data["routes"][0]
        leg = route["legs"][0]

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

        return json.dumps({
            "origin": leg.get("start_address", origin),
            "destination": leg.get("end_address", destination),
            "distance": leg["distance"]["text"],
            "duration": leg["duration"]["text"],
            "mode": mode,
            "steps": steps,
            "summary": route.get("summary", ""),
            "source": "google_maps_mcp",
        }, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"Directions error: {e.response.status_code}"})
    except Exception as e:
        return json.dumps({"error": f"Directions failed: {e}"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Google Maps MCP Server on port %d", _port)
    mcp.run(transport="streamable-http")
