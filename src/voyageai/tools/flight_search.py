"""
Amadeus Flight Search Tool.

Searches for flight offers using the Amadeus Self-Service API.
Uses OAuth2 client_credentials flow for authentication.

API: Amadeus Flight Offers Search v2
Auth: OAuth2 client_credentials (API Key + Secret → Bearer token)
Free Tier: Test environment with free monthly quota, no credit card
Rate Limit: 10 TPS (test), 40 TPS (production)

Example:
    tool = FlightSearchTool()
    result = await tool.execute(
        origin="SEA",
        destination="NRT",
        departure_date="2026-05-01",
        adults=2,
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

# Amadeus test environment endpoints
AMADEUS_AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_FLIGHT_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"


class FlightSearchTool(BaseTool):
    """
    Search for flight offers using the Amadeus API.

    Finds available flights between airports with pricing information.
    Uses the Amadeus test environment (free, limited data).

    Input:
        origin (str): Departure IATA airport code (e.g., "SEA", "JFK", "LAX")
        destination (str): Arrival IATA airport code (e.g., "NRT", "CDG", "LHR")
        departure_date (str): Departure date in YYYY-MM-DD format
        return_date (str): Optional return date for round-trip
        adults (int): Number of adult passengers (default 1)
        max_results (int): Max flight offers to return (default 5)
        travel_class (str): Cabin class: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
        nonstop (bool): Only show non-stop flights

    Output:
        List of flight offers with price, airline, duration, stops.
    """

    name = "search_flights"
    description = (
        "Search for flight offers between airports. Provide IATA airport codes "
        "(e.g., SEA for Seattle, NRT for Tokyo Narita, JFK for New York). "
        "Returns available flights with prices, airlines, duration, and stops. "
        "Can search one-way or round-trip."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "Departure IATA airport code (e.g., 'SEA', 'JFK', 'LAX')",
            },
            "destination": {
                "type": "string",
                "description": "Arrival IATA airport code (e.g., 'NRT', 'CDG', 'LHR')",
            },
            "departure_date": {
                "type": "string",
                "description": "Departure date (YYYY-MM-DD)",
            },
            "return_date": {
                "type": "string",
                "description": "Return date for round-trip (YYYY-MM-DD). Omit for one-way.",
            },
            "adults": {
                "type": "integer",
                "description": "Number of adult passengers (default 1)",
                "default": 1,
            },
            "max_results": {
                "type": "integer",
                "description": "Max flight offers to return (default 5, max 10)",
                "default": 5,
            },
            "travel_class": {
                "type": "string",
                "description": "Cabin class: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST",
            },
            "nonstop": {
                "type": "boolean",
                "description": "Only non-stop flights (default false)",
                "default": False,
            },
        },
        "required": ["origin", "destination", "departure_date"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        timeout: float = 15.0,
    ):
        self.api_key = api_key or settings.amadeus_api_key
        self.api_secret = api_secret or settings.amadeus_api_secret
        self.timeout = timeout
        self._access_token: str | None = None
        self._token_expires_at: float = 0

    async def _get_access_token(self, client: httpx.AsyncClient) -> str:
        """Get OAuth2 access token (cached until expiry)."""
        now = time.time()
        if self._access_token and now < self._token_expires_at - 60:
            return self._access_token

        resp = await client.post(
            AMADEUS_AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.api_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()

        self._access_token = data["access_token"]
        self._token_expires_at = now + data.get("expires_in", 1799)
        logger.info("Amadeus OAuth token obtained (expires in %ds)", data.get("expires_in", 0))
        return self._access_token

    async def execute(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str = "",
        adults: int = 1,
        max_results: int = 5,
        travel_class: str = "",
        nonstop: bool = False,
    ) -> ToolResult:
        """Search for flight offers."""
        start_time = time.time()
        input_args = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "adults": adults,
            "max_results": max_results,
            "travel_class": travel_class,
            "nonstop": nonstop,
        }

        if not self.api_key or not self.api_secret:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Amadeus API credentials not configured (set AMADEUS_API_KEY and AMADEUS_API_SECRET)",
                latency_ms=0,
            )

        max_results = min(max_results, 10)
        origin = origin.upper().strip()
        destination = destination.upper().strip()

        params: dict[str, Any] = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": adults,
            "max": max_results,
            "currencyCode": "USD",
        }

        if return_date:
            params["returnDate"] = return_date
        if travel_class:
            params["travelClass"] = travel_class.upper()
        if nonstop:
            params["nonStop"] = "true"

        try:
            async with httpx.AsyncClient() as client:
                token = await self._get_access_token(client)

                resp = await client.get(
                    AMADEUS_FLIGHT_URL,
                    params=params,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

            # Parse flight offers
            flights = []
            dictionaries = data.get("dictionaries", {})
            carriers = dictionaries.get("carriers", {})

            for offer in data.get("data", [])[:max_results]:
                price = offer.get("price", {})
                itineraries = offer.get("itineraries", [])

                # Parse outbound itinerary
                outbound = itineraries[0] if itineraries else {}
                segments = outbound.get("segments", [])

                airline_codes = list({seg.get("carrierCode", "") for seg in segments})
                airlines = [carriers.get(code, code) for code in airline_codes]

                flights.append({
                    "price": f"{price.get('total', '?')} {price.get('currency', 'USD')}",
                    "airlines": airlines,
                    "duration": outbound.get("duration", ""),
                    "stops": max(0, len(segments) - 1),
                    "departure": segments[0].get("departure", {}) if segments else {},
                    "arrival": segments[-1].get("arrival", {}) if segments else {},
                    "segments_count": len(segments),
                    "is_round_trip": len(itineraries) > 1,
                })

            output = {
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "return_date": return_date or None,
                "flights": flights,
                "count": len(flights),
                "source": "amadeus_test",
                "note": "Prices from Amadeus test environment (may differ from production)",
            }

            logger.info(
                "Flight search %s→%s on %s: %d offers",
                origin, destination, departure_date, len(flights),
            )

            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"Amadeus API error: {e.response.status_code}"
            try:
                err_data = e.response.json()
                errors = err_data.get("errors", [])
                if errors:
                    error_msg = f"Amadeus: {errors[0].get('detail', error_msg)}"
            except Exception:
                pass
            logger.error("Flight search failed: %s", error_msg)
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
                error="Flight search timed out",
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("Flight search failed: %s", e)
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Flight search error: {e}",
                latency_ms=int((time.time() - start_time) * 1000),
            )
