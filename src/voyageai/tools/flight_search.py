"""
Amadeus Flight Search Tool with city-name resolution and web-search fallback.

Primary: Amadeus Self-Service API (Flight Offers Search v2)
Fallback: web_search for flight price estimates when Amadeus fails.

Auth: OAuth2 client_credentials (API Key + Secret → Bearer token)
Free Tier: Test environment with free monthly quota, no credit card
Rate Limit: 10 TPS (test), 40 TPS (production)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from voyageai.config import settings
from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

AMADEUS_AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_FLIGHT_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"

_IATA_RE = re.compile(r"^[A-Z]{3}$")

# Common city/country names → primary IATA airport code.
# Covers cases where the LLM passes a city name instead of an airport code.
_CITY_TO_IATA: dict[str, str] = {
    "TOKYO": "NRT", "東京": "NRT", "東京都": "NRT",
    "OSAKA": "KIX", "大阪": "KIX",
    "KYOTO": "KIX", "京都": "KIX",
    "SEOUL": "ICN", "서울": "ICN",
    "BEIJING": "PEK", "北京": "PEK",
    "SHANGHAI": "PVG", "上海": "PVG",
    "GUANGZHOU": "CAN", "广州": "CAN",
    "SHENZHEN": "SZX", "深圳": "SZX",
    "HONG KONG": "HKG", "香港": "HKG",
    "TAIPEI": "TPE", "台北": "TPE",
    "BANGKOK": "BKK", "กรุงเทพ": "BKK",
    "SINGAPORE": "SIN", "新加坡": "SIN",
    "KUALA LUMPUR": "KUL",
    "HANOI": "HAN", "HO CHI MINH": "SGN", "SAIGON": "SGN",
    "MANILA": "MNL", "JAKARTA": "CGK",
    "MUMBAI": "BOM", "DELHI": "DEL", "NEW DELHI": "DEL",
    "LONDON": "LHR", "PARIS": "CDG",
    "ROME": "FCO", "MILAN": "MXP",
    "MADRID": "MAD", "BARCELONA": "BCN",
    "BERLIN": "BER", "FRANKFURT": "FRA", "MUNICH": "MUC",
    "AMSTERDAM": "AMS", "ZURICH": "ZRH", "VIENNA": "VIE",
    "ISTANBUL": "IST", "ATHENS": "ATH", "LISBON": "LIS",
    "DUBLIN": "DUB", "STOCKHOLM": "ARN", "HELSINKI": "HEL",
    "MOSCOW": "SVO", "ST PETERSBURG": "LED",
    "NEW YORK": "JFK", "LOS ANGELES": "LAX", "SAN FRANCISCO": "SFO",
    "CHICAGO": "ORD", "SEATTLE": "SEA", "MIAMI": "MIA",
    "BOSTON": "BOS", "HOUSTON": "IAH", "DALLAS": "DFW",
    "WASHINGTON": "IAD", "ATLANTA": "ATL", "DENVER": "DEN",
    "LAS VEGAS": "LAS", "ORLANDO": "MCO", "PORTLAND": "PDX",
    "TORONTO": "YYZ", "VANCOUVER": "YVR", "MONTREAL": "YUL",
    "MEXICO CITY": "MEX", "CANCUN": "CUN",
    "SAO PAULO": "GRU", "RIO DE JANEIRO": "GIG",
    "BUENOS AIRES": "EZE", "LIMA": "LIM", "BOGOTA": "BOG",
    "SYDNEY": "SYD", "MELBOURNE": "MEL", "AUCKLAND": "AKL",
    "CAIRO": "CAI", "DUBAI": "DXB", "ABU DHABI": "AUH",
    "DOHA": "DOH", "RIYADH": "RUH",
    "NAIROBI": "NBO", "CAPE TOWN": "CPT", "JOHANNESBURG": "JNB",
}


def _resolve_iata(raw: str) -> str | None:
    """Try to resolve *raw* into a valid 3-letter IATA code.

    Returns the code (uppercase) on success, or ``None`` if unresolvable.
    """
    cleaned = raw.strip().upper()
    if _IATA_RE.match(cleaned):
        return cleaned
    # Try the city lookup (also try the original casing for CJK)
    return _CITY_TO_IATA.get(cleaned) or _CITY_TO_IATA.get(raw.strip())


class FlightSearchTool(BaseTool):
    """Search for flight offers using the Amadeus API with smart fallbacks."""

    name = "search_flights"
    description = (
        "Search for flight offers between airports. You can provide IATA airport codes "
        "(e.g., SEA, NRT, JFK) or city names (e.g., Tokyo, Paris, New York). "
        "Returns available flights with prices, airlines, duration, and stops. "
        "Can search one-way or round-trip."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "Departure IATA airport code or city name (e.g., 'SEA', 'Seattle', 'JFK')",
            },
            "destination": {
                "type": "string",
                "description": "Arrival IATA airport code or city name (e.g., 'NRT', 'Tokyo', 'Paris')",
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
        timeout: float = 10.0,
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

    # ------------------------------------------------------------------
    # Web-search fallback
    # ------------------------------------------------------------------

    async def _web_search_fallback(
        self,
        origin_raw: str,
        destination_raw: str,
        departure_date: str,
        return_date: str,
        adults: int,
        travel_class: str,
        input_args: dict[str, Any],
        start_time: float,
    ) -> ToolResult:
        """Use web_search to find approximate flight info when Amadeus fails."""
        from voyageai.tools.websearch import WebSearchTool

        trip_type = "round-trip" if return_date else "one-way"
        class_str = f" {travel_class}" if travel_class else ""
        query = (
            f"flights from {origin_raw} to {destination_raw} "
            f"{departure_date}{' to ' + return_date if return_date else ''} "
            f"{trip_type}{class_str} price {adults} adult"
        )

        try:
            ws = WebSearchTool()
            ws_result = await ws.execute(
                query=query,
                intent="flight_search",
                max_results=5,
            )

            if not ws_result.success or not ws_result.output:
                return ToolResult(
                    tool_name=self.name,
                    input_args=input_args,
                    output=None,
                    success=False,
                    error="Amadeus unavailable and web search fallback returned no results",
                    latency_ms=int((time.time() - start_time) * 1000),
                )

            results = ws_result.output.get("results", [])
            output = {
                "origin": origin_raw,
                "destination": destination_raw,
                "departure_date": departure_date,
                "return_date": return_date or None,
                "flights": [],
                "count": 0,
                "source": "web_search",
                "web_results": [
                    {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("url", "")}
                    for r in results[:5]
                ],
                "note": (
                    "Flight data from web search (Amadeus API unavailable). "
                    "Prices are approximate — check airline websites for booking."
                ),
            }

            logger.info("Flight web-search fallback for %s→%s: %d results", origin_raw, destination_raw, len(results))

            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("Flight web-search fallback failed: %s", e)
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Amadeus unavailable and web search fallback also failed: {e}",
                latency_ms=int((time.time() - start_time) * 1000),
            )

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------

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
        origin_raw, destination_raw = origin, destination
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

        # --- Resolve IATA codes ---
        resolved_origin = _resolve_iata(origin)
        resolved_dest = _resolve_iata(destination)

        if not resolved_origin or not resolved_dest:
            bad = []
            if not resolved_origin:
                bad.append(f"origin '{origin}'")
            if not resolved_dest:
                bad.append(f"destination '{destination}'")
            logger.warning("Could not resolve IATA code(s): %s — falling back to web search", ", ".join(bad))
            return await self._web_search_fallback(
                origin_raw, destination_raw, departure_date, return_date,
                adults, travel_class, input_args, start_time,
            )

        origin_iata = resolved_origin
        dest_iata = resolved_dest

        if origin_iata != origin.upper().strip() or dest_iata != destination.upper().strip():
            logger.info(
                "Resolved flight codes: %s→%s (from %s→%s)",
                origin_iata, dest_iata, origin, destination,
            )

        if not self.api_key or not self.api_secret:
            return await self._web_search_fallback(
                origin_raw, destination_raw, departure_date, return_date,
                adults, travel_class, input_args, start_time,
            )

        max_results = min(max_results, 10)

        params: dict[str, Any] = {
            "originLocationCode": origin_iata,
            "destinationLocationCode": dest_iata,
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

        # --- Amadeus call with one retry ---
        last_error = ""
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
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

                flights = []
                dictionaries = data.get("dictionaries", {})
                carriers = dictionaries.get("carriers", {})

                for offer in data.get("data", [])[:max_results]:
                    price = offer.get("price", {})
                    itineraries = offer.get("itineraries", [])
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
                    "origin": origin_iata,
                    "destination": dest_iata,
                    "departure_date": departure_date,
                    "return_date": return_date or None,
                    "flights": flights,
                    "count": len(flights),
                    "source": "amadeus_test",
                    "note": "Prices from Amadeus test environment (may differ from production)",
                }

                if origin_iata != origin.strip().upper():
                    output["origin_resolved_from"] = origin
                if dest_iata != destination.strip().upper():
                    output["destination_resolved_from"] = destination

                logger.info(
                    "Flight search %s→%s on %s: %d offers",
                    origin_iata, dest_iata, departure_date, len(flights),
                )

                return ToolResult(
                    tool_name=self.name,
                    input_args=input_args,
                    output=output,
                    success=True,
                    latency_ms=int((time.time() - start_time) * 1000),
                )

            except httpx.HTTPStatusError as e:
                last_error = f"Amadeus API error: {e.response.status_code}"
                try:
                    err_data = e.response.json()
                    errors = err_data.get("errors", [])
                    if errors:
                        last_error = f"Amadeus: {errors[0].get('detail', last_error)}"
                except Exception:
                    pass
                status = e.response.status_code
                if status >= 500 and attempt == 0:
                    logger.warning("Amadeus 5xx (%s), retrying once...", status)
                    continue
                break
            except httpx.TimeoutException:
                last_error = "Flight search timed out"
                if attempt == 0:
                    logger.warning("Amadeus timeout, retrying once...")
                    continue
                break
            except Exception as e:
                last_error = f"Flight search error: {e}"
                break

        # --- Amadeus failed → web-search fallback ---
        logger.warning("Amadeus failed (%s), falling back to web search", last_error)
        return await self._web_search_fallback(
            origin_raw, destination_raw, departure_date, return_date,
            adults, travel_class, input_args, start_time,
        )
