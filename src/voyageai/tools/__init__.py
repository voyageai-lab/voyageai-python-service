"""
VoyageAI Tools Package

This package contains real tool implementations for the AI agent:
- WeatherTool: Weather forecast using Open-Meteo API
- CurrencyTool: Currency conversion using Frankfurter API
- TimeZoneTool: Timezone conversion using Python zoneinfo
- DistanceTool: Distance calculation using Haversine formula
- GeocodeTool: Geocoding using OpenStreetMap Nominatim
- HolidayTool: Public holidays using Nager.Date API

Module 10 additions:
- RateLimiter: Token bucket rate limiting for tool execution
- RateLimitExceeded: Exception for rate limit violations

All tools use free APIs (no API keys required) or local calculations.
"""

from voyageai.tools.base import BaseTool, ToolResult
from voyageai.tools.currency import CurrencyTool
from voyageai.tools.distance import DistanceTool
from voyageai.tools.geocode import GeocodeTool
from voyageai.tools.holiday import HolidayTool
from voyageai.tools.rate_limiter import RateLimitExceeded, RateLimiter, rate_limiter
from voyageai.tools.registry import ToolRegistry, tool_registry
from voyageai.tools.timezone import TimeZoneTool
from voyageai.tools.weather import WeatherTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
    "GeocodeTool",
    "WeatherTool",
    "CurrencyTool",
    "TimeZoneTool",
    "DistanceTool",
    "HolidayTool",
    # Rate limiting (Module 10)
    "RateLimiter",
    "RateLimitExceeded",
    "rate_limiter",
]

