"""
VoyageAI Tools Package

This package contains real tool implementations for the AI agent:
- WeatherTool: Weather forecast using Open-Meteo API
- CurrencyTool: Currency conversion using exchangerate.host
- TimeZoneTool: Timezone conversion using Python zoneinfo
- DistanceTool: Distance calculation using Haversine formula
- GeocodeTool: Geocoding using OpenStreetMap Nominatim
- HolidayTool: Public holidays using Nager.Date API
"""

from voyageai.tools.base import BaseTool, ToolResult

__all__ = ["BaseTool", "ToolResult"]

