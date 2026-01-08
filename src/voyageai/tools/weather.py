"""
Weather Forecast Tool - Get weather forecast for travel planning.

Uses Open-Meteo API (completely free, no API key required).
https://open-meteo.com/

Features:
- 16-day forecast available
- Hourly and daily data
- Multiple weather variables
- No rate limits for reasonable usage

Example:
    tool = WeatherTool()
    result = await tool.execute(
        latitude=35.68,
        longitude=139.76,
        start_date="2026-01-10",
        end_date="2026-01-12"
    )
"""

import logging
import time
from datetime import datetime

import httpx

from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Open-Meteo API endpoint (free, no key needed)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# WMO Weather interpretation codes
# https://open-meteo.com/en/docs#weathervariables
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherTool(BaseTool):
    """
    Get weather forecast for a location.
    
    This tool uses Open-Meteo's free weather API to provide:
    - Daily temperature (min/max)
    - Precipitation probability
    - Weather conditions
    - UV index
    
    API: https://open-meteo.com/
    Rate Limit: None (fair use)
    Cost: Free
    
    Input:
        latitude (float): Location latitude (-90 to 90)
        longitude (float): Location longitude (-180 to 180)
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Output:
        {
            "location": {"latitude": 35.68, "longitude": 139.76},
            "timezone": "Asia/Tokyo",
            "forecast": [
                {
                    "date": "2026-01-10",
                    "temp_max_c": 12.5,
                    "temp_min_c": 4.2,
                    "precipitation_chance": 20,
                    "condition": "Partly cloudy",
                    "uv_index": 3.5,
                    "sunrise": "06:50",
                    "sunset": "16:55"
                },
                ...
            ]
        }
    """
    
    name = "get_weather_forecast"
    description = (
        "Get weather forecast for a destination. Requires latitude and longitude coordinates. "
        "Returns daily temperature, precipitation chance, and conditions. "
        "Use geocode_location first to get coordinates if you only have a location name."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "Latitude coordinate (-90 to 90)"
            },
            "longitude": {
                "type": "number",
                "description": "Longitude coordinate (-180 to 180)"
            },
            "start_date": {
                "type": "string",
                "description": "Start date for forecast (YYYY-MM-DD format)"
            },
            "end_date": {
                "type": "string",
                "description": "End date for forecast (YYYY-MM-DD format)"
            }
        },
        "required": ["latitude", "longitude", "start_date", "end_date"],
        "additionalProperties": False
    }
    
    def __init__(self, timeout: float = 10.0):
        """
        Initialize the weather tool.
        
        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout
    
    def _weather_code_to_text(self, code: int) -> str:
        """Convert WMO weather code to human-readable text."""
        return WEATHER_CODES.get(code, f"Unknown ({code})")
    
    def _validate_dates(self, start_date: str, end_date: str) -> tuple[bool, str]:
        """
        Validate date format and range.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            if end < start:
                return False, "End date must be after start date"
            
            # Open-Meteo supports up to 16 days forecast
            today = datetime.now()
            max_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
            from datetime import timedelta
            max_date += timedelta(days=16)
            
            if end > max_date:
                return False, f"Forecast only available up to 16 days ahead ({max_date.strftime('%Y-%m-%d')})"
            
            return True, ""
            
        except ValueError:
            return False, "Invalid date format. Use YYYY-MM-DD."
    
    async def execute(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str
    ) -> ToolResult:
        """
        Get weather forecast for the specified location and date range.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            ToolResult with weather forecast data
        """
        start_time = time.time()
        input_args = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date
        }
        
        # Validate coordinates
        if not (-90 <= latitude <= 90):
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Latitude must be between -90 and 90",
                latency_ms=0
            )
        
        if not (-180 <= longitude <= 180):
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Longitude must be between -180 and 180",
                latency_ms=0
            )
        
        # Validate dates
        valid, error = self._validate_dates(start_date, end_date)
        if not valid:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=error,
                latency_ms=0
            )
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    OPEN_METEO_URL,
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "start_date": start_date,
                        "end_date": end_date,
                        "daily": ",".join([
                            "temperature_2m_max",
                            "temperature_2m_min",
                            "precipitation_probability_max",
                            "weathercode",
                            "uv_index_max",
                            "sunrise",
                            "sunset"
                        ]),
                        "timezone": "auto",
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
            
            # Parse the daily forecast
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            
            forecast = []
            for i, date in enumerate(dates):
                sunrise = daily.get("sunrise", [None])[i]
                sunset = daily.get("sunset", [None])[i]
                
                forecast.append({
                    "date": date,
                    "temp_max_c": daily.get("temperature_2m_max", [None])[i],
                    "temp_min_c": daily.get("temperature_2m_min", [None])[i],
                    "precipitation_chance": daily.get("precipitation_probability_max", [None])[i],
                    "condition": self._weather_code_to_text(
                        daily.get("weathercode", [0])[i] or 0
                    ),
                    "uv_index": daily.get("uv_index_max", [None])[i],
                    "sunrise": sunrise.split("T")[1][:5] if sunrise else None,
                    "sunset": sunset.split("T")[1][:5] if sunset else None,
                })
            
            output = {
                "location": {"latitude": latitude, "longitude": longitude},
                "timezone": data.get("timezone", "UTC"),
                "forecast": forecast
            }
            
            logger.info(
                f"Weather forecast for ({latitude}, {longitude}): "
                f"{len(forecast)} days retrieved"
            )
            
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000)
            )
            
        except httpx.TimeoutException:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Weather API timed out",
                latency_ms=int((time.time() - start_time) * 1000)
            )
        except httpx.HTTPStatusError as e:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Weather API HTTP error: {e.response.status_code}",
                latency_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Weather error: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000)
            )

