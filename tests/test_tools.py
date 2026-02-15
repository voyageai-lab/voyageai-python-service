"""
Unit tests for VoyageAI tools.

These tests verify:
1. Tool initialization and configuration
2. Input validation
3. Successful execution with real APIs
4. Error handling

Note: These tests call real APIs (Open-Meteo, Nominatim, etc.)
which are all free and don't require API keys.
"""

import pytest
from datetime import datetime, timedelta

from voyageai.tools.base import BaseTool, ToolResult
from voyageai.tools.geocode import GeocodeTool
from voyageai.tools.weather import WeatherTool
from voyageai.tools.currency import CurrencyTool
from voyageai.tools.timezone import TimeZoneTool
from voyageai.tools.distance import DistanceTool
from voyageai.tools.holiday import HolidayTool
from voyageai.tools.registry import ToolRegistry, tool_registry
from voyageai.tools.websearch import WebSearchTool


# ============================================================================
# Base Tool Tests
# ============================================================================

class TestToolResult:
    """Tests for ToolResult model."""
    
    def test_successful_result(self):
        """Test creating a successful tool result."""
        result = ToolResult(
            tool_name="test_tool",
            input_args={"key": "value"},
            output={"result": "data"},
            success=True,
            latency_ms=100
        )
        
        assert result.tool_name == "test_tool"
        assert result.success is True
        assert result.error is None
        assert result.latency_ms == 100
    
    def test_failed_result(self):
        """Test creating a failed tool result."""
        result = ToolResult(
            tool_name="test_tool",
            input_args={},
            output=None,
            success=False,
            error="Something went wrong",
            latency_ms=50
        )
        
        assert result.success is False
        assert result.error == "Something went wrong"


# ============================================================================
# Geocode Tool Tests (uses Nominatim - free, no key)
# ============================================================================

class TestGeocodeTool:
    """Tests for GeocodeTool."""
    
    @pytest.fixture
    def tool(self):
        return GeocodeTool()
    
    def test_openai_function_format(self, tool):
        """Test tool can be converted to OpenAI function format."""
        func = tool.to_openai_function()
        
        assert func["type"] == "function"
        assert func["function"]["name"] == "geocode_location"
        assert "parameters" in func["function"]
    
    @pytest.mark.asyncio
    async def test_geocode_tokyo(self, tool):
        """Test geocoding Tokyo."""
        result = await tool.execute(location="Tokyo, Japan")
        
        assert result.success is True
        assert result.output is not None
        assert "latitude" in result.output
        assert "longitude" in result.output
        # Tokyo is roughly at 35.68, 139.76
        assert 35 < result.output["latitude"] < 36
        assert 139 < result.output["longitude"] < 140
    
    @pytest.mark.asyncio
    async def test_geocode_empty_location(self, tool):
        """Test geocoding with empty location fails gracefully."""
        result = await tool.execute(location="")
        
        assert result.success is False
        assert "empty" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_geocode_nonexistent_location(self, tool):
        """Test geocoding nonexistent location."""
        result = await tool.execute(location="Xyzzy123NotAPlace")
        
        assert result.success is False
        assert "not found" in result.error.lower()


# ============================================================================
# Weather Tool Tests (uses Open-Meteo - free, no key)
# ============================================================================

class TestWeatherTool:
    """Tests for WeatherTool."""
    
    @pytest.fixture
    def tool(self):
        return WeatherTool()
    
    def test_weather_code_translation(self, tool):
        """Test weather code to text translation."""
        assert tool._weather_code_to_text(0) == "Clear sky"
        assert tool._weather_code_to_text(3) == "Overcast"
        assert tool._weather_code_to_text(61) == "Slight rain"
        assert "Unknown" in tool._weather_code_to_text(999)
    
    @pytest.mark.asyncio
    async def test_weather_tokyo(self, tool):
        """Test getting weather for Tokyo."""
        # Use dates within the 16-day forecast window
        today = datetime.now()
        start = today.strftime("%Y-%m-%d")
        end = (today + timedelta(days=2)).strftime("%Y-%m-%d")
        
        result = await tool.execute(
            latitude=35.68,
            longitude=139.76,
            start_date=start,
            end_date=end
        )
        
        assert result.success is True
        assert result.output is not None
        assert "forecast" in result.output
        assert len(result.output["forecast"]) >= 1
        
        # Check forecast structure
        day = result.output["forecast"][0]
        assert "date" in day
        assert "temp_max_c" in day
        assert "condition" in day
    
    @pytest.mark.asyncio
    async def test_weather_invalid_coordinates(self, tool):
        """Test weather with invalid coordinates."""
        result = await tool.execute(
            latitude=999,  # Invalid
            longitude=139.76,
            start_date="2026-01-10",
            end_date="2026-01-12"
        )
        
        assert result.success is False
        assert "latitude" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_weather_invalid_date_range(self, tool):
        """Test weather with end date before start date."""
        result = await tool.execute(
            latitude=35.68,
            longitude=139.76,
            start_date="2026-01-15",
            end_date="2026-01-10"  # Before start
        )
        
        assert result.success is False
        assert "after" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_weather_historical_dates(self, tool):
        """Test weather with past dates uses archive API."""
        result = await tool.execute(
            latitude=35.68,
            longitude=139.76,
            start_date="2024-03-25",
            end_date="2024-03-29"
        )
        
        # Should succeed using archive API instead of returning 400
        assert result.success is True
        assert result.output is not None
        assert result.output["data_source"] == "archive (historical)"
        assert len(result.output["forecast"]) >= 1


# ============================================================================
# Currency Tool Tests (uses Frankfurter - free, no key)
# ============================================================================

class TestCurrencyTool:
    """Tests for CurrencyTool."""
    
    @pytest.fixture
    def tool(self):
        return CurrencyTool()
    
    @pytest.mark.asyncio
    async def test_convert_usd_to_eur(self, tool):
        """Test converting USD to EUR."""
        result = await tool.execute(
            from_currency="USD",
            to_currency="EUR",
            amount=100.0
        )
        
        assert result.success is True
        assert result.output is not None
        assert result.output["from_currency"] == "USD"
        assert result.output["to_currency"] == "EUR"
        assert result.output["original_amount"] == 100.0
        assert result.output["converted_amount"] > 0
        assert result.output["exchange_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_convert_same_currency(self, tool):
        """Test converting same currency returns same amount."""
        result = await tool.execute(
            from_currency="USD",
            to_currency="USD",
            amount=100.0
        )
        
        assert result.success is True
        assert result.output["converted_amount"] == 100.0
        assert result.output["exchange_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_convert_invalid_currency(self, tool):
        """Test converting invalid currency fails."""
        result = await tool.execute(
            from_currency="XYZ",
            to_currency="EUR",
            amount=100.0
        )
        
        assert result.success is False
        assert "unsupported" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_convert_negative_amount(self, tool):
        """Test converting negative amount fails."""
        result = await tool.execute(
            from_currency="USD",
            to_currency="EUR",
            amount=-100.0
        )
        
        assert result.success is False


# ============================================================================
# Timezone Tool Tests (uses Python zoneinfo - no API)
# ============================================================================

class TestTimeZoneTool:
    """Tests for TimeZoneTool."""
    
    @pytest.fixture
    def tool(self):
        return TimeZoneTool()
    
    def test_resolve_iana_timezone(self, tool):
        """Test resolving IANA timezone names."""
        assert tool._resolve_timezone("America/New_York") == "America/New_York"
        assert tool._resolve_timezone("Asia/Tokyo") == "Asia/Tokyo"
    
    def test_resolve_timezone_aliases(self, tool):
        """Test resolving timezone aliases."""
        assert tool._resolve_timezone("PST") == "America/Los_Angeles"
        assert tool._resolve_timezone("JST") == "Asia/Tokyo"
        assert tool._resolve_timezone("TOKYO") == "Asia/Tokyo"
    
    @pytest.mark.asyncio
    async def test_convert_ny_to_tokyo(self, tool):
        """Test converting New York time to Tokyo."""
        result = await tool.execute(
            time="2026-01-10T14:00:00",
            from_timezone="America/New_York",
            to_timezone="Asia/Tokyo"
        )
        
        assert result.success is True
        assert result.output is not None
        assert result.output["original"]["timezone"] == "America/New_York"
        assert result.output["converted"]["timezone"] == "Asia/Tokyo"
        # Tokyo is 14 hours ahead of NY (EST)
        assert result.output["time_difference_hours"] == 14.0
    
    @pytest.mark.asyncio
    async def test_convert_with_alias(self, tool):
        """Test converting with timezone aliases."""
        result = await tool.execute(
            time="14:00",
            from_timezone="PST",
            to_timezone="JST",
            date="2026-01-10"
        )
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_invalid_timezone(self, tool):
        """Test invalid timezone fails."""
        result = await tool.execute(
            time="14:00",
            from_timezone="InvalidTZ",
            to_timezone="Asia/Tokyo"
        )
        
        assert result.success is False
        assert "unknown" in result.error.lower()


# ============================================================================
# Distance Tool Tests (uses Haversine - no API)
# ============================================================================

class TestDistanceTool:
    """Tests for DistanceTool."""
    
    @pytest.fixture
    def tool(self):
        return DistanceTool()
    
    def test_haversine_calculation(self, tool):
        """Test Haversine distance calculation."""
        # Tokyo to Osaka - approximately 400km
        distance = tool._haversine_distance(
            35.68, 139.76,  # Tokyo
            34.69, 135.50   # Osaka
        )
        
        assert 395 < distance < 410  # Should be ~403km
    
    def test_bearing_calculation(self, tool):
        """Test bearing calculation."""
        # Tokyo to Osaka - should be roughly southwest (around 250 degrees)
        bearing = tool._calculate_bearing(
            35.68, 139.76,  # Tokyo
            34.69, 135.50   # Osaka
        )
        
        assert 240 < bearing < 260
    
    @pytest.mark.asyncio
    async def test_distance_tokyo_to_osaka(self, tool):
        """Test calculating distance from Tokyo to Osaka."""
        result = await tool.execute(
            from_latitude=35.68,
            from_longitude=139.76,
            to_latitude=34.69,
            to_longitude=135.50
        )
        
        assert result.success is True
        assert result.output is not None
        assert 395 < result.output["distance_km"] < 410
        assert "estimated_travel_time" in result.output
    
    @pytest.mark.asyncio
    async def test_distance_invalid_latitude(self, tool):
        """Test distance with invalid latitude."""
        result = await tool.execute(
            from_latitude=999,  # Invalid
            from_longitude=139.76,
            to_latitude=34.69,
            to_longitude=135.50
        )
        
        assert result.success is False


# ============================================================================
# Holiday Tool Tests (uses Nager.Date - free, no key)
# ============================================================================

class TestHolidayTool:
    """Tests for HolidayTool."""
    
    @pytest.fixture
    def tool(self):
        return HolidayTool()
    
    @pytest.mark.asyncio
    async def test_holidays_japan_2026(self, tool):
        """Test getting Japanese holidays for 2026."""
        result = await tool.execute(
            country_code="JP",
            year=2026
        )
        
        assert result.success is True
        assert result.output is not None
        assert result.output["country"] == "Japan"
        assert len(result.output["holidays"]) > 10  # Japan has many holidays
        
        # Check for New Year's Day
        holidays = result.output["holidays"]
        new_year = next((h for h in holidays if h["date"] == "2026-01-01"), None)
        assert new_year is not None
    
    @pytest.mark.asyncio
    async def test_holidays_us_2026(self, tool):
        """Test getting US holidays."""
        result = await tool.execute(
            country_code="US",
            year=2026
        )
        
        assert result.success is True
        assert result.output["country_code"] == "US"
    
    @pytest.mark.asyncio
    async def test_holidays_invalid_country(self, tool):
        """Test invalid country code fails."""
        result = await tool.execute(
            country_code="XX",
            year=2026
        )
        
        assert result.success is False
        assert "unsupported" in result.error.lower()


# ============================================================================
# Web Search Tool Tests (uses DuckDuckGo - free, no key)
# ============================================================================

class TestWebSearchTool:
    """Tests for WebSearchTool."""
    
    @pytest.fixture
    def tool(self):
        return WebSearchTool()
    
    def test_openai_function_format(self, tool):
        """Test tool can be converted to OpenAI function format."""
        func = tool.to_openai_function()
        
        assert func["type"] == "function"
        assert func["function"]["name"] == "web_search"
        assert "parameters" in func["function"]
    
    @pytest.mark.asyncio
    async def test_search_tokyo_travel(self, tool):
        """Test searching for Tokyo travel information."""
        result = await tool.execute(query="best cherry blossom spots Tokyo")
        
        assert result.success is True
        assert result.output is not None
        assert "results" in result.output
        assert result.output["query"] == "best cherry blossom spots Tokyo"
    
    @pytest.mark.asyncio
    async def test_search_empty_query(self, tool):
        """Test search with empty query fails."""
        result = await tool.execute(query="")
        
        assert result.success is False
        assert "empty" in result.error.lower()


# ============================================================================
# Tool Registry Tests
# ============================================================================

class TestToolRegistry:
    """Tests for ToolRegistry."""
    
    def test_default_registry_has_all_tools(self):
        """Test that the default registry has all expected tools."""
        expected_tools = [
            "geocode_location",
            "get_weather_forecast",
            "convert_currency",
            "convert_timezone",
            "calculate_distance",
            "get_public_holidays",
            "web_search",
        ]
        
        for tool_name in expected_tools:
            assert tool_registry.get(tool_name) is not None, f"Missing tool: {tool_name}"
    
    def test_get_openai_tools_format(self):
        """Test OpenAI tools format is correct."""
        tools = tool_registry.get_openai_tools()
        
        assert len(tools) >= 7
        
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
    
    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool returns error."""
        result = await tool_registry.execute("nonexistent_tool", {})
        
        assert result.success is False
        assert "unknown" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_tool_via_registry(self):
        """Test executing a tool through the registry."""
        result = await tool_registry.execute("calculate_distance", {
            "from_latitude": 35.68,
            "from_longitude": 139.76,
            "to_latitude": 34.69,
            "to_longitude": 135.50
        })
        
        assert result.success is True
        assert result.output["distance_km"] > 0

