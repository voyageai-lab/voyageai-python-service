"""
Tests for the Responses API Agent Service.

Verifies:
1. Tools are correctly converted to Responses API format
2. Built-in web_search is added when enabled
3. MCP servers are included when configured
4. Function call execution loop works
5. Final itinerary parsing handles JSON extraction
"""

import sys
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Pre-mock chromadb to avoid Python 3.14 pydantic v1 incompatibility
if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
    sys.modules["chromadb.config"] = MagicMock()

from voyageai.services.responses_agent_service import (
    ResponsesAgentService,
    _build_responses_tools,
)


class TestBuildResponsesTools:
    """Tests for tool conversion to Responses API format."""

    def test_converts_tools_to_responses_format(self):
        """Tools should use top-level name/description/parameters, not nested function."""
        tools = _build_responses_tools()
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t
            assert "description" in t
            assert "parameters" in t
            assert "function" not in t

    def test_excludes_xiaohongshu_tools(self):
        """Xiaohongshu tools should not be included."""
        tools = _build_responses_tools()
        names = [t["name"] for t in tools]
        for name in names:
            assert not name.startswith("xiaohongshu__")

    def test_filters_by_selected_names(self):
        """Only selected tool names should be included."""
        tools = _build_responses_tools(["geocode_location", "get_weather_forecast"])
        names = [t["name"] for t in tools]
        assert set(names) == {"geocode_location", "get_weather_forecast"}


class TestResponsesAgentService:
    """Tests for the ResponsesAgentService class."""

    def test_init_defaults(self):
        """Service should initialize with default settings."""
        with patch("voyageai.services.responses_agent_service.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"
            mock_settings.openai_model = "gpt-4o-mini"
            service = ResponsesAgentService()
            assert service.model == "gpt-4o-mini"
            assert service.max_iterations == 10

    def test_parse_itinerary_valid_json(self):
        """Should parse valid JSON from text."""
        service = ResponsesAgentService.__new__(ResponsesAgentService)
        text = '```json\n{"metadata": {"destination": "Tokyo", "start_date": "2026-03-15", "end_date": "2026-03-17", "total_days": 3, "budget": "Medium"}, "days": [{"day_number": 1, "date": "2026-03-15", "theme": "Test", "activities": [{"activity_id": "act-day1-001", "time": "09:00-11:00", "title": "Test", "description": "Desc", "location": {"name": "Place", "latitude": 35.0, "longitude": 139.0}}]}]}\n```'
        result = service._parse_itinerary(text)
        assert result is not None
        assert result.metadata.destination == "Tokyo"

    def test_parse_itinerary_empty(self):
        """Should return None for empty text."""
        service = ResponsesAgentService.__new__(ResponsesAgentService)
        assert service._parse_itinerary("") is None
        assert service._parse_itinerary("no json here") is None

    def test_parse_itinerary_invalid_json(self):
        """Should return None for invalid JSON."""
        service = ResponsesAgentService.__new__(ResponsesAgentService)
        assert service._parse_itinerary("{invalid}") is None


class TestResponsesApiIntegration:
    """Integration-level tests for the Responses API loop."""

    @pytest.mark.asyncio
    async def test_builtin_web_search_included(self):
        """Built-in web_search tool should be in the tools list."""
        from voyageai.config import settings as real_settings
        orig_mcp = real_settings.google_maps_mcp_url
        real_settings.google_maps_mcp_url = ""

        mock_response = MagicMock()
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=200)
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_output_item.content = [MagicMock(text='{"metadata": {"destination": "Test", "start_date": "2026-01-01", "end_date": "2026-01-02", "total_days": 1, "budget": "Low"}, "days": [{"day_number": 1, "date": "2026-01-01", "theme": "T", "activities": [{"activity_id": "a", "time": "09:00", "title": "T", "description": "D", "location": {"name": "N", "latitude": 0, "longitude": 0}}]}]}')]
        mock_response.output = [mock_output_item]
        mock_response.output_text = mock_output_item.content[0].text

        service = ResponsesAgentService()
        service.client = MagicMock()
        service.client.responses = MagicMock()
        service.client.responses.create = AsyncMock(return_value=mock_response)

        try:
            await service.generate_with_tools(requirements="Plan a trip to Test City")
        finally:
            real_settings.google_maps_mcp_url = orig_mcp

        # The first call includes tools; use call_args_list[0]
        first_call = service.client.responses.create.call_args_list[0]
        tools_arg = first_call.kwargs.get("tools", [])
        web_search_tools = [t for t in tools_arg if t.get("type") == "web_search"]
        assert len(web_search_tools) == 1

    @pytest.mark.asyncio
    async def test_mcp_server_included_when_configured(self):
        """MCP server should be in tools when URL is configured."""
        from voyageai.config import settings as real_settings

        mock_response = MagicMock()
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=200)
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_output_item.content = []
        mock_response.output = [mock_output_item]
        mock_response.output_text = ""

        service = ResponsesAgentService()
        service.client = MagicMock()
        service.client.responses = MagicMock()
        service.client.responses.create = AsyncMock(return_value=mock_response)

        # Temporarily override config values
        orig_web = real_settings.enable_builtin_web_search
        orig_mcp = real_settings.google_maps_mcp_url
        real_settings.enable_builtin_web_search = False
        real_settings.google_maps_mcp_url = "http://localhost:8080/mcp"
        try:
            await service.generate_with_tools(requirements="Test")
        finally:
            real_settings.enable_builtin_web_search = orig_web
            real_settings.google_maps_mcp_url = orig_mcp

        # The first call includes tools; use call_args_list[0]
        first_call = service.client.responses.create.call_args_list[0]
        tools_arg = first_call.kwargs.get("tools", [])
        mcp_tools = [t for t in tools_arg if t.get("type") == "mcp"]
        assert len(mcp_tools) == 1
        assert mcp_tools[0]["server_label"] == "google_maps"
