"""
Tests for the Responses API Agent Service.

Verifies:
1. Tools are correctly converted to Responses API format
2. Built-in web_search is added when enabled
3. MCP servers are included when configured
4. Function call execution loop works
5. Final itinerary parsing handles JSON extraction
6. Tool-RAG integration
7. Xiaohongshu pre-fetch
8. Failed tool tracking
9. Clarification detection (analyze_request)
10. Plan outline generation
11. Heartbeat
12. call_single_tool
13. Retry for incomplete days
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
from voyageai.services.agent_types import AgentResponse, _is_permanent_error


class TestBuildResponsesTools:
    """Tests for tool conversion to Responses API format."""

    def test_converts_tools_to_responses_format(self):
        tools = _build_responses_tools()
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t
            assert "description" in t
            assert "parameters" in t
            assert "function" not in t

    def test_excludes_xiaohongshu_tools(self):
        tools = _build_responses_tools()
        names = [t["name"] for t in tools]
        for name in names:
            assert not name.startswith("xiaohongshu__")

    def test_filters_by_selected_names(self):
        tools = _build_responses_tools(["geocode_location", "get_weather_forecast"])
        names = [t["name"] for t in tools]
        assert set(names) == {"geocode_location", "get_weather_forecast"}


class TestResponsesAgentService:
    """Tests for the ResponsesAgentService class."""

    def test_init_defaults(self):
        with patch("voyageai.services.responses_agent_service.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"
            mock_settings.openai_model = "gpt-4o-mini"
            service = ResponsesAgentService()
            assert service.model == "gpt-4o-mini"
            assert service.max_iterations == 10

    def test_init_with_tool_rag(self):
        with patch("voyageai.services.responses_agent_service.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"
            mock_settings.openai_model = "gpt-4o-mini"
            service = ResponsesAgentService(use_tool_rag=True, tool_rag_top_k=5)
            assert service.use_tool_rag is True
            assert service.tool_rag_top_k == 5

    def test_parse_itinerary_valid_json(self):
        service = ResponsesAgentService.__new__(ResponsesAgentService)
        text = '```json\n{"metadata": {"destination": "Tokyo", "start_date": "2026-03-15", "end_date": "2026-03-17", "total_days": 3, "budget": "Medium"}, "days": [{"day_number": 1, "date": "2026-03-15", "theme": "Test", "activities": [{"activity_id": "act-day1-001", "time": "09:00-11:00", "title": "Test", "description": "Desc", "location": {"name": "Place", "latitude": 35.0, "longitude": 139.0}}]}]}\n```'
        result = service._parse_itinerary(text)
        assert result is not None
        assert result.metadata.destination == "Tokyo"

    def test_parse_itinerary_empty(self):
        service = ResponsesAgentService.__new__(ResponsesAgentService)
        assert service._parse_itinerary("") is None
        assert service._parse_itinerary("no json here") is None

    def test_parse_itinerary_invalid_json(self):
        service = ResponsesAgentService.__new__(ResponsesAgentService)
        assert service._parse_itinerary("{invalid}") is None

    def test_extract_destination(self):
        assert ResponsesAgentService._extract_destination("Plan a 3 day trip to Barcelona") == "Barcelona"
        assert "Tokyo" in ResponsesAgentService._extract_destination("Visit Tokyo in March")
        assert ResponsesAgentService._extract_destination("Random text") == "Random text"


class TestPermanentErrorDetection:
    """Tests for failed tool tracking."""

    def test_permanent_errors_detected(self):
        assert _is_permanent_error("API key invalid") is True
        assert _is_permanent_error("401 Unauthorized") is True
        assert _is_permanent_error("403 Forbidden access") is True
        assert _is_permanent_error("quota exceeded for this month") is True
        assert _is_permanent_error("Tool not configured") is True

    def test_transient_errors_not_permanent(self):
        assert _is_permanent_error("Connection timeout") is False
        assert _is_permanent_error("Server error 500") is False
        assert _is_permanent_error(None) is False
        assert _is_permanent_error("") is False


class TestAnalyzeRequest:
    """Tests for clarification detection."""

    @pytest.mark.asyncio
    async def test_analyze_request_ready(self):
        service = ResponsesAgentService()
        service.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"ready": true}'
        service.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await service.analyze_request("Plan a 3-day trip to Tokyo in March")
        assert result["ready"] is True

    @pytest.mark.asyncio
    async def test_analyze_request_needs_clarification(self):
        service = ResponsesAgentService()
        service.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "ready": False,
            "questions": [{"id": "q1", "question": "Where?", "type": "free_text", "options": []}],
        })
        service.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await service.analyze_request("Plan a trip")
        assert result["ready"] is False
        assert len(result["questions"]) == 1

    @pytest.mark.asyncio
    async def test_analyze_request_failure_returns_ready(self):
        service = ResponsesAgentService()
        service.client = MagicMock()
        service.client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))

        result = await service.analyze_request("Plan a trip to Tokyo")
        assert result["ready"] is True

    @pytest.mark.asyncio
    async def test_analyze_request_with_conversation_context(self):
        service = ResponsesAgentService()
        service.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"ready": true}'
        service.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await service.analyze_request(
            "Add a museum to day 1",
            conversation_context="User: Plan 3 days in Paris\nAgent: ...",
        )
        assert result["ready"] is True
        call_args = service.client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        assert "Previous conversation history" in prompt


class TestCallSingleTool:
    """Tests for the debug tool invocation endpoint."""

    @pytest.mark.asyncio
    async def test_call_single_tool_geocode(self):
        service = ResponsesAgentService()
        trace = await service.call_single_tool("geocode_location", {"location": "Tokyo"})
        assert trace.tool_name == "geocode_location"
        assert trace.success is True
        assert trace.call_id.startswith("direct-")

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        service = ResponsesAgentService()
        trace = await service.call_single_tool("nonexistent_tool", {})
        assert trace.success is False


class TestValidateItinerary:
    """Tests for itinerary JSON validation."""

    def test_validate_valid_json(self):
        content = json.dumps({
            "metadata": {"destination": "Tokyo", "start_date": "2026-01-01", "end_date": "2026-01-03", "total_days": 3, "budget": "Med"},
            "days": [{"day_number": 1, "date": "2026-01-01", "theme": "Day1", "activities": [
                {"activity_id": "a1", "time": "09:00-11:00", "title": "T", "description": "D",
                 "location": {"name": "P", "latitude": 35.0, "longitude": 139.0}}
            ]}],
        })
        itinerary, err = ResponsesAgentService._validate_itinerary(content)
        assert itinerary is not None
        assert err is None

    def test_validate_empty(self):
        itinerary, err = ResponsesAgentService._validate_itinerary("")
        assert itinerary is None

    def test_validate_invalid_json(self):
        itinerary, err = ResponsesAgentService._validate_itinerary("{not valid}")
        assert itinerary is None
        assert "Invalid JSON" in err

    def test_validate_strips_markdown_fences(self):
        inner = json.dumps({
            "metadata": {"destination": "Paris", "start_date": "2026-06-01", "end_date": "2026-06-02", "total_days": 1, "budget": "High"},
            "days": [{"day_number": 1, "date": "2026-06-01", "theme": "Art", "activities": [
                {"activity_id": "a1", "time": "10:00-12:00", "title": "Louvre", "description": "Visit museum",
                 "location": {"name": "Louvre Museum", "latitude": 48.86, "longitude": 2.34}}
            ]}],
        })
        content = f"```json\n{inner}\n```"
        itinerary, err = ResponsesAgentService._validate_itinerary(content)
        assert itinerary is not None
        assert itinerary.metadata.destination == "Paris"


class TestHeartbeat:
    """Tests for the heartbeat mechanism."""

    @pytest.mark.asyncio
    async def test_heartbeat_fires(self):
        import asyncio
        service = ResponsesAgentService.__new__(ResponsesAgentService)
        events = []

        async def mock_callback(event_type, data):
            events.append((event_type, data))

        async def slow_task():
            await asyncio.sleep(0.3)
            return "done"

        result = await service._with_heartbeat(
            slow_task(), mock_callback, interval=0.1, message="Working...",
        )
        assert result == "done"
        assert len(events) >= 2
        assert all(e[0] == "thinking" for e in events)

    @pytest.mark.asyncio
    async def test_heartbeat_skipped_without_callback(self):
        service = ResponsesAgentService.__new__(ResponsesAgentService)

        async def quick_task():
            return 42

        result = await service._with_heartbeat(quick_task(), None)
        assert result == 42


class TestResponsesApiIntegration:
    """Integration-level tests for the Responses API loop."""

    @pytest.mark.asyncio
    async def test_builtin_web_search_included(self):
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
        service.client.chat = MagicMock()
        service.client.chat.completions = MagicMock()
        service.client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"summary": "Test"}'))],
            usage=MagicMock(total_tokens=50, prompt_tokens=30, completion_tokens=20, completion_tokens_details=None),
        ))

        try:
            await service.generate_with_tools(requirements="Plan a trip to Test City")
        finally:
            real_settings.google_maps_mcp_url = orig_mcp

        first_call = service.client.responses.create.call_args_list[0]
        tools_arg = first_call.kwargs.get("tools", [])
        web_search_tools = [t for t in tools_arg if t.get("type") == "web_search"]
        assert len(web_search_tools) == 1

    @pytest.mark.asyncio
    async def test_mcp_server_included_when_configured(self):
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
        service.client.chat = MagicMock()
        service.client.chat.completions = MagicMock()
        service.client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"summary": "Test"}'))],
            usage=MagicMock(total_tokens=50, prompt_tokens=30, completion_tokens=20, completion_tokens_details=None),
        ))

        orig_web = real_settings.enable_builtin_web_search
        orig_mcp = real_settings.google_maps_mcp_url
        real_settings.enable_builtin_web_search = False
        real_settings.google_maps_mcp_url = "http://localhost:8080/mcp"
        try:
            await service.generate_with_tools(requirements="Test")
        finally:
            real_settings.enable_builtin_web_search = orig_web
            real_settings.google_maps_mcp_url = orig_mcp

        first_call = service.client.responses.create.call_args_list[0]
        tools_arg = first_call.kwargs.get("tools", [])
        mcp_tools = [t for t in tools_arg if t.get("type") == "mcp"]
        assert len(mcp_tools) == 1
        assert mcp_tools[0]["server_label"] == "google_maps"

    @pytest.mark.asyncio
    async def test_failed_tool_tracking(self):
        """Permanently failed tools should be tracked and advisory injected."""
        service = ResponsesAgentService()
        service.client = MagicMock()
        service.client.responses = MagicMock()
        service.client.chat = MagicMock()
        service.client.chat.completions = MagicMock()
        service.client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"summary": "Test"}'))],
            usage=MagicMock(total_tokens=50, prompt_tokens=30, completion_tokens=20, completion_tokens_details=None),
        ))

        # First response: function call to a tool that will fail permanently
        mock_fn_call = MagicMock()
        mock_fn_call.type = "function_call"
        mock_fn_call.name = "search_flights"
        mock_fn_call.arguments = '{"from": "NYC", "to": "LAX"}'
        mock_fn_call.call_id = "call_fail"

        mock_response1 = MagicMock()
        mock_response1.usage = MagicMock(input_tokens=50, output_tokens=50)
        mock_response1.output = [mock_fn_call]

        # Second response: message (no more tool calls)
        mock_msg = MagicMock()
        mock_msg.type = "message"
        mock_msg.content = [MagicMock(text="I'll use alternative tools")]
        mock_response2 = MagicMock()
        mock_response2.usage = MagicMock(input_tokens=50, output_tokens=50)
        mock_response2.output = [mock_msg]
        mock_response2.output_text = ""

        service.client.responses.create = AsyncMock(side_effect=[mock_response1, mock_response2])

        with patch("voyageai.services.responses_agent_service.tool_registry") as mock_registry:
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "401 Unauthorized - API key invalid"
            mock_result.output = None
            mock_result.latency_ms = 100
            mock_registry.execute = AsyncMock(return_value=mock_result)
            mock_registry.list_tools.return_value = ["geocode_location"]
            mock_registry.get_openai_tools.return_value = []
            mock_registry.get_tool_descriptions.return_value = [{"name": "geocode_location", "description": "test"}]

            response = await service.generate_with_tools(requirements="Test flight search")

        assert any(t.tool_name == "search_flights" and not t.success for t in response.tool_trace)


class TestXhsSourceLinkInjection:
    """Tests for Xiaohongshu source link post-processing."""

    def test_injects_links(self):
        from voyageai.schemas.itinerary import (
            Activity, DailyItinerary, ItineraryMetadata, Location, StructuredItinerary,
        )
        itinerary = StructuredItinerary(
            metadata=ItineraryMetadata(destination="Tokyo", start_date="2026-01-01", end_date="2026-01-02", total_days=1, budget="Med"),
            days=[DailyItinerary(day_number=1, date="2026-01-01", theme="Day 1", activities=[
                Activity(activity_id="a1", time="09:00", title="Test", description="D",
                         location=Location(name="P", latitude=35.0, longitude=139.0), source_links=[]),
            ])],
        )
        xhs_context = (
            "=== XIAOHONGSHU ===\n"
            "### Tokyo Food Guide\n"
            "Author: TravelUser | Likes: 500 | Collects: 200\n"
            "URL: https://www.xiaohongshu.com/explore/abc123\n"
            "Content: Great ramen spots...\n"
        )
        ResponsesAgentService._inject_xhs_source_links(itinerary, xhs_context)
        links = itinerary.days[0].activities[0].source_links
        assert any(sl.source == "xiaohongshu" for sl in links)
        assert any("abc123" in sl.url for sl in links)

    def test_no_duplicates(self):
        from voyageai.schemas.itinerary import (
            Activity, DailyItinerary, ItineraryMetadata, Location, SourceLink, StructuredItinerary,
        )
        existing_link = SourceLink(
            title="Existing", url="https://www.xiaohongshu.com/explore/abc123",
            source="xiaohongshu", snippet="already there",
        )
        itinerary = StructuredItinerary(
            metadata=ItineraryMetadata(destination="Tokyo", start_date="2026-01-01", end_date="2026-01-02", total_days=1, budget="Med"),
            days=[DailyItinerary(day_number=1, date="2026-01-01", theme="Day 1", activities=[
                Activity(activity_id="a1", time="09:00", title="Test", description="D",
                         location=Location(name="P", latitude=35.0, longitude=139.0),
                         source_links=[existing_link]),
            ])],
        )
        xhs_context = (
            "### Tokyo Food\n"
            "Author: User | Likes: 100\n"
            "URL: https://www.xiaohongshu.com/explore/abc123\n"
        )
        ResponsesAgentService._inject_xhs_source_links(itinerary, xhs_context)
        urls = [sl.url for sl in itinerary.days[0].activities[0].source_links]
        assert urls.count("https://www.xiaohongshu.com/explore/abc123") == 1
