"""
Integration tests for the Agent Service.

These tests verify:
1. Agent can call tools correctly
2. Tool results are incorporated into responses
3. Tool trace is captured properly
4. Error handling for tool failures

Note: Tests that call the full agent require OPENAI_API_KEY.
Tests marked with @pytest.mark.requires_openai should be run
with the environment variable set.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from voyageai.services.agent_service import AgentService, AgentResponse
from voyageai.schemas.tool import ToolCallTrace
from voyageai.tools.registry import tool_registry


class TestAgentServiceToolCalling:
    """Tests for AgentService tool calling functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create an agent service instance."""
        return AgentService(max_iterations=5)
    
    @pytest.mark.asyncio
    async def test_call_single_tool_geocode(self, agent):
        """Test calling geocode tool directly through agent."""
        trace = await agent.call_single_tool(
            "geocode_location",
            {"location": "Paris, France"}
        )
        
        assert isinstance(trace, ToolCallTrace)
        assert trace.tool_name == "geocode_location"
        assert trace.success is True
        assert trace.result is not None
        assert "latitude" in trace.result
        # Paris is roughly at 48.86, 2.35
        assert 48 < trace.result["latitude"] < 49
    
    @pytest.mark.asyncio
    async def test_call_single_tool_distance(self, agent):
        """Test calling distance tool through agent."""
        trace = await agent.call_single_tool(
            "calculate_distance",
            {
                "from_latitude": 48.86,
                "from_longitude": 2.35,
                "to_latitude": 51.51,
                "to_longitude": -0.13
            }
        )
        
        assert trace.success is True
        assert trace.result["distance_km"] > 300  # Paris to London ~340km
    
    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, agent):
        """Test calling unknown tool returns error."""
        trace = await agent.call_single_tool(
            "nonexistent_tool",
            {}
        )
        
        assert trace.success is False
        assert "unknown" in trace.error.lower()
    
    @pytest.mark.asyncio
    async def test_call_tool_with_invalid_args(self, agent):
        """Test calling tool with invalid arguments."""
        trace = await agent.call_single_tool(
            "calculate_distance",
            {
                "invalid_param": "value"
            }
        )
        
        assert trace.success is False


class TestAgentServiceMocked:
    """Tests for AgentService with mocked OpenAI responses."""
    
    @pytest.fixture
    def agent(self):
        return AgentService(max_iterations=3)
    
    @pytest.mark.asyncio
    async def test_agent_executes_tool_calls(self, agent):
        """Test that agent executes tool calls from LLM response."""
        # Mock OpenAI response with tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "geocode_location"
        mock_tool_call.function.arguments = '{"location": "Tokyo"}'
        
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=60, completion_tokens=40, completion_tokens_details=None)
        
        # Second response (after tools) - no more tool calls
        mock_message2 = MagicMock()
        mock_message2.content = "Based on the geocoding result..."
        mock_message2.tool_calls = None
        
        mock_choice2 = MagicMock()
        mock_choice2.message = mock_message2
        mock_choice2.finish_reason = "stop"
        
        mock_response2 = MagicMock()
        mock_response2.choices = [mock_choice2]
        mock_response2.usage = MagicMock(total_tokens=150, prompt_tokens=90, completion_tokens=60, completion_tokens_details=None)
        
        # Final structured response
        mock_final_message = MagicMock()
        mock_final_message.content = '''{
            "metadata": {
                "destination": "Tokyo",
                "start_date": "2026-01-10",
                "end_date": "2026-01-10",
                "total_days": 1,
                "budget": "$2000",
                "interests": ["culture"]
            },
            "days": [{
                "day_number": 1,
                "date": "2026-01-10",
                "theme": "Exploration",
                "activities": [{
                    "activity_id": "act-day1-001",
                    "time": "09:00-12:00",
                    "title": "Visit Temple",
                    "description": "Explore historic temple",
                    "location": {
                        "name": "Senso-ji",
                        "latitude": 35.71,
                        "longitude": 139.80,
                        "address": "Tokyo",
                        "place_type": "temple"
                    },
                    "estimated_cost": "$10",
                    "notes": []
                }]
            }],
            "tips": ["Bring umbrella"]
        }'''
        
        mock_final_choice = MagicMock()
        mock_final_choice.message = mock_final_message
        
        mock_final_response = MagicMock()
        mock_final_response.choices = [mock_final_choice]
        mock_final_response.usage = MagicMock(total_tokens=200, prompt_tokens=100, completion_tokens=100, completion_tokens_details=None)

        # Plan outline response (consumed by _generate_plan_outline)
        mock_outline_message = MagicMock()
        mock_outline_message.content = '{"summary": "3-day Tokyo trip", "daily_themes": []}'
        mock_outline_choice = MagicMock()
        mock_outline_choice.message = mock_outline_message
        mock_outline_response = MagicMock()
        mock_outline_response.choices = [mock_outline_choice]
        mock_outline_response.usage = MagicMock(total_tokens=50, prompt_tokens=30, completion_tokens=20, completion_tokens_details=None)
        
        with patch.object(
            agent.client.chat.completions,
            'create',
            new_callable=AsyncMock,
            side_effect=[mock_response, mock_response2, mock_outline_response, mock_final_response]
        ):
            response = await agent.generate_with_tools("Plan a trip to Tokyo")
        
        assert response.success is True
        assert len(response.tool_trace) >= 1
        # The geocode tool should have been called
        geocode_trace = next(
            (t for t in response.tool_trace if t.tool_name == "geocode_location"),
            None
        )
        assert geocode_trace is not None
        assert geocode_trace.success is True


class TestToolRouterIntegration:
    """Integration tests for the tools router."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from httpx import AsyncClient, ASGITransport
        from voyageai.main import app
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    
    @pytest.mark.asyncio
    async def test_list_tools(self, client):
        """Test listing all available tools."""
        async with client:
            response = await client.get("/api/v1/tools")
        
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert data["total"] >= 6
        
        tool_names = [t["name"] for t in data["tools"]]
        assert "geocode_location" in tool_names
        assert "get_weather_forecast" in tool_names
    
    @pytest.mark.asyncio
    async def test_geocode_endpoint(self, client):
        """Test geocode endpoint."""
        async with client:
            response = await client.get(
                "/api/v1/tools/geocode",
                params={"location": "London, UK"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "latitude" in data["data"]
        # London is roughly at 51.5, -0.1
        assert 51 < data["data"]["latitude"] < 52
    
    @pytest.mark.asyncio
    async def test_distance_endpoint(self, client):
        """Test distance endpoint."""
        async with client:
            response = await client.get(
                "/api/v1/tools/distance",
                params={
                    "from_latitude": 35.68,
                    "from_longitude": 139.76,
                    "to_latitude": 34.69,
                    "to_longitude": 135.50
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # Tokyo to Osaka is ~400km
        assert 395 < data["data"]["distance_km"] < 410
    
    @pytest.mark.asyncio
    async def test_timezone_endpoint(self, client):
        """Test timezone endpoint."""
        async with client:
            response = await client.get(
                "/api/v1/tools/timezone",
                params={
                    "time": "14:00",
                    "from_timezone": "PST",
                    "to_timezone": "JST",
                    "date": "2026-01-10"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_invoke_tool_endpoint(self, client):
        """Test generic tool invocation endpoint."""
        async with client:
            response = await client.post(
                "/api/v1/tools/invoke",
                json={
                    "tool_name": "calculate_distance",
                    "arguments": {
                        "from_latitude": 35.68,
                        "from_longitude": 139.76,
                        "to_latitude": 34.69,
                        "to_longitude": 135.50
                    }
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["tool_name"] == "calculate_distance"

