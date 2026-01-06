"""Tests for AI service with mocked OpenAI responses."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voyageai.prompts.templates import ITINERARY_EXAMPLE
from voyageai.services.ai_service import AIService


class TestAIService:
    """Tests for AIService class."""

    @pytest.fixture
    def ai_service(self):
        """Create AIService instance for testing."""
        return AIService()

    @pytest.fixture
    def mock_openai_response(self):
        """Create mock OpenAI response with valid itinerary."""
        # Use MagicMock for synchronous access patterns
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = json.dumps(ITINERARY_EXAMPLE)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    @pytest.mark.asyncio
    async def test_generate_itinerary_success(self, ai_service, mock_openai_response):
        """Test successful itinerary generation with mocked OpenAI."""
        # Use AsyncMock for the create method, returning MagicMock response
        mock_create = AsyncMock(return_value=mock_openai_response)
        
        with patch.object(
            ai_service.client.chat.completions,
            "create",
            mock_create,
        ):
            result = await ai_service.generate_itinerary("3 day trip to Tokyo")

            assert result.metadata.destination == "Tokyo, Japan"
            assert result.metadata.total_days == 3
            assert len(result.days) == 1
            assert result.days[0].day_number == 1
            assert len(result.days[0].activities) == 2

    @pytest.mark.asyncio
    async def test_generate_itinerary_validates_schema(self, ai_service):
        """Test that response is validated against Pydantic schema."""
        invalid_response = MagicMock()
        invalid_choice = MagicMock()
        invalid_message = MagicMock()
        # Missing required fields
        invalid_message.content = json.dumps({"metadata": {"destination": "Tokyo"}})
        invalid_choice.message = invalid_message
        invalid_response.choices = [invalid_choice]

        mock_create = AsyncMock(return_value=invalid_response)

        with patch.object(
            ai_service.client.chat.completions,
            "create",
            mock_create,
        ):
            with pytest.raises(Exception):  # Pydantic validation error
                await ai_service.generate_itinerary("3 day trip to Tokyo")

    @pytest.mark.asyncio
    async def test_generate_itinerary_handles_empty_response(self, ai_service):
        """Test handling of empty/null response from OpenAI."""
        empty_response = MagicMock()
        empty_choice = MagicMock()
        empty_message = MagicMock()
        empty_message.content = None
        empty_choice.message = empty_message
        empty_response.choices = [empty_choice]

        mock_create = AsyncMock(return_value=empty_response)

        with patch.object(
            ai_service.client.chat.completions,
            "create",
            mock_create,
        ):
            with pytest.raises(ValueError, match="Empty response"):
                await ai_service.generate_itinerary("3 day trip to Tokyo")

    @pytest.mark.asyncio
    async def test_generate_itinerary_handles_invalid_json(self, ai_service):
        """Test handling of invalid JSON response."""
        invalid_json_response = MagicMock()
        invalid_json_choice = MagicMock()
        invalid_json_message = MagicMock()
        invalid_json_message.content = "not valid json"
        invalid_json_choice.message = invalid_json_message
        invalid_json_response.choices = [invalid_json_choice]

        mock_create = AsyncMock(return_value=invalid_json_response)

        with patch.object(
            ai_service.client.chat.completions,
            "create",
            mock_create,
        ):
            with pytest.raises(json.JSONDecodeError):
                await ai_service.generate_itinerary("3 day trip to Tokyo")
