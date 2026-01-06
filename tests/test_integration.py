"""Integration tests for API endpoints."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from voyageai.prompts.templates import ITINERARY_EXAMPLE
from voyageai.schemas.itinerary import StructuredItinerary


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        """Test that health endpoint returns 200 OK."""
        response = await client.get("/api/v1/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_up_status(self, client):
        """Test that health endpoint returns UP status."""
        response = await client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "UP"

    @pytest.mark.asyncio
    async def test_health_returns_service_info(self, client):
        """Test that health endpoint returns service metadata."""
        response = await client.get("/api/v1/health")
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data


class TestGenerateEndpoint:
    """Tests for /generate endpoint."""

    @pytest.fixture
    def valid_request(self):
        """Valid generate request payload."""
        return {
            "task_id": "task-001",
            "user_id": "user-001",
            "project_id": "proj-001",
            "requirements": "3 day trip to Tokyo with focus on culture and food",
        }

    @pytest.fixture
    def mock_ai_service(self):
        """Mock AI service to return example itinerary."""
        async def mock_generate(*args, **kwargs):
            return StructuredItinerary.model_validate(ITINERARY_EXAMPLE)
        return mock_generate

    @pytest.mark.asyncio
    async def test_generate_returns_200(self, client, valid_request, mock_ai_service):
        """Test that generate endpoint returns 200 OK with mocked AI."""
        with patch(
            "voyageai.routers.planning.ai_service.generate_itinerary",
            mock_ai_service,
        ):
            response = await client.post("/api/v1/generate", json=valid_request)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_generate_returns_completed_status(
        self, client, valid_request, mock_ai_service
    ):
        """Test that successful generation returns COMPLETED status."""
        with patch(
            "voyageai.routers.planning.ai_service.generate_itinerary",
            mock_ai_service,
        ):
            response = await client.post("/api/v1/generate", json=valid_request)
            data = response.json()
            assert data["status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_generate_returns_itinerary(
        self, client, valid_request, mock_ai_service
    ):
        """Test that successful generation returns itinerary data."""
        with patch(
            "voyageai.routers.planning.ai_service.generate_itinerary",
            mock_ai_service,
        ):
            response = await client.post("/api/v1/generate", json=valid_request)
            data = response.json()
            assert data["itinerary"] is not None
            assert data["itinerary"]["metadata"]["destination"] == "Tokyo, Japan"

    @pytest.mark.asyncio
    async def test_generate_returns_task_id(
        self, client, valid_request, mock_ai_service
    ):
        """Test that response contains the original task_id."""
        with patch(
            "voyageai.routers.planning.ai_service.generate_itinerary",
            mock_ai_service,
        ):
            response = await client.post("/api/v1/generate", json=valid_request)
            data = response.json()
            assert data["task_id"] == valid_request["task_id"]

    @pytest.mark.asyncio
    async def test_generate_returns_processing_time(
        self, client, valid_request, mock_ai_service
    ):
        """Test that response includes processing time."""
        with patch(
            "voyageai.routers.planning.ai_service.generate_itinerary",
            mock_ai_service,
        ):
            response = await client.post("/api/v1/generate", json=valid_request)
            data = response.json()
            assert "processing_time_ms" in data
            assert data["processing_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_generate_handles_ai_failure(self, client, valid_request):
        """Test that AI service failure returns FAILED status."""
        async def mock_fail(*args, **kwargs):
            raise Exception("OpenAI API timeout")

        with patch(
            "voyageai.routers.planning.ai_service.generate_itinerary",
            mock_fail,
        ):
            response = await client.post("/api/v1/generate", json=valid_request)
            data = response.json()
            assert data["status"] == "FAILED"
            assert "error" in data
            assert "timeout" in data["error"]

    @pytest.mark.asyncio
    async def test_generate_validates_request(self, client):
        """Test that invalid request returns 422 validation error."""
        invalid_request = {
            "task_id": "task-001",
            # Missing required fields
        }
        response = await client.post("/api/v1/generate", json=invalid_request)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_generate_requirements_too_long(self, client):
        """Test that overly long requirements returns 422."""
        request = {
            "task_id": "task-001",
            "user_id": "user-001",
            "project_id": "proj-001",
            "requirements": "x" * 2001,  # Over 2000 char limit
        }
        response = await client.post("/api/v1/generate", json=request)
        assert response.status_code == 422

