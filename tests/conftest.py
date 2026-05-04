"""Pytest configuration and fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient

from voyageai.main import app


@pytest.fixture
async def client():
    """Create async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
