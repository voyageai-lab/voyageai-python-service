"""Pytest configuration and fixtures."""

import sys
from unittest.mock import MagicMock

# Pre-mock chromadb to avoid pydantic v1 incompatibility on Python 3.14
if "chromadb" not in sys.modules:
    _mock = MagicMock()
    sys.modules["chromadb"] = _mock
    sys.modules["chromadb.config"] = _mock

import pytest
from httpx import ASGITransport, AsyncClient

from voyageai.main import app


@pytest.fixture
async def client():
    """Create async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


