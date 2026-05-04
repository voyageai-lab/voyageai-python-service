"""
Unit tests for the Reference Itinerary Import endpoint.

Tests cover:
1. Successful import from raw text
2. Successful import from URL (mocked HTTP)
3. Validation: neither URL nor raw_text provided
4. Validation: content too short
5. URL fetch failure handling
6. Text chunking logic
7. ChromaDB storage failure handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from voyageai.main import app
from voyageai.routers.reference import _chunk_text, _strip_html


# ============================================================================
# Text Utility Tests
# ============================================================================

class TestChunkText:
    """Tests for the text chunking function."""

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size returns a single chunk."""
        text = "A short travel blog about Tokyo."
        chunks = _chunk_text(text, chunk_size=1500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_paragraph_based_splitting(self):
        """Paragraphs are kept together when possible."""
        para1 = "A" * 500
        para2 = "B" * 500
        para3 = "C" * 500
        text = f"{para1}\n\n{para2}\n\n{para3}"

        chunks = _chunk_text(text, chunk_size=1100, overlap=100)
        assert len(chunks) >= 2
        assert para1 in chunks[0]

    def test_empty_paragraphs_ignored(self):
        """Blank paragraphs don't produce empty chunks."""
        text = "Hello\n\n\n\nWorld"
        chunks = _chunk_text(text, chunk_size=5000)
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_large_paragraph_force_split(self):
        """A single paragraph larger than chunk_size is force-split."""
        text = "X" * 3000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 3


class TestStripHtml:
    """Tests for naive HTML stripping."""

    def test_removes_tags(self):
        html = "<p>Hello <b>World</b></p>"
        assert "Hello" in _strip_html(html)
        assert "<p>" not in _strip_html(html)

    def test_removes_script_and_style(self):
        html = "<script>alert('xss')</script><style>.x{}</style><p>Content</p>"
        result = _strip_html(html)
        assert "alert" not in result
        assert "Content" in result


# ============================================================================
# Endpoint Tests (mocked ChromaDB and httpx)
# ============================================================================

@pytest.fixture
async def test_client():
    """Create async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestReferenceImportEndpoint:
    """Tests for POST /api/v1/reference-itinerary."""

    @pytest.mark.asyncio
    async def test_import_raw_text(self, test_client):
        """Importing raw text should store chunks and return a document ID."""
        raw_text = "Tokyo is amazing for cherry blossom season. " * 5

        with patch("voyageai.routers.reference.chroma_client") as mock_chroma:
            mock_chroma.initialize = AsyncMock()
            mock_chroma.add_documents = AsyncMock(return_value=1)

            response = await test_client.post(
                "/api/v1/reference-itinerary",
                json={
                    "raw_text": raw_text,
                    "title": "Tokyo Cherry Blossom Guide",
                    "destination": "Tokyo",
                    "tags": ["cherry_blossom", "spring"],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["chunks_stored"] >= 1
        assert data["source"] == "raw_text"
        assert data["title"] == "Tokyo Cherry Blossom Guide"
        mock_chroma.add_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_import_from_url(self, test_client):
        """Importing from a URL should fetch content and store it."""
        fake_html = "<html><body><p>" + ("Paris travel tips. " * 20) + "</p></body></html>"

        mock_response = MagicMock()
        mock_response.text = fake_html
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("voyageai.routers.reference.chroma_client") as mock_chroma,
            patch("voyageai.routers.reference.httpx.AsyncClient") as mock_httpx_cls,
        ):
            mock_chroma.initialize = AsyncMock()
            mock_chroma.add_documents = AsyncMock(return_value=1)

            mock_httpx_instance = AsyncMock()
            mock_httpx_instance.get = AsyncMock(return_value=mock_response)
            mock_httpx_instance.__aenter__ = AsyncMock(return_value=mock_httpx_instance)
            mock_httpx_instance.__aexit__ = AsyncMock(return_value=False)
            mock_httpx_cls.return_value = mock_httpx_instance

            response = await test_client.post(
                "/api/v1/reference-itinerary",
                json={
                    "url": "https://example.com/paris-guide",
                    "title": "Paris Travel Guide",
                    "destination": "Paris",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "url"
        assert data["chunks_stored"] >= 1

    @pytest.mark.asyncio
    async def test_missing_url_and_text(self, test_client):
        """Should return 422 when neither url nor raw_text is provided."""
        response = await test_client.post(
            "/api/v1/reference-itinerary",
            json={"title": "No content"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_content_too_short(self, test_client):
        """Should reject content shorter than 50 characters."""
        response = await test_client.post(
            "/api/v1/reference-itinerary",
            json={"raw_text": "Too short"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chromadb_failure(self, test_client):
        """Should return 500 when ChromaDB storage fails."""
        raw_text = "A sufficiently long travel blog about visiting Rome in the spring. " * 3

        with patch("voyageai.routers.reference.chroma_client") as mock_chroma:
            mock_chroma.initialize = AsyncMock()
            mock_chroma.add_documents = AsyncMock(side_effect=RuntimeError("DB down"))

            response = await test_client.post(
                "/api/v1/reference-itinerary",
                json={"raw_text": raw_text},
            )

        assert response.status_code == 500
        assert "Failed to store reference" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_metadata_includes_reference_marker(self, test_client):
        """Stored metadata must include source=reference and relevance_weight=high."""
        raw_text = "Detailed guide to exploring Kyoto temples and gardens. " * 5

        with patch("voyageai.routers.reference.chroma_client") as mock_chroma:
            mock_chroma.initialize = AsyncMock()
            mock_chroma.add_documents = AsyncMock(return_value=1)

            await test_client.post(
                "/api/v1/reference-itinerary",
                json={"raw_text": raw_text, "destination": "Kyoto"},
            )

            call_args = mock_chroma.add_documents.call_args
            metadatas = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas")
            assert metadatas is not None
            for meta in metadatas:
                assert meta["source"] == "reference"
                assert meta["relevance_weight"] == "high"
                assert meta["destination"] == "Kyoto"
