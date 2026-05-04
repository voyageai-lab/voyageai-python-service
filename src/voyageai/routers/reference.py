"""
Reference Itinerary Import Router

Provides an endpoint to import external travel content (blog posts, articles,
raw text) into the RAG system as reference itineraries. Reference documents
are stored in ChromaDB with elevated relevance metadata so the agent can
prioritize them when generating itineraries.

Workflow:
    1. Client sends a URL or raw text
    2. If URL, content is fetched via httpx
    3. Content is chunked and stored in ChromaDB with source=reference
    4. Document ID is returned for tracking
"""

import hashlib
import logging
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from voyageai.rag.chroma import chroma_client

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_CONTENT_LENGTH = 50_000
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


class ReferenceImportRequest(BaseModel):
    """Request to import a reference itinerary into RAG."""

    url: str | None = Field(default=None, description="URL to fetch content from (blog post, travel article)")
    raw_text: str | None = Field(default=None, description="Raw text content to import directly")
    title: str | None = Field(default=None, description="Optional title for the reference")
    destination: str | None = Field(default=None, description="Destination the reference covers")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering (e.g., 'food', 'budget')")


class ReferenceImportResponse(BaseModel):
    """Response after importing a reference itinerary."""

    document_id: str
    chunks_stored: int
    source: str
    title: str | None
    message: str


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks for better embedding coverage.

    Uses paragraph boundaries when possible to avoid mid-sentence splits.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def _fetch_url_content(url: str) -> str:
    """Fetch and extract text content from a URL."""
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": "VoyageAI/1.0 ReferenceImport"})
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    raw = response.text

    if "text/html" in content_type:
        raw = _strip_html(raw)

    if len(raw) > MAX_CONTENT_LENGTH:
        raw = raw[:MAX_CONTENT_LENGTH]

    return raw.strip()


def _strip_html(html: str) -> str:
    """Naive HTML tag stripping for plain-text extraction."""
    import re

    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@router.post("/reference-itinerary", response_model=ReferenceImportResponse)
async def import_reference_itinerary(request: ReferenceImportRequest) -> ReferenceImportResponse:
    """
    Import a reference itinerary (URL or raw text) into the RAG system.

    The content is chunked and stored in ChromaDB with metadata marking it
    as a reference document with high relevance weight so the agent
    prioritizes it during itinerary generation.

    Example:
        POST /api/v1/reference-itinerary
        {
            "url": "https://example.com/tokyo-travel-blog",
            "title": "Hidden Gems in Tokyo",
            "destination": "Tokyo",
            "tags": ["food", "culture"]
        }
    """
    if not request.url and not request.raw_text:
        raise HTTPException(status_code=422, detail="Provide either 'url' or 'raw_text'")

    source = "url"
    content: str

    if request.url:
        try:
            content = await _fetch_url_content(request.url)
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch URL (HTTP {e.response.status_code}): {request.url}",
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")
    else:
        content = request.raw_text  # type: ignore[assignment]
        source = "raw_text"

    if not content or len(content.strip()) < 50:
        raise HTTPException(status_code=422, detail="Content too short (minimum 50 characters)")

    doc_id = str(uuid.uuid4())
    chunks = _chunk_text(content)

    chunk_ids = [
        hashlib.md5(f"{doc_id}-{i}".encode()).hexdigest()[:16]
        for i in range(len(chunks))
    ]

    metadatas: list[dict[str, Any]] = [
        {
            "document_id": doc_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "source": "reference",
            "source_type": source,
            "source_url": request.url or "",
            "title": request.title or "",
            "destination": request.destination or "",
            "tags": ",".join(request.tags),
            "relevance_weight": "high",
        }
        for i in range(len(chunks))
    ]

    try:
        await chroma_client.initialize()
        await chroma_client.add_documents(
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids,
        )
    except Exception as e:
        logger.error(f"Failed to store reference in ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store reference: {e}")

    logger.info(
        f"Imported reference itinerary doc_id={doc_id} "
        f"chunks={len(chunks)} source={source} title={request.title}"
    )

    return ReferenceImportResponse(
        document_id=doc_id,
        chunks_stored=len(chunks),
        source=source,
        title=request.title,
        message=f"Successfully imported {len(chunks)} chunk(s) into RAG",
    )
