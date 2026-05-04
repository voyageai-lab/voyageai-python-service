"""
Gmail MCP Server — reads travel booking emails via Gmail API.

Exposes tools:
  - search_travel_emails: Search Gmail for hotel, flight, and ticket confirmations
  - get_email_detail: Retrieve full email content by message ID
  - extract_travel_bookings: Parse structured booking info from email content

Security:
  Gmail access requires a valid OAuth2 access token, passed per-request.
  This server does NOT store tokens — the Java backend stores them
  and forwards them to the Python service for each scan request.

Architecture:
  Gmail API → parse emails → extract dates/locations/booking IDs → return structured data
"""

import base64
import json
import logging
import re
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"

TRAVEL_SEARCH_QUERIES = [
    "subject:(booking confirmation OR reservation OR itinerary OR e-ticket)",
    "from:(booking.com OR hotels.com OR airbnb.com OR expedia.com OR agoda.com)",
    "from:(airline OR airways OR airlines OR united OR delta OR american)",
    "subject:(flight confirmation OR boarding pass OR check-in)",
    "subject:(hotel reservation OR hotel confirmation)",
    "subject:(ticket OR admission OR entry) subject:(museum OR park OR tour)",
]


async def search_travel_emails(
    access_token: str,
    query: str | None = None,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search Gmail for travel-related emails.

    Args:
        access_token: OAuth2 access token with gmail.readonly scope.
        query: Optional custom Gmail search query. If None, uses travel-related defaults.
        max_results: Maximum number of emails to return.

    Returns:
        List of email summaries with id, subject, sender, date, and snippet.
    """
    if not query:
        query = " OR ".join(f"({q})" for q in TRAVEL_SEARCH_QUERIES)
        query += " newer_than:90d"

    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{GMAIL_API_BASE}/messages",
            headers=headers,
            params={"q": query, "maxResults": max_results},
        )
        resp.raise_for_status()
        data = resp.json()

    messages = data.get("messages", [])
    if not messages:
        return []

    results = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for msg in messages[:max_results]:
            detail = await _fetch_message_metadata(client, headers, msg["id"])
            if detail:
                results.append(detail)

    return results


async def _fetch_message_metadata(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    message_id: str,
) -> dict[str, Any] | None:
    """Fetch minimal metadata for a single message."""
    try:
        resp = await client.get(
            f"{GMAIL_API_BASE}/messages/{message_id}",
            headers=headers,
            params={"format": "metadata", "metadataHeaders": "Subject,From,Date"},
        )
        resp.raise_for_status()
        msg = resp.json()

        header_map: dict[str, str] = {}
        for h in msg.get("payload", {}).get("headers", []):
            header_map[h["name"].lower()] = h["value"]

        return {
            "id": message_id,
            "subject": header_map.get("subject", ""),
            "from": header_map.get("from", ""),
            "date": header_map.get("date", ""),
            "snippet": msg.get("snippet", ""),
        }
    except Exception as e:
        logger.warning("Failed to fetch message %s: %s", message_id, e)
        return None


async def get_email_detail(
    access_token: str,
    message_id: str,
) -> dict[str, Any]:
    """Get the full text content of an email.

    Args:
        access_token: OAuth2 access token.
        message_id: Gmail message ID.

    Returns:
        Dict with subject, from, date, and plain-text body.
    """
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{GMAIL_API_BASE}/messages/{message_id}",
            headers=headers,
            params={"format": "full"},
        )
        resp.raise_for_status()
        msg = resp.json()

    header_map: dict[str, str] = {}
    for h in msg.get("payload", {}).get("headers", []):
        header_map[h["name"].lower()] = h["value"]

    body_text = _extract_body_text(msg.get("payload", {}))

    return {
        "id": message_id,
        "subject": header_map.get("subject", ""),
        "from": header_map.get("from", ""),
        "date": header_map.get("date", ""),
        "body": body_text[:5000],
    }


def _extract_body_text(payload: dict[str, Any]) -> str:
    """Recursively extract plain-text body from Gmail message payload."""
    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        nested = _extract_body_text(part)
        if nested:
            return nested

    if mime == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            html = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            return _strip_html(html)

    return ""


def _strip_html(html: str) -> str:
    """Naive HTML tag stripping."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_travel_bookings(emails: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse travel booking information from email summaries.

    Extracts structured data: booking type, dates, location, confirmation number.

    Args:
        emails: List of email dicts (from search_travel_emails or get_email_detail).

    Returns:
        List of detected bookings with type, dates, location, and source email ID.
    """
    bookings: list[dict[str, Any]] = []

    for email in emails:
        text = f"{email.get('subject', '')} {email.get('snippet', '')} {email.get('body', '')}"
        text_lower = text.lower()

        booking: dict[str, Any] = {
            "email_id": email["id"],
            "subject": email.get("subject", ""),
            "email_date": email.get("date", ""),
        }

        if any(kw in text_lower for kw in ["hotel", "reservation", "check-in", "check-out", "accommodation", "hostel", "airbnb"]):
            booking["type"] = "hotel"
        elif any(kw in text_lower for kw in ["flight", "boarding", "airline", "airways", "departure", "arrival"]):
            booking["type"] = "flight"
        elif any(kw in text_lower for kw in ["ticket", "admission", "entry", "museum", "tour", "attraction"]):
            booking["type"] = "ticket"
        else:
            booking["type"] = "other"

        dates = _extract_dates(text)
        if dates:
            booking["dates"] = dates

        confirmation = _extract_confirmation_number(text)
        if confirmation:
            booking["confirmation_number"] = confirmation

        location = _extract_location_hints(text)
        if location:
            booking["location_hints"] = location

        bookings.append(booking)

    return bookings


def _extract_dates(text: str) -> list[str]:
    """Extract date-like strings from text."""
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b",
    ]
    dates: list[str] = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    return list(dict.fromkeys(dates))[:6]


def _extract_confirmation_number(text: str) -> str | None:
    """Extract booking confirmation numbers."""
    patterns = [
        r"(?:confirmation|booking|reference|reservation)\s*(?:#|number|no\.?|code)\s*[:.]?\s*([A-Z0-9]{5,12})",
        r"#([A-Z0-9]{5,12})\b",
        r"(?:PNR|Record Locator)\s*[:.]?\s*([A-Z0-9]{5,8})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_location_hints(text: str) -> list[str]:
    """Extract potential location names from booking text."""
    patterns = [
        r"(?:[Hh]otel|[Ss]tay(?:ing)?|[Aa]ccommodation)\s+(?:in|at|near)\s+([A-Za-z][a-zA-Z\s]{2,25})",
        r"(?:to|from|departing|arriving)\s+([A-Z][a-zA-Z\s]{2,25}?)(?:\s+on|\s+at|\s*\()",
        r"(?:located|address)\s*[:.]?\s*(.{10,60}?)(?:\.|,\s*[A-Z]|\n)",
    ]
    hints: list[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        hints.extend(m.strip() for m in matches if m.strip())
    return list(dict.fromkeys(hints))[:5]
