"""
Gmail Trip Detection Router.

Scans a user's Gmail for travel booking confirmations (hotels, flights, tickets)
and returns structured booking information that the frontend can display as
trip suggestions.

Flow:
  1. Java backend sends the user's Gmail access token
  2. This endpoint searches Gmail for travel-related emails
  3. Emails are parsed for booking type, dates, locations, confirmation numbers
  4. Structured results are returned to the Java backend / frontend
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from voyageai.mcp.gmail_server import (
    extract_travel_bookings,
    get_email_detail,
    search_travel_emails,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class GmailScanRequest(BaseModel):
    """Request to scan Gmail for travel bookings."""

    gmail_access_token: str = Field(..., description="OAuth2 access token with gmail.readonly scope")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum emails to scan")
    query: str | None = Field(default=None, description="Custom Gmail search query (uses travel defaults if omitted)")
    include_body: bool = Field(default=False, description="Fetch full email bodies (slower but more accurate)")


class DetectedBooking(BaseModel):
    """A detected travel booking from email."""

    email_id: str
    subject: str
    email_date: str
    type: str  # hotel, flight, ticket, other
    dates: list[str] = Field(default_factory=list)
    confirmation_number: str | None = None
    location_hints: list[str] = Field(default_factory=list)


class GmailScanResponse(BaseModel):
    """Response from Gmail trip detection."""

    bookings: list[DetectedBooking]
    emails_scanned: int
    message: str


@router.post("/gmail/scan-trips", response_model=GmailScanResponse)
async def scan_gmail_trips(request: GmailScanRequest) -> GmailScanResponse:
    """
    Scan user's Gmail for travel booking confirmations.

    Searches for hotel reservations, flight bookings, and event tickets.
    Returns structured booking data that can be matched against existing
    travel projects or used to suggest new ones.

    Called by the Java backend, which provides the user's stored Gmail
    access token (obtained during OAuth2 login with gmail.readonly scope).
    """
    try:
        emails = await search_travel_emails(
            access_token=request.gmail_access_token,
            query=request.query,
            max_results=request.max_results,
        )
    except Exception as e:
        logger.error("Gmail search failed: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to search Gmail: {e}",
        )

    if not emails:
        return GmailScanResponse(
            bookings=[],
            emails_scanned=0,
            message="No travel-related emails found",
        )

    if request.include_body:
        detailed_emails: list[dict[str, Any]] = []
        for email in emails:
            try:
                detail = await get_email_detail(
                    access_token=request.gmail_access_token,
                    message_id=email["id"],
                )
                detailed_emails.append(detail)
            except Exception as e:
                logger.warning("Failed to fetch email %s: %s", email["id"], e)
                detailed_emails.append(email)
        emails = detailed_emails

    bookings_raw = extract_travel_bookings(emails)

    bookings = [
        DetectedBooking(
            email_id=b["email_id"],
            subject=b.get("subject", ""),
            email_date=b.get("email_date", ""),
            type=b.get("type", "other"),
            dates=b.get("dates", []),
            confirmation_number=b.get("confirmation_number"),
            location_hints=b.get("location_hints", []),
        )
        for b in bookings_raw
    ]

    travel_count = sum(1 for b in bookings if b.type != "other")
    logger.info(
        "Gmail scan complete: %d emails, %d travel bookings detected",
        len(emails),
        travel_count,
    )

    return GmailScanResponse(
        bookings=bookings,
        emails_scanned=len(emails),
        message=f"Found {travel_count} travel booking(s) from {len(emails)} email(s)",
    )
