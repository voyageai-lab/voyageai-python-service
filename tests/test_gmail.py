"""
Unit tests for Gmail trip detection.

Tests cover:
1. Email search with mocked Gmail API
2. Email body extraction (plain text and HTML)
3. Travel booking extraction (hotel, flight, ticket)
4. Date extraction from various formats
5. Confirmation number extraction
6. Location hint extraction
7. Gmail scan endpoint (FastAPI)
"""

import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from voyageai.main import app
from voyageai.mcp.gmail_server import (
    _extract_body_text,
    _extract_confirmation_number,
    _extract_dates,
    _extract_location_hints,
    extract_travel_bookings,
)


# ============================================================================
# Text Extraction Tests
# ============================================================================


class TestExtractDates:
    """Tests for date extraction from email text."""

    def test_iso_date(self):
        dates = _extract_dates("Check-in: 2026-05-01, Check-out: 2026-05-05")
        assert "2026-05-01" in dates
        assert "2026-05-05" in dates

    def test_us_date_format(self):
        dates = _extract_dates("Departure: 5/1/2026")
        assert "5/1/2026" in dates

    def test_named_month(self):
        dates = _extract_dates("Arrival: March 15, 2026")
        assert any("March" in d for d in dates)

    def test_no_dates(self):
        dates = _extract_dates("No date information here")
        assert dates == []


class TestExtractConfirmation:
    """Tests for confirmation number extraction."""

    def test_booking_confirmation(self):
        text = "Booking Confirmation #ABC12345"
        result = _extract_confirmation_number(text)
        assert result == "ABC12345"

    def test_pnr(self):
        text = "PNR: XYZABC"
        result = _extract_confirmation_number(text)
        assert result == "XYZABC"

    def test_no_confirmation(self):
        result = _extract_confirmation_number("Just a normal email.")
        assert result is None


class TestExtractLocationHints:
    """Tests for location extraction from booking text."""

    def test_hotel_location(self):
        text = "Hotel in Tokyo Station area"
        hints = _extract_location_hints(text)
        assert any("Tokyo" in h for h in hints)

    def test_flight_destination(self):
        text = "Flight to Paris (CDG) on March 15"
        hints = _extract_location_hints(text)
        assert any("Paris" in h for h in hints)


class TestExtractBodyText:
    """Tests for Gmail message body extraction."""

    def test_plain_text_body(self):
        payload = {
            "mimeType": "text/plain",
            "body": {"data": base64.urlsafe_b64encode(b"Hello, this is a booking.").decode()},
        }
        assert "Hello, this is a booking" in _extract_body_text(payload)

    def test_multipart_with_text(self):
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": base64.urlsafe_b64encode(b"Booking confirmed").decode()},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": base64.urlsafe_b64encode(b"<p>Booking confirmed</p>").decode()},
                },
            ],
        }
        assert "Booking confirmed" in _extract_body_text(payload)

    def test_html_fallback(self):
        payload = {
            "mimeType": "text/html",
            "body": {"data": base64.urlsafe_b64encode(b"<p>Your <b>hotel</b> is booked</p>").decode()},
        }
        result = _extract_body_text(payload)
        assert "hotel" in result
        assert "<p>" not in result


# ============================================================================
# Booking Classification Tests
# ============================================================================


class TestExtractTravelBookings:
    """Tests for travel booking type classification."""

    def test_hotel_detection(self):
        emails = [{"id": "1", "subject": "Hotel Reservation Confirmed", "snippet": "Check-in May 1, check-out May 5"}]
        bookings = extract_travel_bookings(emails)
        assert bookings[0]["type"] == "hotel"

    def test_flight_detection(self):
        emails = [{"id": "2", "subject": "Your Flight Itinerary", "snippet": "Boarding pass for Tokyo"}]
        bookings = extract_travel_bookings(emails)
        assert bookings[0]["type"] == "flight"

    def test_ticket_detection(self):
        emails = [{"id": "3", "subject": "Museum Admission Ticket", "snippet": "Entry to National Museum"}]
        bookings = extract_travel_bookings(emails)
        assert bookings[0]["type"] == "ticket"

    def test_other_detection(self):
        emails = [{"id": "4", "subject": "Thank you for your purchase", "snippet": "Order shipped"}]
        bookings = extract_travel_bookings(emails)
        assert bookings[0]["type"] == "other"

    def test_date_extraction_in_booking(self):
        emails = [{"id": "5", "subject": "Hotel Reservation 2026-05-01", "snippet": "Check-out 2026-05-05"}]
        bookings = extract_travel_bookings(emails)
        assert len(bookings[0].get("dates", [])) >= 1


# ============================================================================
# Endpoint Tests
# ============================================================================


@pytest.fixture
async def test_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestGmailScanEndpoint:
    """Tests for POST /api/v1/gmail/scan-trips."""

    @pytest.mark.asyncio
    async def test_scan_returns_bookings(self, test_client):
        mock_emails = [
            {"id": "msg1", "subject": "Hotel Booking at Hilton Tokyo", "from": "booking@hilton.com", "date": "2026-03-01", "snippet": "Check-in 2026-05-01"},
            {"id": "msg2", "subject": "Flight Confirmation", "from": "noreply@airline.com", "date": "2026-03-01", "snippet": "Flight to Tokyo departing May 1"},
        ]

        with patch("voyageai.routers.gmail.search_travel_emails", new_callable=AsyncMock, return_value=mock_emails):
            response = await test_client.post(
                "/api/v1/gmail/scan-trips",
                json={"gmail_access_token": "fake_token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["emails_scanned"] == 2
        assert len(data["bookings"]) == 2
        types = [b["type"] for b in data["bookings"]]
        assert "hotel" in types
        assert "flight" in types

    @pytest.mark.asyncio
    async def test_scan_no_emails(self, test_client):
        with patch("voyageai.routers.gmail.search_travel_emails", new_callable=AsyncMock, return_value=[]):
            response = await test_client.post(
                "/api/v1/gmail/scan-trips",
                json={"gmail_access_token": "fake_token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["emails_scanned"] == 0
        assert data["bookings"] == []

    @pytest.mark.asyncio
    async def test_scan_gmail_api_failure(self, test_client):
        with patch("voyageai.routers.gmail.search_travel_emails", new_callable=AsyncMock, side_effect=Exception("Auth expired")):
            response = await test_client.post(
                "/api/v1/gmail/scan-trips",
                json={"gmail_access_token": "expired_token"},
            )

        assert response.status_code == 502
        assert "Failed to search Gmail" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_scan_missing_token(self, test_client):
        response = await test_client.post(
            "/api/v1/gmail/scan-trips",
            json={},
        )
        assert response.status_code == 422
