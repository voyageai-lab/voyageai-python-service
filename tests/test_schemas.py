"""Tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from voyageai.schemas.itinerary import (
    Activity,
    DailyItinerary,
    ItineraryMetadata,
    Location,
    StructuredItinerary,
)
from voyageai.schemas.task import GenerateRequest, GenerateResponse, TaskStatus


class TestLocation:
    """Tests for Location schema."""

    def test_valid_location(self):
        """Test creating a valid location."""
        loc = Location(
            name="Tokyo Tower",
            latitude=35.6586,
            longitude=139.7454,
            address="4-2-8 Shiba-koen, Minato City",
        )
        assert loc.name == "Tokyo Tower"
        assert loc.latitude == 35.6586

    def test_latitude_accepts_any_float(self):
        """Test that latitude accepts any float (validation relaxed for LLM output)."""
        loc = Location(name="Test", latitude=91.0, longitude=0)
        assert loc.latitude == 91.0

    def test_extra_fields_ignored(self):
        """Test that extra fields from LLM output are silently ignored."""
        loc = Location(
            name="Test",
            latitude=35.0,
            longitude=139.0,
            extra_field="should be ignored",  # type: ignore[call-arg]
        )
        assert loc.name == "Test"
        assert not hasattr(loc, "extra_field")


class TestActivity:
    """Tests for Activity schema."""

    @pytest.fixture
    def valid_location(self):
        return Location(name="Test Place", latitude=35.0, longitude=139.0)

    def test_valid_activity(self, valid_location):
        """Test creating a valid activity."""
        activity = Activity(
            activity_id="act-day1-001",
            time="09:00-11:00",
            title="Morning Visit",
            description="A nice morning activity",
            location=valid_location,
        )
        assert activity.activity_id == "act-day1-001"

    def test_flexible_activity_id(self, valid_location):
        """Test that activity_id accepts any string (relaxed for LLM output)."""
        activity = Activity(
            activity_id="day1-activity-1",  # Not strict format, but accepted
            time="09:00-11:00",
            title="Test",
            description="Test",
            location=valid_location,
        )
        assert activity.activity_id == "day1-activity-1"

    def test_flexible_time_format(self, valid_location):
        """Test that time accepts any string format (relaxed for LLM output)."""
        activity = Activity(
            activity_id="act-day1-001",
            time="9:00 AM - 11:00 AM",  # Non-strict format, but accepted
            title="Test",
            description="Test",
            location=valid_location,
        )
        assert activity.time == "9:00 AM - 11:00 AM"

    def test_extra_fields_ignored(self, valid_location):
        """Test that extra fields from LLM output are silently ignored."""
        activity = Activity(
            activity_id="act-day1-001",
            time="09:00-11:00",
            title="Test",
            description="Test",
            location=valid_location,
            type="sightseeing",  # type: ignore[call-arg]
            duration="2 hours",  # type: ignore[call-arg]
        )
        assert activity.title == "Test"


class TestDailyItinerary:
    """Tests for DailyItinerary schema."""

    @pytest.fixture
    def valid_activity(self):
        return Activity(
            activity_id="act-day1-001",
            time="09:00-11:00",
            title="Morning Visit",
            description="A nice morning activity",
            location=Location(name="Test", latitude=0, longitude=0),
        )

    def test_valid_daily_itinerary(self, valid_activity):
        """Test creating a valid daily itinerary."""
        day = DailyItinerary(
            day_number=1,
            date="2026-03-15",
            theme="Exploration",
            activities=[valid_activity],
        )
        assert day.day_number == 1
        assert len(day.activities) == 1

    def test_flexible_date_format(self, valid_activity):
        """Test that date accepts any string format (relaxed for LLM output)."""
        day = DailyItinerary(
            day_number=1,
            date="March 15, 2026",  # Non-ISO format, but accepted
            theme="Exploration",
            activities=[valid_activity],
        )
        assert day.date == "March 15, 2026"

    def test_empty_activities(self):
        """Test that empty activities list raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DailyItinerary(
                day_number=1,
                date="2026-03-15",
                theme="Exploration",
                activities=[],  # Empty
            )
        assert "activities" in str(exc_info.value)


class TestGenerateRequest:
    """Tests for GenerateRequest schema."""

    def test_valid_request(self):
        """Test creating a valid generate request."""
        req = GenerateRequest(
            task_id="task-001",
            user_id="user-001",
            project_id="proj-001",
            requirements="3 day trip to Tokyo",
        )
        assert req.task_id == "task-001"

    def test_requirements_max_length(self):
        """Test that requirements over max length raises error."""
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(
                task_id="task-001",
                user_id="user-001",
                project_id="proj-001",
                requirements="x" * 2001,  # Over 2000 chars
            )
        assert "requirements" in str(exc_info.value)


class TestGenerateResponse:
    """Tests for GenerateResponse schema."""

    def test_completed_response(self):
        """Test creating a completed response."""
        resp = GenerateResponse(
            task_id="task-001",
            status=TaskStatus.COMPLETED,
            processing_time_ms=1500,
        )
        assert resp.status == TaskStatus.COMPLETED

    def test_failed_response_with_error(self):
        """Test creating a failed response with error message."""
        resp = GenerateResponse(
            task_id="task-001",
            status=TaskStatus.FAILED,
            error="API timeout",
            processing_time_ms=60000,
        )
        assert resp.status == TaskStatus.FAILED
        assert resp.error == "API timeout"


