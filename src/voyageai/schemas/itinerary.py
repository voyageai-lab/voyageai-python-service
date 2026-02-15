"""
Pydantic models for travel itinerary structured output.

These models are used for TWO purposes:
1. Schema definition for OpenAI strict mode (ai_service.py — standalone path)
2. Post-generation validation for prompt-guided JSON (agent_service.py — agent path)

For purpose #2, constraints are deliberately relaxed (no strict patterns)
so Pydantic accepts minor LLM output variations (e.g., "act-day1-1" vs "act-day1-001").
The prompt guides the format; Pydantic validates the structure.
"""

from pydantic import BaseModel, Field


class Location(BaseModel):
    """Geographic location information."""

    model_config = {"extra": "ignore"}

    name: str = Field(..., description="Location name")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    address: str | None = Field(default=None, description="Full address")
    place_type: str | None = Field(default=None, description="Type of place (restaurant, museum, etc.)")


class Activity(BaseModel):
    """Single activity in the itinerary."""

    model_config = {"extra": "ignore"}

    activity_id: str = Field(
        ...,
        description="Unique activity ID in format 'act-day1-001'",
    )
    time: str = Field(
        ...,
        description="Time range in format '09:00-11:00'",
    )
    title: str = Field(..., description="Activity title")
    description: str = Field(..., description="Detailed description")
    location: Location
    estimated_cost: str | None = Field(default=None, description="Cost estimate (e.g., '$20', 'Free')")
    notes: list[str] = Field(default_factory=list, description="Additional tips or notes")
    # Some LLMs add extra fields like "type", "tips" — model_config extra=ignore handles that


class DailyItinerary(BaseModel):
    """One day's schedule."""

    model_config = {"extra": "ignore"}

    day_number: int = Field(..., ge=1, le=30, description="Day number starting from 1")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    theme: str = Field(..., description="Theme for the day")
    activities: list[Activity] = Field(..., min_length=1, description="List of activities")
    summary: str | None = Field(default=None, description="Optional day summary")


class ItineraryMetadata(BaseModel):
    """Metadata about the trip."""

    model_config = {"extra": "ignore"}

    destination: str = Field(..., description="Main destination")
    start_date: str = Field(..., description="Trip start date")
    end_date: str = Field(..., description="Trip end date")
    total_days: int = Field(..., ge=1, le=30, description="Total number of days")
    budget: str = Field(..., description="Budget level (e.g., 'Medium ($100-200/day)')")
    interests: list[str] = Field(default_factory=list, description="User interests")


class StructuredItinerary(BaseModel):
    """Complete structured itinerary."""

    model_config = {"extra": "ignore"}

    metadata: ItineraryMetadata
    days: list[DailyItinerary] = Field(..., min_length=1)
    tips: list[str] = Field(default_factory=list, description="General travel tips")

