"""Pydantic models for travel itinerary structured output."""

from pydantic import BaseModel, Field


class Location(BaseModel):
    """Geographic location information."""

    name: str = Field(..., description="Location name")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    address: str | None = Field(default=None, description="Full address")
    place_type: str | None = Field(default=None, description="Type of place (restaurant, museum, etc.)")


class Activity(BaseModel):
    """Single activity in the itinerary."""

    activity_id: str = Field(
        ...,
        pattern=r"^act-day\d+-\d{3}$",
        description="Unique activity ID in format 'act-day1-001'",
    )
    time: str = Field(
        ...,
        pattern=r"^\d{2}:\d{2}-\d{2}:\d{2}$",
        description="Time range in format '09:00-11:00'",
    )
    title: str = Field(..., max_length=100, description="Activity title")
    description: str = Field(..., max_length=500, description="Detailed description")
    location: Location
    estimated_cost: str | None = Field(default=None, description="Cost estimate (e.g., '$20', 'Free')")
    notes: list[str] = Field(default_factory=list, description="Additional tips or notes")


class DailyItinerary(BaseModel):
    """One day's schedule."""

    day_number: int = Field(..., ge=1, le=30, description="Day number starting from 1")
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date in YYYY-MM-DD format")
    theme: str = Field(..., max_length=50, description="Theme for the day")
    activities: list[Activity] = Field(..., min_length=1, description="List of activities")


class ItineraryMetadata(BaseModel):
    """Metadata about the trip."""

    destination: str = Field(..., description="Main destination")
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Trip start date")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Trip end date")
    total_days: int = Field(..., ge=1, le=30, description="Total number of days")
    budget: str = Field(..., description="Budget level (e.g., 'Medium ($100-200/day)')")
    interests: list[str] = Field(default_factory=list, description="User interests")


class StructuredItinerary(BaseModel):
    """Complete structured itinerary - target schema for OpenAI structured output."""

    metadata: ItineraryMetadata
    days: list[DailyItinerary] = Field(..., min_length=1)
    tips: list[str] = Field(default_factory=list, description="General travel tips")
    tool_trace: list[dict] = Field(
        default_factory=list,
        description="Track of tool calls made during generation (Module 8)",
    )

