"""
Pydantic models for travel itinerary structured output.

These models are used for TWO purposes:
1. Schema definition for OpenAI strict mode (ai_service.py — standalone path)
2. Post-generation validation for prompt-guided JSON (agent_service.py — agent path)

For purpose #2, constraints are deliberately relaxed (no strict patterns)
so Pydantic accepts minor LLM output variations (e.g., "act-day1-1" vs "act-day1-001").
The prompt guides the format; Pydantic validates the structure.

Extra fields policy:
  All models use `extra = "allow"` so the AI can freely attach additional
  context-specific fields (e.g., distance_km, transport_mode, rating,
  booking_url, weather, highlights, etc.). The frontend renders known fields
  with dedicated UI and surfaces extras dynamically.
"""

from pydantic import BaseModel, Field


class Location(BaseModel):
    """Geographic location information."""

    model_config = {"extra": "allow"}

    name: str = Field(..., description="Location name")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    address: str | None = Field(default=None, description="Full address")
    place_type: str | None = Field(default=None, description="Type of place (restaurant, museum, etc.)")


class SourceLink(BaseModel):
    """A reference link associated with an activity (official site, social media, etc.)."""

    model_config = {"extra": "allow"}

    title: str = Field(..., description="Link title (e.g., 'Official Website', '小红书: 东京美食攻略')")
    url: str = Field(..., description="Full URL")
    source: str = Field(
        ...,
        description="Source type: 'official', 'xiaohongshu', 'foursquare', 'google_maps', 'web_search'",
    )
    snippet: str | None = Field(default=None, description="Brief quote or description from the source")


class Activity(BaseModel):
    """Single activity in the itinerary."""

    model_config = {"extra": "allow"}

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
    website_url: str | None = Field(default=None, description="Official website URL for this activity/place")
    source_links: list[SourceLink] = Field(
        default_factory=list,
        description="Reference links from various sources (official sites, Xiaohongshu, Foursquare, etc.)",
    )


class DailyItinerary(BaseModel):
    """One day's schedule."""

    model_config = {"extra": "allow"}

    day_number: int = Field(..., ge=1, le=30, description="Day number starting from 1")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    theme: str = Field(..., description="Theme for the day")
    activities: list[Activity] = Field(..., min_length=1, description="List of activities")
    summary: str | None = Field(default=None, description="Optional day summary")


class ItineraryMetadata(BaseModel):
    """Metadata about the trip."""

    model_config = {"extra": "allow"}

    destination: str = Field(..., description="Main destination")
    start_date: str = Field(..., description="Trip start date")
    end_date: str = Field(..., description="Trip end date")
    total_days: int = Field(..., ge=1, le=30, description="Total number of days")
    budget: str = Field(..., description="Budget level (e.g., 'Medium ($100-200/day)')")
    interests: list[str] = Field(default_factory=list, description="User interests")


class StructuredItinerary(BaseModel):
    """Complete structured itinerary."""

    model_config = {"extra": "allow"}

    metadata: ItineraryMetadata
    days: list[DailyItinerary] = Field(..., min_length=1)
    tips: list[str] = Field(default_factory=list, description="General travel tips")
