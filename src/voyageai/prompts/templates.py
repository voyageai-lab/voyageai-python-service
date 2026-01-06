"""Prompt templates for travel itinerary generation."""

TRAVEL_ITINERARY_SYSTEM_PROMPT = """You are an expert travel planner with extensive knowledge of destinations worldwide.
Generate detailed, practical travel itineraries based on user requirements.

Guidelines:
- Create realistic schedules with appropriate travel times between locations
- Include a mix of popular attractions and local experiences
- Consider meal times and rest periods
- Provide accurate GPS coordinates for all locations
- Include cost estimates in local currency
- Add practical tips specific to each activity
- Balance indoor and outdoor activities based on typical weather

Output must be valid JSON matching the provided schema exactly."""

TRAVEL_ITINERARY_USER_PROMPT = """Create a detailed travel itinerary based on these requirements:

{requirements}

Additional context:
- Current date for reference: {current_date}
- Response language: English
- Include accurate GPS coordinates for all locations
- Ensure activity IDs follow the pattern: act-day1-001, act-day1-002, etc.
- Time format should be 24-hour (e.g., 09:00-11:00)
- Dates should be in YYYY-MM-DD format"""

# Example for few-shot prompting and schema reference
ITINERARY_EXAMPLE = {
    "metadata": {
        "destination": "Tokyo, Japan",
        "start_date": "2026-03-15",
        "end_date": "2026-03-17",
        "total_days": 3,
        "budget": "Medium ($100-200/day)",
        "interests": ["culture", "food", "technology"],
    },
    "days": [
        {
            "day_number": 1,
            "date": "2026-03-15",
            "theme": "Traditional Tokyo",
            "activities": [
                {
                    "activity_id": "act-day1-001",
                    "time": "09:00-11:00",
                    "title": "Senso-ji Temple Visit",
                    "description": "Explore Tokyo's oldest and most famous Buddhist temple. "
                    "Walk through the iconic Kaminarimon gate and browse the traditional "
                    "Nakamise shopping street.",
                    "location": {
                        "name": "Senso-ji Temple",
                        "latitude": 35.7148,
                        "longitude": 139.7967,
                        "address": "2-3-1 Asakusa, Taito City, Tokyo",
                        "place_type": "temple",
                    },
                    "estimated_cost": "Free",
                    "notes": [
                        "Arrive early to avoid crowds",
                        "Try fortune slips (omikuji) for ¥100",
                        "Best photo spot: in front of the main hall",
                    ],
                },
                {
                    "activity_id": "act-day1-002",
                    "time": "11:30-13:00",
                    "title": "Traditional Japanese Lunch",
                    "description": "Enjoy authentic tempura and soba noodles at a local restaurant "
                    "in the Asakusa area.",
                    "location": {
                        "name": "Daikokuya Tempura",
                        "latitude": 35.7122,
                        "longitude": 139.7946,
                        "address": "1-38-10 Asakusa, Taito City, Tokyo",
                        "place_type": "restaurant",
                    },
                    "estimated_cost": "¥1,500-2,500",
                    "notes": [
                        "Cash only",
                        "Expect a short wait during lunch rush",
                    ],
                },
            ],
        },
    ],
    "tips": [
        "Get a Suica or Pasmo card for easy transportation",
        "Download Google Maps offline for navigation",
        "Carry cash as many small shops don't accept cards",
    ],
    "tool_trace": [],
}

