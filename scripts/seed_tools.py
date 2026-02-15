#!/usr/bin/env python3
"""
Seed Tool Metadata to ChromaDB for Tool-RAG

This script populates the tool_metadata collection with detailed metadata
for each available tool. The example_queries are crucial for semantic
tool selection - they should cover diverse phrasings of when to use each tool.

Usage:
    # From project root with venv activated
    python scripts/seed_tools.py
    
    # Or run directly
    ./scripts/seed_tools.py

The script will:
1. Initialize the ToolRAG collection
2. Clear any existing tool metadata
3. Add all tool definitions with example queries
4. Verify the seeding was successful
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voyageai.schemas.tool_metadata import ToolMetadata
from voyageai.rag.tool_rag import ToolRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Tool metadata definitions with comprehensive example queries
TOOL_DEFINITIONS: list[ToolMetadata] = [
    # =========================================================================
    # Geocode Tool
    # =========================================================================
    ToolMetadata(
        name="geocode_location",
        description=(
            "Convert a location name (city, address, or landmark) to geographic coordinates "
            "(latitude and longitude). Use this before calling weather or distance tools."
        ),
        category="info",
        parameters_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Name of the location to geocode (e.g., 'Tokyo, Japan', 'Eiffel Tower')"
                }
            },
            "required": ["location"],
            "additionalProperties": False
        },
        example_queries=[
            # Direct geocoding requests
            "Where is Tokyo located?",
            "What are the coordinates of Paris?",
            "Find the latitude and longitude of New York City",
            "Get the GPS coordinates for the Eiffel Tower",
            "What's the location of Sydney Opera House?",
            # Implicit geocoding needs (for other tools)
            "I need to check the weather in London",  # Needs geocoding first
            "How far is it from Tokyo to Osaka?",  # Needs geocoding for both
            "What's the distance to Barcelona from Madrid?",
            # Travel planning that needs location
            "I'm planning a trip to Rome",
            "We want to visit Machu Picchu",
            "Looking for things to do in Bangkok",
            # Address lookup
            "Find the location of Times Square, New York",
            "Where exactly is Shibuya Crossing?",
            "Get coordinates for Central Park",
        ],
        rate_limit_per_minute=60,  # Nominatim allows 1/sec
        timeout_seconds=10,
        requires_api_key=False,
        is_enabled=True,
        priority=9,  # High priority - often needed before other tools
    ),
    
    # =========================================================================
    # Weather Tool
    # =========================================================================
    ToolMetadata(
        name="get_weather_forecast",
        description=(
            "Get weather forecast for a destination. Requires latitude and longitude coordinates. "
            "Returns daily temperature, precipitation chance, and conditions. "
            "Use geocode_location first to get coordinates if you only have a location name."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude coordinate (-90 to 90)"
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude coordinate (-180 to 180)"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date for forecast (YYYY-MM-DD format)"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date for forecast (YYYY-MM-DD format)"
                }
            },
            "required": ["latitude", "longitude", "start_date", "end_date"],
            "additionalProperties": False
        },
        example_queries=[
            # Direct weather requests
            "What's the weather like in Tokyo next week?",
            "Will it rain in Paris during my trip?",
            "Check the weather forecast for London",
            "What will the temperature be in Rome in March?",
            "Is it going to be sunny in Barcelona?",
            # Clothing/packing related
            "Should I pack an umbrella for my trip to Seattle?",
            "Do I need warm clothes for visiting Iceland?",
            "What should I pack for a trip to Hawaii?",
            # Activity planning based on weather
            "Will the weather be good for hiking in Switzerland?",
            "Is it beach weather in Thailand right now?",
            "Best days for outdoor sightseeing in New York",
            # Seasonal questions
            "How hot is Singapore in July?",
            "Is December a good time to visit Australia?",
            "What's the rainy season like in Bali?",
            # UV/sun related
            "How strong is the sun in Morocco?",
            "Do I need sunscreen in Dubai?",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=False,  # Open-Meteo is free
        is_enabled=True,
        priority=8,
    ),
    
    # =========================================================================
    # Currency Tool
    # =========================================================================
    ToolMetadata(
        name="convert_currency",
        description=(
            "Convert an amount from one currency to another. "
            "Use this for budget planning and cost estimation. "
            "Supports major world currencies like USD, EUR, JPY, GBP, CNY, etc."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "from_currency": {
                    "type": "string",
                    "description": "Source currency code (e.g., 'USD', 'EUR', 'JPY')"
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency code (e.g., 'USD', 'EUR', 'JPY')"
                },
                "amount": {
                    "type": "number",
                    "description": "Amount to convert"
                }
            },
            "required": ["from_currency", "to_currency", "amount"],
            "additionalProperties": False
        },
        example_queries=[
            # Direct conversion requests
            "How much is 1000 USD in Japanese Yen?",
            "Convert 500 euros to dollars",
            "What's 100 GBP in EUR?",
            "Exchange rate from USD to CNY",
            # Budget planning
            "I have $2000 budget, how much is that in Thai Baht?",
            "How much spending money do I need in euros for Italy?",
            "What's my budget in local currency for Japan?",
            # Price comparison
            "Is $50 a lot for a meal in Tokyo?",
            "How much is 10000 yen in my currency?",
            "Convert hotel price from euros to dollars",
            # Travel money
            "How much cash should I exchange for my trip to Mexico?",
            "What's the current rate for Singapore dollars?",
            "Currency conversion for Korean won",
            # Cost estimation
            "How expensive is the UK compared to US?",
            "Budget conversion for European trip",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=False,  # Frankfurter is free
        is_enabled=True,
        priority=7,
    ),
    
    # =========================================================================
    # Timezone Tool
    # =========================================================================
    ToolMetadata(
        name="convert_timezone",
        description=(
            "Convert a time from one timezone to another. "
            "Useful for flight planning and scheduling. "
            "Accepts IANA timezone names (e.g., 'Asia/Tokyo') or common abbreviations (e.g., 'JST', 'PST')."
        ),
        category="local_computation",
        parameters_schema={
            "type": "object",
            "properties": {
                "time": {
                    "type": "string",
                    "description": "Time to convert (ISO format like '2026-01-10T14:00:00' or 'HH:MM')"
                },
                "from_timezone": {
                    "type": "string",
                    "description": "Source timezone (e.g., 'America/New_York', 'PST', 'Tokyo')"
                },
                "to_timezone": {
                    "type": "string",
                    "description": "Target timezone (e.g., 'Asia/Tokyo', 'JST', 'London')"
                },
                "date": {
                    "type": "string",
                    "description": "Optional date if time is just HH:MM (format: YYYY-MM-DD)"
                }
            },
            "required": ["time", "from_timezone", "to_timezone"],
            "additionalProperties": False
        },
        example_queries=[
            # Flight time planning
            "If I leave NYC at 6pm, what time do I arrive in Tokyo?",
            "My flight lands at 2pm Tokyo time, what time is that in LA?",
            "Convert flight departure time from local to destination time",
            # Meeting/call scheduling
            "What time is 9am PST in London?",
            "Schedule a call between New York and Singapore",
            "When is noon Tokyo time in EST?",
            # General timezone questions
            "What's the time difference between London and Sydney?",
            "How many hours ahead is Japan from California?",
            "Time zone conversion from Paris to Beijing",
            # Jet lag planning
            "What time will it be when I arrive?",
            "Help me understand the time difference",
            "Convert my schedule to local time",
            # Event timing
            "When does the show start in my timezone?",
            "What time should I wake up for a 6am departure?",
        ],
        rate_limit_per_minute=1000,  # Local computation, no API
        timeout_seconds=5,
        requires_api_key=False,
        is_enabled=True,
        priority=6,
    ),
    
    # =========================================================================
    # Distance Tool
    # =========================================================================
    ToolMetadata(
        name="calculate_distance",
        description=(
            "Calculate the distance between two locations using their coordinates. "
            "Also provides estimated travel times for walking, driving, train, and flight. "
            "Use geocode_location first to get coordinates if you only have location names."
        ),
        category="local_computation",
        parameters_schema={
            "type": "object",
            "properties": {
                "from_latitude": {
                    "type": "number",
                    "description": "Starting point latitude (-90 to 90)"
                },
                "from_longitude": {
                    "type": "number",
                    "description": "Starting point longitude (-180 to 180)"
                },
                "to_latitude": {
                    "type": "number",
                    "description": "Destination latitude (-90 to 90)"
                },
                "to_longitude": {
                    "type": "number",
                    "description": "Destination longitude (-180 to 180)"
                }
            },
            "required": ["from_latitude", "from_longitude", "to_latitude", "to_longitude"],
            "additionalProperties": False
        },
        example_queries=[
            # Direct distance questions
            "How far is it from Tokyo to Osaka?",
            "What's the distance between Paris and London?",
            "How many kilometers from Sydney to Melbourne?",
            "Distance from New York to Boston",
            # Travel time estimation
            "How long does it take to drive from LA to San Francisco?",
            "Can I walk from my hotel to the Eiffel Tower?",
            "Train time from Tokyo to Kyoto?",
            "Flight duration between cities",
            # Route planning
            "Is it worth renting a car for this distance?",
            "Should I take a train or fly?",
            "How far apart are the attractions?",
            # Day trip planning
            "Can I do a day trip from Rome to Florence?",
            "Is Barcelona close enough to visit from Madrid?",
            "Distance for a side trip",
            # Multi-city planning
            "Total distance of my itinerary",
            "How far between each destination?",
        ],
        rate_limit_per_minute=1000,  # Local computation, no API
        timeout_seconds=5,
        requires_api_key=False,
        is_enabled=True,
        priority=7,
    ),
    
    # =========================================================================
    # Web Search Tool
    # =========================================================================
    ToolMetadata(
        name="web_search",
        description=(
            "Search the web for up-to-date travel information, guides, tips, "
            "and advisories. Can target specific intents like destination info, "
            "things to do, travel advisory, local events, restaurants, or transportation."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results (default 5)"},
                "intent": {"type": "string", "description": "Search intent: destination_info, things_to_do, travel_advisory, local_events, restaurants, transportation"},
            },
            "required": ["query"],
        },
        example_queries=[
            "What are the top things to do in Kyoto?",
            "Travel advisory for Southeast Asia",
            "Best time to visit Iceland",
            "Local events happening in Barcelona in May",
            "Transportation options in Tokyo",
            "Safety tips for traveling to Mexico",
            "Hidden gems in Portugal",
            "Is it safe to travel to Turkey right now?",
            "Best neighborhoods to stay in Rome",
            "Local food guide for Bangkok",
        ],
        rate_limit_per_minute=30,
        timeout_seconds=10,
        requires_api_key=False,
        is_enabled=True,
        priority=8,
    ),

    # =========================================================================
    # Attractions Tool
    # =========================================================================
    ToolMetadata(
        name="search_attractions",
        description=(
            "Search for tourist attractions, landmarks, museums, and points of interest "
            "near a specific location. Uses OpenStreetMap data. Free, no API key required."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Center latitude"},
                "longitude": {"type": "number", "description": "Center longitude"},
                "radius": {"type": "integer", "description": "Radius in meters (default 5000)"},
                "categories": {"type": "string", "description": "Categories: cultural, historic, natural, museums, etc."},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["latitude", "longitude"],
        },
        example_queries=[
            "Find museums near the Louvre in Paris",
            "Tourist attractions in central Tokyo",
            "Historic landmarks in Rome within walking distance",
            "Nature spots near Reykjavik",
            "What attractions are near my hotel in Barcelona?",
            "Cultural sites in Kyoto",
            "Points of interest near the Colosseum",
            "Beaches near Cancun",
        ],
        rate_limit_per_minute=30,
        timeout_seconds=15,
        requires_api_key=False,
        is_enabled=True,
        priority=7,
    ),

    # =========================================================================
    # Restaurant Search Tool
    # =========================================================================
    ToolMetadata(
        name="search_restaurants",
        description=(
            "Search for restaurants, cafes, and food establishments near a location "
            "using OpenStreetMap data. Can filter by cuisine type. Free, no API key."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Center latitude"},
                "longitude": {"type": "number", "description": "Center longitude"},
                "radius": {"type": "integer", "description": "Radius in meters (default 1000)"},
                "cuisine": {"type": "string", "description": "Cuisine filter (e.g., 'japanese,seafood')"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["latitude", "longitude"],
        },
        example_queries=[
            "Restaurants near Shibuya station",
            "Best ramen shops in Shinjuku",
            "Italian restaurants near the Trevi Fountain",
            "Seafood restaurants near the harbor",
            "Where to eat near my hotel in Paris",
            "Cafes near Central Park",
            "Thai food near Khao San Road",
            "Vegetarian restaurants in Berlin",
        ],
        rate_limit_per_minute=30,
        timeout_seconds=15,
        requires_api_key=False,
        is_enabled=True,
        priority=7,
    ),

    # =========================================================================
    # Foursquare Places Search Tool
    # =========================================================================
    ToolMetadata(
        name="search_places_foursquare",
        description=(
            "Search for places (restaurants, hotels, attractions, museums, cafes, bars, "
            "shops, landmarks) using Foursquare's 100M+ POI database with ratings. "
            "Requires either coordinates or a location name."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "latitude": {"type": "number", "description": "Center latitude"},
                "longitude": {"type": "number", "description": "Center longitude"},
                "near": {"type": "string", "description": "Location name (alternative to lat/lon)"},
                "categories": {"type": "string", "description": "Category filter"},
                "limit": {"type": "integer", "description": "Max results (default 5)"},
            },
            "required": ["query"],
        },
        example_queries=[
            "Best rated restaurants near Times Square",
            "Highly rated sushi in Tokyo",
            "Top hotels near the Eiffel Tower",
            "Popular bars in downtown Austin",
            "Best coffee shops in Seattle",
            "Recommended museums in London",
            "Where to shop in Milan",
            "Nightlife in Barcelona",
            "Rated attractions near me",
            "Best pizza places in New York",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=True,
        is_enabled=True,
        priority=8,
    ),

    # =========================================================================
    # Flight Search Tool
    # =========================================================================
    ToolMetadata(
        name="search_flights",
        description=(
            "Search for flight offers between airports using IATA codes. "
            "Returns prices, airlines, duration, and stops. "
            "Supports one-way and round-trip searches."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Departure IATA code (e.g., SEA)"},
                "destination": {"type": "string", "description": "Arrival IATA code (e.g., NRT)"},
                "departure_date": {"type": "string", "description": "Date YYYY-MM-DD"},
                "return_date": {"type": "string", "description": "Return date YYYY-MM-DD"},
                "adults": {"type": "integer", "description": "Number of passengers"},
            },
            "required": ["origin", "destination", "departure_date"],
        },
        example_queries=[
            "Find flights from Seattle to Tokyo in May",
            "Cheapest flights from JFK to London",
            "How much is a flight from LAX to Paris?",
            "Round trip flights from SFO to NRT",
            "Flight prices from Chicago to Rome",
            "Direct flights to Hawaii from Seattle",
            "Business class flights to Singapore",
            "Compare flight prices for my trip",
            "When is the cheapest time to fly to Japan?",
            "Book a flight from New York to Barcelona",
        ],
        rate_limit_per_minute=30,
        timeout_seconds=15,
        requires_api_key=True,
        is_enabled=True,
        priority=8,
    ),

    # =========================================================================
    # Google Maps Search Places Tool (via MCP)
    # =========================================================================
    ToolMetadata(
        name="googlemaps__search_places",
        description=(
            "Search for places (restaurants, hotels, attractions, museums, cafes, "
            "bars, parks, landmarks) using Google Maps via MCP. Returns names, addresses, "
            "ratings, price levels, and opening hours. Very accurate global data."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "latitude": {"type": "number", "description": "Bias center latitude"},
                "longitude": {"type": "number", "description": "Bias center longitude"},
                "radius": {"type": "integer", "description": "Bias radius in meters (default 5000)"},
                "type": {"type": "string", "description": "Place type filter (restaurant, museum, etc.)"},
                "limit": {"type": "integer", "description": "Max results (default 5)"},
            },
            "required": ["query"],
        },
        example_queries=[
            "Find the best restaurants in Shinjuku Tokyo",
            "Top rated hotels near the Eiffel Tower",
            "Museums to visit in London",
            "Cafes near Central Park New York",
            "Tourist attractions in Barcelona with ratings",
            "Best pizza places in Rome",
            "Highly rated sushi restaurants in Osaka",
            "Parks and gardens in Kyoto",
            "Shopping malls near my hotel in Dubai",
            "Nightlife spots in Berlin with reviews",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=True,
        is_enabled=True,
        priority=9,
    ),

    # =========================================================================
    # Google Maps Directions Tool (via MCP)
    # =========================================================================
    ToolMetadata(
        name="googlemaps__get_directions",
        description=(
            "Get directions between two locations via Google Maps (MCP). "
            "Returns distance, duration, and route steps. "
            "Supports driving, walking, bicycling, and public transit."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Start address or lat,lng"},
                "destination": {"type": "string", "description": "End address or lat,lng"},
                "mode": {"type": "string", "description": "driving, walking, bicycling, transit"},
                "departure_time": {"type": "string", "description": "For transit: 'now' or ISO datetime"},
            },
            "required": ["origin", "destination"],
        },
        example_queries=[
            "How do I get from Shinjuku to Asakusa by train?",
            "Directions from the Louvre to the Eiffel Tower walking",
            "Transit directions from JFK airport to Manhattan",
            "Drive time from Los Angeles to San Francisco",
            "Walking directions between Colosseum and Trevi Fountain",
            "How long is the train ride from Tokyo to Kyoto?",
            "Public transit from my hotel to the airport",
            "Best route from downtown to the beach",
            "Driving directions in Iceland ring road",
            "How to get from one attraction to another",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=True,
        is_enabled=True,
        priority=8,
    ),

    # =========================================================================
    # Holiday Tool
    # =========================================================================
    ToolMetadata(
        name="get_public_holidays",
        description=(
            "Get public holidays for a country in a specific year. "
            "Useful for planning around closures, avoiding crowds, or experiencing local celebrations. "
            "Use ISO country codes like 'JP' for Japan, 'US' for United States, 'FR' for France."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code (e.g., 'JP', 'US', 'GB', 'FR')"
                },
                "year": {
                    "type": "integer",
                    "description": "Year to get holidays for (e.g., 2026)"
                }
            },
            "required": ["country_code", "year"],
            "additionalProperties": False
        },
        example_queries=[
            # Direct holiday requests
            "What are the public holidays in Japan this year?",
            "When is Golden Week in Japan?",
            "List holidays in France for 2026",
            "National holidays in the US",
            # Planning around holidays
            "Will anything be closed during my trip to Italy?",
            "Are there any bank holidays in the UK in March?",
            "Should I avoid traveling during any holidays?",
            # Experiencing local culture
            "What festivals happen in Thailand in April?",
            "Are there any celebrations during my visit to India?",
            "Local holidays to experience in Germany",
            # Avoiding crowds
            "When are the busy holiday periods in Spain?",
            "Dates to avoid when visiting Tokyo",
            "Are there any long weekends in Korea?",
            # Shop/attraction closures
            "Will museums be closed on any holidays?",
            "Bank holiday weekends in London",
            "Days when attractions might be closed",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=False,  # Nager.Date is free
        is_enabled=True,
        priority=6,
    ),
]


async def seed_tools():
    """Seed all tool metadata to ChromaDB."""
    logger.info("Starting tool metadata seeding...")
    
    # Initialize Tool-RAG
    tool_rag = ToolRAG()
    await tool_rag.initialize()
    
    # Clear existing tools
    logger.info("Clearing existing tool metadata...")
    await tool_rag.clear_all()
    
    # Add all tools
    logger.info(f"Seeding {len(TOOL_DEFINITIONS)} tools...")
    count = await tool_rag.add_tools(TOOL_DEFINITIONS)
    
    logger.info(f"Successfully seeded {count} tools")
    
    # Verify by counting
    final_count = tool_rag.count()
    logger.info(f"Verification: {final_count} tools in collection")
    
    # Test a sample query
    logger.info("\nTesting tool selection...")
    test_queries = [
        "What's the weather like in Tokyo?",
        "How far is Paris from London?",
        "Convert 1000 USD to EUR",
    ]
    
    for query in test_queries:
        result = await tool_rag.select_tools(query, top_k=3)
        tools = [t.name for t in result.selected_tools]
        logger.info(f"Query: '{query}'")
        logger.info(f"  Selected tools: {tools}")
        logger.info(f"  Scores: {result.scores}")
    
    logger.info("\nTool metadata seeding complete!")


if __name__ == "__main__":
    asyncio.run(seed_tools())
