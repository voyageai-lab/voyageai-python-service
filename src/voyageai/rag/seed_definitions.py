"""Tool metadata definitions for Tool-RAG auto-seeding.

Shared between the manual seed script and the lazy auto-seed in ToolRAG.
"""

from __future__ import annotations

from voyageai.schemas.tool_metadata import ToolMetadata

TOOL_DEFINITIONS: list[ToolMetadata] = [
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
                "location": {"type": "string", "description": "Name of the location to geocode"}
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        example_queries=[
            "Where is Tokyo located?",
            "What are the coordinates of Paris?",
            "Find the latitude and longitude of New York City",
            "I'm planning a trip to Rome",
            "Looking for things to do in Bangkok",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=False,
        is_enabled=True,
        priority=9,
    ),
    ToolMetadata(
        name="get_weather_forecast",
        description=(
            "Get weather forecast for a destination. Requires latitude and longitude. "
            "Returns daily temperature, precipitation chance, and conditions."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["latitude", "longitude", "start_date", "end_date"],
        },
        example_queries=[
            "What's the weather like in Tokyo next week?",
            "Will it rain in Paris during my trip?",
            "Should I pack an umbrella for my trip to Seattle?",
            "Is it beach weather in Thailand right now?",
            "How hot is Singapore in July?",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=False,
        is_enabled=True,
        priority=8,
    ),
    ToolMetadata(
        name="convert_currency",
        description="Convert an amount from one currency to another for budget planning.",
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["from_currency", "to_currency", "amount"],
        },
        example_queries=[
            "How much is 1000 USD in Japanese Yen?",
            "Convert 500 euros to dollars",
            "What's my budget in local currency for Japan?",
            "Exchange rate from USD to CNY",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=False,
        is_enabled=True,
        priority=7,
    ),
    ToolMetadata(
        name="convert_timezone",
        description="Convert a time from one timezone to another for flight and schedule planning.",
        category="local_computation",
        parameters_schema={
            "type": "object",
            "properties": {
                "time": {"type": "string"},
                "from_timezone": {"type": "string"},
                "to_timezone": {"type": "string"},
            },
            "required": ["time", "from_timezone", "to_timezone"],
        },
        example_queries=[
            "If I leave NYC at 6pm, what time do I arrive in Tokyo?",
            "What time is 9am PST in London?",
            "Time zone conversion from Paris to Beijing",
        ],
        rate_limit_per_minute=1000,
        timeout_seconds=5,
        requires_api_key=False,
        is_enabled=True,
        priority=6,
    ),
    ToolMetadata(
        name="calculate_distance",
        description="Calculate distance between two locations with estimated travel times.",
        category="local_computation",
        parameters_schema={
            "type": "object",
            "properties": {
                "from_latitude": {"type": "number"},
                "from_longitude": {"type": "number"},
                "to_latitude": {"type": "number"},
                "to_longitude": {"type": "number"},
            },
            "required": ["from_latitude", "from_longitude", "to_latitude", "to_longitude"],
        },
        example_queries=[
            "How far is it from Tokyo to Osaka?",
            "Distance between Paris and London",
            "Can I do a day trip from Rome to Florence?",
            "How long does it take to drive from LA to San Francisco?",
        ],
        rate_limit_per_minute=1000,
        timeout_seconds=5,
        requires_api_key=False,
        is_enabled=True,
        priority=7,
    ),
    ToolMetadata(
        name="web_search",
        description=(
            "Search the web for up-to-date travel information, guides, tips, "
            "and advisories."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
                "intent": {"type": "string"},
            },
            "required": ["query"],
        },
        example_queries=[
            "What are the top things to do in Kyoto?",
            "Travel advisory for Southeast Asia",
            "Best time to visit Iceland",
            "Local events happening in Barcelona in May",
            "Safety tips for traveling to Mexico",
        ],
        rate_limit_per_minute=30,
        timeout_seconds=10,
        requires_api_key=False,
        is_enabled=True,
        priority=8,
    ),
    ToolMetadata(
        name="search_attractions",
        description=(
            "Search for tourist attractions, landmarks, museums near a location "
            "using OpenStreetMap data."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "radius": {"type": "integer"},
                "categories": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["latitude", "longitude"],
        },
        example_queries=[
            "Find museums near the Louvre in Paris",
            "Tourist attractions in central Tokyo",
            "Historic landmarks in Rome within walking distance",
            "Cultural sites in Kyoto",
            "Nature spots near Reykjavik",
        ],
        rate_limit_per_minute=30,
        timeout_seconds=15,
        requires_api_key=False,
        is_enabled=True,
        priority=7,
    ),
    ToolMetadata(
        name="search_restaurants",
        description=(
            "Search for restaurants, cafes, and food establishments near a location "
            "using OpenStreetMap data."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "radius": {"type": "integer"},
                "cuisine": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["latitude", "longitude"],
        },
        example_queries=[
            "Restaurants near Shibuya station",
            "Best ramen shops in Shinjuku",
            "Italian restaurants near the Trevi Fountain",
            "Where to eat near my hotel in Paris",
            "Vegetarian restaurants in Berlin",
        ],
        rate_limit_per_minute=30,
        timeout_seconds=15,
        requires_api_key=False,
        is_enabled=True,
        priority=7,
    ),
    ToolMetadata(
        name="search_places_foursquare",
        description=(
            "Search for places using Foursquare's POI database with ratings."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "near": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        example_queries=[
            "Best rated restaurants near Times Square",
            "Top hotels near the Eiffel Tower",
            "Popular bars in downtown Austin",
            "Recommended museums in London",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=True,
        is_enabled=True,
        priority=8,
    ),
    ToolMetadata(
        name="search_flights",
        description=(
            "Search for flight offers between airports using IATA codes."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "departure_date": {"type": "string"},
                "return_date": {"type": "string"},
                "adults": {"type": "integer"},
            },
            "required": ["origin", "destination", "departure_date"],
        },
        example_queries=[
            "Find flights from Seattle to Tokyo in May",
            "Cheapest flights from JFK to London",
            "Round trip flights from SFO to NRT",
            "Flight prices from Chicago to Rome",
        ],
        rate_limit_per_minute=30,
        timeout_seconds=15,
        requires_api_key=True,
        is_enabled=True,
        priority=8,
    ),
    ToolMetadata(
        name="googlemaps__search_places",
        description=(
            "Search for places using Google Maps via MCP. Returns names, addresses, "
            "ratings, price levels, and opening hours."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "radius": {"type": "integer"},
                "type": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        example_queries=[
            "Find the best restaurants in Shinjuku Tokyo",
            "Top rated hotels near the Eiffel Tower",
            "Museums to visit in London",
            "Tourist attractions in Barcelona with ratings",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=True,
        is_enabled=True,
        priority=9,
    ),
    ToolMetadata(
        name="googlemaps__get_place_details",
        description=(
            "Get detailed information about a specific place from Google Maps — "
            "website, phone, hours, reviews."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "place_id": {"type": "string"},
            },
            "required": ["place_id"],
        },
        example_queries=[
            "Get the official website for this restaurant",
            "What are the opening hours of this museum?",
            "Show me reviews for this hotel",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=True,
        is_enabled=True,
        priority=8,
    ),
    ToolMetadata(
        name="googlemaps__get_directions",
        description=(
            "Get directions between two locations via Google Maps. "
            "Supports driving, walking, bicycling, and transit."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "mode": {"type": "string"},
            },
            "required": ["origin", "destination"],
        },
        example_queries=[
            "How do I get from Shinjuku to Asakusa by train?",
            "Transit directions from JFK airport to Manhattan",
            "Walking directions between Colosseum and Trevi Fountain",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=True,
        is_enabled=True,
        priority=8,
    ),
    ToolMetadata(
        name="get_public_holidays",
        description=(
            "Get public holidays for a country in a specific year."
        ),
        category="external_api",
        parameters_schema={
            "type": "object",
            "properties": {
                "country_code": {"type": "string"},
                "year": {"type": "integer"},
            },
            "required": ["country_code", "year"],
        },
        example_queries=[
            "What are the public holidays in Japan this year?",
            "Will anything be closed during my trip to Italy?",
            "When are the busy holiday periods in Spain?",
        ],
        rate_limit_per_minute=60,
        timeout_seconds=10,
        requires_api_key=False,
        is_enabled=True,
        priority=6,
    ),
]
