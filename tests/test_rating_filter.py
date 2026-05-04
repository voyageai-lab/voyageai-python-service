"""
Tests for the Google Maps and Foursquare rating filter feature.

Verifies that:
1. Google Maps MCP server filters places below minimum rating
2. Google Maps MCP server sorts results by rating descending
3. Foursquare tool filters places below minimum rating
4. Config settings for min_rating are respected
5. Places without ratings are NOT filtered out (only low-rated ones)
"""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from voyageai.config import Settings


class TestGoogleMapsRatingFilter:
    """Tests for the Google Maps MCP server rating filter."""

    def _make_google_places_response(self, places_data: list[dict]) -> dict:
        """Helper to build a Google Maps API-style response."""
        places = []
        for p in places_data:
            place = {
                "displayName": {"text": p["name"]},
                "id": p.get("id", "place_123"),
                "formattedAddress": p.get("address", "123 Test St"),
                "location": {
                    "latitude": p.get("lat", 35.68),
                    "longitude": p.get("lon", 139.76),
                },
                "rating": p.get("rating"),
                "userRatingCount": p.get("count", 100),
                "priceLevel": p.get("price", "PRICE_LEVEL_MODERATE"),
                "primaryType": p.get("type", "restaurant"),
                "businessStatus": "OPERATIONAL",
                "googleMapsUri": f"https://maps.google.com/?cid={p.get('id', '123')}",
                "websiteUri": p.get("website", ""),
            }
            places.append(place)
        return {"places": places}

    @pytest.mark.asyncio
    async def test_filters_low_rated_places(self):
        """Places below min_rating should be excluded from results."""
        from voyageai.mcp.google_maps_server import search_places

        mock_resp = self._make_google_places_response([
            {"name": "Great Place", "rating": 4.5},
            {"name": "Bad Place", "rating": 2.0},
            {"name": "OK Place", "rating": 3.8},
            {"name": "Terrible Place", "rating": 1.5},
        ])

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_resp
        mock_http_response.raise_for_status = MagicMock()

        with patch("voyageai.mcp.google_maps_server.GOOGLE_MAPS_API_KEY", "test_key"), \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_http_response):
            result_json = await search_places(
                query="restaurants",
                latitude=35.68,
                longitude=139.76,
                min_rating=3.5,
            )
            result = json.loads(result_json)

        assert result["count"] == 2
        names = [p["name"] for p in result["places"]]
        assert "Great Place" in names
        assert "OK Place" in names
        assert "Bad Place" not in names
        assert "Terrible Place" not in names
        assert result["filtered_below_rating"] == 2

    @pytest.mark.asyncio
    async def test_sorts_by_rating_descending(self):
        """Results should be sorted by rating, highest first."""
        from voyageai.mcp.google_maps_server import search_places

        mock_resp = self._make_google_places_response([
            {"name": "Mid Place", "rating": 4.0},
            {"name": "Best Place", "rating": 4.8},
            {"name": "Good Place", "rating": 4.3},
        ])

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_resp
        mock_http_response.raise_for_status = MagicMock()

        with patch("voyageai.mcp.google_maps_server.GOOGLE_MAPS_API_KEY", "test_key"), \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_http_response):
            result_json = await search_places(
                query="restaurants",
                latitude=35.68,
                longitude=139.76,
                min_rating=0.0,
            )
            result = json.loads(result_json)

        ratings = [p["rating"] for p in result["places"]]
        assert ratings == sorted(ratings, reverse=True)
        assert result["places"][0]["name"] == "Best Place"

    @pytest.mark.asyncio
    async def test_keeps_unrated_places(self):
        """Places without a rating (None) should NOT be filtered out."""
        from voyageai.mcp.google_maps_server import search_places

        mock_resp = self._make_google_places_response([
            {"name": "Rated Place", "rating": 4.5},
            {"name": "Unrated Place", "rating": None},
            {"name": "Low Rated", "rating": 2.0},
        ])

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_resp
        mock_http_response.raise_for_status = MagicMock()

        with patch("voyageai.mcp.google_maps_server.GOOGLE_MAPS_API_KEY", "test_key"), \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_http_response):
            result_json = await search_places(
                query="restaurants",
                latitude=35.68,
                longitude=139.76,
                min_rating=3.5,
            )
            result = json.loads(result_json)

        names = [p["name"] for p in result["places"]]
        assert "Rated Place" in names
        assert "Unrated Place" in names
        assert "Low Rated" not in names

    @pytest.mark.asyncio
    async def test_min_rating_zero_disables_filter(self):
        """Setting min_rating=0 should return all places."""
        from voyageai.mcp.google_maps_server import search_places

        mock_resp = self._make_google_places_response([
            {"name": "High", "rating": 4.5},
            {"name": "Low", "rating": 1.0},
        ])

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_resp
        mock_http_response.raise_for_status = MagicMock()

        with patch("voyageai.mcp.google_maps_server.GOOGLE_MAPS_API_KEY", "test_key"), \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_http_response):
            result_json = await search_places(
                query="restaurants",
                latitude=35.68,
                longitude=139.76,
                min_rating=0.0,
            )
            result = json.loads(result_json)

        assert result["count"] == 2
        assert result["filtered_below_rating"] == 0


class TestFoursquareRatingFilter:
    """Tests for the Foursquare tool rating filter."""

    @pytest.mark.asyncio
    async def test_filters_low_rated_foursquare_places(self):
        """Foursquare places below min_rating should be excluded."""
        from voyageai.tools.foursquare_search import FoursquareSearchTool

        tool = FoursquareSearchTool(api_key="test_key")

        mock_response_data = {
            "results": [
                {
                    "name": "Good Restaurant",
                    "rating": 8.5,
                    "fsq_id": "abc",
                    "location": {"formatted_address": "123 St"},
                    "geocodes": {"main": {"latitude": 35.68, "longitude": 139.76}},
                    "categories": [{"name": "Restaurant"}],
                    "distance": 500,
                },
                {
                    "name": "Bad Restaurant",
                    "rating": 4.0,
                    "fsq_id": "def",
                    "location": {"formatted_address": "456 St"},
                    "geocodes": {"main": {"latitude": 35.69, "longitude": 139.77}},
                    "categories": [{"name": "Restaurant"}],
                    "distance": 800,
                },
                {
                    "name": "Unrated Restaurant",
                    "fsq_id": "ghi",
                    "location": {"formatted_address": "789 St"},
                    "geocodes": {"main": {"latitude": 35.70, "longitude": 139.78}},
                    "categories": [{"name": "Restaurant"}],
                    "distance": 300,
                },
            ]
        }

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response_data
        mock_http_response.raise_for_status = MagicMock()

        with patch("voyageai.tools.foursquare_search.settings") as mock_settings, \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_http_response):
            mock_settings.foursquare_api_key = "test_key"
            mock_settings.foursquare_min_rating = 6.0
            tool.api_key = "test_key"

            result = await tool.execute(
                query="sushi",
                latitude=35.68,
                longitude=139.76,
            )

        assert result.success is True
        names = [p["name"] for p in result.output["places"]]
        assert "Good Restaurant" in names
        assert "Unrated Restaurant" in names
        assert "Bad Restaurant" not in names

    @pytest.mark.asyncio
    async def test_foursquare_sorts_by_rating(self):
        """Foursquare results should be sorted by rating descending."""
        from voyageai.tools.foursquare_search import FoursquareSearchTool

        tool = FoursquareSearchTool(api_key="test_key")

        mock_response_data = {
            "results": [
                {
                    "name": "Mid",
                    "rating": 7.0,
                    "fsq_id": "a",
                    "location": {"formatted_address": "1 St"},
                    "geocodes": {"main": {"latitude": 35.68, "longitude": 139.76}},
                    "categories": [{"name": "Cafe"}],
                    "distance": 100,
                },
                {
                    "name": "Best",
                    "rating": 9.2,
                    "fsq_id": "b",
                    "location": {"formatted_address": "2 St"},
                    "geocodes": {"main": {"latitude": 35.69, "longitude": 139.77}},
                    "categories": [{"name": "Cafe"}],
                    "distance": 200,
                },
            ]
        }

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response_data
        mock_http_response.raise_for_status = MagicMock()

        with patch("voyageai.tools.foursquare_search.settings") as mock_settings, \
             patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_http_response):
            mock_settings.foursquare_api_key = "test_key"
            mock_settings.foursquare_min_rating = 0.0
            tool.api_key = "test_key"

            result = await tool.execute(
                query="cafe",
                latitude=35.68,
                longitude=139.76,
            )

        assert result.success is True
        assert result.output["places"][0]["name"] == "Best"


class TestRatingFilterConfig:
    """Tests for rating filter configuration."""

    def test_default_google_maps_min_rating(self):
        """Default Google Maps min rating should be 3.5."""
        s = Settings(openai_api_key="test")
        assert s.google_maps_min_rating == 3.5

    def test_default_foursquare_min_rating(self):
        """Default Foursquare min rating should be 0.0 (no filter)."""
        s = Settings(openai_api_key="test")
        assert s.foursquare_min_rating == 0.0

    def test_custom_google_maps_min_rating(self):
        """Custom Google Maps min rating via env var."""
        s = Settings(
            openai_api_key="test",
            google_maps_min_rating=4.0,
        )
        assert s.google_maps_min_rating == 4.0
