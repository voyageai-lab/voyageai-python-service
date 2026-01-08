"""
Distance Calculation Tool - Calculate distance and travel time between locations.

Uses the Haversine formula for great-circle distance calculation.
No external API required - runs entirely locally.

The Haversine formula calculates the shortest distance over the earth's
surface, giving an "as-the-crow-flies" distance between two points.

Example:
    tool = DistanceTool()
    result = await tool.execute(
        from_latitude=35.68,
        from_longitude=139.76,
        to_latitude=34.69,
        to_longitude=135.50
    )
"""

import logging
import math
import time

from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Earth's mean radius in kilometers
EARTH_RADIUS_KM = 6371.0

# Conversion factors
KM_TO_MILES = 0.621371

# Average travel speeds (km/h) for estimation
TRAVEL_SPEEDS = {
    "walking": 5.0,       # Average walking speed
    "cycling": 15.0,      # Average cycling speed
    "driving": 80.0,      # Average driving speed (highways)
    "driving_city": 40.0, # Average city driving
    "train": 150.0,       # High-speed rail average
    "flight": 800.0,      # Commercial aviation average
}


class DistanceTool(BaseTool):
    """
    Calculate distance and travel time between two locations.
    
    This tool uses the Haversine formula to calculate the great-circle
    distance between two points on Earth. It also provides estimated
    travel times for different transportation modes.
    
    No external API required - pure mathematical calculation.
    
    Input:
        from_latitude (float): Starting point latitude
        from_longitude (float): Starting point longitude
        to_latitude (float): Destination latitude
        to_longitude (float): Destination longitude
        
    Output:
        {
            "distance_km": 403.5,
            "distance_miles": 250.7,
            "estimated_travel_time": {
                "walking_hours": 80.7,
                "cycling_hours": 26.9,
                "driving_hours": 5.0,
                "train_hours": 2.7,
                "flight_hours": 0.5
            },
            "bearing_degrees": 245.3
        }
    """
    
    name = "calculate_distance"
    description = (
        "Calculate the distance between two locations using their coordinates. "
        "Also provides estimated travel times for walking, driving, train, and flight. "
        "Use geocode_location first to get coordinates if you only have location names."
    )
    parameters_schema = {
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
    }
    
    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate the great-circle distance between two points using Haversine formula.
        
        The Haversine formula determines the shortest distance over the earth's
        surface, giving an "as-the-crow-flies" distance.
        
        Formula:
            a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
            c = 2 * atan2(√a, √(1−a))
            d = R * c
        
        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = (
            math.sin(delta_lat / 2) ** 2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return EARTH_RADIUS_KM * c
    
    def _calculate_bearing(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate the initial bearing (compass direction) from point 1 to point 2.
        
        Returns:
            Bearing in degrees (0-360, where 0=North, 90=East, 180=South, 270=West)
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = (
            math.cos(lat1_rad) * math.sin(lat2_rad) -
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
        )
        
        bearing_rad = math.atan2(x, y)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360
        return (bearing_deg + 360) % 360
    
    def _bearing_to_direction(self, bearing: float) -> str:
        """Convert bearing to cardinal direction."""
        directions = [
            "N", "NNE", "NE", "ENE",
            "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW",
            "W", "WNW", "NW", "NNW"
        ]
        index = round(bearing / 22.5) % 16
        return directions[index]
    
    def _estimate_travel_times(self, distance_km: float) -> dict:
        """
        Estimate travel times for different modes of transportation.
        
        These are rough estimates and don't account for:
        - Actual road distances (usually longer than straight-line)
        - Traffic, stops, or rest breaks
        - Check-in time for flights
        """
        return {
            "walking_hours": round(distance_km / TRAVEL_SPEEDS["walking"], 1),
            "cycling_hours": round(distance_km / TRAVEL_SPEEDS["cycling"], 1),
            "driving_hours": round(distance_km / TRAVEL_SPEEDS["driving"], 1),
            "train_hours": round(distance_km / TRAVEL_SPEEDS["train"], 1),
            "flight_hours": round(distance_km / TRAVEL_SPEEDS["flight"], 2),
        }
    
    async def execute(
        self,
        from_latitude: float,
        from_longitude: float,
        to_latitude: float,
        to_longitude: float
    ) -> ToolResult:
        """
        Calculate distance and travel time between two points.
        
        Args:
            from_latitude: Starting point latitude
            from_longitude: Starting point longitude
            to_latitude: Destination latitude
            to_longitude: Destination longitude
            
        Returns:
            ToolResult with distance and travel time estimates
        """
        start_time = time.time()
        input_args = {
            "from_latitude": from_latitude,
            "from_longitude": from_longitude,
            "to_latitude": to_latitude,
            "to_longitude": to_longitude
        }
        
        # Validate coordinates
        for name, lat in [("from_latitude", from_latitude), ("to_latitude", to_latitude)]:
            if not (-90 <= lat <= 90):
                return ToolResult(
                    tool_name=self.name,
                    input_args=input_args,
                    output=None,
                    success=False,
                    error=f"{name} must be between -90 and 90",
                    latency_ms=0
                )
        
        for name, lon in [("from_longitude", from_longitude), ("to_longitude", to_longitude)]:
            if not (-180 <= lon <= 180):
                return ToolResult(
                    tool_name=self.name,
                    input_args=input_args,
                    output=None,
                    success=False,
                    error=f"{name} must be between -180 and 180",
                    latency_ms=0
                )
        
        try:
            # Calculate distance
            distance_km = self._haversine_distance(
                from_latitude, from_longitude,
                to_latitude, to_longitude
            )
            
            # Calculate bearing
            bearing = self._calculate_bearing(
                from_latitude, from_longitude,
                to_latitude, to_longitude
            )
            
            # Estimate travel times
            travel_times = self._estimate_travel_times(distance_km)
            
            output = {
                "distance_km": round(distance_km, 1),
                "distance_miles": round(distance_km * KM_TO_MILES, 1),
                "estimated_travel_time": travel_times,
                "bearing_degrees": round(bearing, 1),
                "direction": self._bearing_to_direction(bearing),
            }
            
            logger.info(
                f"Distance: ({from_latitude}, {from_longitude}) -> "
                f"({to_latitude}, {to_longitude}) = {distance_km:.1f} km"
            )
            
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000)
            )
            
        except Exception as e:
            logger.error(f"Distance calculation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Calculation error: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000)
            )

