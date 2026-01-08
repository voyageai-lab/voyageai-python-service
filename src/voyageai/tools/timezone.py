"""
Timezone Conversion Tool - Convert times between timezones.

Uses Python's built-in zoneinfo module (no external API needed).
This is useful for flight planning and scheduling activities.

Available since Python 3.9, uses IANA timezone database.

Example:
    tool = TimeZoneTool()
    result = await tool.execute(
        time="2026-01-10T14:00:00",
        from_timezone="America/New_York",
        to_timezone="Asia/Tokyo"
    )
"""

import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo, available_timezones

from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Common timezone aliases for user convenience
TIMEZONE_ALIASES = {
    # US
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    # Europe
    "GMT": "Europe/London",
    "BST": "Europe/London",
    "CET": "Europe/Paris",
    "CEST": "Europe/Paris",
    # Asia
    "JST": "Asia/Tokyo",
    "KST": "Asia/Seoul",
    "CST_CHINA": "Asia/Shanghai",
    "IST": "Asia/Kolkata",
    "SGT": "Asia/Singapore",
    # Australia
    "AEST": "Australia/Sydney",
    "AEDT": "Australia/Sydney",
    # Common cities
    "NEW_YORK": "America/New_York",
    "LOS_ANGELES": "America/Los_Angeles",
    "CHICAGO": "America/Chicago",
    "LONDON": "Europe/London",
    "PARIS": "Europe/Paris",
    "BERLIN": "Europe/Berlin",
    "TOKYO": "Asia/Tokyo",
    "BEIJING": "Asia/Shanghai",
    "SHANGHAI": "Asia/Shanghai",
    "HONG_KONG": "Asia/Hong_Kong",
    "SINGAPORE": "Asia/Singapore",
    "SYDNEY": "Australia/Sydney",
    "DUBAI": "Asia/Dubai",
}


class TimeZoneTool(BaseTool):
    """
    Convert times between timezones.
    
    This tool uses Python's built-in zoneinfo for timezone conversion.
    It's useful for:
    - Planning flight arrivals/departures
    - Scheduling activities across timezones
    - Understanding local time at destination
    
    No external API required - runs entirely locally.
    
    Input:
        time (str): Time to convert (ISO format or HH:MM)
        from_timezone (str): Source timezone (IANA format or alias)
        to_timezone (str): Target timezone (IANA format or alias)
        date (str, optional): Date if time is just HH:MM
        
    Output:
        {
            "original": {
                "time": "2026-01-10T14:00:00",
                "timezone": "America/New_York",
                "utc_offset": "-05:00"
            },
            "converted": {
                "time": "2026-01-11T04:00:00",
                "timezone": "Asia/Tokyo",
                "utc_offset": "+09:00"
            },
            "time_difference_hours": 14.0
        }
    """
    
    name = "convert_timezone"
    description = (
        "Convert a time from one timezone to another. "
        "Useful for flight planning and scheduling. "
        "Accepts IANA timezone names (e.g., 'Asia/Tokyo') or common abbreviations (e.g., 'JST', 'PST')."
    )
    parameters_schema = {
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
    }
    
    def _resolve_timezone(self, tz_input: str) -> str | None:
        """
        Resolve timezone input to IANA timezone name.
        
        Args:
            tz_input: User input (could be IANA name, alias, or city name)
            
        Returns:
            IANA timezone name or None if not found
        """
        tz_upper = tz_input.strip().upper().replace(" ", "_")
        
        # Check aliases first
        if tz_upper in TIMEZONE_ALIASES:
            return TIMEZONE_ALIASES[tz_upper]
        
        # Check if it's already a valid IANA name
        if tz_input in available_timezones():
            return tz_input
        
        # Try case-insensitive match
        tz_lower = tz_input.lower()
        for tz in available_timezones():
            if tz.lower() == tz_lower:
                return tz
        
        # Try partial match (e.g., "Tokyo" -> "Asia/Tokyo")
        for tz in available_timezones():
            if tz_lower in tz.lower():
                return tz
        
        return None
    
    def _parse_time(self, time_str: str, date_str: str | None = None) -> datetime | None:
        """
        Parse time string into datetime.
        
        Supports:
        - ISO format: "2026-01-10T14:00:00"
        - Time only: "14:00" (requires date parameter)
        - Time only: "14:00:00"
        """
        time_str = time_str.strip()
        
        # Try ISO format first
        try:
            return datetime.fromisoformat(time_str)
        except ValueError:
            pass
        
        # Try time-only formats
        for fmt in ["%H:%M", "%H:%M:%S"]:
            try:
                t = datetime.strptime(time_str, fmt)
                if date_str:
                    d = datetime.strptime(date_str, "%Y-%m-%d")
                    return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)
                else:
                    # Use today's date
                    today = datetime.now()
                    return datetime(today.year, today.month, today.day, t.hour, t.minute, t.second)
            except ValueError:
                continue
        
        return None
    
    def _format_offset(self, offset_seconds: int) -> str:
        """Format UTC offset in hours:minutes."""
        hours, remainder = divmod(abs(offset_seconds), 3600)
        minutes = remainder // 60
        sign = "+" if offset_seconds >= 0 else "-"
        return f"{sign}{hours:02d}:{minutes:02d}"
    
    async def execute(
        self,
        time: str,  # noqa: A002 - shadowing builtin is intentional for clarity
        from_timezone: str,
        to_timezone: str,
        date: str | None = None
    ) -> ToolResult:
        """
        Convert time between timezones.
        
        Args:
            time: Time string to convert
            from_timezone: Source timezone
            to_timezone: Target timezone
            date: Optional date if time is just HH:MM
            
        Returns:
            ToolResult with conversion details
        """
        import time as time_module
        start_time_ms = time_module.time()
        
        input_args = {
            "time": time,
            "from_timezone": from_timezone,
            "to_timezone": to_timezone,
            "date": date
        }
        
        # Resolve timezones
        from_tz = self._resolve_timezone(from_timezone)
        if not from_tz:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Unknown timezone: {from_timezone}. Try IANA format like 'America/New_York' or 'Asia/Tokyo'.",
                latency_ms=0
            )
        
        to_tz = self._resolve_timezone(to_timezone)
        if not to_tz:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Unknown timezone: {to_timezone}. Try IANA format like 'America/New_York' or 'Asia/Tokyo'.",
                latency_ms=0
            )
        
        # Parse time
        dt = self._parse_time(time, date)
        if not dt:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Could not parse time: {time}. Use ISO format (2026-01-10T14:00:00) or HH:MM.",
                latency_ms=0
            )
        
        try:
            # Create timezone-aware datetime
            from_zoneinfo = ZoneInfo(from_tz)
            to_zoneinfo = ZoneInfo(to_tz)
            
            dt_from = dt.replace(tzinfo=from_zoneinfo)
            dt_to = dt_from.astimezone(to_zoneinfo)
            
            # Calculate offsets
            from_offset = dt_from.utcoffset()
            to_offset = dt_to.utcoffset()
            
            from_offset_secs = int(from_offset.total_seconds()) if from_offset else 0
            to_offset_secs = int(to_offset.total_seconds()) if to_offset else 0
            
            # Calculate time difference
            diff_hours = (to_offset_secs - from_offset_secs) / 3600
            
            output = {
                "original": {
                    "time": dt_from.strftime("%Y-%m-%dT%H:%M:%S"),
                    "timezone": from_tz,
                    "utc_offset": self._format_offset(from_offset_secs)
                },
                "converted": {
                    "time": dt_to.strftime("%Y-%m-%dT%H:%M:%S"),
                    "timezone": to_tz,
                    "utc_offset": self._format_offset(to_offset_secs)
                },
                "time_difference_hours": diff_hours
            }
            
            logger.info(
                f"Timezone conversion: {dt_from.strftime('%H:%M')} {from_tz} -> "
                f"{dt_to.strftime('%H:%M')} {to_tz} (diff: {diff_hours:+.1f}h)"
            )
            
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time_module.time() - start_time_ms) * 1000)
            )
            
        except Exception as e:
            logger.error(f"Timezone conversion failed: {e}")
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Timezone error: {str(e)}",
                latency_ms=int((time_module.time() - start_time_ms) * 1000)
            )

