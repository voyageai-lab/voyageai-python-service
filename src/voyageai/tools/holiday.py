"""
Public Holiday Tool - Get public holidays for travel planning.

Uses Nager.Date API (free, no API key required).
https://date.nager.at/

This is useful for:
- Avoiding travel during busy holiday periods
- Planning around bank holidays (shops may be closed)
- Understanding local celebrations

Example:
    tool = HolidayTool()
    result = await tool.execute(
        country_code="JP",
        year=2026
    )
"""

import logging
import time
from datetime import datetime

import httpx

from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Nager.Date API endpoint (free, no key needed)
NAGER_DATE_URL = "https://date.nager.at/api/v3"

# Supported countries (ISO 3166-1 alpha-2)
# Full list at: https://date.nager.at/Country
SUPPORTED_COUNTRIES = {
    # Americas
    "US": "United States",
    "CA": "Canada",
    "MX": "Mexico",
    "BR": "Brazil",
    "AR": "Argentina",
    # Europe
    "GB": "United Kingdom",
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "NL": "Netherlands",
    "BE": "Belgium",
    "CH": "Switzerland",
    "AT": "Austria",
    "SE": "Sweden",
    "NO": "Norway",
    "DK": "Denmark",
    "FI": "Finland",
    "IE": "Ireland",
    "PT": "Portugal",
    "PL": "Poland",
    "CZ": "Czech Republic",
    "GR": "Greece",
    # Asia-Pacific
    "JP": "Japan",
    "CN": "China",
    "KR": "South Korea",
    "SG": "Singapore",
    "HK": "Hong Kong",
    "TW": "Taiwan",
    "AU": "Australia",
    "NZ": "New Zealand",
    "IN": "India",
    "TH": "Thailand",
    "MY": "Malaysia",
    "ID": "Indonesia",
    "PH": "Philippines",
    "VN": "Vietnam",
    # Middle East & Africa
    "AE": "United Arab Emirates",
    "IL": "Israel",
    "ZA": "South Africa",
    "EG": "Egypt",
}


class HolidayTool(BaseTool):
    """
    Get public holidays for a country.
    
    This tool uses the Nager.Date API to retrieve:
    - Public/national holidays
    - Bank holidays
    - Optional holidays
    
    Useful for planning around closures and busy periods.
    
    API: https://date.nager.at/
    Rate Limit: None (fair use)
    Cost: Free
    
    Input:
        country_code (str): ISO 3166-1 alpha-2 country code (e.g., "JP", "US")
        year (int): Year to get holidays for
        
    Output:
        {
            "country": "Japan",
            "country_code": "JP",
            "year": 2026,
            "holidays": [
                {
                    "date": "2026-01-01",
                    "name": "New Year's Day",
                    "local_name": "元日",
                    "types": ["Public"],
                    "is_fixed": true
                },
                ...
            ],
            "total_holidays": 16
        }
    """
    
    name = "get_public_holidays"
    description = (
        "Get public holidays for a country in a specific year. "
        "Useful for planning around closures, avoiding crowds, or experiencing local celebrations. "
        "Use ISO country codes like 'JP' for Japan, 'US' for United States, 'FR' for France."
    )
    parameters_schema = {
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
    }
    
    def __init__(self, timeout: float = 10.0):
        """
        Initialize the holiday tool.
        
        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout
    
    async def execute(
        self,
        country_code: str,
        year: int
    ) -> ToolResult:
        """
        Get public holidays for a country.
        
        Args:
            country_code: ISO 3166-1 alpha-2 country code
            year: Year to get holidays for
            
        Returns:
            ToolResult with list of holidays
        """
        start_time = time.time()
        
        # Normalize country code
        country_code = country_code.strip().upper()
        
        input_args = {
            "country_code": country_code,
            "year": year
        }
        
        # Validate country code
        if country_code not in SUPPORTED_COUNTRIES:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Unsupported country: {country_code}. Supported: {', '.join(sorted(SUPPORTED_COUNTRIES.keys()))}",
                latency_ms=0
            )
        
        # Validate year (API typically supports current year ± 5 years)
        current_year = datetime.now().year
        if not (current_year - 5 <= year <= current_year + 5):
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Year must be between {current_year - 5} and {current_year + 5}",
                latency_ms=0
            )
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{NAGER_DATE_URL}/PublicHolidays/{year}/{country_code}",
                    timeout=self.timeout
                )
                response.raise_for_status()
                holidays_raw = response.json()
            
            # Format holidays
            holidays = []
            for h in holidays_raw:
                holidays.append({
                    "date": h.get("date"),
                    "name": h.get("name"),
                    "local_name": h.get("localName"),
                    "types": h.get("types", []),
                    "is_fixed": h.get("fixed", False),
                })
            
            output = {
                "country": SUPPORTED_COUNTRIES[country_code],
                "country_code": country_code,
                "year": year,
                "holidays": holidays,
                "total_holidays": len(holidays)
            }
            
            logger.info(
                f"Retrieved {len(holidays)} holidays for {country_code} in {year}"
            )
            
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=output,
                success=True,
                latency_ms=int((time.time() - start_time) * 1000)
            )
            
        except httpx.TimeoutException:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Holiday API timed out",
                latency_ms=int((time.time() - start_time) * 1000)
            )
        except httpx.HTTPStatusError as e:
            error_msg = f"Holiday API HTTP error: {e.response.status_code}"
            if e.response.status_code == 404:
                error_msg = f"No holiday data available for {country_code} in {year}"
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=error_msg,
                latency_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            logger.error(f"Holiday fetch failed: {e}")
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Holiday error: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000)
            )

