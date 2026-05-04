"""
Currency Conversion Tool - Convert currencies for budget planning.

Uses Frankfurter API (free, open-source, no API key required).
https://www.frankfurter.app/

The API is based on data from the European Central Bank.
Supports ~30 major currencies.

Example:
    tool = CurrencyTool()
    result = await tool.execute(
        from_currency="USD",
        to_currency="JPY",
        amount=1000.0
    )
"""

import logging
import time

import httpx

from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Frankfurter API (free, no key needed, based on ECB data)
FRANKFURTER_URL = "https://api.frankfurter.app"

# Common currency codes for validation
SUPPORTED_CURRENCIES = {
    "AUD", "BGN", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK",
    "EUR", "GBP", "HKD", "HUF", "IDR", "ILS", "INR", "ISK",
    "JPY", "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PLN",
    "RON", "SEK", "SGD", "THB", "TRY", "USD", "ZAR"
}


class CurrencyTool(BaseTool):
    """
    Convert currency and get exchange rates.
    
    This tool uses the Frankfurter API (powered by ECB data) to:
    - Convert amounts between currencies
    - Get current exchange rates
    - Support ~30 major world currencies
    
    API: https://www.frankfurter.app/
    Rate Limit: None (fair use)
    Cost: Free
    
    Input:
        from_currency (str): Source currency code (e.g., "USD")
        to_currency (str): Target currency code (e.g., "JPY")
        amount (float): Amount to convert
        
    Output:
        {
            "from_currency": "USD",
            "to_currency": "JPY",
            "original_amount": 1000.0,
            "converted_amount": 149250.0,
            "exchange_rate": 149.25,
            "rate_date": "2026-01-08"
        }
    """
    
    name = "convert_currency"
    description = (
        "Convert an amount from one currency to another. "
        "Use this for budget planning and cost estimation. "
        "Supports major world currencies like USD, EUR, JPY, GBP, CNY, etc."
    )
    parameters_schema = {
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
    }
    
    def __init__(self, timeout: float = 10.0):
        """
        Initialize the currency tool.
        
        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout
    
    def _normalize_currency(self, code: str) -> str:
        """Normalize currency code to uppercase."""
        return code.strip().upper()
    
    async def execute(
        self,
        from_currency: str,
        to_currency: str,
        amount: float
    ) -> ToolResult:
        """
        Convert currency amount.
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            amount: Amount to convert
            
        Returns:
            ToolResult with conversion details
        """
        start_time = time.time()
        
        # Normalize currency codes
        from_curr = self._normalize_currency(from_currency)
        to_curr = self._normalize_currency(to_currency)
        
        input_args = {
            "from_currency": from_curr,
            "to_currency": to_curr,
            "amount": amount
        }
        
        # Validate currencies
        if from_curr not in SUPPORTED_CURRENCIES:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Unsupported currency: {from_curr}. Supported: {', '.join(sorted(SUPPORTED_CURRENCIES))}",
                latency_ms=0
            )
        
        if to_curr not in SUPPORTED_CURRENCIES:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Unsupported currency: {to_curr}. Supported: {', '.join(sorted(SUPPORTED_CURRENCIES))}",
                latency_ms=0
            )
        
        # Validate amount
        if amount <= 0:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error="Amount must be greater than 0",
                latency_ms=0
            )
        
        # Same currency - no conversion needed
        if from_curr == to_curr:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output={
                    "from_currency": from_curr,
                    "to_currency": to_curr,
                    "original_amount": amount,
                    "converted_amount": amount,
                    "exchange_rate": 1.0,
                    "rate_date": "N/A (same currency)"
                },
                success=True,
                latency_ms=int((time.time() - start_time) * 1000)
            )
        
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(
                    f"{FRANKFURTER_URL}/latest",
                    params={
                        "from": from_curr,
                        "to": to_curr,
                        "amount": amount
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
            
            # Extract conversion result
            rates = data.get("rates", {})
            converted = rates.get(to_curr, 0)
            
            # Calculate the exchange rate
            rate = converted / amount if amount > 0 else 0
            
            output = {
                "from_currency": from_curr,
                "to_currency": to_curr,
                "original_amount": amount,
                "converted_amount": round(converted, 2),
                "exchange_rate": round(rate, 6),
                "rate_date": data.get("date", "unknown")
            }
            
            logger.info(
                f"Currency conversion: {amount} {from_curr} = "
                f"{output['converted_amount']} {to_curr} (rate: {rate:.4f})"
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
                error="Currency API timed out",
                latency_ms=int((time.time() - start_time) * 1000)
            )
        except httpx.HTTPStatusError as e:
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Currency API HTTP error: {e.response.status_code}",
                latency_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            logger.error(f"Currency conversion failed: {e}")
            return ToolResult(
                tool_name=self.name,
                input_args=input_args,
                output=None,
                success=False,
                error=f"Currency error: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000)
            )

