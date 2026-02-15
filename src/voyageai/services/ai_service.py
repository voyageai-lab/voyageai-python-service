"""AI service for generating structured travel itineraries using OpenAI."""

import json
import logging
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from voyageai.config import settings
from voyageai.prompts.templates import (
    ITINERARY_EXAMPLE,
    TRAVEL_ITINERARY_SYSTEM_PROMPT,
    TRAVEL_ITINERARY_USER_PROMPT,
)
from voyageai.schemas.itinerary import StructuredItinerary

logger = logging.getLogger(__name__)


def make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively transform schema for OpenAI's strict structured output mode.
    
    OpenAI's strict mode only supports a limited JSON Schema subset:
      Supported: type, description, enum, const, properties, required, items,
                 anyOf, $ref, $defs, additionalProperties (must be false)
      NOT supported: pattern, format, minimum, maximum, exclusiveMinimum,
                     exclusiveMaximum, minLength, maxLength, minItems, maxItems,
                     minProperties, maxProperties, allOf, oneOf, not, if/then/else
    
    This function:
    1. Adds 'additionalProperties: false' to all objects
    2. Ensures all properties are in 'required' array
    3. Strips unsupported keywords that cause unpredictable behavior
    """
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()

    # Strip keywords unsupported by OpenAI strict mode
    _UNSUPPORTED_KEYWORDS = {
        "pattern", "format",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minLength", "maxLength",
        "minItems", "maxItems",
        "minProperties", "maxProperties",
    }
    for keyword in _UNSUPPORTED_KEYWORDS:
        result.pop(keyword, None)

    # Add additionalProperties: false for object types
    if result.get("type") == "object":
        result["additionalProperties"] = False
        
        # Ensure all properties are in required
        if "properties" in result:
            result["required"] = list(result["properties"].keys())

    # Process nested properties
    if "properties" in result:
        result["properties"] = {
            k: make_strict_schema(v) for k, v in result["properties"].items()
        }

    # Process items in arrays
    if "items" in result:
        result["items"] = make_strict_schema(result["items"])

    # Process $defs (Pydantic puts nested schemas here)
    if "$defs" in result:
        result["$defs"] = {
            k: make_strict_schema(v) for k, v in result["$defs"].items()
        }

    # Process allOf, anyOf, oneOf
    for key in ["allOf", "anyOf", "oneOf"]:
        if key in result:
            result[key] = [make_strict_schema(item) for item in result[key]]

    return result


class AIService:
    """Service for AI-powered itinerary generation."""

    def __init__(self):
        """Initialize the AI service with OpenAI client."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_itinerary(self, requirements: str) -> StructuredItinerary:
        """
        Generate a structured travel itinerary using OpenAI.

        Uses OpenAI's structured output mode (JSON schema) to ensure
        the response matches our Pydantic model exactly.

        Args:
            requirements: User's travel requirements in natural language

        Returns:
            StructuredItinerary: Validated itinerary object

        Raises:
            Exception: If generation fails after retries
        """
        user_prompt = TRAVEL_ITINERARY_USER_PROMPT.format(
            requirements=requirements,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )

        logger.info(f"Generating itinerary for requirements: {requirements[:100]}...")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TRAVEL_ITINERARY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Example output format:\n{json.dumps(ITINERARY_EXAMPLE, indent=2)}",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "travel_itinerary",
                        "schema": make_strict_schema(StructuredItinerary.model_json_schema()),
                        "strict": True,
                    },
                },
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from OpenAI")

            itinerary_data = json.loads(content)

            # Pydantic validation ensures data matches schema
            itinerary = StructuredItinerary.model_validate(itinerary_data)

            logger.info(
                f"Successfully generated {itinerary.metadata.total_days}-day "
                f"itinerary for {itinerary.metadata.destination}"
            )

            return itinerary

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            raise


# Singleton instance for dependency injection
ai_service = AIService()

