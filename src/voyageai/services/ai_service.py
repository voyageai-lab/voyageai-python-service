"""AI service for generating structured travel itineraries using OpenAI."""

import json
import logging
from datetime import datetime

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
                        "schema": StructuredItinerary.model_json_schema(),
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

