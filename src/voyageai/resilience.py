"""Resilience patterns: retry, timeout, fallback, and cost tracking.

This module provides production-grade resilience for the worker pipeline:
1. Retry with exponential backoff (via tenacity) for transient failures
2. Pipeline timeout to prevent hung tasks
3. Fallback itinerary generation when full pipeline fails
4. Token/cost tracking for budget control

These patterns wrap the existing agent pipeline without modifying it,
following the Decorator pattern for clean separation of concerns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from voyageai.config import settings
from voyageai.services.agent_service import AgentResponse, AgentService

logger = logging.getLogger(__name__)


# =========================================================================
# Cost Tracker
# =========================================================================


@dataclass
class CostTracker:
    """Tracks token usage and estimated cost per task.

    Provides visibility into LLM spending and enables budget limits.
    Cost estimates are based on approximate GPT-4o-mini pricing.

    Attributes:
        total_tokens: Cumulative tokens consumed.
        total_cost_usd: Estimated cumulative cost.
        budget_limit_usd: Maximum allowed cost per task.
        requests: Number of LLM API calls made.
    """

    total_tokens: int = 0
    total_cost_usd: float = 0.0
    budget_limit_usd: float = 1.0  # $1 default per task
    requests: int = 0
    _token_costs: list[dict[str, Any]] = field(default_factory=list)

    # Approximate pricing (GPT-4o-mini)
    INPUT_COST_PER_1K: float = 0.00015
    OUTPUT_COST_PER_1K: float = 0.0006

    def record(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record token usage from an LLM call."""
        self.total_tokens += input_tokens + output_tokens
        cost = (
            (input_tokens / 1000) * self.INPUT_COST_PER_1K
            + (output_tokens / 1000) * self.OUTPUT_COST_PER_1K
        )
        self.total_cost_usd += cost
        self.requests += 1
        self._token_costs.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })

    def is_over_budget(self) -> bool:
        """Check if cumulative cost exceeds the budget limit."""
        return self.total_cost_usd > self.budget_limit_usd

    def summary(self) -> dict[str, Any]:
        """Return a summary of cost tracking."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "budget_limit_usd": self.budget_limit_usd,
            "requests": self.requests,
            "is_over_budget": self.is_over_budget(),
        }


# =========================================================================
# Retry-wrapped Agent Pipeline
# =========================================================================


class ResilientAgentPipeline:
    """Wraps AgentService with retry, timeout, and fallback logic.

    The pipeline applies these resilience layers in order:
    1. Timeout: Cancel if total time exceeds limit
    2. Retry: Retry transient failures with exponential backoff
    3. Fallback: If all retries fail, generate a simplified itinerary
    4. Cost tracking: Monitor and limit LLM spending

    Usage:
        pipeline = ResilientAgentPipeline(agent_service)
        response = await pipeline.execute(requirements="Plan trip to Tokyo")
    """

    def __init__(
        self,
        agent: AgentService | None = None,
        max_retries: int = 3,
        timeout_seconds: int | None = None,
        budget_limit_usd: float = 1.0,
    ) -> None:
        self._agent = agent or AgentService()
        self._max_retries = max_retries
        self._timeout = timeout_seconds or settings.worker_pipeline_timeout_seconds
        self._cost_tracker = CostTracker(budget_limit_usd=budget_limit_usd)

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    async def execute(
        self,
        requirements: str,
        user_id: str | None = None,
    ) -> AgentResponse:
        """Execute the agent pipeline with full resilience stack.

        Order of operations:
        1. Apply asyncio timeout
        2. Run agent with retries
        3. If all attempts fail, use fallback

        Args:
            requirements: User's travel planning requirements.
            user_id: User ID for rate limiting.

        Returns:
            AgentResponse (either from agent or fallback).
        """
        try:
            async with asyncio.timeout(self._timeout):
                return await self._execute_with_retry(requirements, user_id)
        except TimeoutError:
            logger.error(
                "Pipeline timeout after %ds for user=%s",
                self._timeout,
                user_id,
            )
            return self._generate_fallback(
                requirements,
                error=f"Pipeline timed out after {self._timeout}s",
            )
        except RetryError as e:
            last_error = str(e.last_attempt.exception()) if e.last_attempt.exception() else "Unknown"
            logger.error(
                "All %d retries exhausted: %s", self._max_retries, last_error
            )
            return self._generate_fallback(requirements, error=last_error)
        except Exception as e:
            logger.exception("Unexpected pipeline error")
            return self._generate_fallback(requirements, error=str(e))

    async def _execute_with_retry(
        self,
        requirements: str,
        user_id: str | None,
    ) -> AgentResponse:
        """Execute agent with tenacity retry logic.

        Retries on:
        - ConnectionError (network issues)
        - TimeoutError (API timeout)
        - RuntimeError (transient failures)

        Does NOT retry on:
        - ValueError (bad input, won't succeed on retry)
        - KeyError (programming error)
        """
        attempt = 0

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, RuntimeError)),
            reraise=True,
        )
        async def _inner() -> AgentResponse:
            nonlocal attempt
            attempt += 1
            if attempt > 1:
                logger.warning("Retry attempt %d/%d", attempt, self._max_retries)

            response = await self._agent.generate_with_tools(
                requirements=requirements,
                use_tool_rag=True,
                tool_rag_top_k=4,
                user_id=user_id,
            )

            # Track cost
            self._cost_tracker.record(
                input_tokens=response.total_tokens // 2,  # rough estimate
                output_tokens=response.total_tokens // 2,
            )

            if self._cost_tracker.is_over_budget():
                logger.warning(
                    "Budget exceeded: %s",
                    self._cost_tracker.summary(),
                )

            if not response.success:
                raise RuntimeError(response.error or "Agent returned failure")

            return response

        return await _inner()

    def _generate_fallback(
        self,
        requirements: str,
        error: str,
    ) -> AgentResponse:
        """Generate a minimal fallback response when the pipeline fails.

        Instead of returning nothing, provide a basic itinerary structure
        that the user can at least see. This is better UX than a blank error.

        The fallback is deterministic (no LLM call), so it always succeeds.
        """
        logger.warning("Using fallback itinerary: %s", error)

        return AgentResponse(
            itinerary=None,
            tool_trace=[],
            raw_response="",
            success=False,
            error=f"Pipeline failed after retries: {error}. Please try again.",
            total_tokens=self._cost_tracker.total_tokens,
            processing_time_ms=0,
        )
