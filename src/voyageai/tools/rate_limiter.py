"""
Rate Limiter for Tool Execution

This module implements a Token Bucket rate limiter to protect against:
1. Runaway API costs (e.g., tool calling loops)
2. External API rate limits (e.g., Nominatim 1 req/sec)
3. Resource exhaustion attacks

Token Bucket Algorithm:
-----------------------
The token bucket is a simple and efficient rate limiting algorithm:

1. Each user/tool combination has a bucket with max_tokens capacity
2. Tokens are consumed on each request (typically 1 per call)
3. Tokens are refilled at a constant rate (refill_rate per second)
4. If bucket is empty, request is rejected

Example:
    max_tokens=60, refill_rate=1 → 60 requests per minute, burst up to 60

Advantages:
- Allows burst traffic up to bucket size
- Smooth rate limiting over time
- Memory efficient (one bucket per user/tool)
- Easy to understand and tune

Implementation Notes:
- Uses in-memory storage (suitable for single-instance)
- For distributed systems, use Redis-based rate limiter
- Async-safe with asyncio locks
- Automatic cleanup of old buckets

Usage:
    limiter = RateLimiter()
    
    # Check if request is allowed
    if await limiter.is_allowed("user123", "get_weather", max_tokens=60):
        result = await weather_tool.execute(...)
    else:
        raise RateLimitExceeded("Too many requests")
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """
    A single token bucket for rate limiting.
    
    Attributes:
        tokens: Current number of available tokens
        max_tokens: Maximum bucket capacity
        refill_rate: Tokens added per second
        last_refill: Timestamp of last refill
    """
    
    tokens: float
    max_tokens: int
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time)
    
    def refill(self) -> None:
        """
        Refill the bucket based on elapsed time.
        
        This is called before each token consumption to ensure
        the bucket has the correct number of tokens.
        """
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        self.refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """
        Calculate time until requested tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Seconds until tokens are available (0 if already available)
        """
        self.refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        user_id: str,
        retry_after: float
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.user_id = user_id
        self.retry_after = retry_after


class RateLimiter:
    """
    In-memory rate limiter using Token Bucket algorithm.
    
    Thread-safe and async-safe implementation that:
    - Tracks rate limits per user/tool combination
    - Supports different limits for different tools
    - Automatically cleans up stale buckets
    
    Design Decisions:
    - In-memory for simplicity (single-instance deployment)
    - Per-tool configuration allows different limits
    - User-level isolation prevents abuse
    - Automatic cleanup prevents memory leaks
    
    For distributed deployments, consider using Redis:
    - Use MULTI/EXEC for atomic bucket operations
    - Or use Redis' built-in rate limiting (redis-cell)
    """
    
    _instance: "RateLimiter | None" = None
    
    def __init__(
        self,
        default_max_tokens: int = 60,
        default_refill_rate: float = 1.0,
        cleanup_interval: int = 300,  # 5 minutes
    ):
        """
        Initialize the rate limiter.
        
        Args:
            default_max_tokens: Default bucket capacity
            default_refill_rate: Default tokens per second
            cleanup_interval: Seconds between bucket cleanup
        """
        self.default_max_tokens = default_max_tokens
        self.default_refill_rate = default_refill_rate
        self.cleanup_interval = cleanup_interval
        
        self._buckets: dict[str, TokenBucket] = {}
        self._tool_configs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()
    
    @classmethod
    def get_instance(cls) -> "RateLimiter":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = RateLimiter()
        return cls._instance
    
    def configure_tool(
        self,
        tool_name: str,
        max_tokens: int,
        refill_rate: float | None = None
    ) -> None:
        """
        Configure rate limit for a specific tool.
        
        Args:
            tool_name: Name of the tool
            max_tokens: Maximum tokens (requests per minute if refill_rate=1)
            refill_rate: Tokens per second (defaults to max_tokens/60)
        """
        if refill_rate is None:
            refill_rate = max_tokens / 60.0  # Convert to per-second rate
        
        self._tool_configs[tool_name] = {
            "max_tokens": max_tokens,
            "refill_rate": refill_rate
        }
        logger.debug(f"Configured rate limit for {tool_name}: {max_tokens} tokens, {refill_rate}/sec")
    
    def _get_bucket_key(self, user_id: str, tool_name: str) -> str:
        """Generate unique key for user/tool bucket."""
        return f"{user_id}:{tool_name}"
    
    def _get_or_create_bucket(self, user_id: str, tool_name: str) -> TokenBucket:
        """
        Get existing bucket or create a new one.
        
        Uses tool-specific configuration if available, otherwise defaults.
        """
        key = self._get_bucket_key(user_id, tool_name)
        
        if key not in self._buckets:
            # Get tool-specific config or defaults
            config = self._tool_configs.get(tool_name, {})
            max_tokens = config.get("max_tokens", self.default_max_tokens)
            refill_rate = config.get("refill_rate", self.default_refill_rate)
            
            self._buckets[key] = TokenBucket(
                tokens=float(max_tokens),  # Start full
                max_tokens=max_tokens,
                refill_rate=refill_rate,
            )
        
        return self._buckets[key]
    
    async def _maybe_cleanup(self) -> None:
        """
        Clean up stale buckets periodically.
        
        Buckets that haven't been used in cleanup_interval seconds
        and are full are removed to free memory.
        """
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return
        
        self._last_cleanup = now
        stale_keys = []
        
        for key, bucket in self._buckets.items():
            # If bucket is full and hasn't been used recently, it's stale
            bucket.refill()
            if bucket.tokens >= bucket.max_tokens:
                # Check if it's been long enough since last activity
                if now - bucket.last_refill > self.cleanup_interval:
                    stale_keys.append(key)
        
        for key in stale_keys:
            del self._buckets[key]
        
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale rate limit buckets")
    
    async def is_allowed(
        self,
        user_id: str,
        tool_name: str,
        tokens: int = 1
    ) -> bool:
        """
        Check if a request is allowed under rate limit.
        
        Args:
            user_id: User identifier
            tool_name: Tool being called
            tokens: Tokens to consume (default 1)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        async with self._lock:
            await self._maybe_cleanup()
            bucket = self._get_or_create_bucket(user_id, tool_name)
            return bucket.consume(tokens)
    
    async def check_and_consume(
        self,
        user_id: str,
        tool_name: str,
        tokens: int = 1
    ) -> None:
        """
        Check rate limit and consume tokens, raising if exceeded.
        
        Args:
            user_id: User identifier
            tool_name: Tool being called
            tokens: Tokens to consume
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        async with self._lock:
            await self._maybe_cleanup()
            bucket = self._get_or_create_bucket(user_id, tool_name)
            
            if not bucket.consume(tokens):
                retry_after = bucket.time_until_available(tokens)
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {tool_name}. Retry after {retry_after:.1f}s",
                    tool_name=tool_name,
                    user_id=user_id,
                    retry_after=retry_after
                )
    
    async def get_status(
        self,
        user_id: str,
        tool_name: str
    ) -> dict[str, Any]:
        """
        Get current rate limit status for a user/tool.
        
        Returns:
            Dict with remaining tokens, max tokens, and refill info
        """
        async with self._lock:
            bucket = self._get_or_create_bucket(user_id, tool_name)
            bucket.refill()
            
            return {
                "user_id": user_id,
                "tool_name": tool_name,
                "tokens_remaining": int(bucket.tokens),
                "max_tokens": bucket.max_tokens,
                "refill_rate_per_second": bucket.refill_rate,
                "time_to_full": bucket.time_until_available(bucket.max_tokens - int(bucket.tokens)),
            }
    
    async def reset(self, user_id: str, tool_name: str | None = None) -> None:
        """
        Reset rate limit for a user (optionally for specific tool).
        
        Args:
            user_id: User identifier
            tool_name: Optional tool name (resets all tools if None)
        """
        async with self._lock:
            if tool_name:
                key = self._get_bucket_key(user_id, tool_name)
                if key in self._buckets:
                    del self._buckets[key]
            else:
                # Reset all buckets for user
                keys_to_delete = [
                    k for k in self._buckets.keys()
                    if k.startswith(f"{user_id}:")
                ]
                for key in keys_to_delete:
                    del self._buckets[key]
        
        logger.info(f"Reset rate limit for user {user_id}" + (f" tool {tool_name}" if tool_name else ""))


# Global singleton instance
rate_limiter = RateLimiter.get_instance()
