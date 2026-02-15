"""
Tool Metadata Schema for Tool-RAG

This module defines the schema for tool metadata that is stored in the
vector database for semantic tool selection. Each tool has:
- Basic info (name, description, category)
- JSON Schema for parameters
- Example queries that demonstrate when to use this tool
- Rate limiting and timeout configuration

The example_queries field is critical for Tool-RAG:
When a user asks a question, we embed the query and find tools whose
example_queries are semantically similar, ensuring we select the
right tools for the task.

Design Rationale:
- example_queries: Multiple examples improve recall (different phrasings)
- category: Enables filtering by tool type (info, calculation, external_api)
- rate_limit_per_minute: Protects against runaway API costs
- timeout_seconds: Ensures responsive user experience
- requires_api_key: For documentation and security auditing
"""

from pydantic import BaseModel, Field


class ToolMetadata(BaseModel):
    """
    Metadata for a tool stored in the vector database.
    
    This schema extends the basic tool definition with:
    1. Example queries for semantic matching
    2. Rate limiting configuration
    3. Categorization for filtering
    
    Example:
        weather_tool_meta = ToolMetadata(
            name="get_weather_forecast",
            description="Get weather forecast for a location",
            category="external_api",
            parameters_schema={...},
            example_queries=[
                "What's the weather like in Tokyo next week?",
                "Will it rain in Paris during my trip?",
                "Should I pack an umbrella for London?"
            ],
            rate_limit_per_minute=30,
            timeout_seconds=10,
            requires_api_key=False,  # Open-Meteo is free
        )
    """
    
    name: str = Field(
        ...,
        description="Unique identifier for the tool (must match tool.name)"
    )
    
    description: str = Field(
        ...,
        description="Human-readable description for the LLM"
    )
    
    category: str = Field(
        ...,
        description="Tool category: 'external_api', 'local_computation', 'info', 'booking'"
    )
    
    parameters_schema: dict = Field(
        default_factory=dict,
        description="JSON Schema for tool parameters"
    )
    
    example_queries: list[str] = Field(
        default_factory=list,
        min_length=1,
        description="Example user queries that should trigger this tool (for embedding)"
    )
    
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Maximum calls per minute per user"
    )
    
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=120,
        description="Maximum execution time before timeout"
    )
    
    requires_api_key: bool = Field(
        default=False,
        description="Whether this tool requires an external API key"
    )
    
    is_enabled: bool = Field(
        default=True,
        description="Whether this tool is currently enabled"
    )
    
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Tool priority (1=lowest, 10=highest) for tie-breaking"
    )

    def get_embedding_text(self) -> str:
        """
        Generate text for embedding.
        
        Combines description and example queries into a single text
        that captures the tool's semantic meaning.
        
        Returns:
            Text optimized for embedding
        """
        examples_text = " | ".join(self.example_queries)
        return f"{self.name}: {self.description}. Examples: {examples_text}"


class ToolSelectionResult(BaseModel):
    """
    Result of tool selection via Tool-RAG.
    
    Includes the selected tools along with explainability
    information showing why each tool was selected.
    """
    
    selected_tools: list[ToolMetadata] = Field(
        default_factory=list,
        description="Tools selected for this query"
    )
    
    query: str = Field(
        ...,
        description="Original user query"
    )
    
    scores: list[float] = Field(
        default_factory=list,
        description="Similarity scores for each selected tool"
    )
    
    total_tools_available: int = Field(
        default=0,
        description="Total number of tools in the registry"
    )
    
    selection_time_ms: int = Field(
        default=0,
        description="Time taken for tool selection"
    )


class RateLimitConfig(BaseModel):
    """
    Rate limiting configuration for tool execution.
    
    Uses token bucket algorithm:
    - Bucket starts full with max_tokens
    - Each request consumes 1 token
    - Tokens refill at refill_rate per second
    - Request blocked if no tokens available
    """
    
    max_tokens: int = Field(
        default=60,
        description="Maximum tokens in bucket (burst capacity)"
    )
    
    refill_rate: float = Field(
        default=1.0,
        description="Tokens added per second"
    )
    
    window_seconds: int = Field(
        default=60,
        description="Time window for rate calculation"
    )


class RateLimitStatus(BaseModel):
    """
    Current rate limit status for a tool/user.
    """
    
    tool_name: str
    user_id: str
    tokens_remaining: int
    next_refill_at: float
    is_limited: bool
