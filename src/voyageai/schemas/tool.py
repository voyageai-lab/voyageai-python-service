"""
Pydantic models for tool-related schemas.

These schemas are used for API requests/responses involving tool calls,
separate from the internal ToolResult used by tools themselves.
"""

from pydantic import BaseModel, Field


class ToolCallTrace(BaseModel):
    """
    Record of a single tool call for observability.
    
    This is stored in the itinerary response to show which tools
    were called and their results, enabling debugging and transparency.
    """
    
    call_id: str = Field(..., description="Unique identifier for this tool call")
    tool_name: str = Field(..., description="Name of the tool that was called")
    arguments: dict = Field(default_factory=dict, description="Arguments passed to the tool")
    result: dict | list | str | None = Field(default=None, description="Tool output")
    success: bool = Field(default=True, description="Whether the call succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
    latency_ms: int = Field(default=0, description="Execution time in milliseconds")


class ToolCallRequest(BaseModel):
    """
    Request to invoke a specific tool directly (for testing).
    """
    
    tool_name: str = Field(..., description="Name of the tool to invoke")
    arguments: dict = Field(default_factory=dict, description="Arguments for the tool")


class ToolCallResponse(BaseModel):
    """
    Response from direct tool invocation.
    """
    
    tool_name: str
    success: bool
    result: dict | list | str | None = None
    error: str | None = None
    latency_ms: int = 0

