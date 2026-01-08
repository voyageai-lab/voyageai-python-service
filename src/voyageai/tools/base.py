"""
Base Tool Abstraction

This module defines the abstract base class for all tools in the VoyageAI system.
Each tool must implement the `execute` method and define its OpenAI function schema.

Design Principles:
1. All tools are async to support non-blocking I/O
2. Tools return structured ToolResult for consistent error handling
3. Each tool defines its own JSON Schema for OpenAI function calling
4. Tools are stateless and can be safely shared across requests

Architecture:
    BaseTool (ABC)
        ├── to_openai_function() -> dict  # Convert to OpenAI tools format
        └── execute(**kwargs) -> ToolResult  # Abstract: Execute the tool
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """
    Structured result from tool execution.
    
    This provides consistent output format across all tools,
    including timing information for performance monitoring.
    
    Attributes:
        tool_name: Name of the executed tool
        input_args: Arguments passed to the tool
        output: The actual result data (type varies by tool)
        success: Whether execution succeeded
        error: Error message if execution failed
        latency_ms: Execution time in milliseconds
    """
    
    tool_name: str = Field(..., description="Name of the executed tool")
    input_args: dict = Field(default_factory=dict, description="Arguments passed to the tool")
    output: Any = Field(default=None, description="Tool output (varies by tool)")
    success: bool = Field(..., description="Whether execution succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
    latency_ms: int = Field(default=0, description="Execution time in milliseconds")


class BaseTool(ABC):
    """
    Abstract base class for all VoyageAI tools.
    
    Each tool must define:
    - name: Unique identifier for the tool
    - description: Human-readable description for the LLM
    - parameters_schema: JSON Schema for input validation
    
    Example:
        class WeatherTool(BaseTool):
            name = "get_weather"
            description = "Get weather forecast"
            parameters_schema = {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
            
            async def execute(self, location: str) -> ToolResult:
                # Implementation
                pass
    """
    
    # Subclasses must define these class attributes
    name: str
    description: str
    parameters_schema: dict
    
    def to_openai_function(self) -> dict:
        """
        Convert tool to OpenAI function calling format.
        
        This format is used in the `tools` parameter of chat completions:
        https://platform.openai.com/docs/guides/function-calling
        
        Returns:
            dict: OpenAI function definition with type, name, description, and parameters
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given arguments.
        
        Implementations should:
        1. Validate input arguments
        2. Perform the actual operation (API call, calculation, etc.)
        3. Return a ToolResult with success=True and output
        4. On error, return ToolResult with success=False and error message
        
        Args:
            **kwargs: Tool-specific arguments matching parameters_schema
            
        Returns:
            ToolResult: Structured result containing output or error
        """
        pass
    
    async def _execute_with_timing(self, **kwargs) -> ToolResult:
        """
        Execute the tool and automatically measure latency.
        
        This is a helper method that wraps execute() with timing.
        """
        start_time = time.time()
        try:
            result = await self.execute(**kwargs)
            # Update latency if not already set
            if result.latency_ms == 0:
                result.latency_ms = int((time.time() - start_time) * 1000)
            return result
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                input_args=kwargs,
                output=None,
                success=False,
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000)
            )

