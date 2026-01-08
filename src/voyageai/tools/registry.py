"""
Tool Registry - Centralized management of all available tools.

The ToolRegistry provides:
- Registration of tool instances
- Lookup by tool name
- Conversion to OpenAI function calling format
- Execution of tools by name

This is the entry point for the agent service to discover and use tools.

Example:
    from voyageai.tools.registry import tool_registry
    
    # Get all tools in OpenAI format
    functions = tool_registry.get_openai_tools()
    
    # Execute a tool
    result = await tool_registry.execute("geocode_location", {"location": "Tokyo"})
"""

import logging
from typing import Any

from voyageai.tools.base import BaseTool, ToolResult
from voyageai.tools.currency import CurrencyTool
from voyageai.tools.distance import DistanceTool
from voyageai.tools.geocode import GeocodeTool
from voyageai.tools.holiday import HolidayTool
from voyageai.tools.timezone import TimeZoneTool
from voyageai.tools.weather import WeatherTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all available tools.
    
    The registry provides a single point of access for:
    1. Discovering available tools
    2. Converting tools to OpenAI function format
    3. Executing tools by name
    
    Design Decisions:
    - Tools are registered at startup, not dynamically
    - Each tool is a singleton (one instance per registry)
    - Thread-safe for concurrent tool execution
    - Failed tool lookups return descriptive errors
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> BaseTool | None:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """
        Get list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_tool_descriptions(self) -> list[dict[str, str]]:
        """
        Get descriptions of all registered tools.
        
        Returns:
            List of dicts with name and description
        """
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]
    
    def get_openai_tools(self) -> list[dict[str, Any]]:
        """
        Get all tools in OpenAI function calling format.
        
        This is the format expected by the OpenAI chat completions API
        in the `tools` parameter.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        return [tool.to_openai_function() for tool in self._tools.values()]
    
    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name.
        
        This is the main entry point for tool execution. It:
        1. Looks up the tool by name
        2. Validates the tool exists
        3. Executes the tool with the provided arguments
        4. Returns the result (success or failure)
        
        Args:
            name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool
            
        Returns:
            ToolResult with execution outcome
        """
        tool = self._tools.get(name)
        
        if tool is None:
            logger.warning(f"Attempted to execute unknown tool: {name}")
            return ToolResult(
                tool_name=name,
                input_args=arguments,
                output=None,
                success=False,
                error=f"Unknown tool: {name}. Available tools: {', '.join(self.list_tools())}",
                latency_ms=0
            )
        
        try:
            logger.info(f"Executing tool: {name} with args: {arguments}")
            result = await tool.execute(**arguments)
            
            if result.success:
                logger.info(f"Tool {name} succeeded in {result.latency_ms}ms")
            else:
                logger.warning(f"Tool {name} failed: {result.error}")
            
            return result
            
        except TypeError as e:
            # Argument mismatch
            logger.error(f"Tool {name} argument error: {e}")
            return ToolResult(
                tool_name=name,
                input_args=arguments,
                output=None,
                success=False,
                error=f"Invalid arguments for {name}: {str(e)}",
                latency_ms=0
            )
        except Exception as e:
            # Unexpected error
            logger.exception(f"Tool {name} unexpected error: {e}")
            return ToolResult(
                tool_name=name,
                input_args=arguments,
                output=None,
                success=False,
                error=f"Tool execution failed: {str(e)}",
                latency_ms=0
            )


def create_default_registry() -> ToolRegistry:
    """
    Create a registry with all default tools registered.
    
    This is the standard way to initialize tools for the VoyageAI service.
    All tools are instantiated with their default configurations.
    
    Returns:
        ToolRegistry with all travel planning tools
    """
    registry = ToolRegistry()
    
    # Register all tools
    registry.register(GeocodeTool())
    registry.register(WeatherTool())
    registry.register(CurrencyTool())
    registry.register(TimeZoneTool())
    registry.register(DistanceTool())
    registry.register(HolidayTool())
    
    logger.info(f"Initialized tool registry with {len(registry.list_tools())} tools")
    
    return registry


# Global singleton registry
# This is initialized at module load time and shared across the application
tool_registry = create_default_registry()

