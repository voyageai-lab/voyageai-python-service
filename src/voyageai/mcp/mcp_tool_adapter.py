"""
MCP Tool Adapter — bridges MCP tools to the BaseTool interface.

Adapts MCP tools discovered from remote MCP servers to the existing
BaseTool interface so they integrate seamlessly with:
- The agent's tool registry (same execute() API)
- Tool-RAG (same name/description/schema)
- The OpenAI function calling format

Each MCP tool is wrapped in an MCPToolAdapter that:
1. Implements the BaseTool interface (name, description, parameters_schema)
2. Calls the MCP server directly via Streamable HTTP for each invocation
3. Parses MCP response content (JSON text → dict)
4. Returns standard ToolResult objects

The adapter makes a fresh HTTP connection per call rather than holding a
persistent session. This avoids event-loop lifecycle issues in the sync
Kafka worker while still using the MCP protocol correctly.

Tool names are prefixed with the server name:
    "googlemaps__search_places", "googlemaps__get_directions"
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from voyageai.mcp.mcp_client import MCPClientManager
from voyageai.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class MCPToolAdapter(BaseTool):
    """Adapts a single MCP tool to the BaseTool interface.

    For Streamable HTTP MCP servers, each tool call opens a fresh session
    to the MCP server. This is slightly more overhead (~20-50ms) but
    avoids the complexity of maintaining a persistent async session across
    sync/async boundaries in the Kafka worker.

    Attributes:
        name: Prefixed tool name, e.g. "googlemaps__search_places"
        description: Tool description from MCP server
        parameters_schema: JSON Schema for tool input (from MCP server)
    """

    def __init__(
        self,
        mcp_server_name: str,
        mcp_tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        server_url: str,
        manager: MCPClientManager | None = None,
    ) -> None:
        self.name = f"{mcp_server_name}__{mcp_tool_name}"
        self.description = description
        self.parameters_schema = input_schema
        self._mcp_server = mcp_server_name
        self._mcp_tool = mcp_tool_name
        self._server_url = server_url
        self._manager = manager  # Kept for backwards compat but not used per-call

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the MCP tool by opening a fresh session to the MCP server.

        Each call:
        1. Opens a Streamable HTTP connection to the MCP server
        2. Initializes the MCP session
        3. Calls the tool
        4. Parses the response
        5. Closes the connection
        """
        start_time = time.time()

        try:
            async with streamablehttp_client(url=self._server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    result = await session.call_tool(self._mcp_tool, arguments=kwargs)

            # Extract text content from MCP response
            content_parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    content_parts.append(item.text)
                elif hasattr(item, "data"):
                    content_parts.append(str(item.data))
                else:
                    content_parts.append(str(item))

            raw_text = "\n".join(content_parts)

            # Check for MCP-level errors
            if result.isError:
                return ToolResult(
                    tool_name=self.name,
                    input_args=kwargs,
                    output=None,
                    success=False,
                    error=raw_text,
                    latency_ms=int((time.time() - start_time) * 1000),
                )

            # Parse JSON output (our MCP servers return JSON strings)
            try:
                output = json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                output = raw_text

            latency = int((time.time() - start_time) * 1000)
            logger.info("MCP tool %s completed in %dms", self.name, latency)

            return ToolResult(
                tool_name=self.name,
                input_args=kwargs,
                output=output,
                success=True,
                latency_ms=latency,
            )

        except Exception as e:
            logger.error("MCP tool %s execution failed: %s", self.name, e)
            return ToolResult(
                tool_name=self.name,
                input_args=kwargs,
                output=None,
                success=False,
                error=f"MCP tool error: {e}",
                latency_ms=int((time.time() - start_time) * 1000),
            )


def create_mcp_tool_adapters(
    manager: MCPClientManager,
) -> list[MCPToolAdapter]:
    """Create BaseTool adapters for all tools from all connected MCP servers.

    Args:
        manager: Connected MCPClientManager with active sessions.
                 Used for tool discovery, not for per-call invocation.

    Returns:
        List of MCPToolAdapter instances ready for tool registry registration.
    """
    adapters: list[MCPToolAdapter] = []

    for tool_info in manager.get_all_tools():
        server_name = tool_info.get("_mcp_server", "unknown")
        original_name = tool_info.get("_mcp_original_name", tool_info.get("name", ""))
        description = tool_info.get("description", f"MCP tool: {original_name}")
        input_schema = tool_info.get("inputSchema", {
            "type": "object",
            "properties": {},
        })

        # Get the server URL for direct HTTP calls
        client = manager.get_client(server_name)
        server_url = client.config.url if client else ""

        adapter = MCPToolAdapter(
            mcp_server_name=server_name,
            mcp_tool_name=original_name,
            description=description,
            input_schema=input_schema,
            server_url=server_url,
            manager=manager,
        )
        adapters.append(adapter)

        logger.info(
            "Created MCP tool adapter: %s (server=%s, tool=%s, url=%s)",
            adapter.name, server_name, original_name, server_url,
        )

    return adapters
