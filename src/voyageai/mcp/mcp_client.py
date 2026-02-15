"""
MCP Client — powered by the official MCP Python SDK.

Connects to MCP servers using the Model Context Protocol and provides
a unified interface for discovering and calling tools.

Supports transports:
- streamable-http: Connect to an HTTP-based MCP server (recommended for Docker)
- stdio: Launch server as a subprocess and communicate via stdin/stdout

This replaces our earlier hand-rolled JSON-RPC implementation with the
official `mcp` SDK's ClientSession, which handles protocol handshake,
capability negotiation, and typed tool invocation.

Usage:
    manager = MCPClientManager()
    await manager.connect_all(mcp_registry.list_enabled_servers())

    tools = manager.get_all_tools()            # For tool discovery
    result = await manager.call_tool(           # For tool invocation
        "googlemaps", "search_places", {"query": "ramen in Tokyo"}
    )
    await manager.disconnect_all()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from voyageai.mcp.mcp_registry import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for a single MCP server connection.

    Wraps the official MCP SDK's ClientSession with lifecycle management.
    The connection is held open via an AsyncExitStack so that the session
    stays alive across multiple tool calls.

    Lifecycle:
        1. connect()     — open transport + initialize session + discover tools
        2. list_tools()  — get available tools (cached from connect)
        3. call_tool()   — invoke a tool
        4. disconnect()  — tear down
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[dict[str, Any]] = []
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Establish connection to the MCP server and discover tools."""
        try:
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

            if self.config.transport in ("http", "sse", "streamable-http"):
                await self._connect_http()
            elif self.config.transport == "stdio":
                await self._connect_stdio()
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")
        except Exception as e:
            logger.error("Failed to connect to MCP server %s: %s", self.config.name, e)
            if self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None

    async def _connect_http(self) -> None:
        """Connect via Streamable HTTP transport (recommended for Docker)."""
        if not self.config.url:
            raise ValueError(
                f"No URL for HTTP MCP server '{self.config.name}'. "
                f"Set MCP_SERVER_{self.config.name}_URL"
            )

        assert self._exit_stack is not None

        logger.info("Connecting to MCP server %s at %s", self.config.name, self.config.url)

        # Open the streamable HTTP transport — stays alive via exit stack
        read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
            streamablehttp_client(url=self.config.url)
        )

        # Create and initialize the session
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        # Discover tools
        tools_result = await self._session.list_tools()
        self._tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            }
            for tool in tools_result.tools
        ]

        self._connected = True
        logger.info(
            "Connected to MCP server %s via HTTP: %d tools discovered",
            self.config.name,
            len(self._tools),
        )
        for t in self._tools:
            logger.info("  MCP tool: %s — %s", t["name"], t["description"][:80])

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport (subprocess)."""
        if not self.config.command:
            raise ValueError(
                f"No command for stdio MCP server '{self.config.name}'. "
                f"Set MCP_SERVER_{self.config.name}_COMMAND"
            )

        assert self._exit_stack is not None

        env = {**os.environ, **self.config.env}
        server_params = StdioServerParameters(
            command=self.config.command[0],
            args=self.config.command[1:] if len(self.config.command) > 1 else [],
            env=env,
        )

        logger.info(
            "Connecting to MCP server %s via stdio: %s",
            self.config.name,
            " ".join(self.config.command),
        )

        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        tools_result = await self._session.list_tools()
        self._tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            }
            for tool in tools_result.tools
        ]

        self._connected = True
        logger.info(
            "Connected to MCP server %s via stdio: %d tools discovered",
            self.config.name,
            len(self._tools),
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return discovered MCP tools."""
        return self._tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke an MCP tool via the official SDK session.

        Args:
            tool_name: Name of the tool to call (server-local name).
            arguments: Tool arguments.

        Returns:
            Dict with tool result content, or error info.
        """
        if not self._connected or not self._session:
            return {"error": f"Not connected to MCP server {self.config.name}"}

        try:
            result = await self._session.call_tool(tool_name, arguments=arguments)

            # Extract text content from the MCP response
            content_parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    content_parts.append(item.text)
                elif hasattr(item, "data"):
                    content_parts.append(str(item.data))
                else:
                    content_parts.append(str(item))

            return {"content": content_parts, "isError": result.isError}

        except Exception as e:
            logger.error("MCP tool %s/%s call failed: %s", self.config.name, tool_name, e)
            return {"error": str(e)}

    async def disconnect(self) -> None:
        """Disconnect and clean up."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning("Error disconnecting MCP server %s: %s", self.config.name, e)
            finally:
                self._exit_stack = None
                self._session = None
                self._connected = False
                logger.info("Disconnected from MCP server %s", self.config.name)


class MCPClientManager:
    """Manages connections to multiple MCP servers.

    Usage:
        manager = MCPClientManager()
        await manager.connect_all(configs)
        tools = manager.get_all_tools()
        result = await manager.call_tool("googlemaps", "search_places", {...})
        await manager.disconnect_all()
    """

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}

    async def connect_server(self, config: MCPServerConfig) -> MCPClient | None:
        """Connect to a single MCP server."""
        client = MCPClient(config)
        await client.connect()

        if client.connected:
            self._clients[config.name] = client
            return client

        logger.warning("Failed to connect to MCP server: %s", config.name)
        return None

    async def connect_all(self, configs: list[MCPServerConfig]) -> int:
        """Connect to all configured MCP servers.

        Returns:
            Number of successfully connected servers.
        """
        connected = 0
        for config in configs:
            if config.enabled:
                client = await self.connect_server(config)
                if client:
                    connected += 1
        logger.info("MCP: %d/%d servers connected", connected, len(configs))
        return connected

    def get_client(self, server_name: str) -> MCPClient | None:
        """Get a connected client by server name."""
        return self._clients.get(server_name)

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all tools from all connected servers, with server name prefix."""
        all_tools = []
        for server_name, client in self._clients.items():
            for tool in client.list_tools():
                prefixed_tool = {
                    **tool,
                    "name": f"{server_name}__{tool['name']}",
                    "_mcp_server": server_name,
                    "_mcp_original_name": tool["name"],
                }
                all_tools.append(prefixed_tool)
        return all_tools

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on a specific MCP server."""
        client = self._clients.get(server_name)
        if not client:
            return {"error": f"MCP server '{server_name}' not connected"}
        return await client.call_tool(tool_name, arguments)

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()
        logger.info("All MCP servers disconnected")
