"""MCP Server Registry.

Manages the configuration and discovery of available MCP servers.
Servers are configured via environment variables or a config file.

Each server entry specifies:
- name: Human-readable server name
- transport: Connection type (stdio, http, sse)
- command/url: How to connect
- env: Optional environment variables for the server process
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    transport: str  # "stdio" | "http" | "sse"
    command: list[str] | None = None  # For stdio transport
    url: str | None = None  # For http/sse transport
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""


# Known MCP server configurations for documentation and quick setup.
# These are the servers supported out of the box; configure via env vars.
#
# ┌──────────────┬──────────────────┬──────────────────────────────────────────┐
# │ Server       │ Transport        │ Description                              │
# ├──────────────┼──────────────────┼──────────────────────────────────────────┤
# │ googlemaps   │ stdio            │ Google Maps: directions, places, geocode │
# │ xiaohongshu  │ streamable-http  │ Xiaohongshu: social media travel search  │
# └──────────────┴──────────────────┴──────────────────────────────────────────┘
#
# Xiaohongshu MCP setup:
#   1. git clone https://github.com/xpzouying/xiaohongshu-mcp
#   2. cd xiaohongshu-mcp && go run .
#   3. Server runs on http://localhost:18060/mcp
#   4. Set env: MCP_SERVERS=xiaohongshu
#              MCP_SERVER_xiaohongshu_TRANSPORT=streamable-http
#              MCP_SERVER_xiaohongshu_URL=http://localhost:18060/mcp
#
# Tools auto-discovered: xiaohongshu__search_feeds, xiaohongshu__get_feed_detail, etc.

KNOWN_SERVERS: dict[str, dict[str, str]] = {
    "googlemaps": {
        "transport": "stdio",
        "description": "Google Maps for directions, places, and geocoding",
    },
    "xiaohongshu": {
        "transport": "streamable-http",
        "url": "http://localhost:18060/mcp",
        "description": "Xiaohongshu social media search for travel recommendations and reviews",
    },
}


class MCPRegistry:
    """Registry of available MCP servers.

    Loads server configurations from environment variables:
        MCP_SERVERS=server1,server2
        MCP_SERVER_server1_TRANSPORT=stdio
        MCP_SERVER_server1_COMMAND=npx,-y,@modelcontextprotocol/server-google-maps
        MCP_SERVER_server1_ENV_GOOGLE_MAPS_API_KEY=xxx

    Known servers (see KNOWN_SERVERS above):
        - googlemaps: Google Maps via stdio transport
        - xiaohongshu: Xiaohongshu social media via streamable-http transport
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load MCP server configs from environment variables."""
        server_names = os.environ.get("MCP_SERVERS", "").strip()
        if not server_names:
            logger.info("No MCP servers configured (MCP_SERVERS not set)")
            return

        for name in server_names.split(","):
            name = name.strip()
            if not name:
                continue

            prefix = f"MCP_SERVER_{name}_"
            transport = os.environ.get(f"{prefix}TRANSPORT", "stdio")
            command_str = os.environ.get(f"{prefix}COMMAND", "")
            url = os.environ.get(f"{prefix}URL", "")
            description = os.environ.get(f"{prefix}DESCRIPTION", f"MCP server: {name}")
            enabled = os.environ.get(f"{prefix}ENABLED", "true").lower() == "true"

            # Collect env vars for the server process
            env: dict[str, str] = {}
            env_prefix = f"{prefix}ENV_"
            for key, value in os.environ.items():
                if key.startswith(env_prefix):
                    env_key = key[len(env_prefix):]
                    env[env_key] = value

            config = MCPServerConfig(
                name=name,
                transport=transport,
                command=command_str.split(",") if command_str else None,
                url=url or None,
                env=env,
                enabled=enabled,
                description=description,
            )

            self._servers[name] = config
            logger.info(
                "Registered MCP server: %s (transport=%s, enabled=%s)",
                name, transport, enabled,
            )

    def get_server(self, name: str) -> MCPServerConfig | None:
        """Get a server config by name."""
        return self._servers.get(name)

    def list_servers(self) -> list[MCPServerConfig]:
        """List all registered servers."""
        return list(self._servers.values())

    def list_enabled_servers(self) -> list[MCPServerConfig]:
        """List only enabled servers."""
        return [s for s in self._servers.values() if s.enabled]

    def register_server(self, config: MCPServerConfig) -> None:
        """Programmatically register a server (for testing)."""
        self._servers[config.name] = config


# Singleton
mcp_registry = MCPRegistry()
