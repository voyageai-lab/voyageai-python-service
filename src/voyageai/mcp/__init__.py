"""MCP (Model Context Protocol) client infrastructure.

This package provides an MCP client that can connect to external MCP servers
and adapt their tools to the existing BaseTool interface. This enables the
VoyageAI agent to call any MCP-compatible travel API.

Architecture:
    MCPClientManager       - Manages connections to multiple MCP servers
    MCPRegistry            - Registry of available MCP servers (from config)
    MCPToolAdapter         - Adapts MCP tools to BaseTool interface
"""
