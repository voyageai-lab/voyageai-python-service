"""Entry point for the Kafka worker process.

Usage:
    python -m voyageai.kafka.run_worker

This starts the Kafka consumer loop that:
1. Connects to configured MCP servers (if any) and registers discovered tools
2. Connects to Kafka and subscribes to planning.request
3. For each message, runs the PlanningWorker pipeline
4. Handles graceful shutdown on SIGTERM/SIGINT
5. (Module 13) Uses structured JSON logging in production
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from voyageai.kafka.consumer import KafkaRequestConsumer
from voyageai.kafka.worker import PlanningWorker
from voyageai.logging_config import setup_logging

# Initialize structured logging (JSON in production, plain text in debug)
setup_logging()
logger = logging.getLogger(__name__)

# Keep a reference to the MCP manager for clean shutdown
_mcp_manager = None


async def _connect_mcp_servers() -> None:
    """Connect to configured MCP servers and register discovered tools."""
    global _mcp_manager

    from voyageai.mcp.mcp_client import MCPClientManager
    from voyageai.mcp.mcp_registry import mcp_registry
    from voyageai.mcp.mcp_tool_adapter import create_mcp_tool_adapters
    from voyageai.tools.registry import tool_registry

    enabled = mcp_registry.list_enabled_servers()
    if not enabled:
        logger.info("No MCP servers configured — skipping MCP initialization")
        return

    logger.info("Connecting to %d MCP server(s)...", len(enabled))
    manager = MCPClientManager()
    connected = await manager.connect_all(enabled)

    if connected == 0:
        logger.warning("No MCP servers connected successfully")
        return

    # Create adapters and register in the tool registry
    adapters = create_mcp_tool_adapters(manager)
    for adapter in adapters:
        try:
            tool_registry.register(adapter)
            logger.info("Registered MCP tool: %s", adapter.name)
        except ValueError as e:
            logger.warning("Skipping MCP tool %s: %s", adapter.name, e)

    _mcp_manager = manager
    logger.info(
        "MCP initialization complete: %d servers, %d tools registered",
        connected,
        len(adapters),
    )


async def _disconnect_mcp_servers() -> None:
    """Disconnect from all MCP servers."""
    global _mcp_manager
    if _mcp_manager:
        await _mcp_manager.disconnect_all()
        _mcp_manager = None


def main() -> None:
    """Main entry point for the Kafka worker."""
    logger.info("Starting VoyageAI Planning Worker...")

    # Connect to MCP servers (async, run in event loop)
    try:
        asyncio.run(_connect_mcp_servers())
    except Exception as e:
        logger.error("MCP initialization failed (non-fatal): %s", e)

    # Create worker and consumer (with clarification reply support)
    worker = PlanningWorker()
    consumer = KafkaRequestConsumer(
        handler=worker.handle_request,
        clarification_handler=worker.handle_clarification_reply,
    )

    # Register signal handlers for graceful shutdown
    def signal_handler(signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, initiating graceful shutdown...", sig_name)
        consumer.shutdown()
        worker.shutdown()
        # Disconnect MCP servers
        try:
            asyncio.run(_disconnect_mcp_servers())
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Start consuming (blocking)
    logger.info("Worker ready, consuming from Kafka...")
    try:
        consumer.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, shutting down...")
    finally:
        consumer.shutdown()
        worker.shutdown()
        try:
            asyncio.run(_disconnect_mcp_servers())
        except Exception:
            pass
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()
