"""gRPC server exposing the same capabilities as the HTTP API.

New, opt-in module added by feat/grpc. It does NOT touch the FastAPI app.
The FastAPI HTTP service keeps running exactly as before; this is a second
transport that delegates to the SAME service layer (ai_service), so the
business logic exists in one place and is reachable over both HTTP and RPC.

Run it as a separate process:
    python -m voyageai.rpc.server        # listens on :50051
"""

from __future__ import annotations

import asyncio
import logging

import grpc

from voyageai.config import settings
from voyageai.rpc import voyage_pb2, voyage_pb2_grpc
from voyageai.services.ai_service import ai_service

logger = logging.getLogger(__name__)


class VoyageServicer(voyage_pb2_grpc.VoyageServicer):
    """gRPC handlers. Thin, just like the HTTP routers: delegate to services."""

    async def Ping(self, request, context):
        # Dependency-free: proves the transport works without external services.
        return voyage_pb2.PingReply(
            message=f"pong: {request.message}",
            service=settings.service_name,
        )

    async def Generate(self, request, context):
        # Reuses the EXACT same service the HTTP /generate endpoint calls.
        try:
            itinerary = await ai_service.generate_itinerary(request.requirements)
            return voyage_pb2.GenerateReply(
                status="COMPLETED",
                itinerary_json=itinerary.model_dump_json(),
            )
        except Exception as e:  # mirror planning.py: never leak a raw crash
            logger.error(f"gRPC Generate failed for task {request.task_id}: {e}")
            return voyage_pb2.GenerateReply(status="FAILED", error=str(e)[:500])


async def serve(port: int = 50051) -> None:
    server = grpc.aio.server()
    voyage_pb2_grpc.add_VoyageServicer_to_server(VoyageServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"gRPC server listening on :{port}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
