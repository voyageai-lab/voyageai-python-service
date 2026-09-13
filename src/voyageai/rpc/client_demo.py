"""Tiny gRPC client to exercise the server.

    python -m voyageai.rpc.client_demo
"""

from __future__ import annotations

import asyncio

import grpc

from voyageai.rpc import voyage_pb2, voyage_pb2_grpc


async def main() -> None:
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = voyage_pb2_grpc.VoyageStub(channel)

        ping = await stub.Ping(voyage_pb2.PingRequest(message="hello from Java"))
        print("Ping ->", ping.message, "|", ping.service)

        gen = await stub.Generate(
            voyage_pb2.GenerateRequest(
                task_id="t1", user_id="u1", requirements="3 days in Tokyo, foodie"
            )
        )
        print("Generate -> status:", gen.status)
        if gen.error:
            print("Generate -> error:", gen.error[:120])


if __name__ == "__main__":
    asyncio.run(main())
