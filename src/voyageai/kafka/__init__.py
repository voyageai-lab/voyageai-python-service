"""Kafka integration for the VoyageAI Python service.

This package implements the Kafka event streaming pipeline:
- schemas: Pydantic models mirroring Java DTOs for cross-language compatibility
- producer: Sends progress and result events to Kafka
- consumer: Consumes planning request events from Kafka
"""
