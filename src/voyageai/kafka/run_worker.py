"""Entry point for the Kafka worker process.

Usage:
    python -m voyageai.kafka.run_worker

This starts the Kafka consumer loop that:
1. Connects to Kafka and subscribes to planning.request
2. For each message, runs the PlanningWorker pipeline
3. Handles graceful shutdown on SIGTERM/SIGINT
"""

from __future__ import annotations

import logging
import signal
import sys

from voyageai.kafka.consumer import KafkaRequestConsumer
from voyageai.kafka.worker import PlanningWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the Kafka worker."""
    logger.info("Starting VoyageAI Planning Worker...")

    # Create worker and consumer
    worker = PlanningWorker()
    consumer = KafkaRequestConsumer(handler=worker.handle_request)

    # Register signal handlers for graceful shutdown
    def signal_handler(signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, initiating graceful shutdown...", sig_name)
        consumer.shutdown()
        worker.shutdown()
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
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()
