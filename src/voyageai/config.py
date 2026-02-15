"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Service Configuration
    service_name: str = "voyageai-python-service"
    service_version: str = "0.1.0"
    debug: bool = False

    # AI Generation Settings
    max_retries: int = 3
    timeout_seconds: int = 60
    max_tokens: int = 4000
    temperature: float = 0.7

    # Kafka Configuration (Module 11)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_group_id: str = "voyageai-python"
    kafka_topic_planning_request: str = "planning.request"
    kafka_topic_planning_progress: str = "planning.progress"
    kafka_topic_planning_result: str = "planning.result"
    # Enable auto-commit for simpler offset management (at-least-once)
    kafka_auto_commit: bool = True
    kafka_auto_commit_interval_ms: int = 5000
    # Session timeout for consumer group rebalancing
    kafka_session_timeout_ms: int = 30000

    # Redis Configuration (Module 12 - Idempotency)
    redis_url: str = "redis://localhost:6379/0"
    # Lock TTL for idempotency guard (in seconds)
    redis_lock_ttl_seconds: int = 3600

    # MongoDB Configuration (Module 12 - Result Storage)
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "voyageai"
    mongodb_collection_results: str = "planning_results"

    # Worker Configuration (Module 12)
    worker_pipeline_timeout_seconds: int = 120


settings = Settings()

