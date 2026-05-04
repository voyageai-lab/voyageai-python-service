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
    openai_model: str = "gpt-4o"
    openai_final_model: str = "o4-mini"

    # Gemini via OpenAI-compatible endpoint
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-2.5-flash"

    # Ollama fallback — local model used when no OpenAI key is configured.
    # Ollama exposes an OpenAI-compatible API at /v1.
    ollama_base_url: str = "http://192.168.1.62:11434/v1"
    ollama_model: str = "gemma4:e2b"
    ollama_enabled: bool = True

    # Service Configuration
    service_name: str = "voyageai-python-service"
    service_version: str = "0.1.0"
    debug: bool = False

    # AI Generation Settings
    max_retries: int = 3
    timeout_seconds: int = 60
    max_tokens: int = 16000
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
    worker_pipeline_timeout_seconds: int = 300
    # Local LLM (Ollama) is much slower than OpenAI; allow a longer budget.
    worker_pipeline_timeout_ollama_seconds: int = 900
    # Must be >= longest pipeline run (ms) or the consumer leaves the group mid-task.
    kafka_max_poll_interval_ms: int = 1_200_000

    # Foursquare Places API (free tier, no credit card)
    foursquare_api_key: str = ""

    # Amadeus Flight/Hotel API (free test environment)
    amadeus_api_key: str = ""
    amadeus_api_secret: str = ""

    # Google Maps API (Places, Directions, Geocoding — $200/mo free credit)
    google_maps_api_key: str = ""

    # Quality filter: minimum rating for place recommendations (0.0 to 5.0)
    google_maps_min_rating: float = 3.5
    foursquare_min_rating: float = 0.0  # Foursquare uses 0-10 scale; 0 = no filter

    # Built-in web search via Responses API (no API key needed, included in token cost)
    enable_builtin_web_search: bool = True
    # MCP servers to connect natively via Responses API
    google_maps_mcp_url: str = "http://localhost:8080/mcp"


settings = Settings()

