"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

