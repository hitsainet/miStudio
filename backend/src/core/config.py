"""Configuration management using Pydantic Settings.

This module provides typed configuration loaded from environment variables.
All settings are validated at startup to fail fast if misconfigured.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings are loaded from .env file or environment variables.
    Validation happens at startup to ensure proper configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database Configuration
    database_url: PostgresDsn = Field(
        description="Async PostgreSQL connection URL (postgresql+asyncpg://...)"
    )
    database_url_sync: str = Field(
        description="Sync PostgreSQL connection URL for Alembic (postgresql://...)"
    )

    # Redis Configuration
    redis_url: RedisDsn = Field(description="Redis connection URL")

    # Celery Configuration
    celery_broker_url: RedisDsn = Field(description="Celery broker URL (Redis)")
    celery_result_backend: RedisDsn = Field(description="Celery result backend URL (Redis)")

    # Data Storage Paths
    data_dir: Path = Field(default=Path("/data"), description="Root data directory")
    hf_home: Path = Field(
        default=Path("/data/huggingface_cache"), description="HuggingFace cache directory"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_workers: int = Field(default=1, ge=1, le=8, description="Number of API workers")
    api_reload: bool = Field(default=True, description="Enable auto-reload in development")
    api_base_url: str = Field(
        default="http://mistudio.mcslab.io", description="Public API base URL"
    )

    # CORS Configuration
    allowed_origins: list[str] = Field(
        default=[
            "http://mistudio.mcslab.io",
            "http://localhost:3000",
            "http://localhost",
            "http://192.168.224.222:3000",
            "http://192.168.224.222",
        ],
        description="Allowed CORS origins",
    )

    # Security
    secret_key: str = Field(
        min_length=32, description="Secret key for signing tokens (min 32 characters)"
    )
    access_token_expire_minutes: int = Field(
        default=30, ge=1, le=10080, description="Access token expiration in minutes"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="Log output format"
    )

    # HuggingFace Configuration
    hf_token: str | None = Field(default=None, description="HuggingFace API token (optional)")
    hf_cache_dir: Path = Field(
        default=Path("/data/huggingface_cache"),
        description="HuggingFace cache directory (same as HF_HOME)",
    )

    # WebSocket Configuration
    websocket_ping_interval: int = Field(
        default=30, ge=10, le=300, description="WebSocket ping interval in seconds"
    )
    websocket_ping_timeout: int = Field(
        default=10, ge=5, le=60, description="WebSocket ping timeout in seconds"
    )
    websocket_emit_url: str = Field(
        default="http://localhost:8000/api/internal/ws/emit",
        description="Internal WebSocket emission endpoint URL for Celery workers"
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(
        default=100, ge=1, le=10000, description="Max requests per minute"
    )
    rate_limit_downloads_per_hour: int = Field(
        default=10, ge=1, le=100, description="Max downloads per hour"
    )

    # Environment
    environment: Literal["development", "production", "test"] = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: str | list[str]) -> list[str]:
        """Parse allowed origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("data_dir", "hf_home", "hf_cache_dir", mode="after")
    @classmethod
    def ensure_path_is_absolute(cls, v: Path) -> Path:
        """Ensure paths are absolute."""
        if not v.is_absolute():
            return v.absolute()
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def is_test(self) -> bool:
        """Check if running in test mode."""
        return self.environment == "test"

    @property
    def datasets_dir(self) -> Path:
        """Get datasets storage directory."""
        return self.data_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        """Get models storage directory."""
        return self.data_dir / "models"

    @property
    def activations_dir(self) -> Path:
        """Get activations storage directory."""
        return self.data_dir / "activations"

    @property
    def checkpoints_dir(self) -> Path:
        """Get checkpoints storage directory."""
        return self.data_dir / "checkpoints"

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.data_dir,
            self.datasets_dir,
            self.models_dir,
            self.activations_dir,
            self.checkpoints_dir,
            self.hf_home,
            self.hf_cache_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()
