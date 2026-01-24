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
    backend_dir: Path = Field(
        default=Path("/app"), description="Backend application directory (where src/ lives)"
    )
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

    # OpenAI Configuration (for feature labeling)
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for GPT-based feature labeling (optional, can also be set per extraction)"
    )

    # Dataset Configuration
    auto_cleanup_after_download: bool = Field(
        default=True,
        description="Automatically cleanup HuggingFace cache and downloads after dataset download"
    )

    # Internal API Configuration (for Celery workers)
    # NOTE: Default is localhost for native mode. Override for containerized deployments:
    #   - Docker Compose: INTERNAL_API_URL=http://backend:8000
    #   - Kubernetes: INTERNAL_API_URL=http://mistudio-backend:8000
    internal_api_url: str = Field(
        default="http://localhost:8000",
        description="Internal API URL for Celery workers to communicate with backend"
    )

    # Ollama/oLLM Configuration
    # NOTE: Default is localhost for native mode. Override for containerized deployments:
    #   - Docker Compose: OLLAMA_URL=http://ollm:11434
    #   - Kubernetes: OLLAMA_URL=http://ollama-proxy:11434 (ExternalName service to ollama namespace)
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API URL for local LLM inference"
    )

    # Neuronpedia Local Instance Configuration
    # NOTE: For K8s deployment, use the service name in the neuronpedia namespace
    neuronpedia_local_db_url: str | None = Field(
        default=None,
        description="PostgreSQL connection URL for local Neuronpedia instance (e.g., postgresql://neuronpedia:pass@neuronpedia-postgres-service.neuronpedia:5432/neuronpedia)"
    )
    neuronpedia_local_url: str | None = Field(
        default=None,
        description="Public URL for local Neuronpedia instance (e.g., http://neuron.mcslab.io)"
    )
    neuronpedia_local_admin_user_id: str = Field(
        default="mistudio-admin",
        description="User ID to use as creator when pushing to local Neuronpedia"
    )


    # WebSocket Configuration
    websocket_ping_interval: int = Field(
        default=30, ge=10, le=300, description="WebSocket ping interval in seconds"
    )
    websocket_ping_timeout: int = Field(
        default=10, ge=5, le=60, description="WebSocket ping timeout in seconds"
    )

    @property
    def websocket_emit_url(self) -> str:
        """WebSocket emission endpoint URL derived from internal_api_url."""
        return f"{self.internal_api_url}/api/internal/ws/emit"

    # System Monitoring Configuration
    system_monitor_interval_seconds: int = Field(
        default=2, ge=1, le=30, description="System metrics collection interval in seconds (via WebSocket)"
    )

    # Extraction Progress Configuration
    extraction_progress_interval_seconds: int = Field(
        default=5, ge=1, le=60, description="Extraction progress WebSocket emission interval in seconds"
    )

    # Steering Configuration
    steering_timeout_seconds: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Timeout for steering generation requests in seconds (default 120s, max 10 minutes)"
    )

    # Token and Feature Filtering Configuration
    # Stage 1: Tokenization-time filtering (conservative, permanent)
    tokenization_filter_enabled: bool = Field(
        default=False,
        description="Enable token filtering during dataset tokenization (permanent filtering)"
    )
    tokenization_filter_mode: Literal["minimal", "conservative"] = Field(
        default="conservative",
        description="Tokenization filter mode: minimal (control chars only), conservative (+ whitespace)"
    )
    tokenization_junk_ratio_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Skip samples if >X% of tokens are junk during tokenization (0.0-1.0)"
    )

    # Stage 2: Extraction-time token filtering (zero-tolerance, affects SAE training)
    extraction_filter_enabled: bool = Field(
        default=False,
        description="Enable token filtering during feature extraction (prevents junk tokens from SAE analysis)"
    )
    extraction_filter_mode: Literal["minimal", "conservative", "standard", "aggressive"] = Field(
        default="standard",
        description="Extraction filter mode: minimal/conservative/standard/aggressive"
    )

    # Stage 3: Pre-labeling feature filtering (aggressive, reversible)
    pre_labeling_filter_enabled: bool = Field(
        default=True,
        description="Enable feature filtering before LLM labeling (saves API costs)"
    )
    pre_labeling_junk_ratio_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Skip features if >X% of top tokens are junk (0.0-1.0)"
    )
    pre_labeling_single_char_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Skip features if >X% of top tokens are single char (0.0-1.0)"
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

    @property
    def run_dir(self) -> Path:
        """Get runtime directory for PID files and temporary logs.

        Uses data_dir/run to ensure it's writable in all deployment modes.
        """
        return self.data_dir / "run"

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
            self.run_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_data_path(self, path: str | Path) -> Path:
        """
        Resolve a data path to an absolute path using data_dir.

        Handles paths stored in the database that may be:
        - Already correct absolute paths (returned as-is if they exist)
        - Docker-style paths starting with "/data/" (converted to use data_dir)
        - Relative paths starting with "data/" (prefix stripped and joined with data_dir)
        - Other relative paths (joined with data_dir directly)

        This is essential for containerized environments where DATA_DIR
        may differ from the path prefix stored in the database.

        Args:
            path: Path string or Path object to resolve

        Returns:
            Absolute Path object
        """
        path_obj = Path(path) if isinstance(path, str) else path
        path_str = str(path_obj)

        # Handle Docker-style absolute paths like "/data/datasets/..."
        # These are absolute in containers but need resolution in native mode
        if path_str.startswith("/data/"):
            # Strip the /data/ prefix and join with actual data_dir
            relative_path = path_str[6:]  # Remove "/data/" prefix
            return self.data_dir / relative_path

        # If it's a real absolute path that exists, use it directly
        if path_obj.is_absolute():
            if path_obj.exists():
                return path_obj
            # If absolute path doesn't exist, try resolving as relative
            # (in case it was stored with wrong absolute prefix)

        # Handle paths stored with "data/" prefix (legacy format)
        if path_str.startswith("data/"):
            path_str = path_str[5:]  # Remove "data/" prefix

        return self.data_dir / path_str


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()
