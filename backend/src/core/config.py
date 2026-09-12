"""Configuration management using Pydantic Settings.

This module provides typed configuration loaded from environment variables.
All settings are validated at startup to fail fast if misconfigured.
"""

import os
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
        default="http://localhost:8000", description="Public API base URL"
    )

    # CORS Configuration
    allowed_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost",
            # THE DEPLOYED HOSTNAMES MUST BE HERE.
            #
            # This list gained a second consumer in MIS-E2E-105: Socket.IO's
            # `cors_allowed_origins` moved from "*" to this setting. Nothing had
            # depended on it before, so it still listed only localhost — and
            # every browser on a real hostname got **403 on the WebSocket
            # upgrade**. socket.io then falls back to polling forever, which
            # LOOKS connected: REST works, the page renders, and only pushed
            # events silently never arrive. Reported 2026-08-25 as "progress
            # only updates when I refresh".
            #
            # A CLI client sends no Origin header and is allowed, so probing
            # with curl or python-socketio does not reproduce it. Only a browser
            # does.
            "http://k8s-mistudio.hitsai.local",
            "https://k8s-mistudio.hitsai.local",
            "http://mistudio.hitsai.local",
            "https://mistudio.hitsai.net",
            "http://mistudio.hitsai.net",
        ],
        description=(
            "Allowed CORS origins. Consumed by Socket.IO's cors_allowed_origins "
            "as well as HTTP CORS — an origin missing here cannot upgrade to a "
            "WebSocket and silently degrades to polling-only."
        ),
    )

    # Security
    secret_key: str = Field(
        min_length=32, description="Secret key for signing tokens (min 32 characters)"
    )
    access_token_expire_minutes: int = Field(
        default=30, ge=1, le=10080, description="Access token expiration in minutes"
    )
    bypass_settings_pin: bool = Field(
        default=False,
        description=(
            "Bypass the Settings panel PIN gate. Set MISTUDIO_BYPASS_PIN=true to recover "
            "from a forgotten PIN — requires filesystem access to the server. "
            "Remove after resetting the PIN and restart the backend."
        ),
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
        description="Public URL for local Neuronpedia instance (e.g., http://neuron.hitsai.local)"
    )
    neuronpedia_local_admin_user_id: str = Field(
        default="cljj57d3c000076ei38vwnv35",
        description="User ID to use as creator when pushing to local Neuronpedia. "
        "Must be in Neuronpedia's PUBLIC_ACTIVATIONS_USER_IDS list for activations to be visible. "
        "Default is the SAELens creator ID which is pre-approved in Neuronpedia."
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

    @property
    def internal_api_secret(self) -> str:
        """Shared secret for internal Celery → backend API calls.

        Derived deterministically from secret_key so no extra env var is needed.
        All containers share the same secret_key, so this is consistent across
        the backend and all Celery workers.
        """
        import hashlib
        import hmac as _hmac
        return _hmac.new(
            self.secret_key.encode("utf-8"),
            b"mistudio-internal-api-v1",
            hashlib.sha256,
        ).hexdigest()

    # System Monitoring Configuration
    system_monitor_interval_seconds: int = Field(
        default=2, ge=1, le=30, description="System metrics collection interval in seconds (via WebSocket)"
    )

    # Extraction Progress Configuration
    extraction_progress_interval_seconds: int = Field(
        default=5, ge=1, le=60, description="Extraction progress WebSocket emission interval in seconds"
    )

    # Steering Configuration
    # Feature 013 (IDL-29): cluster strength-model constants. JSON:
    # {"default": {"a","b","m","M","cohesion_gate"}, "per_sae": {"<sae_id>": {...}}}
    # Per-SAE calibration (MCP validation protocol) writes here — no code change.
    steering_cluster_constants_json: str = Field(
        default="{}",
        description="Overrides for the cluster strength budget model constants",
    )

    steering_timeout_seconds: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Timeout for steering generation requests in seconds (default 120s, max 10 minutes)"
    )

    # Feature 015 (IDL-32/IDL-35): cross-layer steering hazard heuristic. The
    # weight-prior fallback warns when |cos(W_dec(Lᵢ)[:,i], W_enc(Lⱼ)[j,:])| ≥
    # this threshold for a co-steered upstream→downstream pair. Labeled
    # `heuristic` — never causal. Validated circuit edges (rung ≥2) always win.
    steering_hazard_prior_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight-prior threshold above which a cross-layer steering pair is flagged (015 heuristic)",
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

    # Batched labeling: how many features share one miLLM forward pass.
    # 1 disables batching and restores the per-feature request path.
    #
    # 8 is the measured default — 5.59x aggregate throughput on gemma-4-12B-it
    # with 4.8 GB of headroom on a 24 GB card. 12 reaches 7.31x with 1.7 GB
    # left and 16 OOMs, so the ceiling is deliberately below what fits.
    #
    # Bulk labeling ONLY. Batch composition changes greedy output under int8
    # quantisation, so a labeling TRIAL — where the template must be the only
    # variable — has to stay serial. That separation is structural rather than
    # a matter of setting this correctly: LabelingTrialService calls
    # generate_label_from_examples directly and never reaches the batched path.
    labeling_batch_size: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Features per batched labeling request (1 = no batching)"
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
    def jlens_artifacts_dir(self) -> Path:
        """Root of the J-lens artifact registry (PADR IDL-46).

        THE FILESYSTEM IS THE REGISTRY. This directory is what a consumer
        mounts; there is no upload path and no database table that could
        disagree with it. One subdirectory per model slug, each holding exactly
        one `<slug>_jacobian_lens.pt` plus its `config.yaml`.
        """
        return self.data_dir / "jlens"

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

    def resolve_user_path(self, user_path: str | Path) -> Path:
        """
        Resolve a path supplied by an external user (e.g. an API request body).

        Performs **string-only** normalization of the user input — never
        touches the filesystem with raw user data — then performs an
        allow-list containment check against the trusted roots
        (``data_dir``, ``run_dir``, ``hf_cache_dir``). Use this for any path
        that originated from outside the trust boundary; use
        ``resolve_data_path`` for paths read from the database or
        constructed by the system itself.

        Implementation notes:
        - Uses ``os.path.normpath`` (pure-string) instead of ``Path.resolve``
          so we don't probe the filesystem (incl. symlinks) before the
          containment check has succeeded.
        - The ``..`` segment check is performed on the post-normalization
          path under each trusted root; if any segment escapes, normpath's
          collapse will land the candidate outside the root and the
          ``commonpath`` check rejects it.

        Raises:
            ValueError: if the resolved path is not contained inside any
                       trusted root.
        """
        path_str = str(user_path)

        # Strip Docker-style "/data/" or legacy "data/" prefix so the
        # remainder is treated as data_dir-relative regardless of how the
        # caller framed the input.
        if path_str.startswith("/data/"):
            path_str = path_str[6:]
        elif path_str.startswith("data/"):
            path_str = path_str[5:]
        elif path_str.startswith("/"):
            # Bare absolute path (e.g. "/etc/passwd") — strip leading slashes
            # and treat as relative; the containment check will reject it
            # because data_dir/etc/passwd won't exist under any trusted root
            # (and even if it did, normpath wouldn't escape data_dir).
            path_str = path_str.lstrip("/")

        # Pure-string normalization — collapses "..", "." and duplicate slashes
        # without consulting the filesystem.
        data_dir_real = os.path.realpath(str(self.data_dir))
        candidate_str = os.path.normpath(os.path.join(data_dir_real, path_str))

        # Allow-list containment: candidate must live under a trusted root.
        trusted_roots = [
            os.path.realpath(str(self.data_dir)),
            os.path.realpath(str(self.run_dir)),
            os.path.realpath(str(self.hf_cache_dir)),
        ]
        for root in trusted_roots:
            try:
                if os.path.commonpath([candidate_str, root]) == root:
                    return Path(candidate_str)
            except ValueError:
                # commonpath raises if the paths are on different drives (Windows).
                continue
        raise ValueError(
            f"Path {user_path!r} resolves outside the trusted data roots"
        )

    def resolve_deletable_path(self, path: str | Path, *, min_depth: int = 2) -> Path:
        """Resolve a path that is about to be handed to ``rmtree``.

        MIS-E2E-071. ``resolve_data_path`` is documented for "paths read from
        the database or constructed by the system itself" — but the database is
        NOT a trust boundary here: ``raw_path``, ``file_path``,
        ``quantized_path`` and ``tokenized_path`` are all writable through
        unauthenticated create/update endpoints that blind-``setattr`` onto the
        ORM row. So a value reaches this sink having been *stored*, not having
        been *validated*, and ``resolve_data_path`` returns an existing absolute
        path verbatim. ``POST {"raw_path": "/"}`` then ``DELETE`` was two
        ordinary-looking requests.

        Containment alone is not sufficient for a DELETE. ``resolve_user_path``
        maps ``""`` to ``data_dir`` itself and ``"datasets"`` to the directory
        holding every dataset — both pass a containment check and both are
        catastrophic to ``rmtree``. ``min_depth`` is what makes this a deletion
        guard rather than a containment check: the target must be at least that
        many segments BELOW its trusted root, so the roots and their top-level
        category directories cannot be named at all.

        The realpath re-check closes the symlink gap that the deliberately
        string-only ``resolve_user_path`` leaves open: it never touches the
        filesystem, which is correct before containment succeeds, but ``rmtree``
        does traverse intermediate symlinked components.

        Raises:
            ValueError: if the path is empty, escapes the trusted roots, or is
                        too shallow to be a legitimate deletion target.
        """
        if path is None or str(path).strip() in ("", "/", "."):
            raise ValueError(f"Refusing to delete {path!r}: empty or root path")

        trusted_roots = [
            os.path.realpath(str(self.data_dir)),
            os.path.realpath(str(self.run_dir)),
            os.path.realpath(str(self.hf_cache_dir)),
        ]

        # An ALREADY-CONTAINED absolute path is taken as-is.
        #
        # `resolve_user_path` strips the leading slash and re-joins under
        # data_dir, which is right for the untrusted input it was written for
        # but wrong here: the workers store genuinely absolute paths, so
        # `/…/backend/data/models/foo` would become
        # `data_dir/…/backend/data/models/foo` — a path that does not exist, so
        # every real deletion would silently no-op while reporting success.
        # Caught by the cleanup integration tests, which is exactly what they
        # are for.
        raw = os.path.normpath(str(path))
        candidate = None
        if os.path.isabs(raw):
            for root in trusted_roots:
                try:
                    if os.path.commonpath([raw, root]) == root:
                        candidate = Path(raw)
                        break
                except ValueError:
                    continue

        if candidate is None:
            # Relative, or absolute-but-outside (including docker-style
            # "/data/…", which is absolute in a container and not a real root
            # here). Fall through to the untrusted-input semantics, which
            # relativize and then enforce containment.
            candidate = self.resolve_user_path(path)

        # Depth, measured under whichever root contains it.
        for root in trusted_roots:
            try:
                if os.path.commonpath([str(candidate), root]) != root:
                    continue
            except ValueError:
                continue
            rel = os.path.relpath(str(candidate), root)
            depth = len([seg for seg in rel.split(os.sep) if seg not in ("", ".")])
            if depth < min_depth:
                raise ValueError(
                    f"Refusing to delete {path!r}: resolves to {candidate}, only "
                    f"{depth} level(s) below {root} (minimum {min_depth}). A "
                    f"trusted root or a top-level category directory is never a "
                    f"legitimate deletion target."
                )
            break

        # Symlink re-check: only meaningful once the path exists, and only
        # matters for a sink that traverses. A missing path is fine — the caller
        # checks existence — but a path that EXISTS and resolves outside is not.
        if os.path.exists(str(candidate)):
            real = os.path.realpath(str(candidate))
            if not any(
                os.path.commonpath([real, r]) == r
                for r in trusted_roots
                if os.path.exists(r)
            ):
                raise ValueError(
                    f"Refusing to delete {path!r}: {candidate} is a symlink to "
                    f"{real}, outside the trusted data roots"
                )

        return candidate


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()
