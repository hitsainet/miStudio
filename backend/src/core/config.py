"""Configuration management using Pydantic Settings.

This module provides typed configuration loaded from environment variables.
All settings are validated at startup to fail fast if misconfigured.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, PostgresDsn, RedisDsn, field_validator, model_validator
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

    # Redis Configuration
    redis_url: RedisDsn = Field(description="Redis connection URL")

    # Celery Configuration
    celery_broker_url: RedisDsn = Field(description="Celery broker URL (Redis)")
    celery_result_backend: RedisDsn = Field(description="Celery result backend URL (Redis)")
    # Multi-GPU Phase 3 (0xcc/plans/Multi-GPU-Plan.md). "per_card": one solo
    # worker per GPU (workers/gpu_supervisor.py), GPU jobs dispatched to
    # gpu.<uuid> / gpu.auto and run under a GPU lease. "single": one worker
    # consumes every GPU queue, as before Phase 3 (docker-compose, development).
    gpu_worker_mode: Literal["single", "per_card"] = Field(
        default="single",
        description="GPU worker topology: 'single' (one worker) or 'per_card' (one per GPU)",
    )

    # Data Storage Paths
    data_dir: Path = Field(default=Path("/data"), description="Root data directory")
    backend_dir: Path = Field(
        default=Path("/app"), description="Backend application directory (where src/ lives)"
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
    # THE DOCUMENTED NAME IS THE PRIMARY ONE. Every document — the manual, the
    # PADR, the PPRD, this description and the PIN gate's own recovery text —
    # tells an operator who forgot the PIN to set `MISTUDIO_BYPASS_PIN=true`. The
    # field had no alias, so pydantic-settings read only `BYPASS_SETTINGS_PIN`
    # and the documented recovery path did nothing. Both names are accepted;
    # when both are set, `MISTUDIO_BYPASS_PIN` wins (AliasChoices takes the
    # first one present). The field name stays last so `Settings(
    # bypass_settings_pin=...)` keeps working.
    bypass_settings_pin: bool = Field(
        default=False,
        validation_alias=AliasChoices("MISTUDIO_BYPASS_PIN", "bypass_settings_pin"),
        description=(
            "Bypass the Settings panel PIN gate. Set MISTUDIO_BYPASS_PIN=true to recover "
            "from a forgotten PIN — requires filesystem access to the server. "
            "Remove after resetting the PIN and restart the backend. "
            "BYPASS_SETTINGS_PIN is also accepted; MISTUDIO_BYPASS_PIN wins if both are set."
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

    # The judge-prompt library's optional rubric import (PADR IDL-57). Empty = miDataworks is not
    # connected, which is a normal state: the import refuses with a reason and a rubric FILE can
    # still be imported (memory: midataworks-standalone-first). Same variable the MCP pod reads.
    midataworks_api_url: str = Field(
        default="",
        description="miDataworks' base URL, for importing its rubrics as judge prompts",
    )
    # Feature 037/038 (ENH-005 D16, D28): miLLM's base URL, read by the BACKEND (the MCP pod reads its
    # own MILLM_API_URL). The behaviour checklist asks which probes miLLM has armed; spot-checks read
    # its recorded session turns. Empty = miLLM not connected, a normal state: the checklist's
    # "armed" step reads "unknown" and a spot-check refuses with 503 naming miLLM.
    millm_api_url: str = Field(
        default="",
        description="miLLM's base URL, for the behaviour checklist and session spot-checks",
    )
    # Judge-prompt seed files (PADR IDL-57 amendment, 2026-10-21): a folder of
    # `mistudio.judge-prompt/v1` documents loaded at API startup, after the built-in ones in
    # `src/data/judge_prompt_seeds/`. A file whose name and version is already stored with other
    # text is refused with a warning (a version never changes); nothing is ever deleted.
    judge_prompt_seed_dir: Path | None = Field(
        default=None,
        description=(
            "An extra folder of judge-prompt files (mistudio.judge-prompt/v1, the export format) "
            "loaded at startup. Empty: only the built-in starter prompts."
        ),
    )

    # Feature 034 (FR-20; P-08 by analogy): how long a caller-executed agent approval
    # (`millm_load_model`) may wait — pending, or approved and not yet consumed — before it reads
    # `expired`. Steering approvals never expire (operator decision S3-03, 2026-10-06).
    mcp_approval_ttl_hours: int = Field(
        default=24, ge=1,
        description="Hours before a pending or unconsumed model-load approval expires",
    )

    # OpenAI-compatible / Ollama endpoint for local labeling, used when no
    # `ollama_url` app setting is stored. No Ollama is deployed any more (removed
    # 2026-09-13 from the cluster and the compose files); docker-compose.yml points
    # this at miLLM's OpenAI-compatible API instead.
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

    # SAE training's activation buffer (Feature 036, ENH-003 §3.1). These were raw
    # `os.environ` reads in `services/activation_buffer.py`, so no override could reach
    # them; they are fields now, read through `core.runtime_settings.effective()` when each
    # buffer is built (the next training task). The environment names are unchanged.
    activation_read_threads: int = Field(
        default=12,
        ge=1,
        le=256,
        description=(
            "Threads reading extraction rows into the SAE training buffer; capped at the "
            "worker's CPU count when used"
        ),
    )
    activation_prefetch: bool = Field(
        default=True,
        description="Prepare the next training buffer in the background while the current one trains",
    )

    # ── Feature 040 (FR-7): behaviour-detector caps. Runtime keys (RUNTIME_KEYS), read ONLY through
    # `runtime_settings.effective()`; a request past one is refused `over_cap`, never truncated. D32:
    # starting values, provisional until measured. ──────────────────────────────────────────────
    behaviour_max_sub_questions: int = Field(
        default=32, ge=1, le=256, description="Sub-questions one behaviour version may declare"
    )
    behaviour_max_success_criteria: int = Field(
        default=64, ge=1, le=512, description="Success criteria one behaviour version may declare"
    )
    behaviour_max_held_out_columns: int = Field(
        default=16, ge=1, le=64, description="Held-out columns one behaviour version may declare"
    )
    behaviour_max_models: int = Field(
        default=16, ge=1, le=64, description="Models one behaviour version may name"
    )
    rule_set_max_bytes: int = Field(
        default=1024 * 1024, ge=1024, le=16 * 1024 * 1024,
        description="Largest rule-set document (canonical bytes) a behaviour version may carry",
    )
    behaviour_list_page_default: int = Field(
        default=100, ge=1, le=500, description="Behaviours per catalogue page when no limit is given"
    )
    behaviour_list_page_max: int = Field(
        default=500, ge=1, le=500, description="Most behaviours one catalogue page may return"
    )
    # ── Feature 040 S2 ──
    calibration_floor_negatives: int = Field(
        default=2000, ge=1, le=10_000_000,
        description="Fewest calibration negatives a bar may rest on before it is flagged few_negatives",
    )
    breakdown_min_subgroup_rows: int = Field(
        default=30, ge=1, le=1_000_000, description="Groups with fewer rows are flagged too_small in a breakdown"
    )
    breakdown_page_default: int = Field(default=100, ge=1, le=500, description="Breakdown groups per page by default")
    breakdown_page_max: int = Field(default=500, ge=1, le=500, description="Most breakdown groups one page may return")
    breakdown_max_filters: int = Field(default=8, ge=1, le=32, description="Most filters one breakdown may apply")
    group_column_max_values: int = Field(
        default=10_000, ge=2, le=100_000, description="Most distinct values a declared grouping column may hold"
    )
    view_max_group_columns: int = Field(default=32, ge=1, le=256, description="Most grouping columns one view may declare")
    report_list_page_default: int = Field(default=50, ge=1, le=200, description="Success reports per page by default")
    report_list_page_max: int = Field(default=200, ge=1, le=200, description="Most success reports one page may return")
    success_report_max_bytes: int = Field(
        default=1024 * 1024, ge=4096, le=16 * 1024 * 1024, description="Largest stored success report (bytes)"
    )
    # ── Feature 041 (FR-27; T12/T16, D32 starting values): probe families and blend evaluation. Runtime
    # keys, read ONLY through `runtime_settings.effective()`; a request past one is refused
    # `cap_exceeded` naming the setting, never truncated. ──────────────────────────────────────────
    blend_family_page_size: int = Field(
        default=50, ge=1, le=500, description="Family members per page (default and most)"
    )
    blend_max_members: int = Field(default=16, ge=2, le=64, description="Most members one blend evaluation may take")
    blend_max_candidates: int = Field(default=48, ge=1, le=256, description="Most candidates one blend evaluation may rank")
    blend_max_concurrent: int = Field(
        default=2, ge=1, le=8, description="Blend evaluations queued or running at once (independent of the worker's concurrency)"
    )
    blend_folds_default: int = Field(default=5, ge=2, le=100, description="Cross-validation folds of a blend evaluation")
    blend_folds_min: int = Field(default=3, ge=2, le=100, description="Fewest folds a blend evaluation may ask for")
    blend_folds_max: int = Field(default=10, ge=2, le=100, description="Most folds a blend evaluation may ask for")
    blend_cv_min_rows_per_class: int = Field(
        default=20, ge=5, le=1000, description="Fewest rows per class in a held-out fold before cross-validation reads not assessable"
    )
    blend_min_aligned_fraction: float = Field(
        default=0.95, ge=0.5, le=1.0, description="Fewest of its rows a member must share with the others on a view"
    )
    blend_report_page_size: int = Field(default=25, ge=1, le=200, description="Blend reports per page (default and most)")
    blend_scores_page_limit: int = Field(
        default=50_000, ge=1000, le=200_000, description="Most per-row blend values one page may return"
    )
    probe_score_batch_cap: int = Field(
        default=48, ge=1, le=512,
        description="Most offline scores one batch may queue (inputs for one probe; members x inputs for a family)",
    )
    blend_evaluation_time_limit_s: int = Field(
        default=1800, ge=60, le=21_600, description="Seconds a blend evaluation may stay queued or running before it is failed"
    )

    # Environment
    environment: Literal["development", "production", "test"] = Field(
        default="development", description="Application environment"
    )

    @model_validator(mode="after")
    def _blend_folds_ordered(self):
        """Feature 041 (FTID §9): the fold bounds must hold their default — a runtime change that breaks
        `blend_folds_min <= blend_folds_default <= blend_folds_max` is refused, never half-applied."""
        if not (self.blend_folds_min <= self.blend_folds_default <= self.blend_folds_max):
            raise ValueError(
                f"blend_folds_min ({self.blend_folds_min}) <= blend_folds_default ({self.blend_folds_default}) "
                f"<= blend_folds_max ({self.blend_folds_max}) must hold"
            )
        return self

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: str | list[str]) -> list[str]:
        """Parse allowed origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("judge_prompt_seed_dir", mode="before")
    @classmethod
    def _blank_seed_dir_is_none(cls, value):
        """`JUDGE_PROMPT_SEED_DIR=` (set but empty) means none — not `Path("")`, the working
        directory, whose stray *.json files would otherwise be read as judge prompts."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    @field_validator("data_dir", "hf_cache_dir", mode="after")
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


# ─── Runtime environment settings (Feature 036; ENH-003 D2, D4, D6, D7) ──────────────────────────
#
# EVERY `Settings` FIELD IS CLASSIFIED EXACTLY ONCE below, and
# `tests/unit/test_settings_readers.py::test_every_field_is_classified` fails on a field that is
# not, so adding a setting forces the decision: can the operator change it while miStudio runs,
# does it need a restart, is it shown read-only, or is it never on the page at all?


@dataclass(frozen=True)
class RuntimeKey:
    """One setting the operator may override while miStudio runs."""

    label: str
    description: str
    #: When a change takes effect — one of the spec's fixed phrases, shown as the row's badge.
    effect: str
    #: The processes that read it (for the page; the refresh points reach all of them).
    processes: tuple[str, ...]


#: **The settings an operator may change at runtime** through `PUT /api/v1/environment/{key}`
#: (operator decision D2, 2026-10-20). Each is read through `core.runtime_settings.effective()`
#: by every reader, and each reader has a wiring test in `test_environment_readers.py`.
RUNTIME_KEYS: dict[str, RuntimeKey] = {
    "mcp_approval_ttl_hours": RuntimeKey(
        "Agent approval lifetime (hours)",
        "How long an agent's request that waits for your approval (a model load, a probe run) stays "
        "usable before it expires.",
        "next approval check", ("API",),
    ),
    "steering_timeout_seconds": RuntimeKey(
        "Steering time limit (seconds)",
        "How long one steering generation may run. The steering worker stops a generation that runs "
        "30 seconds past this limit.",
        "next generation", ("API", "steering workers"),
    ),
    "steering_hazard_prior_threshold": RuntimeKey(
        "Cross-layer warning threshold",
        "How similar two features' weights must be (0 to 1) before steering them together across "
        "layers is flagged as a possible interaction. A heuristic, never proof.",
        "next request", ("API",),
    ),
    "steering_cluster_constants_json": RuntimeKey(
        "Cluster strength constants (JSON)",
        'Overrides for the cluster strength model, as JSON: {"default": {...}, "per_sae": '
        '{"<sae id>": {...}}}. Keys: a, b, m, M, cohesion_gate, gamma.',
        "next request", ("API",),
    ),
    "labeling_batch_size": RuntimeKey(
        "Labeling batch size",
        "How many features share one request to the labeling model (1 turns batching off). Larger "
        "batches are faster but use more graphics memory.",
        "next labeling job", ("general worker",),
    ),
    "auto_cleanup_after_download": RuntimeKey(
        "Clean up after dataset downloads",
        "Delete the Hugging Face download cache once a dataset has been saved.",
        "next download", ("general worker",),
    ),
    "ollama_url": RuntimeKey(
        "Labeling model service address",
        "Base address of the OpenAI-compatible or Ollama service the Labeling screen lists models "
        "from.",
        "next request", ("API",),
    ),
    "neuronpedia_local_url": RuntimeKey(
        "Local Neuronpedia address",
        "Public address of your own Neuronpedia instance, used for links after a push. Leave empty if "
        "you do not run one.",
        "next request or push", ("API", "general worker"),
    ),
    "neuronpedia_local_admin_user_id": RuntimeKey(
        "Local Neuronpedia creator id",
        "The Neuronpedia user recorded as the creator of pushed features. It must be one Neuronpedia "
        "shows publicly.",
        "next push", ("general worker",),
    ),
    "system_monitor_interval_seconds": RuntimeKey(
        "System monitor interval (seconds)",
        "How often the server measures processor, memory, disk and graphics card use for the System "
        "Monitor.",
        "next monitor tick", ("API",),
    ),
    "log_level": RuntimeKey(
        "Log level",
        "How much the server writes to its logs. DEBUG is the most, CRITICAL the least.",
        "API now; workers at their next task", ("API", "all workers"),
    ),
    "activation_read_threads": RuntimeKey(
        "Training buffer read threads",
        "How many threads read stored activations into the training buffer. Capped at the worker's "
        "processor count.",
        "next training task", ("graphics card workers",),
    ),
    # ── Feature 040 S2 (FR-7, FR-19) ──
    "calibration_floor_negatives": RuntimeKey(
        "Calibration floor (negatives)",
        "Fewest negatives a probe's bar may be cut from before its success criteria read not assessable "
        "(few negatives). A behaviour can set its own floor.",
        "next request", ("API",),
    ),
    "breakdown_min_subgroup_rows": RuntimeKey(
        "Smallest reported group (rows)",
        "Groups with fewer rows are still shown in a breakdown, flagged too small.",
        "next request", ("API",),
    ),
    "breakdown_page_default": RuntimeKey(
        "Breakdown groups per page (default)", "How many groups one breakdown page shows when not asked.",
        "next request", ("API",),
    ),
    "breakdown_page_max": RuntimeKey(
        "Breakdown groups per page (most)", "The most groups one breakdown page may show. A larger page is refused.",
        "next request", ("API",),
    ),
    "breakdown_max_filters": RuntimeKey(
        "Breakdown filters (most)", "The most column filters one breakdown may apply.",
        "next request", ("API",),
    ),
    "group_column_max_values": RuntimeKey(
        "Distinct values per grouping column (most)",
        "A grouping column with more distinct values is refused, so a row id cannot become a group.",
        "next request", ("API",),
    ),
    "view_max_group_columns": RuntimeKey(
        "Grouping columns per view (most)", "The most grouping columns one view may declare.",
        "next request", ("API",),
    ),
    "report_list_page_default": RuntimeKey(
        "Success reports per page (default)", "How many stored success reports one page lists when not asked.",
        "next request", ("API",),
    ),
    "report_list_page_max": RuntimeKey(
        "Success reports per page (most)", "The most stored success reports one page may list.",
        "next request", ("API",),
    ),
    "success_report_max_bytes": RuntimeKey(
        "Largest success report (bytes)", "A success report larger than this is refused, never truncated.",
        "next request", ("API",),
    ),
    # ── Feature 041 (FR-27) ──
    "blend_family_page_size": RuntimeKey(
        "Family members per page",
        "How many members of a behaviour's probe family one page lists, and the most a caller may ask "
        "for. A larger page is refused, naming this setting.",
        "next request", ("API",),
    ),
    "blend_max_members": RuntimeKey(
        "Members per blend evaluation",
        "The most probes one blend evaluation may combine. A larger family is refused, naming this setting; "
        "name fewer members.",
        "next blend evaluation", ("API",),
    ),
    "blend_max_candidates": RuntimeKey(
        "Candidates per blend evaluation",
        "The most candidate blends one evaluation may rank. A longer list is refused, naming this setting.",
        "next blend evaluation", ("API",),
    ),
    "blend_max_concurrent": RuntimeKey(
        "Blend evaluations at once",
        "How many blend evaluations may be queued or running together; one more is refused until one "
        "finishes. Independent of the blend worker's own concurrency (2 when deployed): set above it, the "
        "extra evaluations wait queued rather than run in parallel.",
        "next blend evaluation", ("API",),
    ),
    "blend_folds_default": RuntimeKey(
        "Cross-validation folds (default)",
        "How many folds a blend evaluation's cross-validation uses when the caller does not say.",
        "next blend evaluation", ("API",),
    ),
    "blend_folds_min": RuntimeKey(
        "Cross-validation folds (fewest)",
        "The fewest folds a caller may ask for; fewer give no estimate of the spread.",
        "next blend evaluation", ("API",),
    ),
    "blend_folds_max": RuntimeKey(
        "Cross-validation folds (most)",
        "The most folds a caller may ask for; more thin out the groups each fold holds.",
        "next blend evaluation", ("API",),
    ),
    "blend_cv_min_rows_per_class": RuntimeKey(
        "Rows per class in a fold",
        "A held-out fold with fewer positives or negatives than this makes the cross-validated figures read "
        "not assessable (they are still reported with their counts).",
        "next blend evaluation", ("API", "blend worker"),
    ),
    "blend_min_aligned_fraction": RuntimeKey(
        "Rows shared between members",
        "The share of its scored rows each member must have in common with the others on a view; below it "
        "the evaluation is refused as misaligned.",
        "next blend evaluation", ("API",),
    ),
    "blend_report_page_size": RuntimeKey(
        "Blend reports per page",
        "How many blend reports one page lists, and the most a caller may ask for.",
        "next request", ("API",),
    ),
    "blend_scores_page_limit": RuntimeKey(
        "Blend row values per page (most)",
        "The most per-row values of one blend candidate a single page may return.",
        "next request", ("API",),
    ),
    "probe_score_batch_cap": RuntimeKey(
        "Offline scores per batch (most)",
        "The most offline scores one batch may queue: inputs for one probe, or members times inputs for a "
        "family. Each is a forward pass in the graphics card queue; a larger batch is refused naming both numbers.",
        "next request", ("API",),
    ),
    "blend_evaluation_time_limit_s": RuntimeKey(
        "Blend evaluation time limit (seconds)",
        "A blend evaluation still queued or running after this long is failed with the reason time limit, "
        "freeing its place. It is never restarted.",
        "next janitor sweep", ("beat scheduler",),
    ),
    # ── Feature 040 S1 (FR-7) ──
    "behaviour_max_sub_questions": RuntimeKey(
        "Sub-questions per behaviour version",
        "The most sub-questions one version of a behaviour may declare. A longer list is refused, "
        "naming this setting.",
        "next request", ("API",),
    ),
    "behaviour_max_success_criteria": RuntimeKey(
        "Success criteria per behaviour version",
        "The most success criteria one version of a behaviour may declare. A longer list is refused.",
        "next request", ("API",),
    ),
    "behaviour_max_held_out_columns": RuntimeKey(
        "Held-out columns per behaviour version",
        "The most held-out columns one version of a behaviour may declare. A longer list is refused.",
        "next request", ("API",),
    ),
    "behaviour_max_models": RuntimeKey(
        "Models per behaviour version",
        "The most models one version of a behaviour may name. A longer list is refused.",
        "next request", ("API",),
    ),
    "rule_set_max_bytes": RuntimeKey(
        "Largest rule set (bytes)",
        "The largest rule-set document a behaviour version may carry, measured as its canonical "
        "bytes. A larger one is refused.",
        "next request", ("API",),
    ),
    "behaviour_list_page_default": RuntimeKey(
        "Behaviours per page (default)",
        "How many behaviours one page of the catalogue lists when the caller does not say.",
        "next request", ("API",),
    ),
    "behaviour_list_page_max": RuntimeKey(
        "Behaviours per page (most)",
        "The most behaviours one page of the catalogue may list. A larger page is refused.",
        "next request", ("API",),
    ),
    "activation_prefetch": RuntimeKey(
        "Prepare the next training buffer early",
        "Prepare the next training buffer in the background while the current one trains. Turned off "
        "automatically when memory is short.",
        "next training task", ("graphics card workers",),
    ),
}

#: Shown read-only: changing one needs a restart of the named pod (spec §3.1 "Shown read-only").
RESTART_KEYS: dict[str, str] = {
    "log_format": "mistudio-backend pod",
    "websocket_ping_interval": "mistudio-backend pod",
    "websocket_ping_timeout": "mistudio-backend pod",
    "judge_prompt_seed_dir": "mistudio-backend pod",
}

#: Read-only rows for settings that live in ANOTHER container's environment, which the API process
#: cannot read (FTID TI10): `(environment variable, pod to restart, description)`. D6: the two MCP
#: steering limits stay restart-only in v1 because the MCP pod has no database.
EXTERNAL_RESTART_KEYS: tuple[tuple[str, str, str], ...] = (
    ("CELERY_MAX_TASKS", "mistudio-backend pod",
     "Tasks a worker runs before it is replaced by a fresh process."),
    ("CELERY_CONCURRENCY", "mistudio-backend pod",
     "Tasks a general or processor worker runs at once."),
    ("MCP_TOOL_CATEGORIES", "mistudio-mcp pod",
     "The groups of agent tools the MCP (Model Context Protocol) server offers."),
    ("MILLM_API_URL", "mistudio-mcp pod",
     "miLLM's address, for the agent tools that call miLLM (the backend reads its own copy)."),
    ("MIDATAWORKS_API_URL", "mistudio-mcp pod",
     "miDataworks' address, for the agent tools that call miDataworks."),
    ("MCP_STEERING_MAX_CONCURRENT", "mistudio-mcp pod",
     "How many steering requests one agent may have running at once."),
    ("MCP_STEERING_MAX_NEW_TOKENS", "mistudio-mcp pod",
     "The longest text an agent's steering request may generate."),
)

#: A restart of `mistudio-backend` stops every running job: the five containers share one pod with
#: `strategy: Recreate`. The page says so beside each row that needs it.
BACKEND_RESTART_WARNING = (
    "Restarting the mistudio-backend pod stops every running job, including graphics card work."
)

#: Shown read-only under "From the environment".
SHOWN_READ_ONLY: frozenset[str] = frozenset(
    {
        "gpu_worker_mode",
        "environment",
        "internal_api_url",
        "midataworks_api_url",
        "millm_api_url",
        "bypass_settings_pin",
        "hf_token",
        "openai_api_key",
    }
)

#: Secrets: never served. Shown as "set" / "not set" only, with a link to the API Keys tab, which
#: owns their encrypted flow (D4: "the secret flows keep their documented order").
SECRET_FIELDS: frozenset[str] = frozenset({"hf_token", "openai_api_key"})

#: Never on the page (spec §3.1 "Excluded"): secrets with no page of their own, the database and
#: broker URLs, paths and volumes, CORS.
HIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "secret_key",
        "database_url",
        "redis_url",
        "celery_broker_url",
        "celery_result_backend",
        "neuronpedia_local_db_url",
        "data_dir",
        "backend_dir",
        "hf_cache_dir",
        "allowed_origins",
    }
)

#: **Never writable, named** (spec §2 "Allow-list"). Everything outside `RUNTIME_KEYS` is refused
#: anyway; this list exists so `test_environment_api.py` can try each by name, and so an edit that
#: moves one into `RUNTIME_KEYS` fails there rather than shipping.
NEVER_WRITABLE: frozenset[str] = frozenset(
    {
        "database_url",
        "redis_url",
        "celery_broker_url",
        "celery_result_backend",
        "secret_key",
        "data_dir",
        "backend_dir",
        "hf_cache_dir",
        "allowed_origins",
        "bypass_settings_pin",
        "environment",
        "gpu_worker_mode",
        "internal_api_url",
        "midataworks_api_url",
        "millm_api_url",
        "hf_token",
        "openai_api_key",
        "neuronpedia_local_db_url",
        "judge_prompt_seed_dir",
    }
)


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()
