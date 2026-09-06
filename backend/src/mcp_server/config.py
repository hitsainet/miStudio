"""MCP server configuration (env prefix ``MCP_``)."""

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_CATEGORIES = {
    "read", "groups", "steering", "labeling", "experiments", "profiles", "circuits",
    "jlens", "jobs", "admin",
    # Unified MCP (miLLM Feature 9): opt-in, functional only with
    # MILLM_API_URL set — never in DEFAULT_CATEGORIES.
    "millm_runtime", "millm_clusters", "millm_sensing", "millm_circuits",
}
DEFAULT_CATEGORIES = "read,groups,steering,labeling,experiments,profiles,circuits,jlens,jobs"


class MCPSettings(BaseSettings):
    """Runtime configuration for the miStudio MCP server."""

    model_config = SettingsConfigDict(env_prefix="MCP_", extra="ignore")

    auth_token: str = Field(default="", description="Bearer token required on the HTTP transport")
    allow_anonymous: bool = Field(
        default=False, description="Permit startup without a token (stdio/localhost dev only)"
    )
    host: str = Field(default="0.0.0.0", description="Bind host (LAN-reachable by default)")
    port: int = Field(default=8765)
    tool_categories: str = Field(default=DEFAULT_CATEGORIES)
    steering_max_concurrent: int = Field(default=2, ge=1)
    steering_max_new_tokens: int = Field(default=512, ge=1, le=2048)
    steering_approval: bool = Field(
        default=False, description="Route agent steering through operator approval"
    )

    # Backend base URL — not MCP_-prefixed; matches worker convention.
    @property
    def api_url(self) -> str:
        return os.environ.get("MISTUDIO_API_URL", "http://localhost:8000").rstrip("/")

    #: Per-instance override for `millm_api_url`, so a caller that needs the
    #: millm_* categories to register can say so WITHOUT touching os.environ.
    #: MIS-E2E-115: `mistudio_howto` used `os.environ.setdefault` for this, a
    #: permanent process mutation — after any agent read the docs, the
    #: unauthenticated `/health` endpoint advertised the placeholder URL as the
    #: real configuration for the life of the process.
    millm_api_url_override: str = Field(
        default="",
        description="Overrides MILLM_API_URL for this settings object only",
    )

    # miLLM base URL (Unified MCP, Feature 9). Empty = millm_* categories
    # are skipped at registration even when requested (logged once).
    @property
    def millm_api_url(self) -> str:
        if self.millm_api_url_override:
            return self.millm_api_url_override.rstrip("/")
        return os.environ.get("MILLM_API_URL", "").rstrip("/")

    def enabled_categories(self) -> set[str]:
        requested = {c.strip() for c in self.tool_categories.split(",") if c.strip()}
        unknown = requested - VALID_CATEGORIES
        if unknown:
            raise ValueError(
                f"Unknown MCP tool categories: {sorted(unknown)} (valid: {sorted(VALID_CATEGORIES)})"
            )
        return requested
