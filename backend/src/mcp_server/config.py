"""MCP server configuration (env prefix ``MCP_``)."""

import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# HAND-MAINTAINED, and therefore a second registry that can disagree with the
# real one in `tools/__init__.py`. It cannot be derived here: `tools` imports
# this module, so importing it back would be a cycle. A category missing from
# this set is REFUSED at startup even though it is fully registered, which
# looks exactly like the tools not existing.
#
# `test_reachability.py` pins the two together — see
# TestTheCategoryListsAgree. Add a category in BOTH places or neither.
VALID_CATEGORIES = {
    "read", "groups", "steering", "labeling", "experiments", "profiles", "circuits",
    "jlens", "jobs", "models", "admin",
    # Unified MCP (miLLM Feature 9): opt-in, functional only with
    # MILLM_API_URL set — never in DEFAULT_CATEGORIES.
    "millm_runtime", "millm_clusters", "millm_sensing", "millm_circuits",
    "millm_models",
}
DEFAULT_CATEGORIES = "read,groups,steering,labeling,experiments,profiles,circuits,jlens,jobs,models"


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

    def requested_categories(self) -> set[str]:
        """Exactly what MCP_TOOL_CATEGORIES asked for, recognised or not."""
        return {c.strip() for c in self.tool_categories.split(",") if c.strip()}

    def unknown_categories(self) -> set[str]:
        """Requested names this BUILD does not recognise.

        Surfaced on /health so an unrecognised name is visible rather than
        silent — it is dropped, not honoured, and something has to say so.
        """
        return self.requested_categories() - VALID_CATEGORIES

    def enabled_categories(self) -> set[str]:
        """The recognised categories, with unknown names WARNED AND DROPPED.

        This used to raise, which turned every category addition into a
        guaranteed crashloop window. The manifest and the image ship on
        DIFFERENT SCHEDULES: ArgoCD syncs `k8s/base/mcp.yaml` within minutes
        and the backend image takes about nine, so for that gap the new
        manifest runs against an image whose VALID_CATEGORIES predates it.
        Adding `models` and `millm_models` did exactly this — six restarts, and
        only the rolling update kept a healthy pod serving. Had the old pod
        been evicted in that window the MCP server would have been down
        outright, for a config that becomes valid on its own minutes later.

        Dropping the raise costs nothing for the case it was defending. A typo
        in the deployed manifest is caught in CI by
        `test_every_deployed_category_is_VALID`, which reads the manifest and
        checks it against these names — so the startup crash was a redundant
        second line of defence with an outage attached. What remains for an
        operator setting the variable by hand somewhere else is this warning
        plus `unknown_categories` on /health.

        FAIL-CLOSED IS PRESERVED WHERE IT MATTERS: an unrecognised name enables
        nothing. It is dropped, never honoured.
        """
        requested = self.requested_categories()
        unknown = requested - VALID_CATEGORIES
        if unknown:
            logger.warning(
                "Ignoring unknown MCP tool categories: %s (valid: %s). "
                "Expected transiently while a manifest change leads its image; "
                "if it persists, it is a typo and those tools are OFF.",
                sorted(unknown), sorted(VALID_CATEGORIES),
            )
        return requested & VALID_CATEGORIES
