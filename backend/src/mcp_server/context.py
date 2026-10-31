"""What an external tool module is handed besides its own client (Feature 034, FTDD TD8).

Every external module registers as `register(mcp, client, gate, ctx)`. Before this, a `millm_*`
module received only its miLLM client and the health gate, so it could not read a setting or
reach miStudio's own API — which is exactly what `millm_load_model` needs to ask miStudio's
approval queue before evicting the resident model (FR-20).
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExternalContext:
    """Settings, the miStudio client and the identity this MCP path presents."""

    #: The server's `MCPSettings` (flags such as `probe_gpu_approval`).
    settings: Any
    #: The `MiStudioClient` the native categories use — for miStudio-held approvals.
    mistudio: Any
    #: The product row's identity, e.g. `agent:mistudio-mcp` (FR-12).
    identity: str
