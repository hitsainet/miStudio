"""What an external tool module is handed besides its own client (Feature 034, FTDD TD8).

Every external module registers as `register(mcp, client, gate, ctx)`. Before this, a `millm_*`
module received only its miServer client and the health gate, so it could not read a setting or
reach miWorkbench's own API — which is exactly what `miserver_load_model` needs to ask miWorkbench's
approval queue before evicting the resident model (FR-20).
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExternalContext:
    """Settings, the miWorkbench client and the identity this MCP path presents."""

    #: The server's `MCPSettings` (flags such as `probe_gpu_approval`).
    settings: Any
    #: The `MiStudioClient` the native categories use — for miStudio-held approvals.
    mistudio: Any
    #: The product row's identity, e.g. `agent:miworkbench-mcp` (FR-12; `agent:mistudio-mcp` before the P3 cut).
    identity: str
