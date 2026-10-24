"""
miLLM runtime tools (category: millm_runtime) — Feature 9, Unified MCP.

Status/profile control against a deployed miLLM. All tools are health-gated:
when miLLM is unreachable they return {"unavailable": "millm", "reason": …}
instead of raising (tools are never unregistered — contract §3).
"""

from typing import Annotated, Any, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..context import ExternalContext
from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient


def register(
    mcp: MCPServer, millm: MiLLMClient, gate: HealthGate, ctx: ExternalContext
) -> None:
    @mcp.tool()
    @gated(gate, "millm")
    async def millm_status() -> Any:
        """miLLM runtime status in one call: loaded model, attached SAE,
        inference backend, circuit breakers, and the ACTIVE steering profile
        (id/name/source_kind/intensity/sensing_enabled) or null.

        Two more blocks to read before acting on the resident model:

        * `lease` — the live lease on the resident model (holder, reason, expiry), or null.
          A held lease means another client's load will be REFUSED; the lease ID itself is
          never shown here (only its holder has it).
        * `weights` — weight state and attachment (miLLM Feature 31): `weights_state`,
          `barrier`, `attached`, `attachment_state` (`active` | `stale`). While a trainer
          (miForge) holds the weights or the barrier is up, the served model is being changed
          underneath; `error` replaces the facts when they could not be read — it is not
          "nothing attached".
        * `telemetry` — miLLM's OpenTelemetry export (Feature 038), when that miLLM reports it:
          `enabled`, the non-secret `config`, and per signal (`traces`, `metrics`, `logs`) its
          `state`, `exported` / `failed` / `dropped` counters and `last_success_at`. Confirm an
          export by reading `signals.logs.exported` before and after a request. Absent on a
          miLLM without it; `{"error": …}` means it could not be read, not "disabled"."""
        return await millm.get("/api/health/detailed")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_list_profiles() -> Any:
        """List miLLM steering profiles (manual rows and imported clusters —
        source_kind discriminates)."""
        return await millm.get("/api/profiles")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_activate_profile(profile_id: Annotated[str, Field(description="Cluster-profile id from list_cluster_profiles")],
                                     apply_steering: Annotated[bool, Field(description="Apply the profile's steering immediately on activation")] = True) -> Any:
        """Activate a miLLM profile by id — replaces the live steering.
        Cluster rows enforce the declared-feature-space gate server-side
        (a mismatched SAE refuses with a clear reason)."""
        # The route declares a required request body (009 R1: a body-less
        # POST 422s with 'Field required').
        return await millm.post(f"/api/profiles/{profile_id}/activate",
                                json_body={"apply_steering": bool(apply_steering)})

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_deactivate_profile(profile_id: Annotated[str, Field(description="Cluster-profile id from list_cluster_profiles")]) -> Any:
        """Deactivate a miLLM profile (clears live steering when it is the
        active one)."""
        return await millm.post(f"/api/profiles/{profile_id}/deactivate")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_set_intensity(intensity: Annotated[float, Field(description="Global steering dial. Bounded by the artifact's AUTHORED intensity_range (server-enforced), not a fixed 0-2")],
                                  reapply: Annotated[Optional[bool], Field(description="Re-apply the steering hooks after setting intensity. Omit to keep the default (true)")] = None) -> Any:
        """Set the ACTIVE cluster's persistent intensity dial (lambda). The
        authored intensity_range is enforced server-side; dialing to 0 is
        always allowed. Affects all traffic (global dial) — per-request
        dialing uses the OpenAI-side steering_intensity field instead."""
        # None means "use the default" (True) — bool(None) inverted it
        # for agents that populate optional fields with null (009 R1).
        return await millm.put("/api/clusters/active/intensity",
                               json_body={"intensity": intensity,
                                          "reapply": reapply is not False})

    # ENH-003 D5 (closed 2026-10-21): every MCP surface gets a READ of its app's runtime settings.
    # miLLM hosts no MCP server, so its read lives here. Writes stay off the MCP: the two write
    # routes carry written exemptions in test_mcp_millm_route_parity.py, and the operator changes
    # a value in miLLM's Settings -> Environment.
    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_environment() -> Any:
        """miLLM's runtime settings and where each value comes from (miLLM Feature 35): the value
        in force, the environment's value, `source` (env | override), the group, when a change
        takes effect, and the last change. Secrets appear only as set / not set. Read-only: no MCP
        tool changes a miLLM setting; the operator does, in miLLM's Settings -> Environment."""
        return await millm.get("/api/environment")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_environment_history(
        limit: Annotated[int, Field(description="Newest changes first, at most this many", ge=1, le=500)] = 50,
    ) -> Any:
        """The history of operator changes to miLLM's runtime settings, newest first: key, old
        and new value, who, why and when. Read-only."""
        return await millm.get("/api/environment/history", limit=limit)
