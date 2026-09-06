"""
miLLM runtime tools (category: millm_runtime) — Feature 9, Unified MCP.

Status/profile control against a deployed miLLM. All tools are health-gated:
when miLLM is unreachable they return {"unavailable": "millm", "reason": …}
instead of raising (tools are never unregistered — contract §3).
"""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient


def register(mcp: FastMCP, millm: MiLLMClient, gate: HealthGate) -> None:
    @mcp.tool()
    @gated(gate, "millm")
    async def millm_status() -> Any:
        """miLLM runtime status in one call: loaded model, attached SAE,
        inference backend, circuit breakers, and the ACTIVE steering profile
        (id/name/source_kind/intensity/sensing_enabled) or null."""
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
