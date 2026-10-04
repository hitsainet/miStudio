"""Cluster-profile tools (category: profiles) — Feature 014, IDL-30.

Read/save/export access to durable cluster profiles and the portable
`mistudio.cluster-definition/v1` interchange format. The agent's loop:
discover a promising cluster (groups tools) → tune it (compute_cluster_allocation
+ steer_combined) → save_cluster_profile with narrative → export_cluster_definition
to move it toward MILLM / other consumers.
"""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..client import MiStudioClient
from ..config import MCPSettings


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def list_cluster_profiles(
        sae_id: Annotated[Optional[str], Field(description="miStudio SAE id (sae_xxxxxxxx). NOTE: a circuit-definition's mistudio_sae_id is miLLM's DIFFERENT id form; no tool translates")] = None, search: Annotated[Optional[str], Field(description="Free-text filter over labels and descriptions")] = None
    ) -> Any:
        """List saved cluster profiles (newest first), optionally filtered by SAE
        or a name/display-token substring. Profiles are durable snapshots —
        they survive grouping-index recomputes."""
        params: dict[str, Any] = {}
        if sae_id:
            params["sae_id"] = sae_id
        if search:
            params["search"] = search
        return await client.get("/cluster-profiles", **params)

    @mcp.tool()
    async def get_cluster_profile(profile_id: Annotated[str, Field(description="Cluster-profile id from list_cluster_profiles")]) -> Any:
        """Get one cluster profile: members with tuned strengths, budget
        snapshot (013 allocation), narrative, provenance."""
        return await client.get(f"/cluster-profiles/{profile_id}")

    @mcp.tool()
    async def save_cluster_profile(
        name: Annotated[str, Field(description="Human-readable name")],
        members: Annotated[list[dict[str, Any]], Field(description="[{feature_idx, strength, sign?, similarity?, ...}] — strength is what STEERS. Keep the total modest: ~0.15 x the feature's real max_activation each, total ~3")],
        sae_id: Annotated[Optional[str], Field(description="miStudio SAE id (sae_xxxxxxxx). NOTE: a circuit-definition's mistudio_sae_id is miLLM's DIFFERENT id form; no tool translates")] = None,
        narrative: Annotated[Optional[str], Field(description="Free-text account of what this is claimed to do")] = None,
        display_token: Annotated[Optional[str], Field(description="Token shown as the cluster's headline")] = None,
        source_group_id: Annotated[Optional[str], Field(description="Feature group this profile was derived from")] = None,
        budget: Annotated[Optional[dict[str, Any]], Field(description="{intensity, intensity_range:[lo,hi]} — the dial envelope")] = None,
    ) -> Any:
        """Save a cluster profile (durable, decoupled from recomputable groups).

        members: [{feature_idx, strength, sign?, similarity?, activation_frequency?,
        label?, pinned?}] — max 20, strengths are the TUNED values you validated.
        budget: optional 013 allocation snapshot {B, B_dir, G, formula_id, constants,
        intensity}. Include your steering evidence in narrative (markdown OK)."""
        return await client.post(
            "/cluster-profiles",
            json_body={
                "name": name,
                "members": members,
                "sae_id": sae_id,
                "narrative": narrative,
                "display_token": display_token,
                "source_group_id": source_group_id,
                "budget": budget,
            },
        )

    @mcp.tool()
    async def export_cluster_definition(profile_id: Annotated[str, Field(description="Cluster-profile id from list_cluster_profiles")]) -> Any:
        """Export a profile as portable `mistudio.cluster-definition/v1` JSON —
        the consumer-neutral artifact (never contains secrets or local paths)."""
        return await client.get(f"/cluster-profiles/{profile_id}/export")
