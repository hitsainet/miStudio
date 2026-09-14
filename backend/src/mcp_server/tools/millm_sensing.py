"""
miLLM sensing tools (category: millm_sensing) — Feature 9, Unified MCP.

Cluster co-activation observation: when did the active cluster's members
fire together, on what tokens, with what context.
"""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient


def register(mcp: FastMCP, millm: MiLLMClient, gate: HealthGate) -> None:
    @mcp.tool()
    @gated(gate, "millm")
    async def millm_sensing_status() -> Any:
        """Sensing runtime status: armed cluster, quorum/threshold mode,
        per-request overhead, retention limits, and which clusters have the
        persistent sensing toggle enabled (distinct from armed)."""
        return await millm.get("/api/sensing/status")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_sensing_events(profile_id: Annotated[Optional[str], Field(description="Cluster-profile id from list_cluster_profiles")] = None,
                                   limit: Annotated[int, Field(description="Max rows to return")] = 50,
                                   since: Annotated[Optional[str], Field(description="ISO-8601 timestamp WITH a UTC offset (e.g. 2026-07-21T12:00:00Z). A naive timestamp is refused — it would shift the window silently")] = None) -> Any:
        """Co-activation events newest-first. Each row carries the span,
        fired members with peak activations, score, human summary, the
        ±K token context window (context_text/context_token_ids) when the
        cluster captures context, context_parts {before, span, after} —
        the window pre-split at the fired span so you can quote the prime
        token without re-deriving offsets — and ambient_fired_count, the
        alone-vs-within signal (how many features across the WHOLE SAE
        fired; null when full-width monitoring wasn't co-running — never
        estimated). `since` (ISO-8601 timestamp) time-bounds
        polling so agents fetch only new events. MUST carry an explicit
        UTC offset (e.g. 2026-07-16T12:00:00Z) — naive timestamps are
        rejected because the server would silently assume UTC."""
        if since is not None:
            from datetime import datetime

            try:
                parsed = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                return {"error": f"`since` is not ISO-8601: {since!r}"}
            if parsed.tzinfo is None:
                return {"error": "`since` must carry a UTC offset "
                                 "(e.g. ...T12:00:00Z) — naive timestamps "
                                 "shift the polling window silently"}
        return await millm.get("/api/sensing/events",
                               profile_id=profile_id, limit=limit,
                               since=since)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_sensing_enable(profile_id: Annotated[str, Field(description="Cluster-profile id from list_cluster_profiles")]) -> Any:
        """Enable sensing for a cluster (persists; arms live when that
        cluster is active with an SAE attached)."""
        return await millm.post(f"/api/sensing/{profile_id}/enable")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_sensing_config(profile_id: Annotated[str, Field(description="Cluster-profile id from list_cluster_profiles")],
                                   min_k: Annotated[Optional[int], Field(description="Quorum: how many members must co-fire to record an event. Cannot exceed the sensable member count")] = None,
                                   reset: Annotated[bool, Field(description="Clear the override and return to the default quorum")] = False) -> Any:
        """SET a cluster's sensing quorum (members that must co-fire for an
        event) — this tool WRITES config; use millm_sensing_status to read.
        Provide min_k, OR reset=true to restore the default (ALL members
        with usable thresholds). Calling with neither is refused — omitting
        min_k must not silently wipe an operator's tuned quorum (enh R2).
        Validated server-side against the sensable ceiling."""
        if min_k is None and not reset:
            return {"error": "provide `min_k`, or `reset=true` to restore "
                             "the default quorum"}
        if min_k is not None and reset:
            return {"error": "`min_k` and `reset=true` are contradictory — "
                             "pass exactly one"}
        return await millm.put(f"/api/sensing/{profile_id}/config",
                               json_body={"min_k": None if reset else min_k})

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_sensing_disable(profile_id: Annotated[str, Field(description="Cluster-profile id from list_cluster_profiles")]) -> Any:
        """Disable sensing for a cluster (disarms live if armed)."""
        return await millm.post(f"/api/sensing/{profile_id}/disable")
