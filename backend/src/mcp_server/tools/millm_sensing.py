"""
miLLM sensing tools (category: millm_sensing) — Feature 9, Unified MCP.

Cluster co-activation observation: when did the active cluster's members
fire together, on what tokens, with what context.
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

    @mcp.tool()
    async def millm_sensing_clear(
        profile_id: Annotated[Optional[str], Field(description="Clear only this profile's sensing events")] = None,
        all_profiles: Annotated[bool, Field(description="Clear EVERY profile's sensing events. Required when profile_id is omitted")] = False,
    ) -> Any:
        """Delete recorded sensing events — one profile's, or (explicitly) all of them.
        Irreversible (2026-10-19).

        The scope is REQUIRED: an unscoped call is refused here rather than read as "delete
        everything" (the `millm_circuit_sensing_clear` rule)."""
        if profile_id is None and not all_profiles:
            return {"error": "scope_required",
                    "message": "pass profile_id, or all_profiles=true to clear every profile's events"}
        if profile_id is not None and all_profiles:
            return {"error": "ambiguous_scope",
                    "message": "pass profile_id OR all_profiles=true, not both"}
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await millm.delete("/api/sensing/events", profile_id=profile_id)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_sensing_event(
        event_id: Annotated[int, Field(description="Sensing event id from millm_sensing_events")],
    ) -> Any:
        """⚠ RETURNS DECODED USER PROMPT CONTENT. One co-activation (sensing) event in full,
        including its context window (`context_text` / `context_token_ids`, `context_parts`) —
        the decoded text the user and the model wrote around the fired span.

        Operator decision 2026-10-19: exposed per event. Fetch only the events you need to
        read, and treat the text as user data (do not copy it into logs, reports or other
        systems without reason)."""
        return await millm.get(f"/api/sensing/events/{event_id}")
