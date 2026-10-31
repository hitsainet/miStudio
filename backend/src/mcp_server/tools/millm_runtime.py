"""
miLLM runtime tools (category: millm_runtime) — Feature 9, Unified MCP.

Status/profile control against a deployed miLLM. All tools are health-gated:
when miLLM is unreachable they return {"unavailable": "millm", "reason": …}
instead of raising (tools are never unregistered — contract §3).

`millm_set_telemetry` (miLLM MCP contract v1.23, ENH-006 D75–D78) is the one write of a miLLM
runtime setting reachable here, and only for telemetry keys: miLLM gates an agent's telemetry write
in its own approval queue (`telemetry_write`), so the tool refuses every other key BEFORE sending
anything — an agent's write of a non-telemetry key would run in miLLM at once (ENH-003 D5).
"""

from typing import Annotated, Any, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import BackendError
from ..context import ExternalContext
from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient, error_fields
from .millm_approvals import RELAYED_ACTIONS, ApprovalQueueCheck, relayed

#: Every key `millm_set_telemetry` may name starts with this.
TELEMETRY_PREFIX = "TELEMETRY_"
#: miLLM's environment-only telemetry settings (`runtime_settings.TELEMETRY_ENV_ONLY`, contract
#: v1.23): the receiver's headers, the client key and the certificate paths are secrets or files
#: that must be mounted into the pod, and OTEL_SDK_DISABLED is the SDK's kill switch. miLLM answers
#: 422 SETTING_NOT_RUNTIME for them; refusing here says so without filing a request the operator
#: could only watch fail. Pinned equal to miLLM's set by test_millm_set_telemetry.py (cross-repo).
TELEMETRY_ENV_ONLY: frozenset[str] = frozenset({
    "TELEMETRY_HEADERS", "TELEMETRY_TLS_CLIENT_KEY_FILE", "TELEMETRY_TLS_CA_FILE",
    "TELEMETRY_TLS_CLIENT_CERT_FILE", "OTEL_SDK_DISABLED",
})

#: miLLM's `runtime_settings.REASON_MAX_CHARS`: the longest `reason` `PUT /api/environment` accepts
#: (`EnvironmentSave.reason`, max_length). Pinned equal to miLLM's by test_millm_set_telemetry.py.
REASON_MAX_CHARS = 500


def telemetry_key_refusal(values: dict[str, Any], reset: list[str]) -> Optional[dict[str, Any]]:
    """`None` when every named key is a writable telemetry setting; otherwise the answer to return.

    Checked BEFORE the gate and before any request: a key outside `TELEMETRY_*` is not gated by
    miLLM's `telemetry_write`, so sending it would change a setting with nobody approving it."""
    named = [*values, *reset]
    if not named:
        return {"refused": "nothing_to_change",
                "message": "Name at least one setting in `values` or `reset`. Nothing was sent."}
    for key in named:
        if key in TELEMETRY_ENV_ONLY:
            return {
                "refused": "setting_not_runtime", "key": key,
                "message": (f"{key} is set only in miLLM's environment (a secret, a file mounted "
                            "into the pod, or the SDK's kill switch); it cannot be changed while "
                            "miLLM runs. Nothing was sent."),
            }
        if not isinstance(key, str) or not key.startswith(TELEMETRY_PREFIX):
            return {
                "refused": "not_a_telemetry_setting", "key": key,
                "message": (f"{key!r} is not a telemetry setting. millm_set_telemetry changes only "
                            f"{TELEMETRY_PREFIX}* settings; other miLLM settings are changed by the "
                            "operator on miLLM's Settings page. Nothing was sent."),
            }
    return None


def reason_refusal(reason: Optional[str]) -> Optional[dict[str, Any]]:
    """`None` when `reason` fits miLLM's limit; otherwise the answer to return, before any request.

    miLLM's `EnvironmentSave.reason` is `max_length=REASON_MAX_CHARS` (the change-history column's
    width); a longer one is a 422 the operator never sees. F-3 (2026-10-29)."""
    if reason is not None and len(reason) > REASON_MAX_CHARS:
        return {"refused": "reason_too_long", "length": len(reason), "max": REASON_MAX_CHARS,
                "message": (f"`reason` is {len(reason)} characters; miLLM keeps at most "
                            f"{REASON_MAX_CHARS}. Shorten it and call again. Nothing was sent.")}
    return None


def register(
    mcp: MCPServer, millm: MiLLMClient, gate: HealthGate, ctx: ExternalContext
) -> None:
    #: ENH-008: the fail-closed check that this miLLM gates `telemetry_write` in its own queue.
    queue = ApprovalQueueCheck(millm, gate)

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
    # miLLM hosts no MCP server, so its read lives here. Writes stay the operator's, in miLLM's
    # Settings page, with one exception since miLLM contract v1.23 (ENH-006 D75-D78): telemetry
    # settings, through millm_set_telemetry, which miLLM holds for the operator's approval.
    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_environment() -> Any:
        """miLLM's runtime settings and where each value comes from (miLLM Feature 35): the value
        in force, the environment's value, `source` (env | override), the group, when a change
        takes effect, and the last change. Secrets appear only as set / not set. Read-only. The
        only setting an agent can ask to change is a telemetry one (`TELEMETRY_*`), with
        millm_set_telemetry, and that waits for the operator in miLLM's Approvals page; every
        other setting is changed by the operator on miLLM's Settings page."""
        return await millm.get("/api/environment")

    @mcp.tool()
    async def millm_set_telemetry(
        values: Annotated[dict[str, Any], Field(description="Telemetry settings to change: TELEMETRY_* key -> new value (read the keys, types and limits with millm_get_environment)")],
        reset: Annotated[list[str], Field(description="Telemetry settings (TELEMETRY_* keys) to return to the value miLLM's environment sets")] = [],  # noqa: B006
        reason: Annotated[Optional[str], Field(description=f"Why; shown to the operator and kept in miLLM's change history (at most {REASON_MAX_CHARS} characters)")] = None,
    ) -> Any:
        """Ask to change miLLM's telemetry settings: whether each OpenTelemetry signal is exported,
        where to (the receiver endpoint), sampling, queue and timing. All in one save.

        **This waits for the operator.** The call files the change in **miLLM's** Approvals page
        and returns miLLM's pending answer plus `"decide_in": "millm"`; nothing changes yet. When
        the operator approves, miLLM applies exactly this change once, by itself, and restarts its
        export. Poll `millm_get_approval_status(approval_id)`; never call this again to "use" an
        approval. The operator can also change these settings directly on miLLM's Settings page.

        Only `TELEMETRY_*` settings: any other key is refused here and nothing is sent. Five stay
        set only in miLLM's environment and are refused too: TELEMETRY_HEADERS,
        TELEMETRY_TLS_CLIENT_KEY_FILE, TELEMETRY_TLS_CA_FILE, TELEMETRY_TLS_CLIENT_CERT_FILE and
        OTEL_SDK_DISABLED. An endpoint carrying a user name, password or query is refused by miLLM
        (credentials belong in TELEMETRY_HEADERS); miLLM's refusals come back as
        `{"refused": "millm_refused", "status_code", "code", "message", "details"}`, whatever shape
        miLLM answered in (message always set). A `reason` over 500 characters is refused here
        (`reason_too_long`) and nothing is sent. A miLLM without its own approval queue is
        refused (`millm_has_no_approval_queue`) and nothing is sent."""
        refused = telemetry_key_refusal(values, reset) or reason_refusal(reason)
        if refused is not None:
            return refused
        ok, why = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": why}
        refused = await queue.refusal(RELAYED_ACTIONS["millm_set_telemetry"])
        if refused is not None:
            return refused
        body = {"values": dict(values), "reset": list(reset), "reason": reason}
        try:
            response = await millm.put("/api/environment", json_body=body, full=True)
        except BackendError as exc:
            if exc.status_code == 0:
                gate.invalidate("millm")
                return {"unavailable": "millm", "reason": str(exc)}
            if 400 <= exc.status_code < 500:
                # F-3 (2026-10-29): every refusal shape — envelope, FastAPI `detail`, plain text.
                return {"refused": "millm_refused", "status_code": exc.status_code,
                        **error_fields(exc.status_code, exc.detail)}
            raise
        return relayed(response) or response.body

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_environment_history(
        limit: Annotated[int, Field(description="Newest changes first, at most this many", ge=1, le=500)] = 50,
    ) -> Any:
        """The history of operator changes to miLLM's runtime settings, newest first: key, old
        and new value, who, why and when. Read-only."""
        return await millm.get("/api/environment/history", limit=limit)
