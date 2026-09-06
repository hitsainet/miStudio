"""
miLLM circuit tools (category: millm_circuits) — Feature 20, MCP Circuit Surface.

Circuits are multi-layer graphs over several SAEs, carrying an EVIDENCE RUNG
that says how much is actually known about them. Every tool here transports
that language verbatim and never composes it.

Three agent-facing semantics these tools exist to keep straight — each is
stated in the tool descriptions because an agent reads those, not this file:

  * OBSERVATION IS NOT VALIDATION. Edge sensing records that an upstream
    feature fired and a downstream partner fired after it. That is a
    correlation on live traffic. It NEVER raises a rung: raising one requires
    a causal intervention, which happens in miStudio, not here.
  * ABSENCE OF ROWS IS NOT ABSENCE OF FIRING. An edge with no events may be
    unsensable (its SAE is not attached, its layer is dark). Read
    `unsensable_edges` before concluding a circuit is quiet.
  * `steering: null` MEANS NOT EVALUATED, never "not steering".

Three failure shapes, which need three different responses (R2-07):

  * `{"error": …}`        — YOUR CALL was wrong. Fix the arguments and retry.
  * `{"unavailable": …}`  — miLLM is DOWN. Report it; do not retry in a loop.
  * a tool ERROR with a `code` — the operation was REFUSED for a stated
    reason (`UNVALIDATED_CIRCUIT`, `CIRCUIT_LAYER_CONTENTION`,
    `NO_ACTIVE_CIRCUIT`, `AMBIGUOUS_ACTIVE_CIRCUIT`). Read the code and act on
    it; a retry without changing anything will be refused identically.

No hub tools. Circuit import is inline-only (EC-20.5): a circuit references
several SAEs by id, and importing one from a remote pack without checking those
references is how an agent ends up serving a circuit against the wrong feature
basis.
"""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient


def _validate_since(since: Optional[str]) -> Optional[dict]:
    """Reject a naive `since`. Returns an error dict, or None if fine.

    Copied deliberately from `millm_sensing.py` rather than shared: a naive
    timestamp makes the server silently assume UTC and shift the polling
    window, so both surfaces must reject it identically.
    """
    if since is None:
        return None
    from datetime import datetime

    try:
        parsed = datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError:
        return {"error": f"`since` is not ISO-8601: {since!r}"}
    if parsed.tzinfo is None:
        return {
            "error": "`since` must carry a UTC offset (e.g. ...T12:00:00Z) — "
            "naive timestamps shift the polling window silently"
        }
    return None


def register(mcp: FastMCP, millm: MiLLMClient, gate: HealthGate) -> None:
    # ── Lifecycle ──────────────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_status() -> Any:
        """Which circuits are steering RIGHT NOW, as a list.

        Several circuits can serve at once when their layers are disjoint, so
        this is never a single object. Each row carries `steering` — the
        SERVER's verdict on whether that circuit is genuinely influencing
        generation. Do NOT derive it from `is_active`: an active row can be a
        slice-fallback, unparseable or unattached circuit that steers nothing.
        `steering: null` means NOT EVALUATED, never "not steering".

        Failure shapes: `{"error": …}` means YOUR CALL was wrong,
        `{"unavailable": "millm"}` means the server is DOWN, and a tool error
        with a `code` means the operation was REFUSED for a stated reason.
        Three different responses; do not conflate them.

        `serving_mode` matters. `"full"` means the whole circuit is serving.
        `"slice_fallback"` means only a PER-LAYER PROJECTION is — the SAE set
        was incomplete — so the circuit's own rung does NOT describe what is
        steering, and you must not report it as though the circuit were served.

        While more than one circuit serves, the per-request dial and the
        `X-miLLM-Circuit-Rung` header are BOTH suppressed — no single circuit's
        dial or evidence describes the response. Do not substitute either
        circuit's rung."""
        return await millm.get("/api/circuits/active")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_list_circuits(limit: Annotated[int, Field(description="Max rows to return")] = 50, offset: Annotated[int, Field(description="Rows to skip, for paging")] = 0) -> Any:
        """Imported circuits with their evidence rung.

        `rung_language` is the server's phrase, rendered from the evidence
        ladder — transport it verbatim. Never re-derive a rung locally: a
        circuit's rung is the MINIMUM over its edges, computed server-side, and
        a local guess would overclaim on the first edge it read.

        The ladder: 0 mined, 1 attribution-supported, 2 the validated tier, 3
        faithfulness-tested. Activation is GATED at 2 — anything lower is
        refused unless a human explicitly acknowledges it, so read the rung
        here before attempting to serve.

        Edge OBSERVATION never moves a rung (see
        `millm_circuit_sensing_events`); raising one requires a causal
        intervention, which happens in miStudio."""
        return await millm.get("/api/circuits", limit=limit, offset=offset)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_claims() -> Any:
        """Which circuit holds which LAYER.

        The unit of contention is the layer, not the feature: steering sums
        into one per-layer vector, so two circuits on a layer interact even
        when their features differ. This view makes an activation refusal
        intelligible BEFORE it happens — two circuits can both be active while
        contending for nothing.

        `composed: true` marks a layer carrying more than one circuit. On those
        layers the rung header is omitted."""
        return await millm.get("/api/circuits/claims")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_activate_circuit(
        circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx)")],
        acknowledge_unvalidated: Annotated[bool, Field(description="Serve a circuit below rung 2 (not causally validated). Pass ONLY on explicit human instruction, and report the rung first. Does NOT persist — the intensity dial re-applies the gate")] = False,
        allow_layer_overlap: Annotated[bool, Field(description="Compose with a circuit already holding a layer. Only valid when details.overridable is true; a COLLISION (same layer AND feature) is never overridable")] = False,
    ) -> Any:
        """Serve a circuit.

        Two refusals. Both come back as a TOOL ERROR carrying a structured
        payload — the client raises on any `success:false` envelope — so read
        the `code` and `details` on the error rather than expecting a normal
        result. (F20 R2-06: these were described as "bodies, not errors", which
        is what the SERVER sends and not what the AGENT receives.)

        `UNVALIDATED_CIRCUIT` — the circuit's rung is below 2, so it is not
        causally validated. Re-send with `acknowledge_unvalidated=true` only
        on explicit human instruction; report the rung phrase first.

        `CIRCUIT_LAYER_CONTENTION` — another circuit holds one of these layers.
        Read `details.overridable`:
          * `true` — the layers overlap but the features differ. Composition is
            possible with `allow_layer_overlap=true`, but SURFACE
            `details.measured_hazard` to the human first, INCLUDING its `note`
            ("one model, one fixture — indicative, not exhaustive"). Do not
            present that measurement as a general law.
          * `false` — both circuits steer the SAME feature. NEVER retry;
            `details.override_param` is absent precisely so it cannot be
            guessed. Report `details.colliding_keys` and stop.

        If the refusal names a circuit that `millm_circuit_status` does not
        list, its claim is stuck: use `millm_release_circuit_claims`, NOT a
        retry and NOT the override."""
        return await millm.post(
            f"/api/circuits/{circuit_id}/activate",
            acknowledge_unvalidated=str(bool(acknowledge_unvalidated)).lower(),
            allow_layer_overlap=str(bool(allow_layer_overlap)).lower(),
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_deactivate_circuit(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx)")]) -> Any:
        """Stop serving a circuit and release its layer claims.

        `cleared_steering: false` with a warning means the steering could NOT
        be cleared — the model may still be steering. Report that rather than
        treating deactivation as done."""
        return await millm.post(f"/api/circuits/{circuit_id}/deactivate")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_set_circuit_intensity(
        intensity: Annotated[float, Field(description="Global steering dial. Bounded by the artifact's AUTHORED intensity_range (server-enforced), not a fixed 0-2")], acknowledge_unvalidated: Annotated[bool, Field(description="Serve a circuit below rung 2 (not causally validated). Pass ONLY on explicit human instruction, and report the rung first. Does NOT persist — the intensity dial re-applies the gate")] = False
    ) -> Any:
        """Dial the active circuit's global lambda.

        Refused while SEVERAL circuits serve: no single circuit's dial
        describes the response.

        With NOTHING serving you get `NO_ACTIVE_CIRCUIT`; activate a circuit
        first. With SEVERAL serving you get `AMBIGUOUS_ACTIVE_CIRCUIT`, naming
        them — deactivate all but one. Both surface as a TOOL ERROR with the
        code in its payload, not as a normal result: read the code, do not
        treat it as a transport fault.

        F20 R2-04: a rung<2 circuit re-applies its evidence gate on EVERY dial,
        because re-applying is a fresh arm. Without `acknowledge_unvalidated`
        this tool hit a hard `UNVALIDATED_CIRCUIT` wall with no parameter to
        escape it — an agent that legitimately activated an unvalidated circuit
        was dead-ended on its very next step. Pass it again here, on the same
        explicit human instruction that authorised the activation.

        For a SLICE-FALLBACK circuit the steering belongs to a CLUSTER PROFILE,
        which keeps its own intensity: this dial is recorded but NOT applied,
        and the response says so. Adjust the slice's cluster instead — its dial
        follows cluster rules, including the 0.5 floor."""
        return await millm.put(
            "/api/circuits/active/intensity",
            json_body={
                "intensity": intensity,
                "acknowledge_unvalidated": acknowledge_unvalidated,
            },
        )

    @mcp.tool()
    # NO @gated — same reason as millm_import_circuit below: the decorator runs
    # the gate before the body, which would make the serving check unreachable
    # while miLLM is down. Gate checked manually AFTER validation.
    async def millm_delete_circuit(
        circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx)")], acknowledge_serving: Annotated[bool, Field(description="Delete a circuit that is currently SERVING. Irreversible: stops live steering AND destroys the definition. Export first")] = False
    ) -> Any:
        """Delete an imported circuit permanently.

        If the circuit is currently SERVING, this refuses unless
        `acknowledge_serving=True`. Deleting a serving circuit tears down live
        steering and destroys the definition — there is no undo, and the
        exported artifact goes with it.

        F20 R2-20: this was the only irreversible operation in the module with
        no gating at all, while its sibling destructive tool
        (`millm_circuit_sensing_clear`) requires an explicit scope opt-in. The
        server deletes a live circuit unconditionally, so nothing anywhere in
        the stack stood between a mistyped id and production steering going
        down. An agent told to "clean up the old circuit" that passes a stale
        id would deactivate and destroy a circuit actively steering traffic,
        and the first symptom would be the model quietly behaving differently.

        The check is best-effort and deliberately FAILS OPEN: if the serving
        state cannot be read, the delete proceeds rather than becoming
        unusable during an outage. It is a guard against the plausible mistake
        (a stale id), not a lock against a determined caller.
        """
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}

        unchecked = False
        if not acknowledge_serving:
            serving = await _is_serving(circuit_id)
            if serving is True:
                return {
                    "refused": "circuit_is_serving",
                    "circuit_id": circuit_id,
                    "reason": (
                        f"Circuit {circuit_id} is serving live traffic. "
                        "Deleting it stops steering AND destroys the "
                        "definition — this cannot be undone. Export it first "
                        "(millm_export_circuit) if you may need it back, then "
                        "pass acknowledge_serving=true."
                    ),
                }
            unchecked = serving is None

        result = await millm.delete(f"/api/circuits/{circuit_id}")

        # F20 R3-03: a SILENT fail-open is a guard that lies by omission.
        #
        # R2-20 chose to fail open so cleanup stays possible during an outage,
        # and that is still right — but it said nothing. When /active could not
        # be read, the delete proceeded WITHOUT the protection this tool
        # advertises and the response was byte-identical to a clean delete. The
        # operator whose steering just stopped had no way to connect the two.
        #
        # In the payload rather than a log line, because the agent is the one
        # that has to decide whether to go looking.
        if unchecked and isinstance(result, dict):
            result = {
                **result,
                "guard_skipped": "serving_state_unreadable",
                "warning": (
                    f"Deleted {circuit_id} WITHOUT confirming it was not "
                    "serving — miLLM's /api/circuits/active could not be "
                    "read. If steering has stopped unexpectedly, this is "
                    "where to look."
                ),
            }
        return result

    async def _is_serving(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx)")]) -> Optional[bool]:
        """True/False if known, None if the serving state could not be read.

        Callers must treat None as "unknown" and NOT as "not serving" — see
        the fail-open note in millm_delete_circuit.
        """
        try:
            active = await millm.get("/api/circuits/active")
        except Exception:
            return None
        # `/active` returns a LIST (F19: several circuits can serve at once).
        circuits = active.get("data") if isinstance(active, dict) else active
        if not isinstance(circuits, list):
            return None
        return any(
            str(c.get("id")) == str(circuit_id)
            for c in circuits
            if isinstance(c, dict)
        )

    # ── Import / export ────────────────────────────────────────────────────

    @mcp.tool()
    # NO @gated here, deliberately (millm_clusters.py precedent).
    #
    # The decorator runs the gate BEFORE the body, so an in-body argument check
    # is unreachable while miLLM is down — and an agent debugging its own
    # payload gets told "millm is down", sending it to fix the wrong thing.
    # The gate is checked manually below, AFTER validation.
    async def millm_import_circuit(
        definition: Annotated[dict, Field(description="A portable definition document. `kind` carries NO /v1 suffix")], on_conflict: Annotated[Optional[str], Field(description="'rename' (default — keep both) | 'fail' (refuse if the name exists)")] = None
    ) -> Any:
        """Import a `mistudio.circuit-definition/v1` document (INLINE only).

        No hub import: a circuit references several SAEs by id, and pulling one
        from a remote pack without checking those references is how an agent
        ends up serving a circuit against the wrong feature basis.

        An incompatible definition imports as UNBOUND and refuses only at
        activation, with a per-SAE compatibility matrix. `on_conflict`:
        'rename' (default) or 'fail'."""
        # Argument validation BEFORE the gate check: an agent debugging its own
        # payload must not be told "millm is down" (millm_clusters.py
        # precedent). A gate failure is about the SERVER; this is about the
        # CALL, and conflating them sends the agent to fix the wrong thing.
        if on_conflict not in (None, "rename", "fail"):
            return {"error": "`on_conflict` must be 'rename' or 'fail'"}
        if not isinstance(definition, dict):
            return {"error": "`definition` must be the circuit document object"}
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await millm.post(
            "/api/circuits/import", json_body=definition, on_conflict=on_conflict
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_export_circuit(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx)")]) -> Any:
        """The circuit's ORIGINAL document, byte-for-byte.

        Returned RAW — not in the `{success, data}` envelope — because the
        export is the portable artifact and re-wrapping it would change what a
        round-trip produces. Unwrapping `.data` on this response yields
        `None`."""
        return await millm.raw_get(f"/api/circuits/{circuit_id}/export")

    # ── Recovery ───────────────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_release_circuit_claims(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx)")]) -> Any:
        """Release ONE circuit's stuck layer claims.

        Use when an activation is refused naming a circuit that
        `millm_circuit_status` does not list — its claim outlived its serving.
        Retrying will not help, and `allow_layer_overlap=true` would compose
        against a circuit that is not there.

        Scoped to a single circuit deliberately; there is no "release
        everything". Releasing a circuit that IS still serving leaves it
        steering layers it no longer holds — the response warns when that
        happens."""
        return await millm.post(
            "/api/circuits/claims/release", circuit_id=circuit_id
        )

    # ── Edge sensing ───────────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_status() -> Any:
        """Edge-sensing runtime state for the armed circuit.

        `unsensable_edges` lists edges that CANNOT fire an observation, each
        with a `reason` and `detail` — the actionable half. Report the reason,
        not just the count: it tells the human whether to attach an SAE or
        accept the gap. Read this before concluding a circuit is quiet;
        absence of rows is not absence of firing.

        `requests_sensed == 0` means NO request reached sensing at all — a
        wiring or skip condition, not quiet traffic. `requests_truncated`
        outlives `truncated_layers`: the latter describes only the last drained
        request."""
        return await millm.get("/api/circuit-sensing/status")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_events(
        circuit_id: Annotated[Optional[str], Field(description="miStudio circuit id (circ_xxxxxxxx)")] = None,
        limit: Annotated[int, Field(description="Max rows to return")] = 50,
        since: Annotated[Optional[str], Field(description="ISO-8601 timestamp WITH a UTC offset (e.g. 2026-07-21T12:00:00Z). A naive timestamp is refused — it would shift the window silently")] = None,
    ) -> Any:
        """Observed edge firings, newest first.

        Each row is a CORRELATION on live traffic: an upstream feature fired
        and its downstream partner fired within the token lag. That is an
        OBSERVATION, not a validation — it NEVER raises the edge's rung. Only a
        causal intervention does, and that happens in miStudio.

        `edge_rung_language` is denormalised AS OF OBSERVATION: it describes
        the evidence that was true when the row was written, not today's rung.
        Do not re-read a current rung against an old event.

        `since` MUST carry a UTC offset — naive timestamps shift the window
        silently."""
        bad = _validate_since(since)
        if bad is not None:
            return bad
        return await millm.get(
            "/api/circuit-sensing/events",
            circuit_id=circuit_id,
            limit=limit,
            since=since,
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_event(event_id: Annotated[int, Field(description="Event id (INTEGER) from the events list")]) -> Any:
        """One edge observation in full: both endpoints with their layer,
        feature, position and activation, the observed token lag, and the
        context window.

        The rung language on this row is as-of-observation (see
        `millm_circuit_sensing_events`).

        `event_id` is an INTEGER. F20 R1-05: this advertised `str`, so the
        schema accepted "abc" and the agent got a FastAPI 422 from the route
        instead of a usable error from the tool — and the caller test pinned a
        call shape production would have rejected."""
        return await millm.get(f"/api/circuit-sensing/events/{event_id}")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_enable(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx)")]) -> Any:
        """Enable edge sensing for a circuit (persists; arms when it serves).

        Enabled is operator INTENT and is reported distinctly from `armed`: a
        circuit can be enabled but not armed because it is not active, or
        because its SAE set is not attached.

        If you enable sensing and see no events, call
        `millm_circuit_sensing_status` — it says WHICH of those two it is, and
        they have different remedies (activate the circuit vs attach the SAEs).
        Do not poll `_events` waiting for something that cannot fire."""
        return await millm.post(f"/api/circuit-sensing/{circuit_id}/enable")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_disable(circuit_id: Annotated[str, Field(description="miStudio circuit id (circ_xxxxxxxx)")]) -> Any:
        """Disable edge sensing for a circuit. Recorded events are kept."""
        return await millm.post(f"/api/circuit-sensing/{circuit_id}/disable")

    @mcp.tool()
    # NO @gated: scope validation must run BEFORE the gate (R1-16).
    async def millm_circuit_sensing_clear(
        circuit_id: Annotated[Optional[str], Field(description="miStudio circuit id (circ_xxxxxxxx)")] = None, all_circuits: Annotated[bool, Field(description="Clear observations for EVERY circuit. Mutually exclusive with circuit_id; one of the two is required — there is no default")] = False
    ) -> Any:
        """Delete recorded edge observations. IRREVERSIBLE.

        Pass EITHER `circuit_id` (that circuit only) OR `all_circuits=true`.
        Omitting both is refused rather than treated as "everything".

        F20 R1-16: `circuit_id` alone defaulted to GLOBAL scope, so an agent
        asked to "clear the events for this circuit" that omitted the argument
        wiped every observation in the deployment. The destructive default was
        the quiet one — `millm_release_circuit_claims` already explains why it
        is deliberately not global, and that reasoning applies here.

        This removes the record of what was observed. It does not change any
        rung, because observations never set one."""
        # Validation BEFORE the gate: a scope mistake is about the CALL, and
        # reporting it as "millm is down" sends the agent to fix the wrong
        # thing (same reason `millm_import_circuit` hand-rolls its gate).
        if (circuit_id is None) == (not all_circuits):
            return {
                "error": "specify exactly one scope: `circuit_id` for one "
                         "circuit, or `all_circuits=true` to delete EVERY "
                         "recorded observation. This is irreversible, so "
                         "there is no default."
            }
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        # NOTE (R2-08): global scope is expressed to the server by OMITTING
        # `circuit_id` — there is no `all_circuits` parameter on the route. So
        # this call is byte-identical to the unscoped call the guard above
        # refuses, and the guard is CLIENT-SIDE ONLY.
        #
        # That is the correct division: the server cannot know whether an
        # omission was deliberate, and the agent surface is where the mistake
        # actually happens. Stated here so the next reader does not "fix" the
        # apparent drop by forwarding a parameter the route would reject.
        return await millm.delete(
            "/api/circuit-sensing/events", circuit_id=circuit_id
        )
