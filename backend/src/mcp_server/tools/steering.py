"""
Steering tools (category: steering) — async compare/sweep/combined + mode control.

GPU-heavy operations. Guardrails (Feature 010, BR-3.4):
- max_new_tokens clamped to MCP_STEERING_MAX_NEW_TOKENS
- at most MCP_STEERING_MAX_CONCURRENT unresolved agent steering tasks
- with MCP_STEERING_APPROVAL=true, requests become operator-approval records
  instead of running immediately (poll get_approval_status)
"""

from typing import Annotated, Any, Optional

from pydantic import Field
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from ..client import MiStudioClient
from ..config import MCPSettings


class SteerFeature(BaseModel):
    """One feature to steer with."""

    feature_idx: Annotated[int, Field(description="Feature INDEX within the SAE (0..n_features-1) — not a row id")] = Field(..., ge=0, description="Feature (neuron) index in the SAE")
    layer: Annotated[int, Field(description="Transformer layer this feature lives on; must match the SAE's trained layer")] = Field(..., ge=0, description="Layer the SAE was trained on")
    # Feature 015: per-feature SAE for cross-layer circuits. Omitted ⇒ the
    # request-level sae_id. A feature whose layer ≠ its SAE's trained layer is
    # rejected (own-layer rule), never steered through the wrong decoder.
    sae_id: Optional[str] = Field(
        None, description="SAE for THIS feature (015 multi-layer); defaults to the request sae_id")
    strength: float = Field(
        ..., ge=-300, le=300,
        description="Raw steering coefficient, Neuronpedia scale. Start ±5–20; ±100+ often collapses output",
    )


# Unresolved agent-submitted steering task ids (concurrency guardrail)
_inflight: set[str] = set()


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    def _generation_params(max_new_tokens: Annotated[int, Field(description="Tokens to generate")], temperature: Annotated[float, Field(description="Sampling temperature; 0 = greedy")], top_p: Annotated[float, Field(description="Nucleus sampling threshold")]) -> dict:
        return {
            "max_new_tokens": min(max_new_tokens, settings.steering_max_new_tokens),
            "temperature": temperature,
            "top_p": top_p,
        }

    def _check_concurrency() -> None:
        if len(_inflight) >= settings.steering_max_concurrent:
            raise RuntimeError(
                f"Guardrail: {settings.steering_max_concurrent} agent steering task(s) already "
                "in flight. Poll get_steering_result (which releases the slot) or cancel one."
            )

    async def _submit(tool_name: str, endpoint: str, body: dict) -> Any:
        if settings.steering_approval:
            approval = await client.post(
                "/mcp/approvals", json_body={"tool_name": tool_name, "payload": body}
            )
            return {
                "approval_request_id": approval["id"],
                "status": "pending_approval",
                "hint": "Operator approval required — poll get_approval_status; "
                "once approved it returns the steering task_id",
            }
        _check_concurrency()
        result = await client.post(endpoint, json_body=body)
        task_id = result.get("task_id")
        if task_id:
            _inflight.add(task_id)
        return result

    @mcp.tool()
    async def steering_status() -> Any:
        """Steering service health, circuit-breaker state, and reset path."""
        return await client.get("/steering/status")

    @mcp.tool()
    async def compute_cluster_allocation(
        sae_id: Annotated[str, Field(description="miStudio SAE id (sae_xxxxxxxx). NOTE: a circuit-definition's mistudio_sae_id is miLLM's DIFFERENT id form; no tool translates")],
        members: Annotated[list[dict], Field(description="[{feature_idx, strength, ...}] — strength is what STEERS; keep the total modest (~3 across two layers)")],
        group_cohesion: Annotated[Optional[float], Field(description="Cohesion score carried from the source group")] = None,
        circuit_id: Annotated[Optional[str], Field(description="miStudio circuit id (circ_xxxxxxxx)")] = None,
    ) -> Any:
        """Compute the principled starting strength allocation for steering a
        CLUSTER of features (Feature 013) — or a MULTI-LAYER circuit (Feature
        015). members: [{feature_idx, layer, sae_id?, similarity?,
        activation_frequency?, sign?}] (1-20).

        OWN-LAYER RULE (015): each member is steered through the SAE trained on
        ITS layer — pass `sae_id` per member for cross-layer circuits (omitted ⇒
        the request-level sae_id). A member whose layer ≠ its SAE's trained
        layer is REJECTED (422 listing offenders), never silently mis-served.

        Single-layer requests return the 013 shape unchanged: {B, B_dir, G,
        f_eff, weights, strengths, flags, constants_used}. Multi-layer requests
        (members span >1 layer) return {formula_id, layers: {L: {B, B_dir, G,
        weights, strengths, flags, sae_id}}, hazards: [...], strengths: [...]}.
        Pass `circuit_id` to source hazard evidence from a stored circuit's
        VALIDATED edges (rung>=2 → quantified ES); otherwise the weight-prior
        heuristic labels each hazard 'heuristic' (never causal). VRAM envelope:
        each referenced SAE loads (~130 MB per 8k SAE); ≤4 extra documented.
        Read-only — safe without steering mode.

        HONOR THE FLAGS CONTRACT before seeding steer_combined:
        - 'low_cohesion': the cluster fails the cohesion gate — the UI refuses
          these strengths and keeps per-feature solo baselines; you should too
          unless deliberately testing the gate.
        - 'cancellation': positive members partially cancel (worst pair in
          cancellation_pair) — the blended direction is unreliable.
        - 'approximate': decoder unavailable; constant-budget fallback (G=1).
        - 'default_budget': no activation frequencies known.
        - 'nonunit_decoder': decoder columns deviate from unit norm — the law
          extrapolates; prefer running the calibration protocol first."""
        body: dict = {"sae_id": sae_id, "members": members}
        if group_cohesion is not None:
            body["group_cohesion"] = group_cohesion
        if circuit_id is not None:
            body["circuit_id"] = circuit_id
        return await client.post("/steering/cluster-allocation", json_body=body)

    @mcp.tool()
    async def get_steering_mode() -> Any:
        """Is steering mode active (model+SAE pre-loaded on the GPU)?"""
        return await client.get("/steering/mode")

    @mcp.tool()
    async def enter_steering_mode() -> Any:
        """Enter steering mode — LOADS THE MODEL+SAE ONTO THE GPU.

        OPTIONAL. Every steer_* submit auto-starts the worker if it is not
        running, so this only moves the ~10s cold start out of your first
        task. Calling it is not a prerequisite.

        `exit_steering_mode` IS required when you finish: nothing else reaps
        the worker and it holds VRAM indefinitely.

        Contends for the same single GPU as the circuit pipeline
        (capture/attribution/validation/faithfulness), which uses a separate
        advisory lock that does NOT know about this worker — running both at
        once can OOM as an opaque task failure rather than a clean refusal."""
        return await client.post("/steering/enter-mode")

    @mcp.tool()
    async def exit_steering_mode() -> Any:
        """Exit steering mode and free the GPU memory it held."""
        return await client.post("/steering/exit-mode")

    @mcp.tool()
    async def steer_compare(
        sae_id: Annotated[str, Field(description="miStudio SAE id (sae_xxxxxxxx). NOTE: a circuit-definition's mistudio_sae_id is miLLM's DIFFERENT id form; no tool translates")],
        prompt: Annotated[str, Field(description="Prompt to generate from")],
        features: Annotated[list[SteerFeature], Field(description="Features to steer. Give each its own sae_id for cross-layer work; strengths sum on the residual stream")],
        model_id: Annotated[Optional[str], Field(description="miStudio model row id")] = None,
        max_new_tokens: Annotated[int, Field(description="Tokens to generate")] = 256,
        temperature: Annotated[float, Field(description="Sampling temperature; 0 = greedy")] = 0.7,
        top_p: Annotated[float, Field(description="Nucleus sampling threshold")] = 0.9,
    ) -> Any:
        """Run a steering comparison: 1-4 features, each generating steered output
        side-by-side with an unsteered baseline. GPU-heavy background task —
        returns task_id; poll get_steering_result. Strengths in [-300, 300].

        POLLING IS NOT OPTIONAL: get_steering_result releases an agent
        concurrency slot. Submit without ever polling and you wedge your own
        steering surface until the MCP process restarts.

        UNDER APPROVAL MODE the return shape CHANGES — you get
        {approval_request_id, status: "pending_approval"} instead of a
        task_id, and must poll get_approval_status (which only exists as a
        tool when approval mode is on). Handle both shapes."""
        body = {
            "sae_id": sae_id,
            "model_id": model_id,
            "prompt": prompt,
            "selected_features": [f.model_dump() for f in features],
            "generation_params": _generation_params(max_new_tokens, temperature, top_p),
            "include_unsteered": True,
        }
        return await _submit("steer_compare", "/steering/async/compare", body)

    @mcp.tool()
    async def steer_sweep(
        sae_id: Annotated[str, Field(description="miStudio SAE id (sae_xxxxxxxx). NOTE: a circuit-definition's mistudio_sae_id is miLLM's DIFFERENT id form; no tool translates")],
        prompt: Annotated[str, Field(description="Prompt to generate from")],
        feature_idx: Annotated[int, Field(description="Feature INDEX within the SAE (0..n_features-1) — not a row id")],
        layer: Annotated[int, Field(description="Transformer layer this feature lives on; must match the SAE's trained layer")],
        strength_values: Annotated[list[float], Field(description="Strengths to sweep, e.g. [1,2,5,10]. Read the generations to find your usable ceiling — output collapses well below the +/-300 bound")],
        model_id: Annotated[Optional[str], Field(description="miStudio model row id")] = None,
        max_new_tokens: Annotated[int, Field(description="Tokens to generate")] = 256,
        temperature: Annotated[float, Field(description="Sampling temperature; 0 = greedy")] = 0.7,
        top_p: Annotated[float, Field(description="Nucleus sampling threshold")] = 0.9,
    ) -> Any:
        """Dose-response sweep: one feature generated at each strength value
        (e.g. [0, 5, 20, 50]). The definitive causal-validation tool. GPU-heavy
        background task — returns task_id; poll get_steering_result.

        POLLING IS NOT OPTIONAL: get_steering_result releases an agent
        concurrency slot. Submit without ever polling and you wedge your own
        steering surface until the MCP process restarts.

        UNDER APPROVAL MODE the return shape CHANGES — you get
        {approval_request_id, status: "pending_approval"} instead of a
        task_id, and must poll get_approval_status (which only exists as a
        tool when approval mode is on). Handle both shapes."""
        body = {
            "sae_id": sae_id,
            "model_id": model_id,
            "prompt": prompt,
            "feature_idx": feature_idx,
            "layer": layer,
            "strength_values": [max(-300.0, min(300.0, s)) for s in strength_values],
            "generation_params": _generation_params(max_new_tokens, temperature, top_p),
        }
        return await _submit("steer_sweep", "/steering/async/sweep", body)

    @mcp.tool()
    async def steer_combined(
        sae_id: Annotated[str, Field(description="miStudio SAE id (sae_xxxxxxxx). NOTE: a circuit-definition's mistudio_sae_id is miLLM's DIFFERENT id form; no tool translates")],
        prompt: Annotated[str, Field(description="Prompt to generate from")],
        features: Annotated[list[SteerFeature], Field(description="Features to steer. Give each its own sae_id for cross-layer work; strengths sum on the residual stream")],
        model_id: Annotated[Optional[str], Field(description="miStudio model row id")] = None,
        max_new_tokens: Annotated[int, Field(description="Tokens to generate")] = 256,
        temperature: Annotated[float, Field(description="Sampling temperature; 0 = greedy")] = 0.7,
        top_p: Annotated[float, Field(description="Nucleus sampling threshold")] = 0.9,
    ) -> Any:
        """Apply ALL selected features simultaneously in one generation pass
        (synergy testing). GPU-heavy background task — returns task_id.

        POLLING IS NOT OPTIONAL: get_steering_result releases an agent
        concurrency slot. Submit without ever polling and you wedge your own
        steering surface until the MCP process restarts.

        UNDER APPROVAL MODE the return shape CHANGES — you get
        {approval_request_id, status: "pending_approval"} instead of a
        task_id, and must poll get_approval_status (which only exists as a
        tool when approval mode is on). Handle both shapes.


        MULTI-LAYER (015): give each feature its own `sae_id` to steer a
        cross-layer circuit — every feature steers through the SAE trained on
        ITS layer (own-layer rule; a layer/SAE mismatch is rejected, never
        mis-served). `features_applied` reports each member's layer + SAE so you
        can verify. VRAM envelope: each distinct SAE loads (~130 MB per 8k SAE).
        Compounding/cancellation across layers is surfaced by
        compute_cluster_allocation's hazards — check them before steering."""
        body = {
            "sae_id": sae_id,
            "model_id": model_id,
            "prompt": prompt,
            "selected_features": [f.model_dump() for f in features],
            "generation_params": _generation_params(max_new_tokens, temperature, top_p),
            "include_baseline": True,
        }
        return await _submit("steer_combined", "/steering/async/combined", body)

    @mcp.tool()
    async def get_steering_result(task_id: Annotated[str, Field(description="Steering task id returned by steer_compare/sweep/combined")]) -> Any:
        """Poll an async steering task. Returns status and, when completed, the
        generated outputs and metrics."""
        result = await client.get(f"/steering/async/result/{task_id}")
        # status is nested and terminal states are success/failure/revoked
        # (matching the backend's status_map) — releasing on the wrong key
        # leaked concurrency slots until the guardrail wedged (2026-07-14).
        state = (result.get("status") or {}).get("status") if isinstance(result.get("status"), dict) else result.get("status")
        if state in ("success", "failure", "revoked"):
            _inflight.discard(task_id)
        return result

    @mcp.tool()
    async def cancel_steering_task(task_id: Annotated[str, Field(description="Steering task id returned by steer_compare/sweep/combined")]) -> Any:
        """Cancel a running steering task and free its guardrail slot."""
        result = await client.delete(f"/steering/async/task/{task_id}")
        _inflight.discard(task_id)
        return result

    if settings.steering_approval:
        @mcp.tool()
        async def get_approval_status(approval_request_id: str) -> Any:
            """Poll an operator-approval request. status: pending | approved
            (includes steering_task_id to poll) | denied (includes reason)."""
            return await client.get(f"/mcp/approvals/{approval_request_id}")
