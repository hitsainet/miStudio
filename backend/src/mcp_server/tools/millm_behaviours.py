"""miLLM behaviour probe sets and rule sets (category: millm_probes) — miLLM Feature 039 S1 (§4j) and S2 (§4k).

Proxies of miLLM's routes, under the tool names miLLM's contract (v1.17, §4j and §4k) gives them and its
refusals name in `error.next`. Pure HTTP through the miLLM client (this module imports nothing from the
backend: the `mistudio-mcp` pod has no database). Each tool returns miLLM's answer as it comes; miLLM's
refusals carry the next tool to call.

* Behaviour sets: `millm_list_behaviours`, `millm_arm_behaviour` (all or nothing; `dry_run` first),
  `millm_disarm_behaviour`, and `millm_get_probe_limit_recommendation` (the explained per-model limit the
  `PROBE_BUDGET` refusal points at — a READ). Since miLLM contract v1.25 (ENH-006 D83) the limit is a SLOWDOWN
  BUDGET — probes may slow each token by at most 5% of the model's own time — with no behaviour count
  (`max_behaviours` reads null, "not enforced"; `BEHAVIOUR_LIMIT` is no longer raised).
* Probe capacity (ENH-006 D54, miLLM §4j): `millm_probe_capacity` (READ: measured step time and cost lines, the
  configured and effective limits, and optionally the preview of arming a set), and two calls BEHIND the
  OPERATOR'S APPROVAL IN miLLM (ENH-008) — `millm_request_probe_limits` (changes what every client is served) and
  `millm_measure_probe_capacity` (GPU work on the resident model). Each RELAYS: miLLM files the request in its own
  queue and answers 202 pending, which the tool returns with `decide_in: "millm"`; miLLM runs it once when the
  operator approves (`tools/millm_approvals.py`). A miLLM without that queue is refused, fail closed.
* Rule sets as data: `millm_list_rule_validators`, `millm_validate_rule_set` (every import check, writes
  nothing), `millm_import_rule_set`, `millm_list_rule_sets`, `millm_get_rule_set`, `millm_delete_rule_set`
  (acknowledged), `millm_evaluate_rule_set` (the caller's own text, nothing written), and a behaviour's
  attachments: `millm_get_behaviour_rule_signals`, `millm_set_behaviour_rule_signals`,
  `millm_clear_behaviour_rule_signals`.
* Operator code in an import (miLLM 039 FTASKS 19.1; miLLM §4l): a set that names operator code (a `code_rule`
  signal, or the `{rule_set, code_modules}` envelope) is imported only BEHIND the operator's approval IN miLLM
  (ENH-008): the same relay, with miLLM's `request_digest` over the exact body. A declarative set is one request,
  no approval. `names_operator_code` still decides which: a miLLM that cannot gate operator code is never sent it.

⚠ ATTACHED RULE SIGNALS RUN ON LIVE TRAFFIC while a probe of the behaviour is armed (miLLM §4k): they report
beside the probes and never change them; a declared veto is never applied at serving. Operator code
(`code_rule`) reports `rule_worker_unavailable` (D41). Descriptions use `<behaviour-key>` placeholders only.
"""

import hashlib
from typing import Annotated, Any, Dict, List, Literal, Optional

import rfc8785
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..client import BackendError
from ..context import ExternalContext
from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient
from .millm_approvals import RELAYED_ACTIONS, ApprovalQueueCheck, moved_refusal, relayed

KEY = Annotated[str, Field(description="Behaviour key, e.g. '<behaviour-key>'")]
RULE_SET_ID = Annotated[str, Field(description="miLLM rule-set id (rs_ + 12 hex) from millm_list_rule_sets")]
DOCUMENT = Annotated[Dict[str, Any], Field(description="The `midataworks.rule-set/v1` document itself, unchanged")]
DIGEST = Annotated[Optional[str], Field(pattern=r"^[0-9a-f]{64}$", description="The identity you expect (SHA-256 of the RFC 8785 bytes); a mismatch is refused")]
ORIGIN = Annotated[Optional[Literal["authored", "derived"]], Field(description="authored, or derived from a labelled dataset")]

#: miLLM's accepted range per budget unit (`millm/services/probe_limits.py` `_check_budget`, contract v1.25). miLLM
#: checks it only when the approved request RUNS, so a value outside it would reach the operator and then fail;
#: refusing here sends nothing.
BUDGET_RANGE = {"pct": (0.01, 100.0), "ms": (0.01, 10.0)}


class ProbeBudget(BaseModel):
    """One budget half, exactly miLLM's `BudgetSetting` body (`{value, unit}`, extra fields forbidden).

    `unit` defaults to `pct` (ENH-006 D83): per cent of the model's OWN measured time — decode ms per generated token,
    or prefill ms per 1,000 prompt tokens — so the budget follows the model's next measurement. `ms` is an absolute
    override. The unit is ALWAYS sent, so miLLM never has to guess it.
    """

    model_config = ConfigDict(extra="forbid")

    value: float = Field(description="The budget: per cent of the model's own time (pct) or milliseconds (ms)")
    unit: Literal["pct", "ms"] = Field("pct", description="pct (default): per cent of the model's own measured time; ms: absolute milliseconds")

    @model_validator(mode="after")
    def _in_range(self) -> "ProbeBudget":
        lo, hi = BUDGET_RANGE[self.unit]
        if not lo <= self.value <= hi:
            raise ValueError(f"value must be within {lo:g}–{hi:g} {self.unit}")
        return self


def content_sha256(value: Any) -> str:
    """SHA-256 of the RFC 8785 (JCS) bytes — the identity miStudio and miLLM give a rule-set document
    (`src/rule_signals/canonical.py`, which this pod does not import; equality pinned by a test)."""
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def names_operator_code(body: Any) -> bool:
    """True when the import body names operator code anywhere. FAILS CLOSED: any `code_modules` key, or any
    object anywhere whose `kind` is `code_rule` — not only the top-level `signals` miLLM reads — so a shape this
    check did not foresee is gated, never sent unapproved."""
    if isinstance(body, dict) and "code_modules" in body:
        return True
    stack = [body]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if value.get("kind") == "code_rule":
                return True
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return False


def register(mcp: MCPServer, millm: MiLLMClient, gate: HealthGate, ctx: ExternalContext) -> None:  # noqa: ARG001
    #: ENH-008: the fail-closed check that this miLLM has its own approval queue.
    queue = ApprovalQueueCheck(millm, gate)

    async def _gate_and_queue(tool_name: str) -> Optional[Dict[str, Any]]:
        """`None` when the gated call may be sent: miLLM is up AND gates this action in its own queue."""
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await queue.refusal(RELAYED_ACTIONS[tool_name])

    # ── behaviour probe sets (§4j) ───────────────────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_list_behaviours(
        key: Annotated[Optional[str], Field(description="Only keys starting with this")] = None,
        cursor: Annotated[Optional[str], Field(description="meta.next of the previous page")] = None,
        limit: Annotated[Optional[int], Field(description="Keys per page", ge=1)] = None,
    ) -> Any:
        """One page of the behaviour sets miLLM holds: per key the imported and armed probes, versions, whether
        the set is complete (`"partial: n of m armed"` otherwise), probes stopped by an unload and the attached
        rule sets. Follow `meta.next` until null."""
        return await millm.get("/api/probes/behaviours", key=key, cursor=cursor, limit=limit)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_arm_behaviour(
        key: KEY,
        acknowledge_below_rung2: Annotated[bool, Field(description="Required when any member is below rung 2 (the same acknowledgement as millm_arm_probe)")] = False,
        reason: Annotated[str, Field(description="Why you are arming the set; recorded")] = "",
        windows: Annotated[Optional[List[str]], Field(description="Windows to report: all, prompt, response, last_user")] = None,
        version: Annotated[Optional[int], Field(description="Only this behaviour version's probes (omit: not narrowed)", ge=1)] = None,
        probe_ids: Annotated[Optional[List[str]], Field(description="Only these member probes")] = None,
        dry_run: Annotated[bool, Field(description="Run every gate for every member, arm nothing; answers would_refuse when the set would be refused")] = False,
    ) -> Any:
        """Arm a behaviour's probe set ALL OR NOTHING: each member runs millm_arm_probe's gates (identity, rung,
        scope, parity, cost), then the armed-probe backstop and the whole set's cost against the model's
        slowdown budget are checked once (no behaviour count since miLLM v1.25); a refusal arms none and names
        every member's outcome. Try `dry_run=true` first. Members already armed are untouched. A refusal's `next`
        names the tool that resolves it."""
        body: Dict[str, Any] = {"acknowledge_below_rung2": acknowledge_below_rung2, "reason": reason,
                                "dry_run": dry_run}
        for name, value in (("windows", windows), ("version", version), ("probe_ids", probe_ids)):
            if value is not None:
                body[name] = value
        return await millm.post(f"/api/probes/behaviours/{key}/arm", json_body=body)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_disarm_behaviour(
        key: KEY,
        version: Annotated[Optional[int], Field(description="Only this behaviour version's probes", ge=1)] = None,
        probe_ids: Annotated[Optional[List[str]], Field(description="Only these member probes")] = None,
        reason: Annotated[Optional[str], Field(description="Why; recorded")] = None,
    ) -> Any:
        """Disarm every armed member of a behaviour's set (or the named ones). Non-destructive: definitions and
        recorded results stay; a layer hook another behaviour's armed probe uses stays installed."""
        body: Dict[str, Any] = {}
        for name, value in (("version", version), ("probe_ids", probe_ids), ("reason", reason)):
            if value is not None:
                body[name] = value
        return await millm.post(f"/api/probes/behaviours/{key}/disarm", json_body=body)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_probe_limit_recommendation(
        model_id: Annotated[str, Field(description="miLLM model id (millm_list_models)")],
    ) -> Any:
        """The model's explained probe-limit recommendation (miLLM contract v1.25, ENH-006 D83): probes may slow
        each token by at most 5% of the model's own time (`PROBE_SLOWDOWN_BUDGET_PCT`). `values`:
        `decode_budget_pct` and `prefill_budget_pct` (the percentages), `decode_budget_ms` and `prefill_budget_ms`
        (what they mean on this measurement; null for a half whose step time is not measured, which then uses the
        absolute fallback), and `max_behaviours: null` — there is no behaviour count any more; the cost budget
        alone decides. `capacity` says what fits: `reference_sets` and `reference_set_probes` (how many reference
        sets of the configured shape, and their probes), `dense_probes` (dense probes on one layer), `binding` /
        `dense_binding` (the limit that stops the next one) and `upper_bound` (true while the measurement lacks
        the max / attention lines, so the counts are estimates). Also `budget`, `reasons` in words and numbers,
        the measurement it rests on and `recommendation_digest`. `values: null` when the model has no capacity
        measurement. READ ONLY: to adopt it, millm_request_probe_limits with the digest (miLLM writes the
        percentages, unit pct) — the operator approves it in miLLM."""
        return await millm.get(f"/api/models/{model_id}/probe-limits/recommendation")

    # ── probe capacity (ENH-006 D54; miLLM §4j) ─────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_probe_capacity(
        model_id: Annotated[int, Field(description="miLLM model row id (millm_list_models)")],
        behaviour: Annotated[Optional[str], Field(description="Also preview arming this behaviour's set now, e.g. '<behaviour-key>'")] = None,
        version: Annotated[Optional[int], Field(description="With behaviour: only this version's probes", ge=1)] = None,
        probe_ids: Annotated[Optional[List[str]], Field(description="Or preview arming exactly these probes")] = None,
    ) -> Any:
        """A model's probe capacity in miLLM: the measured step time and probe cost lines, the configured per-model
        limits, the EFFECTIVE limits admission uses, the limits the cost lines imply, and — for the resident model
        — how close the armed set is to each. Since miLLM contract v1.25 (ENH-006 D83) the default budget is 5% of
        the model's own measured decode time per token and prefill time per 1,000 prompt tokens; `effective.sources`
        names where each half came from (a half whose step time is not measured uses the absolute fallback and
        names the measure action). `effective.max_behaviours` is null, "not enforced": there is no behaviour count,
        the cost budget decides how many probes fit (`implied.dense.n`); `max_armed` is only a probe backstop.
        `configured.max_behaviours` may still show an old stored value; it binds nothing. `effective.approval.
        superseded: true` means the limits were approved from a pre-v1.25 recommendation in ms: the 5% budget
        applies instead until the operator approves the new recommendation or sets an override. With `behaviour`
        (and `version`) or `probe_ids`, also the PREVIEW of arming that set now (the projection the arm gate
        computes). READ ONLY. Changing limits is millm_request_probe_limits; measuring is
        millm_measure_probe_capacity (both need the operator)."""
        return await millm.get(f"/api/models/{model_id}/probe-capacity", behaviour=behaviour, version=version,
                               probe_ids=list(probe_ids) if probe_ids is not None else None)

    @mcp.tool()
    async def millm_request_probe_limits(
        model_id: Annotated[int, Field(description="miLLM model row id (millm_list_models)")],
        reason: Annotated[str, Field(description="Why the limits should change; shown to the operator and recorded", min_length=1, max_length=500)],
        recommendation_digest: Annotated[Optional[str], Field(pattern=r"^[0-9a-f]{64}$", description="Approve miLLM's recommendation with this digest (millm_get_probe_limit_recommendation): miLLM writes exactly its recomputed values")] = None,
        max_armed: Annotated[Optional[int], Field(description="Override: the armed-probe backstop on this model (not a behaviour count)", ge=1, le=64)] = None,
        decode_budget: Annotated[Optional[ProbeBudget], Field(description="Override: {value, unit}; unit pct (default) = per cent of the model's own measured decode time per generated token, ms = absolute milliseconds")] = None,
        prefill_budget: Annotated[Optional[ProbeBudget], Field(description="Override: {value, unit}; unit pct (default) = per cent of the model's own measured prefill time per 1,000 prompt tokens, ms = absolute milliseconds")] = None,
        approval_request_id: Annotated[Optional[str], Field(description="DEPRECATED (ENH-008): miLLM now decides its own requests. Passing it is refused with approval_moved_to_millm; poll millm_get_approval_status instead")] = None,
    ) -> Any:
        """Ask for a model's probe limits to change in miLLM; the operator approves it in miLLM's Approvals page.

        The limit is a slowdown budget (miLLM contract v1.25, ENH-006 D83): probes may slow each token by at most N%
        of the model's own time — N per cent of its measured decode time per generated token (`decode_budget`) and
        of its prefill time per 1,000 prompt tokens (`prefill_budget`). miLLM's default N is 5. There is no limit in
        behaviours: the cost budget alone decides how many probes fit (`max_armed` is a probe backstop only).

        It changes what EVERY client is served. EITHER approve miLLM's recommendation (`recommendation_digest` from
        millm_get_probe_limit_recommendation; miLLM recomputes it when the request runs, writes exactly its
        percentages as `{value, unit: "pct"}`, and fails with `RECOMMENDATION_CHANGED` if it moved) OR name
        overrides: each budget `{value, unit}`, unit `pct` (default; 0.01–100) or `ms` (absolute; 0.01–10); an
        omitted value means miLLM's global default. Not both: miLLM ignores values sent beside an approval, so that
        is refused here (`approval_takes_no_values`) and nothing is sent. The call returns miLLM's pending answer
        plus `"decide_in": "millm"` and nothing changes yet; once approved, miLLM writes the limits once, by itself.
        Poll `millm_get_approval_status(approval_id)`. You cannot approve it yourself. A miLLM without its own
        approval queue is refused (`millm_has_no_approval_queue`) and nothing is sent."""
        if approval_request_id is not None:
            return moved_refusal("millm_request_probe_limits")
        overrides = {name: value for name, value in (
            ("max_armed", max_armed),
            ("decode_budget", decode_budget.model_dump() if decode_budget is not None else None),
            ("prefill_budget", prefill_budget.model_dump() if prefill_budget is not None else None),
        ) if value is not None}
        body: Dict[str, Any] = {"reason": reason, "set_by": "mistudio-agent"}
        if recommendation_digest is not None:
            if overrides:
                return {"refused": "approval_takes_no_values", "fields": sorted(overrides),
                        "message": "Approving a recommendation writes exactly its recomputed values; miLLM would "
                                   "ignore these. Send the digest alone, or the overrides alone.",
                        "next": "millm_get_probe_limit_recommendation"}
            body.update(source="recommended-approved", recommendation_digest=recommendation_digest)
        else:
            body["source"] = "operator"
            body.update(overrides)
        refused = await _gate_and_queue("millm_request_probe_limits")
        if refused is not None:
            return refused
        response = await millm.put(f"/api/models/{model_id}/probe-limits", json_body=body, full=True)
        return relayed(response) or response.body

    @mcp.tool()
    async def millm_measure_probe_capacity(
        model_id: Annotated[int, Field(description="miLLM model row id of the RESIDENT model (millm_list_models: loaded)")],
        approval_request_id: Annotated[Optional[str], Field(description="DEPRECATED (ENH-008): miLLM now decides its own requests. Passing it is refused with approval_moved_to_millm; poll millm_get_approval_status instead")] = None,
    ) -> Any:
        """Measure the RESIDENT model's probe capacity on its card(s); the operator approves it in miLLM's Approvals page.

        It measures the step time and probe cost line — GPU work that live chats wait behind. The call returns miLLM's
        pending answer plus `"decide_in": "millm"`; once approved, miLLM measures once, by itself. Poll
        `millm_get_approval_status(approval_id)`, then read millm_probe_capacity or the recommendation. A miLLM
        without its own approval queue is refused (`millm_has_no_approval_queue`) and nothing is sent."""
        if approval_request_id is not None:
            return moved_refusal("millm_measure_probe_capacity")
        refused = await _gate_and_queue("millm_measure_probe_capacity")
        if refused is not None:
            return refused
        response = await millm.post(f"/api/models/{model_id}/probe-capacity/measure", full=True)
        return relayed(response) or response.body

    # ── rule sets as data (§4k) ──────────────────────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_list_rule_validators() -> Any:
        """Every validator miLLM's rule engine registers (id, version, parameter schema, reasons, outputs,
        algorithms) from its live registry, and the forms it evaluates (`inline`, `after_response` for operator
        code, or `unavailable`)."""
        return await millm.get("/api/rule-sets/validators")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_validate_rule_set(document: DOCUMENT, digest: DIGEST = None, origin: ORIGIN = None) -> Any:
        """Every check an import runs — the vendored schema, the pattern dialect, compilation, the set's own
        cases — writing nothing. Answers `{valid, digest, behaviour_key, cases_passed, cases_unverified,
        pending_signals}` or the refusal import would give."""
        return await millm.post("/api/rule-sets/validate", json_body=document, digest=digest, origin=origin)

    @mcp.tool()
    async def millm_import_rule_set(
        document: Annotated[Dict[str, Any], Field(description="The `midataworks.rule-set/v1` document itself, unchanged — or miLLM's envelope {rule_set: <document>, code_modules: {<code_sha256>: <canonical UTF-8>}} to carry operator code")],
        version: Annotated[int, Field(description="The set's version (a new version is a new import)", ge=1)] = 1,
        digest: DIGEST = None,
        origin: ORIGIN = None,
        approval_request_id: Annotated[Optional[str], Field(description="DEPRECATED (ENH-008): miLLM now decides its own requests. Passing it is refused with approval_moved_to_millm; poll millm_get_approval_status instead")] = None,
    ) -> Any:
        """Validate and store one rule set in miLLM (an identical document returns the existing id). Importing
        changes nothing served: a set runs only once attached to a behaviour (millm_set_behaviour_rule_signals)
        that has an armed probe. A DECLARATIVE set is one request, no approval. A set naming OPERATOR CODE (a
        `code_rule` signal, or the envelope carrying `code_modules`) waits for the OPERATOR in **miLLM's**
        Approvals page: the call returns miLLM's pending answer plus `"decide_in": "millm"` (its
        `request_digest` covers exactly this body), and once approved miLLM imports it once, by itself. Poll
        `millm_get_approval_status(approval_id)`. You cannot approve it yourself. A miLLM without its own
        approval queue is refused (`millm_has_no_approval_queue`) and nothing is sent."""
        if approval_request_id is not None:
            return moved_refusal("millm_import_rule_set")
        if not names_operator_code(document):
            ok, why = await gate.check("millm")
            if not ok:
                return {"unavailable": "millm", "reason": why}
        else:
            # ⚠ FAIL CLOSED (ENH-008 §3.2 step 3): operator code is sent only to a miLLM that gates it.
            refused = await _gate_and_queue("millm_import_rule_set")
            if refused is not None:
                return refused
        try:
            # The ONE import request (miLLM §4k/§4l), the same body either way.
            response = await millm.post("/api/rule-sets/import", json_body=document, version=version,
                                        digest=digest, origin=origin, full=True)
        except BackendError as exc:
            if exc.status_code == 0:   # a mid-TTL outage answers `unavailable`, as `gated` does
                gate.invalidate("millm")
                return {"unavailable": "millm", "reason": str(exc)}
            raise
        return relayed(response) or response.body

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_list_rule_sets(
        behaviour: Annotated[Optional[str], Field(description="Only sets for this behaviour key, e.g. '<behaviour-key>'")] = None,
        cursor: Annotated[Optional[str], Field(description="meta.next of the previous page")] = None,
        limit: Annotated[Optional[int], Field(description="Sets per page", ge=1, le=1000)] = None,
    ) -> Any:
        """miLLM's imported rule sets, paged by (behaviour key, id). Follow `meta.next` until null."""
        return await millm.get("/api/rule-sets", behaviour=behaviour, cursor=cursor, limit=limit)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_rule_set(rule_set_id: RULE_SET_ID) -> Any:
        """One imported rule set: its summary, forms per expression, provenance, whether its own cases still
        pass on this build, the behaviours it is attached to, what is stored but never applied at serving
        (vetoes, renderings, decoys, substitutions, outputs) and the document."""
        return await millm.get(f"/api/rule-sets/{rule_set_id}")

    @mcp.tool()
    async def millm_delete_rule_set(
        rule_set_id: RULE_SET_ID,
        acknowledge_delete: Annotated[bool, Field(description="Required: deleting detaches the set from every behaviour (recorded results are kept)")] = False,
        reason: Annotated[Optional[str], Field(description="Why; recorded in each behaviour's history")] = None,
    ) -> Any:
        """Delete an imported rule set. miLLM refuses it while it is attached to a behaviour with an armed probe
        (clear the attachment first); otherwise it is detached from every behaviour and deleted. Refused here
        unless `acknowledge_delete=true`."""
        if not acknowledge_delete:
            return {"refused": "delete_not_acknowledged", "rule_set_id": rule_set_id,
                    "message": "Deleting a rule set detaches it from every behaviour. Re-call with "
                               "acknowledge_delete=true if that is intended."}
        ok, why = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": why}
        return await millm.delete(f"/api/rule-sets/{rule_set_id}", reason=reason)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_evaluate_rule_set(
        rule_set_id: RULE_SET_ID,
        messages: Annotated[Optional[List[Dict[str, Any]]], Field(description="A chat [{role, content}] (or text)")] = None,
        text: Annotated[Optional[str], Field(description="A prompt (or messages)")] = None,
        response: Annotated[Optional[str], Field(description="The reply, for the response window")] = None,
        windows: Annotated[Optional[List[Literal["prompt", "last_user", "response", "all"]]], Field(description="Windows to evaluate (default every window the input has)")] = None,
        return_rule_positions: Annotated[bool, Field(description="Add code-point spans of each match")] = False,
    ) -> Any:
        """Run one stored set on YOUR text, per expression and window: `fires`, `count`, `outcome`. Nothing is
        written. Operator code reports `rule_worker_unavailable`."""
        body: Dict[str, Any] = {"return_rule_positions": return_rule_positions}
        for name, value in (("messages", messages), ("text", text), ("response", response), ("windows", windows)):
            if value is not None:
                body[name] = value
        return await millm.post(f"/api/rule-sets/{rule_set_id}/evaluate", json_body=body)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_get_behaviour_rule_signals(key: KEY) -> Any:
        """A behaviour's attached rule sets in miLLM (and the signals each is narrowed to), with the append-only
        attachment history."""
        return await millm.get(f"/api/probes/behaviours/{key}/rule-signals")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_set_behaviour_rule_signals(
        key: KEY,
        rule_set_ids: Annotated[List[str], Field(description="The rule sets to attach (replaces the list); each set's own behaviour_key must be this key")],
        signals: Annotated[Optional[Dict[str, List[str]]], Field(description="Narrow a set to named expressions: {rule_set_id: [expression, …]}; a set not named keeps every signal")] = None,
        reason: Annotated[Optional[str], Field(description="Why; recorded")] = None,
    ) -> Any:
        """Replace a behaviour's attached rule sets. ⚠ Attached signals run on LIVE traffic while any probe of
        the behaviour is armed, within miLLM's rule time budget, reporting beside the probes (never changing
        them; a declared veto is not applied at serving)."""
        body: Dict[str, Any] = {"rule_set_ids": list(rule_set_ids)}
        if signals is not None:
            body["signals"] = {k: list(v) for k, v in signals.items()}
        if reason is not None:
            body["reason"] = reason
        return await millm.put(f"/api/probes/behaviours/{key}/rule-signals", json_body=body)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_clear_behaviour_rule_signals(
        key: KEY,
        reason: Annotated[Optional[str], Field(description="Why; recorded")] = None,
    ) -> Any:
        """Detach every rule set from a behaviour in miLLM (the sets stay imported; history keeps the change)."""
        return await millm.delete(f"/api/probes/behaviours/{key}/rule-signals", reason=reason)
