"""Operator code rules: documents, approval requests and the operator's decisions (Feature 040 S4; FTDD Appendix C.6).

* `GET  /code-rules/{code_sha256}`                    — the stored document, its functions and its approval state
* `POST /code-rules/{code_sha256}/approval-requests`  — file a request (agents and the UI): the document's own cases
  run first in the isolated runner, at the contract's FLOOR limits, each on two fresh instances; a failing or
  non-deterministic case is refused at once, so the operator only ever sees code that passes. Answers the existing
  pending shape ``{approval_request_id, status: "pending_approval"}``.
* `POST /code-rules/{code_sha256}/revoke`             — OPERATOR ONLY (403 on `X-MiStudio-Agent`), reason required
* `POST /code-rules/{code_sha256}/approve`            — OPERATOR ONLY re-approve, reason required

The FIRST approval is the operator approving the filed request in the approvals banner
(`/mcp/approvals/{id}/approve`, the `code_rule_approve` RECORD branch), which also refuses agents (OSD-56). No MCP
tool revokes or approves (`test_code_rules_api.py` derives that from the live registry).

⚠ NOTHING HERE LOGS A SOURCE, A PARAM OR A TEXT.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....models.agent_approval import AgentApprovalRequest, ApprovalStatus
from ....services import code_rule_runner, code_rules
from ....services.detector_next import DetectorRefused
from ....services.probe_dataset_lifecycle import actor_of
from .mcp_approvals import AGENT_HEADER, _emit_approval_event, requested_by_of

router = APIRouter(prefix="/code-rules", tags=["code-rules"])

SHA = Path(..., pattern=r"^[0-9a-f]{64}$")


class CaseIn(BaseModel):
    """A `CodeRuleCase` of the runner contract (stored beside a code rule, NOT hashed)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    function: str = Field(pattern=r"^[a-z_][a-z0-9_]{0,63}$")
    params: Dict[str, Any] = Field(default_factory=dict)
    text: str = Field(max_length=65_536)
    expect: Dict[str, Any]
    source: Optional[Dict[str, Any]] = None


class ApprovalRequestIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: The document, when miStudio does not hold it yet (it must hash to the path's code_sha256).
    document: Optional[Dict[str, Any]] = None
    cases: List[CaseIn] = Field(min_length=1, max_length=200)


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=2000)


def refuse_agents(request: Request) -> None:
    """Revoke and re-approve are the operator's alone (C.6): any `X-MiStudio-Agent` value is refused."""
    if request.headers.get(AGENT_HEADER) is not None:
        raise HTTPException(status_code=403, detail={
            "code": "code_rule_operator_only",
            "message": "Approving and revoking operator code is the operator's, from miStudio; an agent files a "
                       "request with request_code_rule_approval and waits."})


async def _document(db: AsyncSession, sha: str, inline: Optional[Dict[str, Any]], actor: str):
    if inline is not None:
        got, canonical = code_rules.check_document(inline)
        if got != sha:
            raise DetectorRefused(422, "code_rule_digest_mismatch",
                                  f"the document hashes to {got[:12]}…, not {sha[:12]}…",
                                  {"code_sha256": sha, "computed": got})
        await code_rules.store(db, {sha: (dict(inline), canonical)}, actor)
        await db.commit()
    found = await code_rules.documents_for(db, [sha])
    code_rules.require_documents([sha], found)
    return found[sha]


@router.get("/{code_sha256}")
async def get_code_rule(code_sha256: str = SHA, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    found = await code_rules.documents_for(db, [code_sha256])
    if code_sha256 not in found:
        raise HTTPException(status_code=404, detail={"code": "code_rule_not_found",
                                                     "message": f"no code rule {code_sha256[:12]}… is stored here"})
    document, _canonical = found[code_sha256]
    state = (await code_rules.states(db, [code_sha256]))[code_sha256]
    return {"code_sha256": code_sha256, "runner_contract": document["runner_contract"],
            "functions": document["functions"], "document": document, **state,
            "runner": "available" if code_rule_runner.enabled() else "unavailable"}


@router.post("/{code_sha256}/approval-requests", status_code=202)
async def request_code_rule_approval(body: ApprovalRequestIn, request: Request, code_sha256: str = SHA,
                                     db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    from starlette.concurrency import run_in_threadpool

    actor = actor_of(request)
    try:
        document, canonical = await _document(db, code_sha256, body.document, actor)
        state = (await code_rules.states(db, [code_sha256]))[code_sha256]
        if state["state"] == "approved":
            return {"code_sha256": code_sha256, **state, "status": "approved"}
        if state["state"] == "pending":
            return {"approval_request_id": state["approval_request_id"], "status": "pending_approval",
                    "code_sha256": code_sha256, "duplicate": True}
        settings = code_rule_runner._effective()
        if not str(settings.code_rule_runner_url or "").strip():
            raise DetectorRefused(409, "runner_unavailable",
                                  "this miStudio has no operator-code runner (CODE_RULE_RUNNER_URL), so the code's "
                                  "own cases cannot be run before the operator sees it", {"code_sha256": code_sha256})
        declared = {f.get("name") for f in document.get("functions") or []}
        undeclared = [c.id for c in body.cases if c.function not in declared]
        if undeclared:
            raise DetectorRefused(422, "function_not_declared", "a case names a function the document does not declare",
                                  {"case_ids": undeclared})
        cases = [c.model_dump() for c in body.cases]
        url = str(settings.code_rule_runner_url)
        # 040 S4 debt review: the case run and the inspect share ONE interactive budget (an API/MCP caller waits
        # 60 s at most; they used to wait up to 120 s + 60 s on a runner that accepts and never answers).
        deadline = code_rule_runner._clock() + code_rule_runner.INTERACTIVE_BUDGET_S
        results = await run_in_threadpool(code_rules.run_own_cases, url, code_sha256, canonical, cases)
        if results.get("unavailable") or "refused" in results:
            raise DetectorRefused(503 if results.get("unavailable") else 422, "runner_refused_cases",
                                  "the runner could not run the code's own cases; nothing was filed",
                                  {"code_sha256": code_sha256, "runner": results.get("refused")})
        if results["nondeterministic"]:
            raise DetectorRefused(422, "code_rule_nondeterministic",
                                  "a case gave different answers on two fresh instances; nothing was filed",
                                  {"case_ids": results["nondeterministic"]})
        if results["failing"]:
            raise DetectorRefused(422, "code_rule_cases_failed",
                                  "the code's own cases fail in the runner at the floor limits; nothing was filed",
                                  {"case_ids": results["failing"], "cases": results["cases"]})
        facts = {"functions": document["functions"],
                 "inspect": await run_in_threadpool(code_rules.inspect, url, code_sha256, canonical, None,
                                                    max(1.0, deadline - code_rule_runner._clock())),
                 "case_results": results["cases"], "midataworks_state": None, "requested_by": actor}
        approval = AgentApprovalRequest(tool_name=code_rules.APPROVAL_TOOL,
                                        payload=code_rules.approval_payload(code_sha256, document, facts),
                                        status=ApprovalStatus.PENDING.value,
                                        requested_by=requested_by_of(request))
        db.add(approval)
        await db.commit()
        await db.refresh(approval)
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    _emit_approval_event("created", {"request_id": approval.id, "tool_name": approval.tool_name,
                                     "summary": f"Approve operator code {code_sha256[:12]}… "
                                                f"({', '.join(f.get('name', '?') for f in document['functions'])})"})
    return {"approval_request_id": approval.id, "status": "pending_approval", "code_sha256": code_sha256}


async def _decide(db: AsyncSession, sha: str, action: str, reason: str) -> Dict[str, Any]:
    try:
        return await code_rules.decide(db, sha, action, reason, "operator")
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.post("/{code_sha256}/revoke", dependencies=[Depends(refuse_agents)])
async def revoke_code_rule(body: DecisionIn, code_sha256: str = SHA,
                           db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Operator only. Takes effect at the next use — including the next chunk of a running bulk evaluation."""
    return await _decide(db, code_sha256, "revoke", body.reason)


@router.post("/{code_sha256}/approve", dependencies=[Depends(refuse_agents)])
async def reapprove_code_rule(body: DecisionIn, code_sha256: str = SHA,
                              db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Operator only: approve a hash again after a revocation (the first approval is the approvals banner's)."""
    return await _decide(db, code_sha256, "approve", body.reason)


__all__ = ["router", "refuse_agents"]
