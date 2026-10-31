"""Operator code-rule documents and per-hash operator approval in miStudio (Feature 040 S4; FTDD Appendix C.5, C.6).

* **Documents** (`code_rules`, content-addressed, append-only). A request may carry ``code_rules: [{document,
  cases?}]`` beside an inline or referenced rule set; miStudio validates each against the vendored runner contract's
  CodeRuleDocument schema, hashes it with the contract's rule (`rule_runner.contract.code_sha256`), refuses a
  document no code rule of the set references (`code_rule_unreferenced`) and stores the exact hashed bytes once.
  Referenced documents (fetched by hash from miDataworks) are not available yet: miDataworks has no
  `GET /api/v1/code-rules/by-hash/{sha}` route (verified 2026-10-24) — a missing document is
  `code_rule_document_missing` with the inline `next` (FTID addendum, S4).
* **Approval** (`code_rule_approvals`, append-only, ``origin = 'operator'`` CHECKed). Current state = the latest row
  per hash, READ FRESH at every use: `validate_rule_set` / `evaluate_rule_signals`, a rules evaluation's request,
  and each chunk of a bulk evaluation (`approved_now`, one indexed query). An agent files a request
  (`request_approval` → an `agent_approval_requests` row, tool `code_rule_approve`, after the document's own cases
  pass at the FLOOR limits on two fresh instances each); only the operator decides (`mcp_approvals` RECORD branch,
  and the operator-only revoke / re-approve routes). miDataworks' own approval is information, never a grant here.

⚠ PRIVACY: nothing here logs a source, a param or a text. The approval payload carries the document — the full
source is what the operator reads on the approval card before deciding (C.6).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from ..core.clock import utc_now
from ..models.agent_approval import AgentApprovalRequest, ApprovalStatus
from ..models.code_rule import CodeRule, CodeRuleApproval
from ..rule_runner import contract as runner_contract
from .detector_next import DetectorRefused, next_request_code_rule_approval

logger = logging.getLogger(__name__)

APPROVAL_TOOL = "code_rule_approve"
STATES = ("approved", "revoked", "pending", "unknown")


def code_signals(document: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in (document.get("signals") or []) if isinstance(s, dict) and s.get("kind") == "code_rule"]


def referenced_hashes(document: Mapping[str, Any]) -> List[str]:
    return sorted({str(s.get("code_sha256")) for s in code_signals(document)})


def _document_schema() -> Dict[str, Any]:
    """The contract file IS the CodeRuleDocument JSON schema (its other top-level keys are annotations)."""
    return runner_contract.load_contract()


def check_document(document: Any, *, index: int = 0) -> Tuple[str, bytes]:
    """`(code_sha256, canonical bytes)` of one CodeRuleDocument, or `DetectorRefused` naming the problem."""
    from jsonschema import Draft202012Validator

    errors = sorted(Draft202012Validator(_document_schema()).iter_errors(document), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        raise DetectorRefused(422, "code_rule_invalid",
                              f"code_rules[{index}].document is not a {runner_contract.FORMAT} code-rule document: "
                              f"{first.message[:200]}",
                              {"field": f"code_rules[{index}].document",
                               "pointer": "/" + "/".join(str(p) for p in first.path)})
    try:
        canonical = runner_contract.canonical_document_bytes(document)
    except runner_contract.CanonicalRefused as exc:
        raise DetectorRefused(422, "code_rule_invalid", f"code_rules[{index}].document cannot be canonicalised",
                              {"field": f"code_rules[{index}].document"}) from exc
    return runner_contract.code_sha256(document), canonical


def inline_documents(items: Optional[Sequence[Any]], rule_set: Optional[Mapping[str, Any]]
                     ) -> Dict[str, Tuple[Dict[str, Any], bytes]]:
    """``code_rules`` of a request → ``{sha: (document, canonical)}``; refuses one the set does not reference."""
    out: Dict[str, Tuple[Dict[str, Any], bytes]] = {}
    referenced = set(referenced_hashes(rule_set or {}))
    for i, item in enumerate(items or []):
        document = item.get("document") if isinstance(item, dict) else None
        sha, canonical = check_document(document, index=i)
        if rule_set is not None and sha not in referenced:
            raise DetectorRefused(422, "code_rule_unreferenced",
                                  f"code_rules[{i}] hashes to {sha[:12]}…, which no code_rule signal of the rule set "
                                  f"names", {"field": f"code_rules[{i}]", "code_sha256": sha})
        out[sha] = (dict(document), canonical)
    return out


def _row_values(sha: str, document: Mapping[str, Any], canonical: bytes, actor: str) -> Dict[str, Any]:
    return dict(code_sha256=sha, canonical=canonical, runner_contract=str(document["runner_contract"]),
                functions=list(document["functions"]), source="inline", source_ref=None, fetched_at=None,
                created_by=actor, created_at=utc_now())


async def store(db: Any, documents: Mapping[str, Tuple[Mapping[str, Any], bytes]], actor: str) -> None:
    """INSERT … ON CONFLICT DO NOTHING per document (content-addressed: never rewritten). The caller commits."""
    for sha, (document, canonical) in sorted(documents.items()):
        await db.execute(insert(CodeRule).values(**_row_values(sha, document, canonical, actor))
                         .on_conflict_do_nothing(index_elements=["code_sha256"]))


def _decode(row: CodeRule) -> Dict[str, Any]:
    return json.loads(bytes(row.canonical).decode("utf-8"))


async def documents_for(db: Any, shas: Iterable[str]) -> Dict[str, Tuple[Dict[str, Any], bytes]]:
    wanted = sorted(set(shas))
    if not wanted:
        return {}
    rows = (await db.execute(select(CodeRule).where(CodeRule.code_sha256.in_(wanted)))).scalars().all()
    return {r.code_sha256: (_decode(r), bytes(r.canonical)) for r in rows}


def documents_for_sync(db: Any, shas: Iterable[str]) -> Dict[str, Tuple[Dict[str, Any], bytes]]:
    wanted = sorted(set(shas))
    if not wanted:
        return {}
    rows = db.query(CodeRule).filter(CodeRule.code_sha256.in_(wanted)).all()
    return {r.code_sha256: (_decode(r), bytes(r.canonical)) for r in rows}


def _latest_query(shas: Sequence[str]) -> Any:
    latest = (select(CodeRuleApproval.code_sha256, func.max(CodeRuleApproval.id).label("id"))
              .where(CodeRuleApproval.code_sha256.in_(list(shas))).group_by(CodeRuleApproval.code_sha256).subquery())
    return select(CodeRuleApproval).join(latest, CodeRuleApproval.id == latest.c.id)


def _pending_query(shas: Sequence[str]) -> Any:
    return select(AgentApprovalRequest).where(
        AgentApprovalRequest.tool_name == APPROVAL_TOOL,
        AgentApprovalRequest.status == ApprovalStatus.PENDING.value,
        AgentApprovalRequest.payload["code_sha256"].astext.in_(list(shas)))


def _states(shas: Sequence[str], latest: Sequence[CodeRuleApproval],
            pending: Sequence[AgentApprovalRequest]) -> Dict[str, Dict[str, Any]]:
    by_sha = {r.code_sha256: r for r in latest}
    waiting = {str((p.payload or {}).get("code_sha256")): p.id for p in pending}
    out: Dict[str, Dict[str, Any]] = {}
    for sha in shas:
        row = by_sha.get(sha)
        if row is not None and row.action == "approve":
            out[sha] = {"state": "approved", "approval_row_id": row.id}
        elif sha in waiting:
            out[sha] = {"state": "pending", "approval_request_id": waiting[sha],
                        "revoked_row_id": row.id if row is not None else None}
        elif row is not None:
            out[sha] = {"state": "revoked", "approval_row_id": row.id}
        else:
            out[sha] = {"state": "unknown"}
    return out


async def states(db: Any, shas: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    wanted = sorted(set(shas))
    if not wanted:
        return {}
    latest = (await db.execute(_latest_query(wanted).execution_options(populate_existing=True))).scalars().all()
    pending = (await db.execute(_pending_query(wanted))).scalars().all()
    return _states(wanted, latest, pending)


def approved_now(db: Any, sha: str) -> bool:
    """The FRESH read a bulk evaluation makes before each chunk (C.6): one indexed query, no cache — in a SHORT
    session of its own on the task's engine, so it sees a revocation committed meanwhile without touching the
    task's own transaction."""
    from sqlalchemy.orm import Session

    with Session(bind=db.get_bind()) as fresh:
        action = (fresh.query(CodeRuleApproval.action).filter(CodeRuleApproval.code_sha256 == sha)
                  .order_by(CodeRuleApproval.id.desc()).limit(1).scalar())
    return action == "approve"


def require_approved(found: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse `code_rule_not_approved` naming the first hash that is not approved (C.6)."""
    for sha, state in sorted(found.items()):
        if state["state"] != "approved":
            raise DetectorRefused(409, "code_rule_not_approved",
                                  f"code rule {sha[:12]}… is {state['state']} in this miStudio; the operator approves "
                                  f"operator code per hash before it runs",
                                  {"code_sha256": sha, "state": state["state"]},
                                  next=[next_request_code_rule_approval(sha)])


def require_documents(shas: Iterable[str], documents: Mapping[str, Any]) -> None:
    for sha in sorted(set(shas)):
        if sha not in documents:
            raise DetectorRefused(409, "code_rule_document_missing",
                                  f"the rule set names code {sha[:12]}… and miStudio holds no document for it; send it "
                                  f"inline (code_rules: [{{document}}]) or file it with request_code_rule_approval",
                                  {"code_sha256": sha}, next=[next_request_code_rule_approval(sha)])


# ── the operator's decisions (never an agent's) ──────────────────────────────────────────────────


def decision_row(sha: str, action: str, reason: str, changed_by: str,
                 approval_request_id: Optional[str] = None) -> CodeRuleApproval:
    if action not in ("approve", "revoke"):
        raise ValueError(action)
    reason = (reason or "").strip()
    if not reason:
        raise DetectorRefused(422, "reason_required", "a decision on operator code needs a reason", {"field": "reason"})
    return CodeRuleApproval(code_sha256=sha, action=action, reason=reason[:2000], changed_by=changed_by,
                            origin="operator", approval_request_id=approval_request_id, created_at=utc_now())


async def decide(db: Any, sha: str, action: str, reason: str, changed_by: str) -> Dict[str, Any]:
    """Operator revoke / re-approve (routes refuse `X-MiStudio-Agent`). Commits."""
    if (await db.execute(select(CodeRule.code_sha256).where(CodeRule.code_sha256 == sha))).scalar_one_or_none() is None:
        raise DetectorRefused(404, "code_rule_not_found", f"no code rule {sha[:12]}… is stored here",
                              {"code_sha256": sha})
    row = decision_row(sha, action, reason, changed_by)
    db.add(row)
    await db.commit()
    return {"code_sha256": sha, **(await states(db, [sha]))[sha]}


def approval_payload(sha: str, document: Mapping[str, Any], facts: Mapping[str, Any]) -> Dict[str, Any]:
    return {"code_sha256": sha, "document": dict(document), "facts": dict(facts)}


def record_approval(approval: AgentApprovalRequest, changed_by: str = "operator") -> CodeRuleApproval:
    """The RECORD execution of `mcp_approvals.approve_request`: re-hash the stored document and return the
    `approve` row to add IN THE SAME TRANSACTION that marks the request approved (C.6)."""
    payload = approval.payload or {}
    sha = str(payload.get("code_sha256"))
    document = payload.get("document")
    if not isinstance(document, dict) or runner_contract.code_sha256(document) != sha:
        raise DetectorRefused(409, "code_rule_digest_mismatch",
                              "the request's document no longer hashes to its code_sha256; it cannot be approved",
                              {"code_sha256": sha})
    return decision_row(sha, "approve", f"approved request {approval.id}", changed_by, approval.id)


# ── an agent's request (cases first, at the floor, twice each) ────────────────────────────────────


def _floor_limits() -> Dict[str, int]:
    floor = runner_contract.floor()
    out = {name: int(floor[name]) for name in runner_contract.request_limits() if name in floor}
    out["wall_ms"] = 10_000
    return out


def run_own_cases(base_url: str, sha: str, canonical: bytes, cases: Sequence[Mapping[str, Any]],
                  client: Any = None) -> Dict[str, Any]:
    """018 FR-018.20: each case on TWO fresh instances at the floor limits (`repeat: 2`, `save_check` mode).
    Returns ``{"cases": [...], "failing": [...], "nondeterministic": [...]}``; ``unavailable`` when the runner is.

    040 S4 debt review (2026-10-25): this runs on the API path (the approval-request route, an MCP tool), so it waits
    at most the interactive budget, `INTERACTIVE_BUDGET_S`, never the 120 s it used to.
    The runner answers it within 2 × the floor wall time (its lane backstop plus one call)."""
    from .code_rule_runner import INTERACTIVE_BUDGET_S, http_client

    own = client or http_client(INTERACTIVE_BUDGET_S)
    body = {"contract": runner_contract.FORMAT, "mode": "save_check", "lane": "interactive",
            "limits": _floor_limits(),
            "document": {"code_sha256": sha, "canonical_utf8": canonical.decode("utf-8")},
            "calls": [{"function": c["function"], "params": dict(c.get("params") or {}),
                       "texts": [runner_contract.replace_lone_surrogates(c["text"])], "repeat": 2} for c in cases]}
    try:
        response = own.post(base_url.rstrip("/") + "/v1/run", json=body)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rule runner unreachable (%s) running the cases of %s…", type(exc).__name__, sha[:12])
        return {"unavailable": True}
    if response.status_code != 200:
        code = None
        try:
            code = response.json().get("error", {}).get("code")
        except Exception:  # noqa: BLE001
            pass
        return {"unavailable": response.status_code >= 500, "refused": code or response.status_code}
    results = response.json().get("results") or []
    # Review R2 (S4): `rule_worker_unavailable` is the RUNNER's fault (its lane backstop, a bundle it refused) — no
    # guest can produce it (`REASONS_WITHOUT_VECTOR`). It says nothing about the code, so it is never read as a failing
    # or non-deterministic case: the request is refused as "the runner could not run the cases".
    if any(isinstance(answer, dict) and answer.get("reason") == runner_contract.Reason.RULE_WORKER_UNAVAILABLE.value
           for pair in results if isinstance(pair, list) for answer in pair):
        return {"unavailable": True, "refused": runner_contract.Reason.RULE_WORKER_UNAVAILABLE.value}
    out: Dict[str, Any] = {"cases": [], "failing": [], "nondeterministic": []}
    for case, pair in zip(cases, results):
        first, second = (pair + [None, None])[:2]
        expect = case.get("expect") or {}
        passed = first == expect
        out["cases"].append({"id": case.get("id"), "expect": expect, "got": first, "same_twice": first == second})
        if first != second:
            out["nondeterministic"].append(case.get("id"))
        elif not passed:
            out["failing"].append(case.get("id"))
    return out


def inspect(base_url: str, sha: str, canonical: bytes, client: Any = None,
            timeout_s: Optional[float] = None) -> Dict[str, Any]:
    from .code_rule_runner import INTERACTIVE_BUDGET_S, http_client

    own = client or http_client(INTERACTIVE_BUDGET_S if timeout_s is None else timeout_s)
    try:
        response = own.post(base_url.rstrip("/") + "/v1/inspect", json={
            "contract": runner_contract.FORMAT,
            "document": {"code_sha256": sha, "canonical_utf8": canonical.decode("utf-8")}})
        return dict(response.json()) if response.status_code == 200 else {"unavailable": True}
    except Exception:  # noqa: BLE001
        return {"unavailable": True}
