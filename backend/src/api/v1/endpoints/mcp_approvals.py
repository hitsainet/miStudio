"""
Agent approval endpoints (Feature 010, operator-approval mode).

When the MCP server runs with MCP_STEERING_APPROVAL=true, agent steering
calls land here as pending requests instead of hitting /steering/async/*
directly. An operator approves or denies them (the approvals banner above every panel); on
approval the backend submits the stored payload to the corresponding steering
endpoint itself via an internal self-call — reusing all of that endpoint's
validation, rate limiting, and worker autostart — and records the resulting
steering task id for the agent to poll.

Feature 034 (FR-20) adds CALLER-EXECUTED tools. `millm_load_model` evicts the model miLLM serves,
so an agent's load waits for the operator too — but the backend has no miLLM URL and cannot run
the load itself (T-98). So approving a caller-executed request only marks it `approved`; the MCP
tool re-calls with the approval id, CONSUMES it here (`POST /{id}/consume`, one conditional
UPDATE — the only once-guarantee) and then loads. Approve is the operator's act and consume the
executor's: no MCP code path can approve — and since OSD-56 (2026-10-24) the two decision routes
refuse any caller sending `X-MiStudio-Agent` (403 `approval_operator_only`).

Caller-executed approvals EXPIRE: pending, or approved and unconsumed, for longer than
`mcp_approval_ttl_hours` (default 24). Expiry is lazy — every route runs `expire_stale` first —
so a stale request never reads `pending`. Steering approvals never expire (S3-03).

ENH-008 (2026-10-28): miLLM decides its own requests now, so the four `millm_*` tools no longer file here (the
proxies relay miLLM's pending answer). Each row records who asked (`requested_by`) and who decided (`decided_by`),
`GET` computes `expires_at`, and `decided=true` lists the Approvals page's History, newest decision first.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.runtime_settings import effective
from src.core.deps import get_db
from src.models.agent_approval import AgentApprovalRequest, ApprovalStatus
from src.workers.websocket_emitter import emit_progress

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp/approvals", tags=["mcp-approvals"])

#: The header every MCP request carries (`mcp_server/client.py`); the frontend never sends it.
AGENT_HEADER = "X-MiStudio-Agent"

#: ENH-008: `decided_by` on approve and deny. The suite has no login (operator decision 2026-10-06), so the
#: decider is always "operator"; an expiry records "system".
OPERATOR = "operator"


def refuse_agent_decisions(request: Request) -> None:
    """OSD-56 (040 Appendix C.6, 040-A5): approving or denying is the OPERATOR's act.

    On `/approve` and `/deny` only — never on `POST /mcp/approvals` (agents file requests) or
    `/consume` (the MCP tool spends a caller-executed approval). Any value of the header, even an
    empty or malformed one, is refused, as `refuse_agents` does on the purge routes.
    """
    if request.headers.get(AGENT_HEADER) is not None:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "approval_operator_only",
                "message": (
                    "Approving or denying is the operator's, from miStudio's approvals banner. "
                    "An agent files a request and waits; it can never decide one."
                ),
            },
        )


class Execution(str, Enum):
    """Who carries out an approved request."""

    #: The backend, on approve, by an internal self-call (steering).
    BACKEND = "backend"
    #: The MCP tool that asked, on re-call, after consuming the approval (miLLM model load).
    CALLER = "caller"
    #: Feature 040 S4 (FTDD C.6, 040-A4): nothing is executed or consumed — approving WRITES a durable record (an
    #: operator `approve` row in `code_rule_approvals`, per code hash) in the same transaction that marks the request
    #: approved. It persists until the operator revokes it.
    RECORD = "record"


#: Steering tools. ONE set: `APPROVABLE_TOOLS` is built from it and `expire_stale` excludes it, so
#: the "steering never expires" rule cannot drift from the list of steering tools.
STEERING_TOOLS = frozenset({"steer_compare", "steer_sweep", "steer_combined"})

#: tool name → (who executes it, backend self-call path or None).
APPROVABLE_TOOLS: dict[str, tuple[Execution, Optional[str]]] = {
    **{tool: (Execution.BACKEND, f"/api/v1/steering/async/{tool.removeprefix('steer_')}")
       for tool in sorted(STEERING_TOOLS)},
    # ENH-008 (D1, 2026-10-28): the four `millm_*` tools that used to file here — `millm_load_model`,
    # `millm_request_probe_limits`, `millm_measure_probe_capacity`, `millm_import_rule_set` — are GONE from this
    # table. miLLM decides its own requests in its own queue, and the proxies relay its pending answer; filing one
    # here is refused 422. Requests still waiting at switch-over were expired by `e008_millm_approvals_moved.py`.
    # 2026-10-19 (MCP gap closure): an agent's probe training run and probe evaluation each hold
    # a GPU for minutes to hours, behind which every other GPU job queues. Caller-executed like
    # the miLLM load: approving marks the row, the MCP tool consumes it and submits exactly the
    # payload the operator saw. Gated by `MCP_PROBE_GPU_APPROVAL` (default on).
    "start_probe_training_run": (Execution.CALLER, None),
    "evaluate_probe_monitor": (Execution.CALLER, None),
    # 2026-10-21 (judge-prompt library): an agent's judge run against miLLM is thousands of
    # generations on the GPU that serves miLLM's users, and may load the judge model (evicting the
    # served one). Gated only when the endpoint is miLLM — see `judge_endpoint_is_millm`.
    "start_probe_judge_run": (Execution.CALLER, None),
    # Feature 038 (D34): a spot-check judged by miLLM queues live chats behind every judge call.
    "start_probe_spot_check": (Execution.CALLER, None),
    # Feature 040 S3 (review R1-5): the GPU arm of a recalibration (`recut_windows=true`) re-scores a probe's
    # per-window negatives on a card; the arithmetic arm never asks.
    "recalibrate_probe_monitor": (Execution.CALLER, None),
    # Feature 040 S4 (FTDD Appendix C.6): an agent's request to run operator code (a code rule, by hash) in the
    # isolated runner. Approving records the operator's per-hash approval; it is never consumed.
    "code_rule_approve": (Execution.RECORD, None),
}

#: A view kept for existing references: tool name → steering async endpoint path.
STEERING_ENDPOINTS = {
    tool: endpoint
    for tool, (kind, endpoint) in APPROVABLE_TOOLS.items()
    if kind is Execution.BACKEND
}

CALLER_TOOLS = frozenset(
    tool for tool, (kind, _) in APPROVABLE_TOOLS.items() if kind is Execution.CALLER
)

#: Review R2 (S4): tools whose approval WRITES a durable record. Each is filed only by its own route (operator code:
#: `POST /code-rules/{sha}/approval-requests`, which runs the code's own cases first and stores the exact document),
#: never through `POST /mcp/approvals`; and an approved one is never consumed, so it never "expires unused".
RECORD_TOOLS = frozenset(
    tool for tool, (kind, _) in APPROVABLE_TOOLS.items() if kind is Execution.RECORD
)


class ApprovalCreateRequest(BaseModel):
    tool_name: str
    payload: dict[str, Any]

    @field_validator("tool_name")
    @classmethod
    def _approvable(cls, value: str) -> str:
        if value not in APPROVABLE_TOOLS:
            raise ValueError(
                f"{value!r} is not an approvable tool (one of {sorted(APPROVABLE_TOOLS)})"
            )
        return value


class ConsumeRequest(BaseModel):
    """The executor's claim: this tool, with exactly this payload."""

    tool_name: str
    payload: dict[str, Any]


class ApprovalResponse(BaseModel):
    id: str
    tool_name: str
    payload: dict[str, Any]
    status: str
    reason: Optional[str] = None
    steering_task_id: Optional[str] = None
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    consumed_at: Optional[datetime] = None
    #: ENH-008 (§3.2 step 7): who asked ("mcp", "operator") and who decided ("operator", "system").
    requested_by: Optional[str] = None
    decided_by: Optional[str] = None
    #: ENH-008: when a request that can still lapse will lapse — pending (created + TTL), or approved and not yet
    #: consumed (approved + TTL). NULL for steering (never expires), for an approved record tool (never consumed)
    #: and for every decided row. Computed on read from `mcp_approval_ttl_hours`, never stored.
    expires_at: Optional[datetime] = None
    #: ENH-006 D81: shared by every request approved in one "Approve all" call; NULL otherwise.
    batch_id: Optional[str] = None

    model_config = {"from_attributes": True}


class ApprovalListResponse(BaseModel):
    approvals: list[ApprovalResponse]
    total: int


#: Longest `requested_by` kept (the column's width).
REQUESTED_BY_MAX = 64


def requested_by_of(request: Request) -> str:
    """ENH-008: who filed a request — the `X-MiStudio-Agent` value (the MCP server sends "mcp"), else "operator"
    (the UI sends no header). Stripped and cut to the column; never refused here (filing must not break on a
    label)."""
    raw = request.headers.get(AGENT_HEADER)
    if raw is None:
        return "operator"
    return raw.strip()[:REQUESTED_BY_MAX] or "agent"


def expires_at_of(approval: AgentApprovalRequest, ttl_hours: float) -> Optional[datetime]:
    """When this request will lapse if nothing happens, or `None` (see `ApprovalResponse.expires_at`).

    Mirrors `expire_stale` exactly: steering never lapses; a pending request lapses `ttl` after it was filed; an
    approved CALLER request lapses `ttl` after it was approved unless consumed; a RECORD approval never lapses."""
    if approval.tool_name in STEERING_TOOLS:
        return None
    ttl = timedelta(hours=ttl_hours)
    if approval.status == ApprovalStatus.PENDING.value and approval.created_at is not None:
        return approval.created_at + ttl
    if (approval.status == ApprovalStatus.APPROVED.value and approval.resolved_at is not None
            and approval.tool_name not in RECORD_TOOLS):
        return approval.resolved_at + ttl
    return None


def respond(approval: AgentApprovalRequest, ttl_hours: Optional[float] = None) -> ApprovalResponse:
    """The one serialiser for an approval row (memory: one list needs one serialiser)."""
    if ttl_hours is None:
        ttl_hours = effective().mcp_approval_ttl_hours
    out = ApprovalResponse.model_validate(approval)
    out.expires_at = expires_at_of(approval, ttl_hours)
    return out


class DenyRequest(BaseModel):
    #: ENH-008 (D8c): the UI sends the operator's reason (at most 1000 characters) or
    #: "Rejected by the operator." when the field is left blank.
    reason: Optional[str] = Field(None, max_length=2000)


def _emit_approval_event(event: str, data: dict[str, Any]) -> None:
    emit_progress(channel="mcp/approvals", event=f"approval:{event}", data=data)


async def expire_stale(db: AsyncSession) -> list[str]:
    """Move stale CALLER-EXECUTED approvals to `expired`; return their ids.

    Two conditional statements, both excluding every steering tool (S3-03: steering approvals
    never expire):

    * pending longer than the TTL, by `created_at`;
    * approved and not consumed longer than the TTL, by `resolved_at` — an approval the agent
      never came back for must not stay spendable indefinitely.

    `synchronize_session=False`: a criteria UPDATE otherwise evaluates its WHERE in Python against
    the session's loaded rows (the ORM bulk-statement lesson), comparing an aware cutoff to
    whatever the row holds. The routes re-read rows after this runs.
    """
    # Read per call (Feature 036): an operator's override applies at the next approval check.
    ttl_hours = effective().mcp_approval_ttl_hours
    ttl = timedelta(hours=ttl_hours)
    now = datetime.now(timezone.utc)
    cutoff = now - ttl
    model = AgentApprovalRequest
    pending = (
        update(model)
        .where(
            model.status == ApprovalStatus.PENDING.value,
            model.created_at < cutoff,
            model.tool_name.not_in(sorted(STEERING_TOOLS)),
        )
        .values(
            status=ApprovalStatus.EXPIRED.value,
            resolved_at=now,
            reason=f"expired: pending for more than {ttl_hours} h",
            # ENH-008: nobody decided; the TTL did.
            decided_by="system",
        )
        .returning(model.id)
        .execution_options(synchronize_session=False)
    )
    unconsumed = (
        update(model)
        .where(
            model.status == ApprovalStatus.APPROVED.value,
            model.resolved_at < cutoff,
            model.tool_name.not_in(sorted(STEERING_TOOLS | RECORD_TOOLS)),
        )
        .values(
            status=ApprovalStatus.EXPIRED.value,
            reason=(
                f"expired: approved but not used within {ttl_hours} h"
            ),
        )
        .returning(model.id)
        .execution_options(synchronize_session=False)
    )
    expired = list((await db.execute(pending)).scalars().all())
    expired += list((await db.execute(unconsumed)).scalars().all())
    if expired:
        await db.commit()
        logger.info("expired %d stale agent approval(s)", len(expired))
        for request_id in expired:
            _emit_approval_event("resolved", {"request_id": request_id, "status": "expired"})
    return expired


@router.post("", response_model=ApprovalResponse, status_code=201)
async def create_approval_request(
    request: ApprovalCreateRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a pending approval request (called by the MCP server)."""
    if request.tool_name in RECORD_TOOLS:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "approval_filed_by_its_route",
                "message": (
                    f"{request.tool_name} is filed by its own route, which runs the code's own cases in "
                    "the isolated runner first (request_code_rule_approval); it cannot be filed here."
                ),
            },
        )
    approval = AgentApprovalRequest(
        tool_name=request.tool_name,
        payload=request.payload,
        status=ApprovalStatus.PENDING.value,
        requested_by=requested_by_of(http_request),
    )
    db.add(approval)
    await db.commit()
    await db.refresh(approval)

    _emit_approval_event(
        "created",
        {
            "request_id": approval.id,
            "tool_name": approval.tool_name,
            "summary": _summarize(request.tool_name, request.payload),
        },
    )
    return respond(approval)


@router.get("", response_model=ApprovalListResponse)
async def list_approval_requests(
    status: Optional[str] = Query(
        None, description="Filter: pending|approved|denied|expired|consumed"
    ),
    decided: bool = Query(
        False,
        description="ENH-008: only DECIDED requests (every status but pending), the most recent decision "
                    "first — the Approvals page's History",
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0, description="Skip this many (History's Show older)"),
    db: AsyncSession = Depends(get_db),
):
    await expire_stale(db)
    model = AgentApprovalRequest
    if decided:
        # ⚠ THE HISTORY FILTER: everything that is no longer waiting, newest decision first.
        query = select(model).where(model.status != ApprovalStatus.PENDING.value).order_by(
            model.resolved_at.desc().nulls_last(), model.created_at.desc(), model.id
        )
    else:
        query = select(model).order_by(model.created_at.desc(), model.id)
    if status:
        query = query.where(model.status == status)
    result = await db.execute(
        query.offset(offset).limit(limit).execution_options(populate_existing=True)
    )
    approvals = result.scalars().all()
    ttl_hours = effective().mcp_approval_ttl_hours
    return ApprovalListResponse(
        approvals=[respond(a, ttl_hours) for a in approvals],
        total=len(approvals),
    )


@router.get("/{request_id}", response_model=ApprovalResponse)
async def get_approval_request(request_id: str, db: AsyncSession = Depends(get_db)):
    await expire_stale(db)
    approval = await _get_or_404(db, request_id)
    return respond(approval)


@router.post("/{request_id}/approve", response_model=ApprovalResponse,
             dependencies=[Depends(refuse_agent_decisions)])
async def approve_request(request_id: str, db: AsyncSession = Depends(get_db)):
    """Approve.

    Backend-executed (steering): the backend submits the stored payload itself. Caller-executed
    (`millm_load_model`): the row is marked approved and nothing else happens here — the MCP tool
    consumes it on re-call. An expired request cannot be approved (409).
    """
    await expire_stale(db)
    approval = await _get_or_404(db, request_id)
    if approval.status != ApprovalStatus.PENDING.value:
        raise HTTPException(409, f"Request is {approval.status}, not pending")
    return respond(await _approve_pending(db, approval))


async def _approve_pending(db: AsyncSession, approval: AgentApprovalRequest,
                           batch_id: Optional[str] = None) -> AgentApprovalRequest:
    """THE approve path, shared by `POST /{id}/approve` and `POST /approve-batch` (ENH-006 D81): one request,
    already checked pending, decided by the operator and committed on its own. `batch_id` is stamped on the row
    in the same commit that marks it approved. Raises `HTTPException` when this request cannot be approved."""
    request_id = approval.id
    kind, endpoint = APPROVABLE_TOOLS.get(approval.tool_name, (None, None))
    if kind is Execution.RECORD:
        # ⚠ ONE TRANSACTION (C.6): the operator's `approve` row (origin 'operator', the request id) and the request's
        # `approved` status commit together, after the stored document is re-hashed.
        from ....services import code_rules
        from ....services.detector_next import DetectorRefused

        try:
            row = code_rules.record_approval(approval)
        except DetectorRefused as refusal:
            raise refusal.to_http() from refusal
        db.add(row)
        approval.status = ApprovalStatus.APPROVED.value
        approval.resolved_at = datetime.now(timezone.utc)
        approval.decided_by = OPERATOR
        approval.batch_id = batch_id
        await db.commit()
        await db.refresh(approval)
        _emit_approval_event("resolved", {"request_id": approval.id, "status": "approved"})
        logger.info("Approval %s approved (%s, recorded%s)", request_id, approval.tool_name,
                    f", batch {batch_id}" if batch_id else "")
        return approval
    if kind is Execution.CALLER:
        approval.status = ApprovalStatus.APPROVED.value
        approval.resolved_at = datetime.now(timezone.utc)
        approval.decided_by = OPERATOR
        approval.batch_id = batch_id
        await db.commit()
        await db.refresh(approval)
        _emit_approval_event(
            "resolved", {"request_id": approval.id, "status": "approved"}
        )
        logger.info(f"Approval {request_id} approved ({approval.tool_name}, caller-executed)")
        return approval
    if not endpoint:
        raise HTTPException(422, f"Unknown tool {approval.tool_name}")

    # Internal self-call reuses the steering endpoint's validation, rate
    # limiting, and worker autostart (same pattern as the internal WS emit).
    url = f"{settings.internal_api_url}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, json=approval.payload)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Failed to submit steering task: {e}")

    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if "json" in response.headers.get("content-type", "") else response.text
        raise HTTPException(502, f"Steering submission rejected ({response.status_code}): {detail}")

    task_id = response.json().get("task_id")
    approval.status = ApprovalStatus.APPROVED.value
    approval.steering_task_id = task_id
    approval.resolved_at = datetime.now(timezone.utc)
    approval.decided_by = OPERATOR
    approval.batch_id = batch_id
    await db.commit()
    await db.refresh(approval)

    _emit_approval_event(
        "resolved",
        {"request_id": approval.id, "status": "approved", "steering_task_id": task_id},
    )
    logger.info(f"Approval {request_id} approved → steering task {task_id}")
    return approval


#: ENH-006 D81 / D79: what each batch kind may contain. Never mixed: a batch is ONE kind, and any id of another
#: kind refuses the whole batch. miStudio's queue has no settings-change request (D79), so "settings" is refused.
BATCH_KINDS: dict[str, frozenset[str]] = {"code_rules": RECORD_TOOLS & frozenset({"code_rule_approve"})}

#: Most ids one batch may carry (the list route's maximum page).
BATCH_MAX = 500


class ApproveBatchRequest(BaseModel):
    kind: str
    approval_ids: list[str] = Field(min_length=1, max_length=BATCH_MAX)


class ApproveBatchItem(BaseModel):
    id: str
    status: str  # "approved" | "failed"
    error: Optional[str] = None
    approval: Optional[ApprovalResponse] = None


class ApproveBatchResponse(BaseModel):
    batch_id: str
    kind: str
    results: list[ApproveBatchItem]


def _batch_refusal(code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(422, {"code": code, "message": message, **extra})


def _failure_text(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("code") or detail)
    return str(detail)


@router.post("/approve-batch", response_model=ApproveBatchResponse,
             dependencies=[Depends(refuse_agent_decisions)])
async def approve_batch(body: ApproveBatchRequest, db: AsyncSession = Depends(get_db)) -> ApproveBatchResponse:
    """Approve every listed request of ONE kind with one operator call (ENH-006 D81).

    1. The kind must be one miStudio batches (`code_rules`); `settings` is refused — there are none here (D79).
    2. The WHOLE batch is checked before anything is approved: an id that is missing, not pending, or of another
       kind is named in a 409 and nothing changes.
    3. Each id is approved through `_approve_pending`, the single approve's own path, committed on its own and
       stamped with one shared `batch_id`. A request that fails (a document that no longer hashes to its code, a
       request decided meanwhile) is reported and the rest carry on.
    There is no batch reject.
    """
    if body.kind == "settings":
        raise _batch_refusal("approve_batch_no_settings_requests",
                             "miStudio's queue holds no settings-change requests: runtime settings are the "
                             "operator's, from Settings → Environment, and are never filed for approval.")
    tools = BATCH_KINDS.get(body.kind)
    if tools is None:
        raise _batch_refusal("approve_batch_unknown_kind", f"{body.kind!r} is not a batch kind (one of "
                             f"{sorted(BATCH_KINDS)})")
    ids = list(body.approval_ids)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise _batch_refusal("approve_batch_duplicate_ids", "an id is listed more than once", ids=dupes)

    await expire_stale(db)
    model = AgentApprovalRequest
    found = {
        r.id: r for r in (await db.execute(
            select(model).where(model.id.in_(ids)).execution_options(populate_existing=True))).scalars()
    }
    refused = []
    for rid in ids:
        row = found.get(rid)
        if row is None:
            refused.append({"id": rid, "why": "not found"})
        elif row.tool_name not in tools:
            refused.append({"id": rid, "why": f"is a {row.tool_name} request, not one of kind {body.kind}"})
        elif row.status != ApprovalStatus.PENDING.value:
            refused.append({"id": rid, "why": f"is {row.status}, not pending"})
    if refused:
        raise HTTPException(409, {
            "code": "approve_batch_refused",
            "message": f"{len(refused)} of {len(ids)} ids are not waiting {body.kind} requests; nothing was "
                       "approved",
            "refused": refused,
        })

    batch_id = str(uuid.uuid4())
    ttl_hours = effective().mcp_approval_ttl_hours
    results: list[ApproveBatchItem] = []
    for rid in ids:
        try:
            # Re-read: an earlier item's rollback expires the session's rows.
            approval = await _get_or_404(db, rid)
            if approval.status != ApprovalStatus.PENDING.value:
                raise HTTPException(409, f"Request is {approval.status}, not pending")
            done = await _approve_pending(db, approval, batch_id=batch_id)
            results.append(ApproveBatchItem(id=rid, status="approved", approval=respond(done, ttl_hours)))
        except HTTPException as exc:
            await db.rollback()
            results.append(ApproveBatchItem(id=rid, status="failed", error=_failure_text(exc)))
        except Exception as exc:  # noqa: BLE001 — one item's fault never stops the others
            await db.rollback()
            logger.exception("batch %s: approving %s failed", batch_id, rid)
            results.append(ApproveBatchItem(id=rid, status="failed", error=f"{type(exc).__name__}: {exc}"))
    logger.info("batch %s (%s): %d of %d approved", batch_id, body.kind,
                sum(r.status == "approved" for r in results), len(ids))
    return ApproveBatchResponse(batch_id=batch_id, kind=body.kind, results=results)


@router.post("/{request_id}/deny", response_model=ApprovalResponse,
             dependencies=[Depends(refuse_agent_decisions)])
async def deny_request(
    request_id: str,
    request: DenyRequest,
    db: AsyncSession = Depends(get_db),
):
    await expire_stale(db)
    approval = await _get_or_404(db, request_id)
    if approval.status != ApprovalStatus.PENDING.value:
        raise HTTPException(409, f"Request is {approval.status}, not pending")

    approval.status = ApprovalStatus.DENIED.value
    approval.reason = request.reason
    approval.resolved_at = datetime.now(timezone.utc)
    approval.decided_by = OPERATOR
    await db.commit()
    await db.refresh(approval)

    _emit_approval_event("resolved", {"request_id": approval.id, "status": "denied"})
    return respond(approval)


@router.post("/{request_id}/consume", response_model=ApprovalResponse)
async def consume_request(
    request_id: str,
    request: ConsumeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Spend an approved caller-executed request — once, and only for the payload approved.

    Called by the MCP tool that executes the action (`millm_load_model`), never by the operator.
    Refuses (422) a backend-executed tool, a different tool and a different payload; refuses
    (409, `{status, reason}`) anything not currently `approved` — pending, denied, expired or
    already consumed. The once-guarantee is ONE conditional UPDATE: two concurrent consumes of
    one approval get one 200 and one 409.
    """
    await expire_stale(db)
    approval = await _get_or_404(db, request_id)

    kind, _ = APPROVABLE_TOOLS.get(approval.tool_name, (None, None))
    if kind is not Execution.CALLER:
        raise HTTPException(
            422,
            f"{approval.tool_name} is executed by the backend when approved; there is nothing "
            "to consume",
        )
    if request.tool_name != approval.tool_name:
        raise HTTPException(
            422,
            f"approval {request_id} is for {approval.tool_name}, not {request.tool_name}",
        )
    if request.payload != approval.payload:
        logger.info("consume refused for %s: payload differs from the approved one", request_id)
        raise HTTPException(
            422,
            "the payload differs from the one the operator approved; an approval covers "
            "exactly what was approved",
        )

    model = AgentApprovalRequest
    spent = await db.execute(
        update(model)
        .where(
            model.id == request_id,
            model.status == ApprovalStatus.APPROVED.value,
            model.tool_name == request.tool_name,
        )
        .values(status=ApprovalStatus.CONSUMED.value, consumed_at=datetime.now(timezone.utc))
        .returning(model.id)
        .execution_options(synchronize_session=False)
    )
    if spent.scalar_one_or_none() is None:
        await db.rollback()
        await db.refresh(approval)
        logger.info("consume refused for %s: status %s", request_id, approval.status)
        raise HTTPException(409, {"status": approval.status, "reason": approval.reason})
    await db.commit()
    await db.refresh(approval)
    _emit_approval_event("resolved", {"request_id": approval.id, "status": "consumed"})
    return respond(approval)


async def _get_or_404(db: AsyncSession, request_id: str) -> AgentApprovalRequest:
    # `populate_existing`: `expire_stale` updates with `synchronize_session=False`, so a row this
    # session already holds would otherwise be returned with its pre-expiry status.
    result = await db.execute(
        select(AgentApprovalRequest)
        .where(AgentApprovalRequest.id == request_id)
        .execution_options(populate_existing=True)
    )
    approval = result.scalar_one_or_none()
    if not approval:
        raise HTTPException(404, f"Approval request {request_id} not found")
    return approval


def _summarize(tool_name: str, payload: dict[str, Any]) -> str:
    """One-line payload summary for the operator surface, per tool."""
    if tool_name == "start_probe_training_run":
        return (
            f"Train probes on model {payload.get('model_id')} from probe dataset "
            f"{payload.get('train_dataset_id')} — a GPU run (minutes to hours)"
        )
    if tool_name == "evaluate_probe_monitor":
        sets = payload.get("dataset_ids") or []
        return (
            f"Evaluate probe {payload.get('probe_id')} on {len(sets)} set(s) — a GPU job"
            # Feature 041 T20: the operator sees the sample it approves.
            + (f", a sample of {payload['sample_rows']} rows of each" if payload.get("sample_rows") else "")
        )
    if tool_name == "start_probe_judge_run":
        sets = payload.get("dataset_ids") or []
        return (
            f"Judge {len(sets)} set(s) with {payload.get('model')} on miLLM, up to "
            f"{payload.get('max_rows_per_set')} rows each (one generation per row)"
            + (f", prompt {payload.get('prompt_id')}" if payload.get("prompt_id") else "")
        )
    if tool_name == "recalibrate_probe_monitor":
        return (
            f"Re-score probe {payload.get('probe_id')}'s calibration windows on the GPU and re-cut at "
            f"{payload.get('target_fpr')}" + (" (for its calibration positions)" if payload.get("require_positions") else "")
        )
    if tool_name == "start_probe_spot_check":
        return (
            f"Spot-check {payload.get('size')} turns of {payload.get('behaviour')} with "
            f"{payload.get('model')} on miLLM (one judge call per turn and probe; live chats wait)"
        )
    if tool_name == "code_rule_approve":
        functions = [f.get("name", "?") for f in (payload.get("document") or {}).get("functions") or []]
        return (f"Run operator code {str(payload.get('code_sha256'))[:12]}… ({', '.join(functions)}) in the isolated "
                f"runner — its own cases passed; read its source before approving")
    features = payload.get("selected_features") or payload.get("feature_ids") or []
    prompt = (payload.get("prompt") or "")[:60]
    return f"{len(features)} feature(s), prompt: {prompt!r}"
