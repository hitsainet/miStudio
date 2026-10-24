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
executor's: no MCP code path can approve.

Caller-executed approvals EXPIRE: pending, or approved and unconsumed, for longer than
`mcp_approval_ttl_hours` (default 24). Expiry is lazy — every route runs `expire_stale` first —
so a stale request never reads `pending`. Steering approvals never expire (S3-03).
"""

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, field_validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.runtime_settings import effective
from src.core.deps import get_db
from src.models.agent_approval import AgentApprovalRequest, ApprovalStatus
from src.workers.websocket_emitter import emit_progress

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp/approvals", tags=["mcp-approvals"])

class Execution(str, Enum):
    """Who carries out an approved request."""

    #: The backend, on approve, by an internal self-call (steering).
    BACKEND = "backend"
    #: The MCP tool that asked, on re-call, after consuming the approval (miLLM model load).
    CALLER = "caller"


#: Steering tools. ONE set: `APPROVABLE_TOOLS` is built from it and `expire_stale` excludes it, so
#: the "steering never expires" rule cannot drift from the list of steering tools.
STEERING_TOOLS = frozenset({"steer_compare", "steer_sweep", "steer_combined"})

#: tool name → (who executes it, backend self-call path or None).
APPROVABLE_TOOLS: dict[str, tuple[Execution, Optional[str]]] = {
    **{tool: (Execution.BACKEND, f"/api/v1/steering/async/{tool.removeprefix('steer_')}")
       for tool in sorted(STEERING_TOOLS)},
    "millm_load_model": (Execution.CALLER, None),
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

    model_config = {"from_attributes": True}


class ApprovalListResponse(BaseModel):
    approvals: list[ApprovalResponse]
    total: int


class DenyRequest(BaseModel):
    reason: Optional[str] = None


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
        )
        .returning(model.id)
        .execution_options(synchronize_session=False)
    )
    unconsumed = (
        update(model)
        .where(
            model.status == ApprovalStatus.APPROVED.value,
            model.resolved_at < cutoff,
            model.tool_name.not_in(sorted(STEERING_TOOLS)),
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
    db: AsyncSession = Depends(get_db),
):
    """Create a pending approval request (called by the MCP server)."""
    approval = AgentApprovalRequest(
        tool_name=request.tool_name,
        payload=request.payload,
        status=ApprovalStatus.PENDING.value,
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
    return approval


@router.get("", response_model=ApprovalListResponse)
async def list_approval_requests(
    status: Optional[str] = Query(
        None, description="Filter: pending|approved|denied|expired|consumed"
    ),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    await expire_stale(db)
    query = select(AgentApprovalRequest).order_by(AgentApprovalRequest.created_at.desc())
    if status:
        query = query.where(AgentApprovalRequest.status == status)
    result = await db.execute(query.limit(limit).execution_options(populate_existing=True))
    approvals = result.scalars().all()
    return ApprovalListResponse(
        approvals=[ApprovalResponse.model_validate(a) for a in approvals],
        total=len(approvals),
    )


@router.get("/{request_id}", response_model=ApprovalResponse)
async def get_approval_request(request_id: str, db: AsyncSession = Depends(get_db)):
    await expire_stale(db)
    approval = await _get_or_404(db, request_id)
    return approval


@router.post("/{request_id}/approve", response_model=ApprovalResponse)
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

    kind, endpoint = APPROVABLE_TOOLS.get(approval.tool_name, (None, None))
    if kind is Execution.CALLER:
        approval.status = ApprovalStatus.APPROVED.value
        approval.resolved_at = datetime.now(timezone.utc)
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
    await db.commit()
    await db.refresh(approval)

    _emit_approval_event(
        "resolved",
        {"request_id": approval.id, "status": "approved", "steering_task_id": task_id},
    )
    logger.info(f"Approval {request_id} approved → steering task {task_id}")
    return approval


@router.post("/{request_id}/deny", response_model=ApprovalResponse)
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
    await db.commit()
    await db.refresh(approval)

    _emit_approval_event("resolved", {"request_id": approval.id, "status": "denied"})
    return approval


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
    return approval


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
    if tool_name == "millm_load_model":
        return (
            f"Load miLLM model #{payload.get('model_id')} — evicts the resident model"
        )
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
    if tool_name == "start_probe_spot_check":
        return (
            f"Spot-check {payload.get('size')} turns of {payload.get('behaviour')} with "
            f"{payload.get('model')} on miLLM (one judge call per turn and probe; live chats wait)"
        )
    features = payload.get("selected_features") or payload.get("feature_ids") or []
    prompt = (payload.get("prompt") or "")[:60]
    return f"{len(features)} feature(s), prompt: {prompt!r}"
