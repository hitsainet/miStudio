"""
Agent approval endpoints (Feature 010, operator-approval mode).

When the MCP server runs with MCP_STEERING_APPROVAL=true, agent steering
calls land here as pending requests instead of hitting /steering/async/*
directly. An operator approves or denies them (Steering panel surface); on
approval the backend submits the stored payload to the corresponding steering
endpoint itself via an internal self-call — reusing all of that endpoint's
validation, rate limiting, and worker autostart — and records the resulting
steering task id for the agent to poll.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.deps import get_db
from src.models.agent_approval import AgentApprovalRequest, ApprovalStatus
from src.workers.websocket_emitter import emit_progress

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp/approvals", tags=["mcp-approvals"])

# tool name → steering async endpoint path
STEERING_ENDPOINTS = {
    "steer_compare": "/api/v1/steering/async/compare",
    "steer_sweep": "/api/v1/steering/async/sweep",
    "steer_combined": "/api/v1/steering/async/combined",
}


class ApprovalCreateRequest(BaseModel):
    tool_name: str = Field(..., pattern="^(steer_compare|steer_sweep|steer_combined)$")
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

    model_config = {"from_attributes": True}


class ApprovalListResponse(BaseModel):
    approvals: list[ApprovalResponse]
    total: int


class DenyRequest(BaseModel):
    reason: Optional[str] = None


def _emit_approval_event(event: str, data: dict[str, Any]) -> None:
    emit_progress(channel="mcp/approvals", event=f"approval:{event}", data=data)


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
            "summary": _summarize(request.payload),
        },
    )
    return approval


@router.get("", response_model=ApprovalListResponse)
async def list_approval_requests(
    status: Optional[str] = Query(None, description="Filter: pending|approved|denied|expired"),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    query = select(AgentApprovalRequest).order_by(AgentApprovalRequest.created_at.desc())
    if status:
        query = query.where(AgentApprovalRequest.status == status)
    result = await db.execute(query.limit(limit))
    approvals = result.scalars().all()
    return ApprovalListResponse(
        approvals=[ApprovalResponse.model_validate(a) for a in approvals],
        total=len(approvals),
    )


@router.get("/{request_id}", response_model=ApprovalResponse)
async def get_approval_request(request_id: str, db: AsyncSession = Depends(get_db)):
    approval = await _get_or_404(db, request_id)
    return approval


@router.post("/{request_id}/approve", response_model=ApprovalResponse)
async def approve_request(request_id: str, db: AsyncSession = Depends(get_db)):
    """Approve: backend submits the stored steering payload itself."""
    approval = await _get_or_404(db, request_id)
    if approval.status != ApprovalStatus.PENDING.value:
        raise HTTPException(409, f"Request is {approval.status}, not pending")

    endpoint = STEERING_ENDPOINTS.get(approval.tool_name)
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


async def _get_or_404(db: AsyncSession, request_id: str) -> AgentApprovalRequest:
    result = await db.execute(
        select(AgentApprovalRequest).where(AgentApprovalRequest.id == request_id)
    )
    approval = result.scalar_one_or_none()
    if not approval:
        raise HTTPException(404, f"Approval request {request_id} not found")
    return approval


def _summarize(payload: dict[str, Any]) -> str:
    """One-line payload summary for the operator surface."""
    features = payload.get("selected_features") or payload.get("feature_ids") or []
    prompt = (payload.get("prompt") or "")[:60]
    return f"{len(features)} feature(s), prompt: {prompt!r}"
