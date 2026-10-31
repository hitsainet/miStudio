"""REST surface of session spot-checks (Feature 038 FR-13, FR-17; ENH-005 D28–D35, D39).

Every rule lives in `services/probe_spot_check.py`; these routes translate `SpotCheckRefused` into
``{code, message}``. Without miLLM — not configured, unreachable, or older than its Feature 036 — the start
route answers 503 with a sentence naming miLLM, and `GET /spot-checks/availability` says why, so the UI can
hide the form with the reason (BR-011, FR-17).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ....core.deps import get_db
from ....models.probe_spot_check import ProbeSpotCheck, ProbeSpotCheckItem
from ....schemas.spot_check import SpotCheckCreate, SpotCheckPin
from ....services import millm_read
from ....services import probe_spot_check as spot
from ....services.probe_dataset_lifecycle import actor_of

router = APIRouter(prefix="/probe-monitors", tags=["probe-spot-checks"])

BASE = "/spot-checks"


def _refused(refusal: spot.SpotCheckRefused) -> HTTPException:
    return HTTPException(status_code=refusal.status, detail=refusal.as_detail())


async def _get(db: AsyncSession, check_id: str) -> ProbeSpotCheck:
    row = (await db.execute(select(ProbeSpotCheck).where(ProbeSpotCheck.id == check_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": f"spot-check {check_id} not found"})
    return row


@router.get(f"{BASE}/availability")
async def spot_check_availability() -> Dict[str, Any]:
    """Whether miLLM serves the session routes a spot-check needs, and why not."""
    return await run_in_threadpool(millm_read.spot_check_availability)


@router.post(BASE, status_code=202)
async def start_spot_check(body: SpotCheckCreate, request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    from ....workers.probe_spot_check_tasks import run_probe_spot_check

    try:
        row = await spot.start(db, body.model_dump(), actor_of(request))
    except spot.SpotCheckRefused as refusal:
        raise _refused(refusal) from refusal
    # Dispatch AFTER the commit `start` made: the task refuses without a row.
    task = run_probe_spot_check.delay(spot_check_id=row.id)
    row.celery_task_id = task.id
    await db.commit()
    return {"id": row.id, "status": row.status}


@router.get(BASE)
async def list_spot_checks(
    behaviour: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=200), db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    query = select(ProbeSpotCheck).order_by(ProbeSpotCheck.created_at.desc()).limit(limit)
    if behaviour:
        query = query.where(ProbeSpotCheck.behaviour == behaviour)
    rows = (await db.execute(query)).scalars().all()
    return {"spot_checks": [spot.check_out(r) for r in rows]}


@router.get(f"{BASE}/{{check_id}}")
async def get_spot_check(check_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """The run and its report (`metrics`), led by the statement that the judge is not ground truth."""
    return spot.check_out(await _get(db, check_id))


@router.get(f"{BASE}/{{check_id}}/items")
async def list_spot_check_items(
    check_id: str,
    disagreements_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    await _get(db, check_id)
    query = select(ProbeSpotCheckItem).where(ProbeSpotCheckItem.spot_check_id == check_id)
    if disagreements_only:
        query = query.where(
            ProbeSpotCheckItem.parse_failed.is_(False),
            ProbeSpotCheckItem.not_recorded.is_(False),
            ProbeSpotCheckItem.probe_fired.is_distinct_from(ProbeSpotCheckItem.judge_positive),
        )
    rows = (await db.execute(query.order_by(ProbeSpotCheckItem.id).limit(limit).offset(offset))).scalars().all()
    return {"items": [spot.item_out(r) for r in rows], "total": await spot.count_items(db, check_id)}


@router.post(f"{BASE}/{{check_id}}/cancel", status_code=202)
async def cancel_spot_check(check_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Stops between turns; the turns judged so far are kept, labelled partial (US-5)."""
    from ....core.cancellation import request_cancel

    row = await _get(db, check_id)
    if row.status not in spot.ACTIVE:
        raise HTTPException(status_code=409, detail={"code": "not_running", "message": f"spot-check {check_id} is already {row.status}"})
    request_cancel("probe_spot_check", check_id, celery_task_id=row.celery_task_id)
    return {"id": check_id, "status": "cancelling"}


@router.post(f"{BASE}/{{check_id}}/pin")
async def pin_spot_check(check_id: str, body: SpotCheckPin, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """D39: a pinned run keeps its excerpts past seven days; unpinning restarts the clock."""
    from datetime import timedelta

    from ....core.clock import utc_now

    row = await _get(db, check_id)
    row.pinned = body.pinned
    row.excerpts_expire_at = None if body.pinned else utc_now() + timedelta(days=spot.EXCERPT_DAYS)
    await db.commit()
    return spot.check_out(row)


@router.delete(f"{BASE}/{{check_id}}", status_code=204)
async def delete_spot_check(check_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    """Deletes the run and, by cascade, every item and its text (D35)."""
    row = await _get(db, check_id)
    if row.status in spot.ACTIVE:
        raise HTTPException(status_code=409, detail={"code": "running", "message": "cancel the spot-check before deleting it"})
    await db.delete(row)
    await db.commit()
    return Response(status_code=204)


@router.get(f"{BASE}/{{check_id}}/disagreements.jsonl")
async def export_disagreements(check_id: str, db: AsyncSession = Depends(get_db)) -> StreamingResponse:
    """The disagreeing turns as `mistudio.spot-check-disagreement/v1` lines. Contains recorded user text."""
    row = await _get(db, check_id)
    items = (
        await db.execute(
            select(ProbeSpotCheckItem).where(ProbeSpotCheckItem.spot_check_id == check_id).order_by(ProbeSpotCheckItem.id)
        )
    ).scalars().all()
    lines = list(spot.export_jsonl(row, items))
    return StreamingResponse(
        iter(lines),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{check_id}-disagreements.jsonl"'},
    )
