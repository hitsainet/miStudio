"""Blend reports (Feature 041 S2; FR-6, FR-16 – FR-19; FTDD §5.3).

A report is addressed by its own id (`pbr_…`) across behaviours (T3). Every refusal is
``{code, message, …reasons, next}`` (`BlendRefused`); 5xx bodies carry no internals.

⚠ THE ROW IS COMMITTED BEFORE DISPATCH (`service.start` commits, then this route queues the task), and the
task refuses without a row — so a report can never be queued that nothing can read.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from ....core.deps import get_db
from ....schemas.probe_blend import BlendStart, PruneRows
from ....services import probe_dataset_lifecycle as lifecycle
from ....services.detector_next import DetectorRefused

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/probe-monitors", tags=["probe-blends"])

BASE = "/blend-reports"


def _dispatch(row: Any) -> str:
    from ....workers.probe_blend_tasks import run_blend_evaluation

    return run_blend_evaluation.delay(row.id).id


async def _queue(db: AsyncSession, row: Any) -> Dict[str, Any]:
    try:
        task_id = _dispatch(row)
    except Exception:  # noqa: BLE001 - a broker outage must not leave a queued ghost holding a slot
        logger.exception("could not dispatch blend report %s", row.id)
        row.status, row.failure_code = "failed", "engine_error"
        row.failure_message = "the blend task could not be queued (the task broker is unavailable)"
        await db.commit()
        raise HTTPException(status_code=503, detail={"code": "broker_unavailable",
                                                     "message": "the blend task could not be queued"})
    row.celery_task_id = task_id
    await db.commit()
    return {"report_id": row.id, "status": row.status, "resolved": {
        k: row.resolved.get(k) for k in ("views", "folds", "group_column", "target_fpr", "seed", "flags")},
        "members": [m["probe_id"] for m in row.resolved["members"]], "candidates": row.resolved["candidates"],
        "task_id": task_id}


@router.post(BASE, status_code=202)
async def evaluate_probe_blends(body: BlendStart, request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """FR-6: resolve the members and views, refuse at the call, admit under the concurrency cap, queue."""
    from ....services import behaviour_catalogue as catalogue
    from ....services.probe_blend import service

    try:
        row = await service.start(db, body.model_dump(exclude_none=True), lifecycle.actor_of(request))
    except catalogue.BehaviourRefused as refusal:
        raise HTTPException(status_code=refusal.status, detail=refusal.as_detail()) from refusal
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    return await _queue(db, row)


@router.get(BASE)
async def list_blend_reports(
    behaviour_key: str = Query(...),
    model: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1, description="Reports per page (default and most: blend_report_page_size)"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """FR-16: the behaviour's reports, newest first, with a stable tie (id)."""
    from ....core.runtime_settings import effective
    from ....models.probe_blend import ProbeBlendReport
    from ....services.probe_blend import service

    cap = int(effective().blend_report_page_size)
    size = cap if limit is None else limit
    if size > cap:
        raise HTTPException(422, detail={"code": "cap_exceeded", "message": f"a page of {size} is more than "
                                         f"blend_report_page_size ({cap})", "setting": "blend_report_page_size",
                                         "value": cap, "got": size, "next": []})
    query = select(ProbeBlendReport).where(ProbeBlendReport.behaviour_key == behaviour_key)
    if model:
        query = query.where(ProbeBlendReport.model_id == model)
    rows = (await db.execute(query.order_by(ProbeBlendReport.created_at.desc(), ProbeBlendReport.id.desc())
                             .offset(offset).limit(size + 1))).scalars().all()
    page = rows[:size]
    return {"behaviour_key": behaviour_key, "model": model,
            "reports": [await service.envelope(db, row, include_report=False) for row in page],
            "offset": offset, "limit": size, "next_offset": offset + size if len(rows) > size else None}


@router.get(f"{BASE}/{{report_id}}")
async def get_blend_report(report_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """The envelope: status, failure, `stale` (computed on read), ranking, and the parsed report when done."""
    from ....services.probe_blend import service

    try:
        return await service.envelope(db, await service.get_row(db, report_id))
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal


@router.get(f"{BASE}/{{report_id}}/report.json")
async def get_blend_report_bytes(report_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    """The STORED canonical bytes (never re-serialised; the byte-equality tests read this)."""
    from ....services.probe_blend import service

    try:
        body = service.report_bytes(await service.get_row(db, report_id))
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    return Response(content=body, media_type="application/json")


@router.get(f"{BASE}/{{report_id}}/scores")
async def get_blend_report_scores(
    report_id: str,
    candidate: str = Query(...),
    view_id: str = Query(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(10_000, ge=1, description="Rows per page (most: the runtime setting blend_scores_page_limit)"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """FR-16: one candidate's per-row values on one view, paged in position order; 409 `rows_pruned`."""
    import json

    from ....core.runtime_settings import effective
    from ....services.probe_blend import service

    cap = int(effective().blend_scores_page_limit)
    if limit > cap:
        raise HTTPException(422, detail={"code": "cap_exceeded", "message": f"a page of {limit} rows is more than "
                                         f"blend_scores_page_limit ({cap})", "setting": "blend_scores_page_limit",
                                         "value": cap, "got": limit, "next": []})
    try:
        row = await service.get_row(db, report_id)
        report = json.loads(service.report_bytes(row))
        entry = service.candidate_of(report, candidate)
        service.refuse_view_not_in_report(report, view_id)
        loaded = await run_in_threadpool(service.candidate_rows, row, entry, view_id)
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    total = int(loaded["positions"].shape[0])
    end = min(offset + limit, total)
    values = np.asarray(loaded["values"][offset:end], dtype=np.float64)
    if entry["produces"] == "verdict":
        verdicts = values >= float(entry["params"]["k"])
    else:
        threshold = entry["calibration"]["threshold"]
        verdicts = values >= threshold if threshold is not None else np.zeros(values.shape[0], dtype=bool)
    rows = [{"position": int(p), "value": float(v), "verdict": bool(f), "label": int(lab)}
            for p, v, f, lab in zip(loaded["positions"][offset:end].tolist(), values.tolist(), verdicts.tolist(),
                                    loaded["labels"][offset:end].tolist())]
    return {"report_id": report_id, "candidate": candidate, "view_id": view_id, "total": total, "offset": offset,
            "next_offset": end if end < total else None, "rows": rows}


@router.post(f"{BASE}/{{report_id}}/rerun", status_code=202)
async def rerun_blend_report(report_id: str, request: Request, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """FR-17 / P4: the stored request against today's members and version; linked both ways."""
    from ....services import behaviour_catalogue as catalogue
    from ....services.probe_blend import service

    try:
        row = await service.rerun(db, report_id, lifecycle.actor_of(request))
    except catalogue.BehaviourRefused as refusal:
        raise HTTPException(status_code=refusal.status, detail=refusal.as_detail()) from refusal
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    return await _queue(db, row)


@router.post(f"{BASE}/{{report_id}}/cancel", status_code=202)
async def cancel_blend_evaluation(report_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """OQ-5 (a) / T32: a running report stops between folds or candidates and keeps its inputs; a queued
    one is cancelled at once."""
    from ....core.cancellation import request_cancel
    from ....core.clock import utc_now
    from ....services.probe_blend import service

    try:
        row = await service.get_row(db, report_id)
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    if row.status in ("done", "failed", "cancelled"):
        return {"report_id": report_id, "status": row.status, "detail": f"already {row.status}"}
    task_id = row.celery_task_id
    outcome = await run_in_threadpool(lambda: request_cancel("blend_report", report_id, celery_task_id=task_id))
    await db.refresh(row)
    if row.status == "queued":
        row.status, row.completed_at = "cancelled", utc_now()
        await db.commit()
    return {"report_id": report_id, "status": row.status, "detail": outcome.detail}


@router.post(f"{BASE}/{{report_id}}/prune-rows", dependencies=[Depends(lifecycle.refuse_agents)])
async def prune_blend_report_rows(report_id: str, request: Request, body: Optional[PruneRows] = None,
                                  db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """FR-19: the operator's alone. Deletes the per-row values, keeps `report.json` and the summary."""
    from ....services.probe_blend import service

    try:
        row = await service.prune_rows(db, report_id, lifecycle.actor_of(request))
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal
    return {"report_id": row.id, "rows_pruned_at": row.rows_pruned_at.isoformat() if row.rows_pruned_at else None,
            "rows_pruned_by": row.rows_pruned_by, "reason": (body.reason if body else None)}
