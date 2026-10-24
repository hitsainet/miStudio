"""What is running and which GPUs it is using, in two reads (MCP gap G9, 2026-10-21).

* `GET /operations/active-jobs` — every miStudio job that is queued or running, of every kind
  (`services/active_jobs.py` holds the one list of kinds).
* `GET /operations/gpu-status` — every card NVML sees, by UUID: memory, utilisation, compute
  processes, the miStudio lease and the jobs that recorded the card.

Read-only. The MCP tools `list_active_jobs`, `get_gpu_status` and `gpu_preflight` read these; the
preflight adds miLLM and miForge from the MCP side, because the backend knows neither URL.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.deps import get_db
from src.services import active_jobs as active_jobs_service
from src.services import gpu_status as gpu_status_service

router = APIRouter(prefix="/operations", tags=["operations"])


@router.get("/active-jobs")
async def get_active_jobs(
    kind: Optional[str] = Query(None, description="Only this job kind (see `kinds` in any answer)"),
    gpu: Optional[str] = Query(None, description="Only jobs that recorded this GPU UUID"),
    db: AsyncSession = Depends(get_db),
):
    """Every queued or running miStudio job, newest first. An unknown kind is refused by name."""
    try:
        body = await active_jobs_service.list_active_jobs(db, kind=kind)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if gpu:
        body["jobs"] = active_jobs_service.jobs_on_card(body["jobs"], gpu)
        body["count"] = len(body["jobs"])
        body["gpu"] = gpu
    return body


@router.get("/gpu-status")
async def get_gpu_status(
    gpu: Optional[str] = Query(
        None, description="One card's UUID (GPU-…). Omit for every card. Indices are refused: "
                          "they follow PCI bus order and have moved before"),
    db: AsyncSession = Depends(get_db),
):
    """Live per-card state for every GPU miStudio can see, or one card by UUID."""
    from src.services.gpu_placement import normalise_uuid

    if gpu is not None and not gpu.strip().lower().startswith("gpu-"):
        raise HTTPException(
            422, f"{gpu!r} is not a GPU UUID. Pin cards by UUID (GPU-…), from list_gpus or this "
                 "route's answer; an index can name a different card after a reboot.")
    # nvidia-smi / NVML can take seconds: off the event loop.
    cards, reason = await asyncio.to_thread(gpu_status_service.read_cards)
    leases = await gpu_status_service.live_leases(db)
    jobs = await active_jobs_service.list_active_jobs(db)
    assembled = gpu_status_service.assemble(cards, leases, jobs["jobs"])
    if gpu is not None:
        wanted = normalise_uuid(gpu)
        assembled = [c for c in assembled if normalise_uuid(c["uuid"]) == wanted]
        if not assembled and reason is None:
            raise HTTPException(404, {
                "detail": f"No GPU {gpu} is visible to miStudio",
                "visible": [c["uuid"] for c in cards],
            })
    return {
        "cards": assembled,
        "card_count": len(assembled),
        "reason": reason,
        "job_read_errors": jobs["errors"],
        "read_at": jobs["read_at"],
        "attribution": gpu_status_service.PID_NOTE,
    }
