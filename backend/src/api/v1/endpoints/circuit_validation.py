"""
Circuit validation + faithfulness + manifest endpoints (Feature 017).

Validation runs on a discovery run's own `validation_*` lifecycle (a failed
pass never corrupts the completed discovery). GPU-guarded (shares the single-
GPU circuit guard). Manifests are self-contained + reproducible.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....models.circuit_runs import CircuitDiscoveryRun
from ....models.validation_manifest import ValidationManifest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["circuit-validation"])


class ValidateBody(BaseModel):
    ordering: str = Field("coact", pattern="^(coact|attr)$")
    k: int = Field(20, ge=1, le=200)
    prompts_per_edge: int = Field(8, ge=1, le=64)
    null_samples: int = Field(20, ge=1, le=200)
    percentile: float = Field(95.0, ge=50.0, le=100.0)
    sign_frac: float = Field(0.8, ge=0.0, le=1.0)
    baseline: str = Field("zero", pattern="^(zero|corpus_mean)$")
    seed: int = 0


async def _run_or_404(db, run_id) -> CircuitDiscoveryRun:
    run = (await db.execute(select(CircuitDiscoveryRun).where(
        CircuitDiscoveryRun.id == run_id))).scalar_one_or_none()
    if run is None:
        raise HTTPException(404, f"Discovery run {run_id} not found")
    return run


@router.post("/circuit-discovery/{run_id}/validate", status_code=202)
async def start_validation(run_id: str, body: ValidateBody,
                           db: AsyncSession = Depends(get_db)):
    from ....services.circuit_capture_service import (
        CaptureConflictError, CircuitCaptureService)
    from ....services.circuit_intervention_service import (
        CircuitInterventionService, InterventionConfigError)
    from ....workers.circuit_validation_tasks import validate_circuit_edges

    run = await _run_or_404(db, run_id)
    if run.status != "completed":
        raise HTTPException(409, f"Discovery run is {run.status} — "
                                 f"validation needs a completed run")
    if not run.candidates:
        raise HTTPException(409, "Run has no candidates to validate")
    if run.validation_status in ("pending", "running"):
        raise HTTPException(409, "A validation pass is already in flight")
    # attr ordering is meaningless without a completed attribution pass — else
    # we'd validate the coact order under the "attr" label and the uplift story
    # becomes garbage (R1 Q4).
    if body.ordering == "attr" and run.attribution_status != "completed":
        raise HTTPException(
            409, "ordering='attr' needs a completed attribution pass first — "
                 "run attribution, or validate with ordering='coact'")
    try:
        scope = CircuitInterventionService.create_scope(body.model_dump())
    except InterventionConfigError as e:
        raise HTTPException(422, str(e))

    # Validation loads a model — respect the single-GPU guard; guard + mark in
    # one advisory-locked sync transaction (like attribution).
    from ....api.v1.endpoints.circuit_discovery import _run_sync

    def _guard_and_mark(sync_db):
        CircuitCaptureService.assert_no_active_gpu_run(sync_db)
        row = sync_db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.id == run_id).first()
        row.validation_status = "pending"
        row.validation_progress = 0.0
        row.validation_error = None
        sync_db.commit()

    try:
        await _run_sync(db, _guard_and_mark)
    except CaptureConflictError as e:
        raise HTTPException(409, str(e))
    task = validate_circuit_edges.delay(run.id, scope)
    await db.execute(
        CircuitDiscoveryRun.__table__.update()
        .where(CircuitDiscoveryRun.id == run.id)
        .values(validation_task_id=task.id))
    await db.commit()
    return {"id": run.id, "task_id": task.id, "status": "queued"}


@router.post("/circuit-discovery/{run_id}/validate/cancel")
async def cancel_validation(run_id: str, db: AsyncSession = Depends(get_db)):
    from ....core.celery_app import revoke_task

    run = await _run_or_404(db, run_id)
    if run.validation_status not in ("pending", "running"):
        raise HTTPException(
            409, f"Validation is {run.validation_status} — nothing to cancel")
    run.validation_status = "cancelled"
    await db.commit()
    if run.validation_task_id:
        revoke_task(run.validation_task_id)
    return {"id": run.id, "validation_status": "cancelled"}


# ── manifests ────────────────────────────────────────────────────────────

def _manifest_out(m: ValidationManifest) -> Dict[str, Any]:
    return {"id": m.id, "kind": m.kind, "discovery_run_id": m.discovery_run_id,
            "circuit_id": m.circuit_id,
            "parent_manifest_id": m.parent_manifest_id,
            "payload": m.payload, "created_at": m.created_at}


@router.get("/validation-manifests/{manifest_id}")
async def get_manifest(manifest_id: str, db: AsyncSession = Depends(get_db)):
    from ....services.manifest_service import ManifestService

    m = await ManifestService.get(db, manifest_id)
    if m is None:
        raise HTTPException(404, f"Manifest {manifest_id} not found")
    return _manifest_out(m)


@router.get("/validation-manifests")
async def list_manifests(discovery_run_id: Optional[str] = Query(None),
                         circuit_id: Optional[str] = Query(None),
                         db: AsyncSession = Depends(get_db)):
    from ....services.manifest_service import ManifestService

    rows = await ManifestService.list_by_parent(
        db, discovery_run_id=discovery_run_id, circuit_id=circuit_id)
    return {"manifests": [_manifest_out(m) for m in rows]}


@router.post("/validation-manifests/{manifest_id}/reproduce", status_code=202)
async def reproduce_manifest(manifest_id: str,
                             db: AsyncSession = Depends(get_db)):
    """Re-execute an edge_batch manifest from its payload and store a
    `reproduction` manifest with per-edge deltas + a tolerance verdict — the
    test that a manifest is truly self-contained."""
    from ....services.manifest_service import ManifestService
    from ....workers.circuit_validation_tasks import validate_circuit_edges

    m = await ManifestService.get(db, manifest_id)
    if m is None:
        raise HTTPException(404, f"Manifest {manifest_id} not found")
    if m.kind != "edge_batch":
        raise HTTPException(409, "Only edge_batch manifests are reproducible")
    if not m.discovery_run_id:
        raise HTTPException(409, "Manifest has no discovery run to reproduce on")
    run = await _run_or_404(db, m.discovery_run_id)
    if run.validation_status in ("pending", "running"):
        raise HTTPException(409, "A validation pass is already in flight")
    # Reproduce is a full GPU pass — respect the single-GPU guard (R1 #6/Q6:
    # it previously dispatched without the guard) and guard+mark atomically.
    from ....api.v1.endpoints.circuit_discovery import _run_sync
    from ....services.circuit_capture_service import (
        CaptureConflictError, CircuitCaptureService)

    def _guard_and_mark(sync_db):
        CircuitCaptureService.assert_no_active_gpu_run(sync_db)
        row = sync_db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.id == run.id).first()
        row.validation_status = "pending"
        row.validation_progress = 0.0
        sync_db.commit()

    try:
        await _run_sync(db, _guard_and_mark)
    except CaptureConflictError as e:
        raise HTTPException(409, str(e))
    # Re-run with the SAME scope from the payload; the worker writes a fresh
    # `reproduction` manifest and the verdict is computed against the original.
    # The run's report/candidates are NOT stomped (the worker gates those
    # behind `not reproduce_of` — R1 A4).
    scope = dict(m.payload.get("config") or {})
    scope["reproduce_of"] = manifest_id
    task = validate_circuit_edges.delay(run.id, scope)
    return {"reproduce_of": manifest_id, "task_id": task.id, "status": "queued"}
