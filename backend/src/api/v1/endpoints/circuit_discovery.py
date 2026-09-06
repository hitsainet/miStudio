"""
Circuit capture + discovery + attribution endpoints (Feature 016).

Capture: POST estimate/launch (202) → WS circuit-capture/{id} → list/get/
cancel/delete. Discovery: POST run (202) → WS circuit-discovery/{id} →
get run incl. the first-class report. Attribution: POST sub-route (202).
House conventions: 202 + task id, 409 on concurrent, DB-status cancel.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun

logger = logging.getLogger(__name__)

router = APIRouter(tags=["circuit-discovery"])


# ── schemas ──────────────────────────────────────────────────────────────

class CaptureLayerEntry(BaseModel):
    layer: int = Field(..., ge=0)
    sae_id: str = Field(..., max_length=255)


class AttentionCaptureConfig(BaseModel):
    layers: List[int] = Field(default_factory=list)
    heads: Optional[List[int]] = None  # None = all heads
    top_k: int = Field(4, ge=1, le=32)


class CaptureCreate(BaseModel):
    dataset_id: str = Field(..., max_length=255)
    model_id: Optional[str] = Field(None, max_length=255)
    layers: List[CaptureLayerEntry] = Field(..., min_length=1, max_length=8)
    epsilon: float = Field(0.1, ge=0.0, lt=1.0)
    theta_floor: float = Field(0.01, ge=0.0)
    sample_cap: int = Field(2000, ge=32, le=100_000)
    split_seed: int = 42
    attention_capture: Optional[AttentionCaptureConfig] = None
    confirm: bool = Field(
        False, description="false → probe + estimate only (status 'estimated'); "
                           "true → full capture")


class DiscoverySeedRef(BaseModel):
    """A seed reference: exactly one of feature_idx / cluster_profile_id.

    Typed so a malformed ref (feature_idx null/NaN) is rejected with a 422 at
    submit, not an int(None) TypeError deep in the worker (R1 CR#3)."""
    layer: int = Field(..., ge=0)
    feature_idx: Optional[int] = Field(None, ge=0)
    cluster_profile_id: Optional[str] = Field(None, max_length=64)

    @model_validator(mode="after")
    def _exactly_one(self) -> "DiscoverySeedRef":
        if (self.feature_idx is None) == (self.cluster_profile_id is None):
            raise ValueError(
                "seed ref needs exactly one of feature_idx / cluster_profile_id")
        return self


class DiscoveryCreate(BaseModel):
    capture_run_id: str = Field(..., max_length=36)
    granularity: str = Field("feature", pattern="^(feature|cluster)$")
    mode: str = Field("open", pattern="^(seeded|open)$")
    seed_refs: Optional[List[DiscoverySeedRef]] = Field(
        None, max_length=200,
        description="[{layer, feature_idx}|{layer, cluster_profile_id}]")
    s_min: int = Field(20, ge=1)
    null_shuffles: int = Field(100, ge=10, le=1000)
    null_percentile: float = Field(99.0, ge=50.0, le=100.0)
    fdr_q: float = Field(0.05, gt=0.0, le=0.5)
    cohesion_floor: float = Field(0.3, ge=0.0, le=1.0)
    seed: int = 0
    force: bool = Field(False, description="mine a STALE store anyway")


class AttributionCreate(BaseModel):
    prompt_limit: Optional[int] = Field(None, ge=1, le=256)


# ── serializers ──────────────────────────────────────────────────────────

def _capture_out(run: CircuitCaptureRun) -> Dict[str, Any]:
    m = run.manifest or {}
    return {
        "id": run.id, "status": run.status, "progress": run.progress,
        "error_message": run.error_message,
        "corpus": m.get("corpus"), "model_id": m.get("model_id"),
        "layers": m.get("layers"),
        "split": {k: v for k, v in (m.get("split") or {}).items()
                  if k != "heldout_docs"} | {
                      "heldout_count": len((m.get("split") or {})
                                           .get("heldout_docs", []))},
        "estimate": m.get("estimate"),
        "attention_capture": m.get("attention_capture"),
        "counts": m.get("counts"), "bytes": run.bytes_total,
        "events_total": run.events_total, "stale": run.stale,
        "created_at": run.created_at, "updated_at": run.updated_at,
    }


def _discovery_out(run: CircuitDiscoveryRun, *, include_candidates: bool) -> Dict[str, Any]:
    out = {
        "id": run.id, "capture_run_id": run.capture_run_id,
        "status": run.status, "progress": run.progress,
        "error_message": run.error_message, "params": run.params,
        "report": run.report,
        "candidate_count": len(run.candidates or []),
        # Attribution's own lifecycle (R1 QA-P2) — the discovery status above
        # stays 'completed' regardless of an attribution pass's outcome.
        "attribution_status": run.attribution_status,
        "attribution_progress": run.attribution_progress,
        "attribution_error": run.attribution_error,
        # Validation's own lifecycle (017) — discovery status stays 'completed'.
        "validation_status": run.validation_status,
        "validation_progress": run.validation_progress,
        "validation_error": run.validation_error,
        "created_at": run.created_at, "updated_at": run.updated_at,
    }
    if include_candidates:
        out["candidates"] = run.candidates or []
    return out


async def _capture_or_404(db, run_id) -> CircuitCaptureRun:
    run = (await db.execute(select(CircuitCaptureRun).where(
        CircuitCaptureRun.id == run_id))).scalar_one_or_none()
    if run is None:
        raise HTTPException(404, f"Capture run {run_id} not found")
    return run


async def _discovery_or_404(db, run_id) -> CircuitDiscoveryRun:
    run = (await db.execute(select(CircuitDiscoveryRun).where(
        CircuitDiscoveryRun.id == run_id))).scalar_one_or_none()
    if run is None:
        raise HTTPException(404, f"Discovery run {run_id} not found")
    return run


# ── capture ──────────────────────────────────────────────────────────────

@router.post("/circuit-capture", status_code=202)
async def create_capture(body: CaptureCreate, db: AsyncSession = Depends(get_db)):
    """Create a capture run. confirm=false runs the probe and stops at the
    cost estimate; POST /{id}/confirm launches the full capture."""
    from ....services.circuit_capture_service import (
        CaptureConfigError, CaptureConflictError, CircuitCaptureService)
    from ....workers.circuit_capture_tasks import capture_circuit_activations

    def _create(sync_db):
        # 409-on-concurrent in the SAME transaction as the insert, advisory-
        # locked so two simultaneous requests can't both pass (R1 QA-P1 / R2
        # B2). The estimate path runs a GPU PROBE too, so guard it as well
        # (R2 B5) — not just confirm.
        CircuitCaptureService.assert_no_active_gpu_run(sync_db)
        return CircuitCaptureService.create_run(sync_db, body.model_dump())

    try:
        run = await _run_sync(db, _create)
    except CaptureConfigError as e:
        raise HTTPException(422, str(e))
    except CaptureConflictError as e:
        raise HTTPException(409, str(e))
    task = capture_circuit_activations.delay(run.id, confirmed=body.confirm)
    await db.execute(
        CircuitCaptureRun.__table__.update()
        .where(CircuitCaptureRun.id == run.id)
        .values(celery_task_id=task.id))
    await db.commit()
    return {"id": run.id, "task_id": task.id, "status": "queued",
            "confirmed": body.confirm}


@router.post("/circuit-capture/{run_id}/confirm", status_code=202)
async def confirm_capture(run_id: str, db: AsyncSession = Depends(get_db)):
    """Launch the full capture for an 'estimated' run."""
    from ....services.circuit_capture_service import (
        CaptureConflictError, CircuitCaptureService)
    from ....workers.circuit_capture_tasks import capture_circuit_activations

    run = await _capture_or_404(db, run_id)
    if run.status not in ("estimated", "failed"):
        raise HTTPException(409, f"Run is {run.status} — confirm applies to "
                                 f"'estimated' (or retryable 'failed') runs")

    def _guard(sync_db):
        CircuitCaptureService.assert_no_active_gpu_run(sync_db)

    try:
        await _run_sync(db, _guard)
    except CaptureConflictError as e:
        raise HTTPException(409, str(e))
    task = capture_circuit_activations.delay(run.id, confirmed=True)
    run.celery_task_id = task.id
    run.status = "pending"
    await db.commit()
    return {"id": run.id, "task_id": task.id, "status": "queued"}


@router.get("/circuit-capture")
async def list_captures(limit: int = Query(50, ge=1, le=200),
                        offset: int = Query(0, ge=0),
                        db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(CircuitCaptureRun)
        .order_by(CircuitCaptureRun.created_at.desc())
        .limit(limit).offset(offset))).scalars().all()
    return {"captures": [_capture_out(r) for r in rows],
            "limit": limit, "offset": offset}


@router.get("/circuit-capture/{run_id}")
async def get_capture(run_id: str, db: AsyncSession = Depends(get_db)):
    return _capture_out(await _capture_or_404(db, run_id))


@router.post("/circuit-capture/{run_id}/cancel")
async def cancel_capture(run_id: str, db: AsyncSession = Depends(get_db)):
    from ....core.celery_app import revoke_task

    run = await _capture_or_404(db, run_id)
    if run.status not in ("pending", "running", "estimating"):
        raise HTTPException(409, f"Run is {run.status} — nothing to cancel")
    run.status = "cancelled"
    await db.commit()
    if run.celery_task_id:
        revoke_task(run.celery_task_id)
    return {"id": run.id, "status": "cancelled"}


@router.delete("/circuit-capture/{run_id}")
async def delete_capture(run_id: str, db: AsyncSession = Depends(get_db)):
    from ....services.circuit_capture_service import (
        CaptureConfigError, CircuitCaptureService)

    run = await _capture_or_404(db, run_id)

    def _delete(sync_db):
        row = sync_db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.id == run_id).first()
        CircuitCaptureService.delete_run(sync_db, row)

    try:
        await _run_sync(db, _delete)
    except CaptureConfigError as e:
        raise HTTPException(409, str(e))
    return {"deleted": run_id}


# ── discovery ────────────────────────────────────────────────────────────

@router.post("/circuit-discovery", status_code=202)
async def create_discovery(body: DiscoveryCreate,
                           db: AsyncSession = Depends(get_db)):
    from ....services.circuit_discovery_service import (
        CircuitDiscoveryService, DiscoveryConfigError, DiscoveryConflictError)
    from ....workers.circuit_capture_tasks import run_circuit_discovery

    def _create(sync_db):
        return CircuitDiscoveryService.create_run(sync_db, body.model_dump())

    try:
        run = await _run_sync(db, _create)
    except DiscoveryConfigError as e:
        raise HTTPException(422, str(e))
    except DiscoveryConflictError as e:
        raise HTTPException(409, str(e))
    task = run_circuit_discovery.delay(run.id)
    await db.execute(
        CircuitDiscoveryRun.__table__.update()
        .where(CircuitDiscoveryRun.id == run.id)
        .values(celery_task_id=task.id))
    await db.commit()
    return {"id": run.id, "task_id": task.id, "status": "queued"}


@router.get("/circuit-discovery")
async def list_discoveries(capture_run_id: Optional[str] = Query(None),
                           limit: int = Query(50, ge=1, le=200),
                           offset: int = Query(0, ge=0),
                           db: AsyncSession = Depends(get_db)):
    q = select(CircuitDiscoveryRun).order_by(
        CircuitDiscoveryRun.created_at.desc())
    if capture_run_id:
        q = q.where(CircuitDiscoveryRun.capture_run_id == capture_run_id)
    rows = (await db.execute(q.limit(limit).offset(offset))).scalars().all()
    return {"discoveries": [_discovery_out(r, include_candidates=False)
                            for r in rows],
            "limit": limit, "offset": offset}


@router.get("/circuit-discovery/{run_id}")
async def get_discovery(run_id: str,
                        include_candidates: bool = Query(True),
                        db: AsyncSession = Depends(get_db)):
    """Run + report (+ candidates). The report is the trust surface: null
    method, FDR discipline, replication rate, caps — all first-class."""
    return _discovery_out(await _discovery_or_404(db, run_id),
                          include_candidates=include_candidates)


class BuildCircuitBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    narrative: Optional[str] = Field(None, max_length=10_000)
    # [[up_layer, up_idx, down_layer, down_idx], ...]; empty ⇒ all candidates
    candidate_keys: List[List[int]] = Field(default_factory=list, max_length=400)


@router.post("/circuit-discovery/{run_id}/build-circuit", status_code=201)
async def build_circuit_from_discovery(run_id: str, body: BuildCircuitBody,
                                       db: AsyncSession = Depends(get_db)):
    """Build a circuit from selected candidates of a discovery run (R2 — the
    MISSING discovery→circuit producer). The circuit carries `discovery_run_id`
    so a later validation pass's rung-2 ES propagates onto it (once promoted),
    which is what 015 reads. Created UNPROMOTED; promote + validate separately."""
    from ....services.circuit_service import (
        CircuitService, CircuitValidationError)

    run = await _discovery_or_404(db, run_id)
    if run.status != "completed":
        raise HTTPException(409, f"Discovery run is {run.status} — "
                                 f"build needs a completed run")
    keys = [tuple(k) for k in body.candidate_keys]
    try:
        circuit = await CircuitService.from_candidates(
            db, discovery_run_id=run_id, name=body.name,
            candidate_keys=keys, narrative=body.narrative)
    except CircuitValidationError as e:
        raise HTTPException(422, str(e))
    return {"id": circuit.id, "discovery_run_id": run_id,
            "member_count": len(circuit.members or []),
            "edge_count": len(circuit.edges or []), "rung": circuit.rung}


@router.post("/circuit-discovery/{run_id}/cancel")
async def cancel_discovery(run_id: str, db: AsyncSession = Depends(get_db)):
    from ....core.celery_app import revoke_task

    run = await _discovery_or_404(db, run_id)
    if run.status not in ("pending", "running"):
        raise HTTPException(409, f"Run is {run.status} — nothing to cancel")
    run.status = "cancelled"
    await db.commit()
    if run.celery_task_id:
        revoke_task(run.celery_task_id)
    return {"id": run.id, "status": "cancelled"}


@router.delete("/circuit-discovery/{run_id}")
async def delete_discovery(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await _discovery_or_404(db, run_id)
    if run.status == "running":
        raise HTTPException(409, "Cancel the run before deleting it")
    await db.delete(run)
    await db.commit()
    return {"deleted": run_id}


@router.post("/circuit-discovery/{run_id}/attribution", status_code=202)
async def start_attribution(run_id: str, body: AttributionCreate,
                            db: AsyncSession = Depends(get_db)):
    """Tier-2 gradient attribution pass over the run's candidates (IDL-36)."""
    from ....workers.circuit_capture_tasks import run_circuit_attribution

    run = await _discovery_or_404(db, run_id)
    if run.status != "completed":
        raise HTTPException(409, f"Discovery run is {run.status} — "
                                 f"attribution needs a completed run")
    if not run.candidates:
        raise HTTPException(409, "Run has no candidates to attribute")
    # Separate attribution lifecycle (R1 QA-P2): never overwrite the completed
    # discovery's status. 409 if an attribution pass is already in flight.
    if run.attribution_status in ("pending", "running"):
        raise HTTPException(409, "An attribution pass is already in flight")
    # Attribution loads a model — it's a GPU task and must respect the same
    # single-GPU guard as capture (R2 Q1). The guard + the attribution_status
    # write share ONE advisory-locked transaction so the check-then-mark can't
    # race a concurrent capture/attribution (R2 B2).
    from ....services.circuit_capture_service import (
        CaptureConflictError, CircuitCaptureService)

    def _guard_and_mark(sync_db):
        CircuitCaptureService.assert_no_active_gpu_run(sync_db)
        row = sync_db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.id == run_id).first()
        row.attribution_status = "pending"
        row.attribution_progress = 0.0
        row.attribution_error = None
        sync_db.commit()

    try:
        await _run_sync(db, _guard_and_mark)
    except CaptureConflictError as e:
        raise HTTPException(409, str(e))
    task = run_circuit_attribution.delay(run.id, prompt_limit=body.prompt_limit)
    await db.execute(
        CircuitDiscoveryRun.__table__.update()
        .where(CircuitDiscoveryRun.id == run.id)
        .values(attribution_task_id=task.id))  # own id — discovery's intact (R2 A3)
    await db.commit()
    return {"id": run.id, "task_id": task.id, "status": "queued"}


@router.post("/circuit-discovery/{run_id}/attribution/cancel")
async def cancel_attribution(run_id: str, db: AsyncSession = Depends(get_db)):
    """Cancel an in-flight attribution pass (R2 Q2/B6 — the worker polls
    attribution_status, but nothing set it to cancelled before)."""
    from ....core.celery_app import revoke_task

    run = await _discovery_or_404(db, run_id)
    if run.attribution_status not in ("pending", "running"):
        raise HTTPException(
            409, f"Attribution is {run.attribution_status} — nothing to cancel")
    run.attribution_status = "cancelled"
    await db.commit()
    if run.attribution_task_id:
        revoke_task(run.attribution_task_id)
    return {"id": run.id, "attribution_status": "cancelled"}


# ── sync bridge (create_run/delete_run are worker-shared sync code) ──────

async def _run_sync(db: AsyncSession, fn):
    """Run a sync-session service function via the sync engine in a thread."""
    import anyio

    from ....core.database import get_sync_db

    def _call():
        # `get_sync_db` is @contextmanager-decorated, so calling it returns a
        # _GeneratorContextManager — NOT a generator. `next(...)` on that
        # raises TypeError, so EVERY request through this bridge 500'd:
        # capture, discovery, attribution, validation and their cancels.
        #
        # The endpoints existed, were registered and were documented, and not
        # one of them could ever succeed. Every other caller in the codebase
        # uses `with get_sync_db()`; this one hand-rolled the protocol and got
        # it wrong.
        with get_sync_db() as sync_db:
            return fn(sync_db)

    return await anyio.to_thread.run_sync(_call)


# ── A.4 refinement: which member pairs carry a cluster-level edge ──────────


class RefineClusterEdgeBody(BaseModel):
    """One cluster-level edge, named by its two profiles and their layers."""

    capture_run_id: str = Field(..., min_length=1, max_length=64)
    up_cluster_profile_id: str = Field(..., min_length=1, max_length=64)
    up_layer: int = Field(..., ge=0, le=512)
    down_cluster_profile_id: str = Field(..., min_length=1, max_length=64)
    down_layer: int = Field(..., ge=0, le=512)
    s_min: Optional[int] = Field(None, ge=1, le=100_000)
    null_shuffles: Optional[int] = Field(None, ge=1, le=1000)
    null_percentile: Optional[float] = Field(None, gt=0, lt=100)
    fdr_q: Optional[float] = Field(None, gt=0, lt=1)
    max_null_tested: int = Field(200, ge=1, le=4000)
    seed: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _upstream_is_upstream(self):
        if self.up_layer >= self.down_layer:
            raise ValueError(
                "refinement runs upstream→downstream; up_layer must be less "
                "than down_layer")
        return self


@router.post("/circuit-discovery/refine-cluster-edge")
async def refine_cluster_edge_route(body: RefineClusterEdgeBody):
    """Which member pairs carry a cluster-level edge (Appendix A.4).

    A supernode's activation is `A_C(t) = max_k a_{l,i_k}(t)`, so a cluster
    edge's effect size was measured on a signal that at any token is one
    member's activation. The edge says the two clusters are linked; it does not
    say which members do the linking, and the number cannot be apportioned to
    member pairs by arithmetic. A.4's answer is to MEASURE — the A.3 statistics
    restricted to the two memberships — which is what this runs.

    OFF THE EVENT LOOP. The work is numpy: one hundred circular-shift null
    passes per surviving pair. Running it inline would block every other request
    on the worker for the duration.
    """
    import asyncio

    from ....core.database import get_sync_db
    from ....models.cluster_profile import ClusterProfile
    from ....services import circuit_discovery_service as discovery

    def _work():
        with get_sync_db() as db:
            profs = {
                p.id: p for p in db.query(ClusterProfile).filter(
                    ClusterProfile.id.in_([body.up_cluster_profile_id,
                                           body.down_cluster_profile_id])).all()
            }
            missing = [pid for pid in (body.up_cluster_profile_id,
                                       body.down_cluster_profile_id)
                       if pid not in profs]
            if missing:
                raise LookupError(f"cluster profile(s) not found: {missing}")
            return discovery.refine_cluster_edge(
                db,
                body.capture_run_id,
                {"layer": body.up_layer,
                 "cluster_profile_id": body.up_cluster_profile_id,
                 "members": profs[body.up_cluster_profile_id].members or []},
                {"layer": body.down_layer,
                 "cluster_profile_id": body.down_cluster_profile_id,
                 "members": profs[body.down_cluster_profile_id].members or []},
                s_min=body.s_min,
                null_shuffles=body.null_shuffles,
                null_percentile=body.null_percentile,
                fdr_q=body.fdr_q,
                max_null_tested=body.max_null_tested,
                seed=body.seed,
            )

    try:
        return await asyncio.to_thread(_work)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except discovery.DiscoveryConfigError as e:
        raise HTTPException(422, str(e))
