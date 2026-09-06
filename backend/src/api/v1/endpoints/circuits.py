"""
Circuit endpoints (Feature 018, IDL-33): CRUD, promotion, rung-aware listing,
contract import/export, and per-layer v1 slice export.

Every response carries the rung AND the server-rendered rung_language string
(IDL-35 — the ONE language source; clients never invent causal phrasing).
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....models.circuit import Circuit
from ....schemas.circuit_definition import CircuitDefinitionV1
from ....schemas.evidence_ladder import EvidenceRung, RUNG_NEXT_STEP, rung_language
from ....services.circuit_service import (
    CircuitConcurrencyError, CircuitService, CircuitValidationError)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/circuits", tags=["circuits"])

Granularity = Literal["feature", "cluster"]


class CircuitCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    narrative: Optional[str] = Field(None, max_length=10_000)
    granularity: Granularity = "feature"
    saes: List[Dict[str, Any]]
    members: List[Dict[str, Any]]
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    budget: Optional[Dict[str, Any]] = None
    faithfulness: Optional[Dict[str, Any]] = None
    discovery: Optional[Dict[str, Any]] = None
    discovery_run_id: Optional[str] = Field(None, max_length=36)
    model_id: Optional[str] = Field(None, max_length=255)
    model_hf_id: Optional[str] = Field(None, max_length=500)


class CircuitUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    narrative: Optional[str] = Field(None, max_length=10_000)
    granularity: Optional[Granularity] = None
    saes: Optional[List[Dict[str, Any]]] = None
    members: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    budget: Optional[Dict[str, Any]] = None
    faithfulness: Optional[Dict[str, Any]] = None
    # Optimistic concurrency (017 Task 3.0): when set, a stale value 409s so a
    # panel edit can't silently clobber a validation write. Omitted → no check
    # (back-compatible for callers that don't participate).
    expected_version: Optional[int] = None


class PromoteBody(BaseModel):
    promoted: bool = True  # a badge you can pin is a badge you can unpin


def _summary(circuit: Circuit) -> Dict[str, Any]:
    """Slim list row — full JSONB bodies only on detail fetch (review R1)."""
    rung = EvidenceRung(circuit.rung)
    layers = sorted({m.get("layer") for m in (circuit.members or [])})
    return {
        "id": circuit.id,
        "name": circuit.name,
        "granularity": circuit.granularity,
        "layers": layers,
        "member_count": len(circuit.members or []),
        "edge_count": len(circuit.edges or []),
        "rung": int(rung),
        "rung_language": rung_language(rung),
        "rung_next_step": RUNG_NEXT_STEP[rung],
        "promoted": circuit.promoted,
        "model_id": circuit.model_id,
        "version": circuit.version,  # optimistic-concurrency token (017 Task 3.0)
        "updated_at": circuit.updated_at,
    }


def _out(circuit: Circuit) -> Dict[str, Any]:
    rung = EvidenceRung(circuit.rung)
    return {
        **_summary(circuit),
        "narrative": circuit.narrative,
        "saes": circuit.saes,
        "members": circuit.members,
        "edges": circuit.edges,
        "budget": circuit.budget,
        "faithfulness": circuit.faithfulness,
        "faithfulness_status": getattr(circuit, "faithfulness_status", None),
        "discovery": getattr(circuit, "discovery", None),
        "discovery_run_id": circuit.discovery_run_id,
        "model_hf_id": getattr(circuit, "model_hf_id", None),
        "created_at": circuit.created_at,
    }


async def _get_or_404(db: AsyncSession, circuit_id: str) -> Circuit:
    circuit = await CircuitService.get(db, circuit_id)
    if circuit is None:
        raise HTTPException(status_code=404, detail=f"Circuit {circuit_id} not found")
    return circuit


def _ascii_filename(name: str, suffix: str) -> str:
    """Content-Disposition is latin-1 — non-ASCII names crashed the header."""
    safe = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "-_")) else "-"
                   for ch in name.lower()).strip("-") or "circuit"
    return f"{safe}{suffix}"


@router.get("")
async def list_circuits(
    promoted: Optional[bool] = Query(None),
    min_rung: Optional[int] = Query(None, ge=0, le=3),
    granularity: Optional[Granularity] = Query(None),
    edge_type: Optional[Literal["computed", "persistence", "attention_mediated"]] = Query(
        None, description="Only circuits containing at least one edge of this type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    circuits = await CircuitService.list(
        db, promoted=promoted, min_rung=min_rung, granularity=granularity)
    if edge_type is not None:
        circuits = [c for c in circuits
                    if any(e.get("type") == edge_type for e in (c.edges or []))]
    total = len(circuits)
    page = circuits[offset:offset + limit]
    return {"circuits": [_summary(c) for c in page], "total": total,
            "limit": limit, "offset": offset}


@router.post("", status_code=201)
async def create_circuit(body: CircuitCreate, db: AsyncSession = Depends(get_db)):
    try:
        circuit = await CircuitService.create(db, body.model_dump())
    except CircuitValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _out(circuit)




@router.post("/import", status_code=201)
async def import_circuit(payload: Dict[str, Any],
                         db: AsyncSession = Depends(get_db)):
    """Import a mistudio.circuit-definition/v1 file (BR-013 round-trip).

    Kind-keyed: unknown kinds/major versions are rejected explicitly. Payloads
    over the house 1 MB cap are rejected (R2 B3).
    """
    # The body cap moved to `core.body_limit.BodySizeLimitMiddleware`
    # (MIS-E2E-036). It could not work here: FastAPI parses `payload` before
    # this function's first statement, so the bytes were already read and
    # already in memory by the time the check ran — and a chunked request sends
    # no Content-Length, so the check did not run at all. The middleware meters
    # the receive channel instead, and returns the same 413.
    kind = payload.get("kind")
    if kind != "mistudio.circuit-definition":
        raise HTTPException(
            status_code=422,
            detail=f"Unknown kind {kind!r} — expected 'mistudio.circuit-definition'")
    try:
        defn = CircuitDefinitionV1.model_validate(payload)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(part) for part in first.get("loc", []))
        raise HTTPException(
            status_code=422,
            detail=(f"Invalid circuit definition: {e.error_count()} error(s); "
                    f"first at '{loc}': {first.get('msg', '')}"))
    # Lossless import (R2 B1/B7): model ref, granularity, and the authored
    # created_at all survive the round-trip.
    granularity = "cluster" if any(
        m.member_kind == "cluster_ref" for m in defn.members
    ) else (defn.discovery.granularity if defn.discovery and defn.discovery.granularity
            else "feature")
    try:
        circuit = await CircuitService.create(db, {
            "name": defn.name,
            "narrative": defn.narrative,
            "granularity": granularity,
            "model_id": defn.model.mistudio_model_id,
            "model_hf_id": defn.model.hf_id,
            "created_at": defn.provenance.created_at,
            "saes": [s.model_dump(mode="json") for s in defn.saes],
            "members": [m.model_dump(mode="json") for m in defn.members],
            "edges": [e.model_dump(mode="json") for e in defn.edges],
            "budget": defn.budget.model_dump(mode="json") if defn.budget else None,
            "faithfulness": (defn.faithfulness.model_dump(mode="json")
                             if defn.faithfulness else None),
            # MIS-E2E-037: this key was simply absent. CircuitService.create
            # reads data.get("calibration") and the column has existed all
            # along, so an imported circuit silently lost its whole calibrated
            # band — onset, sweet_spot, cliff, probe_set, and the `provisional`
            # honesty marker — while budget.intensity_range kept the clamped
            # numbers those measurements produced. IDL-37 clause 5 exists so the
            # probe set TRAVELS for a cheap serve-time re-verify; dropping it
            # defeats the decision on the way in while the export honours it on
            # the way out.
            "calibration": (defn.calibration.model_dump(mode="json")
                            if defn.calibration else None),
            "discovery": (defn.discovery.model_dump(mode="json")
                          if defn.discovery else None),
        })
    except CircuitValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _out(circuit)


@router.get("/{circuit_id}")
async def get_circuit(circuit_id: str, db: AsyncSession = Depends(get_db)):
    return _out(await _get_or_404(db, circuit_id))


@router.patch("/{circuit_id}")
async def update_circuit(circuit_id: str, body: CircuitUpdate,
                         db: AsyncSession = Depends(get_db)):
    circuit = await _get_or_404(db, circuit_id)
    data = body.model_dump(exclude_unset=True)
    expected_version = data.pop("expected_version", None)
    # Structural fields cannot be nulled — reject explicitly rather than
    # surfacing a raw pydantic error (review R1 finding #12).
    for field in ("name", "saes", "members", "edges"):
        if field in data and data[field] is None:
            raise HTTPException(status_code=422,
                                detail=f"'{field}' cannot be null")
    try:
        circuit = await CircuitService.update(
            db, circuit, data, expected_version=expected_version)
    except CircuitValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except CircuitConcurrencyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _out(circuit)


@router.delete("/{circuit_id}")
async def delete_circuit(circuit_id: str, db: AsyncSession = Depends(get_db)):
    circuit = await _get_or_404(db, circuit_id)
    await CircuitService.delete(db, circuit)
    return {"deleted": circuit_id}


@router.post("/{circuit_id}/promote")
async def promote_circuit(circuit_id: str, body: Optional[PromoteBody] = None,
                          db: AsyncSession = Depends(get_db)):
    """Badge, not gate (BR-012) — and reversible (pass {"promoted": false})."""
    circuit = await _get_or_404(db, circuit_id)
    promoted = body.promoted if body is not None else True
    return _out(await CircuitService.set_promoted(db, circuit, promoted))


class FaithfulnessBody(BaseModel):
    mode: Literal["necessity", "both"] = "both"
    k_nonmembers: int = Field(256, ge=1, le=4096)
    ablate_all_n: int = Field(1024, ge=1, le=16384)
    n_prompts: int = Field(16, ge=1, le=256)
    seed: int = 0


@router.post("/{circuit_id}/faithfulness", status_code=202)
async def start_faithfulness(circuit_id: str, body: FaithfulnessBody,
                             db: AsyncSession = Depends(get_db)):
    """Launch a faithfulness pass (rung 3) on a circuit: suppress its members
    and measure the necessity/sufficiency of the behavior they drive vs an
    ablate-all proxy. GPU-guarded (shares the single-GPU circuit guard). 404 if
    the circuit is missing; 409 if it has no members. Result + a `faithfulness`
    manifest are the record; circuit.faithfulness is written through the
    contract. Poll get_task_status, then get_circuit for the scores."""
    from ....services.circuit_capture_service import (
        CaptureConflictError, CircuitCaptureService)
    from ....services.circuit_faithfulness_service import FaithfulnessConfigError
    from ....workers.circuit_validation_tasks import run_circuit_faithfulness

    circuit = await _get_or_404(db, circuit_id)
    if not circuit.members:
        raise HTTPException(status_code=409, detail="Circuit has no members")
    if not circuit.discovery_run_id:
        raise HTTPException(
            status_code=409,
            detail="v1 faithfulness needs the circuit's discovery capture "
                   "store for prompts (circuit has no discovery_run_id)")
    try:
        from ....services.circuit_faithfulness_service import (
            CircuitFaithfulnessService)
        config = CircuitFaithfulnessService.create_config(body.model_dump())
    except FaithfulnessConfigError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if circuit.faithfulness_status in ("pending", "running"):
        raise HTTPException(status_code=409,
                            detail="A faithfulness pass is already in flight")
    # Faithfulness loads a model — respect the single-GPU guard AND mark it
    # in-flight in ONE advisory-locked sync transaction so the guard sees it and
    # two runs can't race (R2 B-5), like start_validation does for validation.
    from ....api.v1.endpoints.circuit_discovery import _run_sync

    def _guard_and_mark(sync_db):
        CircuitCaptureService.assert_no_active_gpu_run(sync_db)
        row = sync_db.query(Circuit).filter(Circuit.id == circuit_id).first()
        row.faithfulness_status = "pending"
        sync_db.commit()

    try:
        await _run_sync(db, _guard_and_mark)
    except CaptureConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    task = run_circuit_faithfulness.delay(circuit_id, config)
    await db.execute(
        Circuit.__table__.update().where(Circuit.id == circuit_id)
        .values(faithfulness_task_id=task.id))
    await db.commit()
    return {"circuit_id": circuit_id, "task_id": task.id, "status": "queued"}


class CalibrationBody(BaseModel):
    step_budget: int = Field(10, ge=2, le=40)
    probe_count: int = Field(3, ge=1, le=10)
    margin: float = Field(0.15, ge=0.0, le=1.0)
    seed: int = 0
    # The judge / probe-generation LLM (OpenAI-compatible), carried per request
    # like an enhanced-labeling job. Required for a real run — the correctness
    # cliff cannot be found without a judge.
    judge_endpoint: Optional[str] = None
    judge_model: Optional[str] = None

    @field_validator("judge_endpoint")
    @classmethod
    def _validate_judge_endpoint(cls, v: Optional[str]) -> Optional[str]:
        """MIS-E2E-075: this URL is POSTed to from inside the cluster.

        `judge_endpoint` is free-form per-request input handed straight to an
        `OpenAI` client running in the backend pod, with no validation — not
        even a scheme check. That is a server-side request forgery primitive:
        a caller chooses the host the pod connects to, and the pod can reach
        things the caller cannot, including cloud instance metadata.

        `validate_llm_endpoint_url` is the check the labeling path already
        applies to exactly this kind of field — it permits http/https to public,
        RFC-1918 and loopback addresses, since an internal LLM server is the
        normal case, and blocks metadata addresses and non-http schemes. Reusing
        it rather than writing a second one keeps the two paths from drifting.
        """
        if v is None:
            return v
        from src.utils.url_validation import validate_llm_endpoint_url

        return validate_llm_endpoint_url(v)


@router.post("/{circuit_id}/calibration", status_code=202)
async def start_calibration(circuit_id: str, body: CalibrationBody,
                            db: AsyncSession = Depends(get_db)):
    """Launch a strength-calibration pass (Feature 20 / IDL-37) on a circuit:
    find the usable dial band — onset (min influence above baseline noise) and
    the correctness cliff (max before the model's facts break, judged against
    generated neutral-topic probes) — then clamp the served dial to it. GPU-
    guarded (shares the single-GPU circuit guard, like faithfulness). Badge, not
    gate. 404 if the circuit is missing; 409 if it has no members or a pass is
    already in flight. Poll get_task_status, then get_circuit for the band."""
    from ....services.circuit_calibration_service import CalibrationConfigError
    from ....services.circuit_capture_service import (
        CaptureConflictError, CircuitCaptureService)
    from ....workers.circuit_calibration_tasks import run_circuit_calibration

    circuit = await _get_or_404(db, circuit_id)
    if not circuit.members:
        raise HTTPException(status_code=409, detail="Circuit has no members")
    try:
        from ....services.circuit_calibration_service import (
            CircuitCalibrationService)
        config = CircuitCalibrationService.create_config(body.model_dump())
    except CalibrationConfigError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if circuit.calibration_status in ("pending", "running"):
        raise HTTPException(status_code=409,
                            detail="A calibration pass is already in flight")
    from ....api.v1.endpoints.circuit_discovery import _run_sync

    def _guard_and_mark(sync_db):
        CircuitCaptureService.assert_no_active_gpu_run(sync_db)
        row = sync_db.query(Circuit).filter(Circuit.id == circuit_id).first()
        row.calibration_status = "pending"
        sync_db.commit()

    try:
        await _run_sync(db, _guard_and_mark)
    except CaptureConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    task = run_circuit_calibration.delay(circuit_id, config)
    await db.execute(
        Circuit.__table__.update().where(Circuit.id == circuit_id)
        .values(calibration_task_id=task.id))
    await db.commit()
    return {"circuit_id": circuit_id, "task_id": task.id, "status": "queued"}


@router.post("/calibration-manifests/{manifest_id}/reproduce", status_code=202)
async def reproduce_calibration(manifest_id: str,
                                db: AsyncSession = Depends(get_db)):
    """Re-run a calibration from its manifest and record a reproduction manifest
    with the band-delta verdict (FPRD §8.5). GPU pass — shares the single-GPU
    guard. 404 if the manifest is missing; 409 if it is not a calibration
    manifest or a pass is already in flight on its circuit."""
    from ....services.circuit_calibration_service import CircuitCalibrationService
    from ....services.circuit_capture_service import (
        CaptureConflictError, CircuitCaptureService)
    from ....services.manifest_service import ManifestService
    from ....workers.circuit_calibration_tasks import reproduce_circuit_calibration

    m = await ManifestService.get(db, manifest_id)
    if m is None:
        raise HTTPException(404, f"Manifest {manifest_id} not found")
    if m.kind != "calibration":
        raise HTTPException(
            409, f"Manifest {manifest_id} is {m.kind!r}, not a calibration "
            "manifest — nothing to reproduce")
    circuit = await _get_or_404(db, m.circuit_id)
    if circuit.calibration_status in ("pending", "running"):
        raise HTTPException(409, "A calibration pass is already in flight")
    # Reproduce holds the GPU (sets 'pending') but does NOT re-calibrate, so we
    # remember the circuit's real status and restore it when the task finishes.
    prior_status = circuit.calibration_status
    from ....api.v1.endpoints.circuit_discovery import _run_sync

    def _guard_and_mark(sync_db):
        CircuitCaptureService.assert_no_active_gpu_run(sync_db)
        row = sync_db.query(Circuit).filter(Circuit.id == m.circuit_id).first()
        row.calibration_status = "pending"
        sync_db.commit()

    try:
        await _run_sync(db, _guard_and_mark)
    except CaptureConflictError as e:
        raise HTTPException(409, str(e))
    task = reproduce_circuit_calibration.delay(manifest_id, prior_status)
    await db.execute(
        Circuit.__table__.update().where(Circuit.id == m.circuit_id)
        .values(calibration_task_id=task.id))
    await db.commit()
    return {"reproduces": manifest_id, "task_id": task.id, "status": "queued"}


class RecordArtifact(BaseModel):
    kind: Literal["circuit", "cluster", "features"]
    circuit_id: Optional[str] = None
    cluster_profile_id: Optional[str] = None
    features: Optional[List[Dict[str, Any]]] = None
    model_id: Optional[str] = None


class SteeringSamplesBody(BaseModel):
    artifact: RecordArtifact
    dials: List[float]
    prompts: List[str]
    max_tokens: Optional[int] = None
    seed: int = 0


@router.post("/steering-samples", status_code=202)
async def start_steering_samples(body: SteeringSamplesBody,
                                 db: AsyncSession = Depends(get_db)):
    """Record (dial, prompt, unsteered, steered) transcripts for a circuit,
    cluster, or ad-hoc feature set on the GPU — the raw material for a post-run
    LLM meaning-analysis pass. Judge-free. GPU-guarded (shares the single-GPU
    circuit guard). 422 on bad config/caps; 409 if the GPU is busy. Poll
    get_task_status, then read the steering_samples manifest."""
    from ....models.steering_record_run import SteeringRecordRun
    from ....services.circuit_capture_service import (
        CaptureConflictError, CircuitCaptureService)
    from ....services.steering_recorder_service import (RecordConfigError,
                                                        SteeringRecorderService)
    from ....workers.circuit_record_tasks import run_circuit_record

    config = {"artifact": body.artifact.model_dump(exclude_none=True),
              "dials": body.dials, "prompts": body.prompts,
              "max_tokens": body.max_tokens, "seed": body.seed}
    try:
        config = SteeringRecorderService.create_config(config)
    except RecordConfigError as e:
        raise HTTPException(status_code=422, detail=str(e))

    art = config["artifact"]
    ref = (art.get("circuit_id") or art.get("cluster_profile_id"))
    from ....api.v1.endpoints.circuit_discovery import _run_sync

    def _guard_and_mark(sync_db):
        # Under the advisory lock: refuse if the GPU is busy, else create the
        # in-flight marker atomically so the guard-check and marker-set can't race.
        CircuitCaptureService.assert_no_active_gpu_run(sync_db)
        row = SteeringRecordRun(status="pending", artifact_kind=art["kind"],
                                artifact_ref=ref)
        sync_db.add(row)
        sync_db.commit()
        sync_db.refresh(row)
        return row.id

    try:
        record_run_id = await _run_sync(db, _guard_and_mark)
    except CaptureConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    task = run_circuit_record.delay(record_run_id, config)
    await db.execute(
        SteeringRecordRun.__table__.update()
        .where(SteeringRecordRun.id == record_run_id)
        .values(task_id=task.id))
    await db.commit()
    return {"record_run_id": record_run_id, "task_id": task.id, "status": "queued"}


@router.get("/{circuit_id}/export")
async def export_circuit(circuit_id: str, db: AsyncSession = Depends(get_db)):
    circuit = await _get_or_404(db, circuit_id)
    try:
        defn = CircuitService.to_definition(circuit)
    except CircuitValidationError:
        # Never echo internal validation text on a 500 (security hardening
        # precedent — stack-trace exposure remediation).
        logger.exception("Stored circuit %s fails contract validation", circuit_id)
        raise HTTPException(status_code=500,
                            detail="Stored circuit failed contract validation; see server logs")
    return JSONResponse(
        content=defn.model_dump(mode="json"),
        headers={"Content-Disposition":
                 f'attachment; filename="{_ascii_filename(circuit.name, ".circuit.json")}"'},
    )


@router.post("/{circuit_id}/export-slices")
async def export_circuit_slices(circuit_id: str, db: AsyncSession = Depends(get_db)):
    """BR-014: per-layer cluster-definition/v1 slices (partial renderings).

    The parent rung is computed ONCE from the definition (min over edges) so
    the response and every slice marker agree — the stored column is display
    cache, not evidence truth (review R1 finding #4).
    """
    circuit = await _get_or_404(db, circuit_id)
    try:
        defn = CircuitService.to_definition(circuit)
        slices = CircuitService.slices_of(defn)
    except (CircuitValidationError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    parent_rung = int(defn.displayed_rung())
    return {
        "parent": circuit_id,
        "parent_rung": parent_rung,
        "parent_rung_language": rung_language(parent_rung),
        "slices": [s.model_dump(mode="json") for s in slices],
    }


# ── cancellation ─────────────────────────────────────────────────────────
#
# THREE LIFECYCLES THAT WERE STARTABLE AND NOT STOPPABLE. Faithfulness,
# calibration and the steering recorder each had a launch route, a status
# column and (for faithfulness) a complete cancellation path in the service —
# and no way for an operator to reach any of it. The only stop was restarting
# the pod, which on a `--pool=solo` worker also strands the in-flight
# `acks_late` message for the full 12-hour visibility timeout.

async def _cancel_circuit_stage(circuit_id: str, kind: str, status_field: str,
                                task_field: str, db: AsyncSession):
    """Shared body for the three circuit-scoped stages."""
    from starlette.concurrency import run_in_threadpool

    from ....core.cancellation import request_cancel

    circuit = await _get_or_404(db, circuit_id)
    current = getattr(circuit, status_field, None)
    if current not in ("pending", "running"):
        raise HTTPException(
            409, f"{kind.replace('_', ' ')} is {current or 'not running'} — "
                 f"nothing to cancel")

    outcome = await run_in_threadpool(
        request_cancel, kind, circuit_id,
        reason="cancelled by operator",
        celery_task_id=getattr(circuit, task_field, None),
    )
    return {
        "circuit_id": circuit_id,
        status_field: "cancelled",
        "was_running": outcome.was_running,
        "message": outcome.detail,
    }


@router.post("/{circuit_id}/faithfulness/cancel")
async def cancel_faithfulness(circuit_id: str, db: AsyncSession = Depends(get_db)):
    """Stop a running faithfulness pass at its next prompt boundary."""
    return await _cancel_circuit_stage(
        circuit_id, "circuit_faithfulness", "faithfulness_status",
        "faithfulness_task_id", db)


@router.post("/{circuit_id}/calibration/cancel")
async def cancel_calibration(circuit_id: str, db: AsyncSession = Depends(get_db)):
    """Stop a running strength calibration at its next bisection step."""
    return await _cancel_circuit_stage(
        circuit_id, "circuit_calibration", "calibration_status",
        "calibration_task_id", db)


@router.post("/steering-samples/{run_id}/cancel")
async def cancel_steering_record(run_id: str, db: AsyncSession = Depends(get_db)):
    """Stop a running transcript recording at its next sample."""
    from starlette.concurrency import run_in_threadpool
    from sqlalchemy import select

    from ....core.cancellation import request_cancel
    from ....models.steering_record_run import SteeringRecordRun

    result = await db.execute(
        select(SteeringRecordRun).where(SteeringRecordRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(404, f"Record run {run_id} not found")
    if run.status not in ("pending", "running"):
        raise HTTPException(
            409, f"Recording is {run.status} — nothing to cancel")

    outcome = await run_in_threadpool(
        request_cancel, "steering_record", run_id,
        reason="cancelled by operator", celery_task_id=run.task_id,
    )
    return {"id": run_id, "status": "cancelled",
            "was_running": outcome.was_running, "message": outcome.detail}
