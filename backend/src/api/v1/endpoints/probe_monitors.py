"""REST surface for probe monitors (032 FR-1, FR-5, FR-9, FR-11–FR-14).

⚠ VALIDATION THAT NEEDS DATA LIVES IN THE SERVICE, NOT HERE. The schemas reject what a
request alone can settle (an unknown rule, a stride below 1); the 20-per-class floor, a
layer beyond the model's depth and a missing SAE all need the database or the model, and
they return 422 naming the offending value. A route that guessed at those would produce a
202 for a run that can only ever fail when the worker places it.

⚠ `GET /probes/{id}` ASSEMBLES THE REPORT SERVER-SIDE, including the rung's wording. A
detector's language is the thing most likely to drift above its evidence, and miLLM
mirrors these strings verbatim — so the client is never handed a rung number and left to
phrase it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.config import settings
from ....core.deps import get_db
from ....models.dataset import Dataset
from ....models.probe_monitor import (
    ProbeMonitor,
    ProbeMonitorDataset,
    ProbeMonitorEvaluation,
    ProbeMonitorJudgeRun,
    ProbeMonitorRun,
)
from ....schemas.evidence_ladder import (
    probe_rung_language,
    probe_rung_next_step,
)
from ....schemas.probe_monitor import (
    JudgeRunCreate,
    ProbeDatasetArchiveRequest,
    ProbeDatasetCreate,
    ProbeDatasetPurgePlanRequest,
    ProbeDatasetPurgeRequest,
    ProbeDatasetResponse,
    ProbeDecisionResponse,
    ProbeEvaluationResponse,
    ProbeEvaluationScores,
    ProbeMonitorSummary,
    ProbeReport,
    ProbeRunCreate,
    ProbeRunResponse,
    RecalibrateRequest,
    ScoreRequest,
)
from ....schemas.probe_definition import HEX64
from ....services import probe_dataset_lifecycle as lifecycle
from ....services.probe_decision import (
    backfill_identity,
    digest_match,
    identity_fields,
    needs_backfill,
)
from ....services.gpu_dispatch import gpu_delay
from ..gpu_request import resolve_gpu_request
from ._probe_ids import refuse_blend_report_ids
from ....schemas.probe_blend import ScoreBatch

logger = logging.getLogger(__name__)

from ....services.detector_next import DetectorRefused

# Feature 041 FR-18: a blend report id (`pbr_…`) is refused on EVERY `{probe_id}` route of this router
# (router-level, so a route added later is covered). Removing it fails test_not_a_probe.py.
router = APIRouter(prefix="/probe-monitors", tags=["probe-monitors"],
                   dependencies=[Depends(refuse_blend_report_ids)])


# ── probe datasets ────────────────────────────────────────────────────────────



def requested_dtype_refusal(requested: Optional[str], model: Any) -> Optional[str]:
    """Why a requested `config.dtype` is refused, or None when it is absent or agrees.

    A run loads at the precision the shared rule gives its model row (`ml/native_dtype.py`);
    `config.dtype` cannot change that. So a value that DISAGREES is refused, naming the dtype
    the run would really use, rather than accepted and ignored — which is what this field did
    for a month while the loader hardcoded float16.
    """
    if requested is None:
        return None
    from ....services.activation_service import resolve_model_snapshot
    from ....services.probe_monitor_run import resolve_weights_dir
    from ....ml.native_dtype import resolve_for_snapshot

    try:
        snapshot = resolve_model_snapshot(
            resolve_weights_dir(model.id, model.file_path, model.quantized_path)
        )
        loads_at = resolve_for_snapshot(model.quantization, snapshot).name
    except (ValueError, OSError) as exc:
        return (
            f"config.dtype={requested!r} cannot be checked against model {model.id} ({exc}); "
            "omit it — a run always loads at the precision its model row resolves to"
        )
    if requested != loads_at:
        return (
            f"config.dtype={requested!r}, but model {model.id} "
            f"({getattr(model.quantization, 'value', model.quantization)}) loads at {loads_at}: "
            "the checkpoint's own precision under the shared rule. Omit config.dtype, or change "
            "the model row's quantization."
        )
    return None


async def refuse_precision_mismatch(probe_id: str) -> None:
    """409 NOW, before a GPU task is queued, for work that would mix precisions into a probe.

    The task checks again (`probe_monitor_run.probe_precision`) — this only spares the operator a
    202 that fails minutes later in the worker.
    """
    def _check() -> Optional[str]:
        from ....core.database import SyncSessionLocal
        from ....models.probe_monitor import ProbeMonitorRun
        from ....services.probe_monitor_run import probe_precision

        session = SyncSessionLocal()
        try:
            probe = session.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
            run = probe and session.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).first()
            return probe_precision(session, run)["refusal"] if run is not None else None
        finally:
            session.close()

    refusal = await run_in_threadpool(_check)
    if refusal is not None:
        raise HTTPException(status_code=409, detail={"code": "precision_mismatch", "message": refusal})


async def refuse_render_mismatch(probe_id: str) -> None:
    """409 NOW for work that would mix two renders into a probe (operator, 2026-10-08).

    A probe rendered without the generation prompt — every probe trained before 2026-10-08 —
    cannot be evaluated, GPU re-cut or exported under the served render. The task and the builder
    check again (`probe_monitor_run.probe_render`); this spares a 202 that fails in the worker.
    """
    def _check() -> Optional[str]:
        from ....core.database import SyncSessionLocal
        from ....models.probe_monitor import ProbeMonitorRun
        from ....services.probe_monitor_run import probe_render

        session = SyncSessionLocal()
        try:
            probe = session.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
            if probe is None:
                return None
            run = session.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).first()
            return probe_render(run, probe)["refusal"]
        finally:
            session.close()

    refusal = await run_in_threadpool(_check)
    if refusal is not None:
        raise HTTPException(status_code=409, detail={"code": "render_mismatch", "message": refusal})

@router.post("/datasets", response_model=ProbeDatasetResponse, status_code=201)
async def create_probe_dataset(
    request: ProbeDatasetCreate, db: AsyncSession = Depends(get_db)
) -> Any:
    """Create a label-mapped view over a downloaded dataset.

    The counts are computed NOW and stored. A view that cannot be scored is refused
    here rather than at run submission, because a view nobody can score is not worth
    keeping — and the refusal names both class counts so it can be acted on.
    """
    import uuid as _uuid

    from ....services.probe_monitor_service import (
        ProbeDatasetRefused,
        ResolvedColumns,
        build_view,
        describe_counts,
        load_columns,
        resolve_dataset_path,
    )

    try:
        dataset_uuid = _uuid.UUID(str(request.dataset_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=422, detail=f"{request.dataset_id} is not a dataset id")

    dataset = (
        await db.execute(select(Dataset).where(Dataset.id == dataset_uuid))
    ).scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")

    # Feature 034 (FR-26): an optional miDataworks dataset-version manifest is checked BEFORE
    # anything is read, except the row count, which needs the download.
    facts = None
    expected_rows = None
    if request.dataset_version_manifest is not None:
        from ....services import dataset_version_manifest as manifests

        if not request.split:
            raise HTTPException(
                status_code=422,
                detail="a dataset-version manifest is checked per split; name the split",
            )
        try:
            facts = manifests.validate(request.dataset_version_manifest)
        except manifests.ManifestInvalid as invalid:
            logger.info("probe dataset refused: manifest invalid at %s", invalid.path)
            raise HTTPException(status_code=422, detail=str(invalid))
        expected_rows = facts.row_count(request.split)
        if expected_rows is None:
            logger.info("probe dataset refused: split %r not in manifest", request.split)
            raise HTTPException(
                status_code=409,
                detail=(
                    f"split {request.split!r} is not described by the manifest; it describes "
                    f"{facts.split_names()}"
                ),
            )

    try:
        path = resolve_dataset_path(dataset.raw_path)
        columns = ResolvedColumns(
            input_column=request.input_column,
            label_column=request.label_column,
            pair_column=request.pair_column,
        )
        inputs, labels, pairs, total = load_columns(path, columns, split=request.split)
        # The manifest describes the SPLIT FILE; miStudio's keyword filter and label mapping are
        # its own narrowing (IDL-55 (2)), so the comparison uses the pre-filter total. Kept as its
        # own statement: it is the line a mutation must break.
        if facts is not None and total != expected_rows:
            logger.info("probe dataset refused: row count mismatch on %r", request.split)
            raise HTTPException(
                status_code=409,
                detail=(
                    f"split {request.split!r}: manifest states {expected_rows} rows, "
                    f"downloaded split has {total}"
                ),
            )
        built = build_view(
            inputs,
            labels,
            dict(request.label_mapping),
            keyword_filter=request.keyword_filter.model_dump()
            if request.keyword_filter
            else None,
            pair_values=pairs,
            role=request.role,
        )
    except ProbeDatasetRefused as refused:
        raise HTTPException(status_code=422, detail=str(refused))
    except FileNotFoundError as missing:
        raise HTTPException(status_code=422, detail=str(missing))
    except ValueError as bad:
        raise HTTPException(status_code=422, detail=str(bad))

    # Feature 040 S2 (FR-4, FR-13): the sub-question is checked against the behaviour version the
    # registration CITES (or the current one), and that version is recorded on the view.
    try:
        cited_version = await _cited_behaviour_version(db, request)
        group_columns = None
        if request.group_columns:
            from ....services import view_columns

            limits = view_columns.ColumnLimits.from_settings()
            group_columns = view_columns.declare_on_registration(request.group_columns, limits)
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal

    view = ProbeMonitorDataset(
        name=request.name,
        dataset_id=dataset_uuid,
        config=request.config,
        split=request.split,
        input_column=request.input_column,
        label_column=request.label_column,
        label_mapping=dict(request.label_mapping),
        keyword_filter=request.keyword_filter.model_dump() if request.keyword_filter else None,
        pair_column=request.pair_column,
        role=request.role,
        distribution=request.distribution,
        counts=describe_counts(built),
        dataset_version=(
            facts.record(request.split, expected_rows, total) if facts is not None else None
        ),
        # Feature 037 (FR-15; D7, D37): what the positive class is, as the sender stated it.
        behaviour=request.behaviour,
        positive_meaning=request.positive_meaning,
        # D48 (TI13): a miDataworks send — it carries a verified dataset-version manifest — that names
        # the behaviour counts as the confirmation that these positives ARE that behaviour. Anything
        # else waits for the operator (`POST /datasets/{id}/confirm-positive`).
        positive_labels_confirmed=(
            _midataworks_confirmation(request.behaviour, request.role)
            if facts is not None and request.behaviour
            else None
        ),
        sub_question=request.sub_question,
        behaviour_version=cited_version,
    )
    db.add(view)
    snapshots = None
    if group_columns:
        await db.flush()  # the view's id names its snapshot directory
        from ....services import view_columns

        try:
            snapshots = await view_columns.snapshot_columns(db, view, group_columns, limits)
        except DetectorRefused as refusal:
            await db.rollback()
            raise refusal.to_http() from refusal
        view.group_columns = group_columns
        view.column_snapshots = snapshots
    try:
        await db.commit()
    except BaseException:
        if snapshots:
            from ....services import view_columns

            view_columns.discard(snapshots)
        raise
    await db.refresh(view)
    return view


async def _cited_behaviour_version(db: AsyncSession, request: ProbeDatasetCreate) -> Optional[int]:
    """FR-4: the version a new view cites (named, else current) and its sub-question, checked against THAT
    version's declared keys. An uncatalogued behaviour has no versions: a sub-question is then refused."""
    from ....models.probe_behaviour import ProbeBehaviour, ProbeBehaviourVersion

    if request.behaviour_version is not None and not request.behaviour:
        raise DetectorRefused(422, "invalid_field", "behaviour_version cites a version of `behaviour`; name it",
                              {"field": "behaviour_version", "rule": "requires_behaviour"})
    if request.sub_question is not None and not request.behaviour:
        raise DetectorRefused(422, "invalid_field", "a sub-question belongs to a behaviour; name it",
                              {"field": "sub_question", "rule": "requires_behaviour"})
    if not request.behaviour:
        return None
    head = (await db.execute(select(ProbeBehaviour).where(ProbeBehaviour.key == request.behaviour))).scalar_one_or_none()
    if head is None:
        if request.sub_question is not None or request.behaviour_version is not None:
            raise DetectorRefused(
                422, "sub_question_undeclared",
                f"{request.behaviour!r} is not catalogued, so it declares no sub-questions or versions",
                {"sub_question": request.sub_question, "declared_keys": [], "behaviour_version": None})
        return None
    number = request.behaviour_version or head.current_version
    version = (await db.execute(select(ProbeBehaviourVersion).where(
        ProbeBehaviourVersion.key == request.behaviour, ProbeBehaviourVersion.version == number))).scalar_one_or_none()
    if version is None:
        raise DetectorRefused(404, "version_not_found", f"{request.behaviour!r} has no version {number}",
                              {"behaviour_version": number, "current_version": head.current_version})
    if request.sub_question is not None:
        declared = [s.get("key") for s in (version.sub_questions or [])]
        if request.sub_question not in declared:
            raise DetectorRefused(
                422, "sub_question_undeclared",
                f"version {number} of {request.behaviour!r} declares no sub-question {request.sub_question!r}; "
                f"it declares {declared}",
                {"sub_question": request.sub_question, "declared_keys": declared, "behaviour_version": number})
    return number


def _midataworks_confirmation(behaviour: str, role: str) -> Optional[Dict[str, Any]]:
    """The D48 confirmation a miDataworks send carries; none for a calibration set (no positives)."""
    if role == "calibration":
        return None
    from ....services.behaviour_catalogue import confirmation_record

    return confirmation_record(behaviour, None, "midataworks", "midataworks")


@router.get("/datasets", response_model=List[ProbeDatasetResponse])
async def list_probe_datasets(
    role: Optional[str] = Query(None),
    include_archived: bool = Query(
        False, description="Also list archived views (hidden by default; archive is reversible)"
    ),
    db: AsyncSession = Depends(get_db),
) -> Any:
    query = select(ProbeMonitorDataset).order_by(ProbeMonitorDataset.created_at.desc())
    if role:
        query = query.where(ProbeMonitorDataset.role == role)
    if not include_archived:
        query = query.where(ProbeMonitorDataset.archived_at.is_(None))
    return list((await db.execute(query)).scalars().all())


# ── archive / purge (operator decision 2026-10-21) ────────────────────────────
#
# Archive hides and is reversible (operator or agent). Purge is permanent, ARCHIVED views only,
# refused with every blocker named, and the operator's alone: every purge route depends on
# `refuse_agents`, and no MCP tool reaches one (`test_mcp_route_parity.R_PURGE`).


@router.post("/datasets/{dataset_id}/archive", response_model=ProbeDatasetResponse)
async def archive_probe_dataset(
    dataset_id: str,
    http: Request,
    request: Optional[ProbeDatasetArchiveRequest] = Body(None),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Hide a view from the list and from new runs. Nothing is deleted; unarchive undoes it."""
    actor = lifecycle.actor_of(http)
    view, _ = await lifecycle.archive(
        db, dataset_id, actor, request.reason if request else None
    )
    return view


@router.post("/datasets/{dataset_id}/unarchive", response_model=ProbeDatasetResponse)
async def unarchive_probe_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    view, _ = await lifecycle.unarchive(db, dataset_id)
    return view


@router.post("/datasets/purge/plan", dependencies=[Depends(lifecycle.refuse_agents)])
async def plan_probe_dataset_purge(
    request: ProbeDatasetPurgePlanRequest, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """What purging each view would remove and keep, and everything that blocks it. Changes
    nothing. The UI reads this to decide which checkboxes are selectable."""
    return {"items": await lifecycle.plan(db, request.ids)}


@router.post("/datasets/purge", dependencies=[Depends(lifecycle.refuse_agents)])
async def purge_probe_datasets(
    request: ProbeDatasetPurgeRequest, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Permanently delete exactly these ARCHIVED views, each on its own; one tombstone each."""
    reason = lifecycle.clean_reason(request.reason)
    return await lifecycle.purge_many(db, request.ids, lifecycle.OPERATOR, reason)


@router.get("/dataset-purges", dependencies=[Depends(lifecycle.refuse_agents)])
async def list_probe_dataset_purges(
    limit: int = Query(100, ge=1, le=500), db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """The purge log: one append-only tombstone per permanently deleted view."""
    items = await lifecycle.purge_log(db, limit)
    return {"items": items, "total": len(items)}


@router.get("/datasets/{dataset_id}/rows")
async def browse_probe_dataset_rows(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    columns: Optional[str] = Query(
        None,
        description="Comma-separated OTHER columns to return; the view's input, label and pair "
        "columns are always returned. Omit for every column; an empty value for none.",
    ),
    label: Optional[str] = Query(
        None, description="positive | negative | excluded | filtered_out | unparseable"
    ),
    search: Optional[str] = Query(None, max_length=500, description="Text the input contains"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """A page of a view's rows with each row's MAPPED label, read-only.

    The label comes from the same per-row classifier that produced the view's stored counts, so
    the two cannot disagree; when a label filter forces a full pass the response also carries
    `recount` and `counts_agree`.
    """
    from ....services import probe_dataset_rows as rows
    from ....services.probe_monitor_service import resolve_dataset_path

    view = (
        await db.execute(select(ProbeMonitorDataset).where(ProbeMonitorDataset.id == dataset_id))
    ).scalar_one_or_none()
    if view is None:
        raise HTTPException(status_code=404, detail=f"Probe dataset {dataset_id} not found")
    dataset = (
        await db.execute(select(Dataset).where(Dataset.id == view.dataset_id))
    ).scalar_one_or_none()
    if dataset is None:
        raise HTTPException(
            status_code=409,
            detail="the downloaded dataset behind this view is gone, so its rows cannot be read",
        )
    projection = None
    if columns is not None:
        projection = [c.strip() for c in columns.split(",") if c.strip()]
    try:
        path = resolve_dataset_path(dataset.raw_path)
        page = await run_in_threadpool(
            rows.browse,
            rows.ViewSpec.of(view),
            path,
            offset=offset,
            limit=limit,
            columns=projection,
            label=label,
            search=search,
        )
    except FileNotFoundError as missing:
        raise HTTPException(status_code=409, detail=str(missing))
    except (rows.RowsRefused, ValueError) as refused:
        raise HTTPException(status_code=422, detail=str(refused))
    stored = {k: (view.counts or {}).get(k) for k in ("positive", "negative", "excluded", "filtered_out", "unparseable")}
    page["name"] = view.name
    page["role"] = view.role
    page["dataset_name"] = dataset.name
    page["counts"] = stored
    page["counts_agree"] = None if page["recount"] is None else page["recount"] == stored
    return page


@router.get("/datasets/{dataset_id}", response_model=ProbeDatasetResponse)
async def get_probe_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    """One probe dataset view, with its counts and any miDataworks manifest it was checked against.

    Added 2026-10-19 for the MCP probe surface: the list was the only reader, so an agent that had
    registered a view could confirm it only by paging every view of every role.
    """
    view = (
        await db.execute(
            select(ProbeMonitorDataset).where(ProbeMonitorDataset.id == dataset_id)
        )
    ).scalar_one_or_none()
    if view is None:
        raise HTTPException(status_code=404, detail=f"Probe dataset {dataset_id} not found")
    return view


@router.delete("/datasets/{dataset_id}", dependencies=[Depends(lifecycle.refuse_agents)])
async def delete_probe_dataset(
    dataset_id: str,
    reason: Optional[str] = Query(None, max_length=2000),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Permanently delete ONE archived view — the same purge as `POST /datasets/purge`.

    ⚠ TIGHTENED 2026-10-21. This route used to delete any view nothing TRAINED on, archived or
    not, with no record: a view an evaluation or a calibration had used went silently (the
    evaluation by CASCADE, the calibration provenance by SET NULL). It now refuses a view that is
    not archived (409 `not_archived`) or that anything uses (409 `purge_blocked`, every blocker
    named), writes a tombstone, and is operator-only.
    """
    return await lifecycle.purge_one(
        db, dataset_id, lifecycle.OPERATOR, lifecycle.clean_reason(reason)
    )


# ── runs ──────────────────────────────────────────────────────────────────────


@router.post("/runs", status_code=202)
async def submit_probe_run(
    request: ProbeRunCreate, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Queue a probe run. 202 with the row's id and the Celery task id."""
    from ....models.model import Model
    from ....workers.probe_monitor_tasks import run_probe_monitor

    # BEFORE ANYTHING ELSE: an unknown card is a 400 naming the cards that exist, and
    # `can_split=False` because this job places on ONE card — a 202 for an "all" request
    # would queue work only the worker could refuse.
    gpu_request = resolve_gpu_request(request.gpu, can_split=False)

    model = (
        await db.execute(select(Model).where(Model.id == request.model_id))
    ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    refusal = requested_dtype_refusal(request.config.dtype, model)
    if refusal is not None:
        raise HTTPException(status_code=422, detail=refusal)

    train_view = (
        await db.execute(
            select(ProbeMonitorDataset).where(
                ProbeMonitorDataset.id == request.train_dataset_id
            )
        )
    ).scalar_one_or_none()
    if train_view is None:
        raise HTTPException(
            status_code=404, detail=f"Probe dataset {request.train_dataset_id} not found"
        )
    if train_view.role != "train":
        raise HTTPException(
            status_code=422,
            detail=(
                f"probe dataset {train_view.id} has role '{train_view.role}'; a run must "
                f"train on a 'train' view. The roles are not interchangeable — a "
                f"calibration view has no positives at all."
            ),
        )
    eval_views = []
    for dataset_id in request.eval_dataset_ids:
        view = (
            await db.execute(
                select(ProbeMonitorDataset).where(ProbeMonitorDataset.id == dataset_id)
            )
        ).scalar_one_or_none()
        if view is None:
            raise HTTPException(status_code=404, detail=f"Probe dataset {dataset_id} not found")
        if view.role == "calibration":
            raise HTTPException(
                status_code=422,
                detail=(
                    f"probe dataset {dataset_id} is a calibration view and cannot be an "
                    f"evaluation set: it is not labelled for the concept, so an AUROC "
                    f"over it would measure nothing"
                ),
            )
        eval_views.append(view)
    # Feature 041 FR-1: a run answers ONE question — its training view's sub-question (or the main one).
    try:
        refuse_sub_question_mismatch(train_view, eval_views)
        await refuse_sub_question_not_declared(db, train_view, request.behaviour)
    except DetectorRefused as refusal:
        raise refusal.to_http() from refusal

    await lifecycle.refuse_archived_inputs(
        db,
        [request.train_dataset_id, *request.eval_dataset_ids, request.calibration_dataset_id],
    )

    # 409: one active run per (model, training view). Two concurrent runs would write
    # to different artifact directories but contend for the same card and produce two
    # probes nobody asked to compare.
    active = (
        await db.execute(
            select(ProbeMonitorRun.id).where(
                ProbeMonitorRun.model_id == request.model_id,
                ProbeMonitorRun.train_dataset_id == request.train_dataset_id,
                ProbeMonitorRun.status.in_(("pending", "running")),
            )
        )
    ).scalars().first()
    if active:
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {active} is already {'' }active for this model and training view; "
                f"cancel it before starting another"
            ),
        )

    row = ProbeMonitorRun(
        model_id=request.model_id,
        train_dataset_id=request.train_dataset_id,
        eval_dataset_ids=list(request.eval_dataset_ids),
        calibration_dataset_id=request.calibration_dataset_id,
        config=request.config.model_dump(),
        behaviour=request.behaviour,
        behaviour_label=request.behaviour_label,
        # Feature 040 S2 (FR-13): from the TRAINING view, onto the run and then onto each probe.
        sub_question=train_view.sub_question,
        behaviour_version=train_view.behaviour_version,
        status="pending",
        gpu_request=gpu_request,
        environment={},
    )
    db.add(row)
    # ⚠ COMMITTED BEFORE DISPATCH, AND THE TASK REFUSES WITHOUT A ROW. That order is
    # what makes `missing_row="cancelled"` correct on the cancel scope. When a fix once
    # passed an id without creating the row, every UI extraction was silently refused
    # for a week.
    await db.commit()
    await db.refresh(row)

    try:
        task = gpu_delay(run_probe_monitor, gpu_request)(run_id=row.id)
    except Exception as exc:  # noqa: BLE001 - a broker outage must not leave a ghost row
        logger.exception("Could not dispatch probe run %s", row.id)
        row.status = "failed"
        row.error_message = f"Could not queue the run: {type(exc).__name__}: {exc}"[:2000]
        await db.commit()
        raise HTTPException(
            status_code=503,
            detail="Could not queue the run — the task broker is unavailable",
        )

    row.celery_task_id = task.id
    await db.commit()
    return {"id": row.id, "task_id": task.id, "status": "queued"}


def refuse_sub_question_mismatch(train_view: Any, eval_views: List[Any]) -> None:
    """Feature 041 FR-1: an evaluation view carrying a DIFFERENT sub-question from the training view is
    refused — its labels answer another question, so its AUROC says nothing about this probe.

    OQ-1 (a): a view with NO sub-question (the behaviour's main question) beside a specialist training
    view is allowed; its evaluation is stored and labelled `main`, and does not count towards the rung
    (`probe_monitor_rung.rung_eligibility_of`)."""
    for view in eval_views:
        if view.sub_question is not None and view.sub_question != train_view.sub_question:
            raise DetectorRefused(
                422, "sub_question_mismatch",
                f"evaluation view {view.id} answers sub-question {view.sub_question!r} and the training "
                f"view {train_view.id} answers "
                + (f"sub-question {train_view.sub_question!r}" if train_view.sub_question is not None
                   else "the main question")
                + "; a run is evaluated only on views of its own question (or the main question)",
                {"train_view": train_view.id, "train_sub_question": train_view.sub_question,
                 "view": view.id, "view_sub_question": view.sub_question},
            )


async def refuse_sub_question_not_declared(db: AsyncSession, train_view: Any, behaviour: Optional[str]) -> None:
    """Feature 041 FR-1: the training view's sub-question must be declared by the behaviour version it
    cites — for the behaviour the RUN names (a key declared by one behaviour means nothing to another).
    The version is the view's recorded `behaviour_version` when the run names the view's own behaviour,
    else that behaviour's current version."""
    from ....models.probe_behaviour import ProbeBehaviour, ProbeBehaviourVersion

    key = train_view.sub_question
    if key is None:
        return
    behaviour = behaviour or train_view.behaviour
    declared: List[str] = []
    number: Optional[int] = None
    if behaviour:
        head = (await db.execute(select(ProbeBehaviour).where(ProbeBehaviour.key == behaviour))).scalar_one_or_none()
        if head is not None:
            number = (train_view.behaviour_version if behaviour == train_view.behaviour
                      and train_view.behaviour_version is not None else head.current_version)
            version = (await db.execute(select(ProbeBehaviourVersion).where(
                ProbeBehaviourVersion.key == behaviour, ProbeBehaviourVersion.version == number,
            ))).scalar_one_or_none()
            declared = [s.get("key") for s in ((version.sub_questions if version is not None else None) or [])]
    if key not in declared:
        raise DetectorRefused(
            422, "sub_question_not_declared",
            f"the training view {train_view.id} answers sub-question {key!r}, which "
            + (f"version {number} of {behaviour!r} does not declare; it declares {declared}"
               if number is not None else f"no catalogued version of {behaviour!r} declares"),
            {"train_view": train_view.id, "sub_question": key, "behaviour": behaviour, "version": number,
             "declared": declared},
        )


@router.get("/runs", response_model=List[ProbeRunResponse])
async def list_probe_runs(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    # 035 XR-4 (FR-9, D3, D17): runs TRAINED on this probe dataset — the training set only, ANDed
    # with `status`, newest first, still capped at 200. miDataworks scanned the newest 200 runs and
    # missed every older one. It probes for this parameter NAME in the served OpenAPI
    # (`runs_train_filter`); pinned by `test_dataworks_provenance_names.py`.
    train_dataset_id: Optional[str] = Query(
        None, description="Only runs whose TRAINING set is this probe dataset id"
    ),
    db: AsyncSession = Depends(get_db),
) -> Any:
    query = select(ProbeMonitorRun).order_by(ProbeMonitorRun.created_at.desc()).limit(limit)
    if status:
        query = query.where(ProbeMonitorRun.status == status)
    if train_dataset_id:
        query = query.where(ProbeMonitorRun.train_dataset_id == train_dataset_id)
    return list((await db.execute(query)).scalars().all())


@router.get("/runs/{run_id}", response_model=ProbeRunResponse)
async def get_probe_run(run_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    row = (
        await db.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == run_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Probe monitor run {run_id} not found")
    # Feature 040 S2 (FR-21): both rankings, from the run's record of its selection.
    # FR-19 (FTASKS 9.7): the calibration size of the probe the run selected, None before selection.
    selected = (
        await db.execute(
            select(ProbeMonitor).where(ProbeMonitor.run_id == run_id, ProbeMonitor.selected.is_(True))
        )
    ).scalars().first()
    return ProbeRunResponse.model_validate(row).model_copy(
        update={
            "selection": (row.environment or {}).get("selection"),
            "calibration_size": await _calibration_size(db, selected) if selected is not None else None,
        }
    )


@router.post("/runs/{run_id}/cancel", status_code=202)
async def cancel_probe_run(run_id: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Request a COOPERATIVE stop. The worker notices at its next stage boundary.

    Not a `revoke(terminate=True)`: every worker here is `--pool=solo`, which has no
    pool child to signal and never reads the control queue while busy — so a
    terminating revoke returns cleanly and does nothing at all.
    """
    from ....core.cancellation import request_cancel

    row = (
        await db.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == run_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Probe monitor run {run_id} not found")
    if row.status not in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} is already {row.status}; there is nothing to cancel",
        )
    request_cancel("probe_monitor_run", run_id, celery_task_id=row.celery_task_id)
    return {"id": run_id, "status": "cancelling"}


@router.delete("/runs/{run_id}", status_code=204)
async def delete_probe_run(run_id: str, db: AsyncSession = Depends(get_db)) -> None:
    """Delete a run, its probes, and its artifacts.

    A RUNNING run is STOPPED first. Deleting the row while the worker continues leaves
    it computing over an unlinked artifact directory — which happened here on 2026-09-12
    and held a card for twenty minutes with a model download queued behind it.
    """
    import shutil

    from ....core.cancellation import request_cancel
    from ....core.config import settings

    row = (
        await db.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == run_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Probe monitor run {run_id} not found")
    if row.status in ("pending", "running"):
        request_cancel("probe_monitor_run", run_id, celery_task_id=row.celery_task_id)

    artifact_dir = row.artifact_dir
    await db.delete(row)
    await db.commit()

    if artifact_dir:
        try:
            target = settings.resolve_deletable_path(artifact_dir)
            if target.exists():
                shutil.rmtree(target)
        except ValueError as refused:
            # A stored path is not a trusted one; refusing to delete outside the
            # deletable roots must not fail the request that already removed the row.
            logger.error("Refusing to delete %s: %s", artifact_dir, refused)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", artifact_dir, exc)


# ── probes ────────────────────────────────────────────────────────────────────


def probe_summary(
    probe: Any, run_scope: Optional[str] = None, *, keep_curve: bool, file_missing: bool = False
) -> ProbeMonitorSummary:
    """THE ONE PLACE a probe row becomes a `ProbeMonitorSummary`, for the list AND the report.

    ⚠ TWO ROUTES, ONE SUMMARY (2026-10-08). The report used to build its summary with a bare
    `ProbeMonitorSummary.model_validate(probe)`, so every field this function DERIVES fell back to
    its schema default there: `length_band_count` read 0 on every probe (the list said 4),
    `window_thresholds` read `{}` — which the schema defines as "no per-window bar was placed" —
    `scope` read None and the rung's wording read "". Each was a confident wrong statement, not a
    blank. Both routes now call this, and the derived fields default to None ("not computed"), so a
    summary that skipped this function says so instead of saying zero.

    `keep_curve=False` is the list: `val_metrics["history"]` is one entry per epoch and the list is
    what the panel polls while a run is going, so it is stripped there. The report keeps it — it is
    the one place a reader who wants the curve is already looking.

    ⚠ IT ALSO FILLS THE RUNG'S WORDING, from the ladder module rather than from anything here.
    `RungChip` in the frontend deliberately holds no map from a rung number to a phrase, because
    miLLM mirrors these strings verbatim and a second copy is a second vocabulary that drifts —
    with the drifted one being what a person reads.
    """
    summary = ProbeMonitorSummary.model_validate(probe)
    decisions = getattr(probe, "window_decisions", None) or {}
    update: Dict[str, Any] = {
        "rung_language": probe_rung_language(summary.rung),
        "rung_next_step": probe_rung_next_step(summary.rung),
        # WHICH TOKENS — see the fields' note on `ProbeMonitorSummary`.
        "scope": run_scope,
        "window_thresholds": {
            window: (entry or {}).get("threshold") for window, entry in decisions.items()
        },
        "length_band_count": len(getattr(probe, "length_bands", None) or []),
    }
    # Feature 039 (FR-16): WHICH detector and decision the recorded definition states, and whether
    # that is the decision in force — from the one identity read, never assembled here.
    update.update(identity_fields(probe, file_missing=file_missing))
    metrics = summary.val_metrics or {}
    if not keep_curve and "history" in metrics:
        update["val_metrics"] = {k: v for k, v in metrics.items() if k != "history"}
    return summary.model_copy(update=update)


def summary_without_curve(
    probe: Any, run_scope: Optional[str] = None, *, file_missing: bool = False
) -> ProbeMonitorSummary:
    """The LIST's summary: `probe_summary` with the training curve stripped.

    A separate name so the list's choice can be asserted by walking the AST — and so the report,
    which must keep the curve, can be asserted NOT to call it.
    """
    return probe_summary(probe, run_scope, keep_curve=False, file_missing=file_missing)


async def run_scopes(db: AsyncSession, run_ids: Any) -> Dict[str, str]:
    """`{run_id: scope}` — the scope each run's probes were TRAINED under, for both routes.

    The same default `context_from_row` applies, so a summary states what the run used.
    """
    run_ids = set(run_ids)
    if not run_ids:
        return {}
    rows = (
        await db.execute(
            select(ProbeMonitorRun.id, ProbeMonitorRun.config).where(
                ProbeMonitorRun.id.in_(run_ids)
            )
        )
    ).all()
    return {run_id: str((config or {}).get("scope", "all")) for run_id, config in rows}


async def backfill_pending(db: AsyncSession, probes: List[Any]) -> Dict[str, str]:
    """Recompute the identity of definitions built before 039 among `probes`, once each (B4).

    The work runs in a threadpool with its own sync session (`backfill_identity`, a compare-and-set);
    the rows it wrote are then re-read here so the summaries state them. Returns the per-request
    `file_missing` flags (TI10).
    """
    pending = [probe for probe in probes if needs_backfill(probe)]
    if not pending:
        return {}
    flags = await run_in_threadpool(backfill_identity, [probe.id for probe in pending])
    for probe in pending:
        await db.refresh(probe)
    return flags


def digest_filter(*, detector: Optional[str], decision: Optional[str]) -> List[Any]:
    """The SQL for FR-18's filters: a digest matches the probe's current decision, any history entry
    (JSONB containment over a BOUND parameter), or the last built definition's identity (D-T8)."""
    from sqlalchemy import or_

    clauses = []
    build = ProbeMonitor.definition_build
    if detector:
        clauses.append(or_(
            ProbeMonitor.built_detector_digest == detector,
            build["resolved"]["detector_digest"].astext == detector,
            build["invalidated"]["was_identity"]["detector_digest"].astext == detector,
        ))
    if decision:
        clauses.append(or_(
            ProbeMonitor.decision_digest == decision,
            ProbeMonitor.calibration_history.contains([{"decision_digest": decision}]),
            build["resolved"]["decision_digest"].astext == decision,
            build["invalidated"]["was_identity"]["decision_digest"].astext == decision,
        ))
    return clauses


@router.get("/probes", response_model=List[ProbeMonitorSummary])
async def list_probes(
    run_id: Optional[str] = Query(None),
    selected_only: bool = Query(False),
    # ⚠ FEATURE 039 FR-18: THE JOIN FROM A HELD DIGEST BACK TO THE PROBE. A consumer holding an
    # event's digest — or a file's — finds the probe and the revisions that digest names. Validated
    # as 64 lowercase hex BEFORE any query (422 otherwise).
    detector_digest: Optional[str] = Query(None, pattern=HEX64),
    decision_digest: Optional[str] = Query(None, pattern=HEX64),
    db: AsyncSession = Depends(get_db),
) -> Any:
    query = select(ProbeMonitor).order_by(ProbeMonitor.created_at.desc())
    if run_id:
        query = query.where(ProbeMonitor.run_id == run_id)
    if selected_only:
        query = query.where(ProbeMonitor.selected.is_(True))
    clauses = digest_filter(detector=detector_digest, decision=decision_digest)
    if clauses:
        # ⚠ THE ONE-TIME RECOMPUTE RUNS BEFORE A DIGEST FILTER, NOT ONLY OVER WHAT IT RETURNS (review
        # R1). The filter reads digests from the database, and a pre-039 definition no unfiltered read
        # has touched has none there yet — so a lookup by its own digest answered "no probe".
        pending = query.where(ProbeMonitor.definition_path.is_not(None),
                              ProbeMonitor.built_detector_digest.is_(None))
        await backfill_pending(db, list((await db.execute(pending)).scalars().all()))
    for clause in clauses:
        query = query.where(clause)
    probes = list((await db.execute(query)).scalars().all())
    # Feature 039 (B4): a definition built before digests has its identity recomputed once, here.
    flags = await backfill_pending(db, probes)
    # ⚠ THE TRAINING CURVE IS STRIPPED FROM THE LIST, AND KEPT BY THE REPORT.
    #
    # `val_metrics["history"]` carries one entry per epoch — a few hundred for a normal
    # run — and this endpoint is what the panel polls while a run is going. Returning
    # every probe's full curve on every poll is hundreds of kilobytes for a payload whose
    # job is a table of one line per probe. `GET /probes/{id}` returns it in full, which
    # is where a reader who wants the curve is already looking.
    # The scope each probe was TRAINED under lives on its run; one query for the runs listed.
    scopes = await run_scopes(db, {probe.run_id for probe in probes})
    summaries = [
        summary_without_curve(
            probe, scopes.get(probe.run_id), file_missing=flags.get(probe.id) == "file_missing"
        )
        for probe in probes
    ]
    if detector_digest or decision_digest:
        summaries = [
            summary.model_copy(update={"matched": digest_match(
                probe, detector=detector_digest, decision=decision_digest)})
            for summary, probe in zip(summaries, probes)
        ]
    return summaries


def evaluation_eligibility(probe: Any, view: Any) -> Any:
    """Feature 041 T22: the rung rule's verdict on one evaluation, through the SAME pure predicate
    `recompute_rung` calls (`rung_eligibility_of`), so the label beside an evaluation and the rung
    computed from it cannot disagree."""
    from ....services.probe_monitor_rung import rung_eligibility_of

    return rung_eligibility_of(
        view.role if view is not None else None,
        view.sub_question if view is not None else None,
        probe.sub_question if probe is not None else None,
    )


@router.get("/probes/{probe_id}", response_model=ProbeReport)
async def get_probe_report(probe_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    """The assembled report: evaluations, the rung WITH ITS WORDING, and the SAE pair."""
    from ....schemas.evidence_ladder import probe_rung_language, probe_rung_next_step

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    # Feature 039 (B4): the same one-time recompute as the list.
    flags = await backfill_pending(db, [probe])

    evaluations = list(
        (
            await db.execute(
                select(ProbeMonitorEvaluation)
                .where(ProbeMonitorEvaluation.probe_id == probe_id)
                .order_by(ProbeMonitorEvaluation.created_at.asc())
            )
        ).scalars().all()
    )
    # The dense↔SAE pair: the same run, layer and rule, the other variant. Assembled
    # here so a reader can see what k-sparsity cost without doing the join themselves.
    paired = (
        await db.execute(
            select(ProbeMonitor.id).where(
                ProbeMonitor.run_id == probe.run_id,
                ProbeMonitor.layer == probe.layer,
                ProbeMonitor.rule == probe.rule,
                ProbeMonitor.variant != probe.variant,
            )
        )
    ).scalars().first()
    judge_runs = list(
        (
            await db.execute(
                select(ProbeMonitorJudgeRun).where(ProbeMonitorJudgeRun.probe_id == probe_id)
            )
        ).scalars().all()
    )

    # 033: the dictionary's HuggingFace home, so the export section can say WHY an SAE probe
    # cannot be exported rather than only disabling its button.
    sae_hf_repo = None
    if probe.variant == "sae" and probe.sae_id:
        from ....models.external_sae import ExternalSAE

        sae_row = (
            await db.execute(select(ExternalSAE).where(ExternalSAE.id == probe.sae_id))
        ).scalar_one_or_none()
        if sae_row is not None and sae_row.hf_repo_id and sae_row.hf_filepath:
            sae_hf_repo = sae_row.hf_repo_id

    from ....services.probe_monitor_metrics import threshold_transfer, validation_caveat

    view_ids = {e.dataset_id for e in evaluations}
    views = {
        v.id: v
        for v in (
            await db.execute(select(ProbeMonitorDataset).where(ProbeMonitorDataset.id.in_(view_ids)))
        ).scalars().all()
    } if view_ids else {}
    scopes = await run_scopes(db, {probe.run_id})
    return ProbeReport(
        # THE SAME FUNCTION AS THE LIST, curve kept — see `probe_summary`.
        probe=probe_summary(
            probe, scopes.get(probe.run_id), keep_curve=True,
            file_missing=flags.get(probe.id) == "file_missing",
        ),
        sae_hf_repo=sae_hf_repo,
        # ⚠ Assembled server-side so a reader cannot miss it. A probe reports ONE threshold and
        # one AUROC; read together they suggest one detector with one operating point. The
        # per-set spread is the thing that shows otherwise, and nobody was going to reconstruct
        # it from five ROC curves by hand.
        # ⚠ Beside the evaluations, so `val_auroc` is not read as one of them. It is the max
        # over the epochs and layers the SAME split selected; an in-distribution evaluation view
        # is the held-out number and is scored like any other set.
        validation_caveat=validation_caveat(probe.val_metrics),
        threshold_transfer=threshold_transfer(
            probe.threshold,
            probe.threshold_source,
            [{"metrics": e.metrics} for e in evaluations],
        ),
        rung_language=probe_rung_language(probe.rung),
        rung_next_step=probe_rung_next_step(probe.rung),
        evaluations=[
            _evaluation_response(e, evaluation_eligibility(probe, views.get(e.dataset_id)))
            for e in evaluations
        ],
        paired_probe_id=paired,
        judge_runs=await judge_run_entries(db, judge_runs),
        calibration_size=await _calibration_size(db, probe),
    )


def _evaluation_response(evaluation: Any, eligibility: Any) -> ProbeEvaluationResponse:
    return ProbeEvaluationResponse.model_validate(evaluation).model_copy(update={
        "label_question": eligibility.label_question,
        "counts_towards_rung": eligibility.counts,
    })


async def _calibration_size(db: AsyncSession, probe: Any) -> Dict[str, Any]:
    """FR-19 on the probe report: the floor is the behaviour's own (the version the probe cites, else
    the current one) when its definition sets one, else the runtime setting."""
    from ....core.runtime_settings import effective
    from ....services.calibration_size import calibration_size
    from .behaviour_detectors import _floor_override

    return calibration_size(probe, await _floor_override(db, probe), effective().calibration_floor_negatives)


@router.get("/probes/{probe_id}/decision", response_model=ProbeDecisionResponse)
async def get_probe_decision(probe_id: str) -> Any:
    """The decision in force, in contract form, with its digest and revision (039 FR-15, E1).

    What `millm_recalibrate_probe` reads before it pushes a bar to miLLM: the agent's values must
    equal THIS (OQ-1 (a)), and the digest sent is this one. A probe cut before 039 has its snapshot
    formed here, once (compare-and-set) — so this read is also where a pre-039 probe first gains a
    `decision_digest`.
    """
    from ....services.probe_decision import decision_response

    def _work() -> Optional[Dict[str, Any]]:
        from ....core.database import SyncSessionLocal

        session = SyncSessionLocal()
        try:
            return decision_response(session, probe_id)
        finally:
            session.close()

    out = await run_in_threadpool(_work)
    if out is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    return out


@router.post("/probes/{probe_id}/evaluate", status_code=202)
async def evaluate_probe_endpoint(
    probe_id: str,
    dataset_ids: List[str],
    sample_rows: Optional[int] = Query(None, ge=100, le=200_000, description=(
        "Score a seeded sample of this many rows of each view; the seed is the view and the size, so every "
        "probe evaluated at this size scores the same rows. Required for a calibration view.")),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Evaluate an existing probe on additional sets (this is how rung 2 is reached)."""
    from ....workers.probe_monitor_tasks import evaluate_probe_monitor

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    if not dataset_ids:
        raise HTTPException(status_code=422, detail="name at least one evaluation set")
    # ⚠ FEATURE 041 PT-11: A CALIBRATION VIEW WITHOUT A SAMPLE IS REFUSED. It is unlabelled for the concept
    # and can hold a million rows: evaluating it whole is a full GPU pass for nothing a rung reads.
    if sample_rows is None:
        await refuse_unsampled_calibration(db, probe, list(dataset_ids))
    await refuse_precision_mismatch(probe_id)
    await refuse_render_mismatch(probe_id)
    await lifecycle.refuse_archived_inputs(db, list(dataset_ids))
    task = gpu_delay(evaluate_probe_monitor, "auto")(
        probe_id=probe_id, dataset_ids=list(dataset_ids), sample_rows=sample_rows
    )
    return {"probe_id": probe_id, "task_id": task.id, "status": "queued"}


async def refuse_unsampled_calibration(db: AsyncSession, probe: Any, dataset_ids: List[str]) -> None:
    from ....models.probe_behaviour import ProbeBehaviourVersion, ProbeBehaviour
    from ....schemas.probe_monitor import DEFAULT_CALIBRATION_MAX_ROWS
    from ....services.detector_next import next_evaluate_sampled

    views = (await db.execute(select(ProbeMonitorDataset).where(ProbeMonitorDataset.id.in_(dataset_ids)))).scalars().all()
    for view in views:
        if view.role != "calibration":
            continue
        rows = DEFAULT_CALIBRATION_MAX_ROWS
        key = view.behaviour or probe.behaviour
        if key:
            floor = (await db.execute(
                select(ProbeBehaviourVersion.calibration_floor)
                .join(ProbeBehaviour, ProbeBehaviour.key == ProbeBehaviourVersion.key)
                .where(ProbeBehaviourVersion.key == key,
                       ProbeBehaviourVersion.version == ProbeBehaviour.current_version))).scalar_one_or_none()
            rows = int(floor) if floor else rows
        raise DetectorRefused(
            422, "calibration_view_needs_sample",
            f"view {view.id} is a calibration view; evaluate it on a seeded sample (sample_rows), never whole",
            {"view": view.id, "default_sample_rows": rows},
            next=[next_evaluate_sampled(probe.id, list(dataset_ids), rows)],
        ).to_http()


@router.get(
    "/probes/{probe_id}/evaluations/{evaluation_id}/scores",
    response_model=ProbeEvaluationScores,
)
async def get_evaluation_scores(
    probe_id: str,
    evaluation_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(10_000, ge=1, le=50_000),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """One evaluation's per-row scores, paged, in stored order (035 XR-2; FR-13 – FR-16).

    Order of checks: unknown evaluation or another probe's → 404; recorded before 035 (no saved
    positions) → 409 `positions_not_recorded`, naming the re-evaluate call — nothing is ever served
    from REPLAYED positions (D15); a missing, unreadable, misplaced or misaligned array → 500 naming
    the check, never an empty page (FR-16). A refused evaluation is served, with a null AUROC (D30).
    """
    from ....services.probe_evaluation_scores import (
        EvaluationArtifactError,
        load_evaluation_arrays,
        page,
    )

    evaluation = (
        await db.execute(
            select(ProbeMonitorEvaluation).where(ProbeMonitorEvaluation.id == evaluation_id)
        )
    ).scalar_one_or_none()
    if evaluation is None or evaluation.probe_id != probe_id:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation {evaluation_id} of probe {probe_id} not found",
        )
    if not evaluation.positions_path:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "positions_not_recorded",
                "message": (
                    f"evaluation {evaluation_id} was recorded before miStudio saved row positions, "
                    "so its scores cannot be joined to rows; re-evaluate the probe on this set to "
                    "get an evaluation that can be served"
                ),
                "evaluate": f"POST /api/v1/probe-monitors/probes/{probe_id}/evaluate",
            },
        )

    root = Path(settings.data_dir) / "probe_monitors"
    try:
        loaded = await run_in_threadpool(load_evaluation_arrays, evaluation, root=root)
        rows, next_offset = await run_in_threadpool(page, loaded, offset, limit)
    except EvaluationArtifactError as exc:
        # The full text (paths, the numpy error) goes to the log only: a 5xx body must not carry
        # internals to an unauthenticated caller (`test_no_raw_exception_text_on_5xx`). The
        # response names the evaluation and the failed CHECK, which is what a caller acts on.
        logger.error("scores of evaluation %s cannot be served: %s", evaluation_id, exc.message)
        check = exc.check
        raise HTTPException(
            status_code=500,
            detail={
                "code": "evaluation_artifacts_broken",
                "check": check,
                "evaluation_id": evaluation_id,
                "message": (
                    f"evaluation {evaluation_id}'s stored score arrays failed the {check!r} check "
                    "and cannot be served; see the miStudio log"
                ),
            },
        )

    view = (
        await db.execute(
            select(ProbeMonitorDataset).where(ProbeMonitorDataset.id == evaluation.dataset_id)
        )
    ).scalar_one_or_none()
    source = None
    if view is not None:
        source = (
            await db.execute(select(Dataset).where(Dataset.id == view.dataset_id))
        ).scalar_one_or_none()
    metrics = evaluation.metrics or {}
    reported = metrics.get("auroc") if evaluation.status != "refused" else None
    probe_row = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    eligibility = evaluation_eligibility(probe_row, view)
    return ProbeEvaluationScores(
        label_question=eligibility.label_question,
        counts_towards_rung=eligibility.counts,
        evaluation_id=evaluation.id,
        probe_id=probe_id,
        dataset_id=evaluation.dataset_id,
        source_dataset_id=str(view.dataset_id) if view is not None else "",
        revision=source.revision if source is not None else None,
        status=evaluation.status,
        reported_auroc=float(reported) if reported is not None else None,
        total=loaded.total,
        offset=offset,
        next_offset=next_offset,
        rows=rows,
    )


@router.post("/probes/{probe_id}/recalibrate")
async def recalibrate_probe_endpoint(
    probe_id: str,
    request: RecalibrateRequest,
    recut_windows: bool = Query(
        False,
        description=(
            "Re-score the per-window negatives on the GPU. Only needed for a probe whose "
            "per-window arrays were never persisted; returns 202 and a task id."
        ),
    ),
) -> Dict[str, Any]:
    """Move the operating point without a GPU, or preview the move. 200 — it is arithmetic.

    ⚠ NOT 202, AND NO `gpu_delay`. A threshold is the `(1 - target_fpr)` quantile of an array
    already on disk, so this is microseconds of work on a few thousand floats. Dispatching it
    as a GPU job would make the one part of the pipeline genuinely free to move cost a queue
    position behind an hour-long capture.

    ⚠ EVERY GATE LIVES IN THE SERVICE, NOT HERE. The MCP tool and the GPU task reach the same
    `propose`, and a refusal written at this layer is one the other two bypass — the reasoning
    `build_definition` already states about the evidence gate.
    """
    from ....services.probe_recalibration import RecalibrationRefused, recalibrate_probe

    if recut_windows:
        from ....workers.probe_monitor_tasks import recut_probe_windows

        await refuse_precision_mismatch(probe_id)
        await refuse_render_mismatch(probe_id)
        task = gpu_delay(recut_probe_windows, "auto")(
            probe_id=probe_id,
            target_fpr=request.target_fpr,
            reason=request.reason,
            # ⚠ X6: C3 holds on the GPU arm too. Without this the window re-cut committed over
            # whatever decision was in force when the job finally ran, minutes later.
            expect_decision_digest=request.expect_decision_digest,
        )
        return {"probe_id": probe_id, "task_id": task.id, "status": "queued"}

    def _work() -> Dict[str, Any]:
        from ....core.database import SyncSessionLocal

        session = SyncSessionLocal()
        try:
            return recalibrate_probe(
                session,
                probe_id,
                target_fpr=request.target_fpr,
                commit=not request.preview,
                allow_fire_on_nothing=request.allow_fire_on_nothing,
                reason=request.reason,
                expect_decision_digest=request.expect_decision_digest,
            )
        finally:
            session.close()

    try:
        return await run_in_threadpool(_work)
    except RecalibrationRefused as refusal:
        raise HTTPException(
            status_code=refusal.status,
            # 039 TI4: `extra` carries `decision_changed`'s `expected`, `current` and `next`.
            detail={**refusal.extra, "code": refusal.code, "message": refusal.detail},
        ) from refusal


@router.post("/probes/{probe_id}/score", status_code=202)
async def score_probe(
    probe_id: str, request: ScoreRequest, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Score one input offline (FR-12). 202 — there is no synchronous model load here."""
    from ....workers.probe_monitor_tasks import score_probe_monitor

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    task = gpu_delay(score_probe_monitor, "auto")(
        probe_id=probe_id, text=request.text, messages=request.messages
    )
    return {"probe_id": probe_id, "task_id": task.id, "status": "queued"}


@router.post("/probes/{probe_id}/score-batch", status_code=202)
async def score_probe_batch(probe_id: str, body: ScoreBatch, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Score several inputs offline (Feature 041 TI14, T19): one queued forward pass PER INPUT, the same task as
    `POST …/score`. ⚠ THE CAP IS ENFORCED HERE, from the runtime setting `probe_score_batch_cap` — it lived only
    in the MCP pod as a literal, so a REST caller was unbounded."""
    from ....core.runtime_settings import effective
    from ....workers.probe_monitor_tasks import score_probe_monitor

    cap = int(effective().probe_score_batch_cap)
    if len(body.inputs) > cap:
        raise HTTPException(status_code=422, detail={
            "code": "cap_exceeded", "message": f"{len(body.inputs)} inputs; probe_score_batch_cap allows {cap}",
            "setting": "probe_score_batch_cap", "value": cap, "inputs": len(body.inputs), "next": []})
    probe = (await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    tasks = []
    for index, item in enumerate(body.inputs):
        task = gpu_delay(score_probe_monitor, "auto")(probe_id=probe_id, text=item.text, messages=item.messages)
        tasks.append({"index": index, "task_id": task.id})
    return {"probe_id": probe_id, "tasks": tasks, "submitted": len(tasks)}


@router.get("/probes/{probe_id}/score/{task_id}")
async def get_score_result(
    probe_id: str, task_id: str, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Collect an offline score (FR-12). The second half of `POST …/score`'s contract.

    ⚠ WITHOUT THIS, SCORING WAS WRITE-ONLY. The POST returned a 202 and a task id and
    there was nothing to read it back with, so the whole feature could be exercised and
    never observed — the shape this repo keeps finding: implemented, tested by importing
    the piece directly, and unreachable for any caller.

    A FAILED task reports its REASON rather than an empty trace. An empty trace reads as
    "the probe scored nothing"; a failure reads as "the probe did not run", and those are
    different facts.

    PENDING is reported as-is and never as an error. Celery returns PENDING both for a
    task it has not started and for a task id it has never heard of, so this endpoint
    cannot tell "queued" from "unknown" — and claiming either would be a guess.
    """
    import asyncio

    from ....core.celery_app import celery_app

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")

    def _read():
        result = celery_app.AsyncResult(task_id)
        return result.state, result.info

    # `AsyncResult` touches the broker, which blocks — off the event loop.
    state, info = await asyncio.to_thread(_read)

    if state == "SUCCESS":
        # ⚠ FEATURE 041 PT-8 (pre-existing): this answered ANY task's result under ANY probe id, so a task id
        # read another probe's (or a family's) score. A finished result names its probe; another's is 404.
        if not isinstance(info, dict) or info.get("probe_id") != probe_id:
            raise HTTPException(status_code=404, detail={
                "code": "not_this_probes_task", "next": [],
                "message": f"task {task_id} is not a score of probe {probe_id}"})
        return {"task_id": task_id, "probe_id": probe_id, "status": state, "result": info}
    if state == "FAILURE":
        return {
            "task_id": task_id,
            "probe_id": probe_id,
            "status": state,
            "error": str(info),
            "result": None,
        }
    return {
        "task_id": task_id,
        "probe_id": probe_id,
        "status": state,
        "result": None,
    }


# ── judge runs ────────────────────────────────────────────────────────────────


@router.post("/judge-runs", status_code=202)
async def submit_judge_run(
    request: JudgeRunCreate, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Queue the LLM-monitor baseline. CPU — it is HTTP calls to a served model.

    The prompt is a LIBRARY VERSION (PADR IDL-57): `prompt_id`, or ``stakes-rating/v1`` when
    omitted. It is locked in the same commit that creates the run, and the run records its id, key
    and hash as columns. When a probe is named, the answer says whether this run can count toward
    its rung 3 — a judge of a different behaviour is allowed, and never raises the rung.
    """
    from ....services import judge_prompt_library as library
    from ....utils.url_validation import validate_llm_endpoint_url
    from ....workers.probe_monitor_tasks import run_probe_monitor_judge

    try:
        endpoint = validate_llm_endpoint_url(request.endpoint)
    except ValueError as bad:
        raise HTTPException(status_code=422, detail=str(bad))

    if request.probe_id:
        probe = (
            await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == request.probe_id))
        ).scalar_one_or_none()
        if probe is None:
            raise HTTPException(status_code=404, detail=f"Probe {request.probe_id} not found")

    await lifecycle.refuse_archived_inputs(db, list(request.dataset_ids))

    try:
        prompt = await library.resolve_for_run(db, request.prompt_id)
    except library.PromptRefused as refusal:
        raise HTTPException(status_code=refusal.status, detail=refusal.as_detail()) from refusal
    library.lock(prompt)

    row = ProbeMonitorJudgeRun(
        endpoint=endpoint,
        model=request.model,
        prompt_version=f"{prompt.name}/v{prompt.version}",
        prompt_id=prompt.id,
        prompt_hash=prompt.content_hash,
        dataset_ids=list(request.dataset_ids),
        probe_id=request.probe_id,
        status="pending",
        metrics={},
        # ⚠ PERSISTED, BECAUSE THEY WERE BEING DROPPED HERE. Both are declared and validated on
        # `JudgeRunCreate` and neither reached the row, the task or the service: a run submitted
        # with `max_rows_per_set: 200` judged all 5,737 rows of its five sets, 5.7x the work, and a
        # tightened parse-failure limit was silently the module default.
        max_rows_per_set=request.max_rows_per_set,
        parse_failure_limit=request.parse_failure_limit,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)

    task = run_probe_monitor_judge.delay(judge_run_id=row.id)
    row.celery_task_id = task.id
    await db.commit()
    view = (await library.run_rung_view(db, [row]))[row.id]
    return {
        "id": row.id,
        "task_id": task.id,
        "status": "queued",
        "prompt_id": prompt.id,
        "prompt_version": row.prompt_version,
        "prompt_hash": row.prompt_hash,
        "counts_toward_rung": view["counts_toward_rung"],
        "rung_note": view["rung_note"],
    }


@router.post("/judge-runs/{judge_run_id}/cancel", status_code=202)
async def cancel_judge_run(
    judge_run_id: str, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Stop a judge baseline.

    ⚠ THIS ROUTE WAS MISSING AND THE SCOPE WAS ALREADY REGISTERED — a lifecycle that
    could be started and never stopped, which for the judge means thousands of paid
    calls against a served model for a result nobody is waiting for.
    `test_cancel_registry_completeness` names exactly that: every registered scope needs
    an operator-facing route, or a recorded reason why it has none.
    """
    from ....core.cancellation import request_cancel

    row = (
        await db.execute(
            select(ProbeMonitorJudgeRun).where(ProbeMonitorJudgeRun.id == judge_run_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Judge run {judge_run_id} not found")
    if row.status not in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"judge run {judge_run_id} is already {row.status}",
        )
    request_cancel("probe_monitor_judge", judge_run_id, celery_task_id=row.celery_task_id)
    return {"id": judge_run_id, "status": "cancelling"}


@router.get("/judge-runs", response_model=List[Dict[str, Any]])
async def list_judge_runs(
    probe_id: Optional[str] = Query(None), db: AsyncSession = Depends(get_db)
) -> Any:
    query = select(ProbeMonitorJudgeRun).order_by(ProbeMonitorJudgeRun.created_at.desc())
    if probe_id:
        query = query.where(ProbeMonitorJudgeRun.probe_id == probe_id)
    rows = list((await db.execute(query)).scalars().all())
    return await judge_run_entries(db, rows)


async def judge_run_entries(db: AsyncSession, rows: List[Any]) -> List[Dict[str, Any]]:
    """ONE serialiser for a judge run, for the list and the probe report alike: the run, the
    library prompt it used, and whether it counts toward its probe's rung 3 (PADR IDL-57)."""
    from ....services.judge_prompt_library import run_rung_view

    views = await run_rung_view(db, rows)
    return [
        {
            "id": row.id,
            "endpoint": row.endpoint,
            "model": row.model,
            "prompt_version": row.prompt_version,
            "dataset_ids": row.dataset_ids,
            "probe_id": row.probe_id,
            "status": row.status,
            "progress": row.progress,
            # Rows judged / planned, the current set, rate, ETA, parse failures so far. None until
            # the run reports (`services/live_progress.JudgeProgress`).
            "progress_detail": row.progress_detail,
            "parse_failures": row.parse_failures,
            "max_rows_per_set": row.max_rows_per_set,
            "metrics": row.metrics,
            "error_message": row.error_message,
            "created_at": row.created_at,
            "completed_at": row.completed_at,
            **views[row.id],
        }
        for row in rows
    ]


# ── 033: the definition, its download, and publication ────────────────────────


class BuildDefinitionRequest(BaseModel):
    """What a build takes. `acknowledge_below_rung2` is the FR-5 gate's key."""

    model_config = ConfigDict(extra="forbid")

    acknowledge_below_rung2: Optional[Dict[str, Any]] = None
    vector_count: int = Field(16, ge=8, le=32)
    seed: int = 1337


class PublishDefinitionRequest(BaseModel):
    """⚠ THE TOKEN IS A REQUEST FIELD AND IS NEVER PERSISTED. It reaches the task as an argument
    and is written to no row, no task result and no log line. A token in `task_queue.args` would be
    readable by anyone who can list tasks."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str = Field(min_length=1, max_length=200)
    private: bool = True
    token: Optional[str] = None


@router.post("/probes/{probe_id}/definition", status_code=202)
async def build_definition(
    probe_id: str,
    request: BuildDefinitionRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Build and cache the probe's portable definition (033 FR-7). 202 — it is a GPU job.

    The evidence gate lives in the BUILDER, not here, and deliberately: a build dispatched from the
    MCP tool or re-run by a task must meet the same refusals, and a gate in the endpoint would be
    one the other callers bypass.
    """
    from ....workers.probe_monitor_tasks import build_probe_definition

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    await refuse_precision_mismatch(probe_id)
    await refuse_render_mismatch(probe_id)
    task = gpu_delay(build_probe_definition, "auto")(
        probe_id=probe_id,
        acknowledge_below_rung2=request.acknowledge_below_rung2,
        vector_count=request.vector_count,
        seed=request.seed,
    )
    return {"probe_id": probe_id, "task_id": task.id, "status": "queued"}


@router.get("/probes/{probe_id}/definition")
async def get_definition(probe_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    """The cached definition as JSON, or 409 with what to do about it."""
    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    if not probe.definition_path:
        invalidated = (probe.definition_build or {}).get("invalidated")
        detail = (
            f"probe {probe_id} has no built definition"
            + (
                # Feature 039 (FR-16): the machine `cause` beside the human reason, so an agent can
                # tell a re-cut (`decision`) from a rung change or a build race without parsing prose.
                f"; the previous one was invalidated ({invalidated.get('reason')}; cause: "
                f"{invalidated.get('cause') or 'not recorded'}) because what it stated changed — "
                f"build again"
                if invalidated
                else "; POST to this path to build one"
            )
        )
        raise HTTPException(status_code=409, detail=detail)
    import json
    from pathlib import Path

    path = settings.resolve_data_path(probe.definition_path)
    if not path.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"probe {probe_id} records a definition at {probe.definition_path} and the file is "
                f"gone; build again"
            ),
        )
    return json.loads(path.read_text())


@router.get("/probes/{probe_id}/export")
async def export_definition(probe_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    """Download it as `<slug>.probe.json` (033 FR-7)."""
    from fastapi.responses import FileResponse

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    if not probe.definition_path:
        raise HTTPException(
            status_code=409,
            detail=f"probe {probe_id} has no built definition; POST /probes/{probe_id}/definition",
        )
    path = settings.resolve_data_path(probe.definition_path)
    if not path.exists():
        raise HTTPException(status_code=409, detail="the definition file is gone; build again")
    return FileResponse(
        str(path),
        media_type="application/json",
        filename=f"{probe_id}.probe.json",
    )



def _huggingface_rejects(token: str) -> Optional[str]:
    """`None` when HuggingFace accepts the token; a reason to refuse with when it rejects it.

    ⚠ "A TOKEN IS PRESENT" AND "A TOKEN WORKS" ARE DIFFERENT CLAIMS, and this endpoint used to make
    the first while promising the second. It checked `resolve_hf_token` for a truthy value and
    queued the upload, so an expired token produced a 202, a task that created nothing, and a
    failure at the last step — the exact shape the docstring above says it exists to prevent.

    ⚠ FOUND THE HARD WAY DURING 033 ACCEPTANCE: the token stored in this installation's Settings is
    37 characters, starts with `hf_`, and HuggingFace answers `Invalid user token`. Nothing had ever
    said so, because a public dataset download needs no credential and the only paths that would
    have noticed — gated downloads and publishes — had never run. Two datasets recorded as "gated,
    not downloaded" in the project notes were very likely this and not their gates.

    An authentication failure is a refusal. Anything else — a timeout, DNS, a 5xx — is NOT: it does
    not prove the token is bad, and refusing on it would make the publish path fail closed against
    an outage it has no business having an opinion about.
    """
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).whoami()
        return None
    except Exception as exc:  # noqa: BLE001 - classified below, never swallowed silently
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403) or "Invalid user token" in str(exc):
            logger.warning("HuggingFace rejected the token for a publish request: %s", exc)
            return (
                "HuggingFace rejected this token (it is present but not valid — expired, revoked, "
                "or without write access). Replace it in Settings → API Keys or pass a working one "
                "in the request. Refused here rather than queueing an upload that would fail at "
                "its last step"
            )
        logger.warning(
            "could not verify the HuggingFace token (%s: %s) — proceeding, because this does not "
            "prove the token is bad",
            type(exc).__name__, exc,
        )
        return None

@router.post("/probes/{probe_id}/publish", status_code=202)
async def publish_definition(
    probe_id: str,
    request: PublishDefinitionRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Publish to HuggingFace (033 FR-8). 202.

    ⚠ THE TOKEN IS CHECKED BEFORE ANY WORK IS QUEUED — a 401 here rather than a task that starts,
    uploads nothing and fails at the end. `resolve_hf_token` is the resolver every HuggingFace path
    here shares, so a token stored in Settings works without being repeated in the request.

    ⚠ THAT SENTENCE USED TO READ "the same resolver the SAE upload uses" AND IT WAS FALSE.
    `HuggingFaceSAEService.upload_sae` took a REQUIRED `access_token` straight to `HfApi` and
    resolved nothing, so a stored write token worked for a probe definition and not for an SAE.
    Found while publishing an SAE for 033 acceptance 8.1. It is true now because the upload path
    was changed to match this one — a claim about a neighbour is worth nothing until the neighbour
    is checked.
    """
    from ....services.huggingface_sae_service import resolve_hf_token

    probe = (
        await db.execute(select(ProbeMonitor).where(ProbeMonitor.id == probe_id))
    ).scalar_one_or_none()
    if probe is None:
        raise HTTPException(status_code=404, detail=f"Probe {probe_id} not found")
    if not probe.definition_path:
        raise HTTPException(
            status_code=409,
            detail=f"probe {probe_id} has no built definition; build it before publishing",
        )
    token = resolve_hf_token(request.token)
    if not token:
        raise HTTPException(
            status_code=401,
            detail=(
                "no HuggingFace token: pass one in the request or store it in Settings → API "
                "Keys. Checked before the upload is queued, so this is a refusal rather than a "
                "task that starts and fails at its last step"
            ),
        )
    rejected = _huggingface_rejects(token)
    if rejected:
        raise HTTPException(status_code=401, detail=rejected)
    from ....workers.probe_monitor_tasks import publish_probe_definition

    task = publish_probe_definition.delay(
        probe_id=probe_id,
        repo_id=request.repo_id,
        private=request.private,
        token=token,
    )
    return {"probe_id": probe_id, "task_id": task.id, "status": "queued"}
