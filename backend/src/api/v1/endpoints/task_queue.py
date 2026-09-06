"""
Task Queue API endpoints.

This module provides endpoints for viewing and managing background task operations,
including failed tasks that can be manually retried.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete as sa_delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....services.app_setting_service import AppSettingService
from ....services.task_queue_service import TaskQueueService
from ....schemas.task_queue import (
    TaskQueueResponse,
    TaskQueueListResponse,
    TaskQueueRetryRequest,
    TaskQueueRetryResponse,
)
import asyncio

logger = logging.getLogger(__name__)

#: The `(task_type, entity_type)` pairs `retry_task` can actually dispatch.
#:
#: MIS-E2E-098. This exists so the endpoint can refuse BEFORE
#: `increment_retry_count` wipes the row's failure evidence and commits. Kept
#: in step with the if/elif chain by
#: `test_retry_does_not_destroy_evidence.py::test_every_dispatch_branch_is_listed`,
#: which reads the branches out of the AST — a hand-list that can drift from the
#: code it guards is worth very little.
RETRYABLE_TASK_TYPES: frozenset[tuple[str, str]] = frozenset({
    ("download", "model"),
    ("download", "dataset"),
    ("tokenization", "dataset"),
})


router = APIRouter()


def _serialize_task(task, entity_info, can_retry: bool = True) -> dict:
    """Serialize a TaskQueue ORM row for API responses."""
    return {
        "id": task.id,
        "task_id": task.task_id,
        "task_type": task.task_type,
        "entity_id": task.entity_id,
        "entity_type": task.entity_type,
        "status": task.status,
        "progress": task.progress,
        "error_message": task.error_message,
        "retry_params": task.retry_params,
        "retry_count": task.retry_count,
        "can_retry": can_retry,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        "entity_info": entity_info,
    }


def _federated_row(
    *,
    row_id: str,
    task_type: str,
    entity_id: str,
    entity_type: str,
    status: str,
    progress,
    name: str,
    details: Optional[str] = None,
    error_message: Optional[str] = None,
    created_at=None,
    started_at=None,
    completed_at=None,
    updated_at=None,
) -> dict:
    """Build a task-queue-shaped dict from another job table's row.

    Federated rows are read-only views: they can't be retried or deleted
    through the task-queue endpoints (can_retry=False signals the UI).

    PROGRESS IS 0-100 ON THIS BOUNDARY. The source tables disagree —
    ``trainings.progress`` and ``task_queue.progress`` are percentages while
    ``extraction_jobs.progress`` and ``labeling_jobs.progress`` are fractions —
    so each federator converts before calling this. Callers render the value
    directly as a percentage; a fraction leaking through shows a 98% job as
    "1.0%", which is what it did until 2026-07-26.
    """
    entity_info = {"name": name}
    if details:
        entity_info["details"] = details
    return {
        "id": row_id,
        "task_id": row_id,
        "task_type": task_type,
        "entity_id": entity_id,
        "entity_type": entity_type,
        "status": status,
        "progress": progress,
        "error_message": error_message,
        "retry_params": None,
        "retry_count": 0,
        "can_retry": False,
        "created_at": created_at.isoformat() if created_at else None,
        "started_at": started_at.isoformat() if started_at else None,
        "completed_at": completed_at.isoformat() if completed_at else None,
        "updated_at": updated_at.isoformat() if updated_at else None,
        "entity_info": entity_info,
    }


async def _load_dismissed(db: AsyncSession) -> set:
    """Return the set of (task_type, id) pairs the operator has cleared.

    Federated failures cannot be deleted through this API — they belong to
    other tables — so clearing one records a marker instead. The job row keeps
    its status and error message; only the Monitor listing hides it.
    """
    from ....models.dismissed_operation import DismissedOperation

    result = await db.execute(
        select(DismissedOperation.source_type, DismissedOperation.source_id)
    )
    return {(row.source_type, row.source_id) for row in result}


async def _fetch_federated(db: AsyncSession, statement, params: dict) -> list:
    """Execute a federated query inside a SAVEPOINT and return its rows.

    PostgreSQL aborts the ENTIRE transaction on a failed statement. Each
    federator swallows its own exception, which looks safe but is not: once one
    federated query errors, every federator that runs after it fails with
    ``InFailedSQLTransactionError`` and silently returns nothing, so the whole
    "Active Operations" panel goes blank because of one bad query.

    (That was not hypothetical — ``_federated_trainings`` selected a
    non-existent ``trainings.name`` column, which zeroed out every federator
    after it in both the test and production databases.)

    Running each query in a SAVEPOINT keeps a failure local to the federator
    that caused it; the outer transaction stays usable for the rest.
    """
    try:
        async with db.begin_nested():
            result = await db.execute(statement, params)
            return list(result)
    except Exception:
        logger.exception("Federated task query failed", exc_info=True)
        return []


async def _federated_trainings(db: AsyncSession, statuses: tuple, limit: int = 25) -> list:
    """Fetch trainings in the given statuses as task-queue-shaped rows."""
    from sqlalchemy import text, bindparam

    rows = []
    try:
        # NB: the trainings table has no `name` column — join models for a label.
        result = await _fetch_federated(
            db,
            text("""
                SELECT t.id, t.status, t.progress, t.current_step, t.total_steps,
                       t.error_message, t.created_at, t.started_at, t.completed_at,
                       t.updated_at, m.name AS model_name
                FROM trainings t
                LEFT JOIN models m ON m.id = t.model_id
                WHERE t.status IN :statuses
                ORDER BY t.created_at DESC
                LIMIT :limit
            """).bindparams(bindparam("statuses", expanding=True)),
            {"statuses": list(statuses), "limit": limit},
        )
        for row in result:
            rows.append(_federated_row(
                row_id=row.id,
                task_type="training",
                entity_id=row.id,
                entity_type="training",
                status="running" if row.status in ("running", "initializing") else (
                    "failed" if row.status == "failed" else "queued"
                ),
                progress=row.progress,
                name=f"SAE training ({row.model_name})" if row.model_name else f"Training {row.id}",
                details=f"Step {row.current_step:,}/{row.total_steps:,}" if row.total_steps else None,
                error_message=row.error_message,
                created_at=row.created_at,
                started_at=row.started_at,
                completed_at=row.completed_at,
                updated_at=row.updated_at,
            ))
    except Exception:
        logger.exception("Failed to query trainings for task federation")
    return rows


async def _federated_extractions(db: AsyncSession, statuses: tuple, limit: int = 25) -> list:
    """Fetch extraction jobs in the given statuses as task-queue-shaped rows."""
    from sqlalchemy import text, bindparam

    rows = []
    try:
        result = await _fetch_federated(
            db,
            text("""
                SELECT e.id, e.status, e.progress, e.layer_index, e.hook_type,
                       e.features_extracted, e.total_features, e.error_message,
                       e.created_at, e.completed_at, e.updated_at
                FROM extraction_jobs e
                WHERE e.status IN :statuses
                ORDER BY e.created_at DESC
                LIMIT :limit
            """).bindparams(bindparam("statuses", expanding=True)),
            {"statuses": list(statuses), "limit": limit},
        )
        for row in result:
            layer_part = f"layer {row.layer_index}" if row.layer_index is not None else "features"
            details = None
            if row.total_features:
                details = f"{row.features_extracted or 0}/{row.total_features} features"
            rows.append(_federated_row(
                row_id=row.id,
                task_type="extraction",
                entity_id=row.id,
                entity_type="extraction",
                status="running" if row.status == "extracting" else (
                    "failed" if row.status == "failed" else "queued"
                ),
                # extraction_jobs.progress is a FRACTION; this boundary is 0-100.
                progress=(row.progress or 0.0) * 100,
                name=f"Extraction ({layer_part}{'/' + row.hook_type if row.hook_type else ''})",
                details=details,
                error_message=row.error_message,
                created_at=row.created_at,
                completed_at=row.completed_at,
                updated_at=row.updated_at,
            ))
    except Exception:
        logger.exception("Failed to query extraction jobs for task federation")
    return rows


async def _federated_labeling(db: AsyncSession, statuses: tuple, limit: int = 25) -> list:
    """Fetch labeling jobs in the given statuses as task-queue-shaped rows."""
    from sqlalchemy import text, bindparam

    rows = []
    try:
        result = await _fetch_federated(
            db,
            text("""
                SELECT lj.id, lj.extraction_job_id, lj.labeling_method,
                       lj.openai_compatible_model, lj.openai_model, lj.local_model,
                       lj.status, lj.progress, lj.features_labeled, lj.total_features,
                       lj.error_message, lj.created_at, lj.updated_at
                FROM labeling_jobs lj
                WHERE lj.status IN :statuses
                ORDER BY lj.created_at DESC
                LIMIT :limit
            """).bindparams(bindparam("statuses", expanding=True)),
            {"statuses": list(statuses), "limit": limit},
        )
        for row in result:
            model_name = row.openai_compatible_model or row.openai_model or row.local_model or 'unknown'
            progress_pct = (row.features_labeled / row.total_features * 100) if row.total_features else 0
            rows.append(_federated_row(
                row_id=row.id,
                task_type="labeling",
                entity_id=row.extraction_job_id,
                entity_type="labeling",
                status="running" if row.status == "labeling" else (
                    "failed" if row.status == "failed" else "queued"
                ),
                progress=progress_pct,
                name=f"Labeling ({model_name})",
                details=f"{row.features_labeled}/{row.total_features} features",
                error_message=row.error_message,
                created_at=row.created_at,
                started_at=row.created_at,
                updated_at=row.updated_at,
            ))
    except Exception:
        logger.exception("Failed to query labeling jobs for task federation")
    return rows


async def _federated_pushes(db: AsyncSession, statuses: tuple, limit: int = 25) -> list:
    """Fetch Neuronpedia push jobs in the given statuses as task-queue-shaped rows."""
    from sqlalchemy import text, bindparam

    rows = []
    try:
        result = await _fetch_federated(
            db,
            text("""
                SELECT np.id, np.sae_id, np.status, np.progress,
                       np.features_pushed, np.total_features, np.error_message,
                       np.created_at, np.updated_at,
                       es.name as sae_name
                FROM neuronpedia_pushes np
                LEFT JOIN external_saes es ON es.id = np.sae_id
                WHERE np.status IN :statuses
                ORDER BY np.created_at DESC
                LIMIT :limit
            """).bindparams(bindparam("statuses", expanding=True)),
            {"statuses": list(statuses), "limit": limit},
        )
        for row in result:
            rows.append(_federated_row(
                row_id=row.id,
                task_type="neuronpedia_push",
                entity_id=row.sae_id,
                entity_type="neuronpedia",
                status="running" if row.status in ("pushing", "preparing") else (
                    "failed" if row.status == "failed" else "queued"
                ),
                progress=float(row.progress) if row.progress else 0,
                name="Push to Neuronpedia",
                details=row.sae_name or row.sae_id,
                error_message=row.error_message,
                created_at=row.created_at,
                started_at=row.created_at,
                updated_at=row.updated_at,
            ))
    except Exception:
        logger.exception("Failed to query neuronpedia pushes for task federation")
    return rows


async def _federated_activation_extractions(
    db: AsyncSession, statuses: tuple, limit: int = 25
) -> list:
    """Fetch model activation extractions as task-queue-shaped rows.

    DISTINCT FROM ``_federated_extractions``: that one federates ``extraction_jobs``
    (extracting *features from a trained SAE*). This one federates
    ``activation_extractions`` — the "Extract Activations" flow that pulls
    activations *out of a model* — which is a different table with its own status
    enum and had no federator at all, so a running extraction never appeared in
    /active.

    NOTE ON CASING: ``activation_extractions.status`` is the Postgres enum
    ``extractionstatus`` whose labels are the UPPERCASE Python enum NAMES
    ('EXTRACTING', 'QUEUED', ...), unlike ``extraction_status_enum`` used by
    ``extraction_jobs`` which is lowercase. Comparison is normalized with
    ``upper(...::text)``; callers pass uppercase names.
    """
    from sqlalchemy import text, bindparam

    rows = []
    try:
        result = await _fetch_federated(
            db,
            text("""
                SELECT e.id, e.status, e.progress, e.samples_processed, e.max_samples,
                       e.layer_indices, e.error_message, e.model_id,
                       e.created_at, e.completed_at, e.updated_at,
                       m.name AS model_name
                FROM activation_extractions e
                LEFT JOIN models m ON m.id = e.model_id
                WHERE upper(e.status::text) IN :statuses
                ORDER BY e.created_at DESC
                LIMIT :limit
            """).bindparams(bindparam("statuses", expanding=True)),
            {"statuses": [s.upper() for s in statuses], "limit": limit},
        )
        for row in result:
            status_label = str(row.status).upper()
            details = None
            if row.max_samples:
                details = f"{row.samples_processed or 0}/{row.max_samples} samples"
            layers = row.layer_indices or []
            layer_part = f" (layer{'s' if len(layers) > 1 else ''} {','.join(str(x) for x in layers)})" if layers else ""
            rows.append(_federated_row(
                row_id=row.id,
                task_type="extraction",
                entity_id=row.id,
                entity_type="extraction",
                status="queued" if status_label == "QUEUED" else (
                    "failed" if status_label == "FAILED" else "running"
                ),
                progress=float(row.progress) if row.progress else 0,
                name=f"Extracting activations{layer_part}",
                details=f"{row.model_name} — {details}" if row.model_name and details else (
                    row.model_name or details
                ),
                error_message=row.error_message,
                created_at=row.created_at,
                completed_at=row.completed_at,
                updated_at=row.updated_at,
            ))
    except Exception:
        logger.exception("Failed to query activation extractions for task federation")
    return rows


async def _federated_tokenizations(db: AsyncSession, statuses: tuple, limit: int = 25) -> list:
    """Fetch dataset tokenizations in the given statuses as task-queue-shaped rows.

    A running tokenization has no ``task_queue`` row — the worker only inserts one
    on failure — so without this federator an in-progress tokenization is invisible
    to /active even though the Datasets page shows it running.

    NOTE ON CASING: ``dataset_tokenizations.status`` is the Postgres enum
    ``tokenization_status_enum``, whose labels are UPPERCASE ('QUEUED',
    'PROCESSING', ...) because SQLAlchemy persists the Python enum *name*. This is
    the opposite of ``extraction_status_enum`` ('queued', 'extracting', ...), so
    comparing against lowercase strings here silently matches zero rows. The
    comparison is normalized with ``upper(...::text)`` to be immune to either
    convention; callers pass uppercase names.
    """
    from sqlalchemy import text, bindparam

    rows = []
    try:
        result = await _fetch_federated(
            db,
            text("""
                SELECT t.id, t.status, t.progress, t.max_length, t.error_message,
                       t.dataset_id, t.created_at, t.completed_at, t.updated_at,
                       d.name AS dataset_name
                FROM dataset_tokenizations t
                JOIN datasets d ON d.id = t.dataset_id
                WHERE upper(t.status::text) IN :statuses
                ORDER BY t.created_at DESC
                LIMIT :limit
            """).bindparams(bindparam("statuses", expanding=True)),
            {"statuses": [s.upper() for s in statuses], "limit": limit},
        )
        for row in result:
            status_label = str(row.status).upper()
            details = f"{row.max_length} tokens/seq" if row.max_length else None
            rows.append(_federated_row(
                row_id=row.id,
                task_type="tokenization",
                # Mirrors the failure-path task_queue row written by
                # dataset_tasks.tokenize_dataset_task so the UI renders both alike.
                entity_id=str(row.dataset_id),
                entity_type="dataset",
                status="running" if status_label == "PROCESSING" else (
                    "failed" if status_label == "ERROR" else "queued"
                ),
                progress=float(row.progress) if row.progress else 0,
                name=f"Tokenizing {row.dataset_name}" if row.dataset_name else "Tokenization",
                details=details,
                error_message=row.error_message,
                created_at=row.created_at,
                completed_at=row.completed_at,
                updated_at=row.updated_at,
            ))
    except Exception:
        logger.exception("Failed to query dataset tokenizations for task federation")
    return rows


@router.get("", response_model=TaskQueueListResponse)
async def list_tasks(
    status: Optional[str] = None,
    entity_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all task queue entries with optional filtering.

    Query Parameters:
        status: Filter by status (queued, running, failed, completed, cancelled)
        entity_type: Filter by entity type (model, dataset, training)

    Returns:
        List of task queue entries with entity information
    """
    tasks = await TaskQueueService.get_all_tasks(db, status=status, entity_type=entity_type)

    enriched_tasks = []
    for task in tasks:
        entity_info = await TaskQueueService.get_entity_info(
            db, task.entity_id, task.entity_type
        )
        enriched_tasks.append(_serialize_task(task, entity_info))

    return {"data": enriched_tasks}


@router.get("/failed", response_model=TaskQueueListResponse)
async def list_failed_tasks(db: AsyncSession = Depends(get_db)):
    """
    List all failed operations across job types.

    task_queue entries (model/dataset downloads, tokenizations) are retryable
    through this API. Failed trainings, extractions, labeling jobs, and
    Neuronpedia pushes are federated in read-only (can_retry=False) — they are
    managed from their own panels.

    Returns:
        List of failed tasks with entity information
    """
    tasks = await TaskQueueService.get_failed_tasks(db)

    enriched_tasks = []
    for task in tasks:
        entity_info = await TaskQueueService.get_entity_info(
            db, task.entity_id, task.entity_type
        )
        enriched_tasks.append(_serialize_task(task, entity_info, can_retry=True))

    enriched_tasks.extend(await _federated_trainings(db, ("failed",)))
    enriched_tasks.extend(await _federated_extractions(db, ("failed",)))
    enriched_tasks.extend(await _federated_labeling(db, ("failed",)))
    enriched_tasks.extend(await _federated_pushes(db, ("failed",)))
    # MIS-E2E-101: a failed model activation extraction appeared NOWHERE in the
    # Monitor. It was federated into /active but not here, and unlike the
    # tokenization path — whose exemption above is genuine, its worker really
    # does write a task_queue row on failure — `extract_activations` writes no
    # row at all. (The TaskQueue writes in `model_tasks.py` belong to
    # `download_and_load_model`, a different task in the same module.) So the
    # job vanished from the operator's view the moment it failed.
    #
    # Uppercase for the same reason as the /active call: this table's status is
    # the `extractionstatus` enum, whose labels are the Python enum NAMES.
    enriched_tasks.extend(
        await _federated_activation_extractions(db, ("FAILED",))
    )

    # Hide anything the operator has cleared. Filtered here rather than in each
    # federated query so a source added later is covered without being
    # remembered — forgetting the filter in one of six queries would silently
    # resurrect dismissed rows in that one category only.
    dismissed = await _load_dismissed(db)
    if dismissed:
        enriched_tasks = [
            t for t in enriched_tasks
            if (t["task_type"], t["id"]) not in dismissed
        ]

    # Newest failure first across all sources
    enriched_tasks.sort(
        key=lambda t: t.get("completed_at") or t.get("updated_at") or t.get("created_at") or "",
        reverse=True,
    )

    return {"data": enriched_tasks}


@router.post("/failed/{task_type}/{source_id}/dismiss")
async def dismiss_failed_operation(
    task_type: str,
    source_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Clear one federated failure from the Monitor.

    Federated rows live in other tables and cannot be deleted through this API,
    so this records a marker instead. The training / extraction / labeling job
    / push keeps its status and error message; only the listing hides it.

    Idempotent: dismissing an already-dismissed operation succeeds.

    Retryable task_queue rows are NOT dismissed here — they have a real DELETE
    (``/tasks/{id}``) which removes the queue row outright.
    """
    from ....models.dismissed_operation import DismissedOperation

    existing = await db.get(DismissedOperation, (task_type, source_id))
    if existing is None:
        db.add(DismissedOperation(source_type=task_type, source_id=source_id))
        await db.commit()

    return {"dismissed": True, "task_type": task_type, "source_id": source_id}


@router.post("/failed/dismiss-all")
async def dismiss_all_failed_operations(db: AsyncSession = Depends(get_db)):
    """Clear every federated failure currently listed.

    Deliberately dismisses only what /failed reports RIGHT NOW rather than
    issuing a blanket rule: a failure that appears after this call must still
    surface, or the Monitor would go quietly blind.
    """
    from ....models.dismissed_operation import DismissedOperation

    listing = await list_failed_tasks(db)

    dismissed = 0
    for task in listing["data"]:
        if task.get("can_retry") is not False:
            # Retryable queue rows are deletable; leave them to that path so a
            # "clear all" cannot orphan a row the user could still retry.
            continue
        key = (task["task_type"], task["id"])
        if await db.get(DismissedOperation, key) is None:
            db.add(DismissedOperation(source_type=key[0], source_id=key[1]))
            dismissed += 1

    if dismissed:
        await db.commit()

    return {"dismissed_count": dismissed}


@router.delete("/failed/{task_type}/{source_id}/dismiss")
async def undismiss_failed_operation(
    task_type: str,
    source_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Restore a dismissed operation to the Monitor.

    Dismissal must be reversible — it hides a real failure, and an operator who
    clears the wrong row should not have to touch the database.
    """
    from ....models.dismissed_operation import DismissedOperation

    await db.execute(
        sa_delete(DismissedOperation).where(
            DismissedOperation.source_type == task_type,
            DismissedOperation.source_id == source_id,
        )
    )
    await db.commit()
    return {"dismissed": False, "task_type": task_type, "source_id": source_id}


def _celery_view(task) -> Tuple[bool, Dict[str, Any]]:
    """`(orphaned, live_meta)` for a row, from ONE result-backend read.

    Only ever consults Celery for rows that CLAIM to be running: a queued task
    has not started and legitimately has no heartbeat, and calling it orphaned
    would condemn everything waiting behind a long job.

    THE META COMES BACK RATHER THAN BEING DISCARDED. This used to fetch an
    `AsyncResult`, ask it one question, and throw away `result.info` — which is
    the only place `prompts_seen`, `last_delta` and `stage` exist. The listing
    therefore had a percentage and nothing else, and a J-lens fit rendered as
    "jlens_fit 24.4%" against a raw model id. The read was already being paid
    for; only the answer was being thrown away.
    """
    if getattr(task, "status", None) != "running" or not getattr(task, "task_id", None):
        return False, {}
    try:
        from celery.result import AsyncResult

        from ....core.celery_app import celery_app
        from ....workers.task_heartbeat import (
            looks_abandoned,
            seconds_since_beat,
            seconds_since_row_update,
        )

        result = AsyncResult(task.task_id, app=celery_app)
        info = result.info if isinstance(result.info, dict) else {}
        # Covers the expired-result case too: a worker that died between
        # progress reports leaves Celery answering PENDING with no info, and
        # the heartbeat rule alone reads that as healthy.
        orphaned = looks_abandoned(
            result.state, result.info, seconds_since_row_update(task)
        )
        live = {k: v for k, v in info.items() if k != "heartbeat"}
        age = seconds_since_beat(info)
        if age is not None:
            live["seconds_since_heartbeat"] = round(age, 1)
        return orphaned, live
    except Exception:  # noqa: BLE001 - a listing must not fail on one bad row
        return False, {}


def _progress_details(live: Dict[str, Any]) -> Optional[str]:
    """The one-line subtitle, following the `_federated_row` convention.

    ABSENT FIELDS ARE OMITTED, never rendered as zero. A fit that has not yet
    reported reads as "no counts available"; rendering `0 / 1200` would say the
    fit had done nothing, which is a different and false claim.
    """
    seen = live.get("prompts_seen")
    total = live.get("total_prompts")
    parts = []
    if isinstance(seen, int) and isinstance(total, int) and total > 0:
        parts.append(f"{seen:,} / {total:,} prompts")
    elif isinstance(seen, int):
        parts.append(f"{seen:,} prompts")
    delta = live.get("last_delta")
    if isinstance(delta, (int, float)):
        threshold = live.get("convergence_delta")
        if isinstance(threshold, (int, float)):
            parts.append(f"delta {delta:.2e} (target {threshold:.0e})")
        else:
            parts.append(f"delta {delta:.2e}")
    stage = live.get("stage")
    if isinstance(stage, str) and stage and not parts:
        # A stage with no numbers is still worth saying: "validating" explains
        # a bar that has stopped moving at 53%.
        parts.append(stage)
    return " · ".join(parts) or None


@router.get("/active", response_model=TaskQueueListResponse)
async def list_active_tasks(db: AsyncSession = Depends(get_db)):
    """
    List all active (queued or running) operations across all job types.

    Queries the task_queue table plus trainings, extraction jobs, labeling
    jobs, Neuronpedia push jobs, and dataset tokenizations to provide a
    unified view of all background operations.

    Returns:
        List of active tasks with entity information
    """
    tasks = await TaskQueueService.get_active_tasks(db)

    # MIS-E2E-102: the Celery result-backend reads go OFF the event loop.
    #
    # `_celery_view` builds an `AsyncResult` and reads it — synchronous Redis
    # I/O, up to three round-trips per row — and it was being called directly
    # inside this coroutine. Synchronous I/O in an `async def` blocks the whole
    # event loop, and the Monitor page polls this endpoint continuously: every
    # poll stalled every other request in the process for the duration of N
    # sequential Redis round-trips.
    #
    # `to_thread` moves them to the default executor and `gather` overlaps
    # them, so the cost is one round-trip's latency rather than N, and the loop
    # stays free either way. Order is preserved: gather returns results
    # positionally, and the loop below zips them back against `tasks`.
    celery_views = await asyncio.gather(
        *(asyncio.to_thread(_celery_view, task) for task in tasks)
    ) if tasks else []

    enriched_tasks = []
    for task, (orphaned, live) in zip(tasks, celery_views):
        entity_info = await TaskQueueService.get_entity_info(
            db, task.entity_id, task.entity_type
        )
        row = _serialize_task(task, entity_info)
        # RECONCILE AGAINST THE HEARTBEAT, so this list cannot disagree with
        # /models/tasks/{id}. A row is written when the task is queued and moved
        # by the task itself; a worker that dies writes nothing, so the row sits
        # at whatever progress it had reached — forever. Observed exactly that:
        # a fit killed by a pod roll showed "running 21.5%" in Active Operations
        # while the task endpoint correctly reported ORPHANED.
        #
        # Reported, not rewritten. This endpoint reads; a background janitor is
        # the right place to make the state terminal, and inventing a write here
        # would mean a GET with side effects.
        if orphaned:
            row["status"] = "orphaned"
            row["error_message"] = (
                "the worker running this stopped reporting and is presumed gone "
                "(a deploy, an eviction or a crash). Re-run it."
            )
        if live:
            # Merged into entity_info because it is free-form (`Dict[str, Any]`)
            # and already carries `name`/`repo_id`; `TaskQueueData` is a closed
            # model and would silently DROP a new top-level key.
            merged = dict(row.get("entity_info") or {})
            merged.update(live)
            details = _progress_details(live)
            if details:
                merged["details"] = details
            row["entity_info"] = merged
        enriched_tasks.append(row)

    enriched_tasks.extend(
        await _federated_trainings(db, ("pending", "initializing", "running"))
    )
    enriched_tasks.extend(
        await _federated_extractions(db, ("queued", "extracting"))
    )
    enriched_tasks.extend(
        await _federated_labeling(db, ("queued", "labeling"))
    )
    enriched_tasks.extend(
        await _federated_pushes(db, ("queued", "pushing", "preparing"))
    )
    # Uppercase: tokenization_status_enum labels are the Python enum NAMES.
    # Failed tokenizations are intentionally excluded here — the worker writes a
    # real task_queue row on failure, which /failed already surfaces; federating
    # ERROR too would double-count them.
    enriched_tasks.extend(
        await _federated_tokenizations(db, ("QUEUED", "PROCESSING"))
    )
    # Model activation extractions (distinct from extraction_jobs above). The
    # in-flight set mirrors cleanup_stuck_activations.py, which is the authority
    # on which statuses mean "still running".
    enriched_tasks.extend(
        await _federated_activation_extractions(
            db, ("QUEUED", "LOADING", "EXTRACTING", "SAVING")
        )
    )

    # Newest first across all sources
    enriched_tasks.sort(key=lambda t: t.get("created_at") or "", reverse=True)

    return {"data": enriched_tasks}


@router.get("/{task_queue_id}", response_model=TaskQueueResponse)
async def get_task(
    task_queue_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific task queue entry by ID.

    Args:
        task_queue_id: Task queue entry ID

    Returns:
        Task queue entry with entity information

    Raises:
        HTTPException: If task not found
    """
    task = await TaskQueueService.get_task_by_id(db, task_queue_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task queue entry '{task_queue_id}' not found"
        )

    entity_info = await TaskQueueService.get_entity_info(
        db, task.entity_id, task.entity_type
    )

    return {
        "data": {
            "id": task.id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "entity_id": task.entity_id,
            "entity_type": task.entity_type,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "retry_params": task.retry_params,
            "retry_count": task.retry_count,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "entity_info": entity_info,
        }
    }


@router.post("/{task_queue_id}/retry", response_model=TaskQueueRetryResponse)
async def retry_task(
    task_queue_id: str,
    request: TaskQueueRetryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Retry a failed task with optional parameter overrides.

    This endpoint:
    1. Verifies the task exists and is in failed state
    2. Updates retry count and status
    3. Dispatches a new Celery task with retry parameters
    4. Returns the new task information

    Args:
        task_queue_id: Task queue entry ID
        request: Optional parameter overrides for retry

    Returns:
        Retry status and new task information

    Raises:
        HTTPException: If task not found or not in failed state
    """
    task = await TaskQueueService.get_task_by_id(db, task_queue_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task queue entry '{task_queue_id}' not found"
        )

    if task.status != "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is in '{task.status}' state, can only retry failed tasks"
        )

    # Merge retry_params with any overrides from request
    retry_params = task.retry_params.copy() if task.retry_params else {}
    if request.param_overrides:
        retry_params.update(request.param_overrides)

    # Fetch HF token from encrypted app_settings — never stored in retry_params
    hf_token = await AppSettingService.get_decrypted_value(db, "hf_token")

    # RESOLVE THE BRANCH BEFORE TOUCHING THE ROW (MIS-E2E-098).
    #
    # `increment_retry_count` is not a counter increment. It sets
    # status="queued", error_message=None, progress=0.0, completed_at=None,
    # started_at=None — and then COMMITS. It used to run before the dispatch
    # chain below.
    #
    # So for any `(task_type, entity_type)` pair with no branch, the row was
    # committed as queued with its error message DESTROYED, nothing was
    # dispatched, and the caller got a 400. That row is then permanently
    # stranded: it no longer appears under /failed because its status is not
    # failed, nothing will ever process it because nothing was queued, and
    # pressing Retry again just repeats the sequence. The one artifact saying
    # WHY the job failed is gone on the first click, irrecoverably — and that
    # click also returned an error, so the user has no reason to think anything
    # was written at all.
    #
    # Refusing first costs nothing and leaves the row exactly as it was.
    if (task.task_type, task.entity_type) not in RETRYABLE_TASK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported task type '{task.task_type}' for entity type "
                f"'{task.entity_type}'"
            ),
        )

    # Increment retry count
    await TaskQueueService.increment_retry_count(db, task_queue_id)

    # Dispatch appropriate Celery task based on task type
    if task.task_type == "download" and task.entity_type == "model":
        from ....workers.model_tasks import download_and_load_model

        celery_task = download_and_load_model.delay(
            model_id=task.entity_id,
            repo_id=retry_params.get("repo_id"),
            quantization=retry_params.get("quantization"),
            access_token=hf_token,
            trust_remote_code=retry_params.get("trust_remote_code", False),
        )

        # Update task_queue entry with new Celery task ID
        task.task_id = celery_task.id
        await db.commit()

        logger.info(f"Retried model download task {task_queue_id}, new Celery task: {celery_task.id}")

    elif task.task_type == "download" and task.entity_type == "dataset":
        from ....workers.dataset_tasks import download_dataset_task

        celery_task = download_dataset_task.delay(
            dataset_id=task.entity_id,
            repo_id=retry_params.get("repo_id"),
            access_token=hf_token,
            split=retry_params.get("split"),
            config=retry_params.get("config"),
        )

        task.task_id = celery_task.id
        await db.commit()

        logger.info(f"Retried dataset download task {task_queue_id}, new Celery task: {celery_task.id}")

    elif task.task_type == "tokenization" and task.entity_type == "dataset":
        from ....workers.dataset_tasks import tokenize_dataset_task

        celery_task = tokenize_dataset_task.delay(
            dataset_id=task.entity_id,
            tokenizer_name=retry_params.get("tokenizer_name"),
            max_length=retry_params.get("max_length", 512),
            stride=retry_params.get("stride", 0),
            padding=retry_params.get("padding", "max_length"),
            truncation=retry_params.get("truncation", "longest_first"),
            add_special_tokens=retry_params.get("add_special_tokens", True),
            text_column=retry_params.get("text_column"),
            enable_cleaning=retry_params.get("enable_cleaning", True),
        )

        task.task_id = celery_task.id
        await db.commit()

        logger.info(f"Retried tokenization task {task_queue_id}, new Celery task: {celery_task.id}")

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task type '{task.task_type}' for entity type '{task.entity_type}'"
        )

    return {
        "success": True,
        "message": f"Task retry initiated (attempt {task.retry_count + 1})",
        "task_queue_id": task_queue_id,
        "celery_task_id": task.task_id,
        "retry_count": task.retry_count,
    }


@router.delete("/{task_queue_id}", status_code=204)
async def delete_task(
    task_queue_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a task queue entry.

    This removes the task from the queue. For active tasks, you should
    cancel them first using the cancel endpoint.

    Args:
        task_queue_id: Task queue entry ID

    Raises:
        HTTPException: If task not found
    """
    success = await TaskQueueService.delete_task(db, task_queue_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Task queue entry '{task_queue_id}' not found"
        )

    logger.info(f"Deleted task queue entry {task_queue_id}")
