"""Every miStudio job that is queued or running, of every kind, in one read (MCP gap G9).

WHY THIS EXISTS. Before a GPU run an agent had to ask six tools over five categories whether
anything else was running (probe runs pending, probe runs running, judge runs, extractions,
trainings, the task queue) — and still missed captures, discovery passes, evaluations, the
steering workers and anything a later feature adds. `/task-queue/active` federates six job
tables for the Monitor page and stops there.

THE LIST OF SOURCES IS THE ONE PLACE A JOB KIND IS NAMED. `tests/unit/test_active_jobs.py` reads
the ORM metadata (the authority, not this file) and fails when a table that carries a job
status — a `status`/`*_status` column beside a task id or a GPU column — is neither a source
here nor in `NOT_JOB_TABLES` with a written reason. A new job table cannot be added silently.

READ ERRORS ARE REPORTED, NEVER SWALLOWED. The Monitor page's federation catches every exception
and returns `[]`, so a query that broke reads as "nothing is running" — exactly the answer that
lets an agent start a GPU job on a busy card. Here a failed source is named in `errors` and the
rest of the answer still arrives.

Statuses are compared case-insensitively: some columns store an enum's NAME (`QUEUED`), others
its value (`queued`), and both mean the same thing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobSource:
    """One kind of job: the table it lives in and the statuses that mean "not finished"."""

    kind: str
    table: str
    active: tuple[str, ...]
    status_column: str = "status"
    task_column: Optional[str] = None
    #: A human label for the kind, shown to the agent.
    description: str = ""


#: Every job kind, one row each. `task_queue` is last: it is the generic queue (downloads,
#: J-lens fits, steering requests, finalisation, pruning, grouping) and reports its `task_type`.
JOB_SOURCES: tuple[JobSource, ...] = (
    JobSource("sae_training", "trainings", ("pending", "initializing", "running"),
              task_column="celery_task_id", description="SAE training"),
    JobSource("feature_extraction", "extraction_jobs", ("queued", "extracting"),
              task_column="celery_task_id", description="SAE feature extraction"),
    JobSource("feature_nlp_analysis", "extraction_jobs", ("pending", "processing"),
              status_column="nlp_status", description="NLP analysis of an extraction"),
    JobSource("activation_extraction", "activation_extractions",
              ("queued", "loading", "extracting", "saving"), task_column="celery_task_id",
              description="model activation extraction"),
    JobSource("labeling", "labeling_jobs", ("queued", "labeling"), task_column="celery_task_id",
              description="feature labeling"),
    JobSource("enhanced_labeling", "enhanced_labeling_jobs", ("queued", "running"),
              task_column="celery_task_id", description="enhanced labeling"),
    JobSource("labeling_trial", "labeling_trial_runs", ("queued", "running"),
              description="labeling trial run"),
    JobSource("labeling_resume_sweep", "labeling_resume_sweeps", ("running",),
              description="labeling resume sweep"),
    JobSource("tokenization", "dataset_tokenizations", ("queued", "processing"),
              task_column="celery_task_id", description="dataset tokenization"),
    JobSource("dataset", "datasets", ("downloading", "processing"),
              description="dataset download or processing"),
    JobSource("model", "models", ("downloading", "loading", "quantizing"),
              task_column="celery_task_id", description="model download, load or quantization"),
    JobSource("external_sae", "external_saes", ("pending", "downloading", "converting"),
              description="SAE download or conversion"),
    JobSource("neuronpedia_push", "neuronpedia_pushes", ("queued", "preparing", "pushing"),
              description="Neuronpedia push"),
    JobSource("neuronpedia_export", "neuronpedia_export_jobs",
              ("pending", "computing", "packaging"), description="Neuronpedia export"),
    JobSource("feature_grouping", "feature_grouping_runs", ("pending", "computing"),
              description="feature grouping precompute"),
    JobSource("probe_training", "probe_monitor_runs", ("pending", "running"),
              task_column="celery_task_id", description="probe monitor training run"),
    JobSource("probe_evaluation", "probe_monitor_evaluations", ("pending", "running"),
              description="probe monitor evaluation"),
    JobSource("probe_judge", "probe_monitor_judge_runs", ("pending", "running"),
              task_column="celery_task_id",
              description="probe judge baseline (calls an LLM endpoint; no miStudio GPU)"),
    # Feature 038: a session spot-check (reads miLLM, calls a judge endpoint; no miStudio GPU).
    JobSource("probe_spot_check", "probe_spot_checks", ("pending", "running"),
              task_column="celery_task_id",
              description="behaviour spot-check of recorded miLLM turns (calls a judge endpoint; no miStudio GPU)"),
    # Feature 041: a blend report (CPU only, on the `blend` queue) — counted against blend_max_concurrent.
    JobSource("probe_blend_report", "probe_blend_reports", ("queued", "running"),
              task_column="celery_task_id",
              description="probe blend evaluation (CPU only; reads stored evaluations, no GPU)"),
    JobSource("circuit_capture", "circuit_capture_runs", ("pending", "running"),
              task_column="celery_task_id", description="circuit activation capture"),
    JobSource("circuit_discovery", "circuit_discovery_runs", ("pending", "running"),
              task_column="celery_task_id", description="circuit discovery"),
    JobSource("circuit_attribution", "circuit_discovery_runs", ("pending", "running"),
              status_column="attribution_status", task_column="attribution_task_id",
              description="circuit attribution pass"),
    JobSource("circuit_edge_validation", "circuit_discovery_runs", ("pending", "running"),
              status_column="validation_status", task_column="validation_task_id",
              description="circuit edge validation"),
    JobSource("circuit_faithfulness", "circuits", ("pending", "running"),
              status_column="faithfulness_status", task_column="faithfulness_task_id",
              description="circuit faithfulness"),
    JobSource("circuit_calibration", "circuits", ("pending", "running"),
              status_column="calibration_status", task_column="calibration_task_id",
              description="circuit strength calibration"),
    JobSource("steering_record", "steering_record_runs", ("pending", "running"),
              task_column="task_id", description="steering sample recording"),
    JobSource("task_queue", "task_queue", ("queued", "running"), task_column="task_id",
              description="the generic task queue (downloads, J-lens, steering requests, "
                          "finalisation, pruning, grouping); `task_type` says which"),
)

#: Tables with a status column that are NOT jobs, each with the reason. The completeness test
#: refuses a status-bearing table that is in neither list.
NOT_JOB_TABLES: dict[str, str] = {
    "agent_approval_requests": "an approval request, not work: read it with get_approval_status",
    "features": "`label_status` is one feature's label state; the labeling job is the work",
}

#: The pseudo-kind for live steering card workers (a process per card that holds a model in
#: VRAM between requests). Not a table: read from the PID files the backend keeps.
STEERING_WORKER_KIND = "steering_worker"


def known_kinds() -> list[str]:
    return sorted({s.kind for s in JOB_SOURCES} | {STEERING_WORKER_KIND})


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _table(name: str):
    from ..core.database import Base
    import src.models  # noqa: F401 - registers every table on the metadata

    return Base.metadata.tables[name]


async def _read_source(db: AsyncSession, source: JobSource) -> list[dict[str, Any]]:
    table = _table(source.table)
    cols = table.c
    status = cols[source.status_column]
    wanted = [
        cols[name] for name in (
            "id", "progress", "gpu_uuid", "gpu_uuids", "created_at", "started_at",
            "model_id", "name", "task_type", "entity_id", "entity_type",
        ) if name in cols
    ]
    if source.task_column:
        wanted.append(cols[source.task_column].label("task_id"))
    statement = (
        select(*wanted, status.label("job_status"))
        .where(func.lower(cast(status, String)).in_([s.lower() for s in source.active]))
        .order_by(cols["created_at"].desc())
        .limit(200)
    )
    rows = (await db.execute(statement)).mappings().all()
    out = []
    for row in rows:
        uuids = row.get("gpu_uuids")
        primary = row.get("gpu_uuid")
        job = {
            "kind": source.kind,
            "id": row.get("id"),
            # An Enum column hands back the member: its VALUE is the status, `str()` would be
            # "ModelStatus.DOWNLOADING".
            "status": str(getattr(row["job_status"], "value", row["job_status"])).lower(),
            "progress": row.get("progress"),
            "gpu_uuid": primary,
            "gpu_uuids": list(uuids) if isinstance(uuids, (list, tuple)) else (
                [primary] if primary else []),
            "task_id": row.get("task_id"),
            "created_at": _iso(row.get("created_at")),
            "source_table": source.table,
        }
        for extra in ("model_id", "name", "task_type", "entity_id", "entity_type", "started_at"):
            if extra in row and row.get(extra) is not None:
                job[extra] = _iso(row[extra]) if extra == "started_at" else row[extra]
        out.append(job)
    return out


def steering_worker_jobs(cards: Iterable[Any]) -> list[dict[str, Any]]:
    """A row per card whose steering worker process is alive (it holds VRAM while it lives)."""
    from ..api.v1.steering_workers import card_worker_status

    out = []
    for worker in card_worker_status(cards):
        if not worker.get("worker_pid"):
            continue
        out.append({
            "kind": STEERING_WORKER_KIND,
            "id": f"steering_worker:{worker['card']}",
            "status": "running" if worker.get("busy") else "idle",
            "progress": None,
            "gpu_uuid": worker["card"],
            "gpu_uuids": [worker["card"]],
            "task_id": worker.get("task_id"),
            "pid": worker["worker_pid"],
            "created_at": None,
            "source_table": None,
        })
    return out


async def list_active_jobs(
    db: AsyncSession,
    *,
    kind: Optional[str] = None,
    cards: Optional[Iterable[Any]] = None,
) -> dict[str, Any]:
    """`{jobs, count, kinds, errors, read_at}` — every unfinished job, newest first.

    `kind` narrows to one kind (an unknown kind raises `ValueError` naming the known ones, so a
    typo is never answered "nothing running"). `cards` (GpuCard-like, with `.uuid`) enables the
    steering-worker rows; None reads the live inventory.
    """
    kinds = known_kinds()
    if kind is not None and kind not in kinds:
        raise ValueError(f"{kind!r} is not a job kind. Kinds are {kinds}.")
    jobs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for source in JOB_SOURCES:
        if kind is not None and source.kind != kind:
            continue
        try:
            jobs.extend(await _read_source(db, source))
        except Exception as exc:  # noqa: BLE001 - reported per source, never swallowed
            await db.rollback()
            logger.exception("active jobs: reading %s failed", source.kind)
            errors.append({"kind": source.kind, "error": f"{type(exc).__name__}: {exc}"})
    if kind in (None, STEERING_WORKER_KIND):
        try:
            if cards is None:
                from .gpu_placement import list_cards

                cards = list_cards()
            jobs.extend(steering_worker_jobs(cards))
        except Exception as exc:  # noqa: BLE001
            logger.exception("active jobs: reading steering workers failed")
            errors.append({"kind": STEERING_WORKER_KIND, "error": f"{type(exc).__name__}: {exc}"})
    jobs.sort(key=lambda j: j.get("created_at") or "", reverse=True)
    return {
        "jobs": jobs,
        "count": len(jobs),
        "kinds": kinds,
        "errors": errors,
        "read_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Every miStudio job that is queued or running, of every kind. The same work can "
            "appear twice (a queue entry and its own run row); a row's gpu_uuid is the card the "
            "job recorded, null when the kind records none or it has not been placed yet. "
            "`errors` names any source that could not be read: an error there means the list "
            "may be incomplete, never that nothing is running."
        ),
    }


def jobs_on_card(jobs: Iterable[dict[str, Any]], gpu_uuid: str) -> list[dict[str, Any]]:
    """The jobs that recorded this card (compared by normalised UUID)."""
    from .gpu_placement import normalise_uuid

    target = normalise_uuid(gpu_uuid)
    return [
        job for job in jobs
        if any(normalise_uuid(u) == target for u in (job.get("gpu_uuids") or []) if u)
    ]
