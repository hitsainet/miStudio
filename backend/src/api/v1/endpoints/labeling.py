"""
Feature labeling API endpoints.

Provides REST API for independent semantic labeling of extracted SAE features.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import List, Optional
import httpx
from pydantic import BaseModel, ConfigDict, Field
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select
from src.core.config import settings
from src.core.deps import get_db
from src.services.app_setting_service import AppSettingService
from src.services.labeling_service import LabelingService
from src.services import labeling_eligibility
from src.workers.labeling_tasks import label_features_task, resume_sweep_step
from src.schemas.labeling import (
    LabelingConfigRequest,
    LabelingStatusResponse,
    LabelingListResponse
)
from src.models.extraction_job import ExtractionJob
from src.models.labeling_job import LabelingJob
from src.models.feature import Feature
from src.models.training import Training
from src.models.external_sae import ExternalSAE
from src.models.model import Model

logger = logging.getLogger(__name__)

router = APIRouter()


async def _enrich_labeling_responses(
    db: AsyncSession,
    responses: list[LabelingStatusResponse]
) -> list[LabelingStatusResponse]:
    """Enrich labeling responses with extraction context (model, layer, hook, SAE name)."""
    if not responses:
        return responses

    # Batch-load extraction jobs
    ext_ids = {r.extraction_job_id for r in responses}
    result = await db.execute(
        select(ExtractionJob).where(ExtractionJob.id.in_(ext_ids))
    )
    ext_map = {ej.id: ej for ej in result.scalars().all()}

    # Collect training/SAE IDs for batch loading
    training_ids = {ej.training_id for ej in ext_map.values() if ej.training_id}
    sae_ids = {ej.external_sae_id for ej in ext_map.values() if ej.external_sae_id}

    # Batch-load trainings
    trainings_map = {}
    model_ids = set()
    if training_ids:
        result = await db.execute(select(Training).where(Training.id.in_(training_ids)))
        for t in result.scalars().all():
            trainings_map[t.id] = t
            if t.model_id:
                model_ids.add(t.model_id)

    # Batch-load external SAEs
    saes_map = {}
    if sae_ids:
        result = await db.execute(select(ExternalSAE).where(ExternalSAE.id.in_(sae_ids)))
        for s in result.scalars().all():
            saes_map[s.id] = s
            if hasattr(s, 'model_id') and s.model_id:
                model_ids.add(s.model_id)

    # Batch-load models
    models_map = {}
    if model_ids:
        result = await db.execute(
            select(Model.id, Model.name).where(Model.id.in_(model_ids))
        )
        for row in result.all():
            models_map[row[0]] = row[1]

    # Enrich each response
    for resp in responses:
        ej = ext_map.get(resp.extraction_job_id)
        if not ej:
            continue

        resp.layer_index = ej.layer_index
        resp.hook_type = ej.hook_type

        if ej.training_id and ej.training_id in trainings_map:
            training = trainings_map[ej.training_id]
            resp.model_name = models_map.get(training.model_id, training.model_id)
        elif ej.external_sae_id and ej.external_sae_id in saes_map:
            sae = saes_map[ej.external_sae_id]
            resp.sae_name = getattr(sae, 'name', None) or sae.id
            if hasattr(sae, 'model_id') and sae.model_id:
                resp.model_name = models_map.get(sae.model_id, sae.model_id)

    return responses


class LabelingPanelRequest(LabelingConfigRequest):
    """Apply-mode labeling scoped to an explicit panel of features.

    Separate from LabelingConfigRequest, and `extra="forbid"`, for the same
    reason LabelingTrialRequest is: the parent permits unknown keys, so a
    typo'd `featureIds` there would be dropped in silence and the request would
    execute as a full-extraction run over every feature — 30,712 of them on the
    L46 extraction, roughly five days of serving time.

    Unlike a trial, this DOES write labels onto the feature rows.

    max_length is 2000 rather than the trial path's 200 because Celery kills a
    task at task_time_limit=43200s (12 h) with a soft limit at 36000s (10 h).
    At the ~16 s/feature measured on gemma-4-12B-it, 2000 features is ~8.9 h —
    inside the soft limit with margin. Larger panels must be split into
    several jobs.
    """
    model_config = ConfigDict(extra="forbid")

    feature_ids: List[str] = Field(min_length=1, max_length=2000)


class ResumeSampleScope(str, Enum):
    """Which outcome a resume batch is drawn from.

    A sample must ask ONE question. "Do the failures still fail?" and "does the
    judge work at all?" are different, and a mixed batch of 20 answers neither —
    15 fresh successes and 5 repeat failures reads as 75%, which is true of
    nothing.
    """

    FAILED = "failed"
    PENDING = "pending"


class LabelingFailureReason(BaseModel):
    """One normalised failure reason and what it cost.

    `reason` is the first colon-delimited segment of `label_error` — the
    exception class or the whole fixed sentence — so the variable tail (host,
    port, id) does not split one cause into thousands of singleton groups.
    """

    model_config = ConfigDict(extra="forbid")

    reason: str
    count: int
    #: Real features carrying this reason, so a count can be checked rather than
    #: believed.
    sample_feature_ids: List[str]


class LabelingCoverageResponse(BaseModel):
    """What still needs labeling in one extraction, and what a resume would take.

    COVERAGE IS PART OF THE RESULT, NOT A FOOTNOTE — the shape established by
    the trial path. Without it there is no way to ask "what is left?", which is
    why 16,824 failed features across the estate went unnoticed: they were
    written as fake labels and every existing count read them as finished.
    """

    model_config = ConfigDict(extra="forbid")

    extraction_job_id: str
    total: int
    by_status: dict
    adjudicated: int
    #: Still lacking a verdict, whether or not a resume will take it.
    outstanding: int
    #: What a resume would ACTUALLY take — measured with the batch query's own
    #: predicate, so the button and the job cannot disagree.
    remaining: int
    #: Outstanding features that have used up their retries. Reported rather
    #: than hidden: they still need a label and will not be attempted again
    #: without raising `max_attempts`, and an operator needs both halves or the
    #: arithmetic on screen looks broken.
    exhausted: int = 0
    in_progress: int
    unclassified: int
    stale: Optional[int] = None
    failures_without_a_recorded_reason: int
    caveat: Optional[str] = None
    #: Why the failures failed, commonest first, with real examples. Answers
    #: "will re-running these work?" before spending 32 GPU-hours finding out.
    failure_reasons: List["LabelingFailureReason"] = []
    resume_feature_ids: List[str]


@router.get(
    "/labeling/{extraction_job_id}/coverage",
    response_model=LabelingCoverageResponse,
    summary="What still needs labeling in an extraction",
)
async def get_labeling_coverage(
    extraction_job_id: str,
    resume_limit: int = Query(
        2000,
        ge=1,
        le=2000,
        description=(
            "How many ids to return for the next resume batch. Capped at the "
            "panel route's own limit, which is set by Celery's 10 h soft limit "
            "at the measured ~8 s/feature."
        ),
    ),
    prompt_fingerprint: Optional[str] = Query(
        None,
        description=(
            "With judge_model, also counts and offers features adjudicated by a "
            "DIFFERENT judge. Omit for a routine resume, which must never "
            "revisit a verdict."
        ),
    ),
    judge_model: Optional[str] = Query(None),
    max_attempts: int = Query(
        labeling_eligibility.DEFAULT_MAX_ATTEMPTS,
        ge=1,
        description="Features that have failed this many times stop being offered.",
    ),
    only: Optional[ResumeSampleScope] = Query(
        None,
        description=(
            "Draw the resume batch from ONE outcome. `failed` answers 'do these "
            "still fail?', `pending` answers 'does the judge work?'. Omit for a "
            "normal resume, which takes both. An unrecognised value is rejected "
            "rather than widened to everything."
        ),
    ),
    db: AsyncSession = Depends(get_db),
):
    """Report labeling coverage for an extraction, and the next resume batch.

    Resume needs no endpoint of its own: the ids returned here go straight to
    `POST /labeling/panel`, which already labels an explicit set. What was
    missing was never the ability to label a subset — it was the ability to
    know WHICH subset.
    """
    exists = await db.execute(
        select(ExtractionJob.id).where(ExtractionJob.id == extraction_job_id)
    )
    if exists.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction job {extraction_job_id} not found",
        )

    rows = await db.execute(labeling_eligibility.coverage_query(extraction_job_id))
    counts = {status_value: count for status_value, count in rows.all()}

    # MEASURED with the batch query's own predicate, never inferred from status
    # counts — those know nothing about the attempt cap, which is how a card
    # came to read "Resume 0 of 39".
    eligible_rows = await db.execute(
        labeling_eligibility.eligible_count_query(
            extraction_job_id,
            max_attempts=max_attempts,
            prompt_fingerprint=prompt_fingerprint,
            judge_model=judge_model,
        )
    )
    eligible = int(eligible_rows.scalar_one() or 0)
    exhausted_rows = await db.execute(
        labeling_eligibility.exhausted_count_query(
            extraction_job_id, max_attempts=max_attempts
        )
    )
    exhausted = int(exhausted_rows.scalar_one() or 0)

    summary = labeling_eligibility.summarise(counts, eligible=eligible)

    stale = None
    if prompt_fingerprint is not None and judge_model is not None:
        stale_rows = await db.execute(
            labeling_eligibility.stale_count_query(
                extraction_job_id,
                prompt_fingerprint=prompt_fingerprint,
                judge_model=judge_model,
            )
        )
        stale = int(stale_rows.scalar_one() or 0)

    unreported = await db.execute(
        labeling_eligibility.unreported_failure_count_query(extraction_job_id)
    )
    failures_without_reason = int(unreported.scalar_one() or 0)

    ids = await db.execute(
        labeling_eligibility.resume_batch_query(
            extraction_job_id,
            limit=resume_limit,
            max_attempts=max_attempts,
            prompt_fingerprint=prompt_fingerprint,
            judge_model=judge_model,
            only=only.value if only else None,
        )
    )

    # The breakdown, plus an honest `other` when the tail was capped. Skipped
    # entirely when nothing failed — an empty section reads as "we looked and
    # found nothing", which here would be true but noisy.
    failure_reasons: List[LabelingFailureReason] = []
    if counts.get("failed"):
        reason_rows = await db.execute(
            labeling_eligibility.failure_reason_query(extraction_job_id)
        )
        # One query returns counts AND samples together, at a cost FLAT in group
        # count. See `failure_reason_query` for the measurement — it is ~8 ms
        # slower than 1 + N at today's single group, and decisively faster from
        # three groups on.
        grouped = reason_rows.all()
        for reason, count, samples in grouped:
            failure_reasons.append(
                LabelingFailureReason(
                    reason=reason,
                    count=count,
                    sample_feature_ids=list(samples or []),
                )
            )
        total_failed = await db.execute(
            labeling_eligibility.total_failure_count_query(extraction_job_id)
        )
        remainder = int(total_failed.scalar_one() or 0) - sum(row[1] for row in grouped)
        if remainder > 0:
            # Reported, never truncated silently: a breakdown that does not add
            # up to the failure count is worse than no breakdown.
            failure_reasons.append(
                LabelingFailureReason(
                    reason=f"other ({len(grouped)} commonest reasons shown)",
                    count=remainder,
                    sample_feature_ids=[],
                )
            )

    caveat = None
    if failures_without_reason:
        caveat = (
            f"{failures_without_reason} of the failures carry no recorded reason. "
            "They predate per-feature error capture (or were backfilled from the "
            "old encoding, where a failure was written as a fake label and the "
            "exception text was discarded). Retrying them is the only way to "
            "learn why they failed."
        )

    return LabelingCoverageResponse(
        extraction_job_id=extraction_job_id,
        stale=stale,
        failures_without_a_recorded_reason=failures_without_reason,
        exhausted=exhausted,
        caveat=caveat,
        failure_reasons=failure_reasons,
        resume_feature_ids=[row[0] for row in ids.all()],
        **summary,
    )


class ResumeSweepRequest(BaseModel):
    """Start a multi-batch resume.

    `max_batches` is REQUIRED and has no default anywhere in the chain — not
    here, not on the service, not on the column. An open-ended sweep is a request
    to spend an unknown number of GPU-hours (L46 is ~59 for 27 batches), and it
    must not be reachable by omitting a field.
    """

    model_config = ConfigDict(extra="forbid")

    max_batches: int = Field(
        ge=1,
        le=100,
        description=(
            "How many batches this sweep may run before stopping. At ~8 s per "
            "feature and 2000 per batch, each batch is roughly 4.4 hours."
        ),
    )
    config: LabelingConfigRequest = Field(
        description="The judge configuration every batch reuses, frozen at start."
    )
    #: THE ELIGIBILITY PREDICATE THE SWEEP WAS SIZED WITH.
    #:
    #: `max_batches` is computed by the caller from a coverage read that may
    #: carry a template fingerprint and a judge. Selecting without them makes
    #: every batch take a different set than the one the operator was quoted —
    #: after a template edit, a set up to twenty times smaller — so the sweep
    #: either stops immediately having done nothing or works on the wrong
    #: backlog while the one it was booked for is never touched.
    #:
    #: Frozen on the row with the config, so editing the template mid-sweep
    #: cannot change what a running sweep selects.
    prompt_fingerprint: Optional[str] = Field(
        None,
        description="Fingerprint the batch count was computed against.",
    )
    judge_model: Optional[str] = Field(
        None,
        description="Judge the batch count was computed against.",
    )
    batch_size: int = Field(
        2000,
        ge=1,
        le=2000,
        description="Features per batch. 2000 is ~4.4 h, inside Celery's 10 h soft limit.",
    )


class ResumeSweepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    extraction_job_id: str
    status: str
    batches_done: int
    max_batches: int
    batch_size: int
    #: Counted from what each batch WROTE. A sweep whose batches all failed must
    #: not look complete.
    features_labeled: int
    features_failed: int
    last_labeling_job_id: Optional[str] = None
    cancel_requested_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


def _sweep_response(sweep) -> ResumeSweepResponse:
    return ResumeSweepResponse(
        id=sweep.id,
        extraction_job_id=sweep.extraction_job_id,
        status=sweep.status,
        batches_done=sweep.batches_done,
        max_batches=sweep.max_batches,
        batch_size=sweep.batch_size,
        features_labeled=sweep.features_labeled,
        features_failed=sweep.features_failed,
        last_labeling_job_id=sweep.last_labeling_job_id,
        cancel_requested_at=sweep.cancel_requested_at,
        error_message=sweep.error_message,
        created_at=sweep.created_at,
        completed_at=sweep.completed_at,
    )


@router.post(
    "/labeling/{extraction_job_id}/resume-sweep",
    response_model=ResumeSweepResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Resume labeling across several batches",
)
async def start_resume_sweep(
    extraction_job_id: str,
    body: ResumeSweepRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run up to `max_batches` resume batches, one Celery task each.

    Not a loop: 27 batches of 2000 is ~59 GPU-hours against a 10 h soft limit, so
    each batch is its own task and the sweep row carries the progress.
    """
    from src.services.labeling_sweep_service import LabelingSweepService, SweepError

    # `run_sync` hands the service a real sync Session on the async
    # connection. The service is sync because the Celery task needs it that way,
    # and ONE implementation shared by both is worth more than an async twin
    # that can drift from it.
    try:
        sweep = await db.run_sync(
            lambda sync_db: LabelingSweepService(sync_db).create(
                extraction_job_id,
                max_batches=body.max_batches,
                # The predicate is frozen ALONGSIDE the judge config, so a
                # running sweep keeps selecting the set it was sized for even
                # if the template is edited underneath it.
                config={
                    **body.config.model_dump(exclude_none=True),
                    **(
                        {"prompt_fingerprint": body.prompt_fingerprint}
                        if body.prompt_fingerprint else {}
                    ),
                    **(
                        {"judge_model": body.judge_model}
                        if body.judge_model else {}
                    ),
                },
                batch_size=body.batch_size,
            )
        )
    except SweepError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    resume_sweep_step.delay(sweep.id)
    return _sweep_response(sweep)


@router.get(
    "/labeling/resume-sweeps/{sweep_id}",
    response_model=ResumeSweepResponse,
    summary="How far a resume sweep has got",
)
async def get_resume_sweep(sweep_id: str, db: AsyncSession = Depends(get_db)):
    from src.services.labeling_sweep_service import LabelingSweepService

    sweep = await db.run_sync(
        lambda sync_db: LabelingSweepService(sync_db).get(sweep_id)
    )
    if sweep is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Sweep {sweep_id} not found"
        )
    return _sweep_response(sweep)


@router.post(
    "/labeling/resume-sweeps/{sweep_id}/cancel",
    response_model=ResumeSweepResponse,
    summary="Stop a resume sweep after its current batch",
)
async def cancel_resume_sweep(sweep_id: str, db: AsyncSession = Depends(get_db)):
    """Cooperative. The batch in flight finishes; nothing further is enqueued.

    The worker pool here is `--pool=solo -c 1`, so `revoke(terminate=True)`
    signals a pool child that does not exist and a busy solo worker never reads
    the control queue — it returns cleanly and does nothing. Stopping has to be
    something the task itself checks, and it does, before every batch.
    """
    from src.services.labeling_sweep_service import LabelingSweepService

    sweep = await db.run_sync(
        lambda sync_db: LabelingSweepService(sync_db).request_stop(sweep_id)
    )
    if sweep is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Sweep {sweep_id} not found"
        )
    return _sweep_response(sweep)


async def _fetch_served_models(endpoint: str) -> List[str]:
    """The models an OpenAI-compatible endpoint is serving, sorted.

    A named module-level function rather than an inline `httpx.AsyncClient`
    because a test cannot patch the latter without also patching ITS OWN
    client — both are `httpx.AsyncClient`, so a class-level patch made the
    test client return the mocked model list instead of calling the app, and
    the endpoint under test never ran at all.
    """
    import httpx

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{endpoint.rstrip('/')}/models")
        response.raise_for_status()
        return sorted(
            m.get("id") for m in (response.json().get("data") or []) if m.get("id")
        )


class AvailableJudgesResponse(BaseModel):
    """What the judge endpoint of an existing job is actually serving now.

    Keyed on the JOB, never on a URL from the caller: the endpoint comes from
    the stored row, so this cannot be pointed at an arbitrary host.
    """

    model_config = ConfigDict(extra="forbid")

    labeling_job_id: str
    endpoint: Optional[str] = None
    #: What the job originally used. May no longer be served.
    original_model: Optional[str] = None
    #: True when `original_model` is still available — i.e. a plain resume works.
    original_available: bool = False
    models: List[str] = []
    reachable: bool = True
    detail: Optional[str] = None


@router.get(
    "/labeling/{labeling_job_id}/available-judges",
    response_model=AvailableJudgesResponse,
    summary="Which models this job's endpoint can label with now",
)
async def list_available_judges(
    labeling_job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List the models the job's own labeling endpoint currently serves.

    Resume defaults to the judge a job used, which is right when comparing
    halves of one run. It is useless when that judge is GONE — which is exactly
    when someone needs to resume. A judge removed or renamed between the
    original run and the resume left every remaining feature unlabelable, with
    no way through the UI to choose another.

    This is what lets a resume name a different model deliberately.
    `features.label_model` and `label_prompt_fingerprint` are per-feature, so an
    extraction labelled by two judges stays honest about which produced what.
    """
    job = (
        await db.execute(select(LabelingJob).where(LabelingJob.id == labeling_job_id))
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Labeling job {labeling_job_id} not found",
        )

    original = job.openai_compatible_model or job.openai_model or job.local_model
    endpoint = job.openai_compatible_endpoint
    if not endpoint:
        # Nothing to enumerate — a local or OpenAI-keyed job. Report honestly
        # rather than inventing an empty list that reads as "no models".
        return AvailableJudgesResponse(
            labeling_job_id=labeling_job_id,
            endpoint=None,
            original_model=original,
            original_available=True,
            reachable=False,
            detail="This job does not use an OpenAI-compatible endpoint, so its "
                   "available models cannot be enumerated here.",
        )

    try:
        models = await _fetch_served_models(endpoint)
    except Exception as exc:  # noqa: BLE001 — reported, never raised at the caller
        # A picker that 500s leaves an operator with no route at all. Say the
        # endpoint could not be reached and let them retry.
        return AvailableJudgesResponse(
            labeling_job_id=labeling_job_id,
            endpoint=endpoint,
            original_model=original,
            original_available=False,
            reachable=False,
            detail=f"Could not reach {endpoint}: {type(exc).__name__}",
        )

    return AvailableJudgesResponse(
        labeling_job_id=labeling_job_id,
        endpoint=endpoint,
        original_model=original,
        original_available=bool(original and original in models),
        models=models,
        reachable=True,
    )


@router.post(
    "/labeling/panel",
    response_model=LabelingStatusResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start feature labeling scoped to an explicit panel",
)
async def start_labeling_panel(
    config: LabelingPanelRequest,
    db: AsyncSession = Depends(get_db),
):
    """Label ONLY the listed features, writing the labels to the feature rows.

    Refuses with 422 if any requested id is absent from the extraction: a
    silently-shrunken panel is not the panel that was asked for, and any rate
    computed from it would be wrong.
    """
    return await start_labeling(config, db)


@router.post(
    "/labeling",
    response_model=LabelingStatusResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start feature labeling"
)
async def start_labeling(
    config: LabelingConfigRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Start a semantic labeling job for a completed extraction.

    This creates a labeling job and queues it for async processing. Features
    are labeled independently from extraction, allowing re-labeling without
    re-extraction.

    Args:
        config: Labeling configuration (extraction_job_id, labeling_method, etc.)

    Returns:
        LabelingStatusResponse with job details

    Raises:
        404: Extraction not found
        409: Active labeling already exists for this extraction
        422: Extraction not completed or has no features
    """
    labeling_service = LabelingService(db)

    try:
        # Start labeling job (creates record in QUEUED status)
        labeling_job = await labeling_service.start_labeling(
            extraction_job_id=config.extraction_job_id,
            config=config.model_dump()
        )

        # Queue Celery task for async labeling
        task = label_features_task.delay(labeling_job.id)

        # Update with Celery task ID
        labeling_job.celery_task_id = task.id
        await db.commit()
        await db.refresh(labeling_job)

        logger.info(
            f"Started labeling job {labeling_job.id} for extraction "
            f"{config.extraction_job_id} with task {task.id}"
        )

        return LabelingStatusResponse.model_validate(labeling_job)

    except ValueError as e:
        error_message = str(e)

        # Check for specific error conditions
        if "not found" in error_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        elif "already has an active labeling" in error_message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_message
            )
        else:
            # Must be completed, has features, etc.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_message
            )


@router.get(
    "/labeling/{labeling_job_id}",
    response_model=LabelingStatusResponse,
    summary="Get labeling job status"
)
async def get_labeling_status(
    labeling_job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the status of a specific labeling job.

    Args:
        labeling_job_id: ID of the labeling job

    Returns:
        LabelingStatusResponse with status, progress, and statistics

    Raises:
        404: Labeling job not found
    """
    labeling_service = LabelingService(db)

    labeling_job = await labeling_service.get_labeling_job(labeling_job_id)

    if not labeling_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Labeling job {labeling_job_id} not found"
        )

    response = LabelingStatusResponse.model_validate(labeling_job)
    enriched = await _enrich_labeling_responses(db, [response])
    return enriched[0]


@router.get(
    "/labeling",
    response_model=LabelingListResponse,
    summary="List labeling jobs"
)
async def list_labeling_jobs(
    extraction_job_id: Optional[str] = Query(None, description="Filter by extraction job ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a paginated list of labeling jobs.

    Args:
        extraction_job_id: Optional filter by extraction job ID
        limit: Maximum number of results to return (1-100)
        offset: Number of results to skip for pagination

    Returns:
        LabelingListResponse with list of labeling jobs and metadata
    """
    labeling_service = LabelingService(db)

    # Get labeling jobs
    jobs_list, total = await labeling_service.list_labeling_jobs(
        extraction_job_id=extraction_job_id,
        limit=limit,
        offset=offset
    )

    responses = [LabelingStatusResponse.model_validate(job) for job in jobs_list]
    responses = await _enrich_labeling_responses(db, responses)

    return LabelingListResponse(
        data=responses,
        meta={
            "total": total,
            "limit": limit,
            "offset": offset
        }
    )


@router.post(
    "/labeling/{labeling_job_id}/cancel",
    status_code=status.HTTP_200_OK,
    summary="Cancel labeling job"
)
async def cancel_labeling(
    labeling_job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an active labeling job.

    Args:
        labeling_job_id: ID of the labeling job to cancel

    Returns:
        Success message

    Raises:
        404: Labeling job not found
        409: Labeling job not in cancellable state
    """
    labeling_service = LabelingService(db)

    try:
        await labeling_service.cancel_labeling_job(labeling_job_id)
        logger.info(f"Cancelled labeling job {labeling_job_id}")
        return {"message": "Labeling job cancelled successfully"}
    except ValueError as e:
        error_message = str(e)
        if "not found" in error_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_message
            )
        else:
            # Cannot cancel due to status
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_message
            )


@router.delete(
    "/labeling/{labeling_job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete labeling job"
)
async def delete_labeling(
    labeling_job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a labeling job record.

    This does NOT delete the features or their labels, only the labeling job
    record itself. Feature labels will remain intact.

    If the job is currently active (queued or labeling), it will be automatically
    cancelled before deletion by revoking the Celery task.

    Args:
        labeling_job_id: ID of the labeling job to delete

    Raises:
        404: Labeling job not found
    """
    labeling_service = LabelingService(db)

    try:
        await labeling_service.delete_labeling_job(labeling_job_id)
        logger.info(f"Deleted labeling job {labeling_job_id}")
        return None  # 204 No Content
    except ValueError as e:
        error_message = str(e)
        # Only possible error now is "not found"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_message
        )


@router.post(
    "/extractions/{extraction_id}/label",
    response_model=LabelingStatusResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Label extraction (convenience endpoint)"
)
async def label_extraction(
    extraction_id: str,
    config: LabelingConfigRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Convenience endpoint to start labeling for an extraction.

    This is a shorthand for POST /labeling with extraction_job_id in the body.
    The extraction_id from the URL takes precedence over config.extraction_job_id.

    Args:
        extraction_id: ID of the extraction to label
        config: Labeling configuration (labeling_method, openai_model, etc.)

    Returns:
        LabelingStatusResponse with job details

    Raises:
        404: Extraction not found
        409: Active labeling already exists
        422: Extraction not completed or has no features
    """
    # Override extraction_job_id with URL parameter
    config.extraction_job_id = extraction_id

    # Delegate to main labeling endpoint
    return await start_labeling(config, db)


@router.get(
    "/labeling/models/available",
    summary="List available Ollama models"
)
async def list_available_ollama_models(db: AsyncSession = Depends(get_db)):
    """
    List available Ollama models for local labeling.

    Queries the Ollama API to get all available models. Returns both
    the raw model list and a formatted list with display names.

    Returns:
        Dict with 'models' array containing model information

    Raises:
        503: Ollama service unavailable
    """
    # Resolve Ollama URL: DB setting takes precedence over env var
    from src.models.app_setting import AppSetting
    result = await db.execute(select(AppSetting).where(AppSetting.key == "ollama_url"))
    db_setting = result.scalar_one_or_none()
    ollama_url = (db_setting.value if db_setting else None) or settings.ollama_url

    try:
        # Query Ollama/OpenAI-compatible API for available models
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try OpenAI-compatible /v1/models first, fall back to Ollama /api/tags
            try:
                response = await client.get(f"{ollama_url}/v1/models")
                response.raise_for_status()
                data = response.json()
                models = []
                for model in data.get("data", []):
                    model_name = model.get("id", "")
                    models.append({
                        "name": model_name,
                        "display_name": model_name,
                        "size": 0,
                        "size_gb": 0,
                        "modified_at": "",
                        "details": {},
                    })
                return {"models": models, "total": len(models)}
            except Exception:
                pass
            # Fall back to Ollama /api/tags
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            # Extract model information
            models = []
            for model in data.get("models", []):
                model_name = model.get("name", "")
                model_size = model.get("size", 0)
                modified_at = model.get("modified_at", "")

                # Format size in GB
                size_gb = model_size / (1024**3) if model_size else 0

                # Create display name
                display_name = f"{model_name}"
                if size_gb > 0:
                    display_name += f" ({size_gb:.1f}GB)"

                models.append({
                    "name": model_name,
                    "display_name": display_name,
                    "size": model_size,
                    "size_gb": round(size_gb, 2),
                    "modified_at": modified_at,
                    "details": model.get("details", {})
                })

            return {
                "models": models,
                "total": len(models)
            }

    except httpx.RequestError as e:
        logger.exception("Failed to connect to Ollama")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is not available. Please ensure Ollama is running."
        )
    except Exception as e:
        logger.exception("Error listing Ollama models")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list Ollama models"
        )


class FetchModelsRequest(BaseModel):
    api_key: Optional[str] = None
    endpoint_url: str = "https://api.openai.com/v1"


async def _host_may_receive_stored_key(db: AsyncSession, endpoint_url: str) -> bool:
    """May the OPERATOR's stored key be attached to a request for this URL?

    Only for hosts the operator already designated: the configured
    `openai_compatible_endpoint`, or api.openai.com — the two the key was
    entered for. Every other host gets no stored credential, however reachable
    it is (MIS-E2E-069).

    Compares HOST only. Matching the full URL would let a path or query
    difference defeat it; matching a prefix would admit
    `api.openai.com.evil.tld`.
    """
    from urllib.parse import urlparse

    try:
        target = (urlparse(endpoint_url).hostname or "").lower()
    except ValueError:
        return False
    if not target:
        return False

    allowed = {"api.openai.com"}
    configured = await AppSettingService.get_decrypted_value(
        db, "openai_compatible_endpoint"
    )
    for candidate in (configured,
                      getattr(settings, "openai_compatible_endpoint", None)):
        if candidate:
            host = (urlparse(candidate).hostname or "").lower()
            if host:
                allowed.add(host)
    return target in allowed


@router.post(
    "/labeling/models/openai",
    summary="Fetch available models from OpenAI or compatible endpoint"
)
async def fetch_openai_models(
    request: FetchModelsRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch available models from any OpenAI-compatible endpoint.
    Proxied through the backend to avoid CORS and DNS resolution issues.

    API key resolution order:
      1. request.api_key (explicit in POST body)
      2. Database AppSetting 'openai_api_key' (set via Settings → API Keys)
      3. Environment variable settings.openai_api_key
      4. None (unauthenticated — fine for Ollama / vLLM / miLLM)
    """
    # VALIDATE THE URL FIRST (MIS-E2E-069). `validate_llm_endpoint_url` exists
    # for exactly this and had two call sites in the whole tree — neither of
    # them this one, the only path that attaches a stored credential.
    from src.utils.url_validation import validate_llm_endpoint_url
    try:
        validate_llm_endpoint_url(request.endpoint_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # THE STORED KEY IS NEVER SENT TO A HOST THE REQUEST BODY CHOSE.
    #
    # This used to fall back to the operator's decrypted `openai_api_key`
    # whenever the body omitted `api_key` — so the ABSENCE of a credential in
    # the request is what caused one to be read from the database and sent to
    # `endpoint_url`. A single unauthenticated POST naming any host exfiltrated
    # it, defeating the AES-256-GCM at rest, the masking on every read and the
    # Settings PIN in one call.
    #
    # A caller-supplied key is still honoured — it is the caller's to spend. A
    # STORED key is attached only when the resolved host is one the operator
    # already designated.
    api_key = request.api_key
    if not api_key and await _host_may_receive_stored_key(db, request.endpoint_url):
        from src.models.app_setting import AppSetting
        from src.core.encryption import decrypt_value

        result = await db.execute(
            select(AppSetting).where(AppSetting.key == "openai_api_key")
        )
        db_setting = result.scalar_one_or_none()
        if db_setting and db_setting.value:
            api_key = (
                decrypt_value(db_setting.value, setting_key="openai_api_key")
                if db_setting.is_sensitive
                else db_setting.value
            )
        if not api_key:
            api_key = getattr(settings, "openai_api_key", None)
    elif not api_key:
        logger.info(
            "listing models at %s without a stored credential — the host is not "
            "the configured endpoint or api.openai.com",
            request.endpoint_url,
        )

    # Normalize endpoint URL
    base_url = request.endpoint_url.rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"

    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{base_url}/models",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            models = []

            # OpenAI format: { "data": [{ "id": "gpt-4o-mini", ... }] }
            if "data" in data and isinstance(data["data"], list):
                for m in data["data"]:
                    model_id = m.get("id", "")
                    if model_id:
                        models.append({
                            "id": model_id,
                            "owned_by": m.get("owned_by", ""),
                        })

            # Ollama format: { "models": [{ "name": "gemma2:2b", ... }] }
            elif "models" in data and isinstance(data["models"], list):
                for m in data["models"]:
                    model_name = m.get("name", "") or m.get("id", "")
                    if model_name:
                        models.append({
                            "id": model_name,
                            "owned_by": m.get("owned_by", "ollama"),
                        })

            # Sort: gpt models first, then alphabetical
            models.sort(key=lambda m: (
                0 if m["id"].startswith("gpt-") else 1,
                m["id"]
            ))

            return {
                "models": models,
                "total": len(models)
            }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key."
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"API returned HTTP {e.response.status_code}: {e.response.text[:200]}"
        )
    except httpx.RequestError as e:
        logger.exception("Failed to connect to endpoint")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to {base_url}. Check the endpoint URL."
        )
    except Exception as e:
        logger.exception("Error fetching models")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch models"
        )


# ── prompt-template trials ───────────────────────────────────────────────────
#
# A trial runs ONE template over an explicit feature panel and writes no label.
# Kept on separate routes from /labeling rather than as a flag on it: the apply
# endpoint's whole job is to persist, and a mode flag there would put every
# production labeling run one boolean away from silently not persisting.

class LabelingTrialRequest(BaseModel):
    """Trial configuration.

    extra="forbid" is deliberate and is the reason this is not a field on
    LabelingConfigRequest, which permits unknown keys. A typo'd `featureIds`
    there would be silently dropped and the request would execute as a
    full-extraction APPLY run over every feature in the extraction.
    """
    model_config = ConfigDict(extra="forbid")

    extraction_job_id: str
    feature_ids: List[str] = Field(min_length=1, max_length=200)
    prompt_template_id: Optional[str] = None
    name: Optional[str] = Field(default=None, max_length=200)
    labeling_method: str = "openai_compatible"
    openai_model: Optional[str] = None
    openai_compatible_endpoint: Optional[str] = None
    openai_compatible_model: Optional[str] = None
    batch_size: int = Field(default=10, ge=1, le=50)
    max_tokens: int = Field(default=300, ge=50, le=8000)


@router.post("/labeling/trials", status_code=status.HTTP_201_CREATED,
             summary="Start a prompt-template trial (writes no labels)")
async def start_labeling_trial(body: LabelingTrialRequest,
                               db: AsyncSession = Depends(get_db)):
    from src.services.labeling_trial_service import LabelingTrialService, TrialError
    from src.workers.labeling_tasks import label_features_trial_task

    service = LabelingTrialService(db)
    try:
        run = await service.start_trial(
            body.extraction_job_id, body.feature_ids, body.model_dump()
        )
    except TrialError as exc:
        msg = str(exc)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        if "in-flight" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise HTTPException(status_code=422, detail=msg)

    label_features_trial_task.delay(run.labeling_job_id)
    return {
        "trial_run_id": run.id,
        "labeling_job_id": run.labeling_job_id,
        "panel_id": run.panel_id,
        "status": run.status,
        "writes_labels": False,
    }


@router.get("/labeling/trials", summary="List prompt-template trials")
async def list_labeling_trials(
    extraction_job_id: Optional[str] = None,
    panel_id: Optional[str] = None,
    prompt_template_id: Optional[str] = None,
    limit: int = 50, offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    from src.models.labeling_trial_run import LabelingTrialRun
    q = select(LabelingTrialRun)
    if extraction_job_id:
        q = q.where(LabelingTrialRun.extraction_job_id == extraction_job_id)
    if panel_id:
        q = q.where(LabelingTrialRun.panel_id == panel_id)
    if prompt_template_id:
        q = q.where(LabelingTrialRun.prompt_template_id == prompt_template_id)
    q = q.order_by(LabelingTrialRun.created_at.desc()).limit(
        min(limit, 100)).offset(offset)
    rows = (await db.execute(q)).scalars().all()
    return {"data": [{
        "trial_run_id": r.id, "panel_id": r.panel_id, "name": r.name,
        "prompt_template_id": r.prompt_template_id, "status": r.status,
        "extraction_job_id": r.extraction_job_id,
        "stats": (r.payload or {}).get("stats"),
        "created_at": r.created_at, "completed_at": r.completed_at,
    } for r in rows], "meta": {"limit": limit, "offset": offset}}


@router.get("/labeling/trials/{trial_run_id}", summary="One trial's full result")
async def get_labeling_trial(trial_run_id: str, db: AsyncSession = Depends(get_db)):
    from src.models.labeling_trial_run import LabelingTrialRun
    run = (await db.execute(select(LabelingTrialRun).where(
        LabelingTrialRun.id == trial_run_id))).scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail=f"trial {trial_run_id} not found")
    return {
        "trial_run_id": run.id, "panel_id": run.panel_id, "name": run.name,
        "status": run.status, "error": run.error,
        "prompt_template_id": run.prompt_template_id,
        "extraction_job_id": run.extraction_job_id,
        "payload": run.payload,
        "created_at": run.created_at, "completed_at": run.completed_at,
    }


@router.get("/labeling/trials/compare/{run_a}/{run_b}",
            summary="Compare two trials over the same panel")
async def compare_labeling_trials(run_a: str, run_b: str,
                                  db: AsyncSession = Depends(get_db)):
    from src.models.labeling_trial_run import LabelingTrialRun
    from src.services.labeling_trial_service import LabelingTrialService

    rows = (await db.execute(select(LabelingTrialRun).where(
        LabelingTrialRun.id.in_([run_a, run_b])))).scalars().all()
    found = {r.id: r for r in rows}
    missing = [r for r in (run_a, run_b) if r not in found]
    if missing:
        raise HTTPException(status_code=404, detail=f"trial(s) not found: {missing}")

    result = LabelingTrialService.compare(
        found[run_a].payload or {}, found[run_b].payload or {})
    if not result.get("comparable"):
        # Refusing is the point. Comparing across panels would produce a number
        # that looks like a template difference and is not one.
        raise HTTPException(status_code=409, detail=result["reason"])
    return {"run_a": run_a, "run_b": run_b, **result}
