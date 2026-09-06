"""
Enhanced per-feature labeling endpoints.

POST /features/{feature_id}/label/enhanced   — start a two-pass labeling job
GET  /features/{feature_id}/label/enhanced/latest — get the most recent job for this feature
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.deps import get_db
from src.models.feature import Feature
from src.models.enhanced_labeling_job import EnhancedLabelingJob, EnhancedLabelingStatus
from src.schemas.enhanced_labeling import EnhancedLabelingJobResponse
from src.services.app_setting_service import AppSettingService

logger = logging.getLogger(__name__)

router = APIRouter()

_METHOD_SETTING_KEY = "enhanced_labeling_method"  # "openai" | "openai_compatible"
_ENDPOINT_SETTING_KEY = "openai_compatible_endpoint"
_MODEL_SETTING_KEY = "openai_compatible_model"
_OPENAI_MODEL_SETTING_KEY = "enhanced_labeling_openai_model"
_OPENAI_API_KEY_SETTING_KEY = "openai_api_key"
_WORKERS_SETTING_KEY = "enhanced_labeling_max_workers"
_EXAMPLES_TOTAL = 20  # fixed for now; could become a setting later
_OPENAI_ENDPOINT = "https://api.openai.com/v1"


@router.post(
    "/features/{feature_id}/label/enhanced",
    response_model=EnhancedLabelingJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start an enhanced two-pass labeling job for a single feature",
    tags=["enhanced-labeling"],
)
async def start_enhanced_labeling(
    feature_id: str,
    db: AsyncSession = Depends(get_db),
) -> EnhancedLabelingJobResponse:
    """
    Queue a two-pass enhanced labeling job for the given feature.

    Uses the openai_compatible_endpoint and openai_compatible_model settings
    already configured in the Settings panel. Returns immediately with the
    new job record; subscribe to the WebSocket channel
    `enhanced_labeling/{job_id}` for live progress.

    If a job is already queued or running for this feature, returns it
    (HTTP 200) rather than creating a duplicate.
    """
    # Verify feature exists and lock its row for the duration of this transaction.
    # The row lock serialises concurrent start requests for the same feature so
    # two rapid clicks can't both pass the duplicate-active check and create two
    # jobs (the check + insert below is otherwise a race).
    result = await db.execute(
        select(Feature).where(Feature.id == feature_id).with_for_update()
    )
    feature = result.scalar_one_or_none()
    if not feature:
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")

    # Check for an already-active job (now serialised by the feature-row lock)
    active_result = await db.execute(
        select(EnhancedLabelingJob)
        .where(
            EnhancedLabelingJob.feature_id == feature_id,
            EnhancedLabelingJob.status.in_(
                [EnhancedLabelingStatus.QUEUED.value, EnhancedLabelingStatus.RUNNING.value]
            ),
        )
        .order_by(EnhancedLabelingJob.created_at.desc())
        .limit(1)
    )
    existing = active_result.scalar_one_or_none()
    if existing:
        return EnhancedLabelingJobResponse.model_validate(existing)

    # Resolve method + endpoint/model based on settings
    method = (
        await AppSettingService.get_decrypted_value(db, _METHOD_SETTING_KEY)
        or "openai_compatible"
    )

    if method == "openai":
        # Real OpenAI API — model from dedicated setting, key from API Keys tab
        model = await AppSettingService.get_decrypted_value(db, _OPENAI_MODEL_SETTING_KEY)
        api_key = await AppSettingService.get_decrypted_value(db, _OPENAI_API_KEY_SETTING_KEY)
        if not model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "enhanced_labeling_openai_model must be configured in "
                    "Settings → Labeling → Enhanced Labeling when method is 'openai'."
                ),
            )
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "openai_api_key must be configured in Settings → API Keys "
                    "when enhanced labeling method is 'openai'."
                ),
            )
        endpoint = _OPENAI_ENDPOINT
    elif method == "openai_compatible":
        endpoint = await AppSettingService.get_decrypted_value(db, _ENDPOINT_SETTING_KEY)
        model = await AppSettingService.get_decrypted_value(db, _MODEL_SETTING_KEY)
        if not endpoint or not model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "openai_compatible_endpoint and openai_compatible_model must be configured "
                    "in Settings → Endpoints before using enhanced labeling."
                ),
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown enhanced_labeling_method: {method!r}",
        )

    raw_workers = await AppSettingService.get_decrypted_value(db, _WORKERS_SETTING_KEY)
    workers = int(raw_workers) if raw_workers else 8

    # Create job record
    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    job_id = f"elj_{feature.neuron_index}_{ts_ms}"

    job = EnhancedLabelingJob(
        id=job_id,
        feature_id=feature_id,
        status=EnhancedLabelingStatus.QUEUED.value,
        method=method,
        endpoint=endpoint,
        model=model,
        workers=workers,
        examples_total=_EXAMPLES_TOTAL,
    )
    db.add(job)

    # Mark feature purple immediately so it's visible in the starred filter
    feature.is_favorite = True
    feature.star_color = "purple"

    await db.flush()

    # Commit the job row first so the Celery worker can find it
    await db.commit()
    await db.refresh(job)

    # Queue Celery task and persist its ID
    from src.workers.enhanced_labeling_tasks import enhanced_label_feature_task

    celery_result = enhanced_label_feature_task.delay(job_id)
    job.celery_task_id = celery_result.id
    await db.commit()
    await db.refresh(job)

    logger.info("Queued enhanced labeling job %s for feature %s", job_id, feature_id)
    return EnhancedLabelingJobResponse.model_validate(job)


@router.get(
    "/features/{feature_id}/label/enhanced/latest",
    response_model=EnhancedLabelingJobResponse | None,
    summary="Get the most recent enhanced labeling job for a feature",
    tags=["enhanced-labeling"],
)
async def get_latest_enhanced_labeling_job(
    feature_id: str,
    db: AsyncSession = Depends(get_db),
) -> EnhancedLabelingJobResponse | None:
    """
    Return the most recently created enhanced labeling job for this feature,
    or null if none exists. Used by the modal on open to restore in-progress state.
    """
    result = await db.execute(
        select(EnhancedLabelingJob)
        .where(EnhancedLabelingJob.feature_id == feature_id)
        .order_by(EnhancedLabelingJob.created_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if not job:
        return None
    return EnhancedLabelingJobResponse.model_validate(job)


@router.post(
    "/enhanced-labeling/{job_id}/cancel",
    summary="Stop a running enhanced labeling job",
    tags=["enhanced-labeling"],
)
async def cancel_enhanced_labeling(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Stop a two-pass enhanced labeling job at its next example.

    STARTABLE AND NOT STOPPABLE UNTIL NOW. Pass 1 fans out one LLM call per
    example and pass 2 synthesises them; on a slow endpoint that is minutes of
    billed calls with no way to reach it. The worker is `--pool=solo -c 1`, so
    the job also blocked every other queue for its whole duration.
    """
    from starlette.concurrency import run_in_threadpool
    from sqlalchemy import select

    from ....core.cancellation import request_cancel
    from ....models.enhanced_labeling_job import EnhancedLabelingJob

    result = await db.execute(
        select(EnhancedLabelingJob).where(EnhancedLabelingJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    outcome = await run_in_threadpool(
        request_cancel, "enhanced_labeling", job_id,
        reason="cancelled by operator", celery_task_id=job.celery_task_id,
    )
    if not outcome.requested:
        raise HTTPException(status_code=409, detail=outcome.detail)
    return {"id": job_id, "status": "cancelled", "message": outcome.detail}
