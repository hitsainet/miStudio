"""
Celery task for cross-feature grouping precompute (Feature 010).

Builds the token→feature inverted index and context-similarity groups for one
extraction. CPU-only — routed to the ``low_priority`` queue alongside NLP
analysis. Progress streams on WebSocket channel
``extractions/{extraction_id}/feature-groups``.
"""

import logging
from typing import Any, Dict, Optional

from src.core.celery_app import celery_app
from src.models.extraction_job import ExtractionJob
from src.core.cancellation import OperatorCancelled, cancel_checker
from src.services.feature_grouping_service import FeatureGroupingService
from src.workers.base_task import DatabaseTask, mark_task_queue_entries_completed
from src.workers.websocket_emitter import emit_progress

logger = logging.getLogger(__name__)

TASK_TYPE = "feature_grouping"


def emit_feature_groups_event(extraction_id: str, event: str, data: Dict[str, Any]) -> None:
    """Emit a feature-groups event on the extraction's grouping channel."""
    emit_progress(
        channel=f"extractions/{extraction_id}/feature-groups",
        event=f"feature_groups:{event}",
        data=data,
    )


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.feature_grouping_tasks.compute_feature_groups",
    max_retries=2,
    default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError, OSError),
)
def compute_feature_groups_task(
    self,
    extraction_id: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the grouping index for an extraction."""
    logger.info(f"Starting feature grouping for extraction {extraction_id}")

    with self.get_db() as db:
        extraction = db.query(ExtractionJob).filter(ExtractionJob.id == extraction_id).first()
        if not extraction:
            raise ValueError(f"Extraction job {extraction_id} not found")

        def progress_cb(progress: float, stage: str) -> None:
            emit_feature_groups_event(
                extraction_id,
                "progress",
                {
                    "extraction_id": extraction_id,
                    "progress": round(progress * 100, 1),
                    "stage": stage,
                },
            )

        try:
            service = FeatureGroupingService()
            run = service.compute(
                db, extraction_id, params=params, progress_cb=progress_cb,
                cancel_check_for=lambda rid: cancel_checker(
                    "feature_grouping", rid, db=db),
            )

            mark_task_queue_entries_completed(
                db, entity_id=extraction_id, entity_type="extraction", task_type=TASK_TYPE
            )
            emit_feature_groups_event(
                extraction_id,
                "completed",
                {
                    "extraction_id": extraction_id,
                    "run_id": run.id,
                    "feature_count": run.feature_count,
                    "group_count": run.group_count,
                },
            )
            return {
                "status": "completed",
                "run_id": run.id,
                "feature_count": run.feature_count,
                "group_count": run.group_count,
            }
        except OperatorCancelled as cancelled:
            from src.models.feature_grouping import FeatureGroupingRun, GroupingRunStatus

            row = db.query(FeatureGroupingRun).filter(
                FeatureGroupingRun.id == cancelled.target_id).first()
            if row is not None:
                row.status = GroupingRunStatus.CANCELLED.value
                row.error_message = cancelled.detail[:500]
                db.commit()
            logger.info("Feature grouping %s cancelled", cancelled.target_id)
            return {"status": "cancelled", "run_id": cancelled.target_id,
                    "detail": cancelled.detail}

        except Exception as e:
            logger.error(
                f"Feature grouping failed for extraction {extraction_id}: {e}", exc_info=True
            )
            emit_feature_groups_event(
                extraction_id,
                "failed",
                {"extraction_id": extraction_id, "error": str(e)},
            )
            raise
