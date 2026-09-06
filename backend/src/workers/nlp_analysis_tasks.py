"""
Celery tasks for NLP analysis of feature activation examples.

These tasks run asynchronously to pre-compute NLP analysis (POS tagging, NER,
context patterns, semantic clusters) for features after extraction completes.
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta

from src.core.celery_app import celery_app
from src.workers.base_task import DatabaseTask
from src.services.nlp_analysis_service import NLPAnalysisService
from src.models.feature import Feature
from src.models.feature_activation import FeatureActivation
from src.models.feature_analysis_cache import FeatureAnalysisCache, AnalysisType
from src.models.extraction_job import ExtractionJob
from src.workers.websocket_emitter import emit_progress

logger = logging.getLogger(__name__)


def emit_nlp_analysis_progress(
    extraction_job_id: str,
    event: str,
    data: Dict[str, Any]
) -> None:
    """
    Emit NLP analysis progress via WebSocket.

    Args:
        extraction_job_id: ID of the extraction job being analyzed
        event: Event type ('progress', 'completed', 'failed')
        data: Progress data to emit
    """
    # Frontend expects events prefixed with 'nlp_analysis:'
    prefixed_event = f"nlp_analysis:{event}"
    emit_progress(
        channel=f"nlp_analysis/{extraction_job_id}",
        event=prefixed_event,
        data=data
    )


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.nlp_analysis_tasks.analyze_features_nlp",
    max_retries=3,
    default_retry_delay=60,  # 1-minute back-off between retries
    autoretry_for=(ConnectionError, TimeoutError, OSError),
)
def analyze_features_nlp_task(
    self,
    extraction_job_id: str,
    feature_ids: Optional[List[str]] = None,
    batch_size: int = 100,
    force_reprocess: bool = False
) -> Dict[str, Any]:
    """
    Celery task for computing NLP analysis on feature activation examples.

    This task:
    1. Retrieves all features for an extraction job (or specific feature_ids)
    2. For each feature, retrieves all activation examples
    3. Computes NLP analysis (POS, NER, patterns, clusters)
    4. Stores results directly on Feature.nlp_analysis column (persistent)
    5. Also caches in FeatureAnalysisCache for compatibility
    6. Checks for cancellation between batches and exits cleanly if cancelled

    Args:
        extraction_job_id: ID of the extraction job to analyze
        feature_ids: Optional list of specific feature IDs to analyze
        batch_size: Number of features to process in each batch
        force_reprocess: If True, reprocess all features even if they already have analysis

    Returns:
        Dict with analysis statistics
    """
    logger.info(f"Starting NLP analysis task for extraction {extraction_job_id}")

    with self.get_db() as db:
        try:
            # Verify extraction job exists and is completed
            extraction_job = db.query(ExtractionJob).filter(
                ExtractionJob.id == extraction_job_id
            ).first()

            if not extraction_job:
                raise ValueError(f"Extraction job {extraction_job_id} not found")

            # Update extraction job status to processing
            extraction_job.nlp_status = "processing"
            extraction_job.nlp_progress = 0.0
            extraction_job.nlp_processed_count = 0
            extraction_job.nlp_error_message = None
            db.commit()

            # BIND THE CHILD TO THE PARENT (MIS-E2E-109).
            #
            # The ids branch dropped the extraction scope entirely, while the
            # no-ids branch below scopes correctly. So a caller could POST to a
            # small extraction, pass ids belonging to a large one, and with
            # `force_reprocess: true` overwrite every one of those features'
            # curated `nlp_analysis` and delete its `FeatureAnalysisCache` row —
            # while the progress counters were written onto the PATH extraction.
            # Silent in both directions: the extraction whose data was destroyed
            # shows no activity, and the one showing activity holds none of the
            # results.
            #
            # Of the 11 two-path-parameter routes in this API, all 11 bind child
            # to parent. This is the body-parameter case that was missed.
            if feature_ids:
                features = db.query(Feature).filter(
                    Feature.id.in_(feature_ids),
                    Feature.extraction_job_id == extraction_job_id,
                ).all()

                requested = len(set(feature_ids))
                if len(features) != requested:
                    # Say so rather than silently analysing a subset — a caller
                    # passing foreign ids should learn that, not get a partial
                    # result that looks complete.
                    logger.warning(
                        "NLP analysis for extraction %s: %d of %d requested "
                        "feature ids belong to this extraction; the rest were "
                        "ignored",
                        extraction_job_id, len(features), requested,
                    )
            else:
                features = db.query(Feature).filter(
                    Feature.extraction_job_id == extraction_job_id
                ).order_by(Feature.neuron_index).all()

            total_features = len(features)
            if total_features == 0:
                logger.warning(f"No features found for extraction {extraction_job_id}")
                return {"features_analyzed": 0, "status": "no_features"}

            logger.info(f"Analyzing {total_features} features for extraction {extraction_job_id}")

            # Initialize NLP service
            nlp_service = NLPAnalysisService()

            # Track statistics
            analyzed_count = 0
            cached_count = 0
            error_count = 0
            cache_expiry = timedelta(days=7)

            # Emit initial progress
            emit_nlp_analysis_progress(
                extraction_job_id=extraction_job_id,
                event="progress",
                data={
                    "extraction_job_id": extraction_job_id,
                    "progress": 0.0,
                    "features_analyzed": 0,
                    "total_features": total_features,
                    "status": "analyzing",
                    "message": f"Starting NLP analysis of {total_features} features"
                }
            )

            # Process features in batches
            for batch_start in range(0, total_features, batch_size):
                # Check for cancellation or deletion at the start of each batch
                try:
                    db.refresh(extraction_job)
                except Exception as refresh_error:
                    # Extraction was likely deleted while we were processing
                    logger.warning(f"Extraction {extraction_job_id} may have been deleted: {refresh_error}")
                    return {
                        "status": "aborted",
                        "features_analyzed": analyzed_count,
                        "cached_count": cached_count,
                        "error_count": error_count,
                        "total_features": total_features,
                        "message": "Extraction was deleted during NLP processing"
                    }

                if extraction_job.nlp_status == "cancelled":
                    logger.info(f"NLP analysis cancelled for extraction {extraction_job_id}")
                    emit_nlp_analysis_progress(
                        extraction_job_id=extraction_job_id,
                        event="cancelled",
                        data={
                            "extraction_job_id": extraction_job_id,
                            "progress": analyzed_count / total_features if total_features > 0 else 0,
                            "features_analyzed": analyzed_count,
                            "total_features": total_features,
                            "cached_count": cached_count,
                            "error_count": error_count,
                            "status": "cancelled",
                            "message": f"NLP analysis cancelled. Processed {analyzed_count}/{total_features} features."
                        }
                    )
                    return {
                        "status": "cancelled",
                        "features_analyzed": analyzed_count,
                        "cached_count": cached_count,
                        "error_count": error_count,
                        "total_features": total_features
                    }

                batch_end = min(batch_start + batch_size, total_features)
                batch_features = features[batch_start:batch_end]

                for feature in batch_features:
                    try:
                        # Check if feature already has NLP analysis (skip unless force_reprocess)
                        if not force_reprocess and feature.nlp_analysis is not None and feature.nlp_processed_at is not None:
                            cached_count += 1
                            analyzed_count += 1
                            continue

                        # Retrieve all activation examples for this feature
                        activations = db.query(FeatureActivation).filter(
                            FeatureActivation.feature_id == feature.id
                        ).order_by(FeatureActivation.max_activation.desc()).limit(100).all()

                        if not activations:
                            logger.debug(f"No activations for feature {feature.id}, skipping")
                            analyzed_count += 1
                            continue

                        # Convert activations to example dicts
                        examples = []
                        for act in activations:
                            examples.append({
                                "prefix_tokens": act.prefix_tokens or [],
                                "prime_token": act.prime_token or "",
                                "suffix_tokens": act.suffix_tokens or [],
                                "max_activation": float(act.max_activation)
                            })

                        # Compute NLP analysis
                        analysis_result = nlp_service.analyze_feature(examples, feature.id)

                        now = datetime.now(timezone.utc)

                        # Store directly on Feature model (persistent storage)
                        feature.nlp_analysis = analysis_result
                        feature.nlp_processed_at = now

                        # Also cache for backward compatibility
                        db.query(FeatureAnalysisCache).filter(
                            FeatureAnalysisCache.feature_id == feature.id,
                            FeatureAnalysisCache.analysis_type == AnalysisType.NLP_ANALYSIS
                        ).delete()
                        cache_entry = FeatureAnalysisCache(
                            feature_id=feature.id,
                            analysis_type=AnalysisType.NLP_ANALYSIS,
                            result=analysis_result,
                            computed_at=now,
                            expires_at=now + cache_expiry
                        )
                        db.add(cache_entry)
                        db.commit()

                        analyzed_count += 1

                        # Update extraction job progress
                        extraction_job.nlp_processed_count = analyzed_count
                        extraction_job.nlp_progress = analyzed_count / total_features
                        db.commit()

                    except Exception as e:
                        logger.warning(f"Failed to analyze feature {feature.id}: {e}")
                        error_count += 1
                        analyzed_count += 1
                        db.rollback()

                # Update progress after each batch
                progress = analyzed_count / total_features
                emit_nlp_analysis_progress(
                    extraction_job_id=extraction_job_id,
                    event="progress",
                    data={
                        "extraction_job_id": extraction_job_id,
                        "progress": progress,
                        "features_analyzed": analyzed_count,
                        "total_features": total_features,
                        "cached_count": cached_count,
                        "error_count": error_count,
                        "status": "analyzing",
                        "message": f"Analyzed {analyzed_count}/{total_features} features"
                    }
                )

                logger.info(f"NLP Analysis batch complete: {analyzed_count}/{total_features} features")

            # Update extraction job status to completed
            extraction_job.nlp_status = "completed"
            extraction_job.nlp_progress = 1.0
            extraction_job.nlp_processed_count = analyzed_count
            db.commit()

            # Emit completion
            emit_nlp_analysis_progress(
                extraction_job_id=extraction_job_id,
                event="completed",
                data={
                    "extraction_job_id": extraction_job_id,
                    "progress": 1.0,
                    "features_analyzed": analyzed_count,
                    "total_features": total_features,
                    "cached_count": cached_count,
                    "error_count": error_count,
                    "status": "completed",
                    "message": f"NLP analysis completed for {analyzed_count} features"
                }
            )

            statistics = {
                "features_analyzed": analyzed_count,
                "total_features": total_features,
                "cached_count": cached_count,
                "error_count": error_count,
                "status": "completed"
            }

            logger.info(f"NLP analysis completed for extraction {extraction_job_id}: {statistics}")
            
            # BACKSTOP ONLY. The batch chain is released when the EXTRACTION
            # completes (extraction_service), not here — NLP must never gate the
            # next SAE. This call remains so a batch still advances if that
            # earlier release failed, and it is a no-op in the normal case
            # because _start_next_batch_job claims on celery_task_id IS NULL.
            if extraction_job.batch_id and extraction_job.batch_position:
                _start_next_batch_job(db, extraction_job)
            
            return statistics

        except Exception as e:
            logger.error(
                f"NLP analysis task failed for extraction {extraction_job_id}: {e}",
                exc_info=True
            )

            # Update extraction job status to failed
            try:
                extraction_job.nlp_status = "failed"
                extraction_job.nlp_error_message = str(e)[:500]  # Truncate to 500 chars
                db.commit()
            except Exception:
                db.rollback()

            # Emit failure
            emit_nlp_analysis_progress(
                extraction_job_id=extraction_job_id,
                event="failed",
                data={
                    "extraction_job_id": extraction_job_id,
                    "status": "failed",
                    "error": str(e),
                    "message": f"NLP analysis failed: {e}"
                }
            )

            # Backstop on the failure path too (see the note above).
            try:
                if extraction_job.batch_id and extraction_job.batch_position:
                    _start_next_batch_job(db, extraction_job)
            except Exception as batch_error:
                logger.error(f"Failed to start next batch job after NLP failure: {batch_error}")

            raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.nlp_analysis_tasks.analyze_single_feature_nlp",
    max_retries=3,
    default_retry_delay=60,  # 1-minute back-off between retries
    autoretry_for=(ConnectionError, TimeoutError, OSError),
)
def analyze_single_feature_nlp_task(
    self,
    feature_id: str
) -> Dict[str, Any]:
    """
    Celery task for computing NLP analysis on a single feature.

    Args:
        feature_id: ID of the feature to analyze

    Returns:
        Dict with analysis result
    """
    logger.info(f"Starting NLP analysis for single feature {feature_id}")

    with self.get_db() as db:
        try:
            # Get the feature
            feature = db.query(Feature).filter(Feature.id == feature_id).first()
            if not feature:
                raise ValueError(f"Feature {feature_id} not found")

            # Retrieve all activation examples
            activations = db.query(FeatureActivation).filter(
                FeatureActivation.feature_id == feature_id
            ).order_by(FeatureActivation.max_activation.desc()).limit(100).all()

            if not activations:
                return {"status": "no_activations", "feature_id": feature_id}

            # Convert to example dicts
            examples = []
            for act in activations:
                examples.append({
                    "prefix_tokens": act.prefix_tokens or [],
                    "prime_token": act.prime_token or "",
                    "suffix_tokens": act.suffix_tokens or [],
                    "max_activation": float(act.max_activation)
                })

            # Compute NLP analysis
            nlp_service = NLPAnalysisService()
            analysis_result = nlp_service.analyze_feature(examples, feature_id)

            now = datetime.now(timezone.utc)

            # Store directly on Feature model (persistent storage)
            feature.nlp_analysis = analysis_result
            feature.nlp_processed_at = now

            # Also cache for backward compatibility
            db.query(FeatureAnalysisCache).filter(
                FeatureAnalysisCache.feature_id == feature_id,
                FeatureAnalysisCache.analysis_type == AnalysisType.NLP_ANALYSIS
            ).delete()

            cache_expiry = timedelta(days=7)
            cache_entry = FeatureAnalysisCache(
                feature_id=feature_id,
                analysis_type=AnalysisType.NLP_ANALYSIS,
                result=analysis_result,
                computed_at=now,
                expires_at=now + cache_expiry
            )
            db.add(cache_entry)
            db.commit()

            logger.info(f"NLP analysis completed for feature {feature_id}")
            return {
                "status": "completed",
                "feature_id": feature_id,
                "num_examples": len(examples),
                "analysis": analysis_result
            }

        except Exception as e:
            logger.error(f"NLP analysis failed for feature {feature_id}: {e}", exc_info=True)
            raise


def _start_next_batch_job(db: Session, current_job) -> None:
    """
    Start the next extraction job in a batch.

    Called when the current job's EXTRACTION completes — NLP analysis is
    post-processing and must never gate the next extraction. It used to: the
    chain only advanced from the NLP-completion path, so a batch with auto_nlp
    enabled stalled for the entire NLP pass (measured at 0.72 features/sec, i.e.
    ~12.6 hours for a 32,759-feature SAE) before the next SAE even started.

    IDEMPOTENT: several paths call this (extraction complete, extraction failed,
    NLP complete, NLP-queue failure), so it claims the next job by setting
    celery_task_id under a row lock and only dispatches if it won the claim.
    Without that, two callers would both find the same QUEUED row and dispatch
    the SAME extraction twice.

    Args:
        db: Database session
        current_job: The extraction job that just finished
    """
    from src.models.extraction_job import ExtractionJob, ExtractionStatus
    from src.workers.extraction_tasks import extract_features_from_sae_task
    
    try:
        batch_id = current_job.batch_id
        current_position = current_job.batch_position
        
        logger.info(
            f"Batch {batch_id}: looking for the next queued job after position "
            f"{current_position}"
        )
        
        # Find the next job in the batch and CLAIM it under a row lock.
        # celery_task_id IS NULL is the claim flag: a job that already has one
        # has been dispatched, so a second caller must not dispatch it again.
        # THE NEXT ONE BY ORDER, NOT position + 1 (MIS-E2E-066).
        #
        # `batch_position` comes from `enumerate` over the REQUESTED SAEs, so a
        # skipped one — already extracted, invalid — leaves a GAP. Demanding an
        # exact `current + 1` then found nothing and the whole tail of the batch
        # was stranded: those jobs sat QUEUED until the 3-hour reaper closed
        # them with a "crashed worker" message, which is not what happened. The
        # diagnosis handed to the user pointed at infrastructure for a dispatch
        # arithmetic bug.
        next_job = db.query(ExtractionJob).filter(
            ExtractionJob.batch_id == batch_id,
            ExtractionJob.batch_position > current_position,
            ExtractionJob.status == ExtractionStatus.QUEUED.value,
            ExtractionJob.celery_task_id.is_(None),
        ).order_by(ExtractionJob.batch_position).with_for_update(
            skip_locked=True
        ).first()

        if next_job:
            # Get the config from the job
            config = next_job.config or {}
            sae_id = next_job.external_sae_id
            
            soft_time_limit = config.get("soft_time_limit", 144000)
            time_limit = config.get("time_limit", 172800)
            
            # Queue the task to Celery
            task_result = extract_features_from_sae_task.apply_async(
                args=(sae_id, config),
                soft_time_limit=soft_time_limit,
                time_limit=time_limit
            )
            
            next_job.celery_task_id = task_result.id
            db.commit()
            
            logger.info(
                f"Batch {batch_id}: Started next job {next_job.id} for SAE {sae_id} "
                f"(position {next_job.batch_position}/{next_job.batch_total})"
            )
        else:
            logger.info(f"Batch {batch_id}: No more jobs to process (completed at position {current_position})")
            
    except Exception as e:
        logger.error(f"Error starting next batch job: {e}", exc_info=True)
        # Don't fail the current task - just log the error


