"""
Celery tasks for NLP analysis of feature activation examples.

These tasks run asynchronously to pre-compute NLP analysis (POS tagging, NER,
context patterns, semantic clusters) for features after extraction completes.
"""

import logging
from typing import Dict, Any, List, Optional
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
    emit_progress(
        channel=f"nlp_analysis/{extraction_job_id}",
        event=event,
        data=data
    )


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.nlp_analysis_tasks.analyze_features_nlp",
    max_retries=0,
    autoretry_for=(),
)
def analyze_features_nlp_task(
    self,
    extraction_job_id: str,
    feature_ids: Optional[List[str]] = None,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Celery task for computing NLP analysis on feature activation examples.

    This task:
    1. Retrieves all features for an extraction job (or specific feature_ids)
    2. For each feature, retrieves all activation examples
    3. Computes NLP analysis (POS, NER, patterns, clusters)
    4. Caches results in FeatureAnalysisCache

    Args:
        extraction_job_id: ID of the extraction job to analyze
        feature_ids: Optional list of specific feature IDs to analyze
        batch_size: Number of features to process in each batch

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

            # Get features to analyze
            if feature_ids:
                features = db.query(Feature).filter(
                    Feature.id.in_(feature_ids)
                ).all()
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
                batch_end = min(batch_start + batch_size, total_features)
                batch_features = features[batch_start:batch_end]

                for feature in batch_features:
                    try:
                        # Check if analysis already cached and not expired
                        existing_cache = db.query(FeatureAnalysisCache).filter(
                            FeatureAnalysisCache.feature_id == feature.id,
                            FeatureAnalysisCache.analysis_type == AnalysisType.NLP_ANALYSIS,
                            FeatureAnalysisCache.expires_at > datetime.now(timezone.utc)
                        ).first()

                        if existing_cache:
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

                        # Delete any existing cache entry
                        db.query(FeatureAnalysisCache).filter(
                            FeatureAnalysisCache.feature_id == feature.id,
                            FeatureAnalysisCache.analysis_type == AnalysisType.NLP_ANALYSIS
                        ).delete()

                        # Cache the result
                        now = datetime.now(timezone.utc)
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
            return statistics

        except Exception as e:
            logger.error(
                f"NLP analysis task failed for extraction {extraction_job_id}: {e}",
                exc_info=True
            )

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

            raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="src.workers.nlp_analysis_tasks.analyze_single_feature_nlp",
    max_retries=0,
    autoretry_for=(),
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

            # Delete any existing cache entry
            db.query(FeatureAnalysisCache).filter(
                FeatureAnalysisCache.feature_id == feature_id,
                FeatureAnalysisCache.analysis_type == AnalysisType.NLP_ANALYSIS
            ).delete()

            # Cache the result
            now = datetime.now(timezone.utc)
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
