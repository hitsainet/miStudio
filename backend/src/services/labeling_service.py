"""
Feature labeling service for semantic labeling of SAE features.

This service manages semantic labeling of features extracted from SAE models.
Labeling is independent from extraction, allowing re-labeling without re-extraction.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import desc, select
from collections import defaultdict
import asyncio

from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.labeling_job import LabelingJob, LabelingStatus, LabelingMethod
from src.models.feature import Feature, LabelSource
from src.models.feature_activation import FeatureActivation
from src.core.config import settings
from src.services.local_labeling_service import LocalLabelingService
from src.services.openai_labeling_service import OpenAILabelingService
from src.workers.websocket_emitter import emit_labeling_progress, emit_labeling_result
from src.utils.token_filters import filter_token_stats

logger = logging.getLogger(__name__)


def create_example_tokens_summary(
    token_stats: Dict[str, Dict],
    filter_special: bool = True,
    filter_single_char: bool = True,
    filter_punctuation: bool = True,
    filter_numbers: bool = True,
    filter_fragments: bool = True,
    filter_stop_words: bool = False,
    top_n: int = 7
) -> Optional[Dict]:
    """
    Create example tokens summary from token statistics with filtering.

    Args:
        token_stats: Dict mapping token to {'count': N, 'total_activation': X}
        filter_*: Token filtering flags
        top_n: Number of top tokens to include (default 7)

    Returns:
        Dict with keys: 'tokens', 'counts', 'activations', 'max_activation'
        Returns None if no tokens remain after filtering
    """
    # Apply filters to token_stats
    filtered_stats = filter_token_stats(
        token_stats,
        filter_special=filter_special,
        filter_single_char=filter_single_char,
        filter_punctuation=filter_punctuation,
        filter_numbers=filter_numbers,
        filter_fragments=filter_fragments,
        filter_stop_words=filter_stop_words
    )

    if not filtered_stats:
        return None

    # Sort by count descending
    sorted_tokens = sorted(
        filtered_stats.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:top_n]

    # Extract tokens, counts, and average activations
    tokens = []
    counts = []
    activations = []

    for token, stats in sorted_tokens:
        tokens.append(token)
        counts.append(stats['count'])
        # Calculate average activation: total_activation / count
        avg_activation = stats['total_activation'] / stats['count'] if stats['count'] > 0 else 0.0
        activations.append(float(avg_activation))

    max_activation = max(activations) if activations else 0.0

    return {
        'tokens': tokens,
        'counts': counts,
        'activations': activations,
        'max_activation': float(max_activation)
    }


class LabelingService:
    """
    Service for semantic labeling of SAE features.

    Manages the feature labeling workflow:
    1. Create labeling job for an extraction
    2. Fetch features and their activations
    3. Aggregate token statistics for each feature
    4. Generate semantic labels using OpenAI or local LLM
    5. Update feature names and track labeling job
    6. Emit WebSocket progress events
    """

    def __init__(self, db: Union[AsyncSession, Session]):
        """Initialize labeling service with either async or sync session."""
        self.db = db
        self.is_async = isinstance(db, AsyncSession)

    async def start_labeling(
        self,
        extraction_job_id: str,
        config: Dict[str, Any]
    ) -> LabelingJob:
        """
        Start a feature labeling job for a completed extraction.

        Args:
            extraction_job_id: ID of the extraction to label features from
            config: Labeling configuration (labeling_method, openai_model, prompt_template_id, etc.)

        Returns:
            LabelingJob: Created labeling job record

        Raises:
            ValueError: If extraction not found, not completed, or active labeling exists
        """
        from sqlalchemy import func

        # Validate extraction exists and is completed
        result = await self.db.execute(
            select(ExtractionJob).where(ExtractionJob.id == extraction_job_id)
        )
        extraction_job = result.scalar_one_or_none()

        if not extraction_job:
            raise ValueError(f"Extraction job {extraction_job_id} not found")

        if extraction_job.status != ExtractionStatus.COMPLETED.value:
            raise ValueError(
                f"Extraction {extraction_job_id} must be completed before labeling "
                f"(current status: {extraction_job.status})"
            )

        # Check for active labeling on this extraction
        result = await self.db.execute(
            select(LabelingJob).where(
                LabelingJob.extraction_job_id == extraction_job_id,
                LabelingJob.status.in_([
                    LabelingStatus.QUEUED.value,
                    LabelingStatus.LABELING.value
                ])
            )
        )
        active_labeling = result.scalar_one_or_none()

        if active_labeling:
            raise ValueError(
                f"Extraction {extraction_job_id} already has an active labeling job: "
                f"{active_labeling.id}"
            )

        # Count features to label
        count_result = await self.db.execute(
            select(func.count()).select_from(Feature).where(
                Feature.extraction_job_id == extraction_job_id
            )
        )
        total_features = count_result.scalar_one()

        if total_features == 0:
            raise ValueError(f"Extraction {extraction_job_id} has no features to label")

        # Create labeling job ID: label_{extraction_id}_{timestamp}
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = f"label_{extraction_job_id}_{timestamp}"

        # Create labeling job record
        labeling_job = LabelingJob(
            id=job_id,
            extraction_job_id=extraction_job_id,
            labeling_method=config.get("labeling_method", "openai"),
            openai_model=config.get("openai_model"),
            openai_api_key=config.get("openai_api_key"),
            openai_compatible_endpoint=config.get("openai_compatible_endpoint"),
            openai_compatible_model=config.get("openai_compatible_model"),
            local_model=config.get("local_model"),
            prompt_template_id=config.get("prompt_template_id"),
            filter_special=config.get("filter_special", True),
            filter_single_char=config.get("filter_single_char", True),
            filter_punctuation=config.get("filter_punctuation", True),
            filter_numbers=config.get("filter_numbers", True),
            filter_fragments=config.get("filter_fragments", True),
            filter_stop_words=config.get("filter_stop_words", False),
            save_requests_for_testing=config.get("save_requests_for_testing", False),
            export_format=config.get("export_format", "both"),
            save_poor_quality_labels=config.get("save_poor_quality_labels", False),
            poor_quality_sample_rate=config.get("poor_quality_sample_rate", 1.0),
            status=LabelingStatus.QUEUED.value,
            progress=0.0,
            features_labeled=0,
            total_features=total_features,
            statistics={
                "max_examples": config.get("max_examples")  # Store example count override (None = use template default)
            }
        )

        self.db.add(labeling_job)
        await self.db.commit()
        await self.db.refresh(labeling_job)

        logger.info(
            f"Created labeling job {job_id} for extraction {extraction_job_id} "
            f"with {total_features} features using method: {labeling_job.labeling_method}"
        )

        return labeling_job

    async def _retrieve_top_examples_batch(
        self,
        session: AsyncSession,
        feature_ids: List[str],
        max_examples: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve top-K activation examples for a batch of features.

        Uses PostgreSQL window function (ROW_NUMBER() OVER) to efficiently
        get the top K examples per feature, ordered by max_activation DESC.

        Args:
            session: Async database session
            feature_ids: List of feature IDs to retrieve examples for
            max_examples: Maximum number of examples per feature (K value)

        Returns:
            Dict mapping feature_id to list of example dicts:
            {
                "feature_id_1": [
                    {
                        "sample_index": 123,
                        "max_activation": 0.85,
                        "prefix_tokens": ["token", "sequence"],
                        "prime_token": "prime",
                        "suffix_tokens": ["more", "tokens"],
                        "prime_activation_index": 2,
                        "activations": [0.1, 0.2, 0.85, 0.3],
                        "tokens": ["token", "sequence", "prime", "more"]  # legacy fallback
                    },
                    ...
                ],
                ...
            }
        """
        from sqlalchemy import text

        if not feature_ids:
            return {}

        # SQL query using ROW_NUMBER() window function to get top-K per feature
        query = text("""
            WITH ranked_examples AS (
                SELECT
                    fa.feature_id,
                    fa.sample_index,
                    fa.max_activation,
                    fa.prefix_tokens,
                    fa.prime_token,
                    fa.suffix_tokens,
                    fa.prime_activation_index,
                    fa.activations,
                    fa.tokens,
                    ROW_NUMBER() OVER (
                        PARTITION BY fa.feature_id
                        ORDER BY fa.max_activation DESC, fa.id ASC
                    ) as rank
                FROM feature_activations fa
                WHERE fa.feature_id = ANY(:feature_ids)
            )
            SELECT
                feature_id,
                sample_index,
                max_activation,
                prefix_tokens,
                prime_token,
                suffix_tokens,
                prime_activation_index,
                activations,
                tokens
            FROM ranked_examples
            WHERE rank <= :max_examples
            ORDER BY feature_id, rank;
        """)

        result = await session.execute(
            query,
            {"feature_ids": feature_ids, "max_examples": max_examples}
        )

        # Group examples by feature_id
        examples_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in result:
            feature_id = row.feature_id
            if feature_id not in examples_map:
                examples_map[feature_id] = []

            examples_map[feature_id].append({
                "sample_index": row.sample_index,
                "max_activation": float(row.max_activation),
                "prefix_tokens": row.prefix_tokens or [],
                "prime_token": row.prime_token or "",
                "suffix_tokens": row.suffix_tokens or [],
                "prime_activation_index": row.prime_activation_index,
                "activations": row.activations or [],
                "tokens": row.tokens or []  # legacy fallback
            })

        return examples_map

    def _retrieve_top_examples_batch_sync(
        self,
        session: Session,
        feature_ids: List[str],
        max_examples: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Synchronous version: Retrieve top-K activation examples for a batch of features.

        Uses PostgreSQL window function (ROW_NUMBER() OVER) to efficiently
        get the top K examples per feature, ordered by max_activation DESC.

        Args:
            session: Sync database session
            feature_ids: List of feature IDs to retrieve examples for
            max_examples: Maximum number of examples per feature (K value)

        Returns:
            Dict mapping feature_id to list of example dicts (same format as async version)
        """
        from sqlalchemy import text

        if not feature_ids:
            return {}

        # Same SQL query as async version
        query = text("""
            WITH ranked_examples AS (
                SELECT
                    fa.feature_id,
                    fa.sample_index,
                    fa.max_activation,
                    fa.prefix_tokens,
                    fa.prime_token,
                    fa.suffix_tokens,
                    fa.prime_activation_index,
                    fa.activations,
                    fa.tokens,
                    ROW_NUMBER() OVER (
                        PARTITION BY fa.feature_id
                        ORDER BY fa.max_activation DESC, fa.id ASC
                    ) as rank
                FROM feature_activations fa
                WHERE fa.feature_id = ANY(:feature_ids)
            )
            SELECT
                feature_id,
                sample_index,
                max_activation,
                prefix_tokens,
                prime_token,
                suffix_tokens,
                prime_activation_index,
                activations,
                tokens
            FROM ranked_examples
            WHERE rank <= :max_examples
            ORDER BY feature_id, rank;
        """)

        # Synchronous execute (no await)
        result = session.execute(
            query,
            {"feature_ids": feature_ids, "max_examples": max_examples}
        )

        # Group examples by feature_id (same logic as async version)
        examples_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in result:
            feature_id = row.feature_id
            if feature_id not in examples_map:
                examples_map[feature_id] = []

            examples_map[feature_id].append({
                "sample_index": row.sample_index,
                "max_activation": float(row.max_activation),
                "prefix_tokens": row.prefix_tokens or [],
                "prime_token": row.prime_token or "",
                "suffix_tokens": row.suffix_tokens or [],
                "prime_activation_index": row.prime_activation_index,
                "activations": row.activations or [],
                "tokens": row.tokens or []  # legacy fallback
            })

        return examples_map

    async def _retrieve_bottom_examples_batch(
        self,
        session: AsyncSession,
        feature_ids: List[str],
        num_negative_examples: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve bottom-K activation examples (negative examples) for a batch of features.

        These are examples where the feature has LOW activation, useful for distinguishing
        what the feature does NOT respond to. This helps the LLM understand the feature's
        boundaries and avoid overgeneralization.

        Uses PostgreSQL window function (ROW_NUMBER() OVER) to efficiently
        get the bottom K examples per feature, ordered by max_activation ASC.

        Args:
            session: Async database session
            feature_ids: List of feature IDs to retrieve negative examples for
            num_negative_examples: Number of low-activation examples per feature (default: 5)

        Returns:
            Dict mapping feature_id to list of negative example dicts (same format as positive examples)
        """
        from sqlalchemy import text

        if not feature_ids or num_negative_examples <= 0:
            return {}

        # SQL query using ROW_NUMBER() window function to get bottom-K per feature
        # Note: We order by max_activation ASC to get the LOWEST activations
        query = text("""
            WITH ranked_examples AS (
                SELECT
                    fa.feature_id,
                    fa.sample_index,
                    fa.max_activation,
                    fa.prefix_tokens,
                    fa.prime_token,
                    fa.suffix_tokens,
                    fa.prime_activation_index,
                    fa.activations,
                    fa.tokens,
                    ROW_NUMBER() OVER (
                        PARTITION BY fa.feature_id
                        ORDER BY fa.max_activation ASC, fa.id ASC
                    ) as rank
                FROM feature_activations fa
                WHERE fa.feature_id = ANY(:feature_ids)
            )
            SELECT
                feature_id,
                sample_index,
                max_activation,
                prefix_tokens,
                prime_token,
                suffix_tokens,
                prime_activation_index,
                activations,
                tokens
            FROM ranked_examples
            WHERE rank <= :num_negative_examples
            ORDER BY feature_id, rank;
        """)

        result = await session.execute(
            query,
            {"feature_ids": feature_ids, "num_negative_examples": num_negative_examples}
        )

        # Group negative examples by feature_id
        examples_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in result:
            feature_id = row.feature_id
            if feature_id not in examples_map:
                examples_map[feature_id] = []

            examples_map[feature_id].append({
                "sample_index": row.sample_index,
                "max_activation": float(row.max_activation),
                "prefix_tokens": row.prefix_tokens or [],
                "prime_token": row.prime_token or "",
                "suffix_tokens": row.suffix_tokens or [],
                "prime_activation_index": row.prime_activation_index,
                "activations": row.activations or [],
                "tokens": row.tokens or []  # legacy fallback
            })

        return examples_map

    def _retrieve_bottom_examples_batch_sync(
        self,
        session: Session,
        feature_ids: List[str],
        num_negative_examples: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Synchronous version: Retrieve bottom-K activation examples (negative examples).

        These are examples where the feature has LOW activation, useful for distinguishing
        what the feature does NOT respond to.

        Args:
            session: Sync database session
            feature_ids: List of feature IDs to retrieve negative examples for
            num_negative_examples: Number of low-activation examples per feature (default: 5)

        Returns:
            Dict mapping feature_id to list of negative example dicts (same format as async version)
        """
        from sqlalchemy import text

        if not feature_ids or num_negative_examples <= 0:
            return {}

        # Same SQL query as async version (order by ASC for lowest activations)
        query = text("""
            WITH ranked_examples AS (
                SELECT
                    fa.feature_id,
                    fa.sample_index,
                    fa.max_activation,
                    fa.prefix_tokens,
                    fa.prime_token,
                    fa.suffix_tokens,
                    fa.prime_activation_index,
                    fa.activations,
                    fa.tokens,
                    ROW_NUMBER() OVER (
                        PARTITION BY fa.feature_id
                        ORDER BY fa.max_activation ASC, fa.id ASC
                    ) as rank
                FROM feature_activations fa
                WHERE fa.feature_id = ANY(:feature_ids)
            )
            SELECT
                feature_id,
                sample_index,
                max_activation,
                prefix_tokens,
                prime_token,
                suffix_tokens,
                prime_activation_index,
                activations,
                tokens
            FROM ranked_examples
            WHERE rank <= :num_negative_examples
            ORDER BY feature_id, rank;
        """)

        # Synchronous execute (no await)
        result = session.execute(
            query,
            {"feature_ids": feature_ids, "num_negative_examples": num_negative_examples}
        )

        # Group negative examples by feature_id (same logic as async version)
        examples_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in result:
            feature_id = row.feature_id
            if feature_id not in examples_map:
                examples_map[feature_id] = []

            examples_map[feature_id].append({
                "sample_index": row.sample_index,
                "max_activation": float(row.max_activation),
                "prefix_tokens": row.prefix_tokens or [],
                "prime_token": row.prime_token or "",
                "suffix_tokens": row.suffix_tokens or [],
                "prime_activation_index": row.prime_activation_index,
                "activations": row.activations or [],
                "tokens": row.tokens or []  # legacy fallback
            })

        return examples_map

    def label_features_for_extraction(
        self,
        labeling_job_id: str
    ) -> Dict[str, Any]:
        """
        Execute semantic labeling for features from an extraction job.

        This is the core labeling logic that:
        1. Fetches features and their activations
        2. Aggregates token statistics for each feature (using efficient SQL batching)
        3. Generates semantic labels using specified method
        4. Updates feature names and tracks progress
        5. Calculates statistics and marks job complete

        Args:
            labeling_job_id: ID of the labeling job to execute

        Returns:
            Dict with labeling statistics

        Raises:
            ValueError: If labeling job not found or extraction invalid
        """
        # Fetch labeling job
        labeling_job = self.db.query(LabelingJob).filter(
            LabelingJob.id == labeling_job_id
        ).first()

        if not labeling_job:
            raise ValueError(f"Labeling job {labeling_job_id} not found")

        # Update status to labeling
        labeling_job.status = LabelingStatus.LABELING.value
        labeling_job.updated_at = datetime.now(timezone.utc)
        self.db.commit()

        start_time = datetime.now(timezone.utc)

        try:
            # Fetch extraction job
            extraction_job = self.db.query(ExtractionJob).filter(
                ExtractionJob.id == labeling_job.extraction_job_id
            ).first()

            if not extraction_job:
                raise ValueError(f"Extraction job {labeling_job.extraction_job_id} not found")

            # Fetch all features for this extraction
            all_features = self.db.query(Feature).filter(
                Feature.extraction_job_id == labeling_job.extraction_job_id
            ).order_by(Feature.neuron_index).all()

            if not all_features:
                raise ValueError(f"No features found for extraction {labeling_job.extraction_job_id}")

            # Pre-labeling feature filtering (DISABLED - requires refactoring for context-based approach)
            # TODO: Refactor FeatureFilter to work with activation examples instead of token aggregation.
            # The old token-based filtering is incompatible with the new context-based labeling system.
            # For now, we label all features without pre-filtering.
            logger.info("Pre-labeling filter disabled - labeling all features (context-based approach)")
            features = all_features

            total_features = len(features)
            logger.info(f"Labeling {total_features} features for extraction {labeling_job.extraction_job_id}")

            # Fetch template configuration if specified
            template_config = None
            max_examples = 10  # Default for miStudio Internal
            if labeling_job.prompt_template_id:
                from src.models.labeling_prompt_template import LabelingPromptTemplate
                template = self.db.query(LabelingPromptTemplate).filter(
                    LabelingPromptTemplate.id == labeling_job.prompt_template_id
                ).first()
                if template:
                    # Check for job-level max_examples override in statistics
                    job_max_examples = None
                    if labeling_job.statistics and isinstance(labeling_job.statistics, dict):
                        job_max_examples = labeling_job.statistics.get('max_examples')

                    # Use job override if provided, otherwise use template default
                    max_examples = job_max_examples if job_max_examples is not None else template.max_examples

                    template_config = {
                        'template_type': template.template_type,
                        'max_examples': max_examples,  # Use resolved value (job override or template default)
                        'include_prefix': template.include_prefix,
                        'include_suffix': template.include_suffix,
                        'prime_token_marker': template.prime_token_marker,
                        'include_logit_effects': template.include_logit_effects,
                        'top_promoted_tokens_count': template.top_promoted_tokens_count,
                        'top_suppressed_tokens_count': template.top_suppressed_tokens_count,
                        'is_detection_template': template.is_detection_template
                    }

                    override_msg = f" (job override)" if job_max_examples is not None else ""
                    logger.info(f"Using template: {template.name} (type: {template.template_type}, K={max_examples}{override_msg})")

            # Provide default template_config if no template was specified or found
            if template_config is None:
                template_config = {
                    'template_type': 'mistudio_context',
                    'max_examples': max_examples,
                    'include_prefix': True,
                    'include_suffix': True,
                    'prime_token_marker': '>>>',
                    'include_logit_effects': False,
                    'top_promoted_tokens_count': 10,
                    'top_suppressed_tokens_count': 10,
                    'is_detection_template': False
                }
                logger.info(f"No template specified - using default mistudio_context (K={max_examples})")

            # Retrieve activation examples using efficient SQL batching
            # For NLP analysis, we need ALL examples (up to 100)
            # For display in LLM prompt, we only show top max_examples (default 10)
            BATCH_SIZE = 1000
            NLP_ANALYSIS_EXAMPLES = 100  # Retrieve all examples for comprehensive NLP analysis
            features_examples = []  # Top max_examples for LLM display
            all_features_examples = []  # All examples for NLP analysis
            neuron_indices = []

            # Phase 1: Examples Retrieval with progress tracking
            logger.info(f"Starting examples retrieval phase for {total_features} features (display K={max_examples}, NLP K={NLP_ANALYSIS_EXAMPLES}) in batches of {BATCH_SIZE}")

            for batch_start in range(0, total_features, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_features)
                batch_features = features[batch_start:batch_end]
                batch_size = len(batch_features)

                logger.info(f"Retrieving batch {batch_start//BATCH_SIZE + 1}/{(total_features + BATCH_SIZE - 1)//BATCH_SIZE}: features {batch_start+1}-{batch_end}")

                # Get feature IDs for this batch
                batch_feature_ids = [f.id for f in batch_features]

                # Retrieve ALL examples (up to 100) for NLP analysis
                # Use sync version since Celery worker uses sync session
                all_examples_map = self._retrieve_top_examples_batch_sync(
                    session=self.db,
                    feature_ids=batch_feature_ids,
                    max_examples=NLP_ANALYSIS_EXAMPLES
                )

                # Build ordered lists for labeling (maintain feature order)
                for feature in batch_features:
                    all_examples = all_examples_map.get(feature.id, [])
                    # Store all examples for NLP analysis
                    all_features_examples.append(all_examples)
                    # Store only top max_examples for LLM display
                    features_examples.append(all_examples[:max_examples])
                    neuron_indices.append(feature.neuron_index)

                # Update progress in database
                retrieval_progress = batch_end / total_features
                labeling_job.progress = retrieval_progress * 0.3  # Retrieval is ~30% of total work
                labeling_job.updated_at = datetime.now(timezone.utc)
                self.db.commit()

                # Emit WebSocket progress update
                emit_labeling_progress(
                    labeling_job_id=labeling_job.id,
                    event="labeling:progress",
                    data={
                        "labeling_job_id": labeling_job.id,
                        "extraction_job_id": labeling_job.extraction_job_id,
                        "progress": labeling_job.progress,
                        "features_labeled": 0,
                        "total_features": total_features,
                        "status": "labeling",
                        "phase": "examples_retrieval",
                        "message": f"Retrieved top-{max_examples} examples for {batch_end}/{total_features} features"
                    }
                )

                logger.info(f"Batch {batch_start//BATCH_SIZE + 1} complete: {batch_end}/{total_features} features processed ({retrieval_progress*100:.1f}%)")

            logger.info(f"Examples retrieval complete for {len(features_examples)} features (K={max_examples})")

            # Phase 2: Label Generation with progress tracking
            logger.info("Starting label generation phase")

            # Define progress callback for label generation
            def labeling_progress_callback(current: int, total: int):
                """
                Callback for label generation progress.
                Updates database and emits WebSocket events.

                Args:
                    current: Number of features labeled so far
                    total: Total number of features to label
                """
                # Calculate progress: aggregation was 0-30%, labeling is 30-100%
                labeling_progress = current / total if total > 0 else 0
                overall_progress = 0.3 + (labeling_progress * 0.7)

                # Update database
                labeling_job.progress = overall_progress
                labeling_job.features_labeled = current
                labeling_job.updated_at = datetime.now(timezone.utc)
                self.db.commit()

                # Emit WebSocket progress
                emit_labeling_progress(
                    labeling_job_id=labeling_job.id,
                    event="labeling:progress",
                    data={
                        "labeling_job_id": labeling_job.id,
                        "extraction_job_id": labeling_job.extraction_job_id,
                        "progress": overall_progress,
                        "features_labeled": current,
                        "total_features": total_features,
                        "status": "labeling",
                        "phase": "labeling",
                        "message": f"Generated labels for {current}/{total_features} features"
                    }
                )

            # Initialize appropriate labeling service
            labeling_method = labeling_job.labeling_method
            labels = []

            try:
                if labeling_method == LabelingMethod.LOCAL.value:
                    local_model = labeling_job.local_model or "meta-llama/Llama-3.2-1B"
                    logger.info(f"Initializing local labeling service with model: {local_model}")
                    labeling_service = LocalLabelingService(model_name=local_model)

                    # Load model once for the entire job
                    labeling_service.load_model()

                    try:
                        # Generate and persist labels in batches using context examples
                        # This ensures progress is saved incrementally if the job fails
                        label_source_value = LabelSource.LOCAL_LLM.value
                        labeled_at = datetime.now(timezone.utc)
                        LABEL_BATCH_SIZE = 10

                        logger.info(f"Starting incremental labeling: {total_features} features in batches of {LABEL_BATCH_SIZE}")

                        for batch_start in range(0, total_features, LABEL_BATCH_SIZE):
                            batch_end = min(batch_start + LABEL_BATCH_SIZE, total_features)
                            batch_features = features[batch_start:batch_end]
                            batch_examples = features_examples[batch_start:batch_end]
                            batch_all_examples = all_features_examples[batch_start:batch_end]

                            # Generate labels for this batch (model already loaded)
                            # LOCAL service uses synchronous generation, not async
                            # Pass all examples for NLP analysis to improve labeling
                            batch_labels = []
                            for feature, examples, all_examples in zip(batch_features, batch_examples, batch_all_examples):
                                label = labeling_service.generate_label(
                                    examples=examples,
                                    neuron_index=feature.neuron_index,
                                    feature_id=feature.id,
                                    all_examples=all_examples,  # Pass full 100 examples for NLP analysis
                                    nlp_analysis=feature.nlp_analysis  # Use pre-computed NLP if available
                                )
                                batch_labels.append(label)

                            # Persist this batch immediately
                            for feature, label, examples in zip(batch_features, batch_labels, batch_examples):
                                feature.category = label["category"]
                                feature.name = label["specific"]
                                feature.description = label.get("description", "")
                                feature.label_source = label_source_value
                                feature.labeling_job_id = labeling_job.id
                                feature.labeled_at = labeled_at
                                feature.updated_at = labeled_at

                                # Create example tokens summary from context examples (first 7 prime tokens)
                                prime_tokens = [ex.get('prime_token', '') for ex in examples[:7] if ex.get('prime_token')]
                                example_summary = ', '.join(prime_tokens) if prime_tokens else ''
                                feature.example_tokens_summary = example_summary

                                # Emit individual result for real-time display
                                # Send first 10 full examples with prefix/prime/suffix context
                                example_data = []
                                for ex in examples[:10]:
                                    example_data.append({
                                        "prefix_tokens": ex.get('prefix_tokens', []),
                                        "prime_token": ex.get('prime_token', ''),
                                        "suffix_tokens": ex.get('suffix_tokens', []),
                                        "max_activation": ex.get('max_activation', 0.0)
                                    })

                                emit_labeling_result(
                                    labeling_job_id=labeling_job.id,
                                    feature_data={
                                        "feature_id": feature.neuron_index,
                                        "label": feature.name,
                                        "category": feature.category,
                                        "description": feature.description or "",
                                        "examples": example_data
                                    }
                                )

                            # Commit this batch
                            self.db.commit()

                            # Update progress
                            current_labeled = batch_end
                            labeling_progress_callback(current_labeled, total_features)

                            logger.info(f"Batch {batch_start//LABEL_BATCH_SIZE + 1}/{(total_features + LABEL_BATCH_SIZE - 1)//LABEL_BATCH_SIZE}: Labeled and persisted features {batch_start+1}-{batch_end}/{total_features}")

                        logger.info(f"All {total_features} features labeled and persisted successfully")

                        # Create labels list for statistics calculation (now we need to query back from DB)
                        labels = [{"category": f.category, "specific": f.name} for f in features]

                    finally:
                        # Always unload model to free GPU memory
                        logger.info("Unloading local labeling model from GPU memory")
                        labeling_service.unload_model()

                elif labeling_method == LabelingMethod.OPENAI.value:
                    # Get API key from labeling job, fallback to settings if invalid/missing
                    openai_api_key = labeling_job.openai_api_key

                    # Validate API key - if None, empty, or looks corrupted, use settings
                    if not openai_api_key or len(openai_api_key) > 200 or any(ord(c) > 127 for c in openai_api_key):
                        logger.warning(f"Invalid/missing OpenAI API key in labeling job, using settings.openai_api_key")
                        openai_api_key = getattr(settings, 'openai_api_key', None)

                    if not openai_api_key:
                        raise ValueError("OpenAI API key not provided and not found in settings")

                    openai_model = labeling_job.openai_model or "gpt-4o-mini"

                    # Fetch prompt template if specified
                    system_message = None
                    user_prompt_template = None
                    temperature = 0.3
                    max_tokens = 50
                    top_p = 0.9

                    if labeling_job.prompt_template_id:
                        from src.models.labeling_prompt_template import LabelingPromptTemplate
                        template = self.db.query(LabelingPromptTemplate).filter(
                            LabelingPromptTemplate.id == labeling_job.prompt_template_id
                        ).first()
                        if template:
                            system_message = template.system_message
                            user_prompt_template = template.user_prompt_template
                            temperature = template.temperature
                            max_tokens = template.max_tokens
                            top_p = template.top_p
                            logger.info(f"Using prompt template: {template.name} (ID: {template.id})")

                    # Use default API timeout (120s)
                    # TODO: Add api_timeout column to labeling_jobs table for configurable timeout
                    api_timeout = 120.0

                    logger.info(f"Initializing OpenAI labeling service with model: {openai_model}")
                    labeling_service = OpenAILabelingService(
                        api_key=openai_api_key,
                        model=openai_model,
                        system_message=system_message,
                        user_prompt_template=user_prompt_template,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        timeout=api_timeout,
                        filter_special=labeling_job.filter_special,
                        filter_single_char=labeling_job.filter_single_char,
                        filter_punctuation=labeling_job.filter_punctuation,
                        filter_numbers=labeling_job.filter_numbers,
                        filter_fragments=labeling_job.filter_fragments,
                        filter_stop_words=labeling_job.filter_stop_words,
                        save_requests_for_testing=labeling_job.save_requests_for_testing,
                        export_format=labeling_job.export_format,
                        save_poor_quality_labels=labeling_job.save_poor_quality_labels,
                        poor_quality_sample_rate=labeling_job.poor_quality_sample_rate,
                        save_requests_sample_rate=labeling_job.save_requests_sample_rate,
                        labeling_job_id=labeling_job.id
                    )

                    # Generate and persist labels in batches of 10
                    # This ensures progress is saved incrementally if the job fails
                    label_source_value = LabelSource.OPENAI.value
                    labeled_at = datetime.now(timezone.utc)
                    LABEL_BATCH_SIZE = 10

                    logger.info(f"Starting incremental labeling: {total_features} features in batches of {LABEL_BATCH_SIZE}")

                    for batch_start in range(0, total_features, LABEL_BATCH_SIZE):
                        batch_end = min(batch_start + LABEL_BATCH_SIZE, total_features)
                        batch_features = features[batch_start:batch_end]
                        batch_examples = features_examples[batch_start:batch_end]
                        batch_all_examples = all_features_examples[batch_start:batch_end]

                        # Generate labels for this batch using context-based examples
                        # Create concurrent tasks for all features in batch
                        # Pass all examples for NLP analysis to improve labeling
                        label_tasks = []
                        for feature, examples, all_ex in zip(batch_features, batch_examples, batch_all_examples):
                            task = labeling_service.generate_label_from_examples(
                                examples=examples,
                                template_config=template_config,
                                user_prompt_template=user_prompt_template,
                                system_message=system_message,
                                feature_id=feature.id,
                                neuron_index=feature.neuron_index,
                                logit_effects=None,  # TODO: Implement in Sprint 4
                                all_examples=all_ex,  # Pass full 100 examples for NLP analysis
                                nlp_analysis=feature.nlp_analysis  # Use pre-computed NLP if available
                            )
                            label_tasks.append(task)

                        # Execute all labeling tasks concurrently
                        # Wrap gather in an async function for asyncio.run()
                        async def run_batch():
                            return await asyncio.gather(*label_tasks, return_exceptions=True)

                        batch_labels = asyncio.run(run_batch())

                        # Persist this batch immediately
                        for feature, label, examples in zip(batch_features, batch_labels, batch_examples):
                            # Handle any exceptions
                            if isinstance(label, Exception):
                                logger.error(f"Error generating label for feature {feature.id}: {label}")
                                label = {"category": "error_feature", "specific": f"feature_{feature.neuron_index}", "description": ""}

                            feature.category = label["category"]
                            feature.name = label["specific"]
                            feature.description = label.get("description", "")
                            feature.label_source = label_source_value
                            feature.labeling_job_id = labeling_job.id
                            feature.labeled_at = labeled_at
                            feature.updated_at = labeled_at

                            # Create example tokens summary from context examples (first 7 prime tokens)
                            prime_tokens = [ex.get('prime_token', '') for ex in examples[:7] if ex.get('prime_token')]
                            example_summary = ', '.join(prime_tokens) if prime_tokens else ''
                            feature.example_tokens_summary = example_summary

                            # Emit individual result for real-time display
                            # Send first 10 full examples with prefix/prime/suffix context
                            example_data = []
                            for ex in examples[:10]:
                                example_data.append({
                                    "prefix_tokens": ex.get('prefix_tokens', []),
                                    "prime_token": ex.get('prime_token', ''),
                                    "suffix_tokens": ex.get('suffix_tokens', []),
                                    "max_activation": ex.get('max_activation', 0.0)
                                })

                            emit_labeling_result(
                                labeling_job_id=labeling_job.id,
                                feature_data={
                                    "feature_id": feature.neuron_index,
                                    "label": feature.name,
                                    "category": feature.category,
                                    "description": feature.description or "",
                                    "examples": example_data
                                }
                            )

                        # Commit this batch
                        self.db.commit()

                        # Update progress
                        current_labeled = batch_end
                        labeling_progress_callback(current_labeled, total_features)

                        logger.info(f"Batch {batch_start//LABEL_BATCH_SIZE + 1}/{(total_features + LABEL_BATCH_SIZE - 1)//LABEL_BATCH_SIZE}: Labeled and persisted features {batch_start+1}-{batch_end}/{total_features}")

                    logger.info(f"All {total_features} features labeled and persisted successfully")

                    # Create labels list for statistics calculation (now we need to query back from DB)
                    labels = [{"category": f.category, "specific": f.name} for f in features]

                elif labeling_method == LabelingMethod.OPENAI_COMPATIBLE.value:
                    # OpenAI-compatible endpoint (Ollama, vLLM, etc.)
                    endpoint = labeling_job.openai_compatible_endpoint
                    model_name = labeling_job.openai_compatible_model

                    if not endpoint:
                        raise ValueError("OpenAI-compatible endpoint not provided")
                    if not model_name:
                        raise ValueError("OpenAI-compatible model name not provided")

                    # Fetch prompt template if specified
                    system_message = None
                    user_prompt_template = None
                    temperature = 0.3
                    max_tokens = 50
                    top_p = 0.9

                    if labeling_job.prompt_template_id:
                        from src.models.labeling_prompt_template import LabelingPromptTemplate
                        template = self.db.query(LabelingPromptTemplate).filter(
                            LabelingPromptTemplate.id == labeling_job.prompt_template_id
                        ).first()
                        if template:
                            system_message = template.system_message
                            user_prompt_template = template.user_prompt_template
                            temperature = template.temperature
                            max_tokens = template.max_tokens
                            top_p = template.top_p
                            logger.info(f"Using prompt template: {template.name} (ID: {template.id})")

                    # Use default API timeout (120s)
                    # TODO: Add api_timeout column to labeling_jobs table for configurable timeout
                    api_timeout = 120.0

                    logger.info(f"Initializing OpenAI-compatible labeling service with endpoint: {endpoint}, model: {model_name}")
                    labeling_service = OpenAILabelingService(
                        api_key="dummy-key-not-required",  # Most local endpoints don't require auth
                        model=model_name,
                        base_url=endpoint,
                        system_message=system_message,
                        user_prompt_template=user_prompt_template,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        timeout=api_timeout,
                        filter_special=labeling_job.filter_special,
                        filter_single_char=labeling_job.filter_single_char,
                        filter_punctuation=labeling_job.filter_punctuation,
                        filter_numbers=labeling_job.filter_numbers,
                        filter_fragments=labeling_job.filter_fragments,
                        filter_stop_words=labeling_job.filter_stop_words,
                        save_requests_for_testing=labeling_job.save_requests_for_testing,
                        export_format=labeling_job.export_format,
                        save_poor_quality_labels=labeling_job.save_poor_quality_labels,
                        poor_quality_sample_rate=labeling_job.poor_quality_sample_rate,
                        save_requests_sample_rate=labeling_job.save_requests_sample_rate,
                        labeling_job_id=labeling_job.id
                    )

                    # Generate and persist labels in batches of 10
                    # This ensures progress is saved incrementally if the job fails
                    label_source_value = LabelSource.OPENAI.value  # Use OPENAI source for compatible endpoints
                    labeled_at = datetime.now(timezone.utc)
                    LABEL_BATCH_SIZE = 10

                    logger.info(f"Starting incremental labeling: {total_features} features in batches of {LABEL_BATCH_SIZE}")

                    for batch_start in range(0, total_features, LABEL_BATCH_SIZE):
                        batch_end = min(batch_start + LABEL_BATCH_SIZE, total_features)
                        batch_features = features[batch_start:batch_end]
                        batch_examples = features_examples[batch_start:batch_end]
                        batch_all_examples = all_features_examples[batch_start:batch_end]

                        # Generate labels for this batch using context-based examples
                        # Create concurrent tasks for all features in batch
                        # Pass all examples for NLP analysis to improve labeling
                        label_tasks = []
                        for feature, examples, all_ex in zip(batch_features, batch_examples, batch_all_examples):
                            task = labeling_service.generate_label_from_examples(
                                examples=examples,
                                template_config=template_config,
                                user_prompt_template=user_prompt_template,
                                system_message=system_message,
                                feature_id=feature.id,
                                neuron_index=feature.neuron_index,
                                logit_effects=None,  # TODO: Implement in Sprint 4
                                all_examples=all_ex,  # Pass full 100 examples for NLP analysis
                                nlp_analysis=feature.nlp_analysis  # Use pre-computed NLP if available
                            )
                            label_tasks.append(task)

                        # Execute all labeling tasks concurrently
                        # Wrap gather in an async function for asyncio.run()
                        async def run_batch():
                            return await asyncio.gather(*label_tasks, return_exceptions=True)

                        batch_labels = asyncio.run(run_batch())

                        # Persist this batch immediately
                        for feature, label, examples in zip(batch_features, batch_labels, batch_examples):
                            # Handle any exceptions
                            if isinstance(label, Exception):
                                logger.error(f"Error generating label for feature {feature.id}: {label}")
                                label = {"category": "error_feature", "specific": f"feature_{feature.neuron_index}", "description": ""}

                            feature.category = label["category"]
                            feature.name = label["specific"]
                            feature.description = label.get("description", "")
                            feature.label_source = label_source_value
                            feature.labeling_job_id = labeling_job.id
                            feature.labeled_at = labeled_at
                            feature.updated_at = labeled_at

                            # Create example tokens summary from context examples (first 7 prime tokens)
                            prime_tokens = [ex.get('prime_token', '') for ex in examples[:7] if ex.get('prime_token')]
                            example_summary = ', '.join(prime_tokens) if prime_tokens else ''
                            feature.example_tokens_summary = example_summary

                            # Emit individual result for real-time display
                            # Send first 10 full examples with prefix/prime/suffix context
                            example_data = []
                            for ex in examples[:10]:
                                example_data.append({
                                    "prefix_tokens": ex.get('prefix_tokens', []),
                                    "prime_token": ex.get('prime_token', ''),
                                    "suffix_tokens": ex.get('suffix_tokens', []),
                                    "max_activation": ex.get('max_activation', 0.0)
                                })

                            emit_labeling_result(
                                labeling_job_id=labeling_job.id,
                                feature_data={
                                    "feature_id": feature.neuron_index,
                                    "label": feature.name,
                                    "category": feature.category,
                                    "description": feature.description or "",
                                    "examples": example_data
                                }
                            )

                        # Commit this batch
                        self.db.commit()

                        # Update progress
                        current_labeled = batch_end
                        labeling_progress_callback(current_labeled, total_features)

                        logger.info(f"Batch {batch_start//LABEL_BATCH_SIZE + 1}/{(total_features + LABEL_BATCH_SIZE - 1)//LABEL_BATCH_SIZE}: Labeled and persisted features {batch_start+1}-{batch_end}/{total_features}")

                    logger.info(f"All {total_features} features labeled and persisted successfully")

                    # Create labels list for statistics calculation (now we need to query back from DB)
                    labels = [{"category": f.category, "specific": f.name} for f in features]

                else:
                    raise ValueError(f"Unsupported labeling method: {labeling_method}")

                # Note: Feature persistence now happens incrementally in each method branch above
                logger.info(f"Successfully labeled and persisted {len(features)} features using {labeling_method}")

                # Unload OpenAI-compatible model (Ollama) from VRAM after completion
                if labeling_method == LabelingMethod.OPENAI_COMPATIBLE.value:
                    logger.info("Unloading OpenAI-compatible model from VRAM")
                    asyncio.run(self._unload_ollama_model(
                        labeling_job.openai_compatible_endpoint,
                        labeling_job.openai_compatible_model
                    ))

                # Calculate statistics
                end_time = datetime.now(timezone.utc)
                duration_seconds = (end_time - start_time).total_seconds()

                successfully_labeled = len([l for l in labels if l and l.get("specific") and not l.get("specific").startswith("feature_")])
                failed_labels = len(labels) - successfully_labeled
                avg_label_length = sum(len(l.get("specific", "")) for l in labels) / len(labels) if labels else 0

                statistics = {
                    "total_features": len(features),
                    "successfully_labeled": successfully_labeled,
                    "failed_labels": failed_labels,
                    "avg_label_length": round(avg_label_length, 2),
                    "labeling_duration_seconds": round(duration_seconds, 2),
                    "labeling_method": labeling_method
                }

                # Mark labeling job as completed
                labeling_job.status = LabelingStatus.COMPLETED.value
                labeling_job.progress = 1.0
                labeling_job.features_labeled = len(labels)
                labeling_job.completed_at = end_time
                labeling_job.updated_at = end_time
                labeling_job.statistics = statistics
                self.db.commit()

                logger.info(f"Labeling job {labeling_job_id} completed successfully")

                # Emit completion event via WebSocket
                emit_labeling_progress(
                    labeling_job_id=labeling_job.id,
                    event="labeling:completed",
                    data={
                        "labeling_job_id": labeling_job.id,
                        "extraction_job_id": labeling_job.extraction_job_id,
                        "status": "completed",
                        "features_labeled": len(labels),
                        "total_features": total_features,
                        "statistics": statistics,
                        "message": f"Successfully labeled {successfully_labeled}/{total_features} features in {duration_seconds:.1f}s"
                    }
                )

                return statistics

            except Exception as e:
                logger.error(f"Batch labeling failed: {e}", exc_info=True)
                raise

        except Exception as e:
            logger.error(f"Feature labeling failed for job {labeling_job_id}: {e}", exc_info=True)

            # Mark labeling job as failed
            labeling_job.status = LabelingStatus.FAILED.value
            labeling_job.error_message = str(e)
            labeling_job.updated_at = datetime.now(timezone.utc)
            self.db.commit()

            # Emit failure event via WebSocket
            emit_labeling_progress(
                labeling_job_id=labeling_job.id,
                event="labeling:failed",
                data={
                    "labeling_job_id": labeling_job.id,
                    "extraction_job_id": labeling_job.extraction_job_id,
                    "status": "failed",
                    "error_message": str(e),
                    "message": f"Labeling failed: {str(e)}"
                }
            )

            raise

    async def get_labeling_job(self, labeling_job_id: str) -> Optional[LabelingJob]:
        """
        Get a labeling job by ID.

        Args:
            labeling_job_id: ID of the labeling job

        Returns:
            LabelingJob or None if not found
        """
        result = await self.db.execute(
            select(LabelingJob).where(LabelingJob.id == labeling_job_id)
        )
        return result.scalar_one_or_none()

    async def list_labeling_jobs(
        self,
        extraction_job_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[List[LabelingJob], int]:
        """
        List labeling jobs with optional filtering.

        Args:
            extraction_job_id: Optional filter by extraction job ID
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            Tuple of (list of labeling jobs, total count)
        """
        from sqlalchemy import func

        # Build query
        query = select(LabelingJob).order_by(desc(LabelingJob.created_at))

        if extraction_job_id:
            query = query.where(LabelingJob.extraction_job_id == extraction_job_id)

        # Get total count
        count_query = select(func.count()).select_from(LabelingJob)
        if extraction_job_id:
            count_query = count_query.where(LabelingJob.extraction_job_id == extraction_job_id)

        count_result = await self.db.execute(count_query)
        total = count_result.scalar_one()

        # Get paginated results
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        jobs = result.scalars().all()

        return list(jobs), total

    async def cancel_labeling_job(self, labeling_job_id: str) -> bool:
        """
        Cancel a labeling job.

        Args:
            labeling_job_id: ID of the labeling job to cancel

        Returns:
            True if cancelled successfully

        Raises:
            ValueError: If job not found or not in cancellable state
        """
        result = await self.db.execute(
            select(LabelingJob).where(LabelingJob.id == labeling_job_id)
        )
        labeling_job = result.scalar_one_or_none()

        if not labeling_job:
            raise ValueError(f"Labeling job {labeling_job_id} not found")

        if labeling_job.status not in [LabelingStatus.QUEUED.value, LabelingStatus.LABELING.value]:
            raise ValueError(
                f"Cannot cancel labeling job {labeling_job_id} with status {labeling_job.status}"
            )

        # Revoke the Celery task to stop execution
        if labeling_job.celery_task_id:
            from ..core.celery_app import celery_app
            logger.info(f"Revoking Celery task {labeling_job.celery_task_id} for job {labeling_job_id}")
            celery_app.control.revoke(
                labeling_job.celery_task_id,
                terminate=True,
                signal='SIGTERM'
            )

        # Unload model from VRAM if using OpenAI-compatible endpoint (Ollama)
        if labeling_job.labeling_method == LabelingMethod.OPENAI_COMPATIBLE.value:
            await self._unload_ollama_model(
                labeling_job.openai_compatible_endpoint,
                labeling_job.openai_compatible_model
            )

        labeling_job.status = LabelingStatus.CANCELLED.value
        labeling_job.updated_at = datetime.now(timezone.utc)
        await self.db.commit()

        logger.info(f"Cancelled labeling job {labeling_job_id}")
        return True

    async def _unload_ollama_model(self, endpoint: Optional[str], model_name: Optional[str]) -> None:
        """
        Unload model from Ollama VRAM by sending a request with keep_alive=0.

        Args:
            endpoint: Ollama endpoint URL
            model_name: Model name to unload
        """
        if not endpoint or not model_name:
            return

        try:
            import httpx
            # Extract base URL (remove /v1 or /api suffix if present)
            base_url = endpoint.rstrip('/').replace('/v1', '').replace('/api', '')
            unload_url = f"{base_url}/api/generate"

            async with httpx.AsyncClient(timeout=10.0) as client:
                # Send empty prompt with keep_alive=0 to unload model
                response = await client.post(
                    unload_url,
                    json={
                        "model": model_name,
                        "prompt": "",
                        "keep_alive": 0  # Unload immediately
                    }
                )
                if response.status_code == 200:
                    logger.info(f"Successfully unloaded model {model_name} from VRAM")
                else:
                    logger.warning(f"Failed to unload model {model_name}: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not unload model from VRAM: {e}")
            # Non-critical error, don't raise

    async def delete_labeling_job(self, labeling_job_id: str) -> bool:
        """
        Delete a labeling job.

        This does NOT delete the features or their labels, only the labeling job record.
        Feature labels will remain intact. If the job is active, it will be cancelled first.

        Args:
            labeling_job_id: ID of the labeling job to delete

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If job not found
        """
        from sqlalchemy import update

        result = await self.db.execute(
            select(LabelingJob).where(LabelingJob.id == labeling_job_id)
        )
        labeling_job = result.scalar_one_or_none()

        if not labeling_job:
            raise ValueError(f"Labeling job {labeling_job_id} not found")

        # If job is active, cancel it first (revoke Celery task)
        if labeling_job.status in [LabelingStatus.QUEUED.value, LabelingStatus.LABELING.value]:
            if labeling_job.celery_task_id:
                from ..core.celery_app import celery_app
                logger.info(f"Auto-cancelling active job: revoking Celery task {labeling_job.celery_task_id}")
                celery_app.control.revoke(
                    labeling_job.celery_task_id,
                    terminate=True,
                    signal='SIGTERM'
                )

        # Clear labeling_job_id reference from features
        await self.db.execute(
            update(Feature).where(
                Feature.labeling_job_id == labeling_job_id
            ).values(labeling_job_id=None)
        )

        # Delete labeling job
        await self.db.delete(labeling_job)
        await self.db.commit()

        logger.info(f"Deleted labeling job {labeling_job_id}")
        return True
