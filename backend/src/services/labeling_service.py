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
            status=LabelingStatus.QUEUED.value,
            progress=0.0,
            features_labeled=0,
            total_features=total_features
        )

        self.db.add(labeling_job)
        await self.db.commit()
        await self.db.refresh(labeling_job)

        logger.info(
            f"Created labeling job {job_id} for extraction {extraction_job_id} "
            f"with {total_features} features using method: {labeling_job.labeling_method}"
        )

        return labeling_job

    def _aggregate_token_stats_batch(
        self,
        feature_ids: List[str]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Aggregate token statistics for a batch of features using SQL.

        Uses PostgreSQL JSONB functions to efficiently aggregate token statistics
        directly in the database, avoiding N+1 query problem.

        Args:
            feature_ids: List of feature IDs to aggregate

        Returns:
            Dict mapping feature_id to token_stats dict:
            {
                "feature_id_1": {
                    "token_1": {"count": 10, "total_activation": 5.2, "max_activation": 0.8},
                    "token_2": {"count": 5, "total_activation": 2.1, "max_activation": 0.5}
                },
                ...
            }
        """
        from sqlalchemy import text

        if not feature_ids:
            return {}

        # SQL query using CTEs to aggregate token statistics
        query = text("""
            WITH token_aggregates AS (
                SELECT
                    fa.feature_id,
                    token_elem.token::text as token,
                    COUNT(*) as count,
                    SUM((activation_elem.activation::text)::float) as total_activation,
                    MAX((activation_elem.activation::text)::float) as max_activation
                FROM feature_activations fa,
                    LATERAL jsonb_array_elements(fa.tokens) WITH ORDINALITY AS token_elem(token, token_idx),
                    LATERAL jsonb_array_elements(fa.activations) WITH ORDINALITY AS activation_elem(activation, act_idx)
                WHERE token_elem.token_idx = activation_elem.act_idx
                  AND fa.feature_id = ANY(:feature_ids)
                GROUP BY fa.feature_id, token_elem.token::text
            )
            SELECT
                feature_id,
                jsonb_object_agg(
                    token,
                    jsonb_build_object(
                        'count', count,
                        'total_activation', total_activation,
                        'max_activation', max_activation
                    )
                ) as token_stats
            FROM token_aggregates
            GROUP BY feature_id;
        """)

        result = self.db.execute(query, {"feature_ids": feature_ids})

        # Convert to dict for easy lookup
        token_stats_map = {}
        for row in result:
            feature_id = row.feature_id
            token_stats = row.token_stats or {}
            token_stats_map[feature_id] = token_stats

        return token_stats_map

    def _format_tokens_table(
        self,
        token_stats: Dict[str, Dict[str, float]],
        filter_special: bool = True,
        filter_single_char: bool = True,
        filter_punctuation: bool = True,
        filter_numbers: bool = True,
        filter_fragments: bool = True,
        filter_stop_words: bool = False,
        max_tokens: int = 15
    ) -> str:
        """
        Format token statistics as a table string for prompt replacement.

        Args:
            token_stats: Dict mapping token to stats dict (count, total_activation, max_activation)
            filter_special: Filter special tokens (<s>, </s>, etc.)
            filter_single_char: Filter single character tokens
            filter_punctuation: Filter pure punctuation
            filter_numbers: Filter pure numeric tokens
            filter_fragments: Filter word fragments (BPE subwords)
            filter_stop_words: Filter common stop words
            max_tokens: Maximum number of tokens to include in table

        Returns:
            Formatted table string with filtered tokens
        """
        from src.utils.token_filters import is_junk_token

        # Filter tokens and sort by count
        filtered_tokens = []
        for token, stats in token_stats.items():
            # Clean token for display (remove SentencePiece underscore prefix)
            display_token = token.replace('▁', ' ').strip()
            if not display_token:
                display_token = token

            # Apply filters
            if is_junk_token(
                token,
                filter_special=filter_special,
                filter_single_char=filter_single_char,
                filter_punctuation=filter_punctuation,
                filter_numbers=filter_numbers,
                filter_fragments=filter_fragments,
                filter_stop_words=filter_stop_words
            ):
                continue

            count = stats.get("count", 0)
            filtered_tokens.append((display_token, count))

        # Sort by count descending, then by token ascending
        filtered_tokens.sort(key=lambda x: (-x[1], x[0]))

        # Take top N tokens
        top_tokens = filtered_tokens[:max_tokens]

        # Format as table
        if not top_tokens:
            return "(No tokens found after filtering)"

        lines = []
        for token, count in top_tokens:
            # Format: 'token'                                    → count times
            # Pad token to 40 characters for alignment
            token_str = f"'{token}'"
            padded_token = token_str.ljust(42)
            line = f"{padded_token} → {count} {'time' if count == 1 else 'times'}"
            lines.append(line)

        return '\n'.join(lines)

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

            # Pre-labeling feature filtering (if enabled)
            from src.utils.token_filter import FeatureFilter
            from src.core.config import settings

            if settings.pre_labeling_filter_enabled:
                logger.info(f"Pre-labeling filter enabled - analyzing {len(all_features)} features")

                # Initialize feature filter with configured thresholds
                feature_filter = FeatureFilter(
                    junk_ratio_threshold=settings.pre_labeling_junk_ratio_threshold,
                    single_char_ratio_threshold=settings.pre_labeling_single_char_threshold,
                    min_tokens_for_decision=5
                )

                # Quick token aggregation for filtering decision
                all_feature_ids = [f.id for f in all_features]
                token_stats_map = self._aggregate_token_stats_batch(all_feature_ids)

                # Analyze each feature
                features_to_label = []
                skipped_features = []

                for feature in all_features:
                    token_stats = token_stats_map.get(feature.id, {})
                    if feature_filter.is_junk_feature(token_stats):
                        skipped_features.append(feature)
                    else:
                        features_to_label.append(feature)

                # Mark skipped features as unlabeled junk
                if skipped_features:
                    for feature in skipped_features:
                        feature.name = "unlabeled_junk"
                        feature.category = "system"
                        feature.label_source = "auto"  # Automatically determined by filter
                        feature.labeled_at = datetime.now(timezone.utc)
                    self.db.commit()

                logger.info(
                    f"Pre-labeling filter: {len(features_to_label)} features to label, "
                    f"{len(skipped_features)} junk features skipped "
                    f"({len(skipped_features) / len(all_features) * 100:.1f}% filtered)"
                )

                features = features_to_label
            else:
                logger.info("Pre-labeling filter disabled - labeling all features")
                features = all_features

            total_features = len(features)
            logger.info(f"Labeling {total_features} features for extraction {labeling_job.extraction_job_id}")

            # Aggregate token statistics using efficient SQL batching
            # Process features in batches of 1000 to avoid memory issues and track progress
            BATCH_SIZE = 1000
            features_token_stats = []
            neuron_indices = []

            # Phase 1: Token Aggregation with progress tracking
            logger.info(f"Starting token aggregation phase for {total_features} features in batches of {BATCH_SIZE}")

            for batch_start in range(0, total_features, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_features)
                batch_features = features[batch_start:batch_end]
                batch_size = len(batch_features)

                logger.info(f"Aggregating batch {batch_start//BATCH_SIZE + 1}/{(total_features + BATCH_SIZE - 1)//BATCH_SIZE}: features {batch_start+1}-{batch_end}")

                # Get feature IDs for this batch
                batch_feature_ids = [f.id for f in batch_features]

                # Use SQL to aggregate token stats for entire batch
                token_stats_map = self._aggregate_token_stats_batch(batch_feature_ids)

                # Build ordered lists for labeling (maintain feature order)
                for feature in batch_features:
                    token_stats = token_stats_map.get(feature.id, {})
                    features_token_stats.append(token_stats)
                    neuron_indices.append(feature.neuron_index)

                # Update progress in database
                aggregation_progress = batch_end / total_features
                labeling_job.progress = aggregation_progress * 0.3  # Aggregation is ~30% of total work
                labeling_job.updated_at = datetime.now(timezone.utc)
                self.db.commit()

                # Emit WebSocket progress update
                emit_labeling_progress(
                    labeling_job_id=labeling_job.id,
                    event="progress",
                    data={
                        "labeling_job_id": labeling_job.id,
                        "extraction_job_id": labeling_job.extraction_job_id,
                        "progress": labeling_job.progress,
                        "features_labeled": 0,
                        "total_features": total_features,
                        "status": "labeling",
                        "phase": "aggregation",
                        "message": f"Aggregated token statistics for {batch_end}/{total_features} features"
                    }
                )

                logger.info(f"Batch {batch_start//BATCH_SIZE + 1} complete: {batch_end}/{total_features} features aggregated ({aggregation_progress*100:.1f}%)")

            logger.info(f"Token aggregation complete for {len(features_token_stats)} features")

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
                    event="progress",
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
                        # Generate and persist labels in batches
                        # This ensures progress is saved incrementally if the job fails
                        label_source_value = LabelSource.LOCAL_LLM.value
                        labeled_at = datetime.now(timezone.utc)
                        LABEL_BATCH_SIZE = 10

                        logger.info(f"Starting incremental labeling: {total_features} features in batches of {LABEL_BATCH_SIZE}")

                        for batch_start in range(0, total_features, LABEL_BATCH_SIZE):
                            batch_end = min(batch_start + LABEL_BATCH_SIZE, total_features)
                            batch_features = features[batch_start:batch_end]
                            batch_token_stats = features_token_stats[batch_start:batch_end]
                            batch_neuron_indices = neuron_indices[batch_start:batch_end]

                            # Generate labels for this batch (model already loaded)
                            batch_labels = []
                            for token_stats, neuron_idx in zip(batch_token_stats, batch_neuron_indices):
                                label = labeling_service.generate_label(token_stats, neuron_index=neuron_idx)
                                batch_labels.append({"category": "unknown", "specific": label, "description": ""})

                            # Persist this batch immediately
                            for feature, label, token_stats in zip(batch_features, batch_labels, batch_token_stats):
                                feature.category = label["category"]
                                feature.name = label["specific"]
                                feature.description = label.get("description", "")
                                feature.label_source = label_source_value
                                feature.labeling_job_id = labeling_job.id
                                feature.labeled_at = labeled_at
                                feature.updated_at = labeled_at

                                # Create example tokens summary with filtered top 7 tokens
                                example_summary = create_example_tokens_summary(
                                    token_stats,
                                    filter_special=labeling_job.filter_special,
                                    filter_single_char=labeling_job.filter_single_char,
                                    filter_punctuation=labeling_job.filter_punctuation,
                                    filter_numbers=labeling_job.filter_numbers,
                                    filter_fragments=labeling_job.filter_fragments,
                                    filter_stop_words=labeling_job.filter_stop_words,
                                    top_n=7
                                )
                                feature.example_tokens_summary = example_summary

                                # Emit individual result for real-time display
                                # Extract top 5 tokens for example
                                sorted_tokens = sorted(
                                    token_stats.items(),
                                    key=lambda x: x[1].get("count", 0),
                                    reverse=True
                                )[:5]
                                example_tokens = [token for token, _ in sorted_tokens]

                                emit_labeling_result(
                                    labeling_job_id=labeling_job.id,
                                    feature_data={
                                        "feature_id": feature.neuron_index,
                                        "label": feature.name,
                                        "category": feature.category,
                                        "description": feature.description or "",
                                        "example_tokens": example_tokens
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

                    logger.info(f"Initializing OpenAI labeling service with model: {openai_model}")
                    labeling_service = OpenAILabelingService(
                        api_key=openai_api_key,
                        model=openai_model,
                        system_message=system_message,
                        user_prompt_template=user_prompt_template,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        filter_special=labeling_job.filter_special,
                        filter_single_char=labeling_job.filter_single_char,
                        filter_punctuation=labeling_job.filter_punctuation,
                        filter_numbers=labeling_job.filter_numbers,
                        filter_fragments=labeling_job.filter_fragments,
                        filter_stop_words=labeling_job.filter_stop_words,
                        save_requests_for_testing=labeling_job.save_requests_for_testing,
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
                        batch_token_stats = features_token_stats[batch_start:batch_end]
                        batch_neuron_indices = neuron_indices[batch_start:batch_end]

                        # Generate labels for this batch
                        batch_labels = asyncio.run(labeling_service.batch_generate_labels(
                            features_token_stats=batch_token_stats,
                            neuron_indices=batch_neuron_indices,
                            progress_callback=None,  # We'll handle progress below
                            batch_size=LABEL_BATCH_SIZE
                        ))

                        # Persist this batch immediately
                        for feature, label, token_stats in zip(batch_features, batch_labels, batch_token_stats):
                            feature.category = label["category"]
                            feature.name = label["specific"]
                            feature.description = label.get("description", "")
                            feature.label_source = label_source_value
                            feature.labeling_job_id = labeling_job.id
                            feature.labeled_at = labeled_at
                            feature.updated_at = labeled_at

                            # Create example tokens summary with filtered top 7 tokens
                            example_summary = create_example_tokens_summary(
                                token_stats,
                                filter_special=labeling_job.filter_special,
                                filter_single_char=labeling_job.filter_single_char,
                                filter_punctuation=labeling_job.filter_punctuation,
                                filter_numbers=labeling_job.filter_numbers,
                                filter_fragments=labeling_job.filter_fragments,
                                filter_stop_words=labeling_job.filter_stop_words,
                                top_n=7
                            )
                            feature.example_tokens_summary = example_summary

                            # Emit individual result for real-time display
                            # Extract top 5 tokens for example
                            sorted_tokens = sorted(
                                token_stats.items(),
                                key=lambda x: x[1].get("count", 0),
                                reverse=True
                            )[:5]
                            example_tokens = [token for token, _ in sorted_tokens]

                            emit_labeling_result(
                                labeling_job_id=labeling_job.id,
                                feature_data={
                                    "feature_id": feature.neuron_index,
                                    "label": feature.name,
                                    "category": feature.category,
                                    "description": feature.description or "",
                                    "example_tokens": example_tokens
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
                        filter_special=labeling_job.filter_special,
                        filter_single_char=labeling_job.filter_single_char,
                        filter_punctuation=labeling_job.filter_punctuation,
                        filter_numbers=labeling_job.filter_numbers,
                        filter_fragments=labeling_job.filter_fragments,
                        filter_stop_words=labeling_job.filter_stop_words,
                        save_requests_for_testing=labeling_job.save_requests_for_testing,
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
                        batch_token_stats = features_token_stats[batch_start:batch_end]
                        batch_neuron_indices = neuron_indices[batch_start:batch_end]

                        # Generate labels for this batch
                        batch_labels = asyncio.run(labeling_service.batch_generate_labels(
                            features_token_stats=batch_token_stats,
                            neuron_indices=batch_neuron_indices,
                            progress_callback=None,  # We'll handle progress below
                            batch_size=LABEL_BATCH_SIZE
                        ))

                        # Persist this batch immediately
                        for feature, label, token_stats in zip(batch_features, batch_labels, batch_token_stats):
                            feature.category = label["category"]
                            feature.name = label["specific"]
                            feature.description = label.get("description", "")
                            feature.label_source = label_source_value
                            feature.labeling_job_id = labeling_job.id
                            feature.labeled_at = labeled_at
                            feature.updated_at = labeled_at

                            # Create example tokens summary with filtered top 7 tokens
                            example_summary = create_example_tokens_summary(
                                token_stats,
                                filter_special=labeling_job.filter_special,
                                filter_single_char=labeling_job.filter_single_char,
                                filter_punctuation=labeling_job.filter_punctuation,
                                filter_numbers=labeling_job.filter_numbers,
                                filter_fragments=labeling_job.filter_fragments,
                                filter_stop_words=labeling_job.filter_stop_words,
                                top_n=7
                            )
                            feature.example_tokens_summary = example_summary

                            # Emit individual result for real-time display
                            # Extract top 5 tokens for example
                            sorted_tokens = sorted(
                                token_stats.items(),
                                key=lambda x: x[1].get("count", 0),
                                reverse=True
                            )[:5]
                            example_tokens = [token for token, _ in sorted_tokens]

                            emit_labeling_result(
                                labeling_job_id=labeling_job.id,
                                feature_data={
                                    "feature_id": feature.neuron_index,
                                    "label": feature.name,
                                    "category": feature.category,
                                    "description": feature.description or "",
                                    "example_tokens": example_tokens
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
                    event="completed",
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
                event="failed",
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
