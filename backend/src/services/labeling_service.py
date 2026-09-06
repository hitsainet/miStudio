"""
Feature labeling service for semantic labeling of SAE features.

This service manages semantic labeling of features extracted from SAE models.
Labeling is independent from extraction, allowing re-labeling without re-extraction.
"""

import logging
import os
import random
import uuid
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
from src.core.encryption import decrypt_value, encrypt_value
from src.services.local_labeling_service import LocalLabelingService
from src.services.openai_labeling_service import OpenAILabelingService
from src.workers.websocket_emitter import emit_labeling_progress, emit_labeling_result
from src.utils.token_filters import filter_token_stats
from src.utils.millm_utils import ensure_model_loaded

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

        # Count features to label. When a panel is supplied the count MUST be
        # scoped too: an unscoped count leaves total_features as the whole
        # extraction, so progress crawls to 1% then jumps to 1.0 and every ETA
        # computed from the row is wrong by an order of magnitude.
        panel_ids = list(dict.fromkeys(config.get("feature_ids") or [])) or None
        count_q = select(func.count()).select_from(Feature).where(
            Feature.extraction_job_id == extraction_job_id
        )
        if panel_ids:
            count_q = count_q.where(Feature.id.in_(panel_ids))
        count_result = await self.db.execute(count_q)
        total_features = count_result.scalar_one()

        if total_features == 0:
            raise ValueError(f"Extraction {extraction_job_id} has no features to label")

        if panel_ids and total_features != len(panel_ids):
            # A shrunken panel is not the panel that was requested. Labelling a
            # subset silently would make two runs incomparable and any rate
            # computed from them wrong, so refuse and name the gap.
            raise ValueError(
                f"panel resolved to {total_features} of {len(panel_ids)} requested "
                f"features — the rest are absent from extraction {extraction_job_id}"
            )

        # Create labeling job ID: label_{extraction_id}_{timestamp}_{rand}
        #
        # The timestamp is second-resolution, so two starts within the same second
        # produced the SAME primary key and the insert below died with an opaque
        # IntegrityError->500. The active-job 409 masked this only while the first
        # job was still QUEUED/LABELING — a job that COMPLETED inside the same
        # second left the collision fully exposed. Nothing parses this id, so a
        # short random suffix is safe.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = f"label_{extraction_job_id}_{timestamp}_{uuid.uuid4().hex[:6]}"

        # Create labeling job record
        labeling_job = LabelingJob(
            id=job_id,
            extraction_job_id=extraction_job_id,
            labeling_method=config.get("labeling_method", "openai"),
            openai_model=config.get("openai_model"),
            openai_api_key=encrypt_value(config["openai_api_key"]) if config.get("openai_api_key") else None,
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
            max_tokens=config.get("max_tokens", 300),
            api_timeout=config.get("api_timeout", 120.0),
            status=LabelingStatus.QUEUED.value,
            progress=0.0,
            features_labeled=0,
            total_features=total_features,
            statistics={
                "max_examples": config.get("max_examples"),  # Store example count override (None = use template default)
                "batch_size": config.get("batch_size", 10),  # Features per batch (default 10)
            },
            # A real column, not a statistics key: the completion write replaces
            # `statistics` wholesale, which would erase the panel at the moment
            # the run finished and take reproducibility with it.
            feature_ids=panel_ids,
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

    #: PERMANENT ALIAS, not a subclass. `_LabelingCancelled` is caught by name
    #: in `workers/labeling_tasks.py` and asserted by name in two behavioural
    #: test files; pointing it at `OperatorCancelled` keeps every one of those
    #: working while upgrading it from `Exception` to `BaseException`.
    #:
    #: That upgrade IS the MIS-E2E-058 fix, generalised: labeling's own outer
    #: `except Exception` used to catch this and write FAILED, turning an
    #: operator's deliberate stop into a crash report.
    from ..core.cancellation import OperatorCancelled as _LabelingCancelled

    def _raise_if_cancelled(self, labeling_job_id: str) -> None:
        """Cooperative cancellation check — call once per batch.

        NOW A SHIM over `core.cancellation`. The reasoning that used to be
        spelled out here — solo pool, revoke signals a child that does not
        exist, the main process never services control messages while a task
        runs — lives once in that module's docstring.

        TWO THINGS THE SHIM FIXES FOR FREE.

        `populate_existing()` was already right here (MIS-E2E-057) and is now
        the shared default rather than one service's hard-won knowledge.

        And A DELETED ROW IS NOW A STOP. This returned silently when the row
        was gone — but `delete_labeling_job` DELETES THE ROW as its stop
        signal, so the job ran to completion against a row that no longer
        existed, writing results nobody could read. The `labeling` scope
        carries `missing_row="cancelled"` for exactly that.
        """
        from ..core.cancellation import cancel_checker

        # A FRESH CHECKER PER CALL, WHICH THEREFORE ALWAYS POLLS.
        #
        # `CancelCheck`'s first call always polls, so constructing one here is
        # the same thing the old implementation did: query once per batch. An
        # earlier version of this shim cached the checker per job so the
        # 2-second budget would throttle — and that was a behaviour change
        # dressed as a refactor, because a fast batch loop then ran its whole
        # length inside one window and never re-polled. The time budget exists
        # for per-token loops; a per-batch caller is already at the right
        # granularity, and `min_interval_s` here would be untestable
        # redundancy on top of the first-call rule.
        cancel_checker(
            "labeling", labeling_job_id, db=self.db
        ).raise_if_cancelled(f"labeling job {labeling_job_id}")

    def _label_batch(
        self,
        labeling_service,
        loop,
        batch_features,
        batch_examples,
        batch_all_examples,
        feature_logit_effects,
        template_config,
        user_prompt_template,
        system_message,
    ):
        """Label one batch of features, batched through miLLM when enabled.

        Returns one label per feature in order. Never raises for a single
        feature: the batched client falls back to serial on any batch failure,
        and the serial path returns error labels, so the caller's
        isinstance(label, Exception) guard stays valid either way.

        BULK LABELING ONLY. Batch composition changes greedy output under int8
        quantisation, so a labeling trial must not come through here — and does
        not: LabelingTrialService calls generate_label_from_examples directly.
        """
        requests = []
        for feature, examples, all_ex in zip(
            batch_features, batch_examples, batch_all_examples
        ):
            requests.append({
                "examples": examples,
                "template_config": template_config,
                "user_prompt_template": user_prompt_template,
                "system_message": system_message,
                "feature_id": feature.id,
                "neuron_index": feature.neuron_index,
                "logit_effects": feature_logit_effects.get(feature.id),
                "all_examples": all_ex,
                "nlp_analysis": feature.nlp_analysis,
            })

        batch_size = getattr(settings, "labeling_batch_size", 1) or 1
        can_batch = batch_size > 1 and hasattr(
            labeling_service, "generate_labels_from_examples_batched"
        )

        if can_batch:
            return loop.run_until_complete(
                labeling_service.generate_labels_from_examples_batched(
                    requests, batch_size=batch_size
                )
            )

        # Per-feature requests, concurrent within the shared loop.
        return loop.run_until_complete(
            asyncio.gather(
                *[
                    labeling_service.generate_label_from_examples(**req)
                    for req in requests
                ],
                return_exceptions=True,
            )
        )

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
        # This method uses sync SQLAlchemy Session (.query() calls throughout).
        # It must be called from a Celery worker that injects a sync session,
        # not from an async FastAPI endpoint that uses AsyncSession.
        assert isinstance(self.db, Session), (
            "label_features_for_extraction requires a sync SQLAlchemy Session; "
            f"got {type(self.db).__name__}. Call from a Celery worker, not an async endpoint."
        )

        # Fetch labeling job
        labeling_job = self.db.query(LabelingJob).filter(
            LabelingJob.id == labeling_job_id
        ).first()

        if not labeling_job:
            raise ValueError(f"Labeling job {labeling_job_id} not found")

        # Validate extraction job and features BEFORE transitioning to LABELING.
        # This prevents the job being stuck in LABELING if the extraction is missing.
        extraction_job = self.db.query(ExtractionJob).filter(
            ExtractionJob.id == labeling_job.extraction_job_id
        ).first()
        if not extraction_job:
            raise ValueError(f"Extraction job {labeling_job.extraction_job_id} not found")

        # Chained .filter() rather than .limit()/.join(): the strict Mock in
        # tests/unit/test_labeling_service.py stubs only filter/order_by/all,
        # so keeping this shape means those guards keep guarding.
        _q = self.db.query(Feature).filter(
            Feature.extraction_job_id == labeling_job.extraction_job_id
        )
        # Only a real list of ids counts as a panel. A malformed column (or a
        # test double) is treated as "no panel" rather than being handed to
        # in_(), which raises an opaque ArgumentError deep in SQLAlchemy.
        _panel = labeling_job.feature_ids
        if isinstance(_panel, (list, tuple)) and _panel and all(
            isinstance(i, str) for i in _panel
        ):
            # Panel run. The extraction predicate stays, so a foreign id cannot
            # pull in a feature from a different extraction.
            _q = _q.filter(Feature.id.in_(list(_panel)))
        all_features = _q.order_by(Feature.neuron_index).all()
        if not all_features:
            raise ValueError(f"No features found for extraction {labeling_job.extraction_job_id}")

        # Validation passed — safe to transition to LABELING
        labeling_job.status = LabelingStatus.LABELING.value
        labeling_job.updated_at = datetime.now(timezone.utc)
        self.db.commit()

        start_time = datetime.now(timezone.utc)

        try:
            # NOTE: Batch commits throughout this method mean that if labeling fails
            # mid-run, some features will already be committed as labeled while others
            # are not. The job status is set to FAILED below, but the partial labels
            # remain. A full transactional rollback would require buffering all writes
            # until completion — deferred as a future improvement (high memory cost).

            # All features are retrieved for examples; filtering happens after retrieval
            # using prime tokens from activation examples (context-based approach).
            features = all_features

            total_features = len(features)
            logger.info(f"Labeling {total_features} features for extraction {labeling_job.extraction_job_id}")

            # Fetch template configuration - use specified template or fall back to DB default
            template_config = None
            max_examples = 10  # Default for miStudio Internal
            from src.models.labeling_prompt_template import LabelingPromptTemplate

            template = None
            if labeling_job.prompt_template_id:
                template = self.db.query(LabelingPromptTemplate).filter(
                    LabelingPromptTemplate.id == labeling_job.prompt_template_id
                ).first()
            else:
                # No template specified - look up the default template from DB
                template = self.db.query(LabelingPromptTemplate).filter(
                    LabelingPromptTemplate.is_default == True  # noqa: E712
                ).first()
                if template:
                    logger.info(f"No template specified in job - using DB default: {template.name}")

            # BOUND BEFORE THE BRANCH (MIS-E2E-059).
            #
            # `job_batch_size` was assigned only inside `if template:` and read
            # unconditionally at three later points, so the explicitly-supported
            # "no template found" path died with UnboundLocalError — a labeling
            # run against a deleted template crashed with a Python error instead
            # of falling back, and surfaced as a generic 500 / FAILED job.
            job_max_examples = None
            job_batch_size = 10
            if labeling_job.statistics and isinstance(labeling_job.statistics, dict):
                job_max_examples = labeling_job.statistics.get('max_examples')
                job_batch_size = labeling_job.statistics.get('batch_size', 10)

            if template:
                # Check for job-level overrides in statistics
                if labeling_job.statistics and isinstance(labeling_job.statistics, dict):
                    job_max_examples = labeling_job.statistics.get('max_examples')
                    job_batch_size = labeling_job.statistics.get('batch_size', 10)

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
                    'is_detection_template': template.is_detection_template,
                    'include_nlp_analysis': getattr(template, 'include_nlp_analysis', False),
                }

                override_msg = f" (job override)" if job_max_examples is not None else ""
                logger.info(f"Using template: {template.name} (type: {template.template_type}, K={max_examples}{override_msg})")

            # Provide hardcoded fallback if no template found in DB at all
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
                    'is_detection_template': False,
                    'include_nlp_analysis': False,
                }
                logger.warning(f"No template found (specified or default) - using hardcoded fallback (K={max_examples})")

            # Retrieve activation examples using efficient SQL batching
            # For NLP analysis, we need ALL examples (up to 100)
            # For display in LLM prompt, we only show top max_examples (default 10)
            BATCH_SIZE = 1000
            NLP_ANALYSIS_EXAMPLES = 100  # Retrieve all examples for comprehensive NLP analysis
            include_nlp = template_config.get('include_nlp_analysis', False)
            retrieval_count = NLP_ANALYSIS_EXAMPLES if include_nlp else max_examples
            features_examples = []  # Top max_examples for LLM display
            all_features_examples = []  # All examples for NLP analysis (empty when NLP disabled)
            neuron_indices = []

            # Phase 1: Examples Retrieval with progress tracking
            logger.info(f"Starting examples retrieval phase for {total_features} features (display K={max_examples}, NLP={'enabled (K='+str(retrieval_count)+')' if include_nlp else 'disabled'}) in batches of {BATCH_SIZE}")

            for batch_start in range(0, total_features, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_features)
                batch_features = features[batch_start:batch_end]
                batch_size = len(batch_features)

                logger.info(f"Retrieving batch {batch_start//BATCH_SIZE + 1}/{(total_features + BATCH_SIZE - 1)//BATCH_SIZE}: features {batch_start+1}-{batch_end}")

                # Get feature IDs for this batch
                batch_feature_ids = [f.id for f in batch_features]

                # Retrieve examples: all 100 for NLP analysis, or just max_examples if NLP disabled
                # Use sync version since Celery worker uses sync session
                all_examples_map = self._retrieve_top_examples_batch_sync(
                    session=self.db,
                    feature_ids=batch_feature_ids,
                    max_examples=retrieval_count
                )

                # Build ordered lists for labeling (maintain feature order)
                for feature in batch_features:
                    all_examples = all_examples_map.get(feature.id, [])
                    if include_nlp:
                        # Store all examples for NLP analysis (keep activation order)
                        all_features_examples.append(all_examples)
                        # Shuffle top examples before LLM display to break primacy bias
                        llm_examples = list(all_examples[:max_examples])
                        random.shuffle(llm_examples)
                        features_examples.append(llm_examples)
                    else:
                        # NLP disabled: shuffle all retrieved examples to break primacy bias
                        llm_examples = list(all_examples)
                        random.shuffle(llm_examples)
                        all_features_examples.append([])
                        features_examples.append(llm_examples)
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

            # Apply context-based pre-labeling filter: skip features whose prime tokens
            # are predominantly junk (punctuation, whitespace, single non-alphanumeric chars).
            from src.utils.token_filter import get_feature_filter
            feature_filter = get_feature_filter()
            features, features_examples, all_features_examples, filter_stats = (
                feature_filter.filter_features_from_examples(
                    features, features_examples, all_features_examples
                )
            )
            total_features = len(features)
            logger.info(
                f"Pre-labeling filter: {filter_stats['features_to_label']}/{filter_stats['total_features']} "
                f"features pass ({filter_stats['features_skipped']} skipped as junk, "
                f"{filter_stats['skip_percentage']:.1f}%)"
            )

            # A filter that removed EVERYTHING is a failure, not a completed job.
            #
            # total_features is 0 here, so every `range(0, total_features, ...)`
            # label loop below is a no-op, `labels` stays empty, and the terminal
            # write records COMPLETED / progress=1.0 / features_labeled=0 with
            # avg_label_length=0 — a silent success that looks like a finished run
            # and labeled nothing. Raising converts it into a FAILED job carrying
            # the reason. The smaller the working set the likelier this is, so a
            # scoped trial panel is the case that most needs it.
            if total_features == 0:
                raise ValueError(
                    f"pre-labeling junk filter removed all "
                    f"{filter_stats['total_features']} features; nothing to label"
                )

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
                        LABEL_BATCH_SIZE = job_batch_size

                        logger.info(f"Starting incremental labeling: {total_features} features in batches of {LABEL_BATCH_SIZE}")

                        for batch_start in range(0, total_features, LABEL_BATCH_SIZE):
                            # Stop promptly when the user cancels (see
                            # _raise_if_cancelled: revoke cannot kill a solo-pool task).
                            self._raise_if_cancelled(labeling_job_id)
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
                                # Never overwrite features completed by enhanced labeling
                                if feature.star_color == 'aqua':
                                    logger.debug("Skipping feature %s (star_color=aqua, enhanced labeling result preserved)", feature.id)
                                    continue

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
                    # Decrypt key stored in labeling_job (encrypt_value on write, decrypt here).
                    # decrypt_value() gracefully handles legacy plaintext rows.
                    # Fallback chain: labeling_job → DB app_settings → env.
                    openai_api_key = None
                    if labeling_job.openai_api_key:
                        openai_api_key = decrypt_value(labeling_job.openai_api_key, setting_key="openai_api_key")

                    if not openai_api_key:
                        logger.warning("No API key in labeling job, checking DB app_settings")
                        try:
                            from src.models.app_setting import AppSetting
                            db_setting = self.db.query(AppSetting).filter(AppSetting.key == "openai_api_key").first()
                            if db_setting:
                                openai_api_key = decrypt_value(db_setting.value, setting_key="openai_api_key")
                                logger.info("Using OpenAI API key from DB app_settings")
                        except Exception as e:
                            logger.warning(f"Failed to read API key from DB app_settings: {e}")

                    if not openai_api_key:
                        logger.warning("Falling back to OPENAI_API_KEY environment variable")
                        openai_api_key = getattr(settings, 'openai_api_key', None)

                    if not openai_api_key:
                        raise ValueError("OpenAI API key not provided and not found in settings")

                    openai_model = labeling_job.openai_model or "gpt-4o-mini"

                    # Fetch prompt template - use specified or fall back to DB default
                    system_message = None
                    user_prompt_template = None
                    temperature = 0.3
                    max_tokens = labeling_job.max_tokens or 300
                    top_p = 0.9

                    from src.models.labeling_prompt_template import LabelingPromptTemplate
                    template = None
                    if labeling_job.prompt_template_id:
                        template = self.db.query(LabelingPromptTemplate).filter(
                            LabelingPromptTemplate.id == labeling_job.prompt_template_id
                        ).first()
                    else:
                        template = self.db.query(LabelingPromptTemplate).filter(
                            LabelingPromptTemplate.is_default == True  # noqa: E712
                        ).first()
                        if template:
                            logger.info(f"No template in job - using DB default: {template.name}")

                    if template:
                        system_message = template.system_message
                        user_prompt_template = template.user_prompt_template
                        temperature = template.temperature
                        # THE JOB'S VALUE WINS (MIS-E2E-060).
                        #
                        # `max_tokens` is exposed on the API and in the UI as a
                        # per-job setting and was then unconditionally replaced
                        # by the template's (default 50). A user raising it to
                        # get longer descriptions had the value accepted and
                        # every description still truncated — a control that
                        # appears to work and does nothing.
                        #
                        # The sibling `max_examples` already gets this
                        # precedence right; this is the same rule.
                        if labeling_job.max_tokens:
                            max_tokens = labeling_job.max_tokens
                        else:
                            max_tokens = template.max_tokens
                        top_p = template.top_p
                        logger.info(f"Using prompt template: {template.name} (ID: {template.id})")

                    api_timeout = labeling_job.api_timeout

                    # Generate and persist labels in batches of 10
                    # This ensures progress is saved incrementally if the job fails
                    label_source_value = LabelSource.OPENAI.value
                    labeled_at = datetime.now(timezone.utc)
                    LABEL_BATCH_SIZE = job_batch_size

                    logger.info(f"Starting incremental labeling: {total_features} features in batches of {LABEL_BATCH_SIZE}")

                    # Create event loop BEFORE the OpenAI service so that httpx
                    # AsyncClient and asyncio.Semaphore bind to this loop.
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
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

                        # Pre-load logit effects for all features (one bulk query, avoids N+1)
                        feature_logit_effects: Dict[str, Optional[Dict]] = {}
                        if template_config.get('include_logit_effects'):
                            from src.models.feature_dashboard import FeatureDashboardData
                            n_promoted = template_config.get('top_promoted_tokens_count', 10)
                            n_suppressed = template_config.get('top_suppressed_tokens_count', 10)
                            feature_ids = [f.id for f in features]
                            dashboard_rows = self.db.query(
                                FeatureDashboardData.feature_id,
                                FeatureDashboardData.logit_lens_data
                            ).filter(
                                FeatureDashboardData.feature_id.in_(feature_ids)
                            ).all()
                            for row in dashboard_rows:
                                lens = row.logit_lens_data or {}
                                top_pos = lens.get('top_positive', [])[:n_promoted]
                                top_neg = lens.get('top_negative', [])[:n_suppressed]
                                feature_logit_effects[row.feature_id] = {
                                    'top_promoted': [t['token'] for t in top_pos],
                                    'top_suppressed': [t['token'] for t in top_neg],
                                }
                            logger.info(f"Pre-loaded logit effects for {len(feature_logit_effects)}/{len(feature_ids)} features")

                        for batch_start in range(0, total_features, LABEL_BATCH_SIZE):
                            # Stop promptly when the user cancels (see
                            # _raise_if_cancelled: revoke cannot kill a solo-pool task).
                            self._raise_if_cancelled(labeling_job_id)
                            batch_end = min(batch_start + LABEL_BATCH_SIZE, total_features)
                            batch_features = features[batch_start:batch_end]
                            batch_examples = features_examples[batch_start:batch_end]
                            batch_all_examples = all_features_examples[batch_start:batch_end]

                            # Generate labels for this batch using context-based examples
                            # Create concurrent tasks for all features in batch
                            # Pass all examples for NLP analysis to improve labeling
                            batch_labels = self._label_batch(
                                labeling_service=labeling_service,
                                loop=loop,
                                batch_features=batch_features,
                                batch_examples=batch_examples,
                                batch_all_examples=batch_all_examples,
                                feature_logit_effects=feature_logit_effects,
                                template_config=template_config,
                                user_prompt_template=user_prompt_template,
                                system_message=system_message,
                            )

                            # Persist this batch immediately
                            for feature, label, examples in zip(batch_features, batch_labels, batch_examples):
                                # Handle any exceptions
                                if isinstance(label, Exception):
                                    logger.error(f"Error generating label for feature {feature.id}: {label}")
                                    label = {"category": "error_feature", "specific": f"feature_{feature.neuron_index}", "description": ""}

                                # Never overwrite features completed by enhanced labeling
                                if feature.star_color == 'aqua':
                                    logger.debug("Skipping feature %s (star_color=aqua, enhanced labeling result preserved)", feature.id)
                                    continue

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
                    finally:
                        # Clean up: close httpx client, then shut down the loop
                        loop.run_until_complete(labeling_service._http_client.aclose())
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.close()

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

                    # Fetch prompt template - use specified or fall back to DB default
                    system_message = None
                    user_prompt_template = None
                    temperature = 0.3
                    max_tokens = labeling_job.max_tokens or 300
                    top_p = 0.9

                    from src.models.labeling_prompt_template import LabelingPromptTemplate
                    template = None
                    if labeling_job.prompt_template_id:
                        template = self.db.query(LabelingPromptTemplate).filter(
                            LabelingPromptTemplate.id == labeling_job.prompt_template_id
                        ).first()
                    else:
                        template = self.db.query(LabelingPromptTemplate).filter(
                            LabelingPromptTemplate.is_default == True  # noqa: E712
                        ).first()
                        if template:
                            logger.info(f"No template in job - using DB default: {template.name}")

                    if template:
                        system_message = template.system_message
                        user_prompt_template = template.user_prompt_template
                        temperature = template.temperature
                        # THE JOB'S VALUE WINS (MIS-E2E-060).
                        #
                        # `max_tokens` is exposed on the API and in the UI as a
                        # per-job setting and was then unconditionally replaced
                        # by the template's (default 50). A user raising it to
                        # get longer descriptions had the value accepted and
                        # every description still truncated — a control that
                        # appears to work and does nothing.
                        #
                        # The sibling `max_examples` already gets this
                        # precedence right; this is the same rule.
                        if labeling_job.max_tokens:
                            max_tokens = labeling_job.max_tokens
                        else:
                            max_tokens = template.max_tokens
                        top_p = template.top_p
                        logger.info(f"Using prompt template: {template.name} (ID: {template.id})")

                    api_timeout = labeling_job.api_timeout

                    # Generate and persist labels in batches of 10
                    # This ensures progress is saved incrementally if the job fails
                    label_source_value = LabelSource.OPENAI.value  # Use OPENAI source for compatible endpoints
                    labeled_at = datetime.now(timezone.utc)
                    LABEL_BATCH_SIZE = job_batch_size

                    logger.info(f"Starting incremental labeling: {total_features} features in batches of {LABEL_BATCH_SIZE}")

                    # Ensure the miLLM model is loaded before making inference calls.
                    # No-ops silently for non-miLLM endpoints (Ollama, vLLM, etc.).
                    logger.info(f"Ensuring model {model_name!r} is loaded at {endpoint}")
                    ensure_model_loaded(endpoint, model_name)

                    # Create event loop BEFORE the OpenAI service so that httpx
                    # AsyncClient and asyncio.Semaphore bind to this loop.
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
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

                        # Pre-load logit effects for all features (one bulk query, avoids N+1)
                        feature_logit_effects: Dict[str, Optional[Dict]] = {}
                        if template_config.get('include_logit_effects'):
                            from src.models.feature_dashboard import FeatureDashboardData
                            n_promoted = template_config.get('top_promoted_tokens_count', 10)
                            n_suppressed = template_config.get('top_suppressed_tokens_count', 10)
                            feature_ids = [f.id for f in features]
                            dashboard_rows = self.db.query(
                                FeatureDashboardData.feature_id,
                                FeatureDashboardData.logit_lens_data
                            ).filter(
                                FeatureDashboardData.feature_id.in_(feature_ids)
                            ).all()
                            for row in dashboard_rows:
                                lens = row.logit_lens_data or {}
                                top_pos = lens.get('top_positive', [])[:n_promoted]
                                top_neg = lens.get('top_negative', [])[:n_suppressed]
                                feature_logit_effects[row.feature_id] = {
                                    'top_promoted': [t['token'] for t in top_pos],
                                    'top_suppressed': [t['token'] for t in top_neg],
                                }
                            logger.info(f"Pre-loaded logit effects for {len(feature_logit_effects)}/{len(feature_ids)} features")

                        for batch_start in range(0, total_features, LABEL_BATCH_SIZE):
                            # Stop promptly when the user cancels (see
                            # _raise_if_cancelled: revoke cannot kill a solo-pool task).
                            self._raise_if_cancelled(labeling_job_id)
                            batch_end = min(batch_start + LABEL_BATCH_SIZE, total_features)
                            batch_features = features[batch_start:batch_end]
                            batch_examples = features_examples[batch_start:batch_end]
                            batch_all_examples = all_features_examples[batch_start:batch_end]

                            # Generate labels for this batch using context-based examples
                            # Create concurrent tasks for all features in batch
                            # Pass all examples for NLP analysis to improve labeling
                            batch_labels = self._label_batch(
                                labeling_service=labeling_service,
                                loop=loop,
                                batch_features=batch_features,
                                batch_examples=batch_examples,
                                batch_all_examples=batch_all_examples,
                                feature_logit_effects=feature_logit_effects,
                                template_config=template_config,
                                user_prompt_template=user_prompt_template,
                                system_message=system_message,
                            )

                            # Persist this batch immediately
                            for feature, label, examples in zip(batch_features, batch_labels, batch_examples):
                                # Handle any exceptions
                                if isinstance(label, Exception):
                                    logger.error(f"Error generating label for feature {feature.id}: {label}")
                                    label = {"category": "error_feature", "specific": f"feature_{feature.neuron_index}", "description": ""}

                                # Never overwrite features completed by enhanced labeling
                                if feature.star_color == 'aqua':
                                    logger.debug("Skipping feature %s (star_color=aqua, enhanced labeling result preserved)", feature.id)
                                    continue

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
                    finally:
                        # Clean up: close httpx client, then shut down the loop
                        loop.run_until_complete(labeling_service._http_client.aclose())
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.close()

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

        except LabelingService._LabelingCancelled:
            # A DELIBERATE CANCELLATION IS NOT A FAILURE (MIS-E2E-058).
            #
            # This fell through to the handler below, which set status=FAILED
            # and emitted `labeling:failed` before re-raising — so a user
            # pressing Cancel saw the job reported as broken. Worse,
            # `labeling_tasks.py` carries a comment asserting "the job row is
            # already CANCELLED", which was false precisely because this
            # handler had just overwritten it, so the next reader would not
            # look.
            #
            # Only reachable now that MIS-E2E-057 is fixed: before that the
            # cancellation was never raised at all.
            logger.info(f"Labeling job {labeling_job_id} was cancelled by the user")
            labeling_job.status = LabelingStatus.CANCELLED.value
            labeling_job.updated_at = datetime.now(timezone.utc)
            self.db.commit()

            emit_labeling_progress(
                labeling_job_id=labeling_job.id,
                event="labeling:cancelled",
                data={
                    "labeling_job_id": labeling_job.id,
                    "extraction_job_id": labeling_job.extraction_job_id,
                    "status": LabelingStatus.CANCELLED.value,
                    "message": "Labeling cancelled",
                },
            )
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
        finally:
            # Always release HTTP client file descriptors, whether we succeeded or failed.
            if 'labeling_service' in locals() and hasattr(labeling_service, 'close'):
                try:
                    labeling_service.close()
                except Exception:
                    pass

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

        # THROUGH THE REGISTRY, so the word written here is by construction the
        # word `_raise_if_cancelled` reads. Writing the status inline worked
        # only because both sides happened to spell it the same way — which is
        # exactly what `saes.py` did not, where the endpoint wrote FAILED and a
        # checker looking for "cancelled" could never have seen it.
        from starlette.concurrency import run_in_threadpool

        from ..core.cancellation import request_cancel

        await run_in_threadpool(
            request_cancel, "labeling", labeling_job_id,
            reason="Cancelled by user",
            celery_task_id=getattr(labeling_job, "celery_task_id", None),
        )
        await self.db.refresh(labeling_job)

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
