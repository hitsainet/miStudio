"""
Feature service for CRUD operations and search.

This service provides feature discovery and management capabilities:
- List and search features with filtering and pagination
- Get detailed feature information
- Update feature metadata (name, description, notes)
- Toggle favorite status
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, or_, select, String, exists, literal_column
from sqlalchemy.sql import text

from src.models.feature import Feature, LabelSource
from src.models.feature_activation import FeatureActivation
from src.models.extraction_job import ExtractionJob
from src.schemas.feature import (
    FeatureSearchRequest,
    FeatureResponse,
    FeatureListResponse,
    FeatureDetailResponse,
    FeatureUpdateRequest,
    FeatureStatistics,
    FeatureActivationExample
)
from src.utils.token_filters import analyze_feature_tokens


logger = logging.getLogger(__name__)


class FeatureService:
    """
    Service for feature CRUD operations and search.

    Provides methods for:
    - Listing features with search, filtering, sorting, and pagination
    - Getting detailed feature information
    - Updating feature metadata
    - Managing favorite features
    """

    def __init__(self, db: Union[AsyncSession, Session]):
        """Initialize feature service with either async or sync session."""
        self.db = db

    async def list_features(
        self,
        training_id: str,
        search_params: FeatureSearchRequest
    ) -> FeatureListResponse:
        """
        List features with search, filtering, sorting, and pagination.

        Args:
            training_id: ID of the training to list features for
            search_params: Search parameters (search query, filters, sort, pagination)

        Returns:
            FeatureListResponse with features, pagination info, and statistics
        """
        # Task 9.3: Build base query with training_id filter
        query = select(Feature).where(Feature.training_id == training_id)

        # Task 9.4: Apply search filter if specified
        if search_params.search:
            # Use ILIKE for substring matching (case-insensitive) in:
            # 1. Feature name and description
            # 2. Activation tokens (from feature_activations.tokens JSONB array)
            search_pattern = f'%{search_params.search}%'

            # Subquery to check if any activation tokens match
            # Join FeatureActivation with Feature to filter by training_id
            token_subquery = (
                select(FeatureActivation.feature_id)
                .join(Feature, FeatureActivation.feature_id == Feature.id)
                .where(Feature.training_id == training_id)
                .where(func.cast(FeatureActivation.tokens, String).ilike(search_pattern))
            )

            query = query.where(
                or_(
                    Feature.name.ilike(search_pattern),
                    Feature.description.ilike(search_pattern),
                    Feature.id.in_(token_subquery)
                )
            )

        # Task 9.5: Apply is_favorite filter if specified
        if search_params.is_favorite is not None:
            query = query.where(Feature.is_favorite == search_params.is_favorite)

        # Get total count before pagination
        count_query = select(func.count()).select_from(Feature).where(Feature.training_id == training_id)
        if search_params.search:
            search_pattern = f'%{search_params.search}%'

            # Same token search subquery for count
            token_subquery = (
                select(FeatureActivation.feature_id)
                .join(Feature, FeatureActivation.feature_id == Feature.id)
                .where(Feature.training_id == training_id)
                .where(func.cast(FeatureActivation.tokens, String).ilike(search_pattern))
            )

            count_query = count_query.where(
                or_(
                    Feature.name.ilike(search_pattern),
                    Feature.description.ilike(search_pattern),
                    Feature.id.in_(token_subquery)
                )
            )
        if search_params.is_favorite is not None:
            count_query = count_query.where(Feature.is_favorite == search_params.is_favorite)

        total_result = await self.db.execute(count_query)
        total = total_result.scalar_one()

        # Task 9.6: Apply sorting
        if search_params.sort_by == "activation_freq":
            sort_column = Feature.activation_frequency
        elif search_params.sort_by == "interpretability":
            sort_column = Feature.interpretability_score
        else:  # "feature_id"
            sort_column = Feature.id

        if search_params.sort_order == "asc":
            query = query.order_by(asc(sort_column))
        else:  # "desc"
            query = query.order_by(desc(sort_column))

        # Task 9.7: Apply pagination
        query = query.limit(search_params.limit).offset(search_params.offset)

        # Execute query to get features
        result = await self.db.execute(query)
        features = result.scalars().all()

        # Task 9.8: For each feature, include one example_context
        feature_responses = []
        for feature in features:
            example_context = None

            # Priority 1: Use example_tokens_summary if available (filtered top tokens from labeling)
            if feature.example_tokens_summary:
                summary = feature.example_tokens_summary
                example_context = FeatureActivationExample(
                    tokens=summary.get('tokens', []),
                    activations=summary.get('activations', []),
                    max_activation=summary.get('max_activation', 0.0),
                    sample_index=-1  # -1 indicates this is a summary, not a specific sample
                )
            else:
                # Priority 2: Fall back to max-activating example from FeatureActivation table
                example_query = (
                    select(FeatureActivation)
                    .where(FeatureActivation.feature_id == feature.id)
                    .order_by(desc(FeatureActivation.max_activation))
                    .limit(1)
                )
                example_result = await self.db.execute(example_query)
                example = example_result.scalar_one_or_none()

                if example:
                    # Handle both legacy and new data formats
                    # Legacy: tokens is a dict with all_tokens key (from old extraction code)
                    # New: tokens is a simple list, context window in dedicated columns
                    if isinstance(example.tokens, dict):
                        # Old extraction format: tokens field is a dict
                        tokens_list = example.tokens.get("all_tokens", example.tokens.get("tokens", []))
                        example_context = FeatureActivationExample(
                            tokens=tokens_list,
                            activations=example.activations,
                            max_activation=example.max_activation,
                            sample_index=example.sample_index,
                            prefix_tokens=example.tokens.get("prefix_tokens"),
                            prime_token=example.tokens.get("prime_token"),
                            suffix_tokens=example.tokens.get("suffix_tokens"),
                            prime_activation_index=example.tokens.get("prime_activation_index"),
                            token_positions=example.tokens.get("token_positions")
                        )
                    else:
                        # New extraction format or legacy: tokens is a list, check dedicated columns
                        example_context = FeatureActivationExample(
                            tokens=example.tokens,
                            activations=example.activations,
                            max_activation=example.max_activation,
                            sample_index=example.sample_index,
                            prefix_tokens=example.prefix_tokens,
                            prime_token=example.prime_token,
                            suffix_tokens=example.suffix_tokens,
                            prime_activation_index=example.prime_activation_index
                        )

            feature_response = FeatureResponse(
                id=feature.id,
                training_id=feature.training_id,
                extraction_job_id=feature.extraction_job_id,
                neuron_index=feature.neuron_index,
                category=feature.category,
                name=feature.name,
                description=feature.description,
                label_source=feature.label_source,
                activation_frequency=feature.activation_frequency,
                interpretability_score=feature.interpretability_score,
                max_activation=feature.max_activation,
                mean_activation=feature.mean_activation,
                is_favorite=feature.is_favorite,
                notes=feature.notes,
                created_at=feature.created_at,
                updated_at=feature.updated_at,
                example_context=example_context
            )
            feature_responses.append(feature_response)

        # Task 9.9: Calculate statistics
        # Total features
        total_features_query = select(func.count()).select_from(Feature).where(Feature.training_id == training_id)
        total_features_result = await self.db.execute(total_features_query)
        total_features = total_features_result.scalar_one()

        # Interpretable percentage (interpretability_score > 0.5)
        interpretable_count_query = (
            select(func.count())
            .select_from(Feature)
            .where(Feature.training_id == training_id, Feature.interpretability_score > 0.5)
        )
        interpretable_count_result = await self.db.execute(interpretable_count_query)
        interpretable_count = interpretable_count_result.scalar_one()
        interpretable_percentage = (interpretable_count / total_features * 100) if total_features > 0 else 0.0

        # Average activation frequency
        avg_activation_freq_query = (
            select(func.avg(Feature.activation_frequency))
            .where(Feature.training_id == training_id)
        )
        avg_activation_freq_result = await self.db.execute(avg_activation_freq_query)
        avg_activation_freq_value = avg_activation_freq_result.scalar_one_or_none()
        avg_activation_frequency = float(avg_activation_freq_value) if avg_activation_freq_value else 0.0

        statistics = FeatureStatistics(
            total_features=total_features,
            interpretable_percentage=interpretable_percentage,
            avg_activation_frequency=avg_activation_frequency
        )

        return FeatureListResponse(
            features=feature_responses,
            total=total,
            limit=search_params.limit,
            offset=search_params.offset,
            statistics=statistics
        )

    async def list_features_by_extraction(
        self,
        extraction_job_id: str,
        search_params: FeatureSearchRequest
    ) -> FeatureListResponse:
        """
        List features for a specific extraction job with search, filtering, sorting, and pagination.

        Args:
            extraction_job_id: ID of the extraction job to list features for
            search_params: Search parameters (search query, filters, sort, pagination)

        Returns:
            FeatureListResponse with features, pagination info, and statistics
        """
        # Build base query with extraction_job_id filter
        query = select(Feature).where(Feature.extraction_job_id == extraction_job_id)

        # Apply search filter if specified
        if search_params.search:
            search_pattern = f'%{search_params.search}%'

            # Subquery to check if any activation tokens match
            # Join FeatureActivation with Feature to filter by extraction_job_id
            token_subquery = (
                select(FeatureActivation.feature_id)
                .join(Feature, FeatureActivation.feature_id == Feature.id)
                .where(Feature.extraction_job_id == extraction_job_id)
                .where(func.cast(FeatureActivation.tokens, String).ilike(search_pattern))
            )

            query = query.where(
                or_(
                    Feature.name.ilike(search_pattern),
                    Feature.description.ilike(search_pattern),
                    Feature.id.in_(token_subquery)
                )
            )

        # Apply is_favorite filter if specified
        if search_params.is_favorite is not None:
            query = query.where(Feature.is_favorite == search_params.is_favorite)

        # Get total count before pagination
        count_query = select(func.count()).select_from(Feature).where(Feature.extraction_job_id == extraction_job_id)
        if search_params.search:
            search_pattern = f'%{search_params.search}%'

            # Same token search subquery for count
            token_subquery = (
                select(FeatureActivation.feature_id)
                .join(Feature, FeatureActivation.feature_id == Feature.id)
                .where(Feature.extraction_job_id == extraction_job_id)
                .where(func.cast(FeatureActivation.tokens, String).ilike(search_pattern))
            )

            count_query = count_query.where(
                or_(
                    Feature.name.ilike(search_pattern),
                    Feature.description.ilike(search_pattern),
                    Feature.id.in_(token_subquery)
                )
            )
        if search_params.is_favorite is not None:
            count_query = count_query.where(Feature.is_favorite == search_params.is_favorite)

        total_result = await self.db.execute(count_query)
        total = total_result.scalar_one()

        # Apply sorting
        if search_params.sort_by == "activation_freq":
            sort_column = Feature.activation_frequency
        elif search_params.sort_by == "interpretability":
            sort_column = Feature.interpretability_score
        else:  # "feature_id"
            sort_column = Feature.id

        if search_params.sort_order == "asc":
            query = query.order_by(asc(sort_column))
        else:  # "desc"
            query = query.order_by(desc(sort_column))

        # Apply pagination
        query = query.limit(search_params.limit).offset(search_params.offset)

        # Execute query to get features
        result = await self.db.execute(query)
        features = result.scalars().all()

        # For each feature, include one example_context
        feature_responses = []
        for feature in features:
            example_context = None

            # Priority 1: Use example_tokens_summary if available (filtered top tokens from labeling)
            if feature.example_tokens_summary:
                summary = feature.example_tokens_summary
                example_context = FeatureActivationExample(
                    tokens=summary.get('tokens', []),
                    activations=summary.get('activations', []),
                    max_activation=summary.get('max_activation', 0.0),
                    sample_index=-1  # -1 indicates this is a summary, not a specific sample
                )
            else:
                # Priority 2: Fall back to max-activating example from FeatureActivation table
                example_query = (
                    select(FeatureActivation)
                    .where(FeatureActivation.feature_id == feature.id)
                    .order_by(desc(FeatureActivation.max_activation))
                    .limit(1)
                )
                example_result = await self.db.execute(example_query)
                example = example_result.scalar_one_or_none()

                if example:
                    # Handle both legacy and new data formats
                    # Legacy: tokens is a dict with all_tokens key (from old extraction code)
                    # New: tokens is a simple list, context window in dedicated columns
                    if isinstance(example.tokens, dict):
                        # Old extraction format: tokens field is a dict
                        tokens_list = example.tokens.get("all_tokens", example.tokens.get("tokens", []))
                        example_context = FeatureActivationExample(
                            tokens=tokens_list,
                            activations=example.activations,
                            max_activation=example.max_activation,
                            sample_index=example.sample_index,
                            prefix_tokens=example.tokens.get("prefix_tokens"),
                            prime_token=example.tokens.get("prime_token"),
                            suffix_tokens=example.tokens.get("suffix_tokens"),
                            prime_activation_index=example.tokens.get("prime_activation_index"),
                            token_positions=example.tokens.get("token_positions")
                        )
                    else:
                        # New extraction format or legacy: tokens is a list, check dedicated columns
                        example_context = FeatureActivationExample(
                            tokens=example.tokens,
                            activations=example.activations,
                            max_activation=example.max_activation,
                            sample_index=example.sample_index,
                            prefix_tokens=example.prefix_tokens,
                            prime_token=example.prime_token,
                            suffix_tokens=example.suffix_tokens,
                            prime_activation_index=example.prime_activation_index
                        )

            feature_response = FeatureResponse(
                id=feature.id,
                training_id=feature.training_id,
                extraction_job_id=feature.extraction_job_id,
                neuron_index=feature.neuron_index,
                category=feature.category,
                name=feature.name,
                description=feature.description,
                label_source=feature.label_source,
                activation_frequency=feature.activation_frequency,
                interpretability_score=feature.interpretability_score,
                max_activation=feature.max_activation,
                mean_activation=feature.mean_activation,
                is_favorite=feature.is_favorite,
                notes=feature.notes,
                created_at=feature.created_at,
                updated_at=feature.updated_at,
                example_context=example_context
            )
            feature_responses.append(feature_response)

        # Calculate statistics
        # Total features
        total_features_query = select(func.count()).select_from(Feature).where(Feature.extraction_job_id == extraction_job_id)
        total_features_result = await self.db.execute(total_features_query)
        total_features = total_features_result.scalar_one()

        # Interpretable percentage (interpretability_score > 0.5)
        interpretable_count_query = (
            select(func.count())
            .select_from(Feature)
            .where(Feature.extraction_job_id == extraction_job_id, Feature.interpretability_score > 0.5)
        )
        interpretable_count_result = await self.db.execute(interpretable_count_query)
        interpretable_count = interpretable_count_result.scalar_one()
        interpretable_percentage = (interpretable_count / total_features * 100) if total_features > 0 else 0.0

        # Average activation frequency
        avg_activation_freq_query = (
            select(func.avg(Feature.activation_frequency))
            .where(Feature.extraction_job_id == extraction_job_id)
        )
        avg_activation_freq_result = await self.db.execute(avg_activation_freq_query)
        avg_activation_freq_value = avg_activation_freq_result.scalar_one_or_none()
        avg_activation_frequency = float(avg_activation_freq_value) if avg_activation_freq_value else 0.0

        statistics = FeatureStatistics(
            total_features=total_features,
            interpretable_percentage=interpretable_percentage,
            avg_activation_frequency=avg_activation_frequency
        )

        return FeatureListResponse(
            features=feature_responses,
            total=total,
            limit=search_params.limit,
            offset=search_params.offset,
            statistics=statistics
        )

    async def get_feature_detail(self, feature_id: str) -> Optional[FeatureDetailResponse]:
        """
        Get detailed information about a feature.

        Args:
            feature_id: ID of the feature

        Returns:
            FeatureDetailResponse with computed active_samples field, or None if not found
        """
        # Task 9.10: Load feature record
        feature_query = select(Feature).where(Feature.id == feature_id)
        feature_result = await self.db.execute(feature_query)
        feature = feature_result.scalar_one_or_none()

        if not feature:
            return None

        # Calculate active_samples (activation_frequency * total_evaluation_samples)
        # Get extraction job to find evaluation_samples count
        extraction_job_query = select(ExtractionJob).where(
            ExtractionJob.id == feature.extraction_job_id
        )
        extraction_job_result = await self.db.execute(extraction_job_query)
        extraction_job = extraction_job_result.scalar_one_or_none()

        evaluation_samples = extraction_job.config.get("evaluation_samples", 10000) if extraction_job else 10000
        active_samples = int(feature.activation_frequency * evaluation_samples)

        return FeatureDetailResponse(
            id=feature.id,
            training_id=feature.training_id,
            extraction_job_id=feature.extraction_job_id,
            neuron_index=feature.neuron_index,
            category=feature.category,
            name=feature.name,
            description=feature.description,
            label_source=feature.label_source,
            activation_frequency=feature.activation_frequency,
            interpretability_score=feature.interpretability_score,
            max_activation=feature.max_activation,
            mean_activation=feature.mean_activation,
            is_favorite=feature.is_favorite,
            notes=feature.notes,
            created_at=feature.created_at,
            updated_at=feature.updated_at,
            active_samples=active_samples
        )

    async def update_feature(
        self,
        feature_id: str,
        updates: FeatureUpdateRequest
    ) -> Optional[FeatureResponse]:
        """
        Update feature metadata (name, description, notes).

        Args:
            feature_id: ID of the feature to update
            updates: Fields to update

        Returns:
            Updated FeatureResponse, or None if feature not found
        """
        # Task 9.11: Load feature, validate updates
        feature_query = select(Feature).where(Feature.id == feature_id)
        feature_result = await self.db.execute(feature_query)
        feature = feature_result.scalar_one_or_none()

        if not feature:
            return None

        # Track if name changed to update label_source
        name_changed = False

        # Apply updates
        if updates.name is not None:
            if updates.name != feature.name:
                name_changed = True
            feature.name = updates.name

        if updates.description is not None:
            feature.description = updates.description

        if updates.notes is not None:
            feature.notes = updates.notes

        # Set label_source='user' if name was changed
        if name_changed:
            feature.label_source = LabelSource.USER.value

        # Update timestamp
        feature.updated_at = datetime.now(timezone.utc)

        await self.db.commit()
        await self.db.refresh(feature)

        logger.info(f"Updated feature {feature_id}: name_changed={name_changed}")

        # Return updated feature (without example_context for update response)
        return FeatureResponse(
            id=feature.id,
            training_id=feature.training_id,
            extraction_job_id=feature.extraction_job_id,
            neuron_index=feature.neuron_index,
            category=feature.category,
            name=feature.name,
            description=feature.description,
            label_source=feature.label_source,
            activation_frequency=feature.activation_frequency,
            interpretability_score=feature.interpretability_score,
            max_activation=feature.max_activation,
            mean_activation=feature.mean_activation,
            is_favorite=feature.is_favorite,
            notes=feature.notes,
            created_at=feature.created_at,
            updated_at=feature.updated_at,
            example_context=None
        )

    async def toggle_favorite(self, feature_id: str, is_favorite: bool) -> Optional[bool]:
        """
        Toggle favorite status for a feature.

        Args:
            feature_id: ID of the feature
            is_favorite: New favorite status

        Returns:
            New is_favorite value, or None if feature not found
        """
        # Task 9.12: Load feature, update is_favorite
        feature_query = select(Feature).where(Feature.id == feature_id)
        feature_result = await self.db.execute(feature_query)
        feature = feature_result.scalar_one_or_none()

        if not feature:
            return None

        feature.is_favorite = is_favorite
        feature.updated_at = datetime.now(timezone.utc)

        await self.db.commit()

        logger.info(f"Toggled favorite for feature {feature_id}: is_favorite={is_favorite}")

        return is_favorite

    async def get_feature_examples(
        self,
        feature_id: str,
        limit: int = 100
    ) -> List[FeatureActivationExample]:
        """
        Get max-activating examples for a feature.

        Args:
            feature_id: ID of the feature
            limit: Maximum number of examples to return

        Returns:
            List of max-activating examples with tokens and activations
        """
        examples_query = (
            select(FeatureActivation)
            .where(FeatureActivation.feature_id == feature_id)
            .order_by(desc(FeatureActivation.max_activation))
            .limit(limit)
        )
        examples_result = await self.db.execute(examples_query)
        examples = examples_result.scalars().all()

        result = []
        for example in examples:
            # Check if tokens is an object (enhanced format) or array (legacy format)
            if isinstance(example.tokens, dict):
                # Enhanced format with context window
                result.append(FeatureActivationExample(
                    tokens=example.tokens.get("all_tokens", []),
                    activations=example.activations,
                    max_activation=example.max_activation,
                    sample_index=example.sample_index,
                    prefix_tokens=example.tokens.get("prefix_tokens"),
                    prime_token=example.tokens.get("prime_token"),
                    suffix_tokens=example.tokens.get("suffix_tokens"),
                    prime_activation_index=example.tokens.get("prime_activation_index"),
                    token_positions=example.tokens.get("token_positions")
                ))
            else:
                # New extraction format: tokens is a list, context window in dedicated columns
                result.append(FeatureActivationExample(
                    tokens=example.tokens,
                    activations=example.activations,
                    max_activation=example.max_activation,
                    sample_index=example.sample_index,
                    prefix_tokens=example.prefix_tokens,
                    prime_token=example.prime_token,
                    suffix_tokens=example.suffix_tokens,
                    prime_activation_index=example.prime_activation_index
                ))

        return result

    async def get_feature_token_analysis(
        self,
        feature_id: str,
        apply_filters: bool = True,
        filter_special: bool = True,
        filter_single_char: bool = True,
        filter_punctuation: bool = True,
        filter_numbers: bool = True,
        filter_fragments: bool = True,
        filter_stop_words: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get token analysis for a feature's activation examples.

        Analyzes all tokens from the feature's max-activating examples,
        applies filtering to remove junk tokens, and returns statistics
        and ranked token list.

        Args:
            feature_id: ID of the feature
            apply_filters: Master switch for all filtering (default: True)
            filter_special: Filter special tokens (<s>, </s>, etc.)
            filter_single_char: Filter single character tokens
            filter_punctuation: Filter pure punctuation
            filter_numbers: Filter pure numeric tokens
            filter_fragments: Filter word fragments (BPE subwords)
            filter_stop_words: Filter common stop words (the, and, is, etc.)

        Returns:
            Dictionary with summary statistics and ranked token list, or None if feature not found
        """
        # Verify feature exists
        feature_query = select(Feature).where(Feature.id == feature_id)
        feature_result = await self.db.execute(feature_query)
        feature = feature_result.scalar_one_or_none()

        if not feature:
            return None

        # Query all activations for this feature
        activations_query = (
            select(FeatureActivation)
            .where(FeatureActivation.feature_id == feature_id)
            .order_by(desc(FeatureActivation.max_activation))
        )
        activations_result = await self.db.execute(activations_query)
        activations = activations_result.scalars().all()

        # Extract tokens from each activation
        tokens_list = [activation.tokens for activation in activations if activation.tokens]

        # Analyze tokens using utility function with filter options
        analysis = analyze_feature_tokens(
            tokens_list,
            apply_filters=apply_filters,
            filter_special=filter_special,
            filter_single_char=filter_single_char,
            filter_punctuation=filter_punctuation,
            filter_numbers=filter_numbers,
            filter_fragments=filter_fragments,
            filter_stop_words=filter_stop_words
        )

        return analysis


def get_feature_service(db: Union[AsyncSession, Session]) -> FeatureService:
    """Dependency injection helper for FeatureService."""
    return FeatureService(db)
