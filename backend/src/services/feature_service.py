"""
Feature service for CRUD operations and search.

This service provides feature discovery and management capabilities:
- List and search features with filtering and pagination
- Get detailed feature information
- Update feature metadata (name, description, notes)
- Toggle favorite status
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, or_
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

    def __init__(self, db: Session):
        """Initialize feature service."""
        self.db = db

    def list_features(
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
        query = self.db.query(Feature).filter(Feature.training_id == training_id)

        # Task 9.4: Apply full-text search filter if specified
        if search_params.search:
            # Use PostgreSQL full-text search with to_tsquery
            # The GIN index on (to_tsvector('english', name || ' ' || description)) handles this efficiently
            search_vector = func.to_tsvector('english', Feature.name + ' ' + func.coalesce(Feature.description, ''))
            search_query = func.to_tsquery('english', func.replace(search_params.search, ' ', ' & '))
            query = query.filter(search_vector.op('@@')(search_query))

        # Task 9.5: Apply is_favorite filter if specified
        if search_params.is_favorite is not None:
            query = query.filter(Feature.is_favorite == search_params.is_favorite)

        # Get total count before pagination
        total = query.count()

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
        features = query.all()

        # Task 9.8: For each feature, include one example_context
        feature_responses = []
        for feature in features:
            # Get first max-activating example
            example = (
                self.db.query(FeatureActivation)
                .filter(FeatureActivation.feature_id == feature.id)
                .order_by(desc(FeatureActivation.max_activation))
                .first()
            )

            example_context = None
            if example:
                example_context = FeatureActivationExample(
                    tokens=example.tokens,
                    activations=example.activations,
                    max_activation=example.max_activation,
                    sample_index=example.sample_index
                )

            feature_response = FeatureResponse(
                id=feature.id,
                training_id=feature.training_id,
                extraction_job_id=feature.extraction_job_id,
                neuron_index=feature.neuron_index,
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
        # Get all features for this training (without pagination) for statistics
        all_features_query = self.db.query(Feature).filter(Feature.training_id == training_id)

        # Total features
        total_features = all_features_query.count()

        # Interpretable percentage (interpretability_score > 0.5)
        interpretable_count = all_features_query.filter(Feature.interpretability_score > 0.5).count()
        interpretable_percentage = (interpretable_count / total_features * 100) if total_features > 0 else 0.0

        # Average activation frequency
        avg_activation_freq_result = self.db.query(
            func.avg(Feature.activation_frequency)
        ).filter(Feature.training_id == training_id).scalar()
        avg_activation_frequency = float(avg_activation_freq_result) if avg_activation_freq_result else 0.0

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

    def get_feature_detail(self, feature_id: str) -> Optional[FeatureDetailResponse]:
        """
        Get detailed information about a feature.

        Args:
            feature_id: ID of the feature

        Returns:
            FeatureDetailResponse with computed active_samples field, or None if not found
        """
        # Task 9.10: Load feature record
        feature = self.db.query(Feature).filter(Feature.id == feature_id).first()

        if not feature:
            return None

        # Calculate active_samples (activation_frequency * total_evaluation_samples)
        # Get extraction job to find evaluation_samples count
        extraction_job = self.db.query(ExtractionJob).filter(
            ExtractionJob.id == feature.extraction_job_id
        ).first()

        evaluation_samples = extraction_job.config.get("evaluation_samples", 10000) if extraction_job else 10000
        active_samples = int(feature.activation_frequency * evaluation_samples)

        return FeatureDetailResponse(
            id=feature.id,
            training_id=feature.training_id,
            extraction_job_id=feature.extraction_job_id,
            neuron_index=feature.neuron_index,
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

    def update_feature(
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
        feature = self.db.query(Feature).filter(Feature.id == feature_id).first()

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

        self.db.commit()
        self.db.refresh(feature)

        logger.info(f"Updated feature {feature_id}: name_changed={name_changed}")

        # Return updated feature (without example_context for update response)
        return FeatureResponse(
            id=feature.id,
            training_id=feature.training_id,
            extraction_job_id=feature.extraction_job_id,
            neuron_index=feature.neuron_index,
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

    def toggle_favorite(self, feature_id: str, is_favorite: bool) -> Optional[bool]:
        """
        Toggle favorite status for a feature.

        Args:
            feature_id: ID of the feature
            is_favorite: New favorite status

        Returns:
            New is_favorite value, or None if feature not found
        """
        # Task 9.12: Load feature, update is_favorite
        feature = self.db.query(Feature).filter(Feature.id == feature_id).first()

        if not feature:
            return None

        feature.is_favorite = is_favorite
        feature.updated_at = datetime.now(timezone.utc)

        self.db.commit()

        logger.info(f"Toggled favorite for feature {feature_id}: is_favorite={is_favorite}")

        return is_favorite

    def get_feature_examples(
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
        examples = (
            self.db.query(FeatureActivation)
            .filter(FeatureActivation.feature_id == feature_id)
            .order_by(desc(FeatureActivation.max_activation))
            .limit(limit)
            .all()
        )

        return [
            FeatureActivationExample(
                tokens=example.tokens,
                activations=example.activations,
                max_activation=example.max_activation,
                sample_index=example.sample_index
            )
            for example in examples
        ]


def get_feature_service(db: Session) -> FeatureService:
    """Dependency injection helper for FeatureService."""
    return FeatureService(db)
