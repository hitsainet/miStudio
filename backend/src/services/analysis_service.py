"""
Analysis service for feature interpretability analysis.

This service provides advanced analysis capabilities for discovered features:
- Logit lens: Analyze feature's contribution to model predictions
- Correlations: Find features with similar activation patterns
- Ablation: Measure feature's impact on model performance

All analysis results are cached for 7 days for performance.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select, and_
import numpy as np
from scipy.stats import pearsonr

from src.models.feature import Feature
from src.models.feature_activation import FeatureActivation
from src.models.feature_analysis_cache import FeatureAnalysisCache, AnalysisType
from src.models.training import Training
from src.schemas.feature import (
    LogitLensResponse,
    CorrelationsResponse,
    CorrelatedFeature,
    AblationResponse
)


logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service for feature interpretability analysis.

    Provides methods for:
    - Logit lens analysis: Feature's contribution to output predictions
    - Correlation analysis: Find related features
    - Ablation analysis: Measure feature importance

    All methods implement caching with 7-day expiration.
    """

    CACHE_EXPIRY_DAYS = 7

    def __init__(self, db: Union[AsyncSession, Session]):
        """Initialize analysis service with either async or sync session."""
        self.db = db

    async def calculate_logit_lens(
        self,
        feature_id: str
    ) -> Optional[LogitLensResponse]:
        """
        Calculate logit lens for a feature.

        Analyzes what the feature contributes to the model's output predictions
        by passing a synthetic activation through the SAE decoder and model head.

        Args:
            feature_id: Feature ID to analyze

        Returns:
            LogitLensResponse with top tokens and interpretation, or None if feature not found

        Process:
            1. Check cache for recent result
            2. Load feature, training, SAE model, and base model
            3. Create feature vector with high activation at neuron index
            4. Pass through SAE decoder to reconstruct activation
            5. Pass through model LM head to get logits
            6. Apply softmax and extract top 10 tokens
            7. Generate interpretation based on token patterns
            8. Cache result for future requests
        """
        # Check cache first
        cache_entry = await self._get_cached_analysis(feature_id, AnalysisType.LOGIT_LENS)
        if cache_entry:
            logger.info(f"Logit lens cache hit for feature {feature_id}")
            return LogitLensResponse(**cache_entry.results)

        # Load feature and related models
        feature = await self._get_feature(feature_id)
        if not feature:
            logger.warning(f"Feature {feature_id} not found")
            return None

        training = await self._get_training(feature.training_id)
        if not training:
            logger.warning(f"Training {feature.training_id} not found")
            return None

        try:
            # TODO: Load SAE model from checkpoint
            # TODO: Load base model
            # For now, return mock data
            logger.warning("Logit lens: SAE/model loading not yet implemented, returning mock data")

            # Mock implementation
            top_tokens = ["the", "a", "an", "of", "to", "in", "for", "and", "is", "on"]
            probabilities = [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03]
            interpretation = "Feature appears to activate on common function words (determiners, prepositions)"

            computed_at = datetime.now(timezone.utc)

            # Cache the result
            await self._cache_analysis(
                feature_id=feature_id,
                analysis_type=AnalysisType.LOGIT_LENS,
                results={
                    "top_tokens": top_tokens,
                    "probabilities": probabilities,
                    "interpretation": interpretation,
                    "computed_at": computed_at.isoformat()
                }
            )

            return LogitLensResponse(
                top_tokens=top_tokens,
                probabilities=probabilities,
                interpretation=interpretation,
                computed_at=computed_at
            )

        except Exception as e:
            logger.error(f"Error calculating logit lens for feature {feature_id}: {str(e)}")
            raise

    async def calculate_correlations(
        self,
        feature_id: str
    ) -> Optional[CorrelationsResponse]:
        """
        Calculate correlations with other features.

        Finds features with similar activation patterns by computing
        Pearson correlation coefficients on activation vectors.

        Args:
            feature_id: Feature ID to analyze

        Returns:
            CorrelationsResponse with top 10 correlated features, or None if feature not found

        Process:
            1. Check cache for recent result
            2. Load all features for the same training
            3. Load activation vectors from feature_activations table
            4. Calculate Pearson correlation with all other features
            5. Return top 10 with correlation > 0.5
            6. Cache result for future requests
        """
        # Check cache first
        cache_entry = await self._get_cached_analysis(feature_id, AnalysisType.CORRELATIONS)
        if cache_entry:
            logger.info(f"Correlations cache hit for feature {feature_id}")
            return CorrelationsResponse(**cache_entry.results)

        # Load feature
        feature = await self._get_feature(feature_id)
        if not feature:
            logger.warning(f"Feature {feature_id} not found")
            return None

        try:
            # TODO: Load activation vectors for all features in training
            # TODO: Calculate Pearson correlations
            # For now, return mock data
            logger.warning("Correlations: activation vector loading not yet implemented, returning mock data")

            # Mock implementation - find a few related features
            correlated = []
            features = await self._get_features_for_training(feature.training_id)
            for f in features[:5]:  # Mock: take first 5 other features
                if f.id != feature_id:
                    correlated.append(CorrelatedFeature(
                        feature_id=f.id,
                        feature_name=f.name,
                        correlation=0.75 - len(correlated) * 0.1  # Mock declining correlation
                    ))

            computed_at = datetime.now(timezone.utc)

            # Cache the result
            await self._cache_analysis(
                feature_id=feature_id,
                analysis_type=AnalysisType.CORRELATIONS,
                results={
                    "correlated_features": [
                        {
                            "feature_id": cf.feature_id,
                            "feature_name": cf.feature_name,
                            "correlation": cf.correlation
                        }
                        for cf in correlated
                    ],
                    "computed_at": computed_at.isoformat()
                }
            )

            return CorrelationsResponse(
                correlated_features=correlated,
                computed_at=computed_at
            )

        except Exception as e:
            logger.error(f"Error calculating correlations for feature {feature_id}: {str(e)}")
            raise

    async def calculate_ablation(
        self,
        feature_id: str
    ) -> Optional[AblationResponse]:
        """
        Calculate ablation impact.

        Measures the feature's importance by comparing model performance
        with the feature active vs. ablated (set to zero).

        Args:
            feature_id: Feature ID to analyze

        Returns:
            AblationResponse with perplexity delta and impact score, or None if feature not found

        Process:
            1. Check cache for recent result
            2. Load evaluation samples from dataset
            3. Run model inference with feature active (baseline)
            4. Run model inference with feature ablated
            5. Calculate perplexity for both runs
            6. Compute delta and normalize to impact score (0-1)
            7. Cache result for future requests
        """
        # Check cache first
        cache_entry = await self._get_cached_analysis(feature_id, AnalysisType.ABLATION)
        if cache_entry:
            logger.info(f"Ablation cache hit for feature {feature_id}")
            return AblationResponse(**cache_entry.results)

        # Load feature
        feature = await self._get_feature(feature_id)
        if not feature:
            logger.warning(f"Feature {feature_id} not found")
            return None

        training = await self._get_training(feature.training_id)
        if not training:
            logger.warning(f"Training {feature.training_id} not found")
            return None

        try:
            # TODO: Load dataset samples
            # TODO: Load SAE and base model
            # TODO: Run inference with feature active and ablated
            # TODO: Calculate perplexities
            # For now, return mock data
            logger.warning("Ablation: model inference not yet implemented, returning mock data")

            # Mock implementation
            baseline_perplexity = 15.2
            ablated_perplexity = 18.7
            perplexity_delta = ablated_perplexity - baseline_perplexity
            # Normalize to 0-1 range (assuming max delta of ~10)
            impact_score = min(1.0, perplexity_delta / 10.0)

            computed_at = datetime.now(timezone.utc)

            # Cache the result
            await self._cache_analysis(
                feature_id=feature_id,
                analysis_type=AnalysisType.ABLATION,
                results={
                    "perplexity_delta": perplexity_delta,
                    "impact_score": impact_score,
                    "baseline_perplexity": baseline_perplexity,
                    "ablated_perplexity": ablated_perplexity,
                    "computed_at": computed_at.isoformat()
                }
            )

            return AblationResponse(
                perplexity_delta=perplexity_delta,
                impact_score=impact_score,
                baseline_perplexity=baseline_perplexity,
                ablated_perplexity=ablated_perplexity,
                computed_at=computed_at
            )

        except Exception as e:
            logger.error(f"Error calculating ablation for feature {feature_id}: {str(e)}")
            raise

    # Helper methods

    async def _get_cached_analysis(
        self,
        feature_id: str,
        analysis_type: AnalysisType
    ) -> Optional[FeatureAnalysisCache]:
        """Get cached analysis if available and not expired."""
        expiry_threshold = datetime.now(timezone.utc) - timedelta(days=self.CACHE_EXPIRY_DAYS)

        stmt = select(FeatureAnalysisCache).where(
            and_(
                FeatureAnalysisCache.feature_id == feature_id,
                FeatureAnalysisCache.analysis_type == analysis_type,
                FeatureAnalysisCache.computed_at >= expiry_threshold
            )
        )

        if isinstance(self.db, AsyncSession):
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        else:
            return self.db.execute(stmt).scalar_one_or_none()

    async def _cache_analysis(
        self,
        feature_id: str,
        analysis_type: AnalysisType,
        results: Dict[str, Any]
    ) -> None:
        """Cache analysis results."""
        cache_entry = FeatureAnalysisCache(
            feature_id=feature_id,
            analysis_type=analysis_type,
            results=results,
            computed_at=datetime.now(timezone.utc)
        )

        self.db.add(cache_entry)
        if isinstance(self.db, AsyncSession):
            await self.db.commit()
        else:
            self.db.commit()

    async def _get_feature(self, feature_id: str) -> Optional[Feature]:
        """Get feature by ID."""
        stmt = select(Feature).where(Feature.id == feature_id)

        if isinstance(self.db, AsyncSession):
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        else:
            return self.db.execute(stmt).scalar_one_or_none()

    async def _get_training(self, training_id: str) -> Optional[Training]:
        """Get training by ID."""
        stmt = select(Training).where(Training.id == training_id)

        if isinstance(self.db, AsyncSession):
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        else:
            return self.db.execute(stmt).scalar_one_or_none()

    async def _get_features_for_training(self, training_id: str) -> List[Feature]:
        """Get all features for a training."""
        stmt = select(Feature).where(Feature.training_id == training_id).limit(10)

        if isinstance(self.db, AsyncSession):
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        else:
            return list(self.db.execute(stmt).scalars().all())
