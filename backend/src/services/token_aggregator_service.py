"""
Token Aggregation Service for Neuronpedia Dashboard Data.

This service aggregates tokens across all activation examples for each feature
to find the tokens that consistently cause high activations. Unlike logit lens
(which shows what a feature *promotes*), token aggregation shows what tokens
*activate* the feature across real data.

The aggregation ranks tokens by:
- Total activation: Sum of all activations for that token
- Count: Number of times the token activated the feature
- Mean activation: Average activation strength
- Max activation: Peak activation for the token

This data helps users understand what patterns in real text activate the feature.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.feature import Feature
from ..models.feature_activation import FeatureActivation
from ..models.feature_dashboard import FeatureDashboardData

logger = logging.getLogger(__name__)


@dataclass
class TokenAggregation:
    """Aggregated statistics for a single token across all examples."""
    token: str
    token_id: Optional[int]  # May not always be available
    total_activation: float  # Sum of all activations for this token
    count: int  # Number of times this token activated
    mean_activation: float  # Average activation
    max_activation: float  # Peak activation


@dataclass
class TokenAggregationResult:
    """Result of token aggregation for a single feature."""
    feature_index: int
    top_tokens: List[TokenAggregation]
    total_examples: int  # Number of examples processed


class TokenAggregatorService:
    """
    Service for aggregating top-activating tokens for SAE features.

    Analyzes stored activation examples to find which tokens consistently
    cause high activations for each feature.
    """

    def __init__(self):
        """Initialize the token aggregator service."""
        pass

    async def aggregate_top_tokens(
        self,
        db: AsyncSession,
        feature_id: str,
        k: int = 50,
        min_count: int = 2,
    ) -> TokenAggregationResult:
        """
        Aggregate tokens across all activation examples to find top-activating tokens.

        Args:
            db: Database session
            feature_id: Feature ID (features.id)
            k: Number of top tokens to return
            min_count: Minimum number of occurrences for a token to be included

        Returns:
            TokenAggregationResult with top tokens and statistics
        """
        # Load feature to get neuron_index
        feature = await db.get(Feature, feature_id)
        if not feature:
            raise ValueError(f"Feature not found: {feature_id}")

        # Load all activation examples
        stmt = select(FeatureActivation).where(
            FeatureActivation.feature_id == feature_id
        )
        result = await db.execute(stmt)
        examples = result.scalars().all()

        if not examples:
            logger.warning(f"No activation examples found for feature {feature_id}")
            return TokenAggregationResult(
                feature_index=feature.neuron_index,
                top_tokens=[],
                total_examples=0,
            )

        # Aggregate statistics for each token
        token_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_activation': 0.0,
            'count': 0,
            'max_activation': 0.0,
            'token_id': None,
        })

        total_examples = len(examples)

        for example in examples:
            # FeatureActivation has tokens and activations arrays
            tokens = example.tokens or []
            activations = example.activations or []

            # Handle case where tokens/activations might be stored differently
            if isinstance(tokens, str):
                # Single token case
                tokens = [tokens]
            if not isinstance(activations, list):
                activations = [activations]

            # Process each token-activation pair
            for i, (token, activation) in enumerate(zip(tokens, activations)):
                if activation > 0:  # Only consider non-zero activations
                    stats = token_stats[token]
                    stats['total_activation'] += activation
                    stats['count'] += 1
                    stats['max_activation'] = max(stats['max_activation'], activation)

        # Filter by minimum count and sort by total activation
        filtered_tokens = [
            (token, stats) for token, stats in token_stats.items()
            if stats['count'] >= min_count
        ]

        sorted_tokens = sorted(
            filtered_tokens,
            key=lambda x: x[1]['total_activation'],
            reverse=True
        )[:k]

        # Convert to TokenAggregation objects
        top_tokens = []
        for token, stats in sorted_tokens:
            count = stats['count']
            total_act = stats['total_activation']
            mean_act = total_act / count if count > 0 else 0.0

            top_tokens.append(TokenAggregation(
                token=token,
                token_id=stats.get('token_id'),
                total_activation=total_act,
                count=count,
                mean_activation=mean_act,
                max_activation=stats['max_activation'],
            ))

        return TokenAggregationResult(
            feature_index=feature.neuron_index,
            top_tokens=top_tokens,
            total_examples=total_examples,
        )

    async def aggregate_tokens_batch(
        self,
        db: AsyncSession,
        feature_ids: List[str],
        k: int = 50,
        min_count: int = 2,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, TokenAggregationResult]:
        """
        Aggregate tokens for multiple features.

        Args:
            db: Database session
            feature_ids: List of feature IDs
            k: Number of top tokens per feature
            min_count: Minimum occurrences for a token
            progress_callback: Optional callback(completed, total, message)

        Returns:
            Dictionary mapping feature_id to TokenAggregationResult
        """
        results = {}
        total = len(feature_ids)

        for i, feature_id in enumerate(feature_ids):
            if progress_callback:
                progress_callback(i, total, f"Aggregating tokens for feature {i+1}/{total}")

            try:
                result = await self.aggregate_top_tokens(db, feature_id, k, min_count)
                results[feature_id] = result
            except Exception as e:
                logger.error(f"Error aggregating tokens for {feature_id}: {e}")

        if progress_callback:
            progress_callback(total, total, "Token aggregation complete")

        return results

    async def aggregate_tokens_for_sae(
        self,
        db: AsyncSession,
        sae_id: str,
        k: int = 50,
        min_count: int = 2,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        force_recompute: bool = False,
    ) -> Dict[int, TokenAggregationResult]:
        """
        Aggregate top tokens for all features of an SAE.

        Args:
            db: Database session
            sae_id: External SAE ID
            k: Number of top tokens per feature
            min_count: Minimum occurrences for a token
            progress_callback: Optional callback(completed, total, message)
            force_recompute: If True, ignore cached results

        Returns:
            Dictionary mapping feature index to TokenAggregationResult
        """
        # Get all features for this SAE
        stmt = select(Feature).where(Feature.external_sae_id == sae_id)
        result = await db.execute(stmt)
        features = result.scalars().all()

        if not features:
            logger.warning(f"No features found for SAE {sae_id}")
            return {}

        # Check cache
        results = {}
        remaining_features = []

        if not force_recompute:
            for feature in features:
                cached = await self._get_cached_token_aggregation(db, feature.id)
                if cached:
                    results[feature.neuron_index] = cached
                else:
                    remaining_features.append(feature)
            logger.info(f"Found {len(results)} cached aggregations, {len(remaining_features)} to compute")
        else:
            remaining_features = list(features)

        # Compute remaining
        total = len(remaining_features) + len(results)
        completed = len(results)

        for feature in remaining_features:
            if progress_callback:
                progress_callback(completed, total, f"Aggregating tokens for feature {feature.neuron_index}")

            try:
                result = await self.aggregate_top_tokens(db, feature.id, k, min_count)
                results[feature.neuron_index] = result
                completed += 1
            except Exception as e:
                logger.error(f"Error aggregating tokens for {feature.id}: {e}")

        if progress_callback:
            progress_callback(total, total, "Token aggregation complete")

        return results

    async def _get_cached_token_aggregation(
        self,
        db: AsyncSession,
        feature_id: str,
    ) -> Optional[TokenAggregationResult]:
        """Get cached token aggregation from database."""
        stmt = select(FeatureDashboardData).where(
            FeatureDashboardData.feature_id == feature_id
        )
        result = await db.execute(stmt)
        cached = result.scalar_one_or_none()

        if not cached or not cached.top_tokens:
            return None

        # Get feature to get neuron_index
        feature = await db.get(Feature, feature_id)
        neuron_index = feature.neuron_index if feature else 0

        # Convert cached data to TokenAggregation objects
        top_tokens = []
        for item in cached.top_tokens:
            top_tokens.append(TokenAggregation(
                token=item.get("token", ""),
                token_id=item.get("token_id"),
                total_activation=item.get("total_activation", 0.0),
                count=item.get("count", 0),
                mean_activation=item.get("mean_activation", 0.0),
                max_activation=item.get("max_activation", 0.0),
            ))

        return TokenAggregationResult(
            feature_index=neuron_index,
            top_tokens=top_tokens,
            total_examples=0,  # Not stored in cache
        )

    async def save_token_aggregation_results(
        self,
        db: AsyncSession,
        sae_id: str,
        results: Dict[int, TokenAggregationResult],
    ) -> None:
        """
        Save token aggregation results to the database.

        Args:
            db: Database session
            sae_id: SAE ID
            results: Dictionary mapping feature index to TokenAggregationResult
        """
        # Get features to map index to feature_id
        stmt = select(Feature).where(Feature.external_sae_id == sae_id)
        result = await db.execute(stmt)
        features = {f.neuron_index: f for f in result.scalars().all()}

        for idx, aggregation in results.items():
            feature = features.get(idx)
            if not feature:
                continue

            # Check if dashboard data exists
            stmt = select(FeatureDashboardData).where(
                FeatureDashboardData.feature_id == feature.id
            )
            existing = await db.execute(stmt)
            dashboard_data = existing.scalar_one_or_none()

            # Convert to JSON-serializable format
            top_tokens_json = [
                {
                    "token": ta.token,
                    "token_id": ta.token_id,
                    "total_activation": ta.total_activation,
                    "count": ta.count,
                    "mean_activation": ta.mean_activation,
                    "max_activation": ta.max_activation,
                }
                for ta in aggregation.top_tokens
            ]

            if dashboard_data:
                dashboard_data.top_tokens = top_tokens_json
            else:
                dashboard_data = FeatureDashboardData(
                    feature_id=feature.id,
                    top_tokens=top_tokens_json,
                    computation_version="1.0",
                )
                db.add(dashboard_data)

        await db.commit()
        logger.info(f"Saved token aggregation results for {len(results)} features")


# Global service instance
_token_aggregator_service: Optional[TokenAggregatorService] = None


def get_token_aggregator_service() -> TokenAggregatorService:
    """Get the global token aggregator service instance."""
    global _token_aggregator_service
    if _token_aggregator_service is None:
        _token_aggregator_service = TokenAggregatorService()
    return _token_aggregator_service
