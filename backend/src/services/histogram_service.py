"""
Activation Histogram Generation Service for Neuronpedia Dashboard Data.

This service computes histograms of feature activation values from stored examples.
Histograms show the distribution of activation values across all positions where
the feature fired, which helps users understand:
- How often the feature activates
- The typical activation strength
- Whether activations are concentrated or spread out

The histogram data is used in Neuronpedia feature dashboards to visualize
the activation distribution.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable

import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.feature import Feature
from ..models.feature_activation import FeatureActivation
from ..models.feature_dashboard import FeatureDashboardData

logger = logging.getLogger(__name__)


@dataclass
class HistogramData:
    """Result of histogram computation for a single feature."""
    feature_index: int
    bin_edges: List[float]  # n_bins + 1 edges
    counts: List[int]  # n_bins counts
    total_count: int  # Total number of activations (including zeros)
    nonzero_count: int  # Number of non-zero activations
    mean: float  # Mean of non-zero activations
    std: float  # Standard deviation of non-zero activations
    max_value: float  # Maximum activation value
    log_scale: bool  # Whether bins are log-scaled


class HistogramService:
    """
    Service for computing activation histograms for SAE features.

    Histograms are computed from stored feature activation examples in the database.
    """

    def __init__(self):
        """Initialize the histogram service."""
        pass

    async def compute_histogram(
        self,
        db: AsyncSession,
        feature_id: str,
        n_bins: int = 50,
        log_scale: bool = True,
    ) -> HistogramData:
        """
        Compute histogram of feature activation values from stored examples.

        Args:
            db: Database session
            feature_id: Feature ID (features.id)
            n_bins: Number of histogram bins
            log_scale: Whether to use log-scaled bins (better for sparse activations)

        Returns:
            HistogramData with bin edges, counts, and statistics
        """
        # Load feature to get neuron_index
        feature = await db.get(Feature, feature_id)
        if not feature:
            raise ValueError(f"Feature not found: {feature_id}")

        # Load all activation examples for this feature
        stmt = select(FeatureActivation).where(
            FeatureActivation.feature_id == feature_id
        )
        result = await db.execute(stmt)
        examples = result.scalars().all()

        if not examples:
            logger.warning(f"No activation examples found for feature {feature_id}")
            return HistogramData(
                feature_index=feature.neuron_index,
                bin_edges=[0.0, 1.0],
                counts=[0],
                total_count=0,
                nonzero_count=0,
                mean=0.0,
                std=0.0,
                max_value=0.0,
                log_scale=log_scale,
            )

        # Flatten all activation values from examples
        all_activations = []
        for example in examples:
            # FeatureActivation stores activations as JSONB array
            if example.activations:
                if isinstance(example.activations, list):
                    all_activations.extend(example.activations)
                else:
                    all_activations.append(example.activations)

        if not all_activations:
            return HistogramData(
                feature_index=feature.neuron_index,
                bin_edges=[0.0, 1.0],
                counts=[0],
                total_count=0,
                nonzero_count=0,
                mean=0.0,
                std=0.0,
                max_value=0.0,
                log_scale=log_scale,
            )

        activations = np.array(all_activations, dtype=np.float32)
        total_count = len(activations)

        # Filter to non-zero activations for histogram
        nonzero_acts = activations[activations > 0]
        nonzero_count = len(nonzero_acts)

        if nonzero_count == 0:
            return HistogramData(
                feature_index=feature.neuron_index,
                bin_edges=[0.0, 1.0],
                counts=[0],
                total_count=total_count,
                nonzero_count=0,
                mean=0.0,
                std=0.0,
                max_value=0.0,
                log_scale=log_scale,
            )

        # Compute statistics
        mean_val = float(np.mean(nonzero_acts))
        std_val = float(np.std(nonzero_acts))
        max_val = float(np.max(nonzero_acts))

        # Create bin edges
        if log_scale:
            # Log-spaced bins are better for sparse activations
            log_min = np.log10(max(float(nonzero_acts.min()), 1e-10))
            log_max = np.log10(max_val)
            bin_edges = np.logspace(log_min, log_max, n_bins + 1)
        else:
            # Linear bins
            bin_edges = np.linspace(0, max_val, n_bins + 1)

        # Compute histogram
        counts, _ = np.histogram(nonzero_acts, bins=bin_edges)

        return HistogramData(
            feature_index=feature.neuron_index,
            bin_edges=bin_edges.tolist(),
            counts=counts.tolist(),
            total_count=total_count,
            nonzero_count=nonzero_count,
            mean=mean_val,
            std=std_val,
            max_value=max_val,
            log_scale=log_scale,
        )

    async def compute_histograms_batch(
        self,
        db: AsyncSession,
        feature_ids: List[str],
        n_bins: int = 50,
        log_scale: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, HistogramData]:
        """
        Compute histograms for multiple features.

        Args:
            db: Database session
            feature_ids: List of feature IDs
            n_bins: Number of histogram bins
            log_scale: Whether to use log-scaled bins
            progress_callback: Optional callback(completed, total, message)

        Returns:
            Dictionary mapping feature_id to HistogramData
        """
        results = {}
        total = len(feature_ids)

        for i, feature_id in enumerate(feature_ids):
            if progress_callback:
                progress_callback(i, total, f"Computing histogram for feature {i+1}/{total}")

            try:
                histogram = await self.compute_histogram(
                    db, feature_id, n_bins, log_scale
                )
                results[feature_id] = histogram
            except Exception as e:
                logger.error(f"Error computing histogram for {feature_id}: {e}")

        if progress_callback:
            progress_callback(total, total, "Histogram computation complete")

        return results

    async def compute_histograms_for_sae(
        self,
        db: AsyncSession,
        sae_id: str,
        n_bins: int = 50,
        log_scale: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        force_recompute: bool = False,
    ) -> Dict[int, HistogramData]:
        """
        Compute histograms for all features of an SAE.

        Args:
            db: Database session
            sae_id: External SAE ID
            n_bins: Number of histogram bins
            log_scale: Whether to use log-scaled bins
            progress_callback: Optional callback(completed, total, message)
            force_recompute: If True, ignore cached results

        Returns:
            Dictionary mapping feature index to HistogramData
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
                cached = await self._get_cached_histogram(db, feature.id)
                if cached:
                    results[feature.neuron_index] = cached
                else:
                    remaining_features.append(feature)
            logger.info(f"Found {len(results)} cached histograms, {len(remaining_features)} to compute")
        else:
            remaining_features = list(features)

        # Compute remaining
        total = len(remaining_features) + len(results)
        completed = len(results)

        for feature in remaining_features:
            if progress_callback:
                progress_callback(completed, total, f"Computing histogram for feature {feature.neuron_index}")

            try:
                histogram = await self.compute_histogram(db, feature.id, n_bins, log_scale)
                results[feature.neuron_index] = histogram
                completed += 1
            except Exception as e:
                logger.error(f"Error computing histogram for {feature.id}: {e}")

        if progress_callback:
            progress_callback(total, total, "Histogram computation complete")

        return results

    async def _get_cached_histogram(
        self,
        db: AsyncSession,
        feature_id: str,
    ) -> Optional[HistogramData]:
        """Get cached histogram from database."""
        stmt = select(FeatureDashboardData).where(
            FeatureDashboardData.feature_id == feature_id
        )
        result = await db.execute(stmt)
        cached = result.scalar_one_or_none()

        if not cached or not cached.histogram_data:
            return None

        data = cached.histogram_data
        # Get feature to get neuron_index
        feature = await db.get(Feature, feature_id)
        neuron_index = feature.neuron_index if feature else 0

        return HistogramData(
            feature_index=neuron_index,
            bin_edges=data.get("bin_edges", []),
            counts=data.get("counts", []),
            total_count=data.get("total_count", 0),
            nonzero_count=data.get("nonzero_count", 0),
            mean=data.get("mean", 0.0),
            std=data.get("std", 0.0),
            max_value=data.get("max", 0.0),
            log_scale=data.get("log_scale", True),
        )

    async def save_histogram_results(
        self,
        db: AsyncSession,
        sae_id: str,
        results: Dict[int, HistogramData],
    ) -> None:
        """
        Save histogram results to the database.

        Args:
            db: Database session
            sae_id: SAE ID
            results: Dictionary mapping feature index to HistogramData
        """
        # Get features to map index to feature_id
        stmt = select(Feature).where(Feature.external_sae_id == sae_id)
        result = await db.execute(stmt)
        features = {f.neuron_index: f for f in result.scalars().all()}

        for idx, histogram in results.items():
            feature = features.get(idx)
            if not feature:
                continue

            # Check if dashboard data exists
            stmt = select(FeatureDashboardData).where(
                FeatureDashboardData.feature_id == feature.id
            )
            existing = await db.execute(stmt)
            dashboard_data = existing.scalar_one_or_none()

            histogram_json = {
                "bin_edges": histogram.bin_edges,
                "counts": histogram.counts,
                "total_count": histogram.total_count,
                "nonzero_count": histogram.nonzero_count,
                "mean": histogram.mean,
                "std": histogram.std,
                "max": histogram.max_value,
                "log_scale": histogram.log_scale,
            }

            if dashboard_data:
                dashboard_data.histogram_data = histogram_json
            else:
                dashboard_data = FeatureDashboardData(
                    feature_id=feature.id,
                    histogram_data=histogram_json,
                    computation_version="1.0",
                )
                db.add(dashboard_data)

        await db.commit()
        logger.info(f"Saved histogram results for {len(results)} features")


# Global service instance
_histogram_service: Optional[HistogramService] = None


def get_histogram_service() -> HistogramService:
    """Get the global histogram service instance."""
    global _histogram_service
    if _histogram_service is None:
        _histogram_service = HistogramService()
    return _histogram_service
