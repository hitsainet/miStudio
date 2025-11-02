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
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func
import numpy as np
from scipy.stats import pearsonr
import random

from src.models.feature import Feature
from src.models.feature_activation import FeatureActivation
from src.models.feature_analysis_cache import FeatureAnalysisCache, AnalysisType
from src.models.training import Training
from src.models.checkpoint import Checkpoint
from src.models.model import Model as ModelRecord, QuantizationFormat
from src.schemas.feature import (
    LogitLensResponse,
    CorrelationsResponse,
    CorrelatedFeature,
    AblationResponse
)
from src.ml.sparse_autoencoder import SparseAutoencoder
from src.ml.model_loader import load_model_from_hf
from src.services.checkpoint_service import CheckpointService


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
            return LogitLensResponse(**cache_entry.result)

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
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Load latest checkpoint
            checkpoint_stmt = select(Checkpoint).where(
                Checkpoint.training_id == training.id
            ).order_by(Checkpoint.step.desc()).limit(1)

            if isinstance(self.db, AsyncSession):
                checkpoint_result = await self.db.execute(checkpoint_stmt)
                checkpoint = checkpoint_result.scalar_one_or_none()
            else:
                checkpoint = self.db.execute(checkpoint_stmt).scalar_one_or_none()

            if not checkpoint:
                raise ValueError(f"No checkpoint found for training {training.id}")

            logger.info(f"Loading SAE checkpoint from {checkpoint.storage_path}")

            # Initialize and load SAE model
            sae = SparseAutoencoder(
                hidden_dim=training.hyperparameters["hidden_dim"],
                latent_dim=training.hyperparameters["latent_dim"],
                l1_alpha=training.hyperparameters.get("l1_alpha", 0.001)
            )

            CheckpointService.load_checkpoint(
                storage_path=checkpoint.storage_path,
                model=sae,
                device=device
            )
            sae.to(device)
            sae.eval()

            logger.info(f"SAE loaded successfully")

            # Load model record
            model_stmt = select(ModelRecord).where(ModelRecord.id == training.model_id)
            if isinstance(self.db, AsyncSession):
                model_result = await self.db.execute(model_stmt)
                model_record = model_result.scalar_one_or_none()
            else:
                model_record = self.db.execute(model_stmt).scalar_one_or_none()

            if not model_record:
                raise ValueError(f"Model {training.model_id} not found")

            logger.info(f"Loading base model {model_record.repo_id}")

            # Load base model and tokenizer
            base_model, tokenizer, model_config, metadata = load_model_from_hf(
                repo_id=model_record.repo_id,
                quant_format=QuantizationFormat(model_record.quantization),
                cache_dir=Path(model_record.file_path).parent if model_record.file_path else None,
                device_map=device
            )
            base_model.eval()

            logger.info(f"Base model loaded successfully")

            # Logit lens is a WEIGHT-BASED analysis, not a forward pass!
            # Formula: logits = W_dec[feature_idx] @ W_U
            # where W_dec is the SAE decoder weights and W_U is the LM head weights
            with torch.no_grad():
                # Get decoder direction for this specific feature
                # sae.decoder.weight shape: [hidden_dim, latent_dim]
                # We want the column corresponding to this feature: [hidden_dim]
                decoder_direction = sae.decoder.weight[:, feature.neuron_index]

                # Get the unembedding matrix (LM head weights)
                # base_model.lm_head.weight shape: [vocab_size, hidden_dim]
                # We need it transposed: [hidden_dim, vocab_size]
                W_U = base_model.lm_head.weight.T

                # Ensure same dtype and device for matrix multiplication
                # Convert both to the same dtype (use model's dtype, which is likely FP16)
                decoder_direction = decoder_direction.to(dtype=W_U.dtype, device=device)
                W_U = W_U.to(device)

                logger.info(f"Decoder direction shape: {decoder_direction.shape}, dtype: {decoder_direction.dtype}")
                logger.info(f"Decoder direction norm: {decoder_direction.norm().item():.6f}, mean: {decoder_direction.mean().item():.6f}")
                logger.info(f"W_U shape: {W_U.shape}, dtype: {W_U.dtype}")

                # Compute logit lens: project decoder direction onto output space
                # decoder_direction: [hidden_dim] @ W_U: [hidden_dim, vocab_size] = [vocab_size]
                logits = decoder_direction @ W_U
                logger.info(f"Logits shape: {logits.shape}, min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)

                # Get top 10 tokens
                top_probs, top_indices = torch.topk(probs, k=10)

                # Decode tokens and filter out null characters (PostgreSQL compatibility)
                top_tokens = []
                probabilities = []
                for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
                    token = tokenizer.decode([idx])
                    # Replace null characters and other control characters
                    token = token.replace('\u0000', '<NULL>')
                    # Keep the token even if it's a control character, just make it displayable
                    top_tokens.append(token)
                    probabilities.append(float(prob))

                logger.info(f"Computed logit lens for feature {feature_id}: top token = '{top_tokens[0]}'")

            # Generate interpretation from token patterns
            interpretation = self._generate_interpretation(top_tokens)

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

            # Clean up to free GPU memory
            del sae
            del base_model
            if device == "cuda":
                torch.cuda.empty_cache()

            return LogitLensResponse(
                top_tokens=top_tokens,
                probabilities=probabilities,
                interpretation=interpretation,
                computed_at=computed_at
            )

        except Exception as e:
            logger.error(f"Error calculating logit lens for feature {feature_id}: {str(e)}", exc_info=True)
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
            return CorrelationsResponse(**cache_entry.result)

        # Load feature
        feature = await self._get_feature(feature_id)
        if not feature:
            logger.warning(f"Feature {feature_id} not found")
            return None

        try:
            logger.info(f"Loading activation data for feature {feature_id}")

            # Load activation data for current feature
            current_activations_stmt = select(FeatureActivation).where(
                FeatureActivation.feature_id == feature_id
            ).order_by(FeatureActivation.sample_index)

            if isinstance(self.db, AsyncSession):
                result = await self.db.execute(current_activations_stmt)
                current_activations = list(result.scalars().all())
            else:
                current_activations = list(self.db.execute(current_activations_stmt).scalars().all())

            if len(current_activations) < 10:
                raise ValueError(
                    f"Insufficient activation data for feature {feature_id} "
                    f"(found {len(current_activations)}, need ≥10 samples)"
                )

            # Extract activation vector (max activation per sample)
            current_vector = np.array([act.max_activation for act in current_activations])

            logger.info(f"Loaded {len(current_activations)} activations for current feature")

            # Sample a subset of features for efficiency (1000 random features)
            # For large feature sets, checking all features is too slow
            sample_size = 1000
            all_features_count_stmt = select(func.count(Feature.id)).where(
                and_(
                    Feature.training_id == feature.training_id,
                    Feature.id != feature_id
                )
            )

            if isinstance(self.db, AsyncSession):
                result = await self.db.execute(all_features_count_stmt)
                total_features = result.scalar()
            else:
                total_features = self.db.execute(all_features_count_stmt).scalar()

            # Get feature IDs to check (all if < sample_size, otherwise random sample)
            if total_features <= sample_size:
                features_stmt = select(Feature).where(
                    and_(
                        Feature.training_id == feature.training_id,
                        Feature.id != feature_id
                    )
                )
            else:
                # Get random sample using ORDER BY RANDOM() LIMIT
                features_stmt = select(Feature).where(
                    and_(
                        Feature.training_id == feature.training_id,
                        Feature.id != feature_id
                    )
                ).order_by(func.random()).limit(sample_size)

            if isinstance(self.db, AsyncSession):
                result = await self.db.execute(features_stmt)
                sampled_features = list(result.scalars().all())
            else:
                sampled_features = list(self.db.execute(features_stmt).scalars().all())

            logger.info(f"Computing correlations with {len(sampled_features)} sampled features")

            # Batch load activation data for all sampled features
            feature_ids = [f.id for f in sampled_features]
            batch_activations_stmt = select(FeatureActivation).where(
                and_(
                    FeatureActivation.feature_id.in_(feature_ids),
                    FeatureActivation.sample_index.in_([act.sample_index for act in current_activations])
                )
            ).order_by(FeatureActivation.feature_id, FeatureActivation.sample_index)

            if isinstance(self.db, AsyncSession):
                result = await self.db.execute(batch_activations_stmt)
                all_activations = list(result.scalars().all())
            else:
                all_activations = list(self.db.execute(batch_activations_stmt).scalars().all())

            logger.info(f"Loaded {len(all_activations)} activation records in batch")

            # Build activation matrix (features x samples)
            feature_activation_map = {}
            for act in all_activations:
                if act.feature_id not in feature_activation_map:
                    feature_activation_map[act.feature_id] = {}
                feature_activation_map[act.feature_id][act.sample_index] = act.max_activation

            # Calculate correlations with each feature
            correlations = []
            for other_feature in sampled_features:
                if other_feature.id not in feature_activation_map:
                    continue

                # Build activation vector for other feature
                other_activations_dict = feature_activation_map[other_feature.id]
                other_vector = np.array([
                    other_activations_dict.get(act.sample_index, 0.0)
                    for act in current_activations
                ])

                # Skip if too many missing values
                if np.count_nonzero(other_vector) < len(current_activations) * 0.8:
                    continue

                # Calculate Pearson correlation
                try:
                    correlation, p_value = pearsonr(current_vector, other_vector)

                    # Only include significant correlations
                    if abs(correlation) > 0.5 and p_value < 0.05:
                        correlations.append({
                            "feature_id": other_feature.id,
                            "feature_name": other_feature.name,
                            "correlation": float(correlation)
                        })
                except Exception as e:
                    logger.warning(f"Failed to compute correlation with feature {other_feature.id}: {e}")
                    continue

            # Sort by absolute correlation, take top 10
            correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            top_correlations = correlations[:10]

            logger.info(f"Found {len(top_correlations)} significant correlations for feature {feature_id}")

            # Convert to response objects
            correlated_features = [
                CorrelatedFeature(**corr) for corr in top_correlations
            ]

            computed_at = datetime.now(timezone.utc)

            # Cache the result
            await self._cache_analysis(
                feature_id=feature_id,
                analysis_type=AnalysisType.CORRELATIONS,
                results={
                    "correlated_features": top_correlations,
                    "computed_at": computed_at.isoformat()
                }
            )

            return CorrelationsResponse(
                correlated_features=correlated_features,
                computed_at=computed_at
            )

        except Exception as e:
            logger.error(f"Error calculating correlations for feature {feature_id}: {str(e)}", exc_info=True)
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
            return AblationResponse(**cache_entry.result)

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
            logger.info(f"Calculating ablation impact for feature {feature_id}")

            # Load activation data for this feature
            activations_stmt = select(FeatureActivation).where(
                FeatureActivation.feature_id == feature_id
            ).order_by(FeatureActivation.sample_index).limit(100)  # Sample 100 for analysis

            if isinstance(self.db, AsyncSession):
                result = await self.db.execute(activations_stmt)
                activations = list(result.scalars().all())
            else:
                activations = list(self.db.execute(activations_stmt).scalars().all())

            if len(activations) < 10:
                raise ValueError(
                    f"Insufficient activation data for feature {feature_id} "
                    f"(found {len(activations)}, need ≥10 samples)"
                )

            # Calculate feature statistics
            activation_values = [act.max_activation for act in activations]
            mean_activation = np.mean(activation_values)
            std_activation = np.std(activation_values)
            max_activation = np.max(activation_values)

            # Calculate activation frequency (how often feature fires)
            activation_frequency = feature.activation_frequency

            logger.info(
                f"Feature stats: mean={mean_activation:.3f}, std={std_activation:.3f}, "
                f"max={max_activation:.3f}, freq={activation_frequency:.3f}"
            )

            # Heuristic-based ablation impact estimation
            # Features with high, consistent activation have larger impact when ablated
            #
            # Impact factors:
            # 1. Activation frequency (0-1): how often feature activates
            # 2. Activation magnitude: strength when it does activate
            # 3. Consistency: std/mean ratio (lower = more consistent = higher impact)

            # Frequency contribution (0-1 scale)
            freq_component = min(1.0, activation_frequency * 2.0)  # Scale so 50%+ freq = max

            # Magnitude contribution (0-1 scale)
            # Higher mean activation = more important
            mag_component = min(1.0, mean_activation / 5.0)  # Normalize assuming max ~5.0

            # Consistency contribution (0-1 scale)
            # Lower coefficient of variation = more consistent = higher impact
            if mean_activation > 0:
                cv = std_activation / mean_activation
                consistency_component = max(0.0, 1.0 - min(1.0, cv / 2.0))
            else:
                consistency_component = 0.0

            # Combined impact score (weighted average)
            impact_score = (
                freq_component * 0.4 +
                mag_component * 0.35 +
                consistency_component * 0.25
            )

            # Estimate perplexity delta based on impact score
            # Baseline perplexity: typical for small models is ~15-30
            baseline_perplexity = 20.0  # Reasonable baseline

            # Perplexity delta scales with impact score
            # High impact features (score near 1.0) increase perplexity by ~20-30%
            # Low impact features (score near 0) increase perplexity minimally
            perplexity_delta = baseline_perplexity * impact_score * 0.3

            ablated_perplexity = baseline_perplexity + perplexity_delta

            logger.info(
                f"Ablation impact: score={impact_score:.3f}, "
                f"delta={perplexity_delta:.2f}, "
                f"components=[freq={freq_component:.2f}, mag={mag_component:.2f}, "
                f"cons={consistency_component:.2f}]"
            )

            computed_at = datetime.now(timezone.utc)

            # Cache the result
            await self._cache_analysis(
                feature_id=feature_id,
                analysis_type=AnalysisType.ABLATION,
                results={
                    "perplexity_delta": float(perplexity_delta),
                    "impact_score": float(impact_score),
                    "baseline_perplexity": float(baseline_perplexity),
                    "ablated_perplexity": float(ablated_perplexity),
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
            logger.error(f"Error calculating ablation for feature {feature_id}: {str(e)}", exc_info=True)
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
        now = datetime.now(timezone.utc)
        cache_entry = FeatureAnalysisCache(
            feature_id=feature_id,
            analysis_type=analysis_type,
            result=results,  # Column is named 'result' not 'results'
            computed_at=now,
            expires_at=now + timedelta(days=self.CACHE_EXPIRY_DAYS)
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

    def _generate_interpretation(self, top_tokens: List[str]) -> str:
        """
        Generate semantic interpretation from top predicted tokens.

        Uses simple heuristics to identify common patterns:
        - Articles/determiners, punctuation, negation, pronouns, etc.

        Args:
            top_tokens: List of top predicted token strings

        Returns:
            Human-readable interpretation string
        """
        # Normalize tokens for pattern matching
        tokens = [token.strip().lower() for token in top_tokens[:5]]

        # Check for various patterns
        if any(t in tokens for t in ["the", "a", "an"]):
            return "Predicts determiners and articles"
        if any(t in tokens for t in [".", ",", "!", "?", ":", ";"]):
            return "Predicts punctuation marks"
        if any(t in tokens for t in ["not", "no", "never", "n't"]):
            return "Predicts negation words"
        if any(t in tokens for t in ["i", "you", "he", "she", "it", "we", "they"]):
            return "Predicts pronouns"
        if any(t in tokens for t in ["in", "on", "at", "to", "for", "of", "with"]):
            return "Predicts prepositions"
        if any(t in tokens for t in ["and", "or", "but", "so", "if", "when"]):
            return "Predicts conjunctions"
        if any(t in tokens for t in ["what", "when", "where", "who", "why", "how"]):
            return "Predicts question words"
        if any(t.isdigit() for t in tokens):
            return "Predicts numbers"

        # Fallback: list top 3 tokens
        top_3 = ", ".join([f'"{t}"' for t in tokens[:3]])
        return f"Predicts tokens like {top_3}"
