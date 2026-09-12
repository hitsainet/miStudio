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
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func
import numpy as np
from scipy.stats import pearsonr
import random

from src.core.config import settings
from src.services.feature_provenance import (
    feature_scope_clause,
    resolve_training_id,
)
from src.models.feature import Feature
from src.models.feature_activation import FeatureActivation
from src.models.feature_analysis_cache import FeatureAnalysisCache, AnalysisType
from src.models.training import Training
from src.models.checkpoint import Checkpoint
from src.models.model import Model as ModelRecord, QuantizationFormat
from src.models.external_sae import ExternalSAE
from src.schemas.feature import (
    LogitLensResponse,
    CorrelationsResponse,
    CorrelatedFeature,
    AblationResponse
)
from src.ml.sparse_autoencoder import SparseAutoencoder, create_sae
from src.ml.model_loader import load_model_from_hf
from src.ml.community_format import load_sae_auto_detect
from src.services.checkpoint_service import CheckpointService


logger = logging.getLogger(__name__)



def _validated(tensor, key: str):
    """Refuse a tensor that cannot be an unembedding.

    Suffix matching is more permissive than an exact name, so it earns a shape
    check: an unembedding is 2-D [vocab, d_model] with a vocabulary far larger
    than the hidden size. Without this, picking the wrong tensor would surface
    as an inscrutable matmul error deep in the caller instead of here, where the
    key that was chosen is still in scope.
    """
    if tensor.ndim != 2:
        raise ValueError(
            f"tensor {key!r} is {tensor.ndim}-D, not a [vocab, d_model] "
            f"unembedding matrix"
        )
    rows, cols = tensor.shape
    if rows < cols:
        raise ValueError(
            f"tensor {key!r} has shape {tuple(tensor.shape)}; an unembedding "
            f"should have vocab ({rows}) >= d_model ({cols})"
        )
    return tensor


def _find_unembedding_key(available) -> "str | None":
    """Locate the unembedding tensor across flat AND nested weight layouts.

    The exact-name list covers ordinary text models. It does NOT cover
    multimodal ones, which nest the text tower: gemma-4-12B-it stores its
    unembedding at

        model.language_model.embed_tokens.weight

    alongside model.embed_vision.* and model.embed_audio.*, so a lookup for
    "model.embed_tokens.weight" finds nothing and every logit-lens request on a
    gemma-4 SAE returned 500.

    Falls back to a SUFFIX match, which is what makes nesting irrelevant.
    Matching on the full segment "embed_tokens.weight" is deliberate: the vision
    and audio towers expose "embed_vision.embedding_projection.weight", which a
    looser "embed" search would happily return — and that tensor is not an
    unembedding at all.

    lm_head is preferred over embed_tokens because a model that has both is
    UNTIED, and its true output matrix is lm_head. When several candidates
    match, the shallowest wins: a top-level tensor is the model's own, a deeper
    one belongs to a submodule.
    """
    available = set(available)
    for key in ("lm_head.weight", "model.embed_tokens.weight"):
        if key in available:
            return key
    for suffix in ("lm_head.weight", "embed_tokens.weight"):
        matches = [k for k in available if k.endswith(suffix)]
        if matches:
            return sorted(matches, key=lambda k: (k.count("."), len(k)))[0]
    return None


def load_unembedding_matrix(
    model_dir: "Path", device: str = "cpu"
) -> "torch.Tensor":
    """Load ONLY the unembedding matrix (lm_head), not the whole model.

    Logit lens is `W_dec[feature] @ W_U`. It needs the output embedding matrix
    and nothing else — no layers, no forward pass. Instantiating the full model
    to reach one tensor cost ~17 GB for granite-4.1-8b and made the feature fail
    outright once miLLM was holding the card:

        Out of memory loading ibm-granite/granite-4.1-8b even with FP32.

    W_U here is 100,352 x 4,096 fp16 ~= 820 MB — about 20x less, and it reads
    straight out of the safetensors shard without materialising anything else.

    Falls back to the input embeddings when a model ties them
    (granite-4.1-8b sets tie_word_embeddings=true), which is the same matrix.
    """
    import json

    from safetensors import safe_open

    candidates = ["lm_head.weight", "model.embed_tokens.weight"]

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        key = _find_unembedding_key(weight_map.keys())
        if key is not None:
            with safe_open(str(model_dir / weight_map[key]), framework="pt",
                           device=device) as f:
                return _validated(f.get_tensor(key), key)
        raise ValueError(
            f"No unembedding tensor found in {index_path}; tried {candidates} "
            f"and suffix matches on lm_head.weight / embed_tokens.weight"
        )

    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device=device) as f:
            key = _find_unembedding_key(f.keys())
            if key is not None:
                return _validated(f.get_tensor(key), key)
        raise ValueError(
            f"No unembedding tensor in {single}; tried {candidates} and suffix "
            f"matches on lm_head.weight / embed_tokens.weight"
        )

    raise FileNotFoundError(f"No safetensors weights under {model_dir}")


def resolve_snapshot_dir(cache_dir: "Path", repo_id: str) -> "Optional[Path]":
    """Find the HuggingFace snapshot directory holding a model's weights."""
    hub_name = "models--" + repo_id.replace("/", "--")
    for base in (cache_dir / hub_name, cache_dir):
        snapshots = base / "snapshots"
        if snapshots.is_dir():
            for snap in sorted(snapshots.iterdir()):
                if (snap / "model.safetensors.index.json").exists() or (
                    snap / "model.safetensors"
                ).exists():
                    return snap
        if (base / "model.safetensors.index.json").exists() or (
            base / "model.safetensors"
        ).exists():
            return base
    return None


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

        try:
            # CPU, deliberately, even when a GPU is present.
            #
            # Logit lens is one vector-matrix product: [4096] @ [4096, 100352],
            # about 0.4 GFLOP. It finishes in milliseconds on CPU and gains
            # nothing from the device.
            #
            # Running it on CUDA made it fail whenever anything else held the
            # card. With miLLM serving granite in fp16 (~17.5 GB of 24 GB) every
            # request 500'd. An analysis this small must not compete with
            # serving for VRAM.
            device = "cpu"
            logger.info(f"Using device: {device} (logit lens is CPU-only by design)")

            # Two paths: training-based SAE or external SAE
            sae = None
            model_record = None
            decoder_weight = None

            # RESOLVE THE TRAINING, DON'T READ THE COLUMN (MIS-E2E-135).
            #
            # `feature.training_id` is NULL for essentially every feature here:
            # features are extracted against an entry in the SAE registry, so
            # the provenance lives one hop away on `ExternalSAE.training_id`.
            # Reading the column meant this branch never ran for a real feature.
            resolved_training_id = await resolve_training_id(self.db, feature)

            if resolved_training_id:
                # Path 1: Load from training checkpoint
                training = await self._get_training(resolved_training_id)
                if not training:
                    logger.warning(f"Training {resolved_training_id} not found")
                    return None

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

                # Initialize SAE model using factory
                hp = training.hyperparameters
                architecture_type = hp.get('architecture_type', 'standard')
                logger.info(f"Creating {architecture_type} SAE for logit lens analysis")

                sae = create_sae(
                    architecture_type=architecture_type,
                    hidden_dim=hp["hidden_dim"],
                    latent_dim=hp["latent_dim"],
                    l1_alpha=hp.get("l1_alpha", 0.001),
                    initial_threshold=hp.get("initial_threshold"),
                    bandwidth=hp.get("bandwidth"),
                    sparsity_coeff=hp.get("sparsity_coeff"),
                    normalize_decoder=hp.get("normalize_decoder"),
                    tied_weights=hp.get("tied_weights"),
                    normalize_activations=hp.get("normalize_activations"),
                )

                CheckpointService.load_checkpoint(
                    storage_path=checkpoint.storage_path,
                    model=sae,
                    device=device
                )
                sae.to(device)
                sae.eval()

                # Load model record from training
                model_stmt = select(ModelRecord).where(ModelRecord.id == training.model_id)
                if isinstance(self.db, AsyncSession):
                    model_result = await self.db.execute(model_stmt)
                    model_record = model_result.scalar_one_or_none()
                else:
                    model_record = self.db.execute(model_stmt).scalar_one_or_none()

                if not model_record:
                    raise ValueError(f"Model {training.model_id} not found")

            elif feature.external_sae_id:
                # Path 2: Load from external SAE
                logger.info(f"Loading external SAE {feature.external_sae_id} for logit lens")

                external_sae_stmt = select(ExternalSAE).where(ExternalSAE.id == feature.external_sae_id)
                if isinstance(self.db, AsyncSession):
                    external_sae_result = await self.db.execute(external_sae_stmt)
                    external_sae = external_sae_result.scalar_one_or_none()
                else:
                    external_sae = self.db.execute(external_sae_stmt).scalar_one_or_none()

                if not external_sae:
                    raise ValueError(f"External SAE {feature.external_sae_id} not found")

                if not external_sae.local_path:
                    raise ValueError(f"External SAE {feature.external_sae_id} has no local path")

                # Load SAE using auto-detect
                resolved_sae_path = settings.resolve_data_path(external_sae.local_path)
                logger.info(f"Loading external SAE from {resolved_sae_path}")

                sae_state_dict, sae_config, format_type = load_sae_auto_detect(
                    resolved_sae_path,
                    device=device
                )
                logger.info(f"Loaded external SAE in {format_type} format")

                # Get decoder weights directly from state_dict
                # Community/external SAE format uses 'decoder.weight' key
                if 'decoder.weight' in sae_state_dict:
                    decoder_weight = sae_state_dict['decoder.weight'].to(device)
                elif 'W_dec' in sae_state_dict:
                    decoder_weight = sae_state_dict['W_dec'].to(device)
                else:
                    raise ValueError(f"Could not find decoder weights in SAE state dict. Keys: {sae_state_dict.keys()}")

                logger.info(f"Decoder weight shape: {decoder_weight.shape}")

                # Load model record from external SAE
                if external_sae.model_id:
                    model_stmt = select(ModelRecord).where(ModelRecord.id == external_sae.model_id)
                    if isinstance(self.db, AsyncSession):
                        model_result = await self.db.execute(model_stmt)
                        model_record = model_result.scalar_one_or_none()
                    else:
                        model_record = self.db.execute(model_stmt).scalar_one_or_none()

                if not model_record and external_sae.model_name:
                    # Try to find model by name
                    model_stmt = select(ModelRecord).where(
                        ModelRecord.name.ilike(f"%{external_sae.model_name}%")
                    )
                    if isinstance(self.db, AsyncSession):
                        model_result = await self.db.execute(model_stmt)
                        model_record = model_result.scalar_one_or_none()
                    else:
                        model_record = self.db.execute(model_stmt).scalar_one_or_none()

                if not model_record:
                    raise ValueError(
                        f"Model not found for external SAE. "
                        f"model_id={external_sae.model_id}, model_name={external_sae.model_name}"
                    )

            else:
                logger.warning(f"Feature {feature_id} has no training_id or external_sae_id")
                return None

            logger.info(f"SAE loaded successfully")

            logger.info(f"Loading base model {model_record.repo_id}")

            # Load base model and tokenizer
            # Use local_files_only=True when model is already downloaded to avoid
            # HuggingFace API calls that require authentication for gated models
            resolved_model_path = settings.resolve_data_path(model_record.file_path) if model_record.file_path else None
            model_is_downloaded = resolved_model_path and resolved_model_path.exists()

            # Load ONLY the unembedding matrix, never the full model.
            #
            # This used to instantiate all 8B parameters to reach one tensor.
            # Once miLLM held the card with granite in fp16 (~17.5 GB of 24 GB)
            # there was no room left and EVERY logit-lens request 500'd with
            # "Out of memory loading ibm-granite/granite-4.1-8b even with FP32".
            #
            # The tokenizer is still needed (to turn ids back into strings) but
            # it is CPU-only and cheap.
            W_U_source = None
            if model_is_downloaded:
                snapshot = resolve_snapshot_dir(resolved_model_path, model_record.repo_id)
                if snapshot is not None:
                    logger.info(f"Loading unembedding matrix only from {snapshot}")
                    W_U_source = load_unembedding_matrix(snapshot, device="cpu")

            if W_U_source is None:
                # No local weights to read directly — fall back to the old path
                # so behaviour is unchanged for models that are not cached.
                logger.warning(
                    "Falling back to full model load for logit lens "
                    f"(repo_id={model_record.repo_id}); this needs the whole "
                    "model resident and will fail if the GPU is occupied"
                )
                base_model, tokenizer, model_config, metadata = load_model_from_hf(
                    repo_id=model_record.repo_id,
                    quant_format=QuantizationFormat(model_record.quantization),
                    cache_dir=resolved_model_path,
                    device_map=device,
                    local_files_only=model_is_downloaded,
                )
                base_model.eval()
                W_U_source = base_model.lm_head.weight
            else:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    str(snapshot), local_files_only=True
                )

            logger.info(f"Unembedding matrix ready: {tuple(W_U_source.shape)}")

            # Logit lens is a WEIGHT-BASED analysis, not a forward pass!
            # Formula: logits = W_dec[feature_idx] @ W_U
            # where W_dec is the SAE decoder weights and W_U is the LM head weights
            with torch.no_grad():
                # Get decoder direction for this specific feature
                # Decoder weight shape: [hidden_dim, latent_dim] or [latent_dim, hidden_dim]
                # We want the column corresponding to this feature: [hidden_dim]

                if decoder_weight is not None:
                    # External SAE: decoder_weight already loaded from state_dict
                    # Shape is typically [d_sae, d_model] for community format
                    logger.info(f"Using pre-loaded decoder_weight, shape: {decoder_weight.shape}")
                    if decoder_weight.shape[0] > decoder_weight.shape[1]:
                        # Shape [d_sae, d_model] - rows are features
                        decoder_direction = decoder_weight[feature.neuron_index, :]
                    else:
                        # Shape [d_model, d_sae] - columns are features
                        decoder_direction = decoder_weight[:, feature.neuron_index]
                elif sae is not None:
                    # Training-based SAE: extract from SAE model object
                    # Debug: Log SAE type and available attributes
                    logger.info(f"SAE type: {type(sae).__name__}")
                    logger.info(f"SAE has 'decoder' attr: {hasattr(sae, 'decoder')}")
                    if hasattr(sae, 'decoder'):
                        logger.info(f"sae.decoder type: {type(sae.decoder)}")
                        logger.info(f"sae.decoder is None: {sae.decoder is None}")
                        if sae.decoder is not None:
                            logger.info(f"sae.decoder has 'weight': {hasattr(sae.decoder, 'weight')}")
                    logger.info(f"SAE has 'decoder_weight' attr: {hasattr(sae, 'decoder_weight')}")

                    # Check for JumpReLU's decoder_weight FIRST (before compatibility wrapper)
                    if hasattr(sae, 'decoder_weight') and not isinstance(getattr(sae, 'decoder', None), torch.nn.Linear):
                        # JumpReLU SAE with decoder_weight property - shape [d_model, d_sae]
                        logger.info(f"Using JumpReLU SAE decoder_weight, shape: {sae.decoder_weight.shape}")
                        decoder_direction = sae.decoder_weight[:, feature.neuron_index]
                    elif hasattr(sae, 'decoder') and sae.decoder is not None and hasattr(sae.decoder, 'weight'):
                        # Standard SAE with nn.Linear decoder - shape [hidden_dim, latent_dim]
                        logger.info(f"Using standard SAE decoder.weight, shape: {sae.decoder.weight.shape}")
                        decoder_direction = sae.decoder.weight[:, feature.neuron_index]
                    else:
                        raise ValueError(f"Unknown SAE architecture: cannot find decoder weights")
                else:
                    raise ValueError(f"No SAE model or decoder weights available")

                # Get the unembedding matrix (LM head weights)
                # base_model.lm_head.weight shape: [vocab_size, hidden_dim]
                # We need it transposed: [hidden_dim, vocab_size]
                W_U = W_U_source.T

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

            # Clean up to free GPU memory.
            #
            # base_model only exists on the fallback path now — the fast path
            # never instantiates it — so deleting it unconditionally would
            # NameError and turn a successful computation into a 500.
            del sae
            del W_U_source
            if "base_model" in locals():
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

        Finds features with similar characteristics using a multi-factor similarity
        approach that combines:
        1. Token overlap: Features that activate on similar tokens
        2. Activation statistics: Similar mean/max activation magnitudes
        3. Activation frequency: Similar firing rates

        Note: Traditional Pearson correlation on activation vectors doesn't work well
        because each feature only stores its TOP-K activating samples, which rarely
        overlap with other features' top samples.

        Args:
            feature_id: Feature ID to analyze

        Returns:
            CorrelationsResponse with top 10 correlated features, or None if feature not found
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
            logger.info(f"Computing statistics-based correlations for feature {feature_id}")

            # Extract current feature's token set for comparison
            current_tokens = set()
            if feature.example_tokens_summary:
                if isinstance(feature.example_tokens_summary, list):
                    current_tokens = set(t.strip().lower() for t in feature.example_tokens_summary if t)
                elif isinstance(feature.example_tokens_summary, str):
                    current_tokens = set(t.strip().lower() for t in feature.example_tokens_summary.split(",") if t.strip())

            current_freq = feature.activation_frequency or 0.0
            current_mean = feature.mean_activation or 0.0
            current_max = feature.max_activation or 0.0

            logger.info(f"Current feature: {len(current_tokens)} tokens, freq={current_freq:.3f}, mean={current_mean:.3f}, max={current_max:.3f}")

            # Load other features from the same training.
            # DETERMINISTIC sample (R1 #12): the old `ORDER BY func.random()`
            # gave a different subset on every call — non-reproducible "related
            # features", frozen arbitrarily by the 7-day cache. Order by a
            # stable key so the same query always samples the same features;
            # `sampled` is disclosed in the response.
            sample_size = 2000

            # SCOPE THE PEERS TO THE SAME DICTIONARY.
            #
            # This used to be `Feature.training_id == feature.training_id`.
            # Features are extracted against an SAE in the registry, so
            # `training_id` is NULL for every feature in this product — and
            # `col == None` compiles to `IS NULL`, which matches EVERY feature
            # of EVERY SAE. "Correlated features" were being drawn from other
            # models entirely, then frozen for 7 days by the cache.
            #
            # `source_id` is the Feature model's own answer to "which
            # dictionary is this from": `external_sae_id or training_id`.
            # Same rule, one implementation (`feature_scope_clause`), so the
            # two consumers cannot drift apart.
            if feature.external_sae_id or feature.training_id:
                scope = feature_scope_clause(feature)
            else:
                # Neither set: the feature has no dictionary to compare within.
                # Say so rather than silently comparing against the whole table.
                logger.warning(
                    f"Feature {feature_id} has neither external_sae_id nor "
                    "training_id; cannot scope correlations"
                )
                scope = None

            if scope is None:
                other_features = []
            else:
                features_stmt = select(Feature).where(
                    and_(scope, Feature.id != feature_id)
                ).order_by(Feature.id).limit(sample_size)

                if isinstance(self.db, AsyncSession):
                    result = await self.db.execute(features_stmt)
                    other_features = list(result.scalars().all())
                else:
                    other_features = list(self.db.execute(features_stmt).scalars().all())

            logger.info(f"Comparing with {len(other_features)} other features")

            # Calculate similarity scores for each feature
            similarities = []
            for other in other_features:
                # Extract other feature's tokens
                other_tokens = set()
                if other.example_tokens_summary:
                    if isinstance(other.example_tokens_summary, list):
                        other_tokens = set(t.strip().lower() for t in other.example_tokens_summary if t)
                    elif isinstance(other.example_tokens_summary, str):
                        other_tokens = set(t.strip().lower() for t in other.example_tokens_summary.split(",") if t.strip())

                other_freq = other.activation_frequency or 0.0
                other_mean = other.mean_activation or 0.0
                other_max = other.max_activation or 0.0

                # 1. Token overlap similarity (Jaccard index) - weight: 0.5
                token_similarity = 0.0
                if current_tokens and other_tokens:
                    intersection = len(current_tokens & other_tokens)
                    union = len(current_tokens | other_tokens)
                    if union > 0:
                        token_similarity = intersection / union

                # 2. Activation frequency similarity - weight: 0.2
                freq_similarity = 0.0
                if current_freq > 0 or other_freq > 0:
                    max_freq = max(current_freq, other_freq)
                    if max_freq > 0:
                        freq_similarity = 1.0 - abs(current_freq - other_freq) / max_freq

                # 3. Mean activation similarity - weight: 0.15
                mean_similarity = 0.0
                if current_mean > 0 or other_mean > 0:
                    max_mean = max(current_mean, other_mean)
                    if max_mean > 0:
                        mean_similarity = 1.0 - abs(current_mean - other_mean) / max_mean

                # 4. Max activation similarity - weight: 0.15
                max_similarity = 0.0
                if current_max > 0 or other_max > 0:
                    max_max = max(current_max, other_max)
                    if max_max > 0:
                        max_similarity = 1.0 - abs(current_max - other_max) / max_max

                # Combined weighted similarity score
                # Token overlap is most important for semantic similarity
                combined_similarity = (
                    token_similarity * 0.50 +
                    freq_similarity * 0.20 +
                    mean_similarity * 0.15 +
                    max_similarity * 0.15
                )

                # Only include if similarity is meaningful (>= 0.3)
                # and there's at least some token overlap or statistical similarity
                if combined_similarity >= 0.3 and (token_similarity > 0 or freq_similarity > 0.5):
                    similarities.append({
                        "feature_id": other.id,
                        "feature_name": other.name or f"Feature {other.neuron_index}",
                        # Carried through so the UI can name this feature the
                        # way the rest of the product does, instead of showing
                        # an internal id nobody recognises.
                        "neuron_index": other.neuron_index,
                        "correlation": float(combined_similarity),
                        "_token_sim": token_similarity,
                        "_freq_sim": freq_similarity,
                    })

            # Sort by similarity score, take top 10
            similarities.sort(key=lambda x: x["correlation"], reverse=True)
            top_correlations = similarities[:10]

            # Clean up internal fields before returning
            for corr in top_correlations:
                corr.pop("_token_sim", None)
                corr.pop("_freq_sim", None)

            logger.info(f"Found {len(top_correlations)} similar features for feature {feature_id}")

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
                computed_at=computed_at,
                sampled=len(other_features) >= sample_size,
                sample_size=sample_size,
            )

        except Exception as e:
            logger.error(f"Error calculating correlations for feature {feature_id}: {str(e)}", exc_info=True)
            raise

    async def calculate_ablation(
        self,
        feature_id: str
    ) -> Optional[AblationResponse]:
        """
        Estimate ablation impact from activation STATISTICS (017 remediation).

        IMPORTANT: this is a STATISTICAL ESTIMATE, not a causal measurement —
        it runs NO model inference. It scores importance from the feature's
        activation frequency, magnitude, and consistency, and the response
        carries `method="statistical_estimate"` to say so. The perplexity
        numbers are a heuristic projection, not measured. Real causal ablation
        (suppress the feature, run the model, measure the downstream effect vs
        a null) is the circuit-validation tier — Feature 017, evidence rung 2.
        Never present this estimate as causal evidence.

        Args:
            feature_id: Feature ID to analyze

        Returns:
            AblationResponse (with method="statistical_estimate"), or None if
            the feature is not found.

        Process (statistical, no inference):
            1. Check cache
            2. Load stored activations for the feature
            3. Score from frequency / magnitude / consistency
            4. Project a heuristic perplexity delta (NOT measured)
            5. Cache
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

        # NO training lookup here, deliberately.
        #
        # This estimate is computed entirely from `FeatureActivation` rows and
        # `feature.activation_frequency` — the training is never read. The
        # lookup that used to sit here was a dead precondition, and it failed
        # for every feature in this product: features are extracted against an
        # SAE in the registry, so `Feature.training_id` is NULL and the
        # provenance lives one hop away on `ExternalSAE.training_id`. The
        # endpoint then rendered that `None` as "Feature <id> not found",
        # blaming the feature it had just successfully loaded.
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
                    "computed_at": computed_at.isoformat(),
                    "method": "statistical_estimate",
                }
            )

            return AblationResponse(
                perplexity_delta=perplexity_delta,
                impact_score=impact_score,
                baseline_perplexity=baseline_perplexity,
                ablated_perplexity=ablated_perplexity,
                computed_at=computed_at,
                method="statistical_estimate",
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
        """Cache analysis results, replacing any existing row for this pair.

        UPSERT, not INSERT. `(feature_id, analysis_type)` is UNIQUE, and
        `_get_cached_analysis` filters on `computed_at >= expiry_threshold` —
        so once a row passes CACHE_EXPIRY_DAYS the read stops seeing it while
        the row is still there, and nothing prunes it. A plain INSERT then
        raises a unique violation on every subsequent request, PERMANENTLY:
        logit lens, correlations and ablation all 500 for that feature from
        the expiry onward, and recomputing is exactly what triggers it.

        Two concurrent first-requests for the same feature hit the same wall.

        `ON CONFLICT ... DO UPDATE` also gives the refresh the semantics the
        expiry was always meant to have: a stale entry is replaced by the
        recomputed one.
        """
        now = datetime.now(timezone.utc)
        values = {
            "feature_id": feature_id,
            "analysis_type": analysis_type,
            "result": results,  # Column is named 'result' not 'results'
            "computed_at": now,
            "expires_at": now + timedelta(days=self.CACHE_EXPIRY_DAYS),
        }
        stmt = pg_insert(FeatureAnalysisCache).values(**values).on_conflict_do_update(
            constraint="uq_feature_analysis_cache_feature_type",
            set_={
                "result": values["result"],
                "computed_at": values["computed_at"],
                "expires_at": values["expires_at"],
            },
        )

        if isinstance(self.db, AsyncSession):
            await self.db.execute(stmt)
            await self.db.commit()
        else:
            self.db.execute(stmt)
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
