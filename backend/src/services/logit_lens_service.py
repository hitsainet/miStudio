"""
Logit Lens Computation Service for Neuronpedia Dashboard Data.

This service computes the "logit lens" for SAE features - the top tokens that
each feature promotes or suppresses in the model's output distribution.

Algorithm:
1. Load SAE decoder vectors (W_dec) - the directions each feature represents
2. Load model's unembedding matrix (W_U) - maps hidden states to logits
3. Compute: feature_logits = W_dec @ W_U.T
4. For each feature, find top-k positive and negative tokens

This data is essential for Neuronpedia feature dashboards, showing which tokens
a feature "represents" in the output vocabulary.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from safetensors.torch import load_file

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..core.config import settings
from ..ml.community_format import load_sae_auto_detect, CommunityStandardConfig
from ..models.external_sae import ExternalSAE, SAEStatus
from ..models.feature import Feature
from ..models.feature_dashboard import FeatureDashboardData
from ..models.model import Model

logger = logging.getLogger(__name__)


@dataclass
class LogitLensResult:
    """Result of logit lens computation for a single feature."""
    feature_index: int
    top_positive: List[Dict[str, Any]]  # [{"token": str, "token_id": int, "logit": float}]
    top_negative: List[Dict[str, Any]]  # [{"token": str, "token_id": int, "logit": float}]


class LogitLensService:
    """
    Service for computing logit lens data for SAE features.

    The logit lens shows which tokens each feature promotes (positive logit)
    or suppresses (negative logit) in the model's output distribution.
    """

    # Default batch size for processing features
    DEFAULT_BATCH_SIZE = 512

    # Minimum batch size when OOM occurs
    MIN_BATCH_SIZE = 64

    def __init__(self):
        """Initialize the logit lens service."""
        self._loaded_models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    async def compute_logit_lens_for_sae(
        self,
        db: AsyncSession,
        sae_id: str,
        feature_indices: Optional[List[int]] = None,
        k: int = 20,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        force_recompute: bool = False,
    ) -> Dict[int, LogitLensResult]:
        """
        Compute logit lens data for features of an SAE.

        Args:
            db: Database session
            sae_id: ID of the SAE (external_saes.id)
            feature_indices: List of feature indices to compute, or None for all
            k: Number of top tokens to return (both positive and negative)
            batch_size: Number of features to process at once
            progress_callback: Optional callback(completed, total, message) for progress
            force_recompute: If True, ignore cached results and recompute

        Returns:
            Dictionary mapping feature index to LogitLensResult
        """
        # Load SAE from database
        sae = await db.get(ExternalSAE, sae_id)
        if not sae:
            raise ValueError(f"SAE not found: {sae_id}")

        if sae.status != SAEStatus.READY.value:
            raise ValueError(f"SAE is not ready: {sae.status}")

        if not sae.local_path:
            raise ValueError("SAE has no local path")

        sae_path = Path(sae.local_path)
        if not sae_path.exists():
            raise ValueError(f"SAE path does not exist: {sae.local_path}")

        # Determine model to load - prefer local path over HuggingFace
        model_path = None
        model_hf_name = sae.model_name
        if sae.model_id:
            # Look up the actual model from the database
            model_record = await db.get(Model, sae.model_id)
            if model_record:
                # Check if local paths exist and have model files
                if model_record.quantized_path:
                    qpath = Path(model_record.quantized_path)
                    if qpath.exists() and (qpath / "config.json").exists():
                        model_path = model_record.quantized_path
                if not model_path and model_record.file_path:
                    fpath = Path(model_record.file_path)
                    if fpath.exists() and (fpath / "config.json").exists():
                        model_path = model_record.file_path
                    # Also check for HuggingFace cache directory structure
                    elif fpath.exists():
                        for child in fpath.iterdir():
                            if child.name.startswith("models--") and (child / "snapshots").exists():
                                # Find the latest snapshot
                                snapshots = list((child / "snapshots").iterdir())
                                if snapshots:
                                    model_path = str(snapshots[0])
                                    break
                if not model_hf_name:
                    model_hf_name = model_record.repo_id or model_record.name
        if not model_path and not model_hf_name:
            raise ValueError("SAE has no linked model")

        # Get feature indices if not provided
        if feature_indices is None:
            # Get all features for this SAE from database
            stmt = select(Feature.neuron_index).where(Feature.external_sae_id == sae_id)
            result = await db.execute(stmt)
            feature_indices = [row[0] for row in result.fetchall()]

            if not feature_indices:
                # Fall back to using n_features from SAE metadata
                if sae.n_features:
                    feature_indices = list(range(sae.n_features))
                else:
                    raise ValueError("No features found and SAE has no n_features metadata")

        # Check cache for already computed features
        if not force_recompute:
            cached_results = await self._get_cached_logit_lens(db, sae_id, feature_indices)
            remaining_indices = [i for i in feature_indices if i not in cached_results]
            logger.info(f"Found {len(cached_results)} cached results, {len(remaining_indices)} to compute")
        else:
            cached_results = {}
            remaining_indices = feature_indices

        if not remaining_indices:
            return cached_results

        # Load SAE decoder weights
        logger.info(f"Loading SAE from {sae_path}")
        state_dict, config, _ = load_sae_auto_detect(sae_path, device=self._device)

        # Get decoder weights
        # Community format: W_dec is [d_sae, d_in]
        # miStudio format: decoder.weight is [d_in, d_sae]
        if "W_dec" in state_dict:
            W_dec = state_dict["W_dec"]  # [d_sae, d_in]
        elif "decoder.weight" in state_dict:
            W_dec = state_dict["decoder.weight"].T  # [d_in, d_sae] -> [d_sae, d_in]
        else:
            raise ValueError("Could not find decoder weights in SAE state dict")

        W_dec = W_dec.to(self._device).float()
        logger.info(f"Loaded decoder weights: shape={W_dec.shape}")

        # Load model and get unembedding matrix (prefer local path)
        model_identifier = model_path if model_path else model_hf_name
        model, tokenizer = await self._load_model(model_identifier)
        W_U = self._get_unembedding_matrix(model)
        logger.info(f"Loaded unembedding matrix: shape={W_U.shape}")

        # Compute logit lens in batches
        results = dict(cached_results)
        total = len(remaining_indices)
        completed = len(cached_results)

        for batch_start in range(0, len(remaining_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_indices))
            batch_indices = remaining_indices[batch_start:batch_end]

            if progress_callback:
                progress_callback(
                    completed,
                    total + len(cached_results),
                    f"Computing logit lens for features {batch_start}-{batch_end}"
                )

            try:
                batch_results = await self._compute_batch_logit_lens(
                    W_dec, W_U, tokenizer, batch_indices, k
                )
                results.update(batch_results)
                completed += len(batch_indices)

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM at batch_size={batch_size}, reducing to {batch_size // 2}")
                if batch_size <= self.MIN_BATCH_SIZE:
                    raise RuntimeError("OOM even at minimum batch size")

                # Retry with smaller batch size
                batch_size = batch_size // 2
                torch.cuda.empty_cache()
                batch_results = await self._compute_batch_logit_lens(
                    W_dec, W_U, tokenizer, batch_indices, k
                )
                results.update(batch_results)
                completed += len(batch_indices)

        if progress_callback:
            progress_callback(total + len(cached_results), total + len(cached_results), "Logit lens computation complete")

        return results

    async def _compute_batch_logit_lens(
        self,
        W_dec: torch.Tensor,
        W_U: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        feature_indices: List[int],
        k: int,
    ) -> Dict[int, LogitLensResult]:
        """
        Compute logit lens for a batch of features.

        Args:
            W_dec: Decoder weight matrix [d_sae, d_in]
            W_U: Unembedding matrix [d_in, vocab_size]
            tokenizer: Tokenizer for decoding token IDs
            feature_indices: List of feature indices to process
            k: Number of top tokens to return

        Returns:
            Dictionary mapping feature index to LogitLensResult
        """
        # Get decoder vectors for this batch
        decoder_vectors = W_dec[feature_indices]  # [batch_size, d_in]

        # Compute logits: decoder_vectors @ W_U
        # [batch_size, d_in] @ [d_in, vocab_size] = [batch_size, vocab_size]
        with torch.no_grad():
            feature_logits = decoder_vectors @ W_U

        results = {}
        for i, feature_idx in enumerate(feature_indices):
            logits = feature_logits[i]  # [vocab_size]

            # Top positive (highest logits)
            top_pos_values, top_pos_indices = torch.topk(logits, k)
            top_positive = []
            for idx, val in zip(top_pos_indices.tolist(), top_pos_values.tolist()):
                token = self._safe_decode(tokenizer, idx)
                top_positive.append({
                    "token": token,
                    "token_id": idx,
                    "logit": val,
                })

            # Top negative (lowest logits)
            top_neg_values, top_neg_indices = torch.topk(-logits, k)
            top_negative = []
            for idx, val in zip(top_neg_indices.tolist(), top_neg_values.tolist()):
                token = self._safe_decode(tokenizer, idx)
                top_negative.append({
                    "token": token,
                    "token_id": idx,
                    "logit": -val,  # Convert back to actual (negative) logit
                })

            results[feature_idx] = LogitLensResult(
                feature_index=feature_idx,
                top_positive=top_positive,
                top_negative=top_negative,
            )

        return results

    def _safe_decode(self, tokenizer: PreTrainedTokenizer, token_id: int) -> str:
        """Safely decode a token ID, handling special tokens."""
        try:
            token = tokenizer.decode([token_id])
            # Clean up the token representation
            return token
        except Exception:
            return f"<token_{token_id}>"

    def _get_unembedding_matrix(self, model: PreTrainedModel) -> torch.Tensor:
        """
        Get the unembedding matrix from a model.

        Different models store this in different places:
        - GPT-2: lm_head.weight (vocab_size, d_model)
        - Gemma: model.embed_tokens.weight.T for tied embeddings, or lm_head
        - LLaMA: lm_head.weight

        Returns:
            Unembedding matrix of shape [d_in, vocab_size]
        """
        # Try lm_head first (most common)
        if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
            # lm_head.weight is [vocab_size, d_model]
            return model.lm_head.weight.T.to(self._device).float()  # [d_model, vocab_size]

        # Try model.embed_tokens for tied embeddings
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # embed_tokens.weight is [vocab_size, d_model]
            return model.model.embed_tokens.weight.T.to(self._device).float()

        # Try transformer.wte for GPT-2 style
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            return model.transformer.wte.weight.T.to(self._device).float()

        raise ValueError("Could not find unembedding matrix in model")

    async def _load_model(
        self,
        model_id: str,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a model and tokenizer, with caching."""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        logger.info(f"Loading model: {model_id}")

        # Check if model_id is a local path
        is_local_path = model_id.startswith("/") or model_id.startswith(".")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=is_local_path,
        )

        # Load model (we only need the unembedding matrix, but loading full model
        # ensures compatibility with different architectures)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=is_local_path,
        )

        if self._device == "cpu":
            model = model.to(self._device)

        model.eval()

        self._loaded_models[model_id] = (model, tokenizer)
        logger.info(f"Loaded model {model_id} on {self._device}")

        return model, tokenizer

    async def _get_cached_logit_lens(
        self,
        db: AsyncSession,
        sae_id: str,
        feature_indices: List[int],
    ) -> Dict[int, LogitLensResult]:
        """
        Get cached logit lens results from database.

        Args:
            db: Database session
            sae_id: SAE ID
            feature_indices: Feature indices to look up

        Returns:
            Dictionary mapping feature index to cached LogitLensResult
        """
        results = {}

        # Build feature IDs from SAE ID and indices
        feature_id_prefix = f"feat_sae_{sae_id}_"

        for idx in feature_indices:
            feature_id = f"{feature_id_prefix}{idx}"

            # Look up cached dashboard data
            stmt = select(FeatureDashboardData).where(
                FeatureDashboardData.feature_id == feature_id
            )
            result = await db.execute(stmt)
            cached = result.scalar_one_or_none()

            if cached and cached.logit_lens_data:
                results[idx] = LogitLensResult(
                    feature_index=idx,
                    top_positive=cached.logit_lens_data.get("top_positive", []),
                    top_negative=cached.logit_lens_data.get("top_negative", []),
                )

        return results

    async def save_logit_lens_results(
        self,
        db: AsyncSession,
        sae_id: str,
        results: Dict[int, LogitLensResult],
    ) -> None:
        """
        Save logit lens results to the database.

        Args:
            db: Database session
            sae_id: SAE ID
            results: Dictionary mapping feature index to LogitLensResult
        """
        feature_id_prefix = f"feat_sae_{sae_id}_"

        for idx, result in results.items():
            feature_id = f"{feature_id_prefix}{idx}"

            # Check if dashboard data exists
            stmt = select(FeatureDashboardData).where(
                FeatureDashboardData.feature_id == feature_id
            )
            existing = await db.execute(stmt)
            dashboard_data = existing.scalar_one_or_none()

            logit_lens_json = {
                "top_positive": result.top_positive,
                "top_negative": result.top_negative,
            }

            if dashboard_data:
                # Update existing record
                dashboard_data.logit_lens_data = logit_lens_json
            else:
                # Create new record
                dashboard_data = FeatureDashboardData(
                    feature_id=feature_id,
                    logit_lens_data=logit_lens_json,
                    computation_version="1.0",
                )
                db.add(dashboard_data)

        await db.commit()
        logger.info(f"Saved logit lens results for {len(results)} features")

    def clear_cache(self) -> int:
        """Clear loaded models from cache and free GPU memory."""
        count = len(self._loaded_models)
        for model, _ in self._loaded_models.values():
            del model
        self._loaded_models.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Cleared {count} models from logit lens cache")
        return count


# Global service instance
_logit_lens_service: Optional[LogitLensService] = None


def get_logit_lens_service() -> LogitLensService:
    """Get the global logit lens service instance."""
    global _logit_lens_service
    if _logit_lens_service is None:
        _logit_lens_service = LogitLensService()
    return _logit_lens_service
