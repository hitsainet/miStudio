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
import gc
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
from . import gpu_placement
from .gpu_placement import GpuPlacementError, GpuRequest
from ..ml.community_format import load_sae_auto_detect, CommunityStandardConfig
from ..ml.model_devices import cuda_devices, empty_cache_on
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
        """Initialize the logit lens service.

        There is no service-wide device. The service is a process singleton, and
        in the API process one instance serves every request, each of which may
        name a different card — so the caller passes the device per call and the
        model cache is keyed by (model, device).
        """
        self._loaded_models: Dict[Tuple[str, str], Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}

    async def compute_logit_lens_for_sae(
        self,
        db: AsyncSession,
        sae_id: str,
        feature_indices: Optional[List[int]] = None,
        k: int = 20,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        force_recompute: bool = False,
        *,
        device: torch.device,
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
            device: Where the model, decoder and unembedding live. REQUIRED —
                from :func:`logit_lens_device` in the API process, or
                ``place_job(...).device`` in a worker.

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

        sae_path = settings.resolve_data_path(sae.local_path)
        if not sae_path.exists():
            raise ValueError(f"SAE path does not exist: {sae.local_path}")

        # Determine model to load - prefer local path over HuggingFace
        model_path = None
        model_hf_name = sae.model_name
        #: A downloaded checkpoint to read the unembedding from, when there is one.
        weights_dir = None
        if sae.model_id:
            # Look up the actual model from the database
            model_record = await db.get(Model, sae.model_id)
            if model_record:
                weights_dir = _downloaded_weights_dir(model_record)
                # Check if local paths exist and have model files
                if model_record.quantized_path:
                    qpath = settings.resolve_data_path(model_record.quantized_path)
                    if qpath.exists() and (qpath / "config.json").exists():
                        model_path = str(qpath)
                if not model_path and model_record.file_path:
                    fpath = settings.resolve_data_path(model_record.file_path)
                    if fpath.exists() and (fpath / "config.json").exists():
                        model_path = str(fpath)
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
        state_dict, config, _ = load_sae_auto_detect(sae_path, device=str(device))

        # Get decoder weights
        # After load_sae_auto_detect: W_dec is normalized to [d_in, d_sae]
        # We need [d_sae, d_in] for computing logit lens: feature_logits = W_dec @ W_U
        if "W_dec" in state_dict:
            W_dec = state_dict["W_dec"].T  # [d_in, d_sae] -> [d_sae, d_in]
        elif "decoder.weight" in state_dict:
            W_dec = state_dict["decoder.weight"].T  # [d_in, d_sae] -> [d_sae, d_in]
        else:
            raise ValueError("Could not find decoder weights in SAE state dict")

        W_dec = W_dec.to(device).float()
        logger.info(f"Loaded decoder weights: shape={W_dec.shape} on {device}")

        # THE LENS NEEDS ONE MATRIX, NOT THE MODEL. For a downloaded model the
        # unembedding is read straight from the checkpoint, so the lens runs on
        # one card for a model no card can hold. Only a model known by name
        # alone is loaded whole, on `device`.
        if weights_dir is not None:
            W_U, tokenizer = self._read_unembedding(weights_dir, device)
        else:
            model_identifier = model_path if model_path else model_hf_name
            model, tokenizer = await self._load_model(model_identifier, device)
            W_U = self._get_unembedding_matrix(model, device)
        logger.info(f"Loaded unembedding matrix: shape={W_U.shape}")

        # Compute logit lens in batches
        results = dict(cached_results)
        total = len(remaining_indices)
        completed = len(cached_results)

        from .gpu_job_claim import raise_if_lease_lost

        for batch_start in range(0, len(remaining_indices), batch_size):
            # A LOST GPU LEASE STOPS THE LENS (multi-GPU Phase 3, review round 2). A lens
            # over an SAE's every feature has no cooperative cancellation check, so a job
            # that lost its lease ran on, on a card another job may now hold. Checked at
            # each batch: in a worker it reads the job's claim, in the API process the
            # request's lens lease.
            raise_if_lease_lost("computing the logit lens")
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
                _empty_cache_on(device)
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

    def _get_unembedding_matrix(self, model: PreTrainedModel, device: torch.device) -> torch.Tensor:
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
            return model.lm_head.weight.T.to(device).float()  # [d_model, vocab_size]

        # Try model.embed_tokens for tied embeddings
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # embed_tokens.weight is [vocab_size, d_model]
            return model.model.embed_tokens.weight.T.to(device).float()

        # Try transformer.wte for GPT-2 style
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            return model.transformer.wte.weight.T.to(device).float()

        raise ValueError("Could not find unembedding matrix in model")

    def _read_unembedding(
        self, weights_dir: Path, device: torch.device
    ) -> Tuple[torch.Tensor, PreTrainedTokenizer]:
        """The unembedding ``[d_model, vocab]`` on ``device``, read from a checkpoint, and its tokenizer.

        Loading the whole model to reach ``lm_head`` put every weight on the GPU:
        a model no single card holds could have no logit lens at all, and one
        that fitted stayed on its card in ``_loaded_models`` for the rest of the
        worker's life. The matrix alone is vocab x d_model — 152,064 x 5,120 at
        fp32 is 2.9 GB for a 14B model whose weights are ~29.5 GB. The reader is
        the one the per-feature logit lens already uses (analysis_service), which
        handles nested multimodal layouts and tied embeddings.
        """
        from .analysis_service import load_unembedding_matrix

        weight = load_unembedding_matrix(weights_dir, device="cpu")  # [vocab, d_model]
        tokenizer = AutoTokenizer.from_pretrained(
            str(weights_dir), trust_remote_code=True, local_files_only=True
        )
        return weight.T.to(device).float(), tokenizer

    async def _load_model(
        self,
        model_id: str,
        device: torch.device,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a model and tokenizer onto ``device``, cached per (model, device)."""
        cache_key = (model_id, str(device))
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        logger.info(f"Loading model: {model_id} on {device}")

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
        # ONE card, the one the caller resolved. This was device_map="auto",
        # which spreads the weights over every visible GPU — including a card
        # the request never named — while the unembedding was then moved to
        # "cuda", i.e. whichever card happened to be current.
        on_gpu = device.type == "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": device} if on_gpu else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=is_local_path,
        )

        if not on_gpu:
            model = model.to(device)

        model.eval()

        self._loaded_models[cache_key] = (model, tokenizer)
        logger.info(f"Loaded model {model_id} on {device}")

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
        from sqlalchemy import or_
        from ..models.external_sae import ExternalSAE

        results = {}

        # Get SAE to check for training_id
        sae = await db.get(ExternalSAE, sae_id)
        if not sae:
            return results

        # Build query to get features (check both external_sae_id and training_id)
        if sae.training_id:
            stmt = select(Feature).where(
                or_(
                    Feature.external_sae_id == sae_id,
                    Feature.training_id == sae.training_id
                )
            ).where(Feature.neuron_index.in_(feature_indices))
        else:
            stmt = select(Feature).where(
                Feature.external_sae_id == sae_id
            ).where(Feature.neuron_index.in_(feature_indices))

        result = await db.execute(stmt)
        features = {f.neuron_index: f for f in result.scalars().all()}

        for idx in feature_indices:
            feature = features.get(idx)
            if not feature:
                continue

            # Look up cached dashboard data using actual feature ID
            stmt = select(FeatureDashboardData).where(
                FeatureDashboardData.feature_id == feature.id
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
        from sqlalchemy import or_
        from ..models.external_sae import ExternalSAE

        # Get SAE to check for training_id
        sae = await db.get(ExternalSAE, sae_id)
        if not sae:
            logger.warning(f"SAE not found: {sae_id}")
            return

        # Build query to get features (check both external_sae_id and training_id)
        if sae.training_id:
            stmt = select(Feature).where(
                or_(
                    Feature.external_sae_id == sae_id,
                    Feature.training_id == sae.training_id
                )
            )
        else:
            stmt = select(Feature).where(Feature.external_sae_id == sae_id)

        result = await db.execute(stmt)
        features = {f.neuron_index: f for f in result.scalars().all()}

        saved_count = 0
        for idx, logit_result in results.items():
            # Get actual feature from database
            feature = features.get(idx)
            if not feature:
                # Feature doesn't exist in DB, skip
                continue

            feature_id = feature.id

            # Check if dashboard data exists
            stmt = select(FeatureDashboardData).where(
                FeatureDashboardData.feature_id == feature_id
            )
            existing = await db.execute(stmt)
            dashboard_data = existing.scalar_one_or_none()

            logit_lens_json = {
                "top_positive": logit_result.top_positive,
                "top_negative": logit_result.top_negative,
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
            saved_count += 1

        await db.commit()
        logger.info(f"Saved logit lens results for {saved_count} features")

    def clear_cache(self) -> int:
        """Clear loaded models from cache and free GPU memory."""
        count = len(self._loaded_models)
        devices = {}
        for model, _ in self._loaded_models.values():
            if isinstance(model, torch.nn.Module):
                for device in cuda_devices(model):
                    devices[str(device)] = device
        self._loaded_models.clear()
        gc.collect()

        # The cards the cached models were on, and only those. This walked every
        # card, and entering a card this process never used CREATES a CUDA
        # context there — hundreds of MB taken from whatever job is using it.
        empty_cache_on(devices.values())

        logger.info(f"Cleared {count} models from logit lens cache")
        return count


def _downloaded_weights_dir(model_record: Any) -> Optional[Path]:
    """The snapshot directory holding a downloaded model's raw weights, or None."""
    from .analysis_service import resolve_snapshot_dir

    if not model_record.file_path:
        return None
    raw = settings.resolve_data_path(model_record.file_path)
    if not raw.exists():
        return None
    return resolve_snapshot_dir(raw, model_record.repo_id or "")


def _empty_cache_on(device: torch.device) -> None:
    """Release cached blocks on ``device`` only; a no-op off the GPU."""
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def logit_lens_device(requested: GpuRequest) -> torch.device:
    """The device a logit-lens computation runs on, chosen WITHOUT making it current.

    For the API process (the synchronous dashboard route and the export
    BackgroundTask). ``torch.cuda.set_device`` is per-process and every request
    shares it, so ``place_job`` — which makes the card current — is for Celery
    workers only. Here the card is resolved against live free memory and the
    device is passed explicitly to every load and tensor move.

    ``None`` (an old job with no request) means auto. With no CUDA, auto runs on
    the CPU, as the computation always did; a named card is refused.

    ONE CARD. The lens never splits, so ``"all"`` is refused as a split — with
    the placement's own message — through ``resolve_cards`` without
    ``allow_shard``. ``resolve_card`` read "all" as the name of a card and
    answered "No GPU 'all' on this node".

    Raises:
        GpuPlacementError: the named card does not exist, is not visible to
            this process, there is no CUDA to run a named card on, or the
            request is ``"all"``.
    """
    if not torch.cuda.is_available():
        if gpu_placement.is_auto(requested):
            return torch.device("cpu")
        raise GpuPlacementError(
            f"CUDA is not available here, so GPU {requested!r} cannot be used.",
            requested=requested,
        )
    card = gpu_placement.resolve_cards(requested)[0]
    device = gpu_placement.torch_device(card)
    logger.info("Logit lens placed on %s as %s (requested %r)", card.describe(), device, requested)
    return device


class LogitLensLease:
    """``async with logit_lens_lease(...) as device:`` — a lens in the API PROCESS, under a GPU lease.

    MULTI-GPU PHASE 3 (review round 1, R1-5). The Neuronpedia export's
    BackgroundTask and the synchronous dashboard route compute the logit lens in
    the API process, outside any Celery task, so no lease covered the card: a
    GPU worker could place a job on the card this process was loading a model
    onto. In per-card mode the lens now LEASES its card like any job — through
    ``gpu_job_claim.ClaimContext``, waiting in place at most ``wait_timeout_s``
    (0: a busy card is refused at once) — and releases it, cache first, when the
    block ends however it ends. The process-wide claim (``claiming``) is NOT
    used: an API process serves many requests at once.

    Single mode, and a process without CUDA, are exactly as before:
    :func:`logit_lens_device`.
    """

    def __init__(self, requested: GpuRequest, *, kind: str, wait_timeout_s: Optional[float] = None) -> None:
        self.requested = requested
        self.kind = kind
        self.wait_timeout_s = wait_timeout_s
        self._claim = None
        self._token = None

    async def __aenter__(self) -> torch.device:
        import asyncio

        from .gpu_dispatch import per_card_workers

        if not per_card_workers() or not torch.cuda.is_available():
            return logit_lens_device(self.requested)
        from .gpu_job_claim import DEFAULT_IN_PLACE_TIMEOUT_S, ClaimContext, make_holder

        claim = ClaimContext(
            holder=make_holder(self.kind, None),
            handoff=False,
            wait_timeout_s=DEFAULT_IN_PLACE_TIMEOUT_S if self.wait_timeout_s is None else self.wait_timeout_s,
            release_memory=gpu_placement.empty_cache_on_cards,
        )
        self._claim = claim
        # SHIELDED, AND CLOSED ONLY ONCE THE CLAIM THREAD IS DONE (Phase 3 review round 2).
        # A cancelled await (a client that disconnects, an API shutdown) used to close
        # the claim at once while its thread still ran; that thread then took the lease
        # and started a renewer nothing would ever stop — the card stayed leased, and
        # renewed, until the API process restarted.
        claiming = asyncio.ensure_future(asyncio.to_thread(claim.claim, self.requested))
        try:
            cards = await asyncio.shield(claiming)
            device = gpu_placement.torch_device(cards[0])
        except BaseException:
            self._claim = None
            if claiming.done():
                await asyncio.to_thread(claim.close)
            else:
                import threading

                def close_when_claimed(finished) -> None:
                    if not finished.cancelled():
                        finished.exception()  # retrieved: the caller already has its own error
                    threading.Thread(target=claim.close, daemon=True, name=f"gpu-lens-close-{self.kind}").start()

                claiming.add_done_callback(close_when_claimed)
            raise
        logger.info("Logit lens in the API process leased %s as %s (requested %r)",
                    cards[0].describe(), device, self.requested)
        # The request's claim, for `gpu_job_claim.lease_lost_reason` (review round 2).
        from .gpu_job_claim import _TASK_CLAIM

        self._token = _TASK_CLAIM.set(claim)
        return device

    async def __aexit__(self, *exc_info) -> bool:
        token, self._token = self._token, None
        if token is not None:
            from .gpu_job_claim import _TASK_CLAIM

            try:
                _TASK_CLAIM.reset(token)
            except ValueError:  # exited in another context than it entered
                _TASK_CLAIM.set(None)
        claim, self._claim = self._claim, None
        if claim is not None:
            import asyncio

            await asyncio.to_thread(claim.close)
        return False


def logit_lens_lease(
    requested: GpuRequest, *, kind: str, wait_timeout_s: Optional[float] = None
) -> LogitLensLease:
    """See :class:`LogitLensLease`."""
    return LogitLensLease(requested, kind=kind, wait_timeout_s=wait_timeout_s)


# Global service instance
_logit_lens_service: Optional[LogitLensService] = None


def get_logit_lens_service() -> LogitLensService:
    """Get the global logit lens service instance."""
    global _logit_lens_service
    if _logit_lens_service is None:
        _logit_lens_service = LogitLensService()
    return _logit_lens_service


def _release_idle_logit_lens_models() -> None:
    """Free models the logit lens has cached in THIS process, before a job is placed.

    A worker that ran a dashboard task on a model known only by name kept it in
    ``_loaded_models`` for the rest of its life: a whole model on a card no job
    was using, which Auto then judged fuller, and which the next job placed
    there had to fit beside. Cheap when nothing is cached.
    """
    service = _logit_lens_service
    if service is not None and service._loaded_models:
        service.clear_cache()


gpu_placement.register_idle_release(_release_idle_logit_lens_models)
