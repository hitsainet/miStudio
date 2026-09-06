"""
Steering Service for feature-based model steering.

This module provides the core steering functionality:
1. Loading and managing SAE models for steering
2. Registering forward hooks on transformer layers
3. Modifying activations based on feature strengths
4. Generating steered and unsteered text
5. Computing evaluation metrics (perplexity, coherence, behavioral score)

CRITICAL: This module includes signal handlers and atexit handlers to ensure
GPU memory is properly cleaned up even on abnormal process termination.
"""

import asyncio
import atexit
import gc
import logging
import math
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..core.config import settings
from ..ml.sparse_autoencoder import SparseAutoencoder, create_sae
from ..ml.community_format import (
    load_sae_auto_detect,
    CommunityStandardConfig,
)
from ..schemas.steering import (
    SelectedFeature,
    SteeringComparisonRequest,
    SteeringComparisonResponse,
    GenerationParams,
    AdvancedGenerationParams,
    GenerationMetrics,
    SteeredOutput,
    UnsteeredOutput,
    MultiStrengthResult,
    SteeredOutputMulti,
    SteeringStrengthSweepRequest,
    StrengthSweepResponse,
    StrengthSweepResult,
    CombinedSteeringRequest,
    CombinedSteeringResponse,
    CombinedFeatureApplied,
)
from ..core.clock import utc_now

logger = logging.getLogger(__name__)


# Model architectures whose KV cache is incompatible with steering forward hooks
# and must therefore generate with use_cache=False. Matched (substring) against
# the model's `model_type` and `architectures`. Gemma-2 uses a hybrid
# sliding-window cache that breaks under hooks; every other architecture keeps
# the cache (verified token-identical on LFM2, ~10-15x faster). Extend this list
# if another hybrid-cache model is found to need it.
_CACHE_INCOMPATIBLE_MARKERS: tuple[str, ...] = ("gemma2", "gemma_2", "gemma-2")


# =============================================================================
# GPU CLEANUP ON ABNORMAL EXIT
# =============================================================================
# These handlers ensure GPU memory is freed even when the process is killed
# by a signal or exits abnormally. Without these, zombie processes can hold
# GPU memory indefinitely.


def _emergency_gpu_cleanup():
    """
    Emergency GPU cleanup called on process exit or signal.

    This is a last-resort cleanup that runs independently of any service instance.
    It clears ALL GPU caches across all available GPUs.
    """
    try:
        logger.warning("[Emergency GPU Cleanup] Running emergency GPU cleanup...")
        gc.collect()

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for gpu_id in range(num_gpus):
                try:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"[Emergency GPU Cleanup] GPU {gpu_id} cleanup error: {e}")
            logger.warning(f"[Emergency GPU Cleanup] Cleared cache on {num_gpus} GPU(s)")

        gc.collect()
    except Exception as e:
        # Last resort - try to at least log the error
        try:
            logger.error(f"[Emergency GPU Cleanup] Failed: {e}")
        except Exception:
            pass  # If logging fails, silently ignore


def _signal_handler(signum, frame):
    """
    Signal handler for SIGTERM and SIGINT.

    Runs emergency GPU cleanup before allowing the process to terminate.

    COOPERATIVE when a steering task is executing: raising SystemExit inside
    a running task (the self-exit SIGTERM landing after the next task already
    started) crashed celery's solo pool with the unrecoverable "cannot unpack
    non-iterable ExceptionInfo" TypeError and stranded the in-flight message.
    Instead we defer: mark shutdown_deferred and return; task_postrun in
    steering_tasks.py completes the shutdown once the task has finished.
    """
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)

    try:
        from ..workers import steering_worker_state as worker_state
    except Exception:
        worker_state = None

    if worker_state is not None and worker_state.busy_task_id is not None:
        worker_state.shutdown_deferred = True
        logger.warning(
            f"[Signal Handler] Received {sig_name} mid-task "
            f"{worker_state.busy_task_id} — deferring shutdown to task end"
        )
        return

    logger.warning(f"[Signal Handler] Received {sig_name}, running GPU cleanup...")
    _emergency_gpu_cleanup()

    # Re-raise the signal to allow default handling (process termination)
    # Reset to default handler to avoid infinite loop
    signal.signal(signum, signal.SIG_DFL)
    raise SystemExit(128 + signum)


# Register signal handlers
# Note: These may not work in all contexts (e.g., inside async loops)
# but provide an extra layer of protection
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    logger.info("[Signal Handler] Registered SIGTERM and SIGINT handlers for GPU cleanup")
except Exception as e:
    logger.warning(f"[Signal Handler] Could not register signal handlers: {e}")

# Register atexit handler
# This runs on normal exit, sys.exit(), and unhandled exceptions
atexit.register(_emergency_gpu_cleanup)
logger.info("[atexit] Registered emergency GPU cleanup on exit")


# =============================================================================
# GENERATION WATCHDOG
# =============================================================================
# Monitors generation time and forcefully terminates if stuck
# This prevents zombie processes from holding GPU memory

class GenerationWatchdog:
    """
    Watchdog that monitors generation time and forcefully terminates if stuck.

    When model.generate() hangs (common with certain model/hook combinations),
    the async timeout can't interrupt the blocking call. This watchdog runs in
    a separate thread and will forcefully terminate the process if generation
    exceeds the hard timeout.

    This is aggressive but necessary to prevent zombie processes from holding
    GPU memory indefinitely.
    """

    def __init__(self, hard_timeout: float = 90.0):
        """
        Initialize watchdog.

        Args:
            hard_timeout: Seconds before forceful termination (default 90s)
        """
        self.hard_timeout = hard_timeout
        self._generation_start: Optional[float] = None
        self._generation_active = False
        self._lock = threading.Lock()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_generation(self):
        """Mark that generation has started."""
        with self._lock:
            self._generation_start = time.time()
            self._generation_active = True
            logger.debug(f"[Watchdog] Generation started, timeout in {self.hard_timeout}s")

    def end_generation(self):
        """Mark that generation has completed."""
        with self._lock:
            elapsed = time.time() - self._generation_start if self._generation_start else 0
            self._generation_active = False
            self._generation_start = None
            logger.debug(f"[Watchdog] Generation completed in {elapsed:.1f}s")

    def _watchdog_loop(self):
        """Watchdog thread loop - monitors for stuck generations."""
        while not self._stop_event.wait(timeout=5.0):  # Check every 5 seconds
            with self._lock:
                if self._generation_active and self._generation_start:
                    elapsed = time.time() - self._generation_start
                    if elapsed > self.hard_timeout:
                        logger.error(
                            f"[Watchdog] Generation exceeded hard timeout ({elapsed:.1f}s > {self.hard_timeout}s). "
                            f"Forcefully terminating to prevent zombie process."
                        )
                        # Clean up GPU first
                        try:
                            _emergency_gpu_cleanup()
                        except Exception:
                            pass
                        # Forcefully terminate - let supervisor restart
                        logger.error("[Watchdog] Calling os._exit(1) to terminate process")
                        os._exit(1)
                    elif elapsed > self.hard_timeout * 0.75:
                        logger.warning(
                            f"[Watchdog] Generation taking long ({elapsed:.1f}s), "
                            f"will terminate at {self.hard_timeout}s"
                        )

    def start(self):
        """Start the watchdog thread."""
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._stop_event.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name="SteeringWatchdog"
            )
            self._watchdog_thread.start()
            logger.info(f"[Watchdog] Started with {self.hard_timeout}s hard timeout")

    def stop(self):
        """Stop the watchdog thread."""
        self._stop_event.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
            logger.info("[Watchdog] Stopped")


# Global watchdog instance
_generation_watchdog: Optional[GenerationWatchdog] = None


def get_generation_watchdog() -> GenerationWatchdog:
    """Get or create the global generation watchdog."""
    global _generation_watchdog
    if _generation_watchdog is None:
        from ..core.config import settings
        # Use steering timeout as base, add 30 seconds buffer for hard kill
        hard_timeout = settings.steering_timeout_seconds + 30
        _generation_watchdog = GenerationWatchdog(hard_timeout=hard_timeout)
        _generation_watchdog.start()
    return _generation_watchdog


@dataclass
class FeatureSteeringConfig:
    """Configuration for a single feature's steering."""

    feature_idx: int
    layer: int
    strength: float  # Raw steering coefficient (matches Neuronpedia)
    label: Optional[str] = None
    color: str = "teal"
    # Feature 015: the SAE that steers THIS feature (its own layer's SAE). None
    # in every single-SAE flow, which routes through the request-level SAE.
    sae_id: Optional[str] = None

    @property
    def multiplier(self) -> float:
        """
        Convert strength to activation multiplier.

        Neuronpedia-compatible calibration:
        The strength value IS the raw coefficient used in the formula:
            activations += coefficient * steering_vector

        Examples:
            0 -> no change
            0.07 -> very subtle effect
            1 -> add 1x the feature direction
            80 -> strong effect (80x the feature direction)
            -1 -> subtract 1x the feature direction (suppression)

        This matches Neuronpedia's steering interface exactly.
        """
        return 1 + self.strength


def resolve_decoder_weight(sae_model) -> Optional["torch.Tensor"]:
    """
    Resolve an SAE's decoder weight matrix as [d_model, d_sae].

    Single source of truth for decoder orientation across the steering hook and
    the cluster-allocation service (Feature 013) — the gain computation must see
    exactly the directions the hook will inject.
    """
    if hasattr(sae_model, 'tied_weights') and sae_model.tied_weights:
        # Tied weights: decoder = encoder.weight.T
        return sae_model.encoder.weight.t()  # [d_model, d_sae]
    if hasattr(sae_model, 'decoder_weight') and not isinstance(getattr(sae_model, 'decoder', None), nn.Linear):
        # JumpReLUSAE: decoder_weight property returns [d_model, d_sae]
        return sae_model.decoder_weight
    if hasattr(sae_model, 'decoder') and sae_model.decoder is not None:
        if hasattr(sae_model.decoder, 'weight'):
            return sae_model.decoder.weight  # [d_model, d_sae]
    return None


def resolve_encoder_weight(sae_model) -> Optional["torch.Tensor"]:
    """
    Resolve an SAE's encoder weight matrix as [d_sae, d_model].

    Companion to resolve_decoder_weight — the single orientation source for
    the cross-layer weight prior (IDL-32: cos(W_dec(Li)[:,i], W_enc(Lj)[j,:]))
    and Feature 015's hazard detection. Handles the same format families:
    tied weights, JumpReLU encoder_weight property, and nn.Linear encoders
    (whose .weight is already [d_sae, d_model] by torch convention).
    """
    if hasattr(sae_model, 'tied_weights') and sae_model.tied_weights:
        return sae_model.encoder.weight  # [d_sae, d_model] (Linear convention)
    if hasattr(sae_model, 'encoder_weight') and not isinstance(getattr(sae_model, 'encoder', None), nn.Linear):
        # Defensive branch: no current SAE class defines encoder_weight (JumpReLU
        # resolves via the encoder-compat property below). If a future format adds
        # one, VERIFY its orientation is [d_sae, d_model] before trusting it here.
        return sae_model.encoder_weight
    if hasattr(sae_model, 'encoder') and sae_model.encoder is not None:
        if hasattr(sae_model.encoder, 'weight'):
            return sae_model.encoder.weight  # [d_sae, d_model]
    return None


def load_sae_weights_cpu(sae_path, *, d_model=None, n_features=None,
                         architecture=None):
    """Load JUST the decoder+encoder weight tensors on CPU — for the
    read-only cluster-allocation math (gain G + hazard weight prior). Does NOT
    touch the GPU or the steering-service SAE cache (015 R1 QA-1: the
    allocation endpoint used to force-resident-load N full SAEs onto the 24 GB
    card on a 'read-only' click). Returns (decoder[d_model,d_sae],
    encoder[d_sae,d_model]) or (None, None) on failure."""
    from ..ml.sparse_autoencoder import create_sae

    state_dict, config, _fmt = load_sae_auto_detect(sae_path, device="cpu")
    if config is not None:
        d_in, d_sae = config.d_in, config.d_sae
        arch = config.architecture or "standard"
        normalize = config.normalize_activations or "none"
    else:
        enc = state_dict.get("encoder.weight")
        if enc is not None:
            d_sae, d_in = enc.shape
        else:
            d_in, d_sae = (d_model or 768), (n_features or 8192)
        arch = architecture or "standard"
        normalize = "constant_norm_rescale"
    sae = create_sae(architecture_type=arch, hidden_dim=d_in, latent_dim=d_sae,
                     l1_alpha=0.001, normalize_activations=normalize)
    # STRICT load (matches load_sae): a key mismatch would otherwise leave the
    # decoder at its random init and produce GARBAGE G/prior silently (R2). A
    # mismatch raises → the endpoint's try/except degrades to approximate-G,
    # which is honest, rather than a plausible-looking wrong number.
    missing, unexpected = sae.load_state_dict(state_dict, strict=False)
    # A key-name mismatch (wrong arch class) leaves a weight at its RANDOM INIT
    # while keeping the right SHAPE — so a shape check alone passes on garbage.
    # Reject when ANY weight-bearing param didn't load (missing_keys), which is
    # the actual signal of an arch/format mismatch → the endpoint degrades to
    # approximate-G rather than trusting init noise (R3).
    weighty_missing = [k for k in missing if k.endswith((".weight", "_weight"))
                       or k in ("W_enc", "W_dec")]
    dw = resolve_decoder_weight(sae)
    ew = resolve_encoder_weight(sae)
    if weighty_missing or dw is None or ew is None \
            or dw.shape[0] != d_in or ew.shape[1] != d_in:
        raise ValueError(
            f"SAE weights failed to load cleanly (missing={weighty_missing[:3]}, "
            f"unexpected={list(unexpected)[:3]}) — refusing garbage decoder/encoder")
    sae.eval()  # CPU, fp32 — never .to(cuda), never .half()
    return dw.detach(), ew.detach()


@dataclass
class LoadedSAE:
    """Container for a loaded SAE model and its metadata."""

    model: SparseAutoencoder
    config: Optional[CommunityStandardConfig]
    layer: int
    d_in: int
    d_sae: int
    device: str


@dataclass
class SaeMeta:
    """Load-time metadata for one referenced SAE (Feature 015).

    Threaded from the endpoint (which owns DB access) into generate_combined so
    the worker can resolve EVERY distinct sae_id a multi-layer circuit
    references — not just the request-level one. Single-SAE flows carry exactly
    one entry, so behaviour is unchanged.
    """

    sae_id: str
    sae_path: str
    layer: Optional[int] = None
    d_model: Optional[int] = None
    n_features: Optional[int] = None
    architecture: Optional[str] = None


class SaeLayerMismatchError(Exception):
    """A steered feature targets a layer whose SAE was not the one supplied.

    Raised by resolve_sae_map at SUBMIT time (the endpoint turns it into a 422
    listing offenders) so a mis-serve never reaches the GPU worker. Each
    offender is {feature_idx, layer, sae_id, sae_layer}.
    """

    def __init__(self, offenders: List[Dict[str, Any]]):
        self.offenders = offenders
        detail = ", ".join(
            f"feature {o['feature_idx']} on layer {o['layer']} routed to SAE "
            f"{o['sae_id']} (layer {o['sae_layer']})"
            for o in offenders
        )
        super().__init__(f"feature/SAE layer mismatch: {detail}")


@dataclass
class SteeringContext:
    """Context for an active steering session."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    sae: LoadedSAE
    feature_configs: Dict[int, List[FeatureSteeringConfig]]  # layer -> configs
    hook_handles: List[Any] = field(default_factory=list)

    def cleanup(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


class SteeringService:
    """
    Service class for SAE-based model steering.

    Handles:
    - Loading and caching SAE models
    - Registering steering hooks on transformer layers
    - Generating steered and unsteered text
    - Computing evaluation metrics
    """

    def __init__(self):
        """Initialize the steering service."""
        self._loaded_saes: Dict[str, LoadedSAE] = {}
        self._loaded_models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}
        self._sentence_model = None  # Lazy-loaded for coherence metrics
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def cleanup_gpu(
        self,
        model: Optional[PreTrainedModel] = None,
        device_id: Optional[int] = None,
    ) -> None:
        """
        Clean up GPU memory to prevent memory leaks.

        Supports multi-GPU systems by cleaning all available GPUs or a specific one.

        This should be called:
        - After any error during model operations
        - After completing steering operations
        - When explicitly unloading models

        Args:
            model: Optional model to clear hooks from before cleanup
            device_id: Optional specific GPU to clean (None = all GPUs)
        """
        try:
            # Clear hooks from the model if provided
            if model is not None:
                self._clear_all_model_hooks(model)

            # Force garbage collection first to release Python references
            gc.collect()

            # Clear CUDA cache on all GPUs or specific GPU
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()

                if device_id is not None:
                    # Clean specific GPU
                    if device_id < num_gpus:
                        with torch.cuda.device(device_id):
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        logger.info(f"[GPU Cleanup] GPU {device_id} cache cleared")
                else:
                    # Clean ALL GPUs - critical for multi-GPU systems
                    for gpu_id in range(num_gpus):
                        with torch.cuda.device(gpu_id):
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    logger.info(f"[GPU Cleanup] Cleared cache on {num_gpus} GPU(s)")

            # Second garbage collection pass
            gc.collect()

            # Final CUDA cleanup pass on all GPUs
            if torch.cuda.is_available():
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"[GPU Cleanup] Error during cleanup: {e}")

    def unload_model(self, model_id: str) -> bool:
        """
        Explicitly unload a model from cache and GPU memory.

        Args:
            model_id: The model identifier to unload

        Returns:
            True if model was unloaded, False if not found
        """
        if model_id not in self._loaded_models:
            logger.info(f"[Unload] Model {model_id} not in cache")
            return False

        try:
            model, tokenizer = self._loaded_models.pop(model_id)

            # Clear any hooks
            self._clear_all_model_hooks(model)

            # Move model to CPU first (helps with GPU memory release)
            try:
                model.to("cpu")
            except Exception:
                pass

            # Delete references
            del model
            del tokenizer

            # Clean up GPU
            self.cleanup_gpu()

            logger.info(f"[Unload] Model {model_id} unloaded and GPU cleaned")
            return True

        except Exception as e:
            logger.error(f"[Unload] Error unloading model {model_id}: {e}")
            self.cleanup_gpu()
            return False

    def unload_all_models(self) -> int:
        """
        Unload all cached models and clean GPU memory.

        Returns:
            Number of models unloaded
        """
        model_ids = list(self._loaded_models.keys())
        count = 0

        for model_id in model_ids:
            if self.unload_model(model_id):
                count += 1

        # Final cleanup
        self.cleanup_gpu()
        logger.info(f"[Unload] Unloaded {count} models, GPU cleaned")
        return count

    async def load_sae(
        self,
        sae_path: Path,
        sae_id: str,
        force_reload: bool = False,
        # Fallback metadata from database (used when config is not in checkpoint)
        layer: Optional[int] = None,
        d_model: Optional[int] = None,
        n_features: Optional[int] = None,
        architecture: Optional[str] = None,
    ) -> LoadedSAE:
        """
        Load an SAE from disk.

        Args:
            sae_path: Path to the SAE directory
            sae_id: Unique identifier for caching
            force_reload: Whether to reload even if cached
            layer: Fallback layer from database
            d_model: Fallback hidden dimension from database
            n_features: Fallback latent dimension from database
            architecture: Fallback architecture type from database

        Returns:
            LoadedSAE instance
        """
        if sae_id in self._loaded_saes and not force_reload:
            return self._loaded_saes[sae_id]

        # If force_reload and SAE exists in cache, clean it up first
        if force_reload and sae_id in self._loaded_saes:
            logger.info(f"[Force Reload] Cleaning up existing SAE {sae_id}")
            try:
                old_sae = self._loaded_saes.pop(sae_id)
                # Move SAE model to CPU and delete
                if hasattr(old_sae.model, 'cpu'):
                    old_sae.model.cpu()
                del old_sae
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"[Force Reload] SAE cleanup error: {e}")

        logger.info(f"Loading SAE from {sae_path}")

        # Load SAE weights and config using auto-detect
        state_dict, config, format_type = load_sae_auto_detect(
            sae_path,
            device=self._device,
        )

        # Debug logging
        logger.info(f"SAE format detected: {format_type}")
        logger.info(f"State dict keys: {list(state_dict.keys())}")
        logger.info(f"Config: {config}")

        # Determine dimensions - from config if available, otherwise from weights/database
        if config is not None:
            d_in = config.d_in
            d_sae = config.d_sae
            sae_layer = config.hook_point_layer
            arch_type = config.architecture or "standard"
            normalize = config.normalize_activations or "none"
            l1_coef = config.l1_coefficient or 0.001
        else:
            # Infer dimensions from weights
            # encoder.weight shape is [d_sae, d_in] for miStudio format
            encoder_weight = state_dict.get("encoder.weight")
            if encoder_weight is not None:
                d_sae, d_in = encoder_weight.shape
            else:
                # Use database fallbacks
                d_in = d_model or 768
                d_sae = n_features or 8192

            # Use database fallbacks for other params
            sae_layer = layer or 0
            arch_type = architecture or "standard"
            normalize = "constant_norm_rescale"  # Default for miStudio
            l1_coef = 0.001

            logger.info(f"Using inferred dimensions: d_in={d_in}, d_sae={d_sae}, layer={sae_layer}")

        # Create SAE model
        logger.info(f"Creating SAE with arch_type={arch_type}, d_in={d_in}, d_sae={d_sae}")
        sae_model = create_sae(
            architecture_type=arch_type,
            hidden_dim=d_in,
            latent_dim=d_sae,
            l1_alpha=l1_coef,
            normalize_activations=normalize,
        )
        logger.info(f"Model expects keys: {list(sae_model.state_dict().keys())}")

        # Load weights and ensure correct dtype
        sae_model.load_state_dict(state_dict)
        sae_model.to(self._device)
        # Convert to FP16 if on CUDA to match model dtype
        if self._device == "cuda":
            sae_model.half()
        sae_model.eval()

        loaded = LoadedSAE(
            model=sae_model,
            config=config,
            layer=sae_layer,
            d_in=d_in,
            d_sae=d_sae,
            device=self._device,
        )

        self._loaded_saes[sae_id] = loaded
        logger.info(f"Loaded SAE {sae_id}: d_in={d_in}, d_sae={d_sae}, layer={sae_layer}")

        return loaded

    async def resolve_sae_map(
        self,
        request: "CombinedSteeringRequest",
        sae_meta_map: Dict[str, "SaeMeta"],
        *,
        force_reload: bool = False,
    ) -> Dict[str, LoadedSAE]:
        """Load every DISTINCT SAE a combined request references (Feature 015).

        Each feature steers through the SAE trained on ITS layer: the SAE for a
        feature is ``feature.sae_id`` when present, else the request-level
        ``request.sae_id``. This collects the distinct ids, loads each via the
        EXISTING ``load_sae`` (a cache hit when already resident), and validates
        ``feature.layer == loaded.layer`` for every feature — raising
        ``SaeLayerMismatchError`` listing offenders BEFORE any generation.

        Regression note: when only one distinct sae_id is referenced (every
        single-SAE flow), the returned map has a single entry and the caller's
        behaviour is unchanged.

        Args:
            request: the combined steering request.
            sae_meta_map: {sae_id -> SaeMeta} load metadata for the referenced
                ids (built by the endpoint, which owns DB access).
            force_reload: forwarded to ``load_sae`` (the request-level SAE is
                force-reloaded today to dodge cached-state corruption).

        Returns:
            {sae_id -> LoadedSAE} for exactly the referenced ids.
        """
        referenced: List[str] = []
        for f in request.selected_features:
            sid = f.sae_id or request.sae_id
            if sid not in referenced:
                referenced.append(sid)

        sae_map: Dict[str, LoadedSAE] = {}
        for sid in referenced:
            meta = sae_meta_map.get(sid)
            if meta is None:
                # Endpoint validates existence up-front; this guards the worker
                # path against a request that slipped through with an unknown id.
                raise SaeLayerMismatchError([{
                    "feature_idx": -1, "layer": -1,
                    "sae_id": sid, "sae_layer": None,
                }])
            sae_map[sid] = await self.load_sae(
                Path(meta.sae_path),
                sid,
                force_reload=force_reload,
                layer=meta.layer,
                d_model=meta.d_model,
                n_features=meta.n_features,
                architecture=meta.architecture,
            )

        # Per-feature layer validation against the SAE that will steer it.
        offenders: List[Dict[str, Any]] = []
        for f in request.selected_features:
            sid = f.sae_id or request.sae_id
            loaded = sae_map[sid]
            if f.layer != loaded.layer:
                offenders.append({
                    "feature_idx": f.feature_idx,
                    "layer": f.layer,
                    "sae_id": sid,
                    "sae_layer": loaded.layer,
                })
        if offenders:
            raise SaeLayerMismatchError(offenders)

        return sae_map

    def _find_hf_model_path(self, base_path: Path) -> Optional[Path]:
        """
        Find the actual model path in HuggingFace cache structure.

        HF cache structure is: base_path/models--org--name/snapshots/hash/
        This method finds the most recent snapshot.

        Args:
            base_path: Base path that may contain HF cache structure

        Returns:
            Path to the actual model files or None if not found
        """
        base_path = Path(base_path)

        # Check if there's a models-- subdirectory (HF cache format)
        model_dirs = list(base_path.glob("models--*"))
        if not model_dirs:
            # Not HF cache format, check if it's a direct model directory
            if (base_path / "config.json").exists():
                return base_path
            return None

        # Get the first (should be only one) model directory
        model_dir = model_dirs[0]

        # Find snapshots
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        # Get the most recent snapshot (by directory listing order)
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            return None

        # Return the first snapshot (usually there's only one)
        for snapshot in snapshots:
            if (snapshot / "config.json").exists():
                return snapshot

        return None

    async def load_model(
        self,
        model_id: str,
        model_path: Optional[str] = None,
        force_reload: bool = False,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a transformer model for steering.

        Args:
            model_id: Model identifier (HF model name or local path)
            model_path: Optional local path override
            force_reload: Whether to reload even if cached

        Returns:
            Tuple of (model, tokenizer)
        """
        cache_key = model_id

        if cache_key in self._loaded_models and not force_reload:
            return self._loaded_models[cache_key]

        # If force_reload and model exists in cache, clean it up first
        if force_reload and cache_key in self._loaded_models:
            logger.info(f"[Force Reload] Cleaning up existing model {model_id}")
            try:
                old_model, _ = self._loaded_models.pop(cache_key)
                # Clear hooks before cleanup
                self._clear_all_model_hooks(old_model)
                # Move to CPU and delete
                old_model.cpu()
                del old_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"[Force Reload] Cleanup error: {e}")

        logger.info(f"Loading model {model_id}")

        # Determine path
        load_path = model_path or model_id

        # Check if it's a local path
        local_path = settings.data_dir / "models" / model_id
        if local_path.exists():
            load_path = str(local_path)

        # If model_path is provided, check for HF cache structure
        if model_path:
            actual_model_path = self._find_hf_model_path(Path(model_path))
            if actual_model_path:
                load_path = str(actual_model_path)
                logger.info(f"Found model in HF cache at {load_path}")
            else:
                logger.warning(f"Could not find model files in {model_path}, using as-is")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        # CRITICAL: Check GPU memory BEFORE loading to avoid silent CPU fallback
        if self._device == "cuda" and torch.cuda.is_available():
            gpu_free_mb = torch.cuda.mem_get_info()[0] / 1024**2
            # Rough estimate: need at least 2GB free for small models
            if gpu_free_mb < 2000:
                raise RuntimeError(
                    f"Insufficient GPU memory for model loading. "
                    f"Available: {gpu_free_mb:.0f}MB, Required: ~2000MB minimum. "
                    f"Another process may be holding GPU memory (check for zombie processes)."
                )
            logger.info(f"[GPU Memory] {gpu_free_mb:.0f}MB free before model load")

        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )

        if self._device != "cuda":
            model.to(self._device)

        model.eval()

        # CRITICAL: Verify model is actually on GPU after loading
        # device_map="auto" can silently place model on CPU if GPU memory is low
        if self._device == "cuda":
            try:
                first_param = next(model.parameters())
                if first_param.device.type != "cuda":
                    raise RuntimeError(
                        f"Model failed to load on GPU (found on {first_param.device}). "
                        f"This usually means GPU memory is insufficient or held by another process. "
                        f"Check for zombie processes with: nvidia-smi --query-compute-apps=pid --format=csv"
                    )
                logger.info(f"[GPU Verification] Model confirmed on {first_param.device}")
            except StopIteration:
                logger.warning("[GPU Verification] Model has no parameters to verify device")

        self._loaded_models[cache_key] = (model, tokenizer)
        logger.info(f"Loaded model {model_id}")

        return model, tokenizer

    def _ensure_model_on_gpu(self, model: PreTrainedModel) -> None:
        """
        Verify and ensure model is on GPU after operations.

        Sometimes models can drift to CPU under memory pressure or due to
        automatic device placement. This ensures the model stays on the
        intended device for optimal performance.
        """
        if not torch.cuda.is_available() or self._device == "cpu":
            return

        # Check if any parameter is on CPU when it shouldn't be
        try:
            first_param = next(model.parameters())
            if first_param.device.type == "cpu":
                logger.warning(
                    f"[Model Device] Model found on CPU, moving back to {self._device}"
                )
                model.to(self._device)
                logger.info(f"[Model Device] Model restored to {self._device}")
        except StopIteration:
            pass  # No parameters to check
        except Exception as e:
            logger.warning(f"[Model Device] Could not verify device: {e}")

    def _clear_all_model_hooks(self, model: PreTrainedModel) -> int:
        """
        Clear all forward hooks from transformer layer modules.

        This prevents stale hooks from previous requests that may have timed out
        or failed from contaminating subsequent generations. Critical for ensuring
        unsteered baselines are truly unsteered.

        Uses dynamic layer discovery to find layers, then walks ALL submodules
        of each layer to clear hooks regardless of naming convention.

        Args:
            model: The transformer model to clear hooks from

        Returns:
            Number of hooks cleared
        """
        from ..ml.layer_discovery import discover_transformer_structure

        hooks_cleared = 0

        # Use dynamic discovery to find transformer layers
        try:
            structure = discover_transformer_structure(model)
            layers_module = structure.layers_module
        except ValueError:
            logger.warning("Could not find transformer layers to clear hooks from")
            return 0

        for layer_idx, layer in enumerate(layers_module):
            # Clear hooks on the layer module itself
            if hasattr(layer, "_forward_hooks") and layer._forward_hooks:
                count = len(layer._forward_hooks)
                layer._forward_hooks.clear()
                hooks_cleared += count

            # Walk ALL submodules of this layer (architecture-agnostic)
            for name, submodule in layer.named_modules():
                if name == "":
                    continue  # Skip the layer itself (already handled)
                if hasattr(submodule, "_forward_hooks") and submodule._forward_hooks:
                    count = len(submodule._forward_hooks)
                    submodule._forward_hooks.clear()
                    hooks_cleared += count

        if hooks_cleared > 0:
            logger.warning(
                f"Cleared {hooks_cleared} stale forward hooks from model. "
                "This indicates a previous request did not clean up properly."
            )

        return hooks_cleared

    def _reset_model_state(self, model: PreTrainedModel) -> None:
        """
        Reset all internal model state to ensure clean generation.

        CRITICAL: This ensures prior prompts have NO influence on subsequent generations.
        Must be called before EVERY generation to guarantee context isolation.

        Clears:
        - KV cache (past_key_values)
        - Static cache (for Gemma-2 and similar)
        - Internal cache buffers
        - Any other mutable state

        Args:
            model: The transformer model to reset
        """
        # 1. Clear any past_key_values attribute
        if hasattr(model, "past_key_values"):
            model.past_key_values = None

        # 2. Reset static cache for models that use it (Gemma-2, etc.)
        if hasattr(model, "_cache"):
            model._cache = None

        # 3. Call model's reset_cache method if available (transformers >= 4.38)
        if hasattr(model, "_reset_cache"):
            try:
                model._reset_cache()
                logger.debug("[Model State] Called model._reset_cache()")
            except Exception as e:
                logger.debug(f"[Model State] _reset_cache() not applicable: {e}")

        # 4. For models with HybridCache or StaticCache
        if hasattr(model, "model") and hasattr(model.model, "_cache"):
            model.model._cache = None

        # 5. Clear cache on config level
        if hasattr(model, "config"):
            # Ensure we don't accidentally enable caching
            if hasattr(model.config, "use_cache"):
                # Note: Don't persist this, just check it
                pass

        # 6. Clear per-layer caches (sliding window, attention KV cache, etc.)
        # Use dynamic discovery to find layers for any architecture
        from ..ml.layer_discovery import discover_transformer_structure
        try:
            structure = discover_transformer_structure(model)
            for layer in structure.layers_module:
                # Clear any layer-level cache
                if hasattr(layer, "_cache"):
                    layer._cache = None
                # Walk all submodules to clear attention caches
                for name, submodule in layer.named_modules():
                    if hasattr(submodule, "_cache"):
                        submodule._cache = None
                    if hasattr(submodule, "past_key_value"):
                        submodule.past_key_value = None
        except ValueError:
            pass  # If layers can't be found, skip per-layer cache clearing

        logger.debug("[Model State] Model state reset for clean generation")

    def _get_target_module(
        self,
        model: PreTrainedModel,
        layer: int,
        hook_type: str = "resid_post",
    ) -> Optional[nn.Module]:
        """
        Get the target module for hook registration.

        Uses dynamic layer discovery to support any transformer architecture
        without hardcoded mappings.

        Args:
            model: The transformer model
            layer: Layer index
            hook_type: Type of hook (resid_pre, resid_post, attn, mlp)

        Returns:
            Target module or None if not found
        """
        from ..ml.layer_discovery import discover_transformer_structure

        try:
            structure = discover_transformer_structure(model)
            layers_module = structure.layers_module

            if layer < len(layers_module):
                return layers_module[layer]
            else:
                logger.warning(
                    f"Layer {layer} exceeds model depth {len(layers_module)}"
                )
                return None
        except ValueError as e:
            logger.error(f"Could not discover transformer layers: {e}")
            return None

    def _create_steering_hook(
        self,
        sae: LoadedSAE,
        feature_configs: List[FeatureSteeringConfig],
    ) -> Callable:
        """
        Create a steering hook function using direct steering method.

        For each steered feature, we:
        1. Get the steering vector from the SAE (feature's decoder weights)
        2. Compute steering_coefficient = multiplier - 1 (so multiplier=1 means no change)
        3. Add (steering_coefficient * steering_vector) to ALL token activations

        This direct method applies steering uniformly to all tokens, regardless of
        whether the feature naturally activates on the input. Benefits:
        - Works for sparse features that may not activate on the prompt
        - Consistent results regardless of activation values
        - Simpler and more predictable behavior

        Reference: https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/11859

        IMPORTANT: We use IN-PLACE modification of hidden_states and return the
        original output tuple. This is required for compatibility with Gemma-2 and
        other models that use internal tensor references. Creating new tensors and
        returning a new tuple causes shape mismatches in subsequent layers.

        Args:
            sae: Loaded SAE model
            feature_configs: List of feature steering configurations

        Returns:
            Hook function compatible with PyTorch register_forward_hook
        """
        def steering_hook(module, input, output):
            try:
                # Handle both tuple and single tensor outputs
                # Standard transformers return tuples (hidden_states, ...) but some
                # architectures (e.g., LFM2/Liquid, custom models) return single tensors
                is_tuple = isinstance(output, tuple)
                if is_tuple:
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # MIS-E2E-068(3): DEBUG, and guarded.
                #
                # This was `logger.info` inside the forward hook, so it fired on
                # every forward pass — hundreds of lines per generation, each
                # rebuilding two lists by comprehension purely to format a
                # message that is almost always discarded. The guard matters as
                # much as the level: without it the f-string and both
                # comprehensions are evaluated even when DEBUG is off.
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"[Steering Hook] FIRED on {type(module).__name__}, "
                        f"output_type={'tuple' if is_tuple else 'tensor'}, "
                        f"shape={hidden_states.shape}, dtype={hidden_states.dtype}, "
                        f"features={[c.feature_idx for c in feature_configs]}, "
                        f"strengths={[c.strength for c in feature_configs]}"
                    )

                # Validate shape - ensure we have 3D tensor [batch, seq, hidden]
                if len(hidden_states.shape) != 3:
                    logger.warning(f"[Steering Hook] Unexpected shape: {hidden_states.shape}, skipping")
                    return output

                batch_size, seq_len, hidden_dim = hidden_states.shape
                input_dtype = hidden_states.dtype

                # Validate hidden_dim matches SAE's expected input dimension
                if hidden_dim != sae.d_in:
                    logger.warning(
                        f"[Steering Hook] Hidden dim mismatch: model={hidden_dim}, SAE={sae.d_in}. "
                        f"Skipping steering."
                    )
                    return output

                with torch.no_grad():
                    # Get decoder weights via the shared resolver (kept in
                    # lock-step with the cluster-allocation gain computation).
                    decoder_weight = resolve_decoder_weight(sae.model)

                    if decoder_weight is None:
                        logger.warning("Could not find decoder weights, skipping steering")
                        return output

                    # Compute total steering vector for all features
                    # Using direct steering method: activations += steering_coefficient * steering_vector
                    # This applies steering uniformly to ALL tokens, regardless of feature activation.
                    # Benefits:
                    # - Works even for sparse features that don't activate on the prompt
                    # - Consistent results regardless of activation values
                    total_steering_vector = torch.zeros(hidden_dim, device=hidden_states.device, dtype=input_dtype)

                    # Get SAE dimension for validation
                    sae_dim = decoder_weight.shape[1]  # Number of features in SAE

                    for config in feature_configs:
                        feat_idx = config.feature_idx

                        # CRITICAL: Validate feature index is within SAE bounds
                        if feat_idx >= sae_dim:
                            logger.error(
                                f"[Steering Hook] Feature index {feat_idx} is out of bounds! "
                                f"SAE only has {sae_dim} features (valid indices: 0-{sae_dim-1}). "
                                f"Skipping this feature."
                            )
                            continue

                        # Steering coefficient: multiplier - 1 (so multiplier=1 means no change)
                        steering_coefficient = config.multiplier - 1.0

                        if steering_coefficient == 0:
                            continue  # No change needed

                        # Get the steering vector (decoder direction for this feature)
                        # CRITICAL: Move to hidden_states device/dtype for proper accumulation
                        steering_vector = decoder_weight[:, feat_idx].to(
                            device=hidden_states.device,
                            dtype=input_dtype
                        )  # [d_in]

                        # Accumulate: steering_coefficient * steering_vector
                        total_steering_vector.add_(steering_coefficient * steering_vector)

                    # Broadcast steering vector to all tokens [batch, seq, hidden]
                    # The same steering is applied to every token position
                    delta_3d = total_steering_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

                    # Ensure delta dtype matches input dtype
                    if delta_3d.dtype != input_dtype:
                        delta_3d = delta_3d.to(input_dtype)

                    # Debug: Log steering magnitude on first few calls
                    delta_norm = total_steering_vector.norm().item()
                    if delta_norm > 0:
                        logger.debug(
                            f"[Steering Hook] Applying delta: norm={delta_norm:.4f}, "
                            f"shape={hidden_states.shape}, device={hidden_states.device}"
                        )

                    # CRITICAL: Apply steering delta IN-PLACE
                    # This preserves internal tensor references required by some models (e.g., Gemma-2)
                    hidden_states.add_(delta_3d)

                # Return the ORIGINAL output tuple (hidden_states was modified in place)
                return output

            except Exception as e:
                logger.error(f"[Steering Hook] Error in hook: {e}", exc_info=True)
                # Return original output on error
                return output

        return steering_hook

    def _register_steering_hooks(
        self,
        model: PreTrainedModel,
        sae,
        feature_configs: List[FeatureSteeringConfig],
        *,
        default_sae_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Register steering hooks on the model.

        Supports multi-layer / multi-SAE steering: each feature is steered
        through the SAE trained on ITS OWN layer (Feature 015). Features are
        grouped by ``(sae_id, layer)`` and one hook is registered per group,
        created with THAT group's SAE — so a feature placed on layer L is always
        steered with the SAE whose ``.layer == L`` (no wrong-basis steering).

        Regression guarantee: when ``sae`` is a single ``LoadedSAE`` (every
        solo/compare/single-SAE-combined caller), the grouping collapses to the
        prior group-by-layer and each hook receives that one SAE — byte-identical
        to the pre-015 behaviour.

        Args:
            model: The transformer model.
            sae: EITHER a single ``LoadedSAE`` (legacy single-SAE path) OR a
                ``{sae_id -> LoadedSAE}`` map (Feature 015 multi-SAE path).
            feature_configs: All feature steering configurations. Each config's
                ``sae_id`` (falling back to ``default_sae_id``) selects its SAE
                from the map.
            default_sae_id: The request-level SAE id used when a config carries
                no ``sae_id`` (ignored in the single-SAE overload).

        Returns:
            List of hook handles for cleanup.
        """
        single_sae = sae if isinstance(sae, LoadedSAE) else None
        sae_map: Optional[Dict[str, LoadedSAE]] = None if single_sae is not None else sae

        # Group features by (sae_id, layer). In the single-SAE overload the
        # sae_id is a constant so this is exactly the prior group-by-layer.
        groups: Dict[Tuple[Optional[str], int], List[FeatureSteeringConfig]] = {}
        for config in feature_configs:
            sid = config.sae_id or default_sae_id
            key = (sid, config.layer)
            groups.setdefault(key, []).append(config)

        handles = []

        for (sid, layer), layer_features in groups.items():
            # Resolve the SAE for THIS group's layer.
            if single_sae is not None:
                group_sae = single_sae
            else:
                group_sae = sae_map.get(sid)
                if group_sae is None:
                    logger.warning(
                        f"[Steering] No SAE for id {sid} (layer {layer}), skipping group"
                    )
                    continue

            # Get target module
            target_module = self._get_target_module(model, layer)

            if target_module is None:
                logger.warning(f"Could not find layer {layer} in model, skipping")
                continue

            # Create and register hook with the SAE whose layer matches the group.
            hook_fn = self._create_steering_hook(group_sae, layer_features)
            handle = target_module.register_forward_hook(hook_fn)
            handles.append(handle)

            logger.info(
                f"[Steering] Registered hook on layer {layer} (sae={sid}), "
                f"module type: {type(target_module).__name__}, "
                f"features: {[f.feature_idx for f in layer_features]}"
            )

        return handles

    def _needs_cache_disabled(self, model: PreTrainedModel) -> bool:
        """
        Whether this model's KV cache must be disabled when steering hooks are active.

        Steering hooks add a constant vector to each token's residual stream
        independently of other tokens, so a standard (growing) KV cache yields
        output identical to an uncached forward. Only models with a hook-hostile
        cache — Gemma-2's hybrid sliding-window cache is the known case — need the
        cache turned off. Detected by model_type / architecture name so the list
        stays architecture-agnostic and easy to extend.
        """
        config = getattr(model, "config", None)
        if config is None:
            # No config to reason about — be safe and disable.
            return True
        model_type = (getattr(config, "model_type", "") or "").lower()
        arch_names = " ".join(getattr(config, "architectures", []) or []).lower()
        haystack = f"{model_type} {arch_names}"
        return any(marker in haystack for marker in _CACHE_INCOMPATIBLE_MARKERS)

    async def _generate_text(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        params: GenerationParams,
        advanced_params: Optional[AdvancedGenerationParams] = None,
        disable_cache: bool = False,
    ) -> Tuple[str, int, int]:
        """
        Generate text using the model.

        CRITICAL: This method resets ALL model state before generation to ensure
        prior prompts have NO influence on the output. Each call is guaranteed
        to start with a clean slate.

        Args:
            model: The transformer model
            tokenizer: The tokenizer
            prompt: Input prompt
            params: Generation parameters
            advanced_params: Optional advanced generation parameters
            disable_cache: If True, disable KV cache (needed for some models with hooks)

        Returns:
            Tuple of (generated_text, token_count, generation_time_ms)
        """
        start_time = time.time()

        # CRITICAL: Reset ALL model state BEFORE generation
        # This ensures prior prompts have absolutely NO influence on this generation
        self._reset_model_state(model)

        # Tokenize - fresh tokenization of ONLY the current prompt
        #
        # MIS-E2E-068(2): the window comes from the MODEL, not a constant.
        #
        # This was `max_length=2048 - params.max_new_tokens`, which ignores the
        # real context window and silently truncates. Two ways it bit: on a
        # model with a larger window it discarded prompt the model could have
        # read, and at the schema-allowed `max_new_tokens=2048` the budget went
        # to zero or negative — so the prompt was cut to nothing and the model
        # generated from an empty context, with no error anywhere.
        #
        # A prompt that genuinely does not fit is now REFUSED rather than
        # quietly shortened: a truncated prompt produces a confident answer to
        # a question the user did not ask.
        # `model` is the function's own parameter. I first wrote `self._model`,
        # which does not exist on this service — models live in the
        # `_loaded_models` cache — so every generation raised
        # `'SteeringService' object has no attribute '_model'` and steering
        # failed outright. Caught in production, not by the suite.
        context_window = getattr(getattr(model, "config", None),
                                 "max_position_embeddings", None) \
            or getattr(tokenizer, "model_max_length", None) or 2048
        # Some tokenizers use a sentinel for "no limit".
        if context_window > 1_000_000:
            context_window = getattr(getattr(model, "config", None),
                                     "max_position_embeddings", 2048)

        prompt_budget = context_window - params.max_new_tokens
        if prompt_budget <= 0:
            raise ValueError(
                f"max_new_tokens={params.max_new_tokens} leaves no room for a "
                f"prompt in this model's {context_window}-token context window. "
                f"Reduce max_new_tokens to at most {context_window - 1}."
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
        if inputs["input_ids"].shape[-1] > prompt_budget:
            raise ValueError(
                f"Prompt is {inputs['input_ids'].shape[-1]} tokens but only "
                f"{prompt_budget} fit alongside max_new_tokens="
                f"{params.max_new_tokens} in this model's {context_window}-token "
                f"context window. Shorten the prompt or lower max_new_tokens."
            )

        # Build generation config with sensible defaults
        gen_kwargs = {
            "max_new_tokens": params.max_new_tokens,
            "do_sample": True,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k if params.top_k > 0 else None,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.15,  # Default to prevent degenerate repetition
            # CRITICAL: Explicitly pass None to ensure no stale KV cache is used
            "past_key_values": None,
        }

        # Disable the KV cache ONLY for architectures whose cache is actually
        # incompatible with forward hooks. Steering hooks add a fixed vector to
        # every token's hidden state independently, so for a standard KV cache the
        # cached (incremental) forward produces token-identical output to the
        # uncached one — verified on LFM2 (match_ids=True, ~14x faster). The
        # blanket disable was a Gemma-2 workaround: its hybrid sliding-window
        # cache breaks under hooks. Keep it disabled only there; every other
        # model keeps the cache and generates ~10-15x faster. See
        # _needs_cache_disabled().
        if disable_cache and self._needs_cache_disabled(model):
            gen_kwargs["use_cache"] = False
            logger.debug(
                "KV cache disabled for generation (hook-incompatible cache: "
                f"{getattr(getattr(model, 'config', None), 'model_type', '?')})"
            )

        if params.seed is not None:
            torch.manual_seed(params.seed)

        # Override with advanced params if provided
        if advanced_params:
            gen_kwargs["repetition_penalty"] = advanced_params.repetition_penalty
            gen_kwargs["do_sample"] = advanced_params.do_sample

            if advanced_params.stop_sequences:
                # Convert stop sequences to token IDs
                stop_ids = [
                    tokenizer.encode(seq, add_special_tokens=False)
                    for seq in advanced_params.stop_sequences
                ]
                # Use first token of each stop sequence as eos
                additional_eos = [ids[0] for ids in stop_ids if ids]
                if additional_eos:
                    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + additional_eos

        # Generate with clean state
        # Use watchdog to monitor for hung generation
        watchdog = get_generation_watchdog()
        watchdog.start_generation()

        try:
            # CRITICAL: Synchronize CUDA before generation to ensure clean state
            # This prevents hung operations from prior calls affecting this one
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            # CRITICAL: Synchronize after generation to ensure completion
            # This prevents async CUDA ops from causing issues in subsequent calls
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            # Always mark generation as complete (even on error)
            watchdog.end_generation()

        # Decode (only new tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        generation_time_ms = int((time.time() - start_time) * 1000)
        token_count = len(generated_ids)

        # A model that emits EOS immediately yields token_count>0 and, after
        # skip_special_tokens, an EMPTY string. Every layer above reports that
        # as a normal success, which is worse than an error during CALIBRATION:
        # a strength that collapses the model into instant-EOS looks like a
        # clean run, and a sweep reads it as "this strength is fine".
        # Not an exception — an empty generation is real model behaviour, and
        # raising here would abort a sweep partway through.
        if not generated_text.strip():
            logger.warning(
                "Steered generation produced NO text (%d token(s), all "
                "special/whitespace) in %dms. Treat any metric derived from "
                "this sample as invalid — the usual cause is a steering "
                "strength high enough to collapse the output distribution.",
                token_count, generation_time_ms,
            )

        # CRITICAL: Reset state again after generation to ensure clean slate for next call
        self._reset_model_state(model)

        return generated_text, token_count, generation_time_ms

    async def _calculate_perplexity(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        text: str,
    ) -> float:
        """
        Calculate perplexity of generated text.

        Lower perplexity indicates more fluent/likely text.

        Args:
            model: The transformer model
            tokenizer: The tokenizer
            text: Text to evaluate

        Returns:
            Perplexity score
        """
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        return math.exp(loss.item())

    async def _calculate_coherence(
        self,
        prompt: str,
        generated: str,
    ) -> Optional[float]:
        """
        Calculate semantic coherence between prompt and generation.

        Uses sentence embeddings to measure how topically related
        the generation is to the prompt.

        Args:
            prompt: Original prompt
            generated: Generated text

        Returns:
            Coherence score (0-1, higher = more coherent), or None when the
            embedding model is unavailable. NEVER a placeholder constant.
        """
        # MIS-E2E-063. This returned the CONSTANT 0.5 when the embedding model
        # was unavailable — and `sentence-transformers` is in neither
        # requirements.txt nor the venv, so the import always failed and every
        # coherence score this product has ever displayed was 0.5.
        #
        # The UI renders it as a measured quality score beside real generated
        # text, so a user comparing steering strengths sees 0.5 at every dial
        # and reads it as "coherence is unaffected by strength" — a finding
        # about the model, manufactured by a missing dependency. A constant must
        # never occupy a field the user reads as a measurement; "not measured"
        # is the honest value and both the schema (`Optional[float]`) and the
        # frontend type (`number | null`) already carry it.
        #
        # `except Exception`, not `except ImportError`: the model downloads on
        # first use, and this deployment is offline — so the *normal* failure
        # here was never an ImportError at all, and it aborted the whole
        # steering request instead of degrading.
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._sentence_model.to(self._device)
            except Exception as exc:
                logger.warning(
                    "Coherence not measured — embedding model unavailable (%s: %s). "
                    "Install sentence-transformers to enable it.",
                    type(exc).__name__, exc,
                )
                return None

        with torch.no_grad():
            embeddings = self._sentence_model.encode(
                [prompt, generated],
                convert_to_tensor=True,
                device=self._device,
            )

            # Cosine similarity
            similarity = F.cosine_similarity(
                embeddings[0].unsqueeze(0),
                embeddings[1].unsqueeze(0),
            ).item()

        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        return (similarity + 1) / 2

    async def _calculate_behavioral_score(
        self,
        steered_text: str,
        unsteered_text: str,
        feature_labels: List[str],
    ) -> Optional[float]:
        """
        Calculate behavioral score measuring steering effectiveness.

        Higher score indicates the steering had a noticeable effect
        on the generation while maintaining coherence.

        Args:
            steered_text: Text generated with steering
            unsteered_text: Baseline text without steering
            feature_labels: Labels of steered features for context

        Returns:
            Behavioral score (0-1), or None when it could not be measured.
        """
        # MIS-E2E-063, same field, same rule. This one is worse in kind: the
        # score is meant to say whether steering had an effect, so a constant
        # here reports "steering works, moderately" whatever happened.
        if self._sentence_model is None:
            logger.warning("Behavioral score not measured — embedding model unavailable")
            return None

        with torch.no_grad():
            embeddings = self._sentence_model.encode(
                [steered_text, unsteered_text],
                convert_to_tensor=True,
                device=self._device,
            )

            # Measure difference from baseline
            similarity = F.cosine_similarity(
                embeddings[0].unsqueeze(0),
                embeddings[1].unsqueeze(0),
            ).item()

        # Behavioral score: how different is steered from unsteered?
        # We want some difference (indicating steering worked) but not too much
        # (indicating it didn't break the generation)
        difference = 1 - similarity

        # Optimal difference is around 0.3-0.5
        # Score peaks around 0.4 difference
        optimal_diff = 0.4
        score = 1 - abs(difference - optimal_diff) / optimal_diff

        return max(0, min(1, score))

    async def _compute_metrics(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        generated_text: str,
        token_count: int,
        generation_time_ms: int,
        unsteered_text: Optional[str] = None,
        feature_labels: Optional[List[str]] = None,
    ) -> GenerationMetrics:
        """
        Compute all metrics for a generation.

        Args:
            model: The transformer model
            tokenizer: The tokenizer
            prompt: Original prompt
            generated_text: Generated text
            token_count: Number of tokens generated
            generation_time_ms: Generation time in milliseconds
            unsteered_text: Optional baseline for behavioral score
            feature_labels: Optional feature labels for behavioral analysis

        Returns:
            GenerationMetrics instance
        """
        # Calculate perplexity
        perplexity = await self._calculate_perplexity(
            model, tokenizer, prompt + " " + generated_text
        )

        # Calculate coherence
        coherence = await self._calculate_coherence(prompt, generated_text)

        # Calculate behavioral score if we have baseline
        behavioral_score = None
        if unsteered_text is not None and feature_labels:
            behavioral_score = await self._calculate_behavioral_score(
                generated_text, unsteered_text, feature_labels
            )

        return GenerationMetrics(
            perplexity=perplexity,
            coherence=coherence,
            behavioral_score=behavioral_score,
            token_count=token_count,
            generation_time_ms=generation_time_ms,
        )

    async def generate_comparison(
        self,
        request: SteeringComparisonRequest,
        sae_path: Path,
        model_id: str,
        model_path: Optional[str] = None,
        # SAE metadata from database for fallback
        sae_layer: Optional[int] = None,
        sae_d_model: Optional[int] = None,
        sae_n_features: Optional[int] = None,
        sae_architecture: Optional[str] = None,
        # MIS-E2E-064: every referenced SAE, so each feature is decoded through
        # the dictionary trained on ITS layer. None ⇒ single-SAE flow.
        sae_meta_map: Optional[Dict[str, "SaeMeta"]] = None,
        # Progress callback for Celery tasks
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> SteeringComparisonResponse:
        """
        Generate a steering comparison with steered and unsteered outputs.

        Args:
            request: Steering comparison request
            sae_path: Path to the SAE directory
            model_id: Model identifier
            model_path: Optional local model path
            sae_layer: SAE target layer from database
            sae_d_model: Model hidden dimension from database
            sae_n_features: Number of SAE features from database
            sae_architecture: SAE architecture type from database

        Returns:
            SteeringComparisonResponse with all outputs and metrics
        """
        start_time = time.time()
        comparison_id = f"cmp_{uuid4().hex[:12]}"
        model = None  # Track for cleanup

        # Helper for progress emission
        def emit_progress(percent: int, message: str):
            if progress_callback:
                progress_callback(percent, message)

        try:
            # Load SAE and model
            emit_progress(5, "Loading SAE...")
            # CRITICAL: Force reload SAE to avoid corruption from cached state
            sae = await self.load_sae(
                sae_path,
                request.sae_id,
                force_reload=True,
                layer=sae_layer,
                d_model=sae_d_model,
                n_features=sae_n_features,
                architecture=sae_architecture,
            )

            # MIS-E2E-064. Compare placed its hook at `feature.layer` but always
            # steered through the REQUEST-level SAE, discarding each feature's
            # own `sae_id`. Because `d_model` is uniform across layers the
            # `hidden_dim != sae.d_in` shape guard never fires, so a feature from
            # layer 20's dictionary was decoded through layer 12's SAE and
            # applied at layer 20 — right shape, wrong basis, no error. The
            # combined path was fixed for Feature 015; this one was not.
            steering_saes = {request.sae_id: sae}
            if sae_meta_map:
                steering_saes = await self.resolve_sae_map(
                    request, sae_meta_map, force_reload=True,
                )
                steering_saes.setdefault(request.sae_id, sae)

            emit_progress(15, "Loading model...")
            # CRITICAL: Always force reload the model to avoid corruption from cached state
            # With --pool=solo, the worker doesn't restart between tasks, so cached models
            # can have corrupted state (hooks, KV cache, internal buffers) from prior tasks
            model, tokenizer = await self.load_model(model_id, model_path, force_reload=True)

            # CRITICAL: Clear any stale hooks from previous requests that may have timed out
            # This ensures unsteered baseline is truly unsteered and not contaminated
            # by steering hooks from a previous request that didn't clean up properly
            self._clear_all_model_hooks(model)

            # CRITICAL: Reset all model state to ensure NO context from prior prompts
            # This guarantees each generation starts with a completely clean slate
            self._reset_model_state(model)

            # Use all selected features - duplicates with different strengths are intentional
            # (e.g., same feature at +50 and -50 for A/B comparison)
            # Inject comparison_id into each feature for tracking which job they belong to
            unique_features = [
                SelectedFeature(
                    instance_id=f.instance_id,
                    comparison_id=comparison_id,
                    feature_idx=f.feature_idx,
                    layer=f.layer,
                    strength=f.strength,
                    additional_strengths=f.additional_strengths,
                    label=f.label,
                    color=f.color,
                )
                for f in request.selected_features
            ]

            feature_configs = [
                FeatureSteeringConfig(
                    feature_idx=f.feature_idx,
                    layer=f.layer,
                    strength=f.strength,
                    label=f.label,
                    color=f.color,
                )
                for f in unique_features
            ]

            # Generate unsteered baseline
            unsteered_output = None
            unsteered_text = None

            if request.include_unsteered:
                emit_progress(20, "Generating unsteered baseline...")
                # Disable KV cache for consistency with steered generation
                # Some models (e.g., Gemma-2) behave differently with/without cache
                text, token_count, gen_time = await self._generate_text(
                    model, tokenizer, request.prompt,
                    request.generation_params,
                    request.advanced_params,
                    disable_cache=True,
                )
                unsteered_text = text

                metrics = None
                if request.compute_metrics:
                    metrics = await self._compute_metrics(
                        model, tokenizer, request.prompt,
                        text, token_count, gen_time,
                    )

                unsteered_output = UnsteeredOutput(
                    text=text,
                    metrics=metrics,
                )

            # Check if any feature has additional_strengths (multi-strength mode)
            has_multi_strength = any(
                f.additional_strengths and len(f.additional_strengths) > 0
                for f in unique_features
            )

            if has_multi_strength:
                emit_progress(25, "Generating multi-strength outputs...")
                # Multi-strength mode: generate at multiple strengths per feature
                steered_multi_outputs = await self._generate_multi_strength_outputs(
                    model=model,
                    tokenizer=tokenizer,
                    sae=sae,
                    request=request,
                    unique_features=unique_features,
                    unsteered_text=unsteered_text,
                    progress_callback=progress_callback,
                )

                # Build metrics summary for multi-strength mode
                metrics_summary = None
                if request.compute_metrics and steered_multi_outputs:
                    first_result = steered_multi_outputs[0].primary_result
                    metrics_summary = {
                        "steered_perplexity": first_result.metrics.perplexity if first_result.metrics else None,
                        "unsteered_perplexity": unsteered_output.metrics.perplexity if unsteered_output and unsteered_output.metrics else None,
                        "coherence": first_result.metrics.coherence if first_result.metrics else None,
                        "behavioral_score": first_result.metrics.behavioral_score if first_result.metrics else None,
                    }

                total_time_ms = int((time.time() - start_time) * 1000)

                return SteeringComparisonResponse(
                    comparison_id=comparison_id,
                    sae_id=request.sae_id,
                    model_id=model_id,
                    prompt=request.prompt,
                    unsteered=unsteered_output,
                    steered=[],  # Empty for multi-strength mode
                    steered_multi=steered_multi_outputs,
                    metrics_summary=metrics_summary,
                    total_time_ms=total_time_ms,
                    created_at=utc_now(),
                )

            # Single-strength mode: one output per feature (existing behavior)
            emit_progress(25, "Generating steered outputs...")
            steered_outputs = []
            total_features = len(unique_features)

            for feature_idx, feature in enumerate(unique_features):
                # Calculate progress: 25% to 90% range for generation loop
                loop_progress = 25 + int((feature_idx / total_features) * 65)
                emit_progress(loop_progress, f"Generating feature {feature_idx + 1}/{total_features}...")
                # CRITICAL: Clear stale hooks BEFORE registering new ones
                # This prevents hook accumulation from previous iterations
                self._clear_all_model_hooks(model)

                # CRITICAL: Reset model state BEFORE each feature iteration
                # This guarantees NO context from prior prompt/feature affects this generation
                self._reset_model_state(model)

                # Create config for just this feature
                single_feature_config = [
                    FeatureSteeringConfig(
                        feature_idx=feature.feature_idx,
                        layer=feature.layer,
                        strength=feature.strength,
                        label=feature.label,
                        # MIS-E2E-064: carry the feature's OWN SAE. Dropping it
                        # here is what routed every feature through the
                        # request-level dictionary.
                        sae_id=getattr(feature, "sae_id", None),
                    )
                ]

                # Register steering hooks for this single feature (now on clean
                # model). Passing the MAP — not a single SAE — lets
                # `_register_steering_hooks` group by (sae_id, layer) and decode
                # each feature in its own basis; it already did so for combined.
                handles = self._register_steering_hooks(
                    model, steering_saes, single_feature_config,
                    default_sae_id=request.sae_id,
                )

                try:
                    # Generate with steering for this feature
                    # Disable KV cache because some models (e.g., Gemma-2 with hybrid cache)
                    # are incompatible with forward hooks when caching is enabled
                    text, token_count, gen_time = await self._generate_text(
                        model, tokenizer, request.prompt,
                        request.generation_params,
                        request.advanced_params,
                        disable_cache=True,
                    )

                    metrics = None
                    if request.compute_metrics:
                        feature_label = feature.label or f"Feature {feature.feature_idx}"
                        metrics = await self._compute_metrics(
                            model, tokenizer, request.prompt,
                            text, token_count, gen_time,
                            unsteered_text=unsteered_text,
                            feature_labels=[feature_label],
                        )

                    steered_outputs.append(SteeredOutput(
                        text=text,
                        feature_config=feature,
                        metrics=metrics,
                    ))

                finally:
                    # Clean up hooks
                    for handle in handles:
                        handle.remove()

                    # CRITICAL: Reset state after generation completes or errors
                    # This ensures clean slate for next iteration even if error occurred
                    self._reset_model_state(model)

                    # Clear GPU cache between features to prevent memory fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                logger.debug(
                    f"[Single-Strength] Completed feature {feature_idx + 1}/{len(unique_features)}"
                )

            # Build metrics summary
            metrics_summary = None
            if request.compute_metrics and steered_outputs:
                metrics_summary = {
                    "steered_perplexity": steered_outputs[0].metrics.perplexity if steered_outputs[0].metrics else None,
                    "unsteered_perplexity": unsteered_output.metrics.perplexity if unsteered_output and unsteered_output.metrics else None,
                    "coherence": steered_outputs[0].metrics.coherence if steered_outputs[0].metrics else None,
                    "behavioral_score": steered_outputs[0].metrics.behavioral_score if steered_outputs[0].metrics else None,
                }

            total_time_ms = int((time.time() - start_time) * 1000)

            return SteeringComparisonResponse(
                comparison_id=comparison_id,
                sae_id=request.sae_id,
                model_id=model_id,
                prompt=request.prompt,
                unsteered=unsteered_output,
                steered=steered_outputs,
                steered_multi=None,  # Not in multi-strength mode
                metrics_summary=metrics_summary,
                total_time_ms=total_time_ms,
                created_at=utc_now(),
            )

        except Exception as e:
            logger.error(f"[Steering] Error during generate_comparison: {e}")
            raise

        finally:
            # CRITICAL: Always clean up GPU memory, even on error
            # This prevents zombie processes holding GPU memory
            if model is not None:
                self._clear_all_model_hooks(model)
            self.cleanup_gpu(model)

            # CRITICAL: Ensure model stays on GPU for subsequent requests
            # Memory pressure during generation can sometimes cause drift to CPU
            if model is not None:
                self._ensure_model_on_gpu(model)

            logger.info("[Steering] GPU cleanup completed")

    async def _generate_multi_strength_outputs(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sae: LoadedSAE,
        request: SteeringComparisonRequest,
        unique_features: List[SelectedFeature],
        unsteered_text: Optional[str],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[SteeredOutputMulti]:
        """
        Generate outputs for each feature at multiple strength values.

        For each feature that has additional_strengths, generates text at:
        - The primary strength
        - Each additional strength

        CRITICAL: This method ensures complete context isolation between:
        - Different features (outer loop)
        - Different strengths within a feature (inner loop)

        Each generation starts with:
        1. All stale hooks cleared
        2. Model state fully reset (KV cache, internal buffers)
        3. Fresh hook registration
        4. GPU cache cleared

        Args:
            model: The transformer model
            tokenizer: The tokenizer
            sae: Loaded SAE model
            request: Original steering request
            unique_features: List of features (may include same feature_idx/layer with different strengths)
            unsteered_text: Baseline text for metrics comparison

        Returns:
            List of SteeredOutputMulti, one per feature with multi-strength results
        """
        # Helper for progress emission
        def emit_progress(percent: int, message: str):
            if progress_callback:
                progress_callback(percent, message)

        results = []
        total_features = len(unique_features)
        # Calculate total generations for progress tracking
        total_generations = sum(
            1 + len(f.additional_strengths or [])
            for f in unique_features
        )
        generation_count = 0

        for feature_idx, feature in enumerate(unique_features):
            # CRITICAL: Reset model state at START of each feature iteration
            # This ensures no state leakage from previous feature's generations
            self._reset_model_state(model)
            self._clear_all_model_hooks(model)

            # Collect all strengths to test for this feature
            all_strengths = [feature.strength]  # Primary first
            if feature.additional_strengths:
                all_strengths.extend(feature.additional_strengths)

            # Sort strengths for consistent ordering in results
            all_strengths = sorted(all_strengths)

            logger.info(
                f"[Multi-Strength] Generating for feature {feature.feature_idx} "
                f"at strengths: {all_strengths}"
            )

            # Generate for each strength
            strength_results: List[MultiStrengthResult] = []

            for strength_idx, strength in enumerate(all_strengths):
                # Calculate progress: 25% to 90% range
                generation_count += 1
                loop_progress = 25 + int((generation_count / total_generations) * 65)
                emit_progress(
                    loop_progress,
                    f"Feature {feature_idx + 1}/{total_features} @ strength {strength:.1f}"
                )

                # CRITICAL: Clear any stale hooks BEFORE registering new ones
                # This prevents hook accumulation from previous iterations
                self._clear_all_model_hooks(model)

                # CRITICAL: Reset model state BEFORE each strength iteration
                # This guarantees NO context from prior prompt/strength affects this generation
                self._reset_model_state(model)

                # Create feature config with this strength
                single_feature_config = [
                    FeatureSteeringConfig(
                        feature_idx=feature.feature_idx,
                        layer=feature.layer,
                        strength=strength,
                        label=feature.label,
                        color=feature.color,
                    )
                ]

                # Register steering hooks (now on clean model)
                handles = self._register_steering_hooks(model, sae, single_feature_config)

                try:
                    # Generate with this strength
                    text, token_count, gen_time = await self._generate_text(
                        model, tokenizer, request.prompt,
                        request.generation_params,
                        request.advanced_params,
                        disable_cache=True,
                    )

                    metrics = None
                    if request.compute_metrics:
                        feature_label = feature.label or f"Feature {feature.feature_idx}"
                        metrics = await self._compute_metrics(
                            model, tokenizer, request.prompt,
                            text, token_count, gen_time,
                            unsteered_text=unsteered_text,
                            feature_labels=[feature_label],
                        )

                    strength_results.append(MultiStrengthResult(
                        strength=strength,
                        text=text,
                        metrics=metrics,
                    ))

                finally:
                    # Clean up hooks
                    for handle in handles:
                        handle.remove()

                    # CRITICAL: Reset state after generation completes or errors
                    # This ensures clean slate for next iteration even if error occurred
                    self._reset_model_state(model)

                    # Clear GPU cache between iterations to prevent memory fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                logger.debug(
                    f"[Multi-Strength] Completed strength {strength_idx + 1}/{len(all_strengths)} "
                    f"for feature {feature.feature_idx}"
                )

            # Find primary result (matches original strength)
            primary_idx = all_strengths.index(feature.strength)
            primary_result = strength_results[primary_idx]
            additional_results = [r for i, r in enumerate(strength_results) if i != primary_idx]

            results.append(SteeredOutputMulti(
                feature_config=feature,
                primary_result=primary_result,
                additional_results=additional_results,
            ))

            # Clear GPU cache between features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    async def generate_strength_sweep(
        self,
        request: SteeringStrengthSweepRequest,
        sae_path: Path,
        model_id: str,
        model_path: Optional[str] = None,
        # SAE metadata from database for fallback
        sae_layer: Optional[int] = None,
        sae_d_model: Optional[int] = None,
        sae_n_features: Optional[int] = None,
        sae_architecture: Optional[str] = None,
        # Progress callback for Celery tasks
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> StrengthSweepResponse:
        """
        Generate a strength sweep testing multiple steering strengths.

        Args:
            request: Strength sweep request
            sae_path: Path to the SAE directory
            model_id: Model identifier
            model_path: Optional local model path
            sae_layer: SAE target layer from database
            sae_d_model: Model hidden dimension from database
            sae_n_features: Number of SAE features from database
            sae_architecture: SAE architecture type from database

        Returns:
            StrengthSweepResponse with results for each strength
        """
        start_time = time.time()
        sweep_id = f"sweep_{uuid4().hex[:12]}"
        model = None  # Track for cleanup

        # Helper for progress emission
        def emit_progress(percent: int, message: str):
            if progress_callback:
                progress_callback(percent, message)

        try:
            # Load SAE and model
            emit_progress(5, "Loading SAE...")
            # CRITICAL: Force reload SAE to avoid corruption from cached state
            sae = await self.load_sae(
                sae_path,
                request.sae_id,
                force_reload=True,
                layer=sae_layer,
                d_model=sae_d_model,
                n_features=sae_n_features,
                architecture=sae_architecture,
            )
            emit_progress(15, "Loading model...")
            # CRITICAL: Always force reload the model to avoid corruption from cached state
            model, tokenizer = await self.load_model(model_id, model_path, force_reload=True)

            # CRITICAL: Clear any stale hooks from previous requests that may have timed out
            # This ensures unsteered baseline is truly unsteered
            self._clear_all_model_hooks(model)

            # CRITICAL: Reset all model state to ensure NO context from prior prompts
            self._reset_model_state(model)

            emit_progress(20, "Generating unsteered baseline...")
            # Generate unsteered baseline
            # Disable KV cache for consistency - some models behave differently with/without cache
            text, token_count, gen_time = await self._generate_text(
                model, tokenizer, request.prompt,
                request.generation_params,
                disable_cache=True,
            )

            unsteered_metrics = await self._compute_metrics(
                model, tokenizer, request.prompt,
                text, token_count, gen_time,
            )

            unsteered = UnsteeredOutput(
                text=text,
                metrics=unsteered_metrics,
            )

            # Generate for each strength value
            emit_progress(25, "Starting strength sweep...")
            results = []
            total_strengths = len(request.strength_values)

            for strength_idx, strength in enumerate(request.strength_values):
                # Calculate progress: 25% to 90% range
                loop_progress = 25 + int((strength_idx / total_strengths) * 65)
                emit_progress(loop_progress, f"Testing strength {strength_idx + 1}/{total_strengths}: {strength:.1f}")

                # CRITICAL: Clear stale hooks BEFORE registering new ones
                # This prevents hook accumulation from previous iterations
                self._clear_all_model_hooks(model)

                # CRITICAL: Reset model state BEFORE each strength iteration
                # This guarantees NO context from prior prompt/strength affects this generation
                self._reset_model_state(model)

                # Create feature config
                feature_config = FeatureSteeringConfig(
                    feature_idx=request.feature_idx,
                    layer=request.layer,
                    strength=strength,
                )

                # Register hook (now on clean model)
                handles = self._register_steering_hooks(model, sae, [feature_config])

                try:
                    # Disable KV cache for consistency with unsteered generation
                    text, token_count, gen_time = await self._generate_text(
                        model, tokenizer, request.prompt,
                        request.generation_params,
                        disable_cache=True,
                    )

                    metrics = await self._compute_metrics(
                        model, tokenizer, request.prompt,
                        text, token_count, gen_time,
                        unsteered_text=unsteered.text,
                        feature_labels=[f"Feature {request.feature_idx}"],
                    )

                    results.append(StrengthSweepResult(
                        strength=strength,
                        text=text,
                        metrics=metrics,
                    ))

                finally:
                    # Clean up hooks
                    for handle in handles:
                        handle.remove()

                    # CRITICAL: Reset state after generation completes or errors
                    # This ensures clean slate for next iteration even if error occurred
                    self._reset_model_state(model)

                    # Clear GPU cache between iterations to prevent memory fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                logger.debug(
                    f"[Strength Sweep] Completed strength {strength_idx + 1}/{len(request.strength_values)}"
                )

            total_time_ms = int((time.time() - start_time) * 1000)

            return StrengthSweepResponse(
                sweep_id=sweep_id,
                sae_id=request.sae_id,
                model_id=model_id,
                prompt=request.prompt,
                feature_idx=request.feature_idx,
                layer=request.layer,
                unsteered=unsteered,
                results=results,
                total_time_ms=total_time_ms,
                created_at=utc_now(),
            )

        except Exception as e:
            logger.error(f"[Strength Sweep] Error during generate_strength_sweep: {e}")
            raise

        finally:
            # CRITICAL: Always clean up GPU memory on all GPUs, even on error
            # This prevents zombie processes holding GPU memory
            if model is not None:
                self._clear_all_model_hooks(model)
            self.cleanup_gpu(model)

            # CRITICAL: Ensure model stays on GPU for subsequent requests
            # Memory pressure during generation can sometimes cause drift to CPU
            if model is not None:
                self._ensure_model_on_gpu(model)

            logger.info("[Strength Sweep] GPU cleanup completed")

    async def generate_combined(
        self,
        request: CombinedSteeringRequest,
        sae_path: Path,
        model_id: str,
        model_path: Optional[str] = None,
        # SAE metadata from database for fallback
        sae_layer: Optional[int] = None,
        sae_d_model: Optional[int] = None,
        sae_n_features: Optional[int] = None,
        sae_architecture: Optional[str] = None,
        # Feature 015: load metadata for EVERY distinct sae_id the request
        # references ({sae_id -> SaeMeta}). None ⇒ single-SAE flow, built from
        # the scalar sae_* params below → byte-identical to the pre-015 path.
        sae_meta_map: Optional[Dict[str, "SaeMeta"]] = None,
        # Progress callback for Celery tasks
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> CombinedSteeringResponse:
        """
        Generate text with ALL selected features applied simultaneously.

        Unlike generate_comparison which tests each feature separately, this method
        applies all features together in a single generation pass. This enables:
        - Testing synergistic effects (e.g., "formal" + "positive" = professional tone)
        - Creating complex behavioral changes with multiple influences
        - Exploring feature interactions and emergent behaviors

        The steering vectors from all features are accumulated and applied at once:
            total_steering = Σ (strength_i × decoder_direction_i)

        Args:
            request: Combined steering request with multiple features
            sae_path: Path to the SAE directory
            model_id: Model identifier
            model_path: Optional local model path
            sae_layer: SAE target layer from database
            sae_d_model: Model hidden dimension from database
            sae_n_features: Number of SAE features from database
            sae_architecture: SAE architecture type from database
            progress_callback: Optional callback for progress updates

        Returns:
            CombinedSteeringResponse with combined output and optional baseline
        """
        start_time = time.time()
        combined_id = f"cmb_{uuid4().hex[:12]}"
        model = None  # Track for cleanup

        # Helper for progress emission
        def emit_progress(percent: int, message: str):
            if progress_callback:
                progress_callback(percent, message)

        try:
            # Build the per-SAE load metadata. Absent a caller-supplied map
            # (single-SAE flow), synthesize a one-entry map from the scalar
            # params so downstream code takes the identical multi-SAE codepath
            # with N=1 — no behavioural fork.
            if sae_meta_map is None:
                sae_meta_map = {
                    request.sae_id: SaeMeta(
                        sae_id=request.sae_id,
                        sae_path=str(sae_path),
                        layer=sae_layer,
                        d_model=sae_d_model,
                        n_features=sae_n_features,
                        architecture=sae_architecture,
                    )
                }

            # Load EVERY distinct SAE the request references and validate that
            # each feature's layer matches its SAE. The request-level SAE is
            # force-reloaded (as before) to dodge cached-state corruption;
            # additional SAEs use the cache when resident.
            emit_progress(5, "Loading SAE(s)...")
            sae_map = await self.resolve_sae_map(
                request, sae_meta_map, force_reload=True,
            )
            # Preserve the historical `sae` local for the single-SAE path so the
            # rest of this method reads unchanged when only one SAE is used.
            sae = sae_map[request.sae_id]

            emit_progress(15, "Loading model...")
            # CRITICAL: Always force reload the model to avoid corruption from cached state
            model, tokenizer = await self.load_model(model_id, model_path, force_reload=True)

            # CRITICAL: Clear any stale hooks from previous requests
            self._clear_all_model_hooks(model)

            # CRITICAL: Reset all model state to ensure NO context from prior prompts
            self._reset_model_state(model)

            # Resolve each feature's SAE id ONCE (feature.sae_id ?? request-level)
            # — this is the source of truth threaded into both the hook configs
            # and the applied-summary, so features_applied[].sae_id reflects the
            # SAE actually used at hook time, never merely the request.
            resolved_sae_ids = [
                (f.sae_id or request.sae_id) for f in request.selected_features
            ]

            # Build list of applied features for response (sae_id = the config
            # used at hook time — Feature 015 source of truth).
            features_applied = [
                CombinedFeatureApplied(
                    feature_idx=f.feature_idx,
                    layer=f.layer,
                    sae_id=sid,
                    strength=f.strength,
                    label=f.label,
                    color=f.color,
                )
                for f, sid in zip(request.selected_features, resolved_sae_ids)
            ]

            # Calculate total steering strength (sum of absolute values)
            total_steering_strength = sum(abs(f.strength) for f in request.selected_features)

            # Generate unsteered baseline (if requested)
            baseline_output = None
            baseline_metrics = None
            baseline_text = None

            if request.include_baseline:
                emit_progress(25, "Generating unsteered baseline...")
                # Disable KV cache for consistency with steered generation
                text, token_count, gen_time = await self._generate_text(
                    model, tokenizer, request.prompt,
                    request.generation_params,
                    request.advanced_params,
                    disable_cache=True,
                )
                baseline_text = text
                baseline_output = text

                if request.compute_metrics:
                    baseline_metrics = await self._compute_metrics(
                        model, tokenizer, request.prompt,
                        text, token_count, gen_time,
                    )

            # Build feature configs for ALL features together
            emit_progress(50, f"Generating with {len(request.selected_features)} features combined...")

            # CRITICAL: Clear stale hooks before registering new ones
            self._clear_all_model_hooks(model)

            # CRITICAL: Reset model state before combined generation
            self._reset_model_state(model)

            # Create configs for ALL features, threading each feature's resolved
            # SAE id so _register_steering_hooks groups by (sae_id, layer) and
            # steers each feature through its OWN layer's SAE.
            all_feature_configs = [
                FeatureSteeringConfig(
                    feature_idx=f.feature_idx,
                    layer=f.layer,
                    strength=f.strength,
                    label=f.label,
                    color=f.color,
                    sae_id=sid,
                )
                for f, sid in zip(request.selected_features, resolved_sae_ids)
            ]

            # Register steering hooks for ALL features at once.
            # The existing _create_steering_hook already accumulates multiple
            # features. When exactly one SAE is referenced we pass the single
            # LoadedSAE so the hook path is byte-identical to the pre-015 flow;
            # when several are referenced we pass the whole map + the request's
            # default id so each (sae_id, layer) group steers through its own SAE.
            if len(sae_map) == 1:
                handles = self._register_steering_hooks(model, sae, all_feature_configs)
            else:
                handles = self._register_steering_hooks(
                    model, sae_map, all_feature_configs,
                    default_sae_id=request.sae_id,
                )

            try:
                # Generate with ALL features applied simultaneously
                combined_text, token_count, gen_time = await self._generate_text(
                    model, tokenizer, request.prompt,
                    request.generation_params,
                    request.advanced_params,
                    disable_cache=True,
                )

                # Compute metrics for combined output
                combined_metrics = None
                if request.compute_metrics:
                    feature_labels = [
                        f.label or f"Feature {f.feature_idx}"
                        for f in request.selected_features
                    ]
                    combined_metrics = await self._compute_metrics(
                        model, tokenizer, request.prompt,
                        combined_text, token_count, gen_time,
                        unsteered_text=baseline_text,
                        feature_labels=feature_labels,
                    )

            finally:
                # Clean up hooks
                for handle in handles:
                    handle.remove()

                # CRITICAL: Reset state after generation
                self._reset_model_state(model)

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            emit_progress(90, "Finalizing results...")

            total_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"[Combined Steering] Generated with {len(request.selected_features)} features "
                f"(total strength: {total_steering_strength:.1f}) in {total_time_ms}ms"
            )

            return CombinedSteeringResponse(
                combined_id=combined_id,
                sae_id=request.sae_id,
                model_id=model_id,
                prompt=request.prompt,
                combined_output=combined_text,
                features_applied=features_applied,
                baseline_output=baseline_output,
                combined_metrics=combined_metrics,
                baseline_metrics=baseline_metrics,
                total_steering_strength=total_steering_strength,
                total_time_ms=total_time_ms,
                created_at=utc_now(),
            )

        except Exception as e:
            logger.error(f"[Combined Steering] Error during generate_combined: {e}")
            raise

        finally:
            # CRITICAL: Always clean up GPU memory, even on error
            if model is not None:
                self._clear_all_model_hooks(model)
            self.cleanup_gpu(model)

            # CRITICAL: Ensure model stays on GPU for subsequent requests
            if model is not None:
                self._ensure_model_on_gpu(model)

            logger.info("[Combined Steering] GPU cleanup completed")

    def unload_sae(self, sae_id: str) -> bool:
        """
        Unload a cached SAE from memory and clean up GPU.

        Args:
            sae_id: SAE identifier

        Returns:
            True if unloaded, False if not found
        """
        if sae_id not in self._loaded_saes:
            logger.info(f"[Unload SAE] SAE {sae_id} not in cache")
            return False

        try:
            loaded_sae = self._loaded_saes.pop(sae_id)

            # Move SAE model to CPU first (helps with GPU memory release)
            try:
                if hasattr(loaded_sae.model, 'to'):
                    loaded_sae.model.to("cpu")
            except Exception:
                pass

            # Delete reference
            del loaded_sae

            # Clean up GPU
            self.cleanup_gpu()

            logger.info(f"[Unload SAE] SAE {sae_id} unloaded and GPU cleaned")
            return True

        except Exception as e:
            logger.error(f"[Unload SAE] Error unloading SAE {sae_id}: {e}")
            self.cleanup_gpu()
            return False

    def unload_all_saes(self) -> int:
        """
        Unload all cached SAEs and clean GPU memory.

        Returns:
            Number of SAEs unloaded
        """
        sae_ids = list(self._loaded_saes.keys())
        count = 0

        for sae_id in sae_ids:
            if self.unload_sae(sae_id):
                count += 1

        # Final cleanup
        self.cleanup_gpu()
        logger.info(f"[Unload SAE] Unloaded {count} SAEs, GPU cleaned")
        return count

    def unload_all(self) -> Dict[str, int]:
        """
        Unload all cached models and SAEs, clean all GPUs.

        Returns:
            Dictionary with counts of unloaded models and SAEs
        """
        models_unloaded = self.unload_all_models()
        saes_unloaded = self.unload_all_saes()

        # Final comprehensive cleanup on all GPUs
        self.cleanup_gpu()

        logger.info(
            f"[Unload All] Unloaded {models_unloaded} models, "
            f"{saes_unloaded} SAEs, all GPUs cleaned"
        )

        return {
            "models_unloaded": models_unloaded,
            "saes_unloaded": saes_unloaded,
        }

    def _get_system_vram_usage_gb(self) -> float:
        """
        Get system-wide VRAM usage using pynvml (same as System Monitor).

        This measures TOTAL GPU memory used across ALL processes, not just
        the current process. Falls back to torch.cuda.memory_allocated()
        if pynvml is unavailable.

        Returns:
            VRAM usage in GB
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Primary GPU
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = mem_info.used / (1024 ** 3)
            pynvml.nvmlShutdown()
            return vram_gb
        except Exception as e:
            logger.debug(f"pynvml unavailable, falling back to torch: {e}")
            # Fallback to torch (only measures current process)
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 3)
            return 0.0

    def clear_cache(self) -> dict:
        """
        Clear all cached models and SAEs and free GPU memory system-wide.

        This aggressively clears ALL GPU memory, not just what this service loaded.
        Includes clearing other services' caches and forcing full CUDA cleanup.

        Returns:
            Dict with clearing results including VRAM usage info.
        """
        import gc

        # Get system-wide VRAM usage before clearing (using pynvml like System Monitor)
        vram_before_gb = self._get_system_vram_usage_gb()

        # Log what we're clearing from steering service
        sae_count = len(self._loaded_saes)
        model_count = len(self._loaded_models)
        logger.info(f"Clearing steering cache: {sae_count} SAEs, {model_count} models")

        # Move models to CPU before clearing to help with memory release
        for model_id, (model, tokenizer) in list(self._loaded_models.items()):
            try:
                model.cpu()
                del model
                del tokenizer
            except Exception as e:
                logger.warning(f"Error moving model {model_id} to CPU: {e}")

        # Clear SAEs
        for sae_id, loaded_sae in list(self._loaded_saes.items()):
            try:
                if hasattr(loaded_sae, 'model'):
                    loaded_sae.model.cpu()
                    del loaded_sae.model
            except Exception as e:
                logger.warning(f"Error clearing SAE {sae_id}: {e}")

        # Clear the dictionaries
        self._loaded_saes.clear()
        self._loaded_models.clear()
        self._sentence_model = None

        # Count stray GPU objects (for reporting)
        stray_count = self._count_gpu_objects()

        # Force aggressive garbage collection
        for _ in range(5):
            gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()

        # Get system-wide VRAM usage after clearing
        vram_after_gb = self._get_system_vram_usage_gb()
        vram_freed = max(0, vram_before_gb - vram_after_gb)

        logger.info(f"Cache clear: freed {vram_freed:.2f} GB, {vram_after_gb:.2f} GB remaining")

        was_already_clear = model_count == 0 and sae_count == 0 and stray_count == 0

        return {
            "models_unloaded": model_count,
            "saes_unloaded": sae_count,
            "stray_objects_found": stray_count,
            "vram_before_gb": round(vram_before_gb, 2),
            "vram_after_gb": round(vram_after_gb, 2),
            "vram_freed_gb": round(vram_freed, 2),
            "was_already_clear": was_already_clear,
            "needs_restart": vram_after_gb > 1.0 and vram_freed < 0.5,
        }

    def _count_gpu_objects(self) -> int:
        """
        Count GPU objects that might be holding VRAM (for diagnostic purposes).

        Returns:
            Number of GPU objects found
        """
        import gc

        count = 0

        model_classes = {
            'PreTrainedModel', 'LlamaForCausalLM', 'Gemma2ForCausalLM',
            'GPT2LMHeadModel', 'PhiForCausalLM', 'MistralForCausalLM',
            'Qwen2ForCausalLM', 'SparseAutoencoder', 'JumpReLUSAE',
            'GemmaForCausalLM', 'Gemma2Model', 'AutoModelForCausalLM',
        }

        for obj in gc.get_objects():
            try:
                if not hasattr(obj, '__class__'):
                    continue

                class_name = obj.__class__.__name__

                if class_name in model_classes:
                    logger.info(f"Found stray model: {class_name}")
                    count += 1
                elif class_name == 'Tensor' and hasattr(obj, 'device'):
                    if obj.device.type == 'cuda':
                        count += 1

            except (ReferenceError, TypeError, RuntimeError, OSError, AttributeError):
                pass

        return count


    # =========================================================================
    # SYNCHRONOUS WRAPPERS FOR CELERY TASKS
    # =========================================================================

    def generate_comparison_sync(
        self,
        request_dict: Dict[str, Any],
        sae_path: str,
        model_id: str,
        model_path: Optional[str] = None,
        sae_layer: Optional[int] = None,
        sae_d_model: Optional[int] = None,
        sae_n_features: Optional[int] = None,
        sae_architecture: Optional[str] = None,
        # MIS-E2E-064: JSON {sae_id -> SaeMeta} from the endpoint.
        sae_meta_map: Optional[Dict[str, Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for generate_comparison.

        Used by Celery tasks where async is not needed. Converts request dict
        to Pydantic model and runs the async method synchronously.

        Args:
            request_dict: Serialized SteeringComparisonRequest as dict
            sae_path: Path to SAE weights file
            model_id: Model identifier for HuggingFace loading
            model_path: Optional local path to model weights
            sae_layer: SAE layer index
            sae_d_model: SAE model dimension
            sae_n_features: Number of SAE features
            sae_architecture: SAE architecture type
            progress_callback: Optional callback for progress updates (percent, message)

        Returns:
            Dict representation of SteeringComparisonResponse
        """
        # Convert request_dict back to Pydantic model
        request = SteeringComparisonRequest(**request_dict)

        # MIS-E2E-064: rehydrate the SaeMeta map so each feature can steer
        # through its own layer's dictionary.
        meta_map: Optional[Dict[str, SaeMeta]] = None
        if sae_meta_map:
            meta_map = {sid: SaeMeta(**meta) for sid, meta in sae_meta_map.items()}

        # Run the async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.generate_comparison(
                    request=request,
                    sae_path=Path(sae_path),
                    model_id=model_id,
                    model_path=model_path,
                    sae_layer=sae_layer,
                    sae_d_model=sae_d_model,
                    sae_n_features=sae_n_features,
                    sae_architecture=sae_architecture,
                    sae_meta_map=meta_map,
                    progress_callback=progress_callback,
                )
            )
            # Convert Pydantic model to dict for JSON serialization
            return result.model_dump(mode="json")
        finally:
            loop.close()

    def generate_strength_sweep_sync(
        self,
        request_dict: Dict[str, Any],
        sae_path: str,
        model_id: str,
        model_path: Optional[str] = None,
        sae_layer: Optional[int] = None,
        sae_d_model: Optional[int] = None,
        sae_n_features: Optional[int] = None,
        sae_architecture: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for generate_strength_sweep.

        Used by Celery tasks where async is not needed.

        Args:
            request_dict: Serialized SteeringStrengthSweepRequest as dict
            sae_path: Path to SAE weights file
            model_id: Model identifier for HuggingFace loading
            model_path: Optional local path to model weights
            sae_layer: SAE layer index
            sae_d_model: SAE model dimension
            sae_n_features: Number of SAE features
            sae_architecture: SAE architecture type
            progress_callback: Optional callback for progress updates (percent, message)

        Returns:
            Dict representation of StrengthSweepResponse
        """
        # Convert request_dict back to Pydantic model
        request = SteeringStrengthSweepRequest(**request_dict)

        # Run the async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.generate_strength_sweep(
                    request=request,
                    sae_path=Path(sae_path),
                    model_id=model_id,
                    model_path=model_path,
                    sae_layer=sae_layer,
                    sae_d_model=sae_d_model,
                    sae_n_features=sae_n_features,
                    sae_architecture=sae_architecture,
                    progress_callback=progress_callback,
                )
            )
            # Convert Pydantic model to dict for JSON serialization
            return result.model_dump(mode="json")
        finally:
            loop.close()

    def generate_combined_sync(
        self,
        request_dict: Dict[str, Any],
        sae_path: str,
        model_id: str,
        model_path: Optional[str] = None,
        sae_layer: Optional[int] = None,
        sae_d_model: Optional[int] = None,
        sae_n_features: Optional[int] = None,
        sae_architecture: Optional[str] = None,
        # Feature 015: JSON-serializable {sae_id -> SaeMeta-as-dict} for every
        # distinct SAE the request references. None ⇒ single-SAE flow.
        sae_meta_map: Optional[Dict[str, Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for generate_combined.

        Used by Celery tasks where async is not needed. Converts request dict
        to Pydantic model and runs the async method synchronously.

        Args:
            request_dict: Serialized CombinedSteeringRequest as dict
            sae_path: Path to SAE weights file
            model_id: Model identifier for HuggingFace loading
            model_path: Optional local path to model weights
            sae_layer: SAE layer index
            sae_d_model: SAE model dimension
            sae_n_features: Number of SAE features
            sae_architecture: SAE architecture type
            sae_meta_map: Feature 015 — {sae_id -> SaeMeta dict} for multi-SAE
                requests; None keeps the byte-identical single-SAE path.
            progress_callback: Optional callback for progress updates (percent, message)

        Returns:
            Dict representation of CombinedSteeringResponse
        """
        # Convert request_dict back to Pydantic model
        request = CombinedSteeringRequest(**request_dict)

        # Rehydrate the SaeMeta map (if any) from its JSON dict form.
        meta_map: Optional[Dict[str, SaeMeta]] = None
        if sae_meta_map:
            meta_map = {
                sid: SaeMeta(**meta) for sid, meta in sae_meta_map.items()
            }

        # Run the async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.generate_combined(
                    request=request,
                    sae_path=Path(sae_path),
                    model_id=model_id,
                    model_path=model_path,
                    sae_layer=sae_layer,
                    sae_d_model=sae_d_model,
                    sae_n_features=sae_n_features,
                    sae_architecture=sae_architecture,
                    sae_meta_map=meta_map,
                    progress_callback=progress_callback,
                )
            )
            # Convert Pydantic model to dict for JSON serialization
            return result.model_dump(mode="json")
        finally:
            loop.close()


# Global service instance
_steering_service: Optional[SteeringService] = None


def get_steering_service() -> SteeringService:
    """Get or create the global steering service instance."""
    global _steering_service
    if _steering_service is None:
        _steering_service = SteeringService()
    return _steering_service
