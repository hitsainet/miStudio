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
import signal
import sys
import time
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
)

logger = logging.getLogger(__name__)


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
        except:
            pass  # If logging fails, silently ignore


def _signal_handler(signum, frame):
    """
    Signal handler for SIGTERM and SIGINT.

    Runs emergency GPU cleanup before allowing the process to terminate.
    """
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
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


@dataclass
class FeatureSteeringConfig:
    """Configuration for a single feature's steering."""

    feature_idx: int
    layer: int
    strength: float  # Raw steering coefficient (matches Neuronpedia)
    label: Optional[str] = None
    color: str = "teal"

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
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )

        if self._device != "cuda":
            model.to(self._device)

        model.eval()

        self._loaded_models[cache_key] = (model, tokenizer)
        logger.info(f"Loaded model {model_id}")

        return model, tokenizer

    def _clear_all_model_hooks(self, model: PreTrainedModel) -> int:
        """
        Clear all forward hooks from transformer layer modules.

        This prevents stale hooks from previous requests that may have timed out
        or failed from contaminating subsequent generations. Critical for ensuring
        unsteered baselines are truly unsteered.

        Args:
            model: The transformer model to clear hooks from

        Returns:
            Number of hooks cleared
        """
        hooks_cleared = 0

        # Get all transformer layer modules
        layers_module = None

        # Try different model architectures
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers_module = model.transformer.h
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers_module = model.model.layers
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            layers_module = model.gpt_neox.layers
        elif hasattr(model, "layers"):
            layers_module = model.layers

        if layers_module is None:
            logger.warning("Could not find transformer layers to clear hooks from")
            return 0

        # Clear forward hooks from each layer
        for layer_idx, layer in enumerate(layers_module):
            # Clear hooks on the layer module itself
            if hasattr(layer, "_forward_hooks") and layer._forward_hooks:
                count = len(layer._forward_hooks)
                layer._forward_hooks.clear()
                hooks_cleared += count

            # Also check common submodules that might have hooks
            for submodule_name in ["self_attn", "attn", "mlp", "feed_forward",
                                    "post_attention_layernorm", "ln_2"]:
                if hasattr(layer, submodule_name):
                    submodule = getattr(layer, submodule_name)
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

    def _get_target_module(
        self,
        model: PreTrainedModel,
        layer: int,
        hook_type: str = "resid_post",
    ) -> Optional[nn.Module]:
        """
        Get the target module for hook registration.

        Supports common transformer architectures:
        - GPT-2/GPT-Neo: transformer.h[layer]
        - LLaMA/Mistral/Gemma: model.layers[layer]
        - BLOOM: transformer.h[layer]

        Args:
            model: The transformer model
            layer: Layer index
            hook_type: Type of hook (resid_pre, resid_post, attn, mlp)

        Returns:
            Target module or None if not found
        """
        # Try different model architectures
        module = None

        # GPT-2 style (transformer.h)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
            if layer < len(layers):
                module = layers[layer]

        # LLaMA/Mistral style (model.layers)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            if layer < len(layers):
                module = layers[layer]

        # GPT-NeoX style (gpt_neox.layers)
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            layers = model.gpt_neox.layers
            if layer < len(layers):
                module = layers[layer]

        # Generic fallback - try to find layers attribute
        elif hasattr(model, "layers"):
            if layer < len(model.layers):
                module = model.layers[layer]

        return module

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
                # Output must be a tuple for transformer layers
                if not isinstance(output, tuple):
                    logger.warning("[Steering Hook] Expected tuple output, got single tensor")
                    return output

                hidden_states = output[0]

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
                    # Get decoder weights - handle different SAE architectures
                    decoder_weight = None

                    if hasattr(sae.model, 'tied_weights') and sae.model.tied_weights:
                        # Tied weights: decoder = encoder.weight.T
                        decoder_weight = sae.model.encoder.weight.t()  # [hidden_dim, latent_dim]
                    elif hasattr(sae.model, 'decoder_weight') and not isinstance(getattr(sae.model, 'decoder', None), nn.Linear):
                        # JumpReLUSAE: decoder_weight property returns [d_model, d_sae]
                        decoder_weight = sae.model.decoder_weight  # [hidden_dim, latent_dim]
                    elif hasattr(sae.model, 'decoder') and sae.model.decoder is not None:
                        if hasattr(sae.model.decoder, 'weight'):
                            decoder_weight = sae.model.decoder.weight  # [hidden_dim, latent_dim]

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
        sae: LoadedSAE,
        feature_configs: List[FeatureSteeringConfig],
    ) -> List[Any]:
        """
        Register steering hooks on the model.

        Supports multi-layer steering where each feature can target a different layer.
        Groups features by layer and registers one hook per layer.

        Args:
            model: The transformer model
            sae: Loaded SAE model
            feature_configs: List of all feature steering configurations

        Returns:
            List of hook handles for cleanup
        """
        # Group features by layer
        features_by_layer: Dict[int, List[FeatureSteeringConfig]] = {}
        for config in feature_configs:
            layer = config.layer
            if layer not in features_by_layer:
                features_by_layer[layer] = []
            features_by_layer[layer].append(config)

        handles = []

        for layer, layer_features in features_by_layer.items():
            # Get target module
            target_module = self._get_target_module(model, layer)

            if target_module is None:
                logger.warning(f"Could not find layer {layer} in model, skipping")
                continue

            # Create and register hook
            hook_fn = self._create_steering_hook(sae, layer_features)
            handle = target_module.register_forward_hook(hook_fn)
            handles.append(handle)

            logger.info(
                f"[Steering] Registered hook on layer {layer}, "
                f"module type: {type(target_module).__name__}, "
                f"features: {[f.feature_idx for f in layer_features]}"
            )

        return handles

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

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - params.max_new_tokens,
        ).to(self._device)

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
        }

        # Disable cache if requested (needed for Gemma-2 with forward hooks)
        # Gemma-2's hybrid cache is incompatible with forward hooks
        if disable_cache:
            gen_kwargs["use_cache"] = False
            logger.debug("KV cache disabled for generation (forward hooks active)")

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

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode (only new tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        generation_time_ms = int((time.time() - start_time) * 1000)
        token_count = len(generated_ids)

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
    ) -> float:
        """
        Calculate semantic coherence between prompt and generation.

        Uses sentence embeddings to measure how topically related
        the generation is to the prompt.

        Args:
            prompt: Original prompt
            generated: Generated text

        Returns:
            Coherence score (0-1, higher = more coherent)
        """
        # Lazy load sentence transformer
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._sentence_model.to(self._device)
            except ImportError:
                logger.warning("sentence-transformers not installed, returning default coherence")
                return 0.5

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
    ) -> float:
        """
        Calculate behavioral score measuring steering effectiveness.

        Higher score indicates the steering had a noticeable effect
        on the generation while maintaining coherence.

        Args:
            steered_text: Text generated with steering
            unsteered_text: Baseline text without steering
            feature_labels: Labels of steered features for context

        Returns:
            Behavioral score (0-1)
        """
        if self._sentence_model is None:
            return 0.5

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

        try:
            # Load SAE and model
            sae = await self.load_sae(
                sae_path,
                request.sae_id,
                layer=sae_layer,
                d_model=sae_d_model,
                n_features=sae_n_features,
                architecture=sae_architecture,
            )
            model, tokenizer = await self.load_model(model_id, model_path)

            # CRITICAL: Clear any stale hooks from previous requests that may have timed out
            # This ensures unsteered baseline is truly unsteered and not contaminated
            # by steering hooks from a previous request that didn't clean up properly
            self._clear_all_model_hooks(model)

            # Use all selected features - duplicates with different strengths are intentional
            # (e.g., same feature at +50 and -50 for A/B comparison)
            unique_features = request.selected_features

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
                # Multi-strength mode: generate at multiple strengths per feature
                steered_multi_outputs = await self._generate_multi_strength_outputs(
                    model=model,
                    tokenizer=tokenizer,
                    sae=sae,
                    request=request,
                    unique_features=unique_features,
                    unsteered_text=unsteered_text,
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
                    created_at=datetime.utcnow(),
                )

            # Single-strength mode: one output per feature (existing behavior)
            steered_outputs = []

            for feature in unique_features:
                # Create config for just this feature
                single_feature_config = [
                    FeatureSteeringConfig(
                        feature_idx=feature.feature_idx,
                        layer=feature.layer,
                        strength=feature.strength,
                        label=feature.label,
                    )
                ]

                # Register steering hooks for this single feature
                handles = self._register_steering_hooks(model, sae, single_feature_config)

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
                    # Clean up hooks before next feature
                    for handle in handles:
                        handle.remove()

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
                created_at=datetime.utcnow(),
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
            logger.info("[Steering] GPU cleanup completed")

    async def _generate_multi_strength_outputs(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sae: LoadedSAE,
        request: SteeringComparisonRequest,
        unique_features: List[SelectedFeature],
        unsteered_text: Optional[str],
    ) -> List[SteeredOutputMulti]:
        """
        Generate outputs for each feature at multiple strength values.

        For each feature that has additional_strengths, generates text at:
        - The primary strength
        - Each additional strength

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
        results = []

        for feature in unique_features:
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

            for strength in all_strengths:
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

                # Register steering hooks
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
                    # Clean up hooks before next strength
                    for handle in handles:
                        handle.remove()

            # Find primary result (matches original strength)
            primary_idx = all_strengths.index(feature.strength)
            primary_result = strength_results[primary_idx]
            additional_results = [r for i, r in enumerate(strength_results) if i != primary_idx]

            results.append(SteeredOutputMulti(
                feature_config=feature,
                primary_result=primary_result,
                additional_results=additional_results,
            ))

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

        try:
            # Load SAE and model
            sae = await self.load_sae(
                sae_path,
                request.sae_id,
                layer=sae_layer,
                d_model=sae_d_model,
                n_features=sae_n_features,
                architecture=sae_architecture,
            )
            model, tokenizer = await self.load_model(model_id, model_path)

            # CRITICAL: Clear any stale hooks from previous requests that may have timed out
            # This ensures unsteered baseline is truly unsteered
            self._clear_all_model_hooks(model)

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
            results = []

            for strength in request.strength_values:
                # Create feature config
                feature_config = FeatureSteeringConfig(
                    feature_idx=request.feature_idx,
                    layer=request.layer,
                    strength=strength,
                )

                # Register hook
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
                    for handle in handles:
                        handle.remove()

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
                created_at=datetime.utcnow(),
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
            logger.info("[Strength Sweep] GPU cleanup completed")

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


# Global service instance
_steering_service: Optional[SteeringService] = None


def get_steering_service() -> SteeringService:
    """Get or create the global steering service instance."""
    global _steering_service
    if _steering_service is None:
        _steering_service = SteeringService()
    return _steering_service
