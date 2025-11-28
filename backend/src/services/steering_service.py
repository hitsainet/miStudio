"""
Steering Service for feature-based model steering.

This module provides the core steering functionality:
1. Loading and managing SAE models for steering
2. Registering forward hooks on transformer layers
3. Modifying activations based on feature strengths
4. Generating steered and unsteered text
5. Computing evaluation metrics (perplexity, coherence, behavioral score)
"""

import asyncio
import logging
import math
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
    SteeringStrengthSweepRequest,
    StrengthSweepResponse,
    StrengthSweepResult,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureSteeringConfig:
    """Configuration for a single feature's steering."""

    feature_idx: int
    layer: int
    strength: float  # -100 to +300
    label: Optional[str] = None
    color: str = "teal"

    @property
    def multiplier(self) -> float:
        """
        Convert strength to activation multiplier.

        Mapping:
            -100 -> 0x (full suppression)
            0 -> 1x (no change)
            +100 -> 2x (double)
            +200 -> 3x (triple)
            +300 -> 4x (quadruple)
        """
        return 1 + (self.strength / 100.0)


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
        sae_model = create_sae(
            architecture_type=arch_type,
            hidden_dim=d_in,
            latent_dim=d_sae,
            l1_alpha=l1_coef,
            normalize_activations=normalize,
        )

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
        Create a steering hook function.

        The hook adds steering contributions to the original activations WITHOUT
        doing a full SAE reconstruction (which would corrupt the activations).

        For each steered feature, we:
        1. Encode activations to get feature activation values
        2. Compute the additional contribution: (multiplier - 1) * feature_act * decoder_weight
        3. Add this delta to the original activations

        This preserves the original model activations while adding steering effects.

        Args:
            sae: Loaded SAE model
            feature_configs: List of feature steering configurations

        Returns:
            Hook function compatible with PyTorch register_forward_hook
        """
        def steering_hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()

            # Original shape
            original_shape = hidden_states.shape
            batch_size, seq_len, hidden_dim = original_shape

            # Flatten for SAE processing
            activations = hidden_states.view(-1, hidden_dim)

            with torch.no_grad():
                # Encode to get feature activations
                feature_acts = sae.model.encode(activations)

                # Start with original activations (no reconstruction!)
                steered_activations = activations.clone()

                # Get decoder weights - handle different SAE architectures
                # IMPORTANT: Check decoder_weight FIRST for JumpReLUSAE
                # JumpReLUSAE has both decoder (compat wrapper with TRANSPOSED weights) and decoder_weight (correct shape)
                decoder_weight = None

                if hasattr(sae.model, 'tied_weights') and sae.model.tied_weights:
                    # Tied weights: decoder = encoder.weight.T
                    # encoder.weight shape is [latent_dim, hidden_dim]
                    # We need [hidden_dim, latent_dim] for indexing
                    decoder_weight = sae.model.encoder.weight.t()  # [hidden_dim, latent_dim]
                elif hasattr(sae.model, 'decoder_weight') and not isinstance(getattr(sae.model, 'decoder', None), nn.Linear):
                    # JumpReLUSAE: decoder_weight property returns [d_model, d_sae] (correct shape)
                    # The decoder compat wrapper returns TRANSPOSED weights, so check decoder_weight first
                    decoder_weight = sae.model.decoder_weight  # [hidden_dim, latent_dim]
                elif hasattr(sae.model, 'decoder') and sae.model.decoder is not None:
                    if hasattr(sae.model.decoder, 'weight'):
                        # Standard nn.Linear decoder: weight shape is [hidden_dim, latent_dim]
                        decoder_weight = sae.model.decoder.weight  # [hidden_dim, latent_dim]

                if decoder_weight is None:
                    logger.warning("Could not find decoder weights, using fallback decode method")

                # Add steering delta for each feature
                for config in feature_configs:
                    feat_idx = config.feature_idx
                    # Extra strength: multiplier - 1 (so multiplier=1 means no change)
                    extra_strength = config.multiplier - 1.0

                    if extra_strength == 0:
                        continue  # No change needed

                    # Get the feature activation for all tokens
                    feat_act = feature_acts[:, feat_idx]  # [batch*seq]

                    if decoder_weight is not None:
                        # Get the decoder direction for this feature
                        # For nn.Linear, weight is [d_in, d_sae], so column feat_idx
                        feat_direction = decoder_weight[:, feat_idx]  # [d_in]

                        # Compute steering delta: extra_strength * feat_act * direction
                        # feat_act: [batch*seq], feat_direction: [d_in]
                        # Result: [batch*seq, d_in]
                        steering_delta = extra_strength * feat_act.unsqueeze(-1) * feat_direction.unsqueeze(0)
                    else:
                        # Fallback: create a one-hot and decode it
                        one_hot = torch.zeros_like(feature_acts)
                        one_hot[:, feat_idx] = feat_act * extra_strength
                        steering_delta = sae.model.decode(one_hot) - sae.model.decode(torch.zeros_like(feature_acts))

                    # Add the steering delta to activations
                    steered_activations = steered_activations + steering_delta

            # Reshape back
            steered_hidden = steered_activations.view(original_shape)

            # Return in same format as input
            if other_outputs:
                return (steered_hidden,) + other_outputs
            return steered_hidden

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

            logger.debug(
                f"Registered steering hook on layer {layer} "
                f"for features: {[f.feature_idx for f in layer_features]}"
            )

        return handles

    async def _generate_text(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        params: GenerationParams,
        advanced_params: Optional[AdvancedGenerationParams] = None,
    ) -> Tuple[str, int, int]:
        """
        Generate text using the model.

        Args:
            model: The transformer model
            tokenizer: The tokenizer
            prompt: Input prompt
            params: Generation parameters
            advanced_params: Optional advanced generation parameters

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

        # Convert selected features to configs with deduplication
        seen_features = set()
        unique_features = []
        for f in request.selected_features:
            key = (f.feature_idx, f.layer)
            if key not in seen_features:
                seen_features.add(key)
                unique_features.append(f)
            else:
                logger.warning(f"Duplicate feature ignored: feature_idx={f.feature_idx}, layer={f.layer}")

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
            text, token_count, gen_time = await self._generate_text(
                model, tokenizer, request.prompt,
                request.generation_params,
                request.advanced_params,
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

        # Generate steered outputs - one per feature, each feature tested individually
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
                text, token_count, gen_time = await self._generate_text(
                    model, tokenizer, request.prompt,
                    request.generation_params,
                    request.advanced_params,
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
            metrics_summary=metrics_summary,
            total_time_ms=total_time_ms,
            created_at=datetime.utcnow(),
        )

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

        # Generate unsteered baseline
        text, token_count, gen_time = await self._generate_text(
            model, tokenizer, request.prompt,
            request.generation_params,
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
                text, token_count, gen_time = await self._generate_text(
                    model, tokenizer, request.prompt,
                    request.generation_params,
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

    def unload_sae(self, sae_id: str) -> bool:
        """
        Unload a cached SAE from memory.

        Args:
            sae_id: SAE identifier

        Returns:
            True if unloaded, False if not found
        """
        if sae_id in self._loaded_saes:
            del self._loaded_saes[sae_id]
            return True
        return False

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a cached model from memory.

        Args:
            model_id: Model identifier

        Returns:
            True if unloaded, False if not found
        """
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
            return True
        return False

    def clear_cache(self):
        """Clear all cached models and SAEs and free GPU memory."""
        import gc

        # Log what we're clearing
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

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("CUDA cache cleared")

        logger.info("Steering cache cleared successfully")


# Global service instance
_steering_service: Optional[SteeringService] = None


def get_steering_service() -> SteeringService:
    """Get or create the global steering service instance."""
    global _steering_service
    if _steering_service is None:
        _steering_service = SteeringService()
    return _steering_service
