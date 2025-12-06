"""
Community Standard format utilities for saving and loading SAEs.

This module provides utilities to save/load SAEs in the Community Standard format,
enabling interoperability with popular SAE ecosystems (SAELens, Neuronpedia) and HuggingFace.

Community Standard Format Structure:
    {sae_name}/
    ├── cfg.json              # Configuration/metadata
    ├── sae_weights.safetensors  # Model weights (W_enc, b_enc, W_dec, b_dec)
    └── sparsity.safetensors     # Optional: Feature sparsity statistics

Weight tensor names (Community Standard convention):
    - W_enc: Encoder weights [d_in, d_sae]
    - b_enc: Encoder bias [d_sae]
    - W_dec: Decoder weights [d_sae, d_in]
    - b_dec: Decoder bias [d_in]

Note: The Community Standard uses different tensor shapes than PyTorch nn.Linear:
    - PyTorch Linear(d_in, d_sae): weight is [d_sae, d_in]
    - Community Standard W_enc: weight is [d_in, d_sae] (transposed)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from ..utils.transformerlens_mapping import (
    get_transformerlens_model_id,
    get_transformerlens_hook_name,
)

logger = logging.getLogger(__name__)


@dataclass
class CommunityStandardConfig:
    """
    Configuration for Community Standard compatible SAE.

    This matches the cfg.json format used by major SAE tools for HuggingFace uploads.
    Includes fields required by Neuronpedia for feature dashboard compatibility.
    """
    # Required fields
    model_name: str
    hook_point: str  # e.g., "blocks.0.hook_resid_pre"
    hook_point_layer: int
    d_in: int  # Input dimension (model hidden size)
    d_sae: int  # SAE dimension (number of features)

    # Neuronpedia-compatible field (same as hook_point, for compatibility)
    hook_name: Optional[str] = None  # Will be set from hook_point if not provided

    # Architecture
    architecture: str = "standard"  # standard, gated, jumprelu
    activation_fn_str: str = "relu"  # relu, topk
    normalize_activations: str = "none"  # none, constant_norm_rescale

    # Optional fields
    hook_point_head_index: Optional[int] = None
    context_size: int = 128
    dataset_path: Optional[str] = None

    # Neuronpedia-specific fields
    prepend_bos: bool = True  # Whether BOS token was prepended during training

    # Training parameters (for provenance)
    l1_coefficient: Optional[float] = None
    lr: Optional[float] = None
    total_training_tokens: Optional[int] = None
    train_batch_size: Optional[int] = None

    # Feature scaling
    apply_b_dec_to_input: bool = False
    finetuning_scaling_factor: bool = False

    # Metadata
    dtype: str = "torch.float32"
    device: str = "cpu"

    # miStudio-specific fields (for round-tripping)
    mistudio_training_id: Optional[str] = None
    mistudio_checkpoint_step: Optional[int] = None

    # Additional metadata
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure hook_name is set from hook_point if not provided."""
        if self.hook_name is None:
            self.hook_name = self.hook_point

    @property
    def expansion_factor(self) -> float:
        """Calculate expansion factor (d_sae / d_in)."""
        return self.d_sae / self.d_in if self.d_in > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Add computed field
        d["expansion_factor"] = self.expansion_factor
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommunityStandardConfig":
        """Create from dictionary (e.g., loaded from JSON)."""
        # Remove computed fields that aren't in __init__
        d = d.copy()
        d.pop("expansion_factor", None)

        # Handle unknown fields by storing in extra_metadata
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        extra = {}
        for k in list(d.keys()):
            if k not in known_fields:
                extra[k] = d.pop(k)

        if extra and "extra_metadata" not in d:
            d["extra_metadata"] = extra
        elif extra:
            d["extra_metadata"].update(extra)

        return cls(**d)

    @classmethod
    def from_training_hyperparams(
        cls,
        hyperparams: Dict[str, Any],
        model_name: str,
        layer: int,
        training_id: Optional[str] = None,
        checkpoint_step: Optional[int] = None,
    ) -> "CommunityStandardConfig":
        """
        Create config from miStudio training hyperparameters.

        Args:
            hyperparams: Training hyperparameters dict
            model_name: Name of the target model
            layer: Target layer index
            training_id: Optional training job ID
            checkpoint_step: Optional checkpoint step number

        Returns:
            CommunityStandardConfig instance
        """
        # Map architecture type
        arch_type = hyperparams.get("architecture_type", "standard")

        # Determine activation function
        activation_fn = "relu"
        if arch_type == "jumprelu":
            activation_fn = "jumprelu"
        elif hyperparams.get("top_k_sparsity") is not None:
            activation_fn = "topk"

        # Generate TransformerLens-compatible hook point name
        hook_point = get_transformerlens_hook_name(layer, "resid_post")

        # Map model name to TransformerLens format for Neuronpedia compatibility
        tl_model_name = get_transformerlens_model_id(model_name)

        # Build extra metadata
        extra_metadata = {
            "hf_model_name": model_name,  # Original HuggingFace model name for reference
            "ghost_gradient_penalty": hyperparams.get("ghost_gradient_penalty"),
            "top_k_sparsity": hyperparams.get("top_k_sparsity"),
            "warmup_steps": hyperparams.get("warmup_steps"),
            "weight_decay": hyperparams.get("weight_decay"),
        }

        # Add JumpReLU-specific parameters to metadata
        if arch_type == "jumprelu":
            extra_metadata.update({
                "initial_threshold": hyperparams.get("initial_threshold"),
                "bandwidth": hyperparams.get("bandwidth"),
                "sparsity_coeff": hyperparams.get("sparsity_coeff"),
                "normalize_decoder": hyperparams.get("normalize_decoder"),
            })

        return cls(
            model_name=tl_model_name,  # Use TransformerLens model ID for Neuronpedia compatibility
            hook_point=hook_point,
            hook_name=hook_point,  # Explicitly set hook_name for Neuronpedia
            hook_point_layer=layer,
            d_in=hyperparams.get("hidden_dim", 768),
            d_sae=hyperparams.get("latent_dim", 12288),
            architecture=arch_type,
            activation_fn_str=activation_fn,
            normalize_activations=hyperparams.get("normalize_activations", "none"),
            prepend_bos=True,  # miStudio typically prepends BOS
            l1_coefficient=hyperparams.get("l1_alpha") or hyperparams.get("sparsity_coeff"),
            lr=hyperparams.get("learning_rate"),
            total_training_tokens=hyperparams.get("total_steps", 0) * hyperparams.get("batch_size", 1),
            train_batch_size=hyperparams.get("batch_size"),
            mistudio_training_id=training_id,
            mistudio_checkpoint_step=checkpoint_step,
            extra_metadata=extra_metadata,
        )


def convert_mistudio_to_community_weights(
    state_dict: Dict[str, torch.Tensor],
    tied_weights: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert miStudio SAE state dict to Community Standard weight format.

    Handles two SAE formats:

    1. Standard SAE (SparseAutoencoder, SkipAutoencoder, Transcoder):
        encoder.weight: [d_sae, d_in]
        encoder.bias: [d_sae]
        decoder.weight: [d_in, d_sae]
        decoder.bias: [d_in]
        decoder_bias: [d_in]

    2. JumpReLU SAE:
        W_enc: [d_sae, d_in]
        b_enc: [d_sae]
        W_dec: [d_in, d_sae]
        b_dec: [d_in]
        activation.log_threshold: [d_sae]

    Community Standard uses:
        W_enc: [d_in, d_sae]  (transposed from miStudio)
        b_enc: [d_sae]
        W_dec: [d_sae, d_in]  (transposed from miStudio)
        b_dec: [d_in]
        (JumpReLU) threshold: [d_sae]  (exp of log_threshold)

    Args:
        state_dict: miStudio model state dict
        tied_weights: Whether encoder/decoder weights are tied

    Returns:
        Community Standard format weight dict

    Raises:
        ValueError: If state_dict is empty or doesn't contain expected keys
    """
    # Validate input
    if not state_dict:
        raise ValueError("Cannot convert empty state_dict to Community Standard format")

    community_weights = {}

    # Log input keys for debugging
    logger.debug(f"Converting state_dict with keys: {list(state_dict.keys())}")

    # Detect JumpReLU SAE by presence of W_enc (direct parameter) vs encoder.weight (nn.Linear)
    is_jumprelu = "W_enc" in state_dict
    is_standard = "encoder.weight" in state_dict

    # Validate that we can identify the format
    if not is_jumprelu and not is_standard:
        logger.warning(
            f"Unrecognized state_dict format. Keys: {list(state_dict.keys())}. "
            f"Expected either JumpReLU keys (W_enc, b_enc, ...) or Standard keys (encoder.weight, ...)."
        )

    if is_jumprelu:
        # JumpReLU SAE format
        # W_enc is [d_sae, d_in], transpose to [d_in, d_sae]
        if "W_enc" in state_dict:
            community_weights["W_enc"] = state_dict["W_enc"].t().contiguous()

        # b_enc is [d_sae], direct copy
        if "b_enc" in state_dict:
            community_weights["b_enc"] = state_dict["b_enc"]

        # Decoder weights
        if tied_weights:
            if "W_enc" in state_dict:
                community_weights["W_dec"] = state_dict["W_enc"].contiguous()
        else:
            # W_dec is [d_in, d_sae], transpose to [d_sae, d_in]
            if "W_dec" in state_dict:
                community_weights["W_dec"] = state_dict["W_dec"].t().contiguous()

        # b_dec is [d_in], direct copy
        if "b_dec" in state_dict:
            community_weights["b_dec"] = state_dict["b_dec"]

        # JumpReLU thresholds (stored as log_threshold, convert to actual threshold)
        if "activation.log_threshold" in state_dict:
            community_weights["threshold"] = torch.exp(state_dict["activation.log_threshold"])

    else:
        # Standard SAE format (SparseAutoencoder, SkipAutoencoder, Transcoder)
        # Encoder weights: transpose from [d_sae, d_in] to [d_in, d_sae]
        if "encoder.weight" in state_dict:
            community_weights["W_enc"] = state_dict["encoder.weight"].t().contiguous()

        # Encoder bias: direct copy
        if "encoder.bias" in state_dict:
            community_weights["b_enc"] = state_dict["encoder.bias"]

        # Decoder weights
        if tied_weights:
            # For tied weights, W_dec = W_enc.T
            # Community Standard expects [d_sae, d_in], which is encoder.weight without transpose
            if "encoder.weight" in state_dict:
                community_weights["W_dec"] = state_dict["encoder.weight"].contiguous()
        else:
            # Decoder weights: transpose from [d_in, d_sae] to [d_sae, d_in]
            if "decoder.weight" in state_dict:
                community_weights["W_dec"] = state_dict["decoder.weight"].t().contiguous()

        # Decoder bias: prefer decoder_bias if present, else decoder.bias
        if "decoder_bias" in state_dict:
            community_weights["b_dec"] = state_dict["decoder_bias"]
        elif "decoder.bias" in state_dict:
            community_weights["b_dec"] = state_dict["decoder.bias"]

    # Validate output - conversion should produce at least W_enc and b_enc
    if not community_weights:
        raise ValueError(
            f"Conversion produced empty output. Input state_dict keys: {list(state_dict.keys())}. "
            f"Detected format: {'JumpReLU' if is_jumprelu else 'Standard' if is_standard else 'Unknown'}"
        )

    required_keys = {"W_enc", "b_enc"}
    missing_keys = required_keys - set(community_weights.keys())
    if missing_keys:
        raise ValueError(
            f"Conversion missing required keys: {missing_keys}. "
            f"Output keys: {list(community_weights.keys())}. "
            f"Input keys: {list(state_dict.keys())}"
        )

    logger.debug(f"Converted to Community Standard format with keys: {list(community_weights.keys())}")
    return community_weights


def convert_community_to_mistudio_weights(
    community_weights: Dict[str, torch.Tensor],
    tied_weights: bool = False,
    is_jumprelu: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert Community Standard weight format to miStudio state dict.

    Community Standard uses:
        W_enc: [d_in, d_sae]
        b_enc: [d_sae]
        W_dec: [d_sae, d_in]
        b_dec: [d_in]
        (JumpReLU) threshold: [d_sae]

    miStudio Standard SAE uses PyTorch nn.Linear convention:
        encoder.weight: [d_sae, d_in]
        encoder.bias: [d_sae]
        decoder.weight: [d_in, d_sae]
        decoder.bias: [d_in]
        decoder_bias: [d_in]

    miStudio JumpReLU SAE uses direct parameters:
        W_enc: [d_sae, d_in]
        b_enc: [d_sae]
        W_dec: [d_in, d_sae]
        b_dec: [d_in]
        activation.log_threshold: [d_sae]

    Args:
        community_weights: Community Standard format weight dict
        tied_weights: Whether to use tied weights
        is_jumprelu: Whether to output JumpReLU format (auto-detected if threshold present)

    Returns:
        miStudio model state dict
    """
    # Auto-detect JumpReLU if threshold is present
    if "threshold" in community_weights:
        is_jumprelu = True

    mistudio_weights = {}

    if is_jumprelu:
        # JumpReLU SAE format
        # W_enc: transpose from [d_in, d_sae] to [d_sae, d_in]
        if "W_enc" in community_weights:
            mistudio_weights["W_enc"] = community_weights["W_enc"].t().contiguous()

        # b_enc: direct copy
        if "b_enc" in community_weights:
            mistudio_weights["b_enc"] = community_weights["b_enc"]

        # W_dec: transpose from [d_sae, d_in] to [d_in, d_sae]
        if not tied_weights and "W_dec" in community_weights:
            mistudio_weights["W_dec"] = community_weights["W_dec"].t().contiguous()

        # b_dec: direct copy
        if "b_dec" in community_weights:
            mistudio_weights["b_dec"] = community_weights["b_dec"]

        # JumpReLU thresholds (convert back to log_threshold)
        if "threshold" in community_weights:
            mistudio_weights["activation.log_threshold"] = torch.log(community_weights["threshold"])

    else:
        # Standard SAE format (SparseAutoencoder, SkipAutoencoder, Transcoder)
        # Encoder weights: transpose from [d_in, d_sae] to [d_sae, d_in]
        if "W_enc" in community_weights:
            mistudio_weights["encoder.weight"] = community_weights["W_enc"].t().contiguous()

        # Encoder bias: direct copy
        if "b_enc" in community_weights:
            mistudio_weights["encoder.bias"] = community_weights["b_enc"]

        # Decoder weights (only if not tied)
        if not tied_weights and "W_dec" in community_weights:
            # Transpose from [d_sae, d_in] to [d_in, d_sae]
            mistudio_weights["decoder.weight"] = community_weights["W_dec"].t().contiguous()

        # Decoder bias
        if "b_dec" in community_weights:
            mistudio_weights["decoder_bias"] = community_weights["b_dec"]
            # Also set decoder.bias if not tied
            if not tied_weights:
                mistudio_weights["decoder.bias"] = torch.zeros_like(community_weights["b_dec"])

    return mistudio_weights


def save_sae_community_format(
    model: nn.Module,
    config: CommunityStandardConfig,
    output_dir: Path,
    sparsity: Optional[torch.Tensor] = None,
    tied_weights: bool = False,
) -> None:
    """
    Save an SAE in Community Standard compatible format.

    Creates:
        {output_dir}/
        ├── cfg.json
        ├── sae_weights.safetensors
        └── sparsity.safetensors (optional)

    Args:
        model: PyTorch SAE model
        config: CommunityStandardConfig with metadata
        output_dir: Directory to save files
        sparsity: Optional feature sparsity tensor [d_sae]
        tied_weights: Whether model uses tied weights

    Raises:
        ValueError: If conversion fails or produces invalid output
        RuntimeError: If saved file is suspiciously small
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model state dict
    state_dict = model.state_dict()
    logger.info(
        f"Saving SAE to Community Standard format: model={model.__class__.__name__}, "
        f"state_dict_keys={list(state_dict.keys())}, tied_weights={tied_weights}"
    )

    # Convert to Community Standard format (this will raise ValueError if conversion fails)
    community_weights = convert_mistudio_to_community_weights(state_dict, tied_weights)

    # Log tensor shapes for debugging
    for key, tensor in community_weights.items():
        logger.debug(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")

    # Save weights
    weights_path = output_dir / "sae_weights.safetensors"
    save_file(community_weights, str(weights_path))

    # Verify saved file is not empty/corrupt (safetensors with no data is ~16 bytes)
    file_size = weights_path.stat().st_size
    if file_size < 1000:  # Minimum reasonable size for SAE weights
        raise RuntimeError(
            f"Saved weights file is suspiciously small ({file_size} bytes). "
            f"Expected at least 1KB for valid SAE weights. "
            f"Community weights keys: {list(community_weights.keys())}"
        )

    logger.info(f"Saved Community Standard weights to {weights_path} ({file_size / 1024 / 1024:.2f} MB)")

    # Save config
    config_path = output_dir / "cfg.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Saved Community Standard config to {config_path}")

    # Save sparsity if provided
    if sparsity is not None:
        sparsity_path = output_dir / "sparsity.safetensors"
        save_file({"sparsity": sparsity}, str(sparsity_path))
        logger.info(f"Saved sparsity data to {sparsity_path}")


def load_sae_community_format(
    input_dir: Path,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], CommunityStandardConfig, Optional[torch.Tensor]]:
    """
    Load an SAE from Community Standard format.

    Args:
        input_dir: Directory containing cfg.json and sae_weights.safetensors
        device: Device to load tensors onto

    Returns:
        Tuple of (miStudio state_dict, config, sparsity)
    """
    input_dir = Path(input_dir)

    # Load config
    config_path = input_dir / "cfg.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = CommunityStandardConfig.from_dict(config_dict)

    # Load weights
    weights_path = input_dir / "sae_weights.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    community_weights = load_file(str(weights_path), device=device)

    # Determine if tied weights based on config or weight shapes
    # Community Standard doesn't have explicit tied_weights flag, but we can infer
    # For now, assume not tied (most common case)
    tied_weights = False

    # Convert to miStudio format
    mistudio_weights = convert_community_to_mistudio_weights(community_weights, tied_weights)

    # Load sparsity if present
    sparsity = None
    sparsity_path = input_dir / "sparsity.safetensors"
    if sparsity_path.exists():
        sparsity_data = load_file(str(sparsity_path), device=device)
        sparsity = sparsity_data.get("sparsity")

    return mistudio_weights, config, sparsity


def is_community_format(path: Path) -> bool:
    """
    Check if a path contains Community Standard format files.

    Args:
        path: Directory path to check

    Returns:
        True if cfg.json and sae_weights.safetensors exist
    """
    path = Path(path)
    return (
        (path / "cfg.json").exists() and
        (path / "sae_weights.safetensors").exists()
    )


def is_gemma_scope_format(path: Path) -> bool:
    """
    Check if a path contains Gemma Scope format (params.npz).

    Gemma Scope SAEs from Google use NumPy archives containing:
    - W_enc: [d_in, d_sae]
    - W_dec: [d_sae, d_in]
    - b_enc: [d_sae]
    - b_dec: [d_in]
    - threshold: [d_sae] (for JumpReLU)

    Args:
        path: Directory or file path to check

    Returns:
        True if params.npz exists
    """
    path = Path(path)

    # Check for direct params.npz file
    if path.name == "params.npz" and path.exists():
        return True

    # Check for params.npz in directory
    if path.is_dir() and (path / "params.npz").exists():
        return True

    return False


def load_gemma_scope_format(
    input_path: Path,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], CommunityStandardConfig]:
    """
    Load an SAE from Gemma Scope format (params.npz).

    Gemma Scope weight format:
        W_enc: [d_in, d_sae] - encoder weights
        W_dec: [d_sae, d_in] - decoder weights
        b_enc: [d_sae] - encoder bias
        b_dec: [d_in] - decoder bias (often zero)
        threshold: [d_sae] - JumpReLU thresholds

    Converts to miStudio JumpReLU format:
        W_enc: [d_sae, d_in]
        W_dec: [d_in, d_sae]
        b_enc: [d_sae]
        b_dec: [d_in]
        activation.log_threshold: [d_sae]

    Args:
        input_path: Path to params.npz file or directory containing it
        device: Device to load tensors onto

    Returns:
        Tuple of (miStudio state_dict, config)
    """
    input_path = Path(input_path)

    # Find params.npz
    if input_path.name == "params.npz":
        npz_path = input_path
    else:
        npz_path = input_path / "params.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"params.npz not found at {npz_path}")

    # Load numpy arrays
    with np.load(str(npz_path)) as data:
        # Gemma Scope format: W_enc [d_in, d_sae], W_dec [d_sae, d_in]
        W_enc_np = data["W_enc"]  # [d_in, d_sae]
        W_dec_np = data["W_dec"]  # [d_sae, d_in]
        b_enc_np = data["b_enc"]  # [d_sae]

        # b_dec may or may not exist
        b_dec_np = data.get("b_dec", np.zeros(W_dec_np.shape[1], dtype=W_enc_np.dtype))

        # threshold for JumpReLU
        threshold_np = data.get("threshold", None)

    # Get dimensions
    d_in, d_sae = W_enc_np.shape

    # Convert to PyTorch tensors
    # miStudio JumpReLU format: W_enc [d_sae, d_in], W_dec [d_in, d_sae]
    # So we need to transpose from Gemma Scope format
    W_enc = torch.from_numpy(W_enc_np.T).contiguous().to(device)  # [d_sae, d_in]
    W_dec = torch.from_numpy(W_dec_np.T).contiguous().to(device)  # [d_in, d_sae]
    b_enc = torch.from_numpy(b_enc_np).to(device)
    b_dec = torch.from_numpy(b_dec_np).to(device)

    # Build state dict in miStudio JumpReLU format
    mistudio_weights = {
        "W_enc": W_enc,
        "W_dec": W_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
    }

    # Handle JumpReLU thresholds
    if threshold_np is not None:
        # Convert threshold to log_threshold for JumpReLU activation
        threshold = torch.from_numpy(threshold_np).to(device)
        # Clamp to avoid log(0)
        threshold = torch.clamp(threshold, min=1e-10)
        mistudio_weights["activation.log_threshold"] = torch.log(threshold)

    # Try to extract metadata from path
    # Gemma Scope paths: layer_{N}/width_{width}/average_l0_{l0}/params.npz
    layer = None
    model_name = "google/gemma-2-2b"  # Default assumption for Gemma Scope
    l0_sparsity = None

    path_str = str(input_path)
    import re

    # Extract layer
    layer_match = re.search(r"layer[_/](\d+)", path_str, re.IGNORECASE)
    if layer_match:
        layer = int(layer_match.group(1))

    # Extract L0 sparsity from path
    l0_match = re.search(r"average_l0[_/](\d+)", path_str, re.IGNORECASE)
    if l0_match:
        l0_sparsity = int(l0_match.group(1))

    # Detect model from d_in
    if d_in == 2304:
        model_name = "google/gemma-2-2b"
    elif d_in == 3584:
        model_name = "google/gemma-2-9b"
    elif d_in == 4608:
        model_name = "google/gemma-2-27b"

    # Create config
    config = CommunityStandardConfig(
        model_name=model_name,
        hook_point=f"blocks.{layer or 0}.hook_resid_post",
        hook_point_layer=layer or 0,
        d_in=d_in,
        d_sae=d_sae,
        architecture="jumprelu",
        activation_fn_str="jumprelu",
        normalize_activations="none",
        extra_metadata={
            "source_format": "gemma_scope",
            "l0_sparsity": l0_sparsity,
        },
    )

    logger.info(
        f"Loaded Gemma Scope SAE: d_in={d_in}, d_sae={d_sae}, "
        f"layer={layer}, has_threshold={threshold_np is not None}"
    )

    return mistudio_weights, config


def is_huggingface_sae_format(path: Path) -> Tuple[bool, Optional[Path]]:
    """
    Check if a path contains HuggingFace SAELens format (safetensors with CS keys, no cfg.json).

    This format is used by Gemma Scope SAEs hosted on HuggingFace. The weights are stored
    in safetensors format with Community Standard key names (W_enc, W_dec, b_enc, b_dec)
    but without a cfg.json config file.

    Args:
        path: Directory or file path to check

    Returns:
        Tuple of (is_hf_format, path_to_safetensors_file)
    """
    path = Path(path)

    # Files to look for in order of preference
    sae_filenames = ["sae.safetensors", "sae_weights.safetensors"]

    def _check_safetensors(safetensors_path: Path) -> bool:
        """Check if a safetensors file has Community Standard weight keys."""
        if not safetensors_path.exists():
            return False
        try:
            weights = load_file(str(safetensors_path), device="cpu")
            # Check for Community Standard keys (W_enc and W_dec are required)
            has_cs_keys = "W_enc" in weights and "W_dec" in weights
            # Make sure it's not a miStudio format (no model.* prefix)
            no_model_prefix = not any(k.startswith("model.") for k in weights.keys())
            return has_cs_keys and no_model_prefix
        except Exception:
            return False

    # Check direct file
    if path.suffix == ".safetensors":
        if _check_safetensors(path):
            return True, path
        return False, None

    # Check directory for SAE files (without cfg.json - that's Community Standard)
    if path.is_dir():
        # If cfg.json exists, it's Community Standard format, not HF SAELens
        if (path / "cfg.json").exists():
            return False, None

        # Check for direct SAE files
        for filename in sae_filenames:
            sae_file = path / filename
            if _check_safetensors(sae_file):
                return True, sae_file

        # Check in layer subdirectories (e.g., layer_20/)
        layer_dirs = sorted(path.glob("layer_*"))
        for layer_dir in layer_dirs:
            for filename in sae_filenames:
                sae_file = layer_dir / filename
                if _check_safetensors(sae_file):
                    return True, sae_file

        # Check in nested width/l0 subdirectories (Gemma Scope structure)
        for subdir in path.rglob("*"):
            if subdir.is_dir():
                for filename in sae_filenames:
                    sae_file = subdir / filename
                    if _check_safetensors(sae_file):
                        return True, sae_file

    return False, None


def load_huggingface_sae_format(
    input_path: Path,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], CommunityStandardConfig]:
    """
    Load an SAE from HuggingFace SAELens format (safetensors with CS keys).

    HuggingFace SAELens format is safetensors with Community Standard weight names:
        W_enc: [d_in, d_sae] - encoder weights
        W_dec: [d_sae, d_in] - decoder weights
        b_enc: [d_sae] - encoder bias
        b_dec: [d_in] - decoder bias
        log_threshold or threshold: [d_sae] or scalar - JumpReLU thresholds (optional)

    Converts to miStudio JumpReLU/Standard format:
        W_enc: [d_sae, d_in] (transposed from CS)
        W_dec: [d_in, d_sae] (transposed from CS)
        b_enc: [d_sae]
        b_dec: [d_in]
        activation.log_threshold: [d_sae] (for JumpReLU)

    Args:
        input_path: Path to safetensors file
        device: Device to load tensors onto

    Returns:
        Tuple of (miStudio state_dict, config)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"SAE file not found at {input_path}")

    # Load weights
    weights = load_file(str(input_path), device=device)

    logger.info(f"Loaded HuggingFace SAE with keys: {list(weights.keys())}")

    # Extract weights - Community Standard orientation
    W_enc = weights["W_enc"]  # [d_in, d_sae]
    W_dec = weights["W_dec"]  # [d_sae, d_in]
    b_enc = weights.get("b_enc")  # [d_sae]
    b_dec = weights.get("b_dec")  # [d_in]

    # Get dimensions
    d_in, d_sae = W_enc.shape

    # Check for JumpReLU threshold
    threshold = weights.get("threshold")
    log_threshold = weights.get("log_threshold")
    is_jumprelu = threshold is not None or log_threshold is not None

    # Build miStudio state dict
    mistudio_weights = {}

    if is_jumprelu:
        # JumpReLU format - miStudio expects [d_sae, d_in] for W_enc
        mistudio_weights["W_enc"] = W_enc.t().contiguous()  # [d_sae, d_in]
        mistudio_weights["W_dec"] = W_dec.t().contiguous()  # [d_in, d_sae]
        if b_enc is not None:
            mistudio_weights["b_enc"] = b_enc
        if b_dec is not None:
            mistudio_weights["b_dec"] = b_dec

        # Handle thresholds
        if log_threshold is not None:
            # Already log threshold
            if log_threshold.dim() == 0:
                # Scalar - expand to per-feature
                mistudio_weights["activation.log_threshold"] = log_threshold.expand(d_sae)
            else:
                mistudio_weights["activation.log_threshold"] = log_threshold
        elif threshold is not None:
            # Convert threshold to log_threshold
            threshold_clamped = torch.clamp(threshold, min=1e-10)
            if threshold_clamped.dim() == 0:
                # Scalar - expand to per-feature
                mistudio_weights["activation.log_threshold"] = torch.log(threshold_clamped).expand(d_sae)
            else:
                mistudio_weights["activation.log_threshold"] = torch.log(threshold_clamped)
    else:
        # Standard format - miStudio expects nn.Linear convention [out_features, in_features]
        mistudio_weights["encoder.weight"] = W_enc.t().contiguous()  # [d_sae, d_in]
        if b_enc is not None:
            mistudio_weights["encoder.bias"] = b_enc
        mistudio_weights["decoder.weight"] = W_dec.t().contiguous()  # [d_in, d_sae]
        if b_dec is not None:
            mistudio_weights["decoder_bias"] = b_dec

    # Try to extract metadata from path
    path_str = str(input_path)
    import re

    # Extract layer from path
    layer = None
    layer_match = re.search(r"layer[_/](\d+)", path_str, re.IGNORECASE)
    if layer_match:
        layer = int(layer_match.group(1))

    # Extract L0 sparsity from path
    l0_sparsity = None
    l0_match = re.search(r"average_l0[_/](\d+)", path_str, re.IGNORECASE)
    if l0_match:
        l0_sparsity = int(l0_match.group(1))

    # Detect model from d_in
    model_name = "unknown"
    if d_in == 2304:
        model_name = "google/gemma-2-2b"
    elif d_in == 3584:
        model_name = "google/gemma-2-9b"
    elif d_in == 4608:
        model_name = "google/gemma-2-27b"
    elif d_in == 2048:
        model_name = "google/gemma-2-2b"  # Instruction-tuned variant

    # Create config
    config = CommunityStandardConfig(
        model_name=model_name,
        hook_point=f"blocks.{layer or 0}.hook_resid_post",
        hook_point_layer=layer or 0,
        d_in=d_in,
        d_sae=d_sae,
        architecture="jumprelu" if is_jumprelu else "standard",
        activation_fn_str="jumprelu" if is_jumprelu else "relu",
        normalize_activations="none",
        extra_metadata={
            "source_format": "huggingface_saelens",
            "l0_sparsity": l0_sparsity,
        },
    )

    logger.info(
        f"Loaded HuggingFace SAELens SAE: d_in={d_in}, d_sae={d_sae}, "
        f"layer={layer}, is_jumprelu={is_jumprelu}"
    )

    return mistudio_weights, config


def is_mistudio_format(path: Path) -> bool:
    """
    Check if a path contains miStudio format checkpoint.

    Args:
        path: Directory or file path to check

    Returns:
        True if it's a miStudio checkpoint format
    """
    path = Path(path)

    # Check for direct checkpoint file
    if path.suffix == ".safetensors":
        # Load header to check for model.* keys
        try:
            weights = load_file(str(path), device="cpu")
            return any(k.startswith("model.") for k in weights.keys())
        except Exception:
            return False

    # Check for checkpoint.safetensors in directory
    checkpoint_path = path / "checkpoint.safetensors"
    if checkpoint_path.exists():
        try:
            weights = load_file(str(checkpoint_path), device="cpu")
            return any(k.startswith("model.") for k in weights.keys())
        except Exception:
            return False

    # Check for layer_* subdirectories (multi-layer SAE structure)
    layer_dirs = list(path.glob("layer_*"))
    if layer_dirs:
        for layer_dir in layer_dirs:
            checkpoint_path = layer_dir / "checkpoint.safetensors"
            if checkpoint_path.exists():
                try:
                    weights = load_file(str(checkpoint_path), device="cpu")
                    if any(k.startswith("model.") for k in weights.keys()):
                        return True
                except Exception:
                    continue

    return False


def find_mistudio_checkpoint(path: Path) -> Optional[Path]:
    """
    Find miStudio checkpoint file in various directory structures.

    Handles:
    1. Direct checkpoint file
    2. checkpoint.safetensors in directory
    3. layer_*/checkpoint.safetensors subdirectories

    Args:
        path: Base path to search

    Returns:
        Path to checkpoint file, or None if not found
    """
    path = Path(path)

    # Direct file
    if path.suffix == ".safetensors" and path.exists():
        return path

    # Direct checkpoint in directory
    checkpoint_path = path / "checkpoint.safetensors"
    if checkpoint_path.exists():
        return checkpoint_path

    # Check layer subdirectories
    layer_dirs = sorted(path.glob("layer_*"))
    if layer_dirs:
        # Return first layer's checkpoint (for single-layer SAEs)
        checkpoint_path = layer_dirs[0] / "checkpoint.safetensors"
        if checkpoint_path.exists():
            return checkpoint_path

    return None


def get_hook_name_from_layer(layer: int, hook_type: str = "resid_post") -> str:
    """
    Generate Community Standard hook name from layer index.

    Args:
        layer: Layer index
        hook_type: Type of hook (resid_pre, resid_post, attn, mlp)

    Returns:
        Hook name string like "blocks.12.hook_resid_post"
    """
    return f"blocks.{layer}.hook_{hook_type}"


def get_layer_from_hook_name(hook_name: str) -> int:
    """
    Extract layer index from Community Standard hook name.

    Args:
        hook_name: Hook name like "blocks.12.hook_resid_post"

    Returns:
        Layer index
    """
    # Parse "blocks.{layer}.hook_*" format
    parts = hook_name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return int(parts[1])
        except ValueError:
            pass

    raise ValueError(f"Cannot parse layer from hook name: {hook_name}")


def load_sae_auto_detect(
    path: Path,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Optional[CommunityStandardConfig], str]:
    """
    Auto-detect and load SAE from any supported format.

    Supports:
    1. Community Standard format (cfg.json + sae_weights.safetensors)
    2. Gemma Scope format (params.npz - JumpReLU SAEs from Google)
    3. HuggingFace SAELens format (sae.safetensors with CS keys, no cfg.json)
    4. miStudio checkpoint format (checkpoint.safetensors with model.* keys)
    5. Multi-layer SAE structure (layer_*/checkpoint.safetensors or layer_*/sae.safetensors)

    Args:
        path: Path to SAE directory or checkpoint file
        device: Device to load tensors onto

    Returns:
        Tuple of (state_dict in miStudio format, config or None, format_type)
    """
    path = Path(path)

    logger.info(f"Auto-detecting SAE format at {path}")

    # Check for Community Standard format first (has cfg.json)
    if is_community_format(path):
        logger.info(f"Detected Community Standard format at {path}")
        state_dict, config, _ = load_sae_community_format(path, device)
        return state_dict, config, "community_standard"

    # Check for Gemma Scope format (params.npz)
    if is_gemma_scope_format(path):
        logger.info(f"Detected Gemma Scope format at {path}")
        state_dict, config = load_gemma_scope_format(path, device)
        return state_dict, config, "gemma_scope"

    # Check for HuggingFace SAELens format (safetensors with CS keys, no cfg.json)
    # This handles nested directory structures like layer_20/sae.safetensors
    is_hf_format, sae_file_path = is_huggingface_sae_format(path)
    if is_hf_format and sae_file_path is not None:
        logger.info(f"Detected HuggingFace SAELens format at {sae_file_path}")
        state_dict, config = load_huggingface_sae_format(sae_file_path, device)
        return state_dict, config, "huggingface_saelens"

    # Check for miStudio format (including layer subdirectories)
    if is_mistudio_format(path):
        logger.info(f"Detected miStudio format at {path}")
        checkpoint_path = find_mistudio_checkpoint(path)
        if checkpoint_path is None:
            raise ValueError(f"Could not find checkpoint file in {path}")

        weights = load_file(str(checkpoint_path), device=device)

        # Extract model weights (remove "model." prefix)
        state_dict = {}
        for key, value in weights.items():
            if key.startswith("model."):
                state_dict[key.replace("model.", "")] = value

        return state_dict, None, "mistudio"

    raise ValueError(
        f"Unrecognized SAE format at {path}. "
        f"Expected either Community Standard format (cfg.json + sae_weights.safetensors), "
        f"Gemma Scope format (params.npz), "
        f"HuggingFace SAELens format (sae.safetensors with W_enc/W_dec keys), "
        f"or miStudio format (checkpoint.safetensors with model.* keys)"
    )


def migrate_mistudio_to_community(
    source_path: Path,
    output_dir: Path,
    model_name: str,
    layer: int,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Migrate a miStudio checkpoint to Community Standard format.

    Args:
        source_path: Path to miStudio checkpoint (directory or .safetensors file)
        output_dir: Directory to save Community Standard format files
        model_name: Name of the target model
        layer: Target layer index
        hyperparams: Optional training hyperparameters

    Returns:
        Path to output directory
    """
    # Load miStudio checkpoint
    checkpoint_path = source_path
    if source_path.is_dir():
        checkpoint_path = source_path / "checkpoint.safetensors"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    weights = load_file(str(checkpoint_path), device="cpu")

    # Extract model weights
    model_state = {}
    for key, value in weights.items():
        if key.startswith("model."):
            model_state[key.replace("model.", "")] = value

    if not model_state:
        raise ValueError(f"No model weights found in {checkpoint_path}")

    # Determine dimensions from weights
    encoder_weight = model_state.get("encoder.weight")
    if encoder_weight is None:
        raise ValueError("encoder.weight not found in checkpoint")

    latent_dim, hidden_dim = encoder_weight.shape

    # Create default hyperparams if not provided
    if hyperparams is None:
        hyperparams = {
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
        }

    # Create config
    config = CommunityStandardConfig.from_training_hyperparams(
        hyperparams=hyperparams,
        model_name=model_name,
        layer=layer,
    )

    # Convert to Community Standard format
    tied_weights = "decoder.weight" not in model_state
    community_weights = convert_mistudio_to_community_weights(model_state, tied_weights)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = output_dir / "sae_weights.safetensors"
    save_file(community_weights, str(weights_path))

    # Save config
    config_path = output_dir / "cfg.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    logger.info(f"Migrated miStudio checkpoint to Community Standard format at {output_dir}")

    return str(output_dir)


# Backwards compatibility aliases (deprecated)
SAELensConfig = CommunityStandardConfig
convert_mistudio_to_saelens_weights = convert_mistudio_to_community_weights
convert_saelens_to_mistudio_weights = convert_community_to_mistudio_weights
save_sae_saelens_format = save_sae_community_format
load_sae_saelens_format = load_sae_community_format
is_saelens_format = is_community_format
migrate_mistudio_to_saelens = migrate_mistudio_to_community
