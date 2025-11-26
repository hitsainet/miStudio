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

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


@dataclass
class CommunityStandardConfig:
    """
    Configuration for Community Standard compatible SAE.

    This matches the cfg.json format used by major SAE tools for HuggingFace uploads.
    """
    # Required fields
    model_name: str
    hook_point: str  # e.g., "blocks.0.hook_resid_pre"
    hook_point_layer: int
    d_in: int  # Input dimension (model hidden size)
    d_sae: int  # SAE dimension (number of features)

    # Architecture
    architecture: str = "standard"  # standard, gated, jumprelu
    activation_fn_str: str = "relu"  # relu, topk
    normalize_activations: str = "none"  # none, constant_norm_rescale

    # Optional fields
    hook_point_head_index: Optional[int] = None
    context_size: int = 128
    dataset_path: Optional[str] = None

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
        activation_fn = "relu"
        if hyperparams.get("top_k_sparsity") is not None:
            activation_fn = "topk"

        # Generate hook point name
        hook_point = f"blocks.{layer}.hook_resid_post"

        return cls(
            model_name=model_name,
            hook_point=hook_point,
            hook_point_layer=layer,
            d_in=hyperparams.get("hidden_dim", 768),
            d_sae=hyperparams.get("latent_dim", 12288),
            architecture=arch_type,
            activation_fn_str=activation_fn,
            normalize_activations=hyperparams.get("normalize_activations", "none"),
            l1_coefficient=hyperparams.get("l1_alpha"),
            lr=hyperparams.get("learning_rate"),
            total_training_tokens=hyperparams.get("total_steps", 0) * hyperparams.get("batch_size", 1),
            train_batch_size=hyperparams.get("batch_size"),
            mistudio_training_id=training_id,
            mistudio_checkpoint_step=checkpoint_step,
            extra_metadata={
                "ghost_gradient_penalty": hyperparams.get("ghost_gradient_penalty"),
                "top_k_sparsity": hyperparams.get("top_k_sparsity"),
                "warmup_steps": hyperparams.get("warmup_steps"),
                "weight_decay": hyperparams.get("weight_decay"),
            }
        )


def convert_mistudio_to_community_weights(
    state_dict: Dict[str, torch.Tensor],
    tied_weights: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert miStudio SAE state dict to Community Standard weight format.

    miStudio uses PyTorch nn.Linear convention:
        encoder.weight: [d_sae, d_in]
        encoder.bias: [d_sae]
        decoder.weight: [d_in, d_sae]
        decoder.bias: [d_in]
        decoder_bias: [d_in]

    Community Standard uses:
        W_enc: [d_in, d_sae]  (transposed)
        b_enc: [d_sae]
        W_dec: [d_sae, d_in]  (transposed)
        b_dec: [d_in]

    Args:
        state_dict: miStudio model state dict
        tied_weights: Whether encoder/decoder weights are tied

    Returns:
        Community Standard format weight dict
    """
    community_weights = {}

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

    return community_weights


def convert_community_to_mistudio_weights(
    community_weights: Dict[str, torch.Tensor],
    tied_weights: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert Community Standard weight format to miStudio state dict.

    Community Standard uses:
        W_enc: [d_in, d_sae]
        b_enc: [d_sae]
        W_dec: [d_sae, d_in]
        b_dec: [d_in]

    miStudio uses PyTorch nn.Linear convention:
        encoder.weight: [d_sae, d_in]
        encoder.bias: [d_sae]
        decoder.weight: [d_in, d_sae]
        decoder.bias: [d_in]
        decoder_bias: [d_in]

    Args:
        community_weights: Community Standard format weight dict
        tied_weights: Whether to use tied weights

    Returns:
        miStudio model state dict
    """
    mistudio_weights = {}

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
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model state dict
    state_dict = model.state_dict()

    # Convert to Community Standard format
    community_weights = convert_mistudio_to_community_weights(state_dict, tied_weights)

    # Save weights
    weights_path = output_dir / "sae_weights.safetensors"
    save_file(community_weights, str(weights_path))
    logger.info(f"Saved Community Standard weights to {weights_path}")

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

    return False


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
    2. miStudio checkpoint format (checkpoint.safetensors with model.* keys)

    Args:
        path: Path to SAE directory or checkpoint file
        device: Device to load tensors onto

    Returns:
        Tuple of (state_dict in miStudio format, config or None, format_type)
    """
    path = Path(path)

    # Check for Community Standard format first
    if is_community_format(path):
        state_dict, config, _ = load_sae_community_format(path, device)
        return state_dict, config, "community_standard"

    # Check for miStudio format
    if is_mistudio_format(path):
        checkpoint_path = path
        if path.is_dir():
            checkpoint_path = path / "checkpoint.safetensors"

        weights = load_file(str(checkpoint_path), device=device)

        # Extract model weights (remove "model." prefix)
        state_dict = {}
        for key, value in weights.items():
            if key.startswith("model."):
                state_dict[key.replace("model.", "")] = value

        return state_dict, None, "mistudio"

    raise ValueError(
        f"Unrecognized SAE format at {path}. "
        f"Expected either Community Standard format (cfg.json + sae_weights.safetensors) "
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
