"""
TransformerLens hook name and model ID mapping utilities.

This module provides utilities to map between miStudio/HuggingFace naming conventions
and TransformerLens naming conventions used by Neuronpedia and other tools.

TransformerLens Hook Naming:
    blocks.{layer}.hook_resid_pre    - Residual stream before attention
    blocks.{layer}.hook_resid_post   - Residual stream after MLP (most common)
    blocks.{layer}.hook_mlp_out      - MLP output
    blocks.{layer}.attn.hook_z       - Attention output before projection
    blocks.{layer}.attn.hook_result  - Attention output after projection

Model Naming (HuggingFace → TransformerLens):
    openai-community/gpt2       → gpt2-small
    google/gemma-2-2b           → gemma-2-2b
    EleutherAI/pythia-70m       → pythia-70m
    meta-llama/Llama-3.1-8B     → llama-3.1-8b
"""

import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# Mapping from HuggingFace model IDs to TransformerLens model IDs
HF_TO_TRANSFORMERLENS_MODEL_ID: Dict[str, str] = {
    # GPT-2 family
    "openai-community/gpt2": "gpt2-small",
    "gpt2": "gpt2-small",
    "openai-community/gpt2-medium": "gpt2-medium",
    "gpt2-medium": "gpt2-medium",
    "openai-community/gpt2-large": "gpt2-large",
    "gpt2-large": "gpt2-large",
    "openai-community/gpt2-xl": "gpt2-xl",
    "gpt2-xl": "gpt2-xl",

    # Gemma family
    "google/gemma-2b": "gemma-2b",
    "google/gemma-2b-it": "gemma-2b-it",
    "google/gemma-7b": "gemma-7b",
    "google/gemma-7b-it": "gemma-7b-it",
    "google/gemma-2-2b": "gemma-2-2b",
    "google/gemma-2-2b-it": "gemma-2-2b-it",
    "google/gemma-2-9b": "gemma-2-9b",
    "google/gemma-2-9b-it": "gemma-2-9b-it",
    "google/gemma-2-27b": "gemma-2-27b",
    "google/gemma-2-27b-it": "gemma-2-27b-it",

    # Llama family
    "meta-llama/Llama-2-7b-hf": "llama-2-7b",
    "meta-llama/Llama-2-7b-chat-hf": "llama-2-7b-chat",
    "meta-llama/Llama-2-13b-hf": "llama-2-13b",
    "meta-llama/Llama-3.1-8B": "llama-3.1-8b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b-instruct",
    "meta-llama/Meta-Llama-3.1-8B": "llama-3.1-8b",
    "meta-llama/Llama-3.2-1B": "llama-3.2-1b",
    "meta-llama/Llama-3.2-3B": "llama-3.2-3b",

    # Pythia family (EleutherAI)
    "EleutherAI/pythia-14m": "pythia-14m",
    "EleutherAI/pythia-31m": "pythia-31m",
    "EleutherAI/pythia-70m": "pythia-70m",
    "EleutherAI/pythia-160m": "pythia-160m",
    "EleutherAI/pythia-410m": "pythia-410m",
    "EleutherAI/pythia-1b": "pythia-1b",
    "EleutherAI/pythia-1.4b": "pythia-1.4b",
    "EleutherAI/pythia-2.8b": "pythia-2.8b",
    "EleutherAI/pythia-6.9b": "pythia-6.9b",
    "EleutherAI/pythia-12b": "pythia-12b",

    # GPT-Neo family (EleutherAI)
    "EleutherAI/gpt-neo-125M": "gpt-neo-125m",
    "EleutherAI/gpt-neo-1.3B": "gpt-neo-1.3b",
    "EleutherAI/gpt-neo-2.7B": "gpt-neo-2.7b",
    "EleutherAI/gpt-j-6B": "gpt-j-6b",
    "EleutherAI/gpt-neox-20b": "gpt-neox-20b",

    # Mistral family
    "mistralai/Mistral-7B-v0.1": "mistral-7b",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral-7b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistral-7b-instruct-v0.2",

    # Phi family
    "microsoft/phi-1": "phi-1",
    "microsoft/phi-1_5": "phi-1.5",
    "microsoft/phi-2": "phi-2",

    # Other models
    "facebook/opt-125m": "opt-125m",
    "facebook/opt-350m": "opt-350m",
    "facebook/opt-1.3b": "opt-1.3b",
    "bigscience/bloom-560m": "bloom-560m",
    "bigscience/bloom-1b1": "bloom-1b1",
    "roneneldan/TinyStories-33M": "tiny-stories-33m",
    "roneneldan/TinyStories-1M": "tiny-stories-1m",
}


# Valid hook types for TransformerLens
VALID_HOOK_TYPES = {
    "resid_pre",   # Residual stream before attention
    "resid_post",  # Residual stream after MLP (most common for SAEs)
    "mlp_out",     # MLP output
    "attn_z",      # Attention output (before projection)
    "attn_out",    # Attention output (after projection)
    "attn_result", # Alias for attn_out
}


def get_transformerlens_hook_name(
    layer: int,
    hook_type: str = "resid_post"
) -> str:
    """
    Generate TransformerLens hook name from layer index and hook type.

    Args:
        layer: Layer index (0-indexed)
        hook_type: Type of hook point:
            - "resid_pre": Residual stream before attention
            - "resid_post": Residual stream after MLP (default, most common)
            - "mlp_out": MLP output
            - "attn_z": Attention output (before projection)
            - "attn_out" or "attn_result": Attention output (after projection)

    Returns:
        TransformerLens hook name string

    Examples:
        >>> get_transformerlens_hook_name(12, "resid_post")
        'blocks.12.hook_resid_post'
        >>> get_transformerlens_hook_name(5, "attn_z")
        'blocks.5.attn.hook_z'
    """
    hook_type = hook_type.lower().strip()

    # Normalize hook type
    if hook_type == "attn_result":
        hook_type = "attn_out"

    # Validate hook type
    if hook_type not in VALID_HOOK_TYPES:
        logger.warning(
            f"Unknown hook_type '{hook_type}', defaulting to 'resid_post'. "
            f"Valid types: {VALID_HOOK_TYPES}"
        )
        hook_type = "resid_post"

    # Generate hook name
    hook_mapping = {
        "resid_pre": f"blocks.{layer}.hook_resid_pre",
        "resid_post": f"blocks.{layer}.hook_resid_post",
        "mlp_out": f"blocks.{layer}.hook_mlp_out",
        "attn_z": f"blocks.{layer}.attn.hook_z",
        "attn_out": f"blocks.{layer}.attn.hook_result",
    }

    return hook_mapping.get(hook_type, f"blocks.{layer}.hook_resid_post")


def parse_transformerlens_hook_name(hook_name: str) -> Tuple[int, str]:
    """
    Parse a TransformerLens hook name into layer and hook type.

    Args:
        hook_name: Hook name like "blocks.12.hook_resid_post"

    Returns:
        Tuple of (layer_index, hook_type)

    Raises:
        ValueError: If hook name cannot be parsed

    Examples:
        >>> parse_transformerlens_hook_name("blocks.12.hook_resid_post")
        (12, 'resid_post')
        >>> parse_transformerlens_hook_name("blocks.5.attn.hook_z")
        (5, 'attn_z')
    """
    parts = hook_name.split(".")

    if len(parts) < 3 or parts[0] != "blocks":
        raise ValueError(f"Cannot parse hook name: {hook_name}")

    try:
        layer = int(parts[1])
    except ValueError:
        raise ValueError(f"Cannot extract layer from hook name: {hook_name}")

    # Determine hook type from remaining parts
    if len(parts) == 3 and parts[2].startswith("hook_"):
        hook_type = parts[2].replace("hook_", "")
    elif len(parts) == 4 and parts[2] == "attn" and parts[3].startswith("hook_"):
        hook_type = f"attn_{parts[3].replace('hook_', '')}"
    else:
        raise ValueError(f"Cannot determine hook type from: {hook_name}")

    # Normalize
    if hook_type == "attn_result":
        hook_type = "attn_out"

    return layer, hook_type


def get_transformerlens_model_id(hf_model_name: str) -> str:
    """
    Map HuggingFace model ID to TransformerLens model ID.

    Args:
        hf_model_name: HuggingFace model name (e.g., "openai-community/gpt2")

    Returns:
        TransformerLens model ID (e.g., "gpt2-small")

    Examples:
        >>> get_transformerlens_model_id("openai-community/gpt2")
        'gpt2-small'
        >>> get_transformerlens_model_id("google/gemma-2-2b")
        'gemma-2-2b'
    """
    # Direct lookup
    if hf_model_name in HF_TO_TRANSFORMERLENS_MODEL_ID:
        return HF_TO_TRANSFORMERLENS_MODEL_ID[hf_model_name]

    # Try without organization prefix
    if "/" in hf_model_name:
        short_name = hf_model_name.split("/")[-1]
        if short_name in HF_TO_TRANSFORMERLENS_MODEL_ID:
            return HF_TO_TRANSFORMERLENS_MODEL_ID[short_name]

        # Try lowercase
        if short_name.lower() in HF_TO_TRANSFORMERLENS_MODEL_ID:
            return HF_TO_TRANSFORMERLENS_MODEL_ID[short_name.lower()]

    # Fallback: normalize the name (lowercase, replace underscores)
    normalized = hf_model_name.lower().replace("_", "-")
    if "/" in normalized:
        normalized = normalized.split("/")[-1]

    logger.warning(
        f"No TransformerLens mapping found for '{hf_model_name}', "
        f"using normalized name: '{normalized}'"
    )

    return normalized


def get_hf_model_id(transformerlens_model_id: str) -> Optional[str]:
    """
    Reverse lookup: TransformerLens model ID to HuggingFace model ID.

    Args:
        transformerlens_model_id: TransformerLens model ID (e.g., "gpt2-small")

    Returns:
        HuggingFace model ID or None if not found
    """
    # Reverse mapping
    for hf_id, tl_id in HF_TO_TRANSFORMERLENS_MODEL_ID.items():
        if tl_id == transformerlens_model_id:
            return hf_id
    return None


def infer_hook_type_from_sae_config(sae_config: Dict[str, Any]) -> str:
    """
    Infer hook type from SAE configuration.

    Attempts to determine the hook type from various config fields.

    Args:
        sae_config: SAE configuration dictionary

    Returns:
        Hook type string (e.g., "resid_post")
    """
    # Check for explicit hook_point field
    hook_point = sae_config.get("hook_point", "")
    if hook_point:
        try:
            _, hook_type = parse_transformerlens_hook_name(hook_point)
            return hook_type
        except ValueError:
            pass

    # Check for hook_type field
    hook_type = sae_config.get("hook_type", "")
    if hook_type and hook_type in VALID_HOOK_TYPES:
        return hook_type

    # Check architecture hints
    architecture = sae_config.get("architecture", "")

    # Transcoder SAEs typically hook MLP
    if "transcoder" in architecture.lower():
        return "mlp_out"

    # Attention SAEs
    if "attention" in architecture.lower() or "attn" in architecture.lower():
        return "attn_out"

    # Default: residual stream after MLP (most common for SAEs)
    return "resid_post"


def infer_model_name_from_config(sae_config: Dict[str, Any]) -> Optional[str]:
    """
    Infer model name from SAE configuration.

    Args:
        sae_config: SAE configuration dictionary

    Returns:
        Model name or None if cannot be determined
    """
    # Check various fields
    for field in ["model_name", "model", "model_id", "base_model"]:
        if field in sae_config and sae_config[field]:
            return sae_config[field]

    # Check d_model to infer Gemma model
    d_model = sae_config.get("d_in") or sae_config.get("d_model")
    if d_model:
        gemma_models = {
            2048: "google/gemma-2b",
            2304: "google/gemma-2-2b",
            3072: "google/gemma-7b",
            3584: "google/gemma-2-9b",
            4608: "google/gemma-2-27b",
        }
        if d_model in gemma_models:
            return gemma_models[d_model]

    return None


def build_neuronpedia_config(
    model_name: str,
    layer: int,
    d_in: int,
    d_sae: int,
    hook_type: str = "resid_post",
    architecture: str = "standard",
    activation_fn: str = "relu",
    context_size: int = 128,
    prepend_bos: bool = True,
    normalize_activations: str = "none",
    **extra_fields
) -> Dict[str, Any]:
    """
    Build a complete Neuronpedia-compatible cfg.json configuration.

    This produces the exact format expected by Neuronpedia and SAELens.

    Args:
        model_name: HuggingFace model name
        layer: Layer index
        d_in: Input dimension (model hidden size)
        d_sae: SAE dimension (number of features)
        hook_type: Hook type (resid_post, resid_pre, etc.)
        architecture: SAE architecture (standard, gated, jumprelu)
        activation_fn: Activation function (relu, topk, jumprelu)
        context_size: Context window size
        prepend_bos: Whether to prepend BOS token
        normalize_activations: Activation normalization mode
        **extra_fields: Additional fields to include

    Returns:
        Configuration dictionary
    """
    # Get TransformerLens IDs
    tl_model_id = get_transformerlens_model_id(model_name)
    hook_name = get_transformerlens_hook_name(layer, hook_type)

    config = {
        # Model info
        "model_name": tl_model_id,

        # Hook info
        "hook_name": hook_name,  # Neuronpedia-style field
        "hook_point": hook_name,  # SAELens-style field (same value)
        "hook_point_layer": layer,
        "hook_point_head_index": None,

        # Dimensions
        "d_in": d_in,
        "d_sae": d_sae,

        # Architecture
        "architecture": architecture,
        "activation_fn_str": activation_fn if activation_fn != "jumprelu" else "topk",
        "activation_fn_kwargs": {},

        # Training context
        "context_size": context_size,
        "prepend_bos": prepend_bos,
        "normalize_activations": normalize_activations,

        # Feature scaling (usually false for standard SAEs)
        "apply_b_dec_to_input": False,
        "finetuning_scaling_factor": False,

        # Data type
        "dtype": "float32",
        "device": "cpu",

        # Computed field
        "expansion_factor": d_sae / d_in if d_in > 0 else 0,
    }

    # Add JumpReLU-specific fields
    if architecture == "jumprelu":
        config["activation_fn_str"] = "relu"  # SAELens uses relu for jumprelu
        config["activation_fn_kwargs"] = {}

    # Add extra fields
    config.update(extra_fields)

    return config


def validate_neuronpedia_config(config: Dict[str, Any]) -> Tuple[bool, list]:
    """
    Validate a Neuronpedia configuration for completeness.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Required fields
    required = ["model_name", "hook_name", "d_in", "d_sae"]
    for field in required:
        if field not in config or config[field] is None:
            issues.append(f"Missing required field: {field}")

    # Validate hook_name format
    hook_name = config.get("hook_name", "")
    if hook_name:
        try:
            parse_transformerlens_hook_name(hook_name)
        except ValueError as e:
            issues.append(f"Invalid hook_name format: {e}")

    # Validate dimensions
    d_in = config.get("d_in", 0)
    d_sae = config.get("d_sae", 0)
    if d_in <= 0:
        issues.append(f"Invalid d_in: {d_in}")
    if d_sae <= 0:
        issues.append(f"Invalid d_sae: {d_sae}")

    # Warn about potential issues
    if d_sae < d_in:
        issues.append(f"Warning: d_sae ({d_sae}) < d_in ({d_in}), expansion factor < 1")

    return len(issues) == 0 or all("Warning" in i for i in issues), issues
