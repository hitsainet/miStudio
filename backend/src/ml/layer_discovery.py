"""
Dynamic transformer layer discovery.

This module provides architecture-agnostic detection of transformer layers
by introspecting model structure at runtime rather than relying on hardcoded
architecture mappings.

The key insight is that all transformer architectures share the same fundamental
structure: a sequence of blocks containing attention and MLP/feed-forward modules.
The only differences are naming conventions.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)


# Common naming patterns across transformer architectures
LAYER_CONTAINER_PATTERNS = [
    # (parent_attr, container_attr) tuples to try in order
    ("model", "layers"),          # Llama, Mistral, Gemma, Phi, Qwen3
    ("transformer", "h"),         # GPT-2, Falcon, older Qwen
    ("transformer", "layers"),    # Some variants
    ("gpt_neox", "layers"),       # GPT-NeoX, Pythia
    ("encoder", "layer"),         # BERT-style (for future)
    ("decoder", "layers"),        # T5-style decoder
    ("", "layers"),               # Direct model.layers
    ("", "h"),                    # Direct model.h
]

# ORDERED TUPLES, NOT SETS. Every one of these is consumed by a
# first-match-wins loop (`for pattern in ...: ... break`), so the order decides
# which module gets hooked when a layer has more than one candidate — and a set
# has no order. Python randomises string hashing per process, so as sets these
# genuinely returned different answers in different workers:
#
#     PYTHONHASHSEED=1 -> input_layernorm
#     PYTHONHASHSEED=2 -> post_attention_layernorm
#
# on the same Llama-style layer. That is the residual point activations are
# captured at, so an SAE could be trained on one point by an extraction worker
# and fed the other by a capture worker, with nothing anywhere reporting a
# disagreement. The comments below always described a preference order; these
# are now actually ordered by it.

# Patterns that indicate an attention module
ATTENTION_PATTERNS = (
    "self_attn",       # Llama, Mistral, Gemma, Phi
    "attention",       # Generic
    "attn",            # GPT-2, Falcon
    "self_attention",  # Some variants
    "mha",             # Multi-head attention
    "conv",            # LFM2 uses convolution for sequence mixing
)

# Patterns that indicate an MLP/feed-forward module
MLP_PATTERNS = (
    "mlp",             # Most modern architectures
    "feed_forward",    # Some variants
    "ffn",             # Feed-forward network
    "dense",           # Some variants
    "ff",              # Abbreviated
    "intermediate",    # BERT-style
)

# Patterns for layer normalization (used for residual stream hooks).
# Most-specific and most-preferred first: a post-attention norm is the residual
# point this project means by "residual", and `input_layernorm` is last because
# it is the pre-attention point and only a fallback.
LAYER_NORM_PATTERNS = (
    # Post-attention norms (preferred for residual stream)
    "post_attention_layernorm",  # Llama-style
    "ln_2",                       # GPT-2 style
    "post_layernorm",             # Generic
    "final_layer_norm",           # Some variants
    # Pre-FFN norms
    "ffn_norm",                   # LFM2
    "pre_mlp_layernorm",          # Some variants
    # Other norms
    "operator_norm",              # LFM2
    "layer_norm",                 # Generic
    "ln_1",                       # GPT-2 pre-attention
    "input_layernorm",            # Llama pre-attention
)


@dataclass
class TransformerStructure:
    """Describes the discovered structure of a transformer model."""

    # Path to layers container (e.g., "model.layers" or "transformer.h")
    layers_path: str

    # The actual ModuleList containing transformer layers
    layers_module: nn.ModuleList

    # Number of layers discovered
    num_layers: int

    # Discovered module names within each layer
    attention_module: Optional[str] = None
    mlp_module: Optional[str] = None
    residual_norm_module: Optional[str] = None

    # All discovered layer attributes (for debugging)
    layer_attributes: List[str] = field(default_factory=list)

    # Architecture hint from config (if available)
    architecture_hint: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"TransformerStructure(\n"
            f"  layers_path='{self.layers_path}',\n"
            f"  num_layers={self.num_layers},\n"
            f"  attention='{self.attention_module}',\n"
            f"  mlp='{self.mlp_module}',\n"
            f"  residual_norm='{self.residual_norm_module}'\n"
            f")"
        )


def _get_nested_attr(obj: object, path: str) -> Optional[object]:
    """
    Get a nested attribute using dot notation.

    Args:
        obj: Root object
        path: Dot-separated attribute path (e.g., "model.layers")

    Returns:
        The nested attribute, or None if any part doesn't exist
    """
    if not path:
        return obj

    parts = path.split(".")
    current = obj

    for part in parts:
        if not part:
            continue
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None

    return current


def _find_matching_attr(module: nn.Module, patterns: Sequence[str]) -> Optional[str]:
    """
    Find the attribute matching the EARLIEST pattern in `patterns`.

    MIS-E2E-087. This used to loop `dir(module)` and test membership in
    `patterns`. `dir()` is sorted alphabetically, so the caller's carefully
    ordered preference list was discarded and whichever attribute sorted first
    won. On a Llama-style layer that means `input_layernorm` — the PRE-attention
    norm — is returned where `LAYER_NORM_PATTERNS` asks for
    `post_attention_layernorm`, the residual-stream point. The docstring here
    even asserted that order does not matter, which is how it survived.

    `_is_transformer_layer` (below) already iterates the patterns in order, so
    the two functions disagreed about the same model. They now agree.

    Args:
        module: Module to search
        patterns: Attribute-name patterns, MOST PREFERRED FIRST.

    Returns:
        Matching attribute name, or None if no match
    """
    # Map lower-cased attribute name -> real name, so a pattern can be matched
    # against the module without another pass over dir() per pattern.
    available = {name.lower(): name for name in dir(module)}
    for pattern in patterns:
        attr_name = available.get(pattern.lower())
        if attr_name is None:
            continue
        attr = getattr(module, attr_name, None)
        if attr is not None and isinstance(attr, nn.Module):
            return attr_name
    return None


def _is_transformer_layer(module: nn.Module) -> Tuple[bool, Dict[str, Optional[str]]]:
    """
    Check if a module looks like a transformer layer by examining its children.

    A transformer layer typically has:
    - An attention mechanism (self_attn, attn, attention, etc.)
    - An MLP/feed-forward network (mlp, feed_forward, ffn, etc.)

    Args:
        module: Module to check

    Returns:
        Tuple of (is_transformer_layer, detected_modules_dict)
    """
    children = {name.lower(): name for name in module._modules.keys()}

    detected = {
        "attention": None,
        "mlp": None,
        "residual_norm": None,
    }

    # Helper to check if a name matches a pattern
    # Excludes false positives like "attention" matching "post_attention_layernorm"
    def matches_pattern(child_lower: str, pattern: str, exclude_suffixes: Set[str] = None) -> bool:
        """Check if child matches pattern, excluding false positives."""
        if exclude_suffixes is None:
            exclude_suffixes = set()

        # Exact match is always good
        if child_lower == pattern:
            return True

        # Check if pattern appears in the name
        if pattern not in child_lower:
            return False

        # Exclude names that end with certain suffixes (e.g., layernorm, norm)
        for suffix in exclude_suffixes:
            if child_lower.endswith(suffix):
                return False

        return True

    # Check for attention module (exclude names ending with norm/layernorm)
    norm_suffixes = {"norm", "layernorm", "layer_norm", "_norm"}
    for pattern in ATTENTION_PATTERNS:
        for child_lower, child_actual in children.items():
            if matches_pattern(child_lower, pattern, norm_suffixes):
                detected["attention"] = child_actual
                break
        if detected["attention"]:
            break

    # Check for MLP module (exclude names ending with norm/layernorm)
    for pattern in MLP_PATTERNS:
        for child_lower, child_actual in children.items():
            if matches_pattern(child_lower, pattern, norm_suffixes):
                detected["mlp"] = child_actual
                break
        if detected["mlp"]:
            break

    # Check for layer norm (for residual stream hooks)
    # For norms, we WANT names with norm in them
    for pattern in LAYER_NORM_PATTERNS:
        for child_lower, child_actual in children.items():
            if pattern in child_lower:
                detected["residual_norm"] = child_actual
                break
        if detected["residual_norm"]:
            break

    # A transformer layer should have at least attention OR mlp
    # (some architectures may have variants)
    is_transformer = detected["attention"] is not None or detected["mlp"] is not None

    return is_transformer, detected


def resolve_vocab_size(model: nn.Module) -> Optional[int]:
    """This model's vocabulary size, wherever the config happens to keep it.

    Reported live: two extraction jobs on `gemma-4-12B-it` died with

        Extraction failed: 'Gemma4UnifiedConfig' object has no attribute
        'vocab_size'

    `model.config.vocab_size` is not universal. Unified and multimodal configs
    keep the text-model fields on a SUB-CONFIG — `text_config`, `llm_config`,
    `decoder`, depending on the family — and the top level carries only the
    composition. Reading the attribute directly is the same class of assumption
    this module exists to remove: it names one architecture's layout and fails
    closed on every other.

    Three sources, in order of how specific they are:

      1. `config.vocab_size` — right for the usual decoder-only case.
      2. Any sub-config that has one. Discovered by walking the config's own
         attributes rather than naming `text_config`, so a family this shop has
         not run yet is covered without an edit.
      3. The INPUT EMBEDDING's row count, which is ground truth: a token id is
         usable if and only if the embedding table has a row for it, whatever
         the config says. This is the fallback that cannot be wrong, and it is
         the bound the token-id validation actually needs.

    Returns None only when the model has no embedding table to ask, in which
    case the caller must decide — silently substituting a default would put a
    made-up bound on a real validation check.
    """
    config = getattr(model, "config", None)

    if config is not None:
        direct = getattr(config, "vocab_size", None)
        if isinstance(direct, int) and direct > 0:
            return direct

        # Sub-configs, discovered rather than named.
        for attr in dir(config):
            if attr.startswith("_"):
                continue
            try:
                child = getattr(config, attr)
            except Exception:
                continue
            if child is None or child is config or isinstance(child, (str, bytes)):
                continue
            nested = getattr(child, "vocab_size", None)
            if isinstance(nested, int) and nested > 0:
                logger.info(
                    "vocab_size resolved from config.%s (%s has none at the top "
                    "level)", attr, type(config).__name__,
                )
                return nested

    # Ground truth.
    try:
        embeddings = model.get_input_embeddings()
    except Exception:
        embeddings = None
    if embeddings is not None:
        rows = getattr(embeddings, "num_embeddings", None)
        if isinstance(rows, int) and rows > 0:
            logger.info("vocab_size resolved from the input embedding table (%d)", rows)
            return rows
        weight = getattr(embeddings, "weight", None)
        if weight is not None and getattr(weight, "ndim", 0) >= 1:
            return int(weight.shape[0])

    return None


def discover_transformer_structure(
    model: nn.Module,
    architecture_hint: Optional[str] = None
) -> TransformerStructure:
    """
    Dynamically discover the transformer structure of a model.

    This function introspects the model to find:
    1. The container holding transformer layers (e.g., model.model.layers)
    2. The structure within each layer (attention, mlp, norms)

    Args:
        model: The PyTorch model to analyze
        architecture_hint: Optional hint from model config (e.g., "llama", "gpt2")
                          Used for logging and fallback only

    Returns:
        TransformerStructure describing the discovered model architecture

    Raises:
        ValueError: If no transformer layers can be found
    """
    logger.info(f"Discovering transformer structure (hint: {architecture_hint})")

    layers_module = None
    layers_path = None

    # Strategy 1: Try known container patterns
    for parent_attr, container_attr in LAYER_CONTAINER_PATTERNS:
        if parent_attr:
            path = f"{parent_attr}.{container_attr}"
            container = _get_nested_attr(model, path)
        else:
            path = container_attr
            container = getattr(model, container_attr, None)

        if container is not None and isinstance(container, nn.ModuleList) and len(container) > 0:
            # Verify this looks like transformer layers
            is_transformer, _ = _is_transformer_layer(container[0])
            if is_transformer:
                layers_module = container
                layers_path = path
                logger.info(f"Found transformer layers at '{path}' ({len(container)} layers)")
                break

    # Strategy 2: Deep search through all modules for ModuleList containing transformer layers
    if layers_module is None:
        logger.info("Standard paths didn't work, performing deep search...")
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                is_transformer, _ = _is_transformer_layer(module[0])
                if is_transformer:
                    layers_module = module
                    layers_path = name
                    logger.info(f"Deep search found transformer layers at '{name}' ({len(module)} layers)")
                    break

    # Strategy 3: Look for sequential containers with transformer-like layers
    if layers_module is None:
        logger.info("ModuleList search didn't work, checking Sequential containers...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential) and len(module) > 0:
                is_transformer, _ = _is_transformer_layer(module[0])
                if is_transformer:
                    layers_module = module
                    layers_path = name
                    logger.info(f"Found transformer layers in Sequential at '{name}' ({len(module)} layers)")
                    break

    if layers_module is None:
        # Log model structure to help debug
        top_level_modules = list(model._modules.keys())
        logger.error(f"Could not find transformer layers. Top-level modules: {top_level_modules}")

        # Log one level deeper for debugging
        for name, child in model.named_children():
            child_modules = list(child._modules.keys()) if hasattr(child, '_modules') else []
            logger.error(f"  {name}: {child_modules[:10]}")

        raise ValueError(
            f"Could not discover transformer layers in model. "
            f"Architecture hint: {architecture_hint}. "
            f"Top-level modules: {top_level_modules}. "
            f"Please check if this model has a standard transformer architecture."
        )

    # Analyze the structure of the first layer
    first_layer = layers_module[0]
    _, detected_modules = _is_transformer_layer(first_layer)

    # Get all layer attributes for debugging
    layer_attrs = [attr for attr in dir(first_layer) if not attr.startswith('_')]

    structure = TransformerStructure(
        layers_path=layers_path,
        layers_module=layers_module,
        num_layers=len(layers_module),
        attention_module=detected_modules["attention"],
        mlp_module=detected_modules["mlp"],
        residual_norm_module=detected_modules["residual_norm"],
        layer_attributes=layer_attrs,
        architecture_hint=architecture_hint,
    )

    logger.info(f"Discovered structure: {structure}")

    return structure


def validate_model_for_extraction(model: nn.Module, architecture_hint: Optional[str] = None) -> TransformerStructure:
    """
    Validate that a model can be used for activation extraction.

    This replaces the old architecture whitelist approach with dynamic discovery.

    Args:
        model: The model to validate
        architecture_hint: Optional architecture name from config

    Returns:
        TransformerStructure if valid

    Raises:
        ValueError: If model structure cannot be determined
    """
    structure = discover_transformer_structure(model, architecture_hint)

    # Log warnings for missing components (but don't fail)
    if structure.attention_module is None:
        logger.warning(
            f"Could not identify attention module in layers. "
            f"Attention hooks may not work. Layer attrs: {structure.layer_attributes[:15]}"
        )

    if structure.mlp_module is None:
        logger.warning(
            f"Could not identify MLP module in layers. "
            f"MLP hooks may not work. Layer attrs: {structure.layer_attributes[:15]}"
        )

    if structure.residual_norm_module is None:
        logger.warning(
            f"Could not identify layer norm for residual hooks. "
            f"Residual hooks may fall back to layer output. Layer attrs: {structure.layer_attributes[:15]}"
        )

    return structure


def find_attention_module(layer: nn.Module) -> Optional[nn.Module]:
    """The layer's genuine self-attention module, or None if it has none.

    STRICTER THAN `ATTENTION_PATTERNS`, deliberately. That tuple includes
    `"conv"` because LFM2 uses a short convolution for sequence mixing, which is
    the right call for "what mixes across positions" — but a conv mixer has no
    attention probabilities, so anything reading `[batch, heads, q, k]` off it is
    reading something that does not exist. On a hybrid this is the difference
    between a layer that can be captured and one that cannot.

    Identified structurally rather than by name: a class whose name ends in
    "Attention", or a module carrying both `q_proj` and `k_proj`. That covers
    granite, llama, gemma, mistral, lfm2's attention layers and GPT-2, and
    excludes conv/SSM mixers without needing a list of their names.

    Returns the MODULE, not its attribute name, because the caller wants to hook
    it and the name is per-architecture noise.
    """
    for attr_name in dir(layer):
        if attr_name.startswith("__"):
            continue
        child = getattr(layer, attr_name, None)
        if child is None or not isinstance(child, nn.Module):
            continue
        if type(child).__name__.endswith("Attention"):
            return child
        if hasattr(child, "q_proj") and hasattr(child, "k_proj"):
            return child
    return None


def attention_layer_indices(layers_module: nn.ModuleList) -> List[int]:
    """Which layers actually have attention — checked PER LAYER.

    `discover_transformer_structure` inspects only layer 0 and applies that
    verdict model-wide, which is fine for a dense model and wrong for a hybrid:
    LFM2 has attention on 6 of 16 layers and granite-4.x MoE variants interleave
    Mamba blocks. Anything that needs to know whether a SPECIFIC layer can be
    attention-captured has to look at that layer.
    """
    return [i for i, layer in enumerate(layers_module)
            if find_attention_module(layer) is not None]


def get_hookable_module(
    layer: nn.Module,
    hook_type: str,
    structure: TransformerStructure
) -> Optional[nn.Module]:
    """
    Get the module to hook for a given hook type using discovered structure.

    Args:
        layer: The transformer layer
        hook_type: One of "residual", "mlp", "attention"
        structure: The discovered transformer structure

    Returns:
        Module to hook, or None if not found
    """
    if hook_type == "residual":
        # Try the discovered norm module first
        if structure.residual_norm_module:
            module = getattr(layer, structure.residual_norm_module, None)
            if module is not None:
                return module

        # Fallback: search for any layer norm
        norm_module = _find_matching_attr(layer, LAYER_NORM_PATTERNS)
        if norm_module:
            return getattr(layer, norm_module)

        # Last resort: return the layer itself (hook its output)
        return layer

    elif hook_type == "mlp":
        if structure.mlp_module:
            module = getattr(layer, structure.mlp_module, None)
            if module is not None:
                return module

        # Fallback search
        mlp_attr = _find_matching_attr(layer, MLP_PATTERNS)
        if mlp_attr:
            return getattr(layer, mlp_attr)

        return None

    elif hook_type == "attention":
        if structure.attention_module:
            module = getattr(layer, structure.attention_module, None)
            if module is not None:
                return module

        # Fallback search
        attn_attr = _find_matching_attr(layer, ATTENTION_PATTERNS)
        if attn_attr:
            return getattr(layer, attn_attr)

        return None

    else:
        logger.warning(f"Unknown hook type: {hook_type}")
        return None
