"""
Forward hooks for extracting activations from transformer models.

This module provides functionality to register PyTorch forward hooks on transformer
layers to capture activations during inference. Supports hooks for:
- Residual stream (after layer norm)
- MLP outputs (after feed-forward layers)
- Attention outputs (after self-attention layers)

Architecture support is dynamic - any transformer with standard attention + MLP
blocks is supported via runtime introspection (see layer_discovery.py).
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

import torch
import torch.nn as nn
import numpy as np

from .layer_discovery import (
    discover_transformer_structure,
    get_hookable_module,
    TransformerStructure,
    find_attention_module,
)

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Types of forward hooks supported."""
    RESIDUAL = "residual"
    MLP = "mlp"
    #: The attention module's OUTPUT HIDDEN STATES — `output[0]`, shape
    #: [batch, seq, hidden]. Named "attention" for historical reasons; several
    #: callers depend on this meaning (see `workers/training_tasks.py`).
    ATTENTION = "attention"
    #: The attention PROBABILITIES — `output[1]`, shape [batch, heads, q, k].
    #: A separate type because `create_hook` discards index 1 by design, so
    #: ATTENTION can never yield them however it is pointed.
    ATTENTION_WEIGHTS = "attention_weights"


class HookManager:
    """
    Manages forward hooks for activation extraction from transformer models.

    This class provides methods to register hooks on specific layers, collect
    activations during forward passes, and clean up hooks afterward.

    Uses dynamic layer discovery to support any transformer architecture without
    hardcoded mappings. See layer_discovery.py for introspection logic.

    Attributes:
        model: The PyTorch model to hook
        activations: Dictionary storing captured activations by layer name
        hooks: List of registered hook handles for cleanup
        structure: Discovered transformer structure (populated on first register_hooks call)
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the HookManager with a model.

        Args:
            model: PyTorch model to register hooks on
        """
        self.model = model
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        #: Names of the hooks registered, parallel to `hooks`. `register_hooks`
        #: only raises when NOTHING at all is registered, so on a second call —
        #: which is how attention capture is added on top of residual capture —
        #: a total failure to register is invisible: the residual hooks from the
        #: first call keep the list non-empty. Callers assert on the delta here.
        self.hook_names: List[str] = []
        self.structure: Optional[TransformerStructure] = None

    class AttentionWeightsUnavailable(RuntimeError):
        """The hooked module produced no attention probabilities.

        Overwhelmingly this means the model is running SDPA or flash attention,
        whose kernels never materialise the probability matrix — transformers'
        `sdpa_attention_forward` literally returns `(attn_output, None)`. The
        fix is to load the model with `attn_implementation="eager"`.

        RAISED, NOT LOGGED. The failure this replaces wrote an empty attention
        sidecar that looked like a completed capture, and the only symptom
        downstream was an analysis with nothing in it.
        """

    def create_attention_weights_hook(self, layer_name: str):
        """Capture `output[1]` — the [batch, heads, q, k] probabilities.

        `create_hook` below takes `output[0]` and drops the rest, under a comment
        that names attention weights as the thing being dropped. That is correct
        for hidden-state capture and useless here, which is why this is a
        separate function rather than a flag on that one.

        Kept on-device and NOT moved to CPU: the caller consumes each batch's
        weights within the same loop iteration, and a 128 MiB device-to-host copy
        per layer per batch would dominate the capture.
        """
        def hook_fn(module: nn.Module, input: Tuple[torch.Tensor, ...], output) -> None:
            if not isinstance(output, tuple) or len(output) < 2:
                raise HookManager.AttentionWeightsUnavailable(
                    f"{layer_name}: hooked module {type(module).__name__} returned "
                    f"{type(output).__name__}, not an (output, weights) tuple — it is "
                    "probably a convolution or SSM mixer rather than attention"
                )
            weights = output[1]
            if weights is None:
                raise HookManager.AttentionWeightsUnavailable(
                    f"{layer_name}: {type(module).__name__} returned no attention "
                    "probabilities. SDPA and flash-attention kernels never produce "
                    "them; load the model with attn_implementation=\"eager\"."
                )
            self.activations.setdefault(layer_name, []).append(weights.detach())

        return hook_fn

    def create_hook(self, layer_name: str) -> Callable:
        """
        Create a forward hook function that captures outputs.

        The hook captures the output tensor, detaches it from the computation graph,
        moves it to CPU, and stores it in the activations dictionary.

        Args:
            layer_name: Name identifier for this layer's activations

        Returns:
            Hook function with signature (module, input, output) -> None
        """
        def hook_fn(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            """Forward hook that captures layer output."""
            # Handle tuple outputs (some layers return (tensor, attention_weights))
            if isinstance(output, tuple):
                output = output[0]

            # Detach from computation graph and move to CPU
            activation = output.detach().cpu()

            # Initialize list for this layer if needed
            if layer_name not in self.activations:
                self.activations[layer_name] = []

            # Store activation
            self.activations[layer_name].append(activation)

        return hook_fn

    def register_hooks(
        self,
        layer_indices: List[int],
        hook_types: List[HookType],
        architecture: str
    ) -> None:
        """
        Register hooks on specified layers and hook types.

        Uses dynamic layer discovery to identify transformer layers automatically.
        The architecture parameter is used as a hint for logging but does not
        restrict which models can be used.

        Args:
            layer_indices: List of transformer layer indices to hook (e.g., [0, 5, 10])
            hook_types: List of hook types to register (RESIDUAL, MLP, ATTENTION)
            architecture: Model architecture name (used as hint, not for validation)

        Raises:
            ValueError: If transformer layers cannot be found or no hooks register
        """
        logger.info(f"Registering hooks for architecture={architecture}, layers={layer_indices}, types={hook_types}")

        # Discover transformer structure dynamically
        self.structure = discover_transformer_structure(self.model, architecture_hint=architecture)
        layers_module = self.structure.layers_module

        logger.info(
            f"Discovered {self.structure.num_layers} layers at '{self.structure.layers_path}'. "
            f"Layer components: attention={self.structure.attention_module}, "
            f"mlp={self.structure.mlp_module}, norm={self.structure.residual_norm_module}"
        )

        # Register hooks for each specified layer
        for layer_idx in layer_indices:
            if layer_idx >= len(layers_module):
                logger.warning(f"Layer index {layer_idx} exceeds model depth {len(layers_module)}, skipping")
                continue

            layer = layers_module[layer_idx]

            # Register each requested hook type
            for hook_type in hook_types:
                self._register_hook_for_layer(layer, layer_idx, hook_type, architecture)

        logger.info(f"Registered {len(self.hooks)} hooks total")

        # CRITICAL: Fail if no hooks were registered
        # This prevents silent failures where extraction "completes" but no activations are captured
        if len(self.hooks) == 0:
            raise ValueError(
                f"No hooks were successfully registered for architecture '{architecture}'. "
                f"Requested layers: {layer_indices}, hook_types: {[ht.value for ht in hook_types]}. "
                f"This usually means the model's layer structure doesn't match expected patterns. "
                f"Check that the architecture is correctly detected and layer modules exist."
            )

    # NOTE: _get_layers_module() has been removed.
    # Layer discovery is now handled by layer_discovery.discover_transformer_structure()
    # which supports any transformer architecture without hardcoded mappings.

    def _register_hook_for_layer(
        self,
        layer: nn.Module,
        layer_idx: int,
        hook_type: HookType,
        architecture: str
    ) -> None:
        """
        Register a single hook on a specific layer.

        Uses dynamic discovery to find the appropriate module to hook.

        Args:
            layer: The transformer layer module
            layer_idx: Index of the layer
            hook_type: Type of hook to register
            architecture: Model architecture name (used for logging only)
        """
        hook_name = f"layer_{layer_idx}_{hook_type.value}"

        if hook_type is HookType.ATTENTION_WEIGHTS:
            # STRICT resolution, not `get_hookable_module`. That routes through
            # ATTENTION_PATTERNS, which includes "conv" for LFM2's sequence
            # mixer — a module with no attention probabilities at all. Pointing
            # a weights hook at one produces a confusing error deep in a batch
            # instead of a clear refusal here.
            module_to_hook = find_attention_module(layer)
            if module_to_hook is None:
                logger.warning(
                    f"Layer {layer_idx} has no attention module, so attention "
                    "weights cannot be captured from it"
                )
                return
            hook_fn = self.create_attention_weights_hook(hook_name)
        else:
            # Use dynamic discovery to find the module
            module_to_hook = get_hookable_module(layer, hook_type.value, self.structure)
            if module_to_hook is None:
                logger.warning(f"Could not find module for {hook_type.value} hook on layer {layer_idx}")
                return
            hook_fn = self.create_hook(hook_name)

        # Register and store handle
        handle = module_to_hook.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        self.hook_names.append(hook_name)

        logger.debug(f"Registered {hook_type.value} hook on layer {layer_idx}")

    # NOTE: _get_module_for_hook_type() has been removed.
    # Hook module discovery is now handled by layer_discovery.get_hookable_module()
    # which dynamically finds modules based on common naming patterns.

    def clear_activations(self) -> None:
        """
        Clear all stored activations and explicitly delete tensor references.

        MEMORY FIX: Explicitly delete tensors before clearing the dict to ensure
        Python GC can immediately free GPU memory instead of waiting for dict cleanup.
        """
        # Explicitly delete all tensors in each layer's activation list
        for layer_name in list(self.activations.keys()):
            activation_list = self.activations[layer_name]
            for i in range(len(activation_list)):
                del activation_list[i]
            activation_list.clear()

        # Now clear the dict
        self.activations.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks and clear activations."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.hook_names.clear()
        self.activations.clear()
        logger.info("Removed all hooks and cleared activations")

    def get_activations_as_numpy(self) -> Dict[str, np.ndarray]:
        """
        Get all captured activations as numpy arrays.

        Concatenates activations from multiple forward passes along batch dimension.

        Returns:
            Dictionary mapping layer names to numpy arrays of shape
            [num_samples, seq_len, hidden_dim]
        """
        numpy_activations = {}

        for layer_name, activation_list in self.activations.items():
            if not activation_list:
                continue

            # Stack tensors along batch dimension
            stacked = torch.cat(activation_list, dim=0)

            # Convert to numpy
            numpy_activations[layer_name] = stacked.numpy()

        return numpy_activations

    def __enter__(self) -> "HookManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - automatically remove hooks."""
        self.remove_hooks()
