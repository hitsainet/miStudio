"""
Forward hooks for extracting activations from transformer models.

This module provides functionality to register PyTorch forward hooks on transformer
layers to capture activations during inference. Supports hooks for:
- Residual stream (after layer norm)
- MLP outputs (after feed-forward layers)
- Attention outputs (after self-attention layers)
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Types of forward hooks supported."""
    RESIDUAL = "residual"
    MLP = "mlp"
    ATTENTION = "attention"


class HookManager:
    """
    Manages forward hooks for activation extraction from transformer models.

    This class provides methods to register hooks on specific layers, collect
    activations during forward passes, and clean up hooks afterward.

    Attributes:
        model: The PyTorch model to hook
        activations: Dictionary storing captured activations by layer name
        hooks: List of registered hook handles for cleanup
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

        This method identifies the appropriate module paths for each architecture
        and registers hooks based on the requested types.

        Args:
            layer_indices: List of transformer layer indices to hook (e.g., [0, 5, 10])
            hook_types: List of hook types to register (RESIDUAL, MLP, ATTENTION)
            architecture: Model architecture name (llama, gpt2, etc.)

        Raises:
            ValueError: If architecture is not supported or layers don't exist
        """
        logger.info(f"Registering hooks for architecture={architecture}, layers={layer_indices}, types={hook_types}")

        # Get the transformer layers container for this architecture
        layers_module = self._get_layers_module(architecture)

        if layers_module is None:
            raise ValueError(f"Could not find transformer layers for architecture: {architecture}")

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

    def _get_layers_module(self, architecture: str) -> Optional[nn.ModuleList]:
        """
        Get the transformer layers container for a given architecture.

        Args:
            architecture: Model architecture name

        Returns:
            ModuleList containing transformer layers, or None if not found
        """
        architecture = architecture.lower()

        # Common paths for different architectures
        # LFM2 (LiquidAI) uses Llama-style architecture with model.model.layers
        if architecture in ["llama", "mistral", "mixtral", "gemma", "gemma2", "gemma3", "lfm2"]:
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                return self.model.model.layers
        elif architecture == "gpt2":
            if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                return self.model.transformer.h
        elif architecture in ["gpt_neox", "pythia"]:
            if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
                return self.model.gpt_neox.layers
        elif architecture in ["phi", "phi3", "phi3_v"]:
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                return self.model.model.layers
        elif architecture in ["qwen", "qwen2", "qwen3"]:
            # Qwen models can have layers at transformer.h (Qwen/Qwen2) or model.layers (Qwen3)
            # Try both paths to handle different model variants and quantization wrappers
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                return self.model.model.layers
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                return self.model.transformer.h
            # Log available attributes for debugging
            logger.warning(f"Qwen model structure - model attrs: {dir(self.model)[:10]}")
            if hasattr(self.model, "model"):
                logger.warning(f"Qwen model structure - model.model attrs: {dir(self.model.model)[:10]}")
            if hasattr(self.model, "transformer"):
                logger.warning(f"Qwen model structure - model.transformer attrs: {dir(self.model.transformer)[:10]}")
        elif architecture == "falcon":
            if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                return self.model.transformer.h

        return None

    def _register_hook_for_layer(
        self,
        layer: nn.Module,
        layer_idx: int,
        hook_type: HookType,
        architecture: str
    ) -> None:
        """
        Register a single hook on a specific layer.

        Args:
            layer: The transformer layer module
            layer_idx: Index of the layer
            hook_type: Type of hook to register
            architecture: Model architecture name
        """
        module_to_hook = self._get_module_for_hook_type(layer, hook_type, architecture)

        if module_to_hook is None:
            logger.warning(f"Could not find module for {hook_type.value} hook on layer {layer_idx}")
            return

        # Create hook with descriptive name
        hook_name = f"layer_{layer_idx}_{hook_type.value}"
        hook_fn = self.create_hook(hook_name)

        # Register and store handle
        handle = module_to_hook.register_forward_hook(hook_fn)
        self.hooks.append(handle)

        logger.debug(f"Registered {hook_type.value} hook on layer {layer_idx}")

    def _get_module_for_hook_type(
        self,
        layer: nn.Module,
        hook_type: HookType,
        architecture: str
    ) -> Optional[nn.Module]:
        """
        Get the specific module to hook for a given hook type.

        Args:
            layer: The transformer layer
            hook_type: Type of hook (RESIDUAL, MLP, ATTENTION)
            architecture: Model architecture name

        Returns:
            Module to register hook on, or None if not found
        """
        architecture = architecture.lower()

        if hook_type == HookType.RESIDUAL:
            # Hook after the final layer norm (before residual addition)
            if hasattr(layer, "post_attention_layernorm"):
                return layer.post_attention_layernorm  # Llama-style
            elif hasattr(layer, "ln_2"):
                return layer.ln_2  # GPT-2 style
            elif hasattr(layer, "operator_norm"):
                return layer.operator_norm  # LFM2 (LiquidAI) style
            elif hasattr(layer, "ffn_norm"):
                return layer.ffn_norm  # LFM2 alternative - pre-FFN norm

        elif hook_type == HookType.MLP:
            # Hook the MLP module output
            if hasattr(layer, "mlp"):
                return layer.mlp
            elif hasattr(layer, "feed_forward"):
                return layer.feed_forward

        elif hook_type == HookType.ATTENTION:
            # Hook the attention module output
            if hasattr(layer, "self_attn"):
                return layer.self_attn  # Llama-style
            elif hasattr(layer, "attn"):
                return layer.attn  # GPT-2 style
            elif hasattr(layer, "conv"):
                return layer.conv  # LFM2 (LiquidAI) convolution layers

        # Log available attributes to help debug architecture mismatches
        layer_attrs = [attr for attr in dir(layer) if not attr.startswith('_')]
        logger.warning(
            f"Could not find module for {hook_type.value} hook on {architecture}. "
            f"Available layer attributes: {layer_attrs[:20]}..."  # First 20 to avoid log spam
        )
        return None

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
