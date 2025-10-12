"""
Tests for forward hooks functionality.

This module tests the HookManager class which registers PyTorch forward hooks
on transformer models to capture activations during inference.

NO MOCKING - Uses real PyTorch models and actual forward passes.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List

from src.ml.forward_hooks import HookManager, HookType


# Test fixtures for real transformer models


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class SimpleAttention(nn.Module):
    """Simple attention module for testing."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Final projection
        out = self.out_proj(out)

        return out


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for testing (Llama-style)."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = SimpleAttention(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = SimpleMLP(hidden_dim)

    def forward(self, x):
        # Self-attention block with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        # MLP block with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class SimpleGPT2Layer(nn.Module):
    """Simple GPT-2 style transformer layer for testing."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = SimpleAttention(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = SimpleMLP(hidden_dim)

    def forward(self, x):
        # Attention block
        residual = x
        x = self.ln_1(x)
        x = self.attn(x)
        x = residual + x

        # MLP block
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class SimpleLlamaModel(nn.Module):
    """Simple Llama-style model for testing."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model = nn.ModuleDict({
            'layers': nn.ModuleList([
                SimpleTransformerLayer(hidden_dim) for _ in range(num_layers)
            ])
        })
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return logits


class SimpleGPT2Model(nn.Module):
    """Simple GPT-2 style model for testing."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.ModuleDict({
            'h': nn.ModuleList([
                SimpleGPT2Layer(hidden_dim) for _ in range(num_layers)
            ])
        })
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.wte(input_ids)
        for layer in self.transformer.h:
            x = layer(x)
        logits = self.lm_head(x)
        return logits


# Fixtures


@pytest.fixture
def llama_model():
    """Create a simple Llama-style model for testing."""
    model = SimpleLlamaModel(vocab_size=1000, hidden_dim=64, num_layers=3)
    model.eval()
    return model


@pytest.fixture
def gpt2_model():
    """Create a simple GPT-2 style model for testing."""
    model = SimpleGPT2Model(vocab_size=1000, hidden_dim=64, num_layers=3)
    model.eval()
    return model


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    # Batch of 2 sequences, length 10
    return torch.randint(0, 1000, (2, 10))


# Tests


class TestHookManager:
    """Test suite for HookManager class."""

    def test_hook_manager_initialization(self, llama_model):
        """Test HookManager initializes correctly."""
        hook_manager = HookManager(llama_model)

        assert hook_manager.model == llama_model
        assert hook_manager.activations == {}
        assert hook_manager.hooks == []

    def test_register_residual_hooks_llama(self, llama_model, sample_input):
        """Test registering residual hooks on Llama-style model."""
        with HookManager(llama_model) as hook_manager:
            # Register residual hooks on layers 0 and 2
            hook_manager.register_hooks(
                layer_indices=[0, 2],
                hook_types=[HookType.RESIDUAL],
                architecture="llama"
            )

            # Check hooks were registered
            assert len(hook_manager.hooks) == 2

            # Run forward pass
            with torch.no_grad():
                _ = llama_model(sample_input)

            # Check activations were captured
            assert "layer_0_residual" in hook_manager.activations
            assert "layer_2_residual" in hook_manager.activations

            # Check activation shapes (batch_size=2, seq_len=10, hidden_dim=64)
            for layer_name in ["layer_0_residual", "layer_2_residual"]:
                activations = hook_manager.activations[layer_name]
                assert len(activations) == 1  # One forward pass
                assert activations[0].shape == (2, 10, 64)

    def test_register_mlp_hooks_llama(self, llama_model, sample_input):
        """Test registering MLP hooks on Llama-style model."""
        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0, 1],
                hook_types=[HookType.MLP],
                architecture="llama"
            )

            assert len(hook_manager.hooks) == 2

            with torch.no_grad():
                _ = llama_model(sample_input)

            assert "layer_0_mlp" in hook_manager.activations
            assert "layer_1_mlp" in hook_manager.activations

            # MLP output should have same shape as residual stream
            for layer_name in ["layer_0_mlp", "layer_1_mlp"]:
                activations = hook_manager.activations[layer_name]
                assert activations[0].shape == (2, 10, 64)

    def test_register_attention_hooks_llama(self, llama_model, sample_input):
        """Test registering attention hooks on Llama-style model."""
        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[1],
                hook_types=[HookType.ATTENTION],
                architecture="llama"
            )

            assert len(hook_manager.hooks) == 1

            with torch.no_grad():
                _ = llama_model(sample_input)

            assert "layer_1_attention" in hook_manager.activations

            activations = hook_manager.activations["layer_1_attention"]
            assert activations[0].shape == (2, 10, 64)

    def test_register_multiple_hook_types(self, llama_model, sample_input):
        """Test registering multiple hook types simultaneously."""
        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0, 1],
                hook_types=[HookType.RESIDUAL, HookType.MLP, HookType.ATTENTION],
                architecture="llama"
            )

            # 2 layers × 3 hook types = 6 hooks
            assert len(hook_manager.hooks) == 6

            with torch.no_grad():
                _ = llama_model(sample_input)

            # Check all hooks captured activations
            expected_keys = [
                "layer_0_residual", "layer_0_mlp", "layer_0_attention",
                "layer_1_residual", "layer_1_mlp", "layer_1_attention"
            ]
            for key in expected_keys:
                assert key in hook_manager.activations
                assert len(hook_manager.activations[key]) == 1

    def test_register_hooks_gpt2(self, gpt2_model, sample_input):
        """Test registering hooks on GPT-2 style model."""
        with HookManager(gpt2_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0, 2],
                hook_types=[HookType.RESIDUAL, HookType.MLP],
                architecture="gpt2"
            )

            # 2 layers × 2 hook types = 4 hooks
            assert len(hook_manager.hooks) == 4

            with torch.no_grad():
                _ = gpt2_model(sample_input)

            # Check activations
            assert "layer_0_residual" in hook_manager.activations
            assert "layer_2_residual" in hook_manager.activations
            assert "layer_0_mlp" in hook_manager.activations
            assert "layer_2_mlp" in hook_manager.activations

    def test_multiple_forward_passes(self, llama_model):
        """Test that activations accumulate across multiple forward passes."""
        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0],
                hook_types=[HookType.RESIDUAL],
                architecture="llama"
            )

            # Run 3 forward passes with different batch sizes
            with torch.no_grad():
                _ = llama_model(torch.randint(0, 1000, (2, 10)))
                _ = llama_model(torch.randint(0, 1000, (3, 10)))
                _ = llama_model(torch.randint(0, 1000, (1, 10)))

            # Check we have 3 activation tensors
            activations = hook_manager.activations["layer_0_residual"]
            assert len(activations) == 3
            assert activations[0].shape == (2, 10, 64)
            assert activations[1].shape == (3, 10, 64)
            assert activations[2].shape == (1, 10, 64)

    def test_get_activations_as_numpy(self, llama_model):
        """Test converting activations to numpy arrays."""
        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0, 1],
                hook_types=[HookType.RESIDUAL],
                architecture="llama"
            )

            # Run multiple forward passes
            with torch.no_grad():
                _ = llama_model(torch.randint(0, 1000, (2, 10)))
                _ = llama_model(torch.randint(0, 1000, (3, 10)))

            # Convert to numpy
            numpy_activations = hook_manager.get_activations_as_numpy()

            # Check output
            assert isinstance(numpy_activations, dict)
            assert "layer_0_residual" in numpy_activations
            assert "layer_1_residual" in numpy_activations

            # Check shapes (should concatenate along batch dimension)
            # 2 + 3 = 5 samples total
            for layer_name in ["layer_0_residual", "layer_1_residual"]:
                arr = numpy_activations[layer_name]
                assert isinstance(arr, np.ndarray)
                assert arr.shape == (5, 10, 64)
                assert arr.dtype == np.float32

    def test_clear_activations(self, llama_model, sample_input):
        """Test clearing stored activations."""
        hook_manager = HookManager(llama_model)

        hook_manager.register_hooks(
            layer_indices=[0],
            hook_types=[HookType.RESIDUAL],
            architecture="llama"
        )

        with torch.no_grad():
            _ = llama_model(sample_input)

        # Verify activations were captured
        assert len(hook_manager.activations) > 0

        # Clear and verify
        hook_manager.clear_activations()
        assert len(hook_manager.activations) == 0

        # Cleanup
        hook_manager.remove_hooks()

    def test_context_manager_cleanup(self, llama_model, sample_input):
        """Test that context manager properly cleans up hooks."""
        initial_hooks_count = sum(
            1 for _ in llama_model.modules()
            if hasattr(_, '_forward_hooks') and len(_._forward_hooks) > 0
        )

        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0, 1, 2],
                hook_types=[HookType.RESIDUAL, HookType.MLP],
                architecture="llama"
            )

            with torch.no_grad():
                _ = llama_model(sample_input)

            # Hooks should be registered
            assert len(hook_manager.hooks) == 6

        # After context exit, hooks should be removed
        final_hooks_count = sum(
            1 for _ in llama_model.modules()
            if hasattr(_, '_forward_hooks') and len(_._forward_hooks) > 0
        )

        assert final_hooks_count == initial_hooks_count

    def test_invalid_layer_index(self, llama_model, caplog):
        """Test handling of invalid layer indices."""
        with HookManager(llama_model) as hook_manager:
            # Try to register hook on non-existent layer
            hook_manager.register_hooks(
                layer_indices=[0, 100],  # 100 is out of range
                hook_types=[HookType.RESIDUAL],
                architecture="llama"
            )

            # Should only register hook for valid layer 0
            assert len(hook_manager.hooks) == 1

            # Should log warning about invalid index
            assert "exceeds model depth" in caplog.text

    def test_invalid_architecture(self, llama_model):
        """Test handling of unsupported architecture."""
        with HookManager(llama_model) as hook_manager:
            with pytest.raises(ValueError, match="Could not find transformer layers"):
                hook_manager.register_hooks(
                    layer_indices=[0],
                    hook_types=[HookType.RESIDUAL],
                    architecture="unsupported_architecture"
                )

    def test_activation_detachment(self, llama_model, sample_input):
        """Test that activations are properly detached from computation graph."""
        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0],
                hook_types=[HookType.RESIDUAL],
                architecture="llama"
            )

            # Run forward pass WITH gradients enabled
            _ = llama_model(sample_input)

            # Activations should still be detached (no gradient info)
            activations = hook_manager.activations["layer_0_residual"][0]
            assert not activations.requires_grad
            assert activations.grad_fn is None

    def test_cpu_offloading(self, llama_model, sample_input):
        """Test that activations are moved to CPU."""
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llama_model = llama_model.to(device)
        sample_input = sample_input.to(device)

        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0],
                hook_types=[HookType.RESIDUAL],
                architecture="llama"
            )

            with torch.no_grad():
                _ = llama_model(sample_input)

            # Activations should be on CPU even if model is on GPU
            activations = hook_manager.activations["layer_0_residual"][0]
            assert activations.device.type == "cpu"

    def test_different_sequence_lengths(self, llama_model):
        """Test handling of varying sequence lengths."""
        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0],
                hook_types=[HookType.RESIDUAL],
                architecture="llama"
            )

            # Run forward passes with different sequence lengths
            with torch.no_grad():
                _ = llama_model(torch.randint(0, 1000, (2, 5)))   # seq_len=5
                _ = llama_model(torch.randint(0, 1000, (2, 10)))  # seq_len=10
                _ = llama_model(torch.randint(0, 1000, (2, 20)))  # seq_len=20

            activations = hook_manager.activations["layer_0_residual"]
            assert len(activations) == 3
            assert activations[0].shape == (2, 5, 64)
            assert activations[1].shape == (2, 10, 64)
            assert activations[2].shape == (2, 20, 64)

    def test_numerical_correctness(self, llama_model):
        """Test that hooks don't modify the model's forward pass."""
        sample_input = torch.randint(0, 1000, (2, 10))

        # Get output without hooks
        with torch.no_grad():
            output_without_hooks = llama_model(sample_input)

        # Get output with hooks
        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0, 1, 2],
                hook_types=[HookType.RESIDUAL, HookType.MLP, HookType.ATTENTION],
                architecture="llama"
            )

            with torch.no_grad():
                output_with_hooks = llama_model(sample_input)

        # Outputs should be identical
        assert torch.allclose(output_without_hooks, output_with_hooks, rtol=1e-5)

    def test_memory_efficiency(self, llama_model):
        """Test that activations are properly freed after conversion to numpy."""
        with HookManager(llama_model) as hook_manager:
            hook_manager.register_hooks(
                layer_indices=[0, 1, 2],
                hook_types=[HookType.RESIDUAL],
                architecture="llama"
            )

            # Run many forward passes to accumulate activations
            with torch.no_grad():
                for _ in range(10):
                    _ = llama_model(torch.randint(0, 1000, (4, 20)))

            # Convert to numpy (should concatenate and return)
            numpy_activations = hook_manager.get_activations_as_numpy()

            # Verify shape: 10 batches * 4 samples = 40 total samples
            assert numpy_activations["layer_0_residual"].shape == (40, 20, 64)

            # Original tensor activations should still exist
            assert len(hook_manager.activations["layer_0_residual"]) == 10
