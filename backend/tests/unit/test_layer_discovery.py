"""
Tests for dynamic layer discovery functionality.

This module tests the layer_discovery module which provides architecture-agnostic
detection of transformer layers via runtime introspection.

NO MOCKING - Uses real PyTorch models to verify discovery logic.
"""

import pytest
import torch
import torch.nn as nn

from src.ml.layer_discovery import (
    discover_transformer_structure,
    get_hookable_module,
    TransformerStructure,
    _is_transformer_layer,
    _find_matching_attr,
    ATTENTION_PATTERNS,
    LAYER_NORM_PATTERNS,
    MLP_PATTERNS,
)


# Test model fixtures


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
        return self.out_proj(x)  # Simplified for testing


class LlamaStyleLayer(nn.Module):
    """Llama-style transformer layer."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = SimpleAttention(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = SimpleMLP(hidden_dim)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class GPT2StyleLayer(nn.Module):
    """GPT-2 style transformer layer."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = SimpleAttention(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = SimpleMLP(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CustomArchLayer(nn.Module):
    """Custom architecture layer with unusual naming."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = SimpleAttention(hidden_dim)  # "attention" instead of "self_attn"
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = SimpleMLP(hidden_dim)  # "feed_forward" instead of "mlp"

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class LlamaStyleModel(nn.Module):
    """Llama-style model (model.model.layers pattern)."""

    def __init__(self, num_layers: int = 3, hidden_dim: int = 64, vocab_size: int = 1000):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model = nn.ModuleDict({
            "layers": nn.ModuleList([
                LlamaStyleLayer(hidden_dim) for _ in range(num_layers)
            ])
        })
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.model["layers"]:
            x = layer(x)
        return self.lm_head(x)


class GPT2StyleModel(nn.Module):
    """GPT-2 style model (transformer.h pattern)."""

    def __init__(self, num_layers: int = 3, hidden_dim: int = 64, vocab_size: int = 1000):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.ModuleDict({
            "h": nn.ModuleList([
                GPT2StyleLayer(hidden_dim) for _ in range(num_layers)
            ])
        })
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.wte(x)
        for layer in self.transformer["h"]:
            x = layer(x)
        return self.lm_head(x)


class CustomArchModel(nn.Module):
    """Custom architecture model with deep nesting."""

    def __init__(self, num_layers: int = 3, hidden_dim: int = 64, vocab_size: int = 1000):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        # Unusual nesting: encoder.blocks
        self.encoder = nn.ModuleDict({
            "blocks": nn.ModuleList([
                CustomArchLayer(hidden_dim) for _ in range(num_layers)
            ])
        })
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.encoder["blocks"]:
            x = block(x)
        return self.output(x)


class NonTransformerModel(nn.Module):
    """A model that is NOT a transformer (no attention/MLP blocks)."""

    def __init__(self, hidden_dim: int = 64, vocab_size: int = 1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


# Test fixtures


@pytest.fixture
def llama_model():
    """Create a Llama-style model."""
    return LlamaStyleModel(num_layers=3)


@pytest.fixture
def gpt2_model():
    """Create a GPT-2 style model."""
    return GPT2StyleModel(num_layers=4)


@pytest.fixture
def custom_model():
    """Create a custom architecture model."""
    return CustomArchModel(num_layers=2)


@pytest.fixture
def non_transformer_model():
    """Create a non-transformer model."""
    return NonTransformerModel()


# Tests


class TestTransformerLayerDetection:
    """Tests for _is_transformer_layer function."""

    def test_detects_llama_style_layer(self):
        """Test detection of Llama-style layer."""
        layer = LlamaStyleLayer()
        is_transformer, modules = _is_transformer_layer(layer)

        assert is_transformer is True
        assert modules["attention"] == "self_attn"
        assert modules["mlp"] == "mlp"
        # THE POST-ATTENTION NORM, specifically. This used to accept either,
        # which tolerated the very nondeterminism that made captures
        # irreproducible instead of preventing it.
        assert modules["residual_norm"] == "post_attention_layernorm"

    def test_detects_gpt2_style_layer(self):
        """Test detection of GPT-2 style layer."""
        layer = GPT2StyleLayer()
        is_transformer, modules = _is_transformer_layer(layer)

        assert is_transformer is True
        assert modules["attention"] == "attn"
        assert modules["mlp"] == "mlp"
        # Either ln_1 or ln_2 works for residual hooks
        assert modules["residual_norm"] in ("ln_1", "ln_2")

    def test_detects_custom_style_layer(self):
        """Test detection of custom architecture layer."""
        layer = CustomArchLayer()
        is_transformer, modules = _is_transformer_layer(layer)

        assert is_transformer is True
        assert modules["attention"] == "attention"
        assert modules["mlp"] == "feed_forward"

    def test_rejects_non_transformer(self):
        """Test that non-transformer modules are rejected."""
        linear = nn.Linear(64, 64)
        is_transformer, modules = _is_transformer_layer(linear)

        assert is_transformer is False
        assert modules["attention"] is None
        assert modules["mlp"] is None


class TestDiscoverTransformerStructure:
    """Tests for discover_transformer_structure function."""

    def test_discovers_llama_structure(self, llama_model):
        """Test discovery of Llama-style model structure."""
        structure = discover_transformer_structure(llama_model, "llama")

        assert structure.layers_path == "model.layers"
        assert structure.num_layers == 3
        assert structure.attention_module == "self_attn"
        assert structure.mlp_module == "mlp"
        # See above: pinned, not tolerated.
        assert structure.residual_norm_module == "post_attention_layernorm"

    def test_discovers_gpt2_structure(self, gpt2_model):
        """Test discovery of GPT-2 style model structure."""
        structure = discover_transformer_structure(gpt2_model, "gpt2")

        assert structure.layers_path == "transformer.h"
        assert structure.num_layers == 4
        assert structure.attention_module == "attn"
        assert structure.mlp_module == "mlp"
        # Either ln_1 or ln_2 works for residual hooks
        assert structure.residual_norm_module in ("ln_1", "ln_2")

    def test_discovers_custom_structure(self, custom_model):
        """Test discovery of custom architecture."""
        structure = discover_transformer_structure(custom_model, "custom")

        # Should find the layers via deep search
        assert structure.num_layers == 2
        assert structure.attention_module == "attention"
        assert structure.mlp_module == "feed_forward"

    def test_architecture_hint_is_optional(self, llama_model):
        """Test that architecture hint is truly optional."""
        structure = discover_transformer_structure(llama_model)

        assert structure.num_layers == 3
        assert structure.architecture_hint is None

    def test_wrong_hint_still_works(self, llama_model):
        """Test that a wrong architecture hint doesn't break discovery."""
        # Pass "gpt2" hint for a Llama model - should still work
        structure = discover_transformer_structure(llama_model, "gpt2")

        assert structure.num_layers == 3
        assert structure.attention_module == "self_attn"  # Llama-style

    def test_fails_for_non_transformer(self, non_transformer_model):
        """Test that discovery fails for non-transformer models."""
        with pytest.raises(ValueError, match="Could not discover transformer layers"):
            discover_transformer_structure(non_transformer_model, "unknown")


class TestGetHookableModule:
    """Tests for get_hookable_module function."""

    def test_gets_attention_module(self, llama_model):
        """Test getting attention module for hooking."""
        structure = discover_transformer_structure(llama_model)
        layer = structure.layers_module[0]

        module = get_hookable_module(layer, "attention", structure)

        assert module is not None
        assert isinstance(module, SimpleAttention)

    def test_gets_mlp_module(self, llama_model):
        """Test getting MLP module for hooking."""
        structure = discover_transformer_structure(llama_model)
        layer = structure.layers_module[0]

        module = get_hookable_module(layer, "mlp", structure)

        assert module is not None
        assert isinstance(module, SimpleMLP)

    def test_gets_residual_module(self, llama_model):
        """Test getting residual norm module for hooking."""
        structure = discover_transformer_structure(llama_model)
        layer = structure.layers_module[0]

        module = get_hookable_module(layer, "residual", structure)

        assert module is not None
        assert isinstance(module, nn.LayerNorm)

    def test_falls_back_to_pattern_search(self, custom_model):
        """Test fallback pattern search when structure doesn't have exact match."""
        structure = discover_transformer_structure(custom_model)
        layer = structure.layers_module[0]

        # Should find "attention" via pattern matching
        module = get_hookable_module(layer, "attention", structure)
        assert module is not None

    def test_returns_none_for_unknown_hook_type(self, llama_model):
        """Test that unknown hook types return None."""
        structure = discover_transformer_structure(llama_model)
        layer = structure.layers_module[0]

        module = get_hookable_module(layer, "unknown_type", structure)

        assert module is None


class TestTransformerStructureDataclass:
    """Tests for TransformerStructure dataclass."""

    def test_str_representation(self, llama_model):
        """Test string representation of TransformerStructure."""
        structure = discover_transformer_structure(llama_model, "llama")
        str_repr = str(structure)

        assert "layers_path='model.layers'" in str_repr
        assert "num_layers=3" in str_repr
        assert "attention='self_attn'" in str_repr
        assert "mlp='mlp'" in str_repr


class TestPatternSets:
    """Tests for the pattern constants — content AND order.

    These were sets, and every one of them is consumed by a first-match-wins
    loop, so the order decides which module gets hooked on a layer with more
    than one candidate. A set has no order and Python randomises string hashing
    per process, so the answer differed between workers.
    """

    def test_attention_patterns_comprehensive(self):
        """Test that attention patterns cover common naming."""
        expected = {"self_attn", "attention", "attn", "self_attention", "mha", "conv"}
        assert set(ATTENTION_PATTERNS) == expected

    def test_mlp_patterns_comprehensive(self):
        """Test that MLP patterns cover common naming."""
        expected = {"mlp", "feed_forward", "ffn", "dense", "ff", "intermediate"}
        assert set(MLP_PATTERNS) == expected

    def test_every_pattern_collection_is_ORDERED(self):
        """A set here is a latent nondeterminism bug, not a style choice."""
        for name, patterns in (
            ("ATTENTION_PATTERNS", ATTENTION_PATTERNS),
            ("MLP_PATTERNS", MLP_PATTERNS),
            ("LAYER_NORM_PATTERNS", LAYER_NORM_PATTERNS),
        ):
            assert isinstance(patterns, (tuple, list)), (
                f"{name} is a {type(patterns).__name__}; it is consumed by a "
                "first-match-wins loop, so it must have a defined order"
            )

    def test_the_POST_attention_norm_is_preferred_over_the_pre_attention_one(self):
        """The preference the comments have always claimed.

        A Llama-style layer has both `input_layernorm` (pre-attention) and
        `post_attention_layernorm`. Which one wins is the residual point every
        SAE in this project is trained and served on, so it must be fixed, and
        it must be the post-attention one.
        """
        order = list(LAYER_NORM_PATTERNS)
        assert order.index("post_attention_layernorm") < order.index("input_layernorm")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_model_list(self):
        """Test handling of model with empty layer list."""
        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.ModuleDict({
                    "layers": nn.ModuleList([])  # Empty
                })

        model = EmptyModel()
        with pytest.raises(ValueError, match="Could not discover transformer layers"):
            discover_transformer_structure(model)

    def test_deeply_nested_layers(self):
        """Test discovery of deeply nested layers."""
        class DeeplyNestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.wrapper = nn.ModuleDict({
                    "inner": nn.ModuleDict({
                        "core": nn.ModuleDict({
                            "layers": nn.ModuleList([
                                LlamaStyleLayer() for _ in range(2)
                            ])
                        })
                    })
                })

        model = DeeplyNestedModel()
        structure = discover_transformer_structure(model)

        # Should find layers via deep search
        assert structure.num_layers == 2
        assert structure.attention_module == "self_attn"


class TestDiscoveryIsDeterministicACROSSProcesses:
    """The bug the ordering fix exists for.

    `LAYER_NORM_PATTERNS` was a set, and the selection loop takes the first
    match. Python randomises string hashing per process, so two workers running
    identical code on an identical model chose different residual points. An
    SAE trained by an extraction worker on one point could be fed the other by
    a capture worker, silently, and the only symptom was an implausible
    activation density much later.

    In-process repetition cannot catch this — the seed is fixed once per
    interpreter — so this spawns real subprocesses.
    """

    PROBE = (
        "import torch.nn as nn;"
        "from src.ml.layer_discovery import _is_transformer_layer;"
        "l = nn.Module();"
        "l.input_layernorm = nn.LayerNorm(8);"
        "l.post_attention_layernorm = nn.LayerNorm(8);"
        "l.self_attn = nn.Linear(8, 8);"
        "l.mlp = nn.Linear(8, 8);"
        "print(_is_transformer_layer(l)[1]['residual_norm'])"
    )

    def _choice_under(self, seed: str) -> str:
        import os
        import subprocess
        import sys

        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", self.PROBE],
            capture_output=True, text=True, env=env, timeout=300,
            cwd=str(pathlib.Path(__file__).resolve().parents[2]),
        )
        assert out.returncode == 0, f"probe failed under seed {seed}: {out.stderr[-800:]}"
        return out.stdout.strip().splitlines()[-1]

    def test_the_residual_point_is_the_same_under_every_hash_seed(self):
        seeds = ["0", "1", "2", "3", "4"]
        choices = {seed: self._choice_under(seed) for seed in seeds}

        assert len(set(choices.values())) == 1, (
            "the residual hook target depends on PYTHONHASHSEED, so two workers "
            f"can capture different points of the residual stream: {choices}"
        )
        assert set(choices.values()) == {"post_attention_layernorm"}


import pathlib  # noqa: E402 - used by the subprocess probe above
