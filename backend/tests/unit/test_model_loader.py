"""
Unit tests for model_loader module.

Tests model loading utilities, quantization configurations, architecture validation,
and memory estimation without requiring actual model downloads.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from transformers import BitsAndBytesConfig

from src.ml.model_loader import (
    validate_architecture,
    extract_architecture_config,
    get_quantization_config,
    estimate_model_memory,
    get_fallback_format,
    SUPPORTED_ARCHITECTURES,
    ModelLoadError,
    OutOfMemoryError,
)
from src.models.model import QuantizationFormat


class TestArchitectureValidation:
    """Test architecture validation functionality."""

    def test_validate_supported_architecture(self):
        """Test that supported architectures pass validation."""
        for arch in SUPPORTED_ARCHITECTURES:
            # Should not raise
            validate_architecture(arch)
            validate_architecture(arch.upper())  # Case insensitive

    def test_validate_unsupported_architecture(self):
        """Test that unsupported architectures raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            validate_architecture("unsupported_model")

        with pytest.raises(ValueError, match="Unsupported architecture"):
            validate_architecture("bert")  # Not a causal LM


class TestArchitectureConfigExtraction:
    """Test architecture configuration extraction."""

    def test_extract_basic_config(self):
        """Test extraction of basic configuration fields."""
        mock_config = Mock()
        mock_config.model_type = "llama"
        mock_config.num_hidden_layers = 32
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.intermediate_size = 11008
        mock_config.max_position_embeddings = 2048
        mock_config.vocab_size = 32000

        arch_config = extract_architecture_config(mock_config)

        assert arch_config["model_type"] == "llama"
        assert arch_config["num_hidden_layers"] == 32
        assert arch_config["hidden_size"] == 4096
        assert arch_config["num_attention_heads"] == 32
        assert arch_config["intermediate_size"] == 11008
        assert arch_config["max_position_embeddings"] == 2048
        assert arch_config["vocab_size"] == 32000

    def test_extract_partial_config(self):
        """Test extraction when some fields are missing."""
        mock_config = Mock()
        mock_config.model_type = "gpt2"
        mock_config.num_hidden_layers = 12
        mock_config.hidden_size = 768
        # Missing some optional fields

        # Should not have rope_theta if not present
        delattr(mock_config, "rope_theta")
        delattr(mock_config, "num_key_value_heads")

        arch_config = extract_architecture_config(mock_config)

        assert arch_config["model_type"] == "gpt2"
        assert arch_config["num_hidden_layers"] == 12
        assert "rope_theta" not in arch_config
        assert "num_key_value_heads" not in arch_config

    def test_extract_gqa_config(self):
        """Test extraction of grouped query attention config."""
        mock_config = Mock()
        mock_config.model_type = "llama"
        mock_config.num_hidden_layers = 32
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8  # GQA

        arch_config = extract_architecture_config(mock_config)

        assert arch_config["num_key_value_heads"] == 8


class TestQuantizationConfig:
    """Test quantization configuration generation."""

    def test_fp32_config(self):
        """Test FP32 returns None (no quantization)."""
        config = get_quantization_config(QuantizationFormat.FP32)
        assert config is None

    def test_fp16_config(self):
        """Test FP16 returns None (handled via torch_dtype)."""
        config = get_quantization_config(QuantizationFormat.FP16)
        assert config is None

    def test_q8_config(self):
        """Test Q8 quantization configuration."""
        config = get_quantization_config(QuantizationFormat.Q8)

        assert isinstance(config, BitsAndBytesConfig)
        assert config.load_in_8bit is True
        assert config.llm_int8_threshold == 6.0
        assert config.llm_int8_has_fp16_weight is False

    def test_q4_config(self):
        """Test Q4 quantization configuration."""
        config = get_quantization_config(QuantizationFormat.Q4)

        assert isinstance(config, BitsAndBytesConfig)
        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == torch.float16
        assert config.bnb_4bit_use_double_quant is True
        assert config.bnb_4bit_quant_type == "nf4"

    def test_q2_config(self):
        """Test Q2 quantization configuration (experimental)."""
        config = get_quantization_config(QuantizationFormat.Q2)

        assert isinstance(config, BitsAndBytesConfig)
        assert config.load_in_4bit is True  # Uses 4-bit infrastructure
        assert config.bnb_4bit_quant_type == "fp4"  # More aggressive


class TestMemoryEstimation:
    """Test memory requirement estimation."""

    def test_estimate_fp32_memory(self):
        """Test FP32 memory estimation (4 bytes per param)."""
        params_count = 1_000_000_000  # 1B params
        memory = estimate_model_memory(params_count, QuantizationFormat.FP32)

        # 1B params * 4 bytes = 4GB + 20% overhead = 4.8GB
        expected = int(4_000_000_000 * 1.2)
        assert memory == expected

    def test_estimate_fp16_memory(self):
        """Test FP16 memory estimation (2 bytes per param)."""
        params_count = 1_000_000_000  # 1B params
        memory = estimate_model_memory(params_count, QuantizationFormat.FP16)

        # 1B params * 2 bytes = 2GB + 20% overhead = 2.4GB
        expected = int(2_000_000_000 * 1.2)
        assert memory == expected

    def test_estimate_q8_memory(self):
        """Test Q8 memory estimation (1 byte per param)."""
        params_count = 1_000_000_000  # 1B params
        memory = estimate_model_memory(params_count, QuantizationFormat.Q8)

        # 1B params * 1 byte = 1GB + 20% overhead = 1.2GB
        expected = int(1_000_000_000 * 1.2)
        assert memory == expected

    def test_estimate_q4_memory(self):
        """Test Q4 memory estimation (0.5 bytes per param)."""
        params_count = 1_000_000_000  # 1B params
        memory = estimate_model_memory(params_count, QuantizationFormat.Q4)

        # 1B params * 0.5 bytes = 0.5GB + 20% overhead = 0.6GB
        expected = int(500_000_000 * 1.2)
        assert memory == expected

    def test_estimate_q2_memory(self):
        """Test Q2 memory estimation (0.25 bytes per param)."""
        params_count = 1_000_000_000  # 1B params
        memory = estimate_model_memory(params_count, QuantizationFormat.Q2)

        # 1B params * 0.25 bytes = 0.25GB + 20% overhead = 0.3GB
        expected = int(250_000_000 * 1.2)
        assert memory == expected

    def test_estimate_small_model(self):
        """Test memory estimation for small model."""
        params_count = 124_000_000  # GPT-2 Small: 124M params
        memory = estimate_model_memory(params_count, QuantizationFormat.FP16)

        # 124M * 2 bytes * 1.2 = ~298MB
        expected = int(124_000_000 * 2 * 1.2)
        assert memory == expected


class TestFallbackLogic:
    """Test quantization fallback chain."""

    def test_q2_fallback(self):
        """Test Q2 falls back to Q4."""
        fallback = get_fallback_format(QuantizationFormat.Q2)
        assert fallback == QuantizationFormat.Q4

    def test_q4_fallback(self):
        """Test Q4 falls back to Q8."""
        fallback = get_fallback_format(QuantizationFormat.Q4)
        assert fallback == QuantizationFormat.Q8

    def test_q8_fallback(self):
        """Test Q8 falls back to FP16."""
        fallback = get_fallback_format(QuantizationFormat.Q8)
        assert fallback == QuantizationFormat.FP16

    def test_fp16_fallback(self):
        """Test FP16 falls back to FP32."""
        fallback = get_fallback_format(QuantizationFormat.FP16)
        assert fallback == QuantizationFormat.FP32

    def test_fp32_no_fallback(self):
        """Test FP32 has no fallback."""
        fallback = get_fallback_format(QuantizationFormat.FP32)
        assert fallback is None


class TestSupportedArchitectures:
    """Test that expected architectures are supported."""

    def test_common_architectures_supported(self):
        """Test that common model architectures are in the supported list."""
        expected_architectures = {
            "llama",
            "gpt2",
            "gpt_neox",
            "phi",
            "pythia",
            "mistral",
        }

        for arch in expected_architectures:
            assert arch in SUPPORTED_ARCHITECTURES

    def test_minimum_architectures(self):
        """Test that we support at least 6 architectures."""
        assert len(SUPPORTED_ARCHITECTURES) >= 6
