"""
Model loading and quantization utilities.

This module provides functions for loading language models from HuggingFace,
applying quantization, extracting architecture configuration, and estimating
memory requirements.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)

from ..models.model import QuantizationFormat

logger = logging.getLogger(__name__)


# Supported model architectures
SUPPORTED_ARCHITECTURES = {
    "llama",
    "gpt2",
    "gpt_neox",
    "phi",
    "phi3",  # Phi-3 and Phi-4 models
    "phi3_v",  # Phi-3 Vision models
    "pythia",
    "mistral",
    "mixtral",
    "qwen",
    "qwen3",  # Qwen3 models (added to transformers 2025-03-31)
    "falcon",
}


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class OutOfMemoryError(Exception):
    """Raised when loading fails due to insufficient memory."""
    pass


def validate_architecture(architecture: str) -> None:
    """
    Validate that the model architecture is supported.

    Args:
        architecture: Model architecture name (e.g., "llama", "gpt2")

    Raises:
        ValueError: If architecture is not supported
    """
    # Normalize to lowercase for case-insensitive comparison
    if architecture.lower() not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            f"Supported architectures: {', '.join(sorted(SUPPORTED_ARCHITECTURES))}"
        )


def extract_architecture_config(config: AutoConfig) -> Dict[str, Any]:
    """
    Extract relevant architecture configuration from HuggingFace config.

    Args:
        config: HuggingFace model configuration

    Returns:
        Dictionary containing architecture details
    """
    arch_config = {
        "model_type": config.model_type,
    }

    # Common fields across architectures
    common_fields = [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
        "vocab_size",
        "num_key_value_heads",  # For GQA/MQA
        "hidden_act",
        "initializer_range",
        "layer_norm_eps",
        "use_cache",
        "tie_word_embeddings",
        "rope_theta",  # For RoPE embeddings
    ]

    for field in common_fields:
        if hasattr(config, field):
            arch_config[field] = getattr(config, field)

    return arch_config


def get_quantization_config(quant_format: QuantizationFormat) -> Optional[BitsAndBytesConfig]:
    """
    Get BitsAndBytes quantization configuration for the specified format.

    Args:
        quant_format: Quantization format enum value

    Returns:
        BitsAndBytesConfig for bitsandbytes quantization, or None for FP32/FP16
    """
    if quant_format == QuantizationFormat.FP32:
        return None  # Load in full precision

    elif quant_format == QuantizationFormat.FP16:
        return None  # Will use torch_dtype=torch.float16

    elif quant_format == QuantizationFormat.Q8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    elif quant_format == QuantizationFormat.Q4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    elif quant_format == QuantizationFormat.Q2:
        # Q2 is experimental - use 4-bit with aggressive settings
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",  # More aggressive than nf4
        )

    else:
        raise ValueError(f"Unknown quantization format: {quant_format}")


def estimate_model_memory(
    params_count: int,
    quant_format: QuantizationFormat,
) -> int:
    """
    Estimate memory requirements for a model in bytes.

    Args:
        params_count: Number of model parameters
        quant_format: Quantization format

    Returns:
        Estimated memory requirement in bytes
    """
    # Base memory per parameter based on quantization
    bytes_per_param = {
        QuantizationFormat.FP32: 4,
        QuantizationFormat.FP16: 2,
        QuantizationFormat.Q8: 1,
        QuantizationFormat.Q4: 0.5,
        QuantizationFormat.Q2: 0.25,
    }

    base_memory = params_count * bytes_per_param[quant_format]

    # Add overhead for activations, gradients, optimizer states (roughly 20%)
    overhead = base_memory * 0.2

    return int(base_memory + overhead)


def get_fallback_format(quant_format: QuantizationFormat) -> Optional[QuantizationFormat]:
    """
    Get the next less aggressive quantization format for fallback.

    Args:
        quant_format: Current quantization format

    Returns:
        Next fallback format, or None if no fallback available
    """
    fallback_chain = {
        QuantizationFormat.Q2: QuantizationFormat.Q4,
        QuantizationFormat.Q4: QuantizationFormat.Q8,
        QuantizationFormat.Q8: QuantizationFormat.FP16,
        QuantizationFormat.FP16: QuantizationFormat.FP32,
        QuantizationFormat.FP32: None,
    }

    return fallback_chain.get(quant_format)


def load_model_from_hf(
    repo_id: str,
    quant_format: QuantizationFormat = QuantizationFormat.FP16,
    cache_dir: Optional[Path] = None,
    device_map: str = "auto",
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
    auto_fallback: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoConfig, Dict[str, Any]]:
    """
    Load a language model from HuggingFace Hub with specified quantization.

    Args:
        repo_id: HuggingFace model repository ID (e.g., "meta-llama/Llama-2-7b-hf")
        quant_format: Quantization format to apply
        cache_dir: Directory to cache downloaded models
        device_map: Device mapping strategy ("auto", "cpu", "cuda:0", etc.)
        trust_remote_code: Whether to trust remote code execution
        hf_token: HuggingFace API token for gated models
        auto_fallback: Automatically fallback to less aggressive quantization on OOM

    Returns:
        Tuple of (model, tokenizer, config, metadata dict)

    Raises:
        ModelLoadError: If model loading fails
        OutOfMemoryError: If loading fails due to OOM and auto_fallback is False
        ValueError: If architecture is unsupported
    """
    logger.info(f"Loading model {repo_id} with quantization {quant_format.value}")

    try:
        # Load configuration first to validate architecture
        config = AutoConfig.from_pretrained(
            repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )

        # Validate architecture
        validate_architecture(config.model_type)

        # Extract architecture configuration
        arch_config = extract_architecture_config(config)

        # Get quantization configuration
        quantization_config = get_quantization_config(quant_format)

        # Determine torch dtype
        if quant_format == QuantizationFormat.FP16:
            torch_dtype = torch.float16
        elif quant_format in (QuantizationFormat.Q8, QuantizationFormat.Q4, QuantizationFormat.Q2):
            torch_dtype = torch.float16  # bitsandbytes uses fp16 for compute
        else:
            torch_dtype = torch.float32

        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                config=config,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )

            logger.info(f"Successfully loaded model with {quant_format.value} quantization")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                if auto_fallback:
                    fallback_format = get_fallback_format(quant_format)
                    if fallback_format:
                        logger.warning(
                            f"Out of memory with {quant_format.value}. "
                            f"Falling back to {fallback_format.value}"
                        )
                        # Recursive call with fallback format
                        return load_model_from_hf(
                            repo_id=repo_id,
                            quant_format=fallback_format,
                            cache_dir=cache_dir,
                            device_map=device_map,
                            trust_remote_code=trust_remote_code,
                            hf_token=hf_token,
                            auto_fallback=auto_fallback,
                        )
                    else:
                        raise OutOfMemoryError(
                            f"Out of memory loading {repo_id} even with FP32. "
                            "Model may be too large for available hardware."
                        )
                else:
                    raise OutOfMemoryError(
                        f"Out of memory loading {repo_id} with {quant_format.value}. "
                        f"Try a more aggressive quantization format."
                    )
            else:
                raise

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )

        # Calculate metadata
        params_count = sum(p.numel() for p in model.parameters())
        memory_required = estimate_model_memory(params_count, quant_format)

        metadata = {
            "repo_id": repo_id,
            "quantization": quant_format.value,
            "params_count": params_count,
            "memory_required_bytes": memory_required,
            "architecture": config.model_type,
            "architecture_config": arch_config,
        }

        return model, tokenizer, config, metadata

    except Exception as e:
        if isinstance(e, (OutOfMemoryError, ValueError)):
            raise
        logger.error(f"Failed to load model {repo_id}: {e}")
        raise ModelLoadError(f"Failed to load model {repo_id}: {e}")
