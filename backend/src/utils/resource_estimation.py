"""
Resource estimation utilities for activation extraction.

This module provides functions to estimate GPU memory, disk space, and processing
time requirements for activation extraction jobs before they are started.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def estimate_gpu_memory(
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    sequence_length: int = 512,
    num_hook_types: int = 1,
    dtype_bytes: int = 2,  # FP16 = 2 bytes
    safety_factor: float = 1.5,  # 50% safety margin
    model_params_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Estimate GPU memory requirements for activation extraction.

    GPU memory breakdown:
    1. Model weights: actual parameter count * dtype_bytes
    2. Activations per layer: batch_size * sequence_length * hidden_size
    3. KV cache and attention matrices (for inference)
    4. Temporary buffers: ~20% of activation memory

    Args:
        hidden_size: Model hidden dimension size
        num_layers: Total number of layers in model
        batch_size: Batch size for extraction
        sequence_length: Typical sequence length (default 512)
        num_hook_types: Number of hook types being extracted (1-3)
        dtype_bytes: Bytes per parameter (FP32=4, FP16=2, Q8=1, Q4=0.5)
        safety_factor: Safety multiplier for overhead
        model_params_count: Actual model parameter count (if available, highly recommended)

    Returns:
        Dictionary with memory estimates in various units
    """
    # Model weights memory
    if model_params_count:
        # Use actual parameter count for accurate estimate
        model_memory_bytes = model_params_count * dtype_bytes
    else:
        # Fallback: rough estimate based on architecture
        # Typical transformer: ~12x hidden_size^2 params per layer (attention + FFN + layer norms)
        # This is a very rough approximation and will be inaccurate for large models
        params_per_layer = hidden_size * hidden_size * 12
        total_params = params_per_layer * num_layers
        model_memory_bytes = total_params * dtype_bytes
        logger.warning(
            f"Using rough model size estimate ({total_params:,} params). "
            f"For accurate estimates, provide model_params_count."
        )

    # Activation memory per layer per hook type
    # Each activation is [batch_size, sequence_length, hidden_size]
    activation_size_bytes = batch_size * sequence_length * hidden_size * 4  # FP32 for activations

    # Total activation memory for all selected layers and hook types
    total_activation_bytes = activation_size_bytes * num_hook_types

    # PyTorch overhead: temporary buffers, CUDA context, etc.
    pytorch_overhead_bytes = total_activation_bytes * 0.2

    # Total memory needed
    total_bytes = (model_memory_bytes + total_activation_bytes + pytorch_overhead_bytes) * safety_factor

    # Convert to human-readable units
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)

    # Calculate breakdown
    breakdown = {
        "model_weights_mb": model_memory_bytes / (1024 ** 2),
        "activations_mb": total_activation_bytes / (1024 ** 2),
        "overhead_mb": pytorch_overhead_bytes / (1024 ** 2),
    }

    return {
        "total_bytes": int(total_bytes),
        "total_mb": round(total_mb, 2),
        "total_gb": round(total_gb, 2),
        "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
        "warning": "high" if total_gb > 16 else "medium" if total_gb > 8 else "normal"
    }


def estimate_disk_space(
    num_samples: int,
    num_layers: int,
    hidden_size: int,
    sequence_length: int = 512,
    num_hook_types: int = 1,
    compression_ratio: float = 0.9  # Numpy saves with some compression
) -> Dict[str, Any]:
    """
    Estimate disk space required for saved activations.

    Activations are saved as numpy arrays with shape:
    [num_samples, sequence_length, hidden_size]

    Args:
        num_samples: Number of samples to extract
        num_layers: Number of layers being extracted
        hidden_size: Model hidden dimension size
        sequence_length: Typical sequence length (default 512)
        num_hook_types: Number of hook types being extracted (1-3)
        compression_ratio: Numpy's compression effectiveness (0.9 = 10% compression)

    Returns:
        Dictionary with disk space estimates in various units
    """
    # Size per activation array
    # dtype is FP32 (4 bytes) for saved activations
    bytes_per_array = num_samples * sequence_length * hidden_size * 4

    # Total size for all layers and hook types
    total_bytes_uncompressed = bytes_per_array * num_layers * num_hook_types

    # Apply compression ratio
    total_bytes = total_bytes_uncompressed * compression_ratio

    # Metadata overhead (metadata.json, statistics, etc.)
    metadata_bytes = 1024 * 1024  # ~1 MB

    total_bytes_with_metadata = total_bytes + metadata_bytes

    # Convert to human-readable units
    total_mb = total_bytes_with_metadata / (1024 ** 2)
    total_gb = total_bytes_with_metadata / (1024 ** 3)

    return {
        "total_bytes": int(total_bytes_with_metadata),
        "total_mb": round(total_mb, 2),
        "total_gb": round(total_gb, 2),
        "per_layer_mb": round(total_mb / num_layers, 2) if num_layers > 0 else 0,
        "warning": "high" if total_gb > 50 else "medium" if total_gb > 10 else "normal"
    }


def estimate_processing_time(
    num_samples: int,
    num_layers: int,
    batch_size: int,
    model_params_count: Optional[int] = None,
    hidden_size: Optional[int] = None,
    device_type: str = "cuda",  # "cuda" or "cpu"
) -> Dict[str, Any]:
    """
    Estimate processing time for activation extraction.

    This is a rough heuristic based on:
    - Number of forward passes needed (num_samples / batch_size)
    - Model size (affects inference speed)
    - Device type (GPU vs CPU)

    Args:
        num_samples: Number of samples to extract
        num_layers: Number of layers being extracted
        batch_size: Batch size for extraction
        model_params_count: Total model parameters (if available)
        hidden_size: Model hidden dimension (if params_count not available)
        device_type: "cuda" or "cpu"

    Returns:
        Dictionary with time estimates
    """
    # Calculate number of batches
    num_batches = int(np.ceil(num_samples / batch_size))

    # Estimate seconds per batch based on device and model size
    if model_params_count:
        # Use actual parameter count
        if model_params_count < 1_000_000_000:  # < 1B params
            base_seconds_per_batch = 0.3 if device_type == "cuda" else 3.0
        elif model_params_count < 3_000_000_000:  # 1-3B params
            base_seconds_per_batch = 0.5 if device_type == "cuda" else 5.0
        elif model_params_count < 7_000_000_000:  # 3-7B params
            base_seconds_per_batch = 1.0 if device_type == "cuda" else 10.0
        else:  # 7B+ params
            base_seconds_per_batch = 2.0 if device_type == "cuda" else 20.0
    elif hidden_size:
        # Estimate from hidden size
        if hidden_size < 1024:  # Small models (< 1B)
            base_seconds_per_batch = 0.3 if device_type == "cuda" else 3.0
        elif hidden_size < 2048:  # Medium models (1-3B)
            base_seconds_per_batch = 0.5 if device_type == "cuda" else 5.0
        elif hidden_size < 4096:  # Large models (3-7B)
            base_seconds_per_batch = 1.0 if device_type == "cuda" else 10.0
        else:  # Very large models (7B+)
            base_seconds_per_batch = 2.0 if device_type == "cuda" else 20.0
    else:
        # Fallback to conservative estimate
        base_seconds_per_batch = 1.0 if device_type == "cuda" else 10.0

    # Layer extraction overhead (negligible for modern GPUs)
    layer_overhead_factor = 1.0 + (num_layers * 0.01)

    # Total time estimate
    total_seconds = num_batches * base_seconds_per_batch * layer_overhead_factor

    # Add overhead for saving and statistics calculation (roughly 10% of extraction time)
    total_seconds_with_overhead = total_seconds * 1.1

    # Convert to human-readable format
    total_minutes = total_seconds_with_overhead / 60
    total_hours = total_seconds_with_overhead / 3600

    # Format as string
    if total_hours >= 1:
        time_str = f"{total_hours:.1f} hours"
    elif total_minutes >= 1:
        time_str = f"{total_minutes:.1f} minutes"
    else:
        time_str = f"{int(total_seconds_with_overhead)} seconds"

    return {
        "total_seconds": int(total_seconds_with_overhead),
        "time_str": time_str,
        "estimated_batches": num_batches,
        "seconds_per_batch": round(base_seconds_per_batch, 2),
        "warning": "long" if total_hours > 1 else "medium" if total_minutes > 30 else "normal"
    }


def estimate_extraction_resources(
    model_config: Dict[str, Any],
    extraction_config: Dict[str, Any],
    dataset_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive resource estimation for an extraction job.

    Args:
        model_config: Model configuration including:
            - hidden_size: Hidden dimension
            - num_layers: Number of layers
            - params_count: Total parameters (optional)
            - quantization: Quantization format (FP32, FP16, Q8, Q4)
        extraction_config: Extraction configuration including:
            - layer_indices: List of layers to extract
            - hook_types: List of hook types
            - batch_size: Batch size
            - max_samples: Number of samples
        dataset_config: Optional dataset metadata including:
            - avg_sequence_length: Average sequence length

    Returns:
        Comprehensive resource estimates including GPU memory, disk space, and time
    """
    # Extract parameters
    hidden_size = model_config.get("hidden_size", 768)
    num_total_layers = model_config.get("num_layers", 12)
    params_count = model_config.get("params_count")
    quantization = model_config.get("quantization", "FP16")

    num_selected_layers = len(extraction_config.get("layer_indices", []))
    num_hook_types = len(extraction_config.get("hook_types", []))
    batch_size = extraction_config.get("batch_size", 32)
    max_samples = extraction_config.get("max_samples", 1000)

    # Estimate average sequence length from dataset or use default
    sequence_length = 512
    if dataset_config and "avg_sequence_length" in dataset_config:
        sequence_length = int(dataset_config["avg_sequence_length"])

    # Determine dtype bytes from quantization
    dtype_map = {
        "FP32": 4,
        "FP16": 2,
        "Q8": 1,
        "Q4": 0.5,
        "Q2": 0.25
    }
    dtype_bytes = dtype_map.get(quantization, 2)

    # Calculate estimates
    gpu_memory = estimate_gpu_memory(
        hidden_size=hidden_size,
        num_layers=num_total_layers,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_hook_types=num_hook_types,
        dtype_bytes=dtype_bytes,
        model_params_count=params_count
    )

    disk_space = estimate_disk_space(
        num_samples=max_samples,
        num_layers=num_selected_layers,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        num_hook_types=num_hook_types
    )

    processing_time = estimate_processing_time(
        num_samples=max_samples,
        num_layers=num_selected_layers,
        batch_size=batch_size,
        model_params_count=params_count,
        hidden_size=hidden_size
    )

    # Determine overall warning level
    warnings = []
    if gpu_memory["warning"] == "high":
        warnings.append("High GPU memory usage - consider reducing batch size")
    if disk_space["warning"] == "high":
        warnings.append("Large disk space required - ensure sufficient storage")
    if processing_time["warning"] == "long":
        warnings.append("Long processing time expected")

    return {
        "gpu_memory": gpu_memory,
        "disk_space": disk_space,
        "processing_time": processing_time,
        "warnings": warnings,
        "config_used": {
            "hidden_size": hidden_size,
            "num_layers": num_selected_layers,
            "batch_size": batch_size,
            "max_samples": max_samples,
            "sequence_length": sequence_length
        }
    }


def estimate_training_memory(
    hidden_dim: int,
    latent_dim: int,
    batch_size: int,
    dtype_bytes: int = 4,  # FP32 = 4 bytes
    safety_factor: float = 1.3,  # 30% safety margin
) -> Dict[str, Any]:
    """
    Estimate GPU memory requirements for SAE training.

    Memory breakdown:
    1. Model parameters: (hidden_dim * latent_dim * 2) * dtype_bytes
       - Encoder weights + Decoder weights
    2. Optimizer state (Adam): 2x model parameters (momentum + variance)
    3. Activations: batch_size * (hidden_dim + latent_dim) * dtype_bytes
    4. Gradients: same as model parameters
    5. PyTorch overhead: ~20% of total

    Args:
        hidden_dim: Input/output dimension
        latent_dim: SAE latent dimension (typically 8-32x hidden_dim)
        batch_size: Training batch size
        dtype_bytes: Bytes per parameter (FP32=4, FP16=2)
        safety_factor: Safety multiplier for overhead

    Returns:
        Dictionary with memory estimates in MB/GB
    """
    # Model parameters
    # Encoder: hidden_dim * latent_dim, Decoder: latent_dim * hidden_dim, Biases: hidden_dim + latent_dim
    model_params = (hidden_dim * latent_dim) + (latent_dim * hidden_dim) + hidden_dim + latent_dim
    model_memory_bytes = model_params * dtype_bytes

    # Optimizer state (Adam requires 2x parameters for momentum and variance)
    optimizer_memory_bytes = model_memory_bytes * 2

    # Activations during forward pass
    # Input: batch_size * hidden_dim, Latent: batch_size * latent_dim, Output: batch_size * hidden_dim
    activation_memory_bytes = batch_size * (hidden_dim * 2 + latent_dim) * dtype_bytes

    # Gradients (same size as parameters)
    gradient_memory_bytes = model_memory_bytes

    # Base memory
    base_memory_bytes = model_memory_bytes + optimizer_memory_bytes + activation_memory_bytes + gradient_memory_bytes

    # PyTorch overhead
    overhead_bytes = base_memory_bytes * 0.2

    # Total with safety factor
    total_bytes = (base_memory_bytes + overhead_bytes) * safety_factor

    # Convert to human-readable units
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)

    # Breakdown
    breakdown = {
        "model_params_mb": model_memory_bytes / (1024 ** 2),
        "optimizer_state_mb": optimizer_memory_bytes / (1024 ** 2),
        "activations_mb": activation_memory_bytes / (1024 ** 2),
        "gradients_mb": gradient_memory_bytes / (1024 ** 2),
        "overhead_mb": overhead_bytes / (1024 ** 2),
    }

    # Warning level for Jetson Orin Nano (6GB VRAM)
    warning = "critical" if total_gb > 6 else "high" if total_gb > 4 else "normal"

    return {
        "total_bytes": int(total_bytes),
        "total_mb": round(total_mb, 2),
        "total_gb": round(total_gb, 2),
        "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
        "warning": warning,
        "fits_in_6gb": total_gb <= 6.0,
    }


def estimate_oom_reduced_batch_size(
    current_batch_size: int,
    memory_limit_gb: float = 6.0,
    current_memory_gb: float = None
) -> int:
    """
    Estimate a safe batch size after OOM error.

    Args:
        current_batch_size: Current batch size that caused OOM
        memory_limit_gb: GPU memory limit in GB (default 6GB for Jetson)
        current_memory_gb: Current memory usage if known

    Returns:
        Recommended batch size (halved, minimum 1)
    """
    # Conservative strategy: halve the batch size
    new_batch_size = max(1, current_batch_size // 2)

    logger.info(
        f"OOM detected with batch_size={current_batch_size}. "
        f"Recommending batch_size={new_batch_size}"
    )

    return new_batch_size
