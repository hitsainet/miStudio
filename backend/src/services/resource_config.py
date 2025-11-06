"""
Dynamic resource configuration based on available system resources.
Balances performance with safety to avoid OOM while maximizing throughput.
"""
import psutil
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ResourceConfig:
    """Dynamically calculates optimal resource settings."""
    
    # Safety margins to prevent OOM
    RAM_SAFETY_MARGIN = 0.25  # Reserve 25% of available RAM
    GPU_SAFETY_MARGIN = 0.20  # Reserve 20% of GPU memory
    
    # Per-sample memory estimates (empirically determined)
    RAM_PER_SAMPLE_MB = 2.0  # ~2MB per sample in batch
    # Memory-efficient heap: only stores top-5 token positions per example (~250 bytes/example)
    # NOT full sequences (512 tokens). 250 bytes = 0.25 KB per example.
    RAM_PER_FEATURE_HEAP_KB = 0.25  # ~250 bytes per example (top-5 tokens only)
    
    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """Get current system resource availability."""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=True)
        
        resources = {
            "cpu_cores": cpu_count,
            "total_ram_gb": memory.total / (1024**3),
            "available_ram_gb": memory.available / (1024**3),
            "ram_percent_used": memory.percent,
        }
        
        # GPU resources
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            
            resources.update({
                "gpu_available": True,
                "gpu_name": gpu_props.name,
                "gpu_total_memory_gb": gpu_props.total_memory / (1024**3),
                "gpu_memory_allocated_gb": gpu_memory_allocated,
                "gpu_memory_reserved_gb": gpu_memory_reserved,
                "gpu_memory_available_gb": (gpu_props.total_memory / (1024**3)) - gpu_memory_reserved,
            })
        else:
            resources["gpu_available"] = False
            
        return resources
    
    @classmethod
    def calculate_extraction_config(
        cls,
        num_features: int,
        top_k_examples: int,
        sequence_length: int = 512,
        hidden_dim: int = 768
    ) -> Dict[str, int]:
        """
        Calculate optimal extraction settings based on available resources.
        
        Args:
            num_features: Number of SAE features (latent_dim)
            top_k_examples: Number of examples to store per feature
            sequence_length: Max sequence length
            hidden_dim: Hidden dimension size
            
        Returns:
            Dictionary with optimal settings:
            - batch_size: Samples to process at once
            - num_workers: CPU workers for parallel processing
            - db_commit_batch: Features to commit at once
        """
        resources = cls.get_system_resources()
        
        logger.info(f"Calculating extraction config for {resources['cpu_cores']} cores, "
                   f"{resources['available_ram_gb']:.1f}GB available RAM")
        
        # 1. Calculate batch size based on available RAM and GPU memory
        usable_ram_gb = resources["available_ram_gb"] * (1 - cls.RAM_SAFETY_MARGIN)
        
        # Memory for batch processing (activation tensors, intermediate results)
        batch_memory_overhead_mb = 500  # Base overhead
        per_sample_ram_mb = cls.RAM_PER_SAMPLE_MB * sequence_length / 512  # Scale with seq length
        
        # Memory for feature heap storage
        heap_memory_mb = (num_features * top_k_examples * cls.RAM_PER_FEATURE_HEAP_KB) / 1024
        
        # Available for batches
        available_for_batches_mb = (usable_ram_gb * 1024) - batch_memory_overhead_mb - heap_memory_mb
        max_batch_from_ram = int(available_for_batches_mb / per_sample_ram_mb)
        
        # Constrain by GPU memory if available
        if resources["gpu_available"]:
            gpu_available_gb = resources["gpu_memory_available_gb"] * (1 - cls.GPU_SAFETY_MARGIN)

            # Conservative estimate for transformer forward pass:
            # - Model weights (already loaded): ~1-2GB for GPT-2 class models
            # - Per-sample memory: seq_len * hidden_dim * 4 bytes * num_layers * 3 (input + intermediate + output)
            # - For GPT-2: 512 * 768 * 4 * 12 * 3 ≈ 50MB per sample
            per_sample_gpu_mb = (sequence_length * hidden_dim * 4 * 12 * 3) / (1024**2)  # Assume ~12 layers

            # Reserve space for model and SAE (~2GB)
            available_for_batch_gb = max(0.5, gpu_available_gb - 2.0)
            max_batch_from_gpu = int((available_for_batch_gb * 1024) / per_sample_gpu_mb)

            batch_size = min(max_batch_from_ram, max_batch_from_gpu)
            logger.info(f"GPU-constrained batch size: {max_batch_from_gpu} (available: {gpu_available_gb:.1f}GB)")
        else:
            batch_size = max_batch_from_ram

        # Clamp to reasonable range - more conservative for extraction
        batch_size = max(8, min(batch_size, 64))  # Between 8 and 64 (reduced from 256)
        
        # 2. Calculate number of CPU workers
        # Use 50-75% of cores for CPU-bound feature processing
        # Leave cores for system, database, other services
        max_workers = max(1, int(resources["cpu_cores"] * 0.6))
        
        # Don't exceed what makes sense for workload
        # Too many workers can cause overhead; diminishing returns after ~8
        num_workers = min(max_workers, 8)
        
        # 3. Database commit batch size
        # Larger batches = fewer commits, but more memory
        # Scale with available RAM
        if resources["available_ram_gb"] > 15:
            db_commit_batch = 2000
        elif resources["available_ram_gb"] > 8:
            db_commit_batch = 1000
        else:
            db_commit_batch = 500
            
        config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "db_commit_batch": db_commit_batch,
        }
        
        logger.info(f"Extraction config: batch_size={batch_size}, "
                   f"num_workers={num_workers}, db_commit_batch={db_commit_batch}")
        logger.info(f"Estimated RAM usage: ~{heap_memory_mb + batch_memory_overhead_mb:.0f}MB base + "
                   f"~{per_sample_ram_mb * batch_size:.0f}MB per batch")
        
        return config
    
    @classmethod
    def get_optimal_settings(
        cls,
        training_config: Dict[str, Any],
        extraction_config: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Get optimal settings for extraction based on training and extraction configs.

        Args:
            training_config: Training hyperparameters (latent_dim, hidden_dim, etc.)
            extraction_config: Extraction parameters (top_k_examples, evaluation_samples)

        Returns:
            Optimal extraction settings
        """
        return cls.calculate_extraction_config(
            num_features=training_config.get("latent_dim", 8192),
            top_k_examples=extraction_config.get("top_k_examples", 100),
            sequence_length=extraction_config.get("max_length", 512),
            hidden_dim=training_config.get("hidden_dim", 768)
        )

    @classmethod
    def estimate_resource_usage(
        cls,
        num_features: int,
        top_k_examples: int,
        batch_size: int,
        num_workers: int,
        evaluation_samples: int = 10000,
        sequence_length: int = 512,
        hidden_dim: int = 768
    ) -> Dict[str, Any]:
        """
        Estimate resource usage for given extraction configuration.

        Args:
            num_features: Number of SAE features (latent_dim)
            top_k_examples: Number of examples to store per feature
            batch_size: Samples to process at once
            num_workers: CPU workers for parallel processing
            evaluation_samples: Total samples to evaluate
            sequence_length: Max sequence length
            hidden_dim: Hidden dimension size

        Returns:
            Dictionary with estimated resource usage:
            - estimated_ram_gb: Estimated RAM usage
            - estimated_gpu_gb: Estimated GPU VRAM usage (if available)
            - estimated_duration_minutes: Estimated completion time
            - warnings: List of warning messages
            - errors: List of error messages (resource exhaustion)
        """
        resources = cls.get_system_resources()

        # Calculate memory requirements
        # 1. Heap storage for top-k examples
        heap_memory_mb = (num_features * top_k_examples * cls.RAM_PER_FEATURE_HEAP_KB) / 1024

        # 2. Batch processing memory (activation tensors, intermediate results)
        batch_memory_overhead_mb = 500  # Base overhead
        per_sample_ram_mb = cls.RAM_PER_SAMPLE_MB * sequence_length / 512
        batch_memory_mb = per_sample_ram_mb * batch_size

        # 3. Model and tokenizer memory (rough estimate)
        model_memory_mb = 2000  # ~2GB for typical transformer model

        # Total RAM estimate
        estimated_ram_gb = (heap_memory_mb + batch_memory_overhead_mb + batch_memory_mb + model_memory_mb) / 1024

        # GPU memory estimate (if available)
        estimated_gpu_gb = 0
        if resources["gpu_available"]:
            # Model weights + activations + batch processing
            model_gpu_mb = 2000  # Model on GPU
            per_sample_gpu_mb = (sequence_length * hidden_dim * 4 * 2) / (1024**2)
            batch_gpu_mb = per_sample_gpu_mb * batch_size
            estimated_gpu_gb = (model_gpu_mb + batch_gpu_mb) / 1024

        # Estimate duration (based on empirical data)
        # Approximate processing rate: 150 samples/minute per worker
        samples_per_minute = 150 * num_workers
        estimated_duration_minutes = evaluation_samples / samples_per_minute

        # Validate against available resources
        warnings = []
        errors = []

        # Check RAM
        available_ram_gb = resources["available_ram_gb"]
        if estimated_ram_gb > available_ram_gb:
            errors.append(f"Estimated RAM usage ({estimated_ram_gb:.1f}GB) exceeds available RAM ({available_ram_gb:.1f}GB)")
        elif estimated_ram_gb > available_ram_gb * 0.9:
            warnings.append(f"RAM usage will be very high ({estimated_ram_gb:.1f}GB of {available_ram_gb:.1f}GB available)")

        # Check GPU
        if resources["gpu_available"]:
            available_gpu_gb = resources["gpu_memory_available_gb"]
            if estimated_gpu_gb > available_gpu_gb:
                errors.append(f"Estimated GPU memory ({estimated_gpu_gb:.1f}GB) exceeds available GPU memory ({available_gpu_gb:.1f}GB)")
            elif estimated_gpu_gb > available_gpu_gb * 0.9:
                warnings.append(f"GPU memory usage will be very high ({estimated_gpu_gb:.1f}GB of {available_gpu_gb:.1f}GB available)")

        # Check batch size recommendations
        recommended = cls.calculate_extraction_config(num_features, top_k_examples, sequence_length, hidden_dim)
        if batch_size < recommended["batch_size"] * 0.5:
            warnings.append(f"Batch size ({batch_size}) is significantly below recommended ({recommended['batch_size']}) - extraction will be slower")

        # Check worker count
        if num_workers > resources["cpu_cores"]:
            warnings.append(f"Worker count ({num_workers}) exceeds CPU cores ({resources['cpu_cores']}) - may cause overhead")

        return {
            "estimated_ram_gb": round(estimated_ram_gb, 2),
            "estimated_gpu_gb": round(estimated_gpu_gb, 2) if resources["gpu_available"] else None,
            "estimated_duration_minutes": round(estimated_duration_minutes, 1),
            "warnings": warnings,
            "errors": errors
        }
