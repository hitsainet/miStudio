"""
Service for extracting and managing model activations.

This service handles the orchestration of activation extraction from transformer models,
including dataset loading, hook registration, batch processing, and statistics calculation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import numpy as np
from datasets import load_from_disk

from ..ml.forward_hooks import HookManager, HookType
from ..ml.model_loader import load_model_from_hf
from ..models.model import QuantizationFormat
from ..core.config import settings

logger = logging.getLogger(__name__)


class ActivationExtractionError(Exception):
    """Exception raised during activation extraction."""
    pass


class ActivationService:
    """
    Service for extracting activations from transformer models.

    This service coordinates the extraction process including:
    - Loading models and datasets
    - Registering forward hooks
    - Running batched inference
    - Saving activations to disk
    - Computing statistics
    """

    def __init__(self):
        """Initialize the ActivationService."""
        self.activations_dir = settings.data_dir / "activations"
        self.activations_dir.mkdir(parents=True, exist_ok=True)

    def _log_gpu_memory(self, stage: str, gpu_id: int = 0) -> None:
        """
        Log current GPU memory usage.

        Args:
            stage: Description of the current stage (e.g., "before_load", "after_extraction")
            gpu_id: GPU device ID to check memory for
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)    # GB
            logger.info(f"[GPU {gpu_id} Memory - {stage}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    def _cleanup_model(self, model: torch.nn.Module, gpu_id: int = 0) -> None:
        """
        Explicitly clean up model from GPU memory.

        This ensures GPU memory is freed immediately rather than waiting
        for Python's garbage collector. Critical for sequential extraction jobs.

        Args:
            model: PyTorch model to clean up
            gpu_id: GPU device ID that the model was loaded on
        """
        import gc

        try:
            # Log memory before cleanup
            self._log_gpu_memory("before_cleanup", gpu_id)

            # Synchronize CUDA to ensure all operations are complete
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)

            # AGGRESSIVE CLEANUP: Delete all model parameters and buffers explicitly
            # This is more reliable than model.cpu() when using device_map
            try:
                # Clear all parameters
                for param in model.parameters():
                    param.data = torch.empty(0)
                    if param.grad is not None:
                        param.grad = None

                # Clear all buffers
                for buffer in model.buffers():
                    buffer.data = torch.empty(0)
            except Exception as e:
                logger.warning(f"Error clearing model parameters/buffers: {e}")

            # Try to move model to CPU (may not work with device_map, but worth trying)
            try:
                model.cpu()
            except Exception as e:
                logger.warning(f"model.cpu() failed (expected with device_map): {e}")

            # Delete all module attributes that might hold tensor references
            try:
                for name, child in list(model.named_children()):
                    delattr(model, name)
            except Exception as e:
                logger.warning(f"Error deleting model children: {e}")

            # Delete model reference
            del model

            # Multiple rounds of garbage collection to ensure cleanup
            for _ in range(3):
                gc.collect()

            # Empty CUDA cache on the specific GPU
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    # Also synchronize again after cache clear
                    torch.cuda.synchronize(gpu_id)

            # Log memory after cleanup
            self._log_gpu_memory("after_cleanup", gpu_id)

            logger.info(f"Model cleaned up from GPU {gpu_id} memory")

        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")
            # Still try to clear cache even if other cleanup failed
            if torch.cuda.is_available():
                try:
                    gc.collect()
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    def extract_activations(
        self,
        model_id: str,
        model_path: str,
        architecture: str,
        quantization: QuantizationFormat,
        dataset_path: str,
        layer_indices: List[int],
        hook_types: List[str],
        max_samples: int,
        batch_size: int = 8,
        micro_batch_size: Optional[int] = None,
        extraction_id: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        gpu_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Extract activations from a model using a dataset.

        Args:
            model_id: Model database ID
            model_path: Path to model files
            architecture: Model architecture (llama, gpt2, etc.)
            quantization: Quantization format
            dataset_path: Path to tokenized dataset
            layer_indices: List of layer indices to extract from
            hook_types: List of hook types ('residual', 'mlp', 'attention')
            max_samples: Maximum number of samples to process
            batch_size: Batch size for processing
            micro_batch_size: GPU micro-batch size for memory efficiency (defaults to batch_size)
            extraction_id: Optional extraction ID (generated if not provided)
            progress_callback: Optional callback function(samples_processed, total_samples)
            gpu_id: GPU device ID to use for extraction (default: 0)

        Returns:
            Dictionary with extraction metadata including output_path and statistics

        Raises:
            ActivationExtractionError: If extraction fails
        """
        if extraction_id is None:
            extraction_id = f"ext_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Default micro_batch_size to batch_size if not specified
        if micro_batch_size is None:
            micro_batch_size = batch_size
            logger.info(f"micro_batch_size not specified, defaulting to batch_size={batch_size}")

        logger.info(
            f"Starting activation extraction: {extraction_id} "
            f"(model={model_id}, layers={layer_indices}, hooks={hook_types}, "
            f"max_samples={max_samples}, batch_size={batch_size}, micro_batch_size={micro_batch_size})"
        )

        # Create output directory
        output_dir = self.activations_dir / extraction_id
        output_dir.mkdir(parents=True, exist_ok=True)

        model = None  # Initialize to None for cleanup in finally block
        try:
            # Log GPU memory before loading model
            self._log_gpu_memory("before_load", gpu_id)

            # Load model
            logger.info(f"Loading model from {model_path} to GPU {gpu_id}")
            model, tokenizer = self._load_model(model_path, quantization, gpu_id=gpu_id)
            model.eval()  # Set to evaluation mode

            # Log GPU memory after loading model
            self._log_gpu_memory("after_load", gpu_id)

            # Load dataset
            logger.info(f"Loading dataset from {dataset_path}")
            dataset = self._load_dataset(dataset_path, max_samples)

            # Convert hook type strings to enums
            hook_type_enums = [HookType(ht) for ht in hook_types]

            # Extract activations
            logger.info(f"Extracting activations with hooks: {hook_types}")
            created_at_timestamp = datetime.now().isoformat()
            activations = self._run_extraction(
                model,
                tokenizer,
                dataset,
                architecture,
                layer_indices,
                hook_type_enums,
                batch_size,
                micro_batch_size,
                progress_callback,
                output_dir=output_dir,
                extraction_id=extraction_id,
                model_id=model_id,
                quantization=quantization,
                dataset_path=dataset_path,
                created_at=created_at_timestamp,
            )

            # Log GPU memory after extraction
            self._log_gpu_memory("after_extraction", gpu_id)

            # CRITICAL: Clean up GPU memory immediately after extraction completes
            # Model and hooks are no longer needed for saving/statistics phases
            # This frees ~8-10 GB of GPU memory that would otherwise sit idle
            logger.info(f"Cleaning up GPU memory after extraction (model no longer needed)")
            if model is not None:
                self._cleanup_model(model, gpu_id)
                model = None  # Mark as cleaned up to avoid double cleanup in finally block

            # Save activations to disk
            logger.info(f"Saving activations to {output_dir}")
            saved_files = self._save_activations(output_dir, activations)

            # Calculate statistics
            logger.info("Calculating activation statistics")
            statistics = self._calculate_statistics(activations)

            # Save metadata
            metadata = {
                "extraction_id": extraction_id,
                "model_id": model_id,
                "architecture": architecture,
                "quantization": quantization.value,
                "dataset_path": dataset_path,
                "layer_indices": layer_indices,
                "hook_types": hook_types,
                "max_samples": max_samples,
                "batch_size": batch_size,
                "num_samples_processed": len(dataset),
                "status": "completed",
                "created_at": created_at_timestamp,
                "completed_at": datetime.now().isoformat(),
                "saved_files": saved_files,
                "statistics": statistics,
            }

            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Extraction complete: {extraction_id}")

            return {
                "extraction_id": extraction_id,
                "output_path": str(output_dir),
                "num_samples": len(dataset),
                "saved_files": saved_files,
                "statistics": statistics,
                "metadata_path": str(metadata_path),
            }

        except Exception as e:
            logger.exception(f"Activation extraction failed: {e}")
            raise ActivationExtractionError(f"Extraction failed: {str(e)}") from e

        finally:
            # CRITICAL: Always clean up GPU memory, even if extraction failed
            if model is not None:
                logger.info(f"Cleaning up model for extraction {extraction_id} on GPU {gpu_id}")
                self._cleanup_model(model, gpu_id)
            else:
                logger.info(f"No model to clean up for extraction {extraction_id} (already cleaned or never loaded)")

    def _load_model(
        self,
        model_path: str,
        quantization: QuantizationFormat,
        gpu_id: int = 0
    ) -> tuple[torch.nn.Module, Any]:
        """
        Load model from disk, handling HuggingFace cache structure.

        Args:
            model_path: Path to model files (may contain HF cache structure)
            quantization: Quantization format
            gpu_id: GPU device ID to use (default: 0)

        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import glob

        model_path_obj = Path(model_path)

        # Check if model_path uses HuggingFace cache structure
        # Look for: model_path/models--{org}--{model}/snapshots/{hash}/
        models_dirs = list(model_path_obj.glob("models--*"))

        if models_dirs:
            # HuggingFace cache structure detected
            logger.info(f"Detected HuggingFace cache structure at {model_path}")
            models_dir = models_dirs[0]  # Should only be one

            # Find the snapshot directory (there should be exactly one)
            snapshot_dirs = list((models_dir / "snapshots").glob("*"))
            if not snapshot_dirs:
                raise ActivationExtractionError(
                    f"No snapshots found in HuggingFace cache at {models_dir}/snapshots"
                )

            actual_model_path = str(snapshot_dirs[0])
            logger.info(f"Using snapshot path: {actual_model_path}")
        else:
            # Direct model files (flat structure)
            actual_model_path = model_path
            logger.info(f"Using direct model path: {actual_model_path}")

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise ActivationExtractionError("CUDA is not available. GPU is required for activation extraction.")

        # Validate GPU ID
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus:
            raise ActivationExtractionError(
                f"GPU {gpu_id} not available. System has {num_gpus} GPU(s) (indices 0-{num_gpus-1})."
            )

        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Loading model to device: {device} (CUDA device: {torch.cuda.get_device_name(gpu_id)})")

        # Load model directly to GPU with explicit device parameter
        # IMPORTANT: Use explicit cuda:N instead of device_map="auto" to force GPU placement
        # device_map="auto" with accelerate can offload to CPU if it thinks there's not enough memory
        model = AutoModelForCausalLM.from_pretrained(
            actual_model_path,
            device_map={"": device},  # Force all layers to specified GPU
            torch_dtype=torch.float16,  # Always use FP16 for memory efficiency
            low_cpu_mem_usage=True,  # Minimize CPU memory during loading
        )

        logger.info(f"Model loaded successfully. Device: {model.device}, dtype: {model.dtype}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(actual_model_path)

        return model, tokenizer

    def _load_dataset(self, dataset_path: str, max_samples: int) -> Any:
        """
        Load dataset from disk.

        Args:
            dataset_path: Path to tokenized dataset
            max_samples: Maximum number of samples to load

        Returns:
            Dataset object
        """
        dataset = load_from_disk(dataset_path)

        # Limit to max_samples
        if max_samples > 0 and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        return dataset

    def _write_incremental_metadata(
        self,
        output_dir: Path,
        extraction_id: str,
        model_id: str,
        architecture: str,
        quantization: QuantizationFormat,
        dataset_path: str,
        layer_indices: List[int],
        hook_types: List[str],
        max_samples: int,
        batch_size: int,
        num_samples_processed: int,
        created_at: str,
    ) -> None:
        """
        Write incremental metadata file during extraction.

        Uses atomic write (temp file + rename) to ensure consistency.
        This allows inspection of partial results if extraction crashes.

        Args:
            output_dir: Output directory
            extraction_id: Extraction ID
            model_id: Model ID
            architecture: Model architecture
            quantization: Quantization format
            dataset_path: Dataset path
            layer_indices: Layer indices
            hook_types: Hook types
            max_samples: Maximum samples
            batch_size: Batch size
            num_samples_processed: Number of samples processed so far
            created_at: Creation timestamp
        """
        import tempfile
        import os

        metadata = {
            "extraction_id": extraction_id,
            "model_id": model_id,
            "architecture": architecture,
            "quantization": quantization.value,
            "dataset_path": dataset_path,
            "layer_indices": layer_indices,
            "hook_types": hook_types,
            "max_samples": max_samples,
            "batch_size": batch_size,
            "num_samples_processed": num_samples_processed,
            "status": "in_progress",
            "created_at": created_at,
            "last_updated": datetime.now().isoformat(),
        }

        # Atomic write: write to temp file, then rename
        metadata_path = output_dir / "metadata.json"
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=output_dir,
            delete=False,
            suffix='.tmp'
        ) as temp_file:
            json.dump(metadata, temp_file, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_name = temp_file.name

        # Rename temp file to final name (atomic operation)
        os.replace(temp_name, metadata_path)
        logger.debug(f"Updated incremental metadata: {num_samples_processed}/{max_samples} samples")

    def _run_extraction(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        dataset: Any,
        architecture: str,
        layer_indices: List[int],
        hook_types: List[HookType],
        batch_size: int,
        micro_batch_size: int,
        progress_callback: Optional[callable] = None,
        output_dir: Optional[Path] = None,
        extraction_id: Optional[str] = None,
        model_id: Optional[str] = None,
        quantization: Optional[QuantizationFormat] = None,
        dataset_path: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run activation extraction with hooks using micro-batched inference.

        This method uses a two-level batching strategy:
        1. Outer loop: Process dataset in chunks of batch_size (for progress reporting)
        2. Inner loop: Split each batch into micro-batches of micro_batch_size (for GPU memory)

        Args:
            model: PyTorch model
            tokenizer: Model tokenizer
            dataset: Dataset to process
            architecture: Model architecture
            layer_indices: Layers to hook
            hook_types: Types of hooks to register
            batch_size: Logical batch size for progress tracking (1, 8, 16, 32, 64, 128, 256, 512)
            micro_batch_size: GPU micro-batch size for memory efficiency (must be <= batch_size)
            progress_callback: Optional callback function(samples_processed, total_samples)
            output_dir: Optional output directory for incremental metadata
            extraction_id: Optional extraction ID for metadata
            model_id: Optional model ID for metadata
            quantization: Optional quantization format for metadata
            dataset_path: Optional dataset path for metadata
            created_at: Optional creation timestamp for metadata

        Returns:
            Dictionary mapping layer names to activation arrays
        """
        # Create hook manager
        with HookManager(model) as hook_manager:
            # Register hooks
            hook_manager.register_hooks(layer_indices, hook_types, architecture)

            # Get model's vocabulary size for validation
            vocab_size = model.config.vocab_size
            logger.info(f"Model vocabulary size: {vocab_size}")
            logger.info(f"Processing {len(dataset)} samples with batch_size={batch_size}")

            # Get pad token ID (use eos_token if pad_token not available)
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
                logger.info(f"Using eos_token_id {pad_token_id} as pad_token_id")

            samples_processed = 0

            # Track accumulated activation file paths for each layer
            accumulated_files = {}  # layer_name -> list of temp file paths

            # Process dataset in batches
            with torch.no_grad():
                for batch_start in range(0, len(dataset), batch_size):
                    batch_end = min(batch_start + batch_size, len(dataset))
                    batch_samples = dataset[batch_start:batch_end]

                    # Extract input_ids from batch
                    # NOTE: dataset[start:end] returns a dict with keys as column names
                    # batch_samples = {"input_ids": [[...], [...]], "attention_mask": [[...], [...]]}
                    batch_input_ids = []

                    # Check if batch_samples is a dict (HuggingFace dataset format)
                    if isinstance(batch_samples, dict) and 'input_ids' in batch_samples:
                        # batch_samples['input_ids'] is a list of token ID lists
                        for input_ids in batch_samples['input_ids']:
                            # Convert to list if needed
                            if isinstance(input_ids, torch.Tensor):
                                input_ids = input_ids.tolist()
                            elif hasattr(input_ids, 'tolist'):
                                input_ids = input_ids.tolist()

                            batch_input_ids.append(input_ids)
                    else:
                        # Fallback for other formats (single samples, non-HF datasets)
                        # Iterate over samples
                        if isinstance(batch_samples, list):
                            samples_to_iterate = batch_samples
                        else:
                            samples_to_iterate = [batch_samples]

                        for sample in samples_to_iterate:
                            if isinstance(sample, dict):
                                input_ids = sample.get("input_ids")
                            else:
                                input_ids = sample

                            # Convert to list if needed
                            if isinstance(input_ids, torch.Tensor):
                                input_ids = input_ids.tolist()
                            elif hasattr(input_ids, 'tolist'):
                                input_ids = input_ids.tolist()

                            batch_input_ids.append(input_ids)

                    # Validate and clamp token IDs BEFORE padding
                    # CRITICAL: Also truncate to prevent GPU OOM from very long sequences
                    MAX_SEQ_LENGTH = 512  # Maximum sequence length to prevent GPU OOM
                    cleaned_batch_input_ids = []
                    for idx, input_ids in enumerate(batch_input_ids):
                        # Convert to tensor for validation
                        ids_tensor = torch.tensor(input_ids)

                        # Truncate if too long
                        if len(ids_tensor) > MAX_SEQ_LENGTH:
                            ids_tensor = ids_tensor[:MAX_SEQ_LENGTH]
                            logger.debug(f"Sample {batch_start + idx} truncated from {len(input_ids)} to {MAX_SEQ_LENGTH} tokens")

                        # Clamp to valid vocabulary range
                        max_token = ids_tensor.max().item()
                        min_token = ids_tensor.min().item()

                        if max_token >= vocab_size or min_token < 0:
                            if max_token >= vocab_size:
                                logger.warning(
                                    f"Sample {batch_start + idx} contains token ID {max_token} "
                                    f"(vocab_size={vocab_size}). Clamping to valid range."
                                )
                            if min_token < 0:
                                logger.warning(
                                    f"Sample {batch_start + idx} contains negative token ID {min_token}. "
                                    f"Clamping to valid range."
                                )
                            ids_tensor = torch.clamp(ids_tensor, 0, vocab_size - 1)

                        cleaned_batch_input_ids.append(ids_tensor.tolist())

                    # Find max length in this batch (will be at most MAX_SEQ_LENGTH)
                    max_length = max(len(ids) for ids in cleaned_batch_input_ids)

                    # Pad sequences to max_length and create attention masks
                    padded_input_ids = []
                    attention_masks = []

                    for input_ids in cleaned_batch_input_ids:
                        # Calculate padding needed
                        padding_length = max_length - len(input_ids)

                        # Pad input_ids (pad on the right)
                        padded_ids = input_ids + [pad_token_id] * padding_length
                        padded_input_ids.append(padded_ids)

                        # Create attention mask (1 for real tokens, 0 for padding)
                        attention_mask = [1] * len(input_ids) + [0] * padding_length
                        attention_masks.append(attention_mask)

                    # MICRO-BATCHING: Split batch into micro-batches for GPU memory efficiency
                    # This allows large logical batch sizes while keeping GPU memory usage low
                    micro_batch_activations = {}  # Accumulate activations from all micro-batches

                    for micro_batch_start in range(0, len(padded_input_ids), micro_batch_size):
                        micro_batch_end = min(micro_batch_start + micro_batch_size, len(padded_input_ids))

                        # Get micro-batch slices
                        micro_input_ids = padded_input_ids[micro_batch_start:micro_batch_end]
                        micro_attention_masks = attention_masks[micro_batch_start:micro_batch_end]

                        logger.debug(
                            f"Processing micro-batch {micro_batch_start}-{micro_batch_end} "
                            f"of batch {batch_start}-{batch_end} "
                            f"(micro_batch_size={micro_batch_size})"
                        )

                        # Convert to tensors and move to device
                        input_ids_tensor = torch.tensor(micro_input_ids, dtype=torch.long).to(model.device)
                        attention_mask_tensor = torch.tensor(micro_attention_masks, dtype=torch.long).to(model.device)

                        # Run forward pass (hooks will capture activations for this micro-batch)
                        _ = model(input_ids_tensor, attention_mask=attention_mask_tensor)

                        # Get activations from this micro-batch
                        current_micro_activations = hook_manager.get_activations_as_numpy()

                        # Accumulate activations from this micro-batch
                        for layer_name, activation_array in current_micro_activations.items():
                            if layer_name not in micro_batch_activations:
                                micro_batch_activations[layer_name] = []
                            micro_batch_activations[layer_name].append(activation_array)

                        # Clear hooks to free GPU memory before next micro-batch
                        hook_manager.clear_activations()

                    # Concatenate all micro-batch activations into single batch
                    # This happens in CPU/RAM, not GPU
                    batch_activations = {}
                    for layer_name, activation_list in micro_batch_activations.items():
                        batch_activations[layer_name] = np.concatenate(activation_list, axis=0)

                    # CRITICAL FIX: Save batch activations to disk immediately to prevent memory accumulation

                    # Save each layer's batch to a temporary file
                    for layer_name, activation_array in batch_activations.items():
                        # Initialize list for this layer if first batch
                        if layer_name not in accumulated_files:
                            accumulated_files[layer_name] = []

                        # Save batch activation to temporary file
                        import tempfile
                        import os
                        # Create temp file and close it immediately before numpy writes
                        # This avoids numpy memory-mapping issues with open file handles
                        temp_file = tempfile.NamedTemporaryFile(
                            dir=output_dir,
                            delete=False,
                            suffix=f'_{layer_name}_batch{len(accumulated_files[layer_name])}.npy'
                        )
                        temp_file_path = temp_file.name
                        temp_file.close()  # Close the file handle before numpy writes

                        # Now numpy can write to the closed file without memory-mapping issues
                        np.save(temp_file_path, activation_array)

                        accumulated_files[layer_name].append(temp_file_path)
                        logger.debug(f"Saved batch activation for {layer_name}: {activation_array.shape}")

                    # Note: Activations already cleared after each micro-batch for memory efficiency

                    samples_processed = batch_end

                    # Log progress every 10 samples or every batch (whichever is more frequent)
                    if samples_processed % 10 == 0 or samples_processed == batch_end:
                        logger.info(f"Processed {samples_processed}/{len(dataset)} samples")

                        # Call progress callback if provided
                        if progress_callback:
                            try:
                                progress_callback(samples_processed, len(dataset))
                            except Exception as e:
                                logger.warning(f"Progress callback failed: {e}")

                    # Write incremental metadata every 50 samples
                    if samples_processed % 50 == 0 and output_dir and extraction_id and model_id and quantization and dataset_path and created_at:
                        try:
                            hook_type_strs = [ht.value for ht in hook_types]
                            self._write_incremental_metadata(
                                output_dir=output_dir,
                                extraction_id=extraction_id,
                                model_id=model_id,
                                architecture=architecture,
                                quantization=quantization,
                                dataset_path=dataset_path,
                                layer_indices=layer_indices,
                                hook_types=hook_type_strs,
                                max_samples=len(dataset),
                                batch_size=batch_size,
                                num_samples_processed=samples_processed,
                                created_at=created_at,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to write incremental metadata: {e}")

            # Concatenate all batch files for each layer to create final activations
            # Use memory-efficient chunked concatenation to avoid OOM during large extractions
            logger.info("Concatenating batch activations into final arrays (memory-efficient mode)")
            activations = {}
            for layer_name, temp_files in accumulated_files.items():
                if not temp_files:
                    continue

                logger.info(f"Processing {len(temp_files)} batch files for {layer_name}")

                # Strategy: Use numpy.memmap for zero-copy concatenation
                # 1. Determine final shape by loading first file
                # 2. Pre-allocate output file with final shape
                # 3. Copy batches directly into output file
                # 4. Clean up temp files as we go

                # Load first file to get shape and dtype
                first_array = np.load(temp_files[0])
                sample_shape = first_array.shape[1:]  # Shape without batch dimension
                dtype = first_array.dtype

                # Calculate total samples across all batches
                total_samples = 0
                batch_sizes = []
                for temp_file_path in temp_files:
                    arr = np.load(temp_file_path, mmap_mode='r')  # Memory-mapped read
                    batch_sizes.append(arr.shape[0])
                    total_samples += arr.shape[0]

                # Create output file path
                final_shape = (total_samples,) + sample_shape
                output_file = output_dir / f"{layer_name}_temp_concat.npy"

                logger.info(f"Creating memory-mapped output for {layer_name}: shape={final_shape}, dtype={dtype}")

                # Pre-allocate output array as memory-mapped file
                final_array = np.lib.format.open_memmap(
                    str(output_file),
                    mode='w+',
                    dtype=dtype,
                    shape=final_shape
                )

                # Copy batches into final array in chunks
                current_idx = 0
                for i, (temp_file_path, batch_size) in enumerate(zip(temp_files, batch_sizes)):
                    # Load batch as memory-mapped array (no memory allocation)
                    batch_array = np.load(temp_file_path, mmap_mode='r')

                    # Copy directly into output file
                    final_array[current_idx:current_idx + batch_size] = batch_array
                    current_idx += batch_size

                    # Force flush to disk every 10 batches to free memory
                    if i % 10 == 0:
                        final_array.flush()

                    # Immediately delete temp file to free disk space
                    try:
                        Path(temp_file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")

                # Final flush
                final_array.flush()

                # Load final array (still memory-mapped, not in RAM)
                activations[layer_name] = final_array

                logger.info(f"Concatenated {len(temp_files)} batches for {layer_name}: final shape={final_array.shape}")

                # Clean up any remaining temp files
                for temp_file_path in temp_files:
                    try:
                        if Path(temp_file_path).exists():
                            Path(temp_file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")

        logger.info(f"Extracted activations for {len(activations)} layers")
        return activations

    def _save_activations(
        self,
        output_dir: Path,
        activations: Dict[str, np.ndarray]
    ) -> List[str]:
        """
        Save activations to disk as .npy files.

        Handles both regular numpy arrays and memory-mapped arrays.
        For memory-mapped arrays created during concatenation, renames the file
        instead of copying to avoid memory overhead.

        Args:
            output_dir: Directory to save files
            activations: Dictionary of activations

        Returns:
            List of saved file paths
        """
        saved_files = []

        for layer_name, activation_array in activations.items():
            # Create final filename
            filename = f"{layer_name}.npy"
            filepath = output_dir / filename

            # Check if this is a memory-mapped array from concatenation
            temp_concat_file = output_dir / f"{layer_name}_temp_concat.npy"

            if temp_concat_file.exists() and isinstance(activation_array, np.memmap):
                # This is a memory-mapped array - just rename the file
                # First, ensure all data is flushed to disk
                if hasattr(activation_array, 'flush'):
                    activation_array.flush()

                # Delete reference to allow file rename
                del activation_array

                # Rename temp file to final filename
                temp_concat_file.rename(filepath)
                logger.debug(
                    f"Renamed memory-mapped file for {layer_name} (zero-copy save)"
                )
            else:
                # Regular array - save normally
                np.save(filepath, activation_array)
                logger.debug(
                    f"Saved {layer_name}: shape={activation_array.shape}, "
                    f"dtype={activation_array.dtype}, size={activation_array.nbytes / 1024 / 1024:.2f}MB"
                )

            saved_files.append(filename)

        return saved_files

    def _calculate_statistics(
        self,
        activations: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for extracted activations using chunked processing.

        For large arrays (>1GB), uses memory-mapped files and chunked computation
        to avoid loading the entire array into memory.

        Args:
            activations: Dictionary of activation arrays

        Returns:
            Dictionary mapping layer names to statistics dictionaries
        """
        statistics = {}

        for layer_name, activation_array in activations.items():
            array_size_gb = activation_array.nbytes / (1024 ** 3)

            # For large arrays (>1GB), use chunked processing to avoid memory issues
            if array_size_gb > 1.0:
                logger.info(f"Large array detected ({array_size_gb:.2f} GB), using chunked statistics calculation")

                # Use chunked processing with smaller memory footprint
                chunk_size = 100  # Process 100 samples at a time
                n_samples = activation_array.shape[0]

                # Initialize accumulators
                sum_abs = 0.0
                sum_sq = 0.0
                max_val = float('-inf')
                min_val = float('inf')
                count_near_zero = 0
                total_elements = 0

                # Process in chunks
                for start_idx in range(0, n_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_samples)
                    chunk = activation_array[start_idx:end_idx]

                    # Update statistics
                    abs_chunk = np.abs(chunk)
                    sum_abs += float(abs_chunk.sum())
                    sum_sq += float((chunk ** 2).sum())
                    max_val = max(max_val, float(abs_chunk.max()))
                    min_val = min(min_val, float(abs_chunk.min()))

                    # Count near-zero activations
                    threshold = 0.01
                    count_near_zero += int((abs_chunk < threshold).sum())
                    total_elements += chunk.size

                    # Log progress every 1000 samples
                    if (end_idx % 1000 == 0) or (end_idx == n_samples):
                        logger.debug(f"Processed {end_idx}/{n_samples} samples for statistics")

                # Calculate final statistics
                mean_magnitude = sum_abs / total_elements
                variance = (sum_sq / total_elements) - (mean_magnitude ** 2)
                std_activation = float(np.sqrt(max(0, variance)))  # Avoid negative due to numerical errors
                sparsity = (count_near_zero / total_elements) * 100

                # Handle potential inf values
                if np.isinf(std_activation) or np.isnan(std_activation):
                    std_activation = None
                if np.isinf(mean_magnitude) or np.isnan(mean_magnitude):
                    mean_magnitude = 0.0

                logger.info(f"Statistics calculation complete for {layer_name}")

            else:
                # Small array, use standard numpy operations
                mean_magnitude = float(np.abs(activation_array).mean())
                max_val = float(np.abs(activation_array).max())
                min_val = float(np.abs(activation_array).min())
                std_activation = float(np.std(activation_array))

                # Replace inf/-inf with None for JSON compatibility
                if np.isinf(std_activation) or np.isnan(std_activation):
                    std_activation = None

                # Calculate sparsity
                threshold = 0.01
                sparsity = float((np.abs(activation_array) < threshold).mean() * 100)

            statistics[layer_name] = {
                "shape": list(activation_array.shape),
                "mean_magnitude": mean_magnitude,
                "max_activation": max_val,
                "min_activation": min_val,
                "std_activation": std_activation,
                "sparsity_percent": sparsity,
                "size_mb": float(activation_array.nbytes / 1024 / 1024),
            }

        return statistics

    def get_extraction_info(self, extraction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a completed extraction.

        Args:
            extraction_id: Extraction ID

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self.activations_dir / extraction_id / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return json.load(f)

    def list_extractions(self) -> List[Dict[str, Any]]:
        """
        List all completed extractions.

        Returns:
            List of extraction metadata dictionaries
        """
        extractions = []

        for extraction_dir in self.activations_dir.iterdir():
            if extraction_dir.is_dir():
                metadata_path = extraction_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        extractions.append(json.load(f))

        # Sort by creation time (newest first)
        extractions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return extractions

    def delete_extraction(self, extraction_id: str) -> bool:
        """
        Delete an extraction and all its files.

        Args:
            extraction_id: Extraction ID to delete

        Returns:
            True if deleted successfully, False if not found
        """
        extraction_dir = self.activations_dir / extraction_id

        if not extraction_dir.exists():
            return False

        import shutil
        shutil.rmtree(extraction_dir)

        logger.info(f"Deleted extraction: {extraction_id}")
        return True
