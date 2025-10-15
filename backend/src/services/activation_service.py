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
        extraction_id: Optional[str] = None,
        progress_callback: Optional[callable] = None,
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
            extraction_id: Optional extraction ID (generated if not provided)
            progress_callback: Optional callback function(samples_processed, total_samples)

        Returns:
            Dictionary with extraction metadata including output_path and statistics

        Raises:
            ActivationExtractionError: If extraction fails
        """
        if extraction_id is None:
            extraction_id = f"ext_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(
            f"Starting activation extraction: {extraction_id} "
            f"(model={model_id}, layers={layer_indices}, hooks={hook_types}, "
            f"max_samples={max_samples}, batch_size={batch_size})"
        )

        # Create output directory
        output_dir = self.activations_dir / extraction_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load model
            logger.info(f"Loading model from {model_path}")
            model, tokenizer = self._load_model(model_path, quantization)
            model.eval()  # Set to evaluation mode

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
                progress_callback,
                output_dir=output_dir,
                extraction_id=extraction_id,
                model_id=model_id,
                quantization=quantization,
                dataset_path=dataset_path,
                created_at=created_at_timestamp,
            )

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

    def _load_model(
        self,
        model_path: str,
        quantization: QuantizationFormat
    ) -> tuple[torch.nn.Module, Any]:
        """
        Load model from disk, handling HuggingFace cache structure.

        Args:
            model_path: Path to model files (may contain HF cache structure)
            quantization: Quantization format

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

        device = torch.device("cuda:0")
        logger.info(f"Loading model to device: {device} (CUDA device: {torch.cuda.get_device_name(0)})")

        # Load model directly to GPU with explicit device parameter
        # IMPORTANT: Use device="cuda:0" instead of device_map="auto" to force GPU placement
        # device_map="auto" with accelerate can offload to CPU if it thinks there's not enough memory
        model = AutoModelForCausalLM.from_pretrained(
            actual_model_path,
            device_map={"": device},  # Force all layers to GPU 0
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
        progress_callback: Optional[callable] = None,
        output_dir: Optional[Path] = None,
        extraction_id: Optional[str] = None,
        model_id: Optional[str] = None,
        quantization: Optional[QuantizationFormat] = None,
        dataset_path: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run activation extraction with hooks using batched inference.

        Args:
            model: PyTorch model
            tokenizer: Model tokenizer
            dataset: Dataset to process
            architecture: Model architecture
            layer_indices: Layers to hook
            hook_types: Types of hooks to register
            batch_size: Batch size for processing (1, 8, 16, 32, 64, 128, 256, 512)
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
                    cleaned_batch_input_ids = []
                    for idx, input_ids in enumerate(batch_input_ids):
                        # Convert to tensor for validation
                        ids_tensor = torch.tensor(input_ids)

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

                    # Find max length in this batch
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

                    # Convert to tensors and move to device
                    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long).to(model.device)
                    attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long).to(model.device)

                    # Run forward pass (hooks will capture activations)
                    _ = model(input_ids_tensor, attention_mask=attention_mask_tensor)

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

            # Get activations as numpy arrays
            activations = hook_manager.get_activations_as_numpy()

        logger.info(f"Extracted activations for {len(activations)} layers")
        return activations

    def _save_activations(
        self,
        output_dir: Path,
        activations: Dict[str, np.ndarray]
    ) -> List[str]:
        """
        Save activations to disk as .npy files.

        Args:
            output_dir: Directory to save files
            activations: Dictionary of activations

        Returns:
            List of saved file paths
        """
        saved_files = []

        for layer_name, activation_array in activations.items():
            # Create filename
            filename = f"{layer_name}.npy"
            filepath = output_dir / filename

            # Save as numpy array
            np.save(filepath, activation_array)

            saved_files.append(filename)
            logger.debug(
                f"Saved {layer_name}: shape={activation_array.shape}, "
                f"dtype={activation_array.dtype}, size={activation_array.nbytes / 1024 / 1024:.2f}MB"
            )

        return saved_files

    def _calculate_statistics(
        self,
        activations: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for extracted activations.

        Args:
            activations: Dictionary of activation arrays

        Returns:
            Dictionary mapping layer names to statistics dictionaries
        """
        statistics = {}

        for layer_name, activation_array in activations.items():
            # Calculate statistics
            mean_magnitude = float(np.abs(activation_array).mean())
            max_activation = float(np.abs(activation_array).max())
            min_activation = float(np.abs(activation_array).min())
            std_activation = float(np.std(activation_array))

            # Replace inf/-inf with None for JSON compatibility
            # PostgreSQL JSONB doesn't support Infinity values
            if np.isinf(std_activation):
                std_activation = None

            # Calculate sparsity (percentage of near-zero activations)
            threshold = 0.01
            sparsity = float((np.abs(activation_array) < threshold).mean() * 100)

            statistics[layer_name] = {
                "shape": list(activation_array.shape),
                "mean_magnitude": mean_magnitude,
                "max_activation": max_activation,
                "min_activation": min_activation,
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
