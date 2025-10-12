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
            activations = self._run_extraction(
                model,
                tokenizer,
                dataset,
                architecture,
                layer_indices,
                hook_type_enums,
                batch_size,
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
                "created_at": datetime.now().isoformat(),
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
        Load model from disk.

        Args:
            model_path: Path to model files
            quantization: Quantization format

        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if quantization == QuantizationFormat.FP16 else None,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

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

    def _run_extraction(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        dataset: Any,
        architecture: str,
        layer_indices: List[int],
        hook_types: List[HookType],
        batch_size: int,
    ) -> Dict[str, np.ndarray]:
        """
        Run activation extraction with hooks.

        Args:
            model: PyTorch model
            tokenizer: Model tokenizer
            dataset: Dataset to process
            architecture: Model architecture
            layer_indices: Layers to hook
            hook_types: Types of hooks to register
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping layer names to activation arrays
        """
        # Create hook manager
        with HookManager(model) as hook_manager:
            # Register hooks
            hook_manager.register_hooks(layer_indices, hook_types, architecture)

            # Process dataset in batches
            with torch.no_grad():
                for i in range(0, len(dataset), batch_size):
                    batch_end = min(i + batch_size, len(dataset))
                    batch = dataset[i:batch_end]

                    # Get input_ids from batch
                    if isinstance(batch, dict):
                        input_ids = batch.get("input_ids")
                    else:
                        input_ids = [item["input_ids"] for item in batch]

                    # Convert to tensor and move to device
                    if not isinstance(input_ids, torch.Tensor):
                        input_ids = torch.tensor(input_ids)

                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)

                    input_ids = input_ids.to(model.device)

                    # Run forward pass (hooks will capture activations)
                    _ = model(input_ids)

                    if (i // batch_size + 1) % 10 == 0:
                        logger.info(f"Processed {batch_end}/{len(dataset)} samples")

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
