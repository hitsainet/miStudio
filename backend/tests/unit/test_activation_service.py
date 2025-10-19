"""
Tests for ActivationService.

This module tests the ActivationService class which orchestrates activation
extraction from transformer models.

NO MOCKING - Uses real PyTorch models, real datasets, and actual forward passes.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from datasets import Dataset as HFDataset

from src.services.activation_service import ActivationService, ActivationExtractionError
from src.models.model import QuantizationFormat
from tests.unit.test_forward_hooks import SimpleLlamaModel, SimpleGPT2Model


# Fixtures


@pytest.fixture
def activation_service():
    """Create ActivationService with temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        service = ActivationService()
        # Override activations_dir to use temp directory
        service.activations_dir = Path(tmpdir) / "activations"
        service.activations_dir.mkdir(parents=True, exist_ok=True)
        yield service


@pytest.fixture
def real_model_dir():
    """Create a real model directory with HuggingFace format."""
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "model"
        model_dir.mkdir()

        # Create a tiny Llama config
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
        )

        # Create model from config
        model = LlamaForCausalLM(config)

        # Save model and config
        model.save_pretrained(model_dir)

        # Create a minimal tokenizer (use GPT2 as base, adjust vocab size)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(model_dir)

        yield str(model_dir)


@pytest.fixture
def real_dataset_dir():
    """Create a real HuggingFace dataset directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "dataset"
        dataset_dir.mkdir()

        # Create a real HuggingFace dataset with tokenized text
        data = {
            'input_ids': [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
            ],
            'attention_mask': [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        }

        dataset = HFDataset.from_dict(data)
        dataset.save_to_disk(str(dataset_dir))

        yield str(dataset_dir)


# Tests


class TestActivationService:
    """Test suite for ActivationService."""

    def test_initialization(self, activation_service):
        """Test ActivationService initializes correctly."""
        assert activation_service.activations_dir.exists()
        assert activation_service.activations_dir.is_dir()

    def test_extract_activations_basic(self, activation_service, real_model_dir, real_dataset_dir):
        """Test basic activation extraction with real model and dataset."""
        result = activation_service.extract_activations(
            model_id="test-model-1",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0, 2],
            hook_types=["residual"],
            max_samples=5,
            batch_size=2,
            extraction_id="test-extraction-1"
        )

        # Verify result structure
        assert result["extraction_id"] == "test-extraction-1"
        assert result["num_samples"] == 5
        assert len(result["saved_files"]) == 2  # 2 layers
        assert "statistics" in result
        assert "metadata_path" in result

        # Verify files were created
        output_dir = Path(result["output_path"])
        assert output_dir.exists()
        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "layer_0_residual.npy").exists()
        assert (output_dir / "layer_2_residual.npy").exists()

    def test_extract_multiple_hook_types(self, activation_service, real_model_dir, real_dataset_dir):
        """Test extraction with multiple hook types."""
        result = activation_service.extract_activations(
            model_id="test-model-2",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0, 1],
            hook_types=["residual", "mlp", "attention"],
            max_samples=5,
            batch_size=2
        )

        # Should have 2 layers × 3 hook types = 6 files
        assert len(result["saved_files"]) == 6

        expected_files = [
            "layer_0_residual.npy", "layer_0_mlp.npy", "layer_0_attention.npy",
            "layer_1_residual.npy", "layer_1_mlp.npy", "layer_1_attention.npy"
        ]

        output_dir = Path(result["output_path"])
        for filename in expected_files:
            assert (output_dir / filename).exists()

    def test_max_samples_limit(self, activation_service, real_model_dir, real_dataset_dir):
        """Test that max_samples correctly limits dataset size."""
        # Extract with max_samples=3 (dataset has 5 samples)
        result = activation_service.extract_activations(
            model_id="test-model-3",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=3,
            batch_size=2
        )

        assert result["num_samples"] == 3

        # Load the saved activations and check shape
        import numpy as np
        output_dir = Path(result["output_path"])
        activations = np.load(output_dir / "layer_0_residual.npy")

        # Shape should be (3, 10, 64) - 3 samples, seq_len=10, hidden_dim=64
        assert activations.shape[0] == 3

    def test_batch_size_processing(self, activation_service, real_model_dir, real_dataset_dir):
        """Test that different batch sizes produce consistent shapes and results."""
        # Extract with batch_size=1
        result1 = activation_service.extract_activations(
            model_id="test-model-4a",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=5,
            batch_size=1
        )

        # Extract with batch_size=5 (all at once)
        result2 = activation_service.extract_activations(
            model_id="test-model-4b",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=5,
            batch_size=5
        )

        # Load activations
        import numpy as np
        act1 = np.load(Path(result1["output_path"]) / "layer_0_residual.npy")
        act2 = np.load(Path(result2["output_path"]) / "layer_0_residual.npy")

        # Should have same shape
        assert act1.shape == act2.shape == (5, 10, 64)

        # Both should process all samples
        assert result1["num_samples"] == result2["num_samples"] == 5

    def test_statistics_calculation(self, activation_service, real_model_dir, real_dataset_dir):
        """Test that activation statistics are correctly calculated."""
        result = activation_service.extract_activations(
            model_id="test-model-5",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=5,
            batch_size=2
        )

        stats = result["statistics"]["layer_0_residual"]

        # Verify all statistics are present and valid
        assert "shape" in stats
        assert "mean_magnitude" in stats
        assert "max_activation" in stats
        assert "min_activation" in stats
        assert "std_activation" in stats
        assert "sparsity_percent" in stats
        assert "size_mb" in stats

        # Shape should be [5, 10, 64]
        assert stats["shape"] == [5, 10, 64]

        # All statistics should be non-negative
        assert stats["mean_magnitude"] >= 0
        assert stats["max_activation"] >= 0
        assert stats["min_activation"] >= 0
        assert stats["std_activation"] >= 0
        assert 0 <= stats["sparsity_percent"] <= 100
        assert stats["size_mb"] > 0

    def test_metadata_contents(self, activation_service, real_model_dir, real_dataset_dir):
        """Test that metadata.json contains all required fields."""
        import json

        result = activation_service.extract_activations(
            model_id="test-model-6",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0, 1],
            hook_types=["residual", "mlp"],
            max_samples=5,
            batch_size=2,
            extraction_id="test-extraction-6"
        )

        # Load and verify metadata
        metadata_path = Path(result["metadata_path"])
        assert metadata_path.exists()

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Verify required fields
        assert metadata["extraction_id"] == "test-extraction-6"
        assert metadata["model_id"] == "test-model-6"
        assert metadata["architecture"] == "llama"
        assert metadata["quantization"] == "FP32"
        assert metadata["dataset_path"] == real_dataset_dir
        assert metadata["layer_indices"] == [0, 1]
        assert metadata["hook_types"] == ["residual", "mlp"]
        assert metadata["max_samples"] == 5
        assert metadata["batch_size"] == 2
        assert metadata["num_samples_processed"] == 5
        assert "created_at" in metadata
        assert "saved_files" in metadata
        assert "statistics" in metadata

    def test_get_extraction_info(self, activation_service, real_model_dir, real_dataset_dir):
        """Test retrieving extraction information."""
        # Create an extraction
        result = activation_service.extract_activations(
            model_id="test-model-7",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=5,
            batch_size=2,
            extraction_id="test-extraction-7"
        )

        # Retrieve info
        info = activation_service.get_extraction_info("test-extraction-7")

        assert info is not None
        assert info["extraction_id"] == "test-extraction-7"
        assert info["model_id"] == "test-model-7"

    def test_get_extraction_info_nonexistent(self, activation_service):
        """Test retrieving info for nonexistent extraction."""
        info = activation_service.get_extraction_info("nonexistent-extraction")
        assert info is None

    def test_list_extractions(self, activation_service, real_model_dir, real_dataset_dir):
        """Test listing all extractions."""
        # Create multiple extractions
        for i in range(3):
            activation_service.extract_activations(
                model_id=f"test-model-{i}",
                model_path=real_model_dir,
                architecture="llama",
                quantization=QuantizationFormat.FP32,
                dataset_path=real_dataset_dir,
                layer_indices=[0],
                hook_types=["residual"],
                max_samples=5,
                batch_size=2,
                extraction_id=f"test-extraction-{i}"
            )

        # List all
        extractions = activation_service.list_extractions()

        assert len(extractions) == 3
        extraction_ids = [e["extraction_id"] for e in extractions]
        assert "test-extraction-0" in extraction_ids
        assert "test-extraction-1" in extraction_ids
        assert "test-extraction-2" in extraction_ids

    def test_delete_extraction(self, activation_service, real_model_dir, real_dataset_dir):
        """Test deleting an extraction."""
        # Create extraction
        result = activation_service.extract_activations(
            model_id="test-model-8",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=5,
            batch_size=2,
            extraction_id="test-extraction-8"
        )

        output_dir = Path(result["output_path"])
        assert output_dir.exists()

        # Delete
        success = activation_service.delete_extraction("test-extraction-8")
        assert success is True

        # Verify deleted
        assert not output_dir.exists()
        info = activation_service.get_extraction_info("test-extraction-8")
        assert info is None

    def test_delete_nonexistent_extraction(self, activation_service):
        """Test deleting nonexistent extraction."""
        success = activation_service.delete_extraction("nonexistent")
        assert success is False

    def test_auto_generated_extraction_id(self, activation_service, real_model_dir, real_dataset_dir):
        """Test that extraction_id is auto-generated if not provided."""
        result = activation_service.extract_activations(
            model_id="test-model-9",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=5,
            batch_size=2
            # No extraction_id provided
        )

        # Should have auto-generated ID
        assert "extraction_id" in result
        assert result["extraction_id"].startswith("ext_test-model-9_")

    def test_invalid_dataset_path(self, activation_service, real_model_dir):
        """Test error handling for invalid dataset path."""
        with pytest.raises(FileNotFoundError):
            activation_service._load_dataset("/nonexistent/path", 10)

    def test_architecture_propagation(self, activation_service, real_model_dir, real_dataset_dir):
        """Test that architecture is correctly passed to hooks."""
        # This will fail if architecture isn't properly propagated to HookManager
        result = activation_service.extract_activations(
            model_id="test-model-10",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0, 1, 2],
            hook_types=["residual", "mlp", "attention"],
            max_samples=5,
            batch_size=2
        )

        # Should successfully extract all requested hooks
        assert len(result["saved_files"]) == 9  # 3 layers × 3 hook types

    def test_activation_shapes(self, activation_service, real_model_dir, real_dataset_dir):
        """Test that saved activation arrays have correct shapes."""
        import numpy as np

        result = activation_service.extract_activations(
            model_id="test-model-11",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0, 1],
            hook_types=["residual"],
            max_samples=5,
            batch_size=2
        )

        output_dir = Path(result["output_path"])

        # Load and check shapes
        for layer_idx in [0, 1]:
            activations = np.load(output_dir / f"layer_{layer_idx}_residual.npy")

            # Shape should be [num_samples, seq_len, hidden_dim]
            # = [5, 10, 64]
            assert activations.shape == (5, 10, 64)
            # System uses FP16 for memory efficiency (Phase 18 optimization)
            assert activations.dtype == np.float16

    def test_different_architectures(self, activation_service, real_dataset_dir):
        """Test extraction works with different model architectures."""
        from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test GPT-2 architecture
            gpt2_dir = Path(tmpdir) / "gpt2"
            gpt2_dir.mkdir()

            # Create a tiny GPT-2 config
            config = GPT2Config(
                vocab_size=1000,
                n_embd=64,
                n_layer=3,
                n_head=4,
                n_positions=512,
            )

            # Create model from config
            gpt2_model = GPT2LMHeadModel(config)
            gpt2_model.save_pretrained(gpt2_dir)

            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(gpt2_dir)

            result = activation_service.extract_activations(
                model_id="test-gpt2",
                model_path=str(gpt2_dir),
                architecture="gpt2",
                quantization=QuantizationFormat.FP32,
                dataset_path=real_dataset_dir,
                layer_indices=[0, 1],
                hook_types=["residual", "mlp"],
                max_samples=5,
                batch_size=2
            )

            # Should successfully extract
            assert len(result["saved_files"]) == 4  # 2 layers × 2 hook types
            assert result["num_samples"] == 5

    def test_all_samples_when_max_zero(self, activation_service, real_model_dir, real_dataset_dir):
        """Test that max_samples=0 or negative uses all samples."""
        result = activation_service.extract_activations(
            model_id="test-model-12",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=0,  # Should use all samples
            batch_size=2
        )

        # Should process all 5 samples in dataset
        assert result["num_samples"] == 5

    def test_extraction_consistency(self, activation_service, real_model_dir, real_dataset_dir):
        """Test that running extraction twice produces identical results."""
        import numpy as np

        # First extraction
        result1 = activation_service.extract_activations(
            model_id="test-model-13a",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=5,
            batch_size=2
        )

        # Second extraction (same parameters)
        result2 = activation_service.extract_activations(
            model_id="test-model-13b",
            model_path=real_model_dir,
            architecture="llama",
            quantization=QuantizationFormat.FP32,
            dataset_path=real_dataset_dir,
            layer_indices=[0],
            hook_types=["residual"],
            max_samples=5,
            batch_size=2
        )

        # Load activations
        act1 = np.load(Path(result1["output_path"]) / "layer_0_residual.npy")
        act2 = np.load(Path(result2["output_path"]) / "layer_0_residual.npy")

        # Should be identical
        assert np.array_equal(act1, act2)
