"""
Unit tests for vectorized feature extraction utilities.

Tests the vectorization functions that achieve 10-50x speedup by replacing
sequential Python loops with GPU/CPU SIMD operations.
"""

import numpy as np
import pytest
import torch

from src.services.extraction_vectorized import (
    IncrementalTopKHeap,
    batch_process_features,
    calculate_optimal_vectorization_batch,
    get_vectorization_config,
)


class TestIncrementalTopKHeap:
    """Test IncrementalTopKHeap class for incremental heap construction."""

    def test_initialization(self):
        """Test heap initialization."""
        heap = IncrementalTopKHeap(num_features=100, top_k=10)
        assert heap.num_features == 100
        assert heap.top_k == 10
        assert len(heap.heaps) == 0  # No heaps initially

    def test_add_batch(self):
        """Test adding batch of examples."""
        heap = IncrementalTopKHeap(num_features=10, top_k=5)

        # Add batch of examples
        feature_indices = np.array([0, 1, 2, 0, 1])
        max_activations = np.array([1.5, 2.0, 0.5, 1.0, 2.5])
        examples = [
            {"sample_index": 0, "max_activation": 1.5},
            {"sample_index": 1, "max_activation": 2.0},
            {"sample_index": 2, "max_activation": 0.5},
            {"sample_index": 3, "max_activation": 1.0},
            {"sample_index": 4, "max_activation": 2.5},
        ]

        heap.add_batch(feature_indices, max_activations, examples)

        # Verify heaps were created for 3 unique features (0, 1, 2)
        assert len(heap.heaps) == 3
        # Verify examples were processed
        assert heap.examples_processed == 5

    def test_get_heaps_basic(self):
        """Test getting heaps from incremental data."""
        heap = IncrementalTopKHeap(num_features=3, top_k=2)

        # Add examples for features 0 and 1
        feature_indices = np.array([0, 0, 1, 1, 0])
        max_activations = np.array([1.5, 2.0, 0.5, 1.0, 3.0])
        examples = [
            {"sample_index": i, "max_activation": float(max_activations[i])}
            for i in range(5)
        ]

        heap.add_batch(feature_indices, max_activations, examples)

        # Get final heaps
        final_heaps = heap.get_heaps()

        # Verify structure
        assert len(final_heaps) == 3  # All features should be present

        # Feature 0: should have top-2 examples (3.0, 2.0)
        assert len(final_heaps[0]) == 2
        activations = [act for act, _ in final_heaps[0]]
        assert activations == [3.0, 2.0]  # Sorted descending

        # Feature 1: should have 2 examples (1.0, 0.5)
        assert len(final_heaps[1]) == 2
        activations = [act for act, _ in final_heaps[1]]
        assert activations == [1.0, 0.5]

        # Feature 2: should have 0 examples (never activated)
        assert len(final_heaps[2]) == 0

    def test_get_heaps_respects_top_k(self):
        """Test that only top-k examples are kept per feature."""
        heap = IncrementalTopKHeap(num_features=1, top_k=3)

        # Add 5 examples for feature 0
        feature_indices = np.array([0, 0, 0, 0, 0])
        max_activations = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
        examples = [
            {"sample_index": i, "max_activation": float(max_activations[i])}
            for i in range(5)
        ]

        heap.add_batch(feature_indices, max_activations, examples)

        # Get final heaps
        final_heaps = heap.get_heaps()

        # Verify only top-3 are kept
        assert len(final_heaps[0]) == 3
        activations = [act for act, _ in final_heaps[0]]
        assert activations == [5.0, 4.0, 3.0]  # Top-3 in descending order

    def test_get_heaps_filters_zero_activations(self):
        """Test that zero activations are not stored."""
        heap = IncrementalTopKHeap(num_features=2, top_k=5)

        # Add examples with some zero activations
        feature_indices = np.array([0, 0, 1, 1])
        max_activations = np.array([1.0, 0.0, 2.0, 0.0])
        examples = [
            {"sample_index": i, "max_activation": float(max_activations[i])}
            for i in range(4)
        ]

        heap.add_batch(feature_indices, max_activations, examples)

        # Get final heaps
        final_heaps = heap.get_heaps()

        # Only non-zero activations should be stored
        assert len(final_heaps[0]) == 1  # Only 1.0
        assert len(final_heaps[1]) == 1  # Only 2.0


class TestBatchProcessFeatures:
    """Test batch_process_features function for vectorized processing."""

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing."""
        # Batch of 2 samples, seq_len=4, 3 features
        # Note: Token filtering is applied - single chars ("a") will be filtered
        batch_sae_features = torch.tensor([
            # Sample 0: feature 0 activates at pos 1, feature 2 at pos 2
            [
                [0.0, 0.0, 0.0],  # pos 0
                [2.0, 0.0, 0.0],  # pos 1: feature 0 max at "cat"
                [0.0, 0.0, 3.0],  # pos 2: feature 2 max at "sat"
                [1.0, 0.0, 1.0],  # pos 3
            ],
            # Sample 1: feature 1 activates at pos 1 (not pos 0 to avoid filtered "a")
            [
                [0.0, 1.0, 0.0],  # pos 0: "the" - not max
                [0.0, 4.0, 0.0],  # pos 1: feature 1 max at "dog"
                [0.0, 0.0, 0.0],  # pos 2
                [0.0, 0.0, 0.0],  # pos 3
            ],
        ], dtype=torch.float32)

        token_strings_batch = [
            ["the", "cat", "sat", "down"],
            ["the", "dog", "ran", "fast"],  # Changed "a" to "the" to avoid filtering
        ]

        sample_indices = [0, 1]

        return batch_sae_features, token_strings_batch, sample_indices

    def test_batch_process_features_basic(self, sample_batch):
        """Test basic vectorized feature processing."""
        batch_sae_features, token_strings_batch, sample_indices = sample_batch

        feature_indices, max_activations, examples = batch_process_features(
            batch_sae_features=batch_sae_features,
            token_strings_batch=token_strings_batch,
            sample_indices=sample_indices,
            vectorization_batch_size=2,
            top_k=2
        )

        # Verify results
        # Should find activations for features: 0 (sample 0), 2 (sample 0), 1 (sample 1)
        assert len(feature_indices) == 3
        assert len(max_activations) == 3
        assert len(examples) == 3

        # Convert to lists for easier checking
        feature_indices_list = feature_indices.tolist()
        max_activations_list = max_activations.tolist()

        # Check that we found the expected features
        assert 0 in feature_indices_list  # Feature 0 from sample 0
        assert 2 in feature_indices_list  # Feature 2 from sample 0
        assert 1 in feature_indices_list  # Feature 1 from sample 1

        # Check feature 0 from sample 0 (max activation 2.0 at position 1 "cat")
        for i, feat_idx in enumerate(feature_indices_list):
            if feat_idx == 0:
                assert max_activations_list[i] == pytest.approx(2.0)
                assert examples[i]["sample_index"] == 0
                assert examples[i]["max_activation"] == pytest.approx(2.0)
                break

        # Check feature 2 from sample 0 (max activation 3.0 at position 2 "sat")
        for i, feat_idx in enumerate(feature_indices_list):
            if feat_idx == 2:
                assert max_activations_list[i] == pytest.approx(3.0)
                assert examples[i]["sample_index"] == 0
                assert examples[i]["max_activation"] == pytest.approx(3.0)
                break

        # Check feature 1 from sample 1 (max activation 4.0 at position 1 "dog")
        for i, feat_idx in enumerate(feature_indices_list):
            if feat_idx == 1:
                assert max_activations_list[i] == pytest.approx(4.0)
                assert examples[i]["sample_index"] == 1
                assert examples[i]["max_activation"] == pytest.approx(4.0)
                break

    def test_batch_process_features_vectorization_batch_size(self, sample_batch):
        """Test different vectorization batch sizes."""
        batch_sae_features, token_strings_batch, sample_indices = sample_batch

        # Test with batch_size=1 (process one sample at a time)
        feature_indices_1, max_activations_1, examples_1 = batch_process_features(
            batch_sae_features=batch_sae_features,
            token_strings_batch=token_strings_batch,
            sample_indices=sample_indices,
            vectorization_batch_size=1,
            top_k=2
        )

        # Test with batch_size=2 (process all samples at once)
        feature_indices_2, max_activations_2, examples_2 = batch_process_features(
            batch_sae_features=batch_sae_features,
            token_strings_batch=token_strings_batch,
            sample_indices=sample_indices,
            vectorization_batch_size=2,
            top_k=2
        )

        # Results should be identical regardless of vectorization batch size
        assert len(feature_indices_1) == len(feature_indices_2)
        assert len(max_activations_1) == len(max_activations_2)

    def test_batch_process_features_filters_zero_activations(self):
        """Test that zero activations are not included in results."""
        # Batch with one sample, all zero activations
        batch_sae_features = torch.zeros((1, 4, 3), dtype=torch.float32)
        token_strings_batch = [["the", "cat", "sat", "down"]]
        sample_indices = [0]

        feature_indices, max_activations, examples = batch_process_features(
            batch_sae_features=batch_sae_features,
            token_strings_batch=token_strings_batch,
            sample_indices=sample_indices,
            vectorization_batch_size=1,
            top_k=2
        )

        # Should return no features (all zero)
        assert len(feature_indices) == 0
        assert len(max_activations) == 0
        assert len(examples) == 0


class TestCalculateOptimalVectorizationBatch:
    """Test calculate_optimal_vectorization_batch function."""

    def test_basic_calculation(self):
        """Test basic batch size calculation."""
        # 10 GB available, 16384 features, 512 seq_len
        # Memory per sample: 512 * 16384 * 4 bytes = 32 MB
        # Usable: 10 GB * 0.8 = 8 GB = 8192 MB
        # Batch size: 8192 / 32 = 256
        batch_size = calculate_optimal_vectorization_batch(
            available_vram_gb=10.0,
            latent_dim=16384,
            seq_len=512,
            safety_margin=0.2
        )

        assert batch_size == 256  # Clamped to max

    def test_low_memory(self):
        """Test batch size calculation with low memory."""
        # 2 GB available, 16384 features, 512 seq_len
        # Memory per sample: 32 MB
        # Usable: 2 GB * 0.8 = 1.6 GB = 1638 MB
        # Batch size: 1638 / 32 = 51
        batch_size = calculate_optimal_vectorization_batch(
            available_vram_gb=2.0,
            latent_dim=16384,
            seq_len=512,
            safety_margin=0.2
        )

        assert 32 <= batch_size <= 64  # Should be around 51

    def test_clamping_to_min(self):
        """Test that batch size is clamped to minimum 1."""
        # Very low memory
        batch_size = calculate_optimal_vectorization_batch(
            available_vram_gb=0.01,
            latent_dim=16384,
            seq_len=512,
            safety_margin=0.2
        )

        assert batch_size == 1  # Clamped to minimum

    def test_clamping_to_max(self):
        """Test that batch size is clamped to maximum 256."""
        # Very high memory
        batch_size = calculate_optimal_vectorization_batch(
            available_vram_gb=100.0,
            latent_dim=16384,
            seq_len=512,
            safety_margin=0.2
        )

        assert batch_size == 256  # Clamped to maximum


class TestGetVectorizationConfig:
    """Test get_vectorization_config function."""

    def test_auto_mode(self):
        """Test auto mode calculation."""
        config = {"vectorization_batch_size": "auto"}

        batch_size = get_vectorization_config(
            config=config,
            available_vram_gb=10.0,
            latent_dim=16384,
            seq_len=512
        )

        # Should calculate optimal size
        assert 1 <= batch_size <= 256

    def test_manual_mode(self):
        """Test manual batch size specification."""
        config = {"vectorization_batch_size": 128}

        batch_size = get_vectorization_config(
            config=config,
            available_vram_gb=10.0,
            latent_dim=16384,
            seq_len=512
        )

        assert batch_size == 128

    def test_default_fallback(self):
        """Test fallback to default when config missing."""
        config = {}

        batch_size = get_vectorization_config(
            config=config,
            available_vram_gb=10.0,
            latent_dim=16384,
            seq_len=512
        )

        # Should default to auto mode
        assert 1 <= batch_size <= 256

    def test_invalid_batch_size_fallback(self):
        """Test fallback to 64 when invalid batch size specified."""
        config = {"vectorization_batch_size": 1000}  # > 256 (invalid)

        batch_size = get_vectorization_config(
            config=config,
            available_vram_gb=None,
            latent_dim=None,
            seq_len=None
        )

        assert batch_size == 64  # Fallback

    def test_auto_mode_missing_params_fallback(self):
        """Test fallback when auto mode but missing required params."""
        config = {"vectorization_batch_size": "auto"}

        batch_size = get_vectorization_config(
            config=config,
            available_vram_gb=None,  # Missing
            latent_dim=None,
            seq_len=None
        )

        assert batch_size == 64  # Fallback
