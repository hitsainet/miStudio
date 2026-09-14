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

        feature_indices, max_activations, examples, fired = batch_process_features(
            batch_sae_features=batch_sae_features,
            token_strings_batch=token_strings_batch,
            sample_indices=sample_indices,
            vectorization_batch_size=2,
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
        feature_indices_1, max_activations_1, examples_1, _fired1 = batch_process_features(
            batch_sae_features=batch_sae_features,
            token_strings_batch=token_strings_batch,
            sample_indices=sample_indices,
            vectorization_batch_size=1,
        )

        # Test with batch_size=2 (process all samples at once)
        feature_indices_2, max_activations_2, examples_2, _fired2 = batch_process_features(
            batch_sae_features=batch_sae_features,
            token_strings_batch=token_strings_batch,
            sample_indices=sample_indices,
            vectorization_batch_size=2,
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

        feature_indices, max_activations, examples, fired = batch_process_features(
            batch_sae_features=batch_sae_features,
            token_strings_batch=token_strings_batch,
            sample_indices=sample_indices,
            vectorization_batch_size=1,
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


class TestTrueActivationFrequencyIsCountedPreFilter:
    """A feature's firing rate, counted before the junk-token filter sees it.

    WHY THIS EXISTS
    ---------------
    `features.activation_frequency` is derived from the examples this function
    EMITS, and emission happens after `is_junk_token` has discarded any
    (sample, feature) pair whose peak landed on a filtered token. So the column
    has always meant "fraction of samples where the peak was on a NON-JUNK
    token", not "fraction of samples where the feature fired".

    Five of six filters default on, and `filter_fragments` matches a ~500-entry
    blocklist containing ordinary words — `land`, `field`, `like`, `some`,
    `one`. Measured on feature eb48_00046: that blocklist deletes documents
    whose peak landed on a CONVENTIONAL affix while letting unconventional
    splits through, which is how the feature came to be labelled "technical
    identifiers" when what it detects is a tokenizer seam. The filter was
    selecting for the appearance that produced the label.

    The old column is NOT corrected in place — it feeds the dead-neuron gate and
    a steering auto-baseline whose constants were fitted against it.

    MUTATION CONTROLS:
      C80 count from len(all_examples) (post-filter) instead of the tensor
           -> test_a_junk_prime_still_counts_as_a_firing
      C81 drop the fired_accum accumulation across sub-batches
           -> test_counts_accumulate_across_sub_batches
      C82 re-add `top_k=5` at the caller
           -> TypeError, test_the_dead_top_k_parameter_is_gone
    """

    @staticmethod
    def _batch(activations):
        """(1, seq_len, latent_dim) from a list of per-token feature vectors."""
        return torch.tensor([activations], dtype=torch.float32)

    def test_a_junk_prime_still_counts_as_a_firing(self):
        """C80. The whole point: a filtered peak is still a firing.

        `.` is punctuation, so `filter_punctuation` discards the entire
        (sample, feature) pair — no example is emitted. The feature fired all
        the same, and the true count must say so.
        """
        # One sample, three tokens, one feature. Peak is on the '.' token.
        batch = self._batch([[0.1], [0.9], [0.2]])

        indices, _acts, examples, fired = batch_process_features(
            batch_sae_features=batch,
            token_strings_batch=[["the", ".", "server"]],
            sample_indices=[0],
            vectorization_batch_size=1,
            filter_punctuation=True,
        )

        assert len(examples) == 0, (
            "the junk filter did not discard a punctuation prime; this fixture "
            "cannot exhibit the defect"
        )
        assert len(indices) == 0
        assert fired[0] == 1, (
            "the feature fired and was not counted — the true frequency is "
            "still measuring what survived the filter"
        )

    def test_a_clean_prime_is_counted_once(self):
        """Negative control: the counter must not double-count.

        A counter that incremented per token rather than per sample would pass
        the test above and make every frequency exceed 1.0.
        """
        batch = self._batch([[0.1], [0.9], [0.2]])

        _i, _a, examples, fired = batch_process_features(
            batch_sae_features=batch,
            token_strings_batch=[["the", "server", "ran"]],
            sample_indices=[0],
            vectorization_batch_size=1,
        )

        assert len(examples) == 1
        assert fired[0] == 1

    def test_a_feature_that_never_fires_is_not_counted(self):
        """Zero activations must not register. The test is strict `>`."""
        batch = self._batch([[0.0], [0.0], [0.0]])

        _i, _a, examples, fired = batch_process_features(
            batch_sae_features=batch,
            token_strings_batch=[["the", "server", "ran"]],
            sample_indices=[0],
            vectorization_batch_size=1,
        )
        assert len(examples) == 0
        assert fired[0] == 0

    def test_counts_accumulate_across_sub_batches(self):
        """C81. The accumulator must survive the sub-batch loop.

        `vectorization_batch_size=1` over three samples runs the accumulation
        path three times. An accumulator reset each pass would report 1.
        """
        batch = torch.tensor(
            [[[0.9], [0.1]], [[0.8], [0.1]], [[0.7], [0.1]]],
            dtype=torch.float32,
        )

        _i, _a, _e, fired = batch_process_features(
            batch_sae_features=batch,
            token_strings_batch=[["server", "ran"]] * 3,
            sample_indices=[0, 1, 2],
            vectorization_batch_size=1,
        )

        assert fired[0] == 3, (
            f"expected 3 firings across 3 sub-batches, got {fired[0]}"
        )

    def test_the_count_never_exceeds_the_sample_count(self):
        """The invariant that makes it a FREQUENCY when divided by len(dataset).

        A count above the sample total would produce a frequency above 1.0,
        which `cluster_allocation_service` validates and would reject.
        """
        batch = torch.tensor(
            [[[0.9], [0.8], [0.7]], [[0.6], [0.5], [0.4]]],
            dtype=torch.float32,
        )

        _i, _a, _e, fired = batch_process_features(
            batch_sae_features=batch,
            token_strings_batch=[["a", "b", "c"]] * 2,
            sample_indices=[0, 1],
            vectorization_batch_size=2,
        )
        assert fired[0] <= 2

    def test_the_dead_top_k_parameter_is_gone(self):
        """C82. Removing the PARAMETER is what makes the deletion enforceable.

        `torch.topk(..., k=5, dim=2)` and two `.cpu().numpy()` transfers ran on
        every sub-batch and their results were read nowhere in the repo — the
        top-5 positions WITHIN each document, computed, copied to host, and
        discarded.

        Deleting only the body would leave the parameter as an invitation to
        re-add it, and a source-scrape guard against that fails open — this repo
        has recorded exactly that twice. With the parameter gone, re-adding the
        call is a TypeError, which no scan can miss.
        """
        import inspect

        params = inspect.signature(batch_process_features).parameters
        assert "top_k" not in params, (
            "the dead top_k parameter is back; its results are still unread"
        )

        with pytest.raises(TypeError):
            batch_process_features(
                batch_sae_features=self._batch([[0.9]]),
                token_strings_batch=[["server"]],
                sample_indices=[0],
                vectorization_batch_size=1,
                top_k=5,
            )
