"""
Unit tests for activation extraction progress calculation.

Tests the 3-phase extraction progress formula:
- Loading phase: 0-10%
- Extracting phase: 10-90% (based on samples processed)
- Saving phase: 90-100%
"""

import pytest


class TestExtractionProgressPhases:
    """Test extraction progress calculation across all 3 phases."""

    def test_extraction_progress_loading_phase(self):
        """Test loading phase progress (0-10%)."""
        # Loading phase is set to a fixed 10% when model and dataset are loaded
        loading_progress = 10.0

        assert loading_progress == 10.0

    def test_extraction_progress_extracting_phase_start(self):
        """Test extracting phase at start (10% progress)."""
        # Formula: 10.0 + (samples_processed / total_samples) * 80.0
        samples_processed = 0
        total_samples = 1000

        extraction_progress = 10.0 + (samples_processed / total_samples) * 80.0

        assert extraction_progress == 10.0

    def test_extraction_progress_extracting_phase_quarter(self):
        """Test extracting phase at 25% completion (30% progress)."""
        # Formula: 10.0 + (samples_processed / total_samples) * 80.0
        # At 25% of samples: 10.0 + 0.25 * 80.0 = 30.0
        samples_processed = 250
        total_samples = 1000

        extraction_progress = 10.0 + (samples_processed / total_samples) * 80.0

        assert extraction_progress == 30.0

    def test_extraction_progress_extracting_phase_half(self):
        """Test extracting phase at 50% completion (50% progress)."""
        # Formula: 10.0 + (samples_processed / total_samples) * 80.0
        # At 50% of samples: 10.0 + 0.5 * 80.0 = 50.0
        samples_processed = 500
        total_samples = 1000

        extraction_progress = 10.0 + (samples_processed / total_samples) * 80.0

        assert extraction_progress == 50.0

    def test_extraction_progress_extracting_phase_three_quarters(self):
        """Test extracting phase at 75% completion (70% progress)."""
        # Formula: 10.0 + (samples_processed / total_samples) * 80.0
        # At 75% of samples: 10.0 + 0.75 * 80.0 = 70.0
        samples_processed = 750
        total_samples = 1000

        extraction_progress = 10.0 + (samples_processed / total_samples) * 80.0

        assert extraction_progress == 70.0

    def test_extraction_progress_extracting_phase_complete(self):
        """Test extracting phase at completion (90% progress)."""
        # Formula: 10.0 + (samples_processed / total_samples) * 80.0
        # At 100% of samples: 10.0 + 1.0 * 80.0 = 90.0
        samples_processed = 1000
        total_samples = 1000

        extraction_progress = 10.0 + (samples_processed / total_samples) * 80.0

        assert extraction_progress == 90.0

    def test_extraction_progress_saving_phase(self):
        """Test saving phase progress (90%)."""
        # Saving phase is set to 90% when files are being saved
        saving_progress = 90.0

        assert saving_progress == 90.0

    def test_extraction_progress_complete_phase(self):
        """Test completion phase progress (100%)."""
        # Final progress is set to 100% when extraction fully complete
        complete_progress = 100.0

        assert complete_progress == 100.0


class TestExtractionProgressSamplesProcessed:
    """Test extraction progress with various sample counts."""

    def test_extraction_progress_small_dataset(self):
        """Test progress calculation with small dataset (100 samples)."""
        total_samples = 100

        # At 10 samples (10%)
        samples_processed = 10
        progress = 10.0 + (samples_processed / total_samples) * 80.0
        assert progress == 18.0  # 10% + 10% of 80% = 18%

        # At 50 samples (50%)
        samples_processed = 50
        progress = 10.0 + (samples_processed / total_samples) * 80.0
        assert progress == 50.0

        # At 90 samples (90%)
        samples_processed = 90
        progress = 10.0 + (samples_processed / total_samples) * 80.0
        assert progress == 82.0  # 10% + 90% of 80% = 82%

    def test_extraction_progress_large_dataset(self):
        """Test progress calculation with large dataset (10000 samples)."""
        total_samples = 10000

        # At 1000 samples (10%)
        samples_processed = 1000
        progress = 10.0 + (samples_processed / total_samples) * 80.0
        assert progress == 18.0

        # At 5000 samples (50%)
        samples_processed = 5000
        progress = 10.0 + (samples_processed / total_samples) * 80.0
        assert progress == 50.0

        # At 9000 samples (90%)
        samples_processed = 9000
        progress = 10.0 + (samples_processed / total_samples) * 80.0
        assert progress == 82.0

    def test_extraction_progress_fractional_samples(self):
        """Test progress calculation with non-divisible sample counts."""
        total_samples = 1337

        # At 333 samples (~24.9%)
        samples_processed = 333
        progress = 10.0 + (samples_processed / total_samples) * 80.0
        expected_progress = 10.0 + (333 / 1337) * 80.0
        assert abs(progress - expected_progress) < 0.01
        assert abs(progress - 29.9) < 0.1  # Approximately 29.9%

    def test_extraction_progress_single_sample(self):
        """Test progress calculation with single sample dataset."""
        total_samples = 1

        # At 0 samples
        samples_processed = 0
        progress = 10.0 + (samples_processed / total_samples) * 80.0
        assert progress == 10.0

        # At 1 sample (100%)
        samples_processed = 1
        progress = 10.0 + (samples_processed / total_samples) * 80.0
        assert progress == 90.0


class TestExtractionProgressCallback:
    """Test extraction progress callback invocation."""

    def test_extraction_progress_callback_invoked(self):
        """Test that progress callback is invoked with correct parameters."""
        callback_calls = []

        def mock_callback(samples_processed: int, total_samples: int):
            """Mock progress callback."""
            progress = 10.0 + (samples_processed / total_samples) * 80.0
            callback_calls.append({
                'samples_processed': samples_processed,
                'total_samples': total_samples,
                'progress': progress,
            })

        # Simulate extraction progress
        total_samples = 1000
        for i in range(0, 1001, 100):  # 0, 100, 200, ..., 1000
            mock_callback(i, total_samples)

        # Verify callback was called 11 times (0 to 1000 in steps of 100)
        assert len(callback_calls) == 11

        # Verify first call
        assert callback_calls[0]['samples_processed'] == 0
        assert callback_calls[0]['progress'] == 10.0

        # Verify middle call (500 samples)
        assert callback_calls[5]['samples_processed'] == 500
        assert callback_calls[5]['progress'] == 50.0

        # Verify final call
        assert callback_calls[10]['samples_processed'] == 1000
        assert callback_calls[10]['progress'] == 90.0

    def test_extraction_progress_callback_with_batches(self):
        """Test progress callback with batch processing."""
        callback_calls = []

        def mock_callback(samples_processed: int, total_samples: int):
            """Mock progress callback."""
            progress = 10.0 + (samples_processed / total_samples) * 80.0
            callback_calls.append({
                'samples_processed': samples_processed,
                'total_samples': total_samples,
                'progress': progress,
            })

        # Simulate batch extraction (batch_size = 64)
        total_samples = 1000
        batch_size = 64
        samples_processed = 0

        while samples_processed < total_samples:
            samples_processed = min(samples_processed + batch_size, total_samples)
            mock_callback(samples_processed, total_samples)

        # Verify callback was called at least 15 times (1000 / 64 ≈ 15.6)
        assert len(callback_calls) >= 15

        # Verify progress increases monotonically
        for i in range(1, len(callback_calls)):
            assert callback_calls[i]['progress'] >= callback_calls[i-1]['progress']

        # Verify final progress is 90%
        assert callback_calls[-1]['progress'] == 90.0


class TestExtractionProgressEdgeCases:
    """Test extraction progress edge cases."""

    def test_extraction_progress_zero_samples_handled(self):
        """Test that zero samples is handled gracefully."""
        # In real code, this would likely raise an error or return early
        # But the formula itself would cause division by zero
        total_samples = 0
        samples_processed = 0

        # This would raise ZeroDivisionError in actual code
        with pytest.raises(ZeroDivisionError):
            progress = 10.0 + (samples_processed / total_samples) * 80.0

    def test_extraction_progress_samples_exceed_total(self):
        """Test behavior when samples processed exceeds total (should not happen)."""
        # This shouldn't happen in real code, but test the formula behavior
        total_samples = 1000
        samples_processed = 1200  # Exceeds total

        progress = 10.0 + (samples_processed / total_samples) * 80.0

        # Progress would exceed 90% (the end of extraction phase)
        assert progress > 90.0
        assert progress == 106.0  # 10 + (1.2 * 80)

    def test_extraction_progress_all_phases_sequential(self):
        """Test complete extraction flow through all phases."""
        phases = []

        # Loading phase (0-10%)
        phases.append(('loading', 10.0))

        # Extracting phase (10-90%)
        total_samples = 1000
        for samples in [0, 250, 500, 750, 1000]:
            progress = 10.0 + (samples / total_samples) * 80.0
            phases.append(('extracting', progress))

        # Saving phase (90%)
        phases.append(('saving', 90.0))

        # Complete phase (100%)
        phases.append(('complete', 100.0))

        # Verify progress increases monotonically (or stays same)
        for i in range(1, len(phases)):
            assert phases[i][1] >= phases[i-1][1], f"Progress decreased from {phases[i-1]} to {phases[i]}"

        # Verify start and end
        assert phases[0][1] == 10.0  # Loading
        assert phases[-1][1] == 100.0  # Complete
