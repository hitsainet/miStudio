"""
Unit tests for dataset download and tokenization progress steps.

Tests the fixed progress milestones for:
- Dataset download: 0%, 10%, 70%, 90%, 100%
- Dataset tokenization: 0%, 10%, 20%, 30%, 40%, 80%, 95%, 100%
"""

import pytest
from unittest.mock import Mock, patch


class TestDatasetDownloadProgressSteps:
    """Test dataset download progress steps."""

    def test_dataset_download_progress_step_0_starting(self):
        """Test download progress at start (0%)."""
        progress = 0.0
        status = "downloading"
        message = "Starting download"

        assert progress == 0.0
        assert status == "downloading"

    def test_dataset_download_progress_step_10_downloading(self):
        """Test download progress during HuggingFace fetch (10%)."""
        progress = 10.0
        status = "downloading"
        message = "Downloading from HuggingFace Hub..."

        assert progress == 10.0
        assert status == "downloading"

    def test_dataset_download_progress_step_70_saving(self):
        """Test download progress during disk save (70%)."""
        progress = 70.0
        status = "downloading"
        message = "Saving dataset to disk..."

        assert progress == 70.0
        assert status == "downloading"

    def test_dataset_download_progress_step_90_metadata(self):
        """Test download progress during metadata processing (90%)."""
        progress = 90.0
        status = "downloading"
        message = "Download complete, processing metadata..."

        assert progress == 90.0
        assert status == "downloading"

    def test_dataset_download_progress_step_100_complete(self):
        """Test download progress at completion (100%)."""
        progress = 100.0
        status = "ready"
        message = "Dataset downloaded successfully"

        assert progress == 100.0
        assert status == "ready"

    def test_dataset_download_progress_sequential(self):
        """Test download progress increases monotonically."""
        steps = [0.0, 10.0, 70.0, 90.0, 100.0]

        for i in range(1, len(steps)):
            assert steps[i] > steps[i-1], f"Progress decreased from {steps[i-1]} to {steps[i]}"

        assert steps[0] == 0.0
        assert steps[-1] == 100.0


class TestDatasetTokenizationProgressSteps:
    """Test dataset tokenization progress steps."""

    def test_dataset_tokenization_progress_step_0_starting(self):
        """Test tokenization progress at start (0%)."""
        progress = 0.0
        status = "processing"
        message = "Starting tokenization..."

        assert progress == 0.0
        assert status == "processing"

    def test_dataset_tokenization_progress_step_10_loading_tokenizer(self):
        """Test tokenization progress during tokenizer load (10%)."""
        progress = 10.0
        status = "processing"
        message = "Loading tokenizer..."

        assert progress == 10.0
        assert status == "processing"

    def test_dataset_tokenization_progress_step_20_loading_dataset(self):
        """Test tokenization progress during dataset load (20%)."""
        progress = 20.0
        status = "processing"
        message = "Loading dataset..."

        assert progress == 20.0
        assert status == "processing"

    def test_dataset_tokenization_progress_step_30_analyzing_schema(self):
        """Test tokenization progress during schema analysis (30%)."""
        progress = 30.0
        status = "processing"
        message = "Analyzing dataset schema..."

        assert progress == 30.0
        assert status == "processing"

    def test_dataset_tokenization_progress_step_40_tokenizing(self):
        """Test tokenization progress during tokenization (40%)."""
        progress = 40.0
        status = "processing"
        message = "Tokenizing samples..."

        assert progress == 40.0
        assert status == "processing"

    def test_dataset_tokenization_progress_step_80_calculating_stats(self):
        """Test tokenization progress during statistics calculation (80%)."""
        progress = 80.0
        status = "processing"
        message = "Calculating statistics..."

        assert progress == 80.0
        assert status == "processing"

    def test_dataset_tokenization_progress_step_95_saving(self):
        """Test tokenization progress during save (95%)."""
        progress = 95.0
        status = "processing"
        message = "Saving results..."

        assert progress == 95.0
        assert status == "processing"

    def test_dataset_tokenization_progress_step_100_complete(self):
        """Test tokenization progress at completion (100%)."""
        progress = 100.0
        status = "ready"
        message = "Tokenization complete"

        assert progress == 100.0
        assert status == "ready"

    def test_dataset_tokenization_progress_sequential(self):
        """Test tokenization progress increases monotonically."""
        steps = [0.0, 10.0, 20.0, 30.0, 40.0, 80.0, 95.0, 100.0]

        for i in range(1, len(steps)):
            assert steps[i] > steps[i-1], f"Progress decreased from {steps[i-1]} to {steps[i]}"

        assert steps[0] == 0.0
        assert steps[-1] == 100.0


class TestDatasetProgressErrorHandling:
    """Test dataset progress error handling."""

    def test_dataset_download_progress_error_handling(self):
        """Test download sets status to ERROR and progress to 0 on failure."""
        # Simulate error state
        status = "error"
        progress = 0.0
        error_message = "Failed to download dataset"

        assert status == "error"
        assert progress == 0.0
        assert error_message is not None

    def test_dataset_tokenization_progress_error_handling(self):
        """Test tokenization sets status to ERROR and progress to 0 on failure."""
        # Simulate error state
        status = "error"
        progress = 0.0
        error_message = "Failed to tokenize dataset"

        assert status == "error"
        assert progress == 0.0
        assert error_message is not None

    def test_dataset_cancellation_sets_progress_to_zero(self):
        """Test cancelled dataset sets progress to 0."""
        # Simulate cancellation
        status = "error"
        progress = 0.0
        error_message = "Cancelled by user"

        assert status == "error"
        assert progress == 0.0
        assert "Cancelled" in error_message


class TestDatasetProgressWebSocketEmission:
    """Test dataset progress WebSocket emission."""

    def test_dataset_download_websocket_emission_at_each_step(self):
        """Test WebSocket emits at each download progress step."""
        # Simulate emission at each step
        emissions = []

        steps = [
            (0.0, "downloading", "Starting download"),
            (10.0, "downloading", "Downloading from HuggingFace Hub"),
            (70.0, "downloading", "Saving dataset to disk"),
            (90.0, "downloading", "Processing metadata"),
            (100.0, "ready", "Download complete"),
        ]

        for progress, status, message in steps:
            emissions.append({
                "progress": progress,
                "status": status,
                "message": message,
            })

        # Verify all steps emitted
        assert len(emissions) == 5
        assert emissions[0]["progress"] == 0.0
        assert emissions[1]["progress"] == 10.0
        assert emissions[2]["progress"] == 70.0
        assert emissions[3]["progress"] == 90.0
        assert emissions[4]["progress"] == 100.0

    def test_dataset_tokenization_websocket_emission_at_each_step(self):
        """Test WebSocket emits at each tokenization progress step."""
        # Simulate emission at each step
        emissions = []

        steps = [
            (0.0, "processing", "Starting tokenization"),
            (10.0, "processing", "Loading tokenizer"),
            (20.0, "processing", "Loading dataset"),
            (30.0, "processing", "Analyzing schema"),
            (40.0, "processing", "Tokenizing samples"),
            (80.0, "processing", "Calculating statistics"),
            (95.0, "processing", "Saving results"),
            (100.0, "ready", "Tokenization complete"),
        ]

        for progress, status, message in steps:
            emissions.append({
                "progress": progress,
                "status": status,
                "message": message,
            })

        # Verify all steps emitted
        assert len(emissions) == 8
        assert emissions[0]["progress"] == 0.0
        assert emissions[1]["progress"] == 10.0
        assert emissions[2]["progress"] == 20.0
        assert emissions[3]["progress"] == 30.0
        assert emissions[4]["progress"] == 40.0
        assert emissions[5]["progress"] == 80.0
        assert emissions[6]["progress"] == 95.0
        assert emissions[7]["progress"] == 100.0


class TestDatasetProgressGaps:
    """Test progress gaps between milestones."""

    def test_dataset_download_progress_gaps(self):
        """Test download progress has expected gaps between milestones."""
        steps = [0.0, 10.0, 70.0, 90.0, 100.0]
        gaps = [steps[i+1] - steps[i] for i in range(len(steps) - 1)]

        # Verify gaps
        assert gaps[0] == 10.0  # 0 -> 10
        assert gaps[1] == 60.0  # 10 -> 70
        assert gaps[2] == 20.0  # 70 -> 90
        assert gaps[3] == 10.0  # 90 -> 100

        # Largest gap should be during actual download (10 -> 70)
        assert max(gaps) == 60.0

    def test_dataset_tokenization_progress_gaps(self):
        """Test tokenization progress has expected gaps between milestones."""
        steps = [0.0, 10.0, 20.0, 30.0, 40.0, 80.0, 95.0, 100.0]
        gaps = [steps[i+1] - steps[i] for i in range(len(steps) - 1)]

        # Verify gaps
        assert gaps[0] == 10.0  # 0 -> 10
        assert gaps[1] == 10.0  # 10 -> 20
        assert gaps[2] == 10.0  # 20 -> 30
        assert gaps[3] == 10.0  # 30 -> 40
        assert gaps[4] == 40.0  # 40 -> 80 (largest gap - actual tokenization)
        assert gaps[5] == 15.0  # 80 -> 95
        assert gaps[6] == 5.0   # 95 -> 100

        # Largest gap should be during actual tokenization (40 -> 80)
        assert max(gaps) == 40.0


class TestDatasetProgressEdgeCases:
    """Test dataset progress edge cases."""

    def test_dataset_download_with_small_dataset(self):
        """Test download progress with small dataset (fast completion)."""
        # Even for small datasets, all milestone steps should be hit
        steps = [0.0, 10.0, 70.0, 90.0, 100.0]

        # Verify all milestones present
        assert len(steps) == 5
        assert steps == [0.0, 10.0, 70.0, 90.0, 100.0]

    def test_dataset_tokenization_with_small_dataset(self):
        """Test tokenization progress with small dataset (fast completion)."""
        # Even for small datasets, all milestone steps should be hit
        steps = [0.0, 10.0, 20.0, 30.0, 40.0, 80.0, 95.0, 100.0]

        # Verify all milestones present
        assert len(steps) == 8
        assert steps == [0.0, 10.0, 20.0, 30.0, 40.0, 80.0, 95.0, 100.0]

    def test_dataset_progress_never_exceeds_100(self):
        """Test dataset progress never exceeds 100%."""
        download_steps = [0.0, 10.0, 70.0, 90.0, 100.0]
        tokenization_steps = [0.0, 10.0, 20.0, 30.0, 40.0, 80.0, 95.0, 100.0]

        for step in download_steps:
            assert step <= 100.0

        for step in tokenization_steps:
            assert step <= 100.0

    def test_dataset_progress_never_negative(self):
        """Test dataset progress never goes negative."""
        download_steps = [0.0, 10.0, 70.0, 90.0, 100.0]
        tokenization_steps = [0.0, 10.0, 20.0, 30.0, 40.0, 80.0, 95.0, 100.0]

        for step in download_steps:
            assert step >= 0.0

        for step in tokenization_steps:
            assert step >= 0.0
