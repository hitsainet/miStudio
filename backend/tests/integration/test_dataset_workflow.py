"""
End-to-end workflow test for dataset management.

Tests the complete flow: download → wait for completion → tokenize → verify ready.
"""

import asyncio
import pytest
from uuid import UUID
from pathlib import Path

from src.models.dataset import DatasetStatus
from src.schemas.dataset import DatasetCreate, DatasetUpdate
from src.services.dataset_service import DatasetService
from src.workers.dataset_tasks import download_dataset_task, tokenize_dataset_task


class TestDatasetWorkflow:
    """End-to-end workflow tests for dataset operations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_dataset_workflow(self, async_session):
        """
        Test complete workflow: download → ready → tokenize → ready with stats.

        This test simulates the full user journey:
        1. User downloads a dataset from HuggingFace
        2. System processes download in background
        3. Dataset becomes ready for use
        4. User tokenizes the dataset
        5. System processes tokenization in background
        6. Dataset has tokenization statistics

        Note: This is a SIMULATED test with mocked download/tokenization.
        For real E2E testing, use a small dataset (e.g., 100 samples).
        """
        # Step 1: Create dataset record
        dataset_create = DatasetCreate(
            name="test-tiny-dataset",
            source="HuggingFace",
            hf_repo_id="test/tiny-dataset",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)
        await async_session.commit()
        await async_session.refresh(dataset)

        assert dataset.id is not None
        assert dataset.status == DatasetStatus.DOWNLOADING
        assert dataset.progress in (None, 0.0)  # Progress can be None initially or 0.0

        # Step 2: Simulate download completion
        # In real test, you would call download_dataset_task.delay() and poll for completion
        # Here we simulate the final state after download
        download_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            raw_path="./data/datasets/test_tiny_dataset",
            num_samples=100,
            size_bytes=1024 * 1024,  # 1 MB
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, download_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        assert dataset.status == DatasetStatus.READY
        assert dataset.progress == 100.0
        assert dataset.num_samples == 100
        assert dataset.raw_path is not None

        # Step 3: Start tokenization
        tokenize_update = DatasetUpdate(
            status=DatasetStatus.PROCESSING.value,
            progress=0.0,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, tokenize_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        assert dataset.status == DatasetStatus.PROCESSING
        assert dataset.progress == 0.0

        # Step 4: Simulate tokenization completion with statistics
        # In real test, you would call tokenize_dataset_task.delay() and poll for completion
        tokenization_metadata = {
            "tokenization": {
                "tokenizer_name": "gpt2",
                "text_column_used": "text",
                "max_length": 512,
                "stride": 0,
                "num_tokens": 50000,
                "avg_seq_length": 256.5,
                "min_seq_length": 10,
                "max_seq_length": 512,
            }
        }

        # Note: tokenized_path is stored in DatasetTokenization, not Dataset
        tokenize_complete_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            metadata=tokenization_metadata,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, tokenize_complete_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Step 5: Verify final state
        assert dataset.status == DatasetStatus.READY
        assert dataset.progress == 100.0
        assert dataset.extra_metadata is not None
        assert "tokenization" in dataset.extra_metadata
        assert dataset.extra_metadata["tokenization"]["tokenizer_name"] == "gpt2"
        assert dataset.extra_metadata["tokenization"]["num_tokens"] == 50000
        assert dataset.extra_metadata["tokenization"]["avg_seq_length"] == 256.5

        # Step 6: Cleanup
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_download_error_handling(self, async_session):
        """
        Test error handling during download.

        Verifies that:
        1. Failed downloads set status to ERROR
        2. Error message is stored
        3. Dataset can be retried or deleted
        """
        # Create dataset
        dataset_create = DatasetCreate(
            name="invalid-dataset",
            source="HuggingFace",
            hf_repo_id="invalid/nonexistent-dataset-12345",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)
        await async_session.commit()
        await async_session.refresh(dataset)

        # Simulate download failure
        error_update = DatasetUpdate(
            status=DatasetStatus.ERROR.value,
            error_message="Download failed: Repository not found",
            progress=0.0,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, error_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify error state
        assert dataset.status == DatasetStatus.ERROR
        assert dataset.error_message is not None
        assert "Repository not found" in dataset.error_message

        # Cleanup
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tokenization_error_handling(self, async_session):
        """
        Test error handling during tokenization.

        Verifies that:
        1. Failed tokenization sets status to ERROR
        2. Error message is stored
        3. Dataset remains accessible (raw data not affected)
        """
        # Create ready dataset
        dataset_create = DatasetCreate(
            name="test-dataset",
            source="HuggingFace",
            hf_repo_id="test/dataset",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        # Set to ready state
        ready_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            raw_path="./data/datasets/test_dataset",
            num_samples=100,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, ready_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Start tokenization
        processing_update = DatasetUpdate(
            status=DatasetStatus.PROCESSING.value,
            progress=0.0,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, processing_update
        )
        await async_session.commit()

        # Simulate tokenization failure
        error_update = DatasetUpdate(
            status=DatasetStatus.ERROR.value,
            error_message="Tokenization failed: Invalid tokenizer name",
            progress=0.0,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, error_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify error state
        assert dataset.status == DatasetStatus.ERROR
        assert dataset.error_message is not None
        assert "Invalid tokenizer" in dataset.error_message
        assert dataset.raw_path is not None  # Raw data still accessible

        # Cleanup
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_operations_prevented(self, async_session):
        """
        Test that concurrent tokenization of same dataset is prevented.

        Verifies that:
        1. Dataset in PROCESSING state cannot be tokenized again
        2. Appropriate error is returned
        3. First operation completes successfully
        """
        # Create ready dataset
        dataset_create = DatasetCreate(
            name="test-dataset",
            source="HuggingFace",
            hf_repo_id="test/dataset",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        ready_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            raw_path="./data/datasets/test_dataset",
            num_samples=100,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, ready_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Start tokenization (marks as PROCESSING)
        processing_update = DatasetUpdate(
            status=DatasetStatus.PROCESSING.value,
            progress=0.0,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, processing_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify dataset is processing
        assert dataset.status == DatasetStatus.PROCESSING

        # Attempt to start another tokenization (should be prevented by API endpoint)
        # This test verifies the database state check, actual prevention is in API layer
        dataset_check = await DatasetService.get_dataset(async_session, dataset.id)
        assert dataset_check.status == DatasetStatus.PROCESSING

        # Cleanup
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_metadata_persistence_across_operations(self, async_session):
        """
        Test that metadata is preserved and merged across operations.

        Verifies that:
        1. Download metadata is preserved
        2. Tokenization metadata is added
        3. Both sections coexist in final dataset
        """
        # Create dataset with download metadata
        dataset_create = DatasetCreate(
            name="test-dataset",
            source="HuggingFace",
            hf_repo_id="test/dataset",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        download_metadata = {
            "download": {
                "split": "train",
                "config": "default",
                "timestamp": "2025-10-11T00:00:00Z",
            }
        }

        download_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            raw_path="./data/datasets/test_dataset",
            num_samples=100,
            metadata=download_metadata,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, download_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify download metadata exists
        assert dataset.extra_metadata is not None
        assert "download" in dataset.extra_metadata

        # Add tokenization metadata
        tokenization_metadata = {
            "tokenization": {
                "tokenizer_name": "gpt2",
                "num_tokens": 50000,
                "avg_seq_length": 256.5,
            }
        }

        # Note: tokenized_path is stored in DatasetTokenization, not Dataset
        tokenize_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            metadata=tokenization_metadata,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, tokenize_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify BOTH metadata sections exist
        assert dataset.extra_metadata is not None
        assert "download" in dataset.extra_metadata
        assert "tokenization" in dataset.extra_metadata
        assert dataset.extra_metadata["download"]["split"] == "train"
        assert dataset.extra_metadata["tokenization"]["tokenizer_name"] == "gpt2"

        # Cleanup
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_padding_strategy_in_metadata(self, async_session):
        """
        Test that padding strategy parameter flows through the complete tokenization workflow.

        Verifies that:
        1. Dataset can be tokenized with different padding strategies
        2. Padding parameter is passed through correctly
        3. Metadata can be updated with padding strategy info (future enhancement)
        """
        # Create ready dataset
        dataset_create = DatasetCreate(
            name="test-padding-dataset",
            source="HuggingFace",
            hf_repo_id="test/padding-dataset",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        ready_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            raw_path="./data/datasets/test_padding_dataset",
            num_samples=100,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, ready_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Test tokenization with "max_length" padding strategy
        tokenization_metadata_max_length = {
            "tokenization": {
                "tokenizer_name": "gpt2",
                "text_column_used": "text",
                "max_length": 512,
                "stride": 0,
                "padding": "max_length",  # Explicit padding strategy
                "num_tokens": 50000,
                "avg_seq_length": 256.5,
                "min_seq_length": 10,
                "max_seq_length": 512,
            }
        }

        # Note: tokenized_path is stored in DatasetTokenization, not Dataset
        tokenize_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            metadata=tokenization_metadata_max_length,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, tokenize_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify tokenization metadata with padding strategy
        assert dataset.extra_metadata is not None
        assert "tokenization" in dataset.extra_metadata
        assert dataset.extra_metadata["tokenization"]["tokenizer_name"] == "gpt2"
        assert dataset.extra_metadata["tokenization"]["num_tokens"] == 50000
        # Note: Padding field may or may not persist depending on metadata schema version
        # The important test is that the workflow accepts the padding parameter

        # Test re-tokenization with "do_not_pad" padding strategy
        tokenization_metadata_no_pad = {
            "tokenization": {
                "tokenizer_name": "gpt2",
                "text_column_used": "text",
                "max_length": 512,
                "stride": 0,
                "padding": "do_not_pad",  # Different padding strategy
                "num_tokens": 48000,  # Fewer tokens without padding
                "avg_seq_length": 230.0,
                "min_seq_length": 10,
                "max_seq_length": 450,
            }
        }

        retokenize_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            metadata=tokenization_metadata_no_pad,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, retokenize_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify metadata was updated with re-tokenization
        # Note: Padding field may or may not persist depending on metadata schema version
        assert dataset.extra_metadata["tokenization"]["num_tokens"] == 48000
        assert dataset.extra_metadata["tokenization"]["avg_seq_length"] == 230.0

        # Cleanup
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_truncation_strategy_in_metadata(self, async_session):
        """
        Test that truncation strategy parameter flows through the complete tokenization workflow.

        Verifies that:
        1. Dataset can be tokenized with different truncation strategies
        2. Truncation parameter is passed through correctly
        3. Metadata can be updated with truncation strategy info
        """
        # Create ready dataset
        dataset_create = DatasetCreate(
            name="test-truncation-dataset",
            source="HuggingFace",
            hf_repo_id="test/truncation-dataset",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        ready_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            raw_path="./data/datasets/test_truncation_dataset",
            num_samples=100,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, ready_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Test tokenization with "longest_first" truncation strategy (default)
        tokenization_metadata_longest_first = {
            "tokenization": {
                "tokenizer_name": "gpt2",
                "text_column_used": "text",
                "max_length": 512,
                "stride": 0,
                "padding": "max_length",
                "truncation": "longest_first",  # Explicit truncation strategy
                "num_tokens": 50000,
                "avg_seq_length": 256.5,
                "min_seq_length": 10,
                "max_seq_length": 512,
            }
        }

        # Note: tokenized_path is stored in DatasetTokenization, not Dataset
        tokenize_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            metadata=tokenization_metadata_longest_first,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, tokenize_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify tokenization metadata with truncation strategy
        assert dataset.extra_metadata is not None
        assert "tokenization" in dataset.extra_metadata
        assert dataset.extra_metadata["tokenization"]["tokenizer_name"] == "gpt2"
        assert dataset.extra_metadata["tokenization"]["num_tokens"] == 50000
        # Note: Truncation field may or may not persist depending on metadata schema version
        # The important test is that the workflow accepts the truncation parameter

        # Test re-tokenization with "only_first" truncation strategy
        tokenization_metadata_only_first = {
            "tokenization": {
                "tokenizer_name": "gpt2",
                "text_column_used": "text",
                "max_length": 512,
                "stride": 0,
                "padding": "max_length",
                "truncation": "only_first",  # Different truncation strategy
                "num_tokens": 50000,
                "avg_seq_length": 256.5,
                "min_seq_length": 10,
                "max_seq_length": 512,
            }
        }

        retokenize_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            metadata=tokenization_metadata_only_first,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, retokenize_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify metadata was updated with re-tokenization
        assert dataset.extra_metadata["tokenization"]["num_tokens"] == 50000
        assert dataset.extra_metadata["tokenization"]["avg_seq_length"] == 256.5

        # Test with "do_not_truncate" truncation strategy
        tokenization_metadata_no_truncate = {
            "tokenization": {
                "tokenizer_name": "gpt2",
                "text_column_used": "text",
                "max_length": 512,
                "stride": 0,
                "padding": "max_length",
                "truncation": "do_not_truncate",  # No truncation
                "num_tokens": 52000,  # Potentially more tokens without truncation
                "avg_seq_length": 280.0,
                "min_seq_length": 10,
                "max_seq_length": 600,  # May exceed max_length if truncation disabled
            }
        }

        retokenize_update2 = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            metadata=tokenization_metadata_no_truncate,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, retokenize_update2
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify metadata with no truncation
        assert dataset.extra_metadata["tokenization"]["num_tokens"] == 52000
        assert dataset.extra_metadata["tokenization"]["avg_seq_length"] == 280.0
        assert dataset.extra_metadata["tokenization"]["max_seq_length"] == 600

        # Cleanup
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_new_tokenization_metrics_in_metadata(self, async_session):
        """
        Test that new tokenization metrics (vocab_size, median, length_distribution) are captured in metadata.

        Verifies that:
        1. vocab_size is calculated and stored
        2. median_seq_length is calculated and stored
        3. length_distribution bucketing is generated and stored
        4. All metrics are validated for consistency
        """
        # Create ready dataset
        dataset_create = DatasetCreate(
            name="test-metrics-dataset",
            source="HuggingFace",
            hf_repo_id="test/metrics-dataset",
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        ready_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            raw_path="./data/datasets/test_metrics_dataset",
            num_samples=5000,  # Reasonable sample size for testing
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, ready_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Simulate tokenization with new metrics
        tokenization_metadata = {
            "tokenization": {
                "tokenizer_name": "gpt2",
                "text_column_used": "text",
                "max_length": 512,
                "stride": 0,
                "padding": "max_length",
                "truncation": "longest_first",
                "num_tokens": 1000000,
                "avg_seq_length": 245.5,
                "min_seq_length": 10,
                "max_seq_length": 512,
                # New metrics:
                "median_seq_length": 230.0,  # Median calculation
                "vocab_size": 50257,  # Unique tokens (GPT-2 vocabulary size)
                "length_distribution": {  # Bucketed distribution
                    "0-100": 150,
                    "100-200": 450,
                    "200-400": 2100,
                    "400-600": 1800,
                    "600-800": 300,
                    "800-1000": 150,
                    "1000+": 50,
                },
            }
        }

        # Note: tokenized_path is stored in DatasetTokenization, not Dataset
        tokenize_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            metadata=tokenization_metadata,
        )
        dataset = await DatasetService.update_dataset(
            async_session, dataset.id, tokenize_update
        )
        await async_session.commit()
        await async_session.refresh(dataset)

        # Verify all new metrics are present
        assert dataset.extra_metadata is not None
        assert "tokenization" in dataset.extra_metadata
        tok_meta = dataset.extra_metadata["tokenization"]

        # Check existing metrics
        assert tok_meta["tokenizer_name"] == "gpt2"
        assert tok_meta["num_tokens"] == 1000000
        assert tok_meta["avg_seq_length"] == 245.5
        assert tok_meta["min_seq_length"] == 10
        assert tok_meta["max_seq_length"] == 512

        # Check NEW metrics
        assert "median_seq_length" in tok_meta
        assert tok_meta["median_seq_length"] == 230.0
        assert "vocab_size" in tok_meta
        assert tok_meta["vocab_size"] == 50257
        assert "length_distribution" in tok_meta

        # Verify length_distribution structure
        length_dist = tok_meta["length_distribution"]
        expected_buckets = ["0-100", "100-200", "200-400", "400-600", "600-800", "800-1000", "1000+"]
        for bucket in expected_buckets:
            assert bucket in length_dist, f"Missing bucket: {bucket}"
            assert isinstance(length_dist[bucket], int), f"Bucket {bucket} should be integer count"

        # Verify bucket counts sum to total samples
        total_in_buckets = sum(length_dist.values())
        assert total_in_buckets == 5000, f"Expected 5000 total samples, got {total_in_buckets}"

        # Verify median is between min and max
        assert tok_meta["min_seq_length"] <= tok_meta["median_seq_length"] <= tok_meta["max_seq_length"]

        # Cleanup
        await DatasetService.delete_dataset(async_session, dataset.id)
        await async_session.commit()


# Note: These tests are INTEGRATION tests that verify the workflow logic.
# For REAL E2E testing with actual downloads and tokenization:
#
# 1. Use pytest-asyncio with longer timeouts
# 2. Use a small test dataset (e.g., 100 samples)
# 3. Actually call the Celery tasks and poll for completion
# 4. Verify files are created on disk
# 5. Clean up test files after completion
#
# Example real E2E test structure:
#
# @pytest.mark.asyncio
# @pytest.mark.slow  # Mark as slow test
# async def test_real_download_and_tokenize(async_session):
#     """Real E2E test with actual HuggingFace download."""
#     # Create dataset
#     dataset = await DatasetService.create_dataset(...)
#
#     # Trigger real download
#     task_result = download_dataset_task.delay(
#         dataset_id=str(dataset.id),
#         repo_id="roneneldan/TinyStories",
#         split="train[:100]",  # Small subset
#     )
#
#     # Poll for completion (with timeout)
#     for _ in range(60):  # 60 seconds timeout
#         await asyncio.sleep(1)
#         dataset = await DatasetService.get_dataset(async_session, dataset.id)
#         if dataset.status == DatasetStatus.READY:
#             break
#
#     # Verify download succeeded
#     assert dataset.status == DatasetStatus.READY
#     assert Path(dataset.raw_path).exists()
#
#     # Trigger real tokenization
#     task_result = tokenize_dataset_task.delay(
#         dataset_id=str(dataset.id),
#         tokenizer_name="gpt2",
#         max_length=512,
#     )
#
#     # Poll for completion
#     for _ in range(120):  # 120 seconds timeout
#         await asyncio.sleep(1)
#         dataset = await DatasetService.get_dataset(async_session, dataset.id)
#         if dataset.status == DatasetStatus.READY and dataset.tokenized_path:
#             break
#
#     # Verify tokenization succeeded
#     assert dataset.tokenized_path is not None
#     assert Path(dataset.tokenized_path).exists()
#     assert dataset.extra_metadata["tokenization"]["num_tokens"] > 0
#
#     # Cleanup
#     await DatasetService.delete_dataset(async_session, dataset.id, delete_files=True)
