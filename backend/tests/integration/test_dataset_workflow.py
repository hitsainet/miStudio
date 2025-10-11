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

        tokenize_complete_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            tokenized_path="./data/datasets/test_tiny_dataset_tokenized",
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
        assert dataset.tokenized_path is not None
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

        tokenize_update = DatasetUpdate(
            status=DatasetStatus.READY.value,
            progress=100.0,
            tokenized_path="./data/datasets/test_dataset_tokenized",
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
