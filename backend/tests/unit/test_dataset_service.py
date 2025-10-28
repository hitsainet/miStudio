"""
Unit tests for DatasetService.

Tests dataset management operations including CRUD, progress tracking,
metadata merging, and status updates.
"""

import pytest
import pytest_asyncio
from uuid import UUID, uuid4
from datetime import datetime, UTC

from src.services.dataset_service import DatasetService, deep_merge_metadata
from src.models.dataset import Dataset, DatasetStatus
from src.schemas.dataset import DatasetCreate, DatasetUpdate


@pytest.mark.asyncio
class TestDeepMergeMetadata:
    """Test deep_merge_metadata helper function."""

    async def test_merge_with_empty_existing(self):
        """Test merging into empty existing metadata."""
        existing = {}
        new = {"key1": "value1", "key2": 42}

        result = deep_merge_metadata(existing, new)

        assert result == {"key1": "value1", "key2": 42}

    async def test_merge_with_none_existing(self):
        """Test merging when existing is None."""
        existing = None
        new = {"key1": "value1", "key2": None}

        result = deep_merge_metadata(existing, new)

        # None values should be filtered out
        assert result == {"key1": "value1"}

    async def test_merge_with_empty_new(self):
        """Test merging empty new metadata."""
        existing = {"key1": "value1", "key2": 42}
        new = {}

        result = deep_merge_metadata(existing, new)

        assert result == {"key1": "value1", "key2": 42}

    async def test_none_values_dont_overwrite(self):
        """Test that None values in new dict don't overwrite existing values."""
        existing = {"key1": "existing", "key2": 42}
        new = {"key1": None, "key3": "new"}

        result = deep_merge_metadata(existing, new)

        assert result["key1"] == "existing"  # Not overwritten by None
        assert result["key2"] == 42
        assert result["key3"] == "new"

    async def test_non_none_values_overwrite(self):
        """Test that non-None values do overwrite existing values."""
        existing = {"key1": "old", "key2": 42}
        new = {"key1": "new", "key2": 99}

        result = deep_merge_metadata(existing, new)

        assert result["key1"] == "new"
        assert result["key2"] == 99

    async def test_nested_dict_merge(self):
        """Test recursive merging of nested dictionaries."""
        existing = {
            "top": {
                "nested1": "value1",
                "nested2": "value2"
            },
            "other": "data"
        }
        new = {
            "top": {
                "nested2": "updated",
                "nested3": "new"
            }
        }

        result = deep_merge_metadata(existing, new)

        assert result["top"]["nested1"] == "value1"  # Preserved
        assert result["top"]["nested2"] == "updated"  # Updated
        assert result["top"]["nested3"] == "new"  # Added
        assert result["other"] == "data"  # Preserved

    async def test_list_replacement_not_merge(self):
        """Test that lists are replaced, not merged."""
        existing = {"key": [1, 2, 3]}
        new = {"key": [4, 5]}

        result = deep_merge_metadata(existing, new)

        assert result["key"] == [4, 5]  # Replaced, not merged


@pytest.mark.asyncio
class TestDatasetServiceCreate:
    """Test DatasetService.create_dataset()."""

    async def test_create_dataset_with_hf_repo(self, async_session):
        """Test creating a dataset from HuggingFace."""
        dataset_create = DatasetCreate(
            name="Test Dataset",
            source="HuggingFace",
            hf_repo_id="test/dataset",
            metadata={"split": "train"}
        )

        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        assert dataset is not None
        assert isinstance(dataset.id, UUID)
        assert dataset.name == "Test Dataset"
        assert dataset.source == "HuggingFace"
        assert dataset.hf_repo_id == "test/dataset"
        assert dataset.status == DatasetStatus.DOWNLOADING
        assert dataset.extra_metadata == {"split": "train"}

    async def test_create_dataset_local(self, async_session):
        """Test creating a local dataset (no HF repo)."""
        dataset_create = DatasetCreate(
            name="Local Dataset",
            source="Local",
            raw_path="/data/local-dataset"
        )

        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        assert dataset.status == DatasetStatus.PROCESSING  # Not DOWNLOADING
        assert dataset.hf_repo_id is None
        assert dataset.raw_path == "/data/local-dataset"

    async def test_create_dataset_with_empty_metadata(self, async_session):
        """Test creating dataset with None metadata."""
        dataset_create = DatasetCreate(
            name="No Metadata",
            source="HuggingFace",
            hf_repo_id="test/no-meta"
        )

        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        assert dataset.extra_metadata == {}


@pytest.mark.asyncio
class TestDatasetServiceGet:
    """Test DatasetService.get_dataset() and get_dataset_by_repo_id()."""

    async def test_get_dataset_by_id_success(self, async_session):
        """Test getting a dataset by ID."""
        # Create dataset first
        dataset_create = DatasetCreate(
            name="Test",
            source="HuggingFace",
            hf_repo_id="test/dataset"
        )
        created = await DatasetService.create_dataset(async_session, dataset_create)

        # Get by ID
        fetched = await DatasetService.get_dataset(async_session, created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "Test"

    async def test_get_dataset_not_found(self, async_session):
        """Test getting non-existent dataset returns None."""
        random_uuid = uuid4()
        dataset = await DatasetService.get_dataset(async_session, random_uuid)

        assert dataset is None

    async def test_get_dataset_by_repo_id_success(self, async_session):
        """Test getting dataset by HuggingFace repo ID."""
        dataset_create = DatasetCreate(
            name="Test",
            source="HuggingFace",
            hf_repo_id="test/unique-repo"
        )
        created = await DatasetService.create_dataset(async_session, dataset_create)

        # Get by repo ID
        fetched = await DatasetService.get_dataset_by_repo_id(
            async_session,
            "test/unique-repo"
        )

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.hf_repo_id == "test/unique-repo"

    async def test_get_dataset_by_repo_id_not_found(self, async_session):
        """Test getting dataset by non-existent repo ID returns None."""
        dataset = await DatasetService.get_dataset_by_repo_id(
            async_session,
            "nonexistent/repo"
        )

        assert dataset is None


@pytest.mark.asyncio
class TestDatasetServiceList:
    """Test DatasetService.list_datasets()."""

    async def test_list_all_datasets(self, async_session):
        """Test listing all datasets."""
        # Create multiple datasets
        for i in range(3):
            dataset_create = DatasetCreate(
                name=f"Dataset {i}",
                source="HuggingFace",
                hf_repo_id=f"test/dataset-{i}"
            )
            await DatasetService.create_dataset(async_session, dataset_create)

        # List all
        datasets, total = await DatasetService.list_datasets(async_session)

        assert len(datasets) == 3
        assert total == 3

    async def test_list_datasets_filter_by_status(self, async_session):
        """Test filtering datasets by status."""
        # Create datasets with different statuses
        ds1_create = DatasetCreate(
            name="Downloading",
            source="HuggingFace",
            hf_repo_id="test/ds1"
        )
        ds1 = await DatasetService.create_dataset(async_session, ds1_create)

        ds2_create = DatasetCreate(
            name="Local",
            source="Local",
            raw_path="/data/ds2"
        )
        ds2 = await DatasetService.create_dataset(async_session, ds2_create)

        # Filter by DOWNLOADING
        downloading, total = await DatasetService.list_datasets(
            async_session,
            status=DatasetStatus.DOWNLOADING
        )

        assert len(downloading) == 1
        assert total == 1
        assert downloading[0].id == ds1.id

    async def test_list_datasets_filter_by_source(self, async_session):
        """Test filtering datasets by source."""
        # Create datasets from different sources
        for source in ["HuggingFace", "Local", "HuggingFace"]:
            dataset_create = DatasetCreate(
                name=f"{source} Dataset",
                source=source,
                hf_repo_id=f"test/{source}" if source == "HuggingFace" else None
            )
            await DatasetService.create_dataset(async_session, dataset_create)

        # Filter by HuggingFace
        hf_datasets, total = await DatasetService.list_datasets(
            async_session,
            source="HuggingFace"
        )

        assert len(hf_datasets) == 2
        assert total == 2

    async def test_list_datasets_search(self, async_session):
        """Test searching datasets by name or repo_id."""
        # Create datasets with different names
        names = ["llama-dataset", "gpt-dataset", "llama-v2"]
        for name in names:
            dataset_create = DatasetCreate(
                name=name,
                source="HuggingFace",
                hf_repo_id=f"test/{name}"
            )
            await DatasetService.create_dataset(async_session, dataset_create)

        # Search for "llama"
        llama_datasets, total = await DatasetService.list_datasets(
            async_session,
            search="llama"
        )

        assert len(llama_datasets) == 2
        assert total == 2
        assert all("llama" in ds.name for ds in llama_datasets)

    async def test_list_datasets_pagination(self, async_session):
        """Test pagination of dataset list."""
        # Create 5 datasets
        for i in range(5):
            dataset_create = DatasetCreate(
                name=f"Dataset {i}",
                source="HuggingFace",
                hf_repo_id=f"test/ds-{i}"
            )
            await DatasetService.create_dataset(async_session, dataset_create)

        # Get first page (2 items)
        page1, total = await DatasetService.list_datasets(
            async_session,
            skip=0,
            limit=2
        )

        assert len(page1) == 2
        assert total == 5

        # Get second page (2 items)
        page2, total = await DatasetService.list_datasets(
            async_session,
            skip=2,
            limit=2
        )

        assert len(page2) == 2
        assert total == 5

        # Verify different datasets
        page1_ids = {ds.id for ds in page1}
        page2_ids = {ds.id for ds in page2}
        assert page1_ids.isdisjoint(page2_ids)

    async def test_list_datasets_sorting(self, async_session):
        """Test sorting datasets by created_at."""
        # Create datasets in sequence
        dataset_ids = []
        for i in range(3):
            dataset_create = DatasetCreate(
                name=f"Dataset {i}",
                source="HuggingFace",
                hf_repo_id=f"test/ds-{i}"
            )
            ds = await DatasetService.create_dataset(async_session, dataset_create)
            dataset_ids.append(ds.id)

        # List with default sorting (desc - newest first)
        datasets_desc, _ = await DatasetService.list_datasets(async_session)
        desc_ids = [ds.id for ds in datasets_desc]

        # Should be in reverse order (newest first)
        assert desc_ids == list(reversed(dataset_ids))

        # List with ascending order (oldest first)
        datasets_asc, _ = await DatasetService.list_datasets(
            async_session,
            sort_by="created_at",
            order="asc"
        )
        asc_ids = [ds.id for ds in datasets_asc]

        # Should be in original order
        assert asc_ids == dataset_ids


@pytest.mark.asyncio
class TestDatasetServiceUpdate:
    """Test DatasetService.update_dataset()."""

    async def test_update_dataset_simple_fields(self, async_session):
        """Test updating simple fields."""
        # Create dataset
        dataset_create = DatasetCreate(
            name="Original",
            source="HuggingFace",
            hf_repo_id="test/dataset"
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        # Update
        updates = DatasetUpdate(
            num_samples=1000,
            num_tokens=500000
        )
        updated = await DatasetService.update_dataset(async_session, dataset.id, updates)

        assert updated is not None
        assert updated.num_samples == 1000
        assert updated.num_tokens == 500000
        assert updated.name == "Original"  # Unchanged

    async def test_update_dataset_status(self, async_session):
        """Test updating dataset status."""
        dataset_create = DatasetCreate(
            name="Test",
            source="HuggingFace",
            hf_repo_id="test/dataset"
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        # Update status
        updates = DatasetUpdate(status="READY")
        updated = await DatasetService.update_dataset(async_session, dataset.id, updates)

        assert updated.status == DatasetStatus.READY

    async def test_update_dataset_metadata_merge(self, async_session):
        """Test metadata deep merge on update."""
        # Create with initial metadata
        dataset_create = DatasetCreate(
            name="Test",
            source="HuggingFace",
            hf_repo_id="test/dataset",
            metadata={"field1": "value1", "field2": "value2"}
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        # Update metadata (should merge, not replace)
        updates = DatasetUpdate(
            metadata={"field2": "updated", "field3": "new"}
        )
        updated = await DatasetService.update_dataset(async_session, dataset.id, updates)

        # Check merged result
        assert updated.extra_metadata["field1"] == "value1"  # Preserved
        assert updated.extra_metadata["field2"] == "updated"  # Updated
        assert updated.extra_metadata["field3"] == "new"  # Added

    async def test_update_dataset_metadata_none_skipped(self, async_session):
        """Test that None metadata doesn't overwrite existing."""
        dataset_create = DatasetCreate(
            name="Test",
            source="HuggingFace",
            hf_repo_id="test/dataset",
            metadata={"field1": "value1"}
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        # Update with None metadata (should be skipped)
        updates = DatasetUpdate(metadata=None)
        updated = await DatasetService.update_dataset(async_session, dataset.id, updates)

        # Metadata should be unchanged
        assert updated.extra_metadata == {"field1": "value1"}

    async def test_update_dataset_not_found(self, async_session):
        """Test updating non-existent dataset returns None."""
        random_uuid = uuid4()
        updates = DatasetUpdate(num_samples=1000)
        result = await DatasetService.update_dataset(async_session, random_uuid, updates)

        assert result is None


@pytest.mark.asyncio
class TestDatasetServiceProgressTracking:
    """Test DatasetService.update_dataset_progress()."""

    async def test_update_progress_only(self, async_session):
        """Test updating progress without status change."""
        dataset_create = DatasetCreate(
            name="Test",
            source="HuggingFace",
            hf_repo_id="test/dataset"
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        # Update progress
        updated = await DatasetService.update_dataset_progress(
            async_session,
            dataset.id,
            progress=45.5
        )

        assert updated is not None
        assert updated.progress == 45.5
        assert updated.status == DatasetStatus.DOWNLOADING  # Unchanged

    async def test_update_progress_with_status_change(self, async_session):
        """Test updating progress with status change."""
        dataset_create = DatasetCreate(
            name="Test",
            source="HuggingFace",
            hf_repo_id="test/dataset"
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        # Update progress and status
        updated = await DatasetService.update_dataset_progress(
            async_session,
            dataset.id,
            progress=100.0,
            status=DatasetStatus.READY
        )

        assert updated.progress == 100.0
        assert updated.status == DatasetStatus.READY

    async def test_update_progress_not_found(self, async_session):
        """Test updating progress for non-existent dataset returns None."""
        random_uuid = uuid4()
        result = await DatasetService.update_dataset_progress(
            async_session,
            random_uuid,
            progress=50.0
        )

        assert result is None


@pytest.mark.asyncio
class TestDatasetServiceMarkError:
    """Test DatasetService.mark_dataset_error()."""

    async def test_mark_error_success(self, async_session):
        """Test marking dataset as error."""
        dataset_create = DatasetCreate(
            name="Test",
            source="HuggingFace",
            hf_repo_id="test/dataset"
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        # Mark as error
        errored = await DatasetService.mark_dataset_error(
            async_session,
            dataset.id,
            error_message="Download failed: Network error"
        )

        assert errored is not None
        assert errored.status == DatasetStatus.ERROR
        assert errored.error_message == "Download failed: Network error"

    async def test_mark_error_not_found(self, async_session):
        """Test marking non-existent dataset as error returns None."""
        random_uuid = uuid4()
        result = await DatasetService.mark_dataset_error(
            async_session,
            random_uuid,
            error_message="Some error"
        )

        assert result is None


@pytest.mark.asyncio
class TestDatasetServiceDelete:
    """Test DatasetService.delete_dataset()."""

    async def test_delete_dataset_success(self, async_session):
        """Test deleting a dataset."""
        dataset_create = DatasetCreate(
            name="Test",
            source="HuggingFace",
            hf_repo_id="test/dataset",
            raw_path="/data/raw",
            tokenized_path="/data/tokenized"
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        # Delete
        result = await DatasetService.delete_dataset(async_session, dataset.id)

        assert result is not None
        assert result["deleted"] is True
        assert result["dataset_id"] == str(dataset.id)
        assert result["raw_path"] == "/data/raw"
        assert result["tokenized_path"] == "/data/tokenized"

        # Verify deletion
        fetched = await DatasetService.get_dataset(async_session, dataset.id)
        assert fetched is None

    async def test_delete_dataset_without_paths(self, async_session):
        """Test deleting dataset without file paths."""
        dataset_create = DatasetCreate(
            name="Test",
            source="HuggingFace",
            hf_repo_id="test/dataset"
        )
        dataset = await DatasetService.create_dataset(async_session, dataset_create)

        # Delete
        result = await DatasetService.delete_dataset(async_session, dataset.id)

        assert result["raw_path"] is None
        assert result["tokenized_path"] is None

    async def test_delete_dataset_not_found(self, async_session):
        """Test deleting non-existent dataset returns None."""
        random_uuid = uuid4()
        result = await DatasetService.delete_dataset(async_session, random_uuid)

        assert result is None
