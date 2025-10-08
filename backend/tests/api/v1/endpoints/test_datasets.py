"""
Tests for dataset API endpoints.

This module contains unit tests for all dataset-related API endpoints,
including create, read, update, delete, and list operations.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.dataset import Dataset, DatasetStatus
from src.schemas.dataset import DatasetCreate


pytestmark = pytest.mark.asyncio


class TestCreateDataset:
    """Tests for POST /api/v1/datasets endpoint."""

    async def test_create_dataset_success(self, client: AsyncClient, async_session: AsyncSession):
        """Test successful dataset creation."""
        payload = {
            "name": "Test Dataset",
            "source": "HuggingFace",
            "hf_repo_id": "test/dataset",
            "metadata": {"test": "value"}
        }

        response = await client.post("/api/v1/datasets", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == payload["name"]
        assert data["source"] == payload["source"]
        assert data["hf_repo_id"] == payload["hf_repo_id"]
        assert data["status"] == "downloading"
        assert "id" in data
        assert "created_at" in data

    async def test_create_dataset_local_source(self, client: AsyncClient):
        """Test creating dataset with local source."""
        payload = {
            "name": "Local Dataset",
            "source": "Local",
            "raw_path": "/data/datasets/local/test",
            "metadata": {}
        }

        response = await client.post("/api/v1/datasets", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "processing"
        assert data["raw_path"] == payload["raw_path"]

    async def test_create_dataset_missing_name(self, client: AsyncClient):
        """Test dataset creation fails without name."""
        payload = {
            "source": "HuggingFace"
        }

        response = await client.post("/api/v1/datasets", json=payload)

        assert response.status_code == 422  # Validation error

    async def test_create_dataset_invalid_source(self, client: AsyncClient):
        """Test dataset creation with invalid source."""
        payload = {
            "name": "Test",
            "source": "x" * 100  # Exceeds max length
        }

        response = await client.post("/api/v1/datasets", json=payload)

        assert response.status_code == 422


class TestListDatasets:
    """Tests for GET /api/v1/datasets endpoint."""

    async def test_list_datasets_empty(self, client: AsyncClient):
        """Test listing datasets when none exist."""
        response = await client.get("/api/v1/datasets")

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["pagination"]["total"] == 0

    async def test_list_datasets_with_data(self, client: AsyncClient, async_session: AsyncSession):
        """Test listing datasets with existing data."""
        # Create test datasets
        datasets = [
            Dataset(name=f"Dataset {i}", source="HuggingFace", status=DatasetStatus.READY)
            for i in range(3)
        ]
        async_session.add_all(datasets)
        await async_session.commit()

        response = await client.get("/api/v1/datasets")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        assert data["pagination"]["total"] == 3
        assert data["pagination"]["page"] == 1

    async def test_list_datasets_pagination(self, client: AsyncClient, async_session: AsyncSession):
        """Test dataset pagination."""
        # Create 10 test datasets
        datasets = [
            Dataset(name=f"Dataset {i}", source="HuggingFace", status=DatasetStatus.READY)
            for i in range(10)
        ]
        async_session.add_all(datasets)
        await async_session.commit()

        # Request page 2 with limit 3
        response = await client.get("/api/v1/datasets?page=2&limit=3")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        assert data["pagination"]["page"] == 2
        assert data["pagination"]["total"] == 10
        assert data["pagination"]["total_pages"] == 4
        assert data["pagination"]["has_prev"] is True
        assert data["pagination"]["has_next"] is True

    async def test_list_datasets_filter_by_status(self, client: AsyncClient, async_session: AsyncSession):
        """Test filtering datasets by status."""
        datasets = [
            Dataset(name="Dataset 1", source="HuggingFace", status=DatasetStatus.READY),
            Dataset(name="Dataset 2", source="HuggingFace", status=DatasetStatus.DOWNLOADING),
            Dataset(name="Dataset 3", source="HuggingFace", status=DatasetStatus.READY),
        ]
        async_session.add_all(datasets)
        await async_session.commit()

        response = await client.get("/api/v1/datasets?status=ready")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert all(d["status"] == "ready" for d in data["data"])

    async def test_list_datasets_search(self, client: AsyncClient, async_session: AsyncSession):
        """Test searching datasets by name."""
        datasets = [
            Dataset(name="OpenWebText", source="HuggingFace", status=DatasetStatus.READY),
            Dataset(name="TinyStories", source="HuggingFace", status=DatasetStatus.READY),
            Dataset(name="CodeParrot", source="HuggingFace", status=DatasetStatus.READY),
        ]
        async_session.add_all(datasets)
        await async_session.commit()

        response = await client.get("/api/v1/datasets?search=Tiny")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert "Tiny" in data["data"][0]["name"]

    async def test_list_datasets_sort_by_name(self, client: AsyncClient, async_session: AsyncSession):
        """Test sorting datasets by name."""
        datasets = [
            Dataset(name="Charlie", source="HuggingFace", status=DatasetStatus.READY),
            Dataset(name="Alpha", source="HuggingFace", status=DatasetStatus.READY),
            Dataset(name="Bravo", source="HuggingFace", status=DatasetStatus.READY),
        ]
        async_session.add_all(datasets)
        await async_session.commit()

        response = await client.get("/api/v1/datasets?sort_by=name&order=asc")

        assert response.status_code == 200
        data = response.json()
        names = [d["name"] for d in data["data"]]
        assert names == ["Alpha", "Bravo", "Charlie"]


class TestGetDataset:
    """Tests for GET /api/v1/datasets/{dataset_id} endpoint."""

    async def test_get_dataset_success(self, client: AsyncClient, async_session: AsyncSession):
        """Test getting a dataset by ID."""
        dataset = Dataset(
            name="Test Dataset",
            source="HuggingFace",
            hf_repo_id="test/dataset",
            status=DatasetStatus.READY
        )
        async_session.add(dataset)
        await async_session.commit()
        await async_session.refresh(dataset)

        response = await client.get(f"/api/v1/datasets/{dataset.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(dataset.id)
        assert data["name"] == dataset.name
        assert data["source"] == dataset.source

    async def test_get_dataset_not_found(self, client: AsyncClient):
        """Test getting non-existent dataset."""
        fake_uuid = "12345678-1234-1234-1234-123456789012"
        response = await client.get(f"/api/v1/datasets/{fake_uuid}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_get_dataset_invalid_uuid(self, client: AsyncClient):
        """Test getting dataset with invalid UUID."""
        response = await client.get("/api/v1/datasets/invalid-uuid")

        assert response.status_code == 422  # Validation error


class TestUpdateDataset:
    """Tests for PATCH /api/v1/datasets/{dataset_id} endpoint."""

    async def test_update_dataset_success(self, client: AsyncClient, async_session: AsyncSession):
        """Test successful dataset update."""
        dataset = Dataset(
            name="Original Name",
            source="HuggingFace",
            status=DatasetStatus.DOWNLOADING,
            progress=50.0
        )
        async_session.add(dataset)
        await async_session.commit()
        await async_session.refresh(dataset)

        update_payload = {
            "status": "ready",
            "progress": 100.0
        }

        response = await client.patch(
            f"/api/v1/datasets/{dataset.id}",
            json=update_payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["progress"] == 100.0

    async def test_update_dataset_partial(self, client: AsyncClient, async_session: AsyncSession):
        """Test partial dataset update."""
        dataset = Dataset(
            name="Original",
            source="HuggingFace",
            status=DatasetStatus.READY
        )
        async_session.add(dataset)
        await async_session.commit()
        await async_session.refresh(dataset)

        update_payload = {"name": "Updated Name"}

        response = await client.patch(
            f"/api/v1/datasets/{dataset.id}",
            json=update_payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["status"] == "ready"  # Unchanged

    async def test_update_dataset_not_found(self, client: AsyncClient):
        """Test updating non-existent dataset."""
        fake_uuid = "12345678-1234-1234-1234-123456789012"
        response = await client.patch(
            f"/api/v1/datasets/{fake_uuid}",
            json={"name": "New Name"}
        )

        assert response.status_code == 404

    async def test_update_dataset_invalid_progress(self, client: AsyncClient, async_session: AsyncSession):
        """Test update with invalid progress value."""
        dataset = Dataset(
            name="Test",
            source="HuggingFace",
            status=DatasetStatus.DOWNLOADING
        )
        async_session.add(dataset)
        await async_session.commit()
        await async_session.refresh(dataset)

        update_payload = {"progress": 150.0}  # Invalid: > 100

        response = await client.patch(
            f"/api/v1/datasets/{dataset.id}",
            json=update_payload
        )

        assert response.status_code == 422


class TestDeleteDataset:
    """Tests for DELETE /api/v1/datasets/{dataset_id} endpoint."""

    async def test_delete_dataset_success(self, client: AsyncClient, async_session: AsyncSession):
        """Test successful dataset deletion."""
        dataset = Dataset(
            name="To Delete",
            source="HuggingFace",
            status=DatasetStatus.READY
        )
        async_session.add(dataset)
        await async_session.commit()
        await async_session.refresh(dataset)

        response = await client.delete(f"/api/v1/datasets/{dataset.id}")

        assert response.status_code == 204

        # Verify dataset is deleted
        from sqlalchemy import select
        result = await async_session.execute(
            select(Dataset).where(Dataset.id == dataset.id)
        )
        assert result.scalar_one_or_none() is None

    async def test_delete_dataset_not_found(self, client: AsyncClient):
        """Test deleting non-existent dataset."""
        fake_uuid = "12345678-1234-1234-1234-123456789012"
        response = await client.delete(f"/api/v1/datasets/{fake_uuid}")

        assert response.status_code == 404


class TestDownloadDataset:
    """Tests for POST /api/v1/datasets/download endpoint."""

    async def test_download_dataset_success(self, client: AsyncClient):
        """Test successful dataset download initiation."""
        payload = {
            "repo_id": "roneneldan/TinyStories",
            "split": "train"
        }

        response = await client.post("/api/v1/datasets/download", json=payload)

        assert response.status_code == 202  # Accepted
        data = response.json()
        assert data["hf_repo_id"] == payload["repo_id"]
        assert data["status"] == "downloading"
        assert "id" in data

    async def test_download_dataset_duplicate(self, client: AsyncClient, async_session: AsyncSession):
        """Test downloading dataset that already exists."""
        repo_id = "test/existing"
        dataset = Dataset(
            name="Existing",
            source="HuggingFace",
            hf_repo_id=repo_id,
            status=DatasetStatus.READY
        )
        async_session.add(dataset)
        await async_session.commit()

        payload = {"repo_id": repo_id}

        response = await client.post("/api/v1/datasets/download", json=payload)

        assert response.status_code == 409  # Conflict
        assert "already exists" in response.json()["detail"].lower()

    async def test_download_dataset_invalid_repo_id(self, client: AsyncClient):
        """Test download with invalid repo_id format."""
        payload = {
            "repo_id": "invalid_format"  # Missing '/'
        }

        response = await client.post("/api/v1/datasets/download", json=payload)

        assert response.status_code == 422
        assert "format" in response.json()["detail"][0]["msg"].lower()

    async def test_download_dataset_with_access_token(self, client: AsyncClient):
        """Test download with access token for gated dataset."""
        payload = {
            "repo_id": "private/dataset",
            "access_token": "hf_xxxxxxxxxxxx",
            "split": "validation"
        }

        response = await client.post("/api/v1/datasets/download", json=payload)

        assert response.status_code == 202
        data = response.json()
        # Check that dataset was created successfully
        assert "id" in data
        assert data["hf_repo_id"] == "private/dataset"
        assert data["status"] == "downloading"
