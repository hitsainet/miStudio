"""
Unit tests for ModelService.

Tests model management operations including download initiation,
CRUD operations, progress tracking, status updates, and cleanup.
"""

import pytest
import pytest_asyncio
from datetime import datetime, UTC

from src.services.model_service import ModelService
from src.models.model import Model, ModelStatus, QuantizationFormat
from src.schemas.model import ModelDownloadRequest, ModelUpdate


@pytest.mark.asyncio
class TestModelServiceIDGeneration:
    """Test ModelService.generate_model_id()."""

    async def test_generate_model_id_format(self):
        """Test that generated model ID has correct format."""
        model_id = ModelService.generate_model_id()

        assert model_id.startswith("m_")
        assert len(model_id) == 10  # "m_" + 8 hex chars

    async def test_generate_model_id_uniqueness(self):
        """Test that generated IDs are unique."""
        ids = [ModelService.generate_model_id() for _ in range(10)]

        # All IDs should be unique
        assert len(set(ids)) == 10


@pytest.mark.asyncio
class TestModelServiceInitiateDownload:
    """Test ModelService.initiate_model_download()."""

    async def test_initiate_download_success(self, async_session):
        """Test initiating a model download."""
        download_request = ModelDownloadRequest(
            repo_id="meta-llama/Llama-2-7b-hf",
            quantization=QuantizationFormat.FP16
        )

        model = await ModelService.initiate_model_download(async_session, download_request)

        assert model is not None
        assert model.id.startswith("m_")
        assert model.name == "Llama-2-7b-hf"  # Extracted from repo_id
        assert model.repo_id == "meta-llama/Llama-2-7b-hf"
        assert model.status == ModelStatus.DOWNLOADING
        assert model.progress == 0.0
        assert model.quantization == QuantizationFormat.FP16

    async def test_initiate_download_extracts_model_name(self, async_session):
        """Test that model name is correctly extracted from repo_id."""
        download_request = ModelDownloadRequest(
            repo_id="organization/model-name-v2",
            quantization=QuantizationFormat.Q8  # Q8 is 8-bit quantization
        )

        model = await ModelService.initiate_model_download(async_session, download_request)

        assert model.name == "model-name-v2"

    async def test_initiate_download_different_quantizations(self, async_session):
        """Test initiating downloads with different quantization formats."""
        quantizations = [
            QuantizationFormat.FP32,
            QuantizationFormat.FP16,
            QuantizationFormat.Q8,  # 8-bit quantization
            QuantizationFormat.Q4   # 4-bit quantization
        ]

        for quant in quantizations:
            download_request = ModelDownloadRequest(
                repo_id=f"test/{quant.value}",
                quantization=quant
            )

            model = await ModelService.initiate_model_download(async_session, download_request)

            assert model.quantization == quant


@pytest.mark.asyncio
class TestModelServiceGet:
    """Test ModelService.get_model() and get_model_by_name()."""

    async def test_get_model_by_id_success(self, async_session):
        """Test getting a model by ID."""
        # Create a model first
        download_request = ModelDownloadRequest(
            repo_id="test/model",
            quantization=QuantizationFormat.FP16
        )
        created_model = await ModelService.initiate_model_download(async_session, download_request)

        # Get the model
        fetched_model = await ModelService.get_model(async_session, created_model.id)

        assert fetched_model is not None
        assert fetched_model.id == created_model.id
        assert fetched_model.repo_id == "test/model"

    async def test_get_model_not_found(self, async_session):
        """Test getting a non-existent model returns None."""
        model = await ModelService.get_model(async_session, "m_nonexist")

        assert model is None

    async def test_get_model_by_name_success(self, async_session):
        """Test getting a model by name."""
        download_request = ModelDownloadRequest(
            repo_id="test/my-model",
            quantization=QuantizationFormat.FP16
        )
        created_model = await ModelService.initiate_model_download(async_session, download_request)

        # Get by name
        fetched_model = await ModelService.get_model_by_name(async_session, "my-model")

        assert fetched_model is not None
        assert fetched_model.name == "my-model"
        assert fetched_model.id == created_model.id

    async def test_get_model_by_name_not_found(self, async_session):
        """Test getting non-existent model by name returns None."""
        model = await ModelService.get_model_by_name(async_session, "nonexistent")

        assert model is None


@pytest.mark.asyncio
class TestModelServiceList:
    """Test ModelService.list_models()."""

    async def test_list_all_models(self, async_session):
        """Test listing all models."""
        # Create multiple models
        for i in range(3):
            download_request = ModelDownloadRequest(
                repo_id=f"test/model-{i}",
                quantization=QuantizationFormat.FP16
            )
            await ModelService.initiate_model_download(async_session, download_request)

        # List all models
        models, total = await ModelService.list_models(async_session)

        assert len(models) == 3
        assert total == 3

    async def test_list_models_filter_by_status(self, async_session):
        """Test filtering models by status."""
        # Create models with different statuses
        download_request = ModelDownloadRequest(
            repo_id="test/model-1",
            quantization=QuantizationFormat.FP16
        )
        model1 = await ModelService.initiate_model_download(async_session, download_request)

        download_request2 = ModelDownloadRequest(
            repo_id="test/model-2",
            quantization=QuantizationFormat.FP16
        )
        model2 = await ModelService.initiate_model_download(async_session, download_request2)

        # Mark one as ready
        await ModelService.mark_model_ready(
            async_session,
            model2.id,
            architecture="gpt2",
            params_count=117000000,
            architecture_config={"num_layers": 12},
            memory_required_bytes=500000000,
            disk_size_bytes=450000000,
            file_path="/data/models/model2"
        )

        # Filter by DOWNLOADING status
        downloading_models, total = await ModelService.list_models(
            async_session,
            status=ModelStatus.DOWNLOADING
        )

        assert len(downloading_models) == 1
        assert total == 1
        assert downloading_models[0].id == model1.id

    async def test_list_models_filter_by_quantization(self, async_session):
        """Test filtering models by quantization format."""
        # Create models with different quantizations
        # Note: QuantizationFormat only has FP32 and FP16 (no INT8/INT4)
        for quant in [QuantizationFormat.FP16, QuantizationFormat.FP32, QuantizationFormat.FP16]:
            download_request = ModelDownloadRequest(
                repo_id=f"test/{quant.value}",
                quantization=quant
            )
            await ModelService.initiate_model_download(async_session, download_request)

        # Filter by FP16
        fp16_models, total = await ModelService.list_models(
            async_session,
            quantization=QuantizationFormat.FP16
        )

        assert len(fp16_models) == 2
        assert total == 2
        assert all(m.quantization == QuantizationFormat.FP16 for m in fp16_models)

    async def test_list_models_search(self, async_session):
        """Test searching models by name or repo_id."""
        # Create models with different names
        names = ["llama-model", "gpt-model", "llama-v2"]
        for name in names:
            download_request = ModelDownloadRequest(
                repo_id=f"test/{name}",
                quantization=QuantizationFormat.FP16
            )
            await ModelService.initiate_model_download(async_session, download_request)

        # Search for "llama"
        llama_models, total = await ModelService.list_models(
            async_session,
            search="llama"
        )

        assert len(llama_models) == 2
        assert total == 2
        assert all("llama" in m.name for m in llama_models)

    async def test_list_models_pagination(self, async_session):
        """Test pagination of model list."""
        # Create 5 models
        for i in range(5):
            download_request = ModelDownloadRequest(
                repo_id=f"test/model-{i}",
                quantization=QuantizationFormat.FP16
            )
            await ModelService.initiate_model_download(async_session, download_request)

        # Get first page (2 items)
        page1_models, total = await ModelService.list_models(
            async_session,
            skip=0,
            limit=2
        )

        assert len(page1_models) == 2
        assert total == 5

        # Get second page (2 items)
        page2_models, total = await ModelService.list_models(
            async_session,
            skip=2,
            limit=2
        )

        assert len(page2_models) == 2
        assert total == 5

        # Verify different models on different pages
        page1_ids = {m.id for m in page1_models}
        page2_ids = {m.id for m in page2_models}
        assert page1_ids.isdisjoint(page2_ids)

    async def test_list_models_sorting(self, async_session):
        """Test sorting models by created_at."""
        # Create models in sequence
        model_ids = []
        for i in range(3):
            download_request = ModelDownloadRequest(
                repo_id=f"test/model-{i}",
                quantization=QuantizationFormat.FP16
            )
            model = await ModelService.initiate_model_download(async_session, download_request)
            model_ids.append(model.id)

        # List with default sorting (created_at desc - newest first)
        models_desc, _ = await ModelService.list_models(async_session)
        desc_ids = [m.id for m in models_desc]

        # Should be in reverse order (newest first)
        assert desc_ids == list(reversed(model_ids))

        # List with ascending order (oldest first)
        models_asc, _ = await ModelService.list_models(
            async_session,
            sort_by="created_at",
            order="asc"
        )
        asc_ids = [m.id for m in models_asc]

        # Should be in original order
        assert asc_ids == model_ids


@pytest.mark.asyncio
class TestModelServiceUpdate:
    """Test ModelService.update_model()."""

    async def test_update_model_success(self, async_session):
        """Test updating a model."""
        # Create model
        download_request = ModelDownloadRequest(
            repo_id="test/model",
            quantization=QuantizationFormat.FP16
        )
        model = await ModelService.initiate_model_download(async_session, download_request)

        # Update model
        updates = ModelUpdate(
            architecture="gpt2",
            params_count=117000000
        )
        updated_model = await ModelService.update_model(async_session, model.id, updates)

        assert updated_model is not None
        assert updated_model.architecture == "gpt2"
        assert updated_model.params_count == 117000000

    async def test_update_model_not_found(self, async_session):
        """Test updating non-existent model returns None."""
        updates = ModelUpdate(architecture="gpt2")
        result = await ModelService.update_model(async_session, "m_nonexist", updates)

        assert result is None


@pytest.mark.asyncio
class TestModelServiceProgressTracking:
    """Test ModelService.update_model_progress()."""

    async def test_update_progress_success(self, async_session):
        """Test updating model download progress."""
        # Create model
        download_request = ModelDownloadRequest(
            repo_id="test/model",
            quantization=QuantizationFormat.FP16
        )
        model = await ModelService.initiate_model_download(async_session, download_request)

        # Update progress
        updated_model = await ModelService.update_model_progress(
            async_session,
            model.id,
            progress=45.5
        )

        assert updated_model is not None
        assert updated_model.progress == 45.5
        assert updated_model.status == ModelStatus.DOWNLOADING  # Status unchanged

    async def test_update_progress_with_status_change(self, async_session):
        """Test updating progress with status change."""
        # Create model
        download_request = ModelDownloadRequest(
            repo_id="test/model",
            quantization=QuantizationFormat.FP16
        )
        model = await ModelService.initiate_model_download(async_session, download_request)

        # Update progress and status
        updated_model = await ModelService.update_model_progress(
            async_session,
            model.id,
            progress=100.0,
            status=ModelStatus.READY
        )

        assert updated_model is not None
        assert updated_model.progress == 100.0
        assert updated_model.status == ModelStatus.READY

    async def test_update_progress_not_found(self, async_session):
        """Test updating progress for non-existent model returns None."""
        result = await ModelService.update_model_progress(
            async_session,
            "m_nonexist",
            progress=50.0
        )

        assert result is None


@pytest.mark.asyncio
class TestModelServiceMarkReady:
    """Test ModelService.mark_model_ready()."""

    async def test_mark_ready_success(self, async_session):
        """Test marking a model as ready."""
        # Create model
        download_request = ModelDownloadRequest(
            repo_id="test/model",
            quantization=QuantizationFormat.FP16
        )
        model = await ModelService.initiate_model_download(async_session, download_request)

        # Mark as ready
        ready_model = await ModelService.mark_model_ready(
            async_session,
            model.id,
            architecture="llama",
            params_count=7000000000,
            architecture_config={"num_layers": 32, "hidden_size": 4096},
            memory_required_bytes=14000000000,
            disk_size_bytes=13000000000,
            file_path="/data/models/llama-7b",
            quantized_path="/data/models/llama-7b-fp16"
        )

        assert ready_model is not None
        assert ready_model.status == ModelStatus.READY
        assert ready_model.progress == 100.0
        assert ready_model.architecture == "llama"
        assert ready_model.params_count == 7000000000
        assert ready_model.architecture_config == {"num_layers": 32, "hidden_size": 4096}
        assert ready_model.memory_required_bytes == 14000000000
        assert ready_model.disk_size_bytes == 13000000000
        assert ready_model.file_path == "/data/models/llama-7b"
        assert ready_model.quantized_path == "/data/models/llama-7b-fp16"

    async def test_mark_ready_without_quantized_path(self, async_session):
        """Test marking ready without quantized path (optional field)."""
        download_request = ModelDownloadRequest(
            repo_id="test/model",
            quantization=QuantizationFormat.FP32
        )
        model = await ModelService.initiate_model_download(async_session, download_request)

        ready_model = await ModelService.mark_model_ready(
            async_session,
            model.id,
            architecture="gpt2",
            params_count=117000000,
            architecture_config={},
            memory_required_bytes=500000000,
            disk_size_bytes=450000000,
            file_path="/data/models/gpt2"
        )

        assert ready_model.quantized_path is None

    async def test_mark_ready_not_found(self, async_session):
        """Test marking non-existent model as ready returns None."""
        result = await ModelService.mark_model_ready(
            async_session,
            "m_nonexist",
            architecture="gpt2",
            params_count=117000000,
            architecture_config={},
            memory_required_bytes=500000000,
            disk_size_bytes=450000000,
            file_path="/data/models/gpt2"
        )

        assert result is None


@pytest.mark.asyncio
class TestModelServiceMarkError:
    """Test ModelService.mark_model_error()."""

    async def test_mark_error_success(self, async_session):
        """Test marking a model as error."""
        # Create model
        download_request = ModelDownloadRequest(
            repo_id="test/model",
            quantization=QuantizationFormat.FP16
        )
        model = await ModelService.initiate_model_download(async_session, download_request)

        # Mark as error
        error_model = await ModelService.mark_model_error(
            async_session,
            model.id,
            error_message="Network connection failed"
        )

        assert error_model is not None
        assert error_model.status == ModelStatus.ERROR
        assert error_model.error_message == "Network connection failed"

    async def test_mark_error_not_found(self, async_session):
        """Test marking non-existent model as error returns None."""
        result = await ModelService.mark_model_error(
            async_session,
            "m_nonexist",
            error_message="Some error"
        )

        assert result is None


@pytest.mark.asyncio
class TestModelServiceDelete:
    """Test ModelService.delete_model()."""

    async def test_delete_model_success(self, async_session):
        """Test deleting a model."""
        # Create model
        download_request = ModelDownloadRequest(
            repo_id="test/model",
            quantization=QuantizationFormat.FP16
        )
        model = await ModelService.initiate_model_download(async_session, download_request)

        # Delete model
        result = await ModelService.delete_model(async_session, model.id)

        assert result is not None
        assert result["deleted"] is True
        assert result["model_id"] == model.id
        assert result["file_path"] is not None
        assert result["deleted_extractions"] == 0

        # Verify model deleted
        fetched = await ModelService.get_model(async_session, model.id)
        assert fetched is None

    async def test_delete_model_with_paths(self, async_session):
        """Test that delete returns file paths for cleanup."""
        download_request = ModelDownloadRequest(
            repo_id="test/model",
            quantization=QuantizationFormat.FP16
        )
        model = await ModelService.initiate_model_download(async_session, download_request)

        # Mark as ready with paths
        await ModelService.mark_model_ready(
            async_session,
            model.id,
            architecture="gpt2",
            params_count=117000000,
            architecture_config={},
            memory_required_bytes=500000000,
            disk_size_bytes=450000000,
            file_path="/data/models/test-model",
            quantized_path="/data/models/test-model-fp16"
        )

        # Delete
        result = await ModelService.delete_model(async_session, model.id)

        assert result["file_path"] == "/data/models/test-model"
        assert result["quantized_path"] == "/data/models/test-model-fp16"

    async def test_delete_model_not_found(self, async_session):
        """Test deleting non-existent model returns None."""
        result = await ModelService.delete_model(async_session, "m_nonexist")

        assert result is None


@pytest.mark.asyncio
class TestModelServiceArchitectureInfo:
    """Test ModelService.get_model_architecture_info()."""

    async def test_get_architecture_info_success(self, async_session):
        """Test getting model architecture information."""
        # Create and prepare model
        download_request = ModelDownloadRequest(
            repo_id="test/model",
            quantization=QuantizationFormat.FP16
        )
        model = await ModelService.initiate_model_download(async_session, download_request)

        # Mark as ready with architecture info
        await ModelService.mark_model_ready(
            async_session,
            model.id,
            architecture="llama",
            params_count=7000000000,
            architecture_config={"num_layers": 32, "hidden_size": 4096},
            memory_required_bytes=14000000000,
            disk_size_bytes=13000000000,
            file_path="/data/models/llama-7b"
        )

        # Get architecture info
        info = await ModelService.get_model_architecture_info(async_session, model.id)

        assert info is not None
        assert info["model_id"] == model.id
        assert info["name"] == "model"
        assert info["architecture"] == "llama"
        assert info["params_count"] == 7000000000
        assert info["quantization"] == "FP16"  # Enum value is uppercase
        assert info["architecture_config"] == {"num_layers": 32, "hidden_size": 4096}
        assert info["memory_required_bytes"] == 14000000000
        assert info["disk_size_bytes"] == 13000000000

    async def test_get_architecture_info_not_found(self, async_session):
        """Test getting architecture info for non-existent model returns None."""
        info = await ModelService.get_model_architecture_info(async_session, "m_nonexist")

        assert info is None
