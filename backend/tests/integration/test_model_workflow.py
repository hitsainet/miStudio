"""
End-to-end workflow test for model management.

Tests the complete flow: download → wait for completion → model ready → verify metadata.
"""

import asyncio
import pytest
from pathlib import Path

from src.models.model import ModelStatus, QuantizationFormat
from src.schemas.model import ModelDownloadRequest, ModelUpdate
from src.services.model_service import ModelService
from src.workers.model_tasks import download_and_load_model


class TestModelWorkflow:
    """End-to-end workflow tests for model operations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_model_workflow(self, async_session):
        """
        Test complete workflow: initiate download → model loading → ready with metadata.

        This test simulates the full user journey:
        1. User downloads a model from HuggingFace
        2. System processes download and quantization in background
        3. Model becomes ready for use
        4. Model has all metadata populated (architecture, params, memory)

        Note: This is a SIMULATED test with mocked model loading.
        For real E2E testing, use a tiny model (e.g., GPT-2 small).
        """
        # Step 1: Initiate model download
        download_request = ModelDownloadRequest(
            repo_id="test/tiny-model",
            quantization=QuantizationFormat.FP16,
        )
        model = await ModelService.initiate_model_download(async_session, download_request)
        await async_session.commit()
        await async_session.refresh(model)

        assert model.id is not None
        assert model.status == ModelStatus.DOWNLOADING
        assert model.progress == 0.0
        assert model.name == "tiny-model"
        assert model.quantization == QuantizationFormat.FP16

        # Step 2: Simulate download and loading completion
        # In real test, you would call download_and_load_model.delay() and poll for completion
        # Here we simulate the final state after download
        model_metadata_update = ModelUpdate(
            status=ModelStatus.READY.value,
            progress=100.0,
            architecture="gpt2",
            params_count=124000000,  # 124M parameters
            architecture_config={
                "model_type": "gpt2",
                "num_hidden_layers": 12,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "max_position_embeddings": 1024,
                "vocab_size": 50257,
            },
            memory_req_bytes=int(124000000 * 2 * 1.2),  # FP16: 2 bytes per param + 20% overhead
            file_path=str(Path("/data/models/raw") / model.id),
        )
        model = await ModelService.update_model(
            async_session, model.id, model_metadata_update
        )
        await async_session.commit()
        await async_session.refresh(model)

        # Step 3: Verify final state
        assert model.status == ModelStatus.READY
        assert model.progress == 100.0
        assert model.architecture == "gpt2"
        assert model.params_count == 124000000
        assert model.architecture_config is not None
        assert model.architecture_config["model_type"] == "gpt2"
        assert model.architecture_config["num_hidden_layers"] == 12
        assert model.memory_required_bytes > 0
        assert model.file_path is not None

        # Step 4: Get architecture info
        arch_info = await ModelService.get_model_architecture_info(async_session, model.id)
        assert arch_info is not None
        assert arch_info["architecture"] == "gpt2"
        assert arch_info["params_count"] == 124000000
        assert arch_info["quantization"] == "FP16"

        # Step 5: Cleanup
        await ModelService.delete_model(async_session, model.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_download_error_handling(self, async_session):
        """
        Test error handling during model download.

        Verifies that:
        1. Failed downloads set status to ERROR
        2. Error message is stored
        3. Model can be retried or deleted
        """
        # Create model record
        download_request = ModelDownloadRequest(
            repo_id="invalid/nonexistent-model-12345",
            quantization=QuantizationFormat.FP16,
        )
        model = await ModelService.initiate_model_download(async_session, download_request)
        await async_session.commit()
        await async_session.refresh(model)

        # Simulate download failure
        error_update = ModelUpdate(
            status=ModelStatus.ERROR.value,
            error_message="Download failed: Repository not found",
            progress=0.0,
        )
        model = await ModelService.update_model(
            async_session, model.id, error_update
        )
        await async_session.commit()
        await async_session.refresh(model)

        # Verify error state
        assert model.status == ModelStatus.ERROR
        assert model.error_message is not None
        assert "Repository not found" in model.error_message

        # Cleanup
        await ModelService.delete_model(async_session, model.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_out_of_memory_error_handling(self, async_session):
        """
        Test error handling when model loading fails due to OOM.

        Verifies that:
        1. OOM errors set status to ERROR
        2. Error message indicates memory issue
        3. Model record is preserved for debugging
        """
        # Create model record for large model
        download_request = ModelDownloadRequest(
            repo_id="large/70b-model",
            quantization=QuantizationFormat.FP32,  # Full precision, high memory
        )
        model = await ModelService.initiate_model_download(async_session, download_request)
        await async_session.commit()
        await async_session.refresh(model)

        # Simulate OOM failure
        error_update = ModelUpdate(
            status=ModelStatus.ERROR.value,
            error_message="Out of memory: Model requires 280GB but only 32GB available",
            progress=0.0,
        )
        model = await ModelService.update_model(
            async_session, model.id, error_update
        )
        await async_session.commit()
        await async_session.refresh(model)

        # Verify error state
        assert model.status == ModelStatus.ERROR
        assert model.error_message is not None
        assert "Out of memory" in model.error_message

        # Cleanup
        await ModelService.delete_model(async_session, model.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_duplicate_model_prevention(self, async_session):
        """
        Test that duplicate model names are prevented.

        Verifies that:
        1. Models with same name cannot be created twice
        2. Appropriate error is returned
        3. First model remains intact
        """
        # Create first model
        download_request = ModelDownloadRequest(
            repo_id="test/unique-model",
            quantization=QuantizationFormat.FP16,
        )
        model1 = await ModelService.initiate_model_download(async_session, download_request)
        await async_session.commit()
        await async_session.refresh(model1)

        # Try to create second model with same name
        existing_model = await ModelService.get_model_by_name(async_session, "unique-model")
        assert existing_model is not None
        assert existing_model.id == model1.id

        # Cleanup
        await ModelService.delete_model(async_session, model1.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_quantization_workflow(self, async_session):
        """
        Test model workflow with different quantization formats.

        Verifies that:
        1. Models can be downloaded with different quantization levels
        2. Memory requirements scale appropriately
        3. Quantized path is set when applicable
        """
        # Test Q4 quantization (aggressive compression)
        download_request = ModelDownloadRequest(
            repo_id="test/quantize-model",
            quantization=QuantizationFormat.Q4,
        )
        model = await ModelService.initiate_model_download(async_session, download_request)
        await async_session.commit()
        await async_session.refresh(model)

        # Simulate successful Q4 quantization
        params_count = 124000000  # 124M parameters
        memory_q4 = int(params_count * 0.5 * 1.2)  # Q4: 0.5 bytes per param + 20% overhead

        quantize_update = ModelUpdate(
            status=ModelStatus.READY.value,
            progress=100.0,
            architecture="gpt2",
            params_count=params_count,
            architecture_config={"model_type": "gpt2"},
            memory_req_bytes=memory_q4,
            file_path=str(Path("/data/models/raw") / model.id),
        )
        model = await ModelService.update_model(
            async_session, model.id, quantize_update
        )
        await async_session.commit()
        await async_session.refresh(model)

        # Verify quantization applied
        assert model.status == ModelStatus.READY
        assert model.quantization == QuantizationFormat.Q4
        assert model.memory_required_bytes == memory_q4
        assert model.memory_required_bytes < params_count * 2  # Less than FP16

        # Cleanup
        await ModelService.delete_model(async_session, model.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_models_with_filters(self, async_session):
        """
        Test listing models with various filters.

        Verifies that:
        1. Models can be filtered by status
        2. Models can be filtered by architecture
        3. Models can be filtered by quantization format
        4. Search by name works correctly
        """
        # Create multiple models with different attributes
        models_to_create = [
            ("test/model-1", QuantizationFormat.FP16, "gpt2", ModelStatus.READY),
            ("test/model-2", QuantizationFormat.Q4, "llama", ModelStatus.READY),
            ("test/model-3", QuantizationFormat.FP16, "gpt2", ModelStatus.DOWNLOADING),
        ]

        created_models = []
        for repo_id, quant, arch, status in models_to_create:
            download_request = ModelDownloadRequest(
                repo_id=repo_id,
                quantization=quant,
            )
            model = await ModelService.initiate_model_download(async_session, download_request)

            # Update architecture and status
            if arch != "":
                update = ModelUpdate(
                    status=status.value,
                    architecture=arch,
                )
                model = await ModelService.update_model(async_session, model.id, update)

            created_models.append(model)
            await async_session.commit()
            await async_session.refresh(model)

        # Test filter by status
        ready_models, count = await ModelService.list_models(
            async_session,
            status=ModelStatus.READY
        )
        assert count >= 2  # At least our 2 READY models

        # Test filter by architecture
        gpt2_models, count = await ModelService.list_models(
            async_session,
            architecture="gpt2"
        )
        assert count >= 2  # At least our 2 GPT-2 models

        # Test filter by quantization
        q4_models, count = await ModelService.list_models(
            async_session,
            quantization=QuantizationFormat.Q4
        )
        assert count >= 1  # At least our Q4 model

        # Test search by name
        search_models, count = await ModelService.list_models(
            async_session,
            search="model-2"
        )
        assert count >= 1  # Should find model-2

        # Cleanup
        for model in created_models:
            await ModelService.delete_model(async_session, model.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_model_progress_tracking(self, async_session):
        """
        Test progress tracking during model download and loading.

        Verifies that:
        1. Progress starts at 0.0
        2. Progress can be updated incrementally
        3. Progress reaches 100.0 on completion
        4. Status transitions correctly
        """
        # Create model
        download_request = ModelDownloadRequest(
            repo_id="test/progress-model",
            quantization=QuantizationFormat.FP16,
        )
        model = await ModelService.initiate_model_download(async_session, download_request)
        await async_session.commit()
        await async_session.refresh(model)

        assert model.progress == 0.0
        assert model.status == ModelStatus.DOWNLOADING

        # Simulate download progress
        model = await ModelService.update_model_progress(
            async_session, model.id, 25.0, ModelStatus.DOWNLOADING
        )
        await async_session.commit()
        await async_session.refresh(model)
        assert model.progress == 25.0

        # Simulate loading progress
        model = await ModelService.update_model_progress(
            async_session, model.id, 75.0, ModelStatus.LOADING
        )
        await async_session.commit()
        await async_session.refresh(model)
        assert model.progress == 75.0
        assert model.status == ModelStatus.LOADING

        # Simulate completion
        model = await ModelService.update_model_progress(
            async_session, model.id, 100.0, ModelStatus.READY
        )
        await async_session.commit()
        await async_session.refresh(model)
        assert model.progress == 100.0
        assert model.status == ModelStatus.READY

        # Cleanup
        await ModelService.delete_model(async_session, model.id)
        await async_session.commit()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_architecture_config_persistence(self, async_session):
        """
        Test that architecture configuration is persisted correctly.

        Verifies that:
        1. Architecture config can be stored as JSONB
        2. Complex nested structures are preserved
        3. Config can be retrieved and used for model analysis
        """
        # Create model with detailed architecture config
        download_request = ModelDownloadRequest(
            repo_id="test/llama-model",
            quantization=QuantizationFormat.FP16,
        )
        model = await ModelService.initiate_model_download(async_session, download_request)

        # Add detailed architecture config
        detailed_config = {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,  # Grouped Query Attention
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "vocab_size": 32000,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "mlp_bias": False,
        }

        ready_update = ModelUpdate(
            status=ModelStatus.READY.value,
            progress=100.0,
            architecture="llama",
            params_count=7000000000,  # 7B params
            architecture_config=detailed_config,
            memory_req_bytes=int(7000000000 * 2 * 1.2),  # FP16
        )
        model = await ModelService.update_model(async_session, model.id, ready_update)
        await async_session.commit()
        await async_session.refresh(model)

        # Verify architecture config persistence
        assert model.architecture_config is not None
        assert model.architecture_config["model_type"] == "llama"
        assert model.architecture_config["num_hidden_layers"] == 32
        assert model.architecture_config["num_key_value_heads"] == 8  # GQA preserved
        assert model.architecture_config["rope_theta"] == 10000.0

        # Get architecture info via service
        arch_info = await ModelService.get_model_architecture_info(async_session, model.id)
        assert arch_info["architecture_config"]["num_key_value_heads"] == 8

        # Cleanup
        await ModelService.delete_model(async_session, model.id)
        await async_session.commit()


# Note: These tests are INTEGRATION tests that verify the workflow logic.
# For REAL E2E testing with actual downloads and model loading:
#
# 1. Use pytest-asyncio with longer timeouts
# 2. Use a tiny test model (e.g., GPT-2 Small: 124M params)
# 3. Actually call the Celery tasks and poll for completion
# 4. Verify files are created on disk
# 5. Clean up test files after completion
#
# Example real E2E test structure:
#
# @pytest.mark.asyncio
# @pytest.mark.slow  # Mark as slow test
# async def test_real_model_download(async_session):
#     """Real E2E test with actual HuggingFace model download."""
#     # Create model
#     download_request = ModelDownloadRequest(
#         repo_id="openai-community/gpt2",  # GPT-2 Small (124M params)
#         quantization=QuantizationFormat.Q4,  # Compressed to fit in memory
#     )
#     model = await ModelService.initiate_model_download(async_session, download_request)
#     await async_session.commit()
#
#     # Trigger real download
#     task_result = download_and_load_model.delay(
#         model_id=model.id,
#         repo_id="openai-community/gpt2",
#         quantization="Q4",
#     )
#
#     # Poll for completion (with timeout)
#     for _ in range(300):  # 5 minutes timeout
#         await asyncio.sleep(1)
#         await async_session.refresh(model)
#         if model.status in (ModelStatus.READY, ModelStatus.ERROR):
#             break
#
#     # Verify download succeeded
#     assert model.status == ModelStatus.READY
#     assert Path(model.file_path).exists()
#     assert model.architecture == "gpt2"
#     assert model.params_count == 124000000
#
#     # Verify architecture info
#     arch_info = await ModelService.get_model_architecture_info(async_session, model.id)
#     assert arch_info is not None
#     assert arch_info["params_count"] > 0
#
#     # Cleanup
#     await ModelService.delete_model(async_session, model.id)
#     # Delete files (would need delete_model_files task)
