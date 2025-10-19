"""
Unit tests for CheckpointService.

Tests checkpoint CRUD operations, file I/O, and retention policies.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from uuid import UUID

import pytest
import pytest_asyncio
import torch
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.checkpoint import Checkpoint
from src.models.training import Training
from src.models.model import Model
from src.models.dataset import Dataset
from src.services.checkpoint_service import CheckpointService
from src.ml.sparse_autoencoder import SparseAutoencoder


@pytest_asyncio.fixture
async def test_model(async_session: AsyncSession):
    """Create a test model for checkpoint tests."""
    from src.models.model import ModelStatus, QuantizationFormat

    model = Model(
        id="m_test123",
        name="Test Model",
        repo_id="test/model",
        status=ModelStatus.READY.value,
        quantization=QuantizationFormat.FP16.value,
        architecture="gpt2",
        params_count=117000000,
        architecture_config={"num_hidden_layers": 12, "hidden_size": 768},
    )
    async_session.add(model)
    await async_session.commit()
    return model


@pytest_asyncio.fixture
async def test_dataset(async_session: AsyncSession):
    """Create a test dataset for checkpoint tests."""
    from src.models.dataset import DatasetStatus

    dataset = Dataset(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        name="Test Dataset",
        source="HuggingFace",
        hf_repo_id="test/dataset",
        status=DatasetStatus.READY.value,
        num_samples=1000,
    )
    async_session.add(dataset)
    await async_session.commit()
    return dataset


@pytest_asyncio.fixture
async def test_training(async_session: AsyncSession, test_model, test_dataset):
    """Create a test training for checkpoint tests."""
    from src.models.training import TrainingStatus

    training = Training(
        id="train_test123",
        model_id="m_test123",
        dataset_id="12345678-1234-5678-1234-567812345678",
        status=TrainingStatus.RUNNING.value,
        progress=50.0,
        current_step=5000,
        total_steps=10000,
        hyperparameters={
            "hidden_dim": 768,
            "latent_dim": 16384,
            "l1_alpha": 0.001,
            "learning_rate": 0.0003,
            "batch_size": 4096,
            "total_steps": 10000,
        },
    )
    async_session.add(training)
    await async_session.commit()
    return training


@pytest.mark.asyncio
class TestCheckpointServiceCreate:
    """Test CheckpointService.create_checkpoint()."""

    async def test_create_checkpoint_success(self, async_session, test_training):
        """Test creating a checkpoint record."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
            temp_path = f.name
            f.write(b"fake checkpoint data")

        try:
            checkpoint = await CheckpointService.create_checkpoint(
                db=async_session,
                training_id="train_test123",
                step=5000,
                loss=0.123,
                storage_path=temp_path,
                l0_sparsity=0.05,
                is_best=False,
            )

            # Verify checkpoint created
            assert checkpoint is not None
            assert checkpoint.id.startswith("ckpt_")
            assert checkpoint.training_id == "train_test123"
            assert checkpoint.step == 5000
            assert checkpoint.loss == 0.123
            assert checkpoint.l0_sparsity == 0.05
            assert checkpoint.storage_path == temp_path
            assert checkpoint.is_best is False
            assert checkpoint.file_size_bytes == 20  # len(b"fake checkpoint data")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def test_create_checkpoint_with_metadata(self, async_session, test_training):
        """Test creating checkpoint with extra metadata."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
            temp_path = f.name

        try:
            extra_metadata = {
                "learning_rate": 0.0003,
                "dead_neurons": 100,
                "architecture": "standard",
            }

            checkpoint = await CheckpointService.create_checkpoint(
                db=async_session,
                training_id="train_test123",
                step=1000,
                loss=0.456,
                storage_path=temp_path,
                extra_metadata=extra_metadata,
            )

            assert checkpoint.extra_metadata == extra_metadata
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def test_create_checkpoint_nonexistent_file(self, async_session, test_training):
        """Test creating checkpoint with nonexistent file sets no file size."""
        checkpoint = await CheckpointService.create_checkpoint(
            db=async_session,
            training_id="train_test123",
            step=2000,
            loss=0.789,
            storage_path="/nonexistent/path/checkpoint.safetensors",
        )

        assert checkpoint.file_size_bytes is None

    async def test_create_best_checkpoint(self, async_session, test_training):
        """Test creating checkpoint marked as best."""
        checkpoint = await CheckpointService.create_checkpoint(
            db=async_session,
            training_id="train_test123",
            step=8000,
            loss=0.050,
            storage_path="/tmp/best.safetensors",
            is_best=True,
        )

        assert checkpoint.is_best is True


@pytest.mark.asyncio
class TestCheckpointServiceGet:
    """Test CheckpointService.get_checkpoint()."""

    async def test_get_checkpoint_success(self, async_session, test_training):
        """Test getting checkpoint by ID."""
        # Create checkpoint
        created = await CheckpointService.create_checkpoint(
            db=async_session,
            training_id="train_test123",
            step=3000,
            loss=0.234,
            storage_path="/tmp/test.safetensors",
        )

        # Get checkpoint
        fetched = await CheckpointService.get_checkpoint(async_session, created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.training_id == "train_test123"
        assert fetched.step == 3000
        assert fetched.loss == 0.234

    async def test_get_checkpoint_not_found(self, async_session):
        """Test getting nonexistent checkpoint returns None."""
        checkpoint = await CheckpointService.get_checkpoint(
            async_session, "ckpt_nonexistent"
        )

        assert checkpoint is None


@pytest.mark.asyncio
class TestCheckpointServiceList:
    """Test CheckpointService.list_checkpoints()."""

    async def test_list_checkpoints_empty(self, async_session, test_training):
        """Test listing checkpoints when none exist."""
        checkpoints, total = await CheckpointService.list_checkpoints(
            async_session, "train_test123"
        )

        assert checkpoints == []
        assert total == 0

    async def test_list_checkpoints_multiple(self, async_session, test_training):
        """Test listing multiple checkpoints."""
        # Create 3 checkpoints
        for i, step in enumerate([1000, 2000, 3000]):
            await CheckpointService.create_checkpoint(
                db=async_session,
                training_id="train_test123",
                step=step,
                loss=0.1 * i,
                storage_path=f"/tmp/ckpt_{i}.safetensors",
            )

        checkpoints, total = await CheckpointService.list_checkpoints(
            async_session, "train_test123"
        )

        assert len(checkpoints) == 3
        assert total == 3
        # Should be ordered by step descending (newest first)
        assert checkpoints[0].step == 3000
        assert checkpoints[1].step == 2000
        assert checkpoints[2].step == 1000

    async def test_list_checkpoints_pagination(self, async_session, test_training):
        """Test pagination of checkpoint list."""
        # Create 5 checkpoints
        for i in range(5):
            await CheckpointService.create_checkpoint(
                db=async_session,
                training_id="train_test123",
                step=i * 1000,
                loss=0.1,
                storage_path=f"/tmp/ckpt_{i}.safetensors",
            )

        # Get first page
        page1, total = await CheckpointService.list_checkpoints(
            async_session, "train_test123", skip=0, limit=2
        )

        assert len(page1) == 2
        assert total == 5

        # Get second page
        page2, total = await CheckpointService.list_checkpoints(
            async_session, "train_test123", skip=2, limit=2
        )

        assert len(page2) == 2
        assert total == 5

        # Ensure no overlap
        page1_ids = {c.id for c in page1}
        page2_ids = {c.id for c in page2}
        assert page1_ids.isdisjoint(page2_ids)


@pytest.mark.asyncio
class TestCheckpointServiceBest:
    """Test best checkpoint operations."""

    async def test_get_best_checkpoint_none(self, async_session, test_training):
        """Test getting best checkpoint when none marked."""
        best = await CheckpointService.get_best_checkpoint(
            async_session, "train_test123"
        )

        assert best is None

    async def test_get_best_checkpoint_exists(self, async_session, test_training):
        """Test getting best checkpoint."""
        # Create checkpoints
        await CheckpointService.create_checkpoint(
            db=async_session,
            training_id="train_test123",
            step=1000,
            loss=0.5,
            storage_path="/tmp/ckpt1.safetensors",
            is_best=False,
        )

        best_ckpt = await CheckpointService.create_checkpoint(
            db=async_session,
            training_id="train_test123",
            step=2000,
            loss=0.1,
            storage_path="/tmp/ckpt2.safetensors",
            is_best=True,
        )

        # Get best
        fetched_best = await CheckpointService.get_best_checkpoint(
            async_session, "train_test123"
        )

        assert fetched_best is not None
        assert fetched_best.id == best_ckpt.id
        assert fetched_best.is_best is True

    async def test_update_best_checkpoint(self, async_session, test_training):
        """Test updating which checkpoint is best."""
        # Create two checkpoints
        ckpt1 = await CheckpointService.create_checkpoint(
            db=async_session,
            training_id="train_test123",
            step=1000,
            loss=0.5,
            storage_path="/tmp/ckpt1.safetensors",
            is_best=True,
        )

        ckpt2 = await CheckpointService.create_checkpoint(
            db=async_session,
            training_id="train_test123",
            step=2000,
            loss=0.3,
            storage_path="/tmp/ckpt2.safetensors",
            is_best=False,
        )

        # Update ckpt2 as best
        updated = await CheckpointService.update_best_checkpoint(
            async_session, "train_test123", ckpt2.id
        )

        assert updated is not None
        assert updated.id == ckpt2.id
        assert updated.is_best is True

        # Verify ckpt1 is no longer best
        old_best = await CheckpointService.get_checkpoint(async_session, ckpt1.id)
        assert old_best.is_best is False

    async def test_update_best_checkpoint_not_found(self, async_session, test_training):
        """Test updating nonexistent checkpoint as best returns None."""
        result = await CheckpointService.update_best_checkpoint(
            async_session, "train_test123", "ckpt_nonexistent"
        )

        assert result is None


@pytest.mark.asyncio
class TestCheckpointServiceLatest:
    """Test latest checkpoint operations."""

    async def test_get_latest_checkpoint_none(self, async_session, test_training):
        """Test getting latest checkpoint when none exist."""
        latest = await CheckpointService.get_latest_checkpoint(
            async_session, "train_test123"
        )

        assert latest is None

    async def test_get_latest_checkpoint(self, async_session, test_training):
        """Test getting most recent checkpoint."""
        # Create checkpoints at different steps
        await CheckpointService.create_checkpoint(
            db=async_session,
            training_id="train_test123",
            step=1000,
            loss=0.5,
            storage_path="/tmp/ckpt1.safetensors",
        )

        latest_ckpt = await CheckpointService.create_checkpoint(
            db=async_session,
            training_id="train_test123",
            step=5000,
            loss=0.2,
            storage_path="/tmp/ckpt5.safetensors",
        )

        await CheckpointService.create_checkpoint(
            db=async_session,
            training_id="train_test123",
            step=3000,
            loss=0.3,
            storage_path="/tmp/ckpt3.safetensors",
        )

        # Get latest
        fetched_latest = await CheckpointService.get_latest_checkpoint(
            async_session, "train_test123"
        )

        assert fetched_latest is not None
        assert fetched_latest.id == latest_ckpt.id
        assert fetched_latest.step == 5000


@pytest.mark.asyncio
class TestCheckpointServiceDelete:
    """Test CheckpointService.delete_checkpoint()."""

    async def test_delete_checkpoint_with_file(self, async_session, test_training):
        """Test deleting checkpoint and its file."""
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
            temp_path = f.name
            f.write(b"checkpoint data")

        try:
            # Create checkpoint
            checkpoint = await CheckpointService.create_checkpoint(
                db=async_session,
                training_id="train_test123",
                step=1000,
                loss=0.1,
                storage_path=temp_path,
            )

            assert os.path.exists(temp_path)

            # Delete checkpoint
            deleted = await CheckpointService.delete_checkpoint(
                async_session, checkpoint.id, delete_file=True
            )

            assert deleted is True
            assert not os.path.exists(temp_path)

            # Verify DB record deleted
            fetched = await CheckpointService.get_checkpoint(async_session, checkpoint.id)
            assert fetched is None
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def test_delete_checkpoint_keep_file(self, async_session, test_training):
        """Test deleting checkpoint record but keeping file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
            temp_path = f.name
            f.write(b"checkpoint data")

        try:
            checkpoint = await CheckpointService.create_checkpoint(
                db=async_session,
                training_id="train_test123",
                step=1000,
                loss=0.1,
                storage_path=temp_path,
            )

            # Delete checkpoint but keep file
            deleted = await CheckpointService.delete_checkpoint(
                async_session, checkpoint.id, delete_file=False
            )

            assert deleted is True
            assert os.path.exists(temp_path)  # File still exists
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def test_delete_checkpoint_not_found(self, async_session):
        """Test deleting nonexistent checkpoint returns False."""
        deleted = await CheckpointService.delete_checkpoint(
            async_session, "ckpt_nonexistent"
        )

        assert deleted is False


@pytest.mark.asyncio
class TestCheckpointServiceFileIO:
    """Test checkpoint file save/load operations."""

    def test_save_checkpoint_creates_file(self):
        """Test saving checkpoint creates safetensors file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "checkpoint.safetensors")

            # Create simple model and optimizer
            model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Save checkpoint
            CheckpointService.save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=1000,
                storage_path=storage_path,
            )

            # Verify file created
            assert os.path.exists(storage_path)
            assert os.path.getsize(storage_path) > 0

    def test_save_checkpoint_with_metadata(self):
        """Test saving checkpoint with extra metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "checkpoint.safetensors")

            model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            extra_metadata = {
                "loss": 0.123,
                "l0_sparsity": 0.05,
                "learning_rate": 0.0003,
            }

            # Save with metadata
            CheckpointService.save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=5000,
                storage_path=storage_path,
                extra_metadata=extra_metadata,
            )

            assert os.path.exists(storage_path)

    def test_load_checkpoint_restores_weights(self):
        """Test loading checkpoint restores model weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "checkpoint.safetensors")

            # Create and save model
            original_model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.Adam(original_model.parameters())

            CheckpointService.save_checkpoint(
                model=original_model,
                optimizer=optimizer,
                step=1000,
                storage_path=storage_path,
            )

            # Create new model with different weights
            new_model = torch.nn.Linear(10, 10)
            torch.nn.init.zeros_(new_model.weight)
            torch.nn.init.zeros_(new_model.bias)

            # Load checkpoint
            checkpoint_data = CheckpointService.load_checkpoint(
                storage_path=storage_path,
                model=new_model,
                device="cpu",
            )

            # Verify weights match
            assert torch.allclose(new_model.weight, original_model.weight)
            assert torch.allclose(new_model.bias, original_model.bias)
            assert "model_state" in checkpoint_data
            assert "optimizer_state" in checkpoint_data

    def test_load_checkpoint_nonexistent_file(self):
        """Test loading nonexistent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            CheckpointService.load_checkpoint(
                storage_path="/nonexistent/path.safetensors"
            )

    def test_load_checkpoint_without_model(self):
        """Test loading checkpoint without providing model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "checkpoint.safetensors")

            model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.Adam(model.parameters())

            CheckpointService.save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=1000,
                storage_path=storage_path,
            )

            # Load without model
            checkpoint_data = CheckpointService.load_checkpoint(
                storage_path=storage_path,
                device="cpu",
            )

            assert "model_state" in checkpoint_data
            assert len(checkpoint_data["model_state"]) > 0

    def test_save_load_sae_model(self):
        """Test saving and loading SparseAutoencoder model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "sae_checkpoint.safetensors")

            # Create SAE model
            original_sae = SparseAutoencoder(
                hidden_dim=64,
                latent_dim=256,
                l1_alpha=0.001,
            )
            optimizer = torch.optim.Adam(original_sae.parameters())

            # Save
            CheckpointService.save_checkpoint(
                model=original_sae,
                optimizer=optimizer,
                step=10000,
                storage_path=storage_path,
                extra_metadata={"architecture": "standard"},
            )

            # Create new SAE and load
            new_sae = SparseAutoencoder(
                hidden_dim=64,
                latent_dim=256,
                l1_alpha=0.001,
            )

            checkpoint_data = CheckpointService.load_checkpoint(
                storage_path=storage_path,
                model=new_sae,
                device="cpu",
            )

            # Verify weights loaded
            assert torch.allclose(new_sae.encoder.weight, original_sae.encoder.weight)
            assert torch.allclose(new_sae.encoder.bias, original_sae.encoder.bias)
            assert torch.allclose(new_sae.decoder.weight, original_sae.decoder.weight)
            assert torch.allclose(new_sae.decoder.bias, original_sae.decoder.bias)
            assert torch.allclose(new_sae.decoder_bias, original_sae.decoder_bias)
