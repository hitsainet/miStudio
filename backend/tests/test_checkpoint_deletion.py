#!/usr/bin/env python3
"""
Test script to verify CheckpointService.delete_checkpoint() properly cleans up
multi-layer checkpoint directories.

Usage:
    pytest tests/test_checkpoint_deletion.py -v
"""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text

from src.models.checkpoint import Checkpoint
from src.models.training import Training, TrainingStatus
from src.models.model import Model, ModelStatus, QuantizationFormat
from src.services.checkpoint_service import CheckpointService


@pytest.mark.asyncio
async def test_multilayer_checkpoint_deletion(async_session):
    """
    Test that deleting a multi-layer checkpoint properly cleans up:
    1. The checkpoint file
    2. Empty layer directories
    3. Empty checkpoint step directories
    """
    print("=" * 70)
    print("Testing Multi-Layer Checkpoint Deletion")
    print("=" * 70)

    # Create test checkpoint structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multi-layer checkpoint structure
        checkpoint_dir = temp_path / "checkpoint_500000"
        layer7_dir = checkpoint_dir / "layer_7"
        layer14_dir = checkpoint_dir / "layer_14"
        layer18_dir = checkpoint_dir / "layer_18"

        layer7_dir.mkdir(parents=True)
        layer14_dir.mkdir(parents=True)
        layer18_dir.mkdir(parents=True)

        # Create checkpoint files
        layer7_file = layer7_dir / "checkpoint.safetensors"
        layer14_file = layer14_dir / "checkpoint.safetensors"
        layer18_file = layer18_dir / "checkpoint.safetensors"

        layer7_file.write_text("dummy checkpoint data layer 7")
        layer14_file.write_text("dummy checkpoint data layer 14")
        layer18_file.write_text("dummy checkpoint data layer 18")

        print(f"\nCreated test checkpoint structure at: {checkpoint_dir}")
        print(f"  Layer 7 file:  {layer7_file.exists()}")
        print(f"  Layer 14 file: {layer14_file.exists()}")
        print(f"  Layer 18 file: {layer18_file.exists()}")

        # Use the async_session fixture which provides a clean test database
        db = async_session

        # Create required parent records
        # 1. First create a Model record (required by Training FK)
        test_model_id = f"test_model_{uuid4().hex[:8]}"
        model = Model(
            id=test_model_id,
            name="Test Model",
            architecture="gpt2",
            params_count=124000000,
            quantization=QuantizationFormat.FP32,
            status=ModelStatus.READY,
        )
        db.add(model)
        await db.flush()

        # 2. Create a Training record (required by Checkpoint FK)
        test_training_id = f"test_training_{uuid4().hex[:8]}"
        training = Training(
            id=test_training_id,
            model_id=test_model_id,
            dataset_id=str(uuid4()),  # Fake dataset ID
            status=TrainingStatus.COMPLETED.value,
            total_steps=1000000,
            hyperparameters={"l1_alpha": 0.01},
        )
        db.add(training)
        await db.flush()

        # 3. Create checkpoint records
        ckpt7 = Checkpoint(
            id=f"test_ckpt_layer7_{uuid4().hex[:8]}",
            training_id=test_training_id,
            step=500000,
            loss=0.01,
            storage_path=str(layer7_file),
            file_size_bytes=len("dummy checkpoint data layer 7")
        )
        ckpt14 = Checkpoint(
            id=f"test_ckpt_layer14_{uuid4().hex[:8]}",
            training_id=test_training_id,
            step=500000,
            loss=0.01,
            storage_path=str(layer14_file),
            file_size_bytes=len("dummy checkpoint data layer 14")
        )
        ckpt18 = Checkpoint(
            id=f"test_ckpt_layer18_{uuid4().hex[:8]}",
            training_id=test_training_id,
            step=500000,
            loss=0.01,
            storage_path=str(layer18_file),
            file_size_bytes=len("dummy checkpoint data layer 18")
        )

        db.add(ckpt7)
        db.add(ckpt14)
        db.add(ckpt18)
        await db.commit()

        print("\nCreated 3 checkpoint database records")

        # Test 1: Delete first checkpoint
        print("\n--- Test 1: Delete first checkpoint (layer 7) ---")
        success = await CheckpointService.delete_checkpoint(db, ckpt7.id, delete_file=True)
        print(f"Delete result: {success}")
        print(f"  Layer 7 file exists:  {layer7_file.exists()}")
        print(f"  Layer 7 dir exists:   {layer7_dir.exists()}")
        print(f"  Layer 14 file exists: {layer14_file.exists()}")
        print(f"  Layer 18 file exists: {layer18_file.exists()}")
        print(f"  Checkpoint dir exists: {checkpoint_dir.exists()}")

        assert not layer7_file.exists(), "Layer 7 file should be deleted"
        assert not layer7_dir.exists(), "Layer 7 directory should be deleted (empty)"
        assert layer14_file.exists(), "Layer 14 file should still exist"
        assert layer18_file.exists(), "Layer 18 file should still exist"
        assert checkpoint_dir.exists(), "Checkpoint directory should still exist (not empty)"
        print("  Test 1 passed")

        # Test 2: Delete second checkpoint
        print("\n--- Test 2: Delete second checkpoint (layer 14) ---")
        success = await CheckpointService.delete_checkpoint(db, ckpt14.id, delete_file=True)
        print(f"Delete result: {success}")
        print(f"  Layer 14 file exists: {layer14_file.exists()}")
        print(f"  Layer 14 dir exists:  {layer14_dir.exists()}")
        print(f"  Layer 18 file exists: {layer18_file.exists()}")
        print(f"  Checkpoint dir exists: {checkpoint_dir.exists()}")

        assert not layer14_file.exists(), "Layer 14 file should be deleted"
        assert not layer14_dir.exists(), "Layer 14 directory should be deleted (empty)"
        assert layer18_file.exists(), "Layer 18 file should still exist"
        assert checkpoint_dir.exists(), "Checkpoint directory should still exist (not empty)"
        print("  Test 2 passed")

        # Test 3: Delete third checkpoint (should clean up checkpoint dir)
        print("\n--- Test 3: Delete third checkpoint (layer 18) ---")
        success = await CheckpointService.delete_checkpoint(db, ckpt18.id, delete_file=True)
        print(f"Delete result: {success}")
        print(f"  Layer 18 file exists: {layer18_file.exists()}")
        print(f"  Layer 18 dir exists:  {layer18_dir.exists()}")
        print(f"  Checkpoint dir exists: {checkpoint_dir.exists()}")

        assert not layer18_file.exists(), "Layer 18 file should be deleted"
        assert not layer18_dir.exists(), "Layer 18 directory should be deleted (empty)"
        assert not checkpoint_dir.exists(), "Checkpoint directory should be deleted (empty)"
        print("  Test 3 passed")

    print("\n" + "=" * 70)
    print("All tests passed! Multi-layer checkpoint deletion works correctly.")
    print("=" * 70)
