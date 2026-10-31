#!/usr/bin/env python3
"""
Standalone test to verify checkpoint directory cleanup logic.
Tests the file deletion logic without requiring database setup.

Usage:
    python tests/test_checkpoint_deletion_standalone.py
"""

import tempfile
from pathlib import Path


def test_directory_cleanup_logic():
    """
    Test that the cleanup logic properly removes:
    1. The checkpoint file
    2. Empty layer directories
    3. Empty checkpoint step directories
    """
    print("=" * 70)
    print("Testing Checkpoint Directory Cleanup Logic")
    print("=" * 70)

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

        layer7_file.write_text("dummy data 7")
        layer14_file.write_text("dummy data 14")
        layer18_file.write_text("dummy data 18")

        print(f"\nInitial structure:")
        print(f"  Checkpoint dir: {checkpoint_dir.exists()}")
        print(f"  Layer 7 dir:    {layer7_dir.exists()}")
        print(f"  Layer 14 dir:   {layer14_dir.exists()}")
        print(f"  Layer 18 dir:   {layer18_dir.exists()}")
        print(f"  Layer 7 file:   {layer7_file.exists()}")
        print(f"  Layer 14 file:  {layer14_file.exists()}")
        print(f"  Layer 18 file:  {layer18_file.exists()}")

        # Test 1: Delete layer 7 file and cleanup
        print("\n--- Test 1: Delete layer 7 ---")
        storage_path = layer7_file

        # Step 1: Delete the checkpoint file
        storage_path.unlink()
        print(f"  Deleted file: {storage_path}")

        # Step 2: Delete layer directory if empty
        layer_dir = storage_path.parent
        if layer_dir.exists() and not any(layer_dir.iterdir()):
            layer_dir.rmdir()
            print(f"  Deleted empty layer dir: {layer_dir}")

        # Step 3: Check checkpoint directory (should NOT be deleted - still has files)
        checkpoint_dir_check = layer_dir.parent
        if checkpoint_dir_check.exists() and not any(checkpoint_dir_check.iterdir()):
            checkpoint_dir_check.rmdir()
            print(f"  Deleted empty checkpoint dir: {checkpoint_dir_check}")
        else:
            print(f"  Checkpoint dir still has contents: {list(checkpoint_dir_check.iterdir())}")

        print(f"\nAfter Test 1:")
        print(f"  Layer 7 file:   {layer7_file.exists()}")
        print(f"  Layer 7 dir:    {layer7_dir.exists()}")
        print(f"  Checkpoint dir: {checkpoint_dir.exists()}")

        assert not layer7_file.exists(), "Layer 7 file should be deleted"
        assert not layer7_dir.exists(), "Layer 7 directory should be deleted"
        assert checkpoint_dir.exists(), "Checkpoint directory should still exist"
        print("  ✓ Test 1 passed")

        # Test 2: Delete layer 14
        print("\n--- Test 2: Delete layer 14 ---")
        storage_path = layer14_file
        storage_path.unlink()
        print(f"  Deleted file: {storage_path}")

        layer_dir = storage_path.parent
        if layer_dir.exists() and not any(layer_dir.iterdir()):
            layer_dir.rmdir()
            print(f"  Deleted empty layer dir: {layer_dir}")

        checkpoint_dir_check = layer_dir.parent
        if checkpoint_dir_check.exists() and not any(checkpoint_dir_check.iterdir()):
            checkpoint_dir_check.rmdir()
            print(f"  Deleted empty checkpoint dir: {checkpoint_dir_check}")
        else:
            print(f"  Checkpoint dir still has contents: {list(checkpoint_dir_check.iterdir())}")

        print(f"\nAfter Test 2:")
        print(f"  Layer 14 file:  {layer14_file.exists()}")
        print(f"  Layer 14 dir:   {layer14_dir.exists()}")
        print(f"  Checkpoint dir: {checkpoint_dir.exists()}")

        assert not layer14_file.exists(), "Layer 14 file should be deleted"
        assert not layer14_dir.exists(), "Layer 14 directory should be deleted"
        assert checkpoint_dir.exists(), "Checkpoint directory should still exist"
        print("  ✓ Test 2 passed")

        # Test 3: Delete layer 18 (last file - should delete checkpoint dir)
        print("\n--- Test 3: Delete layer 18 (last file) ---")
        storage_path = layer18_file
        storage_path.unlink()
        print(f"  Deleted file: {storage_path}")

        layer_dir = storage_path.parent
        if layer_dir.exists() and not any(layer_dir.iterdir()):
            layer_dir.rmdir()
            print(f"  Deleted empty layer dir: {layer_dir}")

        checkpoint_dir_check = layer_dir.parent
        if checkpoint_dir_check.exists() and not any(checkpoint_dir_check.iterdir()):
            checkpoint_dir_check.rmdir()
            print(f"  Deleted empty checkpoint dir: {checkpoint_dir_check}")

        print(f"\nAfter Test 3:")
        print(f"  Layer 18 file:  {layer18_file.exists()}")
        print(f"  Layer 18 dir:   {layer18_dir.exists()}")
        print(f"  Checkpoint dir: {checkpoint_dir.exists()}")

        assert not layer18_file.exists(), "Layer 18 file should be deleted"
        assert not layer18_dir.exists(), "Layer 18 directory should be deleted"
        assert not checkpoint_dir.exists(), "Checkpoint directory should be deleted"
        print("  ✓ Test 3 passed")

    print("\n" + "=" * 70)
    print("✓ All tests passed! Cleanup logic works correctly.")
    print("=" * 70)


if __name__ == "__main__":
    test_directory_cleanup_logic()
