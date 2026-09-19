#!/usr/bin/env python3
"""
Checkpoint cleanup script for removing old SAE training checkpoints.

Usage:
    python scripts/cleanup_old_checkpoints.py <training_id> [--keep N] [--dry-run]

Examples:
    # Dry run - see what would be deleted
    python scripts/cleanup_old_checkpoints.py train_36766f85 --dry-run

    # Delete all but the most recent checkpoint
    python scripts/cleanup_old_checkpoints.py train_36766f85 --keep 1

    # Keep the 3 most recent checkpoints
    python scripts/cleanup_old_checkpoints.py train_36766f85 --keep 3
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from src.core.database import AsyncSessionLocal
from src.models.checkpoint import Checkpoint
from src.services.checkpoint_service import CheckpointService


async def cleanup_old_checkpoints(
    training_id: str,
    keep_latest: int = 1,
    dry_run: bool = False
):
    """
    Delete all but the N most recent checkpoints for a training.

    Args:
        training_id: Training ID to clean up
        keep_latest: Number of recent checkpoints to keep (default: 1)
        dry_run: If True, only show what would be deleted without deleting
    """
    async with AsyncSessionLocal() as db:
        # Get all checkpoints sorted by step (descending)
        result = await db.execute(
            select(Checkpoint)
            .where(Checkpoint.training_id == training_id)
            .order_by(Checkpoint.step.desc())
        )
        checkpoints = list(result.scalars().all())

        if not checkpoints:
            print(f"No checkpoints found for training: {training_id}")
            return

        total = len(checkpoints)
        to_keep = checkpoints[:keep_latest]
        to_delete = checkpoints[keep_latest:]

        print("=" * 70)
        print(f"Training ID: {training_id}")
        print(f"Total checkpoints: {total}")
        print("=" * 70)

        if keep_latest > total:
            print(f"\nNothing to delete (only {total} checkpoints exist)")
            return

        # Show checkpoints to keep
        print(f"\nCheckpoints to KEEP ({len(to_keep)}):")
        for ckpt in to_keep:
            file_size_mb = ckpt.file_size_bytes / (1024**2) if ckpt.file_size_bytes else 0
            print(f"  ✓ Step {ckpt.step:>7} | Loss: {ckpt.loss:.6f} | Size: {file_size_mb:>6.1f} MB | {ckpt.id}")

        # Calculate total size to delete
        total_size_bytes = sum(
            ckpt.file_size_bytes for ckpt in to_delete if ckpt.file_size_bytes
        )
        total_size_gb = total_size_bytes / (1024**3)

        print(f"\nCheckpoints to DELETE ({len(to_delete)}):")
        # Show first 5 and last 5
        display_list = to_delete[:5] + ([] if len(to_delete) <= 10 else to_delete[-5:])
        for ckpt in display_list[:5]:
            file_size_mb = ckpt.file_size_bytes / (1024**2) if ckpt.file_size_bytes else 0
            print(f"  ✗ Step {ckpt.step:>7} | Loss: {ckpt.loss:.6f} | Size: {file_size_mb:>6.1f} MB | {ckpt.id}")

        if len(to_delete) > 10:
            print(f"  ... ({len(to_delete) - 10} more checkpoints) ...")
            for ckpt in display_list[-5:]:
                file_size_mb = ckpt.file_size_bytes / (1024**2) if ckpt.file_size_bytes else 0
                print(f"  ✗ Step {ckpt.step:>7} | Loss: {ckpt.loss:.6f} | Size: {file_size_mb:>6.1f} MB | {ckpt.id}")

        print(f"\nTotal disk space to free: {total_size_gb:.2f} GB")

        if dry_run:
            print("\n[DRY RUN] No changes made.")
            return

        # Confirm deletion
        print("\n" + "=" * 70)
        response = input(f"Delete {len(to_delete)} checkpoints? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            print("Aborted.")
            return

        # Delete checkpoints
        print("\nDeleting checkpoints...")
        deleted_count = 0
        failed_count = 0

        for i, ckpt in enumerate(to_delete, 1):
            try:
                success = await CheckpointService.delete_checkpoint(
                    db,
                    ckpt.id,
                    delete_file=True
                )
                if success:
                    deleted_count += 1
                    print(f"  [{i}/{len(to_delete)}] Deleted: {ckpt.id} (step {ckpt.step})")
                else:
                    failed_count += 1
                    print(f"  [{i}/{len(to_delete)}] Failed: {ckpt.id} (not found)")
            except Exception as e:
                failed_count += 1
                print(f"  [{i}/{len(to_delete)}] Error deleting {ckpt.id}: {e}")

        print("\n" + "=" * 70)
        print(f"Completed:")
        print(f"  ✓ Deleted: {deleted_count}/{len(to_delete)} checkpoints")
        if failed_count > 0:
            print(f"  ✗ Failed:  {failed_count} checkpoints")
        print(f"  Freed: ~{total_size_gb:.2f} GB disk space")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old SAE training checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "training_id",
        help="Training ID to clean up (e.g., train_36766f85)"
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=1,
        help="Number of most recent checkpoints to keep (default: 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )

    args = parser.parse_args()

    if args.keep < 1:
        print("Error: --keep must be at least 1")
        sys.exit(1)

    asyncio.run(cleanup_old_checkpoints(
        training_id=args.training_id,
        keep_latest=args.keep,
        dry_run=args.dry_run
    ))


if __name__ == "__main__":
    main()
