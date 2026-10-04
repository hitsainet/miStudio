#!/usr/bin/env python3
"""
Recovery script to fix datasets stuck in PROCESSING status.

This script identifies datasets that have completed tokenization (files exist on disk)
but failed to update the database due to task termination or cleanup errors.

Usage:
    python -m src.scripts.fix_stuck_dataset --dataset-id <uuid>
    python -m src.scripts.fix_stuck_dataset --dataset-name <name>
    python -m src.scripts.fix_stuck_dataset --all
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.database import SyncSessionLocal
from src.models.dataset import Dataset, DatasetStatus
from src.services.tokenization_service import TokenizationService
from src.workers.websocket_emitter import emit_dataset_progress
from src.core.config import settings


def find_tokenized_path(dataset: Dataset) -> Optional[Path]:
    """
    Find the tokenized dataset path by checking common patterns.

    Args:
        dataset: Dataset model instance

    Returns:
        Path to tokenized dataset if found, None otherwise
    """
    if not dataset.raw_path:
        logger.warning(f"Dataset {dataset.id} has no raw_path")
        return None

    raw_path = Path(dataset.raw_path)

    # Pattern 1: Direct tokenized_path (if already set)
    if dataset.tokenized_path:
        tokenized_path = Path(dataset.tokenized_path)
        if not tokenized_path.is_absolute():
            tokenized_path = settings.data_dir / tokenized_path
        if tokenized_path.exists():
            logger.info(f"Found tokenized dataset at: {tokenized_path}")
            return tokenized_path

    # Pattern 2: HF cache format - vietgpt___openwebtext_en
    # The raw path might be: data/datasets/vietgpt___openwebtext_en
    # Check if it exists and has nested structure
    if not raw_path.is_absolute():
        raw_path = settings.data_dir / raw_path.relative_to(Path("data")) if str(raw_path).startswith("data/") else settings.data_dir / raw_path

    logger.info(f"Checking raw_path: {raw_path}")

    if raw_path.exists():
        # Check for HF nested cache: default/0.0.0/hash/
        nested_pattern = list(raw_path.glob("*/*/*/dataset_info.json"))
        if nested_pattern:
            logger.info(f"Found HF nested cache structure in raw_path: {raw_path}")
            return raw_path

        # Check for save_to_disk format
        if (raw_path / "dataset_info.json").exists():
            logger.info(f"Found save_to_disk format at raw_path: {raw_path}")
            return raw_path

    # Pattern 3: Common tokenized suffixes
    potential_paths = [
        raw_path.parent / f"{raw_path.name}_tokenized",
        raw_path.parent / f"{raw_path.name}-tokenized",
        raw_path.with_name(f"{raw_path.stem}_tokenized{raw_path.suffix}"),
    ]

    for path in potential_paths:
        if path.exists():
            logger.info(f"Found tokenized dataset at: {path}")
            return path

    logger.warning(f"Could not find tokenized dataset for {dataset.name}")
    return None


def fix_dataset(db: Session, dataset: Dataset, dry_run: bool = False) -> bool:
    """
    Fix a single dataset stuck in PROCESSING status.

    Args:
        db: Database session
        dataset: Dataset to fix
        dry_run: If True, only report what would be done without making changes

    Returns:
        True if dataset was fixed successfully, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing dataset: {dataset.name} (ID: {dataset.id})")
    logger.info(f"Current status: {dataset.status}, Progress: {dataset.progress}%")
    logger.info(f"{'='*80}")

    # Find tokenized dataset on disk
    tokenized_path = find_tokenized_path(dataset)

    if not tokenized_path:
        logger.error(f"Cannot fix dataset {dataset.name}: Tokenized data not found on disk")
        return False

    try:
        # Load tokenized dataset from disk (memory-efficient)
        logger.info(f"Loading tokenized dataset from: {tokenized_path}")
        tokenized_dataset = TokenizationService.load_dataset_from_disk(str(tokenized_path))
        logger.info(f"Successfully loaded dataset with {len(tokenized_dataset)} samples")

        # Calculate statistics
        logger.info("Calculating statistics (this may take a few minutes for large datasets)...")
        stats = TokenizationService.calculate_statistics(
            tokenized_dataset,
            progress_callback=lambda pct: logger.info(f"  Statistics progress: {pct:.1f}%") if int(pct) % 10 == 0 else None
        )

        logger.info(f"\nCalculated statistics:")
        logger.info(f"  - Samples: {stats['num_samples']:,}")
        logger.info(f"  - Tokens: {stats['num_tokens']:,}")
        logger.info(f"  - Avg sequence length: {stats['avg_seq_length']:.2f}")
        logger.info(f"  - Vocab size: {stats['vocab_size']:,}")
        logger.info(f"  - Min/Max length: {stats['min_seq_length']} / {stats['max_seq_length']}")
        logger.info(f"  - Median length: {stats['median_seq_length']:.2f}")

        if dry_run:
            logger.info("\n[DRY RUN] Would update database with:")
            logger.info(f"  - status: READY")
            logger.info(f"  - progress: 100.0")
            logger.info(f"  - tokenized_path: {tokenized_path}")
            logger.info(f"  - num_samples: {stats['num_samples']}")
            logger.info(f"  - num_tokens: {stats['num_tokens']}")
            logger.info(f"  - avg_seq_length: {stats['avg_seq_length']}")
            logger.info(f"  - vocab_size: {stats['vocab_size']}")
            logger.info(f"\n[DRY RUN] Would emit WebSocket completion event")
            return True

        # Update database
        logger.info("\nUpdating database...")
        dataset.status = DatasetStatus.READY
        dataset.progress = 100.0
        dataset.tokenized_path = str(tokenized_path)
        dataset.num_samples = stats['num_samples']
        dataset.num_tokens = stats['num_tokens']
        dataset.avg_seq_length = stats['avg_seq_length']
        dataset.vocab_size = stats['vocab_size']

        # Update metadata with statistics
        if not dataset.extra_metadata:
            dataset.extra_metadata = {}

        dataset.extra_metadata.update({
            'statistics': stats,
            'recovery_info': {
                'recovered_at': str(dataset.updated_at),
                'recovery_script': 'fix_stuck_dataset.py',
                'reason': 'Dataset completed tokenization but failed to update database'
            }
        })

        db.commit()
        db.refresh(dataset)
        logger.info("✅ Database updated successfully")

        # Emit WebSocket completion event
        logger.info("Emitting WebSocket completion event...")
        try:
            emit_dataset_progress(
                str(dataset.id),
                "completed",
                {
                    'dataset_id': str(dataset.id),
                    'name': dataset.name,
                    'status': 'ready',
                    'progress': 100.0,
                    'num_samples': stats['num_samples'],
                    'num_tokens': stats['num_tokens'],
                    'message': 'Dataset recovery completed'
                }
            )
            logger.info("✅ WebSocket event emitted")
        except Exception as ws_error:
            logger.warning(f"Failed to emit WebSocket event (non-critical): {ws_error}")

        logger.info(f"\n✅ Successfully fixed dataset {dataset.name}")
        logger.info(f"{'='*80}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to fix dataset {dataset.name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main entry point for the recovery script."""
    parser = argparse.ArgumentParser(
        description="Fix datasets stuck in PROCESSING status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix a specific dataset by ID
  python -m src.scripts.fix_stuck_dataset --dataset-id bd7f3c7d-6071-4894-9057-68d1e5244564

  # Fix a specific dataset by name
  python -m src.scripts.fix_stuck_dataset --dataset-name openwebtext_en

  # Fix all stuck datasets
  python -m src.scripts.fix_stuck_dataset --all

  # Dry run (show what would be done without making changes)
  python -m src.scripts.fix_stuck_dataset --all --dry-run
        """
    )

    parser.add_argument(
        '--dataset-id',
        type=str,
        help='UUID of the dataset to fix'
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        help='Name of the dataset to fix'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Fix all datasets stuck in PROCESSING status'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.dataset_id, args.dataset_name, args.all]):
        parser.error("Must specify --dataset-id, --dataset-name, or --all")

    if sum([bool(args.dataset_id), bool(args.dataset_name), args.all]) > 1:
        parser.error("Can only specify one of --dataset-id, --dataset-name, or --all")

    # Create database session
    db = SyncSessionLocal()

    try:
        # Find datasets to fix
        if args.dataset_id:
            datasets = db.query(Dataset).filter(Dataset.id == args.dataset_id).all()
            if not datasets:
                logger.error(f"Dataset with ID {args.dataset_id} not found")
                return 1
        elif args.dataset_name:
            datasets = db.query(Dataset).filter(Dataset.name == args.dataset_name).all()
            if not datasets:
                logger.error(f"Dataset with name '{args.dataset_name}' not found")
                return 1
        else:  # --all
            datasets = db.query(Dataset).filter(Dataset.status == DatasetStatus.PROCESSING).all()
            if not datasets:
                logger.info("No datasets found stuck in PROCESSING status")
                return 0

        logger.info(f"\nFound {len(datasets)} dataset(s) to process")
        if args.dry_run:
            logger.info("[DRY RUN MODE - No changes will be made]\n")

        # Fix each dataset
        success_count = 0
        failure_count = 0

        for dataset in datasets:
            if fix_dataset(db, dataset, dry_run=args.dry_run):
                success_count += 1
            else:
                failure_count += 1

        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total datasets processed: {len(datasets)}")
        logger.info(f"Successfully fixed: {success_count}")
        logger.info(f"Failed: {failure_count}")

        if args.dry_run:
            logger.info("\n[DRY RUN] No changes were made. Run without --dry-run to apply fixes.")

        return 0 if failure_count == 0 else 1

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
