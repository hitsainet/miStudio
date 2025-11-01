#!/usr/bin/env python3
"""
Orphan Data Cleanup Script

Safely removes orphaned training and dataset directories that exist on disk
but are not tracked in the database.

Usage:
    python cleanup_orphans.py --dry-run          # See what will be deleted
    python cleanup_orphans.py --delete-all       # Delete all orphans
    python cleanup_orphans.py --trainings-only   # Delete orphaned trainings
    python cleanup_orphans.py --datasets-only    # Delete orphaned datasets
    python cleanup_orphans.py --interactive      # Ask before each deletion
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Set, List, Tuple
import psycopg2

# Database connection
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "mistudio"
DB_USER = "postgres"
DB_PASSWORD = "devpassword"

# Data directories
DATA_DIR = Path("/home/x-sean/app/miStudio/backend/data")
TRAININGS_DIR = DATA_DIR / "trainings"
DATASETS_DIR = DATA_DIR / "datasets"

# Log file
LOG_FILE = DATA_DIR / f"orphan_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log(message: str):
    """Log message to file and print to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, "a") as f:
        f.write(log_message + "\n")


def get_db_training_ids() -> Set[str]:
    """Get all training IDs from database."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()
    cur.execute("SELECT id FROM trainings;")
    ids = {row[0] for row in cur.fetchall()}
    cur.close()
    conn.close()
    return ids


def get_db_dataset_ids() -> Set[str]:
    """Get all dataset repo_ids and names from database."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()
    # Get both hf_repo_id and name fields
    cur.execute("SELECT hf_repo_id, name FROM datasets;")
    ids = set()
    for row in cur.fetchall():
        repo_id = row[0]
        name = row[1]
        if repo_id:
            # Convert repo_id to directory name format (replace / with _)
            ids.add(repo_id.replace("/", "_"))
        if name:
            ids.add(name)
    cur.close()
    conn.close()
    return ids


def get_disk_training_ids() -> List[str]:
    """Get all training directories from disk."""
    if not TRAININGS_DIR.exists():
        return []
    return [d.name for d in TRAININGS_DIR.iterdir() if d.is_dir()]


def get_disk_dataset_ids() -> List[str]:
    """Get all dataset directories from disk."""
    if not DATASETS_DIR.exists():
        return []
    return [d.name for d in DATASETS_DIR.iterdir() if d.is_dir()]


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        log(f"Error calculating size of {path}: {e}")
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def find_orphaned_trainings() -> List[Tuple[str, Path, int]]:
    """Find orphaned training directories."""
    db_ids = get_db_training_ids()
    disk_ids = get_disk_training_ids()

    orphans = []
    for disk_id in disk_ids:
        if disk_id not in db_ids:
            path = TRAININGS_DIR / disk_id
            size = get_dir_size(path)
            orphans.append((disk_id, path, size))

    return orphans


def find_orphaned_datasets() -> List[Tuple[str, Path, int]]:
    """Find orphaned dataset directories."""
    db_ids = get_db_dataset_ids()
    disk_ids = get_disk_dataset_ids()

    orphans = []
    for disk_id in disk_ids:
        if disk_id not in db_ids:
            path = DATASETS_DIR / disk_id
            size = get_dir_size(path)
            orphans.append((disk_id, path, size))

    return orphans


def delete_directory(path: Path, dry_run: bool = False, interactive: bool = False) -> bool:
    """Delete directory with safety checks."""
    if not path.exists():
        log(f"SKIP: Path does not exist: {path}")
        return False

    if interactive:
        response = input(f"Delete {path}? [y/N]: ").strip().lower()
        if response != 'y':
            log(f"SKIP: User declined deletion of {path}")
            return False

    if dry_run:
        log(f"DRY-RUN: Would delete {path}")
        return False

    try:
        shutil.rmtree(path)
        log(f"DELETED: {path}")
        return True
    except Exception as e:
        log(f"ERROR: Failed to delete {path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Clean up orphaned data directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--trainings-only", action="store_true", help="Only clean orphaned trainings")
    parser.add_argument("--datasets-only", action="store_true", help="Only clean orphaned datasets")
    parser.add_argument("--delete-all", action="store_true", help="Delete all orphans (trainings + datasets)")
    parser.add_argument("--interactive", action="store_true", help="Ask before each deletion")
    args = parser.parse_args()

    if not any([args.dry_run, args.trainings_only, args.datasets_only, args.delete_all, args.interactive]):
        parser.print_help()
        print("\nError: Must specify an action (--dry-run, --delete-all, etc.)")
        return

    log("=" * 80)
    log("Orphan Data Cleanup Script")
    log("=" * 80)

    if args.dry_run:
        log("MODE: DRY RUN (no files will be deleted)")
    elif args.interactive:
        log("MODE: INTERACTIVE (will ask before each deletion)")
    else:
        log("MODE: DELETE")

    # Find orphaned trainings
    training_orphans = []
    if not args.datasets_only:
        log("\nScanning for orphaned trainings...")
        training_orphans = find_orphaned_trainings()
        log(f"Found {len(training_orphans)} orphaned training directories")

        total_training_size = sum(size for _, _, size in training_orphans)
        log(f"Total orphaned training size: {format_size(total_training_size)}")

        if training_orphans:
            log("\nOrphaned Trainings:")
            for train_id, path, size in training_orphans[:10]:  # Show first 10
                log(f"  - {train_id} ({format_size(size)})")
            if len(training_orphans) > 10:
                log(f"  ... and {len(training_orphans) - 10} more")

    # Find orphaned datasets
    dataset_orphans = []
    if not args.trainings_only:
        log("\nScanning for orphaned datasets...")
        dataset_orphans = find_orphaned_datasets()
        log(f"Found {len(dataset_orphans)} orphaned dataset directories")

        total_dataset_size = sum(size for _, _, size in dataset_orphans)
        log(f"Total orphaned dataset size: {format_size(total_dataset_size)}")

        if dataset_orphans:
            log("\nOrphaned Datasets:")
            for dataset_id, path, size in dataset_orphans[:10]:  # Show first 10
                log(f"  - {dataset_id} ({format_size(size)})")
            if len(dataset_orphans) > 10:
                log(f"  ... and {len(dataset_orphans) - 10} more")

    # Summary
    total_orphans = len(training_orphans) + len(dataset_orphans)
    total_size = sum(size for _, _, size in training_orphans) + sum(size for _, _, size in dataset_orphans)

    log("\n" + "=" * 80)
    log(f"SUMMARY: {total_orphans} orphaned directories, {format_size(total_size)} total")
    log("=" * 80)

    if args.dry_run:
        log("\nDRY RUN COMPLETE - No files were deleted")
        log(f"Log file: {LOG_FILE}")
        return

    if not args.delete_all and not args.interactive:
        log("\nTo delete these orphans, re-run with --delete-all or --interactive")
        log(f"Log file: {LOG_FILE}")
        return

    # Confirm deletion
    if not args.interactive:
        print(f"\n⚠️  WARNING: About to delete {total_orphans} directories ({format_size(total_size)})")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            log("ABORTED: User cancelled deletion")
            return

    # Delete orphans
    log("\nDeleting orphans...")
    deleted_count = 0
    deleted_size = 0

    for train_id, path, size in training_orphans:
        if delete_directory(path, dry_run=False, interactive=args.interactive):
            deleted_count += 1
            deleted_size += size

    for dataset_id, path, size in dataset_orphans:
        if delete_directory(path, dry_run=False, interactive=args.interactive):
            deleted_count += 1
            deleted_size += size

    log("\n" + "=" * 80)
    log(f"CLEANUP COMPLETE: Deleted {deleted_count}/{total_orphans} directories")
    log(f"Space reclaimed: {format_size(deleted_size)}")
    log(f"Log file: {LOG_FILE}")
    log("=" * 80)


if __name__ == "__main__":
    main()
