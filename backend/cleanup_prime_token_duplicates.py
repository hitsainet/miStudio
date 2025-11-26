#!/usr/bin/env python3
"""
Standalone script to clean up duplicate prime_tokens in feature_activations table.

Layer 3 of the 3-layer fix for the empty <<>> markers bug.

This script can be run independently of alembic migrations:
    python cleanup_prime_token_duplicates.py [--dry-run]

Options:
    --dry-run    Show what would be fixed without making changes
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text
from src.core.config import settings


def cleanup_duplicates(dry_run: bool = False):
    """
    Clean up duplicate prime_tokens from prefix_tokens and suffix_tokens.

    Args:
        dry_run: If True, only report what would be fixed without making changes
    """
    # Connect to database
    engine = create_engine(settings.database_url)

    with engine.connect() as connection:
        # Count total records
        count_result = connection.execute(text(
            "SELECT COUNT(*) FROM feature_activations WHERE prime_token IS NOT NULL"
        ))
        total_count = count_result.scalar()

        print(f"[Cleanup Script] {'DRY RUN - ' if dry_run else ''}Scanning {total_count} records...")

        # Process in batches
        batch_size = 1000
        fixed_prefix = 0
        fixed_suffix = 0
        processed = 0

        while True:
            result = connection.execute(text("""
                SELECT id, prime_token, prefix_tokens, suffix_tokens
                FROM feature_activations
                WHERE prime_token IS NOT NULL
                ORDER BY id
                LIMIT :batch_size OFFSET :offset
            """), {"batch_size": batch_size, "offset": processed})

            rows = result.fetchall()
            if not rows:
                break

            for row in rows:
                record_id = row[0]
                prime_token = row[1]
                prefix_tokens = row[2]
                suffix_tokens = row[3]

                needs_update = False
                new_prefix = prefix_tokens
                new_suffix = suffix_tokens

                # Check prefix_tokens
                if prefix_tokens and prime_token in prefix_tokens:
                    new_prefix = [t for t in prefix_tokens if t != prime_token]
                    fixed_prefix += 1
                    needs_update = True
                    print(f"  {'[DRY RUN] Would fix' if dry_run else 'Fixed'} prefix duplicate in record {record_id}")

                # Check suffix_tokens
                if suffix_tokens and prime_token in suffix_tokens:
                    new_suffix = [t for t in suffix_tokens if t != prime_token]
                    fixed_suffix += 1
                    needs_update = True
                    print(f"  {'[DRY RUN] Would fix' if dry_run else 'Fixed'} suffix duplicate in record {record_id}")

                # Update if needed (and not dry run)
                if needs_update and not dry_run:
                    connection.execute(text("""
                        UPDATE feature_activations
                        SET prefix_tokens = :prefix_tokens,
                            suffix_tokens = :suffix_tokens
                        WHERE id = :id
                    """), {
                        "id": record_id,
                        "prefix_tokens": json.dumps(new_prefix) if new_prefix else None,
                        "suffix_tokens": json.dumps(new_suffix) if new_suffix else None
                    })

            processed += len(rows)
            if processed % 10000 == 0:
                print(f"  Processed {processed}/{total_count} records...")

        # Commit changes if not dry run
        if not dry_run:
            connection.commit()

        print(f"\n[Cleanup Script] {'DRY RUN ' if dry_run else ''}Complete!")
        print(f"  Records scanned: {processed}")
        print(f"  Prefix duplicates {'would be ' if dry_run else ''}fixed: {fixed_prefix}")
        print(f"  Suffix duplicates {'would be ' if dry_run else ''}fixed: {fixed_suffix}")

        if fixed_prefix + fixed_suffix > 0:
            if dry_run:
                print(f"\n  ⚠️  Would fix {fixed_prefix + fixed_suffix} records. Run without --dry-run to apply changes.")
            else:
                print(f"\n  ✅ Fixed {fixed_prefix + fixed_suffix} records!")
        else:
            print(f"\n  ✅ No duplicates found - data is clean!")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up duplicate prime_tokens in feature_activations table"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes"
    )
    args = parser.parse_args()

    cleanup_duplicates(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
