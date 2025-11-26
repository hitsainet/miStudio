"""cleanup_duplicate_prime_tokens_in_feature_activations

Revision ID: 678f7e8bbeb6
Revises: 92326e3cf5a7
Create Date: 2025-11-25 04:43:32

This migration cleans up corrupt data where prime_token appears duplicated
in prefix_tokens or suffix_tokens arrays. This data corruption can cause
empty <<>> markers in the labeling context formatter.

Layer 3 of the 3-layer fix for the empty markers bug.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
import json
import logging

# revision identifiers, used by Alembic.
revision = '678f7e8bbeb6'
down_revision = '92326e3cf5a7'
branch_labels = None
depends_on = None

logger = logging.getLogger(__name__)


def upgrade() -> None:
    """
    Clean up duplicate prime_tokens from prefix_tokens and suffix_tokens.

    This fixes data corruption where the prime_token (the token with max activation)
    was incorrectly included in the prefix_tokens or suffix_tokens arrays.
    """
    connection = op.get_bind()

    # Count total records to process
    count_result = connection.execute(text(
        "SELECT COUNT(*) FROM feature_activations WHERE prime_token IS NOT NULL"
    ))
    total_count = count_result.scalar()

    print(f"[Layer 3 Migration] Scanning {total_count} feature_activation records for duplicates...")

    # Process in batches to avoid memory issues
    batch_size = 1000
    fixed_prefix = 0
    fixed_suffix = 0
    processed = 0

    while True:
        # Fetch batch of records with non-null prime_token
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
            prefix_tokens = row[2]  # This is already a Python list from JSON
            suffix_tokens = row[3]  # This is already a Python list from JSON

            needs_update = False
            new_prefix = prefix_tokens
            new_suffix = suffix_tokens

            # Check if prime_token is in prefix_tokens
            if prefix_tokens and prime_token in prefix_tokens:
                new_prefix = [t for t in prefix_tokens if t != prime_token]
                fixed_prefix += 1
                needs_update = True
                print(f"  Fixed duplicate in prefix for record {record_id}: removed '{prime_token}'")

            # Check if prime_token is in suffix_tokens
            if suffix_tokens and prime_token in suffix_tokens:
                new_suffix = [t for t in suffix_tokens if t != prime_token]
                fixed_suffix += 1
                needs_update = True
                print(f"  Fixed duplicate in suffix for record {record_id}: removed '{prime_token}'")

            # Update record if needed
            if needs_update:
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

    print(f"[Layer 3 Migration] Complete!")
    print(f"  Total records scanned: {processed}")
    print(f"  Prefix duplicates fixed: {fixed_prefix}")
    print(f"  Suffix duplicates fixed: {fixed_suffix}")

    if fixed_prefix + fixed_suffix > 0:
        print(f"  ⚠️  Found and fixed {fixed_prefix + fixed_suffix} records with duplicate prime_tokens!")
    else:
        print(f"  ✅ No duplicates found - data is clean!")


def downgrade() -> None:
    """
    Downgrade is not possible for data cleanup migrations.
    The original corrupt data cannot be reconstructed.
    """
    print("[Layer 3 Migration] Downgrade not possible - data cleanup cannot be reversed")
    pass
