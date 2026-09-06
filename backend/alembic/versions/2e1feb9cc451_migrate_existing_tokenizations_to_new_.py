"""migrate_existing_tokenizations_to_new_table

Revision ID: 2e1feb9cc451
Revises: 04b58ed9486a
Create Date: 2025-11-08 18:56:01.367198

"""
from typing import Sequence, Union
from datetime import datetime

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = '2e1feb9cc451'
down_revision: Union[str, None] = '04b58ed9486a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Migrate existing tokenization data from datasets table to dataset_tokenizations table.

    For each dataset with tokenized_path:
    1. Extract tokenizer_name from metadata->tokenization->tokenizer_name
    2. Find corresponding model by repo_id
    3. Create tokenization record in dataset_tokenizations
    """
    # Get database connection
    conn = op.get_bind()

    # Query datasets with tokenization data
    query = text("""
        SELECT
            d.id as dataset_id,
            d.tokenized_path,
            d.vocab_size,
            d.num_tokens,
            d.avg_seq_length,
            d.metadata->>'tokenization' as tokenization_json,
            d.created_at
        FROM datasets d
        WHERE d.tokenized_path IS NOT NULL
          AND d.metadata ? 'tokenization'
    """)

    datasets = conn.execute(query).fetchall()

    for dataset in datasets:
        # Parse tokenization metadata
        import json
        tokenization = json.loads(dataset.tokenization_json)
        tokenizer_name = tokenization.get('tokenizer_name')

        if not tokenizer_name:
            print(f"Warning: Dataset {dataset.dataset_id} has tokenized_path but no tokenizer_name in metadata. Skipping.")
            continue

        # Find model by repo_id matching tokenizer_name
        model_query = text("""
            SELECT id
            FROM models
            WHERE repo_id = :tokenizer_name
            LIMIT 1
        """)

        model_result = conn.execute(model_query, {"tokenizer_name": tokenizer_name}).fetchone()

        if not model_result:
            print(f"Warning: No model found with repo_id='{tokenizer_name}' for dataset {dataset.dataset_id}. Skipping.")
            continue

        model_id = model_result[0]

        # Create tokenization ID
        tokenization_id = f"tok_{str(dataset.dataset_id).replace('-', '')}_{model_id}"

        # Insert tokenization record
        insert_query = text("""
            INSERT INTO dataset_tokenizations (
                id,
                dataset_id,
                model_id,
                tokenized_path,
                tokenizer_repo_id,
                vocab_size,
                num_tokens,
                avg_seq_length,
                status,
                progress,
                created_at,
                updated_at,
                completed_at
            ) VALUES (
                :id,
                :dataset_id,
                :model_id,
                :tokenized_path,
                :tokenizer_repo_id,
                :vocab_size,
                :num_tokens,
                :avg_seq_length,
                'READY',
                100.0,
                :created_at,
                :created_at,
                :created_at
            )
            ON CONFLICT (dataset_id, model_id) DO NOTHING
        """)

        conn.execute(insert_query, {
            "id": tokenization_id,
            "dataset_id": dataset.dataset_id,
            "model_id": model_id,
            "tokenized_path": dataset.tokenized_path,
            "tokenizer_repo_id": tokenizer_name,
            "vocab_size": dataset.vocab_size,
            "num_tokens": dataset.num_tokens,
            "avg_seq_length": dataset.avg_seq_length,
            "created_at": dataset.created_at,
        })

        print(f"Migrated tokenization for dataset {dataset.dataset_id} with model {model_id}")


def downgrade() -> None:
    """Refuses. MIS-E2E-039.

    This used to run:

        DELETE FROM dataset_tokenizations
        WHERE dataset_id IN (SELECT id FROM datasets WHERE tokenized_path IS NOT NULL)

    under a docstring promising it *"only removes records that were created by
    this migration"*. It cannot know that. Nothing marks a row as
    migration-created, and `tokenized_path` stays set on the dataset forever —
    so every tokenization a user has made since, for any of those datasets, is
    matched and deleted. The comment called this "safe … since the data still
    exists in the datasets table", but `datasets.tokenized_path` is one path,
    while `dataset_tokenizations` is one row per (dataset, model, max_length):
    the rows it deletes are not reconstructible from what it points at.

    A downgrade that cannot identify its own rows must not guess. Refusing is
    recoverable — the operator can decide what to remove and do it deliberately
    — whereas a wrong DELETE is not.
    """
    raise RuntimeError(
        "Refusing to downgrade 2e1feb9cc451: this migration left no marker "
        "distinguishing the tokenization rows it created from ones users have "
        "made since, and the only available filter (datasets.tokenized_path IS "
        "NOT NULL) matches both. Delete the rows you actually want by hand, "
        "then stamp this revision down with `alembic stamp`."
    )
