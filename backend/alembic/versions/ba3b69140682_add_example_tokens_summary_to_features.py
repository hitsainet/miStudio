"""add_example_tokens_summary_to_features

Revision ID: ba3b69140682
Revises: aed3587bf63f
Create Date: 2025-11-18 10:40:06.527716

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'ba3b69140682'
down_revision: Union[str, None] = 'aed3587bf63f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add example_tokens_summary JSONB column to features table
    # This column stores the top 7 filtered tokens with their counts and activations
    # Format: {'tokens': ['token1', ...], 'counts': [10, ...], 'activations': [1.5, ...], 'max_activation': 2.0}
    op.add_column(
        'features',
        sa.Column('example_tokens_summary', postgresql.JSONB, nullable=True)
    )


def downgrade() -> None:
    # Remove example_tokens_summary column
    op.drop_column('features', 'example_tokens_summary')
