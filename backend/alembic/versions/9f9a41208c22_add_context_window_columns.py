"""add_context_window_columns

Revision ID: 9f9a41208c22
Revises: 126732716f92
Create Date: 2025-11-20 04:59:00.766832

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9f9a41208c22'
down_revision: Union[str, None] = '126732716f92'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add context window columns to extraction_templates, extraction_jobs, and feature_activations tables.

    Context window feature captures tokens before and after the prime token (max activation)
    based on research from Anthropic/OpenAI showing asymmetric windows improve interpretability.
    """
    # Add context window configuration columns to extraction_templates
    op.add_column('extraction_templates',
                  sa.Column('context_prefix_tokens', sa.Integer(), nullable=False, server_default='5'))
    op.add_column('extraction_templates',
                  sa.Column('context_suffix_tokens', sa.Integer(), nullable=False, server_default='3'))

    # Add context window configuration columns to extraction_jobs
    op.add_column('extraction_jobs',
                  sa.Column('context_prefix_tokens', sa.Integer(), nullable=False, server_default='5'))
    op.add_column('extraction_jobs',
                  sa.Column('context_suffix_tokens', sa.Integer(), nullable=False, server_default='3'))

    # Add context window data columns to feature_activations (partitioned table)
    # Note: For partitioned tables, we need to add columns to the parent table
    # and they will automatically be inherited by all partitions
    op.add_column('feature_activations',
                  sa.Column('prefix_tokens', sa.JSON(), nullable=True))
    op.add_column('feature_activations',
                  sa.Column('prime_token', sa.String(), nullable=True))
    op.add_column('feature_activations',
                  sa.Column('suffix_tokens', sa.JSON(), nullable=True))
    op.add_column('feature_activations',
                  sa.Column('prime_activation_index', sa.Integer(), nullable=True))


def downgrade() -> None:
    """
    Remove context window columns from extraction_templates, extraction_jobs, and feature_activations tables.
    """
    # Remove context window data columns from feature_activations
    op.drop_column('feature_activations', 'prime_activation_index')
    op.drop_column('feature_activations', 'suffix_tokens')
    op.drop_column('feature_activations', 'prime_token')
    op.drop_column('feature_activations', 'prefix_tokens')

    # Remove context window configuration columns from extraction_jobs
    op.drop_column('extraction_jobs', 'context_suffix_tokens')
    op.drop_column('extraction_jobs', 'context_prefix_tokens')

    # Remove context window configuration columns from extraction_templates
    op.drop_column('extraction_templates', 'context_suffix_tokens')
    op.drop_column('extraction_templates', 'context_prefix_tokens')
