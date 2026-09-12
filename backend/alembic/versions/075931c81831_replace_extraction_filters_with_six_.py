"""replace_extraction_filters_with_six_boolean_flags

Revision ID: 075931c81831
Revises: ba3b69140682
Create Date: 2025-11-18 11:24:22.056503

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '075931c81831'
down_revision: Union[str, None] = 'ba3b69140682'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Remove old extraction filter columns
    op.drop_column('extraction_jobs', 'extraction_filter_mode')
    op.drop_column('extraction_jobs', 'extraction_filter_enabled')

    # Add 6 new boolean filter columns matching labeling filter structure
    # These filters control which tokens are stored in FeatureActivation records during extraction
    op.add_column('extraction_jobs',
        sa.Column('filter_special', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('extraction_jobs',
        sa.Column('filter_single_char', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('extraction_jobs',
        sa.Column('filter_punctuation', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('extraction_jobs',
        sa.Column('filter_numbers', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('extraction_jobs',
        sa.Column('filter_fragments', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('extraction_jobs',
        sa.Column('filter_stop_words', sa.Boolean(), nullable=False, server_default='false'))


def downgrade() -> None:
    # Remove new filter columns
    op.drop_column('extraction_jobs', 'filter_stop_words')
    op.drop_column('extraction_jobs', 'filter_fragments')
    op.drop_column('extraction_jobs', 'filter_numbers')
    op.drop_column('extraction_jobs', 'filter_punctuation')
    op.drop_column('extraction_jobs', 'filter_single_char')
    op.drop_column('extraction_jobs', 'filter_special')

    # Restore old extraction filter columns
    op.add_column('extraction_jobs',
        sa.Column('extraction_filter_enabled', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('extraction_jobs',
        sa.Column('extraction_filter_mode', sa.String(20), nullable=False, server_default='standard'))
