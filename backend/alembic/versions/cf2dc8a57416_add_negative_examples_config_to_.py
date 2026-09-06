"""add_negative_examples_config_to_labeling_templates

Revision ID: cf2dc8a57416
Revises: 9dc725cba2ad
Create Date: 2025-11-22 12:40:53.507328

Adds negative examples configuration to labeling_prompt_templates:
- include_negative_examples: BOOLEAN NOT NULL DEFAULT TRUE
- num_negative_examples: INTEGER DEFAULT NULL (default: 5)

These fields enable contrastive learning by including low-activation examples
alongside high-activation examples for more precise feature labeling.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cf2dc8a57416'
down_revision: Union[str, None] = '9dc725cba2ad'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add negative examples configuration fields to labeling_prompt_templates table.

    Adds:
    - include_negative_examples: Boolean flag to enable/disable negative examples (default: TRUE)
    - num_negative_examples: Integer count for number of negative examples to retrieve (default: NULL, interpreted as 5)
    """
    # Add include_negative_examples column (NOT NULL with default TRUE)
    op.add_column(
        'labeling_prompt_templates',
        sa.Column('include_negative_examples', sa.Boolean(), nullable=False, server_default='true')
    )

    # Add num_negative_examples column (nullable, default NULL interpreted as 5)
    op.add_column(
        'labeling_prompt_templates',
        sa.Column('num_negative_examples', sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    """
    Remove negative examples configuration fields from labeling_prompt_templates table.
    """
    op.drop_column('labeling_prompt_templates', 'num_negative_examples')
    op.drop_column('labeling_prompt_templates', 'include_negative_examples')
