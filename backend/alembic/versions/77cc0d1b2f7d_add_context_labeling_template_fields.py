"""add_context_labeling_template_fields

Revision ID: 77cc0d1b2f7d
Revises: 9f9a41208c22
Create Date: 2025-11-21 23:07:56.281043

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '77cc0d1b2f7d'
down_revision: Union[str, None] = '9f9a41208c22'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add context-based labeling columns to labeling_prompt_templates table.

    These fields support new prompt templates (miStudio Internal, Anthropic, EleutherAI)
    that use full-context activation examples instead of aggregated token statistics.
    """
    # Template type and configuration
    op.add_column('labeling_prompt_templates',
                  sa.Column('template_type', sa.String(50), nullable=False, server_default='legacy'))
    op.add_column('labeling_prompt_templates',
                  sa.Column('max_examples', sa.Integer(), nullable=False, server_default='10'))

    # Context window configuration
    op.add_column('labeling_prompt_templates',
                  sa.Column('include_prefix', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('labeling_prompt_templates',
                  sa.Column('include_suffix', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('labeling_prompt_templates',
                  sa.Column('prime_token_marker', sa.String(20), nullable=False, server_default='<<>>'))

    # Logit effects configuration (for Anthropic-style template)
    op.add_column('labeling_prompt_templates',
                  sa.Column('include_logit_effects', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('labeling_prompt_templates',
                  sa.Column('top_promoted_tokens_count', sa.Integer(), nullable=True))
    op.add_column('labeling_prompt_templates',
                  sa.Column('top_suppressed_tokens_count', sa.Integer(), nullable=True))

    # Detection/scoring template flag (for EleutherAI-style template)
    op.add_column('labeling_prompt_templates',
                  sa.Column('is_detection_template', sa.Boolean(), nullable=False, server_default='false'))


def downgrade() -> None:
    """
    Remove context-based labeling columns from labeling_prompt_templates table.
    """
    # Remove in reverse order
    op.drop_column('labeling_prompt_templates', 'is_detection_template')
    op.drop_column('labeling_prompt_templates', 'top_suppressed_tokens_count')
    op.drop_column('labeling_prompt_templates', 'top_promoted_tokens_count')
    op.drop_column('labeling_prompt_templates', 'include_logit_effects')
    op.drop_column('labeling_prompt_templates', 'prime_token_marker')
    op.drop_column('labeling_prompt_templates', 'include_suffix')
    op.drop_column('labeling_prompt_templates', 'include_prefix')
    op.drop_column('labeling_prompt_templates', 'max_examples')
    op.drop_column('labeling_prompt_templates', 'template_type')
