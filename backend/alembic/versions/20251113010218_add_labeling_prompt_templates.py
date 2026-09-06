"""add_labeling_prompt_templates

Revision ID: 20251113010218
Revises: 7d94bbc742c5
Create Date: 2025-11-13 01:02:18.000000

Adds labeling_prompt_templates table and prompt_template_id to labeling_jobs.
This enables customizable prompts and API parameters for semantic feature labeling.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20251113010218'
down_revision: Union[str, None] = '7d94bbc742c5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create labeling_prompt_templates table
    op.create_table(
        'labeling_prompt_templates',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),

        # Prompt content
        sa.Column('system_message', sa.Text(), nullable=False),
        sa.Column('user_prompt_template', sa.Text(), nullable=False),

        # API parameters
        sa.Column('temperature', sa.Float(), nullable=False, server_default='0.3'),
        sa.Column('max_tokens', sa.Integer(), nullable=False, server_default='50'),
        sa.Column('top_p', sa.Float(), nullable=False, server_default='0.9'),

        # Metadata
        sa.Column('is_default', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_system', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_by', sa.String(length=255), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),

        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for common queries
    op.create_index('idx_labeling_prompt_templates_is_default', 'labeling_prompt_templates', ['is_default'])
    op.create_index('idx_labeling_prompt_templates_name', 'labeling_prompt_templates', ['name'])
    op.create_index('idx_labeling_prompt_templates_created_at', 'labeling_prompt_templates', ['created_at'])

    # Add prompt_template_id column to labeling_jobs table
    op.add_column('labeling_jobs', sa.Column('prompt_template_id', sa.String(length=255), nullable=True))

    # Add foreign key constraint (ON DELETE RESTRICT to prevent deletion of in-use templates)
    op.create_foreign_key(
        'fk_labeling_jobs_prompt_template_id',
        'labeling_jobs',
        'labeling_prompt_templates',
        ['prompt_template_id'],
        ['id'],
        ondelete='RESTRICT'
    )

    # Create index on prompt_template_id for efficient lookups
    op.create_index('idx_labeling_jobs_prompt_template_id', 'labeling_jobs', ['prompt_template_id'])


def downgrade() -> None:
    # Drop index on labeling_jobs.prompt_template_id
    op.drop_index('idx_labeling_jobs_prompt_template_id', table_name='labeling_jobs')

    # Drop foreign key constraint
    op.drop_constraint('fk_labeling_jobs_prompt_template_id', 'labeling_jobs', type_='foreignkey')

    # Drop prompt_template_id column from labeling_jobs
    op.drop_column('labeling_jobs', 'prompt_template_id')

    # Drop indexes on labeling_prompt_templates
    op.drop_index('idx_labeling_prompt_templates_created_at', table_name='labeling_prompt_templates')
    op.drop_index('idx_labeling_prompt_templates_name', table_name='labeling_prompt_templates')
    op.drop_index('idx_labeling_prompt_templates_is_default', table_name='labeling_prompt_templates')

    # Drop labeling_prompt_templates table
    op.drop_table('labeling_prompt_templates')
