"""create_prompt_templates_table

Revision ID: 7aa4e03d0d11
Revises: 79dd2387ebeb
Create Date: 2025-11-30 14:04:02.389379

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '7aa4e03d0d11'
down_revision: Union[str, None] = '79dd2387ebeb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create prompt_templates table
    op.create_table(
        'prompt_templates',
        sa.Column('id', sa.UUID(), nullable=False, comment='Unique template identifier'),
        sa.Column('name', sa.String(length=255), nullable=False, comment='Template name'),
        sa.Column('description', sa.Text(), nullable=True, comment='Template description'),

        # Prompt content - array of prompt strings stored as JSONB
        sa.Column('prompts', postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment='Array of prompt strings'),

        # Template metadata
        sa.Column('is_favorite', sa.Boolean(), server_default=sa.text('false'), nullable=False, comment='Whether this template is marked as favorite'),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='Array of tag strings for organization'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Record creation timestamp'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Record last update timestamp'),

        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes for efficient queries
    op.create_index('idx_prompt_templates_name', 'prompt_templates', ['name'])
    op.create_index('idx_prompt_templates_is_favorite', 'prompt_templates', ['is_favorite'])
    op.create_index('idx_prompt_templates_created_at', 'prompt_templates', ['created_at'])

    # GIN index for JSONB tags queries
    op.create_index(
        'idx_prompt_templates_tags_gin',
        'prompt_templates',
        ['tags'],
        postgresql_using='gin'
    )

    # GIN index for JSONB prompts queries (in case of content search)
    op.create_index(
        'idx_prompt_templates_prompts_gin',
        'prompt_templates',
        ['prompts'],
        postgresql_using='gin'
    )

    # Create trigger to auto-update updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_prompt_templates_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER trigger_update_prompt_templates_updated_at
        BEFORE UPDATE ON prompt_templates
        FOR EACH ROW
        EXECUTE FUNCTION update_prompt_templates_updated_at();
    """)


def downgrade() -> None:
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS trigger_update_prompt_templates_updated_at ON prompt_templates;")
    op.execute("DROP FUNCTION IF EXISTS update_prompt_templates_updated_at();")

    # Drop indexes
    op.drop_index('idx_prompt_templates_prompts_gin', table_name='prompt_templates', postgresql_using='gin')
    op.drop_index('idx_prompt_templates_tags_gin', table_name='prompt_templates', postgresql_using='gin')
    op.drop_index('idx_prompt_templates_created_at', table_name='prompt_templates')
    op.drop_index('idx_prompt_templates_is_favorite', table_name='prompt_templates')
    op.drop_index('idx_prompt_templates_name', table_name='prompt_templates')

    # Drop table
    op.drop_table('prompt_templates')
