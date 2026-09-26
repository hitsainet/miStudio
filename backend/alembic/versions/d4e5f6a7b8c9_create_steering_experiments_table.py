"""Create steering_experiments table

Revision ID: d4e5f6a7b8c9
Revises: 9c0d1e2f3a4b
Create Date: 2026-01-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = '9c0d1e2f3a4b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create steering_experiments table."""
    op.create_table(
        'steering_experiments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('sae_id', sa.String(255), nullable=False, index=True),
        sa.Column('model_id', sa.String(255), nullable=False, index=True),
        sa.Column('comparison_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('selected_features', postgresql.JSONB(), nullable=False, default=[]),
        sa.Column('generation_params', postgresql.JSONB(), nullable=False, default={}),
        sa.Column('results', postgresql.JSONB(), nullable=False),
        sa.Column('tags', postgresql.JSONB(), nullable=False, default=[]),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Create additional indexes for search
    op.create_index(
        'ix_steering_experiments_created_at',
        'steering_experiments',
        ['created_at'],
        postgresql_using='btree'
    )


def downgrade() -> None:
    """Drop steering_experiments table."""
    op.drop_index('ix_steering_experiments_created_at', 'steering_experiments')
    op.drop_table('steering_experiments')
