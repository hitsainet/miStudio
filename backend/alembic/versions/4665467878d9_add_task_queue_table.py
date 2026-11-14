"""add_task_queue_table

Revision ID: 4665467878d9
Revises: 2d4f3b18cc10
Create Date: 2025-10-22 05:21:00.119079

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4665467878d9'
down_revision: Union[str, None] = '2d4f3b18cc10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create task_queue table
    op.create_table(
        'task_queue',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('task_id', sa.String(), nullable=True),
        sa.Column('task_type', sa.String(), nullable=False),
        sa.Column('entity_id', sa.String(), nullable=False),
        sa.Column('entity_type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_params', sa.JSON(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for common queries
    op.create_index('idx_task_queue_status', 'task_queue', ['status'])
    op.create_index('idx_task_queue_entity', 'task_queue', ['entity_type', 'entity_id'])
    op.create_index('idx_task_queue_task_id', 'task_queue', ['task_id'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_task_queue_task_id', table_name='task_queue')
    op.drop_index('idx_task_queue_entity', table_name='task_queue')
    op.drop_index('idx_task_queue_status', table_name='task_queue')

    # Drop table
    op.drop_table('task_queue')
