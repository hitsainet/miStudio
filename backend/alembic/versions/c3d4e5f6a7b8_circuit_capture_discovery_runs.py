"""circuit_capture_runs + circuit_discovery_runs (Feature 016).

Downgrade drops both tables (capture stores on disk are NOT removed —
they are prunable artifacts owned by the filesystem, not the schema).

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-07-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'circuit_capture_runs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('status', sa.String(16), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('manifest', JSONB(), nullable=False, server_default='{}'),
        sa.Column('store_path', sa.String(1000), nullable=True),
        sa.Column('events_total', sa.Integer(), nullable=True),
        sa.Column('bytes_total', sa.Integer(), nullable=True),
        sa.Column('stale', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('celery_task_id', sa.String(155), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.create_table(
        'circuit_discovery_runs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('capture_run_id', sa.String(36), nullable=False),
        sa.Column('status', sa.String(16), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('params', JSONB(), nullable=False, server_default='{}'),
        sa.Column('report', JSONB(), nullable=True),
        sa.Column('candidates', JSONB(), nullable=True),
        sa.Column('celery_task_id', sa.String(155), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('idx_circuit_discovery_capture', 'circuit_discovery_runs',
                    ['capture_run_id'])


def downgrade() -> None:
    op.drop_index('idx_circuit_discovery_capture', table_name='circuit_discovery_runs')
    op.drop_table('circuit_discovery_runs')
    op.drop_table('circuit_capture_runs')
