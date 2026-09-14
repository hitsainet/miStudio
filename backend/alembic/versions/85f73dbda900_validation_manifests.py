"""validation_manifests (Feature 017).

Self-contained reproducible validation records (edge_batch | faithfulness |
reproduction). Downgrade drops the table.

Revision ID: 85f73dbda900
Revises: 9c8683b365f6
Create Date: 2026-07-20
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = '85f73dbda900'
down_revision: Union[str, None] = '9c8683b365f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'validation_manifests',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('kind', sa.String(24), nullable=False),
        sa.Column('discovery_run_id', sa.String(36), nullable=True),
        sa.Column('circuit_id', sa.String(36), nullable=True),
        sa.Column('parent_manifest_id', sa.String(36), nullable=True),
        sa.Column('payload', JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False,
                  server_default=sa.text('now()')),
    )
    op.create_index('idx_vman_discovery', 'validation_manifests', ['discovery_run_id'])
    op.create_index('idx_vman_circuit', 'validation_manifests', ['circuit_id'])


def downgrade() -> None:
    op.drop_index('idx_vman_circuit', table_name='validation_manifests')
    op.drop_index('idx_vman_discovery', table_name='validation_manifests')
    op.drop_table('validation_manifests')
