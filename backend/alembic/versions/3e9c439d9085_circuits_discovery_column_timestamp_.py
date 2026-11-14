"""circuits discovery column + timestamp server defaults

NOTE: downgrade DROPS the discovery column — discovery provenance is lost on
downgrade (inherent; re-import definitions to restore).

Revision ID: 3e9c439d9085
Revises: 9a7da58fcd50
Create Date: 2026-07-19 21:16:18.372818

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '3e9c439d9085'
down_revision: Union[str, None] = '9a7da58fcd50'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("circuits", sa.Column("discovery", postgresql.JSONB(), nullable=True))
    # Non-ORM inserts (fixtures, backfills) need DB-side timestamps (review R1).
    op.alter_column("circuits", "created_at", server_default=sa.text("now()"))
    op.alter_column("circuits", "updated_at", server_default=sa.text("now()"))


def downgrade() -> None:
    op.alter_column("circuits", "updated_at", server_default=None)
    op.alter_column("circuits", "created_at", server_default=None)
    op.drop_column("circuits", "discovery")
