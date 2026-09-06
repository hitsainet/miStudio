"""create_system_settings_table

Revision ID: e2a5af6a2b11
Revises: 20251113010218
Create Date: 2025-11-16 15:14:09.541188

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e2a5af6a2b11'
down_revision: Union[str, None] = '20251113010218'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
