"""merge app_settings and enhanced_labeling heads

Revision ID: cd6c46abac48
Revises: f3a7b1c2d4e5, t7u8v9w0x1y2
Create Date: 2026-07-11 08:54:50.906308

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cd6c46abac48'
down_revision: Union[str, None] = ('f3a7b1c2d4e5', 't7u8v9w0x1y2')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
