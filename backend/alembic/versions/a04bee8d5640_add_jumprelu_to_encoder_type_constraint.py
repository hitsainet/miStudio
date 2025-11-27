"""add_jumprelu_to_encoder_type_constraint

Revision ID: a04bee8d5640
Revises: 92780ad64734
Create Date: 2025-11-27 00:58:24.155916

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a04bee8d5640'
down_revision: Union[str, None] = '92780ad64734'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the old constraint
    op.drop_constraint('ck_training_templates_encoder_type', 'training_templates', type_='check')

    # Create new constraint with 'jumprelu' added
    op.create_check_constraint(
        'ck_training_templates_encoder_type',
        'training_templates',
        "encoder_type IN ('standard', 'skip', 'transcoder', 'jumprelu')"
    )


def downgrade() -> None:
    # Drop the new constraint
    op.drop_constraint('ck_training_templates_encoder_type', 'training_templates', type_='check')

    # Restore old constraint (note: this will fail if any rows have 'jumprelu')
    op.create_check_constraint(
        'ck_training_templates_encoder_type',
        'training_templates',
        "encoder_type IN ('standard', 'skip', 'transcoder')"
    )
