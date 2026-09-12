"""add_framework_encoder_types

Revision ID: k7l8m9n0o1p2
Revises: j6k7l8m9n0o1
Create Date: 2026-02-12

Expand encoder_type CHECK constraint to include standard_saelens and
standard_anthropic framework types.

The constraint previously allowed: standard, skip, transcoder, jumprelu, topk, gated.
Now it also allows: standard_saelens, standard_anthropic.

'standard' is kept for backward compatibility — the application layer
normalizes it to 'standard_saelens' at validation time.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'k7l8m9n0o1p2'
down_revision: Union[str, None] = 'j6k7l8m9n0o1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Expand encoder_type constraint to include framework-specific types."""
    op.drop_constraint('ck_training_templates_encoder_type', 'training_templates', type_='check')
    op.create_check_constraint(
        'ck_training_templates_encoder_type',
        'training_templates',
        "encoder_type IN ('standard', 'standard_saelens', 'standard_anthropic', "
        "'skip', 'transcoder', 'jumprelu', 'topk', 'gated')"
    )


def downgrade() -> None:
    """Revert to previous constraint (without standard_saelens/standard_anthropic).

    Note: This will fail if any rows use the new encoder_type values.
    """
    op.drop_constraint('ck_training_templates_encoder_type', 'training_templates', type_='check')
    op.create_check_constraint(
        'ck_training_templates_encoder_type',
        'training_templates',
        "encoder_type IN ('standard', 'skip', 'transcoder', 'jumprelu', 'topk', 'gated')"
    )
