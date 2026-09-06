"""expand_encoder_type_constraint

Revision ID: 4a1844011c28
Revises: 56ee0619fe2e
Create Date: 2026-01-14

Add 'topk' and 'gated' to the encoder_type CHECK constraint.
The constraint previously only allowed: standard, skip, transcoder, jumprelu.
Now it allows: standard, skip, transcoder, jumprelu, topk, gated.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4a1844011c28'
down_revision: Union[str, None] = '56ee0619fe2e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Expand encoder_type constraint to include topk and gated architectures.

    Supported encoder types after this migration:
    - standard: Standard SAE with L1 sparsity
    - skip: Skip connection SAE
    - transcoder: Transcoder architecture
    - jumprelu: JumpReLU SAE (DeepMind style)
    - topk: Top-K SAE (selects top K activations)
    - gated: Gated SAE architecture
    """
    # Drop the old constraint (which had standard, skip, transcoder, jumprelu)
    op.drop_constraint('ck_training_templates_encoder_type', 'training_templates', type_='check')

    # Create new constraint with all supported encoder types
    op.create_check_constraint(
        'ck_training_templates_encoder_type',
        'training_templates',
        "encoder_type IN ('standard', 'skip', 'transcoder', 'jumprelu', 'topk', 'gated')"
    )


def downgrade() -> None:
    """
    Revert to previous constraint (jumprelu included but not topk/gated).

    Note: This will fail if any rows have 'topk' or 'gated' encoder_type.
    """
    op.drop_constraint('ck_training_templates_encoder_type', 'training_templates', type_='check')
    op.create_check_constraint(
        'ck_training_templates_encoder_type',
        'training_templates',
        "encoder_type IN ('standard', 'skip', 'transcoder', 'jumprelu')"
    )
