"""add include_nlp_analysis to labeling_prompt_templates

Add configurable toggle for NLP statistical analysis in labeling prompts.
When False (default), NLP analysis is not computed or injected into prompts.

Revision ID: 9e045d0a94ef
Revises: d0549a22d4c6
Create Date: 2026-02-28 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9e045d0a94ef'
down_revision: Union[str, None] = 'd0549a22d4c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE labeling_prompt_templates ADD COLUMN IF NOT EXISTS include_nlp_analysis BOOLEAN NOT NULL DEFAULT false"
    )


def downgrade() -> None:
    op.drop_column('labeling_prompt_templates', 'include_nlp_analysis')
