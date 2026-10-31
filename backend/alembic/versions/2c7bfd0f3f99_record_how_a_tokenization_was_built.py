"""record how a tokenization was built

Revision ID: 2c7bfd0f3f99
Revises: 3e2dc9b2fc44
Create Date: 2026-09-11

WHY. `padding`, `truncation`, `add_special_tokens` and the text column were all
REQUEST-ONLY: passed to the Celery task and then lost. An existing tokenization
therefore carried no record of how it was produced, and two rows that look
identical could have been built differently.

That is not academic. `danidanou/Bloomberg_Financial_News` has columns
`Headline, Journalists, Date, Link, Article` and no `text`/`content`/`chosen`,
so auto-detection fell through to `text_columns[0]` = **Headline**. Every
Bloomberg tokenization in this estate is headlines — mean 17 tokens against
`Article`'s 464, i.e. ~207M tokens of financial prose never used — and nothing
recorded the choice, so nothing could surface it.

Additive and nullable/defaulted throughout: existing rows keep working and
describe themselves as "not recorded" rather than claiming a default they may
not have used. `text_column` is deliberately NULL rather than `'text'` for
exactly that reason.
"""
from alembic import op
import sqlalchemy as sa

revision = '2c7bfd0f3f99'
down_revision = '3e2dc9b2fc44'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Small table (tens of rows); no lock_timeout dance needed, unlike anything
    # touching feature_activations.
    op.add_column('dataset_tokenizations', sa.Column('text_column', sa.String(255), nullable=True))
    # NULLABLE ON PURPOSE, and this was wrong in the first draft of this file.
    #
    # A NOT NULL DEFAULT 'auto' makes every pre-existing row claim it was
    # rendered with the tokenizer's real chat template — the one value that
    # means exactly that. Every historical conversation tokenization in this
    # estate used the `<|role|>` pseudo-markers. A column added to STOP a silent
    # misattribution would have introduced one. NULL means "not recorded", which
    # is the truth for every row that predates this.
    op.add_column('dataset_tokenizations', sa.Column(
        'chat_format', sa.String(20), nullable=True))
    op.add_column('dataset_tokenizations', sa.Column(
        'pack_sequences', sa.Boolean(), nullable=True))
    op.add_column('dataset_tokenizations', sa.Column('padding', sa.String(20), nullable=True))
    op.add_column('dataset_tokenizations', sa.Column('truncation', sa.String(20), nullable=True))
    op.add_column('dataset_tokenizations', sa.Column('add_special_tokens', sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column('dataset_tokenizations', 'add_special_tokens')
    op.drop_column('dataset_tokenizations', 'truncation')
    op.drop_column('dataset_tokenizations', 'padding')
    op.drop_column('dataset_tokenizations', 'pack_sequences')
    op.drop_column('dataset_tokenizations', 'chat_format')
    op.drop_column('dataset_tokenizations', 'text_column')
