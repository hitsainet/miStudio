"""record whether a tokenization's blocks were shuffled

Revision ID: b9d4e7a2c815
Revises: d5a1f3c7e9b2
Create Date: 2026-09-17

WHY. Activation extraction reads a tokenized dataset with
`dataset.select(range(max_samples))` (services/activation_service.py) — a straight
PREFIX from block 0 — and nothing in the tokenization path shuffles: the raw
dataset is read in download order, `dataset.map` preserves order, and
`iter_packed_blocks` emits blocks sequentially.

So every extraction in this estate used the FIRST N blocks of each corpus, in
original order — Bloomberg 9,000 of 138,400 blocks (6.5%), codeparrot 13,500 of
55,926.

MEASURED 2026-09-17, after this migration shipped: neither corpus named above is
actually ordered. Bloomberg scores Spearman -0.06 between row order and article
date (4,025 of 446,762 rows sampled); codeparrot's first quarter covers 30.3% of
its 33,090 repositories. The ordered corpus is OpenHermes-2.5, written one source
at a time — 14 of its 15 labeled sources occupy a single contiguous row range, so
a 10% prefix holds only airoboros2.2 and CamelAI (11.3% of the corpus) and never
reaches glaive-code-assist (18.2%) or the unlabeled 49.6% remainder.

A dictionary trained on one slice looks perfectly healthy — FVU converges, CE
delta is fine, dead latents stay at zero — because the held-out evaluation draws
its rows from AFTER the prefix: adjacent data, same slice, same bias. Nothing
downstream can detect it. These columns put the answer on the row instead of
requiring someone to decode blocks and guess.

NULLABLE, no default, no server_default, no backfill — the convention
2c7bfd0f3f99 settled on after `chat_format` was nearly shipped as
NOT NULL DEFAULT 'auto', which would have made every historical row claim the
tokenizer's real template. NULL here means NOT RECORDED, which is the truth for
every row written before this: all of them are in fact unshuffled, but a row must
not assert a shuffle seed it never had. A sentinel is not available either —
0 is a legitimate seed, and -1 would be a lie of a different shape.
"""
from alembic import op
import sqlalchemy as sa

revision = 'b9d4e7a2c815'
down_revision = 'd5a1f3c7e9b2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Small table (tens of rows); no lock_timeout dance needed, unlike anything
    # touching feature_activations.
    #
    # THE COMMENTS ARE PART OF THE SCHEMA, not decoration. `test_schema_guards`
    # compares the migrated database against the models and reports a
    # `modify_comment` when they disagree — so a migration that adds the column
    # without the comment the model declares leaves the two out of step, and the
    # guard refuses it rather than letting the ratchet absorb it.
    op.add_column('dataset_tokenizations', sa.Column(
        'shuffled', sa.Boolean(), nullable=True,
        comment=(
            "Whether the packed blocks were permuted before being written. NULL "
            "means NOT RECORDED, which is the truth for every row written before "
            "this column existed — those are all unshuffled in fact, but a row "
            "must not claim a shuffle it never had."
        ),
    ))
    op.add_column('dataset_tokenizations', sa.Column(
        'shuffle_seed', sa.Integer(), nullable=True,
        comment=(
            "Seed of the permutation actually used, so the block order is "
            "reproducible. NULL means not recorded, or recorded-and-not-shuffled "
            "(read `shuffled` to tell those apart). No sentinel is available: 0 "
            "is a legitimate seed."
        ),
    ))


def downgrade() -> None:
    op.drop_column('dataset_tokenizations', 'shuffle_seed')
    op.drop_column('dataset_tokenizations', 'shuffled')
