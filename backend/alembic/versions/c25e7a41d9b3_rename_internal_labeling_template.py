"""rename the seeded internal labeling template to the product's new name

Revision ID: c25e7a41d9b3
Revises: 0b576a6b9bda
Create Date: 2026-11-11

ENH-009 suite decision R25 (operator, 2026-11-11). The system labeling template seeded as
``lpt_mistudio_internal`` is shown by name in the labeling UI and by ``list_labeling_templates``; its
name carried the product's old name. This revision renames that ONE row, by id.

Why a new revision and not an edit of the seed file. ``c458030cd014`` seeds the row from
``src/data/templates/mistudio_internal.json`` and ``n1o2p3q4r5s6`` later rewrites its prompts
``WHERE name = '<the seeded name>'``. Changing the name in the JSON would make that later revision
match nothing on a fresh database, silently, and the prompt text it writes is inside the labeling
fingerprint. So the JSON and both old revisions stay exactly as they are; a fresh database seeds the
old name, takes every prompt revision, and is renamed here, ending in the same state as an upgraded one.

* The id ``lpt_mistudio_internal`` and the file name ``mistudio_internal.json`` are frozen (DR-16).
* Only a row that still carries the seeded name is renamed: a name an operator typed is left alone.
* ``name`` is not part of the labeling fingerprint (``labeling_fingerprint.IDENTITY_FIELDS``), so no
  stored label reads as made by a different template.

Data only: one UPDATE of one row by primary key, no lock beyond that row. Downgrade restores the
seeded name under the mirrored guard.
"""

import sqlalchemy as sa

from alembic import op

revision = "c25e7a41d9b3"
down_revision = "0b576a6b9bda"
branch_labels = None
depends_on = None

TEMPLATE_ID = "lpt_mistudio_internal"
SEEDED_NAME = "miStudio Internal - Full Context"
NAME = "miWorkbench Internal - Full Context"

_RENAME = sa.text(
    "UPDATE labeling_prompt_templates SET name = :to_name "
    "WHERE id = :template_id AND is_system = true AND name = :from_name"
)


def upgrade() -> None:
    op.execute(_RENAME.bindparams(template_id=TEMPLATE_ID, from_name=SEEDED_NAME, to_name=NAME))


def downgrade() -> None:
    op.execute(_RENAME.bindparams(template_id=TEMPLATE_ID, from_name=NAME, to_name=SEEDED_NAME))
