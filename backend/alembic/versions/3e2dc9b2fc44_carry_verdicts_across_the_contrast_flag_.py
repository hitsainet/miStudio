"""Carry existing verdicts across the contrast-flag fingerprint change.

THE MIGRATION BEFORE THIS ONE WOULD HAVE ASKED FOR ~118 GPU-HOURS OF NOTHING.

`2a81f3b18250` sets `include_negative_examples = false` on every template, so
that a block which had been inert could not switch itself on estate-wide as a
side effect of being wired. Correct on its own terms — and it moves every
template's `prompt_fingerprint`, because that hash covers every non-identity
column and this is one.

`features.label_prompt_fingerprint` therefore stops matching, and the labeling
card — which always reads coverage with the template's current fingerprint —
reports every succeeded feature as STALE. On L46 that is 53,088 features at
~8 s each: roughly 118 GPU-hours, offered as a single click, to regenerate
labels from prompts that are BYTE-IDENTICAL to the ones that produced them.
Identical because the block never rendered before (no caller) and does not
render now (flag off).

The fingerprint is doing its job; the input to it moved for a reason that did
not change what the judge was sent. So the honest repair is to carry the
verdicts forward, not to relabel them and not to weaken the fingerprint.

WHY THIS IS A SEPARATE REVISION. House style, and more than that: this rewrites
rows in `features`, the largest table this arc touches, while `2a81f3b18250`
rewrites a handful of template rows. They fail differently, they are re-run
differently, and mixing them would make the schema change hostage to the
backfill's runtime.

WHAT IT DOES NOT DO. It does not touch verdicts whose fingerprint matches
neither the old nor the new value — a template that changed for some other
reason is genuinely stale and must stay so. It does not touch NULL
fingerprints, which mean "we do not know which judge produced this" and are
already excluded from routine resume.

Revision ID: 3e2dc9b2fc44
Revises: 2a81f3b18250
Create Date: 2026-09-10
"""
import hashlib
import json

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "3e2dc9b2fc44"
down_revision = "2a81f3b18250"
branch_labels = None
depends_on = None


#: Mirrors `labeling_fingerprint.IDENTITY_FIELDS`. Duplicated deliberately: a
#: migration must describe the schema AS IT WAS WHEN IT RAN, and importing the
#: live module would make this revision's behaviour change under it the next
#: time that denylist is edited.
_IDENTITY_FIELDS = frozenset({
    "id", "name", "description", "is_default", "is_system",
    "created_by", "created_at", "updated_at",
})


def _fingerprint(row_mapping, columns, *, include_negative_examples) -> str:
    """The fingerprint this template WOULD have had at a given flag value."""
    fields = {}
    for column in columns:
        if column in _IDENTITY_FIELDS:
            continue
        value = row_mapping[column]
        if column == "include_negative_examples":
            value = include_negative_examples
        fields[column] = value
    # EVERY ARGUMENT MATCHES `labeling_fingerprint.prompt_fingerprint`.
    #
    # `separators` in particular: the default `", "` / `": "` produce a
    # different byte string than `","` / `":"`, and therefore a different hash —
    # so the UPDATE would match zero rows, print a reassuring "carried 0", and
    # leave the whole estate reading as stale with nothing reporting a problem.
    # A test pins this against the real writer for exactly that reason.
    #
    # No `default=`: the writer refuses non-scalars rather than stringifying
    # them, and silently accepting one here would hash a value the writer never
    # would.
    payload = json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def upgrade() -> None:
    conn = op.get_bind()

    columns = [
        r[0] for r in conn.execute(sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'labeling_prompt_templates' "
            "ORDER BY ordinal_position"
        ))
    ]
    if not columns:
        print(f"[{revision}] no labeling_prompt_templates table; nothing to do")
        return

    rows = conn.execute(sa.text(
        f"SELECT {', '.join(columns)} FROM labeling_prompt_templates"
    )).mappings().all()

    carried = 0
    for row in rows:
        before = _fingerprint(row, columns, include_negative_examples=True)
        after = _fingerprint(row, columns, include_negative_examples=False)
        if before == after:
            continue
        result = conn.execute(
            sa.text(
                "UPDATE features SET label_prompt_fingerprint = :after "
                "WHERE label_prompt_fingerprint = :before"
            ),
            {"after": after, "before": before},
        )
        carried += result.rowcount or 0

    print(
        f"[{revision}] carried {carried} verdict(s) across the contrast-flag "
        f"fingerprint change; those labels were produced by a byte-identical "
        f"prompt and are not stale"
    )


def downgrade() -> None:
    # The exact inverse, and it IS exact: the mapping is one-to-one, computed
    # from the same rows, and no other revision writes these values.
    conn = op.get_bind()

    columns = [
        r[0] for r in conn.execute(sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'labeling_prompt_templates' "
            "ORDER BY ordinal_position"
        ))
    ]
    if not columns:
        return

    rows = conn.execute(sa.text(
        f"SELECT {', '.join(columns)} FROM labeling_prompt_templates"
    )).mappings().all()

    for row in rows:
        before = _fingerprint(row, columns, include_negative_examples=True)
        after = _fingerprint(row, columns, include_negative_examples=False)
        if before == after:
            continue
        conn.execute(
            sa.text(
                "UPDATE features SET label_prompt_fingerprint = :before "
                "WHERE label_prompt_fingerprint = :after"
            ),
            {"after": after, "before": before},
        )
