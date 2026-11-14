"""Hold the weak-example contrast block off until it has been measured.

A DATA MIGRATION, deliberately separate from the schema revisions before it.

`include_negative_examples` defaults True on the column and is True on every
template in the estate — including the ACTIVE default,
`miStudio Brand - NextGen - No NLP - (ACTIVE)`, whose `num_negative_examples` is
NULL and therefore resolves to 5.

That was harmless for as long as the switch was inert: `_retrieve_bottom_
examples_batch` had zero callers, so no prompt ever carried the block. Wiring it
(commit 49b08e43) made every one of those templates start sending five extra
passages per feature — an estate-wide change to what the judge sees, arriving as
a side effect of a bug fix, with no measurement behind it.

That is precisely the intervention this arc's own governing prior warns about.
The 2026-09-01 triage experiment measured that adding material to the prompt
collapsed the judge's refusal rate 38.0% -> 3.5% while label stability FELL
30% -> 17% — recorded as "confabulation, not comprehension". A ~50% increase in
prompt volume is the one channel where that mechanism transfers literally, and
the arc's own plan gates the contrast arm (A3) on the sampling arm (A2) passing
first.

So: OFF on every existing template, and the estate's prompts are byte-identical
to what they were before the wiring. Turning it on is now a deliberate act.

NOT a schema change and NOT a default change. The column default stays True,
because for a template created FROM NOW ON the block is a documented feature
someone opted into. This revision only declines to opt in retroactively on the
operator's behalf.

IDEMPOTENT and re-runnable: guarded on the current value, so re-applying is a
no-op rather than a way to undo an operator's later decision.

Revision ID: 2a81f3b18250
Revises: 9e44fb6b6ba9
Create Date: 2026-09-10
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "2a81f3b18250"
down_revision = "9e44fb6b6ba9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Guarded on `= true` so an operator who has already turned it off, or who
    # turns it on after this runs, is not overridden by a re-run.
    op.execute(
        "UPDATE labeling_prompt_templates "
        "SET include_negative_examples = false, "
        "    updated_at = NOW() "
        "WHERE include_negative_examples = true"
    )


def downgrade() -> None:
    # NO EXACT INVERSE EXISTS, and pretending otherwise would be worse than
    # saying so. Turning every template back on would also switch on any
    # template an operator had deliberately left off before this ran — the flag
    # carries no record of its prior value.
    #
    # The honest downgrade is to do nothing: the block being off is the
    # pre-wiring behaviour, so nothing is lost by staying there, and re-enabling
    # is one UPDATE that the operator can target at the template they mean.
    print(
        "[2a81f3b18250] downgrade is a no-op: include_negative_examples cannot "
        "be restored per-template (the prior value was not recorded), and OFF "
        "is the pre-wiring behaviour. Re-enable deliberately, per template."
    )
