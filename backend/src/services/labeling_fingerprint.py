"""Content-addressed identity for the judge that produced a feature's label.

`features.label_prompt_fingerprint` answers "adjudicated by WHICH judge?", and
that is what lets "already adjudicated" mean something durable. Without it a
resume can only ask whether a verdict exists, not whether it is still the
verdict this template would produce — so improving a prompt would either
invalidate nothing (and quietly serve stale labels forever) or require
hand-written SQL to decide what to redo.

Content-addressed rather than a version counter, for the same reason
`LabelingTrialRun.panel_id` is and PADR IDL-48 pins its scoring prompt: an
editable ruler silently invalidates every prior measurement, and a version
field that someone must remember to bump is exactly the kind of discipline that
is not kept. Templates are editable in place through
`PATCH /labeling-prompt-templates/{id}`, so a stored template id proves nothing
about what was actually sent.

WHY THE FIELD SET IS DERIVED, NOT LISTED.

`labeling_trial_service.freeze_template` hand-lists eighteen columns. That is
the anti-pattern this repo has been bitten by repeatedly — `SURFACES` was
hand-maintained at five files while sixteen modules went unaudited, and
`REQUIRED_TABLES` was a literal dict of seventeen tables while the ORM declared
thirty-six. Both drifted silently, and both drifted in the direction of
reporting success. So the fields here come from the ORM, and only IDENTITY AND
BOOKKEEPING is excluded. A new prompt-affecting column is covered the day it is
added, by default, with nobody remembering anything.

WHY SAMPLING PARAMETERS COUNT.

`temperature`, `top_p` and `max_tokens` do not change what the judge SEES, but
they change what it SAYS, and `max_tokens` can truncate a verdict outright. The
two errors are not symmetric: including a field that did not matter marks some
verdicts stale, and stale verdicts are only re-adjudicated when an operator
explicitly asks — a routine resume never touches them. EXCLUDING a field that
did matter keeps a verdict that this template would no longer produce, forever,
with nothing to show it. Over-inclusion is cheap; under-inclusion is silent.
"""

import hashlib
import json
from typing import Any, Dict, Optional

from ..models.labeling_prompt_template import LabelingPromptTemplate

# Identity and bookkeeping: these change without changing a single byte the
# judge is sent or a single parameter it is called with. Renaming a template or
# marking it default must NOT invalidate the verdicts it already produced.
IDENTITY_FIELDS = frozenset(
    {
        "id",
        "name",
        "description",
        "is_default",
        "is_system",
        "created_by",
        "created_at",
        "updated_at",
    }
)

# The judge when no template row is in play at all. A distinct, stable identity
# rather than NULL: NULL is reserved for "we do not know which judge produced
# this", which is what the backfill writes over historical rows, and a live run
# must never be indistinguishable from that.
_BUILTIN_SENTINEL = "__millm_builtin_default_prompt__"

# Values a fingerprint may be computed over. Deliberately narrow: `json.dumps`
# with `default=str` would happily hash an object's repr, and a repr containing
# a memory address makes the fingerprint differ between two processes reading
# the same row — a bug that would present as "everything is stale, always".
_HASHABLE_TYPES = (str, int, float, bool, type(None))


def prompt_fingerprint_fields(
    template: Optional[LabelingPromptTemplate],
) -> Dict[str, Any]:
    """The exact mapping the fingerprint is taken over.

    Exposed so a test can assert WHAT is covered rather than only that two
    hashes differ, and so an operator can see why a verdict went stale.
    """
    if template is None:
        return {"__template__": _BUILTIN_SENTINEL}

    fields: Dict[str, Any] = {}
    for column in LabelingPromptTemplate.__table__.columns:
        if column.name in IDENTITY_FIELDS:
            continue
        value = getattr(template, column.name)
        if not isinstance(value, _HASHABLE_TYPES):
            raise TypeError(
                f"labeling_prompt_templates.{column.name} is "
                f"{type(value).__name__}, which has no stable textual form. "
                "Add it to IDENTITY_FIELDS if it cannot affect the prompt, or "
                "give it an explicit canonical encoding here — hashing its "
                "repr would make the fingerprint differ between processes."
            )
        fields[column.name] = value
    return fields


def prompt_fingerprint(template: Optional[LabelingPromptTemplate]) -> str:
    """sha256 over every prompt-affecting field of `template`.

    Returns a bare 64-character hex digest, unprefixed: it is stored in
    `features.label_prompt_fingerprint`, which is `String(64)` exactly, so a
    prefix would be silently truncated by the database and collapse distinct
    judges onto one identity.
    """
    payload = json.dumps(
        prompt_fingerprint_fields(template),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
