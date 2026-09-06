"""The labeling-template service. There was no test file for it at all.

Found the hard way: creating a template through the API returned 201 and stored
column defaults for twelve of the twenty fields it accepted, so a template did
not behave the way it was configured and nothing reported a problem.

Mutation controls:
  C54 drop a field from create_template's construction
       -> test_create_persists_every_field_the_schema_accepts
  C55 let a request set is_system=True
       -> test_a_request_cannot_make_itself_a_system_template
"""

import pytest

from src.models.labeling_prompt_template import LabelingPromptTemplate
from src.schemas.labeling_prompt_template import LabelingPromptTemplateCreate
from src.services.labeling_prompt_template_service import (
    LabelingPromptTemplateService,
)

# Non-default values, so a field that is silently dropped shows up as a
# mismatch rather than coincidentally matching the column default.
_DISTINCT = dict(
    name="probe",
    description="d",
    system_message="sys",
    user_prompt_template="body {examples_block}",
    temperature=0.11,
    max_tokens=321,
    top_p=0.77,
    template_type="mistudio_context",
    max_examples=17,
    include_prefix=False,
    include_suffix=False,
    prime_token_marker="[[]]",
    include_logit_effects=True,
    top_promoted_tokens_count=7,
    top_suppressed_tokens_count=8,
    include_negative_examples=False,
    num_negative_examples=3,
    is_detection_template=False,
    include_nlp_analysis=True,
    is_default=False,
)


class _FakeDB:
    def __init__(self): self.added = None
    def add(self, obj): self.added = obj
    async def commit(self): pass
    async def refresh(self, obj): pass


@pytest.mark.asyncio
async def test_create_persists_every_field_the_schema_accepts():
    """C54. Every field the request accepts must reach the row.

    A hand-written constructor list is the defect: the schema grew and the
    constructor did not, so twelve fields were accepted and discarded.
    """
    db = _FakeDB()
    created = await LabelingPromptTemplateService.create_template(
        db, LabelingPromptTemplateCreate(**_DISTINCT))

    dropped = []
    for field, expected in _DISTINCT.items():
        actual = getattr(created, field, None)
        if actual != expected:
            dropped.append(f"{field}: sent {expected!r}, stored {actual!r}")
    assert not dropped, (
        "create_template accepted these fields and did not store them:\n  "
        + "\n  ".join(dropped)
    )


@pytest.mark.asyncio
async def test_a_request_cannot_make_itself_a_system_template():
    """C55. System templates are undeletable and unmodifiable. Deriving the
    field list from the schema must not open a route to setting is_system."""
    db = _FakeDB()
    created = await LabelingPromptTemplateService.create_template(
        db, LabelingPromptTemplateCreate(**_DISTINCT))
    assert created.is_system is False
    assert created.created_by is None


@pytest.mark.asyncio
async def test_the_id_is_generated_not_taken_from_the_request():
    db = _FakeDB()
    created = await LabelingPromptTemplateService.create_template(
        db, LabelingPromptTemplateCreate(**_DISTINCT))
    assert created.id.startswith("lpt_")
    assert len(created.id) == len("lpt_") + 16
