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
    # The arc's own two columns. Without these the clone test compared None
    # against None for both, so adding either to clone_template's denylist —
    # a clone silently reverting the sampling strategy, which is precisely the
    # "clone the baseline, change one field" A/B failure the test is named for
    # — would have stayed green.
    example_sampling="stratified",
    activation_display="percent_of_max",
    is_default=False,
)


class _FakeDB:
    def __init__(self): self.added = None
    def add(self, obj): self.added = obj
    async def commit(self): pass
    async def refresh(self, obj): pass


def test_the_distinct_fixture_covers_every_prompt_affecting_column():
    """The fixture must not silently stop covering a new column.

    `_DISTINCT` is what makes C54 and C56 able to detect a dropped field: a
    value that agrees with its column default proves nothing. This arc added two
    columns and did not update it, so the clone test compared None against None
    for both and the mutation that denylists them passed.

    One assertion closes the class rather than the instance.
    """
    from src.services.labeling_fingerprint import prompt_fingerprint_fields

    covered = set(_DISTINCT)
    needed = set(prompt_fingerprint_fields(_source_row()))
    missing = sorted(needed - covered)
    assert not missing, (
        f"_DISTINCT does not set these prompt-affecting columns, so any test "
        f"using it cannot detect them being dropped: {missing}"
    )


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


# ── clone_template ────────────────────────────────────────────────────────────
#
# C56 re-hand-list the fields in clone_template
#      -> test_clone_carries_every_prompt_affecting_field
#
# `create_template` above was fixed by deriving its field list from the request
# schema, and the comment there explains why. `clone_template` kept its
# hand-written list and kept the bug: it copied eighteen columns and dropped
# `include_negative_examples`, `num_negative_examples` and
# `include_nlp_analysis`.
#
# This matters more than an ordinary copy bug. "Clone the baseline, change one
# field" is how an A/B arm gets built, so a trial constructed that way would
# have moved four variables while its author believed one had moved — and the
# result would have been attributed to the one they changed.


class _FakeResult:
    def __init__(self, row): self._row = row
    def scalar_one_or_none(self): return self._row


class _FakeCloneDB(_FakeDB):
    def __init__(self, source):
        super().__init__()
        self._source = source
    async def execute(self, *_a, **_k): return _FakeResult(self._source)


def _source_row() -> LabelingPromptTemplate:
    """A source whose every column differs from its default.

    A field that agrees by construction cannot show a drop, which is the
    fixture trap this repo keeps rediscovering.
    """
    fields = {k: v for k, v in _DISTINCT.items() if k != "is_default"}
    return LabelingPromptTemplate(
        id="lpt_source", is_system=True, is_default=True, created_by="someone",
        **fields,
    )


@pytest.mark.asyncio
async def test_clone_carries_every_prompt_affecting_field():
    """C56. The fingerprint is the oracle, so this cannot drift from the columns.

    Asserting on a hand-written list of fields here would reproduce the very
    defect under test. `prompt_fingerprint` is DERIVED from
    `LabelingPromptTemplate.__table__.columns` minus an identity denylist, so
    equal fingerprints mean every prompt-affecting column survived the clone —
    including any column added after this test was written.
    """
    from src.services.labeling_fingerprint import (
        prompt_fingerprint,
        prompt_fingerprint_fields,
    )

    source = _source_row()
    db = _FakeCloneDB(source)

    clone = await LabelingPromptTemplateService.clone_template(db, "lpt_source")
    assert clone is not None

    src_fields = prompt_fingerprint_fields(source)
    clone_fields = prompt_fingerprint_fields(clone)
    dropped = [
        f"{k}: source {v!r}, clone {clone_fields.get(k)!r}"
        for k, v in src_fields.items()
        if clone_fields.get(k) != v
    ]
    assert not dropped, (
        "clone_template did not carry these prompt-affecting fields:\n  "
        + "\n  ".join(dropped)
    )
    assert prompt_fingerprint(clone) == prompt_fingerprint(source), (
        "the clone would be treated as a DIFFERENT judge from its source"
    )


@pytest.mark.asyncio
async def test_a_clone_is_never_default_or_system():
    """Cloning a system template must yield an editable, non-default copy.

    The negative control on the test above: deriving the field list from the
    columns must not carry `is_system`/`is_default` across, or cloning a system
    template would mint a second undeletable one.
    """
    source = _source_row()
    clone = await LabelingPromptTemplateService.clone_template(
        _FakeCloneDB(source), "lpt_source")

    assert clone.is_system is False
    assert clone.is_default is False
    assert clone.created_by is None
    assert clone.id != source.id and clone.id.startswith("lpt_")
    assert clone.name == "probe (Copy)"


# ── export / import round trip ────────────────────────────────────────────────
#
# C104 re-hand-list the export fields
#       -> test_export_carries_every_prompt_affecting_field
# C105 re-hand-list the import fields
#       -> test_a_round_trip_preserves_the_fingerprint
# C106 let an import set is_system
#       -> test_an_import_cannot_mint_a_system_template
#
# Export hand-listed EIGHT fields and import hand-listed six, so a round trip
# reset template_type, max_examples, the marker and context switches, both
# negative-example settings, include_nlp_analysis, is_detection_template — and
# this arc's example_sampling and activation_display — to column defaults.
#
# The failure is silent and total: a researcher exports a validated stratified
# template, a colleague imports it, and it runs top_k/absolute. Both believe
# they are running the same judge, and the export JSON carries no fingerprint
# that would contradict them.
#
# The arc's own commit claimed "FIVE hand-written copies … all three now come
# from one list". There were seven. These two were the uncovered ones, and the
# completeness guard reported full coverage over a surface it did not touch.


class _ExportDB(_FakeDB):
    def __init__(self, rows):
        super().__init__()
        self._rows = rows
        self.added = []

    async def execute(self, *_a, **_k):
        rows = self._rows

        class _Scalars:
            def all(self_inner): return rows

        class _Result:
            def scalars(self_inner): return _Scalars()
            def scalar_one_or_none(self_inner): return rows[0] if rows else None

        return _Result()

    def add(self, obj): self.added.append(obj)


@pytest.mark.asyncio
async def test_export_carries_every_prompt_affecting_field():
    """C104. The fingerprint's field set is the oracle, so this cannot drift."""
    from src.services.labeling_fingerprint import prompt_fingerprint_fields

    source = _source_row()
    source.is_system = False  # export only emits non-system templates
    payload = await LabelingPromptTemplateService.export_templates(
        _ExportDB([source]))

    item = payload["templates"][0]
    missing = sorted(set(prompt_fingerprint_fields(source)) - set(item))
    assert not missing, (
        f"export drops these prompt-affecting fields, so an import cannot "
        f"restore them: {missing}"
    )


@pytest.mark.asyncio
async def test_a_round_trip_preserves_the_fingerprint():
    """C105. The property that matters: same judge in, same judge out.

    Asserted on `prompt_fingerprint`, which is derived from the column set — so
    a field added later is covered without editing this test.
    """
    from src.services.labeling_fingerprint import prompt_fingerprint

    source = _source_row()
    source.is_system = False
    exported = await LabelingPromptTemplateService.export_templates(
        _ExportDB([source]))

    db = _ExportDB([])  # name not found -> create path
    await LabelingPromptTemplateService.import_templates(db, exported)

    assert len(db.added) == 1
    imported = db.added[0]
    assert prompt_fingerprint(imported) == prompt_fingerprint(source), (
        "a round trip changed the judge: the imported template would produce "
        "different verdicts than the one that was exported"
    )
    assert imported.example_sampling == "stratified"
    assert imported.activation_display == "percent_of_max"


@pytest.mark.asyncio
async def test_an_import_cannot_mint_a_system_template():
    """C106. Negative control: deriving must not open a privilege route.

    System templates cannot be deleted or edited. An import that could set the
    flag would let a JSON file install an undeletable template.
    """
    source = _source_row()
    source.is_system = False
    exported = await LabelingPromptTemplateService.export_templates(
        _ExportDB([source]))
    # A hand-edited export, which is the realistic attack shape.
    exported["templates"][0]["is_system"] = True
    exported["templates"][0]["is_default"] = True

    db = _ExportDB([])
    await LabelingPromptTemplateService.import_templates(db, exported)

    imported = db.added[0]
    assert imported.is_system is False
    assert imported.is_default is False, (
        "an import seized the default slot, which silently redirects every "
        "later labeling job that specifies no template"
    )


@pytest.mark.asyncio
async def test_the_overwrite_branch_cannot_grant_protection():
    """C106b. The branch NO test drove, and the one holding the is_system guard.

    `test_an_import_cannot_mint_a_system_template` uses `_ExportDB([])`, which
    always answers "not found" — so it only ever exercised the CREATE path. The
    overwrite path is the one that loops `setattr` over derived fields, and
    removing `is_system` from `_NEVER_EXPORTED` plus a natural
    `fields.pop("is_system", None)` gave it
    `setattr(existing_template, "is_system", True)` with the suite green.

    Its companion scrape guard could not catch that either: it asserts the
    strings `existing_template.is_system = ` and `template_data.get("is_system")`
    are absent, and the derived loop contains neither spelling. Third recorded
    instance of a source scrape failing open in this repo.
    """
    existing = _source_row()
    existing.id = "lpt_existing"
    existing.is_system = True          # a protected template
    existing.example_sampling = "top_k"

    db = _ExportDB([existing])
    payload = {
        # The version is required — the import refuses a payload without it,
        # which is correct and which my first fixture tripped over.
        "version": "1.0",
        "templates": [{
            "name": existing.name,
            "system_message": "REPLACED",
            "example_sampling": "stratified",
            "is_system": True,
            "is_default": True,
        }]
    }
    await LabelingPromptTemplateService.import_templates(
        db, payload, overwrite_duplicates=True)

    assert existing.system_message != "REPLACED", (
        "a SYSTEM template was overwritten by an import; the guard must run "
        "before any write"
    )


@pytest.mark.asyncio
async def test_the_overwrite_branch_updates_every_prompt_field():
    """Negative control: the guard must not refuse a legitimate overwrite.

    A branch that refuses everything passes the test above and makes
    `overwrite_duplicates` inert.
    """
    existing = _source_row()
    existing.id = "lpt_existing"
    existing.is_system = False
    existing.example_sampling = "top_k"
    existing.activation_display = "absolute"

    db = _ExportDB([existing])
    payload = {
        "version": "1.0",
        "templates": [{
            "name": existing.name,
            "system_message": "REPLACED",
            "example_sampling": "stratified",
            "activation_display": "percent_of_max",
            # A hand-edited payload asking for protection it may not have.
            "is_system": True,
            "is_default": True,
        }]
    }
    await LabelingPromptTemplateService.import_templates(
        db, payload, overwrite_duplicates=True)

    assert existing.system_message == "REPLACED"
    assert existing.example_sampling == "stratified", (
        "the overwrite dropped a prompt-affecting field — the defect that made "
        "the whole round trip lossy"
    )
    assert existing.activation_display == "percent_of_max"
    assert existing.is_system is False, (
        "an import promoted a user template to protected"
    )


@pytest.mark.asyncio
async def test_an_older_export_does_not_reset_missing_fields():
    """A payload that never carried a field must leave it at its default.

    `.get(key, default)` would overwrite with the default; absence means
    "unknown", not "reset".
    """
    from src.services.labeling_prompt_template_service import _importable_fields

    fields = _importable_fields({"name": "x", "system_message": "s"})
    assert set(fields) == {"name", "system_message"}
    assert "example_sampling" not in fields
