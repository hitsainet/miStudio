"""MIS-E2E-107 — a plain `alias` renames on OUTPUT too.

`DatasetMetadata.dataset_schema` used `alias="schema"`, the exact construct that
`schemas/jspace_contracts.py` and `schemas/cluster_profile.py` both warn about
in writing — having once republished a schema without its wire field and
invalidated every exported document. The lesson was written down and never
applied outside those two modules, and the guard enforcing it iterates only
`JSPACE_KINDS`.

Two separate consequences, both reproduced on pydantic 2.12.5:

  1. The key you got OUT depended on which dump path ran: `model_dump()` emitted
     `dataset_schema` while `model_dump(by_alias=True)` emitted `schema`.
  2. `extra` defaulted to "ignore", so validating stored metadata DROPPED every
     undeclared key — including `task_id`, `task_type` and `lock_key`, which the
     download and tokenize endpoints write into `extra_metadata` and the cancel
     path reads back. A dataset whose metadata passed through this model lost
     the id of the very task someone was trying to cancel.

The stored spelling stays `dataset_schema` because existing rows carry it. What
changed is that it is now the FIELD NAME — stable across every dump path — while
both spellings are accepted on input.
"""

import pytest

from src.schemas.metadata import DatasetMetadata

_SCHEMA = {
    "text_columns": ["text"],
    "column_info": {"text": "string"},
    "all_columns": ["text"],
    "is_multi_column": False,
}


def test_no_field_carries_a_plain_alias():
    """The rule the other two modules already state, applied here.

    A plain `alias` sets BOTH validation and serialization aliases; a
    `validation_alias` sets only the input side, which is what was wanted.
    """
    offenders = [
        name
        for name, field in DatasetMetadata.model_fields.items()
        if field.alias is not None
    ]
    assert not offenders, (
        f"{offenders} carry a plain alias, which renames the field on OUTPUT — "
        f"the construct that invalidated every exported document once already"
    )


@pytest.mark.parametrize("key", ["schema", "dataset_schema"])
def test_both_spellings_are_accepted_on_input(key):
    m = DatasetMetadata.model_validate({key: _SCHEMA})
    assert m.dataset_schema is not None


def test_the_output_key_is_the_same_on_every_dump_path():
    """The defect, stated directly: the key must not depend on how you dump."""
    m = DatasetMetadata.model_validate({"schema": _SCHEMA})
    plain = set(m.model_dump())
    aliased = set(m.model_dump(by_alias=True))
    assert "dataset_schema" in plain
    assert plain == aliased, (
        f"model_dump and model_dump(by_alias=True) disagree: "
        f"{sorted(plain ^ aliased)} — a consumer's key depends on the caller"
    )


def test_metadata_round_trips_without_losing_or_renaming_anything():
    m = DatasetMetadata.model_validate({"schema": _SCHEMA})
    once = m.model_dump()
    twice = DatasetMetadata.model_validate(once).model_dump()
    assert once == twice


@pytest.mark.parametrize("extra_key", ["task_id", "task_type", "lock_key"])
def test_worker_written_keys_survive_validation(extra_key):
    """These are written by the download and tokenize endpoints and read back
    by cancel. `extra: ignore` silently deleted them."""
    payload = {"schema": _SCHEMA, extra_key: "value-1"}
    dumped = DatasetMetadata.model_validate(payload).model_dump()
    assert dumped.get(extra_key) == "value-1", (
        f"{extra_key} was dropped by validation — the cancel path then has no "
        f"task id to revoke"
    )


def test_the_documented_rule_is_still_documented_where_it_was_learned():
    """Negative control for the premise.

    This fix applies a rule stated in two other modules. If those statements
    disappeared, the rule would have no home and the next author would
    reintroduce the alias.
    """
    import inspect

    from src.schemas import jspace_contracts

    doc = inspect.getsource(jspace_contracts)
    assert "renames on output" in doc.lower() or "RENAMES ON OUTPUT" in doc
