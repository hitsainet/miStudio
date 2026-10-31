"""`behaviour` in `mistudio.probe-definition/v1` (Feature 037 FR-10; ENH-005 D23, D25, D50).

What must hold, because each is a claim made to miLLM, which holds documents exported before this field:
* a probe with no key exports `behaviour: null` — and a rebuilt keyless document differs from its
  pre-037 bytes by EXACTLY one trailing `"behaviour": null` member (the field is declared last);
* a document exported before 037 (no `behaviour` key at all) still validates, against the model AND the
  published schema;
* a catalogued key exports the catalogue's current name and version; an uncatalogued key exports the
  probe's label (or the key) with `version: null`;
* the contract's literal slug pattern is the library's (five copies, one rule).

The end-to-end build (catalogued version, `concept` unchanged) is
`test_probe_monitor_pipeline.py::TestTheDefinitionCarriesTheBehaviour`.

The pre-037 fixture `tests/fixtures/probe_definition_pre_behaviour.json` was written by `serialise()` from
`test_probe_definition.definition()` at commit c321b28b, BEFORE the field existed. Do not regenerate it.
"""
from __future__ import annotations

import json
import pathlib
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.schemas.probe_definition import BEHAVIOUR_KEY_PATTERN, BehaviourRef, ProbeDefinitionV1
from src.services.probe_definition_builder import behaviour_ref_for, serialise
from tests.unit.test_probe_definition import definition

ROOT = pathlib.Path(__file__).resolve().parents[3]
PRE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "probe_definition_pre_behaviour.json"
PUBLISHED = ROOT / "docs" / "schemas" / "probe-definition-v1.json"


def test_behaviour_ref_pattern_matches_the_library():
    import importlib.util

    from src.models.probe_behaviour import SLUG_REGEX
    from src.schemas.probe_monitor import BEHAVIOUR_PATTERN, ProbeDatasetCreate
    from src.services.judge_prompt_forms import SLUG_PATTERN

    migration_path = ROOT / "backend" / "alembic" / "versions" / "e7c1a4b9d2f6_behaviour_catalogue.py"
    spec = importlib.util.spec_from_file_location("rev_e7c1_contract", migration_path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    registration = ProbeDatasetCreate.model_json_schema()["properties"]["behaviour"]["anyOf"][0]["pattern"]
    published = json.loads(PUBLISHED.read_text())["$defs"]["BehaviourRef"]["properties"]["key"]["pattern"]
    assert (
        SLUG_PATTERN == BEHAVIOUR_PATTERN == BEHAVIOUR_KEY_PATTERN == SLUG_REGEX
        == migration.SLUG_REGEX == registration == published
    )


class TestTheKeylessDocument:
    def test_keyless_export_adds_only_a_trailing_null_behaviour(self):
        pre = PRE.read_text()
        rebuilt = serialise(ProbeDefinitionV1.model_validate(definition()))
        assert pre.endswith("\n}") and not pre.rstrip().endswith('"behaviour": null\n}')
        # ⚠ Feature 039 (2026-10-23) declared `identity` after `behaviour`, so a rebuild now ends
        # with a second trailing member. Intended (039 FR-4; recorded in
        # 0xcc/reviews/039_implementation_controls_2026-10-23.md). The 037 claim is unchanged: the
        # byte prefix up to the last pre-037 member is untouched and `behaviour` is the next member.
        identity_tail = ',\n  "identity": null\n}'
        assert rebuilt.endswith(identity_tail)
        rebuilt = rebuilt[: -len(identity_tail)] + "\n}"
        # The byte prefix up to the last member is unchanged, and exactly one member is appended.
        assert rebuilt == pre[: -len("\n}")] + ',\n  "behaviour": null\n}'
        parsed = json.loads(rebuilt)
        assert list(parsed)[-1] == "behaviour" and parsed.pop("behaviour") is None
        assert parsed == json.loads(pre)

    def test_pre_change_document_still_validates(self):
        """miLLM holds files exported before 037; they carry no `behaviour` key."""
        import jsonschema

        document = json.loads(PRE.read_text())
        assert "behaviour" not in document
        assert ProbeDefinitionV1.model_validate(document).behaviour is None
        jsonschema.Draft202012Validator(json.loads(PUBLISHED.read_text())).validate(document)

    def test_a_document_with_a_behaviour_validates_against_the_published_schema(self):
        import jsonschema

        document = json.loads(serialise(ProbeDefinitionV1.model_validate(
            {**definition(), "behaviour": {"key": "pii", "name": "Personal data present", "version": 2}}
        )))
        jsonschema.Draft202012Validator(json.loads(PUBLISHED.read_text())).validate(document)
        bad = {**document, "behaviour": {"key": "Not A Slug", "name": "x", "version": 1}}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(json.loads(PUBLISHED.read_text())).validate(bad)

    def test_the_object_carries_no_lifecycle_flag(self):
        """D50: no `archived` in the contract."""
        assert set(BehaviourRef.model_fields) == {"key", "name", "version"}
        with pytest.raises(Exception):
            BehaviourRef.model_validate({"key": "pii", "name": "x", "version": 1, "archived": True})


@pytest.fixture
def sync_db(async_engine):
    url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", "")
    )
    engine = create_engine(url)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


class TestBehaviourRefFor:
    def test_no_key_is_null(self, sync_db):
        assert behaviour_ref_for(sync_db, SimpleNamespace(behaviour=None, behaviour_label="x")) is None

    def test_catalogued_export_carries_version(self, sync_db):
        """After one meaning edit the export says version 2, with version 2's name."""
        from src.models.probe_behaviour import ProbeBehaviour, ProbeBehaviourVersion

        key = f"k-{uuid.uuid4().hex[:8]}"
        sync_db.add(ProbeBehaviour(key=key, current_version=2, created_by="operator"))
        sync_db.flush()
        sync_db.add_all([
            ProbeBehaviourVersion(key=key, version=1, name="Old name", positive_meaning="a", created_by="operator"),
            ProbeBehaviourVersion(key=key, version=2, name="New name", positive_meaning="b", created_by="operator"),
        ])
        sync_db.commit()
        ref = behaviour_ref_for(sync_db, SimpleNamespace(behaviour=key, behaviour_label="Probe label"))
        assert ref.model_dump() == {"key": key, "name": "New name", "version": 2}

    def test_an_uncatalogued_key_exports_its_label_and_no_version(self, sync_db):
        ref = behaviour_ref_for(sync_db, SimpleNamespace(behaviour="not-catalogued", behaviour_label="Refusals"))
        assert ref.model_dump() == {"key": "not-catalogued", "name": "Refusals", "version": None}
        bare = behaviour_ref_for(sync_db, SimpleNamespace(behaviour="not-catalogued", behaviour_label=None))
        assert bare.model_dump() == {"key": "not-catalogued", "name": "not-catalogued", "version": None}
