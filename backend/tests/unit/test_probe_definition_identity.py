"""`identity` in `mistudio.probe-definition/v1` (039 FR-4, FR-5).

What must hold, because miLLM holds documents exported before this field and joins on the digests:
* a document built before 039 (no `identity` key) still validates, against the model AND the
  published schema;
* a rebuilt document differs from its pre-039 bytes by EXACTLY one trailing `identity` member —
  `"identity": null` for a document that records none, the block itself for one that does;
* a carried digest that is not the document's own is refused with a structured reason naming the
  field and both values; an unknown algorithm is refused as `identity_algorithm_unknown`.

The pre-039 fixture `tests/fixtures/probe_definition_pre_identity.json` was written by `serialise()`
from `test_probe_definition.definition()` at commit 272d0423, BEFORE the field existed (sources taken
with `git archive 272d0423`). Do not regenerate it.
"""
from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import ValidationError

from src.schemas import probe_identity as pi
from src.schemas.probe_definition import ProbeDefinitionV1
from src.services.probe_definition_builder import serialise
from tests.unit.test_probe_definition import definition

ROOT = pathlib.Path(__file__).resolve().parents[3]
PRE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "probe_definition_pre_identity.json"
PUBLISHED = ROOT / "docs" / "schemas" / "probe-definition-v1.json"


def _with_identity(revision=3) -> dict:
    document = definition()
    identity = pi.identity_for(ProbeDefinitionV1.model_validate(document), revision)
    document["identity"] = identity.model_dump(mode="json")
    return document


def _flip(hex_digest: str) -> str:
    return ("1" if hex_digest[0] != "1" else "2") + hex_digest[1:]


class TestTheContractIsAdditive:
    def test_pre_039_document_validates(self):
        import jsonschema

        document = json.loads(PRE.read_text())
        assert "identity" not in document
        assert ProbeDefinitionV1.model_validate(document).identity is None
        jsonschema.Draft202012Validator(json.loads(PUBLISHED.read_text())).validate(document)

    def test_rebuild_adds_only_a_trailing_identity(self):
        pre = PRE.read_text()
        assert pre.endswith("\n}") and '"identity"' not in pre
        rebuilt = serialise(ProbeDefinitionV1.model_validate(definition()))
        assert rebuilt == pre[: -len("\n}")] + ',\n  "identity": null\n}'
        parsed = json.loads(rebuilt)
        assert list(parsed)[-1] == "identity" and parsed.pop("identity") is None
        assert parsed == json.loads(pre)

    def test_a_built_identity_is_the_one_trailing_member(self):
        pre = PRE.read_text()
        rebuilt = serialise(ProbeDefinitionV1.model_validate(_with_identity()))
        prefix = pre[: -len("\n}")] + ',\n  "identity": {'
        assert rebuilt.startswith(prefix)
        parsed = json.loads(rebuilt)
        assert list(parsed)[-1] == "identity"
        identity = parsed.pop("identity")
        assert parsed == json.loads(pre)
        assert set(identity) == {"algorithm", "detector_digest", "decision_digest", "decision_revision"}

    def test_a_document_with_an_identity_validates_against_the_published_schema(self):
        import jsonschema

        document = json.loads(serialise(ProbeDefinitionV1.model_validate(_with_identity())))
        jsonschema.Draft202012Validator(json.loads(PUBLISHED.read_text())).validate(document)

    def test_the_published_schema_forbids_extra_identity_keys(self):
        import jsonschema

        document = json.loads(serialise(ProbeDefinitionV1.model_validate(_with_identity())))
        document["identity"]["signed_by"] = "someone"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(json.loads(PUBLISHED.read_text())).validate(document)


class TestAWrongDigestIsRefused:
    def test_the_documents_own_identity_validates(self):
        document = _with_identity(revision=4)
        model = ProbeDefinitionV1.model_validate(document)
        assert model.identity.decision_revision == 4

    @pytest.mark.parametrize("field", ["detector_digest", "decision_digest"])
    def test_wrong_digest_is_refused_with_its_shape(self, field):
        document = _with_identity()
        carried = _flip(document["identity"][field])
        recomputed = document["identity"][field]
        document["identity"][field] = carried
        with pytest.raises(ValidationError) as raised:
            ProbeDefinitionV1.model_validate(document)
        assert pi.identity_refusal(raised.value) == {
            "refused": "identity_mismatch",
            "field": field,
            "carried": carried,
            "recomputed": recomputed,
            "algorithm": "mistudio.probe-identity/1",
        }

    def test_an_edited_document_is_refused_by_its_old_identity(self):
        """The case that matters: a document whose BAR was edited while its identity was not."""
        document = _with_identity()
        old = document["identity"]["decision_digest"]
        document["decision"]["threshold"] = 9.5
        with pytest.raises(ValidationError) as raised:
            ProbeDefinitionV1.model_validate(document)
        detail = pi.identity_refusal(raised.value)
        assert detail["field"] == "decision_digest" and detail["carried"] == old
        assert detail["recomputed"] != old

    def test_an_unknown_algorithm_is_refused_with_its_code(self):
        document = _with_identity()
        document["identity"]["algorithm"] = "mistudio.probe-identity/2"
        with pytest.raises(ValidationError) as raised:
            ProbeDefinitionV1.model_validate(document)
        assert pi.identity_refusal(raised.value) == {
            "refused": "identity_algorithm_unknown",
            "algorithm": "mistudio.probe-identity/2",
            "known": ["mistudio.probe-identity/1"],
        }

    @pytest.mark.parametrize(
        "field,value",
        [("algorithm", "other/1"), ("detector_digest", "ABC"), ("decision_digest", "0" * 63),
         ("decision_revision", 0)],
    )
    def test_a_malformed_block_is_refused_by_the_schema(self, field, value):
        document = _with_identity()
        document["identity"][field] = value
        with pytest.raises(ValidationError) as raised:
            ProbeDefinitionV1.model_validate(document)
        assert pi.identity_refusal(raised.value) is None   # a shape error, not a digest mismatch

    def test_another_validation_error_has_no_identity_refusal(self):
        with pytest.raises(ValidationError) as raised:
            ProbeDefinitionV1.model_validate(definition(scope="nowhere"))
        assert pi.identity_refusal(raised.value) is None
