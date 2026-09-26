"""`docs/schemas/probe-definition-v1.json` stays in lockstep with the pydantic contract (033 1.4).

REGENERATE: `python -m tests.unit.test_probe_definition_schema_sync` from `backend/`, then READ THE
DIFF. The published file is what a consumer validates against, so a regeneration nobody reviewed is
a contract change nobody reviewed.

⚠ WHY A PUBLISHED SCHEMA AT ALL, when the pydantic model is the contract. miLLM validates documents
it did not build, in a process that does not import this package. Without a published schema its
only options are to vendor this module (a second implementation of the same rules, free to drift)
or to trust the document. The cluster and circuit contracts settled this pattern; this follows it.

MUTATION CONTROLS (each verified to fail this file):
  S1  a field's type changed in the contract without regenerating  → the equality test
  S2  a cap constant changed without regenerating                  → the equality test
  S3  the `$id` pointed at a different path                        → the identity test
  S4  a new optional field added without regenerating              → the equality test
"""
import json
from pathlib import Path

from src.schemas.probe_definition import PROBE_DEFINITION_KIND, ProbeDefinitionV1

PUBLISHED = (
    Path(__file__).resolve().parents[3] / "docs" / "schemas" / "probe-definition-v1.json"
)
SCHEMA_ID = (
    "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/"
    "probe-definition-v1.json"
)


def _generate() -> dict:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "miStudio Probe Definition v1",
        "description": (
            "Portable probe monitor (mistudio.probe-definition/v1) — a linear readout over one "
            "decoder layer's residual stream (or a k-sparse SAE basis) plus its normalisation, "
            "combining rule, operating point, evidence rung and reproducible test vectors. "
            "Generated from the pydantic contract in "
            "backend/src/schemas/probe_definition.py; regenerate via "
            "backend/tests/unit/test_probe_definition_schema_sync.py."
        ),
        "$ref": "#/$defs/ProbeDefinitionV1",
        "$defs": {},
    }
    definition = ProbeDefinitionV1.model_json_schema(ref_template="#/$defs/{model}")
    defs = definition.pop("$defs", {})
    schema["$defs"] = {**defs, "ProbeDefinitionV1": definition}
    return schema


def test_the_published_schema_matches_the_contract():
    assert PUBLISHED.exists(), f"published schema missing: {PUBLISHED}"
    published = json.loads(PUBLISHED.read_text())
    assert published == _generate(), (
        "docs/schemas/probe-definition-v1.json is out of sync with "
        "src/schemas/probe_definition.py — regenerate per this module's docstring and REVIEW THE "
        "DIFF before committing; a consumer validates against the published file"
    )


def test_the_schema_id_points_at_the_public_mirror():
    """`hitsainet/miStudio` is the public repo; `Onegaishimas/miStudio` is private, so a `$id`
    pointing there resolves to a 404 for every external consumer."""
    published = json.loads(PUBLISHED.read_text())
    assert published["$id"] == SCHEMA_ID
    assert "hitsainet" in published["$id"]
    assert "Onegaishimas" not in published["$id"]


def test_the_kind_string_carries_the_major_version():
    """A consumer dispatches on `kind` BEFORE parsing, so the version has to be in it."""
    published = json.loads(PUBLISHED.read_text())
    kind = published["$defs"]["ProbeDefinitionV1"]["properties"]["kind"]
    assert kind["const"] == PROBE_DEFINITION_KIND or PROBE_DEFINITION_KIND in kind.get("enum", [])
    assert PROBE_DEFINITION_KIND.endswith("/v1")


def test_the_schema_forbids_unknown_properties():
    """`extra="forbid"` has to survive into the published schema, or a typo'd field is silently
    accepted by the very consumers this file exists for."""
    published = json.loads(PUBLISHED.read_text())
    root = published["$defs"]["ProbeDefinitionV1"]
    assert root.get("additionalProperties") is False, (
        "the published schema permits unknown properties; a misspelled field would validate"
    )


def test_every_nested_model_also_forbids_extras(subtests=None):
    published = json.loads(PUBLISHED.read_text())
    permissive = [
        name
        for name, body in published["$defs"].items()
        if body.get("type") == "object" and body.get("additionalProperties") is not False
    ]
    assert not permissive, f"these nested models permit unknown properties: {permissive}"


if __name__ == "__main__":
    PUBLISHED.write_text(json.dumps(_generate(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {PUBLISHED}")
