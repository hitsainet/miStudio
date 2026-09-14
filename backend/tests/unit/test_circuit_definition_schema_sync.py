"""
Schema-sync test (018 Task 2.1, IDL-33): docs/schemas/circuit-definition-v1.json
must stay in lockstep with src/schemas/circuit_definition.py.

REGEN: run this file's _generate() and dump with indent=2, sort_keys=True
(python -m tests.unit.test_circuit_definition_schema_sync).
"""

import json
from pathlib import Path

from src.schemas.circuit_definition import CircuitDefinitionV1

PUBLISHED = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "circuit-definition-v1.json"


def _generate() -> dict:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/circuit-definition-v1.json",
        "title": "miStudio Circuit Definition v1",
        "description": (
            "Portable circuit definition (mistudio.circuit-definition/v1) — cross-layer "
            "feature circuits with evidence rungs, typed edges, attribution scores, "
            "validation-manifest references, and Tier-2.5-ready position fields (IDL-33). "
            "Generated from the pydantic contract; regenerate via "
            "backend/tests/unit/test_circuit_definition_schema_sync.py instructions."
        ),
        "$ref": "#/$defs/CircuitDefinitionV1",
        "$defs": {},
    }
    def_schema = CircuitDefinitionV1.model_json_schema(ref_template="#/$defs/{model}")
    defs = def_schema.pop("$defs", {})
    schema["$defs"] = {**defs, "CircuitDefinitionV1": def_schema}
    return schema


def test_published_schema_matches_pydantic_contract():
    assert PUBLISHED.exists(), f"Published schema missing: {PUBLISHED}"
    published = json.loads(PUBLISHED.read_text())
    assert published == _generate(), (
        "docs/schemas/circuit-definition-v1.json is out of sync with the pydantic "
        "contract — regenerate per the module docstring and review the diff."
    )


if __name__ == "__main__":
    PUBLISHED.write_text(json.dumps(_generate(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {PUBLISHED}")
