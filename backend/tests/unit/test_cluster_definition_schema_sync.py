"""
Schema-sync test (Feature 014, Task 2.1): the published JSON Schema at
docs/schemas/cluster-definition-v1.json must stay in lockstep with the
pydantic contract in src/schemas/cluster_profile.py.

If this test fails, regenerate the published schema (see REGEN below), review
the diff for breaking changes, and bump the schema_version if the change is
not backward-compatible.
"""

import json
from pathlib import Path

from src.schemas.cluster_profile import ClusterBundleV1, ClusterDefinitionV1

PUBLISHED = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "cluster-definition-v1.json"

# REGEN: python -c of this exact construction, dumped with indent=2, sort_keys=True.


def _generate() -> dict:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/cluster-definition-v1.json",
        "title": "miStudio Cluster Definition v1",
        "description": (
            "Portable cluster definition (mistudio.cluster-definition/v1) — the consumer-neutral "
            "interchange artifact for cluster profiles (IDL-30). Also includes the bundle wrapper "
            "(mistudio.cluster-bundle/v1) under $defs. Generated from the pydantic contract; "
            "regenerate via backend/tests/unit/test_cluster_definition_schema_sync.py instructions."
        ),
        "oneOf": [
            {"$ref": "#/$defs/ClusterDefinitionV1"},
            {"$ref": "#/$defs/ClusterBundleV1"},
        ],
        "$defs": {},
    }
    def_schema = ClusterDefinitionV1.model_json_schema(ref_template="#/$defs/{model}")
    bundle_schema = ClusterBundleV1.model_json_schema(ref_template="#/$defs/{model}")
    for s in (def_schema, bundle_schema):
        defs = s.pop("$defs", {})
        schema["$defs"].update(defs)
    schema["$defs"]["ClusterDefinitionV1"] = def_schema
    schema["$defs"]["ClusterBundleV1"] = bundle_schema
    return schema


def test_published_schema_matches_pydantic_contract():
    assert PUBLISHED.exists(), f"Published schema missing: {PUBLISHED}"
    published = json.loads(PUBLISHED.read_text())
    assert published == _generate(), (
        "docs/schemas/cluster-definition-v1.json is out of sync with "
        "src/schemas/cluster_profile.py — regenerate it (docstring has instructions) "
        "and review the diff for consumer-breaking changes."
    )
