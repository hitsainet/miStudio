"""Schema sync for the J-space interchange kinds (027 tasks 4.1/4.2).

`docs/schemas/jspace-contracts-v1.json` publishes the four kinds declared in
`src/schemas/jspace_contracts.py`, so a consumer (miLLM's mirror, per 027 FTDD §"The mirror
drifts") can pin a byte-identical copy. This test regenerates it from the pydantic models
and fails on any difference.

REGEN: python -m tests.unit.test_jspace_contracts_schema_sync
"""

import json
from pathlib import Path

from src.schemas.jspace_contracts import (
    JLensArtifactDocument,
    ReadoutRecordDocument,
    WatchlistDocument,
    WorkspaceAnnotationDocument,
)

PUBLISHED = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "jspace-contracts-v1.json"

MODELS = {
    "mistudio.jlens-artifact/v1": JLensArtifactDocument,
    "mistudio.workspace-annotation/v1": WorkspaceAnnotationDocument,
    "mistudio.jlens-readout/v1": ReadoutRecordDocument,
    "mistudio.jspace-watchlist/v1": WatchlistDocument,
}


def _generate() -> dict:
    defs: dict = {}
    refs = []
    for kind, model in MODELS.items():
        schema = model.model_json_schema(ref_template="#/$defs/{model}")
        defs.update(schema.pop("$defs", {}))
        defs[model.__name__] = schema
        refs.append({"$ref": f"#/$defs/{model.__name__}"})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/jspace-contracts-v1.json",
        "title": "miStudio J-space interchange kinds v1",
        "description": (
            "The four J-space kinds (BR-021/BR-022): jlens-artifact/v1 (Track A, describes a "
            "mounted directory, no weights), workspace-annotation/v1 (Track B), "
            "jlens-readout/v1 and jspace-watchlist/v1. The version lives in each `kind` "
            "identifier: a consumer must REJECT an unknown kind rather than read a newer "
            "version as this one. Generated from src/schemas/jspace_contracts.py."
        ),
        "oneOf": refs,
        "$defs": defs,
    }


def test_published_schema_matches_the_pydantic_contract():
    assert PUBLISHED.exists(), f"Published schema missing: {PUBLISHED}"
    assert json.loads(PUBLISHED.read_text()) == _generate(), (
        "docs/schemas/jspace-contracts-v1.json is out of sync with "
        "src/schemas/jspace_contracts.py — regenerate per the module docstring"
    )


def test_every_kind_the_parser_reads_is_published():
    from src.services.jspace_documents import DOCUMENT_MODELS

    assert set(DOCUMENT_MODELS) == set(MODELS)


if __name__ == "__main__":
    PUBLISHED.write_text(json.dumps(_generate(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {PUBLISHED}")
