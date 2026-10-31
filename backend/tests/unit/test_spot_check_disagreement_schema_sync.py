"""`docs/schemas/spot-check-disagreement-v1.json` stays in lockstep with `DisagreementRow` (038 FR-11b, D57).

REGENERATE: `python -m tests.unit.test_spot_check_disagreement_schema_sync` from `backend/`, then READ THE
DIFF. miDataworks vendors the published file byte-identical (its ADR-028), so a regeneration nobody
reviewed is a contract change nobody reviewed. Additive only within v1; push order: miStudio first.

Control (C-ROWSCHEMA): a field added to `DisagreementRow` without regenerating → the equality test red.
"""
import json
from pathlib import Path

from src.schemas.spot_check import SPOT_CHECK_DISAGREEMENT_FORMAT, DisagreementRow

PUBLISHED = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "spot-check-disagreement-v1.json"
SCHEMA_ID = (
    "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/spot-check-disagreement-v1.json"
)


def _generate() -> dict:
    """The row's schema AT THE TOP LEVEL (`properties`, `required`), with `ChatMessage` under `$defs`.

    ⚠ TOP LEVEL, NOT A `$ref`: miDataworks derives the export's pinned provenance columns from the
    document's top-level `required` (its `services/sources/spot_check.pinned_columns`), so a `$ref`
    wrapper would publish a schema whose pinned columns read as none and switch recognition off.
    """
    body = DisagreementRow.model_json_schema(ref_template="#/$defs/{model}")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        **body,
        "title": "miStudio Spot-Check Disagreement Row v1",
    }


def test_the_published_schema_matches_the_model():
    assert PUBLISHED.exists(), f"published schema missing: {PUBLISHED}"
    assert json.loads(PUBLISHED.read_text()) == _generate(), (
        "docs/schemas/spot-check-disagreement-v1.json is out of sync with DisagreementRow — regenerate "
        "per this module's docstring and REVIEW THE DIFF; miDataworks vendors this file"
    )


def test_the_format_is_pinned_and_unknown_fields_are_refused():
    published = json.loads(PUBLISHED.read_text())
    root = published
    assert root["properties"]["format"]["const"] == SPOT_CHECK_DISAGREEMENT_FORMAT
    assert {"messages", "session_id", "request_id", "probe_fired", "judge_positive", "prompt_hash"} <= set(root["required"])
    assert root.get("additionalProperties") is False
    assert published["$id"] == SCHEMA_ID and "hitsainet" in published["$id"]
    for name, body in published.get("$defs", {}).items():
        if body.get("type") == "object":
            assert body.get("additionalProperties") is False, name


if __name__ == "__main__":
    PUBLISHED.write_text(json.dumps(_generate(), indent=2) + "\n")
    print(f"wrote {PUBLISHED}")
