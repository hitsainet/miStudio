"""`docs/schemas/probe-result-v1.json` is miLLM's `mi.probe.result/v1`, vendored BYTE-IDENTICALLY
(miLLM 038 FR-38.28, FTASKS 6.1; HANDOFF_PLAN §6 step 4).

miLLM OWNS this schema: it is generated there from `millm/api/schemas/probe_result.py` and published
at `docs/schemas/probe-result-v1.json`. miStudio is a consumer — its MCP tools pass the contract
objects through (`millm_generate`, `millm_score_chat`, `millm_score_probes`, `millm_probe_events`,
`millm_get_probe_event`), and `test_mcp_millm_probe_results.py` validates its miLLM stub's objects
against THIS copy, so the stub cannot drift from the contract unnoticed.

NEVER REGENERATE OR HAND-EDIT IT HERE. When miLLM moves the schema, copy its file across unchanged
(`cp ../miLLM/docs/schemas/probe-result-v1.json docs/schemas/`) in the same release. The pin is red
in whichever repo moves second, by design: from miLLM's S1 push until this copy lands it was red on
any machine with miLLM checked out beside this repo.

Skips LOUDLY when the sibling is absent (the public mirror has no sibling), and FAILS under
`MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1` — an unverified pin is not a verified one.

Controls (038 S3-M, recorded in `0xcc/reviews/038_comiStudio_controls_2026-10-23.md`):
  V1 append one byte to the vendored copy → `test_byte_identical_to_millms_published_schema` red
  V2 point the sibling at an empty directory with the env flag set → red, not skipped
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.support.sibling_repos import REPO_ROOT, millm_repo

VENDORED = REPO_ROOT / "docs" / "schemas" / "probe-result-v1.json"
#: The contract's identity, stated once so a re-vendor of a DIFFERENT document is caught even on
#: the public mirror, where the byte pin skips.
TITLE = "mi.probe.result/v1"
SCHEMA_ID = "https://hitsai.net/schemas/millm/probe-result-v1.json"


def _theirs() -> Path:
    return millm_repo() / "docs" / "schemas" / "probe-result-v1.json"


def test_byte_identical_to_millms_published_schema():
    theirs = _theirs()
    if not theirs.exists():
        if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(
                f"{theirs} is absent and MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1. miLLM publishes "
                "probe-result-v1.json (038 FR-38.28); set MILLM_REPO to its checkout."
            )
        pytest.skip(f"miLLM's published schema is not present ({theirs}); set MILLM_REPO to check it")
    assert VENDORED.read_bytes() == theirs.read_bytes(), (
        "docs/schemas/probe-result-v1.json differs from miLLM's published copy. miLLM owns it: copy "
        "its file across byte-identically in the same release (never regenerate or edit it here)."
    )


def test_the_vendored_copy_is_the_probe_result_contract():
    """Runs everywhere, the mirror included: the file is present and is the document it claims."""
    assert VENDORED.is_file(), f"vendored schema missing: {VENDORED}"
    schema = json.loads(VENDORED.read_text(encoding="utf-8"))
    assert schema["title"] == TITLE
    assert schema["$id"] == SCHEMA_ID
    assert schema["discriminator"]["propertyName"] == "kind"
    assert set(schema["discriminator"]["mapping"]) == {"probe", "rule"}
    # The fields the miStudio tools promise an agent reaches it unchanged (FR-38.27).
    props = schema["$defs"]["ProbeResult"]["properties"]
    for field in ("result_id", "trace_id", "outcome", "reason", "decision", "detector_digest",
                  "persistence", "window", "verdict"):
        assert field in props, f"{field} is not in the vendored ProbeResult"
