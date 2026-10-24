"""miDataworks' rubric schema is VENDORED byte-identical (Feature 037 FR-15b; contract gap C2).

miDataworks OWNS `docs/schemas/midataworks-rubric-v1.json` (`midataworks.rubric/v1`, its 015 FR-015.18).
miStudio carries one copy, the package copy the image reads, beside the dataset-version schema (IDL-55):
`backend/src/schemas/vendored/midataworks-rubric-v1.json`. It is copied, never regenerated or edited
here. Part A's rubric import keeps reading documents defensively (IDL-57 item 7); validating imports
against this copy is not added by 037.

Three states for the cross-repo comparison, through `MIDATAWORKS_REPO` (default: the `miDataworks`
sibling of this checkout): present → the bytes must match; absent → a LOUD skip; absent with
`MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1` → fail. On the public mirror the cross-repo test skips by design;
the pinned digest below still runs there.

Push order: miDataworks first (published at miDataworks 5ed78a1), then this copy.
Control (0xcc/reviews/037_038_implementation_controls_2026-10-21.md, C-RUBRICBYTE): one byte of the
package copy changed → both byte tests red.
"""
from __future__ import annotations

import hashlib
import os
import pathlib

import pytest

from tests.support.sibling_repos import midataworks_repo

REPO = pathlib.Path(__file__).resolve().parents[3]
NAME = "midataworks-rubric-v1.json"
PACKAGE_COPY = REPO / "backend" / "src" / "schemas" / "vendored" / NAME
FLAG = "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"
#: The digest of the file miDataworks published (5ed78a1). Re-vendoring changes it here, on purpose.
PUBLISHED_SHA256 = "e3bc402314162f46fc4c0b5b15da6f57ccdfadbaf866d4ce1b650a0f3fbe77ee"


def _compare_with_sibling(root: pathlib.Path) -> None:
    theirs = root / "docs" / "schemas" / NAME
    if not theirs.is_file():
        message = (f"miDataworks' {NAME} is not at {theirs} (set MIDATAWORKS_REPO to a miDataworks "
                   f"checkout). Skipping is expected on the public mirror; set {FLAG}=1 to fail instead.")
        if os.environ.get(FLAG) == "1":
            pytest.fail(message)
        pytest.skip(message)
    assert PACKAGE_COPY.read_bytes() == theirs.read_bytes(), (
        f"miStudio's vendored {NAME} differs from miDataworks'. Copy it FROM {theirs} INTO "
        f"{PACKAGE_COPY}; never edit or regenerate it here — miDataworks owns the contract."
    )


def test_the_package_copy_is_the_published_file():
    assert hashlib.sha256(PACKAGE_COPY.read_bytes()).hexdigest() == PUBLISHED_SHA256


def test_it_is_byte_identical_to_miDataworks():
    _compare_with_sibling(midataworks_repo())


def test_an_absent_sibling_SKIPS_loudly_and_FAILS_under_the_flag(tmp_path, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(pytest.skip.Exception):
        _compare_with_sibling(tmp_path)
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(pytest.fail.Exception):
        _compare_with_sibling(tmp_path)


def test_it_is_the_rubric_format_with_an_optional_behaviour():
    import json

    schema = json.loads(PACKAGE_COPY.read_text())
    text = json.dumps(schema)
    assert "midataworks.rubric/v1" in text
    assert "behaviour" in text
