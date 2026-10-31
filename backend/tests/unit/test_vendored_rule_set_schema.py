"""miDataworks' rule-set schema and engine vectors are VENDORED byte-identical (Feature 040 FR-28; FTASKS 18.1).

miDataworks 016 OWNS `docs/schemas/midataworks-rule-set-v1.json` and `rule-signal-cases-v1.json`. miStudio
carries two copies of each, both copied, never regenerated:

* `docs/schemas/…` — the vendored contract, compared with miDataworks' file;
* `backend/src/schemas/vendored/…` — the PACKAGE copy the engine reads (the image's build context is
  `backend/`, so `docs/` never reaches it — the MIS-E2E-028 trap).

Three states for the cross-repo comparison, through `MIDATAWORKS_REPO` (default: the `miDataworks` sibling,
`tests/support/sibling_repos.py`): present → the bytes must match; absent → a LOUD skip naming the path and
the flag; absent with `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1` → fail. On the public mirror the cross-repo test
skips by design; the in-repo identity test still runs there.

The engine reads the PACKAGE copy (`rule_signals.document.SCHEMA_PATH`, `CASES_PATH`); the last test pins that.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from tests.support.sibling_repos import midataworks_repo

REPO = pathlib.Path(__file__).resolve().parents[3]
BACKEND = REPO / "backend"
NAMES = ("midataworks-rule-set-v1.json", "rule-signal-cases-v1.json")
FLAG = "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"


def docs_copy(name: str) -> pathlib.Path:
    return REPO / "docs" / "schemas" / name


def package_copy(name: str) -> pathlib.Path:
    return BACKEND / "src" / "schemas" / "vendored" / name


def _compare_with_sibling(root: pathlib.Path, name: str) -> None:
    theirs = root / "docs" / "schemas" / name
    if not theirs.is_file():
        message = (f"miDataworks' {name} is not at {theirs} (set MIDATAWORKS_REPO to a miDataworks checkout). "
                   f"Skipping is expected on the public mirror; set {FLAG}=1 to make this a failure.")
        if os.environ.get(FLAG) == "1":
            pytest.fail(message)
        pytest.skip(message)
    assert package_copy(name).read_bytes() == theirs.read_bytes(), (
        f"miStudio's vendored {name} differs from miDataworks'. Copy miDataworks' file FROM {theirs} INTO "
        f"{docs_copy(name)} and {package_copy(name)} — never edit it here; miDataworks owns the contract.")


@pytest.mark.parametrize("name", NAMES)
def test_the_package_copy_equals_the_docs_copy(name):
    assert docs_copy(name).is_file() and package_copy(name).is_file()
    assert docs_copy(name).read_bytes() == package_copy(name).read_bytes()


@pytest.mark.parametrize("name", NAMES)
def test_it_is_byte_identical_to_miDataworks(name):
    _compare_with_sibling(midataworks_repo(), name)


def test_an_absent_sibling_SKIPS_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(pytest.skip.Exception) as skipped:
        _compare_with_sibling(tmp_path, NAMES[0])
    assert FLAG in str(skipped.value) and NAMES[0] in str(skipped.value)


def test_an_absent_sibling_FAILS_under_the_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(pytest.fail.Exception):
        _compare_with_sibling(tmp_path, NAMES[1])


def test_one_changed_byte_is_red(tmp_path):
    """The pin compares bytes, not parsed JSON: a single flipped byte in a sibling copy fails it."""
    theirs = tmp_path / "docs" / "schemas"
    theirs.mkdir(parents=True)
    data = bytearray(package_copy(NAMES[0]).read_bytes())
    data[-2] = ord(" ") if data[-2] != ord(" ") else ord("\n")
    (theirs / NAMES[0]).write_bytes(bytes(data))
    with pytest.raises(AssertionError):
        _compare_with_sibling(tmp_path, NAMES[0])


def test_the_engine_reads_the_package_copies():
    from src.rule_signals import document

    assert document.SCHEMA_PATH.resolve() == package_copy(NAMES[0]).resolve()
    assert document.CASES_PATH.resolve() == package_copy(NAMES[1]).resolve()
    assert document.schema()["title"] == "miDataworks Rule Set v1"
