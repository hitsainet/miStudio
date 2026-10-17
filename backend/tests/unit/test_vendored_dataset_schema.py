"""miDataworks' dataset-version schema is VENDORED byte-identical, never generated (Feature 034,
FR-24/FR-25/FR-27; SC-9).

miDataworks OWNS `docs/schemas/midataworks-dataset-version-v1.json` (C9). miStudio carries two
copies of it, both copied from miDataworks and never regenerated here:

* `docs/schemas/…` — the vendored contract, compared with miDataworks' file;
* `backend/src/schemas/vendored/…` — the PACKAGE copy the image reads. The backend's build context
  is `backend/`, so `docs/` never reaches the image (the MIS-E2E-028 trap).

Three states for the cross-repo comparison, through `MIDATAWORKS_REPO` (default: the `miDataworks`
sibling of this checkout, `tests/support/sibling_repos.py`):

* present → the bytes must match;
* absent (or present without the file) → a LOUD skip naming the missing path and the flag;
* absent with `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1` → fail.

⚠ ON THE PUBLIC MIRROR (`hitsainet/miStudio`) THE CROSS-REPO TEST SKIPS BY DESIGN: miDataworks is
not checked out there. Do not "fix" that skip — the in-repo identity test below still runs there,
and `sync-to-clean.yml` keeps `docs/schemas/` (asserted below).

Controls (recorded 2026-10-16, task 3.7; each red, each restore verified):
  * change one byte of the package copy → `test_the_package_copy_equals_the_docs_copy` and
    `test_it_is_byte_identical_to_miDataworks` red;
  * point `MIDATAWORKS_REPO` at an empty directory with the flag set →
    `test_an_absent_sibling_FAILS_under_the_flag` asserts the failure path (and the live test
    fails rather than skips).
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

from tests.support.sibling_repos import midataworks_repo

REPO = pathlib.Path(__file__).resolve().parents[3]
BACKEND = REPO / "backend"
NAME = "midataworks-dataset-version-v1.json"
DOCS_COPY = REPO / "docs" / "schemas" / NAME
PACKAGE_COPY = BACKEND / "src" / "schemas" / "vendored" / NAME
FLAG = "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"


def _theirs(root: pathlib.Path) -> pathlib.Path:
    return root / "docs" / "schemas" / NAME


def _compare_with_sibling(root: pathlib.Path) -> None:
    """The cross-repo check, factored so its absent-sibling branch is itself testable."""
    theirs = _theirs(root)
    if not theirs.is_file():
        message = (
            f"miDataworks' {NAME} is not at {theirs} (set MIDATAWORKS_REPO to a miDataworks "
            f"checkout). Skipping is expected on the public mirror; set {FLAG}=1 to make this "
            f"a failure."
        )
        if os.environ.get(FLAG) == "1":
            pytest.fail(message)
        pytest.skip(message)
    assert PACKAGE_COPY.read_bytes() == theirs.read_bytes(), (
        f"miStudio's vendored {NAME} differs from miDataworks'. Copy miDataworks' file FROM "
        f"{theirs} INTO {DOCS_COPY} and {PACKAGE_COPY} — never edit or regenerate it here; "
        "miDataworks owns the contract."
    )


def test_the_package_copy_equals_the_docs_copy():
    assert DOCS_COPY.is_file() and PACKAGE_COPY.is_file()
    assert DOCS_COPY.read_bytes() == PACKAGE_COPY.read_bytes(), (
        "the image reads the package copy and the contract is the docs copy — they must be the "
        "same bytes, both copied from miDataworks"
    )


def test_it_is_byte_identical_to_miDataworks():
    # Skips on hitsainet/miStudio by design (no sibling there) — see the module docstring.
    _compare_with_sibling(midataworks_repo())


def test_an_absent_sibling_SKIPS_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(pytest.skip.Exception) as skipped:
        _compare_with_sibling(tmp_path)
    assert str(_theirs(tmp_path)) in str(skipped.value) and FLAG in str(skipped.value)


def test_an_absent_sibling_FAILS_under_the_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(pytest.fail.Exception):
        _compare_with_sibling(tmp_path)


def test_a_sibling_WITHOUT_the_file_counts_as_absent(tmp_path, monkeypatch):
    (tmp_path / "docs" / "schemas").mkdir(parents=True)
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(pytest.fail.Exception):
        _compare_with_sibling(tmp_path)


def test_the_failure_says_copy_FROM_miDataworks(tmp_path, monkeypatch):
    theirs = _theirs(tmp_path)
    theirs.parent.mkdir(parents=True)
    theirs.write_bytes(PACKAGE_COPY.read_bytes() + b" ")
    with pytest.raises(AssertionError, match="FROM"):
        _compare_with_sibling(tmp_path)


def test_nothing_in_src_WRITES_either_copy():
    """FR-24: vendored, never generated. No call in `backend/src` writes to a path that names the
    file (AST: a `write_text`/`write_bytes`/`open(..., "w")`/`json.dump` in a module that also
    names it)."""
    offenders = []
    for path in sorted((BACKEND / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if NAME not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("write_text", "write_bytes", "dump"):
                offenders.append(f"{path.relative_to(BACKEND)}:{node.lineno}")
            if name == "open" and any(
                isinstance(a, ast.Constant) and isinstance(a.value, str) and "w" in a.value
                for a in node.args[1:]
            ):
                offenders.append(f"{path.relative_to(BACKEND)}:{node.lineno}")
    assert not offenders, offenders


def test_the_dockerignore_does_not_exclude_the_package_copy():
    ignore = (BACKEND / ".dockerignore").read_text().splitlines()
    patterns = [line.strip() for line in ignore if line.strip() and not line.startswith("#")]
    for pattern in patterns:
        bare = pattern.strip("/")
        assert bare not in ("src", "src/schemas", "src/schemas/vendored", "*.json", NAME), pattern


def test_the_mirror_keeps_docs_schemas():
    """FR-27: `sync-to-clean.yml` strips `docs/` except `schemas` and verifies it survives."""
    workflow = (REPO / ".github" / "workflows" / "sync-to-clean.yml").read_text()
    assert "! -name schemas" in workflow
    assert "for p in docs/schemas" in workflow


def test_the_runtime_reads_the_PACKAGE_copy():
    from src.services import dataset_version_manifest as m

    assert m.SCHEMA_FILE == NAME and m.SCHEMA_PACKAGE == "src.schemas.vendored"
    assert m.load_schema() == __import__("json").loads(PACKAGE_COPY.read_text())
