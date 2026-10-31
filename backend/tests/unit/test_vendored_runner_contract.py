"""miDataworks' runner contract and runner cases are VENDORED byte-identical (Feature 040 S4; FTDD Appendix C.2;
FTASKS 25.2; spec D29, D47).

miDataworks 018 OWNS `docs/schemas/midataworks-runner-v1.json` (the contract: guest bundle by digest, limits floor,
WASI table, reply grammar, reasons, hostile classes) and `runner-cases-v1.json` (conformance, hostile, hash, import and
limit vectors). miStudio carries two copies of each, both copied, never regenerated:

* `docs/schemas/…` — the vendored contract, compared with miDataworks' file;
* `backend/src/schemas/vendored/…` — the PACKAGE copy miStudio's runner host (`src/rule_runner/`) and its image read
  (the backend image's build context is `backend/`, so `docs/` never reaches it — the MIS-E2E-028 trap).

Three states for the cross-repo comparison, through `MIDATAWORKS_REPO` (default: the `miDataworks` sibling,
`tests/support/sibling_repos.py`): present → the bytes must match; absent → a LOUD skip naming the path and
the flag; absent with `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1` → fail. On the public mirror the cross-repo test
skips by design; the in-repo identity test still runs there.

D47: the contract states IN THE FILE that its floors are provisional (`limits_floor_status.provisional`) and that
raising one is allowed; miStudio honours that (its `code_rule_*` settings may only be at or above the floor), and the
last tests pin that the file says so and that the runner host reads the package copies.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import pytest

from tests.support.sibling_repos import midataworks_repo

REPO = pathlib.Path(__file__).resolve().parents[3]
BACKEND = REPO / "backend"
NAMES = ("midataworks-runner-v1.json", "runner-cases-v1.json")
PINNED = {
    "midataworks-runner-v1.json": "7a9319b63bf3862c0b313035debda3a6f75e57d088a58217fde5aaf5088ea0bf",
    "runner-cases-v1.json": "e6e0b4b80be0cc28b157f5b28150c0f7156730a5fb2e0fa702bbbf175a036052",
}
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


@pytest.mark.parametrize("name", NAMES)
def test_the_vendored_bytes_are_the_ones_this_slice_was_built_against(name):
    """A re-vendor is a deliberate act: it moves this pin, the runner's identity (`engine_digest`) and the image."""
    assert hashlib.sha256(package_copy(name).read_bytes()).hexdigest() == PINNED[name]


def test_the_floor_is_stated_provisional_and_raising_is_allowed():
    """D47: the file itself says its floors are provisional; within v1 a floor is raised, never lowered."""
    contract = json.loads(package_copy(NAMES[0]).read_text(encoding="utf-8"))
    status = contract["limits_floor_status"]
    assert status["provisional"] is True
    assert "never lowered" in status["change_rule"]
    assert set(contract["limits_floor"]) >= {"call_fuel", "load_fuel", "prelude_fuel", "memory_bytes",
                                             "input_code_points", "output_bytes"}


def test_the_runner_host_reads_the_package_copies():
    from src.rule_runner import contract

    assert contract.CONTRACT_PATH.resolve() == package_copy(NAMES[0]).resolve()
    assert contract.CASES_PATH.resolve() == package_copy(NAMES[1]).resolve()
    assert contract.load_contract()["format"] == "midataworks.runner/v1"
