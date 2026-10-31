"""The text a rule reads equals miLLM's serving windows (Feature 040 FR-33; FTDD §6.2; FTASKS 18.1, 18.4).

`docs/schemas/rule-text-reduction-cases.json` is OWNED by miLLM 039 (it produces the serving windows the text
must equal) and vendored here byte-identically, with a package copy the engine reads. Three states against the
owner through `MILLM_REPO` (default: the `miLLM` sibling): present → byte-equal; absent → loud skip; absent with
`MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1` → fail.

Every case runs through miStudio's own reducer (`rule_signals.reduce.window_texts`, built on `reduce`), never a
copy of miLLM's function. FTASKS 18.4's control: join the messages with ``""`` instead of ``"\\n"`` → red.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from src.rule_signals import reduce as reducer
from tests.support.sibling_repos import millm_repo

REPO = pathlib.Path(__file__).resolve().parents[3]
NAME = "rule-text-reduction-cases.json"
DOCS_COPY = REPO / "docs" / "schemas" / NAME
PACKAGE_COPY = REPO / "backend" / "src" / "schemas" / "vendored" / NAME
FLAG = "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"
CASES = json.loads(PACKAGE_COPY.read_text(encoding="utf-8"))


def _compare_with_sibling(root: pathlib.Path) -> None:
    theirs = root / "docs" / "schemas" / NAME
    if not theirs.is_file():
        message = (f"miLLM's {NAME} is not at {theirs} (set MILLM_REPO to a miLLM checkout). Skipping is "
                   f"expected on the public mirror; set {FLAG}=1 to make this a failure.")
        if os.environ.get(FLAG) == "1":
            pytest.fail(message)
        pytest.skip(message)
    assert PACKAGE_COPY.read_bytes() == theirs.read_bytes(), (
        f"miStudio's vendored {NAME} differs from miLLM's; copy miLLM's file into {DOCS_COPY} and {PACKAGE_COPY}")


def test_the_package_copy_equals_the_docs_copy():
    assert DOCS_COPY.read_bytes() == PACKAGE_COPY.read_bytes()


def test_it_is_byte_identical_to_miLLM():
    _compare_with_sibling(millm_repo())


def test_an_absent_sibling_SKIPS_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(pytest.skip.Exception) as skipped:
        _compare_with_sibling(tmp_path)
    assert FLAG in str(skipped.value)


def test_an_absent_sibling_FAILS_under_the_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(pytest.fail.Exception):
        _compare_with_sibling(tmp_path)


def test_the_format_is_the_one_this_reducer_reads():
    assert CASES["format"] == "millm.rule-text-reduction-cases/v1"
    assert len(CASES["cases"]) >= 10


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["id"])
def test_every_case(case):
    data = case["input"]
    texts, missing = reducer.window_texts(messages=data.get("messages"), prompt=data.get("prompt"),
                                          reply_text=data.get("reply_text"))
    assert texts == case["expect"]["texts"]
    assert missing == case["expect"]["missing"]


def test_each_window_is_the_render_modules_scope():
    """The contract-window → internal-scope map is the render module's (one definition, two readers)."""
    from src.services.probe_monitor_render import CONTRACT_WINDOW_SCOPES

    assert reducer.CONTRACT_WINDOW_SCOPES == CONTRACT_WINDOW_SCOPES
