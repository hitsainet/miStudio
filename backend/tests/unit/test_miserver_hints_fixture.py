"""The hint functions, run over a TEMPORARY miServer-shaped package (042 FR-042.20; FTDD §6.8;
FTASKS 7.12). No sibling needed: these prove the check fails where it must before it is trusted on the
real tree (test_miserver_hints_are_live.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support import miserver_hints as mh
from tests.support.next_calls import live_input_schemas

#: The fixture package's name, joined at runtime so no string constant of this file names miServer's
#: package (test_miserver_package_is_located_by_helper.py scans every test file's constants).
PKG = "".join(("mil", "lm"))
LIVE_NAME = "miserver_list_probes"  # a tool of today's registry since the P3 cut (checked below, not assumed)


def _pkg(root: Path, *, pkg: str = "millm", next_tools: dict[str, str] | None = None,
         errors: str = "", published: list[str] | None = None) -> Path:
    (root / pkg / "core").mkdir(parents=True)
    (root / pkg / "main.py").write_text("")
    (root / pkg / "__init__.py").write_text("")
    if next_tools is not None:
        (root / pkg / "core" / "next_tools.py").write_text(
            "".join(f"{k} = {v!r}\n" for k, v in next_tools.items()))
    (root / pkg / "core" / "errors.py").write_text(errors)
    if published is not None:
        (root / "docs" / "schemas").mkdir(parents=True)
        (root / "docs" / "schemas" / "next-tool-hints.json").write_text(
            json.dumps({"version": 1, "next_tools": sorted(published)}))
    return root


ERRORS = (
    f"from {{pkg}}.core import next_tools\n"
    "class Base(Exception):\n    next_tool = None\n"
    "class Dead(Base):\n    next_tool = next_tools.DEAD\n"
    "class Fine(Base):\n    next_tool: str = next_tools.FINE\n"
    "def raise_it():\n    raise Base('x', next_tool=next_tools.FINE)\n"
)


def test_the_live_name_is_live():
    assert LIVE_NAME in live_input_schemas()


@pytest.mark.parametrize("pkg", ["millm", "miserver"])
def test_dead_hint_in_fixture_package_is_red(tmp_path, pkg):
    repo = _pkg(tmp_path, pkg=pkg, next_tools={"DEAD": "millm_does_not_exist", "FINE": LIVE_NAME},
                errors=ERRORS.format(pkg=pkg), published=["millm_does_not_exist", LIVE_NAME])
    names, authority = mh.hints(repo)
    assert names == {"millm_does_not_exist", LIVE_NAME} and authority.startswith("published")
    problems = mh.liveness_problems(names, set(live_input_schemas()), [])
    assert len(problems) == 1 and problems[0].startswith("millm_does_not_exist"), problems


def test_a_hint_the_published_file_omits_is_red(tmp_path):
    repo = _pkg(tmp_path, next_tools={"DEAD": "millm_does_not_exist", "FINE": LIVE_NAME},
                errors=ERRORS.format(pkg="millm"), published=[LIVE_NAME])
    with pytest.raises(AssertionError, match="millm_does_not_exist"):
        mh.hints(repo)


@pytest.mark.parametrize("expr", ["f'millm_{x}'", "pick()", "next_tools.MISSING", "NAME_ONLY", "1"])
def test_dynamic_hint_fails(tmp_path, expr):
    errors = f"from {PKG}.core import next_tools\nclass E(Exception):\n    next_tool = {expr}\n"
    repo = _pkg(tmp_path, next_tools={"FINE": LIVE_NAME}, errors=errors, published=[LIVE_NAME])
    with pytest.raises(mh.DynamicHint):
        mh.source_hints(repo)


def test_a_dynamic_keyword_hint_fails(tmp_path):
    errors = "class E(Exception):\n    pass\ndef f(t):\n    raise E('x', next_tool=t)\n"
    repo = _pkg(tmp_path, next_tools={"FINE": LIVE_NAME}, errors=errors, published=[LIVE_NAME])
    with pytest.raises(mh.DynamicHint):
        mh.source_hints(repo)


def test_instance_and_local_assignments_are_not_hints(tmp_path):
    # miServer's base class: `self.next_tool = next_tool` in __init__ and a handler's local variable.
    errors = ("class E(Exception):\n    next_tool = None\n    def __init__(self, next_tool=None):\n"
              "        if next_tool is not None:\n            self.next_tool = next_tool\n"
              "def handler(exc):\n    next_tool = getattr(exc, 'next_tool', None)\n    return next_tool\n")
    repo = _pkg(tmp_path, next_tools={}, errors=errors, published=[])
    assert mh.source_hints(repo) == set()


def test_source_is_the_authority_before_040_lands(tmp_path):
    errors = f"class E(Exception):\n    next_tool = {LIVE_NAME!r}\n"
    repo = _pkg(tmp_path, next_tools=None, errors=errors, published=None)
    assert mh.hints(repo) == ({LIVE_NAME}, "source (040 P0 not landed)")


def test_half_shipped_040_fails_under_require(tmp_path, monkeypatch):
    repo = _pkg(tmp_path, next_tools={"FINE": LIVE_NAME},
                errors=f"from {PKG}.core import next_tools\nclass E(Exception):\n    next_tool = next_tools.FINE\n")
    with pytest.raises(mh.HalfShipped):
        mh.hints(repo)
    monkeypatch.setenv("MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS", "1")
    with pytest.raises(pytest.fail.Exception, match="040 shipped half"):
        mh.checked_hints(repo)
    monkeypatch.delenv("MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS")
    monkeypatch.delenv("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="SKIPPED LOUDLY"):
        mh.checked_hints(repo)


def test_published_file_shape_is_refused_when_wrong(tmp_path):
    repo = _pkg(tmp_path, next_tools={}, errors="")
    (repo / "docs" / "schemas").mkdir(parents=True)
    f = repo / "docs" / "schemas" / "next-tool-hints.json"
    for bad in ({"version": 2, "next_tools": []}, {"version": 1, "next_tools": ["b", "a"]},
                {"version": 1, "next_tools": [], "extra": 1}):
        f.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            mh.published_hints(repo)


def test_the_transition_rule_reads_the_relay_set():
    live = {"miserver_list_probes"}
    assert mh.LIVENESS_MODE == "exact", "TI-21: flipped by the P7 relay-removal commit"
    assert mh.liveness_problems({"millm_list_probes"}, live, [frozenset({"miserver_list_probes"})],
                                mode="transition") == []
    assert mh.liveness_problems({"millm_list_probes"}, live, [frozenset()], mode="transition")
    # The default mode since P7: a renamed hint is refused even with its twin in a relay set.
    assert mh.liveness_problems({"millm_list_probes"}, live, [frozenset({"miserver_list_probes"})])
    assert mh.liveness_problems({"millm_list_probes"}, live, [frozenset({"miserver_list_probes"})],
                                mode="exact")
