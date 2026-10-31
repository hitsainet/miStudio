"""Every vendored engine vector passes against miStudio's engine, with ZERO mismatches (Feature 040 FR-28, FR-47 (b)).

`rule-signal-cases-v1.json` is miDataworks' (vendored byte-identically, pinned by
`test_vendored_rule_set_schema.py`). "The vector is the truth" (its own `precedence`): a case red here is a
miStudio defect, never a vector to edit. Every section runs: rule documents and their row cases, refusals
(pattern dialect and documents), renderings, every validator's value checks and scans, and the rule-set
identity (RFC 8785).

⚠ JSON EQUALITY WITH TYPES (`cases.same`): ``True`` is not ``1``. A loose ``==`` let ``1`` pass for ``true``
in miLLM 039's first round; `test_same_compares_json_types` pins it here.

One deliberate difference: under D41 / D44 (2026-10-24; FR-31 of the 040 FPRD amended accordingly) miStudio ACCEPTS a rule set naming operator code
(its dependent expressions are pending, `runner_unavailable`), where `refuse.document.code_rule` states the
producer's refusal at save. That one case is a STRICT xfail: the day the vector or the decision changes,
this file goes red. Under ENH-006 D65 (2026-10-25) `refuse.document.code_rule_in_within_n` is NOT in that set: a
positional read of a code rule is refused here exactly as the vector states (`code_rule_in_within_n`).

FR-28's removal control: the conformance test must call the PRODUCTION engine, not a stub. The AST test at
the bottom reads this file's own call sites and requires calls to `rule_signals.load`, `rule_signals.run_cases`
and `rule_signals.evaluate`.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

import src.rule_signals as rule_signals
from src.rule_signals import canonical, dialect, renderings
from src.rule_signals.cases import same
from src.rule_signals.validators import VALIDATORS, resolve

VECTORS = json.loads((pathlib.Path(rule_signals.__file__).resolve().parents[1] / "schemas" / "vendored"
                      / "rule-signal-cases-v1.json").read_text(encoding="utf-8"))
DOCUMENTS = VECTORS["rule_documents"]
LIMITS = rule_signals.Limits()
#: D41 / D44: miStudio accepts operator code (its 040 FPRD FR-31 amended per ENH-006 D44, 2026-10-24); the producer
#: refuses it at save. ENH-006 D65 (2026-10-25): a within_n over a code rule is refused, so
#: `refuse.document.code_rule_in_within_n` is checked as the refusal it states (never re-add it here).
D41_ACCEPTED = {"refuse.document.code_rule"}


def _row_cases():
    for kind, cases in VECTORS["rule_kinds"].items():
        for case in cases:
            if isinstance(case.get("document"), str):
                yield pytest.param(case, id=f"{kind}:{case['id']}")


def _refusals():
    for case in VECTORS["rule_kinds"]["refusals"]:
        marks = ()
        if case["id"] in D41_ACCEPTED:
            marks = (pytest.mark.xfail(strict=True, reason="ENH-006 D44 (040 FPRD FR-31 amended): operator code is accepted and pending"),)
        yield pytest.param(case, id=case["id"], marks=marks)


def _validator_cases():
    for ref, section in VECTORS["validators"].items():
        for case in section["cases"]:
            yield pytest.param(ref, case, id=case["id"])


@pytest.mark.parametrize("case", list(_row_cases()))
def test_row_case(case):
    compiled = rule_signals.load(DOCUMENTS[case["document"]], LIMITS)
    report = rule_signals.run_cases(compiled, [case])
    assert report.ok, [f.report() for f in report.failures]
    assert not report.unverified


@pytest.mark.parametrize("case", [p for p in _row_cases() if "text" in p.values[0]["input"]])
def test_row_case_through_the_single_text_path(case):
    """The SAME vectors through `evaluate` — the path the API route and the worker call (reduction
    included): its expression counts and `vetoed` must equal the case's."""
    compiled = rule_signals.load(DOCUMENTS[case["document"]], LIMITS)
    verdict = rule_signals.evaluate(compiled, LIMITS, text=case["input"]["text"])
    assert verdict.vetoed is case["expect"]["vetoed"]
    for name, spans in (case["expect"].get("spans") or {}).items():
        assert verdict.expressions[name]["count"] == len(spans), name


def test_every_row_case_document_is_published():
    named = {c["document"] for cases in VECTORS["rule_kinds"].values() for c in cases
             if isinstance(c.get("document"), str)}
    assert named <= set(DOCUMENTS) and named


@pytest.mark.parametrize("case", list(_refusals()))
def test_refusal(case):
    refuse = case["refuse"]
    if "pattern" in refuse:
        refused = dialect.scan(refuse["pattern"])
        assert refused is not None, "the dialect admitted a refused pattern"
        assert (refused.rule, refused.position) == (case["rule"], case["at"])
        return
    with pytest.raises(rule_signals.RuleSetRefused) as caught:
        rule_signals.load(refuse["document"], LIMITS)
    assert caught.value.code == case["rule"]
    if "pointer" in case:
        assert caught.value.details.get("pointer") == case["pointer"]


@pytest.mark.parametrize("case", VECTORS["rule_kinds"]["renderings"], ids=lambda c: c["id"])
def test_rendering(case):
    rendered = renderings.render(case["rendering"], case["input"]["value"])
    assert rendered == case["expect"]["rendered"]
    assert renderings.normalise(case["rendering"], rendered) == case["expect"]["normalised"]


@pytest.mark.parametrize("ref,case", list(_validator_cases()))
def test_validator_case(ref, case):
    vid, version = ref.split("@")
    validator = resolve(vid, int(version))
    params, expect, data = case.get("params", {}), case["expect"], case["input"]
    if "candidates" in expect:
        got = validator.scan(data["text"], params)
        assert len(got) == len(expect["candidates"])
        for candidate, want in zip(got, expect["candidates"], strict=True):
            have = {"start": candidate.start, "end": candidate.end, **vars(candidate.verdict)}
            for field, value in want.items():
                assert same(have[field], value), (field, have[field], value)
        return
    verdict = validator.check(data["value"], data.get("before", ""), data.get("after", ""), params)
    assert same(verdict.valid, expect["valid"]) and same(verdict.reason, expect["reason"])
    for name, value in (expect.get("outputs") or {}).items():
        assert same(getattr(verdict, name), value), name


@pytest.mark.parametrize("case", VECTORS["rule_set_identity"], ids=lambda c: c["id"])
def test_identity(case):
    """FTASKS 18.3: `canonical.py` is pinned by the vendored identity cases (exact bytes and digest)."""
    assert canonical.canonical_bytes(case["document"]).decode("utf-8") == case["canonical_utf8"]
    assert canonical.identity(case["document"]) == case["sha256"]
    assert rule_signals.rule_set_digest(case["document"]) == case["sha256"]


def test_the_registry_and_the_vector_sections_are_the_same_set():
    """Both ways: a registered validator without vectors, or a section no validator answers, is red."""
    assert {f"{i}@{v}" for i, v in VALIDATORS} == set(VECTORS["validators"])
    for ref, section in VECTORS["validators"].items():
        assert section["cases"], f"{ref} has no case"


def test_zero_mismatches_overall():
    """One number for the record: every vector, every section, D41's one case excepted."""
    total = sum(len(c) for c in VECTORS["rule_kinds"].values()) + len(VECTORS["rule_set_identity"]) + sum(
        len(s["cases"]) for s in VECTORS["validators"].values())
    assert total >= 200, f"the vector file shrank to {total} cases"


def test_same_compares_json_types():
    assert not same(True, 1) and not same(1, True) and not same(1, 1.0) and same([1, "a"], [1, "a"])
    assert not same({"a": 1}, {"a": True})


def test_the_conformance_test_calls_the_production_engine():
    """FR-28 removal control: replacing the engine call with a stub (or `False`) must be caught. Reads this
    module's AST: the row tests call `rule_signals.load`, `rule_signals.run_cases` and `rule_signals.evaluate`
    on the production package object imported at the top."""
    tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
    calls = {}
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        names = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == "rule_signals":
                names.add(node.func.attr)
        calls[fn.name] = names
    assert {"load", "run_cases"} <= calls["test_row_case"]
    assert {"load", "evaluate"} <= calls["test_row_case_through_the_single_text_path"]
    imported = [n for n in tree.body if isinstance(n, ast.Import) and any(a.name == "src.rule_signals" and
                                                                          a.asname == "rule_signals" for a in n.names)]
    assert imported, "rule_signals must be the production package"
    assert rule_signals.evaluate.__module__ == "src.rule_signals" and rule_signals.load.__module__ == "src.rule_signals"
