"""The cases file covers the contract, and the escape-form classes are honestly PENDING (Feature 040 S4; ENH-006 D48).

* The vendored ``runner-cases-v1.json`` validates against its own ``case_schema``.
* ``missing_coverage`` is empty: every reason (but the two no guest can produce), every allowed module, every floor
  limit and every hostile class 1 – 19 has a vector — EXCEPT the classes in ``PENDING_HOSTILE_CLASSES`` (1, 2, 3, 4,
  9, 17, 18, 19), the guest-source escape forms the operator writes (D48). miStudio writes none of them.
* No pending class is reported as covered: the set of classes with a vector and the pending set are disjoint and
  together are exactly 1 – 19. Emptying the pending set without vectors turns ``missing_coverage`` red.
* The pending set MIRRORS the producer's (`runner_contract/coverage.py` in miDataworks, loaded from the sibling
  checkout by path; loud skip when absent, fail under ``MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1``).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import jsonschema
import pytest

from mistudio_rule_runner import contract as C

FLAG = "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS"


def test_the_cases_file_validates_against_its_own_schema() -> None:
    cases = C.load_cases()
    jsonschema.Draft202012Validator.check_schema(cases["case_schema"])
    jsonschema.Draft202012Validator(cases["case_schema"]).validate(cases)


def test_every_contract_requirement_has_a_vector_except_the_pending_classes() -> None:
    assert C.missing_coverage(C.load_contract(), C.load_cases()) == {}


def test_no_pending_class_is_reported_covered() -> None:
    covered = C.covered_hostile_classes(C.load_cases())
    assert not (covered & C.PENDING_HOSTILE_CLASSES)
    assert covered | C.PENDING_HOSTILE_CLASSES == frozenset(range(1, 20))
    assert C.PENDING_HOSTILE_CLASSES == frozenset({1, 2, 3, 4, 9, 17, 18, 19})


def test_emptying_the_pending_set_without_vectors_is_red(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C, "PENDING_HOSTILE_CLASSES", frozenset())
    missing = C.missing_coverage(C.load_contract(), C.load_cases())
    assert missing["hostile"] == [1, 2, 3, 4, 9, 17, 18, 19]


def _producer() -> Path:
    root = Path(os.environ.get("MIDATAWORKS_REPO") or Path(__file__).resolve().parents[3] / "miDataworks")
    return root / "backend" / "src" / "runner_contract" / "coverage.py"


def test_the_pending_set_mirrors_the_producers() -> None:
    path = _producer()
    if not path.is_file():
        message = (f"miDataworks' coverage module is not at {path} (set MIDATAWORKS_REPO). Skipping is expected on the "
                   f"public mirror; set {FLAG}=1 to make this a failure.")
        if os.environ.get(FLAG) == "1":
            pytest.fail(message)
        pytest.skip(message)
    spec = importlib.util.spec_from_file_location("_producer_coverage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # standard library only (typing, collections.abc)
    assert C.PENDING_HOSTILE_CLASSES == module.PENDING_HOSTILE_CLASSES
    assert C.REASONS_WITHOUT_VECTOR == module.REASONS_WITHOUT_VECTOR
    assert C.missing_coverage(C.load_contract(), C.load_cases()) == module.missing_coverage(
        C.load_contract(), C.load_cases())
