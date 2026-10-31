"""The hostile checks can SEE a breach (Feature 040 S4; FTDD Appendix C.9; FR-31's mutation control).

`InProcessExecRunner` (`tests/support/in_process_exec_runner.py`) runs code the way FR-31 forbids — ``exec`` in this
process, one namespace across calls, the return value taken raw. Against it, `vector_holds` — the very assertion
`test_code_rule_hostile.py` applies to the real runner — must FAIL for every vendored vector safe to run here:

* class 10 (state carried between calls): the second call answers 2, the vector says 1;
* class 13 (a bool is not an integer): ``True`` is taken as a count;
* class 14 (the count cap): ``10**9`` is not clamped to 1,000;
* class 15 (an unlisted import): ``import os`` loads.

If the harness stops seeing a breach (a check loosened to pass anything), this file is red. Classes 1 – 4 and 9
(the escape forms) are NOT exercised: miStudio writes no escape-form guest source (ENH-006 D48); they are pending.
"""
from __future__ import annotations

import pytest

from src.rule_runner import contract as runner_contract
from tests.support.in_process_exec_runner import SAFE_DOCUMENTS, InProcessExecRunner
from tests.unit.test_code_rule_hostile import vector_holds

CASES = runner_contract.load_cases()
BREACHES = {"hostile.state": 10, "hostile.ret_bool": 13, "hostile.over_cap": 14, "hostile.import_unlisted": 15}


def _signal(case):
    document = CASES["documents"][case["doc"]]
    return {"id": "s", "kind": "code_rule", "code_sha256": runner_contract.code_sha256(document),
            "function": case["function"], "params": case["params"]}, document


@pytest.mark.parametrize("case_id", sorted(BREACHES))
def test_the_forbidden_runner_breaches_and_the_check_sees_it(case_id):
    case = next(c for c in CASES["hostile"] if c["id"] == case_id)
    assert case["number"] == BREACHES[case_id] and case["doc"] in SAFE_DOCUMENTS
    signal, document = _signal(case)
    harness = InProcessExecRunner({signal["code_sha256"]: document})
    answers = harness.run_batch(signal, [case["text"], case["text"]])
    assert not all(vector_holds(case["expect"], got) for got in answers), (case_id, answers)


def test_the_breach_classes_are_not_the_pending_ones():
    assert not set(BREACHES.values()) & runner_contract.PENDING_HOSTILE_CLASSES
