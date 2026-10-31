"""Cases: a rule set's own vectors (FR-29) and the vendored engine vectors (FR-28).

A case states every output column, whether a veto fired and (optionally) an expression's spans; the row
semantics are the producer's (`compose.CompiledRuleSet.evaluate_row`). A failure is reported by case INDEX
and id and by the column or expression that disagreed — never by the text.

⚠ JSON EQUALITY WITH TYPES (`same`): ``True`` is not ``1`` and ``1`` is not ``1.0``. Python's ``==`` would
pass a ``fired`` column emitted as ``1`` (miLLM 039 review R1 found exactly that).

Operator code (D41): a column whose expression names a pending code rule cannot be checked here; it is
listed `unverified`, never compared and never counted as a pass of that column. (A veto or ``field`` over a
code rule is refused at load, ENH-006 D74, so ``vetoed`` is always checked.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .compose import CompiledRuleSet, RuleSignalError
from .document import output_value_key


def same(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return bool(a == b)


@dataclass(frozen=True)
class CaseFailure:
    index: int
    case_id: str
    #: The output column, `vetoed`, `spans:<expression>` or `error` that disagreed.
    where: str

    def report(self) -> Dict[str, Any]:
        return {"case_index": self.index, "case_id": self.case_id, "where": self.where}


@dataclass
class CaseReport:
    passed: int = 0
    failures: List[CaseFailure] = field(default_factory=list)
    unverified: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "failing_cases": [f.report() for f in self.failures],
                "unverified": self.unverified}


def case_input(case: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """`(text, element)`: a value input is `before + value + after` with the value as the element."""
    data = case["input"]
    if "text" in data:
        return data["text"], None
    value = data["value"]
    return data.get("before", "") + value + data.get("after", ""), value


def run_cases(compiled: CompiledRuleSet, cases: Optional[List[Dict[str, Any]]] = None, *,
              stop_at_first: bool = False) -> CaseReport:
    """Run `cases` (default: the document's own). Never raises for a disagreement."""
    cases = compiled.document["cases"] if cases is None else cases
    skip = compiled.pending_columns()
    outputs = compiled.document["outputs"]
    report = CaseReport()
    for index, case in enumerate(cases):
        text, element = case_input(case)
        expect = case["expect"]
        try:
            row = compiled.evaluate_row(text, element)
        except RuleSignalError:
            report.failures.append(CaseFailure(index, case["id"], "error"))
            if stop_at_first:
                return report
            continue
        failure: Optional[CaseFailure] = None
        for out in outputs:
            column = out["column"]
            if column in skip:
                continue
            if not same(row.outputs.get(column), expect["outputs"].get(column)):
                failure = CaseFailure(index, case["id"], column)
                break
        if failure is None and not same(row.vetoed, expect["vetoed"]):
            failure = CaseFailure(index, case["id"], "vetoed")
        if failure is None:
            for name, spans in (expect.get("spans") or {}).items():
                if name in compiled.pending_expressions:
                    continue
                if not same(row.expressions[name]["spans"], [list(s) for s in spans]):
                    failure = CaseFailure(index, case["id"], f"spans:{name}")
                    break
        skipped = sorted(skip)
        if skipped:
            report.unverified[case["id"]] = skipped
        if failure is not None:
            report.failures.append(failure)
            if stop_at_first:
                return report
        else:
            report.passed += 1
    return report


def column_value_key(out: Dict[str, Any]) -> str:
    return output_value_key(out["value"])
