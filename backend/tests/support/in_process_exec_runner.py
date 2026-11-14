"""THE FORBIDDEN RUNNER, for one purpose: proving the hostile tests can see a breach (Feature 040 S4; FTDD Appendix C.9;
FR-31's mutation control).

`InProcessExecRunner` "runs" a code rule the way FR-31 forbids — ``exec`` of its source in THIS process, one
namespace kept across calls, the function's return value taken as the count without the reply grammar, the clamp or
an import list. `test_harness_sees_breaches.py` asserts the hostile checks FAIL against it for every class whose
vendored vector is safe to execute here (10 state, 13 a bool, 14 over the cap, 15 an unlisted import), so a check
that would pass any runner cannot pass silently.

⚠ ONLY FOR THOSE VENDORED DOCUMENTS. It never runs a loop, an allocation, a recursion or a load bomb, and miStudio
writes no escape-form guest source at all (ENH-006 D48: classes 1, 2, 3, 4, 9 and the guest forms of 17 – 19 are
the operator's, pending). Never imported by `src/`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

SAFE_DOCUMENTS = frozenset({"module_counter", "ret_bool", "over_cap", "import_unlisted"})


class InProcessExecRunner:
    available = True

    def __init__(self, documents: Mapping[str, Mapping[str, Any]]) -> None:
        self.documents = dict(documents)  # code_sha256 -> document
        self.namespaces: Dict[str, Dict[str, Any]] = {}

    def identity(self) -> Dict[str, Any]:
        return {"in_process": True}

    def run_batch(self, signal: Mapping[str, Any], texts: Sequence[str]) -> List[Dict[str, Any]]:
        sha = signal["code_sha256"]
        namespace = self.namespaces.get(sha)
        if namespace is None:  # ⚠ loaded ONCE and kept: the state a fresh instance per call would drop
            namespace = {}
            exec(compile(self.documents[sha]["source"], f"<code {sha[:8]}>", "exec"), namespace)  # noqa: S102
            self.namespaces[sha] = namespace
        out: List[Dict[str, Any]] = []
        for text in texts:
            value = namespace[signal["function"]](text, dict(signal.get("params") or {}))
            out.append({"count": value} if isinstance(value, int) else {"reason": "rule_output_invalid"})
        return out
