"""The operator-code seam (Feature 040 FR-31; FTDD §6.1, Appendix C.3; spec D41, D44).

A ``code_rule`` signal names operator code by its content hash. miStudio NEVER runs it in this process (FR-31): only
the isolated runner host may (`src/rule_runner/`, its own Deployment). This package sees the runner only through
the `CodeRuleRunner` protocol — the backend's HTTP client (`services/code_rule_runner.py`) is the one production
implementation; `UnavailableRunner` stands in when this deployment has no runner, or the operator has not turned
``code_rules_enabled`` on.

⚠ D41 / D44 (operator, 2026-10-24): a rule set naming operator code is ACCEPTED — never refused at load — and with no
runner every expression that names it reports PENDING: never ``fires: false`` and never a refusal
(`runner_unavailable`). A POSITIONAL read of one (a ``within_n`` operand, an ``anchor``, a ``span_of``, a veto
``when`` or ``removes``, a ``field`` output) is refused at load instead (ENH-006 D65 / D74; `compose`).

**With a runner (S4).** `run_batch(signal, texts)` answers one result per text, in order: ``{"count": n}`` or
``{"reason": <contract reason>}``. A reason makes that row's rule verdict ``unknown`` with the reason — no match, no
veto, the probe decides (FR-35); a failed code rule is never "did not fire" (018 FR-018.21). `identity()` is what a
verdict is computed under (contract version, bundle digest, the code-rule limits in force): it enters
`rule_signals.engine_digest`, so a cached verdict computed without a runner is never reused once one exists (Review
R1-F8), and a raised limit recomputes (040-A6).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

#: The reason code a pending (operator-code) expression reports in miStudio when no runner can run it (FTDD §5.4).
PENDING_REASON = "runner_unavailable"


class RunnerUnavailable(RuntimeError):
    def __init__(self, signal_id: str):
        super().__init__(f"signal {signal_id!r} is operator code; no runner is available to this miStudio")
        self.signal_id = signal_id


@runtime_checkable
class CodeRuleRunner(Protocol):
    #: False: operator code is pending (D41). True: `run_batch` answers counts or reasons.
    available: bool

    def run_batch(self, signal: Mapping[str, Any], texts: Sequence[str]) -> List[Dict[str, Any]]:
        """One ``code_rule`` signal over ``texts`` (already reduced): one ``{"count"}`` / ``{"reason"}`` per text.
        Raises `RunnerUnavailable` when no runner exists. Never raises for what the code does."""

    def identity(self) -> Optional[Dict[str, Any]]:
        """What verdicts computed through this runner depend on (None: nothing — no code is run)."""


class UnavailableRunner:
    """No runner in this deployment (or code rules switched off): every operator-code signal is pending."""

    available = False

    def run_batch(self, signal: Mapping[str, Any], texts: Sequence[str]) -> List[Dict[str, Any]]:
        raise RunnerUnavailable(str(signal.get("id")))

    def identity(self) -> Optional[Dict[str, Any]]:
        return {"available": False}


DEFAULT_RUNNER = UnavailableRunner()
