"""miStudio's rule-signal engine (Feature 040 S3; FR-27 – FR-31, FR-35; FTDD §6.1, §6.2; FTID §3.7).

A pure function of (rule-set document, text or messages, scope, limits). ⚠ IT IMPORTS NOTHING FROM THE
APP — not ``services``, ``models``, ``api``, ``core``, ``workers``, ``db``, ``ml`` or ``schemas``
(`tests/unit/test_rule_signals_is_self_contained.py` walks every module's imports). Settings arrive as a
`Limits` value; nothing here reads a setting, a clock, a database or the network. The API (one text,
synchronous) and the CPU worker (a whole view) call the same `evaluate`.

Public surface:

* `load(document, limits, runner=…)` — check against the vendored schema, compile (RE2 under the memory cap),
  return a `CompiledRuleSet`. Refusals are `RuleSetRefused` with the producer's code. Cached in a process LRU
  keyed by ``(rule_set_digest, engine_digest(limits))`` — content-addressed, so it cannot leak between
  behaviours (FR-11) and a limit change misses (TI-17).
* `evaluate(compiled, limits, text=… | messages=…, scope=…)` → `RuleVerdict` (FR-27's shape) — `evaluate_many`
  over one input, so the API and the `rules` worker share one path (040-T1).
* `evaluate_many(compiled, limits, inputs, scope=…, runner=…)` (S4, FTDD C.3) — many rows; each ``code_rule``
  signal is sent to the runner ONCE PER CHUNK of up to ``batch_texts`` reduced texts.
* `check_code_rules(compiled, documents, runner_format=…)` (S4) — every code rule has its document, the function is
  declared, the params validate against the document's ``params_schema`` (data only; no code is read here).
* `run_cases(compiled)` — the set's own vectors (FR-29).
* `engine_digest(limits, runner=…)` — what a cached verdict or a stored result was computed under: the engine
  version, the vendored vector files' digests, the verdict-moving limits (TI-17, FR-38) and, for a set with
  operator code, the runner's identity (Review R1-F8).

⚠ THE BUDGET IS DETERMINISTIC (040-T3): a compiled program over ``max_mem`` is refused at load; a reduced
text over ``text_max_code_points`` is ``unknown`` / ``rule_budget_exceeded``; nothing else is ``unknown``
except operator code that no runner can run (``runner_unavailable``, D41) or that failed in the runner (a contract
reason, S4).
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import canonical, dialect
from . import document as _document
from . import reduce as _reduce
from .cases import CaseReport, run_cases
from .compose import CompiledRuleSet, RuleSignalError
from .document import RuleSetRefused
from .runner import (
    DEFAULT_RUNNER,
    PENDING_REASON,
    CodeRuleRunner,
    RunnerUnavailable,
    UnavailableRunner,
)

__all__ = [
    "ENGINE_VERSION", "Limits", "RuleVerdict", "RuleSetRefused", "RuleSignalError", "CompiledRuleSet", "CaseReport",
    "CodeRuleRunner", "RunnerUnavailable", "UnavailableRunner", "load", "evaluate", "evaluate_many", "run_cases",
    "engine_digest", "vector_digests", "rule_set_digest", "check_code_rules",
]

#: Bumped when a verdict this engine computes could change for the same inputs (FR-38: earlier results
#: then read stale, naming the engine digest).
ENGINE_VERSION = 1

#: Reason codes a verdict can carry (FTDD §5.4).
BUDGET_REASON = "rule_budget_exceeded"


@dataclass(frozen=True)
class Limits:
    """The runtime settings the engine obeys, passed in (never read here)."""

    max_mem: int = 8 * 1024 * 1024               # rule_pattern_max_mem_bytes
    text_max_code_points: int = 100_000          # rule_text_max_code_points
    max_signals: int = 1_000                     # rule_set_max_signals
    max_cases: int = 5_000                       # rule_set_max_cases
    max_bytes: int = 1024 * 1024                 # rule_set_max_bytes
    max_messages: int = 256                      # rule_signals_max_messages

    @classmethod
    def from_settings(cls, settings: Any) -> "Limits":
        return cls(
            max_mem=int(settings.rule_pattern_max_mem_bytes),
            text_max_code_points=int(settings.rule_text_max_code_points),
            max_signals=int(settings.rule_set_max_signals),
            max_cases=int(settings.rule_set_max_cases),
            max_bytes=int(settings.rule_set_max_bytes),
            max_messages=int(settings.rule_signals_max_messages),
        )

    def verdict_limits(self) -> Dict[str, int]:
        """The limits that can move a verdict (the others only accept or refuse a set)."""
        return {"max_mem": self.max_mem, "text_max_code_points": self.text_max_code_points}


def _file_sha(path: Any) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_digests() -> Dict[str, str]:
    """The vendored contract files this engine's verdicts are pinned by (read now, not cached: a re-vendor
    must show at once)."""
    return {
        "rule-signal-cases-v1.json": _file_sha(_document.CASES_PATH),
        "rule-text-reduction-cases.json": _file_sha(_document.REDUCTION_CASES_PATH),
        "midataworks-rule-set-v1.json": _file_sha(_document.SCHEMA_PATH),
    }


def engine_digest(limits: Limits, runner: Optional[Dict[str, Any]] = None) -> str:
    """TI-17: SHA-256 of the JCS bytes of {engine version, vendored vector digests, verdict limits}.

    ``runner`` (S4; Review R1-F8): for a set that names operator code, the runner's identity
    (`CodeRuleRunner.identity()`: ``{"available": False}`` with none, else its contract version, bundle digest and
    the code-rule limits in force). It is part of the digest, so verdicts cached while no runner existed — pending,
    `runner_unavailable` — are never reused once one does, and a changed bundle or limit recomputes. ``None`` (a
    set with no code rule) leaves the digest exactly as before S4."""
    body: Dict[str, Any] = {"engine_version": ENGINE_VERSION, "vectors": vector_digests(),
                            "limits": limits.verdict_limits()}
    if runner is not None:
        body["runner"] = runner
    return hashlib.sha256(canonical.canonical_bytes(body)).hexdigest()


def rule_set_digest(document: Any) -> str:
    return canonical.identity(document)


# ── loading, with a content-addressed LRU (TI-18) ────────────────────────────────────────────────

_CACHE_SIZE = 64
_cache: "OrderedDict[tuple, CompiledRuleSet]" = OrderedDict()
_cache_lock = threading.Lock()


def _parse(source: Any, limits: Limits) -> Any:
    if isinstance(source, (bytes, bytearray)):
        return _document.parse_bytes(bytes(source), limits.max_bytes)
    return source


def load(source: Any, limits: Limits, *, runner: CodeRuleRunner = DEFAULT_RUNNER) -> CompiledRuleSet:
    """Check and compile a rule set (the parsed document or its UTF-8 bytes). Raises `RuleSetRefused`."""
    document = _parse(source, limits)
    if not isinstance(document, dict):
        raise RuleSetRefused("rule_set_invalid", "a rule set is a JSON object", {"pointer": "/"})
    try:
        digest = canonical.identity(document)
    except canonical.CanonicalRefused as exc:
        _document.check(document)  # names the JSON pointer (unsafe_integer, not_nfc, …)
        raise RuleSetRefused("unsafe_integer", f"the document cannot be canonicalised: {exc}", {}) from None
    if len(canonical.canonical_bytes(document)) > limits.max_bytes:
        raise RuleSetRefused("over_cap", f"the rule set is over rule_set_max_bytes ({limits.max_bytes})",
                             {"setting": "rule_set_max_bytes", "value": limits.max_bytes})
    key = (digest, engine_digest(limits), limits.max_signals, limits.max_cases)
    cacheable = runner is DEFAULT_RUNNER
    if cacheable:
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None:
                _cache.move_to_end(key)
                return hit
    _document.check(document, max_signals=limits.max_signals, max_cases=limits.max_cases)
    compiled = CompiledRuleSet(document, max_mem=limits.max_mem, runner=runner)
    compiled.digest = digest  # type: ignore[attr-defined]
    if cacheable:
        with _cache_lock:
            _cache[key] = compiled
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)
    return compiled


def cache_clear() -> None:
    with _cache_lock:
        _cache.clear()


# ── evaluating one text or conversation ──────────────────────────────────────────────────────────


@dataclass
class RuleVerdict:
    """FR-27's shape. `rule_match` / `rule_veto` are never None in a returned verdict: an `unknown` verdict
    has both False (no match, no veto — the probe decides, FR-35) and says why in `reason`."""

    rule_match: bool
    rule_veto: bool
    unknown: bool
    reason: Optional[str] = None
    fired: List[Dict[str, Any]] = field(default_factory=list)
    expressions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vetoed: Optional[bool] = None
    missing: Optional[str] = None
    code_points: int = 0
    #: S4: per ``code_rule`` signal id, the runner's ``{"count": n}`` or ``{"reason": r}`` for this row (empty: the
    #: set has none, or no runner ran it).
    code: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_code_rules(compiled: CompiledRuleSet, documents: Mapping[str, Mapping[str, Any]], *,
                     runner_format: str) -> None:
    """S4 (FTDD C.3): every ``code_rule`` signal names a document in ``documents`` (by ``code_sha256``) of the
    runner's contract, its ``function`` is one the document declares and its ``params`` validate against the
    document's ``params_schema`` (data validation only; the source is never read here). Raises `RuleSetRefused`."""
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    for signal in compiled.signals:
        if signal["kind"] != "code_rule":
            continue
        sid, sha = signal["id"], signal["code_sha256"]
        doc = documents.get(sha)
        if doc is None:
            raise RuleSetRefused("code_rule_document_missing",
                                 f"signal {sid!r} names code {sha[:12]}… and no document for it was supplied or stored",
                                 {"signal": sid, "code_sha256": sha})
        if signal.get("runner_contract") != runner_format or doc.get("runner_contract") != runner_format:
            raise RuleSetRefused("runner_contract_unknown",
                                 f"signal {sid!r}: this miStudio's runner serves {runner_format}",
                                 {"signal": sid, "code_sha256": sha, "runner_contract": signal.get("runner_contract")})
        declared = {f.get("name") for f in doc.get("functions") or []}
        if signal["function"] not in declared:
            raise RuleSetRefused("function_not_declared",
                                 f"signal {sid!r}: function {signal['function']!r} is not declared by its document",
                                 {"signal": sid, "code_sha256": sha, "function": signal["function"]})
        try:
            errors = list(Draft202012Validator(doc.get("params_schema") or {}).iter_errors(signal["params"]))
        except SchemaError:
            errors = [None]
        if errors:
            raise RuleSetRefused("code_rule_params_invalid", f"signal {sid!r}: params refused by the document's schema",
                                 {"signal": sid, "code_sha256": sha})


def evaluate_many(compiled: CompiledRuleSet, limits: Limits, inputs: Sequence[Sequence[Any]], *, scope: str = "all",
                  runner: Optional[CodeRuleRunner] = None, spans: bool = False,
                  batch_texts: int = 256) -> List[RuleVerdict]:
    """Many rows (each a list of messages), one verdict each. Raises `RuleSignalError` if a validator raises.

    Each ``code_rule`` signal is sent to ``runner`` once per chunk of up to ``batch_texts`` reduced texts (the rows
    within the text budget), never per row; the counts then enter the composition of every row (C.3). A runner that
    is not available leaves operator code pending (D41); a reason for a row makes that row unknown with it."""
    runner = compiled.runner if runner is None else runner
    reduced = [_reduce.reduce(messages, scope) for messages in inputs]
    eligible = [i for i, r in enumerate(reduced) if len(r.text) <= limits.text_max_code_points]
    code: Dict[int, Dict[str, Dict[str, Any]]] = {i: {} for i in eligible}
    counted = bool(compiled.code_signals) and bool(getattr(runner, "available", False))
    if counted:
        texts = [reduced[i].text for i in eligible]
        size = max(1, int(batch_texts))
        for signal in compiled.signals:
            if signal["kind"] != "code_rule":
                continue
            for start in range(0, len(texts), size):
                chunk = texts[start:start + size]
                results = runner.run_batch(signal, chunk)
                if len(results) != len(chunk):
                    raise RuntimeError(f"the runner answered {len(results)} results for {len(chunk)} texts")
                for offset, result in enumerate(results):
                    code[eligible[start + offset]][signal["id"]] = dict(result)
    out: List[RuleVerdict] = []
    for i, r in enumerate(reduced):
        n = len(r.text)
        if n > limits.text_max_code_points:
            out.append(RuleVerdict(False, False, True, BUDGET_REASON, missing=r.missing, code_points=n))
            continue
        row_code = code.get(i, {})
        failed = next((row_code[s["id"]]["reason"] for s in compiled.signals
                       if s["id"] in row_code and "reason" in row_code[s["id"]]), None)
        if counted and failed is not None:
            out.append(RuleVerdict(False, False, True, str(failed), missing=r.missing, code_points=n, code=row_code))
            continue
        counts = {sid: int(v["count"]) for sid, v in row_code.items()} if counted else None
        out.append(_verdict(compiled, r, compiled.evaluate_row(r.text, code=counts), spans, row_code))
    return out


def _verdict(compiled: CompiledRuleSet, reduced: Any, row: Any, spans: bool,
             code: Dict[str, Dict[str, Any]]) -> RuleVerdict:
    n = len(reduced.text)
    match, veto = compiled.verdicts(row)
    fired: List[Dict[str, Any]] = []
    if spans:
        pending = frozenset() if code else compiled.pending_signals  # with counts nothing is pending (D74)
        for signal in compiled.signals:
            sid = signal["id"]
            if sid in pending:
                fired.append({"signal_id": sid, "kind": signal["kind"], "pending": True, "spans": None})
                continue
            if sid in code:
                if int(code[sid].get("count", 0)) >= 1:  # ⚠ C.3 / 018 Q5 (a): a fired code rule carries spans: null
                    fired.append({"signal_id": sid, "kind": "code_rule", "pending": False, "spans": None,
                                  "count": int(code[sid]["count"])})
                continue
            remaining = row.signals[sid].spans
            if remaining:
                fired.append({"signal_id": sid, "kind": signal["kind"], "pending": False,
                              "spans": [piece for s in remaining for piece in reduced.locate(s.start, s.end)]})
    expressions = {name: {"fires": e["fired"], "pending": e["pending"],
                          "count": None if e["pending"] else e.get("count", len(e["spans"] or [])),
                          "form": compiled.form_of(name)}
                   for name, e in row.expressions.items()}
    if match is None or veto is None:
        return RuleVerdict(False, False, True, PENDING_REASON, fired, expressions,
                           row.vetoed,
                           reduced.missing, n, code)
    return RuleVerdict(bool(match), bool(veto), False, None, fired, expressions, row.vetoed, reduced.missing, n, code)


def evaluate(compiled: CompiledRuleSet, limits: Limits, *, text: Optional[str] = None,
             messages: Optional[Sequence[Any]] = None, scope: str = "all", spans: bool = True,
             runner: Optional[CodeRuleRunner] = None) -> RuleVerdict:
    """One text (or one conversation reduced to `scope`): `evaluate_many` over one input (040-T1)."""
    if messages is None:
        messages = _reduce.as_messages(text if text is not None else "")
    return evaluate_many(compiled, limits, [messages], scope=scope, runner=runner, spans=spans)[0]
