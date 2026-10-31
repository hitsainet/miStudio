"""The evaluator: signals, vetoes, expressions, outputs — and miStudio's two verdicts (Feature 040 FR-27, FR-31).

The ROW semantics are the producer's (the vendored schema's `$comment`; a port of miDataworks'
`backend/src/rules/compose.py`, as miLLM 039 ported it):

1. every signal's spans (pattern / proximity / validator / lexical_weights; a reference is always to an
   EARLIER signal);
2. vetoes: each veto's ``when`` expression is evaluated on the step-1 spans, then every span of each signal
   in ``removes`` that overlaps a ``when`` span (half-open) is removed; ``vetoed`` is true when any was;
3. expressions on the remaining spans (``signal``, ``all``, ``any``, ``not``, ``within_n``);
4. outputs (``fired``, ``count``, ``first``, ``spans``, ``field``).

**miStudio's verdicts (FR-27, FR-33).** ``rule_match`` = some MATCH expression is true, where the match
expressions are those the set's ``fired`` outputs name (else, for a set with no ``fired`` output, every
expression an output names). ``rule_veto`` = ``vetoed and not rule_match`` — a veto holds only when no
match fires. (Planner choice TI-25, recorded in the controls record: the producer's document has no
"match" field; its ``fired`` outputs are the bool columns `rule_set@1` writes, which a view's producer
``rule_columns.rule_match`` names.)

⚠ OPERATOR CODE IS PENDING, NEVER FALSE (D41 / D44). A ``code_rule`` signal is accepted; while no runner is
available (`runner.UnavailableRunner`, everything before S4) it is PENDING, and so is every expression naming it. A
pending expression reports ``pending: true`` and ``fires: null`` — never ``fires: false``. A verdict that depends on
one is ``unknown`` (`runner_unavailable`). Nothing else can be pending: every way a code rule could make ANOTHER
signal, a veto or a ``field`` column pending is a positional read, refused at load (below).

⚠ A POSITIONAL READ OF A CODE RULE IS REFUSED (ENH-006 D65, 2026-10-25, and D74, 2026-10-28; the producer's rule,
miDataworks 018 `rules/compose.py` `_check_code_positions`). A code rule reports a count and never a span, so what
needs its positions could never be decided — "a signal that silently never fires". It is refused at load, before
anything is compiled (`_refuse_positional_code`), in the producer's order:

1. a ``within_n`` operand naming one (``code_rule_in_within_n``, details ``{signal, expression}``, the vendored case
   `refuse.document.code_rule_in_within_n`);
2. a proximity ``anchor`` or a validator ``span_of`` on one (``code_rule_positions_needed``, details
   ``{code_rule, use: "anchor" | "span_of", signal}``);
3. (D74) a veto whose ``when`` names one (``use: "veto_when"``) or whose ``removes`` names one (``use:
   "veto_removes"``), details ``{code_rule, use, veto}``;
4. (D74) a ``field`` output reading one (``use: "field"``), details ``{code_rule, use, column}``.

**With a runner (S4; FTDD Appendix C.3).** `evaluate_row(..., code={signal id: count})` takes the counts the
isolated runner computed for this row (the engine never runs code: FR-31). A code rule reports a COUNT and never a
position (the contract; a fired code rule carries ``spans: null``), so a code-rule term fires when its count is at
least 1 and adds its count to an expression's count, and ``all`` / ``any`` / ``not`` over such terms resolve
(`_counted`). Nothing is pending with a runner (D74 removed the last positional reads, the veto ones). A code
rule that FAILED (a contract reason) never reaches here: the caller makes that row's verdict unknown with the
reason (`rule_signals.evaluate_many`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from jsonschema import Draft202012Validator

from . import dialect, lexical
from .document import (
    RuleSetRefused,
    expression_form,
    output_value_key,
    signal_reference,
    signals_named,
)
from .runner import DEFAULT_RUNNER, CodeRuleRunner
from .validators import ValidatorUnknown, resolve

TEXT = "text"
ELEMENT = "element"


class RuleSignalError(RuntimeError):
    """A validator raised. Carries the signal and validator, NEVER the text."""

    def __init__(self, signal_id: str, validator: str, error: BaseException):
        super().__init__(f"signal {signal_id!r} ({validator}) raised {type(error).__name__}")
        self.signal_id = signal_id
        self.validator = validator


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    source: str = TEXT
    outputs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.source, self.start, self.end)


@dataclass
class SignalResult:
    spans: List[Span]
    candidates: List[Span]


@dataclass
class Row:
    """One row's evaluation. `pending_columns` name what operator code would decide (D41)."""

    vetoed: bool
    outputs: Dict[str, Any]
    expressions: Dict[str, Dict[str, Any]]
    signals: Dict[str, SignalResult]
    pending_columns: FrozenSet[str]


def overlaps(a: Span, b: Span) -> bool:
    return a.source == b.source and a.start < b.end and b.start < a.end


def _sorted_unique(spans: List[Span]) -> List[Span]:
    seen: Dict[Tuple[str, int, int], Span] = {}
    for s in spans:
        seen.setdefault(s.key, s)
    return sorted(seen.values(), key=lambda s: (s.start, s.end, s.source))


def _units(text: str, lo: int, hi: int, unit: str) -> int:
    if unit == "utf8_bytes":
        return len(text[lo:hi].encode("utf-8", "surrogatepass"))
    return hi - lo


def _gap(a: Span, b: Span, unit: str, strings: Mapping[str, str]) -> Optional[int]:
    if a.source != b.source:
        return None
    if a.start < b.end and b.start < a.end:
        return 0
    lo, hi = (a.end, b.start) if a.end <= b.start else (b.end, a.start)
    return _units(strings[a.source], lo, hi, unit)


class CompiledRuleSet:
    """A checked document compiled once: patterns (under the memory cap), validators and their params."""

    def __init__(self, document: Dict[str, Any], *, max_mem: int, runner: CodeRuleRunner = DEFAULT_RUNNER):
        self.document = document
        self.max_mem = int(max_mem)
        self.signals: List[Dict[str, Any]] = list(document["signals"])
        self.expressions: Dict[str, Dict[str, Any]] = dict(document["expressions"])
        self.vetoes: List[Dict[str, Any]] = list(document.get("vetoes") or [])
        self._by_id = {s["id"]: s for s in self.signals}
        self._patterns: Dict[str, Any] = {}
        self._phrases: Dict[str, List[Any]] = {}
        self._lexical: Dict[str, lexical.CompiledLexical] = {}
        self._validators: Dict[str, Any] = {}
        self.runner = runner
        self.code_signals = frozenset(s["id"] for s in self.signals if s["kind"] == "code_rule")
        self._refuse_positional_code()  # D65 / D74: before anything is compiled, as the producer does
        for signal in self.signals:
            self._compile_signal(signal)
        # Without a runner (D41): every code rule, and every expression naming one, is pending. Nothing else can be:
        # every other dependency on a code rule is a positional read, refused above (D65 / D74). With a runner's
        # counts (S4) nothing is pending.
        self.pending_signals: FrozenSet[str] = self.code_signals
        self.pending_expressions = frozenset(
            name for name, expr in self.expressions.items()
            if any(s in self.pending_signals for s in signals_named(expr)))
        self.match_expressions = self._match_expressions()
        #: Match expressions that are also a veto's ``when`` (TI-25's reading counts them as a MATCH, which makes
        #: that veto unable to hold: Review R1 (S3) — reported, never silently resolved; operator question).
        whens = {v["when"] for v in self.vetoes}
        self.match_veto_conditions = tuple(n for n in self.match_expressions if n in whens)
        self._check_fields()

    # ── compile ──────────────────────────────────────────────────────────────────────────────────

    def _pattern(self, signal_id: str, pattern: str, ignore_case: bool) -> Any:
        try:
            return dialect.compile_pattern(pattern, ignore_case, self.max_mem)
        except dialect.PatternRefused as exc:
            raise RuleSetRefused("pattern_refused", f"signal {signal_id!r}: {exc.rule}",
                                 {"signal": signal_id, "construct": exc.rule, "position": exc.position}) from None
        except dialect.PatternTooLarge as exc:
            raise RuleSetRefused(
                "pattern_too_large",
                f"signal {signal_id!r}: the compiled pattern is over {exc.max_mem} bytes (the runtime setting "
                f"rule_pattern_max_mem_bytes)",
                {"signal": signal_id, "bytes": exc.max_mem, "setting": "rule_pattern_max_mem_bytes"},
            ) from None

    @staticmethod
    def _literal(text: str, word_boundary: bool) -> str:
        body = dialect.escape_literal(text)
        return rf"\b{body}\b" if word_boundary else body

    def _compile_signal(self, signal: Dict[str, Any]) -> None:
        kind, sid = signal["kind"], signal["id"]
        if kind == "pattern":
            self._patterns[sid] = self._pattern(sid, signal["pattern"], bool(signal["ignore_case"]))
        elif kind == "proximity":
            self._phrases[sid] = [self._pattern(sid, self._literal(p, signal["word_boundary"]),
                                                bool(signal["ignore_case"])) for p in signal["phrases"]]
        elif kind == "lexical_weights":
            try:
                self._lexical[sid] = lexical.compile_lexical(signal, self.max_mem)
            except dialect.PatternRefused as exc:
                raise RuleSetRefused("pattern_refused", f"signal {sid!r}: {exc.rule}",
                                     {"signal": sid, "construct": exc.rule, "position": exc.position}) from None
            except dialect.PatternTooLarge as exc:
                raise RuleSetRefused("pattern_too_large", f"signal {sid!r}: the term list compiles too large",
                                     {"signal": sid, "bytes": exc.max_mem,
                                      "setting": "rule_pattern_max_mem_bytes"}) from None
        elif kind == "validator":
            ref = signal["validator"]
            try:
                validator = resolve(ref["id"], ref["version"])
            except ValidatorUnknown:
                raise RuleSetRefused("validator_unknown",
                                     f"signal {sid!r}: validator {ref['id']}@{ref['version']} is not in this engine",
                                     {"signal": sid, "validator": ref["id"], "version": ref["version"]}) from None
            errors = sorted(Draft202012Validator(validator.decl.params_schema).iter_errors(signal["params"]), key=str)
            if errors:
                raise RuleSetRefused("validator_params_invalid", f"signal {sid!r}: parameters refused",
                                     {"signal": sid, "validator": validator.decl.ref})
            self._validators[sid] = validator
        elif kind == "code_rule":
            # D41: accepted, and NEVER run in this process (FR-31): its counts come from the isolated runner
            # (`evaluate_row(code=…)`), or it is pending (`_pending`). Nothing here compiles or reads its source.
            pass
        else:  # a kind the vendored schema has and this engine lacks
            raise RuleSetRefused("form_unavailable", f"signal {sid!r}: form {kind!r} is not in this engine",
                                 {"signal": sid, "form": kind})

    def _refuse_positional_code(self) -> None:
        """ENH-006 D65 / D74: refuse every read of a code rule's POSITIONS — a ``within_n`` operand, a proximity
        ``anchor``, a validator ``span_of``, a veto ``when`` or ``removes``, a ``field`` output. The producer's order,
        codes and details (miDataworks `rules/compose.py` `_check_code_positions`); a count-only use (``signal``,
        ``all``, ``any``, ``not``, a ``fired`` / ``count`` output) is accepted and pending until run (D41 / D44)."""
        if not self.code_signals:
            return
        for name, expr in self.expressions.items():
            for found in self._within_n_code_operands(expr):
                raise RuleSetRefused(
                    "code_rule_in_within_n",
                    f"expression {name!r}: within_n needs positions, and code rule {found!r} reports a count, "
                    "never a span",
                    {"signal": found, "expression": name})

        def needs(code_rule: str, use: str, **where: str) -> RuleSetRefused:
            return RuleSetRefused(
                "code_rule_positions_needed",
                f"{use} of code rule {code_rule!r} needs its positions; a code rule reports a count, never a span",
                {"code_rule": code_rule, "use": use, **where})

        for signal in self.signals:
            ref = signal_reference(signal)
            if ref in self.code_signals:
                raise needs(ref, "anchor" if signal["kind"] == "proximity" else "span_of", signal=signal["id"])
        for veto in self.vetoes:  # D74
            for s in signals_named(self.expressions[veto["when"]]):
                if s in self.code_signals:
                    raise needs(s, "veto_when", veto=veto["id"])
            for s in veto["removes"]:
                if s in self.code_signals:
                    raise needs(s, "veto_removes", veto=veto["id"])
        for out in self.document["outputs"]:  # D74
            if output_value_key(out["value"]) == "field" and out["value"]["field"][0] in self.code_signals:
                raise needs(out["value"]["field"][0], "field", column=out["column"])

    def _within_n_code_operands(self, expr: Dict[str, Any]) -> List[str]:
        """The code-rule signals any ``within_n`` inside ``expr`` names as an operand."""
        form = expression_form(expr)
        if form == "within_n":
            body = expr["within_n"]
            return [s for s in signals_named(body["a"]) + signals_named(body["b"]) if s in self.code_signals]
        if form in ("all", "any"):
            return [s for e in expr[form] for s in self._within_n_code_operands(e)]
        if form == "not":
            return self._within_n_code_operands(expr["not"])
        return []

    def _match_expressions(self) -> Tuple[str, ...]:
        outputs = self.document["outputs"]
        fired = [o["value"]["fired"] for o in outputs if output_value_key(o["value"]) == "fired"]
        if fired:
            return tuple(dict.fromkeys(fired))
        named = [o["value"][output_value_key(o["value"])] for o in outputs
                 if output_value_key(o["value"]) != "field"]
        return tuple(dict.fromkeys(named))

    def _field_names(self, signal_id: str) -> set:
        signal = self._by_id[signal_id]
        kind = signal["kind"]
        if kind == "validator":
            return {name for name, _ in self._validators[signal_id].decl.outputs}
        if kind == "proximity":
            return {"distance", "phrase", "side"}
        if kind == "lexical_weights":
            return {"term", "weight"}
        if kind == "code_rule":
            return {o["name"] for o in signal["outputs"]}
        return {"text"}

    def _check_fields(self) -> None:
        for out in self.document["outputs"]:
            if output_value_key(out["value"]) == "field":
                sig, name = out["value"]["field"]
                if name not in self._field_names(sig):
                    raise RuleSetRefused("field_unknown", f"output {out['column']!r}: no output {name!r}",
                                         {"column": out["column"], "signal": sig, "field": name})

    @property
    def has_vetoes(self) -> bool:
        return bool(self.vetoes)

    def pending_columns(self, pending_expressions: Optional[FrozenSet[str]] = None) -> FrozenSet[str]:
        """The columns a pending expression decides. A ``field`` column is never pending: a field of a code rule is
        refused at load (D74), and no other signal can be pending."""
        pending_expressions = self.pending_expressions if pending_expressions is None else pending_expressions
        out = set()
        for o in self.document["outputs"]:
            key = output_value_key(o["value"])
            if key != "field" and o["value"][key] in pending_expressions:
                out.add(o["column"])
        return frozenset(out)

    # ── step 1: signals ──────────────────────────────────────────────────────────────────────────

    def _signal_spans(self, text: str, element: Optional[str]) -> Dict[str, SignalResult]:
        strings = {TEXT: text, ELEMENT: element or ""}
        views: Dict[Tuple[str, bool], dialect.Text] = {}

        def view(source: str, fold: bool) -> dialect.Text:
            key = (source, fold)
            if key not in views:
                raw = strings[source]
                views[key] = dialect.Text(dialect.fold_fullwidth_ascii(raw) if fold else raw)
            return views[key]

        results: Dict[str, SignalResult] = {}
        for signal in self.signals:
            sid, kind = signal["id"], signal["kind"]
            if kind == "code_rule":
                results[sid] = SignalResult([], [])  # never run in-process (D41, FR-31); a code rule has no spans
                continue
            if kind == "pattern":
                source = ELEMENT if signal["reads"] == "element" else TEXT
                if source == ELEMENT and element is None:
                    spans: List[Span] = []
                else:
                    raw = strings[source]
                    spans = [Span(m.start, m.end, source, {"text": raw[m.start:m.end]})
                             for m in dialect.finditer(self._patterns[sid], view(source, bool(signal["fold_fullwidth_ascii"])))]
                results[sid] = SignalResult(spans, spans)
            elif kind == "proximity":
                spans = self._proximity(signal, results[signal["anchor"]].spans, text, view)
                results[sid] = SignalResult(spans, spans)
            elif kind == "validator":
                try:
                    candidates = self._validate(signal, results, text, element)
                except Exception as exc:  # noqa: BLE001 - re-raised named, without the text
                    ref = signal["validator"]
                    raise RuleSignalError(sid, f"{ref['id']}@{ref['version']}", exc) from None
                results[sid] = SignalResult([c for c in candidates if c.outputs.get("valid") is True], candidates)
            elif kind == "lexical_weights":
                found = lexical.spans(self._lexical[sid], lambda fold: view(TEXT, fold))
                spans = [Span(s, e, TEXT, out) for s, e, out in found]
                results[sid] = SignalResult(spans, spans)
        return results

    def _proximity(self, signal: Dict[str, Any], anchors: List[Span], text: str,
                   view: Callable[[str, bool], dialect.Text]) -> List[Span]:
        haystack = view(TEXT, bool(signal["fold_fullwidth_ascii"]))
        occurrences: List[Tuple[int, int, int]] = []
        for order, compiled in enumerate(self._phrases[signal["id"]]):
            occurrences.extend((m.start, m.end, order) for m in dialect.finditer(compiled, haystack))
        side_rule, unit = signal["side"], signal["unit"]
        out: List[Span] = []
        for anchor in anchors:
            if anchor.source != TEXT:
                continue
            best: Optional[Tuple[int, int, int, int, int]] = None
            for s, e, order in occurrences:
                if e <= anchor.start and side_rule in ("before", "either"):
                    gap, side = _units(text, e, anchor.start, unit), 0
                elif s >= anchor.end and side_rule in ("after", "either"):
                    gap, side = _units(text, anchor.end, s, unit), 1
                else:
                    continue  # a phrase overlapping the anchor is ignored
                candidate = (gap, order, side, s, e)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
            # ⚠ THE WINDOW BOUND (FR-27 control): a nearest phrase farther than max_distance yields nothing.
            if best is not None and best[0] <= signal["max_distance"]:
                gap, _o, side, s, e = best
                out.append(Span(anchor.start, anchor.end, TEXT,
                                {"distance": gap, "phrase": text[s:e], "side": "before" if side == 0 else "after"}))
        return out

    def _validate(self, signal: Dict[str, Any], results: Dict[str, SignalResult], text: str,
                  element: Optional[str]) -> List[Span]:
        validator = self._validators[signal["id"]]
        names = [name for name, _ in validator.decl.outputs]
        params = signal["params"]

        def span(start: int, end: int, source: str, verdict: Any) -> Span:
            return Span(start, end, source, {n: getattr(verdict, n) for n in names})

        candidates = signal["candidates"]
        if "span_of" in candidates:
            out = []
            for s in results[candidates["span_of"]].spans:
                string = text if s.source == TEXT else (element or "")
                out.append(span(s.start, s.end, s.source,
                                validator.check(string[s.start:s.end], string[:s.start], string[s.end:], params)))
            return out
        if "element" in candidates:
            if element is None:
                return []
            return [span(0, len(element), ELEMENT, validator.check(element, "", "", params))]
        return [span(c.start, c.end, TEXT, c.verdict) for c in validator.scan(text, params)]

    # ── step 3: expressions ──────────────────────────────────────────────────────────────────────

    def _expr(self, expr: Dict[str, Any], results: Dict[str, SignalResult],
              strings: Mapping[str, str]) -> Tuple[bool, List[Span]]:
        form = expression_form(expr)
        if form == "signal":
            spans = results[expr["signal"]].spans
            return bool(spans), list(spans)
        if form == "all":
            members = [self._expr(e, results, strings) for e in expr["all"]]
            if not all(t for t, _ in members):
                return False, []
            return True, _sorted_unique([s for _, ss in members for s in ss])
        if form == "any":
            members = [self._expr(e, results, strings) for e in expr["any"]]
            if not any(t for t, _ in members):
                return False, []
            return True, _sorted_unique([s for t, ss in members if t for s in ss])
        if form == "not":
            truth, _ = self._expr(expr["not"], results, strings)
            return not truth, []
        body = expr["within_n"]
        _, a_spans = self._expr(body["a"], results, strings)
        _, b_spans = self._expr(body["b"], results, strings)
        hits: List[Span] = []
        for a in a_spans:
            for b in b_spans:
                gap = _gap(a, b, body["unit"], strings)
                if gap is not None and gap <= body["n"]:
                    hits.extend((a, b))
        return bool(hits), _sorted_unique(hits)

    def _counted(self, expr: Dict[str, Any], results: Dict[str, SignalResult], strings: Mapping[str, str],
                 code: Mapping[str, int]) -> Tuple[bool, List[Span], int]:
        """`(truth, positions, code count)`: `_expr` with code-rule terms COUNTED, not placed (S4). With no code
        term inside it equals `_expr` exactly (code count 0). Never called on an expression that is pending."""
        form = expression_form(expr)
        if form == "signal":
            sid = expr["signal"]
            if sid in code:
                return code[sid] >= 1, [], int(code[sid])
            spans = results[sid].spans
            return bool(spans), list(spans), 0
        if form in ("all", "any"):
            members = [self._counted(e, results, strings, code) for e in expr[form]]
            truth = all(t for t, _s, _c in members) if form == "all" else any(t for t, _s, _c in members)
            if not truth:
                return False, [], 0
            kept = members if form == "all" else [m for m in members if m[0]]
            return True, _sorted_unique([sp for _t, ss, _c in kept for sp in ss]), sum(c for _t, _s, c in kept)
        if form == "not":
            truth, _s, _c = self._counted(expr["not"], results, strings, code)
            return not truth, [], 0
        truth, spans = self._expr(expr, results, strings)  # within_n: no code operand here (refused, D65)
        return truth, spans, 0

    def _names_code(self, expr: Dict[str, Any]) -> bool:
        return any(s in self.code_signals for s in signals_named(expr))

    def form_of(self, name: str) -> str:
        expr = self.expressions[name]
        form = expression_form(expr)
        return str(self._by_id[expr["signal"]]["kind"]) if form == "signal" else form

    # ── one row ──────────────────────────────────────────────────────────────────────────────────

    def evaluate_row(self, text: str, element: Optional[str] = None, code: Optional[Mapping[str, int]] = None) -> Row:
        """One row. ``code`` = the runner's COUNT for every code-rule signal of this set (S4), or None (no runner:
        D41 pending). The engine never runs code (FR-31)."""
        if code is not None and set(code) != set(self.code_signals):
            raise ValueError("code counts must name exactly this set's code-rule signals")
        counted = code is not None and bool(self.code_signals)
        pending_expressions: FrozenSet[str] = frozenset() if counted else self.pending_expressions
        strings = {TEXT: text, ELEMENT: element or ""}
        results = self._signal_spans(text, element)
        before_veto = {k: SignalResult(list(v.spans), list(v.candidates)) for k, v in results.items()}
        removed_any = False
        for veto in self.vetoes:
            when_spans = self._expr(self.expressions[veto["when"]], before_veto, strings)[1]
            for signal_id in veto["removes"]:
                result = results[signal_id]
                keep = [s for s in result.spans if not any(overlaps(s, w) for w in when_spans)]
                keep_all = [s for s in result.candidates if not any(overlaps(s, w) for w in when_spans)]
                removed_any = removed_any or len(keep) != len(result.spans)
                result.spans, result.candidates = keep, keep_all
        counts: Dict[str, int] = {}
        if counted:
            evaluated = {}
            for name, expr in self.expressions.items():
                truth, spans, n = self._counted(expr, results, strings, code or {})
                evaluated[name] = (truth, spans)
                counts[name] = len(spans) + n
        else:
            evaluated = {name: self._expr(expr, results, strings) for name, expr in self.expressions.items()}
        outputs: Dict[str, Any] = {}
        for out in self.document["outputs"]:
            value = out["value"]
            key = output_value_key(value)
            if key == "field":
                sig, name = value["field"]
                remaining = sorted(results[sig].candidates, key=lambda s: (s.start, s.end))
                outputs[out["column"]] = remaining[0].outputs.get(name) if remaining else None
                continue
            truth, spans = evaluated[value[key]]
            if counted and key in ("first", "spans") and self._names_code(self.expressions[value[key]]):
                outputs[out["column"]] = None  # a code rule has a count, never a position (spans: null)
            elif key == "fired":
                outputs[out["column"]] = truth
            elif key == "count":
                outputs[out["column"]] = counts.get(value[key], len(spans))
            elif key == "first":
                outputs[out["column"]] = strings[spans[0].source][spans[0].start:spans[0].end] if spans else None
            else:
                outputs[out["column"]] = json.dumps([[s.start, s.end] for s in spans], separators=(",", ":"))
        pending_columns = self.pending_columns(pending_expressions)
        for column in pending_columns:
            outputs[column] = None
        expressions: Dict[str, Dict[str, Any]] = {}
        for name, (truth, spans) in evaluated.items():
            if name in pending_expressions:
                expressions[name] = {"fired": None, "pending": True, "spans": None}
            elif counted and self._names_code(self.expressions[name]):
                expressions[name] = {"fired": truth, "pending": False, "spans": None, "count": counts[name]}
            else:
                expressions[name] = {"fired": truth, "pending": False, "spans": [[s.start, s.end] for s in spans]}
        return Row(vetoed=removed_any, outputs=outputs, expressions=expressions,
                   signals=results, pending_columns=pending_columns)

    def verdicts(self, row: Row) -> Tuple[Optional[bool], Optional[bool]]:
        """(rule_match, rule_veto), each None when operator code would decide it (D41)."""
        truths = [row.expressions[name]["fired"] for name in self.match_expressions]
        if any(t is True for t in truths):
            match: Optional[bool] = True
        elif any(t is None for t in truths):
            match = None
        else:
            match = False
        veto_fired = row.vetoed  # never pending: a veto naming a code rule is refused at load (D74)
        # ⚠ A VETO HOLDS ONLY WHEN NO MATCH FIRES (FR-27): `rule_veto = vetoed and not rule_match`.
        if veto_fired is False or match is True:
            veto: Optional[bool] = False
        elif veto_fired is True and match is False:
            veto = True
        else:
            veto = None
        return match, veto
