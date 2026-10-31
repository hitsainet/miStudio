"""The engine's semantics for operator-code COUNTS (Feature 040 S4; FTDD Appendix C.3; D41, D44; 018 Q5 (a)).

Pure engine, with a recording runner (no HTTP): what a runner's counts change and what they never change.

* A code rule fires when its count is at least 1; ``all`` / ``any`` / ``not`` over it resolve; its count adds to an
  expression's count. A fired code rule carries ``spans: None`` — never ``[]``, never a guessed span.
* Every read of a code rule's POSITIONS is REFUSED at load whatever the runner: a proximity ``anchor``, a validator
  ``span_of``, a ``within_n`` (ENH-006 D65), a veto whose condition names it, a veto that removes from it, a ``field``
  output over it (D74; `test_rule_signals_engine.test_a_positional_read_of_operator_code_is_refused`). With a
  runner nothing is pending.
* A reason for a row makes that row unknown WITH the reason (no match, no veto); other rows are unaffected.
* `evaluate_many` calls the runner once per (signal, chunk) — never per row — with the reduced texts.
* `check_code_rules`: a missing document, an undeclared function, params the document's schema refuses.
* `engine_digest(limits, runner)`: None leaves S3's digest unchanged; an identity moves it (Review R1-F8).
"""
from __future__ import annotations

import pytest

import src.rule_signals as rule_signals
from src.rule_runner import contract as runner_contract
from tests.support import rule_set_fixtures as rf
from tests.support import stub_code_runner as stub
from tests.support.random_behaviour import rng_for

LIMITS = rule_signals.Limits()


class Counting:
    """A runner that ANSWERS from a table (it runs nothing): ``{signal id: {text: result}}``."""

    available = True

    def __init__(self, table, default=None):
        self.table = table
        self.default = default or {"count": 0}
        self.calls = []

    def run_batch(self, signal, texts):
        self.calls.append((signal["id"], list(texts)))
        return [dict(self.table.get(signal["id"], {}).get(t, self.default)) for t in texts]

    def identity(self):
        return {"test": "runner"}


def _ids(request, *prefixes):
    rng = rng_for(request.node.nodeid)
    return rng, [rf.ident(rng, p) for p in prefixes]


def _doc(rng, signals, expressions, outputs, vetoes=None):
    return rf.document(rf.key(rng), rng, signals=signals, expressions=expressions, outputs=outputs, vetoes=vetoes)


def test_counts_resolve_all_any_not_and_add_to_the_count(request):
    rng, (k, p, e_any, e_all, e_not, c1, c2, c3) = _ids(request, "k", "p", "a", "b", "n", "c", "d", "e")
    code = stub.code_signal(k, stub.document())
    doc = _doc(rng, [code, rf.pattern(p, "word")],
               {e_any: {"any": [{"signal": k}, {"signal": p}]}, e_all: {"all": [{"signal": k}, {"signal": p}]},
                e_not: {"not": {"signal": k}}},
               [rf.fired(c1, e_any), rf.fired(c2, e_all), {"column": c3, "type": "int", "value": {"count": e_all}}])
    compiled = rule_signals.load(doc, LIMITS)
    runner = Counting({k: {"word word": {"count": 3}, "none": {"count": 0}}})
    hit = rule_signals.evaluate(compiled, LIMITS, text="word word", runner=runner)
    assert hit.expressions[e_all] == {"fires": True, "pending": False, "count": 5, "form": "all"}
    assert hit.expressions[e_any]["fires"] is True and hit.expressions[e_not]["fires"] is False
    assert {"signal_id": k, "kind": "code_rule", "pending": False, "spans": None, "count": 3} in hit.fired
    row = compiled.evaluate_row("word word", code={k: 3})
    assert row.outputs[c3] == 5 and row.expressions[e_all]["spans"] is None
    miss = rule_signals.evaluate(compiled, LIMITS, text="none", runner=runner)
    assert miss.expressions[e_all]["fires"] is False and miss.expressions[e_not]["fires"] is True
    assert not any(f["signal_id"] == k for f in miss.fired), "a code rule that did not fire is not listed"
    assert (miss.unknown, miss.rule_match) == (False, False)


@pytest.mark.parametrize("through", ["anchor", "span_of", "within_n", "veto", "veto_removes", "field"])
def test_what_needs_a_code_rules_positions_is_refused_with_a_runner(through, request):
    """D65 / D74: every positional read is refused at load whatever the runner (never pending, never a count)."""
    rng, (k, dep, expr, other, other_expr, col, ocol, when) = _ids(request, "k", "d", "e", "o", "f", "c", "g", "w")
    signals = [stub.code_signal(k, stub.document()), rf.pattern(other, "plain")]
    expressions = {other_expr: {"signal": other}}
    vetoes = None
    if through == "anchor":
        signals.append({"id": dep, "kind": "proximity", "anchor": k, "phrases": ["near"], "word_boundary": True,
                        "max_distance": 10, "unit": "code_points", "side": "either", "ignore_case": False,
                        "fold_fullwidth_ascii": False})
        expressions[expr] = {"signal": dep}
    elif through == "span_of":
        signals.append({"id": dep, "kind": "validator", "validator": {"id": "luhn", "version": 1}, "params": {},
                        "candidates": {"span_of": k}})
        expressions[expr] = {"signal": dep}
    elif through == "within_n":
        expressions[expr] = {"within_n": {"a": {"signal": k}, "b": {"signal": other}, "n": 5, "unit": "code_points"}}
    elif through == "veto":
        signals.append(rf.pattern(dep, "[0-9]+"))
        expressions[expr] = {"signal": dep}
        expressions[when] = {"signal": k}
        vetoes = [{"id": rf.ident(rng, "v"), "when": when, "removes": [dep]}]
    elif through == "veto_removes":
        expressions[expr] = {"signal": k}
        vetoes = [{"id": rf.ident(rng, "v"), "when": other_expr, "removes": [k]}]
    else:
        expressions[expr] = {"signal": k}
    outputs = [rf.fired(col, expr), rf.fired(ocol, other_expr)]
    if through == "field":
        outputs.append({"column": rf.ident(rng, "x"), "type": "int", "value": {"field": [k, "hits"]}})
    doc = _doc(rng, signals, expressions, outputs, vetoes)
    rule_signals.cache_clear()
    with pytest.raises(rule_signals.RuleSetRefused) as caught:
        rule_signals.load(doc, LIMITS, runner=Counting({k: {}}, {"count": 2}))
    assert caught.value.code == ("code_rule_in_within_n" if through == "within_n" else "code_rule_positions_needed")
    if through not in ("anchor", "span_of", "within_n"):
        assert caught.value.details["use"] == {"veto": "veto_when"}.get(through, through)


def test_with_a_runner_nothing_is_pending(request):
    """D74: once every positional read is refused, a runner's counts decide every expression of a loadable set —
    no expression, column or veto stays pending (the `_pending(code_known=True)` path D74 made unreachable)."""
    rng, (k, mark, expr, other, when, col, ocol) = _ids(request, "k", "m", "e", "o", "w", "c", "d")
    doc = _doc(rng, [stub.code_signal(k, stub.document()), rf.pattern(mark, "x[0-9]+"), rf.pattern(other, "[0-9]+")],
               {expr: {"any": [{"signal": k}, {"signal": other}]}, when: {"signal": mark}},
               [rf.fired(col, expr), {"column": ocol, "type": "int", "value": {"count": expr}}],
               vetoes=[{"id": rf.ident(rng, "v"), "when": when, "removes": [other]}])
    compiled = rule_signals.load(doc, LIMITS)
    verdict = rule_signals.evaluate(compiled, LIMITS, text="x12", runner=Counting({}, {"count": 0}))
    assert verdict.expressions[expr] == {"fires": False, "pending": False, "count": 0, "form": "any"}
    assert (verdict.unknown, verdict.vetoed, verdict.rule_match, verdict.rule_veto) == (False, True, False, True)
    row = compiled.evaluate_row("x12", code={k: 0})
    assert row.pending_columns == frozenset() and row.outputs[ocol] == 0


def test_a_reason_makes_its_row_unknown_with_it_and_only_that_row(request):
    rng, (k, expr, col) = _ids(request, "k", "e", "c")
    doc = _doc(rng, [stub.code_signal(k, stub.document())], {expr: {"signal": k}}, [rf.fired(col, expr)])
    compiled = rule_signals.load(doc, LIMITS)
    runner = Counting({k: {"bad": {"reason": "rule_timeout"}, "good": {"count": 1}}})
    bad, good = rule_signals.evaluate_many(compiled, LIMITS, [[{"role": "user", "content": "bad"}],
                                                              [{"role": "user", "content": "good"}]], runner=runner)
    assert (bad.unknown, bad.reason, bad.rule_match, bad.rule_veto) == (True, "rule_timeout", False, False)
    assert bad.code == {k: {"reason": "rule_timeout"}}
    assert (good.unknown, good.rule_match) == (False, True)


def test_the_runner_is_called_once_per_signal_and_chunk_with_the_reduced_texts(request):
    rng, (k1, k2, e1, e2, c1, c2) = _ids(request, "k", "l", "e", "f", "c", "d")
    doc = _doc(rng, [stub.code_signal(k1, stub.document("one")), stub.code_signal(k2, stub.document("two"))],
               {e1: {"signal": k1}, e2: {"signal": k2}}, [rf.fired(c1, e1), rf.fired(c2, e2)])
    compiled = rule_signals.load(doc, LIMITS)
    runner = Counting({})
    inputs = [[{"role": "system", "content": "sys"}, {"role": "user", "content": f"u{i}"}] for i in range(7)]
    rule_signals.evaluate_many(compiled, LIMITS, inputs, scope="user", runner=runner, batch_texts=3)
    assert runner.calls == [(k1, ["u0", "u1", "u2"]), (k1, ["u3", "u4", "u5"]), (k1, ["u6"]),
                            (k2, ["u0", "u1", "u2"]), (k2, ["u3", "u4", "u5"]), (k2, ["u6"])]


def test_a_row_over_the_text_budget_never_reaches_the_runner(request):
    rng, (k, expr, col) = _ids(request, "k", "e", "c")
    doc = _doc(rng, [stub.code_signal(k, stub.document())], {expr: {"signal": k}}, [rf.fired(col, expr)])
    small = rule_signals.Limits(text_max_code_points=5)
    runner = Counting({})
    over, ok = rule_signals.evaluate_many(rule_signals.load(doc, small), small,
                                          [[{"role": "user", "content": "x" * 6}], [{"role": "user", "content": "x"}]],
                                          runner=runner)
    assert over.reason == "rule_budget_exceeded" and runner.calls == [(k, ["x"])]


def test_check_code_rules_refuses_what_does_not_fit_its_document(request):
    rng, (k, expr, col) = _ids(request, "k", "e", "c")
    document = stub.document("count_it")
    document["params_schema"] = {"type": "object", "properties": {"needle": {"type": "string"}},
                                 "additionalProperties": False}
    signal = stub.code_signal(k, document, {"needle": "x"})
    doc = _doc(rng, [signal], {expr: {"signal": k}}, [rf.fired(col, expr)])
    compiled = rule_signals.load(doc, LIMITS)
    sha = runner_contract.code_sha256(document)
    rule_signals.check_code_rules(compiled, {sha: document}, runner_format=runner_contract.FORMAT)
    with pytest.raises(rule_signals.RuleSetRefused) as missing:
        rule_signals.check_code_rules(compiled, {}, runner_format=runner_contract.FORMAT)
    assert missing.value.code == "code_rule_document_missing"
    bad_params = _doc(rng, [stub.code_signal(k, document, {"needle": 3})], {expr: {"signal": k}}, [rf.fired(col, expr)])
    with pytest.raises(rule_signals.RuleSetRefused) as params:
        rule_signals.check_code_rules(rule_signals.load(bad_params, LIMITS), {sha: document},
                                      runner_format=runner_contract.FORMAT)
    assert params.value.code == "code_rule_params_invalid"
    undeclared = _doc(rng, [stub.code_signal(k, document, {"needle": "x"}, function="absent")], {expr: {"signal": k}},
                      [rf.fired(col, expr)])
    with pytest.raises(rule_signals.RuleSetRefused) as fn:
        rule_signals.check_code_rules(rule_signals.load(undeclared, LIMITS), {sha: document},
                                      runner_format=runner_contract.FORMAT)
    assert fn.value.code == "function_not_declared"


def test_the_engine_digest_names_the_runner_and_none_keeps_s3s():
    plain = rule_signals.engine_digest(LIMITS)
    assert rule_signals.engine_digest(LIMITS, None) == plain
    off = rule_signals.engine_digest(LIMITS, {"available": False})
    on = rule_signals.engine_digest(LIMITS, {**runner_contract.identity(), "limits": {"call_fuel": 1}})
    raised = rule_signals.engine_digest(LIMITS, {**runner_contract.identity(), "limits": {"call_fuel": 2}})
    assert len({plain, off, on, raised}) == 4
