"""miStudio's verdicts over a rule set: match, veto, pending operator code, the cache (Feature 040 FR-27, FR-31, FR-35;
D41 / D44; FTASKS 17.4, 17.7, 17.8).

All names are random (tests/support/rule_set_fixtures.py). What is pinned:

* `rule_veto` holds only when a veto fires AND no match fires; a masked value vetoes, an unmasked one matches,
  and a text carrying both matches (the "veto despite a match" control).
* D41 / D44: a set naming operator code (`code_rule`) by COUNT LOADS; every expression naming it is PENDING:
  ``fires`` is None, never ``False``; a verdict that depends on it is `unknown` / `runner_unavailable`; an
  expression that does not depend on it still evaluates.
* ENH-006 D65 / D74: a POSITIONAL read of a code rule — a proximity anchor, a validator `span_of`, a `within_n`
  operand (also nested under `not` / `any`), a veto `when` or `removes` (also an anchor on a signal such a veto
  removes from), a `field` output — is REFUSED at load with the producer's code and details, naming the use, in
  the producer's order.
* The runner seam (FR-31): operator code never runs in this process; the engine hands a runner only the signal
  and the reduced texts (S4 counts: `test_code_rule_engine.py`).
* The compiled-set LRU (TI-17): keyed by the set digest AND the engine digest, so a limit change misses.
"""
from __future__ import annotations

import pytest

import src.rule_signals as rule_signals
from src.rule_signals import runner as runner_mod
from tests.support import rule_set_fixtures as rf
from tests.support.random_behaviour import rng_for

LIMITS = rule_signals.Limits()


def _verdict(doc, text):
    return rule_signals.evaluate(rule_signals.load(doc, LIMITS), LIMITS, text=text)


def test_an_unmasked_value_matches_and_a_masked_one_vetoes(request):
    doc = rf.masking_set(request.node.nodeid)
    hit = _verdict(doc, f"x {rf.UNMASKED} y")
    assert (hit.rule_match, hit.rule_veto, hit.unknown) == (True, False, False)
    masked = _verdict(doc, f"x {rf.MASKED} y")
    assert (masked.rule_match, masked.rule_veto, masked.vetoed) == (False, True, True)
    none = _verdict(doc, "nothing here")
    assert (none.rule_match, none.rule_veto) == (False, False)


def test_a_veto_never_holds_while_a_match_fires(request):
    """Both a masked and an unmasked value: the masked span is removed, the unmasked one still matches, so
    `rule_veto` is False although the row was vetoed."""
    doc = rf.masking_set(request.node.nodeid)
    both = _verdict(doc, f"{rf.MASKED} and {rf.UNMASKED}")
    assert both.vetoed is True
    assert (both.rule_match, both.rule_veto) == (True, False)


def test_the_set_cases_pass_and_spans_map_to_messages(request):
    doc = rf.masking_set(request.node.nodeid)
    compiled = rule_signals.load(doc, LIMITS)
    assert rule_signals.run_cases(compiled).ok
    verdict = rule_signals.evaluate(compiled, LIMITS, messages=[
        {"role": "system", "content": "be brief"}, {"role": "user", "content": f"mine is {rf.UNMASKED}"}], scope="user")
    pattern_id = doc["signals"][0]["id"]
    fired = {f["signal_id"]: f for f in verdict.fired}
    assert fired[pattern_id]["spans"] == [{"message_index": 1, "start": 8, "end": 19}]


def test_a_set_without_vetoes_never_vetoes(request):
    doc = rf.masking_set(request.node.nodeid, vetoes=False)
    compiled = rule_signals.load(doc, LIMITS)
    assert not compiled.has_vetoes
    masked = rule_signals.evaluate(compiled, LIMITS, text=f"x {rf.MASKED} y")
    assert (masked.rule_match, masked.rule_veto) == (True, False)


def test_match_expressions_are_the_fired_outputs(request):
    """A `fired` output names the match expression; an expression no `fired` output names does not match."""
    rng = rng_for(request.node.nodeid)
    a, b, ea, eb, col = rf.ident(rng, "a"), rf.ident(rng, "b"), rf.ident(rng, "x"), rf.ident(rng, "y"), rf.ident(rng, "c")
    doc = rf.document(rf.key(rng), rng, signals=[rf.pattern(a, "alpha"), rf.pattern(b, "beta")],
                      expressions={ea: {"signal": a}, eb: {"signal": b}},
                      outputs=[rf.fired(col, ea), {"column": "n_" + col, "type": "int", "value": {"count": eb}}])
    # Review R1 (S3): an expression only a NON-fired output names (a count) is not a match while a fired one exists
    assert _verdict(doc, "beta").rule_match is False
    assert _verdict(doc, "alpha").rule_match is True
    # …and with no fired output at all, every expression an output names is (TI-25's fallback)
    no_fired = dict(doc, outputs=[{"column": "n_" + col, "type": "int", "value": {"count": eb}}])
    assert _verdict(no_fired, "beta").rule_match is True


# ── D41 / D44: operator code is pending, never false, never refused ──────────────────────────────


def _code_set(rng, *, through: str):
    code, dep, expr, col = rf.ident(rng, "k"), rf.ident(rng, "d"), rf.ident(rng, "e"), rf.ident(rng, "c")
    other, other_expr, other_col = rf.ident(rng, "o"), rf.ident(rng, "f"), rf.ident(rng, "g")
    signals = [rf.code_rule_signal(code), rf.pattern(other, "plain")]
    expressions = {other_expr: {"signal": other}}
    vetoes = None
    if through == "direct":
        expressions[expr] = {"signal": code}
    elif through == "anchor":
        signals.append({"id": dep, "kind": "proximity", "anchor": code, "phrases": ["near"], "word_boundary": True,
                        "max_distance": 10, "unit": "code_points", "side": "either", "ignore_case": False,
                        "fold_fullwidth_ascii": False})
        expressions[expr] = {"signal": dep}
    elif through == "span_of":
        signals.append({"id": dep, "kind": "validator", "validator": {"id": "luhn", "version": 1}, "params": {},
                        "candidates": {"span_of": code}})
        expressions[expr] = {"signal": dep}
    elif through == "within_n":
        expressions[expr] = {"within_n": {"a": {"signal": other}, "b": {"signal": code}, "n": 5, "unit": "code_points"}}
    elif through == "within_n_nested":
        expressions[expr] = {"any": [{"signal": other}, {"not": {"within_n": {
            "a": {"signal": code}, "b": {"signal": other}, "n": 5, "unit": "code_points"}}}]}
    elif through == "veto_removes":
        expressions[expr] = {"signal": code}
        vetoes = [{"id": rf.ident(rng, "v"), "when": other_expr, "removes": [code]}]
    elif through == "field":
        expressions[expr] = {"signal": code}
    elif through in ("veto", "anchor_via_veto"):
        signals.append(rf.pattern(dep, "[0-9]+"))
        expressions[expr] = {"signal": dep}
        when = rf.ident(rng, "w")
        expressions[when] = {"signal": code}
        vetoes = [{"id": rf.ident(rng, "v"), "when": when, "removes": [dep]}]
        if through == "anchor_via_veto":  # an anchor on a signal a code-rule veto removes from: the veto is refused
            near = rf.ident(rng, "n")
            signals.append({"id": near, "kind": "proximity", "anchor": dep, "phrases": ["near"], "word_boundary": True,
                            "max_distance": 10, "unit": "code_points", "side": "either", "ignore_case": False,
                            "fold_fullwidth_ascii": False})
            expressions[expr] = {"signal": near}
    outputs = [rf.fired(col, expr), rf.fired(other_col, other_expr)]
    if through == "field":
        outputs.append({"column": rf.ident(rng, "x"), "type": "bool", "value": {"field": [code, "fires"]}})
    doc = rf.document(rf.key(rng), rng, signals=signals, expressions=expressions, outputs=outputs, vetoes=vetoes)
    return doc, expr, other_expr


@pytest.mark.parametrize("through", ["direct"])
def test_operator_code_is_accepted_and_its_dependents_are_pending(through, request):
    doc, expr, other_expr = _code_set(rng_for(request.node.nodeid), through=through)
    compiled = rule_signals.load(doc, LIMITS)  # never refused (D41 / D44)
    assert expr in compiled.pending_expressions
    verdict = rule_signals.evaluate(compiled, LIMITS, text="plain 4111 near 42")
    assert verdict.expressions[expr]["pending"] is True
    assert verdict.expressions[expr]["fires"] is None, "a pending expression must never read fires: false"
    # D74 control D74j: the code rule itself is listed pending in `fired` (never dropped, never a guessed span)
    code_id = next(s["id"] for s in doc["signals"] if s["kind"] == "code_rule")
    assert [f for f in verdict.fired if f["signal_id"] == code_id] == [
        {"signal_id": code_id, "kind": "code_rule", "pending": True, "spans": None}]
    # The independent expression still evaluates and fires, so the row MATCHES despite the pending one.
    assert verdict.expressions[other_expr] == {"fires": True, "pending": False, "count": 1,
                                               "form": "pattern"}
    assert (verdict.rule_match, verdict.unknown) == (True, False)


@pytest.mark.parametrize("through", ["direct"])
def test_a_verdict_that_depends_on_operator_code_is_unknown(through, request):
    doc, expr, _ = _code_set(rng_for(request.node.nodeid), through=through)
    verdict = _verdict(doc, "nothing that fires")
    assert (verdict.unknown, verdict.reason) == (True, "runner_unavailable")
    assert (verdict.rule_match, verdict.rule_veto) == (False, False), "unknown is no match and no veto (FR-35)"


@pytest.mark.parametrize("through,code,use", [
    ("within_n", "code_rule_in_within_n", None),
    ("within_n_nested", "code_rule_in_within_n", None),
    ("anchor", "code_rule_positions_needed", "anchor"),
    ("span_of", "code_rule_positions_needed", "span_of"),
    ("veto", "code_rule_positions_needed", "veto_when"),
    ("anchor_via_veto", "code_rule_positions_needed", "veto_when"),
    ("veto_removes", "code_rule_positions_needed", "veto_removes"),
    ("field", "code_rule_positions_needed", "field"),
])
def test_a_positional_read_of_operator_code_is_refused(through, code, use, request):
    """ENH-006 D65: a code rule reports a count and never a span, so a set that reads its positions is REFUSED at
    load (not accepted with that expression pending forever) — the producer's code and details (miDataworks 018)."""
    doc, expr, _ = _code_set(rng_for(request.node.nodeid), through=through)
    code_id = next(s["id"] for s in doc["signals"] if s["kind"] == "code_rule")
    rule_signals.cache_clear()
    with pytest.raises(rule_signals.RuleSetRefused) as caught:
        rule_signals.load(doc, LIMITS)
    assert caught.value.code == code
    if use is None:
        assert caught.value.details == {"signal": code_id, "expression": expr}
        assert repr(expr) in caught.value.message
    elif use in ("veto_when", "veto_removes"):
        assert caught.value.details == {"code_rule": code_id, "use": use, "veto": doc["vetoes"][0]["id"]}
    elif use == "field":
        column = next(o["column"] for o in doc["outputs"] if "field" in o["value"])
        assert caught.value.details == {"code_rule": code_id, "use": use, "column": column}
    else:
        dep = next(s["id"] for s in doc["signals"] if s["kind"] in ("proximity", "validator"))
        assert caught.value.details == {"code_rule": code_id, "use": use, "signal": dep}
    assert caught.value.message.startswith(f"{use or 'expression'}")


def test_positional_refusals_follow_the_producers_order(request):
    """ENH-006 D74: with several positional reads in one set the FIRST reported is the producer's first
    (miDataworks `_check_code_positions`): within_n, then anchor / span_of, then veto when, veto removes, then field."""
    rng = rng_for(request.node.nodeid)
    code, other, near, e_code, e_other, e_near, e_win, col, fcol = (
        rf.ident(rng, p) for p in ("k", "o", "n", "e", "f", "g", "h", "c", "x"))
    proximity = {"id": near, "kind": "proximity", "anchor": code, "phrases": ["near"], "word_boundary": True,
                 "max_distance": 10, "unit": "code_points", "side": "either", "ignore_case": False,
                 "fold_fullwidth_ascii": False}
    within = {"within_n": {"a": {"signal": other}, "b": {"signal": code}, "n": 5, "unit": "code_points"}}
    veto_when = {"id": rf.ident(rng, "v"), "when": e_code, "removes": [other]}
    veto_removes = {"id": rf.ident(rng, "u"), "when": e_other, "removes": [code]}
    field = {"column": fcol, "type": "bool", "value": {"field": [code, "fires"]}}

    def first_refusal(*, win, anchor, when, removes):
        signals = [rf.code_rule_signal(code), rf.pattern(other, "plain")] + ([proximity] if anchor else [])
        expressions = {e_code: {"signal": code}, e_other: {"signal": other}}
        if anchor:
            expressions[e_near] = {"signal": near}
        if win:
            expressions[e_win] = within
        vetoes = ([veto_when] if when else []) + ([veto_removes] if removes else [])
        doc = rf.document(rf.key(rng), rng, signals=signals, expressions=expressions,
                          outputs=[rf.fired(col, e_other), field], vetoes=vetoes or None)
        rule_signals.cache_clear()
        with pytest.raises(rule_signals.RuleSetRefused) as caught:
            rule_signals.load(doc, LIMITS)
        return caught.value.code, caught.value.details.get("use")

    assert first_refusal(win=True, anchor=True, when=True, removes=True) == ("code_rule_in_within_n", None)
    assert first_refusal(win=False, anchor=True, when=True, removes=True) == ("code_rule_positions_needed", "anchor")
    assert first_refusal(win=False, anchor=False, when=True, removes=True) == (
        "code_rule_positions_needed", "veto_when")
    assert first_refusal(win=False, anchor=False, when=False, removes=True) == (
        "code_rule_positions_needed", "veto_removes")
    assert first_refusal(win=False, anchor=False, when=False, removes=False) == ("code_rule_positions_needed", "field")


def test_the_own_cases_of_a_code_set_list_pending_columns_unverified(request):
    rng = rng_for(request.node.nodeid)
    doc, expr, _ = _code_set(rng, through="direct")
    col = next(o["column"] for o in doc["outputs"] if o["value"]["fired"] == expr)
    other_col = next(o["column"] for o in doc["outputs"] if o["value"]["fired"] != expr)
    doc["cases"] = [rf.case("k1", "plain", {col: True, other_col: True}, False)]
    report = rule_signals.run_cases(rule_signals.load(doc, LIMITS))
    assert report.ok and report.unverified == {"k1": [col]}


def test_operator_code_never_runs_in_this_process(request):
    """FTASKS 17.7 / FR-31 (as amended for S4): the engine hands a runner only the SIGNAL (a hash, a function name,
    params) and the reduced TEXTS — never source — and with no runner the default raises `RunnerUnavailable`."""
    seen = []

    class Recording:
        available = True

        def run_batch(self, signal, texts):
            seen.append((dict(signal), list(texts)))
            return [{"count": 0} for _ in texts]

        def identity(self):
            return {"test": True}

    doc, expr, _ = _code_set(rng_for(request.node.nodeid), through="direct")
    compiled = rule_signals.load(doc, LIMITS)
    verdict = rule_signals.evaluate(compiled, LIMITS, text="plain text", runner=Recording())
    assert verdict.expressions[expr] == {"fires": False, "pending": False, "count": 0, "form": "code_rule"}
    assert len(seen) == 1 and seen[0][1] == ["plain text"]
    assert "source" not in seen[0][0] and set(seen[0][0]) == set(rf.code_rule_signal("x"))
    assert isinstance(runner_mod.DEFAULT_RUNNER, runner_mod.UnavailableRunner)
    with pytest.raises(runner_mod.RunnerUnavailable) as caught:
        runner_mod.DEFAULT_RUNNER.run_batch({"id": "s"}, ["text"])
    assert caught.value.signal_id == "s"


# ── the compiled-set cache (TI-17, TI-18) ─────────────────────────────────────────────────────────


def test_the_cache_is_keyed_by_set_and_engine_digest(request):
    rule_signals.cache_clear()
    doc = rf.masking_set(request.node.nodeid)
    first = rule_signals.load(doc, LIMITS)
    assert rule_signals.load(doc, LIMITS) is first
    other_limits = rule_signals.Limits(text_max_code_points=LIMITS.text_max_code_points + 1)
    assert rule_signals.engine_digest(other_limits) != rule_signals.engine_digest(LIMITS)
    assert rule_signals.load(doc, other_limits) is not first, "a limit change must miss the cache"
    changed = dict(doc, description="Another description.")
    assert rule_signals.load(changed, LIMITS) is not first


def test_the_engine_digest_names_the_vendored_vectors():
    digests = rule_signals.vector_digests()
    assert set(digests) == {"rule-signal-cases-v1.json", "rule-text-reduction-cases.json",
                            "midataworks-rule-set-v1.json"}
    assert all(len(v) == 64 for v in digests.values())


# ── gaps the vendored vectors leave (found by mutation; raised with miDataworks, FR-27) ───────────


def test_the_mod11_primitive_computes_eleven_minus_the_remainder():
    """Control E7 survived the vectors: `nhs@1` has no valid vector (not publicly vectored), so the
    check-digit primitive's final step was unpinned. Hand-computed on constructed bodies (weights 10..2)."""
    from src.rule_signals.validators.algorithms import mod11_weights_10_to_2_digit

    assert mod11_weights_10_to_2_digit("000000001") == "9"   # total 2 → 11 − 2
    assert mod11_weights_10_to_2_digit("000000000") == "0"   # total 0 → 11 → "0"
    assert mod11_weights_10_to_2_digit("000000010") == "8"   # total 3 → 11 − 3
    assert mod11_weights_10_to_2_digit("000000005") == "1"   # total 10 → 11 − 10
    assert mod11_weights_10_to_2_digit("000000006") is None  # total 12 → check 10: no valid value


def test_a_veto_span_that_only_touches_does_not_remove(request):
    """Control E22 survived the vectors: spans are half-open, so a `when` span that ENDS where a removable
    span STARTS does not overlap it and removes nothing (the schema's `$comment`: s1 < e2 and s2 < e1)."""
    rng = rng_for(request.node.nodeid)
    value, mark, hit, when, col = (rf.ident(rng, p) for p in ("v", "m", "e", "w", "c"))
    doc = rf.document(rf.key(rng), rng, signals=[rf.pattern(value, "[0-9]{3}"), rf.pattern(mark, "x")],
                      expressions={hit: {"signal": value}, when: {"signal": mark}}, outputs=[rf.fired(col, hit)],
                      vetoes=[{"id": rf.ident(rng, "z"), "when": when, "removes": [value]}])
    touching = _verdict(doc, "123x")
    assert (touching.vetoed, touching.rule_match) == (False, True)



# ── Review R1 (S3): lines the conformance vectors and the tests above did not reach ───────────────


def test_an_unknown_verdict_still_reports_whether_it_was_vetoed(request):
    """ENH-006 D74: a veto can no longer be pending (a veto naming a code rule is refused at load), so an answer that
    is unknown because an expression is pending still states `vetoed` as the veto decided it, never None."""
    rng = rng_for(request.node.nodeid)
    code, value, mark, e_code, e_value, when, col, vcol = (rf.ident(rng, p) for p in "kvmefwcd")
    doc = rf.document(rf.key(rng), rng, signals=[rf.code_rule_signal(code), rf.pattern(value, "[0-9]+"),
                                                 rf.pattern(mark, "x[0-9]+")],
                      expressions={e_code: {"signal": code}, e_value: {"signal": value}, when: {"signal": mark}},
                      outputs=[rf.fired(col, e_code), rf.fired(vcol, e_value)],
                      vetoes=[{"id": rf.ident(rng, "z"), "when": when, "removes": [value]}])
    vetoed = _verdict(doc, "x12")
    assert (vetoed.unknown, vetoed.vetoed) == (True, True)
    clean = _verdict(doc, "nothing")
    assert (clean.unknown, clean.vetoed) == (True, False)


def test_a_set_with_a_custom_runner_is_never_served_from_the_shared_cache(request):
    """The LRU holds only sets compiled with the default runner: another runner's compile must not be handed to
    (or taken from) default-runner callers."""
    doc = rf.masking_set(request.node.nodeid)
    rule_signals.cache_clear()

    class Other:
        available = False

        def run_batch(self, signal, texts):
            raise runner_mod.RunnerUnavailable(str(signal.get("id")))

        def identity(self):
            return {"available": False}

    custom = rule_signals.load(doc, LIMITS, runner=Other())
    default = rule_signals.load(doc, LIMITS)
    assert custom is not default and default.runner is runner_mod.DEFAULT_RUNNER
    assert rule_signals.load(doc, LIMITS, runner=Other()) is not custom


def test_a_reply_with_no_text_reads_missing_and_last_user_skips_contentless_turns():
    from src.rule_signals import reduce as r

    reduced = r.reduce([{"role": "user", "content": "q"}, {"role": "assistant", "content": None}], "last_assistant")
    assert (reduced.text, reduced.missing) == ("", "no_scored_tokens")
    last = r.reduce([{"role": "user", "content": "a"}, {"role": "user", "content": None}], "last_user")
    assert (last.text, last.missing) == ("a", None)


def test_an_empty_span_is_located_in_its_message(request):
    """A zero-width match (an empty-able pattern) maps to its message too; dropping it lost the span."""
    rng = rng_for(request.node.nodeid)
    sid, expr, col = rf.ident(rng, "p"), rf.ident(rng, "e"), rf.ident(rng, "c")
    doc = rf.document(rf.key(rng), rng, signals=[rf.pattern(sid, "x?")], expressions={expr: {"signal": sid}},
                      outputs=[rf.fired(col, expr)])
    verdict = rule_signals.evaluate(rule_signals.load(doc, LIMITS), LIMITS,
                                    messages=[{"role": "user", "content": "ab"}])
    spans = next(f for f in verdict.fired if f["signal_id"] == sid)["spans"]
    assert spans == [{"message_index": 0, "start": i, "end": i} for i in range(3)]


def test_a_case_that_misstates_vetoed_fails(request):
    doc = rf.masking_set(request.node.nodeid)
    doc["cases"][1]["expect"]["vetoed"] = False  # the masked case IS vetoed
    report = rule_signals.run_cases(rule_signals.load(doc, LIMITS))
    assert [f.where for f in report.failures] == ["vetoed"]


def test_spans_stated_for_a_pending_expression_are_unverified_not_failed(request):
    rng = rng_for(request.node.nodeid)
    doc, expr, other_expr = _code_set(rng, through="direct")
    cols = {o["value"]["fired"]: o["column"] for o in doc["outputs"]}
    case = rf.case("k1", "plain", {cols[expr]: True, cols[other_expr]: True}, False)
    case["expect"]["spans"] = {expr: [[0, 5]], other_expr: [[0, 5]]}
    doc["cases"] = [case]
    report = rule_signals.run_cases(rule_signals.load(doc, LIMITS))
    assert report.ok, [f.report() for f in report.failures]
