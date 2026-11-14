"""The pattern dialect and the deterministic budget (Feature 040 FR-30, FR-35; FTDD §6.1; FTASKS 17.2, 17.6).

* Refused constructs name the construct and the CODE-POINT position, also after multi-byte text.
* RE2 compiles under the memory cap (`rule_pattern_max_mem_bytes`): an over-large program is refused at load
  as `pattern_too_large`, naming the signal — never run. Control: drop the cap → red.
* Spans are code points although matching runs on UTF-8 bytes (one encode per text).
* A reduced text over `rule_text_max_code_points` is `unknown` / `rule_budget_exceeded`; nothing else is
  unknown (FR-35). FR-30's hard wall-clock guard runs a nested-quantifier pattern on a long input in a
  subprocess with a timeout: RE2 is linear, so it finishes; the text cap reports the budget. Control: remove
  the text cap → red.
"""
from __future__ import annotations

import json
import pathlib
import string
import subprocess
import sys

import pytest

import src.rule_signals as rule_signals
from src.rule_signals import dialect
from tests.support import rule_set_fixtures as rf
from tests.support.random_behaviour import rng_for

BACKEND = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("pattern,rule,at", [
    ("日本(a)\\1", "backreference", 5),
    ("é(?=b)", "lookaround", 1),
    ("日(?<!a)b", "lookaround", 1),
    ("é(?>ab)", "atomic", 1),
    ("日a++", "possessive", 2),
    ("é(?i)x", "inline_flag", 1),
    ("日\\p{L}", "unicode_property", 1),
    ("日a{1001}", "repeat_too_large", 2),
    ("日(a{2}){600}", "repeat_too_large", 7),
])
def test_refusals_name_the_construct_at_its_code_point(pattern, rule, at):
    refused = dialect.scan(pattern)
    assert refused is not None and (refused.rule, refused.position) == (rule, at)


def _large(alternatives: int) -> str:
    chars = string.ascii_letters + string.digits
    return "(?:" + "|".join(f"{chars[i % 62]}{chars[(i // 62) % 62]}\\w{{998}}" for i in range(alternatives)) + ")"


def _doc_with(pattern: str, seed: str):
    rng = rng_for(seed)
    sid, expr, col = rf.ident(rng, "p"), rf.ident(rng, "e"), rf.ident(rng, "c")
    return rf.document(rf.key(rng), rng, signals=[rf.pattern(sid, pattern)], expressions={expr: {"signal": sid}},
                       outputs=[rf.fired(col, expr)]), sid


def test_an_over_large_program_is_refused_at_load_naming_the_signal(request):
    doc, sid = _doc_with(_large(20), request.node.nodeid)
    small = rule_signals.Limits(max_mem=1 << 20)
    with pytest.raises(rule_signals.RuleSetRefused) as caught:
        rule_signals.load(doc, small)
    assert caught.value.code == "pattern_too_large"
    assert caught.value.details["signal"] == sid and caught.value.details["setting"] == "rule_pattern_max_mem_bytes"
    # The same pattern under a larger cap compiles: the refusal is the cap, not the pattern.
    assert rule_signals.load(doc, rule_signals.Limits(max_mem=64 << 20)).digest


def test_the_default_cap_refuses_a_pattern_a_larger_cap_admits(request):
    doc, _ = _doc_with(_large(150), request.node.nodeid)
    with pytest.raises(rule_signals.RuleSetRefused):
        rule_signals.load(doc, rule_signals.Limits())
    rule_signals.load(doc, rule_signals.Limits(max_mem=64 << 20))


def test_spans_are_code_points_after_multibyte_text():
    compiled = dialect.compile_pattern("[0-9]+", False, 8 << 20)
    text = dialect.Text("日本 12 é 345 🙂6")
    assert [(m.start, m.end) for m in dialect.finditer(compiled, text)] == [(3, 5), (8, 11), (13, 14)]


def test_an_empty_match_advances_one_code_point_after_multibyte_text():
    compiled = dialect.compile_pattern("\\ba?\\b", False, 8 << 20)
    spans = [(m.start, m.end) for m in dialect.finditer(compiled, dialect.Text("日7x"))]
    assert len(spans) == len(set(spans)), "an empty match was reported twice"


def _producer_spans(pattern: str, text: str):
    """The producer's algorithm (miDataworks / miLLM `dialect.finditer`): the binding's ``str`` path, resumed at
    ``end + 1`` code points after an empty match. Re-written here as the reference, not imported."""
    import re2

    options = re2.Options()
    options.log_errors = False
    compiled = re2.compile(pattern, options)
    out, pos = [], 0
    while pos <= len(text):
        m = compiled.search(text, pos)
        if m is None:
            break
        out.append(m.span())
        pos = m.end() + 1 if m.end() == m.start() else m.end()
    return out


@pytest.mark.parametrize("pattern", ["\\B", "a?", "x*", "\\B\\d?", "(?:a|)", "[^\\x00-\\x7f]?", "\\b\\w*\\b"])
@pytest.mark.parametrize("text", ["fooéébar", "naïve", "a bａｒ", "😀x😀", "foo barß", "日本 12 é", "é"])
def test_empty_matches_inside_multibyte_characters_agree_with_the_producer(pattern, text):
    """Review R1 (S3): RE2 tries every BYTE, so an empty match can land inside a multi-byte character. The
    byte→code-point table left those offsets at 0 (``\\B`` on ``fooéébar`` reported (0, 0) and lost (4, 4)),
    and the resume step counted from the wrong byte; ``count``, ``spans`` and ``within_n`` then differed from
    miDataworks and miLLM. The producer maps such an offset to the next code point."""
    compiled = dialect.compile_pattern(pattern, False, 8 << 20)
    ours = [(m.start, m.end) for m in dialect.finditer(compiled, dialect.Text(text))]
    assert ours == _producer_spans(pattern, text)


def test_classes_are_ascii():
    digits = dialect.compile_pattern("\\d", False, 8 << 20)
    assert list(dialect.finditer(digits, dialect.Text("٣"))) == []  # ARABIC-INDIC DIGIT THREE
    word = dialect.compile_pattern("\\w+", False, 8 << 20)
    assert [(m.start, m.end) for m in dialect.finditer(word, dialect.Text("éa"))] == [(1, 2)]


def test_case_insensitivity_is_per_signal_and_unicode_simple_folding():
    insensitive = dialect.compile_pattern("k", True, 8 << 20)
    assert [(m.start, m.end) for m in dialect.finditer(insensitive, dialect.Text("K"))] == [(0, 1)]
    sensitive = dialect.compile_pattern("k", False, 8 << 20)
    assert list(dialect.finditer(sensitive, dialect.Text("K"))) == []


def test_a_text_over_the_code_point_cap_is_unknown_and_nothing_else_is(request):
    doc, _ = _doc_with("x", request.node.nodeid)
    limits = rule_signals.Limits(text_max_code_points=10)
    compiled = rule_signals.load(doc, limits)
    over = rule_signals.evaluate(compiled, limits, text="x" * 11)
    assert (over.unknown, over.reason, over.rule_match, over.rule_veto) == (True, "rule_budget_exceeded", False, False)
    at = rule_signals.evaluate(compiled, limits, text="x" * 10)
    assert (at.unknown, at.reason, at.rule_match) == (False, None, True)


def test_a_nested_quantifier_on_a_long_input_finishes_within_a_hard_wall_clock(request):
    """FR-30, with the hard guard the FPRD asks for: a subprocess with a timeout (a hang fails the test
    instead of the run). RE2 is linear; the text cap makes the over-budget text `rule_budget_exceeded`."""
    doc, _ = _doc_with("(a+)+(b+)+$", request.node.nodeid)
    code = (
        "import json, sys\n"
        "import src.rule_signals as rs\n"
        "doc = json.loads(sys.argv[1])\n"
        "lim = rs.Limits(text_max_code_points=200_000)\n"
        "c = rs.load(doc, lim)\n"
        "inside = rs.evaluate(c, lim, text='a' * 199_999 + 'c')\n"
        "over = rs.evaluate(c, lim, text='a' * 300_000 + 'c')\n"
        "print(json.dumps([inside.unknown, inside.rule_match, over.unknown, over.reason]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code, json.dumps(doc)], cwd=BACKEND, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-1500:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [False, False, True, "rule_budget_exceeded"]
