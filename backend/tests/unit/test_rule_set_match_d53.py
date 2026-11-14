"""A rule set's explicit positive (miDataworks D53, ENH-006 G19-4): miStudio reads ``match`` as the MATCH.

`midataworks.rule-set/v1` gained an optional ``match: {version: 1, any_of: [expression names]}`` (miDataworks
`5a75301`, re-vendored here). What is pinned:

* a document that states ``match`` is read AS STATED: the `dlp-ssn-us` shape with ``match: [cued]`` no longer
  matches an uncued valid number (the defect D53 closes: the inference read every `fired` output, so ``valid``
  fired alone, against D124), and a named expression no output reads still matches (never widened, never
  narrowed by the outputs);
* a document without ``match`` keeps TI-25's inference, and every surface says which it is (``match_source``);
* the veto still holds only when no match fires, under an explicit match too;
* the producer's document rules: ``match`` absent or an object, never null; each name an expression, once;
  ``version`` 1; no other key.

All names random (`tests/support/rule_set_fixtures.py`).
"""
from __future__ import annotations

import copy

import pytest

import src.rule_signals as rule_signals
from tests.support import rule_set_fixtures as rf

LIMITS = rule_signals.Limits()
API = "/api/v1/probe-monitors"

UNCUED = f"id {rf.UNMASKED} end"
CUED = f"{rf.CUE} {rf.UNMASKED} end"
CUED_MASKED = f"{rf.CUE} {rf.MASKED} end"


def _verdict(doc, text):
    return rule_signals.evaluate(rule_signals.load(doc, LIMITS), LIMITS, text=text)


# ── the positive ─────────────────────────────────────────────────────────────────────────────────


def test_without_match_the_inference_stays_and_says_so(request):
    doc, names = rf.cued_set(request.node.nodeid)
    compiled = rule_signals.load(doc, LIMITS)
    assert compiled.match_source == "inferred"
    assert compiled.match_expressions == (names["valid"], names["cued"])
    # the pre-D53 reading: a valid number with no context matches on its own
    assert _verdict(doc, UNCUED).rule_match is True


def test_an_explicit_match_is_the_positive(request):
    doc, names = rf.cued_set(request.node.nodeid, match=["cued"])
    compiled = rule_signals.load(doc, LIMITS)
    assert compiled.match_source == "explicit"
    assert compiled.match_expressions == (names["cued"],)
    uncued = _verdict(doc, UNCUED)
    assert (uncued.rule_match, uncued.rule_veto) == (False, False), "D124: an uncued valid number is no match"
    assert uncued.expressions[names["valid"]]["fires"] is True  # it still FIRES; it is just not the positive
    assert (_verdict(doc, CUED).rule_match, _verdict(doc, CUED).rule_veto) == (True, False)


def test_an_explicit_match_is_not_widened_or_narrowed_by_the_outputs(request):
    """The only `fired` output reads ``valid``; the match names ``cued``, which no output reads."""
    doc, names = rf.cued_set(request.node.nodeid, match=["cued"], fired_outputs="valid")
    assert names["cued"] not in [o["value"].get("fired") for o in doc["outputs"]]
    assert rule_signals.load(doc, LIMITS).match_expressions == (names["cued"],)
    assert _verdict(doc, UNCUED).rule_match is False
    assert _verdict(doc, CUED).rule_match is True


def test_a_veto_holds_under_an_explicit_match_only_when_no_match_fires(request):
    doc, _ = rf.cued_set(request.node.nodeid, match=["cued"])
    masked = _verdict(doc, CUED_MASKED)
    assert (masked.vetoed, masked.rule_match, masked.rule_veto) == (True, False, True)
    both = _verdict(doc, f"{CUED_MASKED} and {CUED}")
    assert (both.vetoed, both.rule_match, both.rule_veto) == (True, True, False)


def test_an_explicit_match_naming_a_veto_condition_is_flagged_not_hidden(request):
    """miDataworks refuses this at save (`match_includes_veto_condition`); one arriving here is flagged."""
    doc, names = rf.cued_set(request.node.nodeid, match=["cued", "masked"])
    assert rule_signals.load(doc, LIMITS).match_veto_conditions == (names["masked"],)


# ── the producer's document rules ────────────────────────────────────────────────────────────────


def _refused(doc):
    with pytest.raises(rule_signals.RuleSetRefused) as refused:
        rule_signals.load(doc, LIMITS)
    return refused.value


@pytest.mark.parametrize("bad, pointer", [
    ("null", "/match"),
    ("unknown", "/match/any_of"),
    ("duplicate", None),
    ("version_2", None),
    ("extra_key", None),
    ("empty", None),
])
def test_a_malformed_match_is_refused(bad, pointer, request):
    doc, names = rf.cued_set(request.node.nodeid, match=["cued"])
    doc = copy.deepcopy(doc)
    if bad == "null":
        doc["match"] = None
    elif bad == "unknown":
        doc["match"]["any_of"] = [names["cued"], "zz_not_an_expression"]
    elif bad == "duplicate":
        doc["match"]["any_of"] = [names["cued"], names["cued"]]
    elif bad == "version_2":
        doc["match"]["version"] = 2
    elif bad == "extra_key":
        doc["match"]["all_of"] = [names["cued"]]
    elif bad == "empty":
        doc["match"]["any_of"] = []
    refusal = _refused(doc)
    assert refusal.code == "rule_set_invalid", refusal.code
    if pointer is not None:
        assert refusal.details.get("pointer") == pointer, refusal.details
    if bad == "null":  # the producer's own words, not the schema's generic type error
        assert (refusal.message, refusal.details) == ("match is absent or an object, never null",
                                                      {"pointer": "/match"})
    if bad == "duplicate":
        assert refusal.details == {"what": "match expression", "name": names["cued"]}
    if bad == "unknown":
        assert refusal.details.get("expression") == "zz_not_an_expression"


def test_the_vendored_schema_carries_match():
    from src.rule_signals import document

    assert document.schema()["properties"]["match"]["$ref"] == "#/$defs/MatchExpression"
    assert "MATCH (D53)" in document.schema()["$comment"]


# ── every surface says which it is ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("match, source", [(None, "inferred"), (["cued"], "explicit")])
def test_the_validation_record_names_the_source(match, source, request):
    from src.services import rule_set_store

    doc, names = rf.cued_set(request.node.nodeid, match=match)
    record = rule_set_store.validate_for_use(doc)
    assert record["match_source"] == source
    assert record["match_expressions"] == ([names["cued"]] if match else [names["valid"], names["cued"]])


@pytest.mark.parametrize("match, source, fires", [(None, "inferred", True), (["cued"], "explicit", False)])
async def test_evaluate_reads_match_and_names_the_source(client, match, source, fires, request):
    doc, _ = rf.cued_set(request.node.nodeid, match=match)
    answer = await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": UNCUED})
    assert answer.status_code == 200, answer.text
    out = answer.json()
    assert (out["rule_match"], out["match_source"]) == (fires, source)
    assert out["rule_set_digest"] == rule_signals.rule_set_digest(doc)


# ── TI-25's fallback is labelled apart (review of D53, finding 4; tracked debt) ─────────────────────


def _no_fired_set(seed):
    """A set whose only output is a COUNT: the producer's inference (and miLLM's) is empty; TI-25 falls back to
    every expression an output names. Verdicts are unchanged; the label says which reading it is."""
    doc, names = rf.cued_set(seed)
    doc = copy.deepcopy(doc)
    doc["outputs"] = [{"column": "n_" + names["cued"][:40], "type": "int", "value": {"count": names["cued"]}}]
    return doc, names


async def test_the_all_outputs_fallback_is_labelled_on_every_surface(client, request):
    from src.services import rule_set_store

    doc, names = _no_fired_set(request.node.nodeid)
    compiled = rule_signals.load(doc, LIMITS)
    assert (compiled.match_source, compiled.match_expressions) == ("inferred_all_outputs", (names["cued"],))
    assert _verdict(doc, CUED).rule_match is True  # the verdict TI-25 gave before D53: unchanged
    assert rule_set_store.validate_for_use(doc)["match_source"] == "inferred_all_outputs"
    out = (await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": CUED})).json()
    assert (out["rule_match"], out["match_source"]) == (True, "inferred_all_outputs")
