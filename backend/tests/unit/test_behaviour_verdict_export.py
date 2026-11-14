"""The behaviour verdict definition (ENH-006 G19): sliced from the stored report, never recomputed; refused over
budget unless acknowledged (G19-6); the veto scope per member (G19-3) honoured by 041 OFFLINE exactly as the
served combiner honours it, row for row.

Pure: a synthetic report whose `standardisation` block DISAGREES with the candidate's recorded params, so an
export that recomputed or re-read the wrong block cannot pass by construction (global CLAUDE.md: fixtures whose
fields agree by construction hide the bug); and the blend engine on `tests/support/blend_fixtures` arrays.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from src.services import behaviour_verdict as bv
from src.services.detector_identity import jcs_bytes
from src.services.probe_blend import candidates as cand
from src.services.probe_blend import engine
from tests.support import blend_fixtures as bf

REPO = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((REPO / "docs" / "schemas" / "behaviour-verdict-v1.json").read_text())
A, B = "a" * 64, "b" * 64
RULES = {"rule_set_digest": "d" * 64, "engine_digest": "e" * 64}
MATCH = {"source": "explicit", "any_of": ["cued"]}


def report() -> dict:
    cal = lambda realised, threshold=None: {"n_negatives": 1000, "few_negatives": False, "floor": 100,  # noqa: E731
                                            "target_fpr": 0.05, "threshold": threshold, "realised_fpr": realised}
    return {
        "report_id": "pbr_0123456789ab", "model_id": "m_1",
        "behaviour": {"key": "beh-x", "version": 3, "target_fpr": 0.05, "rule_set": {"digest": "d" * 64,
                                                                                       "engine_digest": "e" * 64}},
        "views": {"blend_fit": "v_fit", "tests": [], "calibration": {"id": "v_cal", "sample_rows": 1000}},
        "members": [{"probe_id": "pm_a", "detector_digest": A, "decision_digest": "1" * 64, "decision_revision": 2,
                     "sub_question": "digits"},
                    {"probe_id": "pm_b", "detector_digest": B, "decision_digest": "2" * 64, "decision_revision": 1,
                     "sub_question": "spelled"}],
        # ⚠ deliberately NOT the candidate's params: a recompute from here would differ
        "standardisation": {"pm_a": {"mean": 9.0, "std": 9.0}, "pm_b": {"mean": 7.0, "std": 7.0}},
        "candidates": [
            {"index": 0, "name": "stack", "kind": "stack", "produces": "score",
             "params": {"mean": [1.0, 2.0], "std": [2.0, 4.0], "C": 1.0, "max_iter": 2000, "weights": [0.5, -0.25],
                        "intercept": 0.125},
             "calibration": cal(0.04, 0.25)},
            {"index": 1, "name": "any_of", "kind": "any_of", "produces": "verdict", "params": {"k": 1},
             "calibration": cal(0.05)},
            {"index": 2, "name": "rules_or_stack", "kind": "stack", "produces": "verdict",
             "params": {"mean": [1.0, 2.0], "std": [2.0, 4.0], "C": 1.0, "max_iter": 2000, "weights": [0.5, -0.25],
                        "intercept": 0.125, "rules": RULES},
             "calibration": cal(0.09, 0.75)},
            {"index": 3, "name": "rules_or_any_of", "kind": "any_of", "produces": "verdict",
             "params": {"k": 1, "rules": RULES, "veto_scope": ["pm_a"]}, "calibration": cal(0.03)},
        ],
    }


def build(name: str, *, ack=None, rep=None, windows=("all",)):
    rep = rep or report()
    entry = next(c for c in rep["candidates"] if c["name"] == name)
    return bv.build_definition(rep, entry, report_sha256="f" * 64,
                               member_meta={"pm_a": {"name": "ssn digits", "role": "primary"}},
                               match=MATCH if cand.is_rules_name(name) else None, windows=list(windows),
                               acknowledge_reason=ack, actor="operator")


def test_params_and_bar_are_the_reports_recorded_ones():
    """Control (design list): export recomputed parameters instead of the report's recorded ones → red."""
    d = build("stack")
    assert d["candidate"]["params"] == {"mean": [1.0, 2.0], "std": [2.0, 4.0], "weights": [0.5, -0.25],
                                        "intercept": 0.125}
    assert d["candidate"]["threshold"] == 0.25 and d["candidate"]["k"] is None
    assert d["calibration"]["realised_union_fpr"] == 0.04 and d["calibration"]["report_id"] == "pbr_0123456789ab"
    assert d["rules"] is None and [m["veto_applies"] for m in d["members"]] == [False, False]


def test_the_veto_scope_is_the_reports_per_member_scope():
    d = build("rules_or_any_of")
    assert [m["veto_applies"] for m in d["members"]] == [True, False]
    assert d["rules"] == {**RULES, "match": MATCH, "veto_scope": "per_member"}
    assert d["candidate"] == {"name": "rules_or_any_of", "kind": "any_of", "rules": True, "k": 1, "params": {},
                              "threshold": None}
    # display-only fields carried, never used to bind
    assert d["members"][0]["forms"] == ["digits"] and d["members"][0]["name"] == "ssn digits"


def test_a_report_without_a_scope_vetoes_every_member():
    rep = report()
    del rep["candidates"][3]["params"]["veto_scope"]
    d = build("rules_or_any_of", rep=rep)
    assert [m["veto_applies"] for m in d["members"]] == [True, True] and d["rules"]["veto_scope"] == "all"


def test_over_budget_is_refused_unless_acknowledged():
    with pytest.raises(bv.VerdictRefused) as refused:
        build("rules_or_stack")
    assert refused.value.code == "union_over_budget" and refused.value.status == 409
    assert refused.value.details["realised_union_fpr"] == 0.09 and refused.value.details["target_fpr"] == 0.05
    (call,) = refused.value.next
    assert call["tool"] == "export_behaviour_verdict" and call["needs"] == ["acknowledge_over_budget"]
    assert call["arguments"] == {"key": "beh-x", "model": "m_1", "report_id": "pbr_0123456789ab",
                                 "candidate": "rules_or_stack"}
    d = build("rules_or_stack", ack="  pilot: union measured at 9%, accepted for the dry run ")
    assert d["calibration"]["over_budget"] is True
    assert d["calibration"]["acknowledged"] == {"reason": "pilot: union measured at 9%, accepted for the dry run",
                                                "by": "operator"}
    with pytest.raises(bv.VerdictRefused):
        build("rules_or_stack", ack="   ")


def test_exactly_at_the_target_is_not_over_budget():
    d = build("any_of")
    assert d["calibration"]["over_budget"] is False and d["calibration"]["acknowledged"] is None


def test_refused_candidate_and_scoped_score_kind():
    rep = report()
    rep["candidates"][2]["calibration"]["refused"] = "rules_spend_budget"
    with pytest.raises(bv.VerdictRefused) as refused:
        build("rules_or_stack", rep=rep, ack="x")
    assert refused.value.code == "candidate_refused"
    rep = report()
    rep["candidates"][2]["params"]["veto_scope"] = ["pm_a"]
    with pytest.raises(bv.VerdictRefused) as refused:
        build("rules_or_stack", rep=rep, ack="x")
    assert refused.value.code == "veto_scope_needs_verdict_kind"


@pytest.mark.parametrize("name", ["stack", "any_of", "rules_or_any_of"])
def test_the_definition_validates_and_two_exports_are_byte_identical(name):
    first, second = build(name), build(name)
    jsonschema.validate(first, SCHEMA)
    assert bv.definition_bytes(first) == bv.definition_bytes(second)
    assert bv.definition_bytes(first) == jcs_bytes(json.loads(bv.definition_bytes(first)))      # already canonical
    assert first["members_digest"] == hashlib.sha256(jcs_bytes(sorted([A, B]))).hexdigest()
    assert bv.members_digest([B, A]) == bv.members_digest([A, B])


@pytest.mark.parametrize("name", ["stack", "any_of", "rules_or_any_of"])
def test_every_combiner_vector_reproduces_from_the_definitions_own_fields(name):
    """Read back from the CANONICAL BYTES (what miLLM receives): every vector's expectation is what the combiner
    gives on those fields alone."""
    received = json.loads(bv.definition_bytes(build(name)))
    vectors = received["combiner_vectors"]
    assert len(vectors) >= 5
    for v in vectors:
        assert bv.agrees(bv.combine(received, v["members"], v["rules"]), v["expect"]), v["id"]
    by = {v["id"]: v["expect"] for v in vectors}
    assert by["first_missing_none_fire"]["fires"] is None
    if name == "rules_or_any_of":
        # the in-scope member is vetoed, the out-of-scope one fires through the veto
        assert by["veto_only_0"]["fires"] is False and by["veto_only_0"]["decided_by"] == "vetoed"
        assert by["veto_only_1"]["fires"] is True and by["veto_only_1"]["decided_by"] == "members"
        assert by["match_and_veto"]["decided_by"] == "rule_match"
        assert by["unknown_all_fire"]["rules"] == {"match": None, "veto": None, "unknown": True}


def test_a_vector_off_by_one_verdict_does_not_agree():
    received = json.loads(bv.definition_bytes(build("rules_or_any_of")))
    v = copy.deepcopy(received["combiner_vectors"][0])
    v["expect"]["fires"] = not v["expect"]["fires"]
    assert not bv.agrees(bv.combine(received, v["members"], v["rules"]), v["expect"])


# ── 041 honours the scope OFFLINE exactly as the combiner serves it (G19-3) ───────────────────


def scoped_inputs(seed: int, scope):
    from tests.unit.test_blend_engine import with_rules

    inputs = bf.build(seed, candidates=["any_of"], n_cal=1200, target_fpr=0.05)
    inputs.candidates = [cand.parse_name(n, len(inputs.members), True) for n in ("any_of", "rules_or_any_of")]
    inputs = with_rules(inputs, rule_share=0.004)
    rng = np.random.default_rng(seed + 1)
    for v, rules in inputs.rules.items():                    # veto a third of the unmatched rows on every view
        veto = (rng.random(rules.rule_match.shape[0]) < 0.33) & ~rules.rule_match
        rules.rule_veto = veto
    inputs.veto_scope = scope
    return inputs


@pytest.mark.parametrize("scope", [["pm_a"], None, []], ids=["pm_a_only", "every_member", "no_member"])
def test_the_engines_union_equals_the_combiner_row_for_row(scope):
    """Every aligned row of every view: 041's stored rules ∪ any_of verdict == `combine` on that row's member
    verdicts and rule verdict. Controls: apply the veto to every member when scoped (engine or combiner) → red."""
    inputs = scoped_inputs(41, scope)
    result = engine.run(inputs)
    entry = next(c for c in result["candidates"] if c["name"] == "rules_or_any_of")
    if scope is not None and set(scope) != set(inputs.members):
        assert entry["params"]["veto_scope"] == sorted(scope)
    else:
        assert "veto_scope" not in entry["params"]
    in_scope = set(inputs.members if scope is None else scope)
    definition = {"candidate": {"name": "rules_or_any_of", "kind": "any_of", "rules": True, "k": 1, "params": {},
                                "threshold": None},
                  "members": [{"detector_digest": m, "veto_applies": m in in_scope} for m in inputs.members]}
    differed = 0
    for v in [inputs.blend_fit, *[t["id"] for t in inputs.tests], inputs.calibration]:
        aligned = engine.align(v, inputs.members, inputs.arrays[v], inputs.min_aligned_fraction)
        fires = engine.member_fires_of(inputs.decisions, aligned)
        at = engine.rules_at(inputs.rules[v], aligned.positions)
        stored = np.asarray(result["rows"][v]["values"][entry["index"]]) >= 0.5
        for i in range(aligned.positions.size):
            members_in = {m: {"fires": bool(fires[i, j]), "score": None} for j, m in enumerate(inputs.members)}
            rules_in = {"match": bool(at.match[i]), "veto": bool(at.veto[i]), "unknown": bool(at.unknown[i])}
            served = bv.combine(definition, members_in, rules_in)["fires"]
            assert served == bool(stored[i]), (v, i, members_in, rules_in)
            whole = bool(at.match[i] or (fires[i].any() and not at.veto[i]))
            differed += served != whole
    # a scope that leaves members out must actually change verdicts on these rows (else the test proves nothing)
    assert (differed > 0) == (scope is not None and set(scope) != set(inputs.members))


def test_a_scoped_veto_on_a_score_base_is_refused_by_the_engine():
    inputs = scoped_inputs(42, ["pm_a"])
    inputs.candidates = [cand.parse_name("rules_or_stack", len(inputs.members), True)]
    with pytest.raises(engine.EngineRefused):
        engine.run(inputs)
