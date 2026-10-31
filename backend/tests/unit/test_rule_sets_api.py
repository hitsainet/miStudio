"""Rule-set validation at attach and at use, and the single-text tools (Feature 040 S3; FR-29, FR-32; FTASKS 19.1 – 19.3).

Through the REAL routes. All names random (`tests/support/rule_set_fixtures.py`).

* FR-29: a set with a failing case is refused `rule_set_cases_failed`, naming the case (index, id, column), when
  it is ATTACHED to a behaviour version and when it is USED (a set stored before the check existed is refused at
  use); a passing set attaches and evaluates.
* FR-32: `validate_rule_set` answers `valid: true|false` with the failing cases; `evaluate_rule_signals` returns
  exactly what the engine returns on the same text (route == direct call); caps on messages and text.
* D41 / D44 through the route: a `code_rule` set validates, its dependent expressions are listed pending.
* ENH-006 D65 through the route: a positional read of a code rule (`within_n`, anchor) is refused at validate,
  evaluate and attach; a count-only code-rule set is still accepted with its expressions pending.
* ENH-006 D74 through the route: a veto `when` / `removes` naming a code rule, and a `field` output reading one, are
  refused at validate, evaluate and attach with `code_rule_positions_needed` and the producer's `use` and details.
* No text in any log record (`caplog` sentinel across both routes).
"""
from __future__ import annotations

import logging

import pytest

import src.rule_signals as rule_signals
from tests.support import rule_set_fixtures as rf
from tests.support.random_behaviour import behaviour_body, rng_for

API = "/api/v1/probe-monitors"


def _failing(doc):
    """The set with its first case expecting the wrong verdict."""
    bad = dict(doc)
    first = dict(doc["cases"][0])
    column = next(iter(first["expect"]["outputs"]))
    first["expect"] = {**first["expect"], "outputs": {column: not first["expect"]["outputs"][column]}}
    bad["cases"] = [first] + doc["cases"][1:]
    return bad, first["id"], column


async def test_a_failing_case_is_refused_at_attach_naming_it(client, request):
    rng = rng_for(request.node.nodeid)
    body = behaviour_body(rng)
    doc, case_id, column = _failing(rf.masking_set(request.node.nodeid, body["key"]))
    response = await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": doc}})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "rule_set_cases_failed"
    assert detail["failing_cases"] == [{"case_index": 0, "case_id": case_id, "where": column}]
    assert (await client.get(f"{API}/behaviours/{body['key']}")).status_code == 404, "nothing stored"


async def test_a_new_version_attach_is_checked_too(client, request):
    rng = rng_for(request.node.nodeid)
    body = behaviour_body(rng)
    assert (await client.post(f"{API}/behaviours", json=body)).status_code == 201
    doc, _, _ = _failing(rf.masking_set(request.node.nodeid, body["key"]))
    response = await client.post(f"{API}/behaviours/{body['key']}/versions",
                                 json={"base_version": 1, "rule_set": {"document": doc}})
    assert response.status_code == 422 and response.json()["detail"]["code"] == "rule_set_cases_failed"


async def test_a_stored_set_that_fails_is_refused_at_use(client, async_session, request):
    """A set stored before the check existed (written straight to the rows, as S1 stored it) is refused when
    an evaluation USES it."""
    from src.models.probe_behaviour import ProbeBehaviourVersion
    from src.services import rule_set_store

    rng = rng_for(request.node.nodeid)
    body = behaviour_body(rng)
    good = rf.masking_set(request.node.nodeid, body["key"])
    assert (await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": good}})).status_code == 201
    doc, case_id, _ = _failing(good)
    canonical = rule_signals.canonical.canonical_bytes(doc)
    digest = rule_signals.rule_set_digest(doc)
    await rule_set_store._store(async_session, digest=digest, canonical=canonical, key=body["key"], source="inline",
                                source_ref=None, actor="test", fetched=False)
    from sqlalchemy import update

    from src.models.probe_behaviour import ProbeBehaviour

    async_session.add(ProbeBehaviourVersion(key=body["key"], version=2, created_by="test", name=body["name"],
                                            positive_meaning=body["positive_meaning"], rule_set_digest=digest))
    await async_session.execute(update(ProbeBehaviour).where(ProbeBehaviour.key == body["key"]).values(current_version=2))
    await async_session.commit()
    response = await client.post(f"{API}/rule-signals/evaluate", json={"behaviour_key": body["key"], "text": "x"})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "rule_set_cases_failed" and detail["failing_cases"][0]["case_id"] == case_id
    validated = (await client.post(f"{API}/rule-sets/validate", json={"behaviour_key": body["key"]})).json()
    assert validated["valid"] is False and validated["code"] == "rule_set_cases_failed"


async def test_validate_answers_valid_with_the_engine_facts(client, request):
    doc = rf.masking_set(request.node.nodeid)
    out = (await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})).json()
    assert out["valid"] is True and out["has_vetoes"] is True
    assert out["rule_set_digest"] == rule_signals.rule_set_digest(doc)
    assert out["engine_digest"] == rule_signals.engine_digest(rule_signals.Limits())
    assert out["cases"]["passed"] == 2 and out["pending_expressions"] == []


async def test_validate_answers_a_schema_refusal_as_data(client, request):
    doc = rf.masking_set(request.node.nodeid)
    doc["signals"][0]["pattern"] = "(a)\\1"
    out = (await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})).json()
    assert (out["valid"], out["code"], out["construct"], out["position"]) == (False, "pattern_refused", "backreference", 3)


async def test_a_code_rule_set_validates_and_lists_pending(client, request):
    rng = rng_for(request.node.nodeid)
    code, expr, col = rf.ident(rng, "k"), rf.ident(rng, "e"), rf.ident(rng, "c")
    doc = rf.document(rf.key(rng), rng, signals=[rf.code_rule_signal(code)], expressions={expr: {"signal": code}},
                      outputs=[rf.fired(col, expr)])
    out = (await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})).json()
    assert out["valid"] is True and out["pending_expressions"] == [expr]
    tried = (await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": "anything"})).json()
    assert (tried["unknown"], tried["reason"]) == (True, "runner_unavailable")
    assert tried["expressions"][expr]["fires"] is None and tried["expressions"][expr]["pending"] is True


async def test_a_count_only_code_rule_set_stays_accepted_with_pending_expressions(client, request):
    """ENH-006 D65 keeps D41 / D44 for COUNTS: `signal` / `any` / `not` over a code rule and a `count` output validate,
    every expression naming the code rule is listed pending, and nothing is refused."""
    rng = rng_for(request.node.nodeid)
    code, other = rf.ident(rng, "k"), rf.ident(rng, "p")
    e_sig, e_any, e_not, e_plain = (rf.ident(rng, x) for x in ("e", "a", "n", "q"))
    doc = rf.document(rf.key(rng), rng, signals=[rf.pattern(other, "plain"), rf.code_rule_signal(code)],
                      expressions={e_sig: {"signal": code}, e_any: {"any": [{"signal": code}, {"signal": other}]},
                                   e_not: {"not": {"signal": code}}, e_plain: {"signal": other}},
                      outputs=[rf.fired(rf.ident(rng, "c"), e_sig), rf.fired(rf.ident(rng, "d"), e_any),
                               {"column": rf.ident(rng, "m"), "type": "int", "value": {"count": e_not}},
                               rf.fired(rf.ident(rng, "f"), e_plain)])
    out = (await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})).json()
    assert out["valid"] is True, out
    assert out["pending_expressions"] == sorted([e_sig, e_any, e_not])
    tried = await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": "plain"})
    assert tried.status_code == 200, tried.text
    assert tried.json()["expressions"][e_any]["pending"] is True


def _positional_code_set(rng, behaviour_key=None):
    code, other, expr, col = rf.ident(rng, "k"), rf.ident(rng, "p"), rf.ident(rng, "e"), rf.ident(rng, "c")
    doc = rf.document(behaviour_key or rf.key(rng), rng, signals=[rf.pattern(other, "plain"), rf.code_rule_signal(code)],
                      expressions={expr: {"within_n": {"a": {"signal": other}, "b": {"signal": code}, "n": 5,
                                                       "unit": "code_points"}}},
                      outputs=[rf.fired(col, expr)])
    return doc, code, expr


async def test_a_within_n_over_code_is_refused_by_validate_and_evaluate(client, request):
    """ENH-006 D65 through the REAL routes: validate answers `valid: false` with the producer's code naming the
    expression and the code rule; the evaluation route refuses 422 the same way (never a pending answer)."""
    doc, code, expr = _positional_code_set(rng_for(request.node.nodeid))
    rule_signals.cache_clear()
    response = await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})
    assert response.status_code == 200, response.text
    out = response.json()
    assert (out["valid"], out["code"], out["expression"], out["signal"]) == (False, "code_rule_in_within_n", expr, code)
    assert "pending_expressions" not in out
    tried = await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": "plain"})
    assert tried.status_code == 422, tried.text
    detail = tried.json()["detail"]
    assert (detail["code"], detail["expression"], detail["signal"]) == ("code_rule_in_within_n", expr, code)


async def test_an_anchor_on_code_is_refused_at_attach(client, request):
    """ENH-006 D65 at attach: a behaviour carrying a set whose proximity anchor is a code rule is refused 422
    (`code_rule_positions_needed`, naming the use) and nothing is stored."""
    rng = rng_for(request.node.nodeid)
    body = behaviour_body(rng)
    code, near, expr, col = rf.ident(rng, "k"), rf.ident(rng, "n"), rf.ident(rng, "e"), rf.ident(rng, "c")
    doc = rf.document(body["key"], rng, signals=[rf.code_rule_signal(code), {
        "id": near, "kind": "proximity", "anchor": code, "phrases": ["near"], "word_boundary": True, "max_distance": 10,
        "unit": "code_points", "side": "either", "ignore_case": False, "fold_fullwidth_ascii": False}],
        expressions={expr: {"signal": near}}, outputs=[rf.fired(col, expr)])
    response = await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": doc}})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert (detail["code"], detail["use"], detail["code_rule"], detail["signal"]) == (
        "code_rule_positions_needed", "anchor", code, near)
    assert (await client.get(f"{API}/behaviours/{body['key']}")).status_code == 404, "nothing stored"


def _d74_code_set(rng, use, behaviour_key=None):
    """A set whose only refusal is the D74 `use`: a veto `when`, a veto `removes`, or a `field` over a code rule."""
    code, other, e_code, e_other, col = (rf.ident(rng, p) for p in ("k", "p", "e", "f", "c"))
    expressions = {e_code: {"signal": code}, e_other: {"signal": other}}
    outputs = [rf.fired(col, e_other)]
    vetoes = None
    veto_id = rf.ident(rng, "v")
    if use == "veto_when":
        vetoes = [{"id": veto_id, "when": e_code, "removes": [other]}]
    elif use == "veto_removes":
        vetoes = [{"id": veto_id, "when": e_other, "removes": [code]}]
    else:
        outputs.append({"column": rf.ident(rng, "x"), "type": "bool", "value": {"field": [code, "fires"]}})
    doc = rf.document(behaviour_key or rf.key(rng), rng, signals=[rf.pattern(other, "plain"), rf.code_rule_signal(code)],
                      expressions=expressions, outputs=outputs, vetoes=vetoes)
    where = {"column": outputs[-1]["column"]} if use == "field" else {"veto": veto_id}
    return doc, {"code": "code_rule_positions_needed", "code_rule": code, "use": use, **where}


@pytest.mark.parametrize("use", ["veto_when", "veto_removes", "field"])
async def test_a_d74_read_of_code_is_refused_by_validate_evaluate_and_attach(client, request, use):
    """ENH-006 D74 through the REAL routes, each the producer's code and details: validate answers `valid: false`
    (never a pending list), the evaluation route refuses 422, and attaching it to a behaviour is refused 422 with
    nothing stored."""
    rng = rng_for(request.node.nodeid)
    doc, want = _d74_code_set(rng, use)
    rule_signals.cache_clear()
    out = (await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})).json()
    assert out["valid"] is False and {k: out.get(k) for k in want} == want, out
    assert "pending_expressions" not in out
    tried = await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": "plain"})
    assert tried.status_code == 422, tried.text
    assert {k: tried.json()["detail"].get(k) for k in want} == want
    body = behaviour_body(rng)
    doc, want = _d74_code_set(rng, use, behaviour_key=body["key"])
    response = await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": doc}})
    assert response.status_code == 422, response.text
    assert {k: response.json()["detail"].get(k) for k in want} == want
    assert (await client.get(f"{API}/behaviours/{body['key']}")).status_code == 404, "nothing stored"


async def test_evaluate_equals_the_engine_on_the_same_text(client, request):
    doc = rf.masking_set(request.node.nodeid)
    text = f"a {rf.MASKED} then {rf.UNMASKED} end"
    out = (await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": text})).json()
    direct = rule_signals.evaluate(rule_signals.load(doc, rule_signals.Limits()), rule_signals.Limits(), text=text)
    for field in ("rule_match", "rule_veto", "unknown", "reason", "fired", "expressions", "vetoed"):
        assert out[field] == direct.as_dict()[field], field
    assert out["rule_set_digest"] == rule_signals.rule_set_digest(doc)


async def test_a_veto_condition_read_as_a_match_is_named(client, request):
    """Review R1 (S3), TI-25: a set that also exposes its veto condition as a `fired` column (as a dataset-building
    set may: "is this value masked?") has that condition read as a MATCH, so a masked value matches and the veto
    can never hold. Not resolved here (an operator question); the result names it, and the plain set does not."""
    doc = rf.masking_set(request.node.nodeid)
    plain = (await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": f"id {rf.MASKED} end"})).json()
    assert (plain["rule_match"], plain["rule_veto"]) == (False, True)
    assert "match_includes_veto_condition" not in (plain.get("notes") or [])
    when = doc["vetoes"][0]["when"]
    exposed = dict(doc, outputs=doc["outputs"] + [{"column": "masked_col", "type": "bool", "value": {"fired": when}}],
                   cases=[])
    out = (await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": exposed, "text": f"id {rf.MASKED} end"})).json()
    assert (out["rule_match"], out["rule_veto"]) == (True, False)
    assert when in out["match_expressions"] and "match_includes_veto_condition" in out["notes"]


async def test_evaluate_through_the_behaviour_and_on_messages(client, request):
    rng = rng_for(request.node.nodeid)
    body = behaviour_body(rng)
    doc = rf.masking_set(request.node.nodeid, body["key"])
    assert (await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": doc}})).status_code == 201
    out = (await client.post(f"{API}/rule-signals/evaluate", json={
        "behaviour_key": body["key"], "scope": "last_user",
        "messages": [{"role": "user", "content": f"x {rf.UNMASKED}"}, {"role": "assistant", "content": "ok"},
                     {"role": "user", "content": "nothing"}]})).json()
    assert out["rule_match"] is False and out["scope"] == "last_user"
    out = (await client.post(f"{API}/rule-signals/evaluate", json={
        "behaviour_key": body["key"], "scope": "prompt",
        "messages": [{"role": "user", "content": f"x {rf.UNMASKED}"}, {"role": "assistant", "content": "ok"}]})).json()
    assert out["rule_match"] is True and out["scope"] == "input"
    span = next(f for f in out["fired"] if f["kind"] == "pattern")["spans"]
    assert span == [{"message_index": 0, "start": 2, "end": 13}]


async def test_a_behaviour_without_a_rule_set_is_named(client, request):
    rng = rng_for(request.node.nodeid)
    body = behaviour_body(rng)
    assert (await client.post(f"{API}/behaviours", json=body)).status_code == 201
    response = await client.post(f"{API}/rule-signals/evaluate", json={"behaviour_key": body["key"], "text": "x"})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "no_rule_set"


async def test_the_caps_refuse_naming_the_setting(client, request):
    from src.core import runtime_settings

    doc = rf.masking_set(request.node.nodeid)
    try:
        runtime_settings.apply({"rule_signals_max_messages": 2, "rule_text_max_code_points": 100})
        too_many = await client.post(f"{API}/rule-signals/evaluate", json={
            "rule_set": doc, "messages": [{"role": "user", "content": "a"}] * 3})
        too_long = await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": "x" * 101})
    finally:
        runtime_settings.reset()
    assert too_many.json()["detail"]["setting"] == "rule_signals_max_messages"
    assert too_long.json()["detail"]["setting"] == "rule_text_max_code_points"


async def test_the_engine_limits_are_the_runtime_settings(client, request):
    """The pattern memory cap reaches the engine from the runtime setting (a reader test, 040 §5.6)."""
    from src.core import runtime_settings

    rng = rng_for(request.node.nodeid)
    import string

    chars = string.ascii_letters + string.digits
    large = "(?:" + "|".join(f"{chars[i % 62]}{chars[i // 62]}\\w{{998}}" for i in range(20)) + ")"
    sid, expr, col = rf.ident(rng, "p"), rf.ident(rng, "e"), rf.ident(rng, "c")
    doc = rf.document(rf.key(rng), rng, signals=[rf.pattern(sid, large)], expressions={expr: {"signal": sid}},
                      outputs=[rf.fired(col, expr)])
    assert (await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})).json()["valid"] is True
    try:
        runtime_settings.apply({"rule_pattern_max_mem_bytes": 1024 * 1024})
        out = (await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})).json()
    finally:
        runtime_settings.reset()
    assert (out["valid"], out["code"], out["signal"]) == (False, "pattern_too_large", sid)


async def test_no_text_reaches_a_log_record(client, request, caplog):
    caplog.set_level(logging.DEBUG)
    doc = rf.masking_set(request.node.nodeid)
    sentinel = "zq-sentinel-text-9137"
    await client.post(f"{API}/rule-signals/evaluate", json={"rule_set": doc, "text": f"{sentinel} {rf.UNMASKED}"})
    await client.post(f"{API}/rule-sets/validate", json={"rule_set": doc})
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert sentinel not in joined and rf.UNMASKED not in joined


def test_routes_in_openapi():
    """Removing the router's include line turns this red (FTDD §10.2)."""
    from src.main import app

    paths = app.openapi()["paths"]
    assert "post" in paths["/api/v1/probe-monitors/rule-sets/validate"]
    assert "post" in paths["/api/v1/probe-monitors/rule-signals/evaluate"]
    assert [p for p in paths if p.endswith("/rule-sets/validate")] == ["/api/v1/probe-monitors/rule-sets/validate"]


@pytest.mark.parametrize("route", ["rule-sets/validate", "behaviours/{key}/success-reports"])
async def test_an_uncatalogued_behaviour_is_a_404_not_a_500(client, request, route):
    """Live 2026-10-24: `validate_rule_set` for an uncatalogued key answered 500 — the catalogue's
    `BehaviourRefused` escaped `resolve_version`, which every rule and report route shares."""
    key = behaviour_body(rng_for(request.node.nodeid))["key"]
    path = route.format(key=key)
    body = {"behaviour_key": key} if route == "rule-sets/validate" else {}
    response = await client.post(f"{API}/{path}", json=body)
    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "not_found"
