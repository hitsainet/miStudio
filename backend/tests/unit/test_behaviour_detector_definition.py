"""Detector fields on behaviour versions (Feature 040 FR-1 – FR-5, FR-7; FTASKS 2.3 – 2.6).

Rules are tested twice where it matters: on `behaviour_detector_definition.validate` itself, and through the
REAL route (`POST /behaviours`, `POST /behaviours/{key}/versions`) so a refusal test fails when the
catalogue stops calling `validate` (the 2.5 removal control). Keys, names, sub-question keys, criterion
ids and columns are random slugs seeded from the test id (`tests/support/random_behaviour.py`).
"""
from __future__ import annotations

import pytest

from src.core import runtime_settings
from src.services import behaviour_detector_definition as rules
from src.services.detector_next import DetectorRefused
from tests.support.next_calls import assert_next_calls_are_live
from tests.support.random_behaviour import behaviour_body, rng_for, slug

API = "/api/v1/probe-monitors"
LIMITS = rules.Limits(max_sub_questions=4, max_success_criteria=4, max_held_out_columns=4, max_models=4,
                      rule_set_max_bytes=1 << 20)


@pytest.fixture(autouse=True)
def _fresh_settings():
    runtime_settings.reset()
    yield
    runtime_settings.reset()


@pytest.fixture
def rng(request):
    return rng_for(request.node.nodeid)


def criterion(rng, **overrides):
    out = {"id": slug(rng), "detector": "best_single", "subset": "all", "min_recall": 0.8}
    out.update(overrides)
    return out


def fields(**overrides):
    out = {"sub_questions": None, "success_criteria": None, "held_out_columns": None, "models": None,
           "calibration_floor": None}
    out.update(overrides)
    return out


def refusal(call):
    with pytest.raises(DetectorRefused) as caught:
        call()
    return caught.value


# ── validate(), rule by rule (FR-2) ──────────────────────────────────────────────────────────────


def test_duplicate_sub_question_key(rng):
    key = slug(rng)
    subs = [{"key": key, "question": "q", "positive_meaning": "p"}] * 2
    r = refusal(lambda: rules.validate(fields(sub_questions=subs), None, LIMITS))
    assert (r.code, r.details["field"], r.details["rule"]) == ("invalid_field", "sub_questions[1].key", "duplicate_key")


def test_duplicate_criterion_id(rng):
    c = criterion(rng)
    r = refusal(lambda: rules.validate(fields(success_criteria=[c, dict(c)]), 0.01, LIMITS))
    assert (r.details["field"], r.details["rule"]) == ("success_criteria[1].id", "duplicate_id")


def test_held_out_test_must_be_declared(rng):
    declared, other = slug(rng), slug(rng)
    c = criterion(rng, held_out_test=other)
    r = refusal(lambda: rules.validate(fields(success_criteria=[c], held_out_columns=[declared]), 0.01, LIMITS))
    assert (r.details["field"], r.details["rule"], r.details["declared"]) == (
        "success_criteria[0].held_out_test", "not_declared", [declared])
    rules.validate(fields(success_criteria=[criterion(rng, held_out_test=declared)], held_out_columns=[declared]),
                   0.01, LIMITS)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_target_fpr_bounds(rng, value):
    r = refusal(lambda: rules.validate(fields(success_criteria=[criterion(rng, target_fpr=value)]), None, LIMITS))
    assert (r.details["field"], r.details["rule"]) == ("success_criteria[0].target_fpr", "out_of_range")


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_min_recall_bounds(rng, value):
    r = refusal(lambda: rules.validate(fields(success_criteria=[criterion(rng, min_recall=value)]), 0.01, LIMITS))
    assert (r.details["field"], r.details["rule"]) == ("success_criteria[0].min_recall", "out_of_range")


def test_a_criterion_without_a_target_inherits_the_versions(rng):
    """FR-2 inheritance: valid when the version sets one, and the effective target is the version's."""
    c = criterion(rng)
    rules.validate(fields(success_criteria=[c]), 0.02, LIMITS)
    assert rules.effective_target_fpr(c, 0.02) == 0.02
    assert rules.effective_target_fpr({**c, "target_fpr": 0.005}, 0.02) == 0.005


def test_no_target_anywhere_is_refused_naming_both(rng):
    r = refusal(lambda: rules.validate(fields(success_criteria=[criterion(rng)]), None, LIMITS))
    assert r.details["rule"] == "target_fpr_missing"
    assert r.details["fields"] == ["success_criteria[0].target_fpr", "target_fpr"]


def test_named_probe_needs_a_probe_and_only_it_takes_one(rng):
    r = refusal(lambda: rules.validate(fields(success_criteria=[criterion(rng, detector="named_probe")]), 0.01, LIMITS))
    assert r.details["rule"] == "required_by_named_probe"
    r = refusal(lambda: rules.validate(fields(success_criteria=[criterion(rng, probe_id="pm_x")]), 0.01, LIMITS))
    assert r.details["rule"] == "only_with_named_probe"
    rules.validate(fields(success_criteria=[criterion(rng, detector="named_probe", probe_id="pm_x")]), 0.01, LIMITS)


def test_calibration_floor_at_least_one():
    assert refusal(lambda: rules.validate(fields(calibration_floor=0), None, LIMITS)).details["field"] == "calibration_floor"
    rules.validate(fields(calibration_floor=1), None, LIMITS)


def test_duplicate_held_out_column_and_model(rng):
    column = slug(rng)
    assert refusal(lambda: rules.validate(fields(held_out_columns=[column, column]), None, LIMITS)).details["rule"] == "duplicate_column"
    model = {"model_id": slug(rng)}
    assert refusal(lambda: rules.validate(fields(models=[model, model]), None, LIMITS)).details["rule"] == "duplicate_model"


# ── FR-7 caps: runtime settings, never literals ─────────────────────────────────────────────────

CAP_CASES = [
    ("sub_questions", "behaviour_max_sub_questions", lambda rng, i: {"key": slug(rng), "question": "q", "positive_meaning": "p"}),
    ("success_criteria", "behaviour_max_success_criteria", lambda rng, i: criterion(rng, target_fpr=0.01)),
    ("held_out_columns", "behaviour_max_held_out_columns", lambda rng, i: slug(rng)),
    ("models", "behaviour_max_models", lambda rng, i: {"model_id": slug(rng)}),
]


@pytest.mark.parametrize("field,setting,make", CAP_CASES, ids=[c[0] for c in CAP_CASES])
def test_each_cap_is_the_runtime_setting(rng, field, setting, make):
    """Set the cap at runtime to 2: three entries are refused `over_cap` naming the setting; two pass."""
    runtime_settings.apply({setting: 2})
    limits = rules.Limits.from_settings()
    three = [make(rng, i) for i in range(3)]
    r = refusal(lambda: rules.validate(fields(**{field: three}), None, limits))
    assert (r.code, r.details["setting"], r.details["value"], r.details["got"]) == ("over_cap", setting, 2, 3)
    rules.validate(fields(**{field: three[:2]}), None, limits)


def test_limits_follow_a_runtime_change():
    runtime_settings.apply({"rule_set_max_bytes": 4096})
    assert rules.Limits.from_settings().rule_set_max_bytes == 4096


# ── carry-over and diff (FR-3) ──────────────────────────────────────────────────────────────────


class _Row:
    def __init__(self, **values):
        self.__dict__.update(values)


def test_carry_over_takes_unsent_fields_from_the_named_base(rng):
    names = ("name", "held_out_columns", "success_criteria")
    base = _Row(name="base-name", held_out_columns=[slug(rng)], success_criteria=[criterion(rng)])
    change = {"name": None, "held_out_columns": None, "success_criteria": [criterion(rng)]}
    out = rules.carry_over(base, change, {"success_criteria"}, names)
    assert out["name"] == "base-name" and out["held_out_columns"] == base.held_out_columns
    assert out["success_criteria"] == change["success_criteria"]
    # a field SENT as null is removed (FR-4)
    assert rules.carry_over(base, change, {"held_out_columns"}, names)["held_out_columns"] is None


def test_field_diff_ignores_key_order_and_lists_exact_paths(rng):
    c = criterion(rng, target_fpr=0.01)
    reordered = dict(reversed(list(c.items())))
    before = {"name": "a", "success_criteria": [c], "models": None}
    after = {"name": "a", "success_criteria": [reordered], "models": [{"model_id": "x"}]}
    assert rules.field_diff(before, after, ("name", "success_criteria", "models")) == [
        {"path": "models", "before": None, "after": [{"model_id": "x"}]}]


# ── through the route (FR-1 round trip, FR-2 wiring, FR-3, FR-6 compatibility) ─────────────────


def full_body(rng, **overrides):
    held = slug(rng)
    body = behaviour_body(rng, target_fpr=0.02)
    body.update({
        "sub_questions": [{"key": slug(rng), "question": "q?", "positive_meaning": "yes"}],
        "success_criteria": [
            criterion(rng, held_out_test=held),
            criterion(rng, detector="named_probe", probe_id="pm_abc", subset={"column": slug(rng), "value": "v"},
                      target_fpr=0.005),
        ],
        "held_out_columns": [held],
        "models": [{"model_id": f"{slug(rng)}/{slug(rng)}"}],
        "calibration_floor": 3000,
    })
    body.update(overrides)
    return body


DETECTOR_KEYS = ("sub_questions", "success_criteria", "held_out_columns", "models", "calibration_floor")


async def test_every_field_round_trips_through_rest(client, rng):
    body = full_body(rng)
    made = await client.post(f"{API}/behaviours", json=body)
    assert made.status_code == 201, made.text
    read = (await client.get(f"{API}/behaviours/{body['key']}")).json()["definition"]
    for field in DETECTOR_KEYS:
        assert read[field] == body[field], field
    versions = (await client.get(f"{API}/behaviours/{body['key']}/versions")).json()["versions"]
    assert versions[0]["success_criteria"] == body["success_criteria"]


async def test_a_pre_feature_body_reads_with_no_detector_keys(client, rng):
    """FR-1: a 037-shaped behaviour reads exactly as before — the six keys absent, never null/invalid."""
    body = behaviour_body(rng)
    made = (await client.post(f"{API}/behaviours", json=body)).json()
    assert set(made["definition"]) == {
        "version", "name", "description", "positive_meaning", "negative_meaning", "target_fpr",
        "created_by", "created_at"}
    assert made["warnings"] == []


async def test_declared_empty_is_kept_distinct_from_absent(client, rng):
    body = behaviour_body(rng, held_out_columns=[], sub_questions=[])
    made = (await client.post(f"{API}/behaviours", json=body)).json()
    assert made["definition"]["held_out_columns"] == [] and made["definition"]["sub_questions"] == []
    assert "models" not in made["definition"]


ROUTE_RULES = [
    ("duplicate_key", lambda rng: {"sub_questions": [{"key": "aa-bb", "question": "q", "positive_meaning": "p"}] * 2},
     "sub_questions[1].key"),
    ("duplicate_id", lambda rng: {"success_criteria": [criterion(rng, id="cc-dd", target_fpr=0.1)] * 2},
     "success_criteria[1].id"),
    ("not_declared", lambda rng: {"success_criteria": [criterion(rng, held_out_test="zz", target_fpr=0.1)]},
     "success_criteria[0].held_out_test"),
    ("out_of_range", lambda rng: {"success_criteria": [criterion(rng, target_fpr=1.0)]},
     "success_criteria[0].target_fpr"),
    ("target_fpr_missing", lambda rng: {"success_criteria": [criterion(rng)], "target_fpr": None},
     "success_criteria[0].target_fpr"),
    ("required_by_named_probe", lambda rng: {"success_criteria": [criterion(rng, detector="named_probe", target_fpr=0.1)]},
     "success_criteria[0].probe_id"),
]


@pytest.mark.parametrize("rule,extra,field", ROUTE_RULES, ids=[r[0] for r in ROUTE_RULES])
async def test_each_rule_refuses_through_the_create_route(client, rng, rule, extra, field):
    body = behaviour_body(rng)
    body.update(extra(rng))
    response = await client.post(f"{API}/behaviours", json=body)
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert (detail["code"], detail["rule"], detail["field"]) == ("invalid_field", rule, field)
    assert detail["next"] == []
    assert (await client.get(f"{API}/behaviours/{body['key']}")).status_code == 404  # nothing written


@pytest.mark.parametrize("rule,extra,field", ROUTE_RULES, ids=[r[0] for r in ROUTE_RULES])
async def test_each_rule_refuses_through_the_version_route(client, rng, rule, extra, field):
    body = behaviour_body(rng)
    assert (await client.post(f"{API}/behaviours", json=body)).status_code == 201
    version = {"name": body["name"], "positive_meaning": body["positive_meaning"], **extra(rng)}
    response = await client.post(f"{API}/behaviours/{body['key']}/versions", json=version)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["rule"] == rule


async def test_over_cap_through_the_route(client, rng):
    runtime_settings.apply({"behaviour_max_held_out_columns": 1})
    body = behaviour_body(rng, held_out_columns=[slug(rng), slug(rng)])
    detail = (await client.post(f"{API}/behaviours", json=body)).json()["detail"]
    assert (detail["code"], detail["setting"], detail["value"]) == ("over_cap", "behaviour_max_held_out_columns", 1)


@pytest.mark.parametrize("form", ["rules_or_blend", "blend_alone", "best_single", "rules_alone", "named_probe"])
async def test_each_detector_form_is_accepted(client, rng, form):
    extra = {"probe_id": "pm_x"} if form == "named_probe" else {}
    body = behaviour_body(rng, target_fpr=0.01, success_criteria=[criterion(rng, detector=form, **extra)])
    assert (await client.post(f"{API}/behaviours", json=body)).status_code == 201


@pytest.mark.parametrize("subset", ["all", "rule_invisible_positives", "rule_visible_positives", "FILTER"])
async def test_each_subset_is_accepted(client, rng, subset):
    value = {"column": slug(rng), "value": slug(rng)} if subset == "FILTER" else subset
    body = behaviour_body(rng, target_fpr=0.01, success_criteria=[criterion(rng, subset=value)])
    made = await client.post(f"{API}/behaviours", json=body)
    assert made.status_code == 201 and made.json()["definition"]["success_criteria"][0]["subset"] == value


async def test_an_unknown_form_is_refused_listing_the_allowed(client, rng):
    body = behaviour_body(rng, target_fpr=0.01, success_criteria=[criterion(rng, detector=slug(rng))])
    response = await client.post(f"{API}/behaviours", json=body)
    assert response.status_code == 422
    text = response.text
    for form in ("rules_or_blend", "blend_alone", "best_single", "rules_alone", "named_probe"):
        assert form in text


async def test_a_model_not_on_the_deployment_saves_with_a_live_next(client, rng):
    repo = f"{slug(rng)}/{slug(rng)}"
    body = behaviour_body(rng, models=[{"model_id": repo}])
    made = await client.post(f"{API}/behaviours", json=body)
    assert made.status_code == 201
    warnings = made.json()["warnings"]
    assert [w["code"] for w in warnings] == ["model_not_downloaded"]
    assert warnings[0]["next"] == [{"tool": "download_model", "arguments": {"repo_id": repo}, "why": warnings[0]["next"][0]["why"]}]
    assert_next_calls_are_live(warnings[0]["next"])


async def test_a_column_no_view_declares_is_a_warning(client, rng):
    column = slug(rng)
    made = await client.post(f"{API}/behaviours", json=behaviour_body(rng, held_out_columns=[column]))
    assert made.status_code == 201
    warning = made.json()["warnings"][0]
    assert (warning["code"], warning["column"]) == ("column_not_on_any_view", column)
    assert_next_calls_are_live(warning["next"])


# ── FR-3: a change against a base version ───────────────────────────────────────────────────────


async def _three_versions(client, rng):
    body = full_body(rng)
    key = body["key"]
    assert (await client.post(f"{API}/behaviours", json=body)).status_code == 201
    v2 = await client.post(f"{API}/behaviours/{key}/versions",
                           json={"base_version": 1, "calibration_floor": 4000})
    assert v2.status_code == 201, v2.text
    v3 = await client.post(f"{API}/behaviours/{key}/versions",
                           json={"base_version": 2, "held_out_columns": body["held_out_columns"] + [slug(rng)]})
    assert v3.status_code == 201, v3.text
    return body, key


async def test_one_changed_criterion_keeps_every_other_field_byte_identical(client, rng):
    body, key = await _three_versions(client, rng)
    before = (await client.get(f"{API}/behaviours/{key}")).json()["definition"]
    changed = [dict(before["success_criteria"][0], min_recall=0.55), before["success_criteria"][1]]
    response = await client.post(f"{API}/behaviours/{key}/versions",
                                 json={"base_version": 3, "success_criteria": changed})
    assert response.status_code == 201, response.text
    out = response.json()
    after = out["definition"]
    assert out["base_version"] == 3 and after["version"] == 4
    assert [d["path"] for d in out["diff"]] == ["success_criteria"]
    for field in ("name", "positive_meaning", "target_fpr", "sub_questions", "held_out_columns", "models",
                  "calibration_floor"):
        assert after[field] == before[field], field
    # carried from version 3 (which differs from versions 1 and 2), not from an earlier one
    assert after["calibration_floor"] == 4000 and len(after["held_out_columns"]) == 2


async def test_a_stale_base_is_refused_naming_the_current_version(client, rng):
    body, key = await _three_versions(client, rng)
    response = await client.post(f"{API}/behaviours/{key}/versions", json={"base_version": 2, "calibration_floor": 7})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert (detail["code"], detail["current_version"], detail["base_version"]) == ("stale_base_version", 3, 2)
    assert_next_calls_are_live(detail["next"])


async def test_null_with_a_base_removes_the_field(client, rng):
    body, key = await _three_versions(client, rng)
    out = (await client.post(f"{API}/behaviours/{key}/versions",
                             json={"base_version": 3, "sub_questions": None})).json()
    assert "sub_questions" not in out["definition"]
    assert out["diff"] == [{"path": "sub_questions", "before": body["sub_questions"], "after": None}]


async def test_an_identical_change_is_no_change(client, rng):
    body, key = await _three_versions(client, rng)
    response = await client.post(f"{API}/behaviours/{key}/versions",
                                 json={"base_version": 3, "calibration_floor": 4000})
    assert response.status_code == 422 and response.json()["detail"]["code"] == "no_change"
    assert response.json()["detail"]["next"] == []


async def test_full_replacement_without_a_base_is_037s(client, rng):
    body = behaviour_body(rng)
    key = body["key"]
    await client.post(f"{API}/behaviours", json=body)
    response = await client.post(f"{API}/behaviours/{key}/versions", json={"positive_meaning": "x"})
    assert response.status_code == 422
    assert response.json()["detail"]["rule"] == "required_without_base_version"
    ok = await client.post(f"{API}/behaviours/{key}/versions", json={"name": "n2", "positive_meaning": "x"})
    assert ok.status_code == 201 and ok.json()["definition"]["description"] is None


# ── FR-41 (S1): the checklist's definition step follows real state ──────────────────────────────


async def _definition_step(client, key):
    body = (await client.get(f"{API}/behaviours/{key}/checklist")).json()
    return next(s for s in body["steps"] if s["id"] == "definition_complete")


async def test_definition_step_follows_the_current_version(client, rng):
    body = behaviour_body(rng, target_fpr=0.01)
    key = body["key"]
    await client.post(f"{API}/behaviours", json=body)
    step = await _definition_step(client, key)                       # pre-feature shape: no criteria
    assert step["state"] == "not_done"
    assert step["next"][0]["tool"] == "new_behaviour_version" and step["next"][0]["arguments"] == {
        "key": key, "base_version": 1}
    held = slug(rng)
    await client.post(f"{API}/behaviours/{key}/versions", json={
        "base_version": 1, "held_out_columns": [held], "success_criteria": [criterion(rng, held_out_test=held)]})
    step = await _definition_step(client, key)
    assert (step["state"], step["evidence"]) == ("done", [f"{key}@2"])
    await client.post(f"{API}/behaviours/{key}/versions", json={"base_version": 2, "success_criteria": []})
    assert (await _definition_step(client, key))["state"] == "not_done"   # declared empty is not complete


async def test_definition_step_is_computed_per_version_not_hard_wired(rng):
    """The step reads the version it is given (unit, no route): two versions, two states."""
    from src.services.behaviour_checklist import definition_step

    class V:
        version = 3
        success_criteria = [criterion(rng)]
        held_out_columns = None

    class Empty:
        version = 1
        success_criteria = None
        held_out_columns = None

    assert definition_step("k-x", V())["state"] == "done"
    assert definition_step("k-x", Empty())["state"] == "not_done"
    assert definition_step("k-x", None)["state"] == "not_done"


# ── FR-11: no process-wide current behaviour ─────────────────────────────────────────────────────


def test_two_behaviours_concurrently_give_their_sequential_bytes(request):
    """Validation, carry-over and diff of two random behaviours, interleaved on two threads."""
    from tests.support.concurrency import assert_concurrent_equals_sequential

    rng = rng_for(request.node.nodeid)
    items = []
    for _ in range(2):
        held = slug(rng)
        items.append({"held_out_columns": [held], "success_criteria": [criterion(rng, held_out_test=held)],
                      "sub_questions": [{"key": slug(rng), "question": "q", "positive_meaning": "p"}],
                      "models": [{"model_id": slug(rng)}], "calibration_floor": rng.randint(1, 9999)})
    names = tuple(items[0])

    def evaluate(item):
        rules.validate(item, 0.01, LIMITS)
        base = _Row(**{k: None for k in names})
        merged = rules.carry_over(base, item, set(item), names)
        return {"diff": rules.field_diff({}, merged, names), "columns": rules.columns_named(merged)}

    assert_concurrent_equals_sequential(evaluate, items)
