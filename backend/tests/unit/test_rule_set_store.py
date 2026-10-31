"""Rule sets on behaviour versions, S1 half (Feature 040 FR-1, FR-2, FR-45; FTASKS 2.7).

Through the REAL routes. miDataworks is an `httpx.MockTransport` behind `rule_set_store._transport_factory`,
asserted by request and call count. The identity is miDataworks 016's (RFC 8785 JCS SHA-256 of the document
as received, spec D38-4); `test_the_identity_is_midataworks_own` checks it against 016's published
`rule_set_identity` cases. S3 vendored that file (FTASKS 18.1), so this pin now reads the VENDORED package copy
and always runs; its byte identity with miDataworks' is `test_vendored_rule_set_schema.py`'s.
"""
from __future__ import annotations

import hashlib
import json

import httpx
import pytest
import rfc8785

from src.services import rule_set_store
from tests.support.next_calls import assert_next_calls_are_live
from tests.support.random_behaviour import behaviour_body, rng_for, slug

API = "/api/v1/probe-monitors"


def document(rng, key, **extra):
    """A VALID rule set (S3 validates at attach, FR-29): one pattern, one `fired` output, no cases."""
    sid = "p_" + slug(rng).replace("-", "_")
    doc = {
        "format": "midataworks.rule-set/v1", "name": slug(rng), "behaviour_key": key,
        "description": "d", "signals": [{"id": sid, "kind": "pattern", "pattern": "x", "reads": "text",
                                         "ignore_case": False, "fold_fullwidth_ascii": False}],
        "expressions": {"hit": {"signal": sid}},
        "outputs": [{"column": "hit", "type": "bool", "value": {"fired": "hit"}}], "cases": [],
    }
    doc.update(extra)
    return doc


def identity(doc):
    return hashlib.sha256(rfc8785.dumps(doc)).hexdigest()


class FakeMiDataworks:
    def __init__(self, envelopes=None, fail=False):
        self.envelopes = envelopes or {}
        self.fail = fail
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            raise httpx.ConnectError("refused", request=request)
        name = request.url.path.rsplit("/", 1)[-1]
        key = (name, int(request.url.params["version"]))
        if key not in self.envelopes:
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(200, json=self.envelopes[key])


@pytest.fixture
def rng(request):
    return rng_for(request.node.nodeid)


@pytest.fixture
def no_midataworks(monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "midataworks_api_url", "")


@pytest.fixture
def midataworks(monkeypatch):
    from src.core.config import settings

    fake = FakeMiDataworks()
    monkeypatch.setattr(settings, "midataworks_api_url", "http://dw.test")
    monkeypatch.setattr(rule_set_store, "_transport_factory", lambda: httpx.MockTransport(fake.handler))
    return fake


async def test_an_inline_set_works_with_no_midataworks(client, rng, no_midataworks):
    """FR-45: inline sets need no sibling. Stored once, under 016's identity; since S3 reported with its
    validation by this engine (no longer `unvalidated`)."""
    body = behaviour_body(rng)
    doc = document(rng, body["key"])
    made = await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": doc}})
    assert made.status_code == 201, made.text
    rule_set = made.json()["definition"]["rule_set"]
    validation = rule_set.pop("validation")
    assert rule_set == {"digest": identity(doc), "digest_scheme": "midataworks_rule_set_identity",
                        "source": "inline", "source_ref": None, "behaviour_key": body["key"]}
    assert validation["status"] == "valid" and validation["match_expressions"] == ["hit"]
    assert "unvalidated" not in json.dumps(made.json())


async def test_the_stored_bytes_are_the_canonical_document(client, async_session, rng, no_midataworks):
    body = behaviour_body(rng)
    doc = document(rng, body["key"])
    await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": doc}})
    row = await rule_set_store.get_by_digest(async_session, identity(doc))
    assert bytes(row.bytes) == rfc8785.dumps(doc) and json.loads(row.bytes) == doc


async def test_the_same_set_twice_is_one_row(client, async_session, rng, no_midataworks):
    from sqlalchemy import func, select

    from src.models.rule_set import RuleSet

    body = behaviour_body(rng)
    doc = document(rng, body["key"])
    await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": doc}})
    again = await client.post(f"{API}/behaviours/{body['key']}/versions",
                              json={"base_version": 1, "rule_set": {"document": dict(reversed(list(doc.items())))},
                                    "calibration_floor": 5})
    assert again.status_code == 201, again.text
    assert again.json()["definition"]["rule_set"]["digest"] == identity(doc)  # key order is not identity
    assert (await async_session.execute(select(func.count()).select_from(RuleSet))).scalar_one() == 1


async def test_a_set_for_another_behaviour_is_refused(client, rng, no_midataworks):
    body = behaviour_body(rng)
    response = await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": document(rng, slug(rng))}})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "rule_set_behaviour_mismatch"


async def test_a_document_of_another_format_is_refused(client, rng, no_midataworks):
    body = behaviour_body(rng)
    doc = document(rng, body["key"], format="something-else/v1")
    detail = (await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": doc}})).json()["detail"]
    assert (detail["code"], detail["field"], detail["rule"]) == ("invalid_field", "rule_set.document.format", "unknown_format")


async def test_an_oversized_set_is_refused_by_the_runtime_cap(client, rng, no_midataworks):
    from src.core import runtime_settings

    body = behaviour_body(rng)
    doc = document(rng, body["key"], description="x" * 3000)
    try:
        runtime_settings.apply({"rule_set_max_bytes": 2048})
        detail = (await client.post(f"{API}/behaviours", json={**body, "rule_set": {"document": doc}})).json()["detail"]
    finally:
        runtime_settings.reset()
    assert (detail["code"], detail["setting"], detail["value"]) == ("over_cap", "rule_set_max_bytes", 2048)


async def test_a_reference_is_fetched_once_and_kept(client, rng, midataworks):
    body = behaviour_body(rng)
    doc = document(rng, body["key"])
    ref = {"name": doc["name"], "version": 3, "digest": identity(doc)}
    midataworks.envelopes[(doc["name"], 3)] = {"document": doc, "sha256": identity(doc)}
    made = await client.post(f"{API}/behaviours", json={**body, "rule_set": ref})
    assert made.status_code == 201, made.text
    assert [(r.method, r.url.path, dict(r.url.params)) for r in midataworks.requests] == [
        ("GET", f"/api/v1/rule-sets/{doc['name']}", {"version": "3"})]
    described = made.json()["definition"]["rule_set"]
    assert (described["source"], described["source_ref"], described["digest"]) == ("referenced", ref, ref["digest"])
    # the same reference again: no second fetch (content-addressed)
    again = await client.post(f"{API}/behaviours/{body['key']}/versions",
                              json={"base_version": 1, "rule_set": ref, "calibration_floor": 9})
    assert again.status_code == 201 and len(midataworks.requests) == 1


async def test_a_reference_whose_content_differs_is_refused_naming_both(client, rng, midataworks):
    body = behaviour_body(rng)
    doc = document(rng, body["key"])
    named = "f" * 64
    midataworks.envelopes[(doc["name"], 1)] = {"document": doc, "sha256": identity(doc)}
    detail = (await client.post(f"{API}/behaviours", json={
        **body, "rule_set": {"name": doc["name"], "version": 1, "digest": named}})).json()["detail"]
    assert (detail["code"], detail["requested"], detail["computed"]) == ("rule_set_digest_mismatch", named, identity(doc))


async def test_a_served_digest_that_disagrees_is_refused(client, rng, midataworks):
    body = behaviour_body(rng)
    doc = document(rng, body["key"])
    midataworks.envelopes[(doc["name"], 1)] = {"document": doc, "sha256": "e" * 64}
    detail = (await client.post(f"{API}/behaviours", json={
        **body, "rule_set": {"name": doc["name"], "version": 1, "digest": identity(doc)}})).json()["detail"]
    assert (detail["code"], detail["served"]) == ("rule_set_digest_mismatch", "e" * 64)


@pytest.mark.parametrize("state", ["not_configured", "unreachable", "not_found"])
async def test_an_unresolvable_reference_answers_with_the_inline_call(client, rng, monkeypatch, state):
    from src.core.config import settings

    fake = FakeMiDataworks(fail=state == "unreachable")
    monkeypatch.setattr(settings, "midataworks_api_url", "" if state == "not_configured" else "http://dw.test")
    monkeypatch.setattr(rule_set_store, "_transport_factory", lambda: httpx.MockTransport(fake.handler))
    body = behaviour_body(rng, target_fpr=0.01)
    response = await client.post(f"{API}/behaviours", json={
        **body, "rule_set": {"name": "abc", "version": 1, "digest": "a" * 64}})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "rule_set_unresolvable"
    assert detail["next"][0]["tool"] == "create_behaviour" and detail["next"][0]["needs"] == ["rule_set"]
    assert detail["next"][0]["arguments"]["key"] == body["key"]
    assert "rule_set" not in detail["next"][0]["arguments"]
    assert_next_calls_are_live(detail["next"])
    assert (await client.get(f"{API}/behaviours/{body['key']}")).status_code == 404


async def test_the_version_route_answers_with_its_own_inline_call(client, rng, no_midataworks):
    body = behaviour_body(rng)
    await client.post(f"{API}/behaviours", json=body)
    detail = (await client.post(f"{API}/behaviours/{body['key']}/versions", json={
        "base_version": 1, "rule_set": {"name": "abc", "version": 1, "digest": "a" * 64}})).json()["detail"]
    assert detail["next"][0]["tool"] == "new_behaviour_version"
    assert detail["next"][0]["arguments"] == {"key": body["key"], "base_version": 1}
    assert_next_calls_are_live(detail["next"])


def _identity_cases():
    from src.rule_signals.document import CASES_PATH

    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["rule_set_identity"]


def test_the_identity_is_midataworks_own():
    """Every published `rule_set_identity` case: the bytes and the digest this module stores are 016's
    (FTASKS 18.3: the S1 inline-digest pin now runs on the vendored copy)."""
    from src.services.detector_identity import jcs_bytes, rule_set_identity

    cases = _identity_cases()
    assert len(cases) >= 5
    for case in cases:
        assert jcs_bytes(case["document"]).decode("utf-8") == case["canonical_utf8"], case.get("id")
        assert rule_set_identity(case["document"]) == case["sha256"], case.get("id")


