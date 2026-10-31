"""`millm_recalibrate_probe` is an HTTP push with six checks, and both halves of its payload are load-bearing.

⚠ **THE FIRST VERSION OF THIS TOOL DID THE RE-CUT ITSELF, AND COULD NOT HAVE RUN.**

It imported `src.services.probe_recalibration` and opened a `SyncSessionLocal`. The MCP server is
a SEPARATE deployment with no `DATABASE_URL` — `src.core.config.Settings` fails there with six
missing fields — so the tool was registered in all four layers, payload-asserted, present in the
live registry of the deployed server, and would have raised on its first real call. Every check
passed because every check ran in a process that happened to have a database.

`test_mcp_tools_stay_on_http.py` is the guard that now makes that impossible.

**Feature 039 (FR-15, D13, D-T10)** turned the push into six steps: read miStudio's decision in force
(`ctx.mistudio`, the new `GET …/decision` route), require the agent's bar to EQUAL it, require any
identity the agent passes to agree (and fill the rest), compare the detector miLLM holds, detect
miLLM's request fields from its OpenAPI (cached), then ONE POST carrying miStudio's WHOLE decision —
`calibration` included, because miLLM refuses a digest that does not name the decision it stores
(`PROBE_IDENTITY_MISMATCH`, cross-repo finding 6.4). Zero POSTs on any refusal.

INTENDED CHANGE (FPRD §13, recorded in the 039 controls record): `test_it_issues_exactly_ONE_call`
became exact counts — cold cache 3 GETs + 1 POST, warm cache 2 GETs + 1 POST.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.tools import millm_probes as probe_tools
from tests.support.external_ctx import make_ctx

MILLM_ID = "pr_abc123"
MISTUDIO_ID = "pm_36d1a65f7953"
D = "d" * 64
E = "e" * 64
ALGO = "mistudio.probe-identity/1"

#: miStudio's decision in force, in contract form (calibration included).
DECISION = {
    "threshold": 14.5201,
    "target_fpr": 0.005,
    "realised_fpr": 0.005,
    "threshold_source": "calibration_set",
    "calibration": {"hf_id": "org/cal-set", "config": "default", "split": "test", "revision": None},
    "windows": None,
    "length_bands": None,
}
#: What the agent passes: the re-cut's committed bar (the same numbers).
PROPOSED = {k: DECISION[k] for k in ("threshold", "target_fpr", "realised_fpr", "threshold_source")}

OPENAPI_WITH_FIELDS = {"components": {"schemas": {"ProbeRecalibrationRequest": {"properties": {
    "decision": {}, "mistudio_probe_id": {}, "calibration_id": {}, "reason": {},
    "decision_digest": {}, "decision_revision": {}}}}}}
OPENAPI_OLD = {"components": {"schemas": {"ProbeRecalibrationRequest": {"properties": {
    "decision": {}, "mistudio_probe_id": {}, "calibration_id": {}, "reason": {}}}}}}


@pytest.fixture(autouse=True)
def _cold_cache():
    probe_tools._OPENAPI_CACHE.clear()
    yield
    probe_tools._OPENAPI_CACHE.clear()


def _register(*, current=None, millm_probe=None, openapi=OPENAPI_WITH_FIELDS, mistudio_error=None,
              ttl=60):
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("test")
    client = MagicMock()
    client.base_url = "http://millm.test"
    current = current if current is not None else {
        "probe_id": MISTUDIO_ID, "decision": DECISION, "decision_digest": E,
        "decision_revision": 3, "detector_digest": D}
    millm_probe = millm_probe if millm_probe is not None else {
        "id": MILLM_ID, "detector_digest": D, "identity_algorithm": ALGO}

    async def millm_get(path, **_k):
        if path == "/openapi.json":
            return openapi
        if path == f"/api/probes/{MILLM_ID}":
            return millm_probe
        raise AssertionError(f"unexpected miLLM GET {path}")

    client.get = AsyncMock(side_effect=millm_get)
    client.post = AsyncMock(return_value={"registry_updated": True})
    gate = MagicMock()
    # ⚠ `(ok, reason)`, NOT `None`. `@gated` unpacks the result.
    gate.check = AsyncMock(return_value=(True, None))
    mistudio = MagicMock()
    mistudio.get = AsyncMock(
        side_effect=mistudio_error if mistudio_error is not None else (lambda *a, **k: current))
    probe_tools.register(mcp, client, gate, make_ctx(mistudio, millm_openapi_ttl_seconds=ttl))
    return mcp, client, mistudio


def _fn(mcp):
    try:
        return asyncio.run(mcp.get_tool("millm_recalibrate_probe")).fn
    except Exception:
        return mcp._tool_manager._tools["millm_recalibrate_probe"].fn


def _call(registered=None, **over):
    mcp, client, mistudio = registered or _register()
    kwargs = {"probe_id": MILLM_ID, "mistudio_probe_id": MISTUDIO_ID, "reason": "tighter budget",
              **PROPOSED}
    kwargs.update(over)
    out = asyncio.run(_fn(mcp)(**kwargs))
    return client, mistudio, out


def _gets(client):
    return [c.args[0] for c in client.get.await_args_list]


class TestItIsRegisteredAndStaysOnHttp:
    def test_it_is_in_the_built_server(self):
        mcp, _client, _m = _register()
        names = {t.name for t in asyncio.run(mcp.list_tools())}
        assert "millm_recalibrate_probe" in names

    def test_cold_cache_is_three_GETs_and_ONE_post(self):
        client, mistudio, _out = _call()
        assert mistudio.get.await_count == 1
        assert mistudio.get.await_args.args[0] == f"/probe-monitors/probes/{MISTUDIO_ID}/decision"
        assert _gets(client) == [f"/api/probes/{MILLM_ID}", "/openapi.json"]
        assert client.post.await_count == 1

    def test_warm_cache_is_two_GETs_and_ONE_post(self):
        registered = _register()
        _call(registered)
        client, mistudio, _out = _call(registered)
        assert mistudio.get.await_count == 2              # one per call
        assert _gets(client) == [f"/api/probes/{MILLM_ID}", "/openapi.json", f"/api/probes/{MILLM_ID}"]
        assert client.post.await_count == 2

    def test_ttl_zero_disables_the_cache(self):
        registered = _register(ttl=0)
        _call(registered)
        client, _m, _out = _call(registered)
        assert _gets(client).count("/openapi.json") == 2

    def test_an_expired_entry_is_fetched_again(self, monkeypatch):
        """The cache honours its TTL: past it, miLLM's OpenAPI is read again (a miLLM upgraded to
        take the identity fields is noticed within the TTL, not at the next MCP restart)."""
        clock = [1000.0]
        monkeypatch.setattr(probe_tools.time, "monotonic", lambda: clock[0])
        registered = _register(ttl=60)
        _call(registered)
        clock[0] += 59
        _call(registered)
        clock[0] += 2
        client, _m, _out = _call(registered)
        assert _gets(client).count("/openapi.json") == 2

    def test_it_posts_to_the_probes_own_path_with_json_body(self):
        client, _m, _out = _call()
        assert client.post.await_args.args[0] == f"/api/probes/{MILLM_ID}/recalibrate"
        assert "json_body" in client.post.await_args.kwargs
        assert "params" not in client.post.await_args.kwargs


class TestThePayloadIsMiStudiosWholeDecision:
    def test_the_decision_is_miStudios_calibration_included(self):
        """⚠ The cross-repo finding of 6.4: miLLM digests the decision it STORES; a decision rebuilt
        from the agent's four numbers (no `calibration`) would be refused for every calibration-set
        probe."""
        body = _call()[0].post.await_args.kwargs["json_body"]
        assert body["decision"] == DECISION
        assert body["decision"]["calibration"]["hf_id"] == "org/cal-set"
        assert body["mistudio_probe_id"] == MISTUDIO_ID and body["reason"] == "tighter budget"

    def test_identity_is_filled_from_miStudio_and_calibration_id_is_the_digest(self):
        body = _call()[0].post.await_args.kwargs["json_body"]
        assert body["calibration_id"] == E
        assert body["decision_digest"] == E and body["decision_revision"] == 3

    def test_an_older_miLLM_gets_calibration_id_only(self):
        client, _m, out = _call(_register(
            openapi=OPENAPI_OLD, millm_probe={"id": MILLM_ID}))
        body = client.post.await_args.kwargs["json_body"]
        assert "decision_digest" not in body and "decision_revision" not in body
        assert body["calibration_id"] == E and client.post.await_count == 1
        assert out["detector_unverified"] is True and "predates" in out["detector_unverified_reason"]

    def test_the_MISTUDIO_id_travels_and_is_not_confused_with_miLLMs(self):
        body = _call()[0].post.await_args.kwargs["json_body"]
        assert body["mistudio_probe_id"].startswith("pm_") and body["mistudio_probe_id"] != MILLM_ID

    def test_windows_are_compared_after_stripping_bookkeeping(self):
        """`contract_windows` drops `scores_path`/`lengths_path`; with them stripped the agent's
        windows equal the decision's and the push goes ahead, carrying the contract form only."""
        windows_in_force = {"prompt": {"threshold": 7.0, "target_fpr": 0.005, "realised_fpr": 0.005,
                                       "n_negatives": 2000, "scope": "user", "length_bands": None}}
        current = {"decision": {**DECISION, "windows": windows_in_force}, "decision_digest": E,
                   "decision_revision": 3, "detector_digest": D}
        agent_windows = {"prompt": {**windows_in_force["prompt"], "scores_path": "/data/w.npy"}}
        client, _m, out = _call(_register(current=current), windows=agent_windows)
        assert client.post.await_count == 1
        sent = client.post.await_args.kwargs["json_body"]["decision"]["windows"]
        assert "scores_path" not in sent["prompt"]


class TestEveryRefusalPostsNothing:
    def test_miStudio_unreachable(self):
        client, _m, out = _call(_register(mistudio_error=BackendError(0, "unreachable")))
        assert out["unavailable"] == "mistudio" and client.post.await_count == 0
        assert client.get.await_count == 0

    def test_a_decision_miStudio_cannot_form(self):
        current = {"decision": None, "decision_digest": None, "decision_revision": 1,
                   "detail": "the calibration dataset has no HuggingFace identifier"}
        client, _m, out = _call(_register(current=current))
        assert out["refused"] == "decision_not_formable" and client.post.await_count == 0
        assert out["next"][0]["tool"] == "build_probe_definition"

    @pytest.mark.parametrize("field, value", [
        ("threshold", 14.5202), ("target_fpr", 0.01), ("realised_fpr", 0.0049),
        ("threshold_source", "validation_negatives"),
        ("length_bands", [{"min_tokens": 0, "max_tokens": None, "threshold": 9.0}]),
        ("windows", {"prompt": {"threshold": 7.0}}),
    ])
    def test_values_that_are_not_the_decision_in_force(self, field, value):
        client, _m, out = _call(**{field: value})
        assert out["refused"] == "decision_not_current" and client.post.await_count == 0
        assert [d["field"] for d in out["differences"]] == [field]
        assert out["next"][0]["tool"] == "recalibrate_probe_monitor"
        assert out["next"][0]["arguments"]["preview"] is False

    @pytest.mark.parametrize("field, value", [
        ("decision_digest", "f" * 64), ("decision_revision", 2), ("calibration_id", "pmd_other"),
    ])
    def test_an_identity_that_disagrees(self, field, value):
        client, _m, out = _call(**{field: value})
        assert out["refused"] == "decision_identity_mismatch" and client.post.await_count == 0
        assert out["mismatched"] == [{"field": field, "yours": value,
                                      "in_force": E if field != "decision_revision" else 3}]

    def test_agreeing_identity_is_accepted(self):
        client, _m, _out = _call(decision_digest=E, decision_revision=3, calibration_id=E)
        assert client.post.await_count == 1

    def test_a_different_detector_under_the_same_algorithm(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger=probe_tools.__name__):
            client, _m, out = _call(_register(millm_probe={"id": MILLM_ID, "detector_digest": "a" * 64,
                                                           "identity_algorithm": ALGO}))
        assert out["refused"] == "detector_mismatch" and client.post.await_count == 0
        assert out["detector_digest"] == {"millm": "a" * 64, "mistudio": D}
        assert [c["tool"] for c in out["next"]] == ["export_probe_definition", "millm_import_probe"]
        assert any(r.getMessage().startswith(f"detector_mismatch probe={MILLM_ID}") for r in caplog.records)

    def test_another_algorithm_is_forwarded_unverified(self):
        client, _m, out = _call(_register(millm_probe={"id": MILLM_ID, "detector_digest": "a" * 64,
                                                       "identity_algorithm": "other/2"}))
        assert client.post.await_count == 1
        assert out["detector_unverified"] is True and "other/2" in out["detector_unverified_reason"]

    def test_a_matching_detector_is_verified(self):
        _client, _m, out = _call()
        assert "detector_unverified" not in out


def test_the_next_calls_this_tool_returns_are_live():
    from tests.support.next_calls import assert_next_calls_are_live

    for registered, over in (
        (_register(current={"decision": None, "detail": "x", "decision_digest": None,
                            "decision_revision": 1}), {}),
        (_register(), {"threshold": 1.0}),
        (_register(millm_probe={"id": MILLM_ID, "detector_digest": "a" * 64,
                                "identity_algorithm": ALGO}), {}),
    ):
        _client, _m, out = _call(registered, **over)
        assert_next_calls_are_live(out["next"])
