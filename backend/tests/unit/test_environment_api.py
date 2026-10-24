"""The runtime environment routes (Feature 036; ENH-003 §3.1, §4; FTASKS 2.2, 4.1–4.5).

Driven through the real ASGI app with the test database (``client`` fixture). Every assertion that a
write took effect reads ``runtime_settings.effective()`` — the accessor every reader calls — right after
the response, with no refresh of its own, so it fails when the route's refresh is removed (refresh
point 1).
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import select, text

from src.core import config, runtime_settings
from src.models.app_setting import AppSetting
from src.models.environment_setting import (
    EnvironmentOverride,
    EnvironmentSettingChange,
    HistoryIsAppendOnly,
)

API = "/api/v1/environment"


@pytest.fixture(autouse=True)
def _fresh_cache():
    runtime_settings.reset()
    yield
    runtime_settings.reset()


# ── the routes are served (wiring) ───────────────────────────────────────────────────────────


def test_the_routes_are_served():
    """From the live app's OpenAPI, never `app.routes` (memory: app-routes-is-not-a-route-list).
    Control C-ROUTER: delete `api_router.include_router(environment.router)` → red."""
    from src.main import app

    paths = app.openapi()["paths"]
    assert set(paths[API]) == {"get"}
    assert set(paths[API + "/{key}"]) == {"put", "delete"}
    assert set(paths[API + "/history"]) == {"get"}


# ── listing ──────────────────────────────────────────────────────────────────────────────────


async def test_the_listing_groups_every_shown_setting(client):
    body = (await client.get(API)).json()
    rows = body["settings"]
    runtime = {k for k, r in rows.items() if r["group"] == "runtime"}
    restart = {k for k, r in rows.items() if r["group"] == "restart"}
    environment = {k for k, r in rows.items() if r["group"] == "environment"}

    # 13 from 036; +7 from 040 S1 (the behaviour caps and catalogue page sizes, FR-7); +10 from 040 S2
    # (calibration floor, breakdown, grouping-column and report caps); +13 from 041 (the blend caps, folds,
    # page sizes, time limit and the score-batch cap).
    assert runtime == set(config.RUNTIME_KEYS) and len(runtime) == 43
    assert body["runtime_keys"] == list(config.RUNTIME_KEYS)
    assert restart == set(config.RESTART_KEYS) | {n for n, _, _ in config.EXTERNAL_RESTART_KEYS}
    assert environment == set(config.SHOWN_READ_ONLY)
    assert body["count"] == len(rows)
    # Hidden fields are not on the page at all.
    assert not (set(rows) & config.HIDDEN_FIELDS)

    row = rows["labeling_batch_size"]
    assert row["runtime_writable"] is True and row["source"] == "env"
    assert row["value"] == row["env_value"] == config.settings.labeling_batch_size
    assert row["effect"] == "next labeling job" and row["env_name"] == "LABELING_BATCH_SIZE"
    assert row["label"] and row["description"]

    mcp = rows["MCP_STEERING_MAX_CONCURRENT"]
    assert mcp["source"] == "external" and mcp["value"] is None
    assert mcp["restart"] == "mistudio-mcp pod" and mcp["restart_warning"] is None
    assert rows["CELERY_CONCURRENCY"]["restart_warning"] == config.BACKEND_RESTART_WARNING
    assert rows["bypass_settings_pin"]["env_name"] == "MISTUDIO_BYPASS_PIN"


async def test_no_secret_reaches_the_listing(client, monkeypatch):
    """A sentinel secret in the settings appears nowhere in the response (whole-body search).
    Control C-REDACT: serve `env[key]` for secret rows → red."""
    sentinel_hf = "hf_SENTINEL_e7c1d9a3b5"
    sentinel_oa = "sk-SENTINEL-4b2f8d6e0a"
    sentinel_db = "SENTINELPW91"
    monkeypatch.setattr(config.settings, "hf_token", sentinel_hf)
    monkeypatch.setattr(config.settings, "openai_api_key", sentinel_oa)
    monkeypatch.setattr(config.settings, "internal_api_url", f"http://svc:{sentinel_db}@backend:8000")

    response = await client.get(API)
    text_body = response.text
    for sentinel in (sentinel_hf, sentinel_oa, sentinel_db, config.settings.secret_key):
        assert sentinel not in text_body
    rows = response.json()["settings"]
    assert rows["hf_token"]["secret"] is True and rows["hf_token"]["is_set"] is True
    assert rows["hf_token"]["value"] is None and rows["hf_token"]["env_value"] is None
    assert rows["internal_api_url"]["value"] == "http://svc:<redacted>@backend:8000"


async def test_an_unset_secret_reads_not_set(client, monkeypatch):
    monkeypatch.setattr(config.settings, "openai_api_key", None)
    rows = (await client.get(API)).json()["settings"]
    assert rows["openai_api_key"]["is_set"] is False


# ── writes ───────────────────────────────────────────────────────────────────────────────────


async def test_a_write_reaches_the_accessor_before_the_response_returns(client, async_session):
    """Refresh point 1. Control C-RP1: remove the route's `refresh_async` call → red."""
    assert runtime_settings.effective().labeling_batch_size == config.settings.labeling_batch_size
    response = await client.put(f"{API}/labeling_batch_size", json={"value": 4, "reason": "probe OOM"})
    assert response.status_code == 200, response.text
    assert response.json() == {
        "key": "labeling_batch_size", "value": 4,
        "previous": config.settings.labeling_batch_size, "source": "override",
    }
    assert runtime_settings.effective().labeling_batch_size == 4

    stored = (await async_session.execute(select(EnvironmentOverride))).scalars().all()
    assert [(r.key, r.value, r.updated_by, r.reason) for r in stored] == [
        ("labeling_batch_size", 4, "operator", "probe OOM")
    ]
    listing = (await client.get(API)).json()["settings"]["labeling_batch_size"]
    assert listing["source"] == "override" and listing["value"] == 4
    assert listing["env_value"] == config.settings.labeling_batch_size
    assert listing["updated_by"] == "operator" and listing["last_change"]["action"] == "set"


async def test_reset_returns_the_environment_value(client):
    await client.put(f"{API}/log_level", json={"value": "ERROR"})
    assert runtime_settings.effective().log_level == "ERROR"
    response = await client.delete(f"{API}/log_level", params={"reason": "done"})
    assert response.status_code == 200, response.text
    assert response.json() == {
        "key": "log_level", "value": config.settings.log_level, "previous": "ERROR", "source": "env",
    }
    assert "log_level" not in runtime_settings.overrides()
    assert runtime_settings.effective().log_level == config.settings.log_level


async def test_reset_with_no_override_is_404(client):
    response = await client.delete(f"{API}/labeling_batch_size")
    assert response.status_code == 404
    assert response.json()["detail"].startswith("setting_not_overridden")


async def test_history_records_every_set_and_reset_newest_first(client):
    await client.put(f"{API}/labeling_batch_size", json={"value": 4})
    await client.put(f"{API}/labeling_batch_size", json={"value": 6, "reason": "second"})
    await client.delete(f"{API}/labeling_batch_size")
    changes = (await client.get(f"{API}/history")).json()["changes"]
    assert [(c["action"], c["previous_value"], c["previous_source"], c["new_value"], c["new_source"])
            for c in changes] == [
        ("reset", 6, "override", config.settings.labeling_batch_size, "env"),
        ("set", 4, "override", 6, "override"),
        ("set", config.settings.labeling_batch_size, "env", 4, "override"),
    ]
    assert changes[1]["reason"] == "second" and {c["actor"] for c in changes} == {"operator"}
    one = (await client.get(f"{API}/history", params={"key": "log_level"})).json()
    assert one == {"changes": [], "count": 0}
    assert (await client.get(f"{API}/history", params={"limit": 1})).json()["count"] == 1


async def test_history_is_append_only(client, async_session):
    await client.put(f"{API}/labeling_batch_size", json={"value": 4})
    row = (await async_session.execute(select(EnvironmentSettingChange))).scalars().one()
    row.reason = "rewritten"
    with pytest.raises(HistoryIsAppendOnly):
        await async_session.flush()
    await async_session.rollback()
    # The database refuses it too (the trigger the migration and create_all install).
    with pytest.raises(Exception, match="append-only"):
        await async_session.execute(text("DELETE FROM environment_setting_changes"))
    await async_session.rollback()


# ── refusals ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", sorted(config.NEVER_WRITABLE))
async def test_never_writable_keys_are_refused_by_name(client, key):
    """Moving any of these into RUNTIME_KEYS turns this red (spec §4 "Never-writable, by name")."""
    response = await client.put(f"{API}/{key}", json={"value": "x"})
    assert response.status_code == 422
    assert response.json()["detail"].startswith("setting_not_runtime")
    assert (await client.delete(f"{API}/{key}")).status_code == 422


def test_runtime_keys_and_never_writable_are_disjoint():
    assert not set(config.RUNTIME_KEYS) & config.NEVER_WRITABLE
    assert not set(config.RUNTIME_KEYS) & config.SECRET_FIELDS


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("labeling_batch_size", 33, "less than or equal to 32"),
        ("labeling_batch_size", 0, "greater than or equal to 1"),
        ("steering_timeout_seconds", 20, "greater than or equal to 30"),
        ("steering_hazard_prior_threshold", 1.5, "less than or equal to 1"),
        ("system_monitor_interval_seconds", 31, "less than or equal to 30"),
        ("log_level", "LOUD", "DEBUG"),
        ("activation_read_threads", 0, "greater than or equal to 1"),
        ("mcp_approval_ttl_hours", 0, "greater than or equal to 1"),
        ("ollama_url", "ftp://host", "http"),
        ("ollama_url", "http://169.254.169.254/", "metadata"),
        ("neuronpedia_local_url", "neuron.local", "http"),
        ("neuronpedia_local_admin_user_id", "  ", "empty"),
        ("steering_cluster_constants_json", "{not json", "not valid JSON"),
        ("steering_cluster_constants_json", json.dumps({"default": {"zz": 1}}), "unknown constant"),
        ("steering_cluster_constants_json", json.dumps({"default": {"a": "x"}}), "number"),
    ],
)
async def test_invalid_values_are_refused_with_the_fields_own_message(client, key, value, message):
    response = await client.put(f"{API}/{key}", json={"value": value})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail.startswith(f"invalid_value: {key}:") and message.lower() in detail.lower(), detail
    assert key not in runtime_settings.overrides()


async def test_valid_url_and_json_values_are_accepted(client):
    good = json.dumps({"default": {"a": 3.0}, "per_sae": {"sae_x": {"gamma": 0.5}}})
    assert (await client.put(f"{API}/steering_cluster_constants_json", json={"value": good})).status_code == 200
    assert (await client.put(f"{API}/neuronpedia_local_url", json={"value": None})).status_code == 200
    assert runtime_settings.effective().neuronpedia_local_url is None
    assert (await client.put(f"{API}/ollama_url", json={"value": "http://localhost:8001"})).status_code == 200
    assert runtime_settings.effective().ollama_url == "http://localhost:8001"


async def test_a_long_reason_is_refused(client):
    response = await client.put(f"{API}/log_level", json={"value": "INFO", "reason": "x" * 501})
    assert response.status_code == 422


# ── the generic key/value store refuses runtime keys (FTASKS 4.5) ───────────────────────────


async def test_the_generic_settings_store_refuses_runtime_keys(client, async_session):
    """D4: `ollama_url` moved out of `app_settings`. Control C-GENERIC: remove the refusal → red."""
    put = await client.put("/api/v1/settings", json={"key": "ollama_url", "value": "http://localhost:1"})
    assert put.status_code == 409 and "/api/v1/environment/ollama_url" in put.json()["detail"]
    bulk = await client.put("/api/v1/settings/bulk", json={"settings": [
        {"key": "labeling_default_batch_size", "value": "4"},
        {"key": "log_level", "value": "DEBUG"},
    ]})
    assert bulk.status_code == 409
    assert (await client.delete("/api/v1/settings/ollama_url")).status_code == 409
    rows = (await async_session.execute(select(AppSetting))).scalars().all()
    assert rows == []  # the bulk refusal wrote nothing, not even its valid first item


# ── a stored row that no longer validates is ignored and logged ─────────────────────────────


def test_a_stored_row_that_no_longer_validates_is_ignored(caplog):
    runtime_settings.apply({"labeling_batch_size": 999, "secret_key": "x" * 40, "log_level": "WARNING"})
    assert runtime_settings.overrides() == {"log_level": "WARNING"}
    assert "setting_override_invalid" in caplog.text and "setting_override_ignored" in caplog.text
