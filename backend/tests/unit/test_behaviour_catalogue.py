"""The behaviour catalogue end to end (Feature 037 FR-1 – FR-9, FR-13; ENH-005 D20–D25, D48).

Through the REAL routes and a REAL database wherever the claim is about wiring: that the routes are
served, that adopting a key writes nothing to Part A's rows, that the database itself refuses a second
primary and an edited version, and that Part A's probe-behaviour PATCH writes the catalogue history.

Fixtures differ in EXACTLY the field under test (global CLAUDE.md): probes X and Y differ only in their
``behaviour``; two views differ only in their confirmation.

Negative controls (each broke the named line, ran this file, saw red, restored from saved bytes with a
sha256 check) are recorded in 0xcc/reviews/037_038_implementation_controls_2026-10-21.md.
"""
from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

API = "/api/v1/probe-monitors"
AGENT = {"X-MiStudio-Agent": "mcp"}

PII = {
    "key": "pii",
    "name": "Personal data present",
    "description": "Text that identifies a person",
    "positive_meaning": "The text contains at least one identifying personal data value.",
    "negative_meaning": "No identifying personal data value.",
    "target_fpr": 0.01,
}


@pytest.fixture
def sync_db(async_engine):
    url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", "")
    )
    engine = create_engine(url)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def _base(db):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/nonexistent", architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="corpus", source="local", status=DatasetStatus.READY,
                      raw_path="/nonexistent", extra_metadata={})
    db.add_all([model, dataset])
    db.flush()
    return model, dataset


def _view(db, dataset, *, behaviour=None, confirmed=None, role="train", name="train"):
    from src.models.probe_monitor import ProbeMonitorDataset

    view = ProbeMonitorDataset(
        name=name, dataset_id=dataset.id, input_column="i", label_column="l", label_mapping={},
        role=role, counts={}, behaviour=behaviour,
        positive_labels_confirmed=(
            {"key": confirmed, "version": None, "by": "operator", "at": "2026-10-21T00:00:00+00:00",
             "origin": "operator"} if confirmed else None
        ),
    )
    db.add(view)
    db.flush()
    return view


def _probe(db, *, behaviour, confirmed=None, run_behaviour="same", rung=0):
    """A completed run on its own training view, with one probe. `confirmed` = the view's D48 key."""
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorRun

    model, dataset = _base(db)
    view = _view(db, dataset, confirmed=confirmed)
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[], config={},
                          status="completed", environment={},
                          behaviour=behaviour if run_behaviour == "same" else run_behaviour)
    db.add(run)
    db.flush()
    probe = ProbeMonitor(run_id=run.id, layer=1, rule="mean", rule_params={}, variant="dense",
                         val_metrics={}, behaviour=behaviour, rung=rung, rung_reasons=[])
    db.add(probe)
    db.commit()
    return probe


def _prompt(db, *, behaviour, name=None):
    from src.models.probe_judge_prompt import ProbeJudgePrompt
    from src.services import judge_prompt_library as library
    from src.services.judge_prompt_forms import content_hash

    content = dict(system_prompt="s answer", user_template="{text}",
                   answer_form=library.normalise_answer_form({"kind": "binary"}),
                   max_interaction_chars=8000)
    name = name or f"p-{uuid.uuid4().hex[:8]}"
    row = ProbeJudgePrompt(name=name, version=1, title="t", behaviour=behaviour,
                           content_hash=content_hash(name=name, version=1, **content),
                           source="operator", created_by="operator", **content)
    db.add(row)
    db.commit()
    return row


async def _create(client, **changes):
    response = await client.post(f"{API}/behaviours", json={**PII, **changes})
    assert response.status_code == 201, response.text
    return response.json()


# ── FR-1: the record and its key ────────────────────────────────────────────────────────────


def test_slug_rule_is_the_librarys():
    """Five copies of one rule: the library's (the authority), Part A's schema literal, the catalogue
    model's Postgres regex, the catalogue request schema's, and the migration's frozen copy. Part A's
    `BEHAVIOUR_PATTERN` was a hand copy no test pinned (pre-existing gap, FTDD §6)."""
    import importlib.util
    import pathlib

    from src.models.probe_behaviour import SLUG_REGEX
    from src.schemas.behaviour import KEY_PATTERN
    from src.schemas.probe_monitor import BEHAVIOUR_PATTERN
    from src.services.judge_prompt_forms import SLUG_PATTERN

    path = (pathlib.Path(__file__).resolve().parents[2] / "alembic" / "versions"
            / "e7c1a4b9d2f6_behaviour_catalogue.py")
    spec = importlib.util.spec_from_file_location("rev_e7c1", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert SLUG_PATTERN == BEHAVIOUR_PATTERN == SLUG_REGEX == KEY_PATTERN == migration.SLUG_REGEX


@pytest.mark.asyncio
async def test_a_key_is_catalogued_once_and_must_be_a_slug(client):
    made = await _create(client)
    assert made["key"] == "pii" and made["current_version"] == 1
    assert made["definition"]["positive_meaning"] == PII["positive_meaning"]
    again = await client.post(f"{API}/behaviours", json=PII)
    assert again.status_code == 409 and again.json()["detail"]["code"] == "behaviour_exists"
    for bad in ("Bad Key", "a", "-pii", "pii-", "p_i_i"):
        refused = await client.post(f"{API}/behaviours", json={**PII, "key": bad})
        assert refused.status_code == 422, (bad, refused.text)
        assert refused.json()["detail"]["code"] == "invalid_key"
    too_long = await client.post(f"{API}/behaviours", json={**PII, "key": "x" * 65})
    assert too_long.status_code == 422


def test_the_database_refuses_a_key_that_is_not_a_slug(sync_db):
    with pytest.raises(sa.exc.IntegrityError):
        sync_db.execute(sa.text(
            "INSERT INTO probe_behaviours (key, current_version, created_by, created_at, updated_at) "
            "VALUES ('Bad Key', 1, 'operator', now(), now())"
        ))
    sync_db.rollback()


@pytest.mark.asyncio
async def test_who_asked_is_read_from_the_agent_header(client):
    response = await client.post(f"{API}/behaviours", json=PII, headers=AGENT)
    assert response.status_code == 201
    assert response.json()["created_by"] == "agent:mcp"
    history = (await client.get(f"{API}/behaviours/pii/history")).json()["entries"]
    assert [(e["field"], e["actor"]) for e in history] == [("created", "agent:mcp")]


# ── FR-2: versions are append-only ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_versions_are_append_only_service(client, async_session):
    await _create(client)
    edited = await client.post(
        f"{API}/behaviours/pii/versions",
        json={**{k: v for k, v in PII.items() if k != "key"}, "positive_meaning": "Identifying data."},
    )
    assert edited.status_code == 201, edited.text
    assert edited.json()["current_version"] == 2
    versions = (await client.get(f"{API}/behaviours/pii/versions")).json()["versions"]
    assert [(v["version"], v["positive_meaning"]) for v in versions] == [
        (2, "Identifying data."), (1, PII["positive_meaning"]),
    ]
    same = await client.post(f"{API}/behaviours/pii/versions",
                             json={**{k: v for k, v in PII.items() if k != "key"}, "positive_meaning": "Identifying data."})
    assert same.status_code == 422 and same.json()["detail"]["code"] == "no_change"

    from src.services import behaviour_catalogue as catalogue

    with pytest.raises(catalogue.BehaviourRefused) as refused:
        await catalogue.edit_version(async_session, "pii", 1, {"name": "x"}, "operator")
    assert refused.value.status == 409 and refused.value.code == "version_immutable"


@pytest.mark.asyncio
async def test_versions_are_append_only_db(client, sync_db):
    await _create(client)
    with pytest.raises(sa.exc.DBAPIError, match="append-only"):
        sync_db.execute(sa.text("UPDATE probe_behaviour_versions SET positive_meaning = 'x' WHERE key = 'pii'"))
    sync_db.rollback()
    with pytest.raises(sa.exc.DBAPIError, match="append-only"):
        sync_db.execute(sa.text("DELETE FROM probe_behaviour_versions WHERE key = 'pii'"))
    sync_db.rollback()
    with pytest.raises(sa.exc.DBAPIError, match="append-only"):
        sync_db.execute(sa.text("UPDATE probe_behaviour_history SET actor = 'x' WHERE key = 'pii'"))
    sync_db.rollback()
    with pytest.raises(sa.exc.DBAPIError, match="append-only"):
        sync_db.execute(sa.text("DELETE FROM probe_behaviour_history WHERE key = 'pii'"))
    sync_db.rollback()


# ── FR-3, FR-4: adoption by join; uncatalogued keys ────────────────────────────────────────


def _snapshot(db, table, ids):
    rows = db.execute(sa.text(f"SELECT * FROM {table} WHERE id = ANY(:ids) ORDER BY id"), {"ids": list(ids)})
    return [dict(row._mapping) for row in rows]


@pytest.mark.asyncio
async def test_adoption_writes_nothing(client, sync_db):
    """US-2: Part A records already carrying `pii` appear under it, and not one of their rows changes."""
    probe = _probe(sync_db, behaviour="pii")
    prompt = _prompt(sync_db, behaviour="pii")
    before = {
        "probe_monitors": _snapshot(sync_db, "probe_monitors", [probe.id]),
        "probe_monitor_runs": _snapshot(sync_db, "probe_monitor_runs", [probe.run_id]),
        "probe_judge_prompts": _snapshot(sync_db, "probe_judge_prompts", [prompt.id]),
    }
    made = await _create(client)
    assert [p["id"] for p in made["adopted"]["probes"]] == [probe.id]
    assert [r["id"] for r in made["adopted"]["runs"]] == [probe.run_id]
    assert [p["id"] for p in made["adopted"]["prompts"]] == [prompt.id]
    sync_db.expire_all()
    after = {table: _snapshot(sync_db, table, [r["id"] for r in rows]) for table, rows in before.items()}
    assert after == before, "creating a catalogue row rewrote an adopted Part A row"


@pytest.mark.asyncio
async def test_uncatalogued_keys_counts_each_kind(client, sync_db):
    _prompt(sync_db, behaviour="aa")
    _probe(sync_db, behaviour="bb", run_behaviour=None)
    from src.models.probe_monitor import ProbeMonitorRun

    model, dataset = _base(sync_db)
    view = _view(sync_db, dataset, behaviour="cc")
    sync_db.add(ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[],
                                config={}, status="completed", environment={}, behaviour="dd"))
    sync_db.commit()
    await _create(client, key="dd")
    listed = (await client.get(f"{API}/behaviours")).json()
    by_key = {row["key"]: row for row in listed["uncatalogued"]}
    assert by_key["aa"] == {"key": "aa", "catalogued": False, "prompts": 1, "runs": 0, "probes": 0, "views": 0}
    assert by_key["bb"] == {"key": "bb", "catalogued": False, "prompts": 0, "runs": 0, "probes": 1, "views": 0}
    assert by_key["cc"] == {"key": "cc", "catalogued": False, "prompts": 0, "runs": 0, "probes": 0, "views": 1}
    assert "dd" not in by_key, "a catalogued key is listed as uncatalogued"
    assert [row["key"] for row in listed["behaviours"]] == ["dd"]
    assert listed["behaviours"][0]["counts"] == {"prompts": 0, "runs": 1, "probes": 0, "views": 0}


# ── FR-5: archive, delete ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_used_behaviour_cannot_be_deleted_and_an_unused_one_can(client, sync_db):
    _prompt(sync_db, behaviour="pii")
    await _create(client)
    used = await client.delete(f"{API}/behaviours/pii")
    assert used.status_code == 409 and used.json()["detail"]["details"]["in_use"] == {"prompts": 1}
    await _create(client, key="unused-one")
    assert (await client.delete(f"{API}/behaviours/unused-one")).status_code == 204
    assert (await client.get(f"{API}/behaviours/unused-one")).status_code == 404
    history = (await client.get(f"{API}/behaviours/unused-one/history")).json()["entries"]
    assert [e["field"] for e in history] == ["deleted", "created"], "history must survive a delete"


@pytest.mark.asyncio
async def test_archive_hides_and_unarchive_restores(client):
    await _create(client)
    archived = (await client.post(f"{API}/behaviours/pii/archive", json={"reason": "retired"})).json()
    assert archived["archived"] is True
    assert (await client.get(f"{API}/behaviours")).json()["behaviours"] == []
    everything = (await client.get(f"{API}/behaviours", params={"include_archived": True})).json()
    assert [b["key"] for b in everything["behaviours"]] == ["pii"]
    assert (await client.post(f"{API}/behaviours/pii/unarchive")).json()["archived"] is False


# ── FR-6, FR-9: roles ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_primary_needs_matching_key(client, sync_db):
    """Probes X and Y differ ONLY in their behaviour key."""
    x = _probe(sync_db, behaviour="pii", confirmed="pii")
    y = _probe(sync_db, behaviour="other", confirmed="pii")
    await _create(client)
    accepted = await client.put(f"{API}/behaviours/pii/probes/{x.id}/role", json={"role": "primary"})
    assert accepted.status_code == 200, accepted.text
    refused = await client.put(f"{API}/behaviours/pii/probes/{y.id}/role", json={"role": "candidate"})
    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert detail["code"] == "key_mismatch"
    assert "'other'" in detail["message"] and "'pii'" in detail["message"]


@pytest.mark.asyncio
async def test_primary_needs_a_confirmed_training_view(client, sync_db):
    """Two probes that differ only in whether their training view was confirmed as `pii` (D48)."""
    confirmed = _probe(sync_db, behaviour="pii", confirmed="pii")
    unconfirmed = _probe(sync_db, behaviour="pii", confirmed=None)
    wrong = _probe(sync_db, behaviour="pii", confirmed="high-stakes")
    await _create(client)
    for probe in (unconfirmed, wrong):
        refused = await client.put(f"{API}/behaviours/pii/probes/{probe.id}/role", json={"role": "primary"})
        assert refused.status_code == 422 and refused.json()["detail"]["code"] == "training_view_unconfirmed"
    ok = await client.put(f"{API}/behaviours/pii/probes/{confirmed.id}/role", json={"role": "primary"})
    assert ok.status_code == 200


@pytest.mark.asyncio
async def test_an_archived_behaviour_freezes_roles(client, sync_db):
    probe = _probe(sync_db, behaviour="pii", confirmed="pii")
    await _create(client)
    await client.post(f"{API}/behaviours/pii/archive")
    refused = await client.put(f"{API}/behaviours/pii/probes/{probe.id}/role", json={"role": "primary"})
    assert refused.status_code == 409 and refused.json()["detail"]["code"] == "behaviour_archived"


@pytest.mark.asyncio
async def test_one_primary_per_behaviour_service(client, sync_db):
    first = _probe(sync_db, behaviour="pii", confirmed="pii")
    second = _probe(sync_db, behaviour="pii", confirmed="pii")
    await _create(client)
    assert (await client.put(f"{API}/behaviours/pii/probes/{first.id}/role", json={"role": "primary"})).status_code == 200
    refused = await client.put(f"{API}/behaviours/pii/probes/{second.id}/role", json={"role": "primary"})
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "primary_exists"
    assert first.id in refused.json()["detail"]["message"]
    roles = {p["id"]: p["role"] for p in (await client.get(f"{API}/behaviours/pii")).json()["adopted"]["probes"]}
    assert roles == {first.id: "primary", second.id: "candidate"}


@pytest.mark.asyncio
async def test_one_primary_per_behaviour_db(client, sync_db):
    first = _probe(sync_db, behaviour="pii")
    second = _probe(sync_db, behaviour="pii")
    await _create(client)
    insert = sa.text(
        "INSERT INTO probe_behaviour_probe_roles (probe_id, key, role, updated_by, updated_at) "
        "VALUES (:p, 'pii', 'primary', 'operator', now())"
    )
    sync_db.execute(insert, {"p": first.id})
    sync_db.commit()
    with pytest.raises(sa.exc.IntegrityError):
        sync_db.execute(insert, {"p": second.id})
    sync_db.rollback()


@pytest.mark.asyncio
async def test_a_role_change_writes_one_history_row(client, sync_db):
    probe = _probe(sync_db, behaviour="pii", confirmed="pii")
    await _create(client)
    await client.put(f"{API}/behaviours/pii/probes/{probe.id}/role",
                     json={"role": "retired", "reason": "superseded"}, headers=AGENT)
    entries = (await client.get(f"{API}/behaviours/pii/history")).json()["entries"]
    roles = [e for e in entries if e["field"] == "role"]
    assert len(roles) == 1
    assert (roles[0]["subject_id"], roles[0]["old_value"], roles[0]["new_value"], roles[0]["actor"],
            roles[0]["reason"]) == (probe.id, "candidate", "retired", "agent:mcp", "superseded")


# ── FR-8: Part A's PATCH is recorded ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_behaviour_change_is_recorded(client, sync_db):
    """Part A's `PATCH /probes/{id}/behaviour` moves a probe from `pii` to `high-stakes`: one history row
    on each key, and the role it held under `pii` is dropped (a role belongs to one behaviour, D24)."""
    probe = _probe(sync_db, behaviour="pii", confirmed="pii")
    await _create(client)
    await _create(client, key="high-stakes", name="High stakes")
    await client.put(f"{API}/behaviours/pii/probes/{probe.id}/role", json={"role": "primary"})
    moved = await client.patch(f"{API}/probes/{probe.id}/behaviour",
                               json={"behaviour": "high-stakes"}, headers=AGENT)
    assert moved.status_code == 200, moved.text
    old = [e for e in (await client.get(f"{API}/behaviours/pii/history")).json()["entries"] if e["field"] == "behaviour"]
    new = [e for e in (await client.get(f"{API}/behaviours/high-stakes/history")).json()["entries"] if e["field"] == "behaviour"]
    assert len(old) == 1 and len(new) == 1
    assert old[0]["old_value"] == {"behaviour": "pii", "role": "primary"}
    assert new[0]["new_value"] == {"behaviour": "high-stakes"}
    assert old[0]["actor"] == new[0]["actor"] == "agent:mcp"
    detail = (await client.get(f"{API}/behaviours/pii")).json()
    assert detail["adopted"]["probes"] == []


# ── FR-7, D48: data links and confirmation ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_views_are_listed_by_key_or_link(client, sync_db):
    _, dataset = _base(sync_db)
    keyed = _view(sync_db, dataset, behaviour="pii", name="keyed")
    linked = _view(sync_db, dataset, behaviour=None, name="linked", role="eval")
    _view(sync_db, dataset, behaviour=None, name="unrelated")
    sync_db.commit()
    await _create(client)
    response = await client.post(f"{API}/behaviours/pii/views/{linked.id}", json={"use": "calibration"})
    assert response.status_code == 200, response.text
    views = {v["name"]: v for v in response.json()["adopted"]["views"]}
    assert set(views) == {"keyed", "linked"}
    assert (views["keyed"]["linked_by"], views["keyed"]["use"]) == ("key", "train")
    assert (views["linked"]["linked_by"], views["linked"]["use"]) == ("link", "calibration")
    unlinked = await client.delete(f"{API}/behaviours/pii/views/{linked.id}")
    assert {v["name"] for v in unlinked.json()["adopted"]["views"]} == {"keyed"}
    assert keyed.id


@pytest.mark.asyncio
async def test_confirming_positive_labels_is_the_operators(client, sync_db):
    _, dataset = _base(sync_db)
    view = _view(sync_db, dataset, behaviour="pii")
    sync_db.commit()
    refused = await client.post(f"{API}/datasets/{view.id}/confirm-positive", json={"key": "pii"}, headers=AGENT)
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "confirm_operator_only"
    done = await client.post(f"{API}/datasets/{view.id}/confirm-positive", json={"key": "pii"})
    assert done.status_code == 200, done.text
    record = done.json()["positive_labels_confirmed"]
    assert (record["key"], record["by"], record["origin"]) == ("pii", "operator", "operator")


@pytest.mark.asyncio
async def test_a_calibration_view_has_nothing_to_confirm(client, sync_db):
    _, dataset = _base(sync_db)
    view = _view(sync_db, dataset, role="calibration")
    sync_db.commit()
    refused = await client.post(f"{API}/datasets/{view.id}/confirm-positive", json={"key": "pii"})
    assert refused.status_code == 422 and refused.json()["detail"]["code"] == "calibration_view"


# ── FR-13: the routes are served ───────────────────────────────────────────────────────────


def test_behaviour_routes_in_openapi():
    """Read from the LIVE app's OpenAPI, never `app.routes` (memory: app-routes-is-not-a-route-list)."""
    from src.main import app

    paths = app.openapi()["paths"]
    expected = {
        "/api/v1/probe-monitors/behaviours": {"get", "post"},
        "/api/v1/probe-monitors/behaviours/{key}": {"get", "delete"},
        "/api/v1/probe-monitors/behaviours/{key}/versions": {"get", "post"},
        "/api/v1/probe-monitors/behaviours/{key}/probes/{probe_id}/role": {"put"},
        "/api/v1/probe-monitors/behaviours/{key}/views/{dataset_id}": {"post", "delete"},
        "/api/v1/probe-monitors/behaviours/{key}/archive": {"post"},
        "/api/v1/probe-monitors/behaviours/{key}/unarchive": {"post"},
        "/api/v1/probe-monitors/behaviours/{key}/history": {"get"},
        "/api/v1/probe-monitors/behaviours/{key}/checklist": {"get"},
        "/api/v1/probe-monitors/datasets/{dataset_id}/confirm-positive": {"post"},
    }
    missing = {
        path: sorted(methods - set(paths.get(path, {})))
        for path, methods in expected.items()
        if methods - set(paths.get(path, {}))
    }
    assert not missing, f"behaviour routes not served: {missing}"


@pytest.mark.asyncio
async def test_catalogue_probe_rows_carry_the_identity(client, sync_db):
    """Feature 039 (FR-16, FTASKS 5.8): each adopted probe row carries the six identity fields from
    the one read — here a definition invalidated by a re-cut, so `identity_current` is false."""
    probe = _probe(sync_db, behaviour="pii")
    probe.decision_digest = "f" * 64
    probe.definition_build = {"invalidated": {"cause": "decision", "reason": "threshold recalibrated",
                                              "was_identity": {"detector_digest": "d" * 64,
                                                               "decision_digest": "e" * 64,
                                                               "decision_revision": 2,
                                                               "identity_source": "built"}}}
    sync_db.commit()
    made = await _create(client)
    row = made["adopted"]["probes"][0]
    assert {key: row[key] for key in (
        "detector_digest", "decision_digest", "decision_revision", "identity_current",
        "identity_stale_reason", "identity_source")} == {
        "detector_digest": "d" * 64, "decision_digest": "e" * 64, "decision_revision": 2,
        "identity_current": False, "identity_stale_reason": "decision_recalibrated",
        "identity_source": "built"}
