"""Archive, unarchive and purge of probe dataset VIEWS (operator decision 2026-10-21).

The pattern is miForge's and miDataworks' (shipped 2026-10-20), rebuilt here: ARCHIVE hides and is
reversible; PURGE is permanent, ARCHIVED views only, refused with EVERY blocker named, writes an
append-only tombstone, and is the operator's alone (no MCP tool, and every purge route refuses the
MCP server's header).

Each blocker has its own test, through the real route over the real database, asserting the code
AND that nothing was deleted and no tombstone was written. Wiring tests assert the live registry
and the payload, never that a module imports.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select, text

pytestmark = pytest.mark.asyncio

API = "/api/v1/probe-monitors"
AGENT = {"X-MiStudio-Agent": "mcp"}


# ── fixtures ──────────────────────────────────────────────────────────────────


async def _world(session):
    """A model, one downloaded dataset, and views of each role over it."""
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/x", architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="humor-download", source="local",
                      status=DatasetStatus.READY, raw_path="/x", extra_metadata={})
    session.add_all([model, dataset])
    await session.flush()

    def view(name, role):
        return ProbeMonitorDataset(name=name, dataset_id=dataset.id, input_column="i",
                                   label_column="l", label_mapping={"1": "positive"}, role=role,
                                   counts={"positive": 30, "negative": 30})

    views = {
        "train": view("train view", "train"),
        "other_train": view("other train view", "train"),
        "eval": view("eval view", "eval"),
        "calib": view("calibration view", "calibration"),
        "spare": view("spare view", "eval"),
    }
    session.add_all(views.values())
    await session.commit()
    return model, dataset, {k: v.id for k, v in views.items()}


async def _archive(client, view_id, **kw):
    r = await client.post(f"{API}/datasets/{view_id}/archive", **kw)
    assert r.status_code == 200, r.text
    return r.json()


async def _count(session, model):
    session.expire_all()
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def _exists(session, view_id):
    from src.models.probe_monitor import ProbeMonitorDataset

    session.expire_all()
    return (
        await session.execute(select(ProbeMonitorDataset.id).where(ProbeMonitorDataset.id == view_id))
    ).scalar_one_or_none() is not None


async def _run(session, model, **kw):
    from src.models.probe_monitor import ProbeMonitorRun

    run = ProbeMonitorRun(model_id=model.id, config={}, environment={},
                          status=kw.pop("status", "completed"), **kw)
    session.add(run)
    await session.commit()
    return run.id


async def _refused(client, session, view_id):
    """Purge `view_id`, require 409 purge_blocked, and that NOTHING changed. Returns blocker codes."""
    from src.models.probe_monitor import ProbeMonitorDatasetPurge

    before = await _count(session, ProbeMonitorDatasetPurge)
    r = await client.post(f"{API}/datasets/purge", json={"ids": [view_id]})
    assert r.status_code == 200, r.text
    (result,) = r.json()["results"]
    assert result["outcome"] == "blocked", result
    assert result["code"] == "purge_blocked", result
    assert await _exists(session, view_id), "a refused purge deleted the view"
    assert await _count(session, ProbeMonitorDatasetPurge) == before, "a refusal wrote a tombstone"
    return [b["code"] for b in result["blockers"]], result


# ── the registry is the authority for what must be checked ────────────────────


class TestEveryReferenceIsChecked:
    def test_the_checked_foreign_keys_ARE_the_registry_s(self):
        """Derived from the ORM, not a hand-kept list: a new FK into probe_monitor_datasets turns
        this red until `blockers()` is taught to check it."""
        from src.services import probe_dataset_lifecycle as lc

        assert lc.registry_foreign_keys() == lc.CHECKED_FOREIGN_KEYS

    async def test_an_unknown_reference_fails_closed(self, client, async_session, monkeypatch):
        from src.services import probe_dataset_lifecycle as lc

        _, _, views = await _world(async_session)
        await _archive(client, views["spare"])
        monkeypatch.setattr(
            lc, "registry_foreign_keys",
            lambda: lc.CHECKED_FOREIGN_KEYS | {("new_table", "view_id")},
        )
        codes, result = await _refused(client, async_session, views["spare"])
        assert codes == ["unchecked_reference"]
        assert result["blockers"][0]["table"] == "new_table"

    def test_the_mcp_client_and_the_backend_spell_the_header_the_same(self):
        from src.mcp_server import client as mcp_client
        from src.services import probe_dataset_lifecycle as lc

        assert mcp_client.AGENT_HEADER == lc.AGENT_HEADER

    def test_the_mcp_client_SENDS_the_header_on_every_request(self):
        """The guard on every purge route is only as good as this: an MCP client without the
        header would be read as the operator."""
        from src.mcp_server.client import AGENT_HEADER, AGENT_NAME, MiStudioClient

        client = MiStudioClient("http://backend")
        assert client._http.headers.get(AGENT_HEADER) == AGENT_NAME


# ── archive ───────────────────────────────────────────────────────────────────


class TestArchive:
    async def test_the_list_hides_archived_views_unless_asked(self, client, async_session):
        _, _, views = await _world(async_session)
        await _archive(client, views["spare"])
        listed = {v["id"] for v in (await client.get(f"{API}/datasets")).json()}
        assert views["spare"] not in listed and views["eval"] in listed
        everything = (await client.get(f"{API}/datasets", params={"include_archived": True})).json()
        spare = next(v for v in everything if v["id"] == views["spare"])
        assert spare["archived_at"] and spare["archived_by"] == "operator"
        # Read by id still works: an archived view is hidden, not gone.
        assert (await client.get(f"{API}/datasets/{views['spare']}")).status_code == 200

    async def test_who_and_why_are_recorded_and_unarchive_clears_them(self, client, async_session):
        _, _, views = await _world(async_session)
        body = await _archive(client, views["spare"], json={"reason": "superseded by 20.4"},
                              headers=AGENT)
        assert body["archived_by"] == "agent:mcp"
        assert body["archive_reason"] == "superseded by 20.4"
        again = await _archive(client, views["spare"], json={"reason": "second"})
        assert again["archived_by"] == "agent:mcp", "a re-archive must not rewrite the record"
        assert again["archived_at"] == body["archived_at"]
        r = await client.post(f"{API}/datasets/{views['spare']}/unarchive")
        assert r.status_code == 200
        assert (r.json()["archived_at"], r.json()["archived_by"], r.json()["archive_reason"]) == (None, None, None)
        listed = {v["id"] for v in (await client.get(f"{API}/datasets")).json()}
        assert views["spare"] in listed

    async def test_archive_without_a_body_is_the_operator(self, client, async_session):
        _, _, views = await _world(async_session)
        body = await _archive(client, views["spare"])
        assert (body["archived_by"], body["archive_reason"]) == ("operator", None)

    async def test_an_unknown_view_is_404(self, client, async_session):
        await _world(async_session)
        assert (await client.post(f"{API}/datasets/pmd_nope/archive")).status_code == 404
        assert (await client.post(f"{API}/datasets/pmd_nope/unarchive")).status_code == 404

    async def test_a_malformed_agent_header_is_refused_not_read_as_the_operator(
        self, client, async_session
    ):
        _, _, views = await _world(async_session)
        r = await client.post(f"{API}/datasets/{views['spare']}/archive",
                              headers={"X-MiStudio-Agent": "Robert'); DROP"})
        assert r.status_code == 400
        assert not (await client.get(f"{API}/datasets/{views['spare']}")).json()["archived_at"]


class TestArchivedViewsAreRefusedAsNewInputs:
    """A new run, evaluation or judge baseline naming an archived view is 409 — and nothing is
    queued. Without this, new work could make an archived view unpurgeable behind the operator."""

    async def test_a_training_run(self, client, async_session, monkeypatch):
        from src.api.v1.endpoints import probe_monitors as ep

        model, _, views = await _world(async_session)
        await _archive(client, views["calib"])
        queued = []
        monkeypatch.setattr(ep, "gpu_delay", lambda *a, **k: queued.append(a) or (lambda **kw: None))
        r = await client.post(f"{API}/runs", json={
            "model_id": model.id, "train_dataset_id": views["train"],
            "eval_dataset_ids": [views["eval"]], "calibration_dataset_id": views["calib"],
        })
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail["code"] == "probe_dataset_archived"
        assert detail["archived"] == [views["calib"]]
        assert "calibration view" in detail["message"]
        assert queued == []

    async def test_an_evaluation(self, client, async_session, monkeypatch):
        from src.api.v1.endpoints import probe_monitors as ep
        from src.models.probe_monitor import ProbeMonitor

        model, _, views = await _world(async_session)
        run_id = await _run(async_session, model, train_dataset_id=views["train"], eval_dataset_ids=[])
        probe = ProbeMonitor(run_id=run_id, layer=1, rule="mean", rule_params={}, val_metrics={})
        async_session.add(probe)
        await async_session.commit()
        await _archive(client, views["spare"])

        async def ok(*_a, **_k):
            return None

        monkeypatch.setattr(ep, "refuse_precision_mismatch", ok)
        monkeypatch.setattr(ep, "refuse_render_mismatch", ok)
        queued = []
        monkeypatch.setattr(ep, "gpu_delay", lambda *a, **k: queued.append(a) or (lambda **kw: None))
        r = await client.post(f"{API}/probes/{probe.id}/evaluate", json=[views["eval"], views["spare"]])
        assert r.status_code == 409, r.text
        assert r.json()["detail"]["archived"] == [views["spare"]]
        assert queued == []

    async def test_a_judge_baseline(self, client, async_session, monkeypatch):
        from src.utils import url_validation
        from src.workers import probe_monitor_tasks

        _, _, views = await _world(async_session)
        await _archive(client, views["spare"])
        monkeypatch.setattr(url_validation, "validate_llm_endpoint_url", lambda u: u)
        queued = []
        monkeypatch.setattr(probe_monitor_tasks.run_probe_monitor_judge, "delay",
                            lambda **kw: queued.append(kw))
        r = await client.post(f"{API}/judge-runs", json={
            "endpoint": "http://judge/v1", "model": "j", "dataset_ids": [views["spare"]],
        })
        assert r.status_code == 409, r.text
        assert r.json()["detail"]["archived"] == [views["spare"]]
        assert queued == []


# ── purge: the refusals ───────────────────────────────────────────────────────


class TestPurgeRefusals:
    async def test_a_view_that_is_not_archived(self, client, async_session):
        from src.models.probe_monitor import ProbeMonitorDatasetPurge

        _, _, views = await _world(async_session)
        r = await client.post(f"{API}/datasets/purge", json={"ids": [views["spare"]]})
        (result,) = r.json()["results"]
        assert (result["outcome"], result["code"]) == ("blocked", "not_archived")
        assert "Archive it first" in result["reason"]
        assert await _exists(async_session, views["spare"])
        assert await _count(async_session, ProbeMonitorDatasetPurge) == 0
        r = await client.delete(f"{API}/datasets/{views['spare']}")
        assert r.status_code == 409 and r.json()["detail"]["code"] == "not_archived"

    async def test_a_run_trained_on_it(self, client, async_session):
        model, _, views = await _world(async_session)
        run_id = await _run(async_session, model, train_dataset_id=views["train"], eval_dataset_ids=[])
        await _archive(client, views["train"])
        codes, result = await _refused(client, async_session, views["train"])
        assert codes == ["run_uses_it"]
        assert result["blockers"][0]["run_id"] == run_id
        assert result["blockers"][0]["roles"] == ["train"]

    async def test_a_run_IN_PROGRESS(self, client, async_session):
        model, _, views = await _world(async_session)
        await _run(async_session, model, train_dataset_id=views["train"], eval_dataset_ids=[],
                   status="running")
        await _archive(client, views["train"])
        codes, result = await _refused(client, async_session, views["train"])
        assert codes == ["run_in_progress"]
        assert "running" in result["blockers"][0]["message"]

    async def test_a_run_that_EVALUATES_on_it_through_the_json_list(self, client, async_session):
        """No foreign key here — `eval_dataset_ids` is JSON — so the CASCADE would not even
        notice; the run would be left naming a view that no longer exists."""
        model, _, views = await _world(async_session)
        await _run(async_session, model, train_dataset_id=views["train"],
                   eval_dataset_ids=[views["spare"], views["eval"]])
        await _archive(client, views["eval"])
        codes, result = await _refused(client, async_session, views["eval"])
        assert codes == ["run_uses_it"]
        assert result["blockers"][0]["roles"] == ["eval"]

    async def test_a_run_CALIBRATED_on_it(self, client, async_session):
        model, _, views = await _world(async_session)
        await _run(async_session, model, train_dataset_id=views["train"], eval_dataset_ids=[],
                   calibration_dataset_id=views["calib"])
        await _archive(client, views["calib"])
        codes, result = await _refused(client, async_session, views["calib"])
        assert codes == ["run_uses_it"]
        assert result["blockers"][0]["roles"] == ["calibration"]

    async def test_a_probe_recalibrated_on_it_outside_its_run(self, client, async_session):
        from src.models.probe_monitor import ProbeMonitor

        model, _, views = await _world(async_session)
        run_id = await _run(async_session, model, train_dataset_id=views["train"], eval_dataset_ids=[])
        probe = ProbeMonitor(run_id=run_id, layer=1, rule="mean", rule_params={}, val_metrics={},
                             calibration_dataset_id=views["calib"])
        async_session.add(probe)
        await async_session.commit()
        probe_id = probe.id
        await _archive(client, views["calib"])
        codes, result = await _refused(client, async_session, views["calib"])
        assert codes == ["probe_calibrated_on_it"]
        assert result["blockers"][0]["probe_id"] == probe_id

    async def test_an_evaluation_on_it_outside_its_run(self, client, async_session):
        from src.models.probe_monitor import ProbeMonitor, ProbeMonitorEvaluation

        model, _, views = await _world(async_session)
        run_id = await _run(async_session, model, train_dataset_id=views["train"], eval_dataset_ids=[])
        probe = ProbeMonitor(run_id=run_id, layer=1, rule="mean", rule_params={}, val_metrics={})
        async_session.add(probe)
        await async_session.flush()
        async_session.add_all([
            ProbeMonitorEvaluation(probe_id=probe.id, dataset_id=views["spare"], status="completed",
                                   metrics={}),
            ProbeMonitorEvaluation(probe_id=probe.id, dataset_id=views["spare"], status="refused",
                                   metrics={}),
        ])
        await async_session.commit()
        await _archive(client, views["spare"])
        codes, result = await _refused(client, async_session, views["spare"])
        assert codes == ["evaluation_on_it"]
        assert result["blockers"][0]["evaluations"] == 2

    async def test_a_behaviour_link_to_it(self, client, async_session):
        """Feature 037: an explicit catalogue link would CASCADE away silently; it blocks instead."""
        _, _, views = await _world(async_session)
        made = await client.post(f"{API}/behaviours", json={
            "key": "pii", "name": "PII", "positive_meaning": "identifies a person"})
        assert made.status_code == 201, made.text
        assert (await client.post(f"{API}/behaviours/pii/views/{views['spare']}")).status_code == 200
        await _archive(client, views["spare"])
        codes, result = await _refused(client, async_session, views["spare"])
        assert codes == ["behaviour_link"]
        assert result["blockers"][0]["behaviour"] == "pii"
        assert (await client.delete(f"{API}/behaviours/pii/views/{views['spare']}")).status_code == 200
        r = await client.post(f"{API}/datasets/purge", json={"ids": [views["spare"]]})
        assert r.json()["results"][0]["outcome"] == "purged"

    @pytest.mark.parametrize("status,code", [("completed", "judge_run_uses_it"),
                                             ("running", "judge_run_in_progress")])
    async def test_a_judge_baseline_on_it(self, client, async_session, status, code):
        from src.models.probe_monitor import ProbeMonitorJudgeRun

        _, _, views = await _world(async_session)
        async_session.add(ProbeMonitorJudgeRun(endpoint="http://j", model="j", prompt_version="v1",
                                               dataset_ids=[views["spare"]], status=status, metrics={}))
        await async_session.commit()
        await _archive(client, views["spare"])
        codes, _ = await _refused(client, async_session, views["spare"])
        assert codes == [code]

    @pytest.mark.parametrize("status,consumed,blocks", [
        ("pending", False, True), ("approved", False, True), ("approved", True, False),
        ("denied", False, False),
    ])
    async def test_an_agent_approval_naming_it(self, client, async_session, status, consumed, blocks):
        from src.core.clock import utc_now
        from src.models.agent_approval import AgentApprovalRequest

        _, _, views = await _world(async_session)
        async_session.add(AgentApprovalRequest(
            tool_name="start_probe_training_run", status=status,
            payload={"train_dataset_id": views["spare"], "eval_dataset_ids": []},
            consumed_at=utc_now() if consumed else None,
        ))
        await async_session.commit()
        await _archive(client, views["spare"])
        if blocks:
            codes, _ = await _refused(client, async_session, views["spare"])
            assert codes == ["approval_names_it"]
        else:
            r = await client.post(f"{API}/datasets/purge", json={"ids": [views["spare"]]})
            assert r.json()["results"][0]["outcome"] == "purged"

    async def test_EVERY_blocker_is_named_at_once(self, client, async_session):
        from src.models.probe_monitor import ProbeMonitorJudgeRun

        model, _, views = await _world(async_session)
        await _run(async_session, model, train_dataset_id=views["train"],
                   eval_dataset_ids=[views["eval"]], status="pending")
        await _run(async_session, model, train_dataset_id=views["other_train"],
                   eval_dataset_ids=[views["eval"]])
        async_session.add(ProbeMonitorJudgeRun(endpoint="http://j", model="j", prompt_version="v1",
                                               dataset_ids=[views["eval"]], status="completed",
                                               metrics={}))
        await async_session.commit()
        await _archive(client, views["eval"])
        codes, result = await _refused(client, async_session, views["eval"])
        assert sorted(codes) == ["judge_run_uses_it", "run_in_progress", "run_uses_it"]
        assert result["reason"].count(";") == 2, "the message names all three"


# ── purge: success, batch, plan, log ──────────────────────────────────────────


class TestPurge:
    async def test_a_clean_archived_view_is_purged_with_a_tombstone(self, client, async_session):
        from src.models.dataset import Dataset
        from src.models.probe_monitor import ProbeMonitorDatasetPurge

        _, dataset, views = await _world(async_session)
        dataset_id = str(dataset.id)
        await _archive(client, views["spare"], json={"reason": "test leftover"})
        r = await client.post(f"{API}/datasets/purge",
                              json={"ids": [views["spare"]], "reason": "audit 2026-10-21"})
        body = r.json()
        assert (body["purged"], body["blocked"]) == (1, 0)
        (result,) = body["results"]
        assert result["outcome"] == "purged" and result["name"] == "spare view"
        assert not await _exists(async_session, views["spare"])

        async_session.expire_all()
        (stone,) = (await async_session.execute(select(ProbeMonitorDatasetPurge))).scalars().all()
        assert stone.id == result["tombstone_id"]
        assert (stone.view_id, stone.name, stone.purged_by, stone.reason) == (
            views["spare"], "spare view", "operator", "audit 2026-10-21")
        assert stone.rows_removed == {"probe_monitor_datasets": 1}
        assert stone.removed["archive_reason"] == "test leftover"
        assert stone.removed["dataset_id"] == dataset_id
        assert stone.removed["label_mapping"] == {"1": "positive"}
        # The DOWNLOAD is kept, and the record says why and how many other views read it.
        (kept,) = stone.kept
        assert kept["dataset_id"] == dataset_id and kept["other_views"] == 4
        assert await _count(async_session, Dataset) == 1

    async def test_the_batch_purges_each_on_its_own(self, client, async_session):
        model, _, views = await _world(async_session)
        await _run(async_session, model, train_dataset_id=views["train"], eval_dataset_ids=[])
        for key in ("train", "spare", "calib"):
            await _archive(client, views[key])
        r = await client.post(f"{API}/datasets/purge", json={
            "ids": [views["train"], views["spare"], "pmd_missing", views["calib"], views["spare"]],
        })
        body = r.json()
        outcomes = {x["id"]: (x["outcome"], x.get("code")) for x in body["results"]}
        assert outcomes == {
            views["train"]: ("blocked", "purge_blocked"),
            views["spare"]: ("purged", None),
            "pmd_missing": ("blocked", "not_found"),
            views["calib"]: ("purged", None),
        }
        assert len(body["results"]) == 4, "a repeated id is purged once"
        assert (body["purged"], body["blocked"]) == (2, 2)
        assert await _exists(async_session, views["train"])

    async def test_the_single_delete_route_is_the_same_purge(self, client, async_session):
        from src.models.probe_monitor import ProbeMonitorDatasetPurge

        _, _, views = await _world(async_session)
        await _archive(client, views["spare"])
        r = await client.delete(f"{API}/datasets/{views['spare']}", params={"reason": "why"})
        assert r.status_code == 200, r.text
        assert r.json()["outcome"] == "purged"
        assert not await _exists(async_session, views["spare"])
        assert await _count(async_session, ProbeMonitorDatasetPurge) == 1
        assert (await client.delete(f"{API}/datasets/{views['spare']}")).status_code == 404

    async def test_the_plan_changes_nothing_and_says_what_each_would_do(self, client, async_session):
        model, _, views = await _world(async_session)
        await _run(async_session, model, train_dataset_id=views["train"], eval_dataset_ids=[])
        await _archive(client, views["train"])
        await _archive(client, views["spare"])
        r = await client.post(f"{API}/datasets/purge/plan",
                              json={"ids": [views["train"], views["spare"], views["eval"], "pmd_x"]})
        items = {i["id"]: i for i in r.json()["items"]}
        assert items[views["spare"]]["purgeable"] is True
        assert items[views["spare"]]["removes"] == {"rows": {"probe_monitor_datasets": 1}}
        assert items[views["spare"]]["kept"][0]["what"] == "downloaded dataset"
        assert items[views["train"]]["purgeable"] is False
        assert [b["code"] for b in items[views["train"]]["blockers"]] == ["run_uses_it"]
        assert items[views["eval"]]["not_archived"]["code"] == "not_archived"
        assert items["pmd_x"]["found"] is False
        for v in views.values():
            assert await _exists(async_session, v)

    async def test_the_log_lists_tombstones_newest_first(self, client, async_session):
        _, _, views = await _world(async_session)
        for key in ("spare", "calib"):
            await _archive(client, views[key])
            await client.post(f"{API}/datasets/purge", json={"ids": [views[key]]})
        log = (await client.get(f"{API}/dataset-purges")).json()
        assert [i["view_id"] for i in log["items"]] == [views["calib"], views["spare"]]


class TestTheDeleteItselfIsConditional:
    async def test_a_view_unarchived_under_the_check_is_not_deleted(self, client, async_session, monkeypatch):
        """Belt and braces under the row lock: the DELETE itself requires `archived_at IS NOT NULL`,
        so even a check that read a stale "archived" cannot delete an unarchived view — and the
        tombstone written before it rolls back with it."""
        from types import SimpleNamespace

        from src.models.probe_monitor import ProbeMonitorDatasetPurge
        from src.services import probe_dataset_lifecycle as lc

        _, _, views = await _world(async_session)
        real = lc._view

        async def stale(db, view_id, *, lock=False):
            row = await real(db, view_id, lock=lock)
            fields = {c.name: getattr(row, c.name) for c in row.__table__.columns}
            fields["archived_at"] = lc.utc_now()
            return SimpleNamespace(**fields)

        monkeypatch.setattr(lc, "_view", stale)
        r = await client.post(f"{API}/datasets/purge", json={"ids": [views["spare"]]})
        (result,) = r.json()["results"]
        assert (result["outcome"], result["code"]) == ("blocked", "purge_raced")
        assert await _exists(async_session, views["spare"])
        assert await _count(async_session, ProbeMonitorDatasetPurge) == 0


# ── purge is the operator's alone ─────────────────────────────────────────────

PURGE_CALLS = [
    ("post", "/datasets/purge", {"json": {"ids": ["{v}"]}}),
    ("post", "/datasets/purge/plan", {"json": {"ids": ["{v}"]}}),
    ("delete", "/datasets/{v}", {}),
    ("get", "/dataset-purges", {}),
]


class TestAgentsNeverPurge:
    @pytest.mark.parametrize("method,path,kw", PURGE_CALLS)
    async def test_every_purge_route_refuses_the_agent(self, client, async_session, method, path, kw):
        from src.models.probe_monitor import ProbeMonitorDatasetPurge

        _, _, views = await _world(async_session)
        await _archive(client, views["spare"])
        v = views["spare"]
        kw = {k: ({"ids": [v]} if k == "json" else val) for k, val in kw.items()}
        r = await getattr(client, method)(API + path.replace("{v}", v), headers=AGENT, **kw)
        assert r.status_code == 403, r.text
        assert r.json()["detail"]["code"] == "purge_operator_only"
        assert await _exists(async_session, v)
        assert await _count(async_session, ProbeMonitorDatasetPurge) == 0

    def test_the_live_app_serves_every_purge_route_with_the_guard(self):
        """Read from the app's own routes, not the source: each purge route depends on
        `refuse_agents`. Remove the dependency from one route and this names it."""
        from src.main import app
        from src.services.probe_dataset_lifecycle import refuse_agents

        def walk(routes):
            # FastAPI 0.141 wraps an included router in `_IncludedRouter` (no `.routes`; memory:
            # app-routes-is-not-a-route-list) — descend through `original_router`.
            for route in routes:
                if hasattr(route, "original_router"):
                    yield from walk(route.original_router.routes)
                elif hasattr(route, "dependant"):
                    yield route
                elif getattr(route, "routes", None):
                    yield from walk(route.routes)

        seen, guarded = 0, set()
        for route in walk(app.router.routes):
            seen += 1
            calls = {d.call for d in route.dependant.dependencies}
            if refuse_agents in calls:
                for m in route.methods:
                    guarded.add((m, route.path))
        assert seen > 100, "the walk must reach the app's real routes or this proves nothing"
        want = {("POST", "/probe-monitors/datasets/purge"),
                ("POST", "/probe-monitors/datasets/purge/plan"),
                ("DELETE", "/probe-monitors/datasets/{dataset_id}"),
                ("GET", "/probe-monitors/dataset-purges")}
        guarded = {(m, p[p.index("/probe-monitors"):]) for m, p in guarded if "/probe-monitors" in p}
        served = set(app.openapi()["paths"])
        assert {"/api/v1" + p for _, p in want} <= served
        assert want <= guarded, want - guarded

    def test_no_registered_mcp_tool_purges_or_calls_a_purge_route(self, monkeypatch):
        """The LIVE registry with every category on: no tool name says purge/delete of a view,
        and no tool issues a call to a purge route."""
        from tests.support.mcp_route_map import called_routes, normalise, registered_tools

        names = registered_tools(monkeypatch)
        assert "archive_probe_dataset" in names and "unarchive_probe_dataset" in names
        assert "get_probe_dataset_rows" in names
        assert not [n for n in names if "purge" in n]
        assert "delete_probe_dataset" not in names
        purge_routes = {
            ("POST", normalise("/probe-monitors/datasets/purge")),
            ("POST", normalise("/probe-monitors/datasets/purge/plan")),
            ("DELETE", normalise("/probe-monitors/datasets/{dataset_id}")),
            ("GET", normalise("/probe-monitors/dataset-purges")),
        }
        called = called_routes(names, "mistudio")
        assert ("POST", normalise("/probe-monitors/datasets/{id}/archive")) in called, (
            "the reader must see the archive tool's route, or this test proves nothing"
        )
        assert not purge_routes & set(called)


# ── the tombstone is append-only ──────────────────────────────────────────────


class TestTombstonesAreAppendOnly:
    async def _stone(self, client, async_session):
        from src.models.probe_monitor import ProbeMonitorDatasetPurge

        _, _, views = await _world(async_session)
        await _archive(client, views["spare"])
        await client.post(f"{API}/datasets/purge", json={"ids": [views["spare"]]})
        async_session.expire_all()
        return (await async_session.execute(select(ProbeMonitorDatasetPurge))).scalars().one()

    async def test_the_orm_refuses_an_update_and_a_delete(self, client, async_session):
        from src.models.probe_monitor import PurgeTombstoneIsAppendOnly

        stone = await self._stone(client, async_session)
        stone.reason = "rewritten"
        with pytest.raises(PurgeTombstoneIsAppendOnly):
            await async_session.flush()
        await async_session.rollback()
        stone = await self._stone_again(async_session)
        await async_session.delete(stone)
        with pytest.raises(PurgeTombstoneIsAppendOnly):
            await async_session.flush()
        await async_session.rollback()

    async def _stone_again(self, async_session):
        from src.models.probe_monitor import ProbeMonitorDatasetPurge

        return (await async_session.execute(select(ProbeMonitorDatasetPurge))).scalars().one()

    @pytest.mark.parametrize("sql", [
        "UPDATE probe_monitor_dataset_purges SET reason = 'x'",
        "DELETE FROM probe_monitor_dataset_purges",
    ])
    async def test_the_database_refuses_raw_sql(self, client, async_session, sql):
        from sqlalchemy.exc import DBAPIError

        await self._stone(client, async_session)
        with pytest.raises(DBAPIError, match="append-only"):
            await async_session.execute(text(sql))
        await async_session.rollback()
