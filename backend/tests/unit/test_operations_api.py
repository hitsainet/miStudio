"""G9 backend half: `/operations/active-jobs` and `/operations/gpu-status` (2026-10-21).

Driven through the real ASGI app and the test database (`client`). The kind list is checked
against the ORM METADATA — the authority — so a job table added later without a source fails
here (memory: derive-from-the-registry-not-the-source).

Controls (review record `0xcc/reviews/2026-10-21_mcp-gaps.md`, restored from saved bytes):
  C-ROUTE  delete `api_router.include_router(operations.router)` → test_the_routes_are_served red
  C-PROBE  delete the probe_training JobSource → the probe tests and the completeness test red
  C-SWALLOW make `list_active_jobs` drop a failed source silently → test_a_broken_source_is_NAMED red
  C-LEASE  drop the lease join in `assemble` → test_a_leased_card_names_its_holder red
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest



API = "/api/v1/operations"
CARD_A = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
CARD_B = "GPU-f47ba814-49a2-603f-3595-275284140251"


def test_the_routes_are_served():
    from src.main import app

    paths = app.openapi()["paths"]
    assert set(paths[API + "/active-jobs"]) == {"get"}
    assert set(paths[API + "/gpu-status"]) == {"get"}


def test_every_status_bearing_table_is_a_source_or_a_reasoned_exemption():
    """The completeness guard. A table with a job-like status column (`status` / `*_status`
    beside a task id or a GPU column, or any status column at all) must be listed."""
    import src.models  # noqa: F401
    from src.core.database import Base
    from src.services.active_jobs import JOB_SOURCES, NOT_JOB_TABLES

    covered = {s.table for s in JOB_SOURCES}
    status_tables = {
        t.name for t in Base.metadata.tables.values()
        if any(c == "status" or c.endswith("_status") for c in t.c.keys())
    }
    missing = sorted(status_tables - covered - set(NOT_JOB_TABLES))
    assert not missing, f"job tables with no active-jobs source and no exemption: {missing}"
    assert not set(NOT_JOB_TABLES) & covered
    assert all(len(r) > 20 for r in NOT_JOB_TABLES.values())


def test_every_task_bearing_status_column_is_a_source():
    """Finer: a `*_status` column with a sibling `*_task_id` is its own job (attribution,
    validation, faithfulness, calibration) — one source per column, not per table."""
    import src.models  # noqa: F401
    from src.core.database import Base
    from src.services.active_jobs import JOB_SOURCES, NOT_JOB_TABLES

    sources = {(s.table, s.status_column) for s in JOB_SOURCES}
    missing = []
    for t in Base.metadata.tables.values():
        if t.name in NOT_JOB_TABLES:
            continue
        for col in t.c.keys():
            if col.endswith("_status") and col.replace("_status", "_task_id") in t.c.keys():
                if (t.name, col) not in sources:
                    missing.append((t.name, col))
    assert not missing, missing


def test_every_source_names_a_real_column():
    import src.models  # noqa: F401
    from src.core.database import Base
    from src.services.active_jobs import JOB_SOURCES

    for s in JOB_SOURCES:
        table = Base.metadata.tables[s.table]
        assert s.status_column in table.c, s
        assert "created_at" in table.c, s
        if s.task_column:
            assert s.task_column in table.c, s


async def _seed(session, *, run_status="running", gpu=CARD_A):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import (
        ProbeMonitorDataset, ProbeMonitorJudgeRun, ProbeMonitorRun)

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY,
                  file_path="/x", architecture="LlamaForCausalLM", params_count=1)
    downloading = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="dl",
                        status=ModelStatus.DOWNLOADING, file_path="/y",
                        architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="c", source="local", status=DatasetStatus.READY,
                      raw_path="/x", extra_metadata={})
    session.add_all([model, downloading, dataset])
    await session.flush()
    view = ProbeMonitorDataset(name="V", dataset_id=dataset.id, input_column="i",
                               label_column="l", label_mapping={}, role="train", counts={})
    session.add(view)
    await session.flush()
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[],
                          config={}, status=run_status, environment={}, gpu_uuid=gpu,
                          gpu_uuids=[gpu] if gpu else None, celery_task_id="celery-1")
    done = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[],
                           config={}, status="completed", environment={})
    judge = ProbeMonitorJudgeRun(endpoint="http://judge", model="j", prompt_version="v1",
                                 dataset_ids=[], status="pending")
    session.add_all([run, done, judge])
    await session.commit()
    return {"run": run.id, "done": done.id, "judge": judge.id, "downloading": downloading.id}


async def test_active_jobs_of_several_kinds_in_one_read(client, async_session):
    ids = await _seed(async_session)
    body = (await client.get(API + "/active-jobs")).json()
    by_id = {(j["kind"], j["id"]): j for j in body["jobs"]}

    run = by_id[("probe_training", ids["run"])]
    assert run["status"] == "running"
    assert run["gpu_uuid"] == CARD_A and run["gpu_uuids"] == [CARD_A]
    assert run["task_id"] == "celery-1"
    assert ("probe_judge", ids["judge"]) in by_id
    # Stored as the enum NAME ("DOWNLOADING"); compared case-insensitively.
    assert by_id[("model", ids["downloading"])]["status"] == "downloading"
    assert ("probe_training", ids["done"]) not in by_id, "a completed run is not active"
    assert body["errors"] == [], body["errors"]
    assert body["count"] == len(body["jobs"])
    assert "probe_training" in body["kinds"] and "steering_worker" in body["kinds"]


async def test_every_source_query_runs_against_the_real_schema(client):
    """Each source's SELECT executes (an empty DB is fine): a misspelt column is an error, and
    errors are reported, so this asserts there are none."""
    body = (await client.get(API + "/active-jobs")).json()
    assert body["errors"] == []


async def test_the_kind_and_gpu_filters(client, async_session):
    ids = await _seed(async_session)
    only = (await client.get(API + "/active-jobs", params={"kind": "probe_training"})).json()
    assert {j["kind"] for j in only["jobs"]} == {"probe_training"}
    assert [j["id"] for j in only["jobs"]] == [ids["run"]]
    on_b = (await client.get(API + "/active-jobs", params={"gpu": CARD_B})).json()
    assert on_b["jobs"] == [] and on_b["count"] == 0
    on_a = (await client.get(API + "/active-jobs", params={"gpu": CARD_A.lower()})).json()
    assert [j["id"] for j in on_a["jobs"]] == [ids["run"]]


async def test_an_unknown_kind_is_refused_by_name(client):
    r = await client.get(API + "/active-jobs", params={"kind": "probe_trainng"})
    assert r.status_code == 422
    assert "probe_training" in r.text


async def test_a_broken_source_is_NAMED_not_swallowed(client, monkeypatch):
    from src.services import active_jobs

    broken = active_jobs.JobSource("sae_training", "trainings", ("running",),
                                   status_column="no_such_column")
    monkeypatch.setattr(active_jobs, "JOB_SOURCES", (broken,) + active_jobs.JOB_SOURCES[1:])
    body = (await client.get(API + "/active-jobs")).json()
    assert [e["kind"] for e in body["errors"]] == ["sae_training"]


# ── gpu-status ──────────────────────────────────────────────────────────────────────────────


def _fake_cards():
    def card(uuid, index, used, pids):
        return {"uuid": uuid, "index": index, "name": "NVIDIA Test",
                "memory": {"total_mib": 24576.0, "used_mib": used, "free_mib": 24576.0 - used},
                "utilization": {"gpu_percent": 0, "memory_percent": 0},
                "processes": [{"pid": p, "used_mib": 300.0} for p in pids]}
    return [card(CARD_A, 0, 9000.0, [4242]), card(CARD_B, 1, 0.0, [])], None


async def test_every_card_with_its_owner(client, async_session, monkeypatch):
    from src.models.gpu_lease import GpuLease
    from src.services import gpu_status

    monkeypatch.setattr(gpu_status, "read_cards", lambda: _fake_cards())
    ids = await _seed(async_session)
    now = datetime.now(timezone.utc)
    async_session.add(GpuLease(gpu_uuid=CARD_A, holder=f"probe_monitor_run:{ids['run']}",
                               task_id="celery-1", acquired_at=now, heartbeat_at=now,
                               expires_at=now + timedelta(minutes=10)))
    await async_session.commit()

    body = (await client.get(API + "/gpu-status")).json()
    cards = {c["uuid"]: c for c in body["cards"]}
    assert set(cards) == {CARD_A, CARD_B} and body["card_count"] == 2
    a, b = cards[CARD_A], cards[CARD_B]
    assert a["lease"]["holder"] == f"probe_monitor_run:{ids['run']}"
    assert a["lease"]["kind"] == "probe_monitor_run"
    assert [j["id"] for j in a["jobs"]] == [ids["run"]]
    assert a["in_use"] is True and len(a["in_use_by"]) >= 2
    assert a["processes"][0]["owner"] is None, "a PID from another pod is never guessed"
    assert b["in_use"] is False and b["lease"] is None and b["jobs"] == []


async def test_a_leased_card_names_its_holder_even_with_no_job_row(client, async_session,
                                                                   monkeypatch):
    from src.models.gpu_lease import GpuLease
    from src.services import gpu_status

    monkeypatch.setattr(gpu_status, "read_cards", lambda: _fake_cards())
    now = datetime.now(timezone.utc)
    async_session.add(GpuLease(gpu_uuid=CARD_B, holder="training:train_x", acquired_at=now,
                               heartbeat_at=now, expires_at=now + timedelta(minutes=10)))
    async_session.add(GpuLease(gpu_uuid=CARD_A, holder="training:old", acquired_at=now,
                               heartbeat_at=now, expires_at=now - timedelta(minutes=1)))
    await async_session.commit()
    body = (await client.get(API + "/gpu-status", params={"gpu": CARD_B})).json()
    (card,) = body["cards"]
    assert card["lease"]["holder"] == "training:train_x" and card["in_use"] is True
    a = (await client.get(API + "/gpu-status", params={"gpu": CARD_A})).json()["cards"][0]
    assert a["lease"] is None, "an expired lease holds nothing"


async def test_one_card_by_uuid_an_index_is_refused_an_unknown_uuid_is_404(client, monkeypatch):
    from src.services import gpu_status

    monkeypatch.setattr(gpu_status, "read_cards", lambda: _fake_cards())
    one = (await client.get(API + "/gpu-status", params={"gpu": CARD_A.lower()})).json()
    assert [c["uuid"] for c in one["cards"]] == [CARD_A]
    assert (await client.get(API + "/gpu-status", params={"gpu": "0"})).status_code == 422
    missing = await client.get(API + "/gpu-status", params={"gpu": "GPU-00000000-dead"})
    assert missing.status_code == 404
    assert CARD_A in missing.text


async def test_no_gpu_is_an_answer_not_an_error(client, monkeypatch):
    from src.services import gpu_status

    monkeypatch.setattr(gpu_status, "read_cards", lambda: ([], "NVML could not start (no driver)"))
    body = (await client.get(API + "/gpu-status")).json()
    assert body["cards"] == [] and "no driver" in body["reason"]


def test_read_cards_reads_every_card_nvml_reports():
    """1+n: whatever count NVML gives, never an assumed one."""
    from types import SimpleNamespace

    from src.services.gpu_status import read_cards

    class FakeNVML:
        def __init__(self, n):
            self.n, self.calls = n, []

        def nvmlInit(self):
            self.calls.append("init")

        def nvmlShutdown(self):
            self.calls.append("shutdown")

        def nvmlDeviceGetCount(self):
            return self.n

        def nvmlDeviceGetHandleByIndex(self, i):
            return i

        def nvmlDeviceGetMemoryInfo(self, h):
            return SimpleNamespace(total=24 * 1024**3, used=(h + 1) * 1024**3,
                                   free=(23 - h) * 1024**3)

        def nvmlDeviceGetUtilizationRates(self, h):
            return SimpleNamespace(gpu=10 * h, memory=5)

        def nvmlDeviceGetComputeRunningProcesses(self, h):
            return [SimpleNamespace(pid=100 + h, usedGpuMemory=512 * 1024**2)]

        def nvmlDeviceGetUUID(self, h):
            return f"GPU-{h:08d}".encode()

        def nvmlDeviceGetName(self, h):
            return b"Card"

    for n in (1, 3):
        nvml = FakeNVML(n)
        cards, reason = read_cards(nvml)
        assert reason is None and len(cards) == n
        assert cards[-1]["uuid"] == f"GPU-{n - 1:08d}"
        assert cards[-1]["memory"]["used_mib"] == n * 1024.0
        assert cards[0]["processes"] == [{"pid": 100, "used_mib": 512.0}]
        assert nvml.calls == ["init", "shutdown"]
