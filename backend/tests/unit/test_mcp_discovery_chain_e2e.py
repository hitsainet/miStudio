"""016 FPRD §8.7: an MCP agent completes capture → discovery → attribution → report, without the UI.

`test_mcp_circuits_tools.py` smoke-tests each tool alone against a mocked client, which
proves a tool sends its request and nothing about whether the chain can be walked. This
file walks it the way an agent does:

  * THE LIVE REGISTRY. The server is the real `build_server()`, enabled with the
    categories the deployed manifest (`k8s/base/mcp.yaml`) names, and every call goes
    through an MCP client session (`call_tool` over the protocol), never an imported
    tool function.
  * THE REAL BACKEND. The MCP server's own `MiStudioClient` is pointed at the FastAPI
    app (in process, over ASGI), so every hop is the real route, schema and service
    against the test database.
  * FAKES ONLY WHERE A GPU WOULD BE. The task broker is a recorder; a "worker" then runs
    the REAL Celery task bodies. Inside them only the GPU work is replaced: the capture's
    forward pass (a fixture store is written instead), the card placement, and, for
    attribution, the model and SAE loads (a four-dimensional toy model whose layer-13
    feature 1 feeds layer-14 feature 2). Discovery and the attribution arithmetic run for
    real.

At each hop the test asserts the HTTP request the tool issued, the task it queued (name,
arguments, count) and what the next tool reads back.

What this does not prove: the same chain against the deployed pod on a real model. That
live run is recorded as outstanding in 016 FTASKS Task 6.2.

MUTATION CONTROLS (2026-10-21, each restored by sha256): see 016 FTASKS Task 6.2.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
import yaml
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from src.models.circuit_runs import CircuitCaptureRun
from src.models.dataset import Dataset
from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from src.models.external_sae import ExternalSAE
from src.models.model import Model, ModelStatus, QuantizationFormat
from tests.unit.test_circuit_discovery_service import DOC_LEN, N_DOCS, _build_store

REPO = Path(__file__).resolve().parents[3]
CHAIN_TOOLS = ("start_circuit_capture", "list_circuit_captures", "run_circuit_discovery",
               "get_discovery_results", "run_attribution_pass")
HELDOUT = list(range(32, 40))
D_MODEL, D_SAE, N_LAYERS = 4, 8, 15


def _deployed_categories() -> str:
    """MCP_TOOL_CATEGORIES exactly as the manifest ArgoCD deploys sets it."""
    doc = next(d for d in yaml.safe_load_all((REPO / "k8s/base/mcp.yaml").read_text())
               if d and d.get("kind") == "Deployment")
    env = doc["spec"]["template"]["spec"]["containers"][0]["env"]
    return next(e["value"] for e in env if e["name"] == "MCP_TOOL_CATEGORIES")


# ── the toy model and SAEs (the GPU stand-ins for attribution) ───────────────


class _Block(torch.nn.Module):
    def __init__(self, mix=None):
        super().__init__()
        self.mix = None if mix is None else torch.nn.Parameter(mix, requires_grad=False)

    def forward(self, x):
        return x if self.mix is None else x + x @ self.mix


class _ToyModel(torch.nn.Module):
    """Token 1 writes dim 0, token 2 writes dim 1; layer 14 copies dim 0 into dim 2."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, D_MODEL)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[1, 0] = 1.0
            self.embed.weight[2, 1] = 1.0
        mix = torch.zeros(D_MODEL, D_MODEL)
        mix[0, 2] = 1.0
        self.layers = torch.nn.ModuleList(
            [_Block(mix if i == 14 else None) for i in range(N_LAYERS)])

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


class _ToySAE(torch.nn.Module):
    """Feature k reads and writes one basis direction."""

    normalize_activations = "none"

    def __init__(self, reads):
        super().__init__()
        w = torch.zeros(D_MODEL, D_SAE)
        for feature, dim in reads.items():
            w[dim, feature] = 1.0
        self.W_enc = torch.nn.Parameter(w)
        self.W_dec = torch.nn.Parameter(w.T.clone())

    def encode(self, x):
        return torch.relu(x @ self.W_enc)

    def decode(self, f):
        return f @ self.W_dec


SAE_READS = {13: {1: 0, 3: 1}, 14: {2: 2, 4: 3}}


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def maker(async_engine):
    sync_url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", ""))
    engine = create_engine(sync_url)
    # As `SyncSessionLocal` is configured, so rows outlive the session that made them.
    yield sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    engine.dispose()


@pytest.fixture
def seeded(maker):
    db = maker()
    db.add(Model(id="m_toy", name="toy", architecture="toy", params_count=1, repo_id="toy/toy",
                 quantization=QuantizationFormat.FP16, status=ModelStatus.READY))
    ds = Dataset(name="toy corpus", source="Local")
    db.add(ds)
    db.commit()
    db.add(DatasetTokenization(id="tok_toy", dataset_id=ds.id, model_id="m_toy",
                               tokenized_path="/nonexistent/tok_toy", tokenizer_repo_id="toy/toy",
                               status=TokenizationStatus.READY))
    for layer in (13, 14):
        db.add(ExternalSAE(id=f"sae_chain{layer}", name=f"L{layer}", source="local", status="ready",
                           layer=layer, hook_type="resid_post", model_id="m_toy",
                           local_path=f"/nonexistent/sae{layer}", d_model=D_MODEL, n_features=D_SAE))
    db.commit()
    out = {"dataset_id": str(ds.id)}
    db.close()
    return out


@pytest.fixture
def backend(async_engine, maker, monkeypatch):
    """The real app on the test database; the sync bridge on the same database."""
    from src.core.database import get_db
    from src.main import app

    amaker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def _db():
        async with amaker() as session:
            yield session

    @contextmanager
    def _sync_db():
        session = maker()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    app.dependency_overrides[get_db] = _db
    monkeypatch.setattr("src.core.database.get_sync_db", _sync_db)
    yield SimpleNamespace(app=app, sync_db=_sync_db)
    app.dependency_overrides.pop(get_db, None)


class _Broker:
    """Records every queued task; `drain()` runs them through the REAL task bodies."""

    def __init__(self, sync_db):
        self.queued, self.sent = [], []
        self._sync_db = sync_db

    def delay_for(self, task, gpu_request=None):
        def delay(*args, **kwargs):
            job = {"task": task.name.rsplit(".", 1)[-1], "args": args, "kwargs": kwargs,
                   "gpu_request": gpu_request, "task_obj": task}
            self.queued.append(job)
            self.sent.append({k: v for k, v in job.items() if k != "task_obj"})
            return SimpleNamespace(id=f"celery-{len(self.sent)}")
        return delay

    def drain(self):
        while self.queued:
            job = self.queued.pop(0)
            fake_self = SimpleNamespace(get_db=self._sync_db, request=SimpleNamespace(id="t"))
            job["task_obj"].__wrapped__.__func__(fake_self, *job["args"], **job["kwargs"])


@pytest.fixture
def gpu_fakes(monkeypatch, tmp_path, backend):
    """Broker, placement, capture forward pass and model/SAE loads — nothing else."""
    from src.api.v1.endpoints import circuit_discovery as endpoints
    from src.services.circuit_capture_service import CircuitCaptureService
    from src.workers import circuit_capture_tasks as tasks

    broker = _Broker(backend.sync_db)
    real_discovery_task = tasks.run_circuit_discovery
    monkeypatch.setattr(endpoints, "gpu_delay",
                        lambda task, gpu_request=None: broker.delay_for(task, gpu_request))
    monkeypatch.setattr(tasks, "run_circuit_discovery",
                        SimpleNamespace(delay=broker.delay_for(real_discovery_task),
                                        name=real_discovery_task.name))
    calls = {"capture": [], "placement": [], "model_load": [], "sae_load": [], "events": []}

    placement = SimpleNamespace(device="cpu", is_shard=False,
                                all_devices=(torch.device("cpu"),), gpu_columns=lambda: {})

    def place(gpu_request, **kw):
        calls["placement"].append({"request": gpu_request, "kind": kw.get("kind")})
        return placement, {"request": gpu_request or "auto", "uuid": None}

    monkeypatch.setattr("src.workers.circuit_gpu.place_circuit_job", place)
    monkeypatch.setattr("src.workers.circuit_gpu.circuit_required_mb", lambda *a, **k: None)
    for name in ("emit_circuit_run_progress", "emit_circuit_run_completed", "emit_circuit_run_failed"):
        monkeypatch.setattr(tasks, name, lambda *a, _n=name, **k: calls["events"].append((_n, a[:2])))

    def fake_run_capture(db, run_id, *, confirmed, device, placement=None,
                         cancel_check=None, progress_cb=None):
        """The forward pass. An estimate records one; a confirmed run writes the store."""
        calls["capture"].append({"run_id": run_id, "confirmed": confirmed, "device": device})
        row = db.query(CircuitCaptureRun).filter(CircuitCaptureRun.id == run_id).first()
        manifest = dict(row.manifest)
        if not confirmed:
            manifest["estimate"] = {"minutes": 0.1, "events": 9000, "bytes": 108000}
            row.manifest, row.status = manifest, "estimated"
        else:
            store = tmp_path / run_id
            store.mkdir()
            _build_store(store)
            manifest["split"] = dict(manifest["split"], heldout_docs=HELDOUT)
            manifest["doc_lengths"] = {str(d): DOC_LEN for d in range(N_DOCS)}
            manifest["counts"] = {"documents": N_DOCS, "events": 9000, "tokens": N_DOCS * DOC_LEN}
            row.manifest, row.store_path, row.status = manifest, str(store), "completed"
        db.commit()
        return {"status": row.status}

    monkeypatch.setattr(CircuitCaptureService, "run_capture", staticmethod(fake_run_capture))

    def fake_load_model(**kw):
        calls["model_load"].append(kw.get("repo_id"))
        return _ToyModel(), SimpleNamespace(pad_token_id=0, eos_token_id=0), None, {"model_dtype": "float32"}

    def fake_load_sae(record, device):
        calls["sae_load"].append(record.id)
        return _ToySAE(SAE_READS[record.layer])

    import datasets

    prompts = datasets.Dataset.from_dict({"input_ids": [[1, 1, 2, 0], [2, 1, 3, 1],
                                                        [1, 2, 2, 1], [3, 1, 1, 2],
                                                        [1, 3, 3, 3], [2, 2, 2, 1]]})
    monkeypatch.setattr("src.ml.model_loader.load_model_from_hf", fake_load_model)
    monkeypatch.setattr(datasets, "load_from_disk", lambda path: prompts)
    monkeypatch.setattr("src.services.circuit_capture_service._load_sae_sync", fake_load_sae)
    monkeypatch.setattr("src.ml.layer_discovery.discover_transformer_structure",
                        lambda model: SimpleNamespace(layers_module=model.layers))
    monkeypatch.setattr("src.ml.layer_discovery.get_hookable_module",
                        lambda layer_module, kind, structure: layer_module)
    return SimpleNamespace(broker=broker, calls=calls)


@pytest.fixture
def mcp_server(backend, monkeypatch):
    """The live registry: build_server() with the deployed categories, wired to the app."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    for var in ("MILLM_API_URL", "MIDATAWORKS_API_URL"):
        monkeypatch.delenv(var, raising=False)
    settings = MCPSettings(tool_categories=_deployed_categories(), allow_anonymous=True,
                           _env_file=None)
    mcp, client = build_server(settings, stdio=True)
    requests = []

    async def record(request):
        requests.append((request.method, request.url.path,
                         json.loads(request.content) if request.content else None,
                         dict(request.url.params)))

    client._http = httpx.AsyncClient(  # noqa: SLF001 — the tools hold this client
        transport=httpx.ASGITransport(app=backend.app), base_url="http://test/api/v1",
        event_hooks={"request": [record]})
    return SimpleNamespace(mcp=mcp, requests=requests)


async def _call(session, tool, args):
    result = await session.call_tool(tool, args)
    assert not result.is_error, f"{tool} failed: {result.content}"
    return json.loads(result.content[0].text)


# ── the chain ────────────────────────────────────────────────────────────────


class TestAnAgentWalksTheDiscoveryChain:
    @pytest.mark.asyncio
    async def test_capture_discovery_attribution_report_through_the_live_registry(
            self, seeded, gpu_fakes, mcp_server):
        from mcp import Client

        broker, calls, reqs = gpu_fakes.broker, gpu_fakes.calls, mcp_server.requests
        layers = [{"layer": 13, "sae_id": "sae_chain13"}, {"layer": 14, "sae_id": "sae_chain14"}]

        # mcp 2.x: the in-process client takes the server itself (1.x: a memory-stream session
        # over `_mcp_server`). Still the protocol path: list_tools/call_tool go over JSON-RPC.
        async with Client(mcp_server.mcp) as session:
            listed = {t.name for t in (await session.list_tools()).tools}
            assert set(CHAIN_TOOLS) <= listed, (
                f"not in the deployed registry: {sorted(set(CHAIN_TOOLS) - listed)}")

            # 1. Estimate.
            est = await _call(session, "start_circuit_capture", {
                "dataset_id": seeded["dataset_id"], "layers": layers, "sample_cap": N_DOCS})
            assert est["status"] == "queued" and est["confirmed"] is False
            assert reqs[-1][:2] == ("POST", "/api/v1/circuit-capture")
            assert reqs[-1][2]["confirm"] is False and reqs[-1][2]["layers"] == layers
            assert broker.sent == [{"task": "capture_circuit_activations", "args": (est["id"],),
                                    "kwargs": {"confirmed": False}, "gpu_request": "auto"}]
            broker.drain()
            assert calls["capture"] == [{"run_id": est["id"], "confirmed": False, "device": "cpu"}]

            # 2. The agent reads the estimate back.
            caps = {c["id"]: c for c in (await _call(session, "list_circuit_captures", {}))["captures"]}
            assert caps[est["id"]]["status"] == "estimated"
            assert caps[est["id"]]["estimate"]["minutes"] == 0.1

            # 3. Confirm: a second start with confirm=true runs the full capture.
            full = await _call(session, "start_circuit_capture", {
                "dataset_id": seeded["dataset_id"], "layers": layers, "sample_cap": N_DOCS,
                "confirm": True})
            assert full["confirmed"] is True and full["id"] != est["id"]
            assert broker.sent[-1] == {"task": "capture_circuit_activations", "args": (full["id"],),
                                       "kwargs": {"confirmed": True}, "gpu_request": "auto"}
            broker.drain()
            assert len(calls["capture"]) == 2 and calls["capture"][1]["confirmed"] is True
            caps = {c["id"]: c for c in (await _call(session, "list_circuit_captures", {}))["captures"]}
            assert caps[full["id"]]["status"] == "completed"
            assert caps[full["id"]]["split"]["heldout_count"] == len(HELDOUT)

            # 4. Discovery over the completed capture, run by the real task and service.
            dsc = await _call(session, "run_circuit_discovery", {
                "capture_run_id": full["id"], "granularity": "feature", "mode": "open",
                "s_min": 20, "null_shuffles": 50})
            assert reqs[-1][:2] == ("POST", "/api/v1/circuit-discovery")
            assert reqs[-1][2]["capture_run_id"] == full["id"] and reqs[-1][2]["null_shuffles"] == 50
            assert broker.sent[-1] == {"task": "run_circuit_discovery", "args": (dsc["id"],),
                                       "kwargs": {}, "gpu_request": None}
            broker.drain()

            got = await _call(session, "get_discovery_results", {"run_id": dsc["id"]})
            assert reqs[-1][:2] == ("GET", f"/api/v1/circuit-discovery/{dsc['id']}")
            assert reqs[-1][3] == {"include_candidates": "true"}
            assert got["status"] == "completed"
            top = got["candidates"][0]
            assert (top["up"], top["down"]) == ({"layer": 13, "feature_idx": 1},
                                                {"layer": 14, "feature_idx": 2})
            assert top["attribution"] is None and got["attribution_status"] is None
            assert got["report"]["fdr"]["discipline"] == "benjamini_hochberg"

            # 5. Attribution: the real service, with the toy model in place of the GPU load.
            att = await _call(session, "run_attribution_pass", {"run_id": dsc["id"], "prompt_limit": 4})
            assert att["status"] == "queued"
            assert reqs[-1][:3] == ("POST", f"/api/v1/circuit-discovery/{dsc['id']}/attribution",
                                    {"prompt_limit": 4, "gpu": "auto"})
            assert broker.sent[-1] == {"task": "run_circuit_attribution", "args": (dsc["id"],),
                                       "kwargs": {"prompt_limit": 4, "gpu_request": "auto"},
                                       "gpu_request": None}
            broker.drain()
            assert calls["model_load"] == ["toy/toy"]
            assert sorted(calls["sae_load"]) == ["sae_chain13", "sae_chain14"]

            # 6. The report, retrieved by the agent.
            rep = await _call(session, "get_discovery_results", {"run_id": dsc["id"]})
            assert rep["attribution_status"] == "completed", rep.get("attribution_error")
            top = rep["candidates"][0]
            assert top["attribution"]["score"] > 0
            assert top["attribution"]["rung1_gate"] is True
            assert top["orderings"] == {"coact_rank": 0, "attr_rank": 0}
            assert rep["report"]["attribution"]["prompts"] == 4
            assert "lag-0" in rep["report"]["attribution"]["lag0_disclosure"]

        # Every hop, in order, and nothing else: four tasks, two placements, seven reads/writes.
        assert [s["task"] for s in broker.sent] == [
            "capture_circuit_activations", "capture_circuit_activations",
            "run_circuit_discovery", "run_circuit_attribution"]
        assert [p["kind"] for p in calls["placement"]] == ["capture", "capture", "attribution"]
        assert [(m, p) for m, p, *_ in reqs] == [
            ("POST", "/api/v1/circuit-capture"),
            ("GET", "/api/v1/circuit-capture"),
            ("POST", "/api/v1/circuit-capture"),
            ("GET", "/api/v1/circuit-capture"),
            ("POST", "/api/v1/circuit-discovery"),
            ("GET", f"/api/v1/circuit-discovery/{dsc['id']}"),
            ("POST", f"/api/v1/circuit-discovery/{dsc['id']}/attribution"),
            ("GET", f"/api/v1/circuit-discovery/{dsc['id']}"),
        ]
        assert not [e for e in calls["events"] if e[0] == "emit_circuit_run_failed"]

    def test_the_chain_tools_are_in_the_deployed_category_list(self):
        """Specificity for the registry assertion above: the manifest is what was read."""
        cats = {c.strip() for c in _deployed_categories().split(",")}
        assert "circuits" in cats
