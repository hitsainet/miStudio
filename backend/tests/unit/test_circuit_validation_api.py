"""API pins for circuit validation + manifests (017): 202/409/422, cancel,
manifest retrieval, reproduce gating, faithfulness dispatch."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.circuit import Circuit
from src.models.circuit_runs import CircuitDiscoveryRun
from src.models.validation_manifest import ValidationManifest


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def _run(**kw):
    from datetime import datetime
    return CircuitDiscoveryRun(
        id=kw.pop("id", "dsc1"), capture_run_id="cap1",
        status=kw.pop("status", "completed"), params={},
        candidates=kw.pop("candidates", [{"up": {}, "down": {}}]),
        created_at=datetime(2026, 7, 20), updated_at=datetime(2026, 7, 20), **kw)


class TestValidateAPI:
    def test_needs_completed_run_409(self, client):
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=_run(status="running"))):
            r = client.post("/api/v1/circuit-discovery/dsc1/validate", json={})
        assert r.status_code == 409

    def test_no_candidates_409(self, client):
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=_run(candidates=[]))):
            r = client.post("/api/v1/circuit-discovery/dsc1/validate", json={})
        assert r.status_code == 409

    def test_already_in_flight_409(self, client):
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=_run(validation_status="running"))):
            r = client.post("/api/v1/circuit-discovery/dsc1/validate", json={})
        assert r.status_code == 409

    def test_bad_ordering_422(self, client):
        r = client.post("/api/v1/circuit-discovery/dsc1/validate",
                        json={"ordering": "bogus"})
        assert r.status_code == 422

    def test_cancel_no_validation_409(self, client):
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=_run(validation_status=None))):
            r = client.post("/api/v1/circuit-discovery/dsc1/validate/cancel")
        assert r.status_code == 409

    def test_cancel_running_200(self, client):
        run = _run(validation_status="running", validation_task_id="vt1")
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=run)), \
             patch("src.core.celery_app.revoke_task"):
            r = client.post("/api/v1/circuit-discovery/dsc1/validate/cancel")
        assert r.status_code == 200
        assert r.json()["validation_status"] == "cancelled"


class TestManifestAPI:
    def _m(self, **kw):
        from datetime import datetime
        return ValidationManifest(
            id=kw.pop("id", "vman_1"), kind=kw.pop("kind", "edge_batch"),
            discovery_run_id="dsc1", payload={"config": {"ordering": "coact"}},
            created_at=datetime(2026, 7, 20), **kw)

    def test_get_404(self, client):
        with patch("src.services.manifest_service.ManifestService.get",
                   new=AsyncMock(return_value=None)):
            r = client.get("/api/v1/validation-manifests/vman_x")
        assert r.status_code == 404

    def test_get_200(self, client):
        with patch("src.services.manifest_service.ManifestService.get",
                   new=AsyncMock(return_value=self._m())):
            r = client.get("/api/v1/validation-manifests/vman_1")
        assert r.status_code == 200 and r.json()["kind"] == "edge_batch"

    def test_reproduce_non_edge_batch_409(self, client):
        with patch("src.services.manifest_service.ManifestService.get",
                   new=AsyncMock(return_value=self._m(kind="faithfulness"))):
            r = client.post("/api/v1/validation-manifests/vman_1/reproduce")
        assert r.status_code == 409


def _circuit(**kw):
    from datetime import datetime
    return Circuit(
        id=kw.pop("id", "crc_f1"), name="Faith", granularity="feature",
        saes=[{"mistudio_sae_id": "sae_l13", "layer": 13}],
        members=kw.pop("members", [{"layer": 13, "member_kind": "feature_ref",
                                    "feature": {"feature_idx": 1}}]),
        edges=[], budget=None, faithfulness=None, rung=0, promoted=False,
        discovery_run_id=kw.pop("discovery_run_id", "dsc1"),
        created_at=datetime(2026, 7, 20), updated_at=datetime(2026, 7, 20), **kw)


class TestFaithfulnessAPI:
    def test_missing_circuit_404(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=None)):
            r = client.post("/api/v1/circuits/crc_x/faithfulness", json={})
        assert r.status_code == 404

    def test_no_members_409(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(members=[]))):
            r = client.post("/api/v1/circuits/crc_f1/faithfulness", json={})
        assert r.status_code == 409

    def test_no_discovery_run_409(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(discovery_run_id=None))):
            r = client.post("/api/v1/circuits/crc_f1/faithfulness", json={})
        assert r.status_code == 409

    def test_bad_mode_422(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit())):
            r = client.post("/api/v1/circuits/crc_f1/faithfulness",
                            json={"mode": "bogus"})
        assert r.status_code == 422

    def test_dispatch_202(self, client):
        # The happy path writes faithfulness_task_id via db.execute + commit
        # (R2 B-5) — override get_db with an async-mock session so the write
        # succeeds without a live DB (the endpoint's other db ops are mocked).
        from src.core.database import get_db

        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()

        async def _override():
            yield session

        task = MagicMock(id="task_f1")
        app.dependency_overrides[get_db] = _override
        try:
            with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                       new=AsyncMock(return_value=_circuit())), \
                 patch("src.api.v1.endpoints.circuit_discovery._run_sync",
                       new=AsyncMock(return_value=None)), \
                 patch("src.workers.circuit_validation_tasks."
                       "run_circuit_faithfulness.delay", return_value=task):
                r = client.post("/api/v1/circuits/crc_f1/faithfulness",
                                json={"mode": "both"})
        finally:
            app.dependency_overrides.pop(get_db, None)
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["task_id"] == "task_f1"
        assert body["circuit_id"] == "crc_f1"


class TestFaithfulnessLifecycle:
    """R2 B-5: faithfulness has an in-flight marker so it can't double-run and
    the GPU guard sees it."""

    def test_faithfulness_already_in_flight_409(self, client):
        from unittest.mock import AsyncMock, patch

        from src.models.circuit import Circuit
        c = Circuit(id="crc_f", name="F", granularity="feature",
                    saes=[{"mistudio_sae_id": "s", "layer": 13}],
                    members=[{"layer": 13, "member_kind": "feature_ref",
                              "feature": {"feature_idx": 1, "strength": 0.5}}],
                    edges=[], rung=1, discovery_run_id="dsc1",
                    faithfulness_status="running")
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=c)):
            r = client.post("/api/v1/circuits/crc_f/faithfulness", json={})
        assert r.status_code == 409

    def test_faithfulness_no_members_409(self, client):
        from unittest.mock import AsyncMock, patch

        from src.models.circuit import Circuit
        c = Circuit(id="crc_g", name="G", granularity="feature", saes=[],
                    members=[], edges=[], rung=0)
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=c)):
            r = client.post("/api/v1/circuits/crc_g/faithfulness", json={})
        assert r.status_code == 409
