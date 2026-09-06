"""API pins for /circuits (018 Task 5.1): rung_language on every response,
422 surfaces for contract violations, promote badge-not-gate, slice export."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.circuit import Circuit


def _circuit(**kw):
    from datetime import datetime
    c = Circuit(
        id="crc_test123", name="Test", granularity="feature",
        saes=[{"mistudio_sae_id": "sae_l13", "layer": 13, "n_features": 8192}],
        members=[{"layer": 13, "member_kind": "feature_ref",
                  "feature": {"feature_idx": 1, "strength": 0.5}}],
        edges=[], budget=None, faithfulness=None, rung=kw.pop("rung", 0),
        promoted=kw.pop("promoted", False),
        created_at=datetime(2026, 7, 19), updated_at=datetime(2026, 7, 19), **kw,
    )
    return c


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestRungLanguage:
    def test_every_response_carries_server_rendered_language(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(rung=1))):
            r = client.get("/api/v1/circuits/crc_test123")
        body = r.json()
        assert body["rung"] == 1
        assert body["rung_language"] == "suggested (attribution-supported)"
        assert "causal" not in body["rung_language"]
        assert body["rung_next_step"]

    def test_rung2_language_is_causal(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(rung=2))):
            r = client.get("/api/v1/circuits/crc_test123")
        assert r.json()["rung_language"] == "causally validated (edge)"


class TestContractSurfacing:
    def test_create_violation_is_422(self, client):
        bad = {
            "name": "x",
            "saes": [{"mistudio_sae_id": "s", "layer": 13}],
            "members": [{"layer": 13, "feature": {"feature_idx": i, "strength": 0.1}}
                        for i in range(21)],
        }
        r = client.post("/api/v1/circuits", json=bad)
        assert r.status_code == 422
        assert "per-layer member cap" in r.json()["detail"]

    def test_promote_never_gated_on_rung(self, client):
        c = _circuit(rung=0)
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=c)), \
             patch("src.api.v1.endpoints.circuits.CircuitService.set_promoted",
                   new=AsyncMock(return_value=_circuit(rung=0, promoted=True))):
            r = client.post("/api/v1/circuits/crc_test123/promote")
        assert r.status_code == 200 and r.json()["promoted"] is True

    def test_slices_parent_rung_is_single_source(self, client):
        """R1 fix #4: response rung and slice markers BOTH come from the
        recomputed definition (edge-less circuit => rung 0), never the stored
        column — no contradictory evidence claims in one payload."""
        c = _circuit(rung=1)  # stored column deliberately lies
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=c)):
            r = client.post("/api/v1/circuits/crc_test123/export-slices")
        body = r.json()
        assert body["parent_rung"] == 0  # recomputed, not the stored 1
        assert "causal" not in body["parent_rung_language"]
        assert body["slices"][0]["kind"] == "mistudio.cluster-definition"
        assert "partial_rendering=true" in body["slices"][0]["provenance"]["source_note"]
        assert "parent_rung=0" in body["slices"][0]["provenance"]["source_note"]

    def test_missing_circuit_404(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=None)):
            r = client.get("/api/v1/circuits/crc_nope")
        assert r.status_code == 404


class TestR1Fixes:
    def test_unpromote_supported(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(promoted=True))), \
             patch("src.api.v1.endpoints.circuits.CircuitService.set_promoted",
                   new=AsyncMock(return_value=_circuit(promoted=False))) as sp:
            r = client.post("/api/v1/circuits/crc_test123/promote",
                            json={"promoted": False})
        assert r.status_code == 200 and r.json()["promoted"] is False
        assert sp.await_args.args[2] is False

    def test_import_rejects_unknown_kind(self, client):
        r = client.post("/api/v1/circuits/import",
                        json={"kind": "mistudio.cluster-definition"})
        assert r.status_code == 422
        assert "Unknown kind" in r.json()["detail"]

    def test_list_returns_slim_summaries(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.list",
                   new=AsyncMock(return_value=[_circuit()])):
            r = client.get("/api/v1/circuits?limit=10")
        body = r.json()
        row = body["circuits"][0]
        assert "members" not in row and "edges" not in row  # slim rows
        assert row["member_count"] == 1 and row["layers"] == [13]
        assert body["limit"] == 10 and body["offset"] == 0

    def test_granularity_param_validated(self, client):
        r = client.get("/api/v1/circuits?granularity=features")
        assert r.status_code == 422  # Literal — typos never silently empty

    def test_export_filename_ascii_only(self, client):
        c = _circuit()
        c.name = "Précis ↔ circuit"
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=c)):
            r = client.get("/api/v1/circuits/crc_test123/export")
        cd = r.headers["content-disposition"]
        assert cd.encode("latin-1")  # header must be encodable
        assert "pr-cis" in cd or "pr" in cd

    def test_patch_null_structural_field_rejected_cleanly(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit())):
            r = client.patch("/api/v1/circuits/crc_test123", json={"members": None})
        assert r.status_code == 422
        assert "cannot be null" in r.json()["detail"]


@pytest.fixture
def wired_client(async_engine):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from src.core.database import get_db

    maker = async_sessionmaker(async_engine, class_=AsyncSession,
                               expire_on_commit=False)

    async def _override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db] = _override
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.pop(get_db, None)


class TestRealWiring:
    """R1 TE-1: exercise the REAL service+DB path (no CircuitService mocks)."""

    def test_full_lifecycle_through_real_stack(self, wired_client):
        payload = {
            "name": "Wired circuit",
            "saes": [{"mistudio_sae_id": "sae_l13", "layer": 13, "n_features": 8192},
                     {"mistudio_sae_id": "sae_l14", "layer": 14, "n_features": 8192}],
            "members": [
                {"layer": 13, "feature": {"feature_idx": 1, "strength": 0.5}},
                {"layer": 14, "feature": {"feature_idx": 2, "strength": 0.4}},
            ],
            "edges": [{"up": {"layer": 13, "feature_idx": 1},
                       "down": {"layer": 14, "feature_idx": 2}, "rung": 1}],
            "discovery": {"mode": "seeded", "granularity": "feature"},
            "model_id": "m_lfm25",
        }
        created = wired_client.post("/api/v1/circuits", json=payload)
        assert created.status_code == 201, created.text
        cid = created.json()["id"]
        assert created.json()["discovery"]["mode"] == "seeded"

        listed = wired_client.get("/api/v1/circuits")
        assert any(row["id"] == cid for row in listed.json()["circuits"])

        patched = wired_client.patch(f"/api/v1/circuits/{cid}",
                                     json={"narrative": "updated **markdown**"})
        assert patched.status_code == 200

        promoted = wired_client.post(f"/api/v1/circuits/{cid}/promote")
        assert promoted.json()["promoted"] is True

        slices = wired_client.post(f"/api/v1/circuits/{cid}/export-slices")
        assert slices.status_code == 200
        assert len(slices.json()["slices"]) == 2

        exported = wired_client.get(f"/api/v1/circuits/{cid}/export")
        assert exported.json()["discovery"]["mode"] == "seeded"  # lossless

        # Lossless round-trip (R2 B1): model ref, granularity, created_at.
        assert exported.json()["model"]["mistudio_model_id"] == "m_lfm25"
        reimported = wired_client.post("/api/v1/circuits/import",
                                       json=exported.json())
        assert reimported.status_code == 201, reimported.text
        assert reimported.json()["edges"][0]["rung"] == 1
        assert reimported.json()["model_id"] == "m_lfm25"
        assert reimported.json()["granularity"] == "feature"
        assert reimported.json()["created_at"] == created.json()["created_at"]

        deleted = wired_client.delete(f"/api/v1/circuits/{cid}")
        assert deleted.json()["deleted"] == cid


def _minimal_definition(**overrides):
    """A minimal valid mistudio.circuit-definition/v1 document."""
    doc = {
        "kind": "mistudio.circuit-definition",
        "schema_version": "1",
        "name": "Foreign circuit",
        "saes": [{"mistudio_sae_id": "sae_l13", "layer": 13, "n_features": 8192}],
        "members": [{"layer": 13, "member_kind": "feature_ref",
                     "feature": {"feature_idx": 1, "strength": 0.5}}],
        "edges": [],
    }
    doc.update(overrides)
    return doc


class TestR3Fixes:
    """Pins for the R3 fix wave + the four unpinned R2 fixes (R3-B3)."""

    def test_tz_aware_created_at_imports_cleanly(self, wired_client):
        # R3-B1: "…Z" is the normal foreign-export form; used to 500 in asyncpg.
        doc = _minimal_definition(
            provenance={"created_at": "2026-07-19T10:00:00Z"})
        r = wired_client.post("/api/v1/circuits/import", json=doc)
        assert r.status_code == 201, r.text
        # Stored naive-UTC: same instant, no offset suffix.
        assert r.json()["created_at"].startswith("2026-07-19T10:00:00")
        assert "+" not in r.json()["created_at"]

    def test_hf_id_survives_import_and_reexport(self, wired_client):
        # R3-B2: hf_id is the cross-instance-stable identifier — a foreign
        # file typically has hf_id set and mistudio_model_id null.
        doc = _minimal_definition(
            model={"hf_id": "LiquidAI/LFM2-2.6B", "mistudio_model_id": None})
        r = wired_client.post("/api/v1/circuits/import", json=doc)
        assert r.status_code == 201, r.text
        cid = r.json()["id"]
        assert r.json()["model_hf_id"] == "LiquidAI/LFM2-2.6B"
        exported = wired_client.get(f"/api/v1/circuits/{cid}/export")
        assert exported.json()["model"]["hf_id"] == "LiquidAI/LFM2-2.6B"
        wired_client.delete(f"/api/v1/circuits/{cid}")

    def test_cluster_ref_member_derives_cluster_granularity(self, wired_client):
        # R2 B7 derivation branch, previously unpinned (default masked it).
        doc = _minimal_definition(members=[
            {"layer": 13, "member_kind": "cluster_ref",
             "cluster_profile_id": "clp_x",
             "expanded_members": [{"feature_idx": 1, "strength": 0.5}]},
        ])
        r = wired_client.post("/api/v1/circuits/import", json=doc)
        assert r.status_code == 201, r.text
        assert r.json()["granularity"] == "cluster"
        wired_client.delete(f"/api/v1/circuits/{r.json()['id']}")

    def test_patch_granularity_persists(self, wired_client):
        # R2 B6 fix, previously unpinned.
        created = wired_client.post("/api/v1/circuits", json={
            "name": "Gran", "saes": _minimal_definition()["saes"],
            "members": _minimal_definition()["members"]})
        cid = created.json()["id"]
        patched = wired_client.patch(f"/api/v1/circuits/{cid}",
                                     json={"granularity": "cluster"})
        assert patched.json()["granularity"] == "cluster"
        assert wired_client.get(
            f"/api/v1/circuits/{cid}").json()["granularity"] == "cluster"
        wired_client.delete(f"/api/v1/circuits/{cid}")

    def test_import_413_over_cap(self, client):
        r = client.post("/api/v1/circuits/import", json={"kind": "x"},
                        headers={"Content-Length": str(2_000_000)})
        assert r.status_code == 413

    def test_import_malformed_content_length_is_400(self):
        # R3-B4: int("abc") used to 500. ASGITransport is async-only.
        import asyncio
        import httpx

        async def _run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://test") as c:
                return await c.post(
                    "/api/v1/circuits/import", content=b"{}",
                    headers={"Content-Length": "abc",
                             "Content-Type": "application/json"})

        r = asyncio.run(_run())
        assert r.status_code == 400

    def test_import_422_names_the_failing_field(self, client):
        doc = _minimal_definition()
        doc["members"] = [{"layer": 13, "member_kind": "feature_ref"}]  # no feature
        r = client.post("/api/v1/circuits/import", json=doc)
        assert r.status_code == 422
        assert "first at '" in r.json()["detail"]


class TestOptimisticConcurrencyAPI:
    """017 Task 3.0: PATCH threads expected_version → 409 on stale."""

    def test_patch_stale_version_409(self, wired_client):
        created = wired_client.post("/api/v1/circuits", json={
            "name": "Concur", "saes": _minimal_definition()["saes"],
            "members": _minimal_definition()["members"]})
        cid = created.json()["id"]
        assert created.json()["version"] == 1
        # first edit → version 2
        wired_client.patch(f"/api/v1/circuits/{cid}", json={"narrative": "a"})
        # stale writer still holding version 1
        r = wired_client.patch(f"/api/v1/circuits/{cid}",
                               json={"narrative": "b", "expected_version": 1})
        assert r.status_code == 409
        wired_client.delete(f"/api/v1/circuits/{cid}")

    def test_patch_matching_version_ok(self, wired_client):
        created = wired_client.post("/api/v1/circuits", json={
            "name": "Concur2", "saes": _minimal_definition()["saes"],
            "members": _minimal_definition()["members"]})
        cid = created.json()["id"]
        r = wired_client.patch(f"/api/v1/circuits/{cid}",
                               json={"narrative": "ok", "expected_version": 1})
        assert r.status_code == 200
        assert r.json()["version"] == 2
        wired_client.delete(f"/api/v1/circuits/{cid}")
