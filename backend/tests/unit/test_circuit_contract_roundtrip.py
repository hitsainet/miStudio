"""018 FTASKS 6.1, the non-GPU halves: the shipped schemas accept what the routes
emit, and a definition and its slices re-import to the same document.

Before 2026-10-20 the slices were checked only against the Pydantic model that
produces them (`ClusterDefinitionV1(...)` in `to_layer_slice`) — the producer
grading itself. Consumers (miLLM) validate against the PUBLISHED JSON Schema in
`docs/schemas/`, so that is what these tests read. And the existing round-trip
test compared three fields of a re-imported circuit; here the WHOLE exported
document is compared, minus only what an import is supposed to change.

The circuit is built through `POST /circuit-discovery/{id}/build-circuit` from a
run whose candidates carry every evidence block the builder copies (coactivation,
attribution, effect size, manifest ref) and whose edges the classifier typed, so
the documents are the full-evidence shape a real run produces, not a minimal one.

It also found a defect: importing a slice as a cluster profile dropped its
`provenance.source_note`, the "parent_rung=N; partial_rendering=true — a slice is
NOT the circuit" marker, so the profile re-exported without saying it was a partial
rendering. `ClusterProfileService` now keeps an imported note and re-emits it.

NEGATIVE CONTROLS (2026-10-20, restored byte-identically from saved bytes):
  R1 profile export: source_note=None                                -> 1 red
  R2 profile import: stop storing source_note in imported_from       -> 1 red
  R3 circuit import route: store narrative=None                      -> 1 red
  R4 to_layer_slice: source_note=None (no partial-rendering marker)  -> 1 red
  R5 published cluster-definition-v1.json: kind const changed         -> 2 red
  R6 published circuit-definition-v1.json: kind const changed         -> 1 red
"""

import copy
import datetime as dt
import json
from pathlib import Path

import jsonschema
import pytest
import torch
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from src.models.external_sae import ExternalSAE
from src.models.extraction_job import ExtractionJob
from src.models.feature import Feature

SCHEMAS = Path(__file__).resolve().parents[3] / "docs" / "schemas"
T0 = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)


def _schema(name):
    return json.loads((SCHEMAS / name).read_text())


@pytest.fixture
async def db(async_engine):
    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session


@pytest.fixture
def client(async_engine):
    from fastapi.testclient import TestClient

    from src.core.database import get_db
    from src.main import app

    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def _override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db] = _override
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def _weights(monkeypatch):
    dec = torch.eye(4, 8)          # [d_model, d_sae]
    enc = torch.eye(8, 4)          # [d_sae, d_model]

    def fake(path, *, d_model=None, n_features=None, architecture=None):
        return dec, enc

    monkeypatch.setattr("src.services.steering_service.load_sae_weights_cpu", fake)


async def _built_circuit_id(db, client, tmp_path) -> str:
    for layer in (12, 13):
        path = tmp_path / f"sae_{layer}"
        path.mkdir()
        db.add(ExternalSAE(id=f"sae_rt{layer}", name=f"L{layer}", source="huggingface",
                           status="ready", local_path=str(path), layer=layer,
                           d_model=4, n_features=8))
        db.add(ExtractionJob(id=f"extr_rt{layer}", external_sae_id=f"sae_rt{layer}",
                             config={}, created_at=T0))
    await db.commit()
    for layer, idx, name in ((12, 1, "fear_expression"), (13, 1, "fear_expression"),
                             (12, 2, "death_mention"), (13, 3, "condolence")):
        db.add(Feature(id=f"feat_rt{layer}_{idx}", name=name, label_source="local_llm",
                       neuron_index=idx, extraction_job_id=f"extr_rt{layer}",
                       external_sae_id=f"sae_rt{layer}", activation_frequency=0.1,
                       max_activation=2.0, interpretability_score=0.5,
                       example_tokens_summary="▁fear, ▁afraid" if idx == 1 else name))
    db.add(CircuitCaptureRun(
        id="cap_rt", status="completed", store_path="/x",
        manifest={"model_id": "m_rt", "model_dtype": "bfloat16", "layers": [
            {"layer": 12, "sae_id": "sae_rt12"}, {"layer": 13, "sae_id": "sae_rt13"}]}))
    db.add(CircuitDiscoveryRun(
        id="dsc_rt", capture_run_id="cap_rt", status="completed",
        params={"granularity": "feature", "mode": "seeded"},
        candidates=[
            {"up": {"layer": 12, "feature_idx": 1}, "down": {"layer": 13, "feature_idx": 1},
             "stats": {"pmi": 2.5, "lift": 9.0, "support": 120, "null_pct": 99.5,
                       "spearman": 0.7},
             "replicated_heldout": True,
             "attribution": {"score": 0.41, "method": "raw", "rung1_gate": True,
                             "sign_consistency": 0.9}},
            {"up": {"layer": 12, "feature_idx": 2}, "down": {"layer": 13, "feature_idx": 3},
             "stats": {"pmi": 1.9, "lift": 4.0, "support": 80, "null_pct": 98.0,
                       "spearman": 0.5},
             "replicated_heldout": True,
             "attribution": {"score": 0.3, "method": "raw", "rung1_gate": True,
                             "sign_consistency": 0.8},
             "validated_rung": 2,
             "validation": {"effect_size": 0.62, "manifest_id": "vman_rt"}},
        ]))
    await db.commit()
    built = client.post("/api/v1/circuit-discovery/dsc_rt/build-circuit",
                        json={"name": "Round-trip circuit", "narrative": "**why** it matters"})
    assert built.status_code == 201, built.text
    return built.json()["id"]


def _without(doc, *paths):
    doc = copy.deepcopy(doc)
    for path in paths:
        node = doc
        *head, last = path.split(".")
        for k in head:
            node = node.get(k) or {}
        node.pop(last, None)
    return doc


class TestTheShippedSchemasAcceptTheRoutesOutput:
    @pytest.mark.asyncio
    async def test_export_validates_against_circuit_definition_v1(self, db, client, tmp_path):
        cid = await _built_circuit_id(db, client, tmp_path)
        doc = client.get(f"/api/v1/circuits/{cid}/export").json()
        # the document is the full-evidence shape, not a minimal one
        assert {e["type"] for e in doc["edges"]} == {"persistence", "computed"}
        assert any(e.get("effect_size") == 0.62 for e in doc["edges"])
        jsonschema.validate(doc, _schema("circuit-definition-v1.json"))

    @pytest.mark.asyncio
    async def test_every_slice_validates_against_cluster_definition_v1(self, db, client, tmp_path):
        cid = await _built_circuit_id(db, client, tmp_path)
        r = client.post(f"/api/v1/circuits/{cid}/export-slices")
        assert r.status_code == 200, r.text
        slices = r.json()["slices"]
        assert [s["sae"]["layer"] for s in slices] == [12, 13]
        schema = _schema("cluster-definition-v1.json")
        for s in slices:
            jsonschema.validate(s, schema)
            assert s["kind"] == "mistudio.cluster-definition"


class TestReImportEquality:
    @pytest.mark.asyncio
    async def test_definition_export_import_export_is_identity(self, db, client, tmp_path):
        cid = await _built_circuit_id(db, client, tmp_path)
        first = client.get(f"/api/v1/circuits/{cid}/export").json()

        imported = client.post("/api/v1/circuits/import", json=first)
        assert imported.status_code == 201, imported.text
        second = client.get(f"/api/v1/circuits/{imported.json()['id']}/export").json()

        # Only the export stamp may differ. Everything else — every edge's type,
        # signals, statistics, attribution, effect size and manifest ref, the
        # members, SAEs, model, discovery block — must come back unchanged.
        volatile = ("provenance.exported_at",)
        assert _without(second, *volatile) == _without(first, *volatile)

    @pytest.mark.asyncio
    async def test_slices_import_as_cluster_profiles_and_export_unchanged(self, db, client, tmp_path):
        cid = await _built_circuit_id(db, client, tmp_path)
        slices = client.post(f"/api/v1/circuits/{cid}/export-slices").json()["slices"]

        for s in slices:
            r = client.post("/api/v1/cluster-profiles/import", json={"payload": s})
            assert r.status_code == 200, r.text
            result = r.json()["results"][0]
            assert result["status"] == "imported", result
            again = client.get(
                f"/api/v1/cluster-profiles/{result['profile_id']}/export").json()

            # The cluster export omits nulls (exclude_none) where the slice route
            # keeps them, so compare with nulls dropped on both sides.
            def strip_none(x):
                if isinstance(x, dict):
                    return {k: strip_none(v) for k, v in x.items() if v is not None}
                if isinstance(x, list):
                    return [strip_none(v) for v in x]
                return x

            for field in ("kind", "schema_version", "name", "narrative", "budget"):
                assert strip_none(again.get(field)) == strip_none(s.get(field)), field
            # The SAE ref binds to the local SAE: what the slice named is kept, and the
            # bound row's dims (d_model, n_features) may be filled in.
            sae_sent = strip_none(s["sae"])
            assert {k: again["sae"].get(k) for k in sae_sent} == sae_sent
            # Members: every field the slice carried comes back unchanged, in order.
            # The profile export may ADD display-only enrichment (label, meta) from
            # the bound SAE's Feature rows (bind-at-load, cluster-member-meta rev);
            # it may not change or drop anything the slice said.
            sent = strip_none(s["members"])
            got = strip_none(again["members"])
            assert [m["feature_idx"] for m in got] == [m["feature_idx"] for m in sent]
            for m_sent, m_got in zip(sent, got, strict=True):
                assert {k: m_got.get(k) for k in m_sent} == m_sent
            # the partial-rendering marker survives the round trip
            assert "partial_rendering=true" in again["provenance"]["source_note"]
            jsonschema.validate(again, _schema("cluster-definition-v1.json"))
