"""A circuit built from a discovery run has its edges TYPED, and a cluster run builds.

Two defects in `CircuitService.from_candidates`, found 2026-10-20 while preparing
018 FTASKS 6.1 ("classifier over a real discovery run"):

1. THE CLASSIFIER HAD NO CALLER. `circuit_edge_type_service.classify_edge` was
   implemented, audit-gated and unit-tested, and nothing in production called it.
   Every built edge took the schema default `type: "computed"` with
   `type_signals: null`, so the EdgeTable's type column, the hidden-by-default
   persistence policy and the list's edge-type filter all read a value no
   classifier produced. The live step could never have passed.

2. A CLUSTER-GRANULARITY RUN COULD NOT BE BUILT. Its candidates' endpoints are
   supernodes with no `feature_idx`; the loop skipped them as members and wrote each
   edge endpoint as a feature ref with `feature_idx=None`, which the contract rejects.
   Every "Build circuit" on a cluster run was a 422.

These tests drive the real service (and, for the wiring, the real REST route)
against the test database. Only the SAE weight loader is replaced, with tensors whose
cosines are known.

NEGATIVE CONTROLS (2026-10-20, each restored byte-identically from saved bytes):
  C1 disable the `await CircuitService._classify_edges(...)` call   -> 4 red
  C2 type_circuit_edges: drop the `edge["type"] = ...` line         -> 3 red
  C3 _edge_weight_priors measures no pair                           -> 1 red
  C4 _edge_endpoint always writes a feature ref                     -> 1 red
"""

import datetime as dt

import pytest
import torch
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from src.models.cluster_profile import ClusterProfile
from src.models.external_sae import ExternalSAE
from src.models.extraction_job import ExtractionJob
from src.models.feature import Feature
from src.services.circuit_service import CircuitService

D_MODEL, D_SAE = 4, 8
T0 = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)


@pytest.fixture
async def db(async_engine):
    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session


def _weights():
    """Decoder [d_model, d_sae] and encoder [d_sae, d_model] with known cosines.

    up feature 1's decoder column == down feature 2's encoder row  -> prior 1.0
    up feature 3's decoder column ⟂ down feature 5's encoder row   -> prior 0.0
    """
    dec = torch.zeros(D_MODEL, D_SAE)
    enc = torch.zeros(D_SAE, D_MODEL)
    dec[:, 1] = torch.tensor([1.0, 2.0, 0.0, 0.0])
    enc[2, :] = torch.tensor([1.0, 2.0, 0.0, 0.0])
    dec[:, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    enc[5, :] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    return dec, enc


@pytest.fixture
def loader(monkeypatch):
    """Replace the CPU weight loader; record every call (payload + count)."""
    calls = []

    def fake(path, *, d_model=None, n_features=None, architecture=None):
        calls.append({"path": str(path), "d_model": d_model, "n_features": n_features})
        return _weights()

    monkeypatch.setattr("src.services.steering_service.load_sae_weights_cpu", fake)
    return calls


async def _seed_saes_and_features(db, tmp_path, *, with_files=True):
    for layer in (12, 13):
        path = tmp_path / f"sae_l{layer}"
        if with_files:
            path.mkdir()
        db.add(ExternalSAE(id=f"sae_l{layer}", name=f"L{layer}", source="huggingface",
                           status="ready", local_path=str(path), layer=layer,
                           d_model=D_MODEL, n_features=D_SAE))
        db.add(ExtractionJob(id=f"extr_l{layer}", external_sae_id=f"sae_l{layer}",
                             config={}, created_at=T0))
    await db.commit()

    def feat(layer, idx, name, source, tokens):
        return Feature(
            id=f"feat_l{layer}_{idx}", name=name, label_source=source,
            neuron_index=idx, extraction_job_id=f"extr_l{layer}",
            external_sae_id=f"sae_l{layer}", activation_frequency=0.1,
            max_activation=2.0, interpretability_score=0.5,
            example_tokens_summary=tokens)

    db.add_all([
        # an echo: same concept, same tokens (prior 1.0 from the weights)
        feat(12, 1, "fear_expression", "local_llm", "▁fear, ▁afraid, ▁scared"),
        feat(13, 2, "fear_expression", "local_llm", "▁fear, ▁afraid, ▁scared"),
        # computed: different concept, different tokens, orthogonal weights
        feat(12, 3, "death_mention", "local_llm", "▁died, ▁passed"),
        feat(13, 5, "feature_00005", "auto", "▁condolences, ▁sorry"),
    ])
    await db.commit()


def _run(run_id="dsc_typed", cap_id="cap_typed"):
    cap = CircuitCaptureRun(
        id=cap_id, status="completed", store_path="/x",
        manifest={"model_id": "m", "layers": [
            {"layer": 12, "sae_id": "sae_l12"}, {"layer": 13, "sae_id": "sae_l13"}]})
    run = CircuitDiscoveryRun(
        id=run_id, capture_run_id=cap_id, status="completed",
        params={"granularity": "feature", "mode": "seeded"},
        candidates=[
            {"up": {"layer": 12, "feature_idx": 1}, "down": {"layer": 13, "feature_idx": 2},
             "stats": {"pmi": 3.0, "support": 40}},
            {"up": {"layer": 12, "feature_idx": 3}, "down": {"layer": 13, "feature_idx": 5},
             "stats": {"pmi": 2.0, "support": 30}},
        ])
    return cap, run


class TestTheBuildTypesEveryEdge:
    @pytest.mark.asyncio
    async def test_echo_is_persistence_and_the_rest_is_computed(self, db, tmp_path, loader):
        await _seed_saes_and_features(db, tmp_path)
        cap, run = _run()
        db.add_all([cap, run])
        await db.commit()

        circuit = await CircuitService.from_candidates(
            db, discovery_run_id="dsc_typed", name="Typed", candidate_keys=[])
        by_up = {e["up"]["feature_idx"]: e for e in circuit.edges}

        echo, computed = by_up[1], by_up[3]
        assert echo["type"] == "persistence"
        assert echo["type_signals"]["votes"] == {
            "weight_prior": True, "token_overlap": True, "label_similarity": True}
        assert echo["weight_prior"] == pytest.approx(1.0)
        assert echo["type_signals"]["token_overlap"] == pytest.approx(1.0)
        assert echo["type_signals"]["distinctness"] == pytest.approx(0.0)

        assert computed["type"] == "computed"
        assert computed["weight_prior"] == pytest.approx(0.0)
        # the AUTO placeholder name is not a concept: label signal is absent
        assert computed["type_signals"]["label_method"] == "absent"
        assert computed["type_signals"]["distinctness"] == pytest.approx(1.0)

        # Each layer's SAE loaded exactly once, with its own path and dims.
        assert sorted(c["path"] for c in loader) == sorted(
            [str(tmp_path / "sae_l12"), str(tmp_path / "sae_l13")])
        assert all(c["d_model"] == D_MODEL and c["n_features"] == D_SAE for c in loader)

    @pytest.mark.asyncio
    async def test_an_unmeasured_prior_is_disclosed_as_none_not_zero(self, db, tmp_path, loader):
        """No SAE files on disk: labels and tokens still classify; the prior says
        nobody measured it, rather than claiming a measured 0.0."""
        await _seed_saes_and_features(db, tmp_path, with_files=False)
        cap, run = _run("dsc_noprior", "cap_noprior")
        db.add_all([cap, run])
        await db.commit()

        circuit = await CircuitService.from_candidates(
            db, discovery_run_id="dsc_noprior", name="No prior", candidate_keys=[])
        echo = next(e for e in circuit.edges if e["up"]["feature_idx"] == 1)
        assert loader == []
        assert echo["type_signals"]["weight_prior"] is None
        assert echo["type_signals"]["votes"]["weight_prior"] is False
        assert "weight_prior" not in echo or echo["weight_prior"] is None
        # token overlap + label similarity are still 2 of 3
        assert echo["type"] == "persistence"

    @pytest.mark.asyncio
    async def test_the_build_ROUTE_returns_typed_edges(self, db, wired_client, tmp_path, loader):
        """Wiring at the surface operators and agents use: POST build-circuit, then
        the exported definition carries the classifier's types and signals."""
        await _seed_saes_and_features(db, tmp_path)
        cap, run = _run("dsc_route", "cap_route")
        db.add_all([cap, run])
        await db.commit()
        client = wired_client

        built = client.post("/api/v1/circuit-discovery/dsc_route/build-circuit",
                            json={"name": "Route-built"})
        assert built.status_code == 201, built.text
        exported = client.get(f"/api/v1/circuits/{built.json()['id']}/export").json()
        types = {e["up"]["feature_idx"]: e["type"] for e in exported["edges"]}
        assert types == {1: "persistence", 3: "computed"}
        assert all(e["type_signals"]["thresholds"]["p_hi"] == 0.9 for e in exported["edges"])


class TestUnmeasuredTokensThroughTheBuild:
    @pytest.mark.asyncio
    async def test_features_without_top_tokens_record_none_not_zero(self, db, tmp_path, loader):
        """The live shape (tracker 3.4 A1, 2026-10-20): unlabelled endpoints with
        no `example_tokens_summary`. The built edge must say the token signal was
        not measured."""
        await _seed_saes_and_features(db, tmp_path)
        from sqlalchemy import update
        await db.execute(update(Feature).values(example_tokens_summary=None))
        await db.commit()
        cap, run = _run("dsc_notokens", "cap_notokens")
        db.add_all([cap, run])
        await db.commit()

        circuit = await CircuitService.from_candidates(
            db, discovery_run_id="dsc_notokens", name="No tokens", candidate_keys=[])
        for edge in circuit.edges:
            assert edge["type_signals"]["token_overlap"] is None, edge["type_signals"]
            assert edge["type_signals"]["token_method"] == "absent"
        echo = next(e for e in circuit.edges if e["up"]["feature_idx"] == 1)
        assert echo["type"] == "persistence"  # prior + labels: still 2 of 3


class TestAClusterRunBuilds:
    @pytest.mark.asyncio
    async def test_cluster_endpoints_become_cluster_members_and_edges(self, db, loader):
        db.add(ExternalSAE(id="sae_c12", name="c12", source="huggingface", status="ready",
                           layer=12, d_model=D_MODEL, n_features=D_SAE))
        db.add(ExternalSAE(id="sae_c13", name="c13", source="huggingface", status="ready",
                           layer=13, d_model=D_MODEL, n_features=D_SAE))
        await db.commit()
        db.add(ClusterProfile(id="clp_up", sae_id="sae_c12", name="Up cluster",
                              members=[{"feature_idx": 4, "strength": 1.0},
                                       {"feature_idx": 6, "strength": 0.5}]))
        db.add(ClusterProfile(id="clp_down", sae_id="sae_c13", name="Down cluster",
                              members=[{"feature_idx": 7, "strength": 1.0}]))
        db.add(CircuitCaptureRun(
            id="cap_cl", status="completed", store_path="/x",
            manifest={"model_id": "m", "layers": [
                {"layer": 12, "sae_id": "sae_c12"}, {"layer": 13, "sae_id": "sae_c13"}]}))
        db.add(CircuitDiscoveryRun(
            id="dsc_cl", capture_run_id="cap_cl", status="completed",
            params={"granularity": "cluster", "mode": "seeded"},
            candidates=[{
                "up": {"layer": 12, "cluster_profile_id": "clp_up", "cluster_name": "Up cluster"},
                "down": {"layer": 13, "cluster_profile_id": "clp_down",
                         "cluster_name": "Down cluster"},
                "stats": {"pmi": 2.95, "lift": 19.1, "support": 500, "spearman": 0.84},
                "replicated_heldout": True}]))
        await db.commit()

        circuit = await CircuitService.from_candidates(
            db, discovery_run_id="dsc_cl", name="Cluster circuit", candidate_keys=[])

        assert circuit.granularity == "cluster"
        edge = circuit.edges[0]
        assert edge["up"] == {"layer": 12, "kind": "cluster", "cluster_profile_id": "clp_up",
                              "feature_idx": None}
        assert edge["down"]["kind"] == "cluster"
        assert edge["coactivation"]["lift"] == 19.1
        # the classifier does not type supernodes, and says so
        assert edge["type_signals"]["classified"] is False
        members = {m["cluster_profile_id"]: m for m in circuit.members}
        assert members["clp_up"]["member_kind"] == "cluster_ref"
        assert [m["feature_idx"] for m in members["clp_up"]["expanded_members"]] == [4, 6]
        # nothing to load for a cluster-only circuit
        assert loader == []

    @pytest.mark.asyncio
    async def test_a_deleted_profile_is_a_clear_refusal(self, db, loader):
        from src.services.circuit_service import CircuitValidationError

        db.add(CircuitCaptureRun(id="cap_gone", status="completed", manifest={"layers": []}))
        db.add(CircuitDiscoveryRun(
            id="dsc_gone", capture_run_id="cap_gone", status="completed",
            params={"granularity": "cluster"},
            candidates=[{"up": {"layer": 1, "cluster_profile_id": "clp_missing"},
                         "down": {"layer": 2, "cluster_profile_id": "clp_missing2"}}]))
        await db.commit()
        with pytest.raises(CircuitValidationError, match="clp_missing"):
            await CircuitService.from_candidates(
                db, discovery_run_id="dsc_gone", name="x", candidate_keys=[])


@pytest.fixture
def wired_client(async_engine):
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
