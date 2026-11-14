"""016 FPRD §8.5: a cluster drill-down's member-pair statistics agree with the supernode edge.

A cluster-granularity discovery measures an edge between two SUPERNODES, where a
supernode fires wherever any of its members fires (`A_C(t) = max_k a_k(t)`). The A.4
drill-down (`POST /circuit-discovery/refine-cluster-edge`) then measures the member
pairs of that edge. The criterion is that the two agree. Agreement has five parts,
each asserted here on one fixture capture, with the discovery run by the real
service and the drill-down reached through the real REST route:

  1. SAME FRAME. A member pair is measured over the discovery documents and token
     count the supernode edge used (the capture's own split), never the whole store.
  2. SET ALGEBRA. The supernode edge's support equals the size of the UNION of its
     member pairs' co-firings, and every member pair's support is at most it.
  3. THE CARRIER IS FOUND. When the supernode edge survives, the member pair that
     carries it survives the drill-down, replicates on held-out documents, and is
     more concentrated (higher lift) than the diluted supernode; the decoy member
     pairs do not survive.
  4. NO EDGE, NO CARRIER. When the supernode pair does not survive discovery, the
     drill-down finds no member pair either.
  5. NO RUNG. A drill-down pair is association only and inherits no rung.

The fixture: L13 cluster U = {1 (carrier), 3 (decoy)}, L14 cluster D = {2 (carrier),
4 (decoy)}, L14 cluster D2 = {5, 6} (independent of everything). Feature 2 fires at
five of feature 1's six positions in every document (160 co-firings over the 32
discovery documents); every decoy fires at 10 random positions per document.

MUTATION CONTROLS (2026-10-21, each restored by sha256): see 016 FTASKS Task 6.2.
"""

from contextlib import contextmanager

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from src.models.cluster_profile import ClusterProfile
from src.models.external_sae import ExternalSAE
from src.services.circuit_capture_store import EventReader, open_writers
from src.services.circuit_discovery_service import CircuitDiscoveryService

N_DOCS, DOC_LEN = 40, 96
HELDOUT = list(range(32, 40))
DISCOVERY_DOCS = [d for d in range(N_DOCS) if d not in HELDOUT]
PLANTED_PER_DOC = 5


@pytest.fixture
def engine(async_engine):
    sync_url = async_engine.url.set(
        drivername=async_engine.url.drivername.replace("+asyncpg", "").replace("+psycopg", ""))
    eng = create_engine(sync_url)
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine):
    session = sessionmaker(bind=engine)()
    yield session
    session.rollback()
    session.close()


def _build_store(path):
    rng = np.random.default_rng(21)
    ev13, en13, _ = open_writers(path, 13)
    ev14, en14, _ = open_writers(path, 14)
    for d in range(N_DOCS):
        carrier = rng.choice(DOC_LEN, size=6, replace=False)
        ev13.append(np.full(6, d), carrier, np.full(6, 1), rng.uniform(0.5, 2, 6))
        ev14.append(np.full(PLANTED_PER_DOC, d), carrier[:PLANTED_PER_DOC],
                    np.full(PLANTED_PER_DOC, 2), rng.uniform(0.5, 2, PLANTED_PER_DOC))
        for writer, fid in ((ev13, 3), (ev14, 4), (ev14, 5), (ev14, 6)):
            pos = rng.choice(DOC_LEN, size=10, replace=False)
            writer.append(np.full(10, d), pos, np.full(10, fid), rng.uniform(0.5, 2, 10))
        for en in (en13, en14):
            en.append(np.full(DOC_LEN, d), np.arange(DOC_LEN), rng.uniform(0, 0.2, DOC_LEN))
    for w in (ev13, ev14, en13, en14):
        w.finalize()


def _seed(db, store):
    for layer in (13, 14):
        db.add(ExternalSAE(id=f"sae_dd{layer}", name=f"L{layer}", source="local",
                           status="ready", layer=layer, d_model=4, n_features=8))
    db.commit()
    db.add_all([
        ClusterProfile(id="clp_U", sae_id="sae_dd13", name="U",
                       members=[{"feature_idx": 1, "strength": 1.0},
                                {"feature_idx": 3, "strength": 1.0}]),
        ClusterProfile(id="clp_D", sae_id="sae_dd14", name="D",
                       members=[{"feature_idx": 2, "strength": 1.0},
                                {"feature_idx": 4, "strength": 1.0}]),
        ClusterProfile(id="clp_D2", sae_id="sae_dd14", name="D2",
                       members=[{"feature_idx": 5, "strength": 1.0},
                                {"feature_idx": 6, "strength": 1.0}]),
    ])
    cap = CircuitCaptureRun(
        status="completed", store_path=str(store),
        manifest={
            "corpus": {"dataset_id": "ds_x", "tokenization_id": "tok_x", "sample_cap": N_DOCS},
            "model_id": "m_x",
            "layers": [{"layer": 13, "sae_id": "sae_dd13"}, {"layer": 14, "sae_id": "sae_dd14"}],
            "split": {"method": "per_document", "ratio": 0.8, "seed": 42, "heldout_docs": HELDOUT},
            "doc_lengths": {str(d): DOC_LEN for d in range(N_DOCS)},
            "counts": {"documents": N_DOCS, "events": 0, "tokens": N_DOCS * DOC_LEN},
        })
    db.add(cap)
    db.commit()
    db.refresh(cap)
    return cap


@pytest.fixture
def world(db, tmp_path, engine, monkeypatch):
    """The capture, the three profiles and a completed seeded CLUSTER discovery on U."""
    _build_store(tmp_path)
    cap = _seed(db, tmp_path)
    run = CircuitDiscoveryService.create_run(db, {
        "capture_run_id": cap.id, "granularity": "cluster", "mode": "seeded",
        "seed_refs": [{"layer": 13, "cluster_profile_id": "clp_U"}],
        "s_min": 20, "null_shuffles": 50})
    assert CircuitDiscoveryService.run(db, run.id)["status"] == "completed"
    db.refresh(run)

    # The route opens its own session through `get_sync_db`; point it at this database.
    maker = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)  # as SyncSessionLocal

    @contextmanager
    def _sync_db():
        session = maker()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    monkeypatch.setattr("src.core.database.get_sync_db", _sync_db)
    return {"cap": cap, "run": run, "store": tmp_path}


async def _refine(cap_id, down_profile, down_layer=14):
    from src.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test/api/v1") as c:
        r = await c.post("/circuit-discovery/refine-cluster-edge", json={
            "capture_run_id": cap_id,
            "up_cluster_profile_id": "clp_U", "up_layer": 13,
            "down_cluster_profile_id": down_profile, "down_layer": down_layer,
            "s_min": 20, "null_shuffles": 50})
    assert r.status_code == 200, r.text
    return r.json()


def _edge(run, down_profile):
    hits = [c for c in run.candidates
            if c["up"].get("cluster_profile_id") == "clp_U"
            and c["down"].get("cluster_profile_id") == down_profile]
    assert len(hits) <= 1
    return hits[0] if hits else None


def _member_keys(store, layer, fid):
    ev = EventReader(store, layer).feature_events(fid)
    keys = (ev["doc_id"].astype(np.uint64) << np.uint64(16)) | ev["token_pos"].astype(np.uint64)
    return keys[np.isin(ev["doc_id"], DISCOVERY_DOCS)]


class TestTheDrillDownAgreesWithTheSupernodeEdge:
    @pytest.mark.asyncio
    async def test_the_supernode_edge_exists_and_the_unlinked_pair_does_not(self, world):
        """Precondition for everything below, asserted rather than assumed."""
        edge = _edge(world["run"], "clp_D")
        assert edge is not None, "the planted supernode edge U→D did not survive discovery"
        assert edge["replicated_heldout"] is True
        assert _edge(world["run"], "clp_D2") is None

    @pytest.mark.asyncio
    async def test_member_pairs_are_measured_in_the_supernode_edges_frame(self, world):
        out = await _refine(world["cap"].id, "clp_D")
        carrier = out["member_pairs"][0]["stats"]
        assert carrier["n_tokens"] == len(DISCOVERY_DOCS) * DOC_LEN, (
            "the drill-down counted tokens outside the capture's discovery split")
        assert carrier["n_ud"] == PLANTED_PER_DOC * len(DISCOVERY_DOCS), (
            "the carrier's co-firings include held-out documents")

    @pytest.mark.asyncio
    async def test_supernode_support_is_the_union_of_its_member_pairs(self, world):
        store = world["store"]
        union = set()
        per_pair = {}
        for u in (1, 3):
            for d in (2, 4):
                shared = set(np.intersect1d(_member_keys(store, 13, u), _member_keys(store, 14, d)).tolist())
                per_pair[(u, d)] = len(shared)
                union |= shared
        support = _edge(world["run"], "clp_D")["stats"]["support"]
        assert support == len(union)
        assert max(per_pair.values()) <= support

        out = await _refine(world["cap"].id, "clp_D")
        for p in out["member_pairs"]:
            pair = (p["up"]["feature_idx"], p["down"]["feature_idx"])
            assert p["stats"]["n_ud"] == per_pair[pair] <= support

    @pytest.mark.asyncio
    async def test_the_carrier_pair_survives_and_the_decoys_do_not(self, world):
        out = await _refine(world["cap"].id, "clp_D")
        pairs = [(p["up"]["feature_idx"], p["down"]["feature_idx"]) for p in out["member_pairs"]]
        assert pairs == [(1, 2)], f"expected only the carrier 1→2 to survive, got {pairs}"
        carrier = out["member_pairs"][0]
        assert carrier["replicated"] is True
        assert carrier["up"]["layer"] == 13 and carrier["down"]["layer"] == 14
        edge = _edge(world["run"], "clp_D")
        assert carrier["stats"]["pmi"] > 0 and edge["stats"]["pmi"] > 0
        assert carrier["stats"]["lift"] > edge["stats"]["lift"], (
            "the carrier should be more concentrated than the supernode it is diluted into")
        assert out["members_tested"] == {"up": 2, "down": 2}
        # A decoy pair can fall below the support floor (s_min) and go untested; at most 2 x 2.
        assert 1 <= out["pairs_tested"] <= 4 and out["null_capped"] is False
        assert out["excluded_members"] == []

    @pytest.mark.asyncio
    async def test_no_supernode_edge_means_no_carrying_member_pair(self, world):
        out = await _refine(world["cap"].id, "clp_D2")
        assert out["members_tested"] == {"up": 2, "down": 2}
        assert out["pairs_tested"] > 0, "precondition: the unlinked pairs were tested"
        assert out["member_pairs"] == []

    @pytest.mark.asyncio
    async def test_a_member_pair_inherits_no_rung(self, world):
        out = await _refine(world["cap"].id, "clp_D")
        assert out["member_pairs"]
        for p in out["member_pairs"]:
            assert "rung" not in p and p["measured_on_this_pair"] is True
