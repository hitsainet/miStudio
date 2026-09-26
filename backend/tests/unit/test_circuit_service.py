"""Pins for CircuitService (018 Task 3.2): contract-backed validation, CRUD,
promotion (badge-not-gate), rung recomputation, slices export."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.services.circuit_service import CircuitService, CircuitValidationError


@pytest.fixture
async def db(async_engine):
    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session


def _payload(**overrides):
    base = dict(
        name="Support voice circuit",
        saes=[{"mistudio_sae_id": "sae_l13", "layer": 13, "n_features": 8192},
              {"mistudio_sae_id": "sae_l14", "layer": 14, "n_features": 8192}],
        members=[
            {"layer": 13, "feature": {"feature_idx": 712, "strength": 0.5, "label": "condolences"}},
            {"layer": 14, "feature": {"feature_idx": 231, "strength": 0.4, "label": "reassurance"}},
        ],
        edges=[{
            "up": {"layer": 13, "feature_idx": 712},
            "down": {"layer": 14, "feature_idx": 231},
            "rung": 1,
            "coactivation": {"pmi": 2.0, "support": 30},
        }],
        budget={"layers": {"13": {"B": 1.0}, "14": {"B": 2.4}}, "intensity": 1.0},
    )
    base.update(overrides)
    return base


class TestCRUD:
    @pytest.mark.asyncio
    async def test_create_computes_rung_and_persists(self, db):
        c = await CircuitService.create(db, _payload())
        assert c.id.startswith("crc_")
        assert c.rung == 1  # min over edges
        assert c.promoted is False

    @pytest.mark.asyncio
    async def test_create_rejects_contract_violations(self, db):
        bad = _payload(members=[{"layer": 13, "feature": {"feature_idx": i, "strength": 0.1}}
                                for i in range(21)], edges=[])
        with pytest.raises(CircuitValidationError, match="per-layer member cap"):
            await CircuitService.create(db, bad)

    @pytest.mark.asyncio
    async def test_promote_is_badge_not_gate(self, db):
        c = await CircuitService.create(db, _payload())
        assert c.rung < 2  # NOT causally validated…
        c = await CircuitService.set_promoted(db, c, True)  # …and promotion still succeeds
        assert c.promoted is True

    @pytest.mark.asyncio
    async def test_update_revalidates_and_recomputes_rung(self, db):
        c = await CircuitService.create(db, _payload())
        edges = list(c.edges)
        edges[0] = {**edges[0], "rung": 2, "effect_size": 0.8}
        c = await CircuitService.update(db, c, {"edges": edges})
        assert c.rung == 2
        with pytest.raises(CircuitValidationError):
            await CircuitService.update(db, c, {"edges": [
                {"up": {"layer": 14, "feature_idx": 231},
                 "down": {"layer": 13, "feature_idx": 712}}]})

    @pytest.mark.asyncio
    async def test_recompute_rung_from_stored_edges(self, db):
        c = await CircuitService.create(db, _payload())
        c.edges = [{**c.edges[0], "rung": 3}]
        c = await CircuitService.recompute_rung(db, c)
        assert c.rung == 3


class TestCalibration:
    """IDL-37: apply_calibration ships the band AND clamps the dial — the
    guarantee that a served dial cannot reach the nonsense zone above the cliff."""

    _BAND = {
        "onset": 0.2, "sweet_spot": 0.5, "cliff": 0.6, "provisional": True,
        "probe_set": [{"prompt": "Capital of France?", "expected": "Paris"}],
        "judge_metric_id": "correctness/v1", "step_budget": 10,
    }

    @pytest.mark.asyncio
    async def test_apply_calibration_clamps_the_dial_and_persists_the_band(self, db):
        c = await CircuitService.create(db, _payload())
        c = await CircuitService.apply_calibration(db, c, self._BAND)
        # The band is stored…
        assert c.calibration["sweet_spot"] == 0.5
        assert c.calibration["provisional"] is True
        # …AND the served dial is clamped so it cannot exceed the cliff.
        assert c.budget["intensity_range"] == [0.2, 0.6]
        assert c.budget["intensity"] == 0.5

    @pytest.mark.asyncio
    async def test_the_clamp_preserves_the_rest_of_the_budget(self, db):
        c = await CircuitService.create(db, _payload())  # budget has per-layer B
        c = await CircuitService.apply_calibration(db, c, self._BAND)
        # per-layer budgets and formula are NOT wiped — only the dial envelope moves.
        assert c.budget["layers"]["13"]["B"] == 1.0

    @pytest.mark.asyncio
    async def test_an_inverted_band_is_REFUSED_not_clamped(self, db):
        c = await CircuitService.create(db, _payload())
        with pytest.raises(CircuitValidationError):
            await CircuitService.apply_calibration(
                db, c, {**self._BAND, "onset": 0.7, "sweet_spot": 0.5, "cliff": 0.6})

    @pytest.mark.asyncio
    async def test_calibration_survives_the_export_projection(self, db):
        """The export path (to_definition) must carry calibration — it round-
        trips through _validate, which had to be threaded at that site too."""
        c = await CircuitService.create(db, _payload())
        c = await CircuitService.apply_calibration(db, c, self._BAND)
        defn = CircuitService.to_definition(c)
        assert defn.calibration is not None
        assert defn.calibration.cliff == 0.6
        assert defn.model_dump(mode="json")["calibration"]["onset"] == 0.2

    @pytest.mark.asyncio
    async def test_list_filters(self, db):
        a = await CircuitService.create(db, _payload(name="A"))
        await CircuitService.set_promoted(db, a, True)
        await CircuitService.create(db, _payload(name="B"))
        promoted = await CircuitService.list(db, promoted=True)
        assert [x.name for x in promoted] == ["A"]
        rung1 = await CircuitService.list(db, min_rung=1)
        assert {x.name for x in rung1} == {"A", "B"}


class TestExport:
    @pytest.mark.asyncio
    async def test_to_definition_and_slices(self, db):
        c = await CircuitService.create(db, _payload())
        defn = CircuitService.to_definition(c)
        assert defn.kind == "mistudio.circuit-definition"
        assert defn.provenance.exported_at is not None
        slices = CircuitService.to_slices(c)
        assert [s.sae.layer for s in slices] == [13, 14]
        assert all("partial_rendering=true" in s.provenance.source_note for s in slices)


class TestOptimisticConcurrency:
    """017 Task 3.0: version-based optimistic concurrency + the update()-only
    edge-write rule (018 R2-Q2/R2-A5)."""

    @pytest.mark.asyncio
    async def test_version_starts_at_one_and_bumps_on_update(self, db):
        c = await CircuitService.create(db, _payload())
        assert c.version == 1
        await CircuitService.update(db, c, {"narrative": "edited"})
        assert c.version == 2

    @pytest.mark.asyncio
    async def test_stale_expected_version_409s(self, db):
        from src.services.circuit_service import CircuitConcurrencyError
        c = await CircuitService.create(db, _payload())
        # a writer holding version 1 while another already bumped to 2
        await CircuitService.update(db, c, {"narrative": "first"})  # → v2
        with pytest.raises(CircuitConcurrencyError):
            await CircuitService.update(db, c, {"narrative": "stale"},
                                        expected_version=1)

    @pytest.mark.asyncio
    async def test_matching_version_succeeds(self, db):
        c = await CircuitService.create(db, _payload())
        updated = await CircuitService.update(
            db, c, {"narrative": "ok"}, expected_version=1)
        assert updated.version == 2

    @pytest.mark.asyncio
    async def test_write_edge_validation_recomputes_rung_via_update(self, db):
        """The edge-write path routes through update() so rung recompute runs
        (018 R2-A5) — writing a rung-2 validation result lifts the circuit."""
        c = await CircuitService.create(db, _payload())
        assert c.rung == 1  # the single edge is mined-rung-1
        updated = await CircuitService.write_edge_validation(
            db, c,
            {(13, 712, 14, 231): {"rung": 2, "effect_size": 0.8,
                                  "validation_manifest_ref": "vman_x"}})
        assert updated.rung == 2  # recomputed from the lifted edge
        assert updated.edges[0]["effect_size"] == 0.8
        assert updated.edges[0]["validation_manifest_ref"] == "vman_x"
        assert updated.version == 2  # bumped through update()

    @pytest.mark.asyncio
    async def test_write_edge_validation_respects_version(self, db):
        from src.services.circuit_service import CircuitConcurrencyError
        c = await CircuitService.create(db, _payload())
        await CircuitService.update(db, c, {"narrative": "bump"})  # → v2
        with pytest.raises(CircuitConcurrencyError):
            await CircuitService.write_edge_validation(
                db, c, {(13, 712, 14, 231): {"rung": 2}}, expected_version=1)


class TestPromotedCircuitValidationWriteBack:
    """R1 A1/#2: rung-2 ES must reach a PROMOTED circuit's edges (what 015
    reads). write_edge_validation is the contract-safe path."""

    @pytest.mark.asyncio
    async def test_validated_edge_lifts_promoted_circuit_rung(self, db):
        c = await CircuitService.create(db, _payload(discovery_run_id="dsc_v"))
        await CircuitService.set_promoted(db, c, True)
        assert c.rung == 1
        v_before = c.version
        updated = await CircuitService.write_edge_validation(
            db, c, {(13, 712, 14, 231): {"rung": 2, "effect_size": 0.9,
                                         "validation_manifest_ref": "vman_z"}})
        assert updated.rung == 2
        assert updated.edges[0]["effect_size"] == 0.9
        assert updated.edges[0]["validation_manifest_ref"] == "vman_z"
        assert updated.version == v_before + 1


class TestFromCandidates:
    """R2: the discovery→circuit producer — without it, no circuit carries a
    discovery_run_id so validated ES never reaches a promoted circuit."""

    @pytest.mark.asyncio
    async def test_builds_circuit_carrying_discovery_run_id(self, db):
        from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
        cap = CircuitCaptureRun(
            id="cap_fc", status="completed", store_path="/x",
            manifest={"model_id": "m1", "layers": [
                {"layer": 13, "sae_id": "sae_l13"},
                {"layer": 14, "sae_id": "sae_l14"}]})
        run = CircuitDiscoveryRun(
            id="dsc_fc", capture_run_id="cap_fc", status="completed",
            params={"granularity": "feature", "mode": "seeded"},
            candidates=[{
                "up": {"layer": 13, "feature_idx": 712},
                "down": {"layer": 14, "feature_idx": 231},
                "stats": {"pmi": 2.0, "support": 30, "null_pct": 99.0},
                "replicated_heldout": True,
                "attribution": {"rung1_gate": True},
                "orderings": {"coact_rank": 0, "attr_rank": 0}}])
        db.add(cap)
        db.add(run)
        await db.commit()
        circuit = await CircuitService.from_candidates(
            db, discovery_run_id="dsc_fc", name="Built circuit",
            candidate_keys=[(13, 712, 14, 231)])
        assert circuit.discovery_run_id == "dsc_fc"
        assert circuit.promoted is False  # promotion is a separate act
        assert len(circuit.members) == 2  # both endpoints
        assert circuit.edges[0]["rung"] == 1  # attribution-supported
        assert circuit.edges[0]["up"]["feature_idx"] == 712
        # SAE refs came from the capture manifest
        assert {s["layer"] for s in circuit.saes} == {13, 14}

    @pytest.mark.asyncio
    async def test_empty_selection_raises(self, db):
        from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
        from src.services.circuit_service import CircuitValidationError
        db.add(CircuitCaptureRun(id="cap_e", status="completed",
                                 manifest={"layers": []}))
        db.add(CircuitDiscoveryRun(id="dsc_e", capture_run_id="cap_e",
                                   status="completed", params={}, candidates=[]))
        await db.commit()
        with pytest.raises(CircuitValidationError, match="No matching"):
            await CircuitService.from_candidates(
                db, discovery_run_id="dsc_e", name="x", candidate_keys=[(1, 2, 3, 4)])


class TestFromCandidatesCarriesValidatedES:
    """R3 arc-closure: a circuit built AFTER validation (validate→promote) must
    carry the rung-2 effect_size onto the edge so 015 quantifies the hazard."""

    @pytest.mark.asyncio
    async def test_validated_candidate_edge_has_effect_size(self, db):
        from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
        cap = CircuitCaptureRun(
            id="cap_es", status="completed", store_path="/x",
            manifest={"model_id": "m", "layers": [
                {"layer": 13, "sae_id": "s13"}, {"layer": 14, "sae_id": "s14"}]})
        run = CircuitDiscoveryRun(
            id="dsc_es", capture_run_id="cap_es", status="completed",
            params={"granularity": "feature", "mode": "seeded"},
            candidates=[{
                "up": {"layer": 13, "feature_idx": 1},
                "down": {"layer": 14, "feature_idx": 2},
                "stats": {"pmi": 2.0, "support": 30},
                "validated_rung": 2,
                "validation": {"effect_size": 0.85, "passed": True,
                               "manifest_id": "vman_es"}}])
        db.add(cap); db.add(run)
        await db.commit()
        circuit = await CircuitService.from_candidates(
            db, discovery_run_id="dsc_es", name="Validated circuit",
            candidate_keys=[(13, 1, 14, 2)])
        edge = circuit.edges[0]
        assert edge["rung"] == 2
        assert edge["effect_size"] == 0.85            # ES carried (arc closure)
        assert edge["validation_manifest_ref"] == "vman_es"
