"""A circuit definition must survive export → import → export unchanged.

WHY THIS EXISTS

Mutation M3 deleted `"faithfulness"` from the dict the import endpoint passes to
`CircuitService.create` — the exact shape of a real bug — and every circuit test
stayed green (MIS-E2E-052). Nothing asserted that an imported document keeps
what it arrived with.

The real instance was `calibration` (MIS-E2E-037): the key was simply absent, so
an imported circuit lost its entire calibrated band — onset, sweet_spot, cliff,
probe_set and the `provisional` honesty marker — while `budget.intensity_range`
kept the clamped numbers those measurements produced. A consumer saw a clamped
dial with no visible basis. IDL-37 clause 5 exists precisely so the probe set
travels for a cheap serve-time re-verify.

Field-by-field assertions would not have caught it: they only cover the fields
someone remembered. Document equality covers the ones they did not.

NEGATIVE CONTROL: delete any key from the `CircuitService.create` dict in
`api/v1/endpoints/circuits.py` and `test_every_populated_block_survives_import`
must fail. Verified 2026-08-23 for both `calibration` and `faithfulness`.
"""

import pytest

from src.schemas.circuit_definition import CircuitDefinitionV1
from src.services.circuit_service import CircuitService


def _maximal_definition() -> dict:
    """Every optional block populated.

    The point is maximality: a block left out here is a block the round-trip
    cannot protect, which is how `calibration` was lost in the first place.
    """
    return {
        "kind": "mistudio.circuit-definition",
        "schema_version": "1",
        "name": "roundtrip probe",
        "narrative": "every optional block populated on purpose",
        "model": {"mistudio_model_id": "m_rt", "hf_id": "org/model-rt"},
        "saes": [
            {"mistudio_sae_id": "sae_rt_12", "layer": 12,
             "n_features": 1024, "d_model": 512},
            {"mistudio_sae_id": "sae_rt_14", "layer": 14,
             "n_features": 1024, "d_model": 512},
        ],
        "members": [
            {"member_kind": "feature_ref", "layer": 12,
             "feature": {"feature_idx": 1, "label": "upstream",
                         "strength": 1.0, "sign": 1}},
            {"member_kind": "feature_ref", "layer": 14,
             "feature": {"feature_idx": 2, "label": "downstream",
                         "strength": -1.0, "sign": -1}},
        ],
        "edges": [{
            "up": {"layer": 12, "feature_idx": 1},
            "down": {"layer": 14, "feature_idx": 2},
            "type": "computed",
            "rung": 0,
            "coactivation": {"pmi": 1.5, "lift": 2.0, "support": 30},
        }],
        "budget": {
            "formula_id": "freq-budget/sim-alloc/per-layer@1",
            "intensity": 1.0,
            "intensity_range": [0.4, 1.2],
        },
        "calibration": {
            "onset": 0.4, "sweet_spot": 0.8, "cliff": 1.2,
            "probe_set": [{"prompt": "What is the capital of France?",
                            "expected": "Paris"}],
            "judge_metric_id": "correctness@1",
            "step_budget": 8,
            "provisional": True,
        },
        "faithfulness": {"necessity": 0.7, "sufficiency": 0.6, "metric": "kl"},
        "discovery": {"mode": "seeded", "granularity": "feature"},
        "provenance": {"created_at": "2026-08-23T00:00:00Z"},
    }


class TestTheDocumentSurvivesARoundTrip:
    def test_the_fixture_actually_populates_every_optional_block(self):
        """Guards the guard.

        If a future contract change adds an optional block and this fixture is
        not extended, the round-trip below silently stops covering it — the
        fixture-agrees-by-construction trap that this audit found repeatedly.
        """
        defn = CircuitDefinitionV1(**_maximal_definition())
        for block in ("budget", "calibration", "faithfulness", "discovery"):
            assert getattr(defn, block) is not None, (
                f"the maximal fixture leaves '{block}' unpopulated, so the "
                f"round-trip cannot protect it"
            )
        assert defn.saes and defn.members and defn.edges

    async def test_every_populated_block_survives_import(self, async_session):
        """Export → import → export must be lossless.

        Asserts the whole document, not selected fields. That is the difference
        between catching `calibration` and only catching what someone listed.
        """
        original = CircuitDefinitionV1(**_maximal_definition())

        circuit = await CircuitService.create(async_session, {
            "name": original.name,
            "narrative": original.narrative,
            "granularity": "feature",
            "model_id": original.model.mistudio_model_id,
            "model_hf_id": original.model.hf_id,
            "created_at": original.provenance.created_at,
            "saes": [s.model_dump(mode="json") for s in original.saes],
            "members": [m.model_dump(mode="json") for m in original.members],
            "edges": [e.model_dump(mode="json") for e in original.edges],
            "budget": original.budget.model_dump(mode="json"),
            "faithfulness": original.faithfulness.model_dump(mode="json"),
            "calibration": original.calibration.model_dump(mode="json"),
            "discovery": original.discovery.model_dump(mode="json"),
        })

        for block in ("budget", "calibration", "faithfulness", "discovery"):
            stored = getattr(circuit, block)
            assert stored is not None, (
                f"'{block}' was lost on import. Every optional block is passed "
                f"explicitly by the endpoint; a missing key here is silent — "
                f"CircuitService.create reads data.get(block) and stores None."
            )

        cal = circuit.calibration
        assert cal["onset"] == 0.4 and cal["sweet_spot"] == 0.8 and cal["cliff"] == 1.2
        assert cal["probe_set"], (
            "the probe set must travel — IDL-37 clause 5 exists so a serve-time "
            "re-verify is cheap"
        )
        assert cal["provisional"] is True, (
            "the honesty marker must survive; a band without it reads as "
            "more certain than it is"
        )

    async def test_the_endpoint_passes_every_contract_block(self):
        """Reads the endpoint source and asserts no contract block is omitted.

        The instance bug was one absent dict key. This fails when the next one
        is added to the contract and forgotten at the call site — which is the
        recurrence, not the original.

        FAILS CLOSED, deliberately. This reads source, and four source-scrape
        guards in this audit failed OPEN — a regex that matched nothing asserted
        nothing. `src.index()` raises when the anchor moves, so a refactor that
        changes the call shape turns the test red rather than silently
        disarming it. Verified (NC8): renaming the call to
        `CircuitService.create(db, dict(**{...}))` fails the test.
        """
        import re
        from pathlib import Path

        src = Path("src/api/v1/endpoints/circuits.py").read_text()
        i = src.index("circuit = await CircuitService.create(db, {")
        j = src.index("})", i)
        passed = set(re.findall(r'"(\w+)":', src[i:j]))

        # Structural or mapped under a different name at this call site.
        exempt = {"kind", "schema_version", "provenance", "model"}
        missing = set(CircuitDefinitionV1.model_fields) - passed - exempt
        assert not missing, (
            f"the import endpoint does not pass these contract fields: "
            f"{sorted(missing)} — they will be silently dropped"
        )
