"""The calibration block is an ADDITIVE, nullable contract extension (IDL-37).

The circuit-definition contract crosses to miLLM. The failure mode this repo has
hit repeatedly is a schema change that invalidates already-exported documents or
gets silently dropped on one of the several write/export paths. These tests pin:
  - a document with NO calibration (the shape every existing export has) is valid;
  - a calibration block round-trips through the contract model unchanged;
  - the band ordering onset ≤ sweet_spot ≤ cliff is enforced (an inverted band
    would clamp intensity_range to an empty/inverted interval);
  - calibration survives the SERVICE round-trip AND the export projection — the
    two paths where "threaded through every site or silently dropped" bites.
"""

import pytest


class TestTheCalibrationBlockIsAdditiveAndNullable:
    def test_a_definition_with_no_calibration_is_valid(self):
        """Every circuit exported before this feature has calibration absent.
        Those documents must stay valid, or the export-then-reimport loop breaks
        for the entire existing corpus."""
        from src.schemas.circuit_definition import CircuitDefinitionV1

        defn = CircuitDefinitionV1(
            name="t",
            saes=[{"layer": 12, "sae_id": "sae_abc"}],
            members=[{
                "layer": 12, "member_kind": "feature_ref",
                "feature": {"feature_idx": 1, "strength": 1.0},
            }],
        )
        assert defn.calibration is None
        dumped = defn.model_dump(mode="json")
        assert dumped["calibration"] is None

    def test_an_explicit_null_calibration_is_valid(self):
        from src.schemas.circuit_definition import CircuitDefinitionV1

        defn = CircuitDefinitionV1(
            name="t", calibration=None,
            saes=[{"layer": 12, "sae_id": "sae_abc"}],
            members=[{"layer": 12, "member_kind": "feature_ref",
                      "feature": {"feature_idx": 1, "strength": 1.0}}],
        )
        assert defn.calibration is None

    def test_the_published_schema_validates_a_pre_feature_document(self):
        """Against the GENERATED schema (the artifact consumers check), a
        calibration-absent document must validate."""
        import json
        import pathlib

        import jsonschema

        schema = json.loads((pathlib.Path(__file__).resolve().parents[3]
                             / "docs/schemas/circuit-definition-v1.json").read_text())
        doc = {
            "kind": "mistudio.circuit-definition",
            "name": "t",
            "saes": [{"layer": 12, "mistudio_sae_id": "sae_abc"}],
            "members": [{"layer": 12, "member_kind": "feature_ref",
                         "feature": {"feature_idx": 1, "strength": 1.0}}],
        }
        jsonschema.validate(doc, schema)  # must not raise


class TestTheCalibrationBandRoundTrips:
    def _band(self, **over):
        base = dict(onset=0.2, sweet_spot=0.5, cliff=0.6, provisional=True,
                    probe_set=[{"prompt": "Capital of France?", "expected": "Paris"}],
                    judge_metric_id="correctness/v1", step_budget=10)
        base.update(over)
        return base

    def test_a_full_band_round_trips_unchanged(self):
        from src.schemas.circuit_definition import CircuitCalibration

        c = CircuitCalibration(**self._band())
        out = c.model_dump(mode="json")
        assert out["onset"] == 0.2 and out["sweet_spot"] == 0.5 and out["cliff"] == 0.6
        assert out["provisional"] is True
        assert out["probe_set"][0]["expected"] == "Paris"

    def test_an_inverted_band_is_REJECTED(self):
        """onset > cliff would clamp intensity_range to an inverted interval —
        a served dial with no valid position. Reject at the contract."""
        from pydantic import ValidationError

        from src.schemas.circuit_definition import CircuitCalibration

        with pytest.raises(ValidationError):
            CircuitCalibration(**self._band(onset=0.7, sweet_spot=0.5, cliff=0.6))

    def test_sweet_spot_outside_the_band_is_REJECTED(self):
        from pydantic import ValidationError

        from src.schemas.circuit_definition import CircuitCalibration

        with pytest.raises(ValidationError):
            CircuitCalibration(**self._band(onset=0.2, sweet_spot=0.9, cliff=0.6))

    def test_a_probe_requires_both_prompt_and_expected(self):
        """A probe with no expected answer cannot be judged for correctness."""
        from pydantic import ValidationError

        from src.schemas.circuit_definition import CalibrationProbe

        with pytest.raises(ValidationError):
            CalibrationProbe(prompt="Capital of France?", expected="")
