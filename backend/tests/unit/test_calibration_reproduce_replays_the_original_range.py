"""A calibration reproduction searches the ORIGINAL's dial range, not the clamped one.

Found on the 3090 2026-10-17 (crc_45f22d113063): calibration over [0, 2] found onset 0.4 and,
as designed, clamped the circuit's intensity_range to [onset, cliff] = [0.4, 1.0]. Its
reproduction re-read the circuit's (now clamped) range, put the coarse onset grid on 0.52,
and reported the identical measurement "outside tolerance" (onset delta 0.12 > 0.1).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.services import circuit_calibration_service as svc
from src.services.circuit_calibration_service import CircuitCalibrationService


def _db(original, circuit):
    from src.models.circuit import Circuit
    from src.models.validation_manifest import ValidationManifest

    db = MagicMock()

    def query(model):
        q = MagicMock()
        row = original if model is ValidationManifest else circuit if model is Circuit else None
        q.filter.return_value.first.return_value = row
        return q

    db.query.side_effect = query
    db.refresh.side_effect = lambda man: setattr(man, "id", "vman_repro")
    return db


def _run(original_payload, circuit_range):
    circuit = SimpleNamespace(id="crc_1", budget={"intensity_range": circuit_range})
    original = SimpleNamespace(id="vman_o", kind="calibration", circuit_id="crc_1",
                               payload=original_payload)
    result = SimpleNamespace(onset=0.4, sweet_spot=0.85, cliff=1.0, non_monotone=False,
                             usable_band=True, trace=[])
    fns = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), None)
    with patch.object(CircuitCalibrationService, "_build_generation_fns", return_value=fns), \
         patch.object(svc, "calibrate", return_value=result) as calibrate, \
         patch("src.services.manifest_service.validate_payload"):
        CircuitCalibrationService.reproduce(_db(original, circuit), "vman_o", device="cpu")
    assert calibrate.call_count == 1
    return calibrate.call_args.kwargs["lo"], calibrate.call_args.kwargs["hi"]


PAYLOAD = {"config": {"step_budget": 10, "probe_count": 3, "seed": 0, "margin": 0.15},
           "probes": [{"prompt": "What is the capital of Peru?", "expected": "Lima"}],
           "band": {"onset": 0.4, "sweet_spot": 0.85, "cliff": 1.0}}


def test_the_reproduction_searches_the_recorded_range_not_the_clamped_one():
    assert _run({**PAYLOAD, "dial_bounds": [0.0, 2.0]}, [0.4, 1.0]) == (0.0, 2.0)


def test_a_manifest_without_recorded_bounds_falls_back_to_the_circuit():
    assert _run(dict(PAYLOAD), [0.0, 1.5]) == (0.0, 1.5)
