"""Calibration manifests now carry first-class `transcripts` (Part C).

Two things pinned:
  1. The search trace and the manifest payload carry the GENERATED TEXT per
     (dial, prompt), so a calibration manifest is analysis-ready for an LLM
     meaning pass — not verdict-only.
  2. BACK-COMPAT: `transcripts` is a REQUIRED key on NEW manifests, but
     `validate_payload` runs only on WRITE, so a legacy calibration manifest
     (written before the key existed) is never re-validated and stays
     readable/reproducible. This is the one real risk of the change.
"""

import pytest


class TestTheSearchTraceCarriesGeneratedText:
    def test_cliff_and_sanity_entries_include_the_generation(self):
        from src.services.circuit_calibration_search import calibrate

        def gen_at(dial, prompt):
            return f"OUTPUT@{dial}:{prompt}"

        def baseline_at(prompt, seed):
            return f"OUTPUT@0.0:{prompt}"

        def judge(text, expected):
            dial = float(text.split("@")[1].split(":")[0])
            return "correct" if dial <= 0.6 else "broken"

        def divergence(a, b):
            da = float(a.split("@")[1].split(":")[0])
            db_ = float(b.split("@")[1].split(":")[0])
            s = "x" if "@0.0:" not in a else a
            return abs(da - db_) + (0.02 if a != b and da == db_ else 0.0)

        probes = [{"prompt": "q1", "expected": "a1"}]
        res = calibrate(gen_at, baseline_at, judge, divergence,
                        probes=probes, lo=0.0, hi=1.0, max_steps=10)
        judged = [t for t in res.trace if t.get("phase") in ("cliff", "judge_sanity")]
        assert judged, "no judged trace entries"
        assert all("generation" in t for t in judged), (
            "a judged trace entry is missing the generated text — the manifest "
            "would be verdict-only and useless for meaning-analysis")
        # the recorded text is the actual model output, not the verdict
        assert any(t["generation"].startswith("OUTPUT@") for t in judged)


class TestBackwardCompatibility:
    def test_a_new_calibration_manifest_requires_transcripts(self):
        from src.services.manifest_service import ManifestError, validate_payload

        good = {"probes": [], "config": {}, "band": {}, "trace": [],
                "transcripts": [{"dial": 0.5, "prompt": "q", "generation": "out"}]}
        validate_payload("calibration", good)  # no raise

        legacy_shape = {"probes": [], "config": {}, "band": {}, "trace": []}
        with pytest.raises(ManifestError, match="transcripts"):
            validate_payload("calibration", legacy_shape)   # missing → rejected on WRITE

    def test_validate_payload_is_only_called_on_write_never_on_read(self):
        """The back-compat guarantee rests on this: no read/reproduce path
        re-validates a stored manifest. Asserted structurally over the source so
        a future read-side validate_payload call is caught."""
        import inspect

        from src.services import (circuit_calibration_service,
                                  circuit_faithfulness_service,
                                  circuit_intervention_service, manifest_service)

        for mod in (circuit_calibration_service, circuit_faithfulness_service,
                    circuit_intervention_service):
            src = inspect.getsource(mod)
            for i, line in enumerate(src.splitlines()):
                if "validate_payload(" in line and "def " not in line:
                    # every call must be adjacent to a persist/insert, never a
                    # get/read. Heuristic: the surrounding 6 lines must mention a
                    # ValidationManifest construction or a persist method.
                    window = "\n".join(src.splitlines()[max(0, i - 2):i + 6])
                    assert ("ValidationManifest(" in window
                            or "_persist" in window or "man =" in window), (
                        f"validate_payload in {mod.__name__} is not at a write "
                        f"site (line {i}): {line.strip()}")

    def test_reproduction_reads_a_legacy_manifest_without_transcripts(self):
        """calibration_reproduction_verdict compares BANDS, not transcripts — a
        legacy original manifest (no transcripts) must still be comparable."""
        from src.services.manifest_service import ManifestService

        legacy = {"band": {"onset": 0.2, "sweet_spot": 0.5, "cliff": 0.6}}  # no transcripts
        fresh = {"band": {"onset": 0.2, "sweet_spot": 0.5, "cliff": 0.6},
                 "transcripts": [{"dial": 0.5, "generation": "x"}]}
        v = ManifestService.calibration_reproduction_verdict(legacy, fresh)
        assert v["within_tolerance"] is True   # reproduces the band, transcripts irrelevant
