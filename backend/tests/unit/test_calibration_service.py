"""CircuitCalibrationService orchestration (IDL-37), covered without GPU/LLM.

build_band() is the tested seam: it runs probes → search → band + manifest with
injected generation/judge/divergence, so everything except the GPU generation
loop is unit-covered. Also pins config validation and the manifest's
self-containment.
"""

import pytest

from src.services.circuit_calibration_service import (
    CalibrationConfigError, CircuitCalibrationService, cosine_text_divergence)


class _Circuit:
    """Minimal stand-in with the attributes the service reads."""
    def __init__(self, labels, intensity_range=None):
        self.members = [{"feature": {"label": lb}} for lb in labels]
        self.budget = {"intensity_range": intensity_range} if intensity_range else {}


# Synthetic world: cliff at 0.6, fluent throughout (so only the judge finds it).
def _gen_at(dial, prompt):
    return f"dial={dial}|{prompt}"


def _baseline_at(prompt, seed=0):
    return f"dial=0.0|seed={seed}|{prompt}"


def _judge(text, expected):
    dial = float(text.split("|", 1)[0].split("=")[1])
    return "correct" if dial <= 0.6 else "broken"


def _divergence(a, b):
    da = float(a.split("|", 1)[0].split("=")[1])
    db_ = float(b.split("|", 1)[0].split("=")[1])
    sa = a.split("seed=")[1].split("|")[0] if "seed=" in a else "x"
    sb = b.split("seed=")[1].split("|")[0] if "seed=" in b else "x"
    jitter = 0.02 if sa != sb else 0.0
    return abs(da - db_) + jitter


def _fixed_probes_llm(prompt, max_tokens):
    import json
    return json.dumps([{"prompt": "What is the capital of France?", "expected": "Paris"}])


class TestBuildBand:
    def test_produces_an_ordered_provisional_band_with_a_manifest(self):
        c = _Circuit(["parody", "humor"], intensity_range=[0.0, 1.0])
        out = CircuitCalibrationService.build_band(
            c, gen_at=_gen_at, baseline_at=_baseline_at, judge=_judge,
            divergence=_divergence, llm_call=_fixed_probes_llm,
            config={"step_budget": 12})
        band = out["band"]
        assert band["onset"] <= band["sweet_spot"] <= band["cliff"]
        assert abs(band["cliff"] - 0.6) <= 0.06
        assert band["provisional"] is True            # generated probes
        assert band["judge_metric_id"]
        assert band["probe_set"]                       # travels in the contract

    def test_manifest_is_self_contained_and_valid(self):
        from src.services.manifest_service import validate_payload
        c = _Circuit(["parody"], intensity_range=[0.0, 1.0])
        out = CircuitCalibrationService.build_band(
            c, gen_at=_gen_at, baseline_at=_baseline_at, judge=_judge,
            divergence=_divergence, llm_call=_fixed_probes_llm)
        # Must satisfy the calibration manifest kind's required keys.
        validate_payload("calibration", out["manifest_payload"])  # no raise
        assert out["manifest_payload"]["trace"]        # every judged step recorded

    def test_search_stays_within_the_authored_intensity_range(self):
        # A narrow authored range caps where calibration can look.
        c = _Circuit(["parody"], intensity_range=[0.0, 0.5])
        out = CircuitCalibrationService.build_band(
            c, gen_at=_gen_at, baseline_at=_baseline_at, judge=_judge,
            divergence=_divergence, llm_call=_fixed_probes_llm)
        # Cliff is 0.6 in the judge, but the search cannot exceed hi=0.5.
        assert out["band"]["cliff"] <= 0.5


class TestConfig:
    def test_rejects_tiny_step_budget(self):
        with pytest.raises(CalibrationConfigError):
            CircuitCalibrationService.create_config({"step_budget": 1})

    def test_rejects_zero_probes(self):
        with pytest.raises(CalibrationConfigError):
            CircuitCalibrationService.create_config({"probe_count": 0})

    def test_rejects_out_of_range_margin(self):
        with pytest.raises(CalibrationConfigError):
            CircuitCalibrationService.create_config({"margin": 1.5})


class _FakeCircuit:
    """In-memory circuit for the sync write-back, matching the attributes
    _write_calibration reads/writes without a DB session."""
    def __init__(self):
        self.id = "crc_test"
        self.name = "calib"
        self.narrative = None
        self.saes = [{"mistudio_sae_id": "sae_l12", "layer": 12}]
        self.members = [{"layer": 12, "member_kind": "feature_ref",
                         "feature": {"feature_idx": 1, "strength": 1.0,
                                     "label": "parody"}}]
        self.edges = []
        self.budget = {"intensity_range": [0.0, 1.0], "intensity": 1.0,
                       "layers": {"12": {"B": 1.0}}}
        self.faithfulness = None
        self.calibration = None
        self.version = 3
        self.calibration_status = "running"


class _FakeSyncDB:

    #: R2. A no-op `refresh` was added here in R1 so the completion write would
    #: not raise — and that is precisely what hid the fact that the write it
    #: was smoothing over DISCARDED THE WHOLE RESULT. A fake that makes the
    #: dangerous call harmless is the "fixtures agree by construction" trap in
    #: its purest form. Removed; the code now reads a scalar instead.

    def execute(self, statement, *a, **k):
        """Serve the scalar status read the completion guard performs."""
        class _R:
            def __init__(self, value):
                self._value = value

            def scalar(self):
                return self._value

        return _R(getattr(self._c, "calibration_status", None))

    def __init__(self, circuit):
        self._c = circuit
        self.committed = False

    def query(self, _model):
        db = self

        class _Q:
            def filter(self, *a, **k):
                return self

            def first(self_inner):
                return db._c
        return _Q()

    def commit(self):
        self.committed = True


class TestWriteCalibrationClampsThroughTheContract:
    """The sync write-back (used by the GPU task) clamps the dial AND writes the
    band through the CircuitDefinitionV1 contract — never a raw JSONB mutation,
    mirroring _write_circuit_faithfulness."""

    _BAND = {"onset": 0.2, "sweet_spot": 0.5, "cliff": 0.6, "provisional": True,
             "probe_set": [{"prompt": "Capital of France?", "expected": "Paris"}],
             "judge_metric_id": "x/v1", "step_budget": 10, "non_monotone": False,
             "manifest_ref": "vman_x"}

    def test_it_clamps_the_dial_and_stores_the_band(self):
        c = _FakeCircuit()
        db = _FakeSyncDB(c)
        CircuitCalibrationService._write_calibration(db, c.id, self._BAND)
        assert c.budget["intensity_range"] == [0.2, 0.6]
        assert c.budget["intensity"] == 0.5
        assert c.budget["layers"]["12"]["B"] == 1.0   # per-layer budget preserved
        assert c.calibration["sweet_spot"] == 0.5
        assert c.calibration_status == "completed"
        assert c.version == 4                          # bumped
        assert db.committed

    def test_it_refuses_an_inverted_band(self):
        from src.services.circuit_calibration_service import CalibrationRunError
        c = _FakeCircuit()
        db = _FakeSyncDB(c)
        with pytest.raises(CalibrationRunError):
            CircuitCalibrationService._write_calibration(
                db, c.id, {**self._BAND, "onset": 0.7, "sweet_spot": 0.5, "cliff": 0.6})
        assert c.calibration is None                   # nothing written on refusal


class TestReproductionVerdict:
    """FPRD §8.5: reproducing a calibration compares the band within tolerance."""

    def test_identical_band_is_within_tolerance(self):
        from src.services.manifest_service import ManifestService
        orig = {"band": {"onset": 0.2, "sweet_spot": 0.5, "cliff": 0.6}}
        repro = {"band": {"onset": 0.21, "sweet_spot": 0.49, "cliff": 0.62}}
        v = ManifestService.calibration_reproduction_verdict(orig, repro, tolerance=0.1)
        assert v["within_tolerance"] is True
        assert v["max_delta"] <= 0.1

    def test_a_divergent_band_fails(self):
        from src.services.manifest_service import ManifestService
        orig = {"band": {"onset": 0.2, "sweet_spot": 0.5, "cliff": 0.6}}
        repro = {"band": {"onset": 0.2, "sweet_spot": 0.5, "cliff": 0.95}}
        v = ManifestService.calibration_reproduction_verdict(orig, repro, tolerance=0.1)
        assert v["within_tolerance"] is False
        assert v["max_delta"] > 0.1

    def test_nothing_to_compare_is_not_a_pass(self):
        from src.services.manifest_service import ManifestService
        v = ManifestService.calibration_reproduction_verdict({}, {}, tolerance=0.1)
        assert v["within_tolerance"] is None   # reproducing nothing is not reproducing


class TestJudgeVerdictParsing:
    """R2: `'broken' in verdict` matched 'not broken' → truncated the band."""

    def test_plain_verdicts(self):
        from src.services.circuit_calibration_service import _parse_verdict
        assert _parse_verdict("CORRECT") == "correct"
        assert _parse_verdict("broken") == "broken"
        assert _parse_verdict("Degrading") == "degrading"

    def test_unambiguously_negated_broken_is_correct(self):
        from src.services.circuit_calibration_service import _parse_verdict
        assert _parse_verdict("not broken") == "correct"
        assert _parse_verdict("The response is not broken.") == "correct"

    def test_terse_NO_broken_is_broken_not_correct(self):
        """R3 negative control: 'No, broken.' means no[t correct], BROKEN — the
        'no' must NOT negate the following 'broken'. The R2 fix over-reached by
        treating 'no' as a negator, shipping a broken dial as usable."""
        from src.services.circuit_calibration_service import _parse_verdict
        assert _parse_verdict("No, broken.") == "broken"
        assert _parse_verdict("No — broken") == "broken"
        assert _parse_verdict("No. Broken.") == "broken"

    def test_correct_wins_when_it_appears_first(self):
        from src.services.circuit_calibration_service import _parse_verdict
        assert _parse_verdict("correct, not broken") == "correct"

    def test_unparseable_is_conservative_broken(self):
        from src.services.circuit_calibration_service import _parse_verdict
        assert _parse_verdict("hmmmm") == "broken"


class TestBudgetInvariantIsBackwardCompatible:
    """R2: the intensity∈range invariant must CLAMP legacy budgets, not reject —
    a stored budget with a narrowed range and the default intensity=1.0 must
    still round-trip."""

    def test_legacy_narrow_range_with_default_intensity_is_clamped(self):
        from src.schemas.circuit_definition import CircuitBudget
        b = CircuitBudget(intensity_range=[0.0, 0.5])  # intensity defaults to 1.0
        assert b.intensity == 0.5   # clamped into range, not rejected

    def test_intensity_below_range_is_raised_to_lo(self):
        from src.schemas.circuit_definition import CircuitBudget
        b = CircuitBudget(intensity=0.1, intensity_range=[0.3, 0.8])
        assert b.intensity == 0.3

    def test_a_malformed_range_is_still_rejected(self):
        from pydantic import ValidationError

        from src.schemas.circuit_definition import CircuitBudget
        with pytest.raises(ValidationError):
            CircuitBudget(intensity_range=[0.8, 0.3])   # inverted → real error


class TestDivergence:
    def test_embedder_cosine(self):
        div = cosine_text_divergence(embed=lambda t: [1.0, 0.0] if "a" in t else [0.0, 1.0])
        assert div("a", "a") == pytest.approx(0.0)
        assert div("a", "b") == pytest.approx(1.0)

    def test_jaccard_fallback_when_no_embedder(self):
        div = cosine_text_divergence(embed=None)
        assert div("the cat sat", "the cat sat") == pytest.approx(0.0)
        assert div("cat", "dog") == pytest.approx(1.0)



def _code_only(source: str) -> str:
    """Source with full-line comments removed.

    Six guards in this remediation were satisfied by text inside the comment
    that explained them. Any assertion about the ABSENCE of a construct has to
    ignore the prose describing why it is absent.
    """
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


class TestTheCompletionGuardDoesNotEatTheResult:
    """R2/SEV-1. The guard that refuses to overwrite a cancellation must not
    destroy the run it is completing.

    R1 wrote `db.refresh(circuit)` before the status check. `Session.refresh()`
    EXPIRES the instance BEFORE autoflush, so every un-flushed attribute set
    just above it — the band, the intensity clamp, the version bump — was
    erased and the commit persisted only the status. On EVERY SUCCESSFUL RUN
    the feature silently produced nothing while reporting "completed".

    Nothing caught it because R1 also added a no-op `refresh` to the fake in
    this file, so the test asserted the pre-fix behaviour.
    """

    def test_a_scalar_read_is_used_rather_than_refresh(self):
        """`refresh` on the object being mutated can never be safe here.

        Asserted structurally as well as behaviourally, because the behavioural
        test below depends on a fake and this one does not.
        """
        import inspect

        from src.services.circuit_calibration_service import CircuitCalibrationService

        # Comments stripped — the FIRST version of this assertion matched the
        # comment explaining the fix, so it failed against correct code. That
        # is the seventh instance of this exact trap in this arc, and it
        # appeared inside the test written to catch the trap.
        src = _code_only(inspect.getsource(CircuitCalibrationService))
        assert "db.refresh(circuit)" not in src, (
            "refresh() expires the instance before autoflush, discarding the "
            "band and the clamp this run just computed"
        )

    def test_the_faithfulness_writer_likewise(self):
        import inspect

        from src.services import circuit_faithfulness_service as m

        assert "db.refresh(circuit)" not in _code_only(inspect.getsource(m)), (
            "the faithfulness scores are discarded by refresh() the same way"
        )

    def test_sqlalchemy_really_behaves_this_way(self):
        """The premise, executed rather than asserted — if `refresh` did NOT
        discard pending writes the fix above would be unnecessary."""
        from sqlalchemy import Column, Integer, String, create_engine, select
        from sqlalchemy.orm import declarative_base, sessionmaker

        Base = declarative_base()

        class _T(Base):
            __tablename__ = "t_refresh_probe"
            id = Column(Integer, primary_key=True)
            payload = Column(String)
            status = Column(String)

        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
        session = Session()
        session.add(_T(id=1, payload="old", status="running"))
        session.commit()

        # The R1 shape: refresh discards the pending result.
        row = session.get(_T, 1)
        row.payload = "the band"
        session.refresh(row)
        row.status = "completed"
        session.commit()
        assert session.get(_T, 1).payload == "old", (
            "refresh no longer discards pending writes; re-read the fix, it may "
            "no longer be needed"
        )

        # The replacement: a scalar select leaves them alone.
        session.execute(_T.__table__.delete())
        session.add(_T(id=2, payload="old", status="running"))
        session.commit()
        row = session.get(_T, 2)
        row.payload = "the band"
        fresh = session.execute(select(_T.status).where(_T.id == 2)).scalar()
        if fresh != "cancelled":
            row.status = "completed"
        session.commit()
        assert session.get(_T, 2).payload == "the band"
        assert session.get(_T, 2).status == "completed"
