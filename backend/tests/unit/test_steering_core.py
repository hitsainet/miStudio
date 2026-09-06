"""The unified steering core (Steered Transcript Recorder) + calibration parity.

The core's generation body is MOVED from calibration's _build_generation_fns and
is GPU-only (not unit-tested here — like faithfulness/calibration run()). What IS
tested without a GPU: the per-type member RESOLVERS (pure DB→tuples logic) and the
regression pin that the judge gate did NOT migrate into the shared core.
"""

import inspect

import pytest


class TestHookTargetIsTheWholeLayerNotANormLayer:
    """Hardware E2E (recorder) found that hooking the discovered "residual"
    module — a post-attention RMSNorm on LFM2 — renormalized the steering vector
    AWAY, so steered output was byte-identical to the baseline at every dial. The
    fix hooks the WHOLE decoder-layer output (resid_post), where an added vector
    survives. This is a GPU-only behaviour, so guard it structurally: the core
    must NOT resolve the hook target via get_hookable_module(..., "residual")."""

    def test_the_core_hooks_the_layer_output_not_get_hookable_module(self):
        import inspect

        from src.services import steering_core
        src = inspect.getsource(steering_core.build_steer_generator)
        assert "structure.layers_module[L]" in src, (
            "the hook target must be the whole decoder-layer output (resid_post)")
        # The bad pattern is resolving the target via get_hookable_module — that
        # returns a RMSNorm on LFM2 and steering is renormalized away. (A comment
        # may still mention "residual"; what must not recur is the CALL.)
        assert "get_hookable_module(" not in src, (
            "the core resolves the hook via get_hookable_module(...) again — on "
            "LFM2 that yields a RMSNorm and steering is a hardware-confirmed no-op")


class TestJudgeGateDidNotMigrate:
    """Part A regression pin: the calibration judge gate must still fire BEFORE
    the shared (judge-free) generation core is reached — proving the refactor
    moved the generation half, not the gate."""

    def test_build_generation_fns_still_raises_without_a_judge(self):
        from src.services.circuit_calibration_service import (
            CircuitCalibrationService, CalibrationRunError)

        class _Circuit:
            model_id = "m_x"
            members = [{"layer": 1, "feature": {"feature_idx": 1, "strength": 1.0}}]
            saes = [{"layer": 1, "mistudio_sae_id": "sae_x"}]

        # No judge_endpoint/judge_model → must raise the judge gate, and must do
        # so BEFORE any model load (so the error is the judge message, not a
        # model/SAE-not-found from the core).
        with pytest.raises(CalibrationRunError, match="judge"):
            CircuitCalibrationService._build_generation_fns(
                _Circuit(), db=None, cfg={"seed": 0})

    def test_the_gate_is_in_the_wrapper_not_the_core(self):
        """The shared core must be judge-free — no judge_endpoint reference."""
        from src.services import steering_core
        src = inspect.getsource(steering_core)
        assert "judge_endpoint" not in src and "judge_model" not in src, (
            "the shared steering core references a judge — the gate leaked into "
            "the judge-free half")


class TestCircuitResolver:
    """resolve_circuit_members: nested member.feature.strength, per-layer SAE,
    silent-drop refusal. Uses a fake db so no GPU/model is needed."""

    class _FakeSae:
        pass

    def _fake_db(self, sae_ids):
        # Returns a db whose ExternalSAE query yields a stub for known ids.
        known = set(sae_ids)

        class _Q:
            def __init__(self, sid):
                self._sid = sid

            def filter(self, *a, **k):
                return self

            def first(self_inner):
                return object() if self_inner._sid in known else None

        class _DB:
            def query(self, model):
                # capture the id from the subsequent filter via a closure trick:
                # simpler — return a query that always yields a stub (id checked
                # by the resolver's own missing-SAE branch is exercised separately)
                class _QAll:
                    def filter(self, expr):
                        # crude: pull the bound value from the BinaryExpression
                        try:
                            sid = expr.right.value
                        except Exception:
                            sid = None
                        return _Q(sid)
                return _QAll()
        return _DB()

    def test_resolves_members_with_strength_and_layer(self, monkeypatch):
        from src.services import steering_core

        # Stub the GPU-touching helpers: SAE load + decoder-weight resolution.
        monkeypatch.setattr(steering_core, "_load_wdec_by_layer",
                            lambda ids, db, device: {L: f"W{L}" for L in ids})

        class _Circuit:
            model_id = "m_x"
            saes = [{"layer": 3, "mistudio_sae_id": "sae_a"},
                    {"layer": 5, "mistudio_sae_id": "sae_b"}]
            members = [
                {"layer": 3, "feature": {"feature_idx": 10, "strength": 0.4}},
                {"layer": 5, "feature": {"feature_idx": 20, "strength": 0.6}},
            ]

        model_id, resolved = steering_core.resolve_circuit_members(
            _Circuit(), db=None, device="cpu")
        assert model_id == "m_x"
        assert resolved == [(3, 10, 0.4, "W3"), (5, 20, 0.6, "W5")]

    def test_refuses_a_member_whose_layer_has_no_sae(self, monkeypatch):
        from src.services import steering_core
        from src.services.steering_core import SteeringCoreError

        # Only layer 3 gets a decoder weight; the layer-9 member must be refused.
        monkeypatch.setattr(steering_core, "_load_wdec_by_layer",
                            lambda ids, db, device: {3: "W3"})

        class _Circuit:
            model_id = "m_x"
            saes = [{"layer": 3, "mistudio_sae_id": "sae_a"}]
            members = [
                {"layer": 3, "feature": {"feature_idx": 10, "strength": 0.4}},
                {"layer": 9, "feature": {"feature_idx": 99, "strength": 0.5}},
            ]

        with pytest.raises(SteeringCoreError, match="layer 9"):
            steering_core.resolve_circuit_members(_Circuit(), db=None, device="cpu")


class TestFeatureResolver:
    def test_resolves_ad_hoc_features(self, monkeypatch):
        from src.services import steering_core

        monkeypatch.setattr(steering_core, "_load_wdec_by_layer",
                            lambda ids, db, device: {L: f"W{L}" for L in ids})
        specs = [{"layer": 2, "feature_idx": 7, "strength": 1.5, "sae_id": "s2"},
                 {"layer": 4, "feature_idx": 8, "strength": -0.5, "sae_id": "s4"}]
        model_id, resolved = steering_core.resolve_feature_members(
            specs, model_id="m_y", db=None, device="cpu")
        assert model_id == "m_y"
        assert resolved == [(2, 7, 1.5, "W2"), (4, 8, -0.5, "W4")]

    def test_rejects_two_saes_at_one_layer(self, monkeypatch):
        from src.services import steering_core
        from src.services.steering_core import SteeringCoreError

        monkeypatch.setattr(steering_core, "_load_wdec_by_layer",
                            lambda ids, db, device: {L: f"W{L}" for L in ids})
        specs = [{"layer": 2, "feature_idx": 7, "strength": 1.0, "sae_id": "s2"},
                 {"layer": 2, "feature_idx": 8, "strength": 1.0, "sae_id": "OTHER"}]
        with pytest.raises(SteeringCoreError, match="two different SAEs"):
            steering_core.resolve_feature_members(specs, model_id="m", db=None,
                                                  device="cpu")


class TestClusterResolver:
    def test_applies_sign_to_strength(self, monkeypatch):
        from src.services import steering_core

        monkeypatch.setattr(steering_core, "_load_wdec_by_layer",
                            lambda ids, db, device: {L: f"W{L}" for L in ids})

        class _Prof:
            id = "clp_x"
            model_id = "m_z"
            saes = [{"layer": 6, "mistudio_sae_id": "s6"}]
            members = [{"layer": 6, "feature_idx": 11, "strength": 0.5, "sign": -1}]

        class _DB:
            def query(self, model):
                class _Q:
                    def filter(self, *a, **k):
                        return self

                    def first(self_inner):
                        return _Prof()
                return _Q()

        model_id, resolved = steering_core.resolve_cluster_members(
            "clp_x", db=_DB(), device="cpu")
        assert model_id == "m_z"
        assert resolved == [(6, 11, -0.5, "W6")]   # sign folded into strength

    def test_derives_model_from_sae_when_profile_has_no_model_id(self, monkeypatch):
        """Every CLUSTERS-arc profile persists sae_id but NOT model_id — the
        recorder must derive the model from the SAE, else no existing cluster is
        steerable. (Hardware E2E surfaced this: all 35 profiles had model_id=None.)"""
        from src.services import steering_core

        monkeypatch.setattr(steering_core, "_load_wdec_by_layer",
                            lambda ids, db, device: {L: f"W{L}" for L in ids})

        class _Prof:
            id = "clp_y"
            model_id = None                       # ← the real-data case
            mistudio_sae_id = None
            sae_id = "sae_real"                   # single-SAE profile
            saes = None
            # Real CLUSTERS-arc members carry NO per-member layer — the layer
            # lives on the SAE row. All 35 existing profiles look like this.
            members = [{"feature_idx": 7, "strength": 0.2, "sign": 1}]

        class _SAE:
            model_id = "m_88d55564"
            layer = 13

        class _DB:
            def query(self, model):
                name = getattr(model, "__name__", "")
                class _Q:
                    def filter(self, *a, **k):
                        return self

                    def first(self_inner):
                        return _Prof() if name == "ClusterProfile" else _SAE()
                return _Q()

        model_id, resolved = steering_core.resolve_cluster_members(
            "clp_y", db=_DB(), device="cpu")
        assert model_id == "m_88d55564"           # derived from the SAE
        # layer 13 inherited from the SAE row, NOT the (layerless) member
        assert resolved == [(13, 7, 0.2, "W13")]
