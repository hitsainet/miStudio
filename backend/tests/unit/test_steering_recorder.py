"""SteeringRecorderService: config caps + orchestration (no GPU).

The GPU generation is injected (via monkeypatching the model-load + core), so the
transcript-building loop, the caps, and the manifest payload are all unit-tested.
"""

import pytest

from src.services.steering_recorder_service import (RecordConfigError,
                                                    SteeringRecorderService)


def _cfg(**over):
    base = {"artifact": {"kind": "circuit", "circuit_id": "crc_x"},
            "dials": [0.5, 1.0], "prompts": ["p1", "p2"]}
    base.update(over)
    return base


class TestPerKindValidation:
    """R1: a malformed artifact must 422 up front, before the GPU lock is taken —
    not surface as an opaque KeyError deep in the task."""

    def test_circuit_needs_circuit_id(self):
        with pytest.raises(RecordConfigError, match="circuit_id"):
            SteeringRecorderService.create_config(_cfg(artifact={"kind": "circuit"}))

    def test_cluster_needs_profile_id(self):
        with pytest.raises(RecordConfigError, match="cluster_profile_id"):
            SteeringRecorderService.create_config(_cfg(artifact={"kind": "cluster"}))

    def test_features_needs_model_id_and_features(self):
        with pytest.raises(RecordConfigError, match="model_id"):
            SteeringRecorderService.create_config(_cfg(
                artifact={"kind": "features", "features": [{"layer": 1, "feature_idx": 2}]}))
        with pytest.raises(RecordConfigError, match="features list"):
            SteeringRecorderService.create_config(_cfg(
                artifact={"kind": "features", "model_id": "m"}))

    def test_features_each_needs_layer_and_idx(self):
        with pytest.raises(RecordConfigError, match="layer, feature_idx"):
            SteeringRecorderService.create_config(_cfg(
                artifact={"kind": "features", "model_id": "m",
                          "features": [{"strength": 1.0}]}))

    def test_features_each_needs_sae_id(self):
        # R2: a feature with layer+feature_idx but no sae_id used to pass here and
        # fail deep in the task AFTER the GPU lock + model load.
        with pytest.raises(RecordConfigError, match="sae_id"):
            SteeringRecorderService.create_config(_cfg(
                artifact={"kind": "features", "model_id": "m",
                          "features": [{"layer": 1, "feature_idx": 2, "strength": 1.0}]}))

    def test_features_reject_two_saes_at_one_layer_UP_FRONT(self):
        # R3: the "one SAE per layer" consistency check must 422 at config time,
        # not fail deep in the resolver after the GPU lock + model load.
        with pytest.raises(RecordConfigError, match="two different SAEs"):
            SteeringRecorderService.create_config(_cfg(
                artifact={"kind": "features", "model_id": "m", "features": [
                    {"layer": 2, "feature_idx": 7, "strength": 1.0, "sae_id": "sA"},
                    {"layer": 2, "feature_idx": 8, "strength": 1.0, "sae_id": "sB"}]}))

    def test_non_numeric_dial_is_422_not_500(self):
        with pytest.raises(RecordConfigError, match="not a number"):
            SteeringRecorderService.create_config(_cfg(dials=["abc"]))

    def test_non_numeric_max_tokens_is_422(self):
        with pytest.raises(RecordConfigError, match="max_tokens"):
            SteeringRecorderService.create_config(_cfg(max_tokens="lots"))

    def test_dial_zero_is_dropped_baseline_covers_it(self):
        c = SteeringRecorderService.create_config(_cfg(dials=[0.0, 0.5]))
        assert c["dials"] == [0.5]   # 0.0 dropped — baseline recorded separately

    def test_all_zero_dials_rejected(self):
        with pytest.raises(RecordConfigError, match="above 0"):
            SteeringRecorderService.create_config(_cfg(dials=[0.0]))


class TestConfigCaps:
    def test_rejects_unknown_artifact(self):
        with pytest.raises(RecordConfigError, match="artifact"):
            SteeringRecorderService.create_config(_cfg(artifact={"kind": "bogus"}))

    def test_rejects_empty_dials_or_prompts(self):
        with pytest.raises(RecordConfigError, match="dials"):
            SteeringRecorderService.create_config(_cfg(dials=[]))
        with pytest.raises(RecordConfigError, match="prompts"):
            SteeringRecorderService.create_config(_cfg(prompts=[]))

    def test_dedupes_dials(self):
        c = SteeringRecorderService.create_config(_cfg(dials=[0.5, 0.5, 1.0]))
        assert c["dials"] == [0.5, 1.0]

    def test_rejects_dial_above_ceiling(self):
        with pytest.raises(RecordConfigError, match="servable ceiling"):
            SteeringRecorderService.create_config(_cfg(dials=[2.5]))

    def test_rejects_too_many_dials_or_prompts(self):
        with pytest.raises(RecordConfigError, match="at most 8 dials"):
            SteeringRecorderService.create_config(_cfg(dials=[0.1*i for i in range(1, 10)]))
        with pytest.raises(RecordConfigError, match="at most 8 prompts"):
            SteeringRecorderService.create_config(_cfg(prompts=[f"p{i}" for i in range(9)]))

    def test_rejects_out_of_range_max_tokens(self):
        with pytest.raises(RecordConfigError, match="max_tokens"):
            SteeringRecorderService.create_config(_cfg(max_tokens=999))

    def test_hard_product_cap(self):
        # 8 prompts × (1 + 8 dials) = 72 > 64.
        with pytest.raises(RecordConfigError, match="exceeds the 64"):
            SteeringRecorderService.create_config(_cfg(
                prompts=[f"p{i}" for i in range(8)],
                dials=[0.1*i for i in range(1, 9)]))

    def test_empty_prompt_string_rejected(self):
        with pytest.raises(RecordConfigError, match="non-empty"):
            SteeringRecorderService.create_config(_cfg(prompts=["ok", "   "]))


class TestOrchestration:
    def test_records_baseline_once_and_steered_per_dial(self, monkeypatch):
        import src.services.steering_recorder_service as mod

        # Inject the GPU pieces.
        monkeypatch.setattr(mod, "load_model_and_structure",
                            lambda mid, db: ("MODEL", "TOK", "STRUCT", False, "cpu"))
        monkeypatch.setattr(SteeringRecorderService, "_resolve",
                            classmethod(lambda cls, art, db, dev: ("m_x", [(1, 1, 1.0, "W")])))
        monkeypatch.setattr(SteeringRecorderService, "_artifact_model_id",
                            staticmethod(lambda art, db: "m_x"))
        monkeypatch.setattr(SteeringRecorderService, "_model_hf_id",
                            staticmethod(lambda mid, db: "org/model"))

        def fake_build(model, tok, struct, resolved, *, disable_cache, max_tokens):
            def gen_at(dial, prompt):
                return f"STEER@{dial}:{prompt}"
            def baseline_at(prompt, seed):
                return f"BASE:{prompt}"
            return gen_at, baseline_at
        monkeypatch.setattr(mod, "build_steer_generator", fake_build)

        persisted = {}
        monkeypatch.setattr(SteeringRecorderService, "_persist",
                            staticmethod(lambda db, art, payload: persisted.update(payload) or "vman_rec1"))

        out = SteeringRecorderService.record_samples(
            db=None, config=_cfg(dials=[0.5, 1.0], prompts=["a", "b"]))

        assert out["manifest_ref"] == "vman_rec1"
        assert out["counts"] == {"prompts": 2, "dials": 2, "generations": 6}
        # payload shape: baseline once per prompt, steered per dial.
        t = persisted["transcripts"]
        assert len(t) == 2
        assert t[0]["unsteered_output"] == "BASE:a"
        assert [s["dial"] for s in t[0]["samples"]] == [0.5, 1.0]
        assert t[0]["samples"][0]["steered_output"] == "STEER@0.5:a"
        assert persisted["artifact"] == {"kind": "circuit", "circuit_id": "crc_x"}
        assert persisted["config"]["model_hf_id"] == "org/model"

    def test_dial_zero_reuses_the_baseline_not_a_second_gen(self, monkeypatch):
        import src.services.steering_recorder_service as mod
        monkeypatch.setattr(mod, "load_model_and_structure",
                            lambda mid, db: ("M", "T", "S", False, "cpu"))
        monkeypatch.setattr(SteeringRecorderService, "_resolve",
                            classmethod(lambda cls, art, db, dev: ("m", [(1, 1, 1.0, "W")])))
        monkeypatch.setattr(SteeringRecorderService, "_artifact_model_id",
                            staticmethod(lambda art, db: "m"))
        monkeypatch.setattr(SteeringRecorderService, "_model_hf_id",
                            staticmethod(lambda mid, db: "o/m"))
        calls = {"gen": 0}
        def fake_build(*a, **k):
            def gen_at(dial, prompt):
                calls["gen"] += 1
                return f"S@{dial}"
            def baseline_at(prompt, seed):
                return "BASE"
            return gen_at, baseline_at
        monkeypatch.setattr(mod, "build_steer_generator", fake_build)
        monkeypatch.setattr(SteeringRecorderService, "_persist",
                            staticmethod(lambda db, art, p: "vman_x"))

        SteeringRecorderService.record_samples(
            db=None, config=_cfg(dials=[0.0, 1.0], prompts=["a"]))
        # dial 0.0 must reuse the baseline; only dial 1.0 calls gen_at.
        assert calls["gen"] == 1


class TestClusterModelIdResolution:
    """The recorder loads the model via _artifact_model_id BEFORE _resolve, so
    that path must ALSO derive a cluster's model from its SAE (CLUSTERS-arc
    profiles store sae_id but model_id=None). Fixing only steering_core left this
    parallel copy returning None → 'Model None not found' on hardware."""

    def _db(self, prof, sae):
        class _Q:
            def __init__(self, obj):
                self._obj = obj

            def filter(self, *a, **k):
                return self

            def first(self):
                return self._obj

        class _DB:
            def query(self, model):
                name = getattr(model, "__name__", "")
                return _Q(prof if name == "ClusterProfile" else sae)
        return _DB()

    def test_cluster_model_id_derived_from_sae(self):
        class _Prof:
            id = "clp_z"
            model_id = None            # real-data shape
            mistudio_sae_id = None
            sae_id = "sae_real"
            saes = None
            members = [{"feature_idx": 3, "strength": 0.2}]

        class _SAE:
            model_id = "m_derived"
            layer = 13

        mid = SteeringRecorderService._artifact_model_id(
            {"kind": "cluster", "cluster_profile_id": "clp_z"},
            self._db(_Prof(), _SAE()))
        assert mid == "m_derived"

    def test_record_run_raises_on_model_id_path_mismatch(self, monkeypatch):
        """The two resolution paths must agree — a divergence is a loud error,
        not a silent steer against the wrong model."""
        import src.services.steering_recorder_service as mod
        from src.services.steering_recorder_service import RecordRunError

        monkeypatch.setattr(mod, "load_model_and_structure",
                            lambda mid, db: ("M", "T", "S", False, "cpu"))
        monkeypatch.setattr(SteeringRecorderService, "_artifact_model_id",
                            staticmethod(lambda art, db: "m_loaded"))
        # _resolve disagrees → must raise, never generate.
        monkeypatch.setattr(SteeringRecorderService, "_resolve",
                            classmethod(lambda cls, art, db, dev: ("m_OTHER", [(1, 1, 1.0, "W")])))
        monkeypatch.setattr(mod, "build_steer_generator",
                            lambda *a, **k: (lambda d, p: "x", lambda p, s: "b"))

        with pytest.raises(RecordRunError, match="mismatch"):
            SteeringRecorderService.record_samples(
                db=None, config=_cfg(artifact={"kind": "cluster",
                                               "cluster_profile_id": "clp_z"},
                                     dials=[1.0], prompts=["a"]))


class TestManifestKind:
    def test_steering_samples_kind_validates(self):
        from src.services.manifest_service import validate_payload
        good = {"artifact": {"kind": "circuit", "circuit_id": "c"},
                "dials": [0.5], "prompts": ["p"],
                "transcripts": [{"prompt_index": 0, "unsteered_output": "u",
                                 "samples": [{"dial": 0.5, "steered_output": "s"}]}],
                "config": {"max_tokens": 80, "seed": 0}}
        validate_payload("steering_samples", good)  # no raise

    def test_missing_key_rejected(self):
        from src.services.manifest_service import ManifestError, validate_payload
        with pytest.raises(ManifestError, match="steering_samples"):
            validate_payload("steering_samples", {"dials": [0.5]})

    def test_free_text_may_contain_pathlike_strings(self):
        """R1: a prompt or GENERATION that starts with /data//home/ is legit free
        text, NOT an internal ref — it must not discard a completed GPU run."""
        from src.services.manifest_service import validate_payload
        ok = {"artifact": {"kind": "circuit", "circuit_id": "c"},
              "dials": [0.5], "prompts": ["/home/user please advise"], "config": {},
              "transcripts": [{"unsteered_output": "/data/ is where I keep things",
                               "samples": [{"dial": 0.5,
                                            "steered_output": "/data/x"}]}]}
        validate_payload("steering_samples", ok)  # must NOT raise

    def test_a_real_path_in_a_REF_field_is_still_rejected(self):
        """Specificity: the exemption is only for text fields; a path in a
        non-text field (e.g. an artifact ref) is still caught."""
        from src.services.manifest_service import ManifestError, validate_payload
        bad = {"artifact": {"kind": "circuit", "circuit_id": "/data/leak"},
               "dials": [0.5], "prompts": ["p"], "config": {},
               "transcripts": []}
        with pytest.raises(ManifestError, match="path"):
            validate_payload("steering_samples", bad)

    def test_a_path_NESTED_under_a_text_key_is_still_caught(self):
        """R2 hardening: the exemption is only for STRING values under a text
        key. A dict/list nested under a text-named key is still walked, so a ref
        can't hide inside it."""
        from src.services.manifest_service import ManifestError, validate_payload
        bad = {"artifact": {"kind": "circuit", "circuit_id": "c"},
               "dials": [0.5], "prompts": ["p"], "config": {},
               # 'generation' is a text key, but here its value is a DICT hiding
               # a path — the walker must not skip the whole subtree.
               "transcripts": [{"generation": {"ref": "/data/leak"}}]}
        with pytest.raises(ManifestError, match="path"):
            validate_payload("steering_samples", bad)
