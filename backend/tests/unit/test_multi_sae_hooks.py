"""Unit tests for Feature 015 multi-SAE hook threading + SAE-map resolution.

Pins (no GPU):
  * resolve_sae_map: per-feature ``sae_id`` falls back to the request id;
    distinct ids are all loaded; a missing SAE and a layer mismatch raise a
    structured error listing offenders.
  * _register_steering_hooks: each created hook receives the SAE whose ``.layer``
    equals its group's layer — the whole point of 015 (no wrong-basis steering).
  * regression: the single-SAE overload (a lone ``LoadedSAE``) behaves exactly
    as before — one hook per layer, all sharing that SAE.
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.services.steering_service import (
    SteeringService,
    LoadedSAE,
    SaeMeta,
    SaeLayerMismatchError,
    FeatureSteeringConfig,
)
from src.schemas.steering import CombinedSteeringRequest, SelectedFeature


def _svc():
    # Bypass __init__ (touches the GPU); we exercise pure/mocked methods only.
    s = SteeringService.__new__(SteeringService)
    s._loaded_saes = {}
    return s


def _loaded(layer, d_sae=8192):
    return LoadedSAE(model=object(), config=None, layer=layer, d_in=768,
                     d_sae=d_sae, device="cpu")


def _feat(feature_idx, layer, sae_id=None, strength=50.0):
    return SelectedFeature(feature_idx=feature_idx, layer=layer,
                           sae_id=sae_id, strength=strength)


def _request(features, sae_id="req_sae"):
    return CombinedSteeringRequest(
        sae_id=sae_id, prompt="hi", selected_features=features,
    )


def _meta(sae_id, layer):
    return SaeMeta(sae_id=sae_id, sae_path=f"/tmp/{sae_id}", layer=layer)


# ── resolve_sae_map ─────────────────────────────────────────────────────────

def test_resolve_sae_map_defaults_to_request_sae():
    """A feature with no sae_id routes to the request-level SAE."""
    svc = _svc()
    loaded = {"req_sae": _loaded(13)}

    async def fake_load_sae(path, sid, **kw):
        return loaded[sid]

    svc.load_sae = fake_load_sae
    req = _request([_feat(1, 13), _feat(2, 13)], sae_id="req_sae")
    meta = {"req_sae": _meta("req_sae", 13)}

    sae_map = asyncio.run(svc.resolve_sae_map(req, meta))
    assert set(sae_map) == {"req_sae"}


def test_resolve_sae_map_distinct_ids_all_loaded():
    """Distinct per-feature sae_ids are each loaded once."""
    svc = _svc()
    loaded = {"A": _loaded(13), "B": _loaded(14)}
    calls = []

    async def fake_load_sae(path, sid, **kw):
        calls.append(sid)
        return loaded[sid]

    svc.load_sae = fake_load_sae
    req = _request([_feat(1, 13, "A"), _feat(2, 14, "B"), _feat(3, 13, "A")])
    meta = {"A": _meta("A", 13), "B": _meta("B", 14)}

    sae_map = asyncio.run(svc.resolve_sae_map(req, meta))
    assert set(sae_map) == {"A", "B"}
    # De-duplicated: A loaded once despite two features.
    assert calls == ["A", "B"]


def test_resolve_sae_map_missing_meta_raises():
    svc = _svc()

    async def fake_load_sae(path, sid, **kw):
        return _loaded(13)

    svc.load_sae = fake_load_sae
    req = _request([_feat(1, 13, "A")])
    with pytest.raises(SaeLayerMismatchError):
        asyncio.run(svc.resolve_sae_map(req, {}))  # no meta for "A"


def test_resolve_sae_map_layer_mismatch_lists_offenders():
    """A feature on layer 13 routed to a layer-14 SAE raises, listing it."""
    svc = _svc()
    loaded = {"A": _loaded(14)}  # SAE A is actually layer 14

    async def fake_load_sae(path, sid, **kw):
        return loaded[sid]

    svc.load_sae = fake_load_sae
    req = _request([_feat(7, 13, "A"), _feat(9, 14, "A")])  # feat 7 wrong
    meta = {"A": _meta("A", 14)}

    with pytest.raises(SaeLayerMismatchError) as ei:
        asyncio.run(svc.resolve_sae_map(req, meta))

    offenders = ei.value.offenders
    assert len(offenders) == 1
    o = offenders[0]
    assert o["feature_idx"] == 7 and o["layer"] == 13
    assert o["sae_id"] == "A" and o["sae_layer"] == 14


# ── _register_steering_hooks: hook/SAE pairing pin ──────────────────────────

def _capture_hooks(svc):
    """Patch _create_steering_hook + _get_target_module to capture (sae, group)
    pairs without any torch/model. Returns the captured-pairs list."""
    captured = []

    def fake_create(sae, group):
        captured.append((sae, group))
        return lambda *a, **k: None

    class _Mod:
        def register_forward_hook(self, fn):
            return SimpleNamespace(remove=lambda: None)

    svc._create_steering_hook = fake_create
    svc._get_target_module = lambda model, layer: _Mod()
    return captured


def test_each_hook_gets_the_sae_matching_its_layer():
    """THE 015 invariant: every created hook receives the SAE whose .layer
    equals the group's layer."""
    svc = _svc()
    captured = _capture_hooks(svc)
    sae_map = {"A": _loaded(13), "B": _loaded(14)}
    configs = [
        FeatureSteeringConfig(feature_idx=1, layer=13, strength=50.0, sae_id="A"),
        FeatureSteeringConfig(feature_idx=2, layer=14, strength=50.0, sae_id="B"),
        FeatureSteeringConfig(feature_idx=3, layer=13, strength=50.0, sae_id="A"),
    ]

    svc._register_steering_hooks(object(), sae_map, configs, default_sae_id="A")

    assert len(captured) == 2  # one hook per (sae_id, layer) group
    for sae, group in captured:
        for cfg in group:
            assert sae.layer == cfg.layer  # right SAE for the layer


def test_default_sae_id_used_when_config_has_none():
    """A config with no sae_id routes to the request default."""
    svc = _svc()
    captured = _capture_hooks(svc)
    sae_map = {"req": _loaded(13)}
    configs = [FeatureSteeringConfig(feature_idx=1, layer=13, strength=50.0)]

    svc._register_steering_hooks(object(), sae_map, configs, default_sae_id="req")
    assert len(captured) == 1
    sae, group = captured[0]
    assert sae is sae_map["req"]


def test_single_sae_overload_is_grouped_by_layer_only():
    """Regression: passing a lone LoadedSAE (solo/compare/single-SAE combined)
    groups by layer and hands every hook that one SAE — unchanged behaviour."""
    svc = _svc()
    captured = _capture_hooks(svc)
    sae = _loaded(13)
    configs = [
        FeatureSteeringConfig(feature_idx=1, layer=13, strength=50.0),
        FeatureSteeringConfig(feature_idx=2, layer=13, strength=50.0),
        FeatureSteeringConfig(feature_idx=3, layer=20, strength=50.0),
    ]

    svc._register_steering_hooks(object(), sae, configs)

    # One hook per distinct layer (13, 20); each got the single SAE object.
    assert len(captured) == 2
    for got_sae, group in captured:
        assert got_sae is sae
