"""
Unit tests for KV-cache gating in steered generation.

The steering hook adds a constant vector to each token's residual stream
independently, so a standard KV cache produces output identical to an uncached
forward — verified token-identical on LFM2 (~14x faster). Only Gemma-2's hybrid
sliding-window cache is incompatible with the hooks and must keep use_cache=False.
`_needs_cache_disabled` encodes that gate.
"""

import pytest

from src.services.steering_service import (
    SteeringService,
    _CACHE_INCOMPATIBLE_MARKERS,
)


class _Cfg:
    def __init__(self, model_type, architectures):
        self.model_type = model_type
        self.architectures = architectures


class _Model:
    def __init__(self, config):
        self.config = config


@pytest.fixture
def svc():
    # Bypass __init__ (which touches the GPU) — we only exercise a pure method.
    return SteeringService.__new__(SteeringService)


@pytest.mark.parametrize(
    "model_type,architectures,expected",
    [
        ("gemma2", ["Gemma2ForCausalLM"], True),       # hybrid cache → disable
        ("gemma-2", ["Gemma2ForCausalLM"], True),
        ("lfm2", ["Lfm2ForCausalLM"], False),          # standard cache → keep
        ("llama", ["LlamaForCausalLM"], False),
        ("qwen2", ["Qwen2ForCausalLM"], False),
        ("mistral", ["MistralForCausalLM"], False),
        ("", [], False),                                # unknown but has config
    ],
)
def test_needs_cache_disabled_by_arch(svc, model_type, architectures, expected):
    model = _Model(_Cfg(model_type, architectures))
    assert svc._needs_cache_disabled(model) is expected


def test_detects_gemma2_from_architectures_only(svc):
    # model_type absent but architecture name reveals gemma2
    model = _Model(_Cfg("", ["Gemma2ForCausalLM"]))
    assert svc._needs_cache_disabled(model) is True


def test_missing_config_is_safe_disable(svc):
    model = _Model(None)
    assert svc._needs_cache_disabled(model) is True


def test_markers_are_gemma2_only():
    # Guard against accidental broadening that would slow every model down.
    assert all("gemma" in m for m in _CACHE_INCOMPATIBLE_MARKERS)
