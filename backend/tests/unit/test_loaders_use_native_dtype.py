"""The shared loader hands ONE resolved dtype to the split plan, the bnb config and the load.

`test_no_hard_coded_load_dtype.py` proves no literal reaches a load. This proves the BEHAVIOUR:
given a checkpoint that records bfloat16 (or float16), every consumer of the dtype receives that
dtype, and the loader's metadata — which every artifact records from — says so. A split planned at
one dtype and loaded at another is how a model spills; a bnb compute dtype different from
`torch_dtype` leaves a model internally mixed.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from src.ml import model_loader, split_load
from src.ml.model_loader import QuantizationFormat
from src.ml.native_dtype import load_dtype_record
from src.services import resource_config


class _Loaded(torch.nn.Module):
    def __init__(self, device_map):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.hf_device_map = device_map
        self.config = SimpleNamespace(_attn_implementation="sdpa")


@pytest.fixture
def seen(monkeypatch):
    calls = {"quant": [], "plan": [], "load": []}
    state = {"config": None}

    def get_quantization_config(fmt, compute_dtype=None):
        calls["quant"].append((fmt, compute_dtype))
        return None

    def plan_split_load(config, **kwargs):
        calls["plan"].append(kwargs)
        return None

    def from_pretrained(repo_id, **kwargs):
        calls["load"].append(kwargs)
        return _Loaded({"model.embed_tokens": 0, "model.layers.0": 1})

    monkeypatch.setattr(model_loader.AutoConfig, "from_pretrained", lambda *a, **k: state["config"])
    monkeypatch.setattr(model_loader, "extract_architecture_config", lambda config: {})
    monkeypatch.setattr(model_loader, "get_quantization_config", get_quantization_config)
    monkeypatch.setattr(split_load, "plan_split_load", plan_split_load)
    monkeypatch.setattr(resource_config, "preflight_gpu_capacity", lambda **k: None)
    monkeypatch.setattr(model_loader.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(model_loader.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(torch.cuda, "device", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return calls, state


def _config(dtype):
    return SimpleNamespace(model_type="llama", hidden_size=64, num_hidden_layers=2,
                           vocab_size=100, intermediate_size=128, torch_dtype=dtype)


@pytest.mark.parametrize("recorded,expected", [("bfloat16", torch.bfloat16), ("float16", torch.float16)])
@pytest.mark.parametrize("fmt", [QuantizationFormat.FP16, QuantizationFormat.Q4])
def test_one_dtype_reaches_the_plan_the_bnb_config_and_the_load(seen, recorded, expected, fmt):
    calls, state = seen
    state["config"] = _config(recorded)

    model, _tok, _cfg, metadata = model_loader.load_model_from_hf(
        "org/model", quant_format=fmt, device_map="sequential", max_memory={0: "10GiB", 1: "10GiB"},
    )

    assert calls["quant"] == [(fmt, expected)]
    assert [c["dtype"] for c in calls["plan"]] == [expected]
    assert [c["torch_dtype"] for c in calls["load"]] == [expected]
    assert metadata["model_dtype"] == recorded
    assert metadata["checkpoint_dtype"] == recorded
    assert metadata["dtype_source"] == "config"
    assert load_dtype_record(model)["model_dtype"] == recorded


def test_an_fp32_row_loads_float32_whatever_the_checkpoint_records(seen):
    calls, state = seen
    state["config"] = _config("bfloat16")

    _m, _t, _c, metadata = model_loader.load_model_from_hf("org/model", quant_format=QuantizationFormat.FP32)

    assert [c["torch_dtype"] for c in calls["load"]] == [torch.float32]
    assert metadata["model_dtype"] == "float32" and metadata["activation_storage_dtype"] == "float32"


def test_a_checkpoint_recording_nothing_loads_bfloat16_and_says_it_was_not_recorded(seen):
    calls, state = seen
    state["config"] = SimpleNamespace(model_type="llama", hidden_size=64, num_hidden_layers=2,
                                      vocab_size=100, intermediate_size=128)

    _m, _t, _c, metadata = model_loader.load_model_from_hf("org/model", quant_format=QuantizationFormat.FP16)

    assert [c["torch_dtype"] for c in calls["load"]] == [torch.bfloat16]
    assert metadata["checkpoint_dtype"] is None and metadata["dtype_source"] == "default"
