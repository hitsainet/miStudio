"""A split load hands accelerate a GPU-only budget and refuses anything mapped off the GPUs.

Multi-GPU Phase 2 (0xcc/plans/Multi-GPU-Plan.md): a model that fits no single card
loads with `device_map="sequential"` and `max_memory` = the split's per-card budget,
with no "cpu" key (operator decision 3: transformers loads run on GPUs only).
accelerate still keeps "disk" as a last resort, so a model the cards cannot hold
would load with layers read from disk on every forward pass, hours slower. The
loader refuses that instead. Nothing here needs a GPU: config, weights and
tokenizer are faked (the fake config cannot be built on the meta device, so the
pre-load map in ml/split_load.py is skipped here; it is tested in
test_split_load_maps_onto_the_gpus.py).

AN OUT-OF-MEMORY LOAD IS NEVER RETRIED AT ANOTHER FORMAT (review round 1,
2026-09-14). It used to fall back Q2 -> Q4 -> Q8 -> FP16 -> FP32 by default: every
retry needs more memory than the load that ran out, and one that succeeded gave a
training, an SAE extraction or a steering run a precision nobody asked for.

MUTATION CONTROLS (review round 1; scratchpad p2-r1-core/mutate.py, each alone,
this file run, source restored and checked by sha256):
  O1  the out-of-memory load is not surfaced              -> both out-of-memory tests
  O2  the failed attempt's memory is not released         -> both out-of-memory tests
  O4  a split's release ignores its budget's cards        -> an_out_of_memory_split_is_never_retried...
  O5  the FP32 retry reinstated (the defect)              -> both out-of-memory tests
  O6  a named card's release goes to every visible card   -> an_out_of_memory_load_on_one_card...
"""

import contextlib

import pytest
import torch

from src.ml import model_loader
from src.ml.model_loader import QuantizationFormat
from src.services import resource_config

SPLIT_BUDGET = {1: "20476MiB", 0: "10876MiB"}


class _Config:
    model_type = "gemma"
    hidden_size = 3584
    num_hidden_layers = 48
    vocab_size = 262_144
    intermediate_size = 14_336


class _Loaded(torch.nn.Module):
    def __init__(self, device_map):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.hf_device_map = device_map


class _TokenizerReached(Exception):
    """Raised by the fake tokenizer: the model load itself went through."""


@pytest.fixture
def loader(monkeypatch):
    calls = {"preflight": [], "load": [], "tokenizer": 0}
    state = {"maps": [{"model.embed_tokens": 1, "model.layers.0": 1, "model.layers.47": 0}], "errors": []}

    def preflight(**kwargs):
        calls["preflight"].append(kwargs)

    def from_pretrained(repo_id, **kwargs):
        calls["load"].append(kwargs)
        if state["errors"]:
            raise state["errors"].pop(0)
        return _Loaded(state["maps"].pop(0) if len(state["maps"]) > 1 else state["maps"][0])

    def tokenizer(*args, **kwargs):
        calls["tokenizer"] += 1
        raise _TokenizerReached("tokenizer reached")

    monkeypatch.setattr(model_loader.AutoConfig, "from_pretrained", lambda *a, **k: _Config())
    monkeypatch.setattr(model_loader, "extract_architecture_config", lambda config: {})
    monkeypatch.setattr(model_loader, "get_quantization_config", lambda fmt: None)
    monkeypatch.setattr(resource_config, "preflight_gpu_capacity", preflight)
    monkeypatch.setattr(model_loader.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(model_loader.AutoTokenizer, "from_pretrained", tokenizer)
    # The refusal releases each card's cache; there is no CUDA here.
    monkeypatch.setattr(torch.cuda, "device", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return calls, state


def test_a_split_passes_its_budget_and_preflights_only_its_cards(loader):
    calls, _ = loader

    with pytest.raises(Exception, match="tokenizer reached"):
        model_loader.load_model_from_hf("org/model", device_map="auto", max_memory=SPLIT_BUDGET)

    assert calls["load"][0]["device_map"] == "auto"
    assert calls["load"][0]["max_memory"] == SPLIT_BUDGET
    assert "cpu" not in calls["load"][0]["max_memory"]
    assert calls["preflight"][0]["device"] == (1, 0)


def test_a_default_load_is_unchanged(loader):
    calls, _ = loader

    with pytest.raises(Exception, match="tokenizer reached"):
        model_loader.load_model_from_hf("org/model", device_map="cuda:1")

    assert "max_memory" not in calls["load"][0]
    assert calls["preflight"][0]["device"] == "cuda:1"


@pytest.mark.parametrize("target", ["disk", "cpu"])
def test_a_split_mapped_off_the_gpus_is_refused_before_it_is_used(loader, target):
    calls, state = loader
    state["maps"] = [{"model.embed_tokens": 1, "model.layers.0": 0, "model.layers.47": target}]

    with pytest.raises(Exception, match="does not fit on the GPUs it was split across") as exc:
        model_loader.load_model_from_hf("org/model", device_map="auto", max_memory=SPLIT_BUDGET)

    assert f"model.layers.47 on {target}" in str(exc.value)
    assert calls["tokenizer"] == 0, "the refused model went on to load its tokenizer"


def test_a_single_card_load_is_not_checked_for_offload(loader):
    """No max_memory means the caller named one device; there is no split to check."""
    calls, state = loader
    state["maps"] = [{"": "cpu"}]

    with pytest.raises(Exception, match="tokenizer reached"):
        model_loader.load_model_from_hf("org/model", device_map="cpu")


@pytest.fixture
def released(monkeypatch):
    """The torch index of each card a failed load empties, in order; CUDA "exists" for the release only."""
    cards = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda index: cards.append(index) or contextlib.nullcontext())
    return cards


def test_an_out_of_memory_split_is_never_retried_at_another_format(loader, released):
    """Q4 used to be retried as Q8, FP16 as FP32: more memory than the load that ran out,
    and a model at a precision nobody asked for had one succeeded."""
    calls, state = loader
    state["errors"] = [RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")]

    with pytest.raises(model_loader.OutOfMemoryError, match="at Q4"):
        model_loader.load_model_from_hf(
            "org/model", quant_format=QuantizationFormat.Q4, device_map="sequential", max_memory=SPLIT_BUDGET
        )

    assert len(calls["load"]) == 1, "the out-of-memory load was retried"
    assert calls["tokenizer"] == 0
    assert released == [1, 0], "the failed split did not give back every card it used"


def test_an_out_of_memory_load_on_one_card_gives_that_card_back(loader, released):
    calls, state = loader
    state["errors"] = [torch.cuda.OutOfMemoryError("CUDA out of memory.")]

    with pytest.raises(model_loader.OutOfMemoryError, match="at FP16"):
        model_loader.load_model_from_hf("org/model", quant_format=QuantizationFormat.FP16, device_map="cuda:1")

    assert len(calls["load"]) == 1, "the out-of-memory load was retried"
    assert released == [1]
