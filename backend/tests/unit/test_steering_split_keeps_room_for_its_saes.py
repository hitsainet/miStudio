"""Steering on a split model: placed with its SAEs, and mapped so each layer's card holds its SAE.

Multi-GPU Phase 2, review round 2 (2026-09-14, steering reviewer). Three defects:

1. PLACEMENT OMITTED THE SAEs (`_placement_need_mb`). Every steering request loads its
   SAEs onto the model's card(s), but placement was asked for the model alone. A
   model just under one card's free memory was placed alone on that card and the
   SAEs then did not fit beside it. Calibration already counted its SAEs.
2. EACH LAYER'S SAE HAD NO ROOM ON ITS CARD. Hooks move each SAE to its layer's card;
   the split kept SHARD_RESERVE_MB there. The allowance goes through the planner call
   the placement + loader reviewer added to `load_model` (97fc6e29, which fixed
   steering's split never being mapped at all: measured here too on OLMo-2-1124-13B
   at FP16 with 7,000 + 21,400 MB free, steering's own map spilled 1,585 MiB to disk
   where the planned map loads; scratchpad p2-r2-steer/prove_maps.py).
3. ROUND 1's LIVE RE-BUDGET (756bd9cc) READ FREE MEMORY WITH THE SAEs STILL ON THE
   PLACEMENT'S FIRST CARD, charging them to the card they were about to leave while
   their layer's card kept only the reserve. They now leave the cards first.
Plus "all" on a one-card node (review item D): it unloaded and re-placed the model on
every request, onto the same card.

The fixtures split a model whose first card is cuda:1 while transformers fills cuda:0
first, place SAEs on two different layers with two different sizes, and put every
allowance on a layer the unallowed map gives a card without room for it.

MUTATION CONTROLS (review round 2, 2026-09-14; scratchpad p2-r2-steer/mutate.py with
mutations_w.json and mutations_merge.json, each applied alone, this module +
test_steering_gpu_placement.py + test_every_split_load_is_mapped_first.py run, source
restored and checked by sha256, `git status` unchanged after). All killed:
  B1 compare places without its SAEs          -> test_placement_is_asked_for_..._it_loads[compare]
  B2 sweep places without its SAE             -> ...[sweep]
  B3 combined places without its SAEs         -> ...[combined]
  B4 _placement_need_mb ignores sae_mb        -> all three placement cases
  B5 place_model does not forward sae_mb      -> all three placement cases
  B6 an SAE named twice is counted twice      -> TestTheRequestsSaes::test_each_distinct_sae_is_counted_once
  W1 compare's load gets no SAEs by layer     -> test_the_generation_path_hands_each_layers_sae_to_the_load[compare]
  W2 sweep's load gets no SAEs by layer       -> ...[sweep]
  W3 combined's load gets no SAEs by layer    -> ...[combined]
  W4 _plan_split passes no allowance          -> test_a_split_is_budgeted_with_the_saes_off_the_cards_...,
                                                 test_with_the_real_planner_a_layers_sae_moves_its_layer_...
  W5 load_model's planner call drops the SAEs -> the same two
  W6 the SAEs stay on the cards while the budget is read -> test_a_split_is_budgeted_with_the_saes_off_...
  W7 the moved SAEs' cache is not emptied before the read -> the same
  W8 a dimension-sized SAE counted at 4 bytes, not fp16  -> the same two
  D1 "all" never satisfied by one card        -> test_all_on_a_one_card_node_keeps_the_model_on_that_card
The placement + loader reviewer's controls on the steering lines this re-applied onto
(97fc6e29's P6 plan budget not passed, P7 planner call removed (its line now carries
the SAEs), P8 refusal swallowed, and round 1's R-M10 split load without max_memory and
R-N1 budget not re-read): all killed.

ROUND 1's PINNED SPLIT LINES (6fbc8bb3) RE-RUN on the branch re-based onto c78f0f21
(scratchpad p2-r2-steer/mutations_r1p.json, each alone, restored by sha256, `git diff
HEAD` empty after). All killed:
  R1-P1  "all" satisfied by any split   -> test_all_is_not_satisfied_by_a_split_that_misses_a_visible_card,
                                           test_all_on_a_one_card_node_keeps_the_model_on_that_card (item D's line)
  R1-P2  parameter-level stray check removed -> test_a_parameter_off_the_placements_cards_is_refused_...
  R1-P3  _is_split ignores the device map    -> test_a_dispatched_model_with_no_recorded_placement_...
  R1-P28 a refused split emptied on its first card only -> test_a_refused_split_empties_every_card_...
  R1-P20a/b circuit_required_mb ignores or misreads the row's precision -> test_circuits_run_split TestTheSize (8 each)
  R1-P32 module_device on a non-Module hook target -> test_multi_sae_hooks.py (3)
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config
from transformers.integrations.accelerate import _get_device_map, compute_module_sizes

from src.ml.split_load import SplitDoesNotFit
from src.services import gpu_placement
from src.services import steering_service as steering_module
from src.services.gpu_placement import GpuCard, GpuPlacementError, Placement
from src.services.steering_service import LoadedSAE, SteeringService, request_sae_mb

MIB = 1024**2
TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000)
RTX = GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000)
CUDA0, CUDA1, CPU = torch.device("cuda", 0), torch.device("cuda", 1), torch.device("cpu")

#: The 3090 first (most free); transformers still fills cuda:0 first.
SPLIT = Placement(card=RTX, device=CUDA1, cards=(RTX, TI), devices=(CUDA1, CUDA0),
                  max_memory_mb={1: 21_976, 0: 9_976})
ONE_CARD = Placement(card=RTX, device=CUDA1)

#: ~1.34B parameters at FP16 + 2 GiB headroom = 4,608.0 MiB (worked in
#: test_steering_gpu_placement.py).
SMALL_CONFIG = SimpleNamespace(hidden_size=2048, num_hidden_layers=16, vocab_size=65_536, intermediate_size=8_192)
SMALL_NEED_MB = 4_608.0
#: fp16 SAEs, worked by hand: (2 x d x n + d + n) values x 2 bytes.
#:   A: 4,096 x 16,384 -> 134,238,208 values -> 268,476,416 B
#:   B: 4,096 x 65,536 -> 536,940,544 values -> 1,073,881,088 B
SAE_A_MB = 268_476_416 / MIB
SAE_B_MB = 1_073_881_088 / MIB


class _Stop(Exception):
    """Raised by a fake once it has recorded what it was handed."""


def _loaded_sae(name, layer, d_in, d_sae, events=None):
    model = MagicMock(name=name)
    if events is not None:
        model.to.side_effect = lambda device: events.append(("to", name, str(device)))
    return LoadedSAE(model=model, config=None, layer=layer, d_in=d_in, d_sae=d_sae, device=str(CUDA1))


@pytest.fixture
def cuda(monkeypatch):
    state = SimpleNamespace(events=[], free_mb={0: 11_000, 1: 23_000})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(uuid=[TI_UUID, RTX_UUID][index][len("GPU-"):]))
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())

    def mem_get_info(device=None):
        index = device.index if isinstance(device, torch.device) else device
        state.events.append(("free", index))
        return state.free_mb[index] * MIB, 24_576 * MIB

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(steering_module, "empty_cache_on",
                        lambda devices: state.events.append(("emptied", sorted(str(d) for d in devices))))
    return state


def _service(monkeypatch) -> SteeringService:
    svc = SteeringService()
    monkeypatch.setattr(svc, "_clear_all_model_hooks", lambda model: 0)
    monkeypatch.setattr(svc, "_reset_model_state", lambda model: None)
    return svc


# ── 1. Placement is sized with the request's SAEs ─────────────────────────


REQUESTS = {
    "compare": ("generate_comparison_sync", {
        "sae_id": "sae_A", "model_id": "m", "prompt": "hi",
        "selected_features": [{"feature_idx": 1, "layer": 3, "strength": 5},
                              {"feature_idx": 2, "layer": 7, "strength": 5, "sae_id": "sae_B"}],
        "include_unsteered": False, "compute_metrics": False}),
    "sweep": ("generate_strength_sweep_sync", {
        "sae_id": "sae_A", "model_id": "m", "prompt": "hi",
        "feature_idx": 1, "layer": 3, "strength_values": [1.0, 2.0]}),
    "combined": ("generate_combined_sync", {
        "sae_id": "sae_A", "model_id": "m", "prompt": "hi",
        "selected_features": [{"feature_idx": 1, "layer": 3, "strength": 5},
                              {"feature_idx": 2, "layer": 7, "strength": 5, "sae_id": "sae_B"}],
        "include_baseline": False, "compute_metrics": False}),
}
META_B = {"sae_B": {"sae_id": "sae_B", "sae_path": "/nonexistent/b", "layer": 7,
                    "d_model": 4_096, "n_features": 65_536}}


def _run(svc, kind):
    method, request_dict = REQUESTS[kind]
    kwargs = dict(request_dict=dict(request_dict), sae_path="/nonexistent/a", model_id="m",
                  sae_layer=3, sae_d_model=4_096, sae_n_features=16_384)
    if kind != "sweep":
        kwargs["sae_meta_map"] = META_B
    return getattr(svc, method)(**kwargs)


@pytest.mark.parametrize("kind, expected", [
    ("compare", SMALL_NEED_MB + SAE_A_MB + SAE_B_MB),
    ("sweep", SMALL_NEED_MB + SAE_A_MB),
    ("combined", SMALL_NEED_MB + SAE_A_MB + SAE_B_MB),
])
def test_placement_is_asked_for_the_model_and_every_sae_it_loads(monkeypatch, cuda, kind, expected):
    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: SMALL_CONFIG)
    asked = []

    def place_job(requested="auto", required_mb=None, cards=None, allow_shard=False):
        asked.append(required_mb)
        raise GpuPlacementError("stop after placement")

    monkeypatch.setattr(steering_module, "place_job", place_job)
    svc = _service(monkeypatch)

    with pytest.raises(GpuPlacementError, match="stop after placement"):
        _run(svc, kind)

    assert asked == [pytest.approx(expected, abs=1e-6)]


class TestTheRequestsSaes:
    def test_each_distinct_sae_is_counted_once(self):
        meta = {sid: steering_module.SaeMeta(**m) for sid, m in META_B.items()}
        assert request_sae_mb("sae_A", 4_096, 16_384, ["sae_B", "sae_B", None, "sae_A"], meta) == pytest.approx(
            SAE_A_MB + SAE_B_MB, abs=1e-6)

    def test_an_sae_of_unknown_size_counts_nothing(self):
        assert request_sae_mb("sae_A", None, None, ["sae_Z"], None) == 0.0

    def test_a_loaded_sae_counts_its_own_tensors_at_the_precision_it_holds(self):
        from src.ml.sparse_autoencoder import create_sae

        sae = create_sae("standard", hidden_dim=64, latent_dim=256, normalize_activations="none")
        full = SteeringService._steering_sae_mb(LoadedSAE(sae, None, 3, 64, 256, "cpu"))
        half = SteeringService._steering_sae_mb(LoadedSAE(sae.half(), None, 3, 64, 256, "cpu"))
        assert full > (2 * 64 * 256) * 4 / MIB
        assert half == pytest.approx(full / 2)


# ── 2. Every generation path hands its SAEs, by layer, to the load ─────────


@pytest.mark.parametrize("kind", sorted(REQUESTS))
def test_the_generation_path_hands_each_layers_sae_to_the_load(monkeypatch, cuda, kind):
    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: SMALL_CONFIG)
    monkeypatch.setattr(steering_module, "place_job",
                        lambda requested="auto", required_mb=None, cards=None, allow_shard=False: SPLIT)
    svc = _service(monkeypatch)
    sae_a, sae_b = _loaded_sae("A", 3, 4_096, 16_384), _loaded_sae("B", 7, 4_096, 65_536)

    async def load_sae(*args, **kwargs):
        return sae_a

    async def resolve_sae_map(request, meta_map, force_reload=False):
        return {"sae_A": sae_a, "sae_B": sae_b}

    handed = []

    async def load_model(model_id, model_path=None, force_reload=False, gpu="auto", placement=None, **kwargs):
        handed.append(kwargs)
        raise _Stop

    monkeypatch.setattr(svc, "load_sae", load_sae)
    monkeypatch.setattr(svc, "resolve_sae_map", resolve_sae_map)
    monkeypatch.setattr(svc, "load_model", load_model)

    with pytest.raises(_Stop):
        _run(svc, kind)

    expected = {3: [sae_a]} if kind == "sweep" else {3: [sae_a], 7: [sae_b]}
    assert len(handed) == 1
    assert set(handed[0]) == {"saes_by_layer"}
    got = handed[0]["saes_by_layer"]
    assert {layer: [id(s) for s in group] for layer, group in got.items()} == {
        layer: [id(s) for s in group] for layer, group in expected.items()}


# ── 3. The split load: SAEs off the cards, budget read, split mapped around them ──


class _SplitModel:
    config = SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])


@pytest.fixture
def loading(monkeypatch, cuda):
    """load_model with the split planner and from_pretrained recorded."""
    import transformers

    state = SimpleNamespace(plans=[], loads=[], plan_result=SimpleNamespace(max_memory={1: "20000MiB", 0: "7000MiB"}))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: SMALL_CONFIG)

    def plan_split_load(config, **kwargs):
        state.plans.append(kwargs)
        cuda.events.append(("plan",))
        if isinstance(state.plan_result, Exception):
            raise state.plan_result
        return state.plan_result

    def from_pretrained(path, **kwargs):
        state.loads.append(kwargs)
        raise _Stop

    monkeypatch.setattr("src.ml.split_load.plan_split_load", plan_split_load)
    monkeypatch.setattr(steering_module.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(steering_module.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(pad_token="<pad>", eos_token="</s>"))
    return state


def test_a_split_is_budgeted_with_the_saes_off_the_cards_and_mapped_around_each_layers_sae(monkeypatch, cuda, loading):
    svc = _service(monkeypatch)
    sae_a = _loaded_sae("A", 3, 4_096, 16_384, cuda.events)
    sae_b = _loaded_sae("B", 7, 4_096, 65_536, cuda.events)

    with pytest.raises(_Stop):
        asyncio.run(svc.load_model("big", placement=SPLIT, force_reload=True,
                                   saes_by_layer={3: [sae_a], 7: [sae_b]}))

    first_read = cuda.events.index(("free", 1))
    moved = [e for e in cuda.events[:first_read] if e[0] == "to"]
    assert moved == [("to", "A", "cpu"), ("to", "B", "cpu")], cuda.events
    assert ("emptied", ["cuda:0", "cuda:1"]) in cuda.events[:first_read], cuda.events
    assert len(loading.plans) == 1
    assert loading.plans[0]["max_memory"] == {1: "21976MiB", 0: "9976MiB"}
    assert loading.plans[0]["extra_mb_by_layer"] == {3: pytest.approx(SAE_A_MB, abs=1e-6),
                                                     7: pytest.approx(SAE_B_MB, abs=1e-6)}
    assert [call["max_memory"] for call in loading.loads] == [{1: "20000MiB", 0: "7000MiB"}]


def test_a_split_the_cards_cannot_hold_beside_its_saes_is_refused_before_a_weight_is_read(monkeypatch, cuda, loading):
    loading.plan_result = SplitDoesNotFit("big does not fit on the GPUs it was split across beside 1,024 MiB",
                                          mapped_mb={}, limit_mb={})
    svc = _service(monkeypatch)

    with pytest.raises(RuntimeError, match="does not fit on the GPUs"):
        asyncio.run(svc.load_model("big", placement=SPLIT, force_reload=True,
                                   saes_by_layer={3: [_loaded_sae("A", 3, 4_096, 16_384)]}))

    assert loading.loads == []


def test_a_split_that_cannot_be_mapped_here_loads_with_the_placements_budget(monkeypatch, cuda, loading):
    loading.plan_result = None
    svc = _service(monkeypatch)

    with pytest.raises(_Stop):
        asyncio.run(svc.load_model("big", placement=SPLIT, force_reload=True))

    assert [call["max_memory"] for call in loading.loads] == [{1: "21976MiB", 0: "9976MiB"}]


def test_a_single_card_load_moves_no_sae_and_maps_nothing(monkeypatch, cuda, loading):
    svc = _service(monkeypatch)
    sae = _loaded_sae("A", 3, 4_096, 16_384, cuda.events)

    with pytest.raises(_Stop):
        asyncio.run(svc.load_model("m", placement=ONE_CARD, force_reload=True, saes_by_layer={3: [sae]}))

    assert [e for e in cuda.events if e[0] in ("to", "plan", "emptied")] == []
    assert loading.loads[0]["device_map"] == {"": CUDA1} and "max_memory" not in loading.loads[0]


def _qwen(vocab=64_000):
    return Qwen2Config(vocab_size=vocab, hidden_size=4_096, intermediate_size=11_008, num_hidden_layers=8,
                       num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=False)


def test_with_the_real_planner_a_layers_sae_moves_its_layer_to_a_card_with_room(monkeypatch, cuda):
    """End to end over transformers' real map inference: the steering load receives a
    budget under which layer 2 — whose SAE its card cannot also hold — maps elsewhere."""
    import transformers

    config = _qwen()
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(copy.deepcopy(config), dtype=torch.float16)
    sizes = {name: size / MIB for name, size in compute_module_sizes(skeleton, None, only_modules=False)[0].items()}
    layer = sizes["model.layers.0"]
    budget0 = math.floor(sizes["model.embed_tokens"] + 3.5 * layer)
    budget1 = math.floor(sizes[""] - sizes["model.embed_tokens"] - layer)
    cuda.free_mb = {0: budget0 + gpu_placement.SHARD_RESERVE_MB, 1: budget1 + gpu_placement.SHARD_RESERVE_MB}
    # An fp16 SAE of ~0.8 of a layer: (2 x 4,096 x 17,280 + 4,096 + 17,280) x 2 B = 283,158,272 B.
    sae = _loaded_sae("A", 2, 4_096, 17_280)
    sae_mb = 283_158_272 / MIB

    maps = []

    def from_pretrained(path, **kwargs):
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(copy.deepcopy(config), dtype=kwargs["torch_dtype"])
        maps.append(_get_device_map(model, kwargs["device_map"], dict(kwargs["max_memory"]), None))
        raise _Stop

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: copy.deepcopy(config))
    monkeypatch.setattr("src.services.resource_config.preflight_gpu_capacity", lambda **kwargs: None)
    monkeypatch.setattr(steering_module.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(steering_module.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(pad_token="<pad>", eos_token="</s>"))
    placement = Placement(card=RTX, device=CUDA1, cards=(RTX, TI), devices=(CUDA1, CUDA0),
                          max_memory_mb={1: budget1, 0: budget0})
    svc = _service(monkeypatch)

    with pytest.raises(_Stop):
        asyncio.run(svc.load_model("big", placement=placement, force_reload=True))
    unallowed = maps.pop()
    owner = lambda m, n: m[max((k for k in m if k == "" or n == k or n.startswith(k + ".")), key=len)]  # noqa: E731
    assert owner(unallowed, "model.layers.2") == 0, "precondition: without the SAE, layer 2 maps to cuda:0"
    on0 = sum(sizes.get(name, 0) for name, device in unallowed.items() if device == 0)
    assert on0 + sae_mb > budget0, "precondition: its SAE does not fit beside it there"

    with pytest.raises(_Stop):
        asyncio.run(svc.load_model("big", placement=placement, force_reload=True, saes_by_layer={2: [sae]}))
    final = maps.pop()

    assert owner(final, "model.layers.2") == 1, final
    assert not {d for d in final.values() if not isinstance(d, int)}, final
    on1 = sum(sizes.get(name, 0) for name, device in final.items() if device == 1)
    assert on1 + sae_mb <= budget1


# ── 4. "all" on a node with one card ─────────────────────────────────────


def test_all_on_a_one_card_node_keeps_the_model_on_that_card(monkeypatch, cuda):
    """The only card IS every card. Treated as unsatisfied, each request asking for "all"
    unloaded the model and placed it again, on the same card."""
    monkeypatch.setattr(steering_module, "list_cards", lambda: [RTX])
    placed, unloaded = [], []
    monkeypatch.setattr(steering_module, "place_job",
                        lambda requested="auto", required_mb=None, cards=None, allow_shard=False:
                        placed.append(requested) or ONE_CARD)
    svc = _service(monkeypatch)
    svc._loaded_models["m"] = (object(), object())
    svc._model_placements["m"] = ONE_CARD
    monkeypatch.setattr(svc, "unload_model", lambda model_id: unloaded.append(model_id) or True)

    assert svc.place_model("m", "all") is ONE_CARD
    assert placed == [] and unloaded == []


def test_all_on_a_two_card_node_is_not_satisfied_by_one_card(monkeypatch, cuda):
    monkeypatch.setattr(steering_module, "list_cards", lambda: [TI, RTX])

    assert SteeringService._names_card("all", ONE_CARD) is False
    assert SteeringService._names_card("all", SPLIT) is True
