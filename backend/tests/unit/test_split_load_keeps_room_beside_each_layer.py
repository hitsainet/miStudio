"""A split keeps room for the SAE each layer's card will hold beside that layer.

Multi-GPU Phase 2, review round 2 (2026-09-14). Steering, the steering core
(calibration, the steered transcript recorder) and the circuit passes load each
layer's SAE onto the card that layer was mapped to. A split budget keeps only
SHARD_RESERVE_MB (1 GB) free per card, and transformers fills the lowest-index
card, the 12 GB 3080 Ti on the node, first. Measured with transformers' real map
inference on OLMo-2-1124-13B at FP16 over the node's usual free memory
(11,000 + 23,000 MB; scratchpad p2-r2-steer/prove_maps.py): the planned map puts
layers 0-13 on the 3080 Ti and leaves 525 MiB there, where layer 13's SAE goes —
1,280 MiB at fp16 (steering), 2,560 MiB at fp32 (the core and the circuits). The
job ran out of memory after the whole model had loaded.

`plan_split_load(extra_mb_by_layer=...)` charges each layer's extra MiB to the card
its layer maps to and cuts that card's budget until it holds both. These tests run
the REAL map inference over meta-device models (no weights, no GPU). Every fixture
is a split whose layers span both cards, and every allowance is on a layer the
unallowed map puts on a card WITHOUT room for it — asserted as a precondition, so
no case agrees with the defect by construction.

MUTATION CONTROLS (review round 2, 2026-09-14; scratchpad p2-r2-steer/mutate.py with
mutations_a.json, each alone, this file run, source restored and checked by sha256,
`git status` unchanged after). All killed:
  A1 the budget is never cut for the work beside a layer    -> the_allowance moves/every card/four-bit cases,
                                                               the_loader_maps_a_split_around_the_saes...
  A2 every allowance charged to the lowest-index card       -> moves_that_layer..., every_layers_sae...,
                                                               an_sae_on_a_card_with_room_changes_nothing
  A4 a card still overcommitted after the re-maps is handed on -> a_card_still_overcommitted_..._is_refused
  A5 the allowance is dropped (`extra = {}`)                 -> 8 tests, every allowance case
  A6 load_model_from_hf does not pass extra_mb_by_layer on   -> the_loader_maps_a_split_around_the_saes...
  A18 the re-map also runs for a split with NO allowance      -> SURVIVED, equivalent: the hold-back
      branch handles a model-only overcommit first, and once the hold-back is zero accelerate honours
      its limits, so a model-only excess cannot reach the re-map. Kept because of the finding below.
The callers' wiring controls (A7-A17) are recorded in test_circuits_run_split.py and
test_steering_gpu_placement.py.

ROUND 1 CONTROLS RE-RUN ON THE REWRITTEN LOOP (split_load.py lines this change touches;
this file + test_split_load_maps_onto_the_gpus.py, restored by sha256):
  R1-S1 reclaim = 0, R1-S3 sized without the quantizer, R1-S7 fill ignored, R1-S8 hold-back to the
  highest index, R1-S10 never-refuse branch removed, R1-S11 checkpoint quantizer ignored, and the
  loader's R1-S4 (placement budget passed, not the plan's), R1-S5 (refusal swallowed), R1-S9 (no plan):
  all killed.
  R1-S2 (the overcommit hold-back shrink disabled) SURVIVED the first version of this change, which
  re-mapped ANY overcommitted card: the allowance loop quietly took over the hold-back branch's job
  for splits with no allowance at all, so that branch stopped being load-bearing and every split's
  map could differ from before. Probed on c8fccbcd's own split_load.py + model_loader.py
  (scratchpad p2-r2-steer/s2_on_original.py): killed there. The re-map now runs only for a split
  carrying an allowance; R1-S2 is killed again (test_a_tied_embedding_does_not_overcommit...).

MERGED WITH 0af8f7ce (placement + loader reviewer: a broken planner logs an ERROR). The
conflict was this allowance inside the single try that commit split into "will not build
on meta" (warning) and "internals failed" (error). The allowance's layer lookup is its own
try, reported as unverifiable: finding no decoder layers is a property of the model, not
a broken library. That commit's L1-L4 and its re-runs of R-S1/S3/S7/S10 were re-run on the
resolved file (this module + test_split_planner_breakage_is_loud.py +
test_split_load_maps_onto_the_gpus.py): all killed.

REVIEW ROUND 3 (2026-09-14). Round 2's re-map cut an overcommitted card by its WHOLE excess,
the SAEs' bytes included, so it moved the SAEs' worth of model off the card too and refused a
split that fits: OLMo-2-1124-13B at FP16 on 11,000 + 23,000 MB free with 1,280 MiB beside layers
11-13 (scratchpad p2-r3/probe_overshoot.py found the fitting map with transformers' own
inference; p2-r3/probe_olmo.py shows the refusal before the fix and the plan after). Now the
lowest overcommitted card gives up only the modules at the end of its run that must leave, and
`working_mb_by_layer` charges transient work as a per-card maximum.
Controls (scratchpad p2-r3/mutate.py + mutations_r3.json, each alone, in a private copy of
backend/ with its own test database, restored by sha256 and the copy's tree hash checked):
  N1  the round-2 rule restored (every card cut by its excess) -> the_layers_that_leave_take_their_saes...,
                                                                  the_same_split_is_planned_the_same_way_twice
  N2  the SAEs of leaving modules not taken off with them      -> the same two
  N3  working room summed per card, not its maximum            -> a_card_keeps_room_for_its_largest_layers_work...
  N4  working room left out of which cards are overcommitted   -> work_beside_a_layer..., the_loader_maps_a_split_around_the_work...
  N5  load_model_from_hf does not pass working_mb_by_layer on   -> the_loader_maps_a_split_around_the_work...
  N6  the unused-card warning removed                           -> a_card_left_with_none_of_the_model_is_named
  N6b the minimal move counts every card's modules              -> 9 tests
  N6c the minimal move ignores the working room                 -> work_beside_a_layer..., the_loader_maps_..._work...
All killed. Round 2's controls re-run on the rewritten loop: A1, A2 (re-expressed as
`device = lowest` for the owner-keyed charge), A4, A5, A6; 0af8f7ce's L1-L4 and R-S1/S3/S7/S10:
all killed. N1 and N2 (the defect itself) are caught by the OLMo-shaped fixture and the
determinism case, NOT by the seeded case: its 14 random budgets never land in the narrow band where
cutting the SAEs' bytes too refuses a split that fits. The seeded case checks every refusal against
an oracle over every division of the modules and every plan against each card's real budget; of
these controls it catches N6b.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
import random
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen2Config
from transformers.integrations.accelerate import _get_device_map, compute_module_sizes
from transformers.quantizers.auto import AutoHfQuantizer

from src.ml import model_loader, split_load
from src.ml.model_loader import ModelLoadError, QuantizationFormat
from src.ml.split_load import SplitDoesNotFit, plan_split_load
from src.services import resource_config

MIB = 1024 * 1024


def _config(vocab: int = 64_000) -> Qwen2Config:
    """Eight ~338 MiB (fp16) decoder layers; an embedding and an lm_head of vocab x 4096."""
    return Qwen2Config(
        vocab_size=vocab, hidden_size=4_096, intermediate_size=11_008, num_hidden_layers=8,
        num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=False,
    )


def _sizes_mib(config, quantization_config=None) -> dict:
    quantizer = None
    if quantization_config is not None:
        quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=False)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(copy.deepcopy(config), dtype=torch.float16)
    if quantizer is not None:
        quantizer.preprocess_model(
            model=model, dtype=torch.float16, device_map="sequential", checkpoint_files=None, use_kernels=False
        )
    sizes, _ = compute_module_sizes(model, quantizer, only_modules=False)
    return {name: size / MIB for name, size in sizes.items()}


def _card_of(device_map: dict, layer: int):
    """Worked independently of split_load: the longest map key that owns model.layers.<layer>."""
    name = f"model.layers.{layer}"
    keys = [key for key in device_map if key == "" or key == name or name.startswith(key + ".")]
    return device_map[max(keys, key=len)]


def _mapped(device_map: dict, sizes: dict) -> dict:
    out: dict = {}
    for name, device in device_map.items():
        out[device] = out.get(device, 0.0) + sizes.get(name, 0.0)
    return out


def _budget(mb: float) -> str:
    return f"{math.floor(mb)}MiB"


def _tight_first_card(s: dict, room_layers: float = 0.5):
    """cuda:0 holds the embedding and three layers with `room_layers` of a layer to spare;
    cuda:1 holds the rest with two layers to spare."""
    layer = s["model.layers.0"]
    budget0 = s["model.embed_tokens"] + 3 * layer + room_layers * layer
    budget1 = s[""] - s["model.embed_tokens"] - 3 * layer + 2 * layer
    return {0: _budget(budget0), 1: _budget(budget1)}, layer


class TestTheAllowance:
    def test_an_sae_on_a_layer_its_card_cannot_hold_moves_that_layer_to_a_card_that_can(self):
        config = _config()
        s = _sizes_mib(config)
        budgets, layer = _tight_first_card(s)
        sae_mb = 0.8 * layer

        bare = plan_split_load(config, max_memory=budgets, dtype=torch.float16)
        assert _card_of(bare.device_map, 2) == 0, "the fixture must put layer 2 on the tight card"
        assert bare.mapped_mb[0] + sae_mb > int(budgets[0][:-3]), (
            "precondition: without the allowance layer 2's SAE overcommits cuda:0"
        )

        plan = plan_split_load(config, max_memory=budgets, dtype=torch.float16, extra_mb_by_layer={2: sae_mb})

        assert _card_of(plan.device_map, 2) == 1, plan.device_map
        mapped = _mapped(plan.device_map, s)
        assert not {d for d in plan.device_map.values() if not isinstance(d, int)}, plan.device_map
        assert mapped[0] <= int(budgets[0][:-3]), mapped
        assert mapped[1] + sae_mb <= int(budgets[1][:-3]), (mapped, sae_mb)
        assert plan.extra_mb == {1: math.ceil(sae_mb)}

    def test_every_layers_sae_is_charged_to_its_own_card(self):
        """SAEs on both cards: each card holds its share of the model and its own layers' SAEs."""
        config = _config()
        s = _sizes_mib(config)
        budgets, layer = _tight_first_card(s, room_layers=1.4)
        extra = {1: 0.5 * layer, 6: 0.7 * layer}

        plan = plan_split_load(config, max_memory=budgets, dtype=torch.float16, extra_mb_by_layer=extra)

        mapped = _mapped(plan.device_map, s)
        charged: dict = {}
        for layer_index, mb in extra.items():
            charged[_card_of(plan.device_map, layer_index)] = charged.get(_card_of(plan.device_map, layer_index), 0) + mb
        assert set(charged) == {0, 1}, "the fixture must keep an SAE on each card"
        for index in (0, 1):
            assert mapped.get(index, 0.0) + charged.get(index, 0.0) <= int(budgets[index][:-3]), (index, mapped, charged)
        assert plan.extra_mb == {index: math.ceil(mb) for index, mb in charged.items()}

    def test_an_sae_on_a_card_with_room_changes_nothing(self):
        config = _config()
        s = _sizes_mib(config)
        budgets, layer = _tight_first_card(s)

        bare = plan_split_load(config, max_memory=budgets, dtype=torch.float16)
        plan = plan_split_load(config, max_memory=budgets, dtype=torch.float16, extra_mb_by_layer={7: 0.5 * layer})

        assert _card_of(bare.device_map, 7) == 1
        assert plan.max_memory == bare.max_memory and plan.device_map == bare.device_map
        assert plan.extra_mb == {1: math.ceil(0.5 * layer)}

    def test_a_four_bit_split_keeps_the_allowance_inside_bitsandbytes_fill(self):
        # CI installs a CPU-only torch and no bitsandbytes (backend-tests.yml); the image has both.
        pytest.importorskip("bitsandbytes", reason="bitsandbytes is not installed (CI's CPU-only environment)")
        q4 = model_loader.get_quantization_config(QuantizationFormat.Q4)
        config = _config(vocab=128_000)
        s = _sizes_mib(config, quantization_config=q4)
        layer = s["model.layers.0"]
        on0 = s["model.embed_tokens"] + 3 * layer
        budgets = {0: _budget((on0 + 0.3 * layer) / 0.9), 1: _budget((s[""] - on0 + 3 * layer) / 0.9)}
        sae_mb = 0.9 * layer

        bare = plan_split_load(config, max_memory=budgets, dtype=torch.float16, quantization_config=q4)
        assert _card_of(bare.device_map, 2) == 0
        plan = plan_split_load(
            config, max_memory=budgets, dtype=torch.float16, quantization_config=q4, extra_mb_by_layer={2: sae_mb}
        )

        mapped = _mapped(plan.device_map, s)
        card = _card_of(plan.device_map, 2)
        assert mapped.get(card, 0.0) + sae_mb <= 0.9 * int(budgets[card][:-3]), (mapped, card)
        for index in (0, 1):
            assert mapped.get(index, 0.0) <= 0.9 * int(budgets[index][:-3])


class TestASplitTheSaesDoNotFitIsRefused:
    def test_refused_before_loading_with_the_allowance_in_the_message(self):
        """The last card has 0.2 of a layer to spare and the last layer's SAE is a whole layer:
        there is no card to move it to."""
        config = _config()
        s = _sizes_mib(config)
        layer = s["model.layers.0"]
        budget0 = s["model.embed_tokens"] + 3 * layer + 0.2 * layer
        budgets = {0: _budget(budget0), 1: _budget(s[""] - s["model.embed_tokens"] - 3 * layer + 0.2 * layer)}
        assert plan_split_load(config, max_memory=budgets, dtype=torch.float16) is not None, (
            "precondition: the model alone fits"
        )

        with pytest.raises(SplitDoesNotFit) as refused:
            plan_split_load(config, max_memory=budgets, dtype=torch.float16, extra_mb_by_layer={7: layer})

        message = str(refused.value)
        assert f"{math.ceil(layer):,}" in message or f"{layer:,.0f}" in message, message
        assert "does not fit on the GPUs" in message

    def test_a_card_still_overcommitted_when_the_re_maps_run_out_is_refused(self, monkeypatch):
        """The re-map loop is bounded. A split whose cards still cannot hold their layers'
        SAEs when it ends is refused, never handed to the load as it stands."""
        monkeypatch.setattr(split_load, "ALLOWANCE_PASSES", 0)
        config = _config()
        s = _sizes_mib(config)
        budgets, layer = _tight_first_card(s)

        with pytest.raises(SplitDoesNotFit, match="beside"):
            plan_split_load(config, max_memory=budgets, dtype=torch.float16, extra_mb_by_layer={2: 0.8 * layer})


class TestNothingToChargeIsNotARefusal:
    def test_without_an_allowance_no_layers_are_looked_for(self, monkeypatch):
        def no_layers(model):
            raise AssertionError("layers looked up for a split with no allowance")

        monkeypatch.setattr(split_load, "_layer_names", no_layers)
        config = _config()
        s = _sizes_mib(config)
        budgets, _ = _tight_first_card(s)

        assert plan_split_load(config, max_memory=budgets, dtype=torch.float16) is not None

    def test_a_model_whose_layers_cannot_be_found_is_unverifiable_not_refused(self, monkeypatch):
        monkeypatch.setattr(split_load, "_layer_names", lambda model: None)
        config = _config()
        s = _sizes_mib(config)
        budgets, layer = _tight_first_card(s)

        assert plan_split_load(config, max_memory=budgets, dtype=torch.float16, extra_mb_by_layer={2: layer}) is None


# ── The loader passes the allowance to the plan ─────────────────────────────


class _Loaded(torch.nn.Module):
    def __init__(self, device_map):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.hf_device_map = device_map


class _TokenizerReached(Exception):
    pass


@pytest.fixture
def hub(monkeypatch):
    """from_pretrained computes transformers' real device map from the kwargs it is given."""
    state = SimpleNamespace(config=_config(), maps=[], loads=[])

    def from_pretrained(repo_id, **kwargs):
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(copy.deepcopy(state.config), dtype=kwargs["torch_dtype"])
        max_memory = kwargs.get("max_memory")
        device_map = _get_device_map(model, kwargs["device_map"], dict(max_memory) if max_memory else None, None)
        state.loads.append(kwargs)
        state.maps.append(device_map)
        return _Loaded(device_map)

    def tokenizer(*args, **kwargs):
        raise _TokenizerReached("tokenizer reached")

    monkeypatch.setattr(model_loader.AutoConfig, "from_pretrained", lambda *a, **k: copy.deepcopy(state.config))
    monkeypatch.setattr(resource_config, "preflight_gpu_capacity", lambda **kwargs: None)
    monkeypatch.setattr(model_loader.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(model_loader.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(torch.cuda, "device", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return state


def test_the_loader_maps_a_split_around_the_saes_its_caller_names(hub):
    s = _sizes_mib(hub.config)
    budgets, layer = _tight_first_card(s)
    sae_mb = 0.8 * layer

    with pytest.raises(ModelLoadError, match="tokenizer reached"):
        model_loader.load_model_from_hf(
            "org/model", quant_format=QuantizationFormat.FP16, device_map="sequential",
            max_memory=budgets, extra_mb_by_layer={2: sae_mb},
        )

    assert len(hub.loads) == 1
    final = hub.maps[-1]
    assert _card_of(final, 2) == 1, "the load mapped layer 2 onto the card its SAE does not fit"
    assert _mapped(final, s)[1] + sae_mb <= int(budgets[1][:-3])


def test_a_single_card_load_takes_no_plan_and_ignores_the_allowance(hub, monkeypatch):
    planned = []
    monkeypatch.setattr(split_load, "plan_split_load", lambda *a, **k: planned.append(k))

    with pytest.raises(ModelLoadError, match="tokenizer reached"):
        model_loader.load_model_from_hf(
            "org/model", quant_format=QuantizationFormat.FP16, device_map="meta", extra_mb_by_layer={2: 100.0},
        )

    assert planned == []
    assert "max_memory" not in hub.loads[0]


def test_the_loader_maps_a_split_around_the_work_its_caller_names(hub):
    """Review round 3: working room, like held allowance, reaches the plan from the loader."""
    s = _sizes_mib(hub.config)
    budgets, layer = _tight_first_card(s)

    with pytest.raises(ModelLoadError, match="tokenizer reached"):
        model_loader.load_model_from_hf(
            "org/model", quant_format=QuantizationFormat.FP16, device_map="sequential",
            max_memory=budgets, working_mb_by_layer={2: 0.8 * layer},
        )

    final = hub.maps[-1]
    assert _card_of(final, 2) == 1, "the load mapped layer 2 onto the card its working room does not fit"


# ── Review round 3: only what must leave a card leaves it ────────────────────


#: The fixture's modules at accelerate's no-split granularity, in the order it fills cards.
_ORDER = ["model.embed_tokens", *[f"model.layers.{i}" for i in range(8)], "model.norm", "model.rotary_emb", "lm_head"]


def _layer_of(name: str):
    return int(name.rsplit(".", 1)[1]) if name.startswith("model.layers.") else None


def _charge(run, s, extra, work):
    layers = [_layer_of(name) for name in run if _layer_of(name) is not None]
    return (sum(s.get(name, 0.0) for name in run) + sum(extra.get(L, 0.0) for L in layers)
            + max((work.get(L, 0.0) for L in layers), default=0.0))


def _fits_sequentially(s, budgets, extra, work=None) -> bool:
    """Worked independently of split_load: SOME division of the modules, in order, over the
    cards in index order, where every card holds its run beside its layers' SAEs and the
    largest of its layers' working room."""
    work = work or {}
    cards = sorted(budgets)

    def place(start, position):
        if start == len(_ORDER):
            return True
        if position == len(cards):
            return False
        return any(
            _charge(_ORDER[start:end], s, extra, work) <= int(budgets[cards[position]][:-3]) and place(end, position + 1)
            for end in range(len(_ORDER), start - 1, -1)
        )

    return place(0, 0)


def _holds(device_map, s, budgets, extra, work) -> bool:
    """Every card of a planned map holds its modules beside its layers' SAEs and working room."""
    if any(not isinstance(device, int) for device in device_map.values()):
        return False
    for index in budgets:
        run = [name for name in _ORDER if _owning(device_map, name) == index]
        if _charge(run, s, extra, work) > int(budgets[index][:-3]):
            return False
    return True


def _owning(device_map, name):
    keys = [key for key in device_map if key == "" or key == name or name.startswith(key + ".")]
    return device_map[max(keys, key=len)] if keys else None


class TestOnlyWhatMustLeaveACardLeavesIt:
    """Review round 3. Round 2 cut an overcommitted card's budget by its WHOLE excess, the
    SAEs' bytes included. A layer that leaves takes its SAE with it, so that moved the
    SAEs' worth of MODEL off the card too. On OLMo-2-1124-13B at FP16 over 11,000 + 23,000 MB
    with a decoder beside layers 11-13 (the steering core's case, 1,280 MiB each, and
    steering's fp16 SAEs, the same size) it moved seven layers where two had to go, the
    3090 could not take them, and a split that fits (layer 11 and its decoder on the 3080 Ti:
    8,240 + 1,280 of 9,976 MiB; the rest: 17,921 + 2,560 of 21,976 MiB) was refused before
    loading. Scratchpad p2-r3/probe_overshoot.py found that map with transformers' own
    inference; this fixture has the same shape: SAEs twice a layer's size."""

    def test_the_layers_that_leave_take_their_saes_so_no_more_of_the_model_leaves(self):
        config = _config()
        s = _sizes_mib(config)
        layer, embed = s["model.layers.0"], s["model.embed_tokens"]
        budgets = {0: _budget(embed + 4.5 * layer), 1: _budget(s[""] - embed - 4 * layer + 7 * layer)}
        extra = {1: 2 * layer, 2: 2 * layer, 3: 2 * layer}

        bare = plan_split_load(config, max_memory=budgets, dtype=torch.float16)
        assert [_card_of(bare.device_map, i) for i in range(5)] == [0, 0, 0, 0, 1], (
            "precondition: without the SAEs layers 0-3 fill cuda:0"
        )
        assert _fits_sequentially(s, budgets, extra), "precondition: a map holds the model beside its SAEs"

        plan = plan_split_load(config, max_memory=budgets, dtype=torch.float16, extra_mb_by_layer=extra)

        assert [_card_of(plan.device_map, i) for i in range(5)] == [0, 0, 1, 1, 1], plan.device_map
        assert _holds(plan.device_map, s, budgets, extra, {})

    def test_a_split_is_refused_only_when_no_map_holds_it_and_a_plan_always_holds(self):
        """Seeded cases over both answers: a refusal is checked against every division of
        the modules over the cards, and a plan against each card's real budget."""
        config = _config()
        s = _sizes_mib(config)
        layer, embed = s["model.layers.0"], s["model.embed_tokens"]
        rng = random.Random(20_260_914)
        answers = {"planned": 0, "refused": 0}
        for case in range(14):
            budgets = {0: _budget(embed + rng.uniform(1.0, 7.5) * layer),
                       1: _budget(s[""] - embed - 4 * layer + rng.uniform(-0.5, 8.0) * layer)}
            extra = {L: rng.uniform(0.2, 2.5) * layer for L in rng.sample(range(8), rng.randint(1, 4))}
            work = {L: rng.uniform(0.1, 1.5) * layer for L in rng.sample(range(8), rng.randint(0, 3))}
            try:
                plan = plan_split_load(config, max_memory=budgets, dtype=torch.float16,
                                       extra_mb_by_layer=extra, working_mb_by_layer=work or None)
            except SplitDoesNotFit:
                answers["refused"] += 1
                assert not _fits_sequentially(s, budgets, extra, work), ("refused a split a map holds", case,
                                                                         budgets, extra, work)
                continue
            answers["planned"] += 1
            assert _holds(plan.device_map, s, budgets, extra, work), ("a plan overcommits a card", case,
                                                                       budgets, extra, work, plan.device_map)
        assert answers["planned"] and answers["refused"], answers

    def test_the_same_split_is_planned_the_same_way_twice(self):
        config = _config()
        s = _sizes_mib(config)
        layer, embed = s["model.layers.0"], s["model.embed_tokens"]
        budgets = {0: _budget(embed + 4.5 * layer), 1: _budget(s[""] - embed - 4 * layer + 7 * layer)}
        kwargs = dict(max_memory=budgets, dtype=torch.float16,
                      extra_mb_by_layer={1: 2 * layer, 2: 2 * layer, 3: 2 * layer},
                      working_mb_by_layer={6: 0.5 * layer})

        assert plan_split_load(config, **kwargs) == plan_split_load(config, **kwargs)

    def test_a_card_keeps_room_for_its_largest_layers_work_not_the_sum(self):
        config = _config()
        s = _sizes_mib(config)
        layer, embed = s["model.layers.0"], s["model.embed_tokens"]
        budgets = {0: _budget(embed + 4.5 * layer), 1: _budget(s[""] - embed - 4 * layer + 1.5 * layer)}
        work = {5: 1.2 * layer, 6: 1.2 * layer}

        bare = plan_split_load(config, max_memory=budgets, dtype=torch.float16)
        on1 = _mapped(bare.device_map, s)[1]
        assert _card_of(bare.device_map, 5) == _card_of(bare.device_map, 6) == 1
        assert on1 + 1.2 * layer <= int(budgets[1][:-3]) < on1 + 2.4 * layer, (
            "precondition: cuda:1 holds one layer's working room, not two"
        )

        plan = plan_split_load(config, max_memory=budgets, dtype=torch.float16, working_mb_by_layer=work)

        assert plan.device_map == bare.device_map
        assert plan.working_mb == {1: math.ceil(1.2 * layer)}
        assert plan.extra_mb == {}

    def test_work_beside_a_layer_its_card_cannot_hold_moves_that_layer(self):
        config = _config()
        s = _sizes_mib(config)
        budgets, layer = _tight_first_card(s)

        plan = plan_split_load(config, max_memory=budgets, dtype=torch.float16, working_mb_by_layer={2: 0.8 * layer})

        assert _card_of(plan.device_map, 2) == 1
        assert _holds(plan.device_map, s, budgets, {}, {2: 0.8 * layer})

    def test_a_card_left_with_none_of_the_model_is_named(self, caplog):
        """Three cards. The middle one holds a single layer, and that layer's SAE does not fit
        beside it: the layer moves on, the card is left empty, and the plan says so."""
        config = _config()
        s = _sizes_mib(config)
        layer, embed = s["model.layers.0"], s["model.embed_tokens"]
        budgets = {0: _budget(embed + 3.5 * layer), 1: _budget(1.5 * layer), 2: _budget(s[""] - embed)}
        extra = {3: layer}

        bare = plan_split_load(config, max_memory=budgets, dtype=torch.float16)
        assert [_card_of(bare.device_map, i) for i in (2, 3, 4)] == [0, 1, 2], "precondition: cuda:1 holds layer 3"

        with caplog.at_level(logging.WARNING, logger="src.ml.split_load"):
            plan = plan_split_load(config, max_memory=budgets, dtype=torch.float16, extra_mb_by_layer=extra)

        assert plan.mapped_mb[1] == 0 and _card_of(plan.device_map, 3) == 2
        assert "puts none of the model on cuda:1" in caplog.text
