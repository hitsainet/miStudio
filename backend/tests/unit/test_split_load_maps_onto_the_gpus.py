"""A split the placement accepts is a split the load can map onto its GPUs.

Multi-GPU Phase 2, review round 1 (2026-09-14). `resolve_cards` accepts a split
when its cards' budgets (free memory less SHARD_RESERVE_MB each) cover the job.
transformers 5.15.1 then maps MODULES with its copy of accelerate's
`infer_auto_device_map` (integrations/accelerate.py), which

  1. fills the cards in CUDA INDEX order, whatever order the placement chose them
     in (`get_max_memory` sorts the integer keys);
  2. holds back room for the model's largest layer on the LOWEST-index card
     (`main_devices = [gpus[0], "cpu"]`) — room kept for putting back a layer
     offloaded to the CPU, which a miStudio split (no "cpu" key) never has;
  3. for bitsandbytes, fills each card only to 90% of its budget
     (`adjust_max_memory`);
  4. strands the tail of a card when the next unsplittable block does not fit.

(2) spills past the last card to "disk", and the loader then refused a split
whose budgets held the model — after `from_pretrained` had read every weight
(FP16), or with transformers' own bitsandbytes ValueError suggesting CPU offload.

These tests run the REAL map inference over meta-device models (no weights, no
GPU): `from_pretrained` is replaced by a fake that builds the skeleton and calls
transformers' `_get_device_map` with exactly the kwargs the loader passes, so the
map is transformers', not a fixture's. Card budgets are derived from the model's
measured module sizes, with the smaller card at index 0 as on the node.

Before the fix (18de448c) the first four tests in TestASplitTheBudgetsHoldLoads and
the refusal test were red: "4 module(s) would run from disk" for budgets that hold
the model module for module, and the refusal came after from_pretrained.

REVIEW SWEEP (scratchpad p2-r1-core/sweep.py): six shapes (Qwen2.5-14B, gemma-3 12B
tied and untied, gemma-3 27B, a llama-70B width, a small-layer 262k-vocab model) x
FP16/Q8/Q4 x 7 depths x 5 card sets x Auto/"all", through `resolve_cards` and the
real map inference. With a size equal to the real module sizes, 33 accepted splits
spilled before the fix (Qwen2.5-14B at FP16 on a busy node among them); after it 20
of those load on the GPUs and the rest are refused before loading. 0 plans whose
map leaves the GPUs or exceeds a card's limit; 0 refusals of a split the
placement's own budget maps onto the GPUs. With the download/extraction sizing
(+2 GB over an over-counting parameter estimate) no accepted split spilled.

MUTATION CONTROLS (review round 1, 2026-09-14; scratchpad p2-r1-core/mutate.py,
each alone, this file run, source restored and checked by sha256):
  S1  reclaim = 0 (the hold-back is never returned)       -> the_lowest_index_card..., a_tied_embedding...,
                                                             a_four_bit_split..., a_pre_quantized_checkpoint...,
                                                             TestASplitJudge::a_judge_its_budgets_hold...
  S2  the overcommit shrink disabled                       -> a_tied_embedding_does_not_overcommit...
  S3  the preflight sized without the quantizer            -> a_four_bit_split..., a_judge_its_budgets_hold...
  S4  the loader passes the placement's budget, not the plan's -> the_lowest_index_card..., a_tied_embedding...,
                                                             a_four_bit_split..., a_pre_quantized_checkpoint...
  S5  the loader swallows the refusal                      -> refused_with_the_figures_before_from_pretrained
  S6  the judge never maps its split                       -> both TestASplitJudge tests
  S7  bitsandbytes' 90% fill ignored                       -> a_four_bit_split..., a_pre_quantized_checkpoint...,
                                                             a_judge_its_budgets_hold...
  S8  the hold-back returned to the HIGHEST index          -> 5 tests (the four above + the judge's)
  S9  the loader never maps a split before loading         -> 5 tests, including the refusal
  S10 a reclaimed map that spills is refused without trying transformers' own
                                                          -> a_split_transformers_own_map_holds_is_never_refused
      (survived before that test existed; the sweep with the guard removed found 0 such cases)
  S11 a pre-quantized checkpoint's own quantizer ignored   -> a_pre_quantized_checkpoint...
  S12 the judge swallows its refusal                       -> a_judge_its_budgets_cannot_hold_is_refused...
"""

from __future__ import annotations

import contextlib
import copy
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen2Config
from transformers.integrations.accelerate import _get_device_map, compute_module_sizes
from transformers.quantizers.auto import AutoHfQuantizer

from src.ml import model_loader
from src.ml.model_loader import ModelLoadError, QuantizationFormat
from src.services import gpu_placement as gp
from src.services import resource_config
from src.services.gpu_placement import SHARD_RESERVE_MB, GpuCard

#: CI installs a CPU-only torch and no bitsandbytes (backend-tests.yml); the image has both.
BNB_REASON = "bitsandbytes is not installed (CI's CPU-only environment); the backend image ships it"

MIB = 1024 * 1024
UUID_TI = "GPU-f47ba814-49a2-603f-3595-275284140251"
UUID_RTX = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"


def _config(*, tied: bool = False, vocab: int = 64_000) -> Qwen2Config:
    """Eight ~338 MiB (fp16) decoder layers; an embedding of vocab x 4096."""
    return Qwen2Config(
        vocab_size=vocab, hidden_size=4_096, intermediate_size=11_008, num_hidden_layers=8,
        num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=tied,
    )


def _skeleton(config, dtype, quantization_config=None):
    quantizer = None
    if quantization_config is not None:
        quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=False)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(copy.deepcopy(config), dtype=dtype)
    if quantizer is not None:
        quantizer.preprocess_model(
            model=model, dtype=dtype, device_map="sequential", checkpoint_files=None, use_kernels=False
        )
    return model, quantizer


def _transformers_skeleton(config, dtype, quantization_config, device_map):
    """The skeleton and quantizer as `from_pretrained` picks them: transformers' own
    `get_hf_quantizer`, so a checkpoint's own quantization is honoured as it would be."""
    from transformers.quantizers.auto import get_hf_quantizer

    config = copy.deepcopy(config)
    quantizer, config, _ = get_hf_quantizer(config, quantization_config, device_map, True, {})
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    if quantizer is not None:
        quantizer.preprocess_model(
            model=model, dtype=dtype, device_map=device_map, checkpoint_files=None, use_kernels=False
        )
    return model, quantizer


def _sizes_mib(config, dtype=torch.float16, quantization_config=None) -> dict:
    model, quantizer = _skeleton(config, dtype, quantization_config)
    sizes, _ = compute_module_sizes(model, quantizer, only_modules=False)
    return {name: size / MIB for name, size in sizes.items()}


def _cards(budget0_mb: float, budget1_mb: float) -> list:
    """The 3080 Ti at index 0 and the 3090 at index 1, with these split budgets."""
    return [
        GpuCard(index=0, uuid=UUID_TI, name="RTX 3080 Ti", total_mb=12_288,
                free_mb=math.ceil(budget0_mb) + SHARD_RESERVE_MB),
        GpuCard(index=1, uuid=UUID_RTX, name="RTX 3090", total_mb=24_576,
                free_mb=math.ceil(budget1_mb) + SHARD_RESERVE_MB),
    ]


class _Props:
    def __init__(self, uuid):
        self.uuid = uuid


def _place(cards, required_mb):
    """`place_job` as a worker runs it, over a torch that sees the node's two cards."""
    order = [UUID_TI.removeprefix("GPU-"), UUID_RTX.removeprefix("GPU-")]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.is_available", return_value=True))
        stack.enter_context(patch("torch.cuda.device_count", return_value=2))
        stack.enter_context(
            patch("torch.cuda.get_device_properties", side_effect=lambda i: _Props(order[i]))
        )
        stack.enter_context(patch("torch.cuda.set_device"))
        return gp.place_job("auto", required_mb=required_mb, cards=cards, allow_shard=True)


class _Loaded(torch.nn.Module):
    def __init__(self, device_map):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.hf_device_map = device_map


class _TokenizerReached(Exception):
    """Raised by the fake tokenizer: the model load itself went through."""


def _mapped_mib(device_map: dict, sizes: dict) -> dict:
    mapped: dict = {}
    for name, device in device_map.items():
        # A tied lm_head has no size of its own: it is counted once, on the embedding.
        mapped[device] = mapped.get(device, 0.0) + sizes.get(name, 0.0)
    return mapped


@pytest.fixture
def hub(monkeypatch):
    """from_pretrained computes transformers' real device map from the kwargs it is given."""
    state = SimpleNamespace(config=_config(), loads=[], maps=[])

    def from_pretrained(repo_id, **kwargs):
        model, quantizer = _transformers_skeleton(
            state.config, kwargs["torch_dtype"], kwargs.get("quantization_config"), kwargs["device_map"]
        )
        max_memory = kwargs.get("max_memory")
        device_map = _get_device_map(
            model, kwargs["device_map"], dict(max_memory) if max_memory else None, quantizer
        )
        state.loads.append(kwargs)
        state.maps.append(device_map)
        return _Loaded(device_map)

    def tokenizer(*args, **kwargs):
        raise _TokenizerReached("tokenizer reached")

    monkeypatch.setattr(model_loader.AutoConfig, "from_pretrained", lambda *a, **k: copy.deepcopy(state.config))
    monkeypatch.setattr(resource_config, "preflight_gpu_capacity", lambda **kwargs: None)
    monkeypatch.setattr(model_loader.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(model_loader.AutoTokenizer, "from_pretrained", tokenizer)
    # A refusal empties each card's cache; there is no CUDA here.
    monkeypatch.setattr(torch.cuda, "device", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return state


def _load(placement, quant=QuantizationFormat.FP16):
    return model_loader.load_model_from_hf(
        "org/model", quant_format=quant, device_map=placement.device_map,
        max_memory=placement.max_memory,
    )


def _loads(placement, quant=QuantizationFormat.FP16):
    """The load went through: the fake tokenizer was reached (the loader wraps its error)."""
    with pytest.raises(ModelLoadError, match="tokenizer reached"):
        _load(placement, quant)


class TestASplitTheBudgetsHoldLoads:
    def test_the_lowest_index_card_is_not_short_by_the_largest_layer(self, hub):
        """Budgets that hold the model module for module: the 3080 Ti the embedding and
        three layers, the 3090 the rest. transformers' hold-back took lm_head's size off
        the 3080 Ti, two layers moved to the 3090, and the lm_head went to disk."""
        s = _sizes_mib(hub.config)
        layer = s["model.layers.0"]
        budget0 = s["model.embed_tokens"] + 3 * layer + 1
        budget1 = s[""] - (budget0 - 1) + 1
        cards = _cards(budget0, budget1)
        placement = _place(cards, required_mb=math.floor(s[""]))
        assert placement.is_shard, "the fixture must be a model no single card holds"

        _loads(placement)

        assert hub.loads, "the split was refused before loading"
        final = hub.maps[-1]
        assert not {d for d in final.values() if not isinstance(d, int)}, final
        mapped = _mapped_mib(final, s)
        for index, budget in placement.max_memory_mb.items():
            assert mapped.get(index, 0.0) <= budget, (
                f"cuda:{index} was mapped {mapped.get(index, 0.0):,.0f} MiB against a "
                f"{budget:,} MiB budget — the reclaimed hold-back ate its reserve"
            )

    def test_a_tied_embedding_does_not_overcommit_the_lowest_index_card(self, hub):
        """accelerate recomputes the largest layer as modules are placed, so returning
        the whole initial hold-back lets the 3080 Ti take layers past its budget."""
        hub.config = _config(tied=True, vocab=128_000)
        s = _sizes_mib(hub.config)
        layer = s["model.layers.0"]
        budget0 = s["model.embed_tokens"] + 2 * layer + 0.9 * layer
        budget1 = s[""] - s["model.embed_tokens"] - 2 * layer + 1
        placement = _place(_cards(budget0, budget1), required_mb=math.floor(s[""]))
        assert placement.is_shard

        _loads(placement)

        mapped = _mapped_mib(hub.maps[-1], s)
        assert mapped.get(0, 0.0) <= placement.max_memory_mb[0], mapped
        assert mapped.get(1, 0.0) <= placement.max_memory_mb[1], mapped

    def test_a_four_bit_split_is_sized_as_four_bit(self, hub):
        """Budgets that hold the 4-bit module sizes at bitsandbytes' 90% fill. The
        embedding and lm_head stay fp16 under bitsandbytes, so they dominate."""
        pytest.importorskip("bitsandbytes", reason=BNB_REASON)
        hub.config = _config(vocab=128_000)
        q4 = model_loader.get_quantization_config(QuantizationFormat.Q4)
        s = _sizes_mib(hub.config, quantization_config=q4)
        layer = s["model.layers.0"]
        on0 = s["model.embed_tokens"] + 3 * layer
        budget0 = on0 / 0.9 + 1
        budget1 = (s[""] - on0) / 0.9 + 1
        placement = _place(_cards(budget0, budget1), required_mb=math.floor(s[""] / 0.9))
        assert placement.is_shard

        _loads(placement, QuantizationFormat.Q4)

        assert not {d for d in hub.maps[-1].values() if not isinstance(d, int)}, hub.maps[-1]

    def test_a_pre_quantized_checkpoint_is_sized_by_its_own_quantizer(self, hub):
        """A `-bnb-4bit` checkpoint on an FP16 row: from_pretrained takes the quantizer
        from the checkpoint's config, fills each card to 90%, and the preflight must
        size it the same way rather than wave it through unmapped."""
        pytest.importorskip("bitsandbytes", reason=BNB_REASON)
        q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        config = _config(vocab=128_000)
        config.quantization_config = q4.to_dict()
        hub.config = config
        s = _sizes_mib(_config(vocab=128_000), quantization_config=q4)
        on0 = s["model.embed_tokens"] + 3 * s["model.layers.0"]
        placement = _place(
            _cards(on0 / 0.9 + 1, (s[""] - on0) / 0.9 + 1), required_mb=math.floor(s[""] / 0.9)
        )
        assert placement.is_shard

        _loads(placement)

        assert not {d for d in hub.maps[-1].values() if not isinstance(d, int)}, hub.maps[-1]


class TestASplitTheCardsCannotHoldIsRefusedBeforeLoading:
    def test_refused_with_the_figures_before_from_pretrained(self, hub):
        """The job's estimate undershot: every layer and the lm_head cannot fit. The
        refusal names the map, and no weight is read."""
        s = _sizes_mib(hub.config)
        layer = s["model.layers.0"]
        budget0 = s["model.embed_tokens"] + 3 * layer + 1
        budget1 = s[""] - (budget0 - 1) - 1.5 * layer
        placement = _place(_cards(budget0, budget1), required_mb=math.floor(budget0 + budget1 - 2))
        assert placement.is_shard

        with pytest.raises(ModelLoadError) as refused:
            _load(placement)

        assert hub.loads == [], "the refusal came after from_pretrained had started reading weights"
        message = str(refused.value)
        assert "disk" in message and "cuda:0" in message and "cuda:1" in message, message


def test_a_split_that_cannot_be_mapped_here_loads_as_before(hub, monkeypatch):
    """A model that will not build on the meta device is not a refusal: the load maps
    it itself with the placement's budget, and the post-load check still applies."""
    from src.ml import split_load

    s = _sizes_mib(hub.config)
    placement = _place(_cards(s[""] / 2 + 400, s[""] / 2 + 400), required_mb=math.floor(s[""]))
    assert placement.is_shard

    def unbuildable(*args, **kwargs):
        raise RuntimeError("this architecture cannot be built on meta here")

    monkeypatch.setattr(split_load, "_skeleton", unbuildable)

    _loads(placement)

    assert hub.loads[0]["max_memory"] == placement.max_memory


def test_a_split_transformers_own_map_holds_is_never_refused(monkeypatch):
    """The room returned to the lowest-index card is tried first. If that map spills
    where transformers' own map (nothing returned) does not, the split loads with
    transformers' map rather than being refused. No shape in the review sweep reached
    this, so the first map's spill is injected."""
    from src.ml import split_load

    config = _config()
    model, _ = _skeleton(config, torch.float16)
    monkeypatch.setattr(split_load, "_skeleton", lambda *a, **k: (model, None))
    real_infer = split_load._infer_map
    seen = []

    def infer(model_, max_memory, quantizer):
        seen.append(dict(max_memory))
        device_map = dict(real_infer(model_, max_memory, quantizer))
        if len(seen) == 1:
            device_map[next(reversed(device_map))] = "disk"
        return device_map

    monkeypatch.setattr(split_load, "_infer_map", infer)
    s = _sizes_mib(config)
    budget = {0: f"{math.ceil(s[''] / 2 + 600)}MiB", 1: f"{math.ceil(s[''] / 2 + 600)}MiB"}

    plan = split_load.plan_split_load(config, max_memory=budget, dtype=torch.float16)

    assert len(seen) == 2, seen
    assert seen[0][0] != budget[0], "the first map did not return the hold-back"
    assert plan is not None and plan.max_memory == budget and plan.reclaimed_mb == 0


class TestASplitJudge:
    """The local judge loads through `from_pretrained` directly, so it maps its own split."""

    @pytest.fixture
    def judge_hub(self, monkeypatch):
        from unittest.mock import MagicMock

        pytest.importorskip("bitsandbytes", reason=BNB_REASON)  # every judge here loads Q4

        import src.services.local_labeling_service as labeling

        state = SimpleNamespace(config=_config(vocab=128_000), loads=[], maps=[])

        def from_pretrained(name, **kwargs):
            q4 = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                if kwargs.get("load_in_4bit") else None
            )
            model, quantizer = _skeleton(state.config, kwargs["torch_dtype"], q4)
            max_memory = kwargs.get("max_memory")
            device_map = _get_device_map(
                model, kwargs["device_map"], dict(max_memory) if max_memory else None, quantizer
            )
            state.loads.append(kwargs)
            state.maps.append(device_map)
            return _Loaded(device_map)

        monkeypatch.setattr(labeling.AutoConfig, "from_pretrained", lambda *a, **k: copy.deepcopy(state.config))
        monkeypatch.setattr(labeling.AutoTokenizer, "from_pretrained", MagicMock())
        monkeypatch.setattr(labeling.AutoModelForCausalLM, "from_pretrained", from_pretrained)
        monkeypatch.setattr(labeling, "cuda_devices", lambda model: [torch.device("cuda", 0), torch.device("cuda", 1)])
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
        return state, labeling

    def _judge(self, labeling, placement):
        return labeling.LocalLabelingService(
            model_name="org/judge", device=placement.device,
            device_map=placement.device_map, max_memory=placement.max_memory,
        )

    def _q4_sizes(self, state):
        return _sizes_mib(
            state.config,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        )

    def test_a_judge_its_budgets_hold_loads_on_its_gpus(self, judge_hub):
        state, labeling = judge_hub
        s = self._q4_sizes(state)
        on0 = s["model.embed_tokens"] + 3 * s["model.layers.0"]
        placement = _place(
            _cards(on0 / 0.9 + 1, (s[""] - on0) / 0.9 + 1), required_mb=math.floor(s[""] / 0.9)
        )
        assert placement.is_shard

        judge = self._judge(labeling, placement)
        judge.load_model()

        assert judge.is_loaded
        final = state.maps[-1]
        assert not {d for d in final.values() if not isinstance(d, int)}, final
        mapped = _mapped_mib(final, s)
        for index, budget in placement.max_memory_mb.items():
            assert mapped.get(index, 0.0) <= 0.9 * budget, (index, mapped, budget)

    def test_a_judge_its_budgets_cannot_hold_is_refused_before_loading(self, judge_hub):
        state, labeling = judge_hub
        s = self._q4_sizes(state)
        on0 = s["model.embed_tokens"] + 3 * s["model.layers.0"]
        budget0 = on0 / 0.9 + 1
        budget1 = (s[""] - on0 - 2 * s["model.layers.0"]) / 0.9
        placement = _place(_cards(budget0, budget1), required_mb=math.floor(budget0 + budget1 - 2))
        assert placement.is_shard

        judge = self._judge(labeling, placement)
        with pytest.raises(RuntimeError, match="does not fit on the GPUs") as refused:
            judge.load_model()

        assert state.loads == [], "the judge's refusal came after from_pretrained"
        assert "disk" in str(refused.value)
        assert judge.model is None and not judge.is_loaded
