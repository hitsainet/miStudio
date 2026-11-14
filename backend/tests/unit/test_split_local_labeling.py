"""A local judge no single card holds is loaded across cards, and runs and is released there.

Multi-GPU Phase 2. `LocalLabelingService` takes the placement's `device_map` and
`max_memory`: a split loads across the cards within that GPU-only budget, refuses a load
accelerate still maps to cpu/disk, sends each prompt to the embedding's card,
and counts and releases memory on every card the judge holds. The worker sizes
the judge from a config already on disk so Auto can choose a split at all.

Nothing here needs a GPU or the network.

MUTATION CONTROLS (2026-09-14; each applied alone, this file +
test_gpu_labeling_placement.py + test_local_labeling_context.py run, source
restored and checked by sha256). All went red (LS1–LS5, the worker's side, are
recorded in test_gpu_labeling_placement.py):
  L1  prompt always to self.device                       -> a_split_judges_prompt_goes_to_the_embedding_card
  L2  device_map={"": device} even for a split           -> a_split_judge_loads_across_its_cards...
  L3  max_memory not passed to from_pretrained           -> a_split_judge_loads_across_its_cards...
  L4  no refusal of a load mapped to cpu/disk            -> a_split_judge_mapped_off_the_gpus_is_refused[disk, cpu]
  L5  the judge's cards recorded as [self.device]        -> loads_across_its_cards..., unloading_a_split_judge...
  L6  dispatch hooks not removed on unload               -> unloading_a_split_judge_releases_every_card
  L7  a split budget accepted without its device_map     -> a_split_budget_without_the_splits_device_map_is_refused
  L8  unload empties the first card only                 -> unloading_a_split_judge_releases_every_card
  LJ1 judge config read with local_files_only=False      -> TestTheJudgeIsSized::from_a_config_on_disk_at_four_bits
  LJ2 judge sized at FP16 instead of 4-bit               -> TestTheJudgeIsSized::from_a_config_on_disk_at_four_bits
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import src.services.local_labeling_service as module
from src.services.gpu_placement import GpuCard, Placement
from src.services.local_labeling_service import LocalLabelingService, local_judge_required_mb

CUDA_0 = torch.device("cuda", 0)
CUDA_1 = torch.device("cuda", 1)
SPLIT = Placement(
    card=GpuCard(index=1, uuid="GPU-rtx", name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
    device=CUDA_1,
    cards=(
        GpuCard(index=1, uuid="GPU-rtx", name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
        GpuCard(index=0, uuid="GPU-ti", name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000),
    ),
    devices=(CUDA_1, CUDA_0),
    max_memory_mb={1: 21_976, 0: 9_976},
)
BUDGET = {1: "21976MiB", 0: "9976MiB"}


def _split_judge():
    return LocalLabelingService(
        model_name="org/judge", device=CUDA_1, device_map=SPLIT.device_map, max_memory=SPLIT.max_memory,
    )


def test_a_split_budget_without_the_splits_device_map_is_refused():
    """The budget would be ignored and the judge loaded whole onto the first card."""
    with pytest.raises(ValueError, match="device_map"):
        LocalLabelingService(model_name="org/judge", device=CUDA_1, max_memory=BUDGET)


EXAMPLE = {
    "prefix_tokens": ["In", "the"],
    "prime_token": "garden",
    "suffix_tokens": ["grew"],
    "max_activation": 5.0,
    "sample_index": 0,
}


class _Loaded(torch.nn.Module):
    def __init__(self, device_map):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.hf_device_map = device_map


@pytest.fixture
def hub(monkeypatch):
    """from_pretrained, CUDA memory and cache calls, recorded; the judge "holds" both cards."""
    h = SimpleNamespace(load=[], allocated=[], entered=[], detached=[], maps=[{"model.embed_tokens": 0, "model.layers.0": 1}])

    def from_pretrained(name, **kwargs):
        h.load.append(kwargs)
        return _Loaded(h.maps[0])

    class _DeviceContext:
        def __init__(self, device):
            h.entered.append(device)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", MagicMock())
    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(module, "cuda_devices", lambda model: [CUDA_1, CUDA_0])
    monkeypatch.setattr(module, "detach_dispatch_hooks", h.detached.append)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: h.allocated.append(device) or 0)
    monkeypatch.setattr(torch.cuda, "device", _DeviceContext)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return h


def test_a_split_judge_loads_across_its_cards_and_counts_memory_on_each(hub):
    judge = _split_judge()

    judge.load_model()

    kwargs = hub.load[0]
    assert (kwargs["device_map"], kwargs["max_memory"]) == (SPLIT.device_map, BUDGET)
    assert kwargs["load_in_4bit"] is True
    assert hub.allocated == [CUDA_1, CUDA_0], "memory was counted on only some of the judge's cards"
    assert judge.is_loaded


@pytest.mark.parametrize("target", ["disk", "cpu"])
def test_a_split_judge_mapped_off_the_gpus_is_refused(hub, target):
    hub.maps[0] = {"model.embed_tokens": 0, "model.layers.9": target}
    judge = _split_judge()

    with pytest.raises(RuntimeError, match="does not fit on the GPUs") as exc:
        judge.load_model()

    assert f"model.layers.9 on {target}" in str(exc.value)
    assert judge.model is None and not judge.is_loaded


def test_unloading_a_split_judge_releases_every_card(hub):
    judge = _split_judge()
    judge.load_model()
    model = judge.model
    hub.allocated.clear()

    judge.unload_model()

    assert hub.entered == [CUDA_1, CUDA_0]
    assert hub.allocated == [CUDA_1, CUDA_0]
    assert hub.detached == [model], "the split judge's dispatch hooks were not removed"
    assert judge.model is None


def test_a_split_judges_prompt_goes_to_the_embedding_card():
    class _SplitJudge(torch.nn.Module):
        """The first registered parameter is on "another card" (meta); the embedding is not."""

        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2, device="meta")
            self.embed = torch.nn.Embedding(4, 2)

        def get_input_embeddings(self):
            return self.embed

        def generate(self, **kwargs):
            return torch.zeros(1, 5, dtype=torch.long)

    moved = []

    class _Inputs(dict):
        def to(self, device):
            moved.append(device)
            return self

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "prompt"
    tokenizer.return_value = _Inputs(input_ids=torch.zeros(1, 3, dtype=torch.long))
    tokenizer.decode.return_value = '{"specific": "garden", "category": "semantic", "description": "d"}'

    judge = _split_judge()
    judge.model, judge.tokenizer, judge.is_loaded = _SplitJudge(), tokenizer, True
    label = judge.generate_label(examples=[dict(EXAMPLE)], feature_id="feat_1")

    assert moved == [torch.device("cpu")], "the prompt went to the placement's first card, not the embedding's"
    assert label["specific"] == "garden"


class TestTheJudgeIsSized:
    def test_from_a_config_on_disk_at_four_bits(self, monkeypatch):
        from transformers import LlamaConfig

        from src.ml.model_loader import QuantizationFormat, estimate_model_memory, estimate_parameter_count

        config = LlamaConfig(hidden_size=4096, num_hidden_layers=32, vocab_size=128_256, intermediate_size=14_336)
        seen = {}

        def from_pretrained(name, **kwargs):
            seen.update(name=name, kwargs=kwargs)
            return config

        monkeypatch.setattr(module.AutoConfig, "from_pretrained", from_pretrained)

        size = local_judge_required_mb("phi3")

        expected = estimate_model_memory(estimate_parameter_count(config), QuantizationFormat.Q4) / (1024**2)
        assert size == expected and size > 0
        assert seen["name"] == LocalLabelingService.DEFAULT_MODEL, "the alias was not resolved"
        assert seen["kwargs"]["local_files_only"] is True, "placement waited on the network"

    def test_a_judge_with_no_config_on_disk_has_no_size(self, monkeypatch):
        def missing(name, **kwargs):
            raise OSError("not in the cache")

        monkeypatch.setattr(module.AutoConfig, "from_pretrained", missing)

        assert local_judge_required_mb("org/judge") is None
