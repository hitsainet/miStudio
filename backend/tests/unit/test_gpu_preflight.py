"""A job that cannot fit should be refused before the weights are fetched.

Reported live, 2026-08-23, on the same model as the vocab_size failure:

    Extraction failed: CUDA out of memory. Tried to allocate 120.00 MiB.
    GPU 0 has a total capacity of 23.56 GiB of which 113.06 MiB is free

after 2m47s, 0 samples. The weights had taken the card and the first forward
pass had nowhere to go. gemma-4-12B-it at FP16 is ~22 GB of weights on a
23.56 GB card: it was never going to fit, and the product spent three minutes
finding that out in the least useful way.

Nothing checked. `ResourceConfig.get_optimal_settings` runs AFTER the model is
resident and tunes batch size against system RAM, so it could not have seen
this coming.

The guard lives inside `load_model_from_hf`, not at its ten call sites — a
guard added to one caller and not its siblings is this codebase's most repeated
defect.

2026-09-13, second GPU: the preflight read `mem_get_info(0)` whatever card the
load targeted. With an RTX 3080 Ti at index 0 and the RTX 3090 at index 1, a
job placed on the 3090 was judged against the 3080 Ti. It now measures the
device the load uses: the named card alone, or every visible card together for
device_map="auto" (accelerate spreads the weights over all of them).

MUTATION CONTROLS:
  * loader passes device="auto" instead of device_map -> the loader test fails
  * the named-card branch reads mem_get_info(0)       -> "checked alone" fails
  * "auto" takes the largest card instead of the sum  -> "every visible card" fails
"""

import inspect

import pytest

from src.ml.model_loader import estimate_parameter_count
from src.services.resource_config import (
    VRAMInsufficientError,
    preflight_gpu_capacity,
)

GiB = 1024**3


class _Config:
    hidden_size = 3584
    num_hidden_layers = 48
    vocab_size = 262_144
    intermediate_size = 14_336


class _NestedConfig:
    """Unified/multimodal shape — the same nesting that broke `vocab_size`."""

    class text_config:
        hidden_size = 3584
        num_hidden_layers = 48
        vocab_size = 262_144
        intermediate_size = 14_336


def _fake_cards(monkeypatch, cards, current=0):
    """Patch torch.cuda to report ``cards``: a list of (name, free_gib, total_gib)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(cards))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: cards[index][0])

    def mem_get_info(device=None):
        index = device.index if isinstance(device, torch.device) else device
        _, free, total = cards[index]
        return int(free * GiB), int(total * GiB)

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)


@pytest.fixture
def gpu(monkeypatch):
    """A 23.56 GiB card, matching the reported failure."""
    _fake_cards(monkeypatch, [("NVIDIA GeForce RTX 3090", 23.4, 23.56)])


@pytest.fixture
def two_cards(monkeypatch):
    """The node since 2026-09-13: the smaller card at index 0."""
    _fake_cards(
        monkeypatch,
        [("NVIDIA GeForce RTX 3080 Ti", 11.0, 12.0), ("NVIDIA GeForce RTX 3090", 23.4, 23.56)],
    )


# ── The estimate ───────────────────────────────────────────────────────────

def test_the_parameter_estimate_is_in_the_right_range():
    n = estimate_parameter_count(_Config())
    assert n is not None
    assert 10e9 < n < 14e9, f"{n/1e9:.1f}B is not a plausible 12B estimate"


def test_a_nested_config_is_read_through():
    assert estimate_parameter_count(_NestedConfig()) == estimate_parameter_count(_Config())


def test_an_unreadable_config_yields_None_not_a_guess():
    """None disables the preflight. A preflight that refuses jobs it cannot
    assess is worse than none — it would block every unfamiliar architecture."""

    class _Empty:
        pass

    assert estimate_parameter_count(_Empty()) is None


# ── The refusal ────────────────────────────────────────────────────────────

def test_the_reported_job_is_refused(gpu):
    with pytest.raises(VRAMInsufficientError) as exc:
        preflight_gpu_capacity(
            params_count=estimate_parameter_count(_Config()),
            quantization="FP16",
            device="cuda:0",
            model_name="gemma-4-12B-it",
        )
    message = str(exc.value)
    # The message must carry the arithmetic AND the remedy — a refusal that
    # only says "no" sends the user back to guessing.
    assert "gemma-4-12B-it" in message
    assert "GB of weights" in message
    assert "free of" in message
    assert "Q8" in message or "smaller model" in message


@pytest.mark.parametrize("quantization", ["Q8", "Q4"])
def test_a_quantization_that_fits_is_allowed(gpu, quantization):
    """Negative control for the direction. A preflight that refuses everything
    would pass the test above and break every job on the box."""
    preflight_gpu_capacity(
        params_count=estimate_parameter_count(_Config()),
        quantization=quantization,
        device="cuda:0",
        model_name="gemma-4-12B-it",
    )


def test_a_small_model_is_allowed_at_full_precision(gpu):
    preflight_gpu_capacity(
        params_count=1_200_000_000, quantization="FP16", device="cuda:0", model_name="LFM2.5-1.2B"
    )


def test_unknown_size_does_not_block(gpu):
    preflight_gpu_capacity(
        params_count=None, quantization="FP16", device="cuda:0", model_name="mystery"
    )


def test_no_gpu_does_not_block(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    preflight_gpu_capacity(
        params_count=99_000_000_000, quantization="FP32", device="auto", model_name="huge"
    )


# ── Which card's memory ────────────────────────────────────────────────────

def test_a_named_card_is_checked_alone(two_cards):
    """12B at Q8 needs ~14.3 GB: too much for the 3080 Ti's 11 GB, fine on the 3090."""
    params = estimate_parameter_count(_Config())
    with pytest.raises(VRAMInsufficientError) as exc:
        preflight_gpu_capacity(params_count=params, quantization="Q8", device="cuda:0")
    assert "cuda:0 (NVIDIA GeForce RTX 3080 Ti)" in str(exc.value)
    assert "11.0 GB free of 12.0 GB" in str(exc.value)

    preflight_gpu_capacity(params_count=params, quantization="Q8", device="cuda:1")


def test_a_torch_device_is_read_by_its_index(two_cards):
    import torch

    params = estimate_parameter_count(_Config())
    preflight_gpu_capacity(params_count=params, quantization="Q8", device=torch.device("cuda", 1))
    with pytest.raises(VRAMInsufficientError):
        preflight_gpu_capacity(
            params_count=params, quantization="Q8", device=torch.device("cuda", 0)
        )


def test_auto_counts_every_visible_card(two_cards):
    """12B at FP16 needs ~24.3 GB: no single card has it, both together have 34.4 GB."""
    params = estimate_parameter_count(_Config())
    preflight_gpu_capacity(params_count=params, quantization="FP16", device="auto")
    with pytest.raises(VRAMInsufficientError) as exc:
        preflight_gpu_capacity(params_count=params, quantization="FP16", device="cuda:1")
    assert "cuda:1 (NVIDIA GeForce RTX 3090)" in str(exc.value)


def test_auto_refusal_names_every_card_together(two_cards):
    with pytest.raises(VRAMInsufficientError) as exc:
        preflight_gpu_capacity(params_count=30_000_000_000, quantization="FP16", device="auto")
    assert "the 2 visible GPUs together" in str(exc.value)
    assert "34.4 GB free of 35.6 GB" in str(exc.value)


def test_cuda_without_an_index_is_the_current_device(monkeypatch):
    _fake_cards(
        monkeypatch,
        [("NVIDIA GeForce RTX 3080 Ti", 11.0, 12.0), ("NVIDIA GeForce RTX 3090", 23.4, 23.56)],
        current=1,
    )
    params = estimate_parameter_count(_Config())
    preflight_gpu_capacity(params_count=params, quantization="Q8", device="cuda")


def test_a_cpu_load_is_not_checked(two_cards):
    preflight_gpu_capacity(params_count=99_000_000_000, quantization="FP32", device="cpu")


# ── It is ON the path ──────────────────────────────────────────────────────

def test_the_loader_preflights_before_fetching_weights():
    """In the loader, so all ten call sites inherit it.

    And BEFORE `from_pretrained`: checking after the weights are resident is
    the situation this replaces.
    """
    from src.ml import model_loader

    src = inspect.getsource(model_loader.load_model_from_hf)
    assert "preflight_gpu_capacity(" in src, (
        "the loader does not preflight — every caller can still OOM after "
        "minutes of loading"
    )
    check = src.index("preflight_gpu_capacity(")
    # The CALL, not the return-type annotation — `AutoModelForCausalLM` appears
    # in the signature too, and anchoring on that put `load` before the
    # function body even started.
    load = src.index("AutoModelForCausalLM.from_pretrained(")
    assert check < load, (
        "the preflight runs after the weights are fetched, which is exactly "
        "the failure it exists to prevent"
    )


def test_the_loader_preflights_the_device_it_loads_onto(monkeypatch):
    """Run the loader: the preflight sees the load's device_map, and a refusal
    stops the load before any weights are fetched."""
    from src.ml import model_loader
    from src.services import resource_config

    class _LoadedConfig(_Config):
        model_type = "gemma"

    seen = {}

    def refuse(**kwargs):
        seen.update(kwargs)
        raise VRAMInsufficientError("refused")

    def fetched(*args, **kwargs):
        seen["weights_fetched"] = True
        raise AssertionError("weights fetched after a refusal")

    monkeypatch.setattr(model_loader.AutoConfig, "from_pretrained", lambda *a, **k: _LoadedConfig())
    monkeypatch.setattr(model_loader, "extract_architecture_config", lambda config: {})
    monkeypatch.setattr(resource_config, "preflight_gpu_capacity", refuse)
    monkeypatch.setattr(model_loader.AutoModelForCausalLM, "from_pretrained", fetched)

    with pytest.raises(Exception, match="refused"):
        model_loader.load_model_from_hf("org/model", device_map="cuda:1")

    assert seen["device"] == "cuda:1"
    assert seen["params_count"] == estimate_parameter_count(_Config())
    assert "weights_fetched" not in seen


# ── A split counts its own cards (Multi-GPU Phase 2, 2026-09-14) ───────────

def test_a_split_counts_only_its_own_cards(two_cards):
    """12B at FP16 (~24.3 GB) fits the 3090 and 3080 Ti together, not the 3090 alone.

    A split placement passes its cards' torch indices. A card the split was not
    given must not be counted: "auto" sums every visible card, which would let a
    split over one card pass on memory it cannot use.
    """
    params = estimate_parameter_count(_Config())
    preflight_gpu_capacity(params_count=params, quantization="FP16", device=(1, 0))
    with pytest.raises(VRAMInsufficientError) as exc:
        preflight_gpu_capacity(params_count=params, quantization="FP16", device=(1,))
    assert "GPUs cuda:1 (NVIDIA GeForce RTX 3090) together" in str(exc.value)
