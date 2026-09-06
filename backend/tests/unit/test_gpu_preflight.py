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
"""

import inspect

import pytest

from src.ml.model_loader import estimate_parameter_count
from src.services.resource_config import (
    VRAMInsufficientError,
    preflight_gpu_capacity,
)


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


@pytest.fixture
def gpu(monkeypatch):
    """A 23.56 GiB card, matching the reported failure."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda *a, **k: (int(23.4 * 1024**3), int(23.56 * 1024**3)),
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
        model_name="gemma-4-12B-it",
    )


def test_a_small_model_is_allowed_at_full_precision(gpu):
    preflight_gpu_capacity(
        params_count=1_200_000_000, quantization="FP16", model_name="LFM2.5-1.2B"
    )


def test_unknown_size_does_not_block(gpu):
    preflight_gpu_capacity(params_count=None, quantization="FP16", model_name="mystery")


def test_no_gpu_does_not_block(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    preflight_gpu_capacity(
        params_count=99_000_000_000, quantization="FP32", model_name="huge"
    )


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
