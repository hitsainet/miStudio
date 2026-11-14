"""A steering SAE reaches a card as the fp16 copy placement charges for, and a split's stays off the cards.

Multi-GPU Phase 2, review round 3 (2026-09-14). Round 2 (e78a8940) sized a request's
SAEs at fp16, as steering holds them, for placement and for each layer's card of a
split. But `SteeringService.load_sae` read the state dict ONTO the placement's card at
fp32 and built the fp32 module there before halving it: four times the fp16 SAE on
that card at once. Every generation path loads its SAEs before `load_model`
force-reloads the model the cache kept, so the kept model still filled the card. And
a split's SAEs were loaded onto its first card only for `load_model` to move them to
the CPU again before reading the budget.

Now the state dict is read on the CPU, the module halved there, and only the fp16 copy
moves: to the card for one card, nowhere for a split, whose hooks put each SAE on its
layer's card.

The SAE here is a real `create_sae` module loading a real state dict; only its `.to`
is recorded, since there is no GPU.

MUTATION CONTROLS (review round 3, 2026-09-14; scratchpad p2-r3/mutate.py + mutations_r3.json,
each alone in a private copy of backend/, restored by sha256). All killed:
  N23 the state dict read onto the placement's card again     -> both tests
  N24 the module halved after it moves, not before            -> test_on_one_card_only_the_fp16_copy_moves_to_the_card
  N25 a split's SAE moved to the placement's card             -> test_a_splits_sae_stays_on_the_cpu_until_its_hook_places_it
Round 2's W6 (the SAEs left on the cards while the split's budget is read) and W7 (their cache
not emptied) re-run against this change over test_every_split_load_is_mapped_first,
test_steering_split_keeps_room_for_its_saes and test_steering_gpu_placement: both still killed,
one test red each. `load_model` still moves an SAE cached on a card from an earlier request.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from src.ml.sparse_autoencoder import create_sae
from src.services import steering_service as steering_module
from src.services.gpu_placement import GpuCard, Placement
from src.services.steering_service import SteeringService

TI = GpuCard(index=0, uuid="GPU-f47ba814-49a2-603f-3595-275284140251", name="NVIDIA GeForce RTX 3080 Ti",
             total_mb=12_288, free_mb=11_000)
RTX = GpuCard(index=1, uuid="GPU-247aa582-0d1b-e161-8156-983ed1fefc57", name="NVIDIA GeForce RTX 3090",
              total_mb=24_576, free_mb=23_000)
CUDA0, CUDA1 = torch.device("cuda", 0), torch.device("cuda", 1)
SPLIT = Placement(card=RTX, device=CUDA1, cards=(RTX, TI), devices=(CUDA1, CUDA0),
                  max_memory_mb={1: 21_976, 0: 9_976})
ONE_CARD = Placement(card=RTX, device=CUDA1)


@pytest.fixture
def sae_on_disk(monkeypatch):
    """A real 8 x 32 SAE's state dict read by `load_sae_auto_detect`, and every move of the module recorded."""
    seen = {"read_on": [], "moves": []}
    source = create_sae("standard", hidden_dim=8, latent_dim=32, normalize_activations="constant_norm_rescale")
    state = {name: tensor.clone() for name, tensor in source.state_dict().items()}

    def load_sae_auto_detect(path, device="cpu"):
        seen["read_on"].append(str(device))
        return {name: tensor.to(device) for name, tensor in state.items()}, None, "mistudio"

    def recording_create_sae(**kwargs):
        module = create_sae(**kwargs)

        def to(device, *args, **more):
            seen["moves"].append((str(device), sorted({str(p.dtype) for p in module.parameters()})))
            return module

        module.to = to
        return module

    monkeypatch.setattr(steering_module, "load_sae_auto_detect", load_sae_auto_detect)
    monkeypatch.setattr(steering_module, "create_sae", recording_create_sae)
    return seen


def _load(placement):
    service = SteeringService()
    service._device = placement.device
    service._placement = placement
    return asyncio.run(service.load_sae(Path("/saes/sae_1"), "sae_1", layer=3, architecture="standard"))


def test_on_one_card_only_the_fp16_copy_moves_to_the_card(sae_on_disk):
    loaded = _load(ONE_CARD)

    assert sae_on_disk["read_on"] == ["cpu"], "the fp32 state dict was read onto the card"
    assert sae_on_disk["moves"] == [("cuda:1", ["torch.float16"])], "the card was handed the fp32 module"
    assert loaded.device == "cuda:1"
    assert {p.dtype for p in loaded.model.parameters()} == {torch.float16}


def test_a_splits_sae_stays_on_the_cpu_until_its_hook_places_it(sae_on_disk):
    loaded = _load(SPLIT)

    assert sae_on_disk["read_on"] == ["cpu"]
    assert all(device == "cpu" for device, _dtypes in sae_on_disk["moves"]), sae_on_disk["moves"]
    assert loaded.device == "cpu"
    assert {p.dtype for p in loaded.model.parameters()} == {torch.float16}
