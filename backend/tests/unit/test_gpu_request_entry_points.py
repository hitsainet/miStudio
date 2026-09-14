"""The two places a job meets the GPU resolver: submit and start.

* Submit (API): ``resolve_request`` turns the request into what the row stores —
  ``"auto"`` or a UUID. An index becomes the UUID it names NOW, because a queued
  job may start after a card is added and the indices shift. An unknown card is
  a 400 from ``resolve_gpu_request``, before any row exists.
* Start (worker): ``place_job`` chooses against live free memory and makes the
  card CURRENT, so code that asks CUDA for "the current device" (bitsandbytes,
  bare ``torch.cuda`` calls) lands on the chosen card instead of index 0.

MUTATION CONTROLS:
  * resolve_request returns the index unchanged       -> "index becomes the UUID" fails
  * place_job drops torch.cuda.set_device             -> "made current" fails
  * place_job returns cpu for a named card with no CUDA -> "refused without CUDA" fails
  * GPU_REQUEST_PATTERN accepts any string            -> "rejects" fails
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

from src.api.v1.gpu_request import resolve_gpu_request
from src.schemas.gpu import AUTO, GPU_REQUEST_DESCRIPTION, GpuRequestStr
from src.services import gpu_placement
from src.services.gpu_placement import GpuCard, GpuPlacementError, place_job, resolve_request

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"

CARDS = [
    GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000),
    GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
]


# ── Submit ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("requested", [None, "", "auto", " AUTO "])
def test_auto_is_stored_as_auto(requested):
    assert resolve_request(requested, cards=CARDS) == AUTO


def test_auto_needs_no_inventory(monkeypatch):
    """The API must not read NVML just to store "auto"."""
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: pytest.fail("NVML read for auto"))
    assert resolve_request("auto") == AUTO


@pytest.mark.parametrize("requested", [1, "1"])
def test_an_index_becomes_the_uuid_it_names_now(requested):
    assert resolve_request(requested, cards=CARDS) == RTX_UUID


def test_a_uuid_is_stored_as_nvml_reports_it():
    assert resolve_request(RTX_UUID.lower(), cards=CARDS) == RTX_UUID
    assert resolve_request(RTX_UUID[len("GPU-"):], cards=CARDS) == RTX_UUID


def test_submit_does_not_judge_free_memory():
    """A queued job is judged when it starts; a busy card can still be named."""
    busy = [CARDS[0], GpuCard(1, RTX_UUID, "NVIDIA GeForce RTX 3090", 24_576, free_mb=10)]
    assert resolve_request(RTX_UUID, cards=busy) == RTX_UUID


def test_an_unknown_card_is_refused_with_the_inventory():
    with pytest.raises(GpuPlacementError) as exc:
        resolve_request(5, cards=CARDS)
    assert "RTX 3090" in str(exc.value) and "RTX 3080 Ti" in str(exc.value)


def test_the_endpoint_helper_turns_an_unknown_card_into_a_400(monkeypatch):
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)
    with pytest.raises(HTTPException) as exc:
        resolve_gpu_request("GPU-00000000-0000-0000-0000-000000000000", can_split=True)
    assert exc.value.status_code == 400
    assert "No GPU" in exc.value.detail
    assert resolve_gpu_request("0", can_split=True) == TI_UUID


# ── The request field ──────────────────────────────────────────────────────

class _Request(BaseModel):
    gpu: GpuRequestStr = Field(AUTO, description=GPU_REQUEST_DESCRIPTION)


def test_the_field_defaults_to_auto():
    assert _Request().gpu == "auto"


@pytest.mark.parametrize("value", ["auto", "0", "12", RTX_UUID, RTX_UUID.lower(), RTX_UUID[4:], " auto "])
def test_the_field_accepts(value):
    assert _Request(gpu=value).gpu == value.strip()


@pytest.mark.parametrize("value", ["cuda:0", "cuda", "GPU-nothex", "any", "123", "-1", "auto; rm"])
def test_the_field_rejects(value):
    with pytest.raises(ValidationError):
        _Request(gpu=value)


# ── Start ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cuda(monkeypatch):
    """Torch sees both cards; returns the list of devices made current."""
    import torch

    made_current = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    # torch reports the UUID without NVML's prefix.
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid=[TI_UUID, RTX_UUID][index][len("GPU-"):]),
    )
    monkeypatch.setattr(torch.cuda, "set_device", made_current.append)
    return made_current


def test_auto_is_placed_on_the_most_free_card_and_made_current(cuda):
    import torch

    placement = place_job("auto", cards=CARDS)
    assert placement.uuid == RTX_UUID
    assert placement.device == torch.device("cuda", 1)
    assert cuda == [torch.device("cuda", 1)]


def test_a_named_card_is_placed_and_made_current(cuda):
    import torch

    placement = place_job(TI_UUID, required_mb=4_000, cards=CARDS)
    assert (placement.uuid, placement.device) == (TI_UUID, torch.device("cuda", 0))
    assert cuda == [torch.device("cuda", 0)]


def test_a_named_card_without_room_is_refused_and_nothing_is_made_current(cuda):
    with pytest.raises(GpuPlacementError, match="Choose another GPU or Auto"):
        place_job(TI_UUID, required_mb=15_000, cards=CARDS)
    assert cuda == []


def test_auto_without_cuda_runs_on_the_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    placement = place_job("auto", cards=CARDS)
    assert (placement.card, placement.device, placement.uuid) == (None, torch.device("cpu"), None)


def test_a_named_card_is_refused_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(GpuPlacementError, match="CUDA is not available"):
        place_job(RTX_UUID, cards=CARDS)
