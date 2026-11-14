"""GPU choice: Auto takes the most free card that fits; an explicit card is honoured or refused.

Written for the two-card node (RTX 3080 Ti at index 0, RTX 3090 at index 1) and
for the next card, whatever its index.
"""

import sys
import types
from unittest.mock import patch

import pytest

from src.services import gpu_placement as gp
from src.services.gpu_placement import GpuCard, GpuPlacementError

UUID_3080TI = "GPU-f47ba814-49a2-603f-3595-275284140251"
UUID_3090 = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"

TWO_CARDS = [
    GpuCard(index=0, uuid=UUID_3080TI, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_900),
    GpuCard(index=1, uuid=UUID_3090, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=24_100),
]


class TestAuto:
    @pytest.mark.parametrize("requested", [None, "auto", "AUTO", "", "  auto "])
    def test_takes_the_card_with_the_most_free_memory(self, requested):
        assert gp.resolve_card(requested, cards=TWO_CARDS).uuid == UUID_3090

    def test_skips_a_card_that_cannot_fit_the_job(self):
        busy_3090 = [TWO_CARDS[0], GpuCard(1, UUID_3090, "RTX 3090", 24_576, 3_000)]
        assert gp.resolve_card("auto", required_mb=8_000, cards=busy_3090).uuid == UUID_3080TI

    def test_most_free_wins_even_when_both_fit(self):
        busy_3090 = [TWO_CARDS[0], GpuCard(1, UUID_3090, "RTX 3090", 24_576, 10_000)]
        assert gp.resolve_card("auto", required_mb=2_000, cards=busy_3090).uuid == UUID_3080TI

    def test_equal_free_memory_prefers_the_lower_index(self):
        tie = [GpuCard(0, "GPU-a", "a", 24_000, 5_000), GpuCard(1, "GPU-b", "b", 24_000, 5_000)]
        assert gp.resolve_card("auto", cards=tie).index == 0

    def test_a_third_card_is_used_with_no_code_change(self):
        three = TWO_CARDS + [GpuCard(2, "GPU-new", "RTX 4090", 24_576, 24_500)]
        assert gp.resolve_card("auto", cards=three).uuid == "GPU-new"

    def test_no_card_fits_is_refused_with_the_figures(self):
        with pytest.raises(GpuPlacementError) as raised:
            gp.resolve_card("auto", required_mb=30_000, cards=TWO_CARDS)

        details = raised.value.details()
        assert details["required_mb"] == 30_000
        assert [card["free_mb"] for card in details["cards"]] == [11_900, 24_100]
        assert "30,000 MB" in str(raised.value)


class TestExplicit:
    @pytest.mark.parametrize("requested", [0, "0"])
    def test_an_index_is_honoured(self, requested):
        assert gp.resolve_card(requested, cards=TWO_CARDS).uuid == UUID_3080TI

    @pytest.mark.parametrize("requested", [UUID_3090, UUID_3090.lower(), UUID_3090.removeprefix("GPU-")])
    def test_a_uuid_is_honoured_with_or_without_prefix(self, requested):
        assert gp.resolve_card(requested, cards=TWO_CARDS).index == 1

    def test_an_explicit_card_without_room_is_refused_not_swapped(self):
        with pytest.raises(GpuPlacementError) as raised:
            gp.resolve_card(0, required_mb=16_000, cards=TWO_CARDS)

        assert "GPU 0" in str(raised.value)
        assert raised.value.required_mb == 16_000

    @pytest.mark.parametrize("requested", [7, "GPU-00000000-0000-0000-0000-000000000000", True])
    def test_a_card_that_is_not_there_is_refused(self, requested):
        with pytest.raises(GpuPlacementError):
            gp.resolve_card(requested, cards=TWO_CARDS)


class TestNoGpus:
    def test_resolving_with_no_gpus_is_an_error(self):
        with pytest.raises(GpuPlacementError, match="No GPU is visible"):
            gp.resolve_card("auto", cards=[])


def _fake_pynvml(cards, init_error=None):
    module = types.SimpleNamespace()
    module.shutdowns = 0

    def nvmlInit():
        if init_error:
            raise init_error

    def nvmlShutdown():
        module.shutdowns += 1

    module.nvmlInit = nvmlInit
    module.nvmlShutdown = nvmlShutdown
    module.nvmlDeviceGetCount = lambda: len(cards)
    module.nvmlDeviceGetHandleByIndex = lambda index: cards[index]
    module.nvmlDeviceGetMemoryInfo = lambda card: types.SimpleNamespace(
        total=card["total_mb"] * 1024 * 1024, free=card["free_mb"] * 1024 * 1024
    )
    module.nvmlDeviceGetUUID = lambda card: card["uuid"]
    module.nvmlDeviceGetName = lambda card: card["name"]
    return module


class TestListCards:
    def test_reads_every_card_from_nvml(self, monkeypatch):
        fake = _fake_pynvml([
            {"uuid": UUID_3080TI.encode(), "name": b"NVIDIA GeForce RTX 3080 Ti", "total_mb": 12_288, "free_mb": 11_900},
            {"uuid": UUID_3090, "name": "NVIDIA GeForce RTX 3090", "total_mb": 24_576, "free_mb": 24_100},
        ])
        monkeypatch.setitem(sys.modules, "pynvml", fake)

        assert gp.list_cards() == TWO_CARDS
        assert fake.shutdowns == 1

    def test_no_driver_means_no_cards(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([], init_error=RuntimeError("NVML Shared Library Not Found")))

        assert gp.list_cards() == []

    def test_live_resolution_reads_nvml(self, monkeypatch):
        monkeypatch.setattr(gp, "list_cards", lambda: TWO_CARDS)

        assert gp.resolve_card().uuid == UUID_3090


class TestTorchDevice:
    def _cuda(self, uuids_in_cuda_order):
        props = [types.SimpleNamespace(uuid=uuid) for uuid in uuids_in_cuda_order]
        return patch.multiple(
            "torch.cuda",
            is_available=lambda: True,
            device_count=lambda: len(props),
            get_device_properties=lambda index: props[index],
        )

    def test_matches_by_uuid_not_by_index(self):
        # CUDA enumerates the 3090 first here (fastest-first, or a restricted
        # CUDA_VISIBLE_DEVICES) while NVML calls it index 1.
        with self._cuda([UUID_3090.removeprefix("GPU-"), UUID_3080TI.removeprefix("GPU-")]):
            device = gp.torch_device(TWO_CARDS[1])

        assert (device.type, device.index) == ("cuda", 0)

    def test_a_uuid_string_works_too(self):
        with self._cuda([UUID_3080TI, UUID_3090]):
            assert gp.torch_device(UUID_3090).index == 1

    def test_a_card_this_process_cannot_see_is_an_error(self):
        with self._cuda([UUID_3080TI]):
            with pytest.raises(GpuPlacementError, match="not visible"):
                gp.torch_device(UUID_3090)

    def test_no_cuda_is_an_error(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(GpuPlacementError, match="CUDA is not available"):
                gp.torch_device(UUID_3090)
