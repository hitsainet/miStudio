"""A model no single GPU can hold is split across GPUs — only where the job can run split.

Multi-GPU Phase 2 (0xcc/plans/Multi-GPU-Plan.md, decision D4): one card whenever
one card fits; otherwise the FEWEST cards, most free first, whose budgets (free
memory less SHARD_RESERVE_MB each) cover the job; otherwise refused with the
figures. `"all"` is an explicit choice — honoured by a job that can split,
refused by one that cannot. Written for the node's RTX 3080 Ti at index 0 beside
the RTX 3090 at index 1, and for the next card.
"""

from unittest.mock import patch

import pytest

from src.schemas import gpu as gpu_schema
from src.services import gpu_placement as gp
from src.services.gpu_placement import ALL, SHARD_RESERVE_MB, GpuCard, GpuPlacementError

UUID_3080TI = "GPU-f47ba814-49a2-603f-3595-275284140251"
UUID_3090 = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"

TI = GpuCard(index=0, uuid=UUID_3080TI, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_900)
RTX = GpuCard(index=1, uuid=UUID_3090, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=21_500)
TWO = [TI, RTX]

#: gemma-4-12B-it at bf16 with headroom: more than either card, less than both.
TOO_BIG_FOR_ONE = 26_000


class TestOneCardWheneverOneFits:
    def test_a_job_that_fits_one_card_is_never_split(self):
        assert gp.resolve_cards("auto", required_mb=8_000, cards=TWO, allow_shard=True) == (RTX,)

    def test_a_job_that_cannot_split_gets_the_single_card_answer(self):
        with pytest.raises(GpuPlacementError, match="No single GPU"):
            gp.resolve_cards("auto", required_mb=TOO_BIG_FOR_ONE, cards=TWO, allow_shard=False)

    def test_a_named_card_is_never_split(self):
        with pytest.raises(GpuPlacementError, match="GPU 0"):
            gp.resolve_cards(0, required_mb=TOO_BIG_FOR_ONE, cards=TWO, allow_shard=True)


class TestAutoSplit:
    def test_too_big_for_one_card_splits_most_free_first(self):
        assert gp.resolve_cards("auto", required_mb=TOO_BIG_FOR_ONE, cards=TWO, allow_shard=True) == (RTX, TI)

    def test_takes_only_as_many_cards_as_the_job_needs(self):
        spare = GpuCard(index=2, uuid="GPU-spare", name="RTX 4090", total_mb=24_576, free_mb=20_000)
        # 21,500 + 20,000 less two reserves covers 26 GB; the 3080 Ti stays free.
        assert gp.resolve_cards("auto", required_mb=TOO_BIG_FOR_ONE, cards=TWO + [spare], allow_shard=True) == (RTX, spare)

    def test_budgets_keep_the_reserve_on_every_card(self):
        exactly = (RTX.free_mb - SHARD_RESERVE_MB) + (TI.free_mb - SHARD_RESERVE_MB)
        assert gp.resolve_cards("auto", required_mb=exactly, cards=TWO, allow_shard=True) == (RTX, TI)
        with pytest.raises(GpuPlacementError) as raised:
            gp.resolve_cards("auto", required_mb=exactly + 1, cards=TWO, allow_shard=True)
        assert raised.value.required_mb == exactly + 1
        assert [card["free_mb"] for card in raised.value.details()["cards"]] == [11_900, 21_500]
        assert f"{SHARD_RESERVE_MB:,} MB kept free" in str(raised.value)

    def test_one_card_has_nothing_to_split_across(self):
        with pytest.raises(GpuPlacementError, match="No single GPU"):
            gp.resolve_cards("auto", required_mb=TOO_BIG_FOR_ONE, cards=[RTX], allow_shard=True)

    def test_an_unknown_size_is_never_split(self):
        assert gp.resolve_cards("auto", required_mb=None, cards=TWO, allow_shard=True) == (RTX,)


class TestAll:
    @pytest.mark.parametrize("requested", ["all", "ALL", " all "])
    def test_all_is_every_card_most_free_first(self, requested):
        assert gp.resolve_cards(requested, required_mb=2_000, cards=TWO, allow_shard=True) == (RTX, TI)

    def test_all_is_refused_by_a_job_that_cannot_split(self):
        with pytest.raises(GpuPlacementError, match="cannot run split"):
            gp.resolve_cards(ALL, required_mb=2_000, cards=TWO, allow_shard=False)

    def test_all_is_refused_when_every_card_together_is_too_small(self):
        with pytest.raises(GpuPlacementError, match="All 2 GPUs together"):
            gp.resolve_cards(ALL, required_mb=40_000, cards=TWO, allow_shard=True)

    def test_all_is_recorded_as_all_at_submit(self):
        assert gp.resolve_request("all", cards=TWO) == ALL

    def test_the_request_schema_accepts_all_and_matches_placement(self):
        import re

        assert gpu_schema.ALL == gp.ALL
        assert re.match(gpu_schema.GPU_REQUEST_PATTERN, "all")


class _Props:
    def __init__(self, uuid):
        self.uuid = uuid


def _torch_sees(order):
    """torch indices in `order` (UUIDs, no prefix as torch reports them)."""
    return [
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=len(order)),
        patch("torch.cuda.get_device_properties", side_effect=lambda index: _Props(order[index])),
        patch("torch.cuda.set_device"),
    ]


class TestPlaceJob:
    def _place(self, torch_order, **kwargs):
        patches = _torch_sees(torch_order)
        for p in patches:
            p.start()
        try:
            return gp.place_job(cards=TWO, **kwargs)
        finally:
            for p in patches:
                p.stop()

    def test_a_split_carries_every_card_its_devices_and_a_gpu_only_budget(self):
        placement = self._place(
            [UUID_3080TI.removeprefix("GPU-"), UUID_3090.removeprefix("GPU-")],
            requested="auto", required_mb=TOO_BIG_FOR_ONE, allow_shard=True,
        )
        assert placement.is_shard
        assert placement.card == RTX and str(placement.device) == "cuda:1"
        assert placement.cards == (RTX, TI)
        assert [str(d) for d in placement.all_devices] == ["cuda:1", "cuda:0"]
        # Sequential: "auto" rebalances max_memory to ~model/N per card and can
        # map a model the budgets hold to disk (transformers _get_device_map).
        assert placement.device_map == "sequential"
        assert placement.max_memory == {
            1: f"{RTX.free_mb - SHARD_RESERVE_MB}MiB",
            0: f"{TI.free_mb - SHARD_RESERVE_MB}MiB",
        }
        assert "cpu" not in placement.max_memory and "disk" not in placement.max_memory
        assert placement.gpu_columns() == {"gpu_uuid": UUID_3090, "gpu_uuids": [UUID_3090, UUID_3080TI]}

    def test_budget_keys_are_torch_indices_not_nvml_indices(self):
        # CUDA_VISIBLE_DEVICES reordered: torch sees the 3090 first.
        placement = self._place(
            [UUID_3090.removeprefix("GPU-"), UUID_3080TI.removeprefix("GPU-")],
            requested="all", required_mb=2_000, allow_shard=True,
        )
        assert set(placement.max_memory) == {0, 1}
        assert placement.max_memory[0] == f"{RTX.free_mb - SHARD_RESERVE_MB}MiB"

    def test_a_single_card_placement_reads_exactly_as_before(self):
        placement = self._place(
            [UUID_3080TI.removeprefix("GPU-"), UUID_3090.removeprefix("GPU-")],
            requested="auto", required_mb=8_000, allow_shard=True,
        )
        assert not placement.is_shard
        assert placement.cards == () and placement.max_memory is None
        assert placement.device_map == "cuda:1"
        assert placement.gpu_columns() == {"gpu_uuid": UUID_3090, "gpu_uuids": None}

    def test_the_default_never_splits(self):
        with pytest.raises(GpuPlacementError, match="No single GPU"):
            self._place(
                [UUID_3080TI.removeprefix("GPU-"), UUID_3090.removeprefix("GPU-")],
                requested="auto", required_mb=TOO_BIG_FOR_ONE,
            )
