"""A base model's size, and its budget on a split beside the SAE that works on the first card.

`Placement.max_memory` hands every card of a split to the model, less a 1 GB
reserve each. On-the-fly SAE training and SAE feature extraction keep their SAE on
the placement's first card, and accelerate fills that card up to its budget — so
without a carve-out the model takes the memory the SAE was placed there to use.

MUTATION CONTROLS (2026-09-14; each applied alone, this file +
test_split_sae_feature_extraction.py + test_training_gpu_placement.py run,
source restored and checked by sha256). All went red:
  B1 carve from min(budgets) (card 0) not the SAE's card     -> sae_share_comes_off_the_sae_card_only, refusal,
                                                                extraction and training split tests
  B2 no refusal when the SAE's card cannot hold the SAE      -> an_sae_card_that_cannot_hold_the_sae_is_refused
  B3 the carve-out is not applied                            -> sae_share..., extraction and training split tests
  B4 base_model_mb ignores the quantization (always FP16)    -> a_base_model_is_sized_at_its_quantization[FP32, Q8, Q4]

A Q4 ROW COUNTS ITS PARAMETERS PACKED (review round 1, 2026-09-14): sized from its
architecture_config instead. Controls (scratchpad p2-r1-core/mutate.py, each alone,
this file + the two harness files run, source restored and checked by sha256):
  Q1 no format is treated as packed                       -> a_packed_row_is_sized_from_its_architecture[Q4, Q2],
                                                                the training and SAE-extraction Q4 placement tests
  Q2 the description replaces the row's count (no max)    -> a_packed_row_is_never_sized_below_its_own_count
  Q3 training does not pass architecture_config           -> test_training_gpu_placement::a_four_bit_base_model...
  Q4 SAE extraction does not pass architecture_config     -> test_split_sae_feature_extraction::a_four_bit_model...
  Q5 every format is treated as packed                    -> an_unpacked_row_keeps_its_own_count

REVIEW ROUND 2 (2026-09-14). The packed-count fix reached training and SAE extraction
only; circuits and J-lens sized a Q4 row from its packed count (`params_for_sizing` is
now the one place every placement sizer reads a row's count). And the architecture
estimate counted one MLP a layer, so an MoE row (Mixtral-8x7B) was still sized at half.
Controls (scratchpad p2-r2-place/mutate.py, each alone, source restored by sha256):
  Q6  circuit_required_mb passes no architecture_config   -> test_circuits_run_split::a_four_bit_row...
  Q7  estimate_weights_mb passes no architecture_config     -> test_jlens_split_gpu::a_four_bit_row...
  Q8  experts beyond one MLP are not counted               -> a_packed_mixture_of_experts..., shared_and_routed...
  Q9  shared experts are not counted                       -> shared_and_routed_experts_are_both_counted
  Q10 moe_intermediate_size ignored                        -> shared_and_routed_experts_are_both_counted
  Q11 circuit_required_mb bypasses params_for_sizing        -> test_circuits_run_split::a_four_bit_row...
All red. Round 1's Q1-Q5 re-run on the moved lines (Q2 is now `return max(...)`, Q5
`if fmt in PACKED_FORMATS`): all red, Q1 now also in the circuit and J-lens tests.
"""

import math

import pytest
import torch

from src.services.base_model_budget import (
    SAE_WORKING_RESERVE_MB,
    base_model_mb,
    budget_beside_sae,
    inference_sae_card_mb,
)
from src.services.gpu_placement import GpuCard, GpuPlacementError, Placement

MiB = 1024**2

TI = GpuCard(index=0, uuid="GPU-ti", name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000)
RTX = GpuCard(index=1, uuid="GPU-rtx", name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000)

#: As place_job builds it: most free first, so the SAE's card is torch index 1.
#: With the SAE's card at index 0 a carve-out from "the first key" would pass.
SPLIT = Placement(
    card=RTX, device=torch.device("cuda", 1), cards=(RTX, TI),
    devices=(torch.device("cuda", 1), torch.device("cuda", 0)),
    max_memory_mb={1: 21_976, 0: 9_976},
)
ONE = Placement(card=RTX, device=torch.device("cuda", 1))


def test_the_sae_share_comes_off_the_sae_card_only():
    assert budget_beside_sae(SPLIT, 3_000.4) == {1: "18975MiB", 0: "9976MiB"}


def test_a_single_card_placement_has_no_budget():
    assert budget_beside_sae(ONE, 3_000) is None


def test_the_budget_never_offers_the_cpu():
    assert "cpu" not in budget_beside_sae(SPLIT, 3_000)


def test_the_placement_is_left_as_it_was():
    budget_beside_sae(SPLIT, 3_000)
    assert SPLIT.max_memory_mb == {1: 21_976, 0: 9_976}


def test_an_sae_card_that_cannot_hold_the_sae_is_refused_with_the_figures():
    with pytest.raises(GpuPlacementError, match="SAE's card") as exc:
        budget_beside_sae(SPLIT, 22_000)
    assert "21,976 MB budget" in str(exc.value)


@pytest.mark.parametrize("quantization, bytes_per_param", [("FP16", 2), ("FP32", 4), ("Q8", 1), ("Q4", 0.5)])
def test_a_base_model_is_sized_at_its_quantization(quantization, bytes_per_param):
    params = 1_000_000_000
    assert base_model_mb(params, quantization) == int(params * bytes_per_param * 1.2) / MiB


def test_the_rows_own_enum_is_accepted():
    from src.models.model import QuantizationFormat

    assert base_model_mb(1_000_000_000, QuantizationFormat.FP16) == base_model_mb(1_000_000_000, "FP16")


@pytest.mark.parametrize("params", [None, 0, -5, True, "7B", 7.0])
def test_an_unknown_parameter_count_has_no_size(params):
    assert base_model_mb(params, "FP16") is None


def test_an_unknown_format_has_no_size():
    assert base_model_mb(1_000_000_000, "GGUF") is None


def test_an_inference_sae_needs_its_weights_and_working_memory():
    weights = (2 * 2304 * 16384 + 2304 + 16384) * 4 / MiB
    assert inference_sae_card_mb(2304, 16384) == pytest.approx(weights + SAE_WORKING_RESERVE_MB)
    assert math.isclose(inference_sae_card_mb(None, None), SAE_WORKING_RESERVE_MB)


# ── A Q4 row counts its parameters packed (review round 1, 2026-09-14) ─────
#
# `params_count` is counted off the model the download loaded at the row's
# quantization. A quantized bitsandbytes 4-bit weight reports half its elements
# (Params4bit of a 64x128 weight: numel 4,096), so a Q4 model was sized at about a
# quarter of its weights and Auto could pick a card, or a split, that cannot hold it.

LLAMA_8B = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 128_256, "intermediate_size": 14_336}


def _described(architecture):
    from types import SimpleNamespace

    from src.ml.model_loader import estimate_parameter_count

    return estimate_parameter_count(SimpleNamespace(**architecture))


@pytest.mark.parametrize("quantization", ["Q4", "Q2"])
def test_a_packed_row_is_sized_from_its_architecture(quantization):
    from src.ml.model_loader import QuantizationFormat, estimate_model_memory

    described = _described(LLAMA_8B)
    expected = estimate_model_memory(described, QuantizationFormat(quantization)) / MiB
    assert base_model_mb(described // 2, quantization, LLAMA_8B) == expected


def test_an_unpacked_row_keeps_its_own_count():
    assert base_model_mb(8_030_000_000, "FP16", LLAMA_8B) == base_model_mb(8_030_000_000, "FP16")
    assert base_model_mb(8_030_000_000, "Q8", LLAMA_8B) == base_model_mb(8_030_000_000, "Q8")


def test_a_packed_row_is_never_sized_below_its_own_count():
    tiny = {"hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 100, "intermediate_size": 128}
    assert base_model_mb(8_000_000_000, "Q4", tiny) == base_model_mb(8_000_000_000, "Q4")


@pytest.mark.parametrize("architecture", [None, {}, {"model_type": "llama"}])
def test_a_packed_row_without_a_usable_description_keeps_its_count(architecture):
    assert base_model_mb(4_000_000_000, "Q4", architecture) == base_model_mb(4_000_000_000, "Q4")


# ── A mixture-of-experts row (review round 2, 2026-09-14) ──────────────────
#
# The dense description counts ONE MLP a layer, so an 8-expert model is described
# at about a sixth of its weights. The packed count is about half, so the max of
# the two still sized a Q4 Mixtral-8x7B at half its weights.

#: Mixtral-8x7B's shape. Worked by hand:
#:   attention 32 x 4*4096^2              =  2,147,483,648
#:   experts   32 x 8 x 3*4096*14336      = 45,097,156,608
#:   embedding + head 2 x 32000 x 4096   =    262,144,000
#:   total                                = 47,506,784,256
MIXTRAL = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32_000,
           "intermediate_size": 14_336, "num_local_experts": 8, "num_experts_per_tok": 2}
MIXTRAL_PARAMS = 47_506_784_256
#: What the Q4 download counts: the linear weights at half, the fp16 embeddings in full.
MIXTRAL_PACKED = (MIXTRAL_PARAMS - 262_144_000) // 2 + 262_144_000


def test_a_packed_mixture_of_experts_row_counts_every_expert():
    from src.services.base_model_budget import params_for_sizing

    assert params_for_sizing(MIXTRAL_PACKED, "Q4", MIXTRAL) == MIXTRAL_PARAMS


def test_shared_and_routed_experts_are_both_counted():
    """DeepSeek/Qwen-MoE spell it n_routed_experts / num_experts with a moe_intermediate_size,
    beside always-on shared experts. 4 x (4*1024^2 + 16*3*1024*512 + 2*3*1024*512) + 2*1000*1024."""
    from src.services.base_model_budget import params_for_sizing

    deepseek = {"hidden_size": 1024, "num_hidden_layers": 4, "vocab_size": 1000, "intermediate_size": 4096,
                "n_routed_experts": 16, "n_shared_experts": 2, "moe_intermediate_size": 512}
    expected = 4 * (4 * 1024**2 + 16 * 3 * 1024 * 512 + 2 * 3 * 1024 * 512) + 2 * 1000 * 1024
    assert params_for_sizing(1, "Q4", deepseek) == expected

    qwen = {"hidden_size": 1024, "num_hidden_layers": 4, "vocab_size": 1000, "intermediate_size": 4096,
            "num_experts": 16, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 2048}
    expected = 4 * (4 * 1024**2 + 16 * 3 * 1024 * 512 + 3 * 1024 * 2048) + 2 * 1000 * 1024
    assert params_for_sizing(1, "Q4", qwen) == expected


def test_an_unpacked_mixture_of_experts_row_keeps_its_own_count():
    from src.services.base_model_budget import params_for_sizing

    assert params_for_sizing(MIXTRAL_PARAMS, "Q8", MIXTRAL) == MIXTRAL_PARAMS
    assert params_for_sizing(MIXTRAL_PARAMS, "FP16", MIXTRAL) == MIXTRAL_PARAMS


def test_a_single_expert_description_is_a_dense_model():
    from src.services.base_model_budget import params_for_sizing

    assert params_for_sizing(1, "Q4", {**LLAMA_8B, "num_local_experts": 1}) == _described(LLAMA_8B)
