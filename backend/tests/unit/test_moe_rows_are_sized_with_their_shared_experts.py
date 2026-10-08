"""A quantized MoE row is sized from its architecture with every expert it holds, shared ones included.

Multi-GPU Phase 2, review round 3 (2026-09-14). Round 2 (301026cc) sized a Q4 row from its
recorded `architecture_config` and counted routed and shared experts, by the field names
Mixtral, Qwen-MoE and DeepSeek use. Granite's shared MLP (`shared_intermediate_size`,
granitemoeshared and granitemoehybrid) and ERNIE's experts (`moe_num_experts`,
`moe_num_shared_experts`) were neither recorded nor counted: a Q4 granite-4.0-h-small was
sized without about 4B of its 32B parameters (3 x 4,096 x 8,192 a layer over 40 layers).

The oracle is not a hand formula: it is the parameter count of the model each config
class BUILDS, on the meta device, from transformers' own implementation. The estimate
must never fall below it by more than the router and norm weights it does not model.

MUTATION CONTROLS (review round 3, 2026-09-14; scratchpad p2-r3/mutate.py + mutations_r3.json,
each alone in a private copy of backend/, restored by sha256):
  N37  SHAPE_FIELDS drops shared_intermediate_size           -> [granitemoeshared], [granitemoehybrid]
  N38  the shared width read only as shared_expert_...       -> the same two
  N40  the shared count read only as n_shared_experts        -> [ernie4_5_moe]
  N40b SHAPE_FIELDS drops moe_num_shared_experts             -> [ernie4_5_moe]
  N39  moe_num_experts dropped from the routed spellings     -> SURVIVED, and it was equivalent: ERNIE's config
       answers num_experts, which SHAPE_FIELDS already records. The spelling was removed (6064a51d) rather
       than kept unpinned.
Round 2's Q8, Q10, R-Q1, R-Q2, R-Q5 re-run, and Q9 re-expressed on the new shared-count line
(`shared = 0`): all killed.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from src.ml.model_loader import extract_architecture_config
from src.services.base_model_budget import params_for_sizing

COMMON = dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
              vocab_size=1000, intermediate_size=512, tie_word_embeddings=False)

#: Each config class's own spelling of its experts, with a shared part where it has one.
CASES = {
    "granitemoeshared": dict(num_local_experts=8, shared_intermediate_size=384),
    "granitemoehybrid": dict(num_local_experts=8, shared_intermediate_size=384, layer_types=["attention"] * 4),
    "ernie4_5_moe": dict(moe_num_experts=8, moe_intermediate_size=128, moe_num_shared_experts=2,
                         moe_layer_start_index=0, moe_layer_end_index=3),
    # Already counted by round 2; kept so a change for the new spellings cannot lose them.
    "mixtral": dict(num_local_experts=8),
    "qwen2_moe": dict(num_experts=8, moe_intermediate_size=128, shared_expert_intermediate_size=384),
    "deepseek_v3": dict(n_routed_experts=8, moe_intermediate_size=128, n_shared_experts=2, first_k_dense_replace=0,
                        q_lora_rank=None, kv_lora_rank=64, qk_rope_head_dim=16, qk_nope_head_dim=32, v_head_dim=64),
}

#: What the estimate does not model and may miss: router gates and norms, under 1% at these widths.
TOLERANCE = 0.97


@pytest.mark.parametrize("model_type", sorted(CASES))
def test_a_packed_row_is_described_with_every_expert_its_model_builds(model_type):
    config = CONFIG_MAPPING[model_type](**{**COMMON, **CASES[model_type]})
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(copy.deepcopy(config), dtype=torch.float16)
    built = sum(p.numel() for p in model.parameters())

    # A packed count of 1 leaves the description as the size.
    described = params_for_sizing(1, "Q4", extract_architecture_config(config))

    assert described >= TOLERANCE * built, (model_type, described, built, described / built)
