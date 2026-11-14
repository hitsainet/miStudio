"""OSD-37c — the GPU preflight sizes vision and multimodal configs (2026-10-16).

`estimate_parameter_count` required `vocab_size`, which no vision config has, so it
returned None and `preflight_gpu_capacity` skipped — for exactly the family vision work
loads. And it read each field from the FIRST nested sub-config carrying it, so
`Blip2Config` sized itself from `qformer_config`: ~0.16B for a ~3.7B model, a ~23x
underestimate that passed preflight and then OOMed. Real transformers configs here, not
namespaces shaped to agree with the code.
"""

import pytest
from transformers import Blip2Config, CLIPConfig, CLIPVisionConfig, LlamaConfig, OPTConfig

from src.ml.model_loader import estimate_parameter_count
from src.services import resource_config
from src.services.resource_config import VRAMInsufficientError, preflight_gpu_capacity


def _blip2_opt_2p7b() -> Blip2Config:
    """Salesforce/blip2-opt-2.7b's shape: ViT-g vision, a Q-Former, OPT-2.7B text."""
    text = OPTConfig(hidden_size=2560, num_hidden_layers=32, ffn_dim=10240,
                     num_attention_heads=32, vocab_size=50272)
    return Blip2Config(text_config=text.to_dict())


def test_a_plain_decoder_is_sized_as_before():
    c = LlamaConfig(hidden_size=2048, num_hidden_layers=16, intermediate_size=8192, vocab_size=64000)
    assert estimate_parameter_count(c) == 16 * (4 * 2048**2 + 3 * 2048 * 8192) + 2 * 64000 * 2048


def test_a_vision_config_is_sized_not_skipped():
    est = estimate_parameter_count(CLIPVisionConfig())  # ViT-B/32, ~88M real
    assert est is not None
    assert 80e6 <= est <= 160e6


def test_blip2_counts_every_tower():
    est = estimate_parameter_count(_blip2_opt_2p7b())
    assert est is not None
    # ~3.8B real. Rounded UP on purpose (3·h·i MLP, untied embeddings); never the
    # ~0.16B the Q-Former alone gives.
    assert 3.0e9 <= est <= 6.5e9, est


def test_a_two_tower_clip_counts_both():
    clip = CLIPConfig()
    text_only = estimate_parameter_count(clip.text_config)
    vision_only = estimate_parameter_count(clip.vision_config)
    assert estimate_parameter_count(clip) == text_only + vision_only


def test_the_preflight_now_refuses_a_vision_model_that_cannot_fit(monkeypatch):
    """The wiring that matters: a vision estimate reaches the OOM guard and it bites."""
    monkeypatch.setattr(
        resource_config, "_usable_gpu_memory",
        lambda device: (1 * 1024**3, 24 * 1024**3, "GPU 0 (test)"),
    )
    with pytest.raises(VRAMInsufficientError, match="does not fit"):
        preflight_gpu_capacity(
            params_count=estimate_parameter_count(_blip2_opt_2p7b()),
            quantization="FP16", device="cuda:0", model_name="blip2-opt-2.7b",
        )
