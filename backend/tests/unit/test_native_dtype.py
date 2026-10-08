"""The precision rule: a row loads at the checkpoint's own 16-bit dtype, never a hardcoded one.

The table in `docs/schemas/native-dtype-cases.json` is the rule. miLLM tests its own resolver
against the same file, and `test_the_case_table_is_identical_in_millm` keeps the two copies
byte-identical — so the producer and the server cannot drift onto different rules, which is the
defect this module exists to end (probes fitted on float16 activations, served over bfloat16).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from src.ml.native_dtype import (
    checkpoint_dtype_of,
    normalise_dtype_name,
    resolve_for_config,
    resolve_for_snapshot,
    resolve_load_dtype,
)

REPO = Path(__file__).resolve().parents[3]
CASES_PATH = REPO / "docs" / "schemas" / "native-dtype-cases.json"
MILLM_CASES = Path(os.environ.get("MILLM_REPO", "/home/x-sean/app/miLLM")) / "docs" / "schemas" / "native-dtype-cases.json"
CASES = json.loads(CASES_PATH.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['quantization']}-{c['checkpoint_dtype']}-pq{c['pre_quantized']}")
def test_every_case_in_the_shared_table(case):
    resolved = resolve_load_dtype(
        case["quantization"], case["checkpoint_dtype"], pre_quantized=case["pre_quantized"]
    )
    assert resolved.name == case["loads_at"]
    assert resolved.torch_dtype is getattr(torch, case["loads_at"])
    assert resolved.storage_name == case["storage_dtype"]
    assert resolved.source == case["source"]
    assert resolved.checkpoint_dtype == case["checkpoint_dtype"]


def test_the_table_covers_every_row_and_every_checkpoint_dtype():
    """A rule tested on half its inputs is a rule half-pinned."""
    seen = {(c["quantization"], c["checkpoint_dtype"]) for c in CASES if not c["pre_quantized"]}
    for quant in ("FP32", "FP16", "Q8", "Q4", "Q2"):
        for ckpt in ("bfloat16", "float16", "float32", None):
            assert (quant, ckpt) in seen


def test_the_case_table_is_identical_in_millm():
    """⚠ TWO RESOLVERS, ONE RULE. If miLLM's copy differs, the two repos load different dtypes."""
    if not MILLM_CASES.exists():
        if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(f"miLLM's copy of the case table is missing at {MILLM_CASES}")
        pytest.skip("miLLM checkout not present")
    assert MILLM_CASES.read_bytes() == CASES_PATH.read_bytes()


class TestTheBf16CheckpointThisWasFoundOn:
    def test_llama_3_1_fp16_row_loads_bfloat16(self):
        """The case that failed parity: an FP16 row of a checkpoint published in bfloat16."""
        resolved = resolve_for_config("FP16", {"model_type": "llama", "torch_dtype": "bfloat16"})
        assert resolved.torch_dtype is torch.bfloat16
        assert resolved.as_record() == {
            "model_dtype": "bfloat16",
            "checkpoint_dtype": "bfloat16",
            "dtype_source": "config",
            "activation_storage_dtype": "float16",
            "quantization": "FP16",
        }

    def test_a_q4_row_computes_at_the_checkpoints_dtype_not_float16(self):
        assert resolve_load_dtype("Q4", "bfloat16").torch_dtype is torch.bfloat16

    def test_fp32_row_loads_float32_and_stores_float32(self):
        resolved = resolve_load_dtype("FP32", "bfloat16")
        assert resolved.torch_dtype is torch.float32
        assert resolved.storage_name == "float32"


class TestReadingTheCheckpoint:
    def test_read_order_dtype_then_torch_dtype_then_text_config(self):
        assert checkpoint_dtype_of({"dtype": "float16", "torch_dtype": "bfloat16"}) == ("float16", "config")
        assert checkpoint_dtype_of({"torch_dtype": "bfloat16"}) == ("bfloat16", "config")
        assert checkpoint_dtype_of({"text_config": {"torch_dtype": "bfloat16"}}) == ("bfloat16", "text_config")
        assert checkpoint_dtype_of({"text_config": {"dtype": "float16"}}) == ("float16", "text_config")

    def test_nothing_recorded_is_None_not_a_default(self):
        """⚠ A default presented as the checkpoint's dtype would be a fabricated fact."""
        assert checkpoint_dtype_of({}) == (None, "default")
        assert checkpoint_dtype_of(None) == (None, "default")
        resolved = resolve_for_config("FP16", {})
        assert resolved.checkpoint_dtype is None and resolved.source == "default"
        assert resolved.name == "bfloat16"

    def test_a_transformers_config_object_without_a_recorded_dtype_reads_as_nothing(self, tmp_path):
        from transformers import AutoConfig

        small = dict(model_type="llama", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=4, vocab_size=100)
        (tmp_path / "config.json").write_text(json.dumps(small))
        assert checkpoint_dtype_of(AutoConfig.from_pretrained(tmp_path)) == (None, "default")
        (tmp_path / "config.json").write_text(json.dumps(dict(small, torch_dtype="bfloat16")))
        assert checkpoint_dtype_of(AutoConfig.from_pretrained(tmp_path)) == ("bfloat16", "config")

    def test_snapshot_reading(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"torch_dtype": "float16"}))
        assert resolve_for_snapshot("FP16", tmp_path).torch_dtype is torch.float16
        assert resolve_for_snapshot("FP16", tmp_path / "missing").source == "default"

    @pytest.mark.parametrize("value,expected", [
        (torch.bfloat16, "bfloat16"), ("torch.bfloat16", "bfloat16"), ("bf16", "bfloat16"),
        ("half", "float16"), ("float", "float32"), ("int8", None), (None, None),
    ])
    def test_normalisation(self, value, expected):
        assert normalise_dtype_name(value) == expected

    def test_an_unknown_quantization_is_refused(self):
        with pytest.raises(ValueError):
            resolve_load_dtype("Q3", "bfloat16")


def test_a_pre_quantized_config_is_detected_by_the_resolver():
    """An FP32 row on a GPTQ checkpoint resolves 16-bit (shared table v2)."""
    assert resolve_for_config("FP32", {"torch_dtype": "bfloat16", "quantization_config": {"quant_method": "gptq"}}).name == "bfloat16"
    assert resolve_for_config("FP32", {"torch_dtype": "bfloat16"}).name == "float32"
