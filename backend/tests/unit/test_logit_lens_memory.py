"""Logit lens must not need the base model, or the GPU.

REPORTED 2026-07-30: "none of the per feature logit lens screens are working on
any of the labeled extractions." Every request returned 500:

    Out of memory loading ibm-granite/granite-4.1-8b even with FP32.
    Model may be too large for available hardware.

The computation is `W_dec[feature] @ W_U` — the code says so in a comment — and
the ONLY thing it used the base model for was `base_model.lm_head.weight`. It
instantiated all 8B parameters (~17 GB) to reach one 822 MB tensor. Once miLLM
held the card serving granite in fp16 (17.5 GB of 24 GB) there was no room and
the feature stopped working entirely.

Verified on the real weights: lm_head.weight is (100352, 4096) bf16, 822 MB,
and reads out of the shard in 1.34s.

MUTATION CONTROLS:
  * make logit lens use cuda again        -> device test fails
  * load the full model on the fast path  -> no-full-load test fails
  * drop the tied-embedding fallback      -> tied test fails
  * delete base_model unconditionally     -> cleanup test fails
"""

import inspect
import json

import pytest
import torch

from src.services import analysis_service
from src.services.analysis_service import (
    load_unembedding_matrix,
    resolve_snapshot_dir,
)


def _write_shard(path, tensors):
    from safetensors.torch import save_file

    save_file(tensors, str(path))


class TestItLoadsOnlyTheUnembedding:
    def test_reads_lm_head_from_a_sharded_model(self, tmp_path):
        _write_shard(tmp_path / "s1.safetensors", {
            "lm_head.weight": torch.zeros(8, 4),
            "model.layers.0.mlp.weight": torch.zeros(4, 4),
        })
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "lm_head.weight": "s1.safetensors",
                "model.layers.0.mlp.weight": "s1.safetensors",
            }
        }))

        W = load_unembedding_matrix(tmp_path)
        assert tuple(W.shape) == (8, 4)

    def test_falls_back_to_tied_input_embeddings(self, tmp_path):
        """granite-4.1-8b sets tie_word_embeddings=true; some exports then ship
        only model.embed_tokens.weight, which is the same matrix."""
        _write_shard(tmp_path / "s1.safetensors", {
            "model.embed_tokens.weight": torch.zeros(6, 3),
        })
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"model.embed_tokens.weight": "s1.safetensors"}
        }))

        W = load_unembedding_matrix(tmp_path)
        assert tuple(W.shape) == (6, 3)

    def test_reads_a_single_file_model(self, tmp_path):
        _write_shard(tmp_path / "model.safetensors", {
            "lm_head.weight": torch.zeros(5, 2),
        })
        W = load_unembedding_matrix(tmp_path)
        assert tuple(W.shape) == (5, 2)

    def test_raises_clearly_when_absent(self, tmp_path):
        _write_shard(tmp_path / "model.safetensors", {"other.weight": torch.zeros(2, 2)})
        with pytest.raises(ValueError, match="unembedding"):
            load_unembedding_matrix(tmp_path)

    def test_missing_weights_is_not_a_silent_none(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_unembedding_matrix(tmp_path)


class TestSnapshotResolution:
    def test_finds_the_hf_snapshot_directory(self, tmp_path):
        snap = tmp_path / "models--ibm-granite--granite-4.1-8b" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.safetensors.index.json").write_text("{}")

        found = resolve_snapshot_dir(tmp_path, "ibm-granite/granite-4.1-8b")
        assert found == snap

    def test_returns_none_when_nothing_is_cached(self, tmp_path):
        assert resolve_snapshot_dir(tmp_path, "ibm-granite/granite-4.1-8b") is None


class TestItDoesNotCompeteWithServing:
    def test_logit_lens_runs_on_cpu(self):
        """A 0.4 GFLOP product must not contend for VRAM with the served model."""
        src = inspect.getsource(analysis_service.AnalysisService.calculate_logit_lens)
        assert 'device = "cpu"' in src, (
            "logit lens selects cuda again; it then fails whenever miLLM holds "
            "the card, which is the reported outage"
        )
        assert 'torch.cuda.is_available()' not in src.split("Two paths")[0], (
            "device is chosen from GPU availability rather than pinned to CPU"
        )

    def test_the_fast_path_never_instantiates_the_model(self):
        src = inspect.getsource(analysis_service.AnalysisService.calculate_logit_lens)
        fast = src[: src.index("Falling back to full model load")]
        assert "load_model_from_hf" not in fast, (
            "the fast path loads the whole model to reach one tensor"
        )
        assert "load_unembedding_matrix" in fast

    def test_cleanup_tolerates_the_fast_path(self):
        """base_model exists only on the fallback; deleting it unconditionally
        would turn a successful computation into a 500."""
        src = inspect.getsource(analysis_service.AnalysisService.calculate_logit_lens)
        assert '"base_model" in locals()' in src, (
            "cleanup deletes base_model unconditionally and will NameError on "
            "the fast path"
        )
