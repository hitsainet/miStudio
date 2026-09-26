"""OSD-24 — an unsizeable fit is placed SHORT, so read the checkpoint first.

The recorded claim was that the GPU preflight "returns None and silently skips".
It does not skip silently — it warns by model name. The real defect is what
happens next: the caller falls back to the FIXED 2 GiB activation headroom, while
a fit needs that cap PLUS roughly 0.41 GB of fp32 accumulators per captured layer
(measured: 3.63 GB of working set for four layers at d_model 3840). So an unsized
fit is placed short by the accumulator term and dies mid-backward on a card that
looked like it had room.

The dimensions were never actually unavailable: `architecture_config` on the row
can be incomplete, but `config.json` is on disk. Reading it is reporting what the
model is, not inventing a dimension.
"""
import json
from unittest.mock import patch

import pytest

from src.services import jlens_model_registry as registry


class _Row:
    def __init__(self, arch=None, quantization="Q8"):
        self.id, self.repo_id = "m_1", "org/model"
        self.architecture_config = arch
        self.quantization = quantization


COMPLETE = {
    "hidden_size": 3840, "num_hidden_layers": 48,
    "vocab_size": 262144, "num_attention_heads": 16,
}


@pytest.fixture
def snapshot(tmp_path):
    """A checkpoint directory the registry's own resolver will find."""
    def _write(config):
        (tmp_path / "config.json").write_text(json.dumps(config))
        return tmp_path
    return _write


def _size(row, snapshot_dir=None, prompts=("hello world",), layers=(42, 43, 44, 45)):
    with patch.object(registry, "locate_weights", return_value=("org/model", str(snapshot_dir or "/nope"))), \
         patch("src.services.analysis_service.resolve_snapshot_dir", return_value=snapshot_dir), \
         patch.object(registry, "tokenizer_for", side_effect=registry.ModelNotAvailable("no tokenizer")):
        return registry.estimate_fit_working_mb(row, prompts, layers=layers)


class TestTheRowIsUsedWhenItCan:

    def test_a_complete_row_sizes_without_reading_the_disk(self, snapshot):
        with patch.object(registry, "_dims_from_checkpoint") as never:
            with patch.object(registry, "tokenizer_for",
                              side_effect=registry.ModelNotAvailable("no tokenizer")):
                got = registry.estimate_fit_working_mb(
                    _Row(arch=COMPLETE), ["hello"], layers=(42, 43)
                )
        assert got and got > 0
        never.assert_not_called()


class TestTheCheckpointFillsTheGap:

    def test_an_empty_row_is_sized_from_config_json(self, snapshot):
        directory = snapshot(COMPLETE)
        got = _size(_Row(arch={}), snapshot_dir=directory)
        assert got is not None, (
            "the dimensions were on disk; returning None places the fit short by "
            "the accumulator term"
        )
        assert got > 0

    def test_it_agrees_with_a_complete_row(self, snapshot):
        directory = snapshot(COMPLETE)
        from_disk = _size(_Row(arch={}), snapshot_dir=directory)
        from_row = _size(_Row(arch=COMPLETE), snapshot_dir=directory)
        assert from_disk == from_row

    def test_a_partially_filled_row_is_completed_not_discarded(self, snapshot):
        directory = snapshot(COMPLETE)
        partial = {"hidden_size": 3840, "num_hidden_layers": 48}   # vocab/heads missing
        assert _size(_Row(arch=partial), snapshot_dir=directory) is not None

    def test_a_multimodal_checkpoint_is_read_under_text_config(self, snapshot):
        """gemma-4-12B keeps the text stack's dimensions nested.

        Reading only the top level finds nothing — the same nesting that once made
        the J-lens final norm resolve to None and fall back to plain RMS silently.
        """
        directory = snapshot({"model_type": "gemma4", "text_config": COMPLETE})
        assert _size(_Row(arch={}), snapshot_dir=directory) is not None


class TestItStillRefusesToInventADimension:

    def test_no_row_and_no_config_returns_none(self, snapshot):
        directory = snapshot({"model_type": "mystery"})
        assert _size(_Row(arch={}), snapshot_dir=directory) is None

    def test_an_unreadable_snapshot_returns_none(self):
        assert _size(_Row(arch={}), snapshot_dir=None) is None

    def test_the_warning_names_the_missing_fields_and_the_shortfall(self, snapshot, caplog):
        directory = snapshot({"hidden_size": 3840})
        with caplog.at_level("WARNING"):
            assert _size(_Row(arch={}), snapshot_dir=directory) is None
        message = caplog.text
        assert "num_hidden_layers" in message and "vocab_size" in message
        assert "0.41 GB per captured layer" in message, (
            "the warning must say the headroom is SHORT, not merely that sizing failed"
        )


class TestTheDimensionReader:

    def test_it_takes_only_positive_integers(self, snapshot):
        directory = snapshot({"hidden_size": "3840", "num_hidden_layers": 0,
                              "vocab_size": True, "num_attention_heads": 16})
        with patch.object(registry, "locate_weights", return_value=("org/model", str(directory))), \
             patch("src.services.analysis_service.resolve_snapshot_dir", return_value=directory):
            found = registry._dims_from_checkpoint(_Row())
        assert found == {"num_attention_heads": 16}, (
            "a string, a zero and a bool are not dimensions"
        )
