"""The integrity hash must describe the file the probe actually reads.

⚠ `_sae_weights_sha256` hashed `sorted(path.glob("**/*.safetensors"))[0]` — the
lexicographically first file in the whole SAE tree. One repository publishes a dictionary per
layer, and **`layer_10` sorts before `layer_9`**, so a probe trained on layer 9 could be pinned to
layer 10's bytes.

The consumer then fetches the file the probe NAMES (`sae.path`), hashes it, and is told it is
corrupt. That is precisely the failure the function's own docstring is written against — *"a wrong
integrity hash is worse than no export, because it makes a mismatch look like corruption at the
consumer's end"* — arriving through sort order rather than through a missing file.

Found 2026-09-27 while investigating an unrelated k-sparse divergence. It did not fire on the
shipped probe (the repo holds one layer, and the sha matched at arm time), which is why nothing
caught it: the defect is invisible until a multi-layer dictionary is exported.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from src.services.probe_definition_builder import ProbeExportRefused, _sae_weights_sha256


def _row(local_path: str, hf_filepath: str | None):
    return SimpleNamespace(id="sae_1", local_path=local_path, hf_filepath=hf_filepath)


def _write(path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def multilayer(tmp_path, monkeypatch):
    """A dictionary published per layer, with the ordering trap present.

    ⚠ `layer_9` is the one we want and `layer_10` sorts FIRST. A fixture with only
    layer_1/layer_2 would be sorted in the same order as intended and would pass against the
    defect — agreeing by construction, which is how this estate's bugs usually survive review.
    """
    from src.core.config import settings

    # ⚠ `data_dir` is patched, NOT `resolve_data_path`. Stubbing the resolver would test my
    # stub instead of the resolution production actually performs.
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    root = tmp_path / "sae_multi"
    digests = {
        "layer_9": _write(root / "layer_9" / "sae_weights.safetensors", b"NINE" * 64),
        "layer_10": _write(root / "layer_10" / "sae_weights.safetensors", b"TEN!" * 64),
        "layer_11": _write(root / "layer_11" / "sae_weights.safetensors", b"ELVN" * 64),
    }
    return root, digests


class TestTheHashDescribesTheLayerTheProbeReads:
    def test_layer_9_is_hashed_even_though_layer_10_sorts_first(self, multilayer):
        _root, digests = multilayer
        assert sorted(["layer_9", "layer_10"])[0] == "layer_10", (
            "the fixture no longer contains the ordering trap it exists to reproduce"
        )
        assert _sae_weights_sha256(_row("sae_multi", "layer_9")) == digests["layer_9"]

    @pytest.mark.parametrize("layer", ["layer_9", "layer_10", "layer_11"])
    def test_every_layer_resolves_to_its_own_bytes(self, multilayer, layer):
        _root, digests = multilayer
        assert _sae_weights_sha256(_row("sae_multi", layer)) == digests[layer]

    def test_an_unscoped_row_with_several_layers_is_REFUSED(self, multilayer):
        """No `hf_filepath` means nothing says which layer — so there is no basis to choose.

        Refusing is the same judgement the function already makes when there is no local copy:
        a guessed integrity hash is worse than no export.
        """
        with pytest.raises(ProbeExportRefused) as exc:
            _sae_weights_sha256(_row("sae_multi", None))
        assert "which one this probe read" in str(exc.value)

    def test_a_single_layer_directory_still_works_without_scoping(self, tmp_path, monkeypatch):
        """The common case must not need `hf_filepath` — most dictionaries hold one file."""
        from src.core.config import settings

        monkeypatch.setattr(settings, "data_dir", tmp_path)
        root = tmp_path / "sae_one"
        digest = _write(root / "sae_weights.safetensors", b"ONLY" * 64)
        assert _sae_weights_sha256(_row("sae_one", None)) == digest

    def test_a_file_path_is_hashed_directly(self, tmp_path, monkeypatch):
        from src.core.config import settings

        monkeypatch.setattr(settings, "data_dir", tmp_path)
        digest = _write(tmp_path / "w.safetensors", b"FILE" * 64)
        assert _sae_weights_sha256(_row("w.safetensors", None)) == digest

    def test_no_weights_at_all_is_refused(self, tmp_path, monkeypatch):
        from src.core.config import settings

        monkeypatch.setattr(settings, "data_dir", tmp_path)
        (tmp_path / "empty").mkdir()
        with pytest.raises(ProbeExportRefused):
            _sae_weights_sha256(_row("empty", None))
