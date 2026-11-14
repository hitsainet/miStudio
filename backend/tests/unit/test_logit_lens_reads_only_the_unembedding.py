"""The logit lens reads one matrix from the checkpoint, so it never needs a model split across cards.

`compute_logit_lens_for_sae` loaded the WHOLE model onto one card to reach
`lm_head`. For a model no single card holds (Multi-GPU Phase 2's acceptance model
is ~29.5 GB at bf16) there could be no logit lens at all, and a model that did fit
stayed on its card in `_loaded_models` for the rest of the worker's life.

For a downloaded model the unembedding is now read straight from the safetensors
(the reader the per-feature logit lens in analysis_service already uses), so the
lens needs vocab x d_model on one card. Only a model known by name alone is still
loaded whole. And placement now frees a model the lens left cached.

MUTATION CONTROLS (2026-09-14; each applied alone, this file +
test_gpu_logit_lens_placement.py run, source restored and checked by sha256).
All went red:
  LL1 weights_dir = None (always load the whole model)  -> a_downloaded_model_is_never_loaded_whole, lands_on_the_callers_device
  LL2 the unembedding is not moved to `device`          -> the_unembedding_lands_on_the_callers_device
  LL3 the idle release is not registered                -> the_release_is_registered..., a_cached_model_is_freed...
  LL4 clear_cache empties no card                       -> a_cached_model_is_freed_on_its_own_card_only
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from safetensors.torch import save_file

from src.models.external_sae import ExternalSAE, SAEStatus
from src.models.model import Model
from src.services import gpu_placement
from src.services import logit_lens_service as module
from src.services.logit_lens_service import LogitLensService


@pytest.fixture
def downloaded(tmp_path, monkeypatch):
    """An SAE whose model is on disk with only its unembedding in the checkpoint."""
    weights = tmp_path / "model"
    weights.mkdir()
    lm_head = torch.randn(10, 3)  # [vocab, d_model]
    save_file({"lm_head.weight": lm_head}, str(weights / "model.safetensors"))
    sae_dir = tmp_path / "sae"
    sae_dir.mkdir()

    sae = SimpleNamespace(
        status=SAEStatus.READY.value, local_path=str(sae_dir), model_id="m_1", model_name=None,
        n_features=4, training_id=None,
    )
    model_row = SimpleNamespace(
        id="m_1", file_path=str(weights), quantized_path=None, repo_id="org/base", name="base",
    )
    db = MagicMock()
    db.get = AsyncMock(side_effect=lambda cls, key: {ExternalSAE: sae, Model: model_row}[cls])

    monkeypatch.setattr(
        module, "load_sae_auto_detect", lambda path, device: ({"W_dec": torch.randn(3, 4)}, None, None)
    )
    tokenizers = []
    monkeypatch.setattr(
        module.AutoTokenizer, "from_pretrained",
        lambda name, **kwargs: tokenizers.append((name, kwargs)) or MagicMock(),
    )

    service = LogitLensService()
    service._load_model = AsyncMock(side_effect=AssertionError("the whole model was loaded"))
    seen = {}

    async def batch(W_dec, W_U, tokenizer, indices, k):
        seen["W_U"] = W_U
        seen["W_dec_device"] = W_dec.device
        return {}

    service._compute_batch_logit_lens = batch
    return SimpleNamespace(
        db=db, service=service, seen=seen, lm_head=lm_head, weights=weights, tokenizers=tokenizers,
    )


def _compute(h, device):
    return asyncio.run(
        h.service.compute_logit_lens_for_sae(h.db, "sae_1", [0, 1], force_recompute=True, device=device)
    )


def test_a_downloaded_model_is_never_loaded_whole(downloaded):
    _compute(downloaded, torch.device("cpu"))

    assert downloaded.service._load_model.await_count == 0
    torch.testing.assert_close(downloaded.seen["W_U"], downloaded.lm_head.T.float())
    assert downloaded.tokenizers == [
        (str(downloaded.weights), {"trust_remote_code": True, "local_files_only": True})
    ]


def test_the_unembedding_lands_on_the_callers_device(downloaded):
    """On `meta`, so a matrix left on the CPU shows up as the wrong device."""
    meta = torch.device("meta")

    _compute(downloaded, meta)

    assert downloaded.seen["W_U"].device == meta
    assert tuple(downloaded.seen["W_U"].shape) == (3, 10)
    assert downloaded.seen["W_dec_device"] == meta


class TestPlacementFreesAModelTheLensLeftCached:
    def test_the_release_is_registered_where_placement_runs_it(self):
        assert module._release_idle_logit_lens_models in gpu_placement._IDLE_RELEASERS

    def test_a_cached_model_is_freed_on_its_own_card_only(self, monkeypatch):
        service = LogitLensService()
        service._loaded_models[("org/base", "cuda:1")] = (torch.nn.Linear(2, 2), object())
        emptied = []
        monkeypatch.setattr(module, "_logit_lens_service", service)
        monkeypatch.setattr(module, "cuda_devices", lambda model: [torch.device("cuda", 1)])
        monkeypatch.setattr(module, "empty_cache_on", lambda devices: emptied.append(list(devices)))

        gpu_placement.release_idle_gpu_memory()

        assert service._loaded_models == {}
        assert emptied == [[torch.device("cuda", 1)]], "a card the cached model was not on was touched"

    def test_an_empty_cache_is_left_alone(self, monkeypatch):
        service = LogitLensService()
        monkeypatch.setattr(module, "_logit_lens_service", service)
        monkeypatch.setattr(service, "clear_cache", lambda: pytest.fail("cleared an empty cache"))

        module._release_idle_logit_lens_models()
