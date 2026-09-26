"""OSD-22 — an OOM *mid-backward* must not leave the fit's model on the card.

The tracker recorded this as NOT FIXED: after an OOM the worker "sat idle holding
23.6 GB" and only a restart reclaimed it. Re-verified 2026-09-25 rather than
fixed, because the release has since been rebuilt — `release_card` clears the
propagating exception's traceback frames and then calls `clear_cache`, which is
exactly the OOM case, since the traceback holds every frame between the task and
the raise and one of those frames holds the model.

Nothing covered the FIT path, though: the existing weakref test drives
`compute_readout`. So this proves it where the tracker says it failed, with an
OOM raised from inside the fit rather than from the load.

Why a weakref and not "was clear_cache called": the recorded lesson from the
first attempt at this release is that three tests passed by asserting
`clear_cache` was CALLED while 2,608 MiB of weights stayed resident. Being called
is not being effective.
"""
import gc
import types
import weakref
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.services.gpu_placement import Placement
from src.workers.jlens_fit_tasks import fit_jlens_artifact

RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"


class _Model:
    pass


class _Loaded:
    """A real object — MagicMock children keep their parents alive."""

    def __init__(self):
        self.model = _Model()
        self.tokenizer = self.structure = self.unembedding = None
        self.name = "org/model"


@contextmanager
def _fit_worker(*, fit_raises):
    """Run `fit_jlens_artifact` with NVML, the database and the load faked."""
    holder, seen, events = {}, {}, []

    record = MagicMock()
    record.id, record.repo_id = "m_1", "org/model"
    record.params_count, record.quantization = 7_000_000_000, "Q8"
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = record

    @contextmanager
    def fake_db():
        yield db

    def fake_load(model_record, capture_device="cpu", placement=None):
        loaded = _Loaded()
        holder["ref"] = weakref.ref(loaded.model)
        return loaded

    def fake_clear_cache():
        """Stand in for the registry's cache drop, and observe what survives."""
        gc.collect()
        ref = holder.get("ref")
        seen["alive"] = ref is not None and ref() is not None
        events.append("release")

    def fake_fit_and_publish(task, *, loaded=None, **_kw):
        # A frame that HOLDS the model, like the real fit's helpers do, so the
        # traceback keeps it alive unless the release clears those frames.
        held = loaded.model            # noqa: F841 - held on purpose
        raise fit_raises

    fit_jlens_artifact.push_request(id="t-fit-oom")
    try:
        with patch("src.core.database.get_sync_db", fake_db), patch(
            "src.services.gpu_placement.place_job",
            lambda requested, required_mb=None, allow_shard=False: Placement(
                card=MagicMock(uuid=RTX_UUID, index=0), device="cuda:0"
            ),
        ), patch("src.workers.jlens_progress.record_gpu", lambda *a, **k: True), patch(
            "src.services.jlens_model_registry.load_for_readout", fake_load
        ), patch(
            "src.services.jlens_model_registry.clear_cache", fake_clear_cache
        ), patch("src.workers.jlens_progress.update_row", MagicMock()), patch(
            "src.workers.jlens_fit_tasks._fit_and_publish", fake_fit_and_publish
        ), patch.object(fit_jlens_artifact, "update_state", MagicMock()):
            yield types.SimpleNamespace(seen=seen, events=events)
    finally:
        fit_jlens_artifact.pop_request()


def _oom():
    """The real class when torch has it, a stand-in when it does not."""
    import torch

    return getattr(torch.cuda, "OutOfMemoryError", RuntimeError)(
        "CUDA out of memory. Tried to allocate 2.00 GiB"
    )


class TestAnOomMidFitReleasesEverything:

    def test_the_card_is_released_at_all(self):
        with _fit_worker(fit_raises=_oom()) as run:
            with pytest.raises(Exception, match="out of memory"):
                fit_jlens_artifact.run(model_id="m_1", prompts=["hello"], gpu_request=RTX_UUID)
        assert run.events == ["release"], (
            "the fit's finally must release the card on the OOM path — this is the "
            "path the tracker recorded as leaving 23.6 GB resident"
        )

    def test_nothing_still_references_the_model_when_the_cache_is_dropped(self):
        with _fit_worker(fit_raises=_oom()) as run:
            with pytest.raises(Exception, match="out of memory"):
                fit_jlens_artifact.run(model_id="m_1", prompts=["hello"], gpu_request=RTX_UUID)
        assert "alive" in run.seen, "the release never ran"
        assert run.seen["alive"] is False, (
            "the model was still referenced when the cache was dropped, so "
            "empty_cache had no free blocks to return and the weights stayed on "
            "the card — the traceback's frames were not cleared"
        )

    def test_it_holds_for_an_ordinary_failure_too(self):
        """Not OOM-specific: any raise through the fit must release."""
        with _fit_worker(fit_raises=RuntimeError("validation refused the fit")) as run:
            with pytest.raises(RuntimeError, match="validation refused"):
                fit_jlens_artifact.run(model_id="m_1", prompts=["hello"], gpu_request=RTX_UUID)
        assert run.seen.get("alive") is False
