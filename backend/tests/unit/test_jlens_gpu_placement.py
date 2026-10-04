"""J-lens jobs run on the GPU the caller chose, and their rows say which.

Every J-space GPU job took `"cuda"` — the CURRENT device, index 0 — or, for the
readout family, "any resident copy, else the CPU". On 2026-09-13 the node gained
a second card and the new 12 GB RTX 3080 Ti took index 0, so every fit,
acquisition and intervention moved onto it without anyone deciding so.
Plan: 0xcc/plans/Multi-GPU-Plan.md, Phase 1.

Asserted per task family, by RUNNING the endpoints and tasks with only NVML,
the database and the model load faked:

  (a) the endpoint RESOLVES the request — an index becomes the UUID it names
      now — and hands it to the task, asserted by payload and call count;
  (b) an unknown card is a 400, and nothing is queued or recorded;
  (c) the worker places the job, writes the card to its row BEFORE the load,
      loads on the placement's device, and releases the card even when the
      load dies part-way;
  (d) a GpuPlacementError fails the task with its own message and nothing loads;
  (e) the resolved request rides in `retry_params`, so a re-run reuses the card.

The acquisition's worker cases live in test_jlens_acquire_worker.py, beside the
harness that can run it.

"all" (multi-GPU Phase 2, review round 1, 2026-09-14): a fit runs on ONE card, so
its route refuses "all" at submit instead of queueing a job the worker can only
refuse; every other J-lens job splits and takes "all" to its task. Controls
(each alone, run red, restored byte-identically and checked by sha256):
  K1 `resolve_gpu_request` loses its refusal branch -> TestAFitRefusesAllWhenItIsSubmitted::test_all_is_a_400_...
     (and the training and logit-lens refusals in their own modules)
  K4 the fit route stops passing can_split=False      -> the same case
"""

from __future__ import annotations

import asyncio
import importlib
import types
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi import HTTPException
from pydantic import ValidationError

from src.services import gpu_placement
from src.services.gpu_placement import GpuCard, GpuPlacementError, Placement

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
UNKNOWN_UUID = "GPU-00000000-0000-0000-0000-000000000000"

#: The node as NVML reports it. The 3080 Ti is index 0 and has LESS free memory,
#: so index "0" arriving as its UUID cannot be confused with Auto's choice.
CARDS = [
    GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000),
    GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000),
]

#: Index 1, so a task that falls back to the current device (cuda:0) is visible.
RTX_DEVICE = torch.device("cuda", 1)

#: What a task placed on the 3090 hands the model load: the PLACEMENT, not its
#: device, so a split placement reaches the loader with its cards and budgets.
RTX_PLACEMENT = Placement(card=CARDS[1], device=RTX_DEVICE)


@pytest.fixture
def two_cards(monkeypatch):
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: list(CARDS))


# ── Endpoints ──────────────────────────────────────────────────────────────


def _db_with_model():
    row = MagicMock()
    row.repo_id = "org/model"
    result = MagicMock()
    result.scalar_one_or_none.return_value = row
    db = MagicMock()

    async def execute(_query):
        return result

    db.execute = execute
    return db


def _gpu(gpu):
    """No `gpu` key at all when None, so the DEFAULT is what is exercised."""
    return {} if gpu is None else {"gpu": gpu}


def _fit(gpu):
    from src.api.v1.endpoints.jlens import FitRequest, fit

    return fit(FitRequest(model_id="m_1", prompts=["a", "b"], **_gpu(gpu)), db=_db_with_model())


def _revalidate(gpu):
    from src.api.v1.endpoints.jlens import RevalidateRequest, revalidate_staged

    return revalidate_staged(RevalidateRequest(model_id="m_1", **_gpu(gpu)), db=_db_with_model())


def _band(gpu):
    from src.api.v1.endpoints.jlens import BandReportRequest, compute_band_report

    return compute_band_report(
        BandReportRequest(model_id="m_1", prompts=["a"], control_seed=7, **_gpu(gpu)),
        db=_db_with_model(),
    )


def _readout(gpu):
    from src.api.v1.endpoints.jlens import readout
    from src.schemas.jlens import ReadoutRequest

    return readout(
        ReadoutRequest(model_id="m_1", prompt="hello", **_gpu(gpu)), db=_db_with_model()
    )


def _probe(gpu):
    from src.api.v1.endpoints.jlens import probe
    from src.schemas.jlens import ProbeRequest

    return probe(
        ProbeRequest(model_id="m_1", prompt="hello", tokens=[" dog"], **_gpu(gpu)),
        db=_db_with_model(),
    )


def _acquire(gpu):
    from src.api.v1.endpoints.jlens import AcquireRequest, acquire_artifact

    return acquire_artifact(
        AcquireRequest(
            model_id="m_1", repo_id="org/lenses", path_in_repo="a/lens.pt", **_gpu(gpu)
        ),
        db=_db_with_model(),
    )


def _intervention(gpu):
    from src.api.v1.endpoints.jlens import InterventionRequest, run_intervention

    return run_intervention(
        InterventionRequest(
            model_id="m_1",
            prompt="hello",
            primitive="additive",
            layers=[1],
            direction_token=" dog",
            **_gpu(gpu),
        ),
        db=_db_with_model(),
    )


def _nothing():
    return []


def _a_staged_artifact():
    """The revalidate route refuses without one, before it would queue."""
    service = MagicMock()
    service._ref_for.return_value = types.SimpleNamespace(slug="model")
    return [
        patch("src.services.jlens_artifact_service.JLensArtifactService", return_value=service)
    ]


def _room_to_acquire():
    """The acquire route's three synchronous refusals, all passing."""
    service = MagicMock()
    service._ref_for.return_value = None
    return [
        patch("src.services.jlens_model_registry.locate_weights", return_value=("org/model", None)),
        patch("src.services.jlens_acquire_service.check_free_space"),
        patch("src.services.jlens_artifact_service.JLensArtifactService", return_value=service),
    ]


#: (route call, task, how it is queued, row type, patches the route needs)
ENDPOINTS = [
    pytest.param(_fit, "src.workers.jlens_fit_tasks.fit_jlens_artifact", "delay", "jlens_fit", _nothing, id="fit"),
    pytest.param(_revalidate, "src.workers.jlens_fit_tasks.revalidate_staged_artifact", "delay", "jlens_revalidate", _a_staged_artifact, id="revalidate"),
    pytest.param(_band, "src.workers.jlens_band_tasks.compute_band_report_task", "delay", "jlens_band_report", _nothing, id="band"),
    pytest.param(_readout, "src.workers.jlens_readout_tasks.compute_readout", "delay", "jlens_readout", _nothing, id="readout"),
    pytest.param(_probe, "src.workers.jlens_probe_tasks.compute_probe", "delay", "jlens_probe", _nothing, id="probe"),
    pytest.param(_acquire, "src.workers.jlens_acquire_tasks.acquire_jlens_artifact", "apply_async", "jlens_acquire", _room_to_acquire, id="acquire"),
    pytest.param(_intervention, "src.workers.jlens_intervention_tasks.run_intervention_task", "delay", "jlens_intervention", _nothing, id="intervention"),
]


class TestAFitRefusesAllWhenItIsSubmitted:
    """A fit never splits (its batched backward needs workspace a split does not
    leave), so "all" was accepted with a 202, queued behind whatever else holds
    the GPU queue, and refused only when the worker placed it. A request doomed
    at submit is refused at submit — the acquire route's own doctrine."""

    def test_all_is_a_400_and_nothing_is_queued_or_recorded(self, two_cards):
        with _queued("src.workers.jlens_fit_tasks.fit_jlens_artifact", _nothing) as (task, open_row):
            with pytest.raises(HTTPException) as refused:
                asyncio.run(_fit("all"))

        assert refused.value.status_code == 400
        assert "cannot run split across GPUs" in refused.value.detail
        assert task.delay.call_count == 0
        assert open_row.call_count == 0

    @pytest.mark.parametrize(
        "call, task_path, via, row_type, extra",
        [p for p in ENDPOINTS if p.id != "fit"],
    )
    def test_every_other_j_lens_job_takes_all_to_its_task(self, two_cards, call, task_path, via, row_type, extra):
        """They run split, so the refusal must not reach them."""
        with _queued(task_path, extra) as (task, _open_row):
            asyncio.run(call("all"))

        count, kwargs = _sent(task, via)
        assert count == 1
        assert kwargs["gpu_request"] == "all"


@contextmanager
def _queued(task_path, extra_patches):
    with ExitStack() as stack:
        task = stack.enter_context(patch(task_path))
        task.delay.return_value = MagicMock(id="t-1")
        task.apply_async.return_value = MagicMock(id="t-1")
        open_row = stack.enter_context(patch("src.workers.jlens_progress.open_row"))
        for extra in extra_patches():
            stack.enter_context(extra)
        yield task, open_row


def _sent(task, via):
    """(call count, the kwargs the task will run with).

    The acquisition queues through `apply_async(kwargs=...)` so its token can be
    redacted from the message headers; everything else uses `.delay(**kwargs)`.
    """
    method = getattr(task, via)
    if method.call_args is None:
        return method.call_count, None
    kwargs = method.call_args.kwargs
    return method.call_count, (kwargs["kwargs"] if via == "apply_async" else kwargs)


@pytest.mark.parametrize("call, task_path, via, row_type, extra", ENDPOINTS)
class TestTheEndpointResolvesTheRequest:
    def test_an_index_reaches_the_task_as_the_UUID_it_names_now(
        self, two_cards, call, task_path, via, row_type, extra
    ):
        """(a) and (e). A queued job may start after a card is added and the
        indices shift, so the index must become a UUID at submit."""
        with _queued(task_path, extra) as (task, open_row):
            asyncio.run(call("0"))

        count, kwargs = _sent(task, via)
        assert count == 1, "the route queued nothing, or queued twice"
        assert kwargs["gpu_request"] == TI_UUID
        assert open_row.call_count == 1
        assert open_row.call_args.args == (row_type, "m_1", "t-1")
        assert set(open_row.call_args.kwargs) == {"retry_params"}
        # The readout ALSO stores whether it unloads its model — asserted on its
        # own in TestTheReadoutChoosesWhetherToUnload. Every family stores `gpu`.
        retry = {k: v for k, v in open_row.call_args.kwargs["retry_params"].items() if k != "unload_after"}
        assert retry == {"gpu": TI_UUID}

    def test_auto_is_the_default_and_reads_no_inventory(
        self, monkeypatch, call, task_path, via, row_type, extra
    ):
        monkeypatch.setattr(
            gpu_placement, "list_cards", lambda: pytest.fail("NVML was read to store auto")
        )
        with _queued(task_path, extra) as (task, open_row):
            asyncio.run(call(None))

        count, kwargs = _sent(task, via)
        assert count == 1
        assert kwargs["gpu_request"] == "auto"
        retry = {k: v for k, v in open_row.call_args.kwargs["retry_params"].items() if k != "unload_after"}
        assert retry == {"gpu": "auto"}

    def test_an_unknown_card_is_a_400_and_nothing_is_queued(
        self, two_cards, call, task_path, via, row_type, extra
    ):
        """(b) The 400 names the cards that exist, so the caller can pick one."""
        with _queued(task_path, extra) as (task, open_row):
            with pytest.raises(HTTPException) as refused:
                asyncio.run(call(UNKNOWN_UUID))

        assert refused.value.status_code == 400
        assert "RTX 3090" in refused.value.detail
        assert "RTX 3080 Ti" in refused.value.detail
        assert _sent(task, via)[0] == 0, "a job was queued for a card that does not exist"
        assert open_row.call_count == 0


REQUESTS = [
    pytest.param("src.api.v1.endpoints.jlens", "FitRequest", {"model_id": "m", "prompts": ["a"]}, id="fit"),
    pytest.param("src.api.v1.endpoints.jlens", "RevalidateRequest", {"model_id": "m"}, id="revalidate"),
    pytest.param("src.api.v1.endpoints.jlens", "BandReportRequest", {"model_id": "m", "prompts": ["a"], "control_seed": 1}, id="band"),
    pytest.param("src.schemas.jlens", "ReadoutRequest", {"model_id": "m", "prompt": "x"}, id="readout"),
    pytest.param("src.schemas.jlens", "ProbeRequest", {"model_id": "m", "prompt": "x", "tokens": ["a"]}, id="probe"),
    pytest.param("src.api.v1.endpoints.jlens", "AcquireRequest", {"model_id": "m", "repo_id": "o/r", "path_in_repo": "a.pt"}, id="acquire"),
    pytest.param(
        "src.api.v1.endpoints.jlens",
        "InterventionRequest",
        {"model_id": "m", "prompt": "x", "primitive": "additive", "layers": [1]},
        id="intervention",
    ),
]


@pytest.mark.parametrize("module, name, fields", REQUESTS)
def test_every_request_defaults_to_auto_and_refuses_a_device_string(module, name, fields):
    """`cuda:0` is exactly the hard-coding being removed; it is not a card name."""
    request = getattr(importlib.import_module(module), name)
    assert request(**fields).gpu == "auto"
    with pytest.raises(ValidationError):
        request(**fields, gpu="cuda:0")


# ── Workers ────────────────────────────────────────────────────────────────


WORKERS = [
    pytest.param("src.workers.jlens_fit_tasks", "fit_jlens_artifact", {"model_id": "m_1", "prompts": ["a"]}, id="fit"),
    pytest.param("src.workers.jlens_fit_tasks", "revalidate_staged_artifact", {"model_id": "m_1"}, id="revalidate"),
    pytest.param("src.workers.jlens_band_tasks", "compute_band_report_task", {"model_id": "m_1", "prompts": ["a"], "control_seed": 7}, id="band"),
    pytest.param("src.workers.jlens_readout_tasks", "compute_readout", {"model_id": "m_1", "prompt": "hello"}, id="readout"),
    pytest.param("src.workers.jlens_probe_tasks", "compute_probe", {"model_id": "m_1", "prompt": "hello", "tokens": [" dog"]}, id="probe"),
    pytest.param(
        "src.workers.jlens_intervention_tasks",
        "run_intervention_task",
        {"model_id": "m_1", "prompt": "hello", "primitive": "additive", "layers": [1], "direction_token": " dog"},
        id="intervention",
    ),
]


def _task(module, name):
    return getattr(importlib.import_module(module), name)


@contextmanager
def _worker(task, *, place=None, load=None, release=None):
    """Run a J-space task with NVML, the database and the model load faked.

    `events` is the order in which the task meets the GPU: place, record, load,
    release. By default the load dies with a CUDA OOM, which is both the path
    the release matters most on and a way to stop before the real work.
    """
    events = []
    placed = []
    record = MagicMock()
    record.id, record.repo_id = "m_1", "org/model"
    # A size the estimate can read without touching disk (a quantized row), so
    # `required_mb` is a real number every task must pass on.
    record.params_count, record.quantization = WORKER_PARAMS, "Q8"
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = record

    @contextmanager
    def fake_db():
        yield db

    def fake_place_job(requested, required_mb=None, allow_shard=False):
        events.append(("place", requested))
        placed.append({"required_mb": required_mb, "allow_shard": allow_shard})
        if place is not None:
            return place(requested)
        return Placement(card=CARDS[1], device=RTX_DEVICE)

    def fake_record_gpu(task_id, gpu_uuid, gpu_uuids=None, attempts=10):
        events.append(
            ("record", task_id, gpu_uuid)
            if gpu_uuids is None
            else ("record", task_id, gpu_uuid, gpu_uuids)
        )
        return True

    def fake_load(model_record, capture_device="cpu", placement=None):
        events.append(("load", capture_device if placement is None else placement))
        if load is not None:
            return load()
        raise RuntimeError("CUDA out of memory while loading the weights")

    rows = MagicMock()
    task.push_request(id="t-gpu")
    try:
        with patch("src.core.database.get_sync_db", fake_db), patch(
            "src.services.gpu_placement.place_job", fake_place_job
        ), patch("src.workers.jlens_progress.record_gpu", fake_record_gpu), patch(
            "src.services.jlens_model_registry.load_for_readout", fake_load
        ), patch(
            "src.services.jlens_model_registry.clear_cache",
            release or (lambda: events.append(("release",))),
        ), patch("src.workers.jlens_progress.update_row", rows), patch.object(
            task, "update_state", MagicMock()
        ):
            yield types.SimpleNamespace(events=events, rows=rows, placed=placed)
    finally:
        task.pop_request()


#: The `_worker` record's size: 7B parameters at Q8, 1.1 bytes each.
WORKER_PARAMS = 7_000_000_000
#: What placement is asked for: those weights plus the 2 GiB of activation headroom
#: `place_on_card` adds (review round 2), so a task sized at its weights alone disagrees.
WORKER_REQUIRED_MB = WORKER_PARAMS * 1.1 / (1024 * 1024) + 2 * 1024

#: Which J-space tasks may SPLIT a model no single card holds. The fit may not:
#: its batched backward needs workspace a split's per-card reserve cannot hold
#: (see `fit_jlens_artifact`). Everything else only runs forward passes through
#: split-safe code.
MAY_SPLIT = {
    "fit_jlens_artifact": False,
    "revalidate_staged_artifact": True,
    "compute_band_report_task": True,
    "compute_readout": True,
    "compute_probe": True,
    "run_intervention_task": True,
}


@pytest.mark.parametrize("module, name, kwargs", WORKERS)
class TestTheWorkerRunsOnThePlacedCard:
    def test_the_card_is_recorded_BEFORE_the_load_and_the_load_uses_its_device(
        self, module, name, kwargs
    ):
        """(c). The release is asserted on a load that DIES: the load used to sit
        outside every task's guarded block, so its allocations had no release."""
        task = _task(module, name)
        with _worker(task) as run:
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                task.run(**kwargs, gpu_request=RTX_UUID)

        assert run.events == [
            ("place", RTX_UUID),
            ("record", "t-gpu", RTX_UUID),
            ("load", RTX_PLACEMENT),
            ("release",),
        ]

    def test_a_card_that_cannot_take_the_job_FAILS_the_task_with_its_reason(
        self, module, name, kwargs
    ):
        """(d). No fallback to another card or to the CPU, and no retry."""
        reason = (
            "GPU 1 (NVIDIA GeForce RTX 3090, 900 of 24,576 MB free) cannot take "
            "this job: it needs ~15,000 MB. Choose another GPU or Auto."
        )

        def refuse(requested):
            raise GpuPlacementError(reason, requested=requested)

        task = _task(module, name)
        with _worker(task, place=refuse) as run:
            with pytest.raises(GpuPlacementError, match="Choose another GPU or Auto"):
                task.run(**kwargs, gpu_request=RTX_UUID)

        assert run.events == [("place", RTX_UUID)], "something ran after the refusal"
        failed = [c for c in run.rows.call_args_list if c.kwargs.get("status") == "failed"]
        assert len(failed) == 1, run.rows.call_args_list
        assert failed[0].args == ("t-gpu",)
        assert reason in failed[0].kwargs["error_message"]

    def test_a_cpu_placement_releases_nothing(self, module, name, kwargs):
        """No card is held, and the single-entry cache keeps a CPU copy on purpose."""
        task = _task(module, name)
        cpu = Placement(card=None, device=torch.device("cpu"))
        with _worker(task, place=lambda _requested: cpu) as run:
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                task.run(**kwargs)

        assert run.events == [
            ("place", "auto"),
            ("record", "t-gpu", None),
            ("load", cpu),
        ]

    def test_the_task_says_whether_it_may_split_and_how_big_its_model_is(
        self, module, name, kwargs
    ):
        """Phase 2. Without `required_mb` Auto cannot know a model fits no single
        card, so it never splits; without `allow_shard` it may not. The fit is
        the one task that must stay on one card.

        MUTATION CONTROLS (2026-09-14, each red, restored by sha256):
          M14 readout places with allow_shard=False -> [readout] fails
          M15 fit places with allow_shard=True -> [fit] fails
          M20 readout places with required_mb=None -> [readout] fails
        """
        task = _task(module, name)
        with _worker(task) as run:
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                task.run(**kwargs, gpu_request="auto")

        assert len(run.placed) == 1, run.placed
        assert run.placed[0]["allow_shard"] is MAY_SPLIT[name]
        assert run.placed[0]["required_mb"] == pytest.approx(WORKER_REQUIRED_MB)


class TestTheReleaseFreesTheModelOnAFailure:
    """Being called is not being effective.

    A readout that dies mid-stream raises through `_read_out`, whose frame holds
    a `ReadoutService` built with `model=loaded.model`. The traceback keeps that
    frame alive, so a release that only nulls the task's own `loaded` frees
    nothing on the card. `release_card` clears the traceback's frames first.
    """

    def test_nothing_holds_the_model_when_the_card_is_released(self):
        import gc
        import weakref

        from src.workers.jlens_readout_tasks import compute_readout

        holder, seen = {}, {}

        class _Model:
            pass

        class _Loaded:
            """A real object; MagicMock children keep their parents alive."""

            def __init__(self):
                self.model = _Model()
                self.tokenizer = self.structure = self.unembedding = None
                self.name = "org/model"

        def make_loaded():
            loaded = _Loaded()
            holder["ref"] = weakref.ref(loaded.model)
            return loaded

        class _Service:
            """Keeps the model like the real ReadoutService, then dies capturing."""

            def __init__(self, model=None, **_kw):
                self.model = model

            def stream(self, *_a, **_kw):
                raise RuntimeError("CUDA out of memory during capture")

        def release():
            gc.collect()
            ref = holder.get("ref")
            seen["alive"] = ref is not None and ref() is not None

        with _worker(compute_readout, load=make_loaded, release=release), patch(
            "src.services.jlens_readout_service.ReadoutService", _Service
        ):
            with pytest.raises(RuntimeError, match="during capture"):
                compute_readout.run(model_id="m_1", prompt="hello", gpu_request=RTX_UUID)

        assert "alive" in seen, "the card was never released"
        assert seen["alive"] is False, (
            "the model was still referenced when the card was released — the "
            "traceback kept `_read_out`'s frame, and its ReadoutService, alive"
        )


# ── Readout: loaded or unloaded between readouts ────────────────────────────


class TestTheReadoutPlan:
    """`plan_readout`, the pure decision the readout task acts on."""

    def test_nothing_resident_is_placed(self):
        from src.workers.jlens_readout_tasks import ReadoutPlan, plan_readout

        assert plan_readout(None, None, "auto") == ReadoutPlan(reuse=False, release_first=False)

    def test_a_cpu_copy_is_placed_without_freeing_anything(self):
        from src.workers.jlens_readout_tasks import ReadoutPlan, plan_readout

        assert plan_readout("cpu", None, RTX_UUID) == ReadoutPlan(reuse=False, release_first=False)

    @pytest.mark.parametrize("requested", ["auto", None, RTX_UUID, RTX_UUID.lower()])
    def test_a_gpu_copy_is_reused_for_auto_or_for_its_own_card(self, requested):
        from src.workers.jlens_readout_tasks import ReadoutPlan, plan_readout

        assert plan_readout("cuda:1", CARDS[1], requested) == ReadoutPlan(reuse=True, release_first=False)

    def test_a_gpu_copy_is_freed_before_placing_on_another_card(self):
        from src.workers.jlens_readout_tasks import ReadoutPlan, plan_readout

        assert plan_readout("cuda:1", CARDS[1], TI_UUID) == ReadoutPlan(reuse=False, release_first=True)

    def test_a_gpu_copy_on_a_card_nvml_does_not_list_is_freed_then_placed(self):
        from src.workers.jlens_readout_tasks import ReadoutPlan, plan_readout

        assert plan_readout("cuda:1", None, "auto") == ReadoutPlan(reuse=False, release_first=True)


@contextmanager
def _readout_worker(*, resident=None, resident_card=None, place=None, read_out=None):
    """`_worker` for the readout: a resident copy (or none) and a readout that succeeds."""
    from src.workers.jlens_readout_tasks import compute_readout

    loaded = object()
    with _worker(compute_readout, place=place, load=lambda: loaded) as run, patch(
        "src.services.jlens_model_registry.resident_device_for", lambda record: resident
    ), patch(
        "src.services.gpu_placement.card_for_device", lambda device, cards=None: resident_card
    ), patch(
        "src.services.gpu_placement.list_cards", lambda: list(CARDS)
    ), patch(
        "src.services.gpu_placement.make_current",
        lambda device: run.events.append(("current", torch.device(device))),
    ), patch(
        "src.workers.jlens_readout_tasks._read_out",
        read_out or (lambda self, **kwargs: {"meta": {"kind": "meta"}, "tokens": []}),
    ):
        yield compute_readout, run


class TestTheReadoutChoosesWhetherToUnload:
    """`unload_after`: the user's choice between a fast next readout and a free card.

    User decision 2026-09-13: "give the user the option of toggling an unload
    between readouts". The default keeps the model loaded; a failure always frees.

    MUTATION CONTROLS (2026-09-13, each red, restored byte-identically):
      U1 the `finally` releases unconditionally (keep_loaded ignored)
           -> the default-keeps and Auto-reuses tests fail
      U2 plan_readout never reuses (Auto with a resident copy places anyway)
           -> the plan reuse tests and the Auto-reuses worker test fail
      U3 a different named card does not free the resident copy first
           -> the plan and the naming-another-card worker test fail
      U4 the endpoint drops `unload_after` from the task kwargs
           -> test_the_choice_reaches_the_task_and_the_row fails
      U5 keep_loaded is decided BEFORE the readout runs (a failure keeps the model)
           -> test_a_failed_readout_frees_the_card_even_when_asked_to_keep_it fails
    """

    def test_by_default_a_successful_readout_leaves_the_model_loaded(self):
        with _readout_worker() as (task, run):
            task.run(model_id="m_1", prompt="hello", gpu_request=RTX_UUID)

        assert run.events == [
            ("place", RTX_UUID),
            ("record", "t-gpu", RTX_UUID),
            ("load", RTX_PLACEMENT),
        ], "the model was freed although the caller asked to keep it loaded"

    def test_unload_after_frees_the_card_when_the_readout_ends(self):
        with _readout_worker() as (task, run):
            task.run(model_id="m_1", prompt="hello", gpu_request=RTX_UUID, unload_after=True)

        assert run.events == [
            ("place", RTX_UUID),
            ("record", "t-gpu", RTX_UUID),
            ("load", RTX_PLACEMENT),
            ("release",),
        ]

    def test_a_failed_readout_frees_the_card_even_when_asked_to_keep_it(self):
        def dies(self, **kwargs):
            raise RuntimeError("CUDA out of memory during capture")

        with _readout_worker(read_out=dies) as (task, run):
            with pytest.raises(RuntimeError, match="during capture"):
                task.run(model_id="m_1", prompt="hello", gpu_request=RTX_UUID, unload_after=False)

        assert run.events[-1] == ("release",), "a failed readout stranded its model on the card"

    def test_auto_reuses_the_copy_left_loaded_instead_of_placing_a_second(self):
        with _readout_worker(resident="cuda:1", resident_card=CARDS[1]) as (task, run):
            task.run(model_id="m_1", prompt="hello", gpu_request="auto")

        assert run.events == [
            ("current", RTX_DEVICE),
            ("record", "t-gpu", RTX_UUID),
            ("load", RTX_PLACEMENT),
        ], "Auto placed a second copy instead of reusing the one left on the 3090"

    def test_naming_another_card_frees_the_resident_copy_before_placing(self):
        on_ti = Placement(card=CARDS[0], device=torch.device("cuda", 0))
        with _readout_worker(
            resident="cuda:1", resident_card=CARDS[1], place=lambda requested: on_ti
        ) as (task, run):
            task.run(model_id="m_1", prompt="hello", gpu_request=TI_UUID)

        assert run.events == [
            ("release",),
            ("place", TI_UUID),
            ("record", "t-gpu", TI_UUID),
            ("load", on_ti),
        ]

    @pytest.mark.parametrize("unload_after", [False, True])
    def test_the_choice_reaches_the_task_and_the_row(self, unload_after):
        from src.api.v1.endpoints.jlens import readout
        from src.schemas.jlens import ReadoutRequest

        with _queued("src.workers.jlens_readout_tasks.compute_readout", _nothing) as (task, open_row):
            asyncio.run(
                readout(
                    ReadoutRequest(model_id="m_1", prompt="hello", unload_after=unload_after),
                    db=_db_with_model(),
                )
            )

        assert task.delay.call_count == 1
        assert task.delay.call_args.kwargs["unload_after"] is unload_after
        assert open_row.call_args.kwargs == {
            "retry_params": {"gpu": "auto", "unload_after": unload_after}
        }

    def test_the_request_defaults_to_keeping_the_model_loaded(self):
        from src.schemas.jlens import ReadoutRequest

        assert ReadoutRequest(model_id="m", prompt="x").unload_after is False


@contextmanager
def _resident(key):
    """The real single-entry cache holding a copy under `key`, restored afterwards."""
    from src.services import jlens_model_registry as registry

    saved = registry._CACHE._entry
    registry._CACHE._entry = types.SimpleNamespace(key=key) if key else None
    try:
        yield registry
    finally:
        registry._CACHE._entry = saved


@contextmanager
def _real_placement(seen):
    """The REAL `place_job` on a faked two-card node; `seen` gets the cache key
    as it stood when the inventory was read."""
    from src.services import jlens_model_registry as registry

    def inventory():
        seen.append(registry._CACHE.loaded_key)
        return list(CARDS)

    with patch.object(gpu_placement, "list_cards", inventory), patch.object(
        gpu_placement, "torch_device", lambda card: torch.device("cuda", card.index)
    ), patch("torch.cuda.is_available", lambda: True), patch(
        "torch.cuda.set_device", lambda device: None
    ), patch("torch.cuda.empty_cache", lambda: None), patch(
        # The release empties each card the copy named; there are no cards here.
        "src.ml.model_devices.empty_cache_on", lambda devices: None
    ):
        yield


class TestAnyPlacementFreesAnIdleReadoutCopy:
    """A readout's kept copy never skews another job's card choice.

    Review round 2 (2026-09-13). `unload_after=False` leaves the model on its
    card in the worker that runs EVERY `extraction`-queue job. Placement reads
    NVML, which counts that copy: a fit, probe, band report or intervention
    placed on Auto chose the OTHER card before its own load evicted the copy
    (gemma-4-12B Q8, 12.8 GB, onto the 12 GB 3080 Ti), and an activation
    extraction or circuit run, which never touches the J-lens cache, ran for
    hours beside it.

    MUTATION CONTROLS (round 2, each red, restored):
      P1 `place_job` without `release_idle_gpu_memory()`
           -> the Auto, named-card and J-space-sibling tests fail
      P2 the registry's `_register_with_placement()` call deleted
           -> the same three fail
      P3 `release_idle_gpu_copy` drops a CPU copy too
           -> test_a_cpu_copy_is_kept fails
    """

    @pytest.mark.parametrize("requested", ["auto", RTX_UUID], ids=["auto", "named"])
    def test_the_copy_is_freed_before_the_inventory_is_read(self, requested):
        seen = []
        with _resident("m_1@cuda:1") as registry, _real_placement(seen):
            placement = gpu_placement.place_job(requested)
            assert registry._CACHE.loaded_key is None, "the idle copy is still on the card"

        assert seen == [None], (
            f"the cards were judged with the idle copy still resident: {seen}")
        assert placement.card.uuid == RTX_UUID

    def test_a_j_space_sibling_task_frees_it_through_its_own_placement(self):
        seen = []
        with _resident("m_1@cuda:1") as registry, _real_placement(seen), patch(
            "src.workers.jlens_progress.record_gpu", lambda *a, **k: True
        ):
            from src.workers import jlens_progress

            jlens_progress.place_on_card("t-fit", "auto")
            assert registry._CACHE.loaded_key is None

        assert seen == [None]

    def test_a_cpu_copy_is_kept(self):
        seen = []
        with _resident("m_1@cpu") as registry, _real_placement(seen):
            gpu_placement.place_job("auto")
            assert registry._CACHE.loaded_key == "m_1@cpu"

        assert seen == ["m_1@cpu"]


class TestAKeptCopyIsTheSamePrecision:
    """A copy loaded at one quantization is not reused for another.

    Review round 2 (2026-09-13). Precision is switched by Re-download on the
    SAME model id, in another worker, so a cache keyed by id alone handed a
    readout the Q4 copy it had kept and reported it as FP16.

    MUTATION CONTROL (round 2, red, restored): `_model_key` returns the id alone
      -> test_a_copy_at_another_precision_is_not_the_resident_copy fails
    """

    @staticmethod
    def _record(quantization):
        return types.SimpleNamespace(
            id="m_1", repo_id="org/model", file_path=None, quantization=quantization
        )

    def test_a_copy_at_another_precision_is_not_the_resident_copy(self):
        from src.services import jlens_model_registry as registry

        keys = []
        with patch.object(
            registry._CACHE, "get_or_load", lambda key, loader: keys.append(key)
        ):
            registry.load_for_readout(self._record("Q4"), capture_device="cuda:1")

        with _resident(keys[0]):
            assert registry.resident_device_for(self._record("Q4")) == "cuda:1"
            assert registry.resident_device_for(self._record("FP16")) is None, (
                "a readout of the FP16 row would reuse the copy loaded at Q4")


# ── The row ────────────────────────────────────────────────────────────────


@pytest.fixture
def task_rows(monkeypatch):
    """A real `task_queue` table, in memory, behind the sync session the helpers use."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from src.models.task_queue import TaskQueue

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    TaskQueue.__table__.create(engine)
    session_factory = sessionmaker(bind=engine)

    @contextmanager
    def fake_sync_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr("src.core.database.get_sync_db", fake_sync_db)
    yield session_factory
    engine.dispose()


def _row(session_factory, task_id):
    from src.models.task_queue import TaskQueue

    session = session_factory()
    try:
        return session.query(TaskQueue).filter_by(task_id=task_id).one()
    finally:
        session.close()


class TestTheRowCarriesTheCardAndTheRequest:
    def test_open_row_stores_the_resolved_request_for_a_rerun(self, task_rows):
        """(e) on the real row, not on a mock of the call."""
        from src.workers import jlens_progress

        jlens_progress.open_row(
            jlens_progress.FIT, "m_1", "t-1", retry_params={"gpu": RTX_UUID}
        )
        row = _row(task_rows, "t-1")
        assert row.retry_params == {"gpu": RTX_UUID}
        assert row.gpu_uuid is None, "a queued job has not been placed on a card yet"

    def test_placing_the_job_writes_the_card_to_the_row(self, task_rows, monkeypatch):
        from src.workers import jlens_progress

        jlens_progress.open_row(jlens_progress.FIT, "m_1", "t-1", retry_params={"gpu": "auto"})
        monkeypatch.setattr(
            gpu_placement,
            "place_job",
            lambda requested, required_mb=None, allow_shard=False: Placement(
                card=CARDS[1], device=RTX_DEVICE
            ),
        )

        placement = jlens_progress.place_on_card("t-1", "auto")

        assert placement.device == RTX_DEVICE
        assert _row(task_rows, "t-1").gpu_uuid == RTX_UUID

    def test_a_row_opened_after_the_worker_starts_still_gets_its_card(
        self, task_rows, monkeypatch
    ):
        """The route opens the row AFTER `.delay()`, so the worker can arrive first."""
        import time

        from src.workers import jlens_progress

        waits = []

        def the_route_catches_up(seconds):
            waits.append(seconds)
            if len(waits) == 1:
                jlens_progress.open_row(jlens_progress.READOUT, "m_1", "t-late")

        monkeypatch.setattr(time, "sleep", the_route_catches_up)

        assert jlens_progress.record_gpu("t-late", RTX_UUID) is True
        assert waits, "the row existed before the first write, so no race was exercised"
        assert _row(task_rows, "t-late").gpu_uuid == RTX_UUID
