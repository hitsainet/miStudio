"""Multi-GPU Phase 3's GPU leases, composed with the Phase 4 rolling buffer's review round 2.

Buffer round 2 (221020f6) releases a training's host-side memory in
``TrainingTask.after_return``: the activation stream is closed, then ``gc.collect()``,
``torch.cuda.empty_cache()`` and PyTorch's pinned host cache. It also closes any
stream a previous training left on the task before ``train_sae_task`` measures
memory. Phase 3 releases a job's GPU leases when ``@gpu_job``'s ``claiming()`` exits.
Ported together (2026-09-14), the order between them was nobody's to pin.

THE ORDER, AND WHY (changed by Phase 3 review round 1, finding R1-1). ``claiming()``
wraps the task body and Celery calls ``after_return`` only once the body has
returned, so the port released the leases BEFORE any buffer release ran. A finished
training's GPU buffer and cached blocks were then still allocated while its card was
offered: an Auto job read the card fuller than it was, and a job NAMING the card was
refused. So the claim now releases the memory FIRST, while the cards are still leased
(``ClaimContext._release_memory`` -> ``gpu_job.release_job_memory`` ->
``TrainingTask.release_job_memory``), and only then releases the leases:

* the lease release never waits for ever on a buffer close.
  ``RollingActivationBuffer.close`` joins its read pools, so the release is bounded
  (``RELEASE_MEMORY_TIMEOUT_S``); past it the lease is released anyway, with a warning;
* a failing buffer close skips nothing: ``_close_activation_stream`` logs the failure,
  the cache and pinned host releases after it still run, and so does the lease release.

``after_return`` still runs afterwards and finds nothing left to close.

A stream left by a PREVIOUS run is closed at the start of ``train_sae_task``, before
its first ``place_job``: this execution holds no lease yet, so that close delays
only this job.

MUTATION CONTROLS (2026-09-14, the port; each applied alone, this module run red,
restored byte-identically and checked by sha256):
  Composition (new with this module):
    Z1  _close_activation_stream lets a failing close escape (its try/except removed)
          -> test_the_leases_go_before_after_return_and_a_failing_close_skips_nothing
  Buffer round 2 controls re-run against these tests:
    Z3  (R2-afterreturn) after_return never empties the pinned host cache
          -> test_the_leases_go_before_after_return_and_a_failing_close_skips_nothing
    Z4  (R2-closeprev) train_sae_task no longer closes a buffer left on the task
          -> test_a_leftover_stream_is_closed_before_the_training_places
  Phase 3 control re-run against these tests:
    M3  close() does not release
          -> test_the_leases_go_before_after_return_and_a_failing_close_skips_nothing

REVIEW ROUND 1 (2026-09-14, finding R1-1, fix 3b857811): the order test was renamed
test_the_memory_goes_before_the_leases_and_a_failing_close_skips_nothing and now
requires the releases INSIDE the claim. It was red on c78f0f21's sources. Controls,
each alone, restored by sha256, `git diff` clean — all red:
    C-R1-1a close() skips the memory release
    C-R1-1d the @gpu_job wrapper does not wire release_memory
    C-R1-1e TrainingTask.release_job_memory does nothing
    C-R1-1f release_job_memory skips empty_cache_on_cards (re-run after the rename)
  Re-run on the changed test:
    Z1  _close_activation_stream lets a failing close escape (the claim's release
        stops at the close; empty_cache and the pinned release run only after the leases)
    Z3  after_return never empties the pinned host cache
    M3  close() does not release the leases
"""

import ast
import contextlib
import pathlib
from types import SimpleNamespace

import pytest
import torch
from celery import Celery

from src.core.config import settings
from src.services import activation_buffer
from src.services import gpu_job_claim as C
from src.services import gpu_leases, gpu_placement
from src.services.gpu_placement import GpuCard
from src.workers import gpu_job as G
from src.workers.gpu_supervisor import WORKER_GPU_ENV
from src.workers.training_tasks import TrainingTask
from tests.unit.gpu_lease_db import clear, lease_engine, session_factory

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
CARDS = [
    GpuCard(0, TI_UUID, "NVIDIA GeForce RTX 3080 Ti", 12_288, 11_000),
    GpuCard(1, RTX_UUID, "NVIDIA GeForce RTX 3090", 24_576, 23_000),
]
SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(scope="module")
def engine():
    eng = lease_engine("mistudio_test_gpu_leases_compose_buffer")
    yield eng
    eng.dispose()


@pytest.fixture
def node(engine, monkeypatch):
    """Per-card mode, the 3090's worker, real leases, the REAL place_job; every release recorded."""
    clear(engine)
    db = session_factory(engine)
    events = []

    def live():
        with db() as s:
            return gpu_leases.live_leases(s)

    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    monkeypatch.setenv(WORKER_GPU_ENV, RTX_UUID)
    monkeypatch.setattr(C, "_sync_session", db)
    monkeypatch.setattr(C, "list_cards", lambda: list(CARDS))
    monkeypatch.setattr(C, "host_available_mb", lambda: 1_000_000.0)
    monkeypatch.setattr(G, "park_job", lambda *a, **k: pytest.fail(f"the job was parked: {a}"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty_cache", live())))
    # The claim's release empties each leased card under its own device context.
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    # A card or its UUID, as the real torch_device takes.
    index_of = {card.uuid: card.index for card in CARDS}
    monkeypatch.setattr(
        gpu_placement, "torch_device",
        lambda card: torch.device("cuda", index_of[getattr(card, "uuid", card)]),
    )
    monkeypatch.setattr(
        activation_buffer, "_empty_pinned_host_cache", lambda device=None: events.append(("pinned", live()))
    )
    return SimpleNamespace(db=db, live=live, events=events)


class _FailingStream:
    def __init__(self, node):
        self._node = node

    def close(self):
        self._node.events.append(("close", self._node.live()))
        raise RuntimeError("the activation buffer's close failed")


def test_the_memory_goes_before_the_leases_and_a_failing_close_skips_nothing(node):
    """Through Celery's REAL trace: body -> claiming() exit (memory, then leases) -> after_return.

    A private Celery app, so the live registry other tests scan never sees this task."""
    app = Celery("p3-buffer-order", set_as_current=False)

    @app.task(bind=True, base=TrainingTask, name="tests.p3.training_release_order")
    @G.gpu_job("training")
    def train(self, job_id, gpu_request="auto"):
        gpu_placement.place_job(gpu_request, required_mb=4_000)
        node.events.append(("body", node.live()))
        self._activation_stream = _FailingStream(node)
        return "trained"

    result = train.apply(args=["job-1"], kwargs={"gpu_request": "auto"}, task_id="tid-order")

    assert result.successful(), result.traceback
    (kind, leased), *released = node.events
    assert kind == "body" and list(leased) == [RTX_UUID] and leased[RTX_UUID].startswith("training:tid-order:")
    assert released == [
        # Inside the claim, the card still leased: the buffer, the card's cache, pinned host memory.
        ("close", leased), ("empty_cache", leased), ("pinned", leased),
        # after_return, the leases gone: nothing left to close; the caches again, harmlessly.
        ("empty_cache", {}), ("pinned", {}),
    ], (
        "the card was released before the job's memory, or a failing close skipped a release "
        f"after it: {released}"
    )
    assert node.live() == {}
    assert train._activation_stream is None


def _calls(func: ast.AST, name: str) -> list:
    return sorted(
        node.lineno for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "attr", None) or getattr(node.func, "id", None)) == name
    )


def test_a_leftover_stream_is_closed_before_the_training_places():
    """A stream a previous run left on the task is closed while this execution holds no
    lease: its close CALL precedes the first place_job CALL (calls, never text)."""
    tree = ast.parse((SRC / "workers" / "training_tasks.py").read_text())
    train = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_sae_task"
    )
    closes, places = _calls(train, "_close_activation_stream"), _calls(train, "place_job")
    assert places, "the walk found no place_job call in train_sae_task"
    assert closes and closes[0] < places[0], (
        f"train_sae_task closes a leftover stream at lines {closes}, first placing at line {places[0]}"
    )
