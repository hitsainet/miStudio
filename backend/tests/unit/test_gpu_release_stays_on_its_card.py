"""A job's end-of-claim memory release never opens a CUDA context on a card it did not lease.

Measured on mcs-lnxhost02 (2026-09-14, miStudio Phase 3/4 hardware acceptance): a
training on the RTX 3090 (cuda:1) left a 250 MiB CUDA context on the RTX 3080 Ti
(cuda:0) as it finished. The claim runs its bounded memory release on a helper
thread (`ClaimContext._release_memory`), and a thread that never selected a card
has device 0 current. A controlled run in the backend container isolated the
call: `torch._C._host_emptyCache()` on such a thread added 253 MiB to cuda:0;
the same call on the thread that had selected cuda:1 added nothing, and neither
did empty_cache on a named device, the memory stats, synchronize or pinned
allocation.

So the pinned-host release is run from the job's own card, and a job that held
no card (a hand-off) pinned nothing and releases nothing.

MUTATION CONTROLS (2026-09-14; each applied alone, this module run red, restored
byte-identically and checked by sha256):
  H1  _empty_pinned_host_cache calls the release without entering the device
        -> test_the_pinned_release_runs_inside_the_card_it_is_given
  H2  release_job_memory empties the pinned cache without a device
        -> test_a_release_on_the_claims_thread_names_the_leased_card
  H3  release_job_memory empties the pinned cache when nothing was held
        -> test_a_job_that_held_no_card_releases_no_pinned_memory
  H4  RollingActivationBuffer.close() empties the pinned cache without a device
        -> test_the_buffer_close_names_its_own_card
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import textwrap
import threading
from types import SimpleNamespace

import pytest
import torch

from src.services import activation_buffer as AB
from src.services import gpu_placement
from src.workers import gpu_job as G

RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
CUDA1 = torch.device("cuda", 1)


def test_the_pinned_release_runs_inside_the_card_it_is_given(monkeypatch):
    events = []

    @contextlib.contextmanager
    def device(target):
        events.append(("enter", target))
        yield
        events.append(("exit", target))

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: events.append("release"), raising=False)

    AB._empty_pinned_host_cache(device=CUDA1)

    assert events == [("enter", CUDA1), "release", ("exit", CUDA1)]


def test_without_a_device_the_release_keeps_the_callers_current_card(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "device", lambda target: pytest.fail("entered a device nobody named"))
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: events.append("release"), raising=False)

    AB._empty_pinned_host_cache()

    assert events == ["release"]


@pytest.fixture
def release_path(monkeypatch):
    """release_job_memory with every step but the pinned release stubbed."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_placement, "empty_cache_on_cards", lambda uuids: None)
    monkeypatch.setattr(G, "release_idle_copies_at_claim_end", lambda: None)
    monkeypatch.setattr(
        gpu_placement, "torch_device",
        lambda card: CUDA1 if getattr(card, "uuid", card) == RTX_UUID else pytest.fail(f"unknown card {card}"),
    )
    monkeypatch.setattr(AB, "_empty_pinned_host_cache", lambda device=None: calls.append(device))
    return calls


def test_a_release_on_the_claims_thread_names_the_leased_card(release_path):
    task = SimpleNamespace(release_job_memory=lambda: None)
    failures = []

    def on_a_helper_thread():  # as ClaimContext._release_memory runs it
        try:
            G.release_job_memory(task, (RTX_UUID,))
        except BaseException as exc:  # noqa: BLE001 - reported below
            failures.append(exc)

    thread = threading.Thread(target=on_a_helper_thread)
    thread.start()
    thread.join(10)

    assert not failures, failures
    assert release_path == [CUDA1]


def test_a_job_that_held_no_card_releases_no_pinned_memory(release_path):
    G.release_job_memory(SimpleNamespace(), ())

    assert release_path == []


def test_the_buffer_close_names_its_own_card():
    """close() passes the buffer's CUDA device to the pinned release.

    Asserted as the CALL in close()'s body: building a buffer on a CUDA device needs a
    GPU this suite does not have."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(AB.RollingActivationBuffer.close)))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_empty_pinned_host_cache"
    ]
    assert len(calls) == 1, "close() must release the pinned cache exactly once"
    device = {kw.arg: kw.value for kw in calls[0].keywords}.get("device")
    assert isinstance(device, ast.Call) and getattr(device.func, "id", None) == "_cuda_device_of", (
        "close() releases the pinned cache without naming its card"
    )
    assert [ast.unparse(arg) for arg in device.args] == ["self.storage_device", "self.train_device"]


def test_the_buffers_card_is_its_first_cuda_device():
    cpu = torch.device("cpu")

    assert AB._cuda_device_of(cpu, CUDA1) == CUDA1
    assert AB._cuda_device_of(torch.device("cuda", 0), CUDA1) == torch.device("cuda", 0)
    assert AB._cuda_device_of(cpu, cpu) is None
