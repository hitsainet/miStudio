"""Steering runs on the GPU the caller chose — Multi-GPU Plan, Phase 1.

The node gained a second card and it took NVML index 0. Steering had been
written for one: the API spawned its worker with ``CUDA_VISIBLE_DEVICES="0"``,
the service loaded with ``device_map="auto"`` against ``mem_get_info()`` of the
current device, and VRAM reporting read NVML handle 0. All of it moved to the
12 GB card without anyone deciding so.

What is pinned here, behaviourally (this workstation has no GPU, so torch.cuda,
NVML and the card inventory are faked):

  (a) each async endpoint dispatches the RESOLVED ``gpu`` — an index becomes the
      UUID it names — exactly once;
  (b) an unknown card is a 400 and nothing is spawned or dispatched;
  (c) the spawned steering worker's environment carries no CUDA_VISIBLE_DEVICES;
  (d) the model cache: Auto keeps a loaded model where it is, a different named
      card reloads onto that card, a fresh load puts ``from_pretrained`` on the
      placed device, and every generation path places BEFORE loading SAEs;
  (e) a GpuPlacementError fails the task with its message — no other card;
  (f) VRAM is read from the model's card, and cleanup touches only cards
      steering used.

MUTATION CONTROLS (each applied, run, required red, restored byte-identically):
  M1 restore env["CUDA_VISIBLE_DEVICES"] = "0" in _spawn_steering_worker
       -> test_the_spawned_worker_sees_every_card[*] fail
  M2 Auto re-places a loaded model (drop `is_auto(gpu) or` in place_model)
       -> test_auto_keeps_the_loaded_model_where_it_is fails
  M3 a named card never reloads (condition -> `if True`)
       -> test_a_different_named_card_reloads_onto_that_card fails
  M4 compare dispatches request.model_dump() without the resolved gpu
       -> test_the_endpoint_dispatches_the_resolved_card[compare-*] fails
  M5 VRAM reads nvmlDeviceGetHandleByIndex(0)
       -> test_vram_is_read_from_the_models_card fails
  M6 device_map={"": device} -> "auto"
       -> test_a_fresh_load_goes_onto_the_placed_device fails
  M7 generation paths place with "auto" instead of request.gpu
       -> test_the_generation_path_runs_on_the_requested_card[*] fail
  M8 cuda_devices() returns every visible card
       -> test_cleanup_synchronizes_only_the_cards_steering_used and
          test_a_fresh_service_synchronizes_nothing fail
  M9 compare skips resolve_gpu_request (raw value passes)
       -> test_an_unknown_card_is_a_400_before_anything_is_spawned_or_queued[compare] fails
  M10 a refused card falls back to place_job(AUTO)
       -> test_a_card_without_room_fails_the_task_and_nothing_moves_elsewhere[*] fail
All ten were run on 2026-09-13 against 7fa592e3: each red, each restored with
`git checkout`, `git diff --quiet` clean after every one.

A named card that cannot hold the model is refused with the arithmetic before
any weights load (preflight_gpu_capacity on the placed device; place_job only
had the 2 GB floor, because the size is unknown until the path is resolved):
  M11 load_model skips the size preflight
       -> test_a_model_the_named_card_cannot_hold_is_refused_before_any_weights_load
          and test_a_fresh_load_goes_onto_the_placed_device fail
  M12 the preflight measures device="auto" (every card together)
       -> the same two fail
Both run 2026-09-13 on main after df880182, each restored byte-identically.

MULTI-GPU PHASE 2 — a model no single card holds is SPLIT (sections g, h).
Steering places against the model's size with allow_shard=True; a split loads
with the placement's own device_map and max_memory, never a spelled strategy.
The split fixture puts the embedding on cuda:0 while the placement's first card
is cuda:1 (accelerate fills max_memory in index order), so a first-parameter or
`model.device` check cannot agree with it by construction.
  M1  SAE stays on self._device, not its hooked layer's
        -> test_each_sae_moves_to_its_hooked_layers_device
  M2  generation inputs back to self._device
        -> test_generation_inputs_go_to_the_embedding_and_every_card_is_synchronized
  M3  perplexity inputs back to self._device -> test_perplexity_inputs_go_to_the_embedding
  M4  core generator inputs back to model.device
        -> test_the_core_generator_sends_inputs_to_the_embedding
  M5  first-parameter device check restored (9 red, incl.)
        -> test_a_model_no_single_card_holds_loads_split_over_both_cards
  M6  cuda_devices() from each placement's first device only
        -> test_cleanup_covers_every_card_of_a_split (+2)
  M7  allow_shard dropped from place_model
        -> test_a_model_no_single_card_holds_loads_split_over_both_cards (+4)
  M8  placement asked for the 2 GB floor, not the model's size (7 red)
  M9  gpu_uuids not filled -> test_the_generation_path_reports_every_card_of_a_split[*]
  M10 max_memory dropped from the split load
        -> test_a_model_no_single_card_holds_loads_split_over_both_cards
  M11 split model moved to the CPU on release
        -> test_a_forced_reload_of_a_split_model_detaches_the_old_copy (+4)
  M12 _ensure_model_on_gpu pulls a split onto one card
        -> test_a_split_model_is_never_pulled_onto_one_card
  M13 a named card satisfied by a split starting on it
        -> test_a_named_card_is_not_satisfied_by_a_split_that_starts_on_it
  M14 off-GPU module check removed -> test_a_split_that_left_the_cards_is_refused_and_released[*]
  M15 generation synchronizes the first card only
        -> test_generation_inputs_go_to_the_embedding_and_every_card_is_synchronized
  M16 VRAM read from the first card only -> test_vram_is_summed_over_every_card_of_a_split
  M17 cleanup task empties only the current device
        -> test_the_cleanup_task_measures_every_card_of_a_split
        (SURVIVED first; the emptied-devices assertion was added, then red)
  M18 core SAE on the fallback device -> test_each_members_decoder_loads_on_its_layers_device
  M19 core split load without max_memory -> test_the_core_loader_splits_only_a_split_placement[True]
  M20 recorder drops the layer structure / M21 drops the placement
        -> test_a_split_recording_hands_the_placement_to_the_loader_and_the_layers_to_the_resolver
  M22 recorder task does not record gpu_uuids / M23 does not forward the placement
  M28 manifest gpu record lacks the split's cards
        -> test_the_recorder_task_records_every_card_of_a_split
  M24 recorder _resolve drops the structure -> test_the_recorder_resolver_forwards_the_layer_structure
  M25 steering tasks empty only the current device
        -> test_a_steering_task_releases_cache_on_every_card_of_a_split[*]
        (SURVIVED first; that test was written, then red)
  M26 split preflight measures the first card only
        -> test_a_model_no_single_card_holds_loads_split_over_both_cards
  M27 "all" never keeps a split -> test_all_keeps_a_split_that_already_spans_every_card
  M29 layer_device ignores the layer -> test_each_members_decoder_loads_on_its_layers_device
All 29 run 2026-09-14 by a script that checked each target occurred the expected
number of times, restored every file byte-identically (sha256), and compared the
tree's `git diff` hash before and after. M10/M11/M13/M17/M19/M25 were re-run after
the split device_map assertions stopped naming a strategy; all red.

REVIEW ROUND 1 (2026-09-14, sections i and j). Fix controls — each applied alone,
the module run red, restored and checked by sha256, `git status` clean after:
  R1-N1 load_model no longer re-reads a split's budget (call -> `pass`)
        -> test_a_split_is_budgeted_from_the_memory_free_when_its_weights_load,
           test_the_first_split_load_budgets_the_memory_the_saes_already_took
  R1-N2 the re-read budget is discarded (`return placement`) -> the same two
  R1-N3 the re-read budget ignores SHARD_RESERVE_MB
        -> the same two + test_a_model_no_single_card_holds_loads_split_over_both_cards
  R1-N4 unload_model cleans without the unloaded model's cards (`cleanup_gpu()`)
        -> test_unloading_a_model_releases_its_cards_while_another_placement_is_current
  R1-N5 cleanup_gpu's second pass empties only cuda_devices() -> the same test
Probes of load-bearing lines that SURVIVED the widened steering suite (260 tests)
before this round, each now killed by its own test:
  R1-P1  "all" satisfied by any split -> test_all_is_not_satisfied_by_a_split_that_misses_a_visible_card
  R1-P2  parameter-level stray check removed
        -> test_a_parameter_off_the_placements_cards_is_refused_even_when_the_map_says_gpu
  R1-P3  _is_split ignores the model's own device map
        -> test_a_dispatched_model_with_no_recorded_placement_is_never_moved_to_the_cpu
  R1-P28 a refused split emptied on its first card only -> test_a_refused_split_empties_every_card_it_was_given
  R1-P32 revert 48fd53ea (module_device on a non-Module hook target)
        -> test_multi_sae_hooks.py (3 tests); that previous-round fix is pinned.

MULTI-GPU PHASE 3, PORTED ONTO MAIN (2026-09-14, section k). Phase 3's GPU leases
composed with review round 1's load-time split budget, on a THREE-card node so that
"only the leased cards" cannot hold by construction. Each applied alone, run red,
restored and checked by sha256:
  X7  the load-time split budget reads every visible card, not the placement's
        -> test_a_split_is_budgeted_at_load_only_on_the_cards_its_claim_leased
  M1  (Phase 3) place_job resolves over every card instead of claiming -> the same test
  M3  (Phase 3) close() does not release                               -> the same test
A defect the composition exposed, fixed in its own commit: a model the cache KEPT was
reused by `place_model` without a lease, so a surviving worker's next request reloaded
(and re-budgeted) it on cards its claim never held:
  SR1 place_model's reuse branch no longer leases the kept cards
        -> test_a_kept_split_is_leased_again_before_the_next_request_reloads_it,
           test_a_kept_split_is_never_reloaded_onto_a_card_another_job_now_holds

REVIEW ROUND 2, STEERING REVIEWER (2026-09-14). Each layer's SAE goes on that layer's
card and a split keeps only SHARD_RESERVE_MB free per card, so the split load is told
each layer's SAE size (ml/split_load.py `extra_mb_by_layer`). Controls (scratchpad
p2-r2-steer/mutate.py; each alone, restored byte-identically by sha256; all killed):
  A8  SAEs sized at 2 bytes, not _load_sae_sync's fp32  -> test_the_core_split_load_is_mapped_to_hold_each_layers_sae
                                                          (+ the four circuit cases)
  A14 the core split load gets no allowance             -> test_the_core_split_load_is_mapped_to_hold_each_layers_sae
  A16 the recorder's split load gets no SAE ids         -> test_a_split_recording_hands_the_placement_to_the_loader_...

REVIEW ROUND 3 (2026-09-14): THE CORE KEEPS ONLY DECODERS. Calibration and the recorder steer
with W_dec alone, but built each whole SAE on its layer's card first, and the split was mapped
for the whole SAE: with the round-2 overshoot also fixed, OLMo-2-1124-13B at FP16 with decoders
beside layers 11-13 on the node's 11,000 + 23,000 MB fits; charged whole, it was refused. Each
SAE is now built on the CPU and only its decoder moves. Controls (scratchpad p2-r3/mutate.py +
mutations_r3.json, each alone in a private copy of backend/, restored by sha256), all killed:
  N7  the SAE built on its layer's device again         -> test_each_members_decoder_loads_on_its_layers_device
  N7b the decoder not moved to its layer's device       -> the same test
  N8  the core's split sized for the whole SAE          -> test_the_core_split_load_is_mapped_to_hold_each_layers_sae
  N8b sae_mb_by_layer ignores decoder_only              -> the same test
Round 2's A8 (re-expressed on the new signature), A14 (re-expressed on the decoder-only call),
A16 and A17 re-run: all killed.

PHASE 3 REVIEW ROUND 1 (2026-09-14, supervisor/workers/deployment), item 3: an Auto
request no longer waits for a kept model's card that another job now holds — the kept
weights are what that job needs — it releases the model and places again; a named card
still waits and refuses. Each alone, restored byte-identically (sha256), all red:
  SR1 (re-run as a negative control of the previous round's fix) -> still red
  SR2 the release also applies to a named card (`is_auto` dropped)
        -> test_a_named_card_still_waits_for_its_kept_model_and_refuses_with_the_reason
  SR3 the busy check disabled (`busy = []`)
        -> test_an_auto_request_releases_a_kept_split_on_a_busy_card_and_places_anew,
           test_an_auto_request_releases_a_kept_single_card_model_on_a_busy_card
  SR4 gpu_job_claim.cards_held_by_other_jobs returns nothing
        -> test_an_auto_request_releases_a_kept_split_on_a_busy_card_and_places_anew
"""

import asyncio
import contextlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi import HTTPException

from src.services import gpu_placement
from src.services import steering_service as steering_module
from src.services.gpu_placement import GpuCard, GpuPlacementError, Placement
from src.services.steering_service import MIN_FREE_MB_FOR_MODEL_LOAD, SteeringService

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"

TI = GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000)
RTX = GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000)
CARDS = [TI, RTX]

CUDA0 = torch.device("cuda", 0)
CUDA1 = torch.device("cuda", 1)


# ── Fakes ──────────────────────────────────────────────────────────────────


class FakeCuda:
    """Records every device-taking torch.cuda call a steering path makes."""

    def __init__(self):
        self.set_device = []
        self.synchronize = []
        self.mem_get_info = []
        self.memory_allocated = []
        self.allocated_bytes = {CUDA0: 0, CUDA1: 0}


@pytest.fixture
def cuda(monkeypatch):
    fake = FakeCuda()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda index: SimpleNamespace(uuid=[TI_UUID, RTX_UUID][index][len("GPU-"):]),
    )
    monkeypatch.setattr(torch.cuda, "set_device", fake.set_device.append)

    def synchronize(device=None):
        fake.synchronize.append(device)

    def mem_get_info(device=None):
        fake.mem_get_info.append(device)
        return (20_000 * 1024**2, 24_000 * 1024**2)

    def memory_allocated(device=None):
        fake.memory_allocated.append(device)
        return fake.allocated_bytes.get(device, 0)

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(torch.cuda, "memory_allocated", memory_allocated)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: CARDS[index].name)
    return fake


#: ~1.2B parameters: ~2.2 GB at FP16, fits either card.
SMALL_CONFIG = SimpleNamespace(
    hidden_size=2048, num_hidden_layers=16, vocab_size=65_536, intermediate_size=8_192
)
#: ~6.7B parameters: ~12.5 GB of FP16 weights + 2 GB headroom — the 3090 holds it,
#: the 3080 Ti does not.
MEDIUM_CONFIG = SimpleNamespace(
    hidden_size=4096, num_hidden_layers=32, vocab_size=32_000, intermediate_size=11_008
)
#: ~13B parameters: ~24.2 GB of FP16 weights + 2 GB headroom. Neither card holds
#: it (3090: 23,000 MB free); their split budgets (21,976 + 9,976 MB) do.
BIG_CONFIG = SimpleNamespace(
    hidden_size=5120, num_hidden_layers=40, vocab_size=32_000, intermediate_size=13_824
)

#: What placement is asked for, worked BY HAND from the preflight's arithmetic
#: (FP16 = 2 bytes/param, + 2 GiB headroom) rather than recomputed by the code
#: under test, so a changed formula cannot agree with itself:
#:   SMALL:  1,342,177,280 params -> 2,684,354,560 + 2,147,483,648 B = 4,608.0 MiB
#:   MEDIUM: 6,738,149,376 params -> 13,476,298,752 + 2,147,483,648 B = 14,900.0 MiB
#:   BIG:   13,015,449,600 params -> 26,030,899,200 + 2,147,483,648 B = 26,873.6 MiB
SMALL_NEED_MB = 4_608.0
MEDIUM_NEED_MB = 14_900.0
BIG_NEED_MB = 28_178_382_848 / 1024**2


class FakeModel:
    """A 'model' whose parameters sit on the device(s) it was loaded onto.

    A split model lists one parameter per device IN THE ORDER accelerate fills
    them: it sorts ``max_memory`` by torch index, so the embedding — the first
    registered parameter — lands on cuda:0 even when the placement's first
    (most free) card is cuda:1. A fixture with the embedding on
    ``placement.device`` would agree with a first-parameter check by
    construction.
    """

    def __init__(self, device, devices=None, hf_device_map=None):
        self.device = device
        self.devices = list(devices) if devices else [device]
        self.moved_to = []
        self.config = SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map

    def parameters(self):
        for device in self.devices:
            yield SimpleNamespace(device=device)

    def to(self, device):
        self.moved_to.append(device)
        return self

    def cpu(self):
        return self.to("cpu")

    def eval(self):
        return self


def _split_model(*, stray=None):
    """What accelerate returns for a split over cuda:1 + cuda:0: embedding on cuda:0."""
    device_map = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 1, "lm_head": 1}
    devices = [CUDA0, CUDA0, CUDA1, CUDA1]
    if stray is not None:
        device_map["lm_head"] = stray
        devices[-1] = torch.device(stray) if stray != "disk" else torch.device("meta")
    return FakeModel(CUDA0, devices=devices, hf_device_map=device_map)


class _Calls(list):
    """A list of loader calls that can carry the fixture's knobs."""


@pytest.fixture
def loader(monkeypatch):
    """Fake from_pretrained: one card for a {"": device} map, a split for a strategy string.

    Keyed on the TYPE, never a literal: the split strategy is the placement's
    (`Placement.device_map`), and the core changed it once already.
    """
    calls = _Calls()
    state = SimpleNamespace(stray=None)

    def from_pretrained(path, **kwargs):
        calls.append(kwargs)
        device_map = kwargs.get("device_map")
        if isinstance(device_map, str):
            return _split_model(stray=state.stray)
        device = device_map[""] if isinstance(device_map, dict) else torch.device("cpu")
        return FakeModel(device)

    import transformers

    tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="</s>")
    monkeypatch.setattr(steering_module.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(steering_module.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: SMALL_CONFIG)
    calls.state = state
    return calls


#: The split a 13B-class FP16 model gets on this node: the 3090 first (most
#: free), the 3080 Ti second, each budgeted its free memory less 1024 MB.
SPLIT = Placement(
    card=RTX, device=CUDA1, cards=(RTX, TI), devices=(CUDA1, CUDA0),
    max_memory_mb={1: 21_976, 0: 9_976},
)


@pytest.fixture
def placer(monkeypatch):
    """Fake place_job. Auto answers `auto_card` (or SPLIT when `split`); a named card answers itself."""
    state = SimpleNamespace(calls=[], allow_shard=[], auto_card=RTX, split=False)

    def place_job(requested="auto", required_mb=None, cards=None, allow_shard=False):
        state.calls.append((requested, required_mb))
        state.allow_shard.append(allow_shard)
        if gpu_placement.is_auto(requested) and state.split:
            torch.cuda.set_device(SPLIT.device)
            return SPLIT
        if gpu_placement.is_auto(requested):
            card = state.auto_card
        else:
            card = gpu_placement.find_card(requested, CARDS)
        device = torch.device("cuda", card.index)
        torch.cuda.set_device(device)
        return Placement(card=card, device=device)

    monkeypatch.setattr(steering_module, "place_job", place_job)
    return state


def _service(monkeypatch) -> SteeringService:
    svc = SteeringService()
    # Hook discovery walks a real transformer; these fakes have no layers.
    monkeypatch.setattr(svc, "_clear_all_model_hooks", lambda model: 0)
    monkeypatch.setattr(svc, "_reset_model_state", lambda model: None)
    return svc


# ── (d) The model cache ────────────────────────────────────────────────────


def test_a_fresh_load_goes_onto_the_placed_device(monkeypatch, cuda, loader, placer):
    svc = _service(monkeypatch)

    model, _ = asyncio.run(svc.load_model("m", gpu=RTX_UUID))

    assert placer.calls == [(RTX_UUID, SMALL_NEED_MB)]
    assert len(loader) == 1
    assert loader[0]["device_map"] == {"": CUDA1}
    assert loader[0]["torch_dtype"] == torch.float16
    assert model.device == CUDA1
    # The size preflight (by index) and the free-memory re-check both read THAT
    # card, never the current device.
    assert cuda.mem_get_info == [1, CUDA1]
    assert svc._device == CUDA1
    assert svc._model_placements["m"].uuid == RTX_UUID


def _free_by_card(ti_free_mb, rtx_free_mb):
    """mem_get_info that answers per card, by index or torch.device."""

    def mem_get_info(device=None):
        index = device.index if isinstance(device, torch.device) else device
        free_mb = {0: ti_free_mb, 1: rtx_free_mb}[index]
        return free_mb * 1024**2, CARDS[index].total_mb * 1024**2

    return mem_get_info


def test_a_model_the_named_card_cannot_hold_is_refused_before_any_weights_load(
    monkeypatch, cuda, loader, placer
):
    """~6.7B at FP16 needs ~14.5 GB: refused on the 3080 Ti with the arithmetic,
    instead of running out of memory partway through from_pretrained."""
    import transformers

    from src.services.resource_config import VRAMInsufficientError

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: MEDIUM_CONFIG)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _free_by_card(11_000, 23_000))
    svc = _service(monkeypatch)

    with pytest.raises(VRAMInsufficientError) as exc:
        asyncio.run(svc.load_model("seven_b", gpu=TI_UUID))

    assert "cuda:0 (NVIDIA GeForce RTX 3080 Ti)" in str(exc.value)
    assert loader == [], "weights were fetched for a model the card cannot hold"
    assert "seven_b" not in svc._loaded_models


def test_the_same_model_loads_on_the_card_that_can_hold_it(monkeypatch, cuda, loader, placer):
    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: MEDIUM_CONFIG)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _free_by_card(11_000, 23_000))
    svc = _service(monkeypatch)

    model, _ = asyncio.run(svc.load_model("seven_b", gpu=RTX_UUID))

    assert model.device == CUDA1
    assert [call["device_map"] for call in loader] == [{"": CUDA1}]


def test_an_unreadable_config_leaves_only_the_free_memory_floor(monkeypatch, cuda, loader, placer):
    """Unknown size must not block a load: the 2 GB floor is all that applies."""
    import transformers

    def unreadable(*args, **kwargs):
        raise OSError("no config.json")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", unreadable)
    svc = _service(monkeypatch)

    model, _ = asyncio.run(svc.load_model("mystery", gpu=TI_UUID))

    assert model.device == CUDA0
    assert [call["device_map"] for call in loader] == [{"": CUDA0}]


def test_auto_keeps_the_loaded_model_where_it_is(monkeypatch, cuda, loader, placer):
    """Auto is satisfied by the loaded model, even when a fresh Auto choice would
    now pick a different card — so re-placing would be visible."""
    svc = _service(monkeypatch)
    first, _ = asyncio.run(svc.load_model("m", gpu=RTX_UUID))
    placer.auto_card = TI  # what a fresh Auto would choose now

    again, _ = asyncio.run(svc.load_model("m", gpu="auto"))

    assert again is first
    assert len(loader) == 1, "Auto reloaded a model that was already loaded"
    assert len(placer.calls) == 1, "Auto re-placed a model that was already loaded"
    assert first.moved_to == []
    assert svc._device == CUDA1
    assert cuda.set_device[-1] == CUDA1


def test_a_forced_reload_under_auto_stays_on_the_models_card(monkeypatch, cuda, loader, placer):
    """Every generation path force-reloads; under Auto that reload lands where
    the model already was, not wherever a fresh Auto choice would put it."""
    svc = _service(monkeypatch)
    asyncio.run(svc.load_model("m", gpu=RTX_UUID))
    placer.auto_card = TI

    asyncio.run(svc.load_model("m", gpu="auto", force_reload=True))

    assert [call["device_map"] for call in loader] == [{"": CUDA1}, {"": CUDA1}]
    assert len(placer.calls) == 1


def test_a_different_named_card_reloads_onto_that_card(monkeypatch, cuda, loader, placer):
    svc = _service(monkeypatch)
    first, _ = asyncio.run(svc.load_model("m", gpu=RTX_UUID))

    moved, _ = asyncio.run(svc.load_model("m", gpu=TI_UUID))

    assert moved is not first
    assert first.moved_to == ["cpu"], "the model on the old card was not unloaded"
    assert placer.calls == [
        (RTX_UUID, SMALL_NEED_MB),
        (TI_UUID, SMALL_NEED_MB),
    ]
    assert [call["device_map"] for call in loader] == [{"": CUDA1}, {"": CUDA0}]
    assert svc._model_placements["m"].uuid == TI_UUID
    assert svc._device == CUDA0


@pytest.mark.parametrize("same_card", [RTX_UUID.lower(), RTX_UUID[len("GPU-"):], "1"])
def test_naming_the_card_the_model_is_on_keeps_it(monkeypatch, cuda, loader, placer, same_card):
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)
    svc = _service(monkeypatch)
    first, _ = asyncio.run(svc.load_model("m", gpu=RTX_UUID))

    again, _ = asyncio.run(svc.load_model("m", gpu=same_card))

    assert again is first
    assert len(loader) == 1 and len(placer.calls) == 1


# ── (d) Every generation path places first and reports the card ────────────


def _loaded_sae(device):
    return steering_module.LoadedSAE(
        model=MagicMock(), config=None, layer=3, d_in=8, d_sae=16, device=str(device),
    )


REQUESTS = {
    "compare": (
        "generate_comparison_sync",
        {
            "sae_id": "sae_A", "model_id": "m", "prompt": "hi",
            "selected_features": [{"feature_idx": 1, "layer": 3, "strength": 5}],
            "include_unsteered": False, "compute_metrics": False,
        },
    ),
    "sweep": (
        "generate_strength_sweep_sync",
        {
            "sae_id": "sae_A", "model_id": "m", "prompt": "hi",
            "feature_idx": 1, "layer": 3, "strength_values": [1.0, 2.0],
        },
    ),
    "combined": (
        "generate_combined_sync",
        {
            "sae_id": "sae_A", "model_id": "m", "prompt": "hi",
            "selected_features": [{"feature_idx": 1, "layer": 3, "strength": 5}],
            "include_baseline": False, "compute_metrics": False,
        },
    ),
}


@pytest.mark.parametrize("kind", sorted(REQUESTS))
def test_the_generation_path_runs_on_the_requested_card(monkeypatch, cuda, loader, placer, kind):
    """request_dict["gpu"] -> place_model -> SAEs and model on that card ->
    gpu_uuid in the result. The SAEs load BEFORE the model, so the device they
    see proves placement came first."""
    svc = _service(monkeypatch)
    sae_devices = []

    async def load_sae(*args, **kwargs):
        sae_devices.append(svc._device)
        return _loaded_sae(svc._device)

    async def resolve_sae_map(request, meta_map, force_reload=False):
        sae_devices.append(svc._device)
        return {request.sae_id: _loaded_sae(svc._device)}

    monkeypatch.setattr(svc, "load_sae", load_sae)
    monkeypatch.setattr(svc, "resolve_sae_map", resolve_sae_map)
    monkeypatch.setattr(svc, "_register_steering_hooks", lambda *a, **k: [])
    monkeypatch.setattr(svc, "_generate_text", AsyncMock(return_value=("text", 1, 1)))
    monkeypatch.setattr(
        svc, "_compute_metrics",
        AsyncMock(return_value=steering_module.GenerationMetrics(token_count=1, generation_time_ms=1)),
    )

    method, request_dict = REQUESTS[kind]
    result = getattr(svc, method)(
        request_dict={**request_dict, "gpu": TI_UUID},
        sae_path="/nonexistent/sae",
        model_id="m",
    )

    assert placer.calls == [(TI_UUID, SMALL_NEED_MB)]
    assert sae_devices == [CUDA0], "the SAE loaded before the card was chosen"
    assert [call["device_map"] for call in loader] == [{"": CUDA0}]
    assert result["gpu_uuid"] == TI_UUID


# ── (e) A placement refusal fails the task with its message ────────────────


TASK_KWARGS = {
    "sae_id": "sae_A", "model_id": "m", "sae_path": "/nonexistent/sae",
    "sae_layer": 3, "sae_d_model": 8, "sae_n_features": 16, "sae_architecture": "standard",
}


@pytest.mark.parametrize("kind", sorted(REQUESTS))
def test_a_card_without_room_fails_the_task_and_nothing_moves_elsewhere(
    monkeypatch, cuda, loader, kind
):
    from src.workers import steering_tasks

    busy_ti = GpuCard(0, TI_UUID, TI.name, TI.total_mb, free_mb=500)
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: [busy_ti, RTX])
    svc = _service(monkeypatch)
    sae_load = AsyncMock()
    monkeypatch.setattr(svc, "load_sae", sae_load)
    monkeypatch.setattr(svc, "resolve_sae_map", sae_load)
    monkeypatch.setattr(steering_module, "get_steering_service", lambda: svc)
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=SimpleNamespace(returncode=1, stdout="")))
    emitted = MagicMock(return_value=True)
    monkeypatch.setattr(steering_tasks, "emit_steering_progress", emitted)

    task = {
        "compare": steering_tasks.steering_compare_task,
        "sweep": steering_tasks.steering_sweep_task,
        "combined": steering_tasks.steering_combined_task,
    }[kind]
    _, request_dict = REQUESTS[kind]
    result = task.apply(kwargs={**TASK_KWARGS, "request_dict": {**request_dict, "gpu": TI_UUID}})

    assert result.failed()
    assert isinstance(result.result, GpuPlacementError)
    assert "cannot take this job" in str(result.result)
    errors = [c.kwargs.get("error") for c in emitted.call_args_list if c.kwargs.get("error")]
    assert errors == [str(result.result)]
    # Refused, not substituted: no SAE, no model, no card chosen.
    assert loader == [] and sae_load.await_count == 0
    assert svc._placement is None
    assert cuda.set_device == []


# ── (f) VRAM and cleanup read the model's card ─────────────────────────────


def _fake_pynvml(monkeypatch, *, used_by_uuid, used_by_index, init_error=None):
    calls = SimpleNamespace(by_uuid=[], by_index=[])
    module = types.ModuleType("pynvml")

    def nvmlInit():
        if init_error:
            raise init_error

    def by_uuid(uuid):
        calls.by_uuid.append(uuid)
        return ("uuid", uuid)

    def by_index(index):
        calls.by_index.append(index)
        return ("index", index)

    def memory_info(handle):
        kind, key = handle
        used = used_by_uuid[key] if kind == "uuid" else used_by_index[key]
        return SimpleNamespace(used=used)

    module.nvmlInit = nvmlInit
    module.nvmlShutdown = lambda: None
    module.nvmlDeviceGetHandleByUUID = by_uuid
    module.nvmlDeviceGetHandleByIndex = by_index
    module.nvmlDeviceGetMemoryInfo = memory_info
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return calls


def test_vram_is_read_from_the_models_card(monkeypatch, cuda):
    calls = _fake_pynvml(
        monkeypatch,
        used_by_uuid={RTX_UUID: 7 * 1024**3, TI_UUID: 3 * 1024**3},
        used_by_index={0: 3 * 1024**3, 1: 7 * 1024**3},
    )
    svc = SteeringService()
    svc._placement = Placement(card=RTX, device=CUDA1)

    assert svc._get_system_vram_usage_gb() == pytest.approx(7.0)
    assert calls.by_uuid == [RTX_UUID]
    assert calls.by_index == []


def test_without_nvml_vram_falls_back_to_this_process_on_the_models_card(monkeypatch, cuda):
    _fake_pynvml(monkeypatch, used_by_uuid={}, used_by_index={}, init_error=RuntimeError("no driver"))
    cuda.allocated_bytes[CUDA1] = 2 * 1024**3
    svc = SteeringService()
    svc._placement = Placement(card=RTX, device=CUDA1)

    assert svc._get_system_vram_usage_gb() == pytest.approx(2.0)
    assert cuda.memory_allocated == [CUDA1]


def test_before_any_placement_there_is_no_card_to_report(monkeypatch, cuda):
    calls = _fake_pynvml(monkeypatch, used_by_uuid={}, used_by_index={0: 9 * 1024**3})
    assert SteeringService()._get_system_vram_usage_gb() == 0.0
    assert calls.by_index == [] and calls.by_uuid == []


def test_cleanup_synchronizes_only_the_cards_steering_used(monkeypatch, cuda):
    """A synchronize creates a CUDA context on the card it names. The worker now
    sees every card, so sweeping all of them would take memory on the card
    steering never touched."""
    svc = _service(monkeypatch)
    svc._model_placements["m"] = Placement(card=RTX, device=CUDA1)
    svc._placement = svc._model_placements["m"]

    svc.synchronize()
    svc.cleanup_gpu()

    assert cuda.synchronize == [CUDA1, CUDA1]


def test_a_fresh_service_synchronizes_nothing(monkeypatch, cuda):
    SteeringService().synchronize()
    assert cuda.synchronize == []


def test_the_cleanup_task_measures_the_models_card(monkeypatch, cuda):
    from src.workers import steering_tasks

    svc = _service(monkeypatch)
    svc._loaded_models["m"] = (FakeModel(CUDA1), object())
    svc._model_placements["m"] = Placement(card=RTX, device=CUDA1)
    svc._placement = svc._model_placements["m"]
    cuda.allocated_bytes[CUDA1] = 5 * 1024**3
    cuda.allocated_bytes[CUDA0] = 1 * 1024**3

    def unload_all():
        cuda.allocated_bytes[CUDA1] = 1 * 1024**3
        return {"models_unloaded": 1, "saes_unloaded": 0}

    monkeypatch.setattr(svc, "unload_all", unload_all)
    monkeypatch.setattr(steering_module, "get_steering_service", lambda: svc)

    result = steering_tasks.cleanup_steering_gpu.apply().get()

    assert result["memory_freed_gb"] == pytest.approx(4.0)
    assert cuda.memory_allocated == [CUDA1, CUDA1]


# ── (a)(b) The endpoints ───────────────────────────────────────────────────


def _sae_row():
    sae = MagicMock()
    sae.status = "ready"
    sae.local_path = "saes/sae_A"
    sae.layer, sae.d_model, sae.n_features, sae.architecture = 3, 8, 16, "standard"
    sae.model_id = sae.model_name = "m"
    return sae


def _endpoint_request(kind, gpu):
    from src.schemas.steering import (
        CombinedSteeringRequest,
        SteeringComparisonRequest,
        SteeringStrengthSweepRequest,
    )

    schema = {
        "compare": SteeringComparisonRequest,
        "sweep": SteeringStrengthSweepRequest,
        "combined": CombinedSteeringRequest,
    }[kind]
    return schema(**REQUESTS[kind][1], gpu=gpu)


ENDPOINTS = {
    "compare": ("submit_async_steering_comparison", "steering_compare_task"),
    "sweep": ("submit_async_strength_sweep", "steering_sweep_task"),
    "combined": ("submit_async_combined_steering", "steering_combined_task"),
}


@contextlib.contextmanager
def _endpoint_environment(task_name):
    http_request = MagicMock()
    http_request.client = MagicMock(host="127.0.0.1")
    http_request.headers = {}
    ensure = AsyncMock(return_value=(True, 123))
    with patch("src.api.v1.endpoints.steering._rate_limiter") as limiter, \
         patch("src.api.v1.endpoints.steering._ensure_steering_worker_running", new=ensure), \
         patch("src.api.v1.endpoints.steering._guard_steering_dispatch", new=AsyncMock()), \
         patch("src.api.v1.endpoints.steering.SAEManagerService.get_sae",
               new=AsyncMock(return_value=_sae_row())), \
         patch("src.api.v1.endpoints.steering.resolve_referenced_saes",
               new=AsyncMock(return_value={})), \
         patch("src.api.v1.endpoints.steering.ModelService.get_model",
               new=AsyncMock(return_value=None)), \
         patch("src.api.v1.endpoints.steering.settings") as settings, \
         patch(f"src.workers.steering_tasks.{task_name}") as task:
        limiter.is_allowed.return_value = True
        settings.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        task.apply_async.return_value = MagicMock(id="task-1")
        yield SimpleNamespace(http_request=http_request, ensure=ensure, task=task)


@pytest.mark.parametrize("requested,dispatched", [("1", RTX_UUID), ("auto", "auto"), (TI_UUID.lower(), TI_UUID)])
@pytest.mark.parametrize("kind", sorted(ENDPOINTS))
def test_the_endpoint_dispatches_the_resolved_card(monkeypatch, kind, requested, dispatched):
    from src.api.v1.endpoints import steering as endpoints

    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)
    handler_name, task_name = ENDPOINTS[kind]
    with _endpoint_environment(task_name) as env:
        asyncio.run(getattr(endpoints, handler_name)(
            _endpoint_request(kind, requested), env.http_request, db=MagicMock(),
        ))

    assert env.task.apply_async.call_count == 1
    request_dict = env.task.apply_async.call_args.kwargs["kwargs"]["request_dict"]
    assert request_dict["gpu"] == dispatched
    assert request_dict["prompt"] == "hi"


@pytest.mark.parametrize("kind", sorted(ENDPOINTS))
def test_an_unknown_card_is_a_400_before_anything_is_spawned_or_queued(monkeypatch, kind):
    from src.api.v1.endpoints import steering as endpoints

    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)
    handler_name, task_name = ENDPOINTS[kind]
    with _endpoint_environment(task_name) as env:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(getattr(endpoints, handler_name)(
                _endpoint_request(kind, "GPU-00000000-0000-0000-0000-000000000000"),
                env.http_request, db=MagicMock(),
            ))

    assert exc.value.status_code == 400
    assert "RTX 3090" in exc.value.detail
    assert env.task.apply_async.call_count == 0
    assert env.ensure.await_count == 0


# ── (c) The spawned worker sees every card ─────────────────────────────────


@pytest.mark.parametrize("entry", ["_ensure_steering_worker_running", "enter_steering_mode"])
def test_the_spawned_worker_sees_every_card(monkeypatch, tmp_path, entry):
    from src.api.v1.endpoints import steering as endpoints
    from src.workers import steering_worker_state

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("MISTUDIO_SPAWN_SENTINEL", "inherited")
    monkeypatch.setattr(steering_worker_state, "_marker_path", lambda: tmp_path / "busy.json")
    liveness = iter([(False, None)])
    monkeypatch.setattr(endpoints, "_is_steering_worker_running", lambda: next(liveness, (True, 4321)))
    monkeypatch.setattr(endpoints, "PID_FILE", str(tmp_path / "steering.pid"))
    monkeypatch.setattr(endpoints, "STEERING_LOG", str(tmp_path / "steering.log"))

    with patch.object(endpoints.subprocess, "Popen") as popen, \
         patch.object(endpoints.asyncio, "sleep", new=AsyncMock()):
        popen.return_value.pid = 98765
        try:
            asyncio.run(getattr(endpoints, entry)())
        finally:
            endpoints._SPAWNED_WORKER_PIDS.discard(98765)

    assert popen.call_count == 1
    argv, kwargs = popen.call_args.args[0], popen.call_args.kwargs
    assert argv[argv.index("-Q") + 1] == "steering"
    env = kwargs["env"]
    assert "CUDA_VISIBLE_DEVICES" not in env, f"the worker is pinned: {env.get('CUDA_VISIBLE_DEVICES')!r}"
    assert env["MISTUDIO_SPAWN_SENTINEL"] == "inherited"


def test_steering_mode_memory_is_summed_over_every_card(monkeypatch):
    """It read the first line — GPU 0 — so exit-mode reported freeing nothing
    after killing a worker that held gigabytes on another card."""
    from src.api.v1.endpoints import steering as endpoints

    monkeypatch.setattr(
        endpoints.subprocess, "run",
        MagicMock(return_value=SimpleNamespace(returncode=0, stdout="1200\n9000\n")),
    )
    assert endpoints._get_gpu_memory_mb() == 10200


# ── MCP: the tools send the card ───────────────────────────────────────────


def _steering_tools():
    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import steering as steering_tools

    steering_tools._inflight.clear()
    mcp = FastMCP("test")
    client = MagicMock()
    client.post = AsyncMock(return_value={"task_id": "t-1"})
    steering_tools.register(mcp, client, MCPSettings(allow_anonymous=True, steering_approval=False))
    return mcp, client, steering_tools


MCP_CALLS = {
    "steer_compare": ("/steering/async/compare", {
        "sae_id": "sae_A", "prompt": "hi",
        "features": [{"feature_idx": 1, "layer": 3, "strength": 5}],
    }),
    "steer_sweep": ("/steering/async/sweep", {
        "sae_id": "sae_A", "prompt": "hi", "feature_idx": 1, "layer": 3, "strength_values": [1, 2],
    }),
    "steer_combined": ("/steering/async/combined", {
        "sae_id": "sae_A", "prompt": "hi",
        "features": [{"feature_idx": 1, "layer": 3, "strength": 5}],
    }),
}


@pytest.mark.parametrize("gpu", [RTX_UUID, None])
@pytest.mark.parametrize("tool", sorted(MCP_CALLS))
def test_the_mcp_steering_tools_send_the_card(tool, gpu):
    mcp, client, steering_tools = _steering_tools()
    path, arguments = MCP_CALLS[tool]
    if gpu is not None:
        arguments = {**arguments, "gpu": gpu}
    try:
        asyncio.run(mcp.call_tool(tool, arguments))
    finally:
        steering_tools._inflight.clear()

    assert client.post.await_count == 1
    assert client.post.await_args.args[0] == path
    assert client.post.await_args.kwargs["json_body"]["gpu"] == (gpu or "auto")


# ── (g) Multi-GPU Phase 2: a model no single card holds runs split ─────────


class _TinySplit(torch.nn.Module):
    """A model whose first registered parameter is on "the other card" (meta)
    and whose embedding is on the input card (cpu) — as on a real split, where
    ``model.device`` (the first parameter's) is not where inputs belong."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 4, device="meta")
        self.embed = torch.nn.Embedding(10, 4, device="cpu")
        self.config = SimpleNamespace(
            max_position_embeddings=64, model_type="llama", architectures=["LlamaForCausalLM"]
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.embed

    def generate(self, **kwargs):
        return torch.tensor([[5, 6, 7, 8]])

    def forward(self, **kwargs):
        return SimpleNamespace(loss=torch.tensor(0.0))


class _Tokenizer:
    """Records the device every encoded batch is moved to."""

    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 64

    def __init__(self):
        self.moved_to = []

    def __call__(self, text, **kwargs):
        moved_to = self.moved_to

        class _Batch(dict):
            def to(self, device):
                moved_to.append(torch.device(device))
                return self

        return _Batch(input_ids=torch.tensor([[5, 6]]), attention_mask=torch.tensor([[1, 1]]))

    def decode(self, ids, skip_special_tokens=True):
        return "text"


def test_a_model_no_single_card_holds_loads_split_over_both_cards(monkeypatch, cuda, loader):
    """The REAL place_job: placed against the 2 GB floor, or without
    allow_shard, this ~26.9 GB model is put on one card or refused."""
    import transformers

    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: BIG_CONFIG)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _free_by_card(11_000, 23_000))
    svc = _service(monkeypatch)

    model, _ = asyncio.run(svc.load_model("big", gpu="auto"))

    assert loader == [{
        "torch_dtype": torch.float16,
        "device_map": svc._model_placements["big"].device_map,
        "max_memory": {1: "21976MiB", 0: "9976MiB"},
        "trust_remote_code": True,
    }]
    assert svc._model_placements["big"].uuids == [RTX_UUID, TI_UUID]
    assert svc._device == CUDA1 and cuda.set_device[-1] == CUDA1
    assert {p.device for p in model.parameters()} == {CUDA0, CUDA1}


def test_a_model_one_card_holds_keeps_the_single_card_load(monkeypatch, cuda, loader):
    """Allowing a split changes nothing for a model that fits: one card, and
    from_pretrained receives exactly the Phase 1 arguments."""
    import transformers

    monkeypatch.setattr(gpu_placement, "list_cards", lambda: CARDS)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: MEDIUM_CONFIG)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _free_by_card(11_000, 23_000))
    svc = _service(monkeypatch)

    asyncio.run(svc.load_model("seven_b", gpu="auto"))

    assert loader == [{"torch_dtype": torch.float16, "device_map": {"": CUDA1}, "trust_remote_code": True}]
    assert svc._model_placements["seven_b"].gpu_columns() == {"gpu_uuid": RTX_UUID, "gpu_uuids": None}


def test_placement_is_asked_for_the_models_size_and_allowed_to_split(monkeypatch, cuda, loader, placer):
    svc = _service(monkeypatch)

    asyncio.run(svc.load_model("m", gpu="auto"))

    assert placer.calls == [("auto", SMALL_NEED_MB)]
    assert placer.allow_shard == [True]


def test_an_unknown_size_is_placed_against_the_floor(monkeypatch, cuda, loader, placer):
    import transformers

    def unreadable(*args, **kwargs):
        raise OSError("no config.json")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", unreadable)
    svc = _service(monkeypatch)

    asyncio.run(svc.load_model("mystery", gpu=TI_UUID))

    assert placer.calls == [(TI_UUID, MIN_FREE_MB_FOR_MODEL_LOAD)]


@pytest.mark.parametrize("stray,named", [("cpu", "lm_head on cpu"), ("disk", "lm_head on disk")])
def test_a_split_that_left_the_cards_is_refused_and_released(
    monkeypatch, cuda, loader, placer, stray, named
):
    placer.split = True
    loader.state.stray = stray
    detached = []
    monkeypatch.setattr(steering_module, "detach_dispatch_hooks", detached.append)
    svc = _service(monkeypatch)

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(svc.load_model("big", gpu="auto"))

    assert named in str(exc.value)
    assert "big" not in svc._loaded_models and "big" not in svc._model_placements
    assert [call["device_map"] for call in loader] == [SPLIT.device_map]
    assert len(detached) == 1, "the refused split was not released"


def test_generation_inputs_go_to_the_embedding_and_every_card_is_synchronized(monkeypatch, cuda):
    from src.schemas.steering import GenerationParams

    monkeypatch.setattr(
        steering_module, "get_generation_watchdog",
        lambda: SimpleNamespace(start_generation=lambda: None, end_generation=lambda: None),
    )
    svc = _service(monkeypatch)
    model, tokenizer = _TinySplit(), _Tokenizer()
    svc._loaded_models["big"] = (model, tokenizer)
    svc._model_placements["big"] = SPLIT
    svc._use_placement(SPLIT)

    asyncio.run(svc._generate_text(model, tokenizer, "hi", GenerationParams(max_new_tokens=4)))

    assert tokenizer.moved_to == [torch.device("cpu")], "inputs did not go to the embedding's device"
    assert cuda.synchronize == [CUDA1, CUDA0, CUDA1, CUDA0]


def test_perplexity_inputs_go_to_the_embedding(monkeypatch, cuda):
    svc = _service(monkeypatch)
    model, tokenizer = _TinySplit(), _Tokenizer()
    svc._use_placement(SPLIT)

    asyncio.run(svc._calculate_perplexity(model, tokenizer, "some text"))

    assert tokenizer.moved_to == [torch.device("cpu")]


def test_each_sae_moves_to_its_hooked_layers_device(monkeypatch, cuda):
    svc = _service(monkeypatch)
    layers = torch.nn.ModuleList([
        torch.nn.Linear(4, 4, device="cpu"),   # layer 0: the card the SAEs loaded on
        torch.nn.Linear(4, 4, device="meta"),  # layer 1: "the other card"
    ])
    monkeypatch.setattr(svc, "_get_target_module", lambda model, layer: layers[layer])

    def sae(layer):
        return steering_module.LoadedSAE(
            model=torch.nn.Linear(4, 8, device="cpu"), config=None,
            layer=layer, d_in=4, d_sae=8, device="cpu",
        )

    near, far = sae(0), sae(1)
    configs = [
        steering_module.FeatureSteeringConfig(feature_idx=1, layer=0, strength=2.0, sae_id="near"),
        steering_module.FeatureSteeringConfig(feature_idx=1, layer=1, strength=2.0, sae_id="far"),
    ]
    handles = svc._register_steering_hooks(object(), {"near": near, "far": far}, configs)
    try:
        assert {p.device.type for p in far.model.parameters()} == {"meta"} and far.device == "meta"
        assert {p.device.type for p in near.model.parameters()} == {"cpu"} and near.device == "cpu"
    finally:
        for handle in handles:
            handle.remove()


def test_a_split_model_is_never_pulled_onto_one_card(monkeypatch, cuda):
    svc = _service(monkeypatch)
    # First parameter on the CPU: the case the single-card check "restores".
    model = FakeModel(CUDA0, devices=[torch.device("cpu"), CUDA0, CUDA1],
                      hf_device_map={"model.embed_tokens": "cpu", "model.layers.0": 0, "lm_head": 1})
    svc._loaded_models["big"] = (model, object())
    svc._model_placements["big"] = SPLIT
    svc._use_placement(SPLIT)

    svc._ensure_model_on_gpu(model)

    assert model.moved_to == []


def test_a_single_card_model_found_on_the_cpu_is_still_restored(monkeypatch, cuda):
    svc = _service(monkeypatch)
    model = FakeModel(CUDA1, devices=[torch.device("cpu")])
    svc._loaded_models["m"] = (model, object())
    svc._model_placements["m"] = Placement(card=RTX, device=CUDA1)
    svc._use_placement(svc._model_placements["m"])

    svc._ensure_model_on_gpu(model)

    assert model.moved_to == [CUDA1]


def test_unloading_a_split_model_detaches_its_hooks_and_never_moves_it(monkeypatch, cuda):
    detached = []
    monkeypatch.setattr(steering_module, "detach_dispatch_hooks", detached.append)
    svc = _service(monkeypatch)
    model = _split_model()
    svc._loaded_models["big"] = (model, object())
    svc._model_placements["big"] = SPLIT
    svc._use_placement(SPLIT)

    assert svc.unload_model("big") is True

    assert model.moved_to == [] and detached == [model]
    assert cuda.synchronize == [CUDA0, CUDA1], "cleanup did not cover both cards of the split"


def test_a_forced_reload_of_a_split_model_detaches_the_old_copy(monkeypatch, cuda, loader, placer):
    placer.split = True
    detached = []
    monkeypatch.setattr(steering_module, "detach_dispatch_hooks", detached.append)
    svc = _service(monkeypatch)
    first, _ = asyncio.run(svc.load_model("big", gpu="auto"))

    asyncio.run(svc.load_model("big", gpu="auto", force_reload=True))

    assert first.moved_to == [] and detached == [first]
    assert [call["device_map"] for call in loader] == [SPLIT.device_map, SPLIT.device_map]


def test_cleanup_covers_every_card_of_a_split(monkeypatch, cuda):
    emptied = []
    monkeypatch.setattr(steering_module, "empty_cache_on", lambda devices: emptied.append(list(devices)))
    svc = _service(monkeypatch)
    svc._model_placements["big"] = SPLIT
    svc._placement = SPLIT

    svc.synchronize()
    svc.cleanup_gpu()

    assert cuda.synchronize == [CUDA0, CUDA1, CUDA0, CUDA1]
    assert emptied == [[CUDA0, CUDA1]]


def test_the_cleanup_task_measures_every_card_of_a_split(monkeypatch, cuda):
    from src.workers import steering_tasks

    svc = _service(monkeypatch)
    svc._loaded_models["big"] = (_split_model(), object())
    svc._model_placements["big"] = SPLIT
    svc._placement = SPLIT
    cuda.allocated_bytes[CUDA1] = 20 * 1024**3
    cuda.allocated_bytes[CUDA0] = 9 * 1024**3

    def unload_all():
        cuda.allocated_bytes[CUDA1] = 1 * 1024**3
        cuda.allocated_bytes[CUDA0] = 1 * 1024**3
        return {"models_unloaded": 1, "saes_unloaded": 0}

    monkeypatch.setattr(svc, "unload_all", unload_all)
    monkeypatch.setattr(steering_module, "get_steering_service", lambda: svc)
    emptied = []
    # The task imports it at call time, so patch it where it is defined.
    monkeypatch.setattr("src.ml.model_devices.empty_cache_on",
                        lambda devices: emptied.append(list(devices)))

    result = steering_tasks.cleanup_steering_gpu.apply().get()

    assert result["memory_freed_gb"] == pytest.approx(27.0)
    assert cuda.memory_allocated == [CUDA0, CUDA1, CUDA0, CUDA1]
    assert emptied == [[CUDA0, CUDA1]], "the cleanup task released only the current device"


def test_vram_is_summed_over_every_card_of_a_split(monkeypatch, cuda):
    calls = _fake_pynvml(
        monkeypatch,
        used_by_uuid={RTX_UUID: 20 * 1024**3, TI_UUID: 7 * 1024**3},
        used_by_index={},
    )
    svc = SteeringService()
    svc._placement = SPLIT

    assert svc._get_system_vram_usage_gb() == pytest.approx(27.0)
    assert calls.by_uuid == [RTX_UUID, TI_UUID]


def test_without_nvml_vram_falls_back_over_every_card_of_a_split(monkeypatch, cuda):
    _fake_pynvml(monkeypatch, used_by_uuid={}, used_by_index={}, init_error=RuntimeError("no driver"))
    cuda.allocated_bytes[CUDA1] = 2 * 1024**3
    cuda.allocated_bytes[CUDA0] = 1 * 1024**3
    svc = SteeringService()
    svc._placement = SPLIT

    assert svc._get_system_vram_usage_gb() == pytest.approx(3.0)
    assert cuda.memory_allocated == [CUDA1, CUDA0]


@pytest.mark.parametrize("kind", sorted(REQUESTS))
def test_the_generation_path_reports_every_card_of_a_split(monkeypatch, cuda, loader, placer, kind):
    placer.split = True
    detached = []
    monkeypatch.setattr(steering_module, "detach_dispatch_hooks", detached.append)
    svc = _service(monkeypatch)

    async def load_sae(*args, **kwargs):
        return _loaded_sae(svc._device)

    async def resolve_sae_map(request, meta_map, force_reload=False):
        return {request.sae_id: _loaded_sae(svc._device)}

    monkeypatch.setattr(svc, "load_sae", load_sae)
    monkeypatch.setattr(svc, "resolve_sae_map", resolve_sae_map)
    monkeypatch.setattr(svc, "_register_steering_hooks", lambda *a, **k: [])
    monkeypatch.setattr(svc, "_generate_text", AsyncMock(return_value=("text", 1, 1)))
    monkeypatch.setattr(
        svc, "_compute_metrics",
        AsyncMock(return_value=steering_module.GenerationMetrics(token_count=1, generation_time_ms=1)),
    )

    method, request_dict = REQUESTS[kind]
    result = getattr(svc, method)(
        request_dict={**request_dict, "gpu": "auto"}, sae_path="/nonexistent/sae", model_id="m",
    )

    assert placer.allow_shard == [True]
    assert [call["device_map"] for call in loader] == [SPLIT.device_map]
    assert result["gpu_uuid"] == RTX_UUID
    assert result["gpu_uuids"] == [RTX_UUID, TI_UUID]


def test_a_named_card_is_not_satisfied_by_a_split_that_starts_on_it(monkeypatch, cuda, loader, placer):
    placer.split = True
    detached = []
    monkeypatch.setattr(steering_module, "detach_dispatch_hooks", detached.append)
    svc = _service(monkeypatch)
    first, _ = asyncio.run(svc.load_model("m", gpu="auto"))
    placer.split = False

    moved, _ = asyncio.run(svc.load_model("m", gpu=RTX_UUID))

    assert moved is not first and detached == [first]
    assert [call["device_map"] for call in loader] == [SPLIT.device_map, {"": CUDA1}]
    assert svc._model_placements["m"].gpu_columns() == {"gpu_uuid": RTX_UUID, "gpu_uuids": None}


def test_all_keeps_a_split_that_already_spans_every_card(monkeypatch, cuda, loader, placer):
    monkeypatch.setattr(steering_module, "list_cards", lambda: CARDS)
    placer.split = True
    svc = _service(monkeypatch)
    first, _ = asyncio.run(svc.load_model("m", gpu="auto"))

    again, _ = asyncio.run(svc.load_model("m", gpu="all"))

    assert again is first
    assert len(loader) == 1 and len(placer.calls) == 1


# ── (h) steering_core and the recorder run a split model ──────────────────


class _Stop(Exception):
    pass


class _RowSession:
    """A sync session whose every query answers one row."""

    def __init__(self, row):
        self.row = row
        self.commits = 0

    def query(self, *args):
        return self

    def filter(self, *args, **kwargs):
        return self

    def populate_existing(self):
        return self

    def first(self):
        return self.row

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


@pytest.mark.parametrize("split", [True, False])
def test_the_core_loader_splits_only_a_split_placement(split):
    from src.services.steering_core import load_model_and_structure

    model_rec = SimpleNamespace(repo_id="org/big", quantization="FP16", file_path=None)
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        raise _Stop

    placement = SPLIT if split else Placement(card=RTX, device=CUDA1)
    with patch("src.ml.model_loader.load_model_from_hf", side_effect=fake_load):
        with pytest.raises(_Stop):
            load_model_and_structure("m_1", _RowSession(model_rec), CUDA1, placement=placement)

    if split:
        assert seen["device_map"] == SPLIT.device_map
        assert seen["max_memory"] == {1: "21976MiB", 0: "9976MiB"}
    else:
        assert seen["device_map"] == CUDA1
        assert "max_memory" not in seen


class _SaeRows:
    """A sync session answering the model row for Model and each SAE row by its id."""

    def __init__(self, model_rec, saes):
        self._model_rec, self._saes, self._model = model_rec, saes, None

    def query(self, model):
        self._model = model
        return self

    def filter(self, criterion):
        self._key = getattr(getattr(criterion, "right", None), "value", None)
        return self

    def first(self):
        from src.models.external_sae import ExternalSAE

        return self._saes.get(self._key) if self._model is ExternalSAE else self._model_rec


def test_the_core_split_load_is_mapped_to_hold_each_layers_sae():
    """Review round 2. Calibration and the recorder resolve each layer's SAE onto that
    layer's card after the load; a split keeps only SHARD_RESERVE_MB free per card, so
    the load is told each layer's SAE size — at fp32, as `_load_sae_sync` builds it.
    Two different widths, so a size charged to the wrong layer disagrees.

    Review round 3: the DECODER's size. The core moves only each decoder to its
    layer's card; charging the encoder too refused OLMo-2-13B with decoders beside
    layers 11-13 on the node's cards, a split that fits."""
    from src.services.steering_core import load_model_and_structure

    model_rec = SimpleNamespace(repo_id="org/big", quantization="FP16", file_path=None)
    saes = {"sae_a": SimpleNamespace(d_model=5_120, n_features=65_536),
            "sae_b": SimpleNamespace(d_model=5_120, n_features=16_384)}
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        raise _Stop

    with patch("src.ml.model_loader.load_model_from_hf", side_effect=fake_load):
        with pytest.raises(_Stop):
            load_model_and_structure("m_1", _SaeRows(model_rec, saes), CUDA1, placement=SPLIT,
                                     sae_ids_by_layer={13: "sae_a", 30: "sae_b"})

    # By hand: 5,120 x 65,536 x 4 B = 1,342,177,280 B; 5,120 x 16,384 x 4 B = 335,544,320 B.
    assert seen["extra_mb_by_layer"] == {13: pytest.approx(1_342_177_280 / 2**20, abs=1e-9),
                                         30: pytest.approx(335_544_320 / 2**20, abs=1e-9)}


def test_each_members_decoder_loads_on_its_layers_device(monkeypatch):
    """Review round 3: every SAE is BUILT on the CPU and only its decoder moves to its
    layer's device. It used to be built on the layer's card, encoder and all, and a
    split had to keep room there for an encoder the core throws away."""
    from src.services import circuit_capture_service
    from src.services import steering_service as ss
    from src.services.steering_core import resolve_feature_members

    loaded = {}
    decoders = {0: torch.randn(4, 8), 1: torch.randn(4, 8)}

    def fake_load_sae(record, device):
        loaded[record.layer] = str(device)
        return record

    monkeypatch.setattr(circuit_capture_service, "_load_sae_sync", fake_load_sae)
    monkeypatch.setattr(ss, "resolve_decoder_weight", lambda sae: decoders[sae.layer])
    rows = iter([SimpleNamespace(layer=0), SimpleNamespace(layer=1)])
    db = SimpleNamespace(query=lambda model: SimpleNamespace(
        filter=lambda *a, **k: SimpleNamespace(first=lambda: next(rows))))
    structure = SimpleNamespace(layers_module=torch.nn.ModuleList([
        torch.nn.Linear(4, 4, device="cpu"), torch.nn.Linear(4, 4, device="meta"),
    ]))
    specs = [
        {"layer": 0, "feature_idx": 1, "strength": 1.0, "sae_id": "sae_near"},
        {"layer": 1, "feature_idx": 2, "strength": 1.0, "sae_id": "sae_far"},
    ]

    # The fallback device is a sentinel, so a layer's device must come from the
    # loaded model's layer, never from the placement's first card.
    _, members = resolve_feature_members(specs, "m_1", db, "first-card", structure=structure)

    assert loaded == {0: "cpu", 1: "cpu"}, "an SAE was built on its layer's card, encoder and all"
    assert [(L, W.device.type, tuple(W.shape)) for (L, _i, _s, W) in members] == [(0, "cpu", (4, 8)),
                                                                                   (1, "meta", (4, 8))]
    assert torch.equal(members[0][3], decoders[0])


def test_the_core_generator_sends_inputs_to_the_embedding():
    from src.services.steering_core import build_steer_generator

    model, tokenizer = _TinySplit(), _Tokenizer()
    structure = SimpleNamespace(layers_module=torch.nn.ModuleList([torch.nn.Linear(4, 4)]))
    _gen_at, baseline_at = build_steer_generator(
        model, tokenizer, structure, [(0, 1, 1.0, torch.zeros(4, 8))],
        disable_cache=False, max_tokens=3,
    )

    baseline_at("hi", 0)

    assert tokenizer.moved_to == [torch.device("cpu")], "inputs went to model.device, not the embedding"


def _recorder_fakes(monkeypatch, seen):
    import src.services.steering_recorder_service as mod
    from src.services.steering_recorder_service import SteeringRecorderService

    def fake_load(model_id, db, device, placement=None, **kwargs):
        seen["load"] = (device, placement)
        if kwargs:
            seen["load_kwargs"] = kwargs
        return ("M", "T", "STRUCT", False, device)

    def fake_resolve(cls, artifact, db, device, structure=None):
        seen["resolve"] = (device, structure)
        return ("m_1", [(1, 1, 1.0, "W")])

    monkeypatch.setattr(mod, "load_model_and_structure", fake_load)
    monkeypatch.setattr(SteeringRecorderService, "_resolve", classmethod(fake_resolve))
    monkeypatch.setattr(SteeringRecorderService, "_artifact_model_id", staticmethod(lambda art, db: "m_1"))
    monkeypatch.setattr(SteeringRecorderService, "_model_hf_id", staticmethod(lambda mid, db: "org/m"))
    monkeypatch.setattr(mod, "build_steer_generator", lambda *a, **k: (lambda d, p: "S", lambda p, s: "B"))
    monkeypatch.setattr(SteeringRecorderService, "_persist", staticmethod(lambda db, art, payload: "vman_s"))
    return SteeringRecorderService


RECORD_CONFIG = {"artifact": {"kind": "circuit", "circuit_id": "crc_1"}, "dials": [0.5], "prompts": ["hi"]}

#: The circuit RECORD_CONFIG names: two SAEs, at layers 4 and 7.
RECORDED_CIRCUIT = SimpleNamespace(id="crc_1", model_id="m_1", saes=[
    {"layer": 4, "mistudio_sae_id": "sae_l4"}, {"layer": 7, "mistudio_sae_id": "sae_l7"}])


def test_a_split_recording_hands_the_placement_to_the_loader_and_the_layers_to_the_resolver(monkeypatch):
    seen = {}
    recorder = _recorder_fakes(monkeypatch, seen)

    recorder.record_samples(_RowSession(RECORDED_CIRCUIT), dict(RECORD_CONFIG), device=CUDA1,
                            gpu={"uuid": RTX_UUID}, placement=SPLIT)

    # The split is loaded to hold, beside each layer, the SAE its resolver puts there.
    assert seen == {"load": (CUDA1, SPLIT), "resolve": (CUDA1, "STRUCT"),
                    "load_kwargs": {"sae_ids_by_layer": {4: "sae_l4", 7: "sae_l7"}}}


def test_a_single_card_recording_keeps_its_exact_load_call(monkeypatch):
    seen = {}
    recorder = _recorder_fakes(monkeypatch, seen)

    recorder.record_samples(_RowSession(RECORDED_CIRCUIT), dict(RECORD_CONFIG), device=CUDA1,
                            gpu={"uuid": RTX_UUID}, placement=Placement(card=RTX, device=CUDA1))

    assert seen == {"load": (CUDA1, None), "resolve": (CUDA1, None)}


def test_the_recorder_resolver_forwards_the_layer_structure(monkeypatch):
    import src.services.steering_recorder_service as mod
    from src.services.steering_recorder_service import SteeringRecorderService

    calls = []
    monkeypatch.setattr(mod, "resolve_feature_members",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or ("m", []))
    artifact = {"kind": "features", "model_id": "m", "features": [{"layer": 1}]}

    SteeringRecorderService._resolve(artifact, None, CUDA1, structure="STRUCT")
    SteeringRecorderService._resolve(artifact, None, CUDA1)

    assert calls[0][1] == {"structure": "STRUCT"}
    assert calls[1] == (([{"layer": 1}], "m", None, CUDA1), {})


def test_the_recorder_task_records_every_card_of_a_split(monkeypatch):
    from src.services.steering_recorder_service import SteeringRecorderService
    from src.workers import circuit_record_tasks as tasks

    row = SimpleNamespace(id="srr_1", status="pending", gpu_request="all", gpu_uuid=None,
                          gpu_uuids=None, error=None, manifest_ref=None)
    session = _RowSession(row)
    asked = []

    def fake_place_job(requested="auto", required_mb=None, cards=None, allow_shard=False):
        asked.append({"requested": requested, "allow_shard": allow_shard})
        return SPLIT

    monkeypatch.setattr("src.workers.circuit_gpu.place_job", fake_place_job)
    seen = {}

    def fake_record(db, config, *, device, gpu=None, placement=None, **kwargs):
        seen.update(device=device, gpu=gpu, placement=placement, uuids_at_load=row.gpu_uuids)
        return {"manifest_ref": "vman_s"}

    fake_self = SimpleNamespace(get_db=lambda: contextlib.nullcontext(session))
    with patch.object(SteeringRecorderService, "record_samples", side_effect=fake_record), \
         patch.object(tasks, "emit_circuit_run_completed"):
        tasks.run_circuit_record.run.__func__(fake_self, "srr_1", {})

    assert asked == [{"requested": "all", "allow_shard": True}]
    assert row.gpu_uuid == RTX_UUID
    assert row.gpu_uuids == [RTX_UUID, TI_UUID]
    assert seen["uuids_at_load"] == [RTX_UUID, TI_UUID], "the cards were not recorded before the load"
    assert seen["placement"] is SPLIT and seen["device"] == CUDA1
    assert seen["gpu"]["uuids"] == [RTX_UUID, TI_UUID] and seen["gpu"]["uuid"] == RTX_UUID
    assert row.status == "completed"


@pytest.mark.parametrize("kind,task_name,method", [
    ("compare", "steering_compare_task", "generate_comparison_sync"),
    ("sweep", "steering_sweep_task", "generate_strength_sweep_sync"),
    ("combined", "steering_combined_task", "generate_combined_sync"),
])
def test_a_steering_task_releases_cache_on_every_card_of_a_split(monkeypatch, cuda, kind, task_name, method):
    """At task start and in its `finally`: a bare empty_cache() released only
    the current device, the first card of a split model."""
    from src.workers import steering_tasks

    svc = _service(monkeypatch)
    svc._model_placements["big"] = SPLIT
    svc._placement = SPLIT
    emptied = []
    monkeypatch.setattr(steering_module, "empty_cache_on", lambda devices: emptied.append(list(devices)))
    monkeypatch.setattr(svc, method, lambda **kwargs: {"ok": True})
    monkeypatch.setattr(steering_module, "get_steering_service", lambda: svc)
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=SimpleNamespace(returncode=1, stdout="")))
    monkeypatch.setattr(steering_tasks, "emit_steering_progress", MagicMock(return_value=True))

    result = getattr(steering_tasks, task_name).apply(
        kwargs={**TASK_KWARGS, "request_dict": REQUESTS[kind][1]})

    assert result.successful(), result.result
    assert emptied == [[CUDA0, CUDA1], [CUDA0, CUDA1]]


# ── (i) Review round 1: budgets read at load time; unload releases its own cards ──


def test_a_split_is_budgeted_from_the_memory_free_when_its_weights_load(monkeypatch, cuda, loader, placer):
    """Every steering request force-reloads its model on the placement the cache
    kept, so a budget read at the FIRST placement was reused for every later
    load. Here another process takes 6 GB of the 3080 Ti between two requests:
    the stale 9,976 MB budget lets accelerate fill that card past what it has,
    and the load dies part-way instead of being mapped honestly (or refused)."""
    placer.split = True
    monkeypatch.setattr(torch.cuda, "mem_get_info", _free_by_card(11_000, 23_000))
    svc = _service(monkeypatch)
    asyncio.run(svc.load_model("big", gpu="auto"))

    monkeypatch.setattr(torch.cuda, "mem_get_info", _free_by_card(5_000, 23_000))
    asyncio.run(svc.load_model("big", gpu="auto", force_reload=True))

    assert [call["max_memory"] for call in loader] == [
        {1: "21976MiB", 0: "9976MiB"},
        {1: "21976MiB", 0: "3976MiB"},
    ]
    kept = svc._model_placements["big"]
    assert kept.max_memory_mb == {1: 21_976, 0: 3_976}
    assert kept.uuids == [RTX_UUID, TI_UUID], "the budget refresh must keep the split's cards"
    assert placer.calls == [("auto", SMALL_NEED_MB)], "a reload under Auto must not re-place"


def test_the_first_split_load_budgets_the_memory_the_saes_already_took(monkeypatch, cuda, loader, placer):
    """SAEs load onto the placement's first card AFTER placement read the cards
    and BEFORE the model loads. The budget handed to accelerate must be read
    after them, or that card is budgeted memory the SAEs already hold."""
    placer.split = True
    # The placement saw 23,000 MB free on the 3090; the SAEs then took 1,500 MB.
    monkeypatch.setattr(torch.cuda, "mem_get_info", _free_by_card(11_000, 21_500))
    svc = _service(monkeypatch)

    asyncio.run(svc.load_model("big", gpu="auto"))

    assert [call["max_memory"] for call in loader] == [{1: "20476MiB", 0: "9976MiB"}]


def test_unloading_a_model_releases_its_cards_while_another_placement_is_current(monkeypatch, cuda):
    """`unload_model` pops the model's placement before cleanup, and cleanup reads
    the cards from the placements that REMAIN plus the current one. When the last
    request used another model, the unloaded model's cards were never synchronized
    or emptied, so their cached blocks stayed reserved and NVML — which placement
    reads — kept counting them."""
    emptied = []
    monkeypatch.setattr(steering_module, "empty_cache_on", lambda devices: emptied.append(list(devices)))
    monkeypatch.setattr(steering_module, "detach_dispatch_hooks", lambda model: None)
    svc = _service(monkeypatch)
    one_card = Placement(card=RTX, device=CUDA1)
    svc._loaded_models["big"] = (_split_model(), object())
    svc._model_placements["big"] = SPLIT
    svc._loaded_models["small"] = (FakeModel(CUDA1), object())
    svc._model_placements["small"] = one_card
    svc._use_placement(one_card)

    assert svc.unload_model("big") is True

    assert CUDA0 in cuda.synchronize, "the unloaded split's other card was never synchronized"
    assert emptied and CUDA0 in emptied[-1], "the unloaded split's other card was never emptied"


# ── (k) Multi-GPU Phase 3 composed with review round 1: the load-time budget stays on the leased cards ──

#: A third card, with the most free memory on the node, leased by ANOTHER job.
#: On a two-card node a split leases every card, so "inside the leased cards"
#: would hold by construction; this card is what can tell the two apart.
X_UUID = "GPU-0a1b2c3d-0000-4000-8000-000000000003"
X_CARD = GpuCard(index=2, uuid=X_UUID, name="NVIDIA RTX A6000", total_mb=48_000, free_mb=40_000)
OTHER_HOLDER = "training:someone-else:00000000"


def _free_on_three_cards(device=None):
    index = device.index if isinstance(device, torch.device) else device
    return {0: 11_000, 1: 23_000, 2: 40_000}[index] * 1024**2, [12_288, 24_576, 48_000][index] * 1024**2


@pytest.fixture
def leased_node(monkeypatch, cuda):
    """Per-card mode on a three-card node, real Postgres leases, the other job holding the A6000."""
    import transformers

    from src.core.config import settings
    from src.services import gpu_leases
    from tests.unit.gpu_lease_db import clear, lease_engine, session_factory

    engine = lease_engine("mistudio_test_steering_leases")
    clear(engine)
    db = session_factory(engine)
    monkeypatch.setattr(settings, "gpu_worker_mode", "per_card")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _free_on_three_cards)
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: [TI, RTX, X_CARD])
    monkeypatch.setattr(gpu_placement, "torch_device", lambda card: torch.device("cuda", card.index))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: BIG_CONFIG)
    with db() as s:
        assert gpu_leases.acquire(s, [X_UUID], OTHER_HOLDER, task_id="other-task")

    def live():
        with db() as s:
            return gpu_leases.live_leases(s)

    budgets = []
    original = SteeringService.__dict__["_budget_split_from_free_memory"].__func__

    def recording(placement):
        budgeted = original(placement)
        budgets.append({"cards": sorted(budgeted.max_memory_mb), "leases": live()})
        return budgeted

    monkeypatch.setattr(SteeringService, "_budget_split_from_free_memory", staticmethod(recording))
    yield SimpleNamespace(db=db, live=live, budgets=budgets)
    engine.dispose()


def _steering_claim(node, task_id, *, wait_timeout_s=60.0):
    from src.services import gpu_job_claim

    return gpu_job_claim.ClaimContext(
        holder=gpu_job_claim.make_holder("steering", task_id), task_id=task_id,
        worker_uuid=RTX_UUID, handoff=False, wait_timeout_s=wait_timeout_s,
        session=node.db, inventory=lambda: [TI, RTX, X_CARD],
        available_mb=lambda: 1_000_000.0, sleep=lambda seconds: None,
    )


def test_a_split_is_budgeted_at_load_only_on_the_cards_its_claim_leased(monkeypatch, loader, leased_node):
    """Review round 1 re-reads a split's budget from free memory as its weights load;
    Phase 3 leases the cards. The re-read must stay on the leased cards: the A6000
    has the most free memory on the node, and another job holds it."""
    from src.services import gpu_job_claim

    svc = _service(monkeypatch)
    claim = _steering_claim(leased_node, "tid-1")

    with gpu_job_claim.claiming(claim):
        asyncio.run(svc.load_model("big", gpu="auto"))

    assert [call["max_memory"] for call in loader] == [{1: "21976MiB", 0: "9976MiB"}]
    assert leased_node.budgets == [{
        "cards": [0, 1],
        "leases": {RTX_UUID: claim.holder, TI_UUID: claim.holder, X_UUID: OTHER_HOLDER},
    }], "the load-time budget read a card the job had not leased, or read it after its lease went"
    assert svc._model_placements["big"].uuids == [RTX_UUID, TI_UUID]
    assert leased_node.live() == {X_UUID: OTHER_HOLDER}


def _load_under_a_claim(node, svc, task_id):
    from src.services import gpu_job_claim

    claim = _steering_claim(node, task_id)
    with gpu_job_claim.claiming(claim):
        asyncio.run(svc.load_model("big", gpu="auto"))
    return claim


def test_a_kept_split_is_leased_again_before_the_next_request_reloads_it(monkeypatch, loader, leased_node):
    """Every generation path force-reloads its model on the placement the cache KEPT,
    and round 1 budgets that reload from each card's free memory. A worker that
    outlives its task keeps the placement while the next request's claim holds
    nothing: the kept cards must be leased by that request before it reloads."""
    from src.services import gpu_job_claim

    svc = _service(monkeypatch)
    _load_under_a_claim(leased_node, svc, "tid-1")
    assert leased_node.live() == {X_UUID: OTHER_HOLDER}

    claim = _steering_claim(leased_node, "tid-2")
    with gpu_job_claim.claiming(claim):
        placement = svc.place_model("big", "auto")
        held = leased_node.live()
        asyncio.run(svc.load_model("big", force_reload=True, placement=placement))

    both = {RTX_UUID: claim.holder, TI_UUID: claim.holder, X_UUID: OTHER_HOLDER}
    assert held == both, "the kept model's cards were not leased by the request that reloads them"
    assert len(loader) == 2
    assert leased_node.budgets[-1] == {"cards": [0, 1], "leases": both}
    assert leased_node.live() == {X_UUID: OTHER_HOLDER}


def test_a_kept_split_is_never_reloaded_onto_a_card_another_job_now_holds(monkeypatch, loader, leased_node):
    from src.services import gpu_job_claim, gpu_leases

    svc = _service(monkeypatch)
    _load_under_a_claim(leased_node, svc, "tid-1")
    another = "circuit_capture:another-job:11111111"
    with leased_node.db() as s:
        assert gpu_leases.acquire(s, [TI_UUID], another, task_id="another-task")

    claim = _steering_claim(leased_node, "tid-2", wait_timeout_s=0.0)
    with gpu_job_claim.claiming(claim):
        with pytest.raises(GpuPlacementError, match="Waited"):
            svc.place_model("big", "auto")

    assert len(loader) == 1, "the kept model was reloaded onto a card another job holds"
    assert leased_node.live() == {X_UUID: OTHER_HOLDER, TI_UUID: another}
    # Review round 1: Auto released the kept copy rather than keep it on the card
    # the other job holds; nothing else fits the model here, so the request waits.
    assert "big" not in svc._loaded_models


def test_an_auto_request_releases_a_kept_split_on_a_busy_card_and_places_anew(monkeypatch, loader, leased_node):
    """Review round 1, item 3. The kept split's 3080 Ti is now another job's, and the
    A6000 is idle with room for the whole model. Waiting for the 3080 Ti could only
    end in a refusal after the steering wait: the kept weights are the very memory
    the other job needs. Auto never asked for that card, so the copy is released and
    the request placed again."""
    from src.services import gpu_job_claim, gpu_leases

    svc = _service(monkeypatch)
    _load_under_a_claim(leased_node, svc, "tid-1")
    another = "circuit_capture:another-job:11111111"
    with leased_node.db() as s:
        assert gpu_leases.acquire(s, [TI_UUID], another, task_id="another-task")
        gpu_leases.release(s, OTHER_HOLDER)

    claim = _steering_claim(leased_node, "tid-2", wait_timeout_s=0.0)
    with gpu_job_claim.claiming(claim):
        placement = svc.place_model("big", "auto")
        held = leased_node.live()

    assert "big" not in svc._loaded_models, "the kept copy stayed on the card another job holds"
    assert placement.uuids == [X_UUID]
    assert held == {X_UUID: claim.holder, TI_UUID: another}


def test_a_named_card_still_waits_for_its_kept_model_and_refuses_with_the_reason(monkeypatch, loader, leased_node):
    """The other half of item 3: a request that NAMES the kept model's card is honoured
    or refused, never moved — it waits (bounded) and then says which card was held."""
    from src.services import gpu_job_claim, gpu_leases

    svc = _service(monkeypatch)
    svc._loaded_models["kept"] = (FakeModel(CUDA0), object())
    svc._model_placements["kept"] = Placement(card=TI, device=CUDA0)
    another = "circuit_capture:another-job:11111111"
    with leased_node.db() as s:
        assert gpu_leases.acquire(s, [TI_UUID], another, task_id="another-task")

    claim = _steering_claim(leased_node, "tid-2", wait_timeout_s=0.0)
    with gpu_job_claim.claiming(claim):
        with pytest.raises(GpuPlacementError, match=r"Waited .*held by another job"):
            svc.place_model("kept", TI_UUID)

    assert "kept" in svc._loaded_models, "a request naming the card released the model instead of waiting"
    assert leased_node.live() == {X_UUID: OTHER_HOLDER, TI_UUID: another}


def test_an_auto_request_releases_a_kept_single_card_model_on_a_busy_card(monkeypatch, loader, leased_node):
    from src.services import gpu_job_claim, gpu_leases

    svc = _service(monkeypatch)
    svc._loaded_models["kept"] = (FakeModel(CUDA0), object())
    svc._model_placements["kept"] = Placement(card=TI, device=CUDA0)
    with leased_node.db() as s:
        assert gpu_leases.acquire(s, [TI_UUID], "circuit_capture:another-job:11111111", task_id="another-task")
        gpu_leases.release(s, OTHER_HOLDER)

    claim = _steering_claim(leased_node, "tid-2", wait_timeout_s=0.0)
    with gpu_job_claim.claiming(claim):
        placement = svc.place_model("kept", "auto")

    assert "kept" not in svc._loaded_models
    assert placement.uuids == [X_UUID]


# ── (j) Review round 1: load-bearing lines no test caught (mutation probes) ──


def test_all_is_not_satisfied_by_a_split_that_misses_a_visible_card(monkeypatch, cuda, loader, placer):
    """P1. "all" means every card. With a third card visible, a split over two of
    them is NOT what was asked, so the model is placed again — a two-card fixture
    agrees with "any split satisfies all" by construction."""
    third = GpuCard(index=2, uuid="GPU-3a3a3a3a-3a3a-3a3a-3a3a-3a3a3a3a3a3a",
                    name="NVIDIA GeForce RTX 3060", total_mb=12_288, free_mb=12_000)
    monkeypatch.setattr(steering_module, "list_cards", lambda: [TI, RTX, third])
    placed = []
    monkeypatch.setattr(
        steering_module, "place_job",
        lambda requested="auto", required_mb=None, cards=None, allow_shard=False:
            placed.append(requested) or SPLIT,
    )
    svc = _service(monkeypatch)
    svc._loaded_models["big"] = (_split_model(), object())
    svc._model_placements["big"] = SPLIT
    unloaded = []
    monkeypatch.setattr(svc, "unload_model", lambda model_id: unloaded.append(model_id) or True)

    assert svc._names_card("all", SPLIT) is False
    svc.place_model("big", "all")

    assert unloaded == ["big"], "a split missing a visible card was kept for 'all'"
    assert placed == ["all"]


def test_a_parameter_off_the_placements_cards_is_refused_even_when_the_map_says_gpu(
    monkeypatch, cuda, loader, placer
):
    """P2. `hf_device_map` names where accelerate MEANT modules to go; a tensor
    can still sit elsewhere (a tied or late-registered parameter). The refusal
    reads the parameters too, not only the map."""
    placer.split = True
    stray = FakeModel(CUDA0, devices=[CUDA0, CUDA1, torch.device("cpu")],
                      hf_device_map={"model.embed_tokens": 0, "model.layers.0": 0, "lm_head": 1})
    stray.named_parameters = lambda: iter([
        ("model.embed_tokens.weight", SimpleNamespace(device=CUDA0)),
        ("model.layers.0.weight", SimpleNamespace(device=CUDA1)),
        ("lm_head.rotary_cache", SimpleNamespace(device=torch.device("cpu"))),
    ])
    monkeypatch.setattr(steering_module.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: stray)
    monkeypatch.setattr(steering_module, "detach_dispatch_hooks", lambda model: None)
    svc = _service(monkeypatch)

    with pytest.raises(RuntimeError) as refused:
        asyncio.run(svc.load_model("big", gpu="auto"))

    assert "lm_head.rotary_cache on cpu" in str(refused.value)
    assert "big" not in svc._loaded_models


def test_a_refused_split_empties_every_card_it_was_given(monkeypatch, cuda, loader, placer):
    """P28. The refused copy's weights sat on both cards; emptying only the first
    left the other card's blocks reserved."""
    placer.split = True
    loader.state.stray = "disk"
    emptied = []
    monkeypatch.setattr(steering_module, "empty_cache_on", lambda devices: emptied.append(list(devices)))
    monkeypatch.setattr(steering_module, "detach_dispatch_hooks", lambda model: None)
    svc = _service(monkeypatch)

    with pytest.raises(RuntimeError):
        asyncio.run(svc.load_model("big", gpu="auto"))

    assert emptied == [[CUDA1, CUDA0]], "the refused split was released from its first card only"


def test_a_dispatched_model_with_no_recorded_placement_is_never_moved_to_the_cpu(monkeypatch, cuda):
    """P3. `_is_split` falls back to the model's own device map when no placement
    is known, so a dispatched model is released where it is rather than copied
    whole into host memory against its dispatch hooks."""
    detached = []
    monkeypatch.setattr(steering_module, "detach_dispatch_hooks", detached.append)
    svc = _service(monkeypatch)
    model = _split_model()

    svc._release_model(model, None)

    assert model.moved_to == [] and detached == [model]
