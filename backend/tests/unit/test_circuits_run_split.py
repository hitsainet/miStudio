"""Circuit runs on a model split across GPUs: which may split, and that they run right when they do.

Multi-GPU Phase 2 (0xcc/plans/Multi-GPU-Plan.md). A model that fits no single
card loads with the placement's accelerate device map over several. The circuit
services were written for one card — every SAE on the placed device, input ids
on `model.device`, a decoder column converted to the hidden state's dtype but not
its device — so on a split each was a device-mismatch error at the first layer
on another card.

HOW A SECOND CARD IS FAKED WITHOUT ONE. `meta` stands in for "another card": a
meta tensor meeting a non-scalar CPU tensor in an elementwise op raises the same
"not on the expected device" error CUDA raises, and accelerate's real
`AlignDevicesHook` carries hidden states onto meta layers. Two fixtures, so that
neither agrees with a defect by construction:

  * `embedding_not_first` — every layer computes on the CPU, but the FIRST
    registered parameter is on meta (as a multimodal checkpoint's tower on
    another card would be), so `model.device` is meta while the embedding is on
    the CPU. Numbers are real, and each run is compared with an unsplit twin.
  * `later_layers_elsewhere` — layers 2-3, the final norm and the head on meta.
    Nothing there can be read back, so it proves WHERE things go (each SAE on
    its layer's device), not values.

Meta cannot carry data back to the CPU, so a move from an SAE's card BACK to a
layer's card cannot be exercised here; the attribution hook therefore does not
move at all and relies on the SAE sitting on its layer's card, which is pinned.
`_OneDevicePerOp` makes any op mixing devices raise, including `F.linear`, which
meta alone lets through.

MUTATION CONTROLS (2026-09-14; each applied alone, this module run red,
restored byte-identically and checked by sha256; 34 of 34 killed):
  M1  attribution SAE loaded onto the job's device  -> TestEachSaeSitsOnItsLayersCard::test_attribution
  M2  validation SAE onto the job's device          -> TestEachSaeSitsOnItsLayersCard::test_validation
  M3  faithfulness SAE onto the job's device        -> TestEachSaeSitsOnItsLayersCard::test_faithfulness
  M4  capture SAE onto the job's device             -> TestEachSaeSitsOnItsLayersCard::test_capture
  M5  suppress_directional: d_i dtype-only `.to`    -> TestSuppressionMeetsTheHiddenStateOnItsCard::test_directional
  M6  suppress_feature_list: d_i dtype-only `.to`   -> ...::test_feature_list
  M7  suppress_directional: delta not moved         -> ...::test_directional
  M8  suppress_feature_list: delta not moved        -> ...::test_feature_list
  M9  attribution inputs to model.device            -> TestInputsGoToTheEmbeddingsCard::test_attribution_scores_and_every_cards_peak
  M10 validation edge inputs to model.device        -> ...::test_validation_edge
  M11 validation null inputs to model.device        -> ...::test_validation_null
  M12 faithfulness inputs to model.device           -> ...::test_faithfulness_behavior
  M13 capture probe inputs to model.device          -> ...::test_the_capture_probe
  M14 capture batch inputs to model.device          -> ...::test_the_full_capture_hands_the_batch_the_embeddings_card
  M15-M18 capture/attribution/validation/faithfulness task drops allow_shard
                                                    -> TestSplitSafeRunsOptIn (its own case each)
  M19 capture row records gpu_uuid only             -> ...::test_capture_splits_and_records_every_card_on_its_row
  M20 card record drops a split's uuids             -> TestTheCardRecord + the three pass cases
  M21 split loaded onto its first card (no map/budget) -> TestTheLoadTakesThePlacement (all four)
  M22 attribution peak stats on the first card only -> ...::test_attribution_scores_and_every_cards_peak
  M23a-d a service skips release_model before cleanup -> TestEachSaeSitsOnItsLayersCard (its own case each)
  M24 capture task places without a size            -> ...::test_capture_splits_and_records_every_card_on_its_row
  M25/M26 validation/faithfulness encoder reads on the hidden state's device
                                                    -> TestEncodersReadOnTheSaesCard encoder cases
  M27/M28 validation/faithfulness downstream encode off the SAE's device
                                                    -> TestEncodersReadOnTheSaesCard downstream cases
  M29/M30 capture probe/batch activations to the job's/inputs' device
                                                    -> TestTheCaptureEncodesOnEachLayersSae
  M31 size ignores the SAEs                         -> TestTheSize + every opt-in case
  M32 RETIRED: it pinned calibration to one card, and calibration now splits
  M33 discovery run's capture manifest not followed -> TestTheSize + the three pass cases

CALIBRATION AND RECORDER SPLIT (2026-09-14; run over this module plus
test_steering_gpu_placement, test_circuits_run_on_the_chosen_gpu,
test_steering_recorder and test_calibration_service; 16 of 16 killed):
  M34/M36 calibration run/reproduce task drops allow_shard -> ...::test_calibration_splits_sized_...
                                                    / ...::test_calibration_reproduce_splits_sized_...
  M35 calibration task places without a size        -> ...::test_calibration_splits_sized_from_the_saes_it_steers_with
  M37/M38 run/reproduce task withholds the placement -> the same two cases
  M48 task hands a single card its placement        -> ...::test_a_calibration_on_one_card_keeps_its_exact_call
  M39/M40 service run/reproduce withholds it        -> TestCalibrationGeneratesOnTheSplit::test_run_and_reproduce_...
  M41 split resolves without the loaded structure   -> ...::test_the_generation_loads_split_and_resolves_...
  M42 split loads without the placement             -> the same case
  M47 service hands a single card a placement kwarg -> test_circuits_run_on_the_chosen_gpu (fake without it)
  M43 recorder task drops allow_shard               -> the recorder cases here + test_steering_gpu_placement
  M49 recorder task places without a size           -> ...::test_the_recorder_splits_sized_from_its_artifact (all three)
  M44/M50 recorder sizes a cluster/circuit with no SAEs -> its own artifact case
  M45 steering size ignores the SAEs                -> both calibration cases + all three recorder cases

REVIEW ROUND 1 (2026-09-14; each applied alone against this module, run red,
restored byte-identically and checked by sha256; 4 of 4 killed). Two gaps, both
fixtures agreeing with a defect by construction:
  * every fixture was FP16, whose 2 bytes a parameter is also the size lookup's
    fallback, so sizing that ignored the quantization passed the whole suite
    (256 green across the circuit, steering, recorder and calibration modules);
  * every circuit's members covered every captured layer, so faithfulness —
    sized from the capture, loading only its circuit's layers — agreed with the
    right size. Fixed: `circuit_faithfulness_manifest`, from the pass's own
    `expand_circuit_members`.
  R1  weights sized at 2 bytes/param whatever the quantization
                                                    -> TestTheSize::test_the_weights_are_sized_at_the_models_own_quantization (all three)
  R2  faithfulness task sized from the capture manifest again
                                                    -> TestTheSize::test_faithfulness_is_sized_from_the_layers_its_circuit_sits_on
  R3  the faithfulness size keeps every captured layer
                                                    -> ...::test_faithfulness_is_sized_from_... + ...::test_the_size_counts_the_layers_the_pass_loads_an_sae_for
  R4  the pass expands its members inline, not through expand_circuit_members
                                                    -> TestTheSize::test_the_pass_expands_its_members_through_that_one_definition

REVIEW ROUND 1, STEERING REVIEWER (2026-09-14; each applied alone, this module run red, restored
byte-identically and checked by sha256, `git status` clean after):
  Calibration (run, reproduce) and the recorder returned without releasing their
  model — TestASteeringJobReturnsItsCards:
  R1-N6  release_circuit_job does not clear the traceback's frames -> the three [failed-*] cases
  R1-N7  cleanup given the first card only                -> all six split cases
  R1-N8  calibration task's finally drops the release     -> [succeeded-calibration], [failed-calibration]
  R1-N9  reproduce task's finally drops the release       -> [succeeded-reproduce], [failed-reproduce]
  R1-N10 recorder task's finally drops the release        -> both record cases + test_a_single_card_job_cleans_its_one_card
  R1-N12 release_circuit_job releases nothing             -> all seven cases
  A probe that SURVIVED 503 circuit/calibration/recorder tests before this round,
  because every fixture was an FP16 row (2 bytes a parameter = the fallback):
  R1-P20a size ignores the row's precision (always 2.0)   -> TestTheSize::test_a_quantized_or_full_precision_row_...[*]
  R1-P20b precision read as str(QuantizationFormat.Q4), which is 'QuantizationFormat.Q4'
                                                          -> the same three cases

REVIEW ROUND 2, PLACEMENT + LOADER (2026-09-14). A Q4 row's params_count is the packed
count (aaa233de fixed it for training and SAE extraction only); circuits sized a Q4
model at about half its weights. Controls, each alone, restored by sha256:
  Q6  circuit_required_mb passes no architecture_config -> TestTheSize::test_a_four_bit_row_...
  Q11 circuit_required_mb reads params_count directly    -> TestTheSize::test_a_four_bit_row_...
Both red; round 1's Q1 (no format packed) re-run -> the same test, red.

REVIEW ROUND 2, STEERING REVIEWER (2026-09-14): EACH LAYER'S SAE IS MAPPED FOR.
Every pass here puts each layer's SAE on that layer's card, and a split keeps only
SHARD_RESERVE_MB free per card — less than a 12B-class SAE (ml/split_load.py). The
load is now told each layer's SAE size (`extra_mb_by_layer`). The fixtures use three
captured layers and a circuit on two of them, so an allowance read from the capture
disagrees with faithfulness's. Controls (scratchpad p2-r2-steer/mutate.py with
mutations_a.json; each alone, restored byte-identically by sha256; all killed):
  A7  model_load_kwargs drops the sized allowance       -> TestTheLoadTakesThePlacement::test_a_split_is_mapped_to_hold_each_layers_sae_beside_it[*] (4)
  A8  SAEs sized at 2 bytes, not _load_sae_sync's fp32  -> the same four + test_steering_gpu_placement core case
  A9  capture passes no SAE ids                         -> ...[capture]
  A10 attribution passes no SAE ids                     -> ...[attribution]
  A11 validation passes no SAE ids                      -> ...[validation]
  A12 faithfulness passes no SAE ids                    -> ...[faithfulness]
  A13 faithfulness allowance read from the capture      -> ...[faithfulness]
  A15 calibration's split load gets no SAE ids          -> TestCalibrationGeneratesOnTheSplit::test_the_generation_loads_split_...
  A17 circuit_sae_ids_by_layer keeps one SAE            -> the same case
(A14 and A16 — the core loader and the recorder — are in test_steering_gpu_placement.py.)

REVIEW ROUND 3 (2026-09-14): WHAT EACH PASS HOLDS AND WORKS WITH. Round 2 charged every layer's
whole fp32 SAE and nothing for an encode's codes (capture: 8 x 512 tokens, 5 GiB beside a
65,536-feature SAE's layer against a 1 GiB reserve) or attribution's codes kept for the
backward; and the steering jobs (calibration, the recorder) keep only each decoder. Controls
(scratchpad p2-r3/mutate.py + mutations_r3.json, each alone in a private copy, restored by sha256):
  N9   steering manifests lose sae_matrices=1                 -> both calibration cases + the three recorder cases
  N9b  circuit_required_mb ignores sae_matrices                -> the same five
  N10  the placement size drops the encode working memory     -> capture, validation, faithfulness sizes (4)
  N11  the placement size drops attribution's retained codes  -> ...[attribution]
  N11b the working memory summed over layers, not the largest -> the same four as N10
  N12  capture task passes no encode_tokens                   -> test_capture_splits_and_records_every_card_on_its_row
  N13  attribution task passes no backward_tokens             -> ...[attribution]
  N14  validation task passes no encode_tokens                -> ...[validation]
  N15  faithfulness task passes no encode_tokens              -> ...[faithfulness] + TestTheSize faithfulness case
  N16  capture service loads with no encode_tokens            -> test_a_split_keeps_room_for_the_codes_each_layer_encodes[capture]
  N17  attribution service loads with no backward_tokens      -> test_a_split_is_mapped_to_hold_each_layers_sae_beside_it[attribution]
  N18  intervention service loads with no encode_tokens       -> test_a_split_keeps_room_for_the_codes...[validation]
  N19  faithfulness service loads with no encode_tokens       -> ...[faithfulness]
  N20  model_load_kwargs drops working_mb_by_layer            -> the three encode cases
  N20b capture's encode sized for one document, not a batch   -> the capture size + the capture encode case
  N22b sae_mb_by_layer drops the backward's retained codes     -> ...[attribution]
All killed. Round 2's A7, A8 (re-expressed on the new signature), A9-A13 (re-expressed for the
calls that now carry encode/backward tokens), A15, A17 and round 2's Q6, Q11 re-run: all killed.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode

from src.ml.model_devices import input_device, module_device
from src.services.gpu_placement import GpuCard, Placement

TI_UUID = "GPU-f47ba814-49a2-603f-3595-275284140251"
RTX_UUID = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
TI = GpuCard(index=0, uuid=TI_UUID, name="NVIDIA GeForce RTX 3080 Ti", total_mb=12_288, free_mb=11_000)
RTX = GpuCard(index=1, uuid=RTX_UUID, name="NVIDIA GeForce RTX 3090", total_mb=24_576, free_mb=23_000)
CUDA1, CUDA0 = torch.device("cuda", 1), torch.device("cuda", 0)
CPU, META = torch.device("cpu"), torch.device("meta")

#: A split over both cards, 3090 first — constructible without CUDA.
SPLIT = Placement(card=RTX, device=CUDA1, cards=(RTX, TI), devices=(CUDA1, CUDA0),
                  max_memory_mb={1: 21_976, 0: 9_976})
ONE_CARD = Placement(card=RTX, device=CUDA1)

D_MODEL, D_SAE = 32, 48


class _Stop(Exception):
    """Raised by a fake once it has recorded what it was handed."""


# ── strict devices ───────────────────────────────────────────────────────


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _tensors(item)


class _OneDevicePerOp(TorchFunctionMode):
    """Any op whose non-scalar tensor arguments span devices raises, as it would across CUDA cards."""

    TRANSFERS = {torch.Tensor.to, torch.Tensor.copy_, torch.Tensor.cpu,
                 torch._has_compatible_shallow_copy_type}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in self.TRANSFERS:
            devices = {t.device for t in _tensors((args, kwargs)) if t.dim() > 0}
            if len(devices) > 1:
                raise RuntimeError(f"{getattr(func, '__name__', func)} mixes devices {sorted(map(str, devices))}")
        return func(*args, **kwargs)


# ── models, SAEs, data ───────────────────────────────────────────────────


def _llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=D_MODEL, intermediate_size=64,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)
    return LlamaForCausalLM(config).eval()


def _move_behind_a_dispatch_hook(module, device):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    module.to(device)
    add_hook_to_module(module, AlignDevicesHook(execution_device=torch.device(device), io_same_device=False))


def embedding_not_first():
    model = _llama()
    model.register_parameter(
        "a_tower_on_another_card",
        torch.nn.Parameter(torch.empty(1, device="meta"), requires_grad=False))
    _move_behind_a_dispatch_hook(model.lm_head, "meta")
    assert model.device == META and input_device(model) == CPU
    return model


def later_layers_elsewhere():
    model = _llama()
    for module in (model.model.layers[2], model.model.layers[3], model.model.norm, model.lm_head):
        _move_behind_a_dispatch_hook(module, "meta")
    return model


def _sae(seed):
    from src.ml.sparse_autoencoder import create_sae

    torch.manual_seed(100 + seed)
    sae = create_sae("standard", hidden_dim=D_MODEL, latent_dim=D_SAE, normalize_activations="none")
    with torch.no_grad():
        # Fire most features, so a run that reads nothing cannot pass as equal.
        sae.encoder.bias.fill_(1.0)
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae.eval()


def _dataset():
    from datasets import Dataset

    return Dataset.from_dict({"input_ids": [
        [5, 9, 11, 3, 7, 2], [8, 1, 4, 6, 10, 12], [3, 3, 9, 1, 2, 5]]})


TOKENIZER = SimpleNamespace(pad_token_id=0, eos_token_id=0)


class _Reader:
    """An EventReader over planted events: {feature: [(doc, pos, act), ...]}."""

    def __init__(self, events):
        self._events = events
        self.feature_ids = sorted(events)

    def feature_events(self, feature):
        rows = self._events.get(int(feature), [])
        out = np.zeros(len(rows), dtype=[("doc_id", "u4"), ("token_pos", "u2"), ("act", "f4")])
        for i, row in enumerate(rows):
            out[i] = row
        return out

    def feature_activation_mass(self):
        return {f: float(sum(a for _d, _p, a in rows)) for f, rows in self._events.items()}


# ── a database that answers by model and id ──────────────────────────────


class _Query:
    def __init__(self, rows):
        self._rows = rows
        self._key = None

    def filter(self, *criteria):
        for criterion in criteria:
            value = getattr(getattr(criterion, "right", None), "value", None)
            if value is not None:
                self._key = value
        return self

    def populate_existing(self):
        return self

    def first(self):
        if isinstance(self._rows, dict):
            return self._rows.get(self._key)
        return self._rows

    def all(self):
        return list(self._rows.values()) if isinstance(self._rows, dict) else [self._rows]


class _Db:
    def __init__(self, rows):
        self.rows = rows
        self.commits = 0

    def query(self, model):
        return _Query(self.rows.get(model))

    def execute(self, *_a, **_k):
        return MagicMock()

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def refresh(self, _obj):
        pass

    def add(self, _obj):
        pass


def _fake_task(db):
    task = MagicMock()

    @contextmanager
    def _get_db():
        yield db

    task.get_db = _get_db
    return task


def _raw(celery_task):
    return celery_task.__wrapped__.__func__


def _rows(*, layers=(1, 2), params=1_000_000_000, sae_dims=(4096, 16_384)):
    """Rows for one capture → discovery → circuit chain, and its model and SAEs."""
    from src.models.circuit import Circuit
    from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
    from src.models.dataset_tokenization import DatasetTokenization
    from src.models.external_sae import ExternalSAE
    from src.models.model import Model, QuantizationFormat

    manifest = {
        "model_id": "m_1",
        "corpus": {"tokenization_id": "tok_1", "sample_cap": 3},
        "layers": [{"layer": L, "sae_id": f"sae_l{L}", "epsilon": 0.0, "theta_floor": 0.0}
                   for L in layers],
        "estimate": {"minutes": 1.0},
    }
    capture = SimpleNamespace(id="cap_1", manifest=manifest, store_path="circuit_captures/cap_1",
                              gpu_request=RTX_UUID, gpu_uuid=None, gpu_uuids=None,
                              status="pending", error_message=None, progress=None)
    up, down = layers[0], layers[-1]
    discovery = SimpleNamespace(
        id="dsc_1", capture_run_id="cap_1", params={}, report=None,
        candidates=[
            {"up": {"layer": up, "feature_idx": 0}, "down": {"layer": down, "feature_idx": 4},
             "stats": {"pmi": 1.0}, "orderings": {"coact_rank": 0}},
            {"up": {"layer": up, "feature_idx": 3}, "down": {"layer": down, "feature_idx": 7},
             "stats": {"pmi": -1.0}, "orderings": {"coact_rank": 1}},
        ],
        attribution_status="pending", attribution_progress=None, attribution_error=None,
        validation_status="pending", validation_progress=None, validation_error=None)
    circuit = SimpleNamespace(
        id="crc_1", discovery_run_id="dsc_1", model_id="m_1", version=1,
        members=[{"layer": up, "member_kind": "feature_ref", "feature": {"feature_idx": 0}},
                 {"layer": down, "member_kind": "feature_ref", "feature": {"feature_idx": 4}}],
        saes=[{"layer": L, "mistudio_sae_id": f"sae_l{L}"} for L in layers],
        faithfulness_status="pending", calibration_status="pending")
    model = SimpleNamespace(id="m_1", repo_id="org/tiny", file_path=None, params_count=params,
                            quantization=QuantizationFormat.FP16)
    saes = {f"sae_l{L}": SimpleNamespace(id=f"sae_l{L}", layer=L, model_id="m_1",
                                         d_model=sae_dims[0], n_features=sae_dims[1])
            for L in layers}
    return {
        CircuitCaptureRun: {"cap_1": capture},
        CircuitDiscoveryRun: {"dsc_1": discovery},
        Circuit: {"crc_1": circuit},
        Model: {"m_1": model},
        DatasetTokenization: {"tok_1": SimpleNamespace(id="tok_1", tokenized_path="datasets/tok_1")},
        ExternalSAE: saes,
    }


def _expected_mb(params=1_000_000_000, sae_dims=(4096, 16_384), n_saes=2, matrices=2,
                 extra_per_sae_mb=0.0, working_mb=0.0):
    """The preflight's own figures: FP16 weights + 2 GB headroom, plus fp32 encoder+decoder per SAE
    (``matrices=1``: the decoder a steering job keeps), plus what the pass does with them."""
    return (params * 2.0 / 2**20 + 2.0 * 1024 + n_saes * matrices * sae_dims[0] * sae_dims[1] * 4 / 2**20
            + n_saes * extra_per_sae_mb + working_mb)


#: Review round 3, by hand for the fixture's 16,384-feature SAEs. Capture encodes 8 x 512 =
#: 4,096 tokens: 5 units x 4,096 x 16,384 x 4 B = 1,342,177,280 B. Validation and faithfulness
#: encode one 512-token prompt: 5 x 512 x 16,384 x 4 B = 167,772,160 B. Attribution keeps 8 units
#: of one prompt's codes beside every hooked layer: 8 x 512 x 16,384 x 4 B = 268,435,456 B each.
CAPTURE_WORKING_MB = 1_342_177_280 / 2**20
PROMPT_WORKING_MB = 167_772_160 / 2**20
ATTRIBUTION_RETAINED_MB = 268_435_456 / 2**20


# ═════════════════════ WHICH RUN TYPES MAY SPLIT ══════════════════════════


@pytest.fixture
def placing(monkeypatch):
    """`place_job` hands every job the split and records what it was asked."""
    calls = []

    def fake_place_job(requested="auto", required_mb=None, cards=None, allow_shard=False):
        calls.append({"requested": requested, "required_mb": required_mb, "allow_shard": allow_shard})
        return SPLIT

    monkeypatch.setattr("src.workers.circuit_gpu.place_job", fake_place_job)
    return calls


SPLIT_RECORD = {"request": RTX_UUID, "uuid": RTX_UUID, "name": "NVIDIA GeForce RTX 3090",
                "uuids": [RTX_UUID, TI_UUID]}


class TestSplitSafeRunsOptIn:
    """Every circuit run places with `allow_shard=True` and a size; the service receives
    the whole placement; the card record names every card."""

    def test_capture_splits_and_records_every_card_on_its_row(self, placing):
        from src.models.circuit_runs import CircuitCaptureRun
        from src.services.circuit_capture_service import CircuitCaptureService
        from src.workers import circuit_capture_tasks as tasks

        rows = _rows()
        row = rows[CircuitCaptureRun]["cap_1"]
        db = _Db(rows)
        seen = []

        def fake_run_capture(db_, run_id, *, confirmed, device, placement=None,
                             cancel_check=None, progress_cb=None):
            seen.append((device, placement, row.gpu_uuid, row.gpu_uuids, db.commits))
            return {"status": "estimated"}

        with patch.object(CircuitCaptureService, "run_capture", side_effect=fake_run_capture), \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.capture_circuit_activations)(_fake_task(db), "cap_1")

        assert placing == [{"requested": RTX_UUID,
                            "required_mb": pytest.approx(_expected_mb(working_mb=CAPTURE_WORKING_MB)),
                            "allow_shard": True}]
        assert len(seen) == 1
        device, placement, uuid, uuids, commits = seen[0]
        assert device == CUDA1 and placement is SPLIT
        assert (uuid, uuids) == (RTX_UUID, [RTX_UUID, TI_UUID]), "the split's cards were not recorded"
        assert commits >= 1, "the cards must be committed before the model load"

    @pytest.mark.parametrize("task_name, service_path, args, work", [
        ("circuit_capture_tasks.run_circuit_attribution",
         "src.services.circuit_attribution_service.CircuitAttributionService.run", ("dsc_1",),
         {"extra_per_sae_mb": ATTRIBUTION_RETAINED_MB}),
        ("circuit_validation_tasks.validate_circuit_edges",
         "src.services.circuit_intervention_service.CircuitInterventionService.run", ("dsc_1", {"k": 1}),
         {"working_mb": PROMPT_WORKING_MB}),
        ("circuit_validation_tasks.run_circuit_faithfulness",
         "src.services.circuit_faithfulness_service.CircuitFaithfulnessService.run", ("crc_1", {}),
         {"working_mb": PROMPT_WORKING_MB}),
    ], ids=["attribution", "validation", "faithfulness"])
    def test_the_pass_splits_sized_from_its_capture(self, placing, task_name, service_path, args, work):
        import importlib

        module_name, attr = task_name.split(".")
        tasks = importlib.import_module(f"src.workers.{module_name}")
        db = _Db(_rows())
        with patch(service_path, return_value={"status": "completed"}) as run, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(getattr(tasks, attr))(_fake_task(db), *args, gpu_request=RTX_UUID)

        assert placing == [{"requested": RTX_UUID, "required_mb": pytest.approx(_expected_mb(**work)),
                            "allow_shard": True}]
        assert run.call_count == 1
        assert run.call_args.kwargs["placement"] is SPLIT
        assert run.call_args.kwargs["device"] == CUDA1
        assert run.call_args.kwargs["gpu"] == SPLIT_RECORD

    def test_calibration_splits_sized_from_the_saes_it_steers_with(self, placing):
        from src.models.circuit import Circuit
        from src.services.circuit_calibration_service import CircuitCalibrationService
        from src.workers import circuit_calibration_tasks as tasks

        rows = _rows()
        # The capture recorded two layers; the circuit steers with one SAE. A
        # size read from the capture would count both.
        rows[Circuit]["crc_1"].saes = [{"layer": 2, "mistudio_sae_id": "sae_l2"}]
        with patch.object(CircuitCalibrationService, "run", return_value={"band": None}) as run, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.run_circuit_calibration)(_fake_task(_Db(rows)), "crc_1", {}, gpu_request=RTX_UUID)

        # Review round 3: the decoder alone, which is all the core keeps on a card.
        assert placing == [{"requested": RTX_UUID,
                            "required_mb": pytest.approx(_expected_mb(n_saes=1, matrices=1)),
                            "allow_shard": True}]
        assert run.call_count == 1
        assert run.call_args.kwargs["placement"] is SPLIT
        assert run.call_args.kwargs["device"] == CUDA1
        assert run.call_args.kwargs["gpu"] == SPLIT_RECORD

    def test_calibration_reproduce_splits_sized_from_its_circuit(self, placing):
        from src.models.validation_manifest import ValidationManifest
        from src.services.circuit_calibration_service import CircuitCalibrationService
        from src.workers import circuit_calibration_tasks as tasks

        rows = _rows()
        rows[ValidationManifest] = {"vman_k": SimpleNamespace(id="vman_k", circuit_id="crc_1")}
        with patch.object(CircuitCalibrationService, "reproduce", return_value={}) as reproduce, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.reproduce_circuit_calibration)(_fake_task(_Db(rows)), "vman_k", "completed",
                                                      gpu_request=RTX_UUID)

        assert placing == [{"requested": RTX_UUID, "required_mb": pytest.approx(_expected_mb(matrices=1)),
                            "allow_shard": True}]
        assert reproduce.call_count == 1
        assert reproduce.call_args.kwargs["placement"] is SPLIT
        assert reproduce.call_args.kwargs["gpu"] == SPLIT_RECORD

    def test_a_calibration_on_one_card_keeps_its_exact_call(self, monkeypatch):
        from src.services.circuit_calibration_service import CircuitCalibrationService
        from src.workers import circuit_calibration_tasks as tasks

        monkeypatch.setattr("src.workers.circuit_gpu.place_job",
                            lambda requested="auto", required_mb=None, cards=None, allow_shard=False: ONE_CARD)
        with patch.object(CircuitCalibrationService, "run", return_value={"band": None}) as run, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.run_circuit_calibration)(_fake_task(_Db(_rows())), "crc_1", {}, gpu_request=RTX_UUID)

        assert "placement" not in run.call_args.kwargs

    @pytest.mark.parametrize("artifact, n_saes", [
        ({"kind": "circuit", "circuit_id": "crc_1"}, 2),
        ({"kind": "features", "model_id": "m_1",
          "features": [{"layer": 1, "feature_idx": 0, "strength": 1.0, "sae_id": "sae_l1"},
                       {"layer": 2, "feature_idx": 4, "strength": 1.0, "sae_id": "sae_l2"}]}, 2),
        ({"kind": "cluster", "cluster_profile_id": "cp_1"}, 1),
    ], ids=["circuit", "features", "cluster"])
    def test_the_recorder_splits_sized_from_its_artifact(self, placing, artifact, n_saes):
        from src.models.cluster_profile import ClusterProfile
        from src.models.steering_record_run import SteeringRecordRun
        from src.services.steering_recorder_service import SteeringRecorderService
        from src.workers import circuit_record_tasks as tasks

        rows = _rows()
        run_row = SimpleNamespace(id="srr_1", status="pending", gpu_request="all", gpu_uuid=None,
                                  gpu_uuids=None, error=None, manifest_ref=None)
        rows[SteeringRecordRun] = {"srr_1": run_row}
        # A CLUSTERS-arc profile: one sae_id, no model_id, members without a layer.
        rows[ClusterProfile] = {"cp_1": SimpleNamespace(
            id="cp_1", sae_id="sae_l1", model_id=None, saes=None,
            members=[{"feature_idx": 0, "strength": 1.0}])}
        config = {"artifact": artifact, "dials": [0.5], "prompts": ["The weather is"]}
        seen = []

        def fake_record(db, config_, *, device, gpu=None, placement=None, **kwargs):
            seen.append((device, gpu, placement, run_row.gpu_uuids))
            return {"manifest_ref": "vman_s"}

        with patch.object(SteeringRecorderService, "record_samples", side_effect=fake_record), \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.run_circuit_record)(_fake_task(_Db(rows)), "srr_1", config)

        assert placing == [{"requested": "all",
                            "required_mb": pytest.approx(_expected_mb(n_saes=n_saes, matrices=1)),
                            "allow_shard": True}]
        assert len(seen) == 1
        device, gpu, placement, uuids_at_load = seen[0]
        assert device == CUDA1 and placement is SPLIT
        assert gpu["uuids"] == [RTX_UUID, TI_UUID]
        assert uuids_at_load == [RTX_UUID, TI_UUID], "the cards were not recorded before the load"

    def test_a_recorder_that_cannot_be_sized_still_places(self, placing):
        from src.models.steering_record_run import SteeringRecordRun
        from src.services.steering_recorder_service import SteeringRecorderService
        from src.workers import circuit_record_tasks as tasks

        rows = _rows()
        rows[SteeringRecordRun] = {"srr_1": SimpleNamespace(
            id="srr_1", status="pending", gpu_request="all", gpu_uuid=None, gpu_uuids=None,
            error=None, manifest_ref=None)}
        config = {"artifact": {"kind": "circuit", "circuit_id": "crc_gone"},
                  "dials": [0.5], "prompts": ["The weather is"]}

        with patch.object(SteeringRecorderService, "record_samples",
                          return_value={"manifest_ref": "vman_s"}), \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.run_circuit_record)(_fake_task(_Db(rows)), "srr_1", config)

        assert placing == [{"requested": "all", "required_mb": None, "allow_shard": True}]


class TestASteeringJobReturnsItsCards:
    """Calibration (run and reproduce) and the recorder load a model through
    steering_core and returned WITHOUT releasing it — unlike capture, attribution,
    validation and faithfulness, which all clean up in a `finally`. They run on the
    shared `extraction` worker, so the model's blocks stayed reserved on every card
    of the job and NVML, which the next job's placement reads, went on counting them.

    The cleanup must run AFTER the model is unreachable: on a failure the traceback
    holds the service's frames, and the model in them, so those frames are cleared
    first. `held` is a weak reference to a model the fake service keeps in a local.
    """

    def _run(self, kind, *, fail, monkeypatch, placement=SPLIT):
        import weakref

        from src.models.steering_record_run import SteeringRecordRun
        from src.models.validation_manifest import ValidationManifest
        from src.services.circuit_calibration_service import CircuitCalibrationService
        from src.services.steering_recorder_service import SteeringRecorderService
        from src.workers import circuit_calibration_tasks as calibration_tasks
        from src.workers import circuit_record_tasks as record_tasks

        monkeypatch.setattr(
            "src.workers.circuit_gpu.place_job",
            lambda requested="auto", required_mb=None, cards=None, allow_shard=False: placement)
        rows = _rows()
        rows[SteeringRecordRun] = {"srr_1": SimpleNamespace(
            id="srr_1", status="pending", gpu_request="all", gpu_uuid=None, gpu_uuids=None,
            error=None, manifest_ref=None)}
        rows[ValidationManifest] = {"vman_k": SimpleNamespace(id="vman_k", circuit_id="crc_1")}
        held, cleanups = [], []

        def service(*_args, **_kwargs):
            model = torch.nn.Linear(2, 2)
            held.append(weakref.ref(model))
            if fail:
                raise RuntimeError("generation died")
            return {"manifest_ref": "vman_s", "band": None}

        def cleanup(models_to_cleanup=None, context="unknown", device=None):
            cleanups.append({"model_gone": held[-1]() is None if held else None,
                             "device": device, "models": models_to_cleanup})

        monkeypatch.setattr("src.services.extraction_service.cleanup_gpu_memory", cleanup)
        target, call = {
            "calibration": (
                patch.object(CircuitCalibrationService, "run", side_effect=service),
                lambda db: _raw(calibration_tasks.run_circuit_calibration)(
                    _fake_task(db), "crc_1", {}, gpu_request=RTX_UUID)),
            "reproduce": (
                patch.object(CircuitCalibrationService, "reproduce", side_effect=service),
                lambda db: _raw(calibration_tasks.reproduce_circuit_calibration)(
                    _fake_task(db), "vman_k", "completed", gpu_request=RTX_UUID)),
            "record": (
                patch.object(SteeringRecorderService, "record_samples", side_effect=service),
                lambda db: _raw(record_tasks.run_circuit_record)(
                    _fake_task(db), "srr_1",
                    {"artifact": {"kind": "circuit", "circuit_id": "crc_1"},
                     "dials": [0.5], "prompts": ["hi"]})),
        }[kind]
        with target, \
             patch.object(calibration_tasks, "emit_circuit_run_completed"), \
             patch.object(calibration_tasks, "emit_circuit_run_failed"), \
             patch.object(record_tasks, "emit_circuit_run_completed"), \
             patch.object(record_tasks, "emit_circuit_run_failed"):
            if fail:
                with pytest.raises(RuntimeError, match="generation died"):
                    call(_Db(rows))
            else:
                call(_Db(rows))
        return cleanups

    @pytest.mark.parametrize("kind", ["calibration", "reproduce", "record"])
    @pytest.mark.parametrize("fail", [False, True], ids=["succeeded", "failed"])
    def test_every_card_of_the_job_is_cleaned_once_the_model_is_unreachable(self, kind, fail, monkeypatch):
        cleanups = self._run(kind, fail=fail, monkeypatch=monkeypatch)

        assert len(cleanups) == 1, f"{kind} did not clean up its cards once ({cleanups})"
        assert list(cleanups[0]["device"]) == [CUDA1, CUDA0], "cleanup must cover every card of the split"
        assert cleanups[0]["model_gone"] is True, "cleanup ran while something still held the model"

    def test_a_single_card_job_cleans_its_one_card(self, monkeypatch):
        cleanups = self._run("record", fail=False, monkeypatch=monkeypatch, placement=ONE_CARD)

        assert [list(c["device"]) for c in cleanups] == [[CUDA1]]


class TestCalibrationGeneratesOnTheSplit:
    """The calibration service hands a split's placement to the loader and the loaded
    structure to the resolver, so each layer's decoder lands on that layer's card."""

    def test_run_and_reproduce_hand_the_placement_on(self):
        from src.models.circuit import Circuit
        from src.models.validation_manifest import ValidationManifest
        from src.services.circuit_calibration_service import CircuitCalibrationService

        rows = _rows()
        rows[ValidationManifest] = {"vman_k": SimpleNamespace(
            id="vman_k", circuit_id="crc_1", kind="calibration", payload={"config": {}})}
        rows[Circuit]["crc_1"].members = [{"layer": 1}]
        handed = []

        def fake_fns(circuit_, db, cfg, *, device, placement=None):
            handed.append((device, placement))
            raise _Stop

        with patch.object(CircuitCalibrationService, "_build_generation_fns", side_effect=fake_fns):
            with pytest.raises(_Stop):
                CircuitCalibrationService.run(_Db(rows), "crc_1", {}, device=CUDA1, gpu={},
                                              placement=SPLIT)
            with pytest.raises(_Stop):
                CircuitCalibrationService.reproduce(_Db(rows), "vman_k", device=CUDA1, gpu={},
                                                    placement=SPLIT)

        assert handed == [(CUDA1, SPLIT), (CUDA1, SPLIT)]

    def test_the_generation_loads_split_and_resolves_against_the_loaded_structure(self, monkeypatch):
        from src.services import steering_core
        from src.services.circuit_calibration_service import CircuitCalibrationService

        calls = {}

        def fake_load(model_id, db, device, placement=None, sae_ids_by_layer=None):
            calls["load"] = (model_id, device, placement)
            calls["sae_ids_by_layer"] = sae_ids_by_layer
            return "MODEL", "TOKENIZER", "STRUCTURE", False, CUDA1

        def fake_resolve(circuit, db, device, structure=None):
            calls["resolve"] = (device, structure)
            raise _Stop

        monkeypatch.setattr(steering_core, "load_model_and_structure", fake_load)
        monkeypatch.setattr(steering_core, "resolve_circuit_members", fake_resolve)
        # Two layers with two SAEs, neither at layer 0 or 1: an allowance keyed by
        # position, or naming one SAE for every layer, disagrees with this.
        circuit = SimpleNamespace(model_id="m_1", saes=[
            {"layer": 5, "mistudio_sae_id": "sae_l5"}, {"layer": 9, "mistudio_sae_id": "sae_l9"}])
        with pytest.raises(_Stop):
            CircuitCalibrationService._build_generation_fns(
                circuit, None,
                {"seed": 0, "judge_endpoint": "http://judge/v1", "judge_model": "j"},
                device=CUDA1, placement=SPLIT)
        # The SAEs its resolver will put on each layer's card go to the split load.
        assert calls.pop("sae_ids_by_layer") == {5: "sae_l5", 9: "sae_l9"}

        assert calls == {"load": ("m_1", CUDA1, SPLIT), "resolve": (CUDA1, "STRUCTURE")}


class TestTheCardRecord:
    def test_a_single_card_record_is_unchanged(self):
        from src.workers.circuit_gpu import gpu_record

        assert gpu_record("auto", ONE_CARD) == {"request": "auto", "uuid": RTX_UUID,
                                                "name": "NVIDIA GeForce RTX 3090"}

    def test_a_split_record_lists_every_card_in_fill_order(self):
        from src.workers.circuit_gpu import gpu_record

        assert gpu_record(RTX_UUID, SPLIT) == SPLIT_RECORD


class TestTheSize:
    def test_weights_headroom_and_every_sae(self):
        from src.workers.circuit_gpu import circuit_required_mb, discovery_capture_manifest

        db = _Db(_rows(layers=(1, 2, 3), params=8_000_000_000, sae_dims=(3840, 65_536)))
        manifest = discovery_capture_manifest(db, "dsc_1")
        assert circuit_required_mb(db, manifest) == pytest.approx(
            _expected_mb(params=8_000_000_000, sae_dims=(3840, 65_536), n_saes=3))

    @pytest.mark.parametrize("fmt, per_param", [("Q4", 0.6), ("Q8", 1.1), ("FP32", 4.0)])
    def test_a_quantized_or_full_precision_row_is_sized_at_its_own_precision(self, fmt, per_param):
        """P20a/P20b. Every other fixture here is an FP16 row, whose 2 bytes a parameter
        equals the fallback, so a size that ignored the row's precision — or read
        `str(QuantizationFormat.Q4)`, which is 'QuantizationFormat.Q4', not 'Q4' —
        agreed with the right one by construction. Sized at FP16, a Q4 model is
        3.3x too big: Auto splits a model one card holds, or refuses it."""
        from src.models.model import Model, QuantizationFormat
        from src.workers.circuit_gpu import circuit_required_mb, discovery_capture_manifest

        rows = _rows(params=8_000_000_000)
        rows[Model]["m_1"].quantization = QuantizationFormat(fmt)
        db = _Db(rows)
        saes_mb = 2 * 2 * 4096 * 16_384 * 4 / 2**20
        assert circuit_required_mb(db, discovery_capture_manifest(db, "dsc_1")) == pytest.approx(
            8_000_000_000 * per_param / 2**20 + 2.0 * 1024 + saes_mb)

    #: An 8B-class description: 32 x (4*4096^2 + 3*4096*14336) + 2*128256*4096
    #: = 7,784,628,224 + 1,050,673,152 = 8,835,301,376 parameters, worked by hand.
    ARCH_8B = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 128_256,
               "intermediate_size": 14_336, "model_type": "llama"}
    ARCH_8B_PARAMS = 8_835_301_376

    def test_a_four_bit_row_is_sized_from_its_architecture_not_its_packed_count(self):
        """Review round 2. A Q4 row's params_count is counted off the model loaded at
        Q4, where bitsandbytes packs two values a byte, so it reports about half the
        parameters (aaa233de). That fix reached training and SAE extraction only; a
        circuit job sized a Q4 model at ~half its weights, and Auto could place it on a
        card, or choose a split, that cannot hold it."""
        from src.models.model import Model, QuantizationFormat
        from src.workers.circuit_gpu import circuit_required_mb, discovery_capture_manifest

        rows = _rows(params=4_940_000_000)
        rows[Model]["m_1"].quantization = QuantizationFormat.Q4
        rows[Model]["m_1"].architecture_config = dict(self.ARCH_8B)
        db = _Db(rows)
        saes_mb = 2 * 2 * 4096 * 16_384 * 4 / 2**20
        assert circuit_required_mb(db, discovery_capture_manifest(db, "dsc_1")) == pytest.approx(
            self.ARCH_8B_PARAMS * 0.6 / 2**20 + 2.0 * 1024 + saes_mb)

    def test_an_eight_bit_row_keeps_its_own_count(self):
        """bitsandbytes int8 stores one value a byte, so a Q8 row counts every parameter."""
        from src.models.model import Model, QuantizationFormat
        from src.workers.circuit_gpu import circuit_required_mb, discovery_capture_manifest

        rows = _rows(params=4_940_000_000)
        rows[Model]["m_1"].quantization = QuantizationFormat.Q8
        rows[Model]["m_1"].architecture_config = dict(self.ARCH_8B)
        db = _Db(rows)
        saes_mb = 2 * 2 * 4096 * 16_384 * 4 / 2**20
        assert circuit_required_mb(db, discovery_capture_manifest(db, "dsc_1")) == pytest.approx(
            4_940_000_000 * 1.1 / 2**20 + 2.0 * 1024 + saes_mb)

    def test_the_circuit_reaches_its_capture_through_its_discovery_run(self):
        from src.workers.circuit_gpu import circuit_capture_manifest

        db = _Db(_rows())
        assert circuit_capture_manifest(db, "crc_1")["model_id"] == "m_1"

    def test_an_unknown_model_gives_no_size_rather_than_a_guess(self):
        from src.workers.circuit_gpu import circuit_required_mb

        assert circuit_required_mb(_Db({}), {"model_id": "m_gone", "layers": []}) is None
        assert circuit_required_mb(_Db({}), None) is None

    @pytest.mark.parametrize("quantization, bytes_per_param", [
        ("Q4", 0.6), ("Q8", 1.1), ("FP32", 4.0),
    ])
    def test_the_weights_are_sized_at_the_models_own_quantization(self, quantization, bytes_per_param):
        """Every other fixture here is FP16, whose 2 bytes a parameter is ALSO the
        lookup's fallback — so sizing that ignored the quantization passed them all,
        and a Q4 model was sized at FP16, over three times its weights. Auto then
        splits, or refuses, a model one card holds."""
        from src.models.model import Model, QuantizationFormat
        from src.workers.circuit_gpu import circuit_required_mb, discovery_capture_manifest

        rows = _rows(params=8_000_000_000)
        rows[Model]["m_1"].quantization = QuantizationFormat(quantization)
        db = _Db(rows)

        fp16 = _expected_mb(params=8_000_000_000)
        expected = fp16 - 8_000_000_000 * 2.0 / 2**20 + 8_000_000_000 * bytes_per_param / 2**20
        assert circuit_required_mb(db, discovery_capture_manifest(db, "dsc_1")) == pytest.approx(expected)

    def test_faithfulness_is_sized_from_the_layers_its_circuit_sits_on(self, placing):
        """The capture recorded three layers; the circuit's members sit on 1 and 3.
        The pass loads an SAE for those two only, so a size read from the capture
        counted an SAE it never loads. Every other case here builds a circuit whose
        members cover every captured layer, where the two sizes agree by construction."""
        from src.services.circuit_faithfulness_service import CircuitFaithfulnessService
        from src.workers import circuit_validation_tasks as tasks

        db = _Db(_rows(layers=(1, 2, 3)))
        with patch.object(CircuitFaithfulnessService, "run", return_value={"status": "completed"}) as run, \
             patch.object(tasks, "emit_circuit_run_completed"):
            _raw(tasks.run_circuit_faithfulness)(_fake_task(db), "crc_1", {}, gpu_request=RTX_UUID)

        assert placing == [{"requested": RTX_UUID,
                            "required_mb": pytest.approx(_expected_mb(n_saes=2, working_mb=PROMPT_WORKING_MB)),
                            "allow_shard": True}]
        assert run.call_count == 1

    def test_the_size_counts_the_layers_the_pass_loads_an_sae_for(self):
        """ONE definition: the layers the size counts are the layers the service
        expands its members to, cluster members included."""
        from src.models.cluster_profile import ClusterProfile
        from src.models.circuit import Circuit
        from src.services.circuit_faithfulness_service import expand_circuit_members
        from src.workers.circuit_gpu import circuit_faithfulness_manifest

        rows = _rows(layers=(1, 2, 3, 4))
        rows[Circuit]["crc_1"].members = [
            {"layer": 2, "member_kind": "cluster_ref", "cluster_profile_id": "cp_1"},
            {"layer": 3, "member_kind": "cluster_ref", "cluster_profile_id": "cp_empty"},
            {"layer": 4, "member_kind": "feature_ref", "feature": {"feature_idx": 9}},
        ]
        rows[ClusterProfile] = {
            "cp_1": SimpleNamespace(id="cp_1", members=[{"feature_idx": 5}]),
            "cp_empty": SimpleNamespace(id="cp_empty", members=[]),
        }
        db = _Db(rows)

        loads = sorted(expand_circuit_members(db, rows[Circuit]["crc_1"]))
        sized = [e["layer"] for e in circuit_faithfulness_manifest(db, "crc_1")["layers"]]
        assert loads == [2, 4]
        assert sized == loads

    def test_the_pass_expands_its_members_through_that_one_definition(self):
        """The equality above is only worth something if the pass itself CALLS the
        expansion the size reads. Read from the AST — a call, not a name in text."""
        import ast
        import inspect

        from src.services import circuit_faithfulness_service as service

        tree = ast.parse(inspect.getsource(service))
        run = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                   and cls.name == "CircuitFaithfulnessService"
                   for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "expand_circuit_members"]
        assert len(calls) == 1, "the pass expands its members some other way than the size does"
        assert [a.id for a in calls[0].args if isinstance(a, ast.Name)] == ["db", "circuit"]


# ═════════════════════ THE LOAD TAKES THE SPLIT ═══════════════════════════


def _load_until_the_model(kind, placement, monkeypatch, layers=(1, 2)):
    """Run a service until its model load, and return the kwargs the loader received."""
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    db = _Db(_rows(layers=layers))
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        raise _Stop

    common = [patch("datasets.load_from_disk", return_value=_dataset()),
              patch("src.ml.model_loader.load_model_from_hf", side_effect=fake_load),
              patch("src.services.extraction_service.cleanup_gpu_memory"),
              patch("src.services.circuit_capture_store.EventReader"),
              patch("src.services.circuit_intervention_service.EventReader")]
    for p in common:
        p.start()
    try:
        with pytest.raises(_Stop):
            if kind == "capture":
                from src.services.circuit_capture_service import CircuitCaptureService

                CircuitCaptureService.run_capture(db, "cap_1", confirmed=True,
                                                  device=placement.device, placement=placement)
            elif kind == "attribution":
                from src.services.circuit_attribution_service import CircuitAttributionService

                CircuitAttributionService.run(db, "dsc_1", device=placement.device, placement=placement)
            elif kind == "validation":
                from src.services.circuit_intervention_service import CircuitInterventionService

                CircuitInterventionService.run(db, "dsc_1", CircuitInterventionService.create_scope({"k": 1}),
                                               device=placement.device, placement=placement)
            else:
                from src.services.circuit_faithfulness_service import CircuitFaithfulnessService

                with patch.object(CircuitFaithfulnessService, "_select_prompts", return_value=[0]):
                    CircuitFaithfulnessService.run(db, "crc_1", {}, device=placement.device,
                                                   placement=placement)
    finally:
        for p in common:
            p.stop()
    return seen


KINDS = ["capture", "attribution", "validation", "faithfulness"]


#: One fp32 SAE of the `_rows` fixture (4,096 x 16,384), worked by hand: encoder and
#: decoder 2 x 4,096 x 16,384 + both biases 4,096 + 16,384 = 134,238,208 values x 4 bytes.
FIXTURE_SAE_MB = 536_952_832 / 2**20

#: The layers whose SAE each pass puts on that layer's card: the capture's three for a
#: pass over the capture, and the circuit's two (members at 1 and 3) for faithfulness —
#: so a faithfulness allowance read from the capture disagrees.
SAE_LAYERS = {"capture": (1, 2, 3), "attribution": (1, 2, 3), "validation": (1, 2, 3),
              "faithfulness": (1, 3)}


class TestTheLoadTakesThePlacement:
    @pytest.mark.parametrize("kind", KINDS)
    def test_a_split_loads_with_the_placements_map_and_budget(self, kind, monkeypatch):
        seen = _load_until_the_model(kind, SPLIT, monkeypatch, layers=(1, 2, 3))

        # The placement owns the map strategy; the service passes it through.
        assert seen["device_map"] == SPLIT.device_map
        assert seen["device_map"] != str(SPLIT.device), "the split was loaded onto its first card"
        assert seen["max_memory"] == {1: "21976MiB", 0: "9976MiB"}

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_split_is_mapped_to_hold_each_layers_sae_beside_it(self, kind, monkeypatch):
        """Review round 2. Each pass puts every layer's SAE on that layer's card, and a
        split keeps only SHARD_RESERVE_MB free per card, less than a 12B-class SAE.
        The load is told each layer's SAE size, so the split is mapped around them."""
        seen = _load_until_the_model(kind, SPLIT, monkeypatch, layers=(1, 2, 3))

        # Review round 3: attribution also keeps each layer's codes for its backward.
        held = FIXTURE_SAE_MB + (ATTRIBUTION_RETAINED_MB if kind == "attribution" else 0.0)
        assert seen["extra_mb_by_layer"] == {layer: pytest.approx(held) for layer in SAE_LAYERS[kind]}

    @pytest.mark.parametrize("kind, working", [
        ("capture", CAPTURE_WORKING_MB), ("attribution", None),
        ("validation", PROMPT_WORKING_MB), ("faithfulness", PROMPT_WORKING_MB),
    ])
    def test_a_split_keeps_room_for_the_codes_each_layer_encodes(self, kind, working, monkeypatch):
        """Review round 3. An encode allocates its codes beside the layer it reads, on that
        layer's card, one layer at a time. Capture's 4,096-token batch on a 65,536-feature
        SAE is 4 GiB, and a split keeps 1 GiB free per card. Attribution's codes are held
        for the backward instead, so they are in each layer's allowance above."""
        seen = _load_until_the_model(kind, SPLIT, monkeypatch, layers=(1, 2, 3))

        if working is None:
            assert "working_mb_by_layer" not in seen
        else:
            assert seen["working_mb_by_layer"] == {layer: pytest.approx(working) for layer in SAE_LAYERS[kind]}

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_single_card_load_is_unchanged(self, kind, monkeypatch):
        seen = _load_until_the_model(kind, ONE_CARD, monkeypatch, layers=(1, 2, 3))

        assert seen["device_map"] == CUDA1
        assert "max_memory" not in seen
        assert "extra_mb_by_layer" not in seen


# ═════════════════════ EACH SAE ON ITS LAYER'S CARD ═══════════════════════


def _hooks_left(model):
    return [name for name, module in model.named_modules() if hasattr(module, "_hf_hook")]


def _route(kind, next_step, monkeypatch):
    """Run a service on `later_layers_elsewhere` with layers (1, 3) up to `next_step`.

    Returns (devices each SAE was loaded onto, what `next_step` recorded,
    dispatch hooks still attached when cleanup ran).
    """
    model = later_layers_elsewhere()
    rows = _rows(layers=(1, 3))
    db = _Db(rows)
    loads, cleaned = {}, {}
    saes = {1: _sae(1), 3: _sae(3)}

    def fake_load_sae(record, device):
        loads[record.layer] = torch.device(device)
        return saes[record.layer].to(device)

    def fake_cleanup(models, context=None, device=None):
        cleaned["hooks"] = _hooks_left(model)

    patches = [patch("datasets.load_from_disk", return_value=_dataset()),
               patch("src.ml.model_loader.load_model_from_hf", return_value=(model, TOKENIZER, None, {})),
               patch("src.services.circuit_capture_service._load_sae_sync", side_effect=fake_load_sae),
               patch("src.services.extraction_service.cleanup_gpu_memory", side_effect=fake_cleanup),
               patch("src.services.circuit_capture_store.EventReader"),
               patch("src.services.circuit_intervention_service.EventReader")]
    recorded = {}
    for p in patches:
        p.start()
    try:
        recorded = next_step(db, model)
    finally:
        for p in patches:
            p.stop()
    return loads, recorded, cleaned.get("hooks")


class TestEachSaeSitsOnItsLayersCard:
    """`_load_sae_sync` is handed each layer's device — cpu for layer 1, meta for layer 3."""

    EXPECTED = {1: CPU, 3: META}

    def test_attribution(self, monkeypatch):
        from src.services import circuit_attribution_service as mod

        codes = {}

        def record_codes(state, groups):
            codes.update({L: f.device for L, f in state.codes.items()})
            return {}

        def step(db, model):
            with patch.object(mod, "attribute_prompt", side_effect=record_codes):
                mod.CircuitAttributionService.run(db, "dsc_1", prompt_limit=1, device=CPU)
            return codes

        loads, codes, hooks = _route("attribution", step, monkeypatch)
        assert loads == self.EXPECTED
        # The hooks ran on each layer's hidden state with the SAE beside it.
        assert codes == self.EXPECTED
        assert hooks == [], "the split model's dispatch hooks were not removed before cleanup"

    def test_validation(self, monkeypatch):
        from src.services.circuit_intervention_service import CircuitInterventionService

        seen = {}

        def record(model, structure, ghm, saes, *rest):
            seen.update({L: module_device(s) for L, s in saes.items()})
            raise _Stop

        def step(db, model):
            with patch.object(CircuitInterventionService, "_validate_edge", side_effect=record), \
                 pytest.raises(_Stop):
                CircuitInterventionService.run(db, "dsc_1", CircuitInterventionService.create_scope({"k": 1}),
                                               device=CPU)
            return seen

        loads, seen, hooks = _route("validation", step, monkeypatch)
        assert loads == self.EXPECTED and seen == self.EXPECTED
        assert hooks == []

    def test_faithfulness(self, monkeypatch):
        from src.services.circuit_faithfulness_service import CircuitFaithfulnessService as F

        seen = {}

        def record(*, saes, **_kw):
            seen.update({L: module_device(s) for L, s in saes.items()})
            raise _Stop

        def step(db, model):
            with patch.object(F, "_select_prompts", return_value=[0]), \
                 patch.object(F, "_top_features", return_value=[]), \
                 patch.object(F, "_top_nonmembers", return_value=[]), \
                 patch.object(F, "_behavior", side_effect=record), \
                 pytest.raises(_Stop):
                F.run(db, "crc_1", {}, device=CPU)
            return seen

        loads, seen, hooks = _route("faithfulness", step, monkeypatch)
        assert loads == self.EXPECTED and seen == self.EXPECTED
        assert hooks == []

    def test_capture(self, monkeypatch):
        from src.services import circuit_capture_service as cap

        seen = {}

        def record(model, tokenizer, dataset, saes, *rest, **kw):
            seen.update({L: module_device(s) for L, s in saes.items()})
            raise _Stop

        def step(db, model):
            with patch.object(cap, "_probe", side_effect=record), \
                 patch.object(cap, "_sae_fingerprint", return_value="fp"), \
                 pytest.raises(_Stop):
                cap.CircuitCaptureService.run_capture(db, "cap_1", confirmed=False, device=CPU)
            return seen

        loads, seen, hooks = _route("capture", step, monkeypatch)
        assert loads == self.EXPECTED and seen == self.EXPECTED
        assert hooks == []


class TestTheCaptureEncodesOnEachLayersSae:
    """Hooked activations come back to host RAM; each layer's go to THAT layer's SAE."""

    def _encodes(self, drive, monkeypatch):
        from src.services import circuit_capture_service as cap

        seen = []

        def record(sae, acts):
            seen.append((acts.device, module_device(sae)))
            raise _Stop

        monkeypatch.setattr(cap, "_encode_layer", record)
        with pytest.raises(_Stop):
            drive(cap)
        return seen

    def test_the_probe(self, monkeypatch):
        saes = {1: _sae(1).to("meta"), 2: _sae(2)}
        seen = self._encodes(lambda cap: cap._probe(_llama(), TOKENIZER, _dataset(), saes, [1, 2], CPU),
                             monkeypatch)
        assert seen == [(META, META)]

    def test_the_capture_batch(self, monkeypatch):
        saes = {1: _sae(1).to("meta"), 2: _sae(2)}
        ids = torch.tensor([[5, 9, 11, 3]])
        writers = {L: (MagicMock(), MagicMock(), None) for L in (1, 2)}
        seen = self._encodes(lambda cap: cap._capture_batch(
            _llama(), saes, [1, 2], writers, ids, torch.ones_like(ids), 0, [4],
            epsilon_by_layer={1: 0.0, 2: 0.0}, floor_by_layer={1: 0.0, 2: 0.0},
            probe_max={}, attn_cfg=None), monkeypatch)
        assert seen == [(META, META)]


# ═════════════════════ HOOK MATH MEETS THE HIDDEN STATE ═══════════════════


class TestSuppressionMeetsTheHiddenStateOnItsCard:
    """A decoder column and an activation from another card land on the hidden state's card."""

    def test_directional(self):
        from src.services.circuit_intervention_hooks import suppress_directional

        hidden = torch.zeros(1, 3, 8, device="meta")
        out = suppress_directional(hidden, torch.randn(8, 16), 2, torch.ones(1, 3), a_base=0.5,
                                   positions=torch.ones(1, 3, dtype=torch.bool))
        assert out.device == META and out.shape == (1, 3, 8)

    def test_feature_list(self):
        from src.services.circuit_intervention_hooks import suppress_feature_list

        hidden = torch.zeros(1, 3, 8, device="meta")
        out = suppress_feature_list(hidden, torch.randn(8, 16), [2, 5],
                                    [torch.ones(1, 3), torch.ones(1, 3)],
                                    positions=torch.ones(1, 3, dtype=torch.bool))
        assert out.device == META and out.shape == (1, 3, 8)

    @pytest.mark.parametrize("hidden_dtype", [torch.float32, torch.float16])
    def test_on_one_device_the_arithmetic_is_bit_identical_to_before(self, hidden_dtype):
        """The single-card formula, including an fp32 activation against an fp16 residual."""
        from src.services.circuit_intervention_hooks import suppress_directional, suppress_feature_list

        torch.manual_seed(3)
        W = torch.randn(8, 16)
        base = torch.randn(2, 5, 8).to(hidden_dtype)
        a1, a2 = torch.rand(2, 5) * 3, torch.rand(2, 5) * 3
        mask = torch.rand(2, 5) > 0.3

        expected = base.clone()
        expected.sub_(((a1 - 0.25) * mask.to(a1.dtype)).unsqueeze(-1) * W[:, 4].to(hidden_dtype))
        got = suppress_directional(base.clone(), W, 4, a1, a_base=0.25, positions=mask)
        assert torch.equal(got, expected)

        total = torch.zeros_like(base)
        for idx, a in ((4, a1), (9, a2)):
            total = total + a.unsqueeze(-1) * W[:, idx].to(hidden_dtype)
        expected_list = base.clone()
        expected_list.sub_(total)
        assert torch.equal(suppress_feature_list(base.clone(), W, [4, 9], [a1, a2]), expected_list)


class TestEncodersReadOnTheSaesCard:
    def test_the_intervention_encoder(self):
        from src.services.circuit_intervention_service import CircuitInterventionService

        sae, hidden = _sae(1).to("meta"), torch.randn(1, 4, D_MODEL)
        enc = CircuitInterventionService._encoder_for(sae, 3)
        with _OneDevicePerOp():
            a = enc(hidden)
        assert a.device == META and a.shape == (1, 4)

    def test_the_faithfulness_encoder(self):
        from src.services.circuit_faithfulness_service import CircuitFaithfulnessService

        sae, hidden = _sae(1).to("meta"), torch.randn(1, 4, D_MODEL)
        enc = CircuitFaithfulnessService._multi_encoder(sae, [3, 5])
        with _OneDevicePerOp():
            a = enc(hidden)
        assert [t.device for t in a] == [META, META]

    def test_the_intervention_downstream_read(self):
        from src.services.circuit_intervention_service import CircuitInterventionService

        model, sae = _llama(), _sae(1).to("meta")
        ids = torch.tensor([[5, 9, 11, 3]])
        with _OneDevicePerOp():
            a = CircuitInterventionService._downstream_acts(
                model, model.model.layers[1], sae, 4, ids, torch.ones_like(ids))
        assert a.device == META and a.shape == (1, 4)

    def test_the_faithfulness_downstream_read(self):
        from src.services.circuit_faithfulness_service import CircuitFaithfulnessService

        model, sae = _llama(), _sae(1).to("meta")
        ids = torch.tensor([[5, 9, 11, 3]])
        with _OneDevicePerOp():
            a = CircuitFaithfulnessService._downstream_sum(
                model, model.model.layers[1], sae, [4, 7], ids, torch.ones_like(ids))
        assert a.device == META and a.shape == (1, 4)


# ═════════════════════ INPUTS ON THE EMBEDDING'S CARD — REAL NUMBERS ══════


def _structure(model):
    from src.ml.layer_discovery import discover_transformer_structure

    return discover_transformer_structure(model)


class TestInputsGoToTheEmbeddingsCard:
    """On `embedding_not_first`, `model.device` is meta. Each run must match its unsplit twin exactly."""

    READERS_EVENTS = {
        1: {0: [(0, 1, 2.0), (0, 3, 1.5), (1, 2, 1.0)],
            5: [(1, 1, 1.2), (2, 0, 0.9)], 6: [(0, 2, 1.1), (2, 4, 1.3)]},
        2: {4: [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 0.5)]},
    }

    def _edge_args(self, model):
        from src.ml.layer_discovery import get_hookable_module
        from src.services.circuit_intervention_service import CircuitInterventionService
        from src.services.steering_service import resolve_decoder_weight

        readers = {L: _Reader(events) for L, events in self.READERS_EVENTS.items()}
        scope = CircuitInterventionService.create_scope({"k": 1, "prompts_per_edge": 3, "null_samples": 2})
        return (model, _structure(model), get_hookable_module, {1: _sae(1), 2: _sae(2)},
                readers, scope, resolve_decoder_weight)

    def _validate_edge(self, model):
        from src.services.circuit_intervention_service import CircuitInterventionService

        model, structure, ghm, saes, readers, scope, rdw = self._edge_args(model)
        return CircuitInterventionService._validate_edge(
            model, structure, ghm, saes, readers, _dataset(), TOKENIZER,
            {"layer": 1, "feature_idx": 0}, {"layer": 2, "feature_idx": 4},
            scope, {1: "sae_l1", 2: "sae_l2"}, CPU, rdw)

    def _null(self, model):
        from src.services.circuit_intervention_service import CircuitInterventionService

        model, structure, ghm, saes, readers, scope, rdw = self._edge_args(model)
        return CircuitInterventionService._null_effect_sizes(
            model, structure, ghm, saes, readers[1], readers[2], _dataset(), TOKENIZER,
            1, 2, 4, [0, 1], 0.5, scope, rdw, exclude=0)

    def test_validation_edge(self):
        clean = self._validate_edge(_llama())

        assert clean["n_prompts"] == 2 and clean["effect_size"] != 0.0, "the fixture measured nothing"
        assert self._validate_edge(embedding_not_first()) == clean

    def test_validation_null(self):
        """The shuffled non-edge null runs its own forward passes, with its own input line."""
        clean = self._null(_llama())

        assert len(clean) == 2 and any(es != 0.0 for es in clean), "the null measured nothing"
        assert self._null(embedding_not_first()) == clean

    def _behavior(self, model, suppress):
        from src.ml.layer_discovery import get_hookable_module
        from src.services.circuit_capture_service import _pad_batch
        from src.services.circuit_faithfulness_service import CircuitFaithfulnessService
        from src.services.steering_service import resolve_decoder_weight

        return CircuitFaithfulnessService._behavior(
            suppress=suppress, model=model, structure=_structure(model),
            get_hookable_module=get_hookable_module, saes={1: _sae(1), 2: _sae(2)}, readers={},
            dataset=_dataset(), tokenizer=TOKENIZER, doc_ids=[0, 1, 2], down_layer=2,
            down_features=[4, 7], device=CPU, resolve_decoder_weight=resolve_decoder_weight,
            cancel_check=None, _pad_batch=_pad_batch)

    def test_faithfulness_behavior(self):
        clean = self._behavior(_llama(), {1: [0, 3]})
        assert clean != self._behavior(_llama(), {}), "suppression changed nothing — the fixture is inert"
        assert self._behavior(embedding_not_first(), {1: [0, 3]}) == clean

    def _attribute(self, model, *, placement, device, monkeypatch):
        from src.services.circuit_attribution_service import CircuitAttributionService

        peaks = {str(CUDA1): 100 * 2**20, str(CUDA0): 50 * 2**20}
        resets = []
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d=None: resets.append(d))
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: peaks[str(d)])
        rows = _rows()
        db = _Db(rows)
        saes = {1: _sae(1), 2: _sae(2)}
        with patch("datasets.load_from_disk", return_value=_dataset()), \
             patch("src.ml.model_loader.load_model_from_hf", return_value=(model, TOKENIZER, None, {})), \
             patch("src.services.circuit_capture_service._load_sae_sync",
                   side_effect=lambda record, d: saes[record.layer].to(d)), \
             patch("src.services.extraction_service.cleanup_gpu_memory"):
            CircuitAttributionService.run(db, "dsc_1", prompt_limit=3, device=device, placement=placement)
        from src.models.circuit_runs import CircuitDiscoveryRun

        run = rows[CircuitDiscoveryRun]["dsc_1"]
        return [c["attribution"]["score"] for c in run.candidates], run.report["attribution"], resets

    def test_attribution_scores_and_every_cards_peak(self, monkeypatch):
        clean, _report, _ = self._attribute(_llama(), placement=None, device=CPU, monkeypatch=monkeypatch)
        split, report, resets = self._attribute(embedding_not_first(), placement=SPLIT, device=CUDA1,
                                                monkeypatch=monkeypatch)

        assert all(score != 0.0 for score in clean), "attribution measured nothing"
        assert split == clean
        assert resets == [CUDA1, CUDA0], "peak stats were not reset on every card of the split"
        assert report["peak_vram_mb"] == 150.0
        assert report["peak_vram_mb_by_device"] == {"cuda:1": 100.0, "cuda:0": 50.0}

    def test_the_capture_probe(self):
        from src.services import circuit_capture_service as cap

        def probe(model):
            probe_max, events, tokens = cap._probe(model, TOKENIZER, _dataset(), {1: _sae(1), 2: _sae(2)},
                                                   [1, 2], CPU)
            return {L: t.tolist() for L, t in probe_max.items()}, events, tokens

        clean = probe(_llama())
        assert clean[1] > 0 and clean[2] == 18
        assert probe(embedding_not_first()) == clean

    def test_the_full_capture_hands_the_batch_the_embeddings_card(self, monkeypatch, tmp_path):
        from src.services import circuit_capture_service as cap

        model = embedding_not_first()
        seen = []

        def record(model_, saes, layers, writers, input_ids, mask, *rest, **kw):
            seen.append((input_ids.device, mask.device))
            raise _Stop

        monkeypatch.setattr(cap, "captures_dir", lambda: tmp_path)
        monkeypatch.setattr(cap, "MIN_FREE_DISK_BYTES", 0)
        saes = {1: _sae(1), 2: _sae(2)}
        with patch("datasets.load_from_disk", return_value=_dataset()), \
             patch("src.ml.model_loader.load_model_from_hf", return_value=(model, TOKENIZER, None, {})), \
             patch.object(cap, "_load_sae_sync", side_effect=lambda record, d: saes[record.layer].to(d)), \
             patch.object(cap, "_capture_batch", side_effect=record), \
             patch("src.services.extraction_service.cleanup_gpu_memory"), \
             pytest.raises(_Stop):
            cap.CircuitCaptureService.run_capture(_Db(_rows()), "cap_1", confirmed=True, device=CPU)

        assert seen == [(CPU, CPU)]
