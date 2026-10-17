"""A J-lens FIT is placed for the memory its batched backward takes, not a forward pass's 2 GiB.

Multi-GPU Phase 2, review round 3 (2026-09-14). Round 2 (a6dd8d07) placed every J-lens
task at its weights plus the loader preflight's fixed 2 GiB of activation headroom. That
is a forward pass's allowance. A fit keeps the whole forward graph for a backward that
differentiates chunks of output dimensions at once (`JacobianFitter._fit_one`,
`is_grads_batched=True`, `retain_graph=True`), and holds the gradients of every captured
layer together. Bounded from the fitter's own arithmetic and checked against the real
`_fit_one` under torch's MemTracker, the default fit (550-character prompts, about 128
tokens, every layer to the penultimate captured) needs 3.1 GiB beyond the weights on
gemma-4-12b-it and 2.9 GiB on OLMo-2-13B. A fit may not split, so Auto put one on a card
with 2 GiB to spare and it ran out of memory after loading.

`jlens_fitter.fit_working_bytes` is the bound; `jlens_model_registry.estimate_fit_working_mb`
applies it to a model row and the longest prompt, read with the model's own tokenizer;
the fit task hands it to `jlens_progress.place_on_card` as `headroom_mb`.

MUTATION CONTROLS (review round 3, 2026-09-14; scratchpad p2-r3/mutate.py + mutations_r3.json,
each alone in a private copy of backend/, restored by sha256). All killed:
  N26 place_on_card ignores headroom_mb                        -> the 3090 refusal + larger_of[3164]
  N27 headroom_mb replaces 2 GiB instead of raising it         -> larger_of[1000-2048]
  N28 the fit task passes no headroom_mb                       -> the_fit_task_places_with_its_own_estimate...
  N29 the estimate reads the first prompt, not the longest     -> the_estimate_reads..., the fit task case
  N30 the estimate ignores the layers the fit captures         -> the_layers_it_fits..., the fit task case
  N30b the estimate ignores target_layer="final"               -> the_layers_it_fits_and_its_target...
  N30c every row's activations sized at 4 bytes                -> four estimate cases
  N30d an unreadable tokenizer counts one token                -> ..._counts_a_token_a_character
  N31 the graph term at (6 d + 2 i)                            -> the 12B figure
  N32 the logits term dropped                                  -> the 12B figure
  N33 the captured gradients counted for one layer             -> the 12B figure + the_bound_holds[256-1024-8-...-7]
  N34 the fp32 temporaries dropped                             -> the 12B figure + the_bound_holds[...-7]
  N35 one layer's batched backward dropped                     -> the 12B figure + five the_bound_holds cases
  N36 the fp32 cotangent dropped                               -> the 12B figure
Round 2's C1 (the headroom line reverted, re-expressed on `needed_mb`) and C2 (place_job handed
the bare weights) re-run: both killed (14 and 10 red across this module,
test_jlens_placement_keeps_activation_headroom, test_jlens_gpu_placement, test_jlens_split_gpu).
N32 and N36 are caught only by the 12B figure, not by a measured shape: on the small models
MemTracker can run, the logits and the cotangent are margin beside the graph and the backward. The
figure pins them so the bound cannot quietly lose a term that matters at 12B scale. N33 and N34 are
also caught by the measured 8-layer shape with seven captured layers.
"""

from __future__ import annotations

import contextlib
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed._tools.mem_tracker import MemTracker
from transformers import AutoModelForCausalLM, LlamaConfig

from src.ml.jlens_fitter import JacobianFitter, fit_working_bytes
from src.ml.layer_discovery import discover_transformer_structure
from src.services import gpu_placement, jlens_model_registry
from src.services.gpu_placement import GpuCard, Placement
from src.workers import jlens_progress

MIB = 1024 * 1024


# ── The bound holds against the real fitter ─────────────────────────────────


class _Tokens:
    def __init__(self, seq_len, vocab):
        self.seq_len, self.vocab = seq_len, vocab

    def __call__(self, prompt, return_tensors="pt"):
        generator = torch.Generator().manual_seed(0)
        return {"input_ids": torch.randint(0, self.vocab, (1, self.seq_len), generator=generator)}


def _measured(d, i, n_layers, seq_len, vocab, heads, n_captured, dtype):
    config = LlamaConfig(hidden_size=d, intermediate_size=i, num_hidden_layers=n_layers, num_attention_heads=heads,
                         num_key_value_heads=heads, vocab_size=vocab, tie_word_embeddings=False,
                         attn_implementation="sdpa")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(copy.deepcopy(config), dtype=dtype).eval()
    target = n_layers - 2
    captured = list(range(target + 1 - n_captured, target + 1))
    fitter = JacobianFitter(model, _Tokens(seq_len, vocab), discover_transformer_structure(model), min_prompts=1)
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    tracker = MemTracker()
    tracker.track_external(model)
    with tracker:
        fitter._fit_one("prompt", captured)
    peak = tracker.get_tracker_snapshot("peak")
    total = sum(value for device, categories in peak.items() for name, value in categories.items()
                if str(device) == "cpu" and name == "Total")
    return total - weights


@pytest.mark.parametrize("d, i, n_layers, seq_len, vocab, heads, n_captured, dtype, activation_bytes", [
    (128, 512, 4, 64, 2000, 4, 3, torch.bfloat16, 2),
    (128, 512, 4, 128, 2000, 4, 3, torch.bfloat16, 2),
    (256, 1024, 8, 128, 4000, 8, 7, torch.bfloat16, 2),
    (256, 1024, 8, 128, 4000, 8, 2, torch.bfloat16, 2),
    (128, 512, 4, 128, 2000, 4, 3, torch.float32, 4),
    # A narrow model with a wide vocabulary and one captured layer: the peak is the
    # forward's end, the head's logits beside the graph, not the backward.
    (64, 128, 4, 256, 64_000, 4, 1, torch.bfloat16, 2),
])
def test_the_bound_holds_the_real_fits_peak_without_being_decorative(
    d, i, n_layers, seq_len, vocab, heads, n_captured, dtype, activation_bytes
):
    measured = _measured(d, i, n_layers, seq_len, vocab, heads, n_captured, dtype)
    bound = fit_working_bytes(d_model=d, intermediate=i, layers_to_target=n_layers - 1, n_captured=n_captured,
                              max_seq_len=seq_len, vocab=vocab, heads=heads, activation_bytes=activation_bytes)

    assert measured <= bound, f"the fit took {measured / MIB:.1f} MiB, over its bound of {bound / MIB:.1f}"
    assert bound <= 3 * measured, f"a bound of {bound / MIB:.1f} MiB for {measured / MIB:.1f} MiB refuses fits that fit"


def test_a_12b_fit_at_the_default_prompt_length_needs_more_than_a_forward_passs_headroom():
    """gemma-4-12b-it's text stack at 128 tokens, every layer to the penultimate captured."""
    bound = fit_working_bytes(d_model=3840, intermediate=15_360, layers_to_target=47, n_captured=47,
                              max_seq_len=128, vocab=262_144, heads=16, activation_bytes=2)

    assert bound / MIB > 2 * 1024
    assert bound / MIB == pytest.approx(3_164, abs=40)


# ── A model row and its prompts ─────────────────────────────────────────────

GEMMA = {"hidden_size": 3840, "num_hidden_layers": 48, "vocab_size": 262_144, "num_attention_heads": 16,
         "intermediate_size": 15_360}


def _row(**arch):
    return SimpleNamespace(id="m_1", repo_id="org/gemma", quantization="Q8", params_count=12_000_000_000,
                           file_path=None, architecture_config=dict(GEMMA, **arch))


class _WordTokenizer:
    def __call__(self, text):
        return {"input_ids": list(range(len(text.split())))}


@pytest.fixture
def words(monkeypatch):
    monkeypatch.setattr(jlens_model_registry, "tokenizer_for", lambda record: _WordTokenizer())


def test_the_estimate_reads_the_rows_dimensions_and_its_longest_prompt_in_tokens(words):
    got = jlens_model_registry.estimate_fit_working_mb(_row(), ["one two three", "one two three four five"])

    assert got == pytest.approx(fit_working_bytes(
        d_model=3840, intermediate=15_360, layers_to_target=47, n_captured=47, max_seq_len=5,
        vocab=262_144, heads=16, activation_bytes=2) / MIB)


def test_the_layers_it_fits_and_its_target_change_the_estimate(words):
    got = jlens_model_registry.estimate_fit_working_mb(_row(), ["a b c d"], layers=[40, 41], target_layer="final")

    assert got == pytest.approx(fit_working_bytes(
        d_model=3840, intermediate=15_360, layers_to_target=48, n_captured=2, max_seq_len=4,
        vocab=262_144, heads=16, activation_bytes=2) / MIB)


def test_a_row_without_its_dimensions_gets_no_estimate(words):
    assert jlens_model_registry.estimate_fit_working_mb(_row(num_attention_heads=None), ["a"]) is None


def test_a_prompt_whose_tokenizer_cannot_be_read_counts_a_token_a_character(monkeypatch):
    def unavailable(record):
        raise jlens_model_registry.ModelNotAvailable("not downloaded")

    monkeypatch.setattr(jlens_model_registry, "tokenizer_for", unavailable)

    got = jlens_model_registry.estimate_fit_working_mb(_row(), ["abcdefgh"])

    assert got == pytest.approx(fit_working_bytes(
        d_model=3840, intermediate=15_360, layers_to_target=47, n_captured=47, max_seq_len=8,
        vocab=262_144, heads=16, activation_bytes=2) / MIB)


# ── The placement asks for it ───────────────────────────────────────────────

TI = GpuCard(index=0, uuid="GPU-f47ba814-49a2-603f-3595-275284140251", name="NVIDIA GeForce RTX 3080 Ti",
             total_mb=12_288, free_mb=11_000)
RTX = GpuCard(index=1, uuid="GPU-247aa582-0d1b-e161-8156-983ed1fefc57", name="NVIDIA GeForce RTX 3090",
              total_mb=24_576, free_mb=23_000)


@pytest.mark.parametrize("headroom_mb, added_mb", [(3_164.0, 3_164.0), (1_000.0, 2_048.0), (None, 2_048.0)])
def test_the_placement_takes_the_larger_of_the_headroom_and_a_forward_passs(headroom_mb, added_mb):
    asked = []

    def place_job(requested, required_mb=None, allow_shard=False):
        asked.append(required_mb)
        return Placement(card=RTX, device=torch.device("cuda", 1))

    with patch.object(gpu_placement, "place_job", place_job), \
         patch.object(jlens_progress, "record_gpu", lambda *a, **k: True):
        jlens_progress.place_on_card("t-fit", "auto", required_mb=20_000.0, headroom_mb=headroom_mb)

    assert asked == [20_000.0 + added_mb]


@contextlib.contextmanager
def _two_cards():
    order = [TI.uuid[len("GPU-"):], RTX.uuid[len("GPU-"):]]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.is_available", return_value=True))
        stack.enter_context(patch("torch.cuda.device_count", return_value=2))
        stack.enter_context(patch("torch.cuda.get_device_properties",
                                  side_effect=lambda index: SimpleNamespace(uuid=order[index])))
        stack.enter_context(patch("torch.cuda.set_device"))
        stack.enter_context(patch.object(gpu_placement, "list_cards", lambda: [TI, RTX]))
        stack.enter_context(patch.object(jlens_progress, "record_gpu", lambda *a, **k: True))
        yield


def test_a_fit_the_3090_holds_only_with_a_forward_passs_headroom_is_refused_not_started():
    """20,000 MB of weights: with 2 GiB the 3090's 23,000 MB free takes the fit, and its
    backward then runs out of memory; with the fit's own 3,164 MiB no card holds it, and a
    fit may not split."""
    with _two_cards():
        forward_only = jlens_progress.place_on_card("t-fit", "auto", required_mb=20_000.0)
        assert not forward_only.is_shard and forward_only.uuid == RTX.uuid, "precondition"
        with pytest.raises(gpu_placement.GpuPlacementError, match="No single GPU"):
            jlens_progress.place_on_card("t-fit", "auto", required_mb=20_000.0, headroom_mb=3_164.0)


class _Stop(Exception):
    pass


def test_the_fit_task_places_with_its_own_estimate_for_its_prompts_and_layers(words):
    from src.workers.jlens_fit_tasks import fit_jlens_artifact

    record = _row()
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = record
    placed = []

    @contextlib.contextmanager
    def fake_db():
        yield db

    def place_on_card(task_id, gpu_request, **kwargs):
        placed.append(kwargs)
        raise _Stop()

    prompts = ["one two three", "one two three four five six"]
    fit_jlens_artifact.push_request(id="t-fit")
    try:
        with patch("src.core.database.get_sync_db", fake_db), \
             patch.object(jlens_progress, "place_on_card", place_on_card), \
             patch.object(jlens_progress, "update_row", MagicMock()):
            with pytest.raises(_Stop):
                fit_jlens_artifact.run(model_id="m_1", prompts=prompts, layers=[44, 45], target_layer="penultimate")
    finally:
        fit_jlens_artifact.pop_request()

    assert placed == [{
        "required_mb": pytest.approx(jlens_model_registry.estimate_weights_mb(record)),
        "headroom_mb": pytest.approx(fit_working_bytes(
            d_model=3840, intermediate=15_360, layers_to_target=47, n_captured=2, max_seq_len=6,
            vocab=262_144, heads=16, activation_bytes=2) / MIB),
    }]
