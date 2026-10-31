"""Held-out evaluation is chunked, exact, bounded in memory, and drawn from every source.

SAE training remediation item 2 (2026-09-15). The in-training held-out check
encoded every held-out token in ONE pass — ~6 GiB per latent-sized tensor at
100,000 tokens and 16,384 latents, an OOM beside a training buffer that was
caught and logged, so the run simply had no held-out number. And the "sample"
was each extraction's lowest positions until the cap: the first rows of the
first extraction.

These tests hold the replacement to four promises:

* chunking changes nothing — every metric of a chunked pass equals the
  unchunked pass, and each FVU equals an independent oracle of its own formula
  (fixtures include a constant-offset dimension, where the two formulas differ,
  so a swap of the keys cannot pass);
* the legacy ``fvu`` is the number ``JumpReLUSAE.forward`` reports, so stored
  history keeps its meaning;
* one chunk's allocations stay under ``holdout_eval_peak_bytes``, measured on
  CPU tensors for a 16,384-latent x 3-layer configuration;
* the sample is split across sources by weight (equal when unset) and taken as
  whole rows in a seeded order, never lowest-first.

MUTATION CONTROLS (2026-09-15). Each broke one line, ran the listed tests, and was
restored from the original bytes with the sha256 and `git diff` re-checked. All KILLED.
  D1   holdout_quotas proportional to size when no weights are set (weights=None)
         -> test_quotas_are_equal_without_weights_not_proportional_to_size,
            test_a_source_that_cannot_fill_its_share_hands_the_rest_on
  D2   held-out rows taken lowest first (range instead of holdout_row_order)
         -> test_whole_rows_in_a_seeded_order_not_lowest_first,
            test_the_order_depends_on_the_seed_and_the_source, and the cached E2E
            test_each_extraction_supplies_its_weighted_share_of_whole_held_out_rows
  D5   the two FVU keys swapped in the result
         -> test_each_fvu_is_its_own_formula[x4], transcoder, legacy-forward tests (6)
  D6   the centred denominator computed with one global mean
         -> test_each_fvu_is_its_own_formula[x4], transcoder (5)
  D7   Σx² overwritten per chunk instead of accumulated
         -> every chunk-invariance and formula test (10)
  D8   the chunk size ignored (one pass over every token)
         -> test_a_16k_latent_three_layer_evaluation_stays_under_the_bound, eval-mode tests
  D9   the evaluation run under enable_grad
         -> test_it_runs_in_eval_mode_without_grad_and_restores_the_mode[x2]
  D10  the model's training mode never restored
         -> test_it_runs_in_eval_mode_..._restores_the_mode[True],
            test_the_mode_is_restored_when_the_forward_fails
  D36  the last held-out row not cut to the quota
         -> cached E2E test_each_extraction_supplies_its_weighted_share_of_whole_held_out_rows
  CONS1 (review R1-B) the accumulator never feeds FvuSums, now the one FVU definition
         -> test_each_fvu_is_its_own_formula, transcoder and chunk-invariance tests (6)
  CONS2 (review R1-B) the legacy and centred values swapped in the accumulator's result
         -> the same 6
Wiring controls for the training task (D3, D4, D11, D12, D14, D37) are recorded in
test_training_data_path_e2e.py.
"""

import weakref

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from src.ml.sparse_autoencoder import create_sae
from src.services import holdout_evaluation as HE

CPU = torch.device("cpu")


def _sae(arch: str, d: int = 12, latent: int = 48, seed: int = 0):
    torch.manual_seed(seed)
    kwargs = dict(architecture_type=arch, hidden_dim=d, latent_dim=latent, l1_alpha=1e-3)
    if arch == "jumprelu":
        kwargs.update(initial_threshold=0.05, sparsity_coeff=1e-3)
    if arch == "topk":
        kwargs.update(top_k=5)
    return create_sae(**kwargs)


def _data(n: int = 257, d: int = 12, seed: int = 1) -> torch.Tensor:
    """Tokens whose dimensions have very different means and scales.

    Dimension 0 sits on a large constant offset: the legacy FVU subtracts ONE
    scalar mean from every element, so that offset inflates its denominator and
    it reads far lower than the per-dimension (centred) FVU. A fixture without it
    lets the two formulas agree by construction.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g) * torch.linspace(0.2, 3.0, d)
    x[:, 0] += 40.0
    x[:, 1] -= 7.0
    return x


def _oracle(model, x, transcoder=False):
    """The unchunked metrics, computed directly in float64 from one forward pass."""
    with torch.no_grad():
        x_hat, f, _ = model(x, x, return_loss=False) if transcoder else model(x, return_loss=False)
    x64, xh64 = x.double(), x_hat.double()
    fvu = (x64 - xh64).var() / (x64.var() + 1e-8)
    mu = x64.mean(dim=0, keepdim=True)
    centred = ((x64 - xh64) ** 2).sum() / ((x64 - mu) ** 2).sum()
    active = (f != 0)
    return {
        "fvu": float(fvu), "fvu_centred": float(centred),
        "l0_mean": float(active.sum(dim=1).double().mean()),
        "firing_counts": active.sum(dim=0),
    }


ARCHITECTURES = ["jumprelu", "standard", "topk", "skip"]


class TestChunkingChangesNothing:
    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_every_chunk_size_gives_the_unchunked_result(self, arch):
        model = _sae(arch)
        x = _data()
        whole = HE.evaluate_holdout(model, x, CPU, chunk_tokens=x.shape[0])
        for chunk in (1, 7, 64, 256):
            part = HE.evaluate_holdout(model, x, CPU, chunk_tokens=chunk)
            assert part["n_tokens"] == whole["n_tokens"] == x.shape[0]
            for name in ("fvu", "fvu_centred", "l0_mean", "l0_sparsity", "loss_reconstruction", "loss_zero"):
                assert part[name] == pytest.approx(whole[name], rel=1e-6, abs=1e-12), (arch, chunk, name)
            assert torch.equal(part["firing_counts"], whole["firing_counts"]), (arch, chunk)

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_each_fvu_is_its_own_formula(self, arch):
        model = _sae(arch)
        x = _data()
        result = HE.evaluate_holdout(model, x, CPU, chunk_tokens=10)
        oracle = _oracle(model, x)
        assert result["fvu"] == pytest.approx(oracle["fvu"], rel=1e-6)
        assert result["fvu_centred"] == pytest.approx(oracle["fvu_centred"], rel=1e-6)
        assert result["l0_mean"] == pytest.approx(oracle["l0_mean"], rel=1e-12)
        assert torch.equal(result["firing_counts"], oracle["firing_counts"])

    def test_the_fixture_separates_the_two_fvus(self):
        """Precondition for the test above: here a swapped key is a large error."""
        model = _sae("jumprelu")
        oracle = _oracle(model, _data())
        assert oracle["fvu_centred"] > 3 * oracle["fvu"], oracle

    def test_the_legacy_fvu_is_what_the_jumprelu_forward_reports(self):
        """Stored `fvu` values keep their meaning: the same number, chunked."""
        model = _sae("jumprelu")
        x = _data()
        with torch.no_grad():
            _, _, losses = model(x, return_loss=True)
        result = HE.evaluate_holdout(model, x, CPU, chunk_tokens=16)
        assert result["fvu"] == pytest.approx(float(losses["fvu"]), rel=1e-4)
        assert result["l0_mean"] == pytest.approx(float(losses["l0_mean"]), rel=1e-6)
        assert result["loss_reconstruction"] == pytest.approx(float(losses["loss_reconstruction"]), rel=1e-4)
        assert result["loss_zero"] == pytest.approx(float(losses["loss_zero"]), rel=1e-4)
        assert result["loss_l0"] == pytest.approx(float(losses["loss_l0"]), rel=1e-6)

    def test_a_transcoder_is_evaluated_against_itself(self):
        model = _sae("transcoder")
        x = _data()
        whole = HE.evaluate_holdout(model, x, CPU, chunk_tokens=x.shape[0], transcoder=True)
        part = HE.evaluate_holdout(model, x, CPU, chunk_tokens=9, transcoder=True)
        oracle = _oracle(model, x, transcoder=True)
        assert part["fvu_centred"] == pytest.approx(whole["fvu_centred"], rel=1e-6)
        assert part["fvu_centred"] == pytest.approx(oracle["fvu_centred"], rel=1e-6)


class TestTheEvaluationLeavesTrainingAlone:
    @pytest.mark.parametrize("training", [True, False])
    def test_it_runs_in_eval_mode_without_grad_and_restores_the_mode(self, training):
        model = _sae("jumprelu")
        model.train(training)
        seen = []
        handle = model.register_forward_hook(
            lambda m, i, o: seen.append((m.training, torch.is_grad_enabled()))
        )
        try:
            HE.evaluate_holdout(model, _data(n=40), CPU, chunk_tokens=16)
        finally:
            handle.remove()
        assert seen == [(False, False)] * 3, seen
        assert model.training is training

    def test_the_mode_is_restored_when_the_forward_fails(self):
        model = _sae("jumprelu")
        model.train()

        def boom(*a, **k):
            raise RuntimeError("out of memory")

        handle = model.register_forward_pre_hook(boom)
        try:
            with pytest.raises(RuntimeError):
                HE.evaluate_holdout(model, _data(n=8), CPU, chunk_tokens=4)
        finally:
            handle.remove()
        assert model.training is True


class _LiveTensorBytes(TorchDispatchMode):
    """Peak bytes of CPU tensor storage allocated inside the mode and still alive.

    Every operator's outputs are tracked by their storage; a storage is released
    when the last tracked tensor on it is collected. Storages that existed before
    (the SAE's parameters, the held-out inputs) are excluded, so the peak is what
    the evaluation itself allocates.
    """

    def __init__(self, preexisting):
        super().__init__()
        self._exclude = set(preexisting)
        self._live = {}
        self.current = 0
        self.peak = 0

    def _track(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        storage = tensor.untyped_storage()
        ptr = storage.data_ptr()
        if ptr == 0 or ptr in self._exclude:
            return
        entry = self._live.get(ptr)
        if entry is None:
            entry = self._live[ptr] = [storage.nbytes(), set()]
            self.current += entry[0]
            self.peak = max(self.peak, self.current)
        if id(tensor) not in entry[1]:
            entry[1].add(id(tensor))
            weakref.finalize(tensor, self._release, ptr, id(tensor))

    def _release(self, ptr, tensor_id):
        entry = self._live.get(ptr)
        if entry is None:
            return
        entry[1].discard(tensor_id)
        if not entry[1]:
            self.current -= entry[0]
            del self._live[ptr]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for tensor in out if isinstance(out, (tuple, list)) else (out,):
            self._track(tensor)
        return out


def _storages(*tensors):
    return {t.untyped_storage().data_ptr() for t in tensors}


class TestTheMemoryIsBounded:
    """16,384 latents x 3 layers, measured on CPU allocations.

    The stated bound is ``holdout_eval_peak_bytes(chunk, d, latents)`` — per
    chunk, because layers are evaluated one after another. For the production
    16K configuration (d = 2,048, the default 2,048-token chunk) that is
    ``holdout_eval_peak_bytes(2048, 2048, 16384)`` = 768 MiB, against ~18 GiB of
    latent-sized tensors for the unchunked 100,000-token pass it replaced. The
    width here is small only to keep the SAEs' weights small; the latent terms,
    which dominate, are at full size.
    """

    LATENTS = 16_384
    WIDTH = 64
    CHUNK = 256
    LAYERS = 3

    def _run(self, chunk):
        saes = [_sae("jumprelu", d=self.WIDTH, latent=self.LATENTS, seed=s) for s in range(self.LAYERS)]
        held = [_data(n=3 * self.CHUNK + 17, d=self.WIDTH, seed=s) for s in range(self.LAYERS)]
        preexisting = set()
        for sae in saes:
            preexisting |= _storages(*sae.parameters(), *sae.buffers())
        preexisting |= _storages(*held)
        meter = _LiveTensorBytes(preexisting)
        with meter:
            for sae, x in zip(saes, held):
                HE.evaluate_holdout(sae, x, CPU, chunk_tokens=chunk)
        return meter.peak

    def test_a_16k_latent_three_layer_evaluation_stays_under_the_bound(self):
        peak = self._run(self.CHUNK)
        bound = HE.holdout_eval_peak_bytes(self.CHUNK, self.WIDTH, self.LATENTS)
        assert 0 < peak <= bound, f"peak {peak / 2**20:.1f} MiB over the {bound / 2**20:.1f} MiB bound"
        assert HE.holdout_eval_peak_bytes(2048, 2048, 16384) == 768 * 2**20

    def test_the_meter_sees_an_unchunked_pass_break_the_bound(self):
        """NEGATIVE CONTROL: a meter that measured nothing would pass the test above."""
        tokens = 3 * self.CHUNK + 17
        peak = self._run(tokens)
        assert peak > HE.holdout_eval_peak_bytes(self.CHUNK, self.WIDTH, self.LATENTS)


class TestTheSampleSpansEverySource:
    def test_quotas_follow_the_weights(self):
        assert HE.holdout_quotas([10_000, 10_000, 10_000], 4_000, [2.0, 1.0, 1.0]) == [2_000, 1_000, 1_000]

    def test_quotas_are_equal_without_weights_not_proportional_to_size(self):
        assert HE.holdout_quotas([90_000, 10_000], 10_000) == [5_000, 5_000]

    def test_a_source_that_cannot_fill_its_share_hands_the_rest_on(self):
        assert HE.holdout_quotas([1_000, 50_000], 10_000) == [1_000, 9_000]

    def test_a_source_with_no_held_out_tokens_gets_none(self):
        assert HE.holdout_quotas([0, 50_000, 50_000], 10_000) == [0, 5_000, 5_000]

    @staticmethod
    def _held(rows, seq, drop_every=0):
        flat = np.arange(rows * seq, dtype=np.int64)
        if drop_every:
            flat = flat[(flat % seq) % drop_every != 0]
        return flat

    def test_whole_rows_in_a_seeded_order_not_lowest_first(self):
        seq = 16
        held = self._held(200, seq) + 5 * seq  # held rows 5..204
        picked = HE.select_holdout_positions(held, seq, 10 * seq, seed=3, source_index=0)
        rows = np.unique(picked // seq)
        assert picked.size == 10 * seq
        assert rows.size == 10, "whole rows are taken, not scattered positions"
        assert not np.array_equal(rows, np.arange(5, 15)), "the lowest held-out rows were taken first"
        for row in rows:
            assert np.array_equal(picked[picked // seq == row], np.arange(row * seq, (row + 1) * seq))
        assert np.array_equal(picked, np.sort(picked))

    def test_the_quota_is_met_exactly_by_cutting_the_last_row(self):
        seq = 16
        held = self._held(50, seq, drop_every=3)  # rows with holes, as a real mask leaves them
        picked = HE.select_holdout_positions(held, seq, 100, seed=1, source_index=2)
        assert picked.size == 100
        assert np.isin(picked, held).all()

    def test_the_order_depends_on_the_seed_and_the_source(self):
        seq = 8
        held = self._held(400, seq)
        a = HE.select_holdout_positions(held, seq, 5 * seq, seed=1, source_index=0)
        b = HE.select_holdout_positions(held, seq, 5 * seq, seed=1, source_index=0)
        c = HE.select_holdout_positions(held, seq, 5 * seq, seed=2, source_index=0)
        e = HE.select_holdout_positions(held, seq, 5 * seq, seed=1, source_index=1)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c) and not np.array_equal(a, e)

    def test_a_quota_larger_than_the_source_takes_everything_once(self):
        seq = 4
        held = self._held(6, seq)
        picked = HE.select_holdout_positions(held, seq, 10_000, seed=0, source_index=0)
        assert np.array_equal(picked, held)
