"""The SAE's cost to the MODEL, not to its own reconstruction objective.

WHY THIS FILE EXISTS. Nothing ever put an SAE reconstruction back into the
model. Every reported number — FVU, L0, dead count — lives in the SAE's own
space, and reconstruction error is not uniformly important: an SAE can reach a
low FVU and still wreck the next-token distribution, because the directions that
carry the output are a small share of the variance.

The failure mode this file guards hardest is the FORWARD HOOK. A hook left
attached changes every subsequent forward pass, so the symptom is a slow
unexplained quality drift, not an error.

REMEDIATION ITEM 6 (2026-09-15). The ablation baseline substituted ``b_dec`` — a
vector in the SAE's NORMALISED space (per-token norm sqrt(d) under
``constant_norm_rescale``) — as a RAW activation, so "ablated" measured an
arbitrary perturbation. The baseline is now the layer's mean activation over the
evaluation's REAL tokens; zero ablation is kept for reference. ``TestMeanAblation``
is the guard: its fixture puts ``b_dec`` and the raw mean far apart, and puts
padding activations far from real ones, so both old defects change the answer.

MUTATION CONTROLS — Phase 2, the SAE on its own card (2026-09-14; carried over,
re-pointed at the new functions). All went red:
  S1 reconstruct_at_layer runs sae(x) on the layer's card    -> the_activation_goes_to_the_sae..., the_evaluation_feeds_the_sae...
  S2 reconstruction left on the SAE's card (dtype only)      -> the_activation_goes_to_the_sae_and_the_reconstruction_comes_back
  S3 mean left on the SAE's card (dtype only)                -> the_mean_comes_to_the_layers_card
  S4 the evaluation inlines sae(x) instead of the helper     -> the_evaluation_feeds_the_sae_on_the_sae_card

MUTATION CONTROLS — remediation item 6 (2026-09-15; each applied alone, this file run,
source restored and verified by sha256). All went red:
  A1 stats and mean over every position (padding included)  -> TestMeanAblation::it_is_the_real_token_mean..., padding_does_not_enter_the_mean,
                                                               a_perfect_sae_costs_a_real_model_nothing
  A2 ablation substitutes the SAE's b_dec again             -> it_is_not_the_saes_normalised_bias, it_is_the_real_token_mean..., precision,
                                                               the_evaluation_feeds_the_sae..., answer_does_not_depend_on_batching
  A3 no dtype cast into the SAE                             -> TestPrecision::test_a_half_precision_model_meets_a_full_precision_sae
  A4 splice hook never removed                              -> all four TestTheHooksAlwaysComeOff cases that splice, plus three others
  A5 base CE from the last batch only (= instead of +=)     -> a_perfect_sae_costs_a_real_model_nothing, answer_does_not_depend_on_batching
  A7 mean left on its own card                              -> TestMeanAblation::test_the_mean_comes_to_the_layers_card
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.services.sae_evaluation import (
    evaluate_spliced_layers,
    loss_recovered,
    mean_ablation_at_layer,
    prediction_mask,
    reconstruct_at_layer,
)


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _TinyModel(nn.Module):
    """A real nn.Module so hooks behave as they do in production."""

    def __init__(self, vocab=11, d=4, scale=1.0):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, d)
        self.layer = nn.Linear(d, d)
        self.head = nn.Linear(d, vocab)
        self.scale = scale
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.forward_calls += 1
        h = self.layer(self.embed(input_ids)) * self.scale
        return _Out(self.head(h))


class _PerfectSAE(nn.Module):
    """Round-trips exactly: the reconstruction costs the model nothing. Two latents per token."""

    def __init__(self, d=4):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        z = torch.zeros(*x.shape[:-1], 6, device=x.device)
        z[..., :2] = 1.0
        return (x, z, {})


class _ZeroSAE(nn.Module):
    """Reconstructs everything as zeros — exactly as bad as zero ablation."""

    def __init__(self, d=4):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return (torch.zeros_like(x), None, {})


def _single(model, sae, input_ids, attention_mask):
    return evaluate_spliced_layers(
        model, {0: sae}, {0: model.layer}, lambda: [(input_ids, attention_mask)]
    )


@pytest.fixture
def batch():
    torch.manual_seed(0)
    input_ids = torch.randint(0, 11, (2, 6))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 4:] = 0  # one padded row, so masking is exercised
    return input_ids, attention_mask


def _llama(dtype=torch.float32):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2,
    )
    return LlamaForCausalLM(config).eval().to(dtype)


def _llama_batch():
    ids = torch.randint(0, 64, (3, 12), generator=torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    mask[2, 7:] = 0
    return ids, mask


class TestTheHooksAlwaysComeOff:
    """The most dangerous defect here is silent and permanent."""

    def test_no_hook_survives_the_measurement(self, batch):
        model = _TinyModel()
        _single(model, _PerfectSAE(), *batch)
        assert len(model.layer._forward_hooks) == 0, (
            "a forward hook is still attached; every later forward pass is now silently modified"
        )

    def test_the_model_is_unchanged_afterwards(self, batch):
        model = _TinyModel()
        input_ids, attention_mask = batch
        with torch.no_grad():
            before = model(input_ids, attention_mask=attention_mask).logits.clone()
        _single(model, _ZeroSAE(), *batch)
        with torch.no_grad():
            after = model(input_ids, attention_mask=attention_mask).logits
        torch.testing.assert_close(before, after)

    @pytest.mark.parametrize("fail_on_call", [1, 2], ids=["capture_pass", "splice_pass"])
    def test_the_hooks_come_off_even_when_the_sae_raises(self, batch, fail_on_call):
        model = _TinyModel()

        class _Exploding(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                if self.calls >= fail_on_call:
                    raise RuntimeError("boom")
                return (x, None, {})

        with pytest.raises(RuntimeError):
            _single(model, _Exploding(), *batch)
        assert len(model.layer._forward_hooks) == 0

    def test_training_mode_is_restored(self, batch):
        model = _TinyModel()
        model.train()
        _single(model, _PerfectSAE(), *batch)
        assert model.training, "the model was left in eval mode"

    def test_a_real_model_keeps_no_hooks(self):
        model = _llama()
        ids, mask = _llama_batch()
        saes = {L: _PerfectSAE(32) for L in (1, 2)}
        evaluate_spliced_layers(model, saes, {L: model.model.layers[L] for L in saes},
                                lambda: [(ids, mask)])
        assert sum(len(layer._forward_hooks) for layer in model.model.layers) == 0


class TestTheNumbersMeanSomething:

    def test_a_perfect_sae_costs_a_real_model_nothing(self):
        model = _llama()
        ids, mask = _llama_batch()
        saes = {L: _PerfectSAE(32) for L in (1, 2)}
        out = evaluate_spliced_layers(model, saes, {L: model.model.layers[L] for L in saes},
                                      lambda: [(ids[:2], mask[:2]), (ids[2:], mask[2:])])
        for layer in out["layers"]:
            assert layer["ce_delta"] == pytest.approx(0.0, abs=1e-5)
            assert layer["kl"] == pytest.approx(0.0, abs=1e-6)
            assert layer["loss_recovered_vs_mean"] == pytest.approx(1.0, abs=1e-4)
            assert layer["l0"] == pytest.approx(2.0)
            assert layer["fvu_centred"] == pytest.approx(0.0, abs=1e-9)
        assert out["all_layers_spliced"]["ce_delta"] == pytest.approx(0.0, abs=1e-5)
        assert out["all_layers_spliced"]["layers"] == [1, 2]
        real = int(mask.sum())
        assert out["tokens"] == real
        assert out["predicted_tokens"] == int(prediction_mask(ids, mask).sum())

    def test_an_sae_that_outputs_zeros_recovers_nothing_against_zero_ablation(self, batch):
        """NEGATIVE CONTROL: as bad as the zero floor must score exactly 0 against it."""
        out = _single(_TinyModel(), _ZeroSAE(), *batch)
        layer = out["layers"][0]
        assert layer["ce_spliced"] == pytest.approx(layer["ce_zero_ablated"], abs=1e-6)
        assert layer["loss_recovered_vs_zero"] == pytest.approx(0.0, abs=1e-4)

    def test_every_measurement_is_taken(self, batch):
        """Per batch: one capture pass, then base + (spliced, mean, zero) per layer + all-spliced."""
        model = _TinyModel()
        input_ids, attention_mask = batch
        evaluate_spliced_layers(
            model, {0: _PerfectSAE()}, {0: model.layer},
            lambda: [(input_ids[:1], attention_mask[:1]), (input_ids[1:], attention_mask[1:])],
        )
        assert model.forward_calls == 2 + 2 * (1 + 3 + 1)

    def test_the_answer_does_not_depend_on_how_blocks_are_batched(self):
        """Sums, not per-batch means: batches with different real-token counts must agree."""
        model = _llama()
        ids, mask = _llama_batch()
        sae = _NoisySAE(32)
        modules = {1: model.model.layers[1]}
        one = evaluate_spliced_layers(model, {1: sae}, modules, lambda: [(ids, mask)])
        three = evaluate_spliced_layers(
            model, {1: sae}, modules, lambda: [(ids[i:i + 1], mask[i:i + 1]) for i in range(3)]
        )
        for key in ("ce_spliced", "ce_mean_ablated", "ce_zero_ablated", "kl", "fvu_centred", "fvu_legacy"):
            assert three["layers"][0][key] == pytest.approx(one["layers"][0][key], rel=1e-5), key
        assert three["ce_base"] == pytest.approx(one["ce_base"], rel=1e-6)
        # A ratio of two near-equal CE differences on a random model amplifies the
        # float noise of padded-batch attention; the sums above are what batching
        # could corrupt, so the ratio gets an absolute tolerance.
        assert three["layers"][0]["loss_recovered_vs_mean"] == pytest.approx(
            one["layers"][0]["loss_recovered_vs_mean"], abs=1e-3
        )

    def test_undefined_numbers_are_none_not_nan(self):
        """JSONB has no NaN: a collapsed denominator must be stored as None."""
        out = _single(_TinyModel(scale=0.0), _PerfectSAE(), *_padded_pair())
        layer = out["layers"][0]
        assert layer["loss_recovered_vs_mean"] is None
        assert all(not (isinstance(v, float) and math.isnan(v)) for v in layer.values())


class _NoisySAE(nn.Module):
    """A deterministic imperfect reconstruction."""

    def __init__(self, d):
        super().__init__()
        torch.manual_seed(3)
        self.proj = nn.Parameter(torch.eye(d) + 0.3 * torch.randn(d, d))

    def forward(self, x):
        return (x @ self.proj.T, None, {})


def _padded_pair():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    return ids, mask


class TestMeanAblation:
    """The baseline is the layer's mean over REAL tokens, in the space the model reads."""

    PAD = 0

    def _fixture(self):
        # A layer output at a large raw scale (like resid_post outliers), and pad
        # tokens whose activations sit far from the real ones.
        model = _TinyModel(vocab=11, d=4, scale=40.0)
        with torch.no_grad():
            model.embed.weight[self.PAD] = 25.0
        ids, mask = _padded_pair()
        return model, ids, mask

    def _layer_outputs(self, model, ids):
        captured = {}
        handle = model.layer.register_forward_hook(lambda m, i, o: captured.__setitem__("x", o.detach()))
        try:
            with torch.no_grad():
                model(ids)
        finally:
            handle.remove()
        return captured["x"]

    def _ce_with(self, model, ids, mask, replacement):
        handle = model.layer.register_forward_hook(lambda m, i, o: replacement.expand_as(o))
        try:
            with torch.no_grad():
                logits = model(ids, attention_mask=mask).logits
        finally:
            handle.remove()
        valid = (mask[:, 1:] * mask[:, :-1]).bool()
        losses = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1), reduction="none").reshape(valid.shape)
        return float(losses[valid].mean())

    def test_it_is_the_real_token_mean_of_the_raw_layer_output(self):
        model, ids, mask = self._fixture()
        x = self._layer_outputs(model, ids)
        real_mean = x[mask.bool()].mean(0)

        out = _single(model, _PerfectSAE(), ids, mask)

        expected = self._ce_with(model, ids, mask, real_mean)
        assert out["layers"][0]["ce_mean_ablated"] == pytest.approx(expected, rel=1e-5)

    def test_padding_does_not_enter_the_mean(self):
        """NEGATIVE CONTROL on the fixture: a mean over every position gives a different CE."""
        model, ids, mask = self._fixture()
        x = self._layer_outputs(model, ids)
        all_positions_mean = x.reshape(-1, x.shape[-1]).mean(0)

        out = _single(model, _PerfectSAE(), ids, mask)

        polluted = self._ce_with(model, ids, mask, all_positions_mean)
        assert abs(out["layers"][0]["ce_mean_ablated"] - polluted) > 1e-3

    def test_it_is_not_the_saes_normalised_bias(self):
        """The old baseline: b_dec at the normalised scale, substituted as a raw activation."""
        model, ids, mask = self._fixture()
        sae = _PerfectSAE()
        with torch.no_grad():
            sae.b_dec.copy_(torch.full((4,), 2.0 ** 0.5 / 2))  # norm sqrt(d)/... a normalised-scale vector

        out = _single(model, sae, ids, mask)

        old_baseline = self._ce_with(model, ids, mask, sae.b_dec.detach())
        assert abs(out["layers"][0]["ce_mean_ablated"] - old_baseline) > 1e-3

    def test_the_mean_comes_to_the_layers_card(self):
        mean = _OnCard(torch.device("cuda", 1), torch.float32)
        x = _OnCard(torch.device("cuda", 0), torch.float16)
        ablated = mean_ablation_at_layer(mean, x)
        assert (ablated.device, ablated.dtype) == (torch.device("cuda", 0), torch.float16)


class TestPrecision:

    def test_a_half_precision_model_meets_a_full_precision_sae(self):
        """The base model runs in fp16/bf16 and the SAE in float32: F.linear refuses mixed dtypes.

        On CPU a float64 model stands in for the mismatch; the evaluation must cast
        the activation to the SAE's dtype and the reconstruction back.
        """
        model = _llama(torch.float64)
        ids, mask = _llama_batch()

        class _LinearSAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.eye(32))  # float32

            def forward(self, x):
                return (F.linear(x, self.W), None, {})

        out = evaluate_spliced_layers(model, {1: _LinearSAE()}, {1: model.model.layers[1]},
                                      lambda: [(ids, mask)])
        assert out["layers"][0]["ce_delta"] == pytest.approx(0.0, abs=1e-4)


class TestLossRecovered:

    def test_it_is_scale_free(self):
        assert loss_recovered(2.0, 2.5, 4.0) == pytest.approx(0.75)
        assert loss_recovered(8.0, 8.5, 10.0) == pytest.approx(0.75)

    def test_a_collapsed_denominator_is_nan_not_a_perfect_score(self):
        assert math.isnan(loss_recovered(2.0, 2.0, 2.0))

    def test_a_layer_whose_ablation_helps_is_reported_honestly(self):
        """Not clamped to [0, 1]: on an untrained or redundant layer ablation can beat the real activation."""
        assert loss_recovered(baseline=3.0, spliced=3.5, ablated=2.0) == pytest.approx(1.5)
        assert loss_recovered(baseline=2.0, spliced=5.0, ablated=4.0) < 0.0


class TestPaddingIsExcluded:

    def test_padded_positions_do_not_enter_the_cross_entropy(self, batch):
        """Including them measures how well the model predicts PAD."""
        model = _TinyModel()
        input_ids, attention_mask = batch
        masked = _single(model, _PerfectSAE(), input_ids, attention_mask)
        unmasked = _single(model, _PerfectSAE(), input_ids, None)
        assert masked["ce_base"] != pytest.approx(unmasked["ce_base"], abs=1e-9)

    def test_a_prediction_counts_only_when_the_token_and_its_successor_are_real(self):
        ids = torch.zeros(1, 5, dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        assert prediction_mask(ids, mask).tolist() == [[True, True, False, False]]

    def test_a_pad_token_does_not_predict_its_real_successor(self):
        """REVIEW R1-C C9. The case above is right-padded, where `mask[:, 1:]` alone
        gives the same answer, so dropping the `mask[:, :-1]` half survived the
        suite. A LEFT-padded row is where the two differ: the last pad predicts the
        first real token, and that prediction is made from padding.

        NEGATIVE CONTROL: `return mask[:, 1:]` -> RED here.
        """
        ids = torch.zeros(2, 5, dtype=torch.long)
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 0]])
        assert prediction_mask(ids, mask).tolist() == [
            [False, False, True, True],
            [True, True, True, False],
        ]


def test_all_layers_spliced_substitutes_every_layer_not_only_the_last():
    """REVIEW R1-C C10. Every multi-layer fixture used a PERFECT SAE at each layer,
    where splicing all of them, only the last, or none gives one number, so an
    all-layers splice that kept only the last layer survived the suite. Here the
    first layer's SAE reconstructs zeros and the second is perfect: splicing both
    must cost exactly what the first alone costs, and the last alone costs nothing.

    NEGATIVE CONTROL: splice only `layers[-1]` in the all-layers pass -> RED here.
    """
    model = _llama()
    ids, mask = _llama_batch()
    saes = {1: _ZeroSAE(32), 2: _PerfectSAE(32)}
    out = evaluate_spliced_layers(model, saes, {L: model.model.layers[L] for L in saes},
                                  lambda: [(ids, mask)])
    first, last = out["layers"]
    assert last["ce_delta"] == pytest.approx(0.0, abs=1e-5)
    # 0.028 nats on this random model: small, and still ~2,800x the 1e-5 equality below,
    # so splicing only the last layer (which leaves the untouched CE) cannot pass it.
    assert abs(first["ce_delta"]) > 1e-3, "the fixture must make the first layer's splice costly"
    assert out["all_layers_spliced"]["ce"] == pytest.approx(first["ce_spliced"], rel=1e-5)
    assert out["all_layers_spliced"]["kl"] == pytest.approx(first["kl"], rel=1e-4)


def test_each_loss_recovered_is_against_its_own_ablation():
    """REVIEW R1-C C30. Computing `loss_recovered_vs_mean` against the ZERO-ablated CE
    survived the suite: the only fixtures reading it used a perfect SAE, which scores
    1.0 against any ablation. Here the SAE is noisy, and the fixture guarantees the two
    ablations cost different amounts, so each ratio has exactly one right denominator.

    NEGATIVE CONTROL: `loss_recovered(ce_base, ce_spliced, ce_zero)` for vs_mean -> RED here.
    """
    model = _llama()
    ids, mask = _llama_batch()
    out = evaluate_spliced_layers(model, {1: _NoisySAE(32)}, {1: model.model.layers[1]},
                                  lambda: [(ids, mask)])
    [layer] = out["layers"]
    base, spliced = out["ce_base"], layer["ce_spliced"]
    mean, zero = layer["ce_mean_ablated"], layer["ce_zero_ablated"]
    # A random tiny model sits near a uniform output, so the two ablations are close in
    # CE (4.1631 against 4.1589 nats); their DENOMINATORS against the base still differ,
    # and that is what the ratios read.
    assert abs(mean - zero) > 1e-3 and abs(spliced - base) > 1e-4, "the fixture must separate the ablations"
    assert layer["loss_recovered_vs_mean"] == pytest.approx(loss_recovered(base, spliced, mean), rel=1e-9)
    assert layer["loss_recovered_vs_zero"] == pytest.approx(loss_recovered(base, spliced, zero), rel=1e-9)
    assert layer["loss_recovered_vs_mean"] != pytest.approx(layer["loss_recovered_vs_zero"], rel=1e-2)


class _OnCard:
    """A tensor stand-in that knows only its device and dtype; `.to` returns the moved copy.

    A CPU-only machine has no second real device to split across, and `meta`
    cannot be copied back out of, so the routing is checked on these.
    """

    def __init__(self, device, dtype=torch.float16):
        self.device = torch.device(device)
        self.dtype = dtype

    def to(self, *args, device=None, dtype=None):
        for arg in args:
            if isinstance(arg, torch.dtype):
                dtype = arg
            else:
                device = arg
        return _OnCard(self.device if device is None else device, self.dtype if dtype is None else dtype)

    def expand_as(self, other):
        return _OnCard(self.device, self.dtype)


class _SaeOnMeta(nn.Module):
    """An SAE held on "another card" (meta); it records the device of what it is given."""

    def __init__(self, output=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(1, device="meta"))
        self.seen = []
        self._output = output

    def forward(self, x):
        self.seen.append(x.device)
        out = self._output if self._output is not None else torch.zeros(x.shape)
        return (out, None, {})


class TestOnASplitModelTheSaeRunsOnItsOwnCard:
    """The layer a CE hook sits on can be on another card than the SAE.

    The SAE was trained on the job's first card; accelerate may have put the
    layer anywhere. `sae(x)` there is a device mismatch inside the encoder, which
    would record "evaluation failed" — no CE for any SAE off the first card.
    """

    def test_the_activation_goes_to_the_sae_and_the_reconstruction_comes_back(self):
        sae = _SaeOnMeta(output=_OnCard("meta", torch.float32))
        x = _OnCard(torch.device("cuda", 0), torch.float16)

        x_hat = reconstruct_at_layer(sae, x)

        assert sae.seen == [torch.device("meta")], "the SAE was run on the layer's card"
        assert (x_hat.device, x_hat.dtype) == (torch.device("cuda", 0), torch.float16), (
            "the reconstruction was left on the SAE's card; the next layer reads it on this one"
        )

    def test_the_evaluation_feeds_the_sae_on_the_sae_card(self, batch):
        """Real tensors through the real hooks: every SAE call, both passes, uses the routing."""
        model = _TinyModel()
        sae = _SaeOnMeta()

        out = _single(model, sae, *batch)

        assert sae.seen and set(sae.seen) == {torch.device("meta")}
        assert len(model.layer._forward_hooks) == 0
        assert math.isfinite(out["layers"][0]["ce_spliced"])
