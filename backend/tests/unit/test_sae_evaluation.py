"""The SAE's cost to the MODEL, not to its own reconstruction objective.

WHY THIS FILE EXISTS. Nothing ever put an SAE reconstruction back into the
model. Every reported number — FVU, L0, dead count — lives in the SAE's own
space, and reconstruction error is not uniformly important: an SAE can reach
FVU 0.02 and still wreck the next-token distribution, because the directions
that carry the output are a small share of the variance.

The failure mode this file guards hardest is the FORWARD HOOK. A hook left
attached changes every subsequent forward pass, so the symptom is a slow
unexplained quality drift in the steps that follow, not an error.

MUTATION CONTROLS — Phase 2, the SAE on its own card (2026-09-14; each applied
alone, this file run, source restored and checked by sha256). All went red:
  S1 reconstruct_at_layer runs sae(x) on the layer's card    -> the_activation_goes_to_the_sae..., spliced_ce_delta_feeds_the_sae...
  S2 reconstruction left on the SAE's card (dtype only)      -> the_activation_goes_to_the_sae_and_the_reconstruction_comes_back
  S3 ablation bias left on the SAE's card (dtype only)       -> the_ablation_bias_comes_to_the_layers_card
  S4 spliced_ce_delta inlines sae(x) instead of the helper   -> spliced_ce_delta_feeds_the_sae_on_the_sae_card
"""

import math

import pytest
import torch
import torch.nn as nn

from src.services.sae_evaluation import (
    loss_recovered,
    spliced_ce_delta,
)


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _TinyModel(nn.Module):
    """A real nn.Module so hooks behave as they do in production."""

    def __init__(self, vocab=11, d=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layer = nn.Linear(d, d)
        self.head = nn.Linear(d, vocab)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.forward_calls += 1
        h = self.layer(self.embed(input_ids))
        return _Out(self.head(h))


class _PerfectSAE(nn.Module):
    """Round-trips exactly: the reconstruction costs the model nothing."""

    def __init__(self, d=4):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return (x, None, {})


class _DestructiveSAE(nn.Module):
    """Returns the bias for everything — as bad as ablation."""

    def __init__(self, d=4):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return (self.b_dec.expand_as(x), None, {})


@pytest.fixture
def batch():
    torch.manual_seed(0)
    input_ids = torch.randint(0, 11, (2, 6))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 4:] = 0  # one padded row, so masking is exercised
    return input_ids, attention_mask


class TestTheHookIsAlwaysRemoved:
    """The most dangerous defect here is silent and permanent."""

    def test_no_hook_survives_the_measurement(self, batch):
        model = _TinyModel()
        spliced_ce_delta(model, _PerfectSAE(), model.layer, *batch)
        assert len(model.layer._forward_hooks) == 0, (
            "a forward hook is still attached; every later forward pass, "
            "including training steps, is now silently modified"
        )

    def test_the_model_is_unchanged_afterwards(self, batch):
        """Same inputs must give the same logits before and after."""
        model = _TinyModel()
        input_ids, attention_mask = batch
        with torch.no_grad():
            before = model(input_ids, attention_mask=attention_mask).logits.clone()
        spliced_ce_delta(model, _DestructiveSAE(), model.layer, *batch)
        with torch.no_grad():
            after = model(input_ids, attention_mask=attention_mask).logits
        torch.testing.assert_close(before, after)

    def test_the_hook_comes_off_even_when_the_forward_raises(self, batch):
        model = _TinyModel()

        class _Exploding(nn.Module):
            def forward(self, x):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            spliced_ce_delta(model, _Exploding(), model.layer, *batch)
        assert len(model.layer._forward_hooks) == 0

    def test_training_mode_is_restored(self, batch):
        model = _TinyModel()
        model.train()
        spliced_ce_delta(model, _PerfectSAE(), model.layer, *batch)
        assert model.training, "the model was left in eval mode"


class TestTheNumbersMeanSomething:

    def test_a_perfect_sae_costs_the_model_nothing(self, batch):
        model = _TinyModel()
        out = spliced_ce_delta(model, _PerfectSAE(), model.layer, *batch)
        assert out["ce_delta"] == pytest.approx(0.0, abs=1e-5)
        assert out["loss_recovered"] == pytest.approx(1.0, abs=1e-4)

    def test_a_destructive_sae_recovers_nothing(self, batch):
        """NEGATIVE CONTROL: an SAE no better than ablation must score ~0."""
        model = _TinyModel()
        out = spliced_ce_delta(model, _DestructiveSAE(), model.layer, *batch)
        assert out["loss_recovered"] == pytest.approx(0.0, abs=1e-4)

    def test_a_layer_whose_ablation_helps_is_reported_honestly(self):
        """The score is NOT clamped to [0, 1], and that is deliberate.

        On a layer that contributes nothing useful — an untrained model, or a
        genuinely redundant layer — ablation can score BETTER than the real
        activation, making the denominator negative. That is a real and useful
        finding about the layer. Clamping would hide it behind a plausible
        number, which is the failure mode this whole arc keeps removing.

        (Written after the first version of this test asserted
        `ce_ablated >= ce_spliced` as an invariant and a randomly-initialised
        fixture disproved it.)
        """
        # Ablating is CHEAPER than the baseline (2.0 < 3.0): the layer was
        # hurting. The reconstruction is worse still, so the score exceeds 1.
        score = loss_recovered(baseline=3.0, spliced=3.5, ablated=2.0)
        assert score == pytest.approx(1.5), (
            "the score was clamped; a layer whose ablation helps must be "
            "reported as it is, not flattened into [0, 1]"
        )
        # And a reconstruction worse than ablation scores below zero.
        assert loss_recovered(baseline=2.0, spliced=5.0, ablated=4.0) < 0.0

    def test_all_three_measurements_are_taken(self, batch):
        """Three forwards: baseline, spliced, ablated. Fewer means one is faked."""
        model = _TinyModel()
        spliced_ce_delta(model, _PerfectSAE(), model.layer, *batch)
        assert model.forward_calls == 3


class TestLossRecovered:

    def test_it_is_scale_free(self):
        assert loss_recovered(2.0, 2.5, 4.0) == pytest.approx(0.75)
        assert loss_recovered(8.0, 8.5, 10.0) == pytest.approx(0.75)

    def test_a_collapsed_denominator_is_nan_not_a_perfect_score(self):
        """If ablating costs nothing, "how much did we avoid" has no answer.

        Returning 1.0 would award a perfect score to a layer that does nothing.
        """
        assert math.isnan(loss_recovered(2.0, 2.0, 2.0))


class TestPaddingIsExcluded:

    def test_padded_positions_do_not_enter_the_cross_entropy(self, batch):
        """Including them measures how well the model predicts PAD."""
        model = _TinyModel()
        input_ids, attention_mask = batch
        masked = spliced_ce_delta(model, _PerfectSAE(), model.layer, input_ids, attention_mask)
        unmasked = spliced_ce_delta(model, _PerfectSAE(), model.layer, input_ids, None)
        assert masked["ce_baseline"] != pytest.approx(unmasked["ce_baseline"], abs=1e-9)


class TestTheEvaluationIsReachable:
    """It shipped with ZERO callers — rounds 3 and 4 both recorded that.

    The cached-activation path never loads a base model, which is why it was
    unwired. It now runs once after the checkpoint is written: the model load
    is affordable at that point, and a failure there cannot cost a training that
    already succeeded.
    """

    @staticmethod
    def _calls(module):
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(module))
        out = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute):
                    out.add(n.func.attr)
                elif isinstance(n.func, ast.Name):
                    out.add(n.func.id)
        return out

    def test_the_trainer_calls_the_evaluator(self):
        from src.workers import training_tasks

        assert "_evaluate_spliced_ce" in self._calls(training_tasks), (
            "the spliced-CE evaluation has no caller; every number reported "
            "about an SAE stays inside the SAE's own space"
        )

    def test_the_evaluator_calls_spliced_ce_delta(self):
        from src.workers import training_tasks

        assert "spliced_ce_delta" in self._calls(training_tasks), (
            "the evaluator does not actually measure anything"
        )

    def test_it_runs_after_the_checkpoint_is_saved(self):
        """A model load that OOMs must not cost a completed training."""
        import inspect

        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        saved = src.index("Saved Community Standard checkpoint")
        # The CALL, not the `def` — which is defined above the task and would
        # otherwise always compare as "before".
        evaluated = src.index("_evaluate_spliced_ce(\n                self,")
        assert saved < evaluated, (
            "the evaluation runs before the checkpoint is written, so a failure "
            "there loses the trained SAE"
        )

    def test_failure_cannot_fail_the_training(self):
        import inspect

        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        window = src[src.index("_evaluate_spliced_ce(\n                self,"):]
        assert "training is safe" in window[:900], (
            "a failed evaluation propagates and fails a finished training"
        )

    def test_it_is_configurable(self):
        from src.schemas.training import TrainingHyperparameters

        assert "evaluate_ce_delta" in TrainingHyperparameters.model_fields


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

    def __init__(self, output=None, b_dec=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(1, device="meta"))
        self.seen = []
        self._output = output
        if b_dec is not None:
            self.b_dec = b_dec

    def forward(self, x):
        self.seen.append(x.device)
        out = self._output if self._output is not None else torch.zeros(x.shape)
        return (out, None, {})


class TestOnASplitModelTheSaeRunsOnItsOwnCard:
    """The layer a CE hook sits on can be on another card than the SAE.

    The SAE was trained on the job's first card; accelerate may have put the
    layer anywhere. `sae(x)` there is a device mismatch inside the encoder, which
    the trainer logs as "evaluation failed" — no CE for any SAE off the first card.
    """

    def test_the_activation_goes_to_the_sae_and_the_reconstruction_comes_back(self):
        from src.services.sae_evaluation import reconstruct_at_layer

        sae = _SaeOnMeta(output=_OnCard("meta", torch.float32))
        x = _OnCard(torch.device("cuda", 0), torch.float16)

        x_hat = reconstruct_at_layer(sae, x)

        assert sae.seen == [torch.device("meta")], "the SAE was run on the layer's card"
        assert (x_hat.device, x_hat.dtype) == (torch.device("cuda", 0), torch.float16), (
            "the reconstruction was left on the SAE's card; the next layer reads it on this one"
        )

    def test_the_ablation_bias_comes_to_the_layers_card(self):
        from src.services.sae_evaluation import ablation_at_layer

        sae = _SaeOnMeta(b_dec=_OnCard(torch.device("cuda", 1), torch.float32))
        x = _OnCard(torch.device("cuda", 0), torch.float16)

        ablated = ablation_at_layer(sae, x)

        assert (ablated.device, ablated.dtype) == (torch.device("cuda", 0), torch.float16)

    def test_spliced_ce_delta_feeds_the_sae_on_the_sae_card(self, batch):
        """Real tensors through the real hook: the measurement uses the routing."""
        model = _TinyModel()
        sae = _SaeOnMeta()

        out = spliced_ce_delta(model, sae, model.layer, *batch)

        assert sae.seen == [torch.device("meta")]
        assert len(model.layer._forward_hooks) == 0
        assert math.isfinite(out["ce_spliced"])
