"""C and D are pure refactors. They must not move a single float.

Two performance findings from the 2026-09-29 audit change HOW work is arranged, not what is
computed: reusing standardised batches across rules (D), and reusing one forward across probes
that share a layer (C). The honest way to show a refactor of that kind is sound is not "the
numbers look similar" — it is that the weights are **bit-identical**.

This estate has the determinism to support that: SC-4 reproduced a run to sixteen significant
figures across two separately-rolled pods. So `torch.equal`, not `allclose`. A refactor of
batching has no licence to move a float, and `allclose` would hide exactly the drift worth
catching — an accumulation reordered, a mask applied in a different place, a standardisation
recomputed from a different set of rows.

⚠ THE HARNESS IS WRITTEN BEFORE THE REFACTORS, DELIBERATELY. Written afterwards it would assert
equivalence against numbers produced by the code being changed, which proves only that the new
code agrees with itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.services.probe_monitor_trainer import (
    _standardisation,
    _standardised_batches,
    plan_length_buckets,
    train_rule,
)

D_MODEL = 32


def make_rows(n=60, seed=0):
    """Rows of DELIBERATELY VARIED length, so more than one length bucket is formed.

    ⚠ A fixture whose rows are all one length lands in a single batch, and both refactors then
    pass trivially — the bucketing never engages and there is only one forward to share. The
    spread is the part that makes this test able to fail.
    """
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=D_MODEL).astype(np.float32)
    rows, labels = [], []
    for i in range(n):
        y = i % 2
        length = int(rng.integers(5, 60))          # 12x spread
        block = rng.normal(0.0, 1.0, size=(length, D_MODEL)).astype(np.float32)
        if y:
            block += 0.4 * direction
        rows.append(block)
        labels.append(y)
    return rows, labels


@pytest.fixture
def data():
    rows, labels = make_rows()
    split = int(len(rows) * 0.75)
    return rows[:split], labels[:split], rows[split:], labels[split:]


class TestTheFixtureCanActuallyExerciseTheRefactors:
    """A harness that cannot fail is decoration. These pin its preconditions."""

    def test_the_rows_form_more_than_one_length_bucket(self, data):
        train_rows, _, _, _ = data
        widths = [r.shape[0] for r in train_rows]
        # A small budget, so bucketing engages at test scale the way it does at run scale.
        buckets = plan_length_buckets(widths, budget_slots=400)
        assert len(buckets) > 1, (
            f"all {len(widths)} rows landed in one bucket, so batch reuse cannot be tested; "
            f"widths {min(widths)}..{max(widths)}"
        )

    def test_the_widths_actually_vary(self, data):
        train_rows, _, _, _ = data
        widths = {r.shape[0] for r in train_rows}
        assert len(widths) > 5, f"only {len(widths)} distinct widths — padding is not exercised"


class TestTrainingIsDeterministic:
    """The premise everything else rests on. If training is not reproducible at fixed seed,
    bit-identity cannot be used to verify anything and this whole approach is void."""

    @pytest.mark.parametrize("rule", ["mean", "max", "softmax"])
    def test_the_same_inputs_give_bit_identical_weights(self, data, rule):
        tr, trl, va, val = data
        a = train_rule(rule, tr, trl, va, val, seed=1337, epochs=25)
        b = train_rule(rule, tr, trl, va, val, seed=1337, epochs=25)

        assert torch.equal(a.weight, b.weight), "training is not reproducible at a fixed seed"
        assert a.bias == b.bias
        assert a.val_auroc == b.val_auroc
        assert a.best_epoch == b.best_epoch

    def test_a_DIFFERENT_seed_gives_different_weights(self, rule="attention", data=None):
        """⚠ Specificity. If every seed gave the same answer the test above would pass against
        training that ignores its inputs."""
        tr, trl, va, val = make_rows()[0][:45], None, None, None
        rows, labels = make_rows()
        split = int(len(rows) * 0.75)
        a = train_rule(rule, rows[:split], labels[:split], rows[split:], labels[split:],
                       seed=1, epochs=25)
        b = train_rule(rule, rows[:split], labels[:split], rows[split:], labels[split:],
                       seed=2, epochs=25)
        assert not torch.equal(a.weight, b.weight)


class TestBatchReuseIsEquivalent:
    """Finding D: the standardised batches are rebuilt per rule from the same rows and the same
    statistics. Passing them in must produce exactly what building them inside produced."""

    def test_prebuilt_batches_give_bit_identical_weights(self, data):
        tr, trl, va, val = data
        baseline = train_rule("mean", tr, trl, va, val, seed=1337, epochs=25)

        # What the caller will hoist: statistics and batches depend only on the TRAIN rows.
        mean, std = _standardisation(tr)
        device = torch.device("cpu")
        train_batches = _standardised_batches(tr, mean, std, device)
        val_batches = _standardised_batches(va, mean, std, device)

        reused = train_rule(
            "mean", tr, trl, va, val, seed=1337, epochs=25,
            prebuilt=(mean, std, train_batches, val_batches),
        )
        assert torch.equal(baseline.weight, reused.weight)
        assert baseline.bias == reused.bias
        assert baseline.val_auroc == reused.val_auroc
        assert baseline.best_epoch == reused.best_epoch

    def test_the_statistics_come_from_TRAIN_only(self, data):
        """⚠ The hazard the hoist introduces. Building the statistics over train+validation
        would leak the validation distribution's scale into the model — and it would still
        train, and still look fine."""
        tr, _trl, va, _val = data
        train_only, _ = _standardisation(tr)
        both, _ = _standardisation(list(tr) + list(va))
        assert not torch.equal(train_only, both), (
            "the fixture cannot distinguish train-only statistics from train+val ones, so it "
            "cannot catch that leak"
        )

    @pytest.mark.parametrize("rule", ["mean", "max", "softmax", "rolling_mean_max"])
    def test_every_rule_is_equivalent_under_reuse(self, data, rule):
        """One shared batch set feeds every rule, so each has to be checked."""
        tr, trl, va, val = data
        baseline = train_rule(rule, tr, trl, va, val, seed=1337, epochs=15)
        mean, std = _standardisation(tr)
        device = torch.device("cpu")
        reused = train_rule(
            rule, tr, trl, va, val, seed=1337, epochs=15,
            prebuilt=(mean, std,
                      _standardised_batches(tr, mean, std, device),
                      _standardised_batches(va, mean, std, device)),
        )
        assert torch.equal(baseline.weight, reused.weight), rule
        assert baseline.val_auroc == reused.val_auroc, rule


class TestTheHoistedStatisticsComeFromTrainOnly:
    """⚠ FOUND BY A CONTROL, NOT BY READING — AND MY OWN NOTE ABOUT IT WAS WRONG.

    Finding D hoists `_standardisation` out of the per-rule loop. That moves the call to a place
    where it is easy to widen: `_standardisation(list(train_rows) + list(val_rows))` is a
    one-token edit, it trains, it converges, and the model is now fitted to statistics that saw
    the validation distribution's scale. The validation AUROC then decides early stopping and the
    layer, so the leak flatters the numbers that make the decisions.

    A mutation doing exactly that survived the WHOLE suite — 8829 tests green — after I had
    recorded in the commit that it was "guarded by the pipeline test". It was not guarded by
    anything.

    Asserted on the ARGUMENT of the call, which is what changes: `train_rows` alone is a
    `Name`, and any widening makes it a `BinOp`, a `Call` or a longer argument list.
    """

    @staticmethod
    def _standardisation_calls():
        import ast
        import inspect

        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run.execute_probe_run))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_standardisation"
        ]

    def test_the_scan_finds_the_call(self):
        """A source scan that matches nothing asserts nothing."""
        assert self._standardisation_calls(), (
            "no _standardisation call in execute_probe_run — the hoist has moved and this guard "
            "has stopped looking"
        )

    def test_it_is_given_the_train_rows_and_nothing_else(self):
        for call in self._standardisation_calls():
            assert len(call.args) == 1, (
                f"_standardisation called with {len(call.args)} positional arguments; the "
                f"statistics must come from the training rows alone"
            )
            argument = call.args[0]
            assert isinstance(argument, ast.Name), (
                f"_standardisation is called on a {type(argument).__name__} rather than a plain "
                f"name — a concatenation or comprehension here is how validation rows get into "
                f"the training statistics"
            )
            assert argument.id == "train_rows", (
                f"_standardisation is called on {argument.id!r}, not train_rows"
            )

    def test_val_rows_is_not_mentioned_in_the_statistics_expression(self):
        """Belt and braces: catches a widening that keeps the argument a single Name by
        rebinding `train_rows` itself just above the call."""
        import ast
        import inspect

        from src.services import probe_monitor_run

        source = inspect.getsource(probe_monitor_run.execute_probe_run)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "train_rows" for t in node.targets
                )
            ):
                mentioned = {
                    n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)
                }
                assert "val_rows" not in mentioned, (
                    "train_rows is assigned from an expression mentioning val_rows, so the "
                    "training statistics see the validation distribution"
                )


import ast  # noqa: E402  (used by the AST tests above)
