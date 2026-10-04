"""The metrics that were computed every step and thrown away.

WHY THIS FILE EXISTS. `training_metrics` has columns that were NEVER written:
`loss_reconstructed`, `loss_zero`, `l1_sparsity`, `grad_norm`,
`samples_per_second`. `log_metric` accepted some of them as parameters and no
call site passed any, so the columns were permanently NULL — and a comment in
`sparse_autoencoder` asserted that `loss_zero` *was* persisted.

`l0_mean` is the one that mattered most: the active-feature COUNT was computed
on every forward pass and discarded, while the column that did exist,
`l0_sparsity`, holds a FRACTION of d_sae. The UI compares it against a stated
target of "10-100", which is a count. 0.0094 at d_sae=8192 is ~77 features.
"""

import inspect

import pytest

from src.models.training_metric import TrainingMetric
from src.workers.training_tasks import TrainingTask


class TestLogMetricAcceptsThem:

    @pytest.mark.parametrize(
        "name",
        ["loss_reconstructed", "loss_zero", "l0_mean", "l1_sparsity",
         "grad_norm", "samples_per_second"],
    )
    def test_the_parameter_exists(self, name):
        params = inspect.signature(TrainingTask.log_metric).parameters
        assert name in params, f"log_metric cannot record {name}"


class TestTheColumnsExist:

    @pytest.mark.parametrize(
        "name",
        ["loss_reconstructed", "loss_zero", "l0_mean", "l1_sparsity",
         "grad_norm", "samples_per_second", "fvu", "l0_sparsity"],
    )
    def test_the_column_exists(self, name):
        assert name in TrainingMetric.__table__.columns, f"no column {name}"

    def test_l0_mean_and_l0_sparsity_are_both_present(self):
        """They are different quantities and both are needed.

        `l0_sparsity` is a fraction of d_sae; `l0_mean` is a count per token.
        Keeping only the fraction is what let a UI target of "10-100" sit beside
        a stored value of 0.0094.
        """
        cols = TrainingMetric.__table__.columns
        assert "l0_sparsity" in cols and "l0_mean" in cols


class TestTheValuesAreActuallyPassed:
    """Accepting a parameter nobody passes is the defect, not the fix."""

    @staticmethod
    def _log_metric_calls():
        import ast

        from src.workers import training_tasks

        tree = ast.parse(inspect.getsource(training_tasks))
        calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "log_metric"
            ):
                calls.append({kw.arg for kw in node.keywords})
        return calls

    def test_every_call_site_passes_the_recovered_metrics(self):
        calls = self._log_metric_calls()
        assert calls, "no log_metric call sites found — the scan is broken"
        for i, kwargs in enumerate(calls):
            for name in ("loss_reconstructed", "loss_zero", "l0_mean", "l1_sparsity"):
                assert name in kwargs, (
                    f"log_metric call {i} does not pass {name}; the column would "
                    f"stay NULL exactly as before"
                )

    @staticmethod
    def _kwargs_with_literal_none():
        """Each call site's kwargs -> whether the value is a literal `None`.

        Round 4, M-4: the assertion above checks only that the keyword NAME
        appears, so `l0_mean=None` satisfies it and the column the migration
        exists for stays NULL forever. This repo's own checklist says to assert
        the PAYLOAD, not that a call happened.
        """
        import ast

        from src.workers import training_tasks

        tree = ast.parse(inspect.getsource(training_tasks))
        out = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "log_metric"
            ):
                out.append({
                    kw.arg: (
                        isinstance(kw.value, ast.Constant) and kw.value.value is None
                    )
                    for kw in node.keywords
                })
        return out

    def test_the_recovered_metrics_are_not_hardcoded_none(self):
        for i, kwargs in enumerate(self._kwargs_with_literal_none()):
            for name in ("loss_reconstructed", "loss_zero", "l0_mean", "l1_sparsity"):
                assert not kwargs.get(name, False), (
                    f"log_metric call {i} passes {name}=None literally, so the "
                    f"column stays NULL while the name assertion above passes"
                )

    def test_both_the_aggregate_and_per_layer_rows_are_covered(self):
        """There are two writers and fixing one is this repo's known anti-pattern."""
        assert len(self._log_metric_calls()) >= 2
