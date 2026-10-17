"""One forward serves several probes — and only ones that read the same layer.

Finding C of the 2026-09-29 audit: evaluation ran the model once per (probe, set). With
`top_n_layers=1` every probe reads the SAME layer, so every forward after the first recomputed
activations that already existed. Measured on real runs: 1164 s for one probe over five sets,
2388.6 s for two — linear in probes, for one layer's work. A four-rule run paid ~80 minutes where
~20 would do, which is most of the reason nobody uses the sweep the feature exists to support.

⚠ THE LAYER EQUALITY IS A PRECONDITION, NOT AN ASSUMPTION, and it is the whole risk of this
change. With `top_n_layers > 1` probes do NOT share a layer. Scoring one probe against another
layer's activations yields plausible numbers in the wrong basis — no metric would look wrong, no
test of AUROC would fail, and the exported document would carry weights fitted at one layer
beside vectors produced at another. So the shared path REFUSES a mixed group rather than picking
a layer, and the caller groups.

The other half is that `forward_scores` stayed a wrapper rather than becoming a second
implementation: this module's contract is that evaluation, offline scoring and 033's test-vector
generation score through the SAME code, so an exported definition's vectors come from whatever
produced its metrics.
"""

from __future__ import annotations

import pytest
import torch

from src.ml.probe_monitor_model import ProbeHead

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _head(layer: int, weight: list[float]) -> ProbeHead:
    return ProbeHead(weight=torch.tensor(weight), bias=0.0, layer=layer)


class TestMixedLayersAreRefused:
    """The guard that makes grouping a requirement rather than a convention."""

    def test_two_layers_in_one_group_raises(self):
        from src.services.probe_monitor_capture import ScoreSpec, forward_scores_many

        specs = [
            ScoreSpec(head=_head(3, [1.0, 0.0]), rule="mean"),
            ScoreSpec(head=_head(7, [1.0, 0.0]), rule="mean"),
        ]
        with pytest.raises(ValueError, match="one forward serves one layer"):
            forward_scores_many(object(), [object()], specs)

    def test_the_refusal_names_the_layers(self):
        """An error that does not say which layers collided sends the reader to the wrong place."""
        from src.services.probe_monitor_capture import ScoreSpec, forward_scores_many

        specs = [
            ScoreSpec(head=_head(3, [1.0]), rule="mean"),
            ScoreSpec(head=_head(7, [1.0]), rule="mean"),
        ]
        with pytest.raises(ValueError) as exc:
            forward_scores_many(object(), [object()], specs)
        assert "3" in str(exc.value) and "7" in str(exc.value)

    def test_one_layer_repeated_is_fine(self):
        """⚠ Specificity. Refusing every multi-probe group would make the optimisation
        unreachable while still passing the test above."""
        from src.services.probe_monitor_capture import ScoreSpec, forward_scores_many

        specs = [
            ScoreSpec(head=_head(5, [1.0]), rule="mean"),
            ScoreSpec(head=_head(5, [1.0]), rule="max"),
        ]
        # Empty examples short-circuits before the model is touched, so this reaches the layer
        # check and returns rather than needing a real model.
        assert forward_scores_many(object(), [], specs) == [[], []]

    def test_no_specs_is_refused(self):
        from src.services.probe_monitor_capture import forward_scores_many

        with pytest.raises(ValueError, match="no probes"):
            forward_scores_many(object(), [], [])


class TestOneResultPerProbe:
    def test_the_shape_is_per_spec(self):
        from src.services.probe_monitor_capture import ScoreSpec, forward_scores_many

        specs = [ScoreSpec(head=_head(1, [1.0]), rule="mean") for _ in range(3)]
        out = forward_scores_many(object(), [], specs)
        assert len(out) == 3, "one result list per probe, so a caller can zip them with the group"


class TestTheSingleProbeFormIsAWrapper:
    """⚠ NOT A SECOND IMPLEMENTATION. Two scoring paths would be two detectors and only one of
    them measured — the module's own contract. When the multi-probe form was added for
    performance, the single-probe form became one spec through the same loop."""

    def test_forward_scores_delegates(self):
        import ast
        import inspect

        from src.services import probe_monitor_capture

        tree = ast.parse(inspect.getsource(probe_monitor_capture.forward_scores))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "forward_scores_many" in called, (
            f"forward_scores does not delegate — it is a second scoring path; calls: {sorted(called)}"
        )

    def test_it_does_not_register_its_own_hooks(self):
        """The sharpest form of the same check: a second implementation would have to hook the
        model itself. If this ever grows a `HookManager`, it has forked."""
        import inspect

        from src.services import probe_monitor_capture

        source = inspect.getsource(probe_monitor_capture.forward_scores)
        assert "HookManager" not in source
        assert "register_hooks" not in source


class TestTheRunGroupsByLayer:
    """Wiring. The guard refuses a mixed group, so the caller MUST group or the run breaks —
    but it must also actually share within a group, which is the point of the change."""

    @staticmethod
    def _source():
        import inspect

        from src.services import probe_monitor_run

        return inspect.getsource(probe_monitor_run.execute_probe_run)

    def test_the_evaluating_stage_groups_before_calling(self):
        assert "by_layer" in self._source(), (
            "the evaluating stage does not group probes by layer; a multi-layer run would be "
            "refused outright by forward_scores_many"
        )

    def test_it_calls_the_grouped_evaluator(self):
        import ast
        import inspect

        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run.execute_probe_run))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_evaluate_probes_on_sets" in called

    def test_the_old_per_probe_evaluator_is_gone(self):
        """Leaving it would let a future caller reintroduce the per-probe forward silently."""
        from src.services import probe_monitor_run

        assert not hasattr(probe_monitor_run, "_evaluate_probe_on_sets")

    def test_the_reevaluation_path_uses_the_same_function(self):
        """`evaluate_probe` re-scores an existing probe. Two paths would let a re-evaluation
        disagree with the original about how a probe is scored."""
        import ast
        import inspect

        from src.services import probe_monitor_run

        tree = ast.parse(inspect.getsource(probe_monitor_run.evaluate_probe))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_evaluate_probes_on_sets" in called
