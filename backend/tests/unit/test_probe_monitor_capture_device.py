"""The capture loops reconcile their mask to the ACTIVATION's device, not the model's.

⚠ WHY THIS FILE EXISTS. The second Stage 1 acceptance run loaded Llama-3.1-8B on the
3090, rendered 8,000 rows and died 30% in:

    RuntimeError: Expected all tensors to be on the same device, but found at least two
    devices, cuda:0 and cpu!

`HookManager`'s forward hook stores `output.detach().cpu()` — deliberately, because
keeping seven swept layers' activations in VRAM would not fit. So on a GPU run the
model's parameters are on cuda:0 while every activation the loops receive is on the CPU.
All three loops built their mask and index tensors from `next(model.parameters()).device`.

`test_probe_monitor_capture.py` could not have caught it, and the reason is the trap
this repo keeps paying for: those tests run a real tiny Llama ON THE CPU, where the
model's device and the hook's device are the same object. The two tensors agreed BY
CONSTRUCTION, so nine mutation controls passed over a loop that could not run on the
hardware it was written for.

So the decision is a two-line pure function and it is tested on tensors whose devices
genuinely differ — `meta` against `cpu`, which needs no GPU and which torch rejects with
the same error class. The CALL is asserted by walking each loop's AST, because a
substring search for `_align` matches the comments above each call (five times in this
repo, and this file would have been the sixth).

MUTATION CONTROLS (each verified to fail this file):
  M134  `_align` returns `tensor` unchanged              → the cross-device tests
  M135  the call dropped from `capture_pooled`           → the AST wiring test
  M136  the call dropped from `capture_tokens`           → the AST wiring test
  M137  the call dropped from `forward_scores`           → the AST wiring test
  M138  `torch.arange(..., device=device)` restored      → the arange test
"""
import ast
import inspect

import pytest
import torch

from src.services import probe_monitor_capture
from src.services.probe_monitor_capture import _align

LOOPS = ("capture_pooled", "capture_tokens", "forward_scores")


class TestAlignMovesTheMaskToTheActivation:
    def test_a_mask_on_another_device_is_moved(self):
        activation = torch.zeros(2, 3, device="meta")
        mask = torch.ones(2, 3, dtype=torch.bool)
        assert _align(mask, activation).device == activation.device

    def test_the_arithmetic_then_works(self):
        """The failing expression itself. Without the move this raises; `meta` is the
        stand-in for `cuda:0` so the test needs no GPU."""
        activation = torch.zeros(2, 4, 3, device="meta")
        mask = torch.ones(2, 4, dtype=torch.bool)
        product = activation * _align(mask, activation).unsqueeze(-1)
        assert product.device == activation.device

    def test_the_unaligned_arithmetic_really_does_raise(self):
        """The premise, asserted rather than assumed: if `meta` silently accepted a CPU
        operand, every test above would pass against the defect."""
        activation = torch.zeros(2, 4, 3, device="meta")
        mask = torch.ones(2, 4, dtype=torch.bool)
        with pytest.raises(RuntimeError):
            _ = activation * mask.unsqueeze(-1)

    def test_a_mask_already_on_the_right_device_is_returned_untouched(self):
        """Not an optimisation — `.to()` on a same-device tensor is a no-op copy, and
        returning the identity keeps the bool mask's identity for the `any()` checks."""
        activation = torch.zeros(2, 3)
        mask = torch.ones(2, 3, dtype=torch.bool)
        assert _align(mask, activation) is mask

    def test_it_does_not_change_the_dtype(self):
        """A mask that arrives as bool must stay bool: `activation[position][keep]`
        selects with a bool mask and GATHERS with an integer one, which would silently
        take row 1 of every row instead of its scored tokens."""
        activation = torch.zeros(2, 3, device="meta")
        mask = torch.ones(2, 3, dtype=torch.bool)
        assert _align(mask, activation).dtype is torch.bool


class TestEveryLoopActuallyAlignsIt:
    def _calls_in(self, name):
        function = getattr(probe_monitor_capture, name)
        tree = ast.parse(inspect.getsource(function).lstrip())
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_align"
        ]

    @pytest.mark.parametrize("name", LOOPS)
    def test_the_loop_calls_align(self, name):
        assert self._calls_in(name), f"{name} never calls _align, so its mask stays on the model's device"

    @pytest.mark.parametrize("name", LOOPS)
    def test_it_aligns_against_the_activation_and_not_something_else(self, name):
        """A call passing the wrong second argument compiles, passes the test above and
        reproduces the defect — `_align(mask, ids)` would move the mask back onto the
        model's device."""
        targets = set()
        for call in self._calls_in(name):
            assert len(call.args) == 2, f"{name}: _align takes (tensor, like)"
            like = call.args[1]
            assert isinstance(like, ast.Name), f"{name}: aligned against {ast.dump(like)}"
            targets.add(like.id)
        assert targets <= {"activation", "hidden"}, (
            f"{name} aligns against {sorted(targets)}; the authority is the activation"
        )

    def test_the_ast_walk_can_tell_a_call_from_a_mention(self):
        """The control for the walk: `_align` named in a comment and a string must not
        register, which is exactly how a source scrape passes over a missing call."""

        def decoy():
            # _align(mask, activation)
            return "_align(mask, activation)"

        tree = ast.parse(inspect.getsource(decoy).lstrip())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_align"
        ]
        assert not calls


class TestNoIndexTensorIsBuiltOnTheModelsDevice:
    """`torch.arange(len(indices), device=device)` indexed a CPU activation from the
    model's device — the same defect in a second expression, which an `_align` check
    alone would not see."""

    def test_capture_pooled_builds_its_arange_on_the_activations_device(self):
        tree = ast.parse(inspect.getsource(probe_monitor_capture.capture_pooled).lstrip())
        offenders = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "arange":
                continue
            for keyword in node.keywords:
                if keyword.arg != "device":
                    continue
                # Accept `activation.device`; reject the bare name `device`.
                if isinstance(keyword.value, ast.Name):
                    offenders.append(keyword.value.id)
        assert not offenders, (
            f"torch.arange built on {offenders}; index tensors must come from the "
            f"activation's device"
        )
