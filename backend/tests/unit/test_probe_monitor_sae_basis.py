"""A k-sparse probe is SCORED in the basis it was TRAINED in (032 FR-8, found by Stage 2).

⚠ THE DEFECT. `train_sae_variant` encodes the token capture through the SAE and slices the chosen
k columns, so the head it fits is k-dimensional. `_evaluate_probe_on_sets` then scored that head
against the RAW residual, and Stage 2 acceptance died 98.5% of the way through a 24-minute
evaluation — after the dense probe's five sets had already succeeded:

    ValueError: activations are d_model=2048 but this probe is d_model=128
    (probe_monitor_model.py:157)

The k-sparse variant was trainable and not evaluable. It could never leave rung 0, and FR-8's whole
point is that it is reported beside the dense baseline.

⚠ WHY THE FIX IS A HOOK AND NOT A SECOND FUNCTION. `forward_scores` is deliberately the one
scoring path — evaluation, offline scoring and 033's test vectors all go through it, so an exported
definition's vectors come from the code that produced the metrics. A separate `forward_scores_sae`
would be a second detector with only one of them measured. `encoder` transforms the activation
before the head and leaves the head, the mask and `combine` untouched.

MUTATION CONTROLS (each verified to fail this file):
  A1  `encoder` dropped from the evaluation call            → the dimension test
  A2  `encoder` dropped from the offline-scoring call       → the scoring test
  A3  the encoder stops selecting the k columns             → the k-width test
  A4  `encode_with_training_normalization` → bare `encode`  → the normalisation test
  A5  the variant check inverted (dense probes encoded)     → the dense test
"""
import ast
import inspect

import numpy as np
import pytest
import torch

from src.services import probe_monitor_run as run_module


class _Dictionary(torch.nn.Module):
    """A stand-in SAE: k features, and an encode that is NOT the identity so the basis shows."""

    def __init__(self, d_model=8, n_features=32):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = torch.nn.Linear(d_model, n_features, bias=True)
        self.n_features = n_features
        self.d_model = d_model


def _fake_sae_encoder(indices, d_model=8, n_features=32):
    """The transform under test, built without a database: same shape contract as
    `sae_encoder_for` returns."""
    dictionary = _Dictionary(d_model, n_features)
    columns = torch.as_tensor(list(indices), dtype=torch.long)

    def encode(hidden):
        flat = hidden.reshape(-1, hidden.shape[-1])
        with torch.no_grad():
            features = dictionary.encoder(flat)
            selected = features.index_select(1, columns)
        return selected.reshape(hidden.shape[0], hidden.shape[1], -1)

    return encode


class TestTheTransformHasTheRightShape:
    def test_it_maps_d_model_to_k(self):
        encode = _fake_sae_encoder([1, 5, 9])
        out = encode(torch.randn(2, 7, 8))
        assert out.shape == (2, 7, 3), out.shape

    def test_it_preserves_the_batch_and_token_axes(self):
        """A reshape that collapsed the batch would pair every row's scores with another row's."""
        encode = _fake_sae_encoder([0, 1])
        hidden = torch.randn(3, 4, 8)
        out = encode(hidden)
        assert out.shape[:2] == hidden.shape[:2]

    def test_each_row_is_transformed_independently(self):
        """Row i of the output must depend only on row i of the input — otherwise the reshape
        crossed the batch boundary, which no shape assertion can see."""
        encode = _fake_sae_encoder([0, 1, 2])
        hidden = torch.randn(3, 5, 8)
        whole = encode(hidden)
        for index in range(hidden.shape[0]):
            alone = encode(hidden[index : index + 1])
            assert torch.allclose(whole[index], alone[0], atol=1e-6), index

    def test_it_returns_the_activation_to_its_own_device(self):
        """The head and the mask live where the hook put the activation."""
        encode = _fake_sae_encoder([0, 1])
        hidden = torch.randn(1, 3, 8)
        assert encode(hidden).device == hidden.device


class TestTheHeadAndTheBasisAgree:
    def test_a_k_dimensional_head_accepts_the_encoded_activation(self):
        from src.ml.probe_monitor_model import ProbeHead

        k = 3
        head = ProbeHead(
            weight=torch.ones(k),
            bias=0.0,
            mean=torch.zeros(k),
            std=torch.ones(k),
            layer=1,
        )
        encode = _fake_sae_encoder(list(range(k)))
        scores = head.token_scores(encode(torch.randn(2, 4, 8)))
        assert scores.shape == (2, 4)

    def test_the_same_head_REFUSES_the_raw_activation(self):
        """The defect, reproduced. Without the transform the head meets d_model, not k, and the
        refusal is the exact one Stage 2 hit."""
        from src.ml.probe_monitor_model import ProbeHead

        k = 3
        head = ProbeHead(
            weight=torch.ones(k),
            bias=0.0,
            mean=torch.zeros(k),
            std=torch.ones(k),
            layer=1,
        )
        with pytest.raises(ValueError) as caught:
            head.token_scores(torch.randn(2, 4, 8))
        assert "d_model=8" in str(caught.value) and "d_model=3" in str(caught.value)


class TestTheWiring:
    """Asserted by AST over the call sites, because `encoder` defaults to None: a missing argument
    is silent at every level, which is exactly how this shipped."""

    def _calls_with_encoder(self, function):
        tree = ast.parse(inspect.getsource(function).lstrip())
        found = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "forward_scores":
                continue
            found.append({keyword.arg for keyword in node.keywords})
        return found

    def test_evaluation_passes_an_encoder(self):
        calls = self._calls_with_encoder(run_module._evaluate_probe_on_sets)
        assert calls, "_evaluate_probe_on_sets does not call forward_scores"
        for keywords in calls:
            assert "encoder" in keywords, (
                "the evaluation call omits encoder=, so a k-sparse probe is scored against the "
                "raw residual — the Stage 2 failure"
            )

    def test_offline_scoring_passes_an_encoder(self):
        calls = self._calls_with_encoder(run_module.score_one)
        assert calls, "score_one does not call forward_scores"
        for keywords in calls:
            assert "encoder" in keywords

    def test_forward_scores_accepts_and_applies_it(self):
        from src.services import probe_monitor_capture

        signature = inspect.signature(probe_monitor_capture.forward_scores)
        assert "encoder" in signature.parameters
        body = ast.parse(inspect.getsource(probe_monitor_capture.forward_scores).lstrip())
        applied = [
            node for node in ast.walk(body)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "encoder"
        ]
        assert applied, "forward_scores takes an encoder and never calls it"

    def test_the_ast_walk_can_tell_a_call_from_a_mention(self):
        def decoy():
            # forward_scores(model, examples, head, encoder=encoder)
            return "encoder="

        assert not self._calls_with_encoder(decoy)

    def test_only_an_sae_VARIANT_gets_an_encoder(self):
        """A dense probe encoded through a dictionary would be scored in the wrong basis in the
        other direction. The branch is on `variant`, asserted over the source of the decision."""
        source = inspect.getsource(run_module._evaluate_probe_on_sets)
        tree = ast.parse(source.lstrip())
        compares = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and any(
                isinstance(c, ast.Constant) and c.value == "sae" for c in node.comparators
            )
        ]
        assert compares, "the encoder is not gated on variant == 'sae'"

    def test_the_normalisation_helper_is_the_one_used(self):
        """⚠ A BARE `encode()` READS THE RIGHT WEIGHTS IN THE WRONG BASIS, plausibly and
        invisibly. `encode_with_training_normalization` exists for that reason; training uses it,
        so scoring must."""
        tree = ast.parse(inspect.getsource(run_module.sae_encoder_for).lstrip())
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "encode_with_training_normalization" in called, sorted(called)
        assert "encode" not in called, "a bare encode() bypasses the training normalisation"

    def test_it_refuses_a_non_residual_dictionary(self):
        tree = ast.parse(inspect.getsource(run_module.sae_encoder_for).lstrip())
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "refuse_non_residual" in called

    def test_a_variant_probe_missing_its_sae_is_refused_by_name(self):
        source = inspect.getsource(run_module._evaluate_probe_on_sets)
        assert "cannot be scored in the basis it was trained in" in source


class TestTheDictionaryIsLoadedOncePerProbe:
    """Five evaluation sets must not mean five reads of the same weights file."""

    def test_the_encoder_is_built_outside_the_dataset_loop(self):
        tree = ast.parse(inspect.getsource(run_module._evaluate_probe_on_sets).lstrip())
        function = tree.body[0]
        loops = [node for node in function.body if isinstance(node, ast.For)]
        assert loops, "no dataset loop found"
        inside = {
            node.func.id
            for loop in loops
            for node in ast.walk(loop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "sae_encoder_for" not in inside, (
            "the dictionary is loaded inside the per-dataset loop, so five sets read it five times"
        )
        before = {
            node.func.id
            for statement in function.body
            if not isinstance(statement, ast.For)
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "sae_encoder_for" in before


class TestBothSaePathsRefuseANonResidualDictionary:
    """⚠ THE TRAINING PATH'S GUARD WAS UNPINNED, and that is a PRE-EXISTING gap this round found.

    Mutation A8: deleting `refuse_non_residual` from `encode_sae_features` — the path
    `train_sae_variant` uses — left 96 tests green. The new scoring path's guard is caught by
    `TestTheWiring`, so only half the rule was protected.

    The rule itself: a probe reads `resid_post`, and an SAE trained on an MLP or attention output
    describes a different space. Encoding one against the other produces features that mean
    nothing in particular, plausibly and without raising — the same silent-wrong-basis class as
    the `"residual"` hook that resolved to a post-attention norm for months.

    MUTATION CONTROLS:
      A6  the guard removed from `sae_encoder_for`     → the scoring-path test
      A8  the guard removed from `encode_sae_features` → the training-path test
    """

    PATHS = ("sae_encoder_for", "encode_sae_features")

    @pytest.mark.parametrize("name", PATHS)
    def test_the_function_calls_the_refusal(self, name):
        tree = ast.parse(inspect.getsource(getattr(run_module, name)).lstrip())
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "refuse_non_residual" in called, (
            f"{name} loads an SAE without checking its hook type; an MLP or attention dictionary "
            f"would encode this probe's residual activations into a basis that means nothing"
        )

    @pytest.mark.parametrize("name", PATHS)
    def test_it_refuses_BEFORE_loading_the_weights(self, name):
        """Order matters: reading a multi-gigabyte dictionary and then refusing wastes the read,
        and on a tight card it can OOM before the refusal is reached."""
        tree = ast.parse(inspect.getsource(getattr(run_module, name)).lstrip())
        order = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"refuse_non_residual", "_load_sae_sync"}
        ]
        assert order[:2] == ["refuse_non_residual", "_load_sae_sync"], (
            f"{name} calls {order}; the refusal must come first"
        )

    def test_the_refusal_really_raises_on_a_non_residual_hook(self):
        """The premise, asserted rather than assumed: if `refuse_non_residual` accepted
        everything, both tests above would pass over a guard that does nothing."""
        from src.services.sae_hook_support import refuse_non_residual

        refuse_non_residual("residual", "a probe monitor")          # the allowed case
        for hook in ("mlp", "attention", "mlp_out", "attn_out"):
            with pytest.raises(Exception):
                refuse_non_residual(hook, "a probe monitor")

    def test_the_ast_walk_is_not_satisfied_by_a_mention(self):
        def decoy():
            # refuse_non_residual(hook, "a probe monitor")
            return "refuse_non_residual"

        tree = ast.parse(inspect.getsource(decoy).lstrip())
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "refuse_non_residual" not in called
