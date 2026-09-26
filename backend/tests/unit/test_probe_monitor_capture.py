"""Capture loops, against a REAL (tiny, random) causal LM — no GPU, no download.

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M107  the pooled mean divides by the row WIDTH instead of the scored count
  M108  the last-scored index is found from the LEFT
  M109  `plan_batches` sizes by row count instead of tokens
  M110  the scored mask is dropped (padding reaches the pooling)
  M111  `capture_tokens` appends instead of pre-sizing from the offsets
  M112  an empty-mask row is zero-filled instead of reported
  M113  `forward_scores` passes the scores as the attention logits
  M114  `capture_pooled` stops checking the registered hook count
  M115  `with_oom_retry` retries forever instead of once

⚠ WHY A REAL MODEL. The thing under test is that hooks fire at the decoder layer's
output and that a mask lines up with what the model actually produced. A stubbed
"model" that returns a crafted tensor makes both true by construction — and the
capture point is the one thing on this estate that was silently wrong for months
(`"residual"` resolved to a post-attention norm until 2026-09-12; every SAE learned a
signal miLLM never reads, and nothing looked broken). These tests run a 4-layer
random Llama on CPU and compare against `output_hidden_states`, which is the only
independent witness to where `resid_post` is.
"""
import numpy as np
import pytest
import torch

from src.services.probe_monitor_capture import (
    DEFAULT_TOKEN_BUDGET,
    ProbeCaptureOOM,
    capture_pooled,
    capture_tokens,
    count_scored_tokens,
    forward_scores,
    plan_batches,
    with_oom_retry,
)
from src.services.probe_monitor_render import RenderedExample

D_MODEL = 32
N_LAYERS = 4


@pytest.fixture(scope="module")
def tiny_model():
    """A real `LlamaForCausalLM` with random weights: 4 layers, d=32."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=D_MODEL,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def _example(ids, roles=None):
    roles = roles or ["user"] * len(ids)
    return RenderedExample(
        input_ids=list(ids),
        token_roles=list(roles),
        token_message=[0] * len(ids),
        text="",
    )


def _examples(n=6, length=5):
    return [_example(list(range(1, length + 1))) for _ in range(n)]


class TestTheHookFiresAtTheDECODERLAYEROutput:
    def test_the_pooled_vector_matches_output_hidden_states(self, tiny_model):
        """⚠ THE ONE INDEPENDENT WITNESS. `output_hidden_states[L + 1]` IS the output
        of decoder layer L, and comparing against it is the only way to prove the hook
        is at `resid_post` and not at a norm inside the block. Reading the code cannot:
        a norm's output has the right shape and plausible values, which is exactly how
        every SAE here was trained on the wrong signal for months."""
        ids = [1, 2, 3, 4, 5]
        examples = [_example(ids)]
        captured = capture_pooled(tiny_model, examples, [2], scope="all")

        with torch.no_grad():
            reference = tiny_model(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                output_hidden_states=True,
            )
        expected = reference.hidden_states[3][0].to(torch.float32)   # layer 2's OUTPUT
        assert torch.allclose(
            captured.mean[2][0], expected.mean(dim=0), atol=1e-4
        ), "the pooled mean does not match decoder layer 2's output"
        assert torch.allclose(captured.last[2][0], expected[-1], atol=1e-4)

    def test_it_is_NOT_the_post_attention_norm(self, tiny_model):
        """Proves the comparison above discriminates. If the two points agreed, the
        test would pass against the defect it exists to catch."""
        ids = [1, 2, 3, 4, 5]
        with torch.no_grad():
            hidden = tiny_model(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                output_hidden_states=True,
            ).hidden_states[3][0]
        layer = tiny_model.model.layers[2]
        normed = layer.post_attention_layernorm(hidden)
        assert not torch.allclose(hidden, normed, atol=1e-3), (
            "on this model the layer output and the post-attention norm are "
            "indistinguishable, so the test above cannot detect a capture at the norm"
        )

    def test_every_requested_layer_is_captured_in_ONE_pass(self, tiny_model):
        """The whole point of pooled mode: a full sweep costs one pass, not one per
        layer."""
        captured = capture_pooled(tiny_model, _examples(4), [0, 1, 2, 3])
        assert sorted(captured.mean) == [0, 1, 2, 3]
        assert all(captured.mean[layer].shape == (4, D_MODEL) for layer in range(4))

    def test_a_layer_beyond_the_model_is_REFUSED_not_skipped(self, tiny_model):
        """`register_hooks` warns and skips an out-of-range layer, so without this
        check the run would complete having captured fewer layers than it swept.

        ⚠ IT MATCHES THE GUARD'S OWN WORDING, NOT "out of range". The first version
        matched that phrase — which `_layer_activation`'s fallback ALSO contains — so
        deleting the guard left the test green while the run failed later and less
        clearly. Found by a mutation that survived.
        """
        with pytest.raises(RuntimeError, match="hook registration mismatch"):
            capture_pooled(tiny_model, _examples(2), [0, 99])

    def test_that_message_is_DISTINCT_from_the_missing_activation_fallback(self):
        """Prove the two errors cannot be confused, or the test above is back to
        passing against either."""
        import inspect

        from src.services import probe_monitor_capture as module

        guard = inspect.getsource(module.capture_pooled)
        fallback = inspect.getsource(module._layer_activation)
        assert "hook registration mismatch" in guard
        assert "hook registration mismatch" not in fallback


class TestThePoolingHonoursTheMask:
    def test_the_mean_divides_by_the_SCORED_COUNT_not_the_row_width(self, tiny_model):
        """Dividing by the padded width makes the value depend on the batch's longest
        row — the defect that cost this estate an entire SAE corpus, where 48% of every
        batch was padding."""
        short = _example([1, 2])
        long = _example(list(range(1, 21)))
        batched = capture_pooled(tiny_model, [short, long], [1])
        alone = capture_pooled(tiny_model, [short], [1])
        assert torch.allclose(batched.mean[1][0], alone.mean[1][0], atol=1e-4), (
            "the short row's mean changed when a longer row shared its batch, so "
            "padding is reaching the pooling"
        )

    def test_the_last_vector_is_the_last_REAL_token(self, tiny_model):
        short = _example([1, 2])
        long = _example(list(range(1, 21)))
        batched = capture_pooled(tiny_model, [short, long], [1])
        alone = capture_pooled(tiny_model, [short], [1])
        assert torch.allclose(batched.last[1][0], alone.last[1][0], atol=1e-4), (
            "the last-token vector landed on a pad position"
        )

    def test_a_role_scope_changes_what_is_pooled(self, tiny_model):
        mixed = RenderedExample(
            input_ids=[1, 2, 3, 4],
            token_roles=["user", "user", "assistant", "assistant"],
            token_message=[0, 0, 1, 1],
            text="",
        )
        every = capture_pooled(tiny_model, [mixed], [1], scope="all")
        assistant = capture_pooled(tiny_model, [mixed], [1], scope="assistant")
        assert not torch.allclose(every.mean[1][0], assistant.mean[1][0], atol=1e-5), (
            "the scope did not reach the pooling, so an 'assistant' probe would train "
            "on the user's tokens too"
        )

    def test_an_empty_mask_row_is_REPORTED_not_zero_filled(self, tiny_model):
        """A zero vector is a legitimate activation, so an all-zero row would train as
        if it were data."""
        nothing = RenderedExample(
            input_ids=[1, 2, 3],
            token_roles=["user", "user", "user"],
            token_message=[0, 0, 0],
            text="",
        )
        captured = capture_pooled(tiny_model, [nothing], [1], scope="assistant")
        assert captured.empty_rows == [0]

    def test_an_empty_row_is_counted_ONCE_not_per_layer(self, tiny_model):
        nothing = RenderedExample(
            input_ids=[1, 2], token_roles=["user", "user"], token_message=[0, 0], text=""
        )
        captured = capture_pooled(tiny_model, [nothing], [0, 1, 2], scope="assistant")
        assert captured.empty_rows == [0], f"counted {len(captured.empty_rows)} times"


class TestBatchesAreSizedByTOKENS:
    def test_a_long_row_gets_a_smaller_batch(self):
        short = [_example([1, 2]) for _ in range(20)]
        batches = plan_batches(short, token_budget=20)
        assert all(len(b) * 2 <= 20 for b in batches)

        long = [_example(list(range(1, 11))) for _ in range(20)]
        long_batches = plan_batches(long, token_budget=20)
        assert max(len(b) for b in long_batches) < max(len(b) for b in batches), (
            "batches are sized by row count, so the longest row in the set decides "
            "whether the pass OOMs"
        )

    def test_the_budget_accounts_for_PADDING(self):
        """A batch costs rows x widest, not the sum of lengths."""
        examples = [_example([1]), _example(list(range(1, 51)))]
        batches = plan_batches(examples, token_budget=60)
        assert len(batches) == 2, (
            "a 1-token row and a 50-token row were batched together under a 60-token "
            "budget; padded, that batch is 100 tokens"
        )

    def test_every_row_appears_exactly_once(self):
        examples = _examples(37, length=3)
        flat = [i for batch in plan_batches(examples, token_budget=12) for i in batch]
        assert sorted(flat) == list(range(37))

    def test_a_row_wider_than_the_budget_still_gets_its_own_batch(self):
        """Dropping it would remove exactly the hardest examples from an evaluation,
        and truncating is the render's decision, recorded on the row."""
        examples = [_example(list(range(1, 101)))]
        assert plan_batches(examples, token_budget=10) == [[0]]

    def test_a_zero_budget_is_refused(self):
        with pytest.raises(ValueError, match="token_budget"):
            plan_batches(_examples(2), token_budget=0)

    def test_the_default_budget_is_declared_not_magic(self):
        assert DEFAULT_TOKEN_BUDGET >= 1024


class TestTokenCaptureIsPreSized:
    def test_the_file_is_exactly_as_long_as_the_offsets_say(self, tiny_model, tmp_path):
        """A growing file has no moment at which its length is known, so a crash
        leaves a file whose size cannot be checked against the offsets."""
        examples = _examples(5, length=4)
        destination = tmp_path / "tokens.bin"
        capture = capture_tokens(tiny_model, examples, 1, destination)
        expected_rows = int(capture.offsets[-1])
        assert expected_rows == 5 * 4
        assert destination.stat().st_size == expected_rows * D_MODEL * 2   # fp16

    def test_the_offsets_locate_each_row(self, tiny_model, tmp_path):
        examples = [_example([1, 2]), _example([1, 2, 3, 4, 5])]
        capture = capture_tokens(tiny_model, examples, 1, tmp_path / "t.bin")
        assert capture.row(0).shape == (2, D_MODEL)
        assert capture.row(1).shape == (5, D_MODEL)

    def test_the_stored_rows_match_a_direct_forward(self, tiny_model, tmp_path):
        ids = [1, 2, 3]
        capture = capture_tokens(tiny_model, [_example(ids)], 2, tmp_path / "t.bin")
        with torch.no_grad():
            reference = tiny_model(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
                output_hidden_states=True,
            ).hidden_states[3][0]
        stored = torch.tensor(np.asarray(capture.row(0)), dtype=torch.float32)
        # fp16 storage, so the tolerance is fp16's, not fp32's.
        assert torch.allclose(stored, reference.to(torch.float32), atol=2e-2)

    def test_only_SCORED_tokens_are_stored(self, tiny_model, tmp_path):
        mixed = RenderedExample(
            input_ids=[1, 2, 3, 4],
            token_roles=["user", "user", "assistant", "assistant"],
            token_message=[0, 0, 1, 1],
            text="",
        )
        capture = capture_tokens(
            tiny_model, [mixed], 1, tmp_path / "t.bin", scope="assistant"
        )
        assert int(capture.offsets[-1]) == 2, "unscored positions were stored"

    def test_a_set_with_no_scored_tokens_is_REFUSED(self, tiny_model, tmp_path):
        nothing = RenderedExample(
            input_ids=[1, 2], token_roles=["user", "user"], token_message=[0, 0], text=""
        )
        with pytest.raises(ValueError, match="no scored tokens"):
            capture_tokens(tiny_model, [nothing], 1, tmp_path / "t.bin", scope="assistant")

    def test_count_scored_tokens_needs_no_forward_pass(self):
        """It pre-sizes the memmap, so it must be computable during render."""
        total, offsets = count_scored_tokens([_example([1, 2]), _example([1, 2, 3])])
        assert total == 5
        assert list(offsets) == [0, 2, 5]


class TestForwardScoresIsTheONEScoringPath:
    def _head(self, layer=1):
        from src.ml.probe_monitor_model import ProbeHead

        torch.manual_seed(1)
        # `layer` is a FIELD, not an attribute a caller bolts on: `ProbeHead` is
        # frozen, and the layer is part of the probe's identity — the same weights over
        # a different layer are a different detector. A head whose layer is known only
        # to its caller cannot be serialised or served.
        return ProbeHead(
            weight=torch.randn(D_MODEL),
            bias=0.25,
            mean=torch.zeros(D_MODEL),
            std=torch.ones(D_MODEL),
            attention_query=torch.randn(D_MODEL),
            layer=layer,
        )

    def test_it_returns_one_aggregate_and_the_per_token_trace(self, tiny_model):
        rows = forward_scores(
            tiny_model, _examples(3, length=4), self._head(), rule="mean"
        )
        assert len(rows) == 3
        assert all(row.n_scored == 4 for row in rows)
        assert all(len(row.token_scores) == 4 for row in rows)

    def test_the_aggregate_agrees_with_the_pure_rule_on_the_same_scores(self, tiny_model):
        """Ties the streaming path to the tested pure function: if they disagree, the
        metrics and the exported definition describe different detectors."""
        from src.ml.probe_monitor_model import combine_sequence

        rows = forward_scores(tiny_model, _examples(2, length=5), self._head(), rule="mean")
        for row in rows:
            assert row.aggregate == pytest.approx(
                combine_sequence("mean", row.token_scores), abs=1e-5
            )

    def test_max_and_mean_disagree_so_the_rule_REACHES_the_aggregate(self, tiny_model):
        examples = _examples(2, length=5)
        by_mean = forward_scores(tiny_model, examples, self._head(), rule="mean")
        by_max = forward_scores(tiny_model, examples, self._head(), rule="max")
        assert [r.aggregate for r in by_mean] != [r.aggregate for r in by_max]

    def test_attention_uses_the_LOGITS_not_the_scores(self, tiny_model):
        """⚠ `attention` weights tokens by `softmax(q·ẑ)` while the value stays `w·ẑ`.
        Passing the scores as the logits silently turns it into `softmax` at tau=1 — a
        different detector under the same name."""
        examples = _examples(2, length=6)
        head = self._head()
        by_attention = forward_scores(tiny_model, examples, head, rule="attention")
        by_softmax = forward_scores(tiny_model, examples, head, rule="softmax")
        assert [r.aggregate for r in by_attention] != [r.aggregate for r in by_softmax]

    def test_the_scope_reaches_the_scores(self, tiny_model):
        mixed = RenderedExample(
            input_ids=[1, 2, 3, 4],
            token_roles=["user", "user", "assistant", "assistant"],
            token_message=[0, 0, 1, 1],
            text="",
        )
        assert forward_scores(tiny_model, [mixed], self._head(), rule="mean",
                              scope="assistant")[0].n_scored == 2

    def test_a_head_with_no_layer_is_refused(self, tiny_model):
        from src.ml.probe_monitor_model import ProbeHead

        head = ProbeHead(weight=torch.randn(D_MODEL))
        with pytest.raises(ValueError, match="no layer"):
            forward_scores(tiny_model, _examples(1), head, rule="mean")

    def test_an_empty_input_returns_nothing_rather_than_raising(self, tiny_model):
        assert forward_scores(tiny_model, [], self._head(), rule="mean") == []

    def test_the_rows_come_back_in_INPUT_order(self, tiny_model):
        """Batching reorders work; a score list that does not line up with the labels
        produces an AUROC over shuffled pairs, which looks like a weak probe."""
        examples = [_example([1]), _example(list(range(1, 31))), _example([2, 3])]
        rows = forward_scores(tiny_model, examples, self._head(), rule="mean",
                              token_budget=32)
        assert [row.index for row in rows] == [0, 1, 2]
        assert [row.n_scored for row in rows] == [1, 30, 2]


class TestTheOOMRetryHappensOnce:
    def test_it_halves_the_budget_and_succeeds(self):
        seen = []

        def run(budget):
            seen.append(budget)
            if len(seen) == 1:
                raise torch.cuda.OutOfMemoryError("out of memory")
            return "done"

        assert with_oom_retry(run, token_budget=1000) == "done"
        assert seen == [1000, 500]

    def test_a_second_OOM_FAILS_and_names_the_card(self):
        """"CUDA out of memory" without the card is unactionable on a two-card node."""

        def always(budget):
            raise torch.cuda.OutOfMemoryError("out of memory")

        with pytest.raises(ProbeCaptureOOM, match="twice"):
            with_oom_retry(always, token_budget=1000)

    def test_it_does_NOT_retry_forever(self):
        calls = []

        def always(budget):
            calls.append(budget)
            raise torch.cuda.OutOfMemoryError("out of memory")

        with pytest.raises(ProbeCaptureOOM):
            with_oom_retry(always, token_budget=8)
        assert len(calls) == 2, f"retried {len(calls)} times"

    def test_a_non_OOM_error_is_not_retried(self):
        calls = []

        def broken(budget):
            calls.append(budget)
            raise ValueError("a real bug")

        with pytest.raises(ValueError, match="a real bug"):
            with_oom_retry(broken, token_budget=100)
        assert len(calls) == 1, "a genuine bug was retried as if it were an OOM"
