"""A corpus of one repeated token must not be reported as a successful tokenization.

MEASURED 2026-09-11. `teknium/OpenHermes-2.5` was tokenized with
text_column="conversations" — the name of the RAW conversation column, whose
rendered text lands in a different column. The tokenizer therefore received
structured data, produced nothing per document, and the packer emitted only its
EOS separators: 490 blocks of 2048 in which every token was `<|im_end|>`.

The run succeeded. `status: ready`. The log said "99.8% real tokens". The
statistics block reported 490 full blocks, all in the `1000+` length bucket,
from 1,001,551 documents. Every one of those numbers was TRUE — occupancy counts
EOS as real because the attention mask does.

The only number that knew was `unique_tokens_used: 1`, and nothing read it.
Bloomberg, tokenized correctly minutes earlier, reported 41,715.
"""

import ast
import inspect

import pytest

from src.services.corpus_sanity import (
    MIN_TOKENS_TO_JUDGE,
    MIN_UNIQUE_TOKENS,
    DegenerateCorpusError,
    check_token_diversity,
)


class TestItCatchesTheObservedFailure:

    def test_the_openhermes_corpus_is_refused(self):
        """The exact numbers from the production run."""
        with pytest.raises(DegenerateCorpusError) as exc:
            check_token_diversity(1, 1_003_520, context="text_column='conversations'")

        message = str(exc.value)
        assert "1,003,520" in message
        assert "conversations" in message, "name the likely cause"

    def test_the_bloomberg_corpus_is_accepted(self):
        """NEGATIVE CONTROL — tokenized correctly minutes earlier. A guard that
        refused everything would pass the test above."""
        check_token_diversity(41_715, 275_918_848)

    def test_a_handful_of_tokens_is_still_refused(self):
        with pytest.raises(DegenerateCorpusError):
            check_token_diversity(MIN_UNIQUE_TOKENS - 1, 1_000_000)

    def test_the_threshold_itself_passes(self):
        check_token_diversity(MIN_UNIQUE_TOKENS, 1_000_000)


class TestItDoesNotFailRunsForTheWrongReason:

    def test_a_tiny_corpus_is_exempt(self):
        """A deliberate fixture or a smoke test can legitimately be this small.
        Failing those would make the guard something people switch off."""
        check_token_diversity(1, MIN_TOKENS_TO_JUDGE - 1)

    def test_the_exemption_ends_where_it_should(self):
        with pytest.raises(DegenerateCorpusError):
            check_token_diversity(1, MIN_TOKENS_TO_JUDGE)

    @pytest.mark.parametrize(
        "unique,total",
        [(None, 1_000_000), (1, None), (None, None)],
    )
    def test_unknown_statistics_are_silent(self, unique, total):
        """"I could not compute this" is a different problem from "this is
        broken", and turning the first into the second fails runs for the wrong
        reason."""
        check_token_diversity(unique, total)


class TestTheGuardIsReachable:
    """A capability is not shipped until a test FAILS when its wiring is removed."""

    def test_the_tokenize_task_calls_it(self):
        from src.workers import dataset_tasks

        tree = ast.parse(inspect.getsource(dataset_tasks))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "tokenize_dataset_task":
                called = {
                    sub.func.id
                    for sub in ast.walk(node)
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                }
                assert "check_token_diversity" in called
                return
        raise AssertionError("tokenize_dataset_task not found")

    def test_it_is_given_the_unique_count_and_not_the_vocab_size(self):
        """`stats["vocab_size"]` is OVERWRITTEN with the tokenizer's full vocab
        two lines above the call — 64,400 for this tokenizer. Passing that
        instead of `unique_tokens_used` would make the guard pass on every
        corpus including the broken one, while looking correct."""
        from src.workers import dataset_tasks

        tree = ast.parse(inspect.getsource(dataset_tasks))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "check_token_diversity"
            ):
                first = node.args[0]
                assert isinstance(first, ast.Call), "expected stats.get(...)"
                key = first.args[0]
                assert isinstance(key, ast.Constant)
                assert key.value == "unique_tokens_used", (
                    f"guard is reading {key.value!r}; vocab_size is the "
                    "tokenizer's full vocabulary and is always large"
                )
                return
        raise AssertionError("no check_token_diversity call found")
