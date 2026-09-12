"""The fourteen mutations that survived round 4 — simultaneously.

Round 4 applied all fourteen at once and the full suite returned
`4770 passed | 7 failed`, byte-identical to baseline: the arc's core data-path
behaviour could be reverted wholesale and nothing noticed.

Every one was a guard that watched the line which COMPUTES rather than the one
which APPLIES, or a source scrape satisfied by a different occurrence. This file
replaces them with behaviour.
"""

import ast
import inspect

import numpy as np
import pytest

from src.services.tokenization_service import TokenizationService


def _calls(module) -> set:
    tree = ast.parse(inspect.getsource(module))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute):
                out.add(n.func.attr)
            elif isinstance(n.func, ast.Name):
                out.add(n.func.id)
    return out


class TestPackingChoosesTheRightTokenizerArguments:
    """H8. Inverting one ternary turns 36 genuinely-full blocks into 1,002
    blocks whose mask CLAIMS 100% real over 3.5%-real data."""

    def test_packing_disables_padding_and_truncation(self):
        padding, truncation = TokenizationService.resolve_padding_and_truncation(
            True, "max_length", "longest_first"
        )
        assert padding == "do_not_pad", (
            "packing with padding on pads each document first, which is exactly "
            "what packing undoes"
        )
        assert truncation is False, (
            "truncating discards the tail that should start the next block"
        )

    def test_not_packing_leaves_the_request_alone(self):
        """NEGATIVE CONTROL — inverting the ternary must not pass this."""
        assert TokenizationService.resolve_padding_and_truncation(
            False, "max_length", "longest_first"
        ) == ("max_length", "longest_first")

    def test_the_worker_uses_the_resolver(self):
        from src.workers import dataset_tasks

        assert "resolve_padding_and_truncation" in _calls(dataset_tasks)


class TestTheTextColumnOverrideIsHonoured:
    """H9. The arc's single most-cited motivating defect: Bloomberg tokenized on
    `Headline` while `Article` — 41x the content — went unused."""

    def test_an_explicit_column_beats_auto_detection(self):
        assert TokenizationService.resolve_text_column(
            "Article", "Headline", ["Headline", "Article"]
        ) == "Article"

    def test_auto_detection_is_used_when_nothing_is_requested(self):
        assert TokenizationService.resolve_text_column(
            None, "Headline", ["Headline", "Article"]
        ) == "Headline"

    def test_a_missing_column_is_refused_not_silently_ignored(self):
        """Falling back to detection IS the defect."""
        with pytest.raises(ValueError, match="Article"):
            TokenizationService.resolve_text_column(
                "Article", "Headline", ["Headline", "Body"]
            )

    def test_the_worker_uses_the_resolver(self):
        from src.workers import dataset_tasks

        assert "resolve_text_column" in _calls(dataset_tasks)


class TestTheMaskIsAppliedNotJustComputed:
    """H4 and M-7. The previous guards scraped for the line that COMPUTES the
    mask; the line that APPLIES it was unguarded, so the result could be
    discarded with the suite green."""

    def test_the_shared_builder_returns_masks_that_exclude_padding(self):
        from src.services import activation_mask

        _, attn, lengths, _, _ = activation_mask.build_padded_batch(
            [[7] * 3 + [0] * 13], [[1] * 3 + [0] * 13], 16, pad_token_id=0
        )
        assert lengths == [3]
        assert sum(attn[0]) == 3, "the mask marks padding as real"

    def test_the_training_loop_indexes_by_the_mask(self):
        """`acts_flat = kept` is the line that applies it."""
        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        assert "acts_flat = kept" in src, (
            "the on-the-fly mask is computed and its result discarded — R1's H3 "
            "restored, ~97% PAD on the Bloomberg tokenization"
        )


class TestTheMixtureWeightsReachTheAllocator:
    """H5. Round 2's M5 survived three times: the guard was satisfied by an
    unrelated `normalise_weights(requested_weights, ...)` call below it."""

    def test_the_allocator_is_called_with_the_requested_weights(self):
        from src.workers import training_tasks

        tree = ast.parse(inspect.getsource(training_tasks))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "allocate_tokens"
            ):
                names = [
                    a.id for a in node.args if isinstance(a, ast.Name)
                ]
                assert "requested_weights" in names, (
                    "allocate_tokens is called without the requested weights, so "
                    "the mixture always follows availability"
                )
                return
        pytest.fail("allocate_tokens is never called")


class TestTheHoldoutIsEvaluatedNotJustDeleted:
    """H1. `holdout_by_extraction` was populated and never read, so the knob
    deleted 10% of the corpus and produced no out-of-sample number at all."""

    def test_the_held_out_activations_are_gathered(self):
        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        assert "holdout_activations" in src

    def test_a_held_out_metric_row_is_written(self):
        """Structural, not a string match.

        My first version of this test scraped for the log message and the
        mutation renamed it — the assertion then matched a DIFFERENT occurrence
        elsewhere in the module and survived. The evidence that the holdout is
        measured is that a metric row is written for it, marked by a negative
        `layer_idx` so held-out and in-sample rows are never averaged together.
        """
        from src.workers import training_tasks

        tree = ast.parse(inspect.getsource(training_tasks))
        marked = False
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "log_metric"
            ):
                continue
            for kw in node.keywords:
                if kw.arg == "layer_idx" and isinstance(kw.value, ast.BinOp):
                    # `-1 - int(sae_key[0])` — the held-out marker.
                    marked = True
        assert marked, (
            "no log_metric call marks a row as held-out, so the reserved tokens "
            "are never recorded and the split only deletes training data"
        )

    def test_the_holdout_is_evaluated_under_no_grad(self):
        """Evaluating inside the training graph would leak it into the update."""
        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        idx = src.index("holdout_activations.items()")
        window = src[idx:idx + 1200]
        assert "torch.no_grad()" in window
        assert ".eval()" in window


class TestTokenizationSelectionEverywhere:
    """H6. Wired to two of FOUR sites — one of the misses was SAE feature
    extraction, which produces every feature's example contexts."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "src.workers.model_tasks",
            "src.workers.training_tasks",
            "src.services.extraction_service",
            "src.services.circuit_capture_service",
        ],
    )
    def test_every_site_uses_the_selector(self, module_path):
        import importlib

        module = importlib.import_module(module_path)
        assert "select_tokenization_for_model" in _calls(module), (
            f"{module_path} picks a tokenization by row order, so which context "
            f"window it reads is decided by Postgres"
        )


class TestTheRetryUsesTheIdItAdvertises:
    """Round 4, L-3: the retry generated `new_extraction_id`, logged it,
    returned it to the caller, and opened a WebSocket channel on it — then never
    passed it to the task, so the worker minted its own.

    The client watched a channel nothing published to, and the response named a
    row that does not exist.
    """

    @staticmethod
    def _retry_delay_kwargs():
        from src.api.v1.endpoints import models

        tree = ast.parse(inspect.getsource(models))
        out = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "delay"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "extract_activations"
            ):
                out.append({
                    kw.arg: (
                        kw.value.id if isinstance(kw.value, ast.Name) else None
                    )
                    for kw in node.keywords
                })
        return out

    def test_the_generated_id_is_passed_to_the_task(self):
        calls = self._retry_delay_kwargs()
        assert calls, "no extract_activations.delay call found"

        using_generated = [
            c for c in calls if c.get("extraction_id") == "new_extraction_id"
        ]
        assert using_generated, (
            "the retry does not pass the extraction_id it generated, so the "
            "worker creates a different row than the one the API returned and "
            "the WebSocket channel is published to by nobody"
        )

    def test_the_budget_knobs_reach_both_extraction_call_sites(self):
        """M-6 again, from the call side rather than the schema side."""
        calls = self._retry_delay_kwargs()
        assert len(calls) >= 2, "expected an extract and a retry call site"
        for i, c in enumerate(calls):
            assert "max_seq_length" in c, f"call site {i} drops max_seq_length"
            assert "micro_batch_token_budget" in c, (
                f"call site {i} drops micro_batch_token_budget"
            )
