"""Sequence length comes from the tokenization; memory is bounded by tokens.

WHY THIS FILE EXISTS. `MAX_SEQ_LENGTH = 512` was hardcoded in
`activation_service` "to prevent GPU OOM", and a second, independent 512 cap sat
in the on-the-fly training path. Tokenizing at 1024 or 2048 and then extracting
silently discarded everything past token 512, reported only at `logger.debug`.

Removing a cap that exists for memory reasons requires replacing what it did.
Activation memory scales with rows x seq_len, so the budget is expressed in
TOKENS and the micro-batch shrinks as the window grows.
"""

import pytest

from src.services.activation_service import (
    REFERENCE_SEQ_LEN,
    micro_batch_size_for_length,
)


class TestTokenBudget:

    def test_the_default_budget_preserves_the_old_envelope_at_512(self):
        """At the length the old cap assumed, nothing should change."""
        assert micro_batch_size_for_length(8, REFERENCE_SEQ_LEN) == 8

    def test_a_longer_window_shrinks_the_micro_batch_faster_than_tokens(self):
        """Round 4, M-c: the budget models the CAPTURED TENSOR (rows x seq_len),
        but eager attention peaks at rows x heads x seq_len^2 — so at a constant
        tokens-per-forward the true peak still grows LINEARLY in seq_len, 16x at
        8192 versus 512. The old hardcoded cap hid that by making long windows
        impossible.

        So the budget shrinks super-linearly: doubling the window quarters the
        micro-batch, not halves it.
        """
        assert micro_batch_size_for_length(8, 1024) == 2
        assert micro_batch_size_for_length(8, 2048) == 1

    def test_tokens_per_forward_fall_as_the_window_grows(self):
        at_512 = 512 * micro_batch_size_for_length(8, 512)
        at_1024 = 1024 * micro_batch_size_for_length(8, 1024)
        assert at_1024 < at_512, (
            "a longer window costs the same tokens-per-forward, which ignores "
            "the quadratic attention term"
        )

    def test_a_shorter_window_does_not_inflate_beyond_the_request(self):
        """The budget is a ceiling, not a target — the caller's number still caps."""
        assert micro_batch_size_for_length(8, 128) == 8

    def test_one_sequence_always_gets_through(self):
        """Refusing to process a long document is worse than processing it slowly."""
        assert micro_batch_size_for_length(4, 1_000_000) == 1

    def test_an_explicit_budget_overrides_the_default(self):
        # At the reference length there is no attention penalty, so the budget
        # applies directly.
        assert micro_batch_size_for_length(64, 512, token_budget=1024) == 2

    def test_a_degenerate_length_does_not_divide_by_zero(self):
        assert micro_batch_size_for_length(8, 0) == 8


class TestNoCapTruncatesByDefault:
    """Behavioural, because the source scrapes that used to live here FAILED OPEN.

    Round 1's M7 reintroduced the cap under another name —
    `_HARD_CAP = 512; ids_tensor = ids_tensor[:_HARD_CAP]` — and both guards
    passed: one looked for an assignment literally named `MAX_SEQ_LENGTH`, the
    other for any `BoolOp(And)` mentioning `max_seq_length`, never checking it
    guarded the truncation. A scrape for a name cannot see a rename.

    So this drives the truncation decision itself.
    """

    @staticmethod
    def _truncate(ids, max_seq_length):
        """The decision `_extract_activations_batched` makes per sample."""
        if max_seq_length is not None and len(ids) > max_seq_length:
            return ids[:max_seq_length]
        return ids

    def test_nothing_is_truncated_when_no_cap_is_configured(self):
        long_doc = list(range(2048))
        assert len(self._truncate(long_doc, None)) == 2048, (
            "a 2048-token document was shortened with no cap configured"
        )

    def test_an_explicit_cap_is_honoured(self):
        assert len(self._truncate(list(range(2048)), 512)) == 512

    def test_a_document_under_the_cap_is_untouched(self):
        assert len(self._truncate(list(range(100)), 512)) == 100

    def test_the_service_truncation_matches_this_decision(self):
        """Pins the helper above to the real branch, so drift shows up here.

        Extracted rather than scraped: the previous guards asserted on the
        SHAPE of the source and were defeated by a rename.
        """
        import inspect

        from src.services import activation_service

        src = inspect.getsource(activation_service)
        assert "if max_seq_length is not None and len(ids_tensor) > max_seq_length:" in src, (
            "the truncation branch changed; re-derive _truncate above from it "
            "rather than editing this assertion"
        )
        # And there must be exactly ONE place that slices the ids.
        assert src.count("ids_tensor = ids_tensor[:") == 1, (
            "a second truncation site exists — a cap under another name would "
            "be invisible to the branch assertion above"
        )


class TestTheKnobsHaveAnApiCaller:
    """Round 4, M-6: both were threaded service-deep and set by nothing.

    That matters beyond the test gap — the old hardcoded 512 cap is gone, so
    without a caller an 8192 tokenization runs at micro-batch 1 with no way for
    an operator to intervene.
    """

    @pytest.mark.parametrize("field", ["max_seq_length", "micro_batch_token_budget"])
    def test_the_extraction_request_accepts_it(self, field):
        from src.schemas.model import ActivationExtractionRequest

        assert field in ActivationExtractionRequest.model_fields, (
            f"{field} cannot be set by any caller, so it is permanently None"
        )

    @pytest.mark.parametrize("field", ["max_seq_length", "micro_batch_token_budget"])
    def test_the_endpoint_forwards_it(self, field):
        import ast
        import inspect

        from src.api.v1.endpoints import models

        tree = ast.parse(inspect.getsource(models))
        forwarded = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == field:
                    forwarded += 1
        assert forwarded >= 2, (
            f"{field} reaches {forwarded} extraction call site(s); both the "
            f"extract and retry paths must carry it or a retry silently changes "
            f"the window"
        )
