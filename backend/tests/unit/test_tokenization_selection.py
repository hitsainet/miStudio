"""An extraction must run on ITS OWN model's token ids.

WHY THIS FILE EXISTS. `extract_activations` took `dataset.tokenizations[0]` on
an unordered relationship, with a comment justifying it as "the most common case
is one tokenization per dataset". Datasets here are tokenized for several
models, so the row Postgres happened to return first decided which tokenizer's
ids a model was fed. `activation_service` then clamps out-of-range ids to
`vocab_size - 1` — collapsing them onto one token — and only logs per sample.

MEASURED IN PRODUCTION 2026-09-11, and worse than the clamping suggests:
**20 of the 22 extractions on disk** had consumed another model's tokenization.
Comparing vocabularies id-by-id, agreement is 0.00% for every pair involved —
0 of 64,402 between LFM2.5-1.2B-Instruct and LFM2.5-2.6B, which share a vendor
and a family name. Bloomberg row 0 was written as

    'Ivory Coast Keeps Cocoa Export Tax Below 22%, Document Shows'

and read by LFM2.5-1.2B as

    '<|file_end|>\x1b129 express展<|reserved_330|>ショ<|reserved_63|> nood
     impression actividades<|reserved_219|> theseeten regression her iss'

So the clamped 3.7% was merely the part that warned. **The dangerous direction
is the quiet one:** when the tokenization's vocabulary is SMALLER than the
model's, every id is in range, nothing clamps, nothing warns, and the model
reads different text at full volume. Four of the six gemma-4-12B extractions
were that case. The selector must therefore refuse on IDENTITY, never on
whether ids happen to fit.
"""

import pytest

from src.services.activation_service import ActivationExtractionError
from src.workers.model_tasks import select_tokenization_for_model


class _Tok:
    def __init__(self, id, model_id, max_length=512, tokenized_path="/data/x"):
        self.id = id
        self.model_id = model_id
        self.max_length = max_length
        self.tokenized_path = tokenized_path


class TestItPicksThisModelsTokenization:

    def test_the_matching_model_wins_regardless_of_order(self):
        """The defect was order-dependence, so the fix must not be."""
        wrong = _Tok("tok_a", "m_2_6B")
        right = _Tok("tok_b", "m_1_2B")

        for order in ([wrong, right], [right, wrong]):
            assert select_tokenization_for_model(order, "m_1_2B", "ds_1") is right

    def test_it_refuses_when_no_tokenization_matches(self):
        """Refusing is the whole point — the fallback IS the bug."""
        with pytest.raises(ActivationExtractionError) as exc:
            select_tokenization_for_model([_Tok("tok_a", "m_other")], "m_1_2B", "ds_1")

        message = str(exc.value)
        assert "m_1_2B" in message
        assert "m_other" in message, "the error must say what IS available"

    def test_it_refuses_an_empty_or_missing_relationship(self):
        for value in ([], None):
            with pytest.raises(ActivationExtractionError):
                select_tokenization_for_model(value, "m_1_2B", "ds_1")

    def test_a_tokenization_with_no_path_does_not_count_as_a_match(self):
        """A row without a path cannot be read; matching on model alone would
        pick it and then fail later with a less useful message."""
        with pytest.raises(ActivationExtractionError):
            select_tokenization_for_model(
                [_Tok("tok_a", "m_1_2B", tokenized_path=None)], "m_1_2B", "ds_1"
            )

    def test_the_longest_context_wins_among_this_models_tokenizations(self):
        short = _Tok("tok_a", "m_1_2B", max_length=512)
        long = _Tok("tok_b", "m_1_2B", max_length=2048)
        assert select_tokenization_for_model([short, long], "m_1_2B", "ds_1") is long
        assert select_tokenization_for_model([long, short], "m_1_2B", "ds_1") is long

    def test_ties_are_broken_deterministically_not_by_row_order(self):
        a = _Tok("tok_a", "m_1_2B", max_length=512)
        b = _Tok("tok_b", "m_1_2B", max_length=512)
        assert (
            select_tokenization_for_model([a, b], "m_1_2B", "ds_1")
            is select_tokenization_for_model([b, a], "m_1_2B", "ds_1")
        )

    def test_a_null_max_length_does_not_crash_the_comparison(self):
        weird = _Tok("tok_a", "m_1_2B", max_length=None)
        normal = _Tok("tok_b", "m_1_2B", max_length=512)
        assert select_tokenization_for_model([weird, normal], "m_1_2B", "ds_1") is normal


class TestTheQuietDirectionIsRefusedToo:
    """The clamping warning only fires when the tokenizer's vocab is LARGER than
    the model's. Four production extractions went the other way and were silent.

    The selector must not have any notion of "the ids fit, so it is fine" — it
    matches on model identity alone. These tests fail if anyone ever adds a
    vocab-range escape hatch.
    """

    def test_a_smaller_vocab_tokenization_is_still_refused(self):
        """granite ids into gemma: every id in range, no clamp, no warning."""
        granite_tok = _Tok("tok_granite", "m_granite_8b")

        with pytest.raises(ActivationExtractionError) as exc:
            select_tokenization_for_model([granite_tok], "m_gemma_12b", "ds_1")

        assert "m_granite_8b" in str(exc.value)

    def test_same_vendor_and_family_is_not_a_match(self):
        """LFM2.5-1.2B-Instruct and LFM2.5-2.6B agree on 0 of 64,402 ids. A
        selector that relaxed to a repo-name prefix would reintroduce the
        exact 15 extractions this fix exists for."""
        sibling = _Tok("tok_2_6B", "m_lfm25_2_6b")

        with pytest.raises(ActivationExtractionError):
            select_tokenization_for_model([sibling], "m_lfm25_1_2b_instruct", "ds_1")

    def test_it_still_returns_the_match_when_one_is_present(self):
        """NEGATIVE CONTROL — a selector that refused everything would pass both
        tests above. This one must not."""
        sibling = _Tok("tok_2_6B", "m_lfm25_2_6b")
        mine = _Tok("tok_1_2B", "m_lfm25_1_2b_instruct")

        assert (
            select_tokenization_for_model([sibling, mine], "m_lfm25_1_2b_instruct", "ds_1")
            is mine
        )


class TestAnOverrideMustNotPointBackAtAConsumedColumn:
    """R: naming the raw conversation column produced a corpus of pure EOS.

    `teknium/OpenHermes-2.5` has a `conversations` column of ShareGPT turns.
    Preprocessing renders it through the chat template into a NEW column, and
    detection then points at that. Passing text_column="conversations" — the
    obviously correct-looking choice, and the one the UI's dropdown invites
    because it lists the RAW columns — overrode the rendered text with the raw
    list-of-dicts. Result: 490 blocks, every token `<|im_end|>`, reported READY.
    """

    def test_naming_the_consumed_column_resolves_to_the_rendered_one(self):
        from src.services.tokenization_service import TokenizationService

        resolved = TokenizationService.resolve_text_column(
            "conversations",
            "text",                       # what preprocessing rendered into
            ["conversations", "text"],
            rendered_from="conversations",
        )
        assert resolved == "text"

    def test_without_preprocessing_the_same_name_is_honoured(self):
        """NEGATIVE CONTROL. A guard that always returned `detected` would pass
        the test above and silently undo the Bloomberg fix."""
        from src.services.tokenization_service import TokenizationService

        assert TokenizationService.resolve_text_column(
            "Article", "Headline", ["Headline", "Article"], rendered_from=None
        ) == "Article"

    def test_a_different_override_still_wins_after_preprocessing(self):
        """Only the CONSUMED column is redirected; any other explicit choice is
        still the operator's to make."""
        from src.services.tokenization_service import TokenizationService

        assert TokenizationService.resolve_text_column(
            "system_prompt",
            "text",
            ["conversations", "text", "system_prompt"],
            rendered_from="conversations",
        ) == "system_prompt"

    def test_the_worker_passes_the_consumed_column(self):
        """Reachability — the parameter is useless if the caller never fills it."""
        import ast
        import inspect

        from src.workers import dataset_tasks

        tree = ast.parse(inspect.getsource(dataset_tasks))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "resolve_text_column"
            ):
                names = {kw.arg for kw in node.keywords}
                assert "rendered_from" in names, (
                    "the worker must tell the resolver which column "
                    "preprocessing consumed"
                )
                return
        raise AssertionError("no resolve_text_column call found")
