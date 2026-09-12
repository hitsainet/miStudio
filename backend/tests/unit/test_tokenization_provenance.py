"""How a tokenization was built must be recorded, and recorded truthfully.

WHY THIS FILE EXISTS. `padding`, `truncation`, `add_special_tokens` and the text
column were request-only: passed to the Celery task and lost. An existing
tokenization carried no record of how it was produced. That is how
`danidanou/Bloomberg_Financial_News` — columns Headline, Journalists, Date,
Link, Article — came to be tokenized on `Headline` (mean 17 tokens) while
`Article` (mean 464, ~207M tokens) went unused, with nothing able to surface it.

Round 2 found the columns had been ADDED and never WRITTEN, which is the same
defect wearing a schema, and that `chat_format` was NOT NULL DEFAULT 'auto' —
so every historical row would claim it used the tokenizer's real chat template,
when every one of them used the `<|role|>` pseudo-markers.
"""

import inspect

import pytest

from src.models.dataset_tokenization import DatasetTokenization
from src.services.tokenization_service import TokenizationService


class TestTheColumnsDoNotLieAboutOldRows:

    def test_chat_format_is_nullable_so_null_means_not_recorded(self):
        col = DatasetTokenization.__table__.columns["chat_format"]
        assert col.nullable, (
            "chat_format is NOT NULL, so every pre-existing row claims a value. "
            "'auto' means 'used the tokenizer's real template' — which is false "
            "for every row written before this column existed."
        )
        assert col.server_default is None, (
            "a server default backfills a claim onto rows that predate it"
        )

    def test_pack_sequences_is_nullable_for_the_same_reason(self):
        col = DatasetTokenization.__table__.columns["pack_sequences"]
        assert col.nullable and col.server_default is None

    @pytest.mark.parametrize(
        "name",
        ["text_column", "chat_format", "pack_sequences", "padding",
         "truncation", "add_special_tokens"],
    )
    def test_the_column_exists(self, name):
        assert name in DatasetTokenization.__table__.columns


class TestTheWorkerActuallyWritesThem:
    """A column added and never written is the same defect wearing a schema."""

    @staticmethod
    def _row_construction_kwargs():
        import ast

        from src.workers import dataset_tasks

        tree = ast.parse(inspect.getsource(dataset_tasks))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "DatasetTokenization"
            ):
                return {kw.arg for kw in node.keywords}
        return set()

    @pytest.mark.parametrize(
        "name",
        ["text_column", "chat_format", "pack_sequences", "padding",
         "truncation", "add_special_tokens",
         "remove_all_punctuation", "custom_filter_chars"],
    )
    def test_the_row_is_created_with_it(self, name):
        kwargs = self._row_construction_kwargs()
        assert kwargs, "no DatasetTokenization(...) construction found"
        assert name in kwargs, (
            f"{name} is never written, so the column stays at its default "
            f"forever and records nothing"
        )

    def test_the_resolved_text_column_is_recorded_after_detection(self):
        """A row built by auto-detection must still say which column it read."""
        from src.workers import dataset_tasks

        src = inspect.getsource(dataset_tasks)
        assert "row.text_column = TokenizationService.resolve_recorded_text_column(" in src, (
            "the resolved column is no longer written back, so a row built by "
            "auto-detection stays silent about which column it read — the "
            "original Bloomberg failure"
        )


class TestThePathCarriesMaxLength:
    """M7 from round 2: this fix was wired and had no test at all, so reverting
    it to the pre-commit expression left the whole suite green."""

    def test_two_lengths_get_two_directories(self):
        a = TokenizationService.tokenized_dataset_path("/data/ds/corpus", "m_1", 512)
        b = TokenizationService.tokenized_dataset_path("/data/ds/corpus", "m_1", 2048)
        assert a != b, (
            "a 512 and a 2048 tokenization of the same dataset resolve to ONE "
            "directory; the second silently overwrites the first while both DB "
            "rows keep pointing at it"
        )

    def test_two_models_get_two_directories(self):
        a = TokenizationService.tokenized_dataset_path("/data/ds/corpus", "m_1", 512)
        b = TokenizationService.tokenized_dataset_path("/data/ds/corpus", "m_2", 512)
        assert a != b

    def test_the_length_is_in_the_name(self):
        p = TokenizationService.tokenized_dataset_path("/data/ds/corpus", "m_1", 2048)
        assert "2048" in p.name
        assert p.parent.name == "ds"

    def test_it_is_deterministic(self):
        args = ("/data/ds/corpus", "m_1", 512)
        assert TokenizationService.tokenized_dataset_path(*args) == \
            TokenizationService.tokenized_dataset_path(*args)


class TestTheRetryPathCallsSomethingThatExists:
    """The API returned 202 and the worker then died on a TypeError."""

    def test_the_retry_passes_model_id_not_tokenizer_name(self):
        from src.api.v1.endpoints import task_queue

        src = inspect.getsource(task_queue)
        assert "model_id=model_id_for_retry" in src, (
            "the tokenization retry does not pass model_id, which the task "
            "requires; .delay() serialises kwargs unchecked so this fails only "
            "in the worker"
        )
        assert "tokenizer_name=retry_params" not in src, (
            "tokenizer_name is not a parameter of tokenize_dataset_task"
        )
        # Round 3, M7: the fix repaired only NEW rows. Every failure written
        # before model_id was recorded — i.e. the whole existing backlog, which
        # is exactly what anyone would click retry on — still returned 202 and
        # died in the worker.
        assert "model_id_for_retry = retry_params.get" in src
        assert "status_code=422" in src, (
            "a retry that cannot supply model_id must be refused at the API, "
            "not accepted and failed in the worker"
        )

    def test_the_retry_carries_the_arcs_parameters(self):
        """A retry that silently changes what the tokenization IS is worse than
        one that fails: the provenance columns then record plausible lies."""
        import inspect

        from src.api.v1.endpoints import task_queue

        src = inspect.getsource(task_queue)
        for field in ("chat_format", "pack_sequences", "remove_all_punctuation",
                      "custom_filter_chars", "return_attention_mask"):
            assert f"{field}=retry_params.get" in src, (
                f"retrying drops {field}, so a packed/chat-templated tokenization "
                f"comes back padded and plain"
            )

    def test_the_failure_blob_records_model_id(self):
        from src.workers import dataset_tasks

        src = inspect.getsource(dataset_tasks)
        assert '"model_id": model_id,' in src, (
            "a retry cannot pass model_id if the failure never recorded it"
        )


class TestTheDecisionsAreFunctionsNotInlineBranches:
    """Round 3 proved both of these were guarded only by source scrapes that
    fail open: deleting the BOS wiring, and writing `text_column = None`, each
    left the full 4777-test suite green.

    Extracting the decision makes it testable; asserting the CALL (by AST, in
    test_wiring_reachable) makes removing it red. A scrape did neither.
    """

    def test_bos_is_suppressed_only_when_the_template_emits_one(self):
        class _Emits:
            chat_template = "t"
            bos_token_id = 1

            def apply_chat_template(self, messages, **kw):
                return "<s>x"

            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": ([1] if text.startswith("<s>") else []) + [9]}

        class _DoesNot(_Emits):
            def apply_chat_template(self, messages, **kw):
                return "x"

        r = TokenizationService.resolve_add_special_tokens
        assert r(True, _Emits(), used_template=True) is False, (
            "a template that emits BOS must suppress the tokenizer's, or every "
            "conversation starts with a doubled BOS"
        )
        assert r(True, _DoesNot(), used_template=True) is True, (
            "NEGATIVE CONTROL: suppressing unconditionally would strip BOS from "
            "templates that do not emit one"
        )

    def test_plain_rendering_never_suppresses_bos(self):
        """Plain text has no template-emitted BOS to double."""
        class _Emits:
            chat_template = "t"
            bos_token_id = 1

            def apply_chat_template(self, messages, **kw):
                return "<s>x"

            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": [1, 9]}

        assert TokenizationService.resolve_add_special_tokens(
            True, _Emits(), used_template=False
        ) is True

    def test_a_caller_that_asked_for_no_special_tokens_is_respected(self):
        assert TokenizationService.resolve_add_special_tokens(
            False, object(), used_template=True
        ) is False

    def test_a_conversation_row_records_the_SOURCE_column(self):
        """`text_column` is the synthetic output name by this point; recording it
        reintroduces the Bloomberg failure in a new place."""
        r = TokenizationService.resolve_recorded_text_column
        assert r(True, {"source_column": "messages"}, "text") == "messages"

    def test_a_plain_row_records_the_column_it_read(self):
        r = TokenizationService.resolve_recorded_text_column
        assert r(False, None, "Article") == "Article"

    def test_it_never_records_nothing(self):
        """Writing None is what round 3's surviving mutation did."""
        r = TokenizationService.resolve_recorded_text_column
        assert r(True, {}, "text") == "text"
        assert r(True, None, "text") == "text"


class TestARetryRebuildsTheProvenance:
    """Round 4, M-2: on a retry the row already exists, so the creation branch
    never runs and the columns keep describing the PREVIOUS attempt.

    A failed packed, chat-templated tokenization retried with different
    parameters came back unpacked and plain while the row still claimed
    `pack_sequences=true` — a column added to stop a silent misattribution,
    producing one.
    """

    def test_the_existing_row_branch_rewrites_every_provenance_column(self):
        import inspect

        from src.workers import dataset_tasks

        src = inspect.getsource(dataset_tasks)
        for field in ("chat_format", "pack_sequences", "padding", "truncation",
                      "add_special_tokens", "text_column"):
            assert f"tokenization_obj.{field} = " in src, (
                f"a retry leaves {field} describing the previous attempt"
            )

    def test_the_retry_endpoint_forwards_the_packing_parameters(self):
        """Rebuilding the row is useless if the parameters never arrive."""
        import inspect

        from src.api.v1.endpoints import task_queue

        src = inspect.getsource(task_queue)
        assert "chat_format=retry_params.get" in src
        assert "pack_sequences=retry_params.get" in src
