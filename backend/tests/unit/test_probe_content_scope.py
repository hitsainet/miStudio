"""Scope `content`: score the messages' own text and nothing the chat template added (2026-10-05).

On Llama-3.1 a single-message row is a ~25-token injected system block, a role header and the text;
`all` and `user` both average over all of it, so for a 20-token joke more than half of what a
`mean` reads is the same boilerplate on every row. These tests use a REAL fast tokenizer and a real
Jinja template that injects a preamble the same way (see `test_probe_monitor_render._tokenizer`).

MUTATION CONTROLS (each run, each red, restored and re-grepped — recorded in the review):
  C1  `scored_mask('content')` ignores the content mask (scores the message span)
  C2  `_content_mask` skips the ids-must-match check
  C3  truncation does not slice the content mask
  C4  `_row_mask` stops routing through `effective_scope` (fallback row raises)
  C5  `render_all` stops counting `content_mask_unavailable`
  C6  `_contract_scope` defaults an unknown scope to 'all' again
  C7  the content search runs over the whole render from the start (matches the preamble)
"""
import pytest

from src.services.probe_monitor_render import (
    _content_mask,
    effective_scope,
    render_all,
    render_messages,
)
from tests.unit.test_probe_monitor_render import _tokenizer

pytest.importorskip("tokenizers")

#: Injects a constant system block before every conversation, as Llama-3.1 does with no system message.
PREAMBLE = (
    "<sys> system a b c d e f </sys> "
    "{% for m in messages %}<turn> {{ m['role'] }} {{ m['content'] }} </turn> {% endfor %}"
)
#: Rewrites the content, so it can no longer be found in the render.
REWRITES_CONTENT = (
    "{% for m in messages %}<turn> {{ m['role'] }} {{ m['content'] | replace('funds', 'safe') }}"
    " </turn> {% endfor %}"
)


def _scored_tokens(tokenizer, rendered, scope):
    mask = rendered.scored_mask(scope)
    return [tokenizer.convert_ids_to_tokens(i) for i, keep in zip(rendered.input_ids, mask) if keep]


class TestContentIsTheMessagesOwnText:
    def test_a_single_turn_scores_only_its_text_not_the_injected_preamble(self):
        tokenizer = _tokenizer(PREAMBLE)
        rendered = render_messages(tokenizer, [{"role": "user", "content": "transfer funds"}])
        assert _scored_tokens(tokenizer, rendered, "content") == ["transfer", "funds"]
        # The point of the scope: `all` and `user` both read the preamble and the header.
        assert "<sys>" in _scored_tokens(tokenizer, rendered, "all")
        assert "<sys>" in _scored_tokens(tokenizer, rendered, "user")

    def test_every_message_contributes_its_text_and_no_headers(self):
        tokenizer = _tokenizer(PREAMBLE)
        rendered = render_messages(tokenizer, [
            {"role": "user", "content": "transfer funds"},
            {"role": "assistant", "content": "no"},
            {"role": "user", "content": "hello world"},
        ])
        assert _scored_tokens(tokenizer, rendered, "content") == ["transfer", "funds", "no", "hello", "world"]

    def test_text_that_also_appears_in_the_preamble_is_found_in_its_own_message(self):
        """'a' is a preamble word here, as 'Today' is in Llama's injected block."""
        tokenizer = _tokenizer(PREAMBLE)
        rendered = render_messages(tokenizer, [{"role": "user", "content": "a"}])
        scored = [i for i, keep in enumerate(rendered.scored_mask("content")) if keep]
        assert len(scored) == 1
        assert tokenizer.convert_ids_to_tokens(rendered.input_ids[scored[0]]) == "a"
        assert scored[0] > rendered.input_ids.index(tokenizer.convert_tokens_to_ids("</sys>"))

    def test_truncation_keeps_the_tail_of_the_content_mask_too(self):
        tokenizer = _tokenizer(PREAMBLE)
        messages = [{"role": "user", "content": "transfer funds"}, {"role": "assistant", "content": "no"}]
        full = render_messages(tokenizer, messages)
        cut = render_messages(tokenizer, messages, max_length=len(full.input_ids) - 3)
        assert cut.truncated
        assert len(cut.content_mask) == len(cut.input_ids)
        assert cut.content_mask == full.content_mask[3:]


class TestItFallsBackToAllAndSaysSo:
    def test_content_the_template_rewrote_has_no_mask_and_is_scored_as_all(self):
        tokenizer = _tokenizer(REWRITES_CONTENT)
        rendered = render_messages(tokenizer, [{"role": "user", "content": "transfer funds"}])
        assert rendered.content_mask is None
        assert effective_scope(rendered, "content") == "all"
        with pytest.raises(ValueError, match="no content mask"):
            rendered.scored_mask("content")

    def test_the_capture_mask_applies_the_same_fallback(self):
        from src.services.probe_monitor_capture import _row_mask

        tokenizer = _tokenizer(REWRITES_CONTENT)
        rendered = render_messages(tokenizer, [{"role": "user", "content": "transfer funds"}])
        assert _row_mask(rendered, "content") == rendered.scored_mask("all")

    def test_the_render_pass_counts_fallback_rows_and_sizes_tokens_by_what_is_scored(self):
        good, bad = _tokenizer(PREAMBLE), _tokenizer(REWRITES_CONTENT)
        rows = [[{"role": "user", "content": "transfer funds"}]]
        _, summary = render_all(good, rows, scope="content")
        assert summary.content_mask_unavailable == 0 and summary.scored_tokens == 2
        results, summary = render_all(bad, rows, scope="content")
        assert summary.content_mask_unavailable == 1
        assert summary.scored_tokens == sum(results[0].scored_mask("all"))
        assert summary.as_dict()["content_mask_unavailable"] == 1

    def test_offsets_that_disagree_with_the_real_encoding_are_not_trusted(self):
        class Disagrees:
            def __call__(self, text, **_):
                return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 1), (1, 2), (2, 3)]}

        assert _content_mask(Disagrees(), "abc", [9, 9, 9], [{"role": "user", "content": "abc"}], [3]) is None

    def test_a_tokenizer_without_offsets_gives_no_mask(self):
        class Slow:
            def __call__(self, text, **kwargs):
                if kwargs.get("return_offsets_mapping"):
                    raise NotImplementedError("slow tokenizers have no offsets")
                return {"input_ids": [1]}

        assert _content_mask(Slow(), "a", [1], [{"role": "user", "content": "a"}], [1]) is None


class TestTheRunAcceptsItAndExportRefusesIt:
    def test_a_run_may_be_scoped_content(self):
        from src.schemas.probe_monitor import ProbeRunConfig

        assert ProbeRunConfig(scope="content").scope == "content"

    def test_export_refuses_content_rather_than_calling_it_all(self):
        from src.services.probe_definition_builder import ProbeExportRefused, _contract_scope

        with pytest.raises(ProbeExportRefused, match="cannot express"):
            _contract_scope("content")

    def test_export_refuses_any_scope_it_does_not_know(self):
        from src.services.probe_definition_builder import ProbeExportRefused, _contract_scope

        with pytest.raises(ProbeExportRefused):
            _contract_scope("a_scope_added_next_year")
        assert _contract_scope("all") == "all" and _contract_scope("user") == "prompt"
