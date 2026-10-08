"""Token filters must see byte-level BPE tokens as the words they are.

THE DEFECT. `cleaned = token.replace('▁', '').strip().lower()` stripped the
SentencePiece word-boundary marker and nothing else. Byte-level BPE — GPT-2,
Llama 3, LFM2, and every model trained on this estate — encodes a leading space
as 'Ġ' (U+0120). So the comparison text for 'Ġthe' stayed 'Ġthe', which is in
neither STOP_WORDS nor WORD_FRAGMENTS, and `filter_stop_words=True` filtered
nothing at all on real data.

This was not a corner case. Measured on the 16k L11 extraction, the stored prime
tokens are 'Ġthe' (11,266), 'Ġand' (11,182), 'Ġto' (9,324), 'Ġof' (7,978),
'Ġis' (6,092) — roughly 18% of all stored evidence in the top fifteen alone, and
every one of them invisible to the filter that exists to remove them.

WHY 28 EXISTING TESTS MISSED IT. Every fixture in `test_token_filters.py` used
either a bare word ('the') or the SentencePiece form ('▁the'). Neither is what
the tokenizer in production emits. The fixtures agreed with the defect by
construction — so the suite was green, the feature was documented, and the flag
did nothing.

MUTATION CONTROLS (each applied alone; suite must go red):
  B1  drop `.replace('Ġ', '')` from normalize_token_text
        -> `test_byte_level_stop_words_are_filtered` fails
  B2  drop `.replace('▁', '')`
        -> `test_sentencepiece_form_still_works` fails
  B3  drop `.replace('Ċ', '')`
        -> `test_newline_marker_is_stripped` fails
  B4  inline the old one-marker logic back into analyze_feature_tokens
        -> `test_analyze_counts_byte_level_stop_words` fails
        (NOTE: `filter_token_stats` DELEGATES to is_junk_token and has no copy
         of its own, so a control aimed there is void -- that is how the first
         version of B4 survived)
  B5  inline the old one-marker logic back into clean_token_display
        -> `test_display_strips_the_byte_level_marker` fails
  B6  make normalize_token_text return the token unchanged
        -> most of this file fails, and `test_content_words_survive` still passes
           (which is the point: that test alone cannot detect the defect)
"""

import pytest

from src.utils.token_filters import (
    STOP_WORDS,
    analyze_feature_tokens,
    clean_token_display,
    decode_byte_level_token,
    filter_token_stats,
    is_junk_token,
    normalize_token_text,
)

#: The exact forms stored by the production tokenizer, taken from a real
#: extraction rather than invented — these are the top stop-word prime tokens
#: on `extr_20260919_200304_sae_sae_9a4d`, with their row counts.
REAL_STORED_STOP_TOKENS = ['Ġthe', 'Ġand', 'Ġto', 'Ġof', 'Ġis', 'Ġin', 'Ġa']

#: Content words in the same byte-level form. These must NEVER be filtered —
#: without them a filter that discards everything would pass every other test.
REAL_STORED_CONTENT_TOKENS = ['Ġneural', 'Ġcocoa', 'Ġbanking', 'Ġanswer']


class TestNormalisation:
    def test_strips_byte_level_space_marker(self):
        assert normalize_token_text('Ġthe') == 'the'

    def test_sentencepiece_form_still_works(self):
        """The one convention that DID work must keep working."""
        assert normalize_token_text('▁the') == 'the'

    def test_newline_marker_is_stripped(self):
        assert normalize_token_text('Ċ') == ''

    def test_bare_word_is_unchanged(self):
        assert normalize_token_text('the') == 'the'

    def test_case_is_preserved_for_callers_to_decide(self):
        """Callers lowercase for set comparison; display must not be mangled."""
        assert normalize_token_text('ĠHello') == 'Hello'


class TestStopWordsOnRealTokens:
    @pytest.mark.parametrize('token', REAL_STORED_STOP_TOKENS)
    def test_byte_level_stop_words_are_filtered(self, token):
        assert is_junk_token(token, filter_stop_words=True) is True

    @pytest.mark.parametrize('token', REAL_STORED_STOP_TOKENS)
    def test_and_are_kept_when_the_flag_is_off(self, token):
        """It must be the FLAG doing this, not blanket junk classification."""
        assert is_junk_token(token, filter_stop_words=False) is False

    @pytest.mark.parametrize('token', REAL_STORED_CONTENT_TOKENS)
    def test_content_words_survive(self, token):
        assert is_junk_token(token, filter_stop_words=True) is False

    def test_chat_scaffolding_is_not_a_stop_word(self):
        """Recorded, not fixed: 'assistant' is chat scaffolding, not a stop word.

        It is the second-largest degenerate class after function words and this
        filter does not address it. Pinning the fact so a later change to
        STOP_WORDS is a deliberate decision rather than a surprise.
        """
        assert 'assistant' not in STOP_WORDS
        assert is_junk_token('assistant', filter_stop_words=True) is False


class TestTheOtherCallersAgree:
    def test_analyze_counts_byte_level_stop_words(self):
        """`analyze_feature_tokens` keeps its OWN copy of the normalisation.

        It does not call `is_junk_token` — it re-implements the branch chain and
        tallies why each token was dropped. So it carried the same missing
        marker independently, and its `stop_words` tally read 0 on real data
        however the flag was set.

        This is the control the first version of this file missed: the agreement
        test below exercises `filter_token_stats`, which DELEGATES to
        `is_junk_token` and therefore cannot drift. Mutating the normalisation
        here changes nothing there, so B4 survived. Asserting the tally is what
        makes it bite.
        """
        result = analyze_feature_tokens(
            [['Ġthe', 'Ġand', 'Ġneural']],
            apply_filters=True,
            filter_stop_words=True,
        )

        stats = result['summary']['filter_stats']
        assert stats['stop_words'] == 2, (
            f"expected 'Ġthe' and 'Ġand' counted as stop words, got {stats}"
        )

    def test_analyze_leaves_content_words_alone(self):
        """Negative control for the tally — it must not count everything."""
        result = analyze_feature_tokens(
            [['Ġneural', 'Ġcocoa']],
            apply_filters=True,
            filter_stop_words=True,
        )

        assert result['summary']['filter_stats']['stop_words'] == 0

    def test_stats_agree_with_the_filter(self):
        """`filter_token_stats` must classify exactly as `is_junk_token` does.

        It delegates today, which is why this cannot drift — kept as a guard in
        case someone re-inlines the logic, which is exactly how the three copies
        came about.
        """
        tokens = REAL_STORED_STOP_TOKENS + REAL_STORED_CONTENT_TOKENS
        stats = {t: {'count': 1} for t in tokens}

        kept = filter_token_stats(stats, filter_stop_words=True)

        for token in tokens:
            expected_kept = not is_junk_token(
                token,
                filter_special=True,
                filter_single_char=True,
                filter_punctuation=True,
                filter_numbers=True,
                filter_fragments=True,
                filter_stop_words=True,
            )
            assert (token in kept) is expected_kept, (
                f'{token!r}: stats and is_junk_token disagree'
            )

    def test_display_strips_the_byte_level_marker(self):
        assert clean_token_display('Ġthe') == 'the'

    def test_display_falls_back_to_the_original_when_empty(self):
        """A bare marker has no word behind it; showing nothing would be worse."""
        assert clean_token_display('Ġ') == 'Ġ'
        assert clean_token_display('▁') == '▁'


class TestByteLevelDecoding:
    """Stripping the marker is not decoding.

    `normalize_token_text` gives STOP_WORDS the word it needs and does nothing
    for a token whose bytes are non-ASCII. Those stayed mojibake everywhere a
    token is SHOWN — the feature browser, and the judge's prompt.

    MUTATION CONTROLS:
      D1  make decode_byte_level_token return its input
            -> `test_decodes_the_real_second_most_common_token` fails
      D2  drop the decode call from clean_token_display
            -> `test_browser_shows_real_text` fails
      D3  drop the decode call from LabelingContextFormatter._clean_token
            -> `test_the_judge_reads_real_text` fails
      D4  drop the leading-whitespace lstrip in _clean_token
            -> `test_the_judge_reads_real_text` fails on 'Ġthe'
      D5  make the decoder raise instead of returning the input
            -> `test_non_byte_level_tokens_pass_through` fails
    """

    def test_decodes_the_real_second_most_common_token(self):
        # 3,015 rows on extr_20260920_070000_sae_sae_7c5b, displayed as mojibake.
        assert decode_byte_level_token('âĢĻs') == '’s'
        assert decode_byte_level_token('âĢĻt') == '’t'

    def test_decodes_the_space_marker_to_a_space(self):
        assert decode_byte_level_token('Ġthe') == ' the'

    def test_non_byte_level_tokens_pass_through(self):
        """Fails OPEN: mangling a token would be worse than returning it."""
        assert decode_byte_level_token('▁the') == '▁the'
        assert decode_byte_level_token('<|endoftext|>') == '<|endoftext|>'

    def test_browser_shows_real_text(self):
        assert clean_token_display('âĢĻs') == '’s'
        assert clean_token_display('Ġthe') == 'the'

    def test_browser_keeps_its_existing_behaviour(self):
        assert clean_token_display('▁Hello') == 'Hello'
        assert clean_token_display('Ġ') == 'Ġ'   # empty after cleaning -> original

    def test_the_judge_reads_real_text(self):
        """This text goes INTO THE PROMPT. A mangled token is a label defect."""
        from src.services.labeling_context_formatter import LabelingContextFormatter

        assert LabelingContextFormatter._clean_token('âĢĻs') == '’s'
        assert LabelingContextFormatter._clean_token('Ġthe') == 'the'
        # unchanged behaviour for the conventions it already handled
        assert LabelingContextFormatter._clean_token('▁world') == 'world'
        assert LabelingContextFormatter._clean_token('##ing') == 'ing'
