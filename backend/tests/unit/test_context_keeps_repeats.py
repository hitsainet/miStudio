"""Repeated occurrences of the prime token must survive into the prompt.

The formatter used to DELETE every context token matching the prime, logging
"🚨 DUPLICATE BUG". The premise — that a prime token should never appear in its
own context window — is false, and the deletion destroyed the evidence that
identifies a feature.

Every case below is a REAL example from
extr_20260828_080834_sae_sae_39cc_002, reproduced from what the old code
actually did to it.
"""

import pytest

from src.services.labeling_context_formatter import LabelingContextFormatter as F


def _ctx(prefix, prime, suffix):
    return F._format_single_example(
        {"prefix_tokens": prefix, "prime_token": prime, "suffix_tokens": suffix},
        marker_left="<<", marker_right=">>",
        include_prefix=True, include_suffix=True, example_idx=1,
    )


class TestRepeatsSurvive:
    def test_python_self_is_not_mangled(self):
        """A Python-idiom feature must not be shown broken Python.

        Old output: `def __init__(, value):<< self>>.value = value`
        """
        out = _ctx(["▁def", "▁__init__", "(", "self", ",", "▁value", ")", ":"],
                   "▁self",
                   [".", "value", "▁=", "▁value", "▁def", "▁__str__", "(", "self", ")"])
        assert out.count("self") >= 3, (
            f"context lost its other `self` occurrences: {out!r}"
        )
        assert "(," not in out, f"deletion left broken syntax: {out!r}"

    def test_a_repeated_name_keeps_its_sentence_subject(self):
        """Old output dropped the subject: `us. was one of GOD's chosen`"""
        out = _ctx(["▁remind", "▁us", ".", "David", "▁was", "▁one", "▁of"],
                   "▁David", ["▁had", "▁to", "▁hide"])
        assert out.count("David") >= 2, f"lost a David: {out!r}"

    def test_a_compound_word_is_not_broken(self):
        """`Reverse-Flash` is a DIFFERENT entity that merely shares a substring.

        Old output: `the Reverse- is faster`
        """
        out = _ctx(["▁the", "▁Reverse", "-", "Flash", "▁is", "▁faster"],
                   "▁Flash", ["▁does", "▁not"])
        assert "Reverse" in out
        assert out.count("Flash") >= 2, (
            f"deleted the Flash inside Reverse-Flash: {out!r}"
        )

    def test_an_adjacent_phrase_survives(self):
        """Old output: `Job Description` -> `Job`"""
        out = _ctx(["▁engineer"], "Description",
                   [":", "▁Job", "▁Description", "▁Join", "▁Hired"])
        assert out.count("Description") >= 2, f"lost the phrase: {out!r}"


class TestNormalFormattingIsUnchanged:
    def test_the_prime_is_still_marked_exactly_once(self):
        out = _ctx(["▁the"], "▁cat", ["▁sat"])
        assert out.count("<<") == 1 and out.count(">>") == 1

    def test_a_non_repeating_context_is_unaffected(self):
        out = _ctx(["▁the", "▁quick"], "▁brown", ["▁fox"])
        for w in ("quick", "brown", "fox"):
            assert w in out


class TestTheAlarmIsGone:
    def test_the_formatter_no_longer_deletes_matching_tokens(self):
        """Source guard: the filter comprehensions must not come back.

        Narrow by design — the behavioural tests above are the real cover. This
        catches a revert that reintroduces the deletion under a new name.
        """
        src = open("src/services/labeling_context_formatter.py").read()
        assert "DUPLICATE BUG" not in src, (
            "the duplicate-removal alarm is back; repetition is signal"
        )
        assert "if t != prime_token and" not in src, (
            "the deleting filter comprehension has been reintroduced"
        )
