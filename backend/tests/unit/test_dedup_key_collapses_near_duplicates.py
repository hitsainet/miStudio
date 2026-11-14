"""OSD-13 — the same passage twice is one piece of evidence, not two.

`_content_key` hashed exact token tuples, so a passage repeated with different
indentation or a `#` comment leader counted as distinct and could fill most of a
feature's top-k with one text. Seen by hand on L11: a feature whose six strongest
examples were all the same " publication is switched on" passage.

MEASURED on 600 random L11 features (2026-09-25) before changing anything:
normalising collapses examples for 13 features (2.2%), reclaiming 82 of 15,000
slots — 0.55% overall, but a median of 6 and a maximum of 15 slots on the
features it touches. Low total yield, high local damage, and it lands on exactly
the features whose labels read as confident nonsense.
"""
from src.services.extraction_vectorized import _content_key, _key_form


def _example(prefix, prime, suffix):
    return {"prefix_tokens": list(prefix), "prime_token": prime,
            "suffix_tokens": list(suffix)}


class TestNearDuplicatesCollapse:

    def test_indentation_alone_is_not_new_evidence(self):
        a = _example(["Ġif", "Ġautomatic"], "Ġpublication", ["Ġis", "Ġon"])
        b = _example(["Ġif", "ĠĠĠĠ", "Ġautomatic"], "Ġpublication", ["Ġis", "Ġon"])
        assert _content_key(a) == _content_key(b)

    def test_a_comment_leader_alone_is_not_new_evidence(self):
        a = _example(["Ġif", "Ġautomatic"], "Ġpublication", ["Ġis", "Ġon"])
        b = _example(["#", "Ġif", "Ġautomatic"], "Ġpublication", ["Ġis", "Ġon"])
        assert _content_key(a) == _content_key(b)

    def test_the_byte_level_marker_is_not_part_of_the_identity(self):
        assert _key_form(["Ġfoo"]) == _key_form(["foo"])


class TestDistinctEvidenceSurvives:
    """BR-006: passages differing only by a name are separate evidence."""

    def test_a_byline_differing_by_name_stays_distinct(self):
        a = _example(["Ġstory", ":"], "ĠSmith", ["Ġat", "Ġbloomberg"])
        b = _example(["Ġstory", ":"], "ĠJones", ["Ġat", "Ġbloomberg"])
        assert _content_key(a) != _content_key(b)

    def test_a_different_prime_on_the_same_window_stays_distinct(self):
        a = _example(["Ġthe", "Ġcat"], "Ġsat", ["Ġdown"])
        b = _example(["Ġthe", "Ġcat"], "Ġdown", ["Ġsat"])
        assert _content_key(a) != _content_key(b)

    def test_case_is_still_evidence(self):
        """Deliberately NOT folded: `Once` and `once` can be different features."""
        assert _key_form(["ĠOnce"]) != _key_form(["Ġonce"])

    def test_punctuation_is_still_evidence(self):
        assert _key_form(["Ġend", "."]) != _key_form(["Ġend"])

    def test_the_whole_token_list_branch_normalises_too(self):
        a = {"tokens": ["Ġa", "ĠĠ", "Ġb"], "prime_activation_index": 2}
        b = {"tokens": ["Ġa", "Ġb"], "prime_activation_index": 2}
        assert _content_key(a) == _content_key(b)

    def test_the_prime_position_still_separates_them(self):
        a = {"tokens": ["Ġa", "Ġb"], "prime_activation_index": 0}
        b = {"tokens": ["Ġa", "Ġb"], "prime_activation_index": 1}
        assert _content_key(a) != _content_key(b)


class TestItStillFailsOpen:
    """A quality filter must never discard evidence when it cannot decide."""

    def test_an_example_with_nothing_to_key_on_returns_none(self):
        assert _content_key({}) is None

    def test_a_window_of_only_whitespace_does_not_collapse_everything(self):
        """Two examples that normalise to empty windows but differ in prime."""
        a = _example(["ĠĠ"], "Ġalpha", ["ĠĠ"])
        b = _example(["ĠĠ"], "Ġbeta", ["ĠĠ"])
        assert _content_key(a) != _content_key(b)
