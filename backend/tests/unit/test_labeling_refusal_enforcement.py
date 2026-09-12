"""The model's reported evidence is binding, not advisory.

Every case here is a real row from the 2026-08-30 gemma-4-12B-it run over
extraction extr_20260828_184419_sae_sae_5bad, where 107 of 390 features got the
single label `proper_noun_entities` and 49 of 120 declared-uninterpretable
features carried a confident name anyway.
"""

import pytest

from src.services.openai_labeling_service import (
    MIN_FIT_RATIO,
    REFUSAL_LABEL,
    _enforce_refusal,
    _fit_ratio,
)


def _label(**kw):
    base = {
        "category": "semantic",
        "specific": "proper_noun_entities",
        "description": "d",
        "fit_count": None,
        "confidence": None,
    }
    base.update(kw)
    return base


class TestFitCountIsBinding:
    def test_a_poor_self_reported_fit_becomes_a_refusal(self):
        """The model cited its own evidence; that evidence overrides its label."""
        out = _enforce_refusal(_label(fit_count="3/10"))
        assert out["category"] == REFUSAL_LABEL
        assert out["specific"] == REFUSAL_LABEL

    def test_a_good_fit_is_left_alone(self):
        out = _enforce_refusal(_label(fit_count="9/10"))
        assert out["category"] == "semantic"
        assert out["specific"] == "proper_noun_entities"

    def test_the_boundary_is_inclusive_of_half(self):
        """5/10 is 'half fit', which the template does not ask to refuse."""
        assert _enforce_refusal(_label(fit_count="5/10"))["specific"] == (
            "proper_noun_entities"
        )
        assert _enforce_refusal(_label(fit_count="4/10"))["specific"] == REFUSAL_LABEL

    def test_the_original_verdict_is_preserved_for_audit(self):
        out = _enforce_refusal(_label(fit_count="2/10", confidence="high"))
        assert out["fit_count"] == "2/10"
        assert out["confidence"] == "high"

    def test_a_denominator_other_than_ten_still_works(self):
        assert _enforce_refusal(_label(fit_count="1/8"))["specific"] == REFUSAL_LABEL
        assert _enforce_refusal(_label(fit_count="7/8"))["specific"] != REFUSAL_LABEL


class TestAbsentEvidenceIsNotARefusal:
    """Templates predating fit_count must keep working.

    Treating "no number" as "bad fit" would turn every label from every legacy
    template into a refusal — a far worse failure than the one being fixed.
    """

    @pytest.mark.parametrize("value", [None, "", "N/10", "unknown", "10", "/"])
    def test_unreadable_fit_count_leaves_the_verdict_standing(self, value):
        out = _enforce_refusal(_label(fit_count=value))
        assert out["specific"] == "proper_noun_entities"

    def test_zero_denominator_does_not_divide(self):
        assert _fit_ratio("0/0") is None
        assert _enforce_refusal(_label(fit_count="0/0"))["specific"] != REFUSAL_LABEL


class TestTheCategoryNameContradiction:
    def test_an_uninterpretable_category_forces_an_uninterpretable_name(self):
        """Downstream reads `specific`, so a confident name there is the claim.

        Real row: category=uninterpretable, name=proper_noun_entities, whose own
        description read "without a unifying theme".
        """
        out = _enforce_refusal(
            _label(category="uninterpretable", specific="proper_noun_entities")
        )
        assert out["specific"] == REFUSAL_LABEL

    @pytest.mark.parametrize(
        "name", ["none", "None", "null", "N/A", "unknown", ""]
    )
    def test_null_words_do_not_become_feature_names(self, name):
        """A real row stored the literal string 'none' as a feature name."""
        out = _enforce_refusal(_label(specific=name))
        assert out["specific"] == REFUSAL_LABEL
        assert out["category"] == REFUSAL_LABEL

    def test_a_normal_label_is_untouched(self):
        out = _enforce_refusal(_label(category="semantic", specific="fda_mentions"))
        assert out["category"] == "semantic"
        assert out["specific"] == "fda_mentions"


class TestAppliedAtTheParser:
    """Both parser return paths must enforce, not just the JSON one."""

    def test_json_path_enforces(self):
        from src.services.openai_labeling_service import OpenAILabelingService
        from unittest.mock import patch

        with patch.object(OpenAILabelingService, "__init__", lambda self: None):
            svc = OpenAILabelingService()
        out = svc._parse_dual_label(
            '{"category":"semantic","specific":"proper_noun_entities",'
            '"description":"d","fit_count":"2/10"}',
            "fallback",
        )
        assert out["specific"] == REFUSAL_LABEL

    def test_plaintext_fallback_path_enforces(self):
        from src.services.openai_labeling_service import OpenAILabelingService
        from unittest.mock import patch

        with patch.object(OpenAILabelingService, "__init__", lambda self: None):
            svc = OpenAILabelingService()
        out = svc._parse_dual_label(
            'category: semantic specific: proper_noun_entities '
            'description: "d" fit_count: "1/10"',
            "fallback",
        )
        assert out["specific"] == REFUSAL_LABEL, (
            "the non-JSON path skipped enforcement; a model that breaks JSON "
            "would bypass the guard entirely"
        )


class TestAgainstTheRealRun:
    """Replay of the distribution that prompted this fix."""

    def test_the_observed_bad_rows_all_become_refusals(self):
        observed = [
            {"category": "uninterpretable", "specific": "noun_phrase_entities"},
            {"category": "uninterpretable", "specific": "transitional_phrases"},
            {"category": "uninterpretable", "specific": "proper_noun_entities"},
            {"category": "uninterpretable", "specific": "proper_name_activation"},
            {"category": "uninterpretable", "specific": "descriptive_adjectives"},
            {"category": "uninterpretable", "specific": "none"},
        ]
        for row in observed:
            out = _enforce_refusal(_label(**row))
            assert out["specific"] == REFUSAL_LABEL, row

    def test_the_crisp_feature_survives(self):
        """idx100 fired on FDA ten times over; it must keep its label."""
        out = _enforce_refusal(
            _label(category="semantic", specific="fda_regulatory_mentions",
                   fit_count="10/10", confidence="high")
        )
        assert out["specific"] == "fda_regulatory_mentions"
