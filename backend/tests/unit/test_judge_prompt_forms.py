"""Answer forms, templates and the content hash of library judge prompts (PADR IDL-57, D2).

Every parser is held to the stances the single pinned prompt was held to: an unreadable reply is
None and is COUNTED, never a default; out of range is refused, never clamped; prose containing an
answer is not an answer. Controls are recorded in 0xcc/reviews/judge_prompt_library_2026-10-21.md.
"""
import hashlib
import json

import pytest
from pydantic import ValidationError

from src.services import judge_prompt_forms as forms
from src.services.judge_prompt_forms import (
    LEGACY_LAYOUT,
    TemplateInvalid,
    content_hash,
    describe_answer_form,
    load_answer_form,
    parse_answer,
    render_user_message,
    reply_shape_warnings,
    validate_user_template,
)

RATING = {"kind": "rating", "min": 1, "max": 10, "positive_from": 6}
BINARY = {"kind": "binary"}
LABELS = {"kind": "labels", "labels": ["none", "mild", "severe"], "positive_labels": ["mild", "severe"]}


class TestTheRatingForm:
    @pytest.mark.parametrize(
        "reply,value,positive",
        [
            ('{"rating": 7, "reason": "money"}', 7, True),
            ('```json\n{"rating": 2}\n```', 2, False),
            ('Sure! {"rating": 6, "reason": "x"}', 6, True),
            ("5", 5, False),
            ("  10  ", 10, True),
            ('{"rating": "8"}', 8, True),
        ],
    )
    def test_it_reads_real_reply_shapes(self, reply, value, positive):
        answer = parse_answer(RATING, reply)
        assert answer is not None
        assert (answer.value, answer.score, answer.positive) == (value, float(value), positive)

    @pytest.mark.parametrize(
        "reply",
        [None, "", "  ", '{"score": 5}', "{}", "there are 3 risks here", '{"rating": true}',
         '{"rating": 7.5}', '"7"', "[7]", "7 out of 10"],
    )
    def test_unreadable_is_None_never_a_midpoint(self, reply):
        assert parse_answer(RATING, reply) is None

    @pytest.mark.parametrize("value", [0, 11, 99])
    def test_out_of_range_is_REFUSED_not_clamped(self, value):
        assert parse_answer(RATING, json.dumps({"rating": value})) is None

    def test_the_boundary_is_positive_from_inclusive(self):
        assert parse_answer(RATING, "5").positive is False
        assert parse_answer(RATING, "6").positive is True

    def test_a_custom_key_and_range(self):
        form = {"kind": "rating", "min": 0, "max": 4, "positive_from": 3, "json_key": "severity"}
        assert parse_answer(form, '{"severity": 0}').value == 0
        assert parse_answer(form, '{"rating": 3}') is None
        assert parse_answer(form, '{"severity": 5}') is None

    def test_verdict_line(self):
        form = {**RATING, "reply": "verdict_line"}
        assert parse_answer(form, "thinking…\nVERDICT: 9").value == 9
        assert parse_answer(form, "VERDICT: 9\nmore text") is None, "the LAST line is the verdict"
        assert parse_answer(form, '{"rating": 9}') is None

    @pytest.mark.parametrize(
        "bad",
        [
            {"kind": "rating", "min": 5, "max": 5, "positive_from": 5},
            {"kind": "rating", "min": 1, "max": 10, "positive_from": 1},
            {"kind": "rating", "min": 1, "max": 10, "positive_from": 11},
            {"kind": "rating", "min": 1, "max": 10, "positive_from": 6, "extra": 1},
        ],
    )
    def test_impossible_forms_are_refused(self, bad):
        with pytest.raises(ValidationError):
            load_answer_form(bad)


class TestTheBinaryForm:
    @pytest.mark.parametrize(
        "reply,positive",
        [
            ('{"answer": "yes", "reason": "an email"}', True),
            ('{"answer": "No"}', False),
            ("yes", True),
            ("Yes.", True),
            ('"no"', False),
            ('{"answer": true}', True),
            ('{"answer": false}', False),
            ('Here you go: {"answer": "yes"}', True),
        ],
    )
    def test_it_reads_real_reply_shapes(self, reply, positive):
        answer = parse_answer(BINARY, reply)
        assert answer is not None and answer.positive is positive
        assert answer.score == (1.0 if positive else 0.0)

    @pytest.mark.parametrize(
        "reply",
        ["Yes, it contains an email address", "maybe", '{"answer": "unsure"}', "", None,
         '{"verdict": "yes"}', "1"],
    )
    def test_prose_and_other_words_are_not_answers(self, reply):
        assert parse_answer(BINARY, reply) is None

    def test_positive_can_be_NO(self):
        """"Is this free of personal data?" — then 'no' is the behaviour."""
        form = {"kind": "binary", "positive_answer": "no"}
        assert parse_answer(form, "no").positive is True
        assert parse_answer(form, "yes").positive is False

    def test_custom_tokens_and_case_sensitivity(self):
        form = {"kind": "binary", "yes_token": "PII", "no_token": "CLEAN", "case_sensitive": True}
        assert parse_answer(form, "PII").positive is True
        assert parse_answer(form, "pii") is None
        assert parse_answer({**form, "case_sensitive": False}, "pii").positive is True

    def test_the_tokens_must_differ(self):
        with pytest.raises(ValidationError):
            load_answer_form({"kind": "binary", "yes_token": "Yes", "no_token": "yes"})


class TestTheLabelForm:
    def test_each_label_maps_to_its_side(self):
        assert parse_answer(LABELS, '{"label": "none"}').positive is False
        assert parse_answer(LABELS, '{"label": "Severe"}').positive is True
        assert parse_answer(LABELS, "mild").value == "mild"

    def test_an_unknown_label_is_None(self):
        assert parse_answer(LABELS, '{"label": "moderate"}') is None
        assert parse_answer(LABELS, "it is mild") is None

    def test_numeric_labels(self):
        form = {"kind": "labels", "labels": ["1", "2", "3"], "positive_labels": ["3"]}
        assert parse_answer(form, '{"label": 3}').positive is True
        assert parse_answer(form, "2").positive is False

    @pytest.mark.parametrize(
        "bad",
        [
            {"kind": "labels", "labels": ["a"], "positive_labels": ["a"]},
            {"kind": "labels", "labels": ["a", "A"], "positive_labels": ["a"]},
            {"kind": "labels", "labels": ["a", "b"], "positive_labels": ["c"]},
            {"kind": "labels", "labels": ["a", "b"], "positive_labels": ["a", "b"]},
        ],
    )
    def test_impossible_label_sets_are_refused(self, bad):
        with pytest.raises(ValidationError):
            load_answer_form(bad)


class TestTheTemplate:
    MESSAGES = [
        {"role": "user", "content": "my email is a@b.c"},
        {"role": "assistant", "content": "noted"},
    ]

    def test_each_placeholder_is_filled(self):
        text = render_user_message(
            "{interaction}|{text}|{last_user_message}|{last_assistant_message}",
            self.MESSAGES,
            max_chars=8000,
        )
        assert text == (
            "user: my email is a@b.c\nassistant: noted|my email is a@b.c\n\nnoted|"
            "my email is a@b.c|noted"
        )

    def test_a_rows_own_braces_are_never_expanded(self):
        """One pass: a row containing "{interaction}" is text, not a second substitution."""
        text = render_user_message("{text}", [{"role": "user", "content": "{interaction}"}], max_chars=8000)
        assert text == "{interaction}"

    def test_truncation_keeps_the_END(self):
        text = render_user_message(
            "{text}", [{"role": "user", "content": "x" * 500 + "THE END"}], max_chars=200
        )
        assert text.startswith("…") and text.endswith("THE END")

    @pytest.mark.parametrize("bad", ["", "no placeholder", "{interaction} {conversation}", "{Text}"])
    def test_unknown_or_missing_placeholders_are_refused(self, bad):
        with pytest.raises(TemplateInvalid):
            validate_user_template(bad)

    def test_json_in_a_template_is_not_a_placeholder(self):
        assert validate_user_template('Reply {"answer": "yes"}.\n{text}') == ["text"]


class TestTheHash:
    def test_the_LEGACY_layout_hashes_exactly_as_prompt_hash_always_did(self):
        """The stakes seed's hash must equal every pre-library run's recorded hash."""
        from src.services.probe_monitor_judge import JUDGE_SYSTEM_PROMPT, prompt_hash

        expected = hashlib.sha256(f"stakes-rating/v1\n{JUDGE_SYSTEM_PROMPT}".encode()).hexdigest()
        got = content_hash(
            name="stakes-rating",
            version=1,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_template=LEGACY_LAYOUT["user_template"],
            answer_form=LEGACY_LAYOUT["answer_form"],
            max_interaction_chars=LEGACY_LAYOUT["max_interaction_chars"],
        )
        assert got == expected == prompt_hash()
        # The value recorded by the one production judge run (pmj_da0220514811, read 2026-10-21).
        assert got.startswith("31dde28dcd8b")

    @pytest.mark.parametrize(
        "change",
        [
            {"user_template": "Conversation:\n{interaction}"},
            {"max_interaction_chars": 4000},
            {"answer_form": {**LEGACY_LAYOUT["answer_form"], "positive_from": 7}},
            {"answer_form": {**LEGACY_LAYOUT["answer_form"], "reply": "verdict_line"}},
            {"version": 2},
            {"system_prompt": "other"},
        ],
    )
    def test_anything_that_changes_what_is_sent_or_read_moves_it(self, change):
        base = dict(
            name="stakes-rating", version=1, system_prompt="s",
            user_template=LEGACY_LAYOUT["user_template"],
            answer_form=LEGACY_LAYOUT["answer_form"],
            max_interaction_chars=LEGACY_LAYOUT["max_interaction_chars"],
        )
        assert content_hash(**{**base, **change}) != content_hash(**base)

    def test_defaults_written_or_omitted_hash_the_same(self):
        short = {"kind": "binary"}
        full = forms.normalise_answer_form(short)
        args = dict(name="p", version=1, system_prompt="s", user_template="{text}",
                    max_interaction_chars=8000)
        assert content_hash(answer_form=short, **args) == content_hash(answer_form=full, **args)


def test_the_forms_describe_themselves_in_plain_words():
    assert "6 or higher counts as positive" in describe_answer_form(RATING)
    assert '"yes" counts as positive' in describe_answer_form(BINARY)
    assert "positive: 'mild', 'severe'" in describe_answer_form(LABELS)


def test_a_prompt_that_never_asks_for_the_key_is_warned():
    assert reply_shape_warnings("Say yes or no.", "{text}", BINARY)
    assert not reply_shape_warnings('Reply {"answer": "yes" or "no"}.', "{text}", BINARY)
