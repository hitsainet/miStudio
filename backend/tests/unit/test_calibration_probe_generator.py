"""Probes must be NEUTRAL-topic and falsifiable (IDL-37).

The whole point of the generator: a probe on the circuit's OWN concept has no
falsifiable answer, so the judge could never detect the correctness cliff. These
pin that the generator (a) targets unrelated topics, (b) drops any probe that
drifted onto the concept, and (c) degrades safely to neutral fallbacks rather
than calibrating against an undetectable cliff.
"""

import json

from src.services.circuit_probe_generator import (
    _FALLBACK_PROBES, _looks_on_concept, generate_probes)


class TestNeutrality:
    def test_a_probe_on_the_concept_is_recognised_as_on_concept(self):
        assert _looks_on_concept(
            {"prompt": "Tell me a parody of a news report", "expected": "..."},
            ["parody", "comedians", "humor"]) is True

    def test_a_neutral_probe_is_not_flagged(self):
        assert _looks_on_concept(
            {"prompt": "What is the capital of France?", "expected": "Paris"},
            ["parody", "comedians", "humor"]) is False

    def test_generated_probes_that_drift_onto_the_concept_are_DROPPED(self):
        """The LLM returns a mix; only the neutral ones survive."""
        def fake_llm(prompt, max_tokens):
            return json.dumps([
                {"prompt": "Write a parody about weather", "expected": "n/a"},   # on-concept
                {"prompt": "What is the capital of Japan?", "expected": "Tokyo"},  # neutral
            ])

        probes = generate_probes(["parody", "humor"], llm_call=fake_llm, n=3)
        assert probes == [{"prompt": "What is the capital of Japan?", "expected": "Tokyo"}]

    def test_all_on_concept_falls_back_to_NEUTRAL_probes(self):
        """If every generated probe is on-concept, calibrating against them
        would be blind — fall back to the neutral static set instead."""
        def fake_llm(prompt, max_tokens):
            return json.dumps([
                {"prompt": "Write a comedic parody", "expected": "n/a"},
                {"prompt": "Tell a humor joke about comedians", "expected": "n/a"},
            ])

        probes = generate_probes(["parody", "humor", "comedians"], llm_call=fake_llm, n=3)
        assert probes == _FALLBACK_PROBES[:3]
        # and the fallbacks are themselves neutral to this concept
        for p in probes:
            assert not _looks_on_concept(p, ["parody", "humor", "comedians"])


class TestNeutralityDoesNotOverMatch:
    """Round-1: the old length>=4 prefix match dropped neutral probes containing
    generic label words, and MISSED short concept words. Fixed via a stopword
    set + whole-word match."""

    def test_a_generic_label_word_does_NOT_drop_a_neutral_probe(self):
        # Labels contain the generic word "language"; a neutral geography probe
        # legitimately mentions "language" and must NOT be flagged on-concept.
        assert _looks_on_concept(
            {"prompt": "What language is spoken in Brazil?", "expected": "Portuguese"},
            ["informal language and tone", "references to jokes"]) is False

    def test_generic_labels_do_not_nuke_all_neutral_probes(self):
        import json
        def fake_llm(prompt, max_tokens):
            return json.dumps([
                {"prompt": "What language is spoken in Brazil?", "expected": "Portuguese"},
                {"prompt": "What is the capital of Japan?", "expected": "Tokyo"},
            ])
        probes = generate_probes(
            ["informal language and tone", "general references"],
            llm_call=fake_llm, n=3)
        # Both survive — neither was wrongly flagged — so the LLM output is used,
        # NOT the static fallback.
        assert len(probes) == 2
        assert probes != _FALLBACK_PROBES[:2]

    def test_a_SHORT_concept_word_is_still_detected(self):
        # 'pun' (3 chars) was skipped by the old len>=4 gate → on-concept probes
        # survived → undetectable cliff. Now caught.
        assert _looks_on_concept(
            {"prompt": "Tell me a pun about cats", "expected": "n/a"},
            ["pun", "gag", "wit"]) is True

    def test_a_neutral_probe_with_short_concepts_present_is_kept(self):
        assert _looks_on_concept(
            {"prompt": "What is the capital of Peru?", "expected": "Lima"},
            ["pun", "gag", "wit"]) is False


class TestParseToleratesStrayBrackets:
    def test_footnote_brackets_before_the_array(self):
        raw = 'See [1] for context. Here: [{"prompt":"Q","expected":"A"}]'
        probes = generate_probes(["parody"], llm_call=lambda p, m: raw, n=1)
        assert probes == [{"prompt": "Q", "expected": "A"}]

    def test_trailing_bracket_note_after_the_array(self):
        raw = '[{"prompt":"What is 2+2?","expected":"4"}] and a note [end]'
        probes = generate_probes(["parody"], llm_call=lambda p, m: raw, n=1)
        assert probes == [{"prompt": "What is 2+2?", "expected": "4"}]


class TestRobustness:
    def test_no_llm_uses_neutral_fallback(self):
        probes = generate_probes(["parody"], llm_call=None, n=2)
        assert probes == _FALLBACK_PROBES[:2]

    def test_garbage_llm_output_falls_back(self):
        probes = generate_probes(["parody"], llm_call=lambda p, m: "not json at all", n=3)
        assert probes == _FALLBACK_PROBES[:3]

    def test_fenced_json_is_parsed(self):
        def fake_llm(prompt, max_tokens):
            return ('Here you go:\n```json\n'
                    '[{"prompt":"What is 2+2?","expected":"4"}]\n```')
        probes = generate_probes(["parody"], llm_call=fake_llm, n=1)
        assert probes == [{"prompt": "What is 2+2?", "expected": "4"}]

    def test_every_probe_carries_prompt_and_expected(self):
        """A probe missing its expected answer cannot be judged; it is dropped."""
        def fake_llm(prompt, max_tokens):
            return json.dumps([
                {"prompt": "What is the capital of Peru?", "expected": "Lima"},
                {"prompt": "Missing answer?"},  # no expected → dropped
            ])
        probes = generate_probes(["parody"], llm_call=fake_llm, n=3)
        assert probes == [{"prompt": "What is the capital of Peru?", "expected": "Lima"}]
