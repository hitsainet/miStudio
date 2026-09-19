"""
Unit tests for EnhancedLabelingService.

Tests the two-pass labeling logic without making real HTTP calls or DB connections.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call

from src.services.enhanced_labeling_service import (
    EnhancedLabelingService,
    EnhancedLabelingError,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def service():
    svc = EnhancedLabelingService(
        endpoint="http://test-llm.local/v1",
        model="test-model",
        workers=4,
    )
    # Mock the OpenAI SDK client so tests don't make real HTTP calls.
    svc._openai = MagicMock()
    # Backwards-compatible alias for the old tests that referenced ._client.post
    svc._client = MagicMock()
    svc._client.post = svc._openai.chat.completions.create
    return svc


def _make_llm_response(content: str, status_code: int = 200):
    """Build a minimal OpenAI-SDK-shaped response mock."""
    resp = MagicMock()
    # OpenAI SDK chat completion shape: resp.choices[0].message.content
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = "stop"
    resp.usage = MagicMock()
    resp.usage.model_dump = MagicMock(return_value={})
    # Legacy httpx-like attributes for any tests that still poke at them
    resp.status_code = status_code
    resp.text = content if status_code >= 400 else ""
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": content}}]
    })
    return resp


def _activation_row(prime: str, prefix: list, suffix: list, act: float = 5.0) -> dict:
    return {
        "prime_token": prime,
        "prefix_tokens": prefix,
        "suffix_tokens": suffix,
        "max_activation": act,
    }


# ── token cleaning ─────────────────────────────────────────────────────────────

class TestCleanToken:
    def test_bpe_space_prefix(self):
        assert EnhancedLabelingService._clean_token("\u0120thirty") == " thirty"

    def test_plain_token_unchanged(self):
        assert EnhancedLabelingService._clean_token("hello") == "hello"

    def test_mojibake_right_quote(self):
        assert EnhancedLabelingService._clean_token("\u00e2\u0080\u0099") == "'"

    def test_empty_string(self):
        assert EnhancedLabelingService._clean_token("") == ""


class TestJoinTokens:
    def test_joins_and_cleans(self):
        result = EnhancedLabelingService._join_tokens(["\u0120hello", "\u0120world"])
        assert result == " hello world"

    def test_empty_list(self):
        assert EnhancedLabelingService._join_tokens([]) == ""


# ── JSON parsing ──────────────────────────────────────────────────────────────

class TestParseJson:
    def test_clean_json(self):
        raw = '{"name": "test_slug", "category": "test_cat", "description": "desc.", "confidence": "high", "reasoning": "reason text"}'
        result = EnhancedLabelingService._parse_json(raw)
        assert result["name"] == "test_slug"
        assert result["confidence"] == "high"

    def test_json_with_code_fence(self):
        raw = '```json\n{"name": "foo", "category": "bar", "description": "d.", "confidence": "medium", "reasoning": "r."}\n```'
        result = EnhancedLabelingService._parse_json(raw)
        assert result["name"] == "foo"

    def test_fallback_regex(self):
        raw = 'Some preamble. "name": "ordinal_numbers", "confidence": "high"'
        result = EnhancedLabelingService._parse_json(raw)
        assert result["name"] == "ordinal_numbers"
        assert result["confidence"] == "high"

    def test_unparseable_raises(self):
        with pytest.raises(EnhancedLabelingError):
            EnhancedLabelingService._parse_json("not json at all, no fields")


# ── notes builder ─────────────────────────────────────────────────────────────

class TestBuildNotes:
    def test_contains_reasoning(self):
        row = {"prime_token": "of", "max_activation": 7.1}
        summaries = [(row, "preposition linking university to location")]
        notes = EnhancedLabelingService._build_notes("The reasoning here.", summaries)
        assert "The reasoning here." in notes

    def test_contains_summary_table(self):
        row = {"prime_token": "of", "max_activation": 7.1}
        summaries = [(row, "preposition linking university to location")]
        notes = EnhancedLabelingService._build_notes("reasoning", summaries)
        assert "preposition linking university to location" in notes
        assert "| Act | Token | Summary |" in notes

    def test_separator_present(self):
        row = {"prime_token": "x", "max_activation": 1.0}
        notes = EnhancedLabelingService._build_notes("r", [(row, "s")])
        assert "---" in notes

    def test_pipe_escaping_in_summary(self):
        row = {"prime_token": "x", "max_activation": 1.0}
        notes = EnhancedLabelingService._build_notes("r", [(row, "a | b")])
        assert "a \\| b" in notes


# ── call_llm retry logic ──────────────────────────────────────────────────────

class TestCallLlm:
    def test_success_on_first_attempt(self, service):
        service._client.post.return_value = _make_llm_response("hello")
        result = service._call_llm("prompt", 80)
        assert result == "hello"
        service._client.post.assert_called_once()

    def test_retries_on_failure_then_succeeds(self, service):
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = Exception("500")
        ok_resp = _make_llm_response("ok")
        service._client.post.side_effect = [Exception("conn error"), ok_resp]
        with patch("time.sleep"):
            result = service._call_llm("prompt", 80, retries=3)
        assert result == "ok"
        assert service._client.post.call_count == 2

    def test_exhausted_retries_raise(self, service):
        service._client.post.side_effect = Exception("always fails")
        with patch("time.sleep"):
            with pytest.raises(EnhancedLabelingError, match="LLM call failed after"):
                service._call_llm("prompt", 80, retries=2)


# ── run() integration (all mocked) ────────────────────────────────────────────

class TestRun:
    def _make_rows(self, n: int = 5) -> list[dict]:
        return [
            _activation_row(
                prime=f"\u0120token{i}",
                prefix=["\u0120the", "\u0120quick"],
                suffix=["\u0120brown", "\u0120fox"],
                act=float(8 - i),
            )
            for i in range(n)
        ]

    def test_run_calls_pass1_and_pass2(self, service):
        rows = self._make_rows(3)
        per_example_resp = _make_llm_response("Token is a number word.")
        synthesis_resp = _make_llm_response(
            '{"name":"number_words","category":"numerics",'
            '"description":"Fires on number tokens.","confidence":"high",'
            '"reasoning":"All examples are numbers."}'
        )
        # First 3 calls are pass-1; last call is synthesis
        service._client.post.side_effect = [
            per_example_resp, per_example_resp, per_example_resp,
            synthesis_resp,
        ]

        result = service.run(activation_rows=rows, max_examples=3)

        assert result["name"] == "number_words"
        assert result["category"] == "numerics"
        assert "Fires on number tokens." in result["description"]
        assert "All examples are numbers." in result["notes"]
        assert len(result["pass1_summaries"]) == 3

    def test_progress_callback_called(self, service):
        rows = self._make_rows(3)
        llm_resp = _make_llm_response("summary")
        synthesis_resp = _make_llm_response(
            '{"name":"n","category":"c","description":"d.","confidence":"high","reasoning":"r."}'
        )
        service._client.post.side_effect = [llm_resp, llm_resp, llm_resp, synthesis_resp]

        calls = []
        service.run(rows, max_examples=3, progress_cb=lambda n, t: calls.append((n, t)))

        assert len(calls) == 3
        assert calls[-1] == (3, 3)

    def test_empty_rows_raises(self, service):
        with pytest.raises(EnhancedLabelingError, match="No activation examples"):
            service.run(activation_rows=[], max_examples=20)

    def test_max_examples_clips_rows(self, service):
        rows = self._make_rows(10)
        llm_resp = _make_llm_response("s")
        synthesis_resp = _make_llm_response(
            '{"name":"n","category":"c","description":"d.","confidence":"high","reasoning":"r."}'
        )
        # Only 3 pass-1 calls + 1 synthesis even though we pass 10 rows
        service._client.post.side_effect = [llm_resp] * 3 + [synthesis_resp]
        result = service.run(rows, max_examples=3)
        assert len(result["pass1_summaries"]) == 3

    def test_failed_pass1_example_still_completes(self, service):
        rows = self._make_rows(2)
        synthesis_resp = _make_llm_response(
            '{"name":"n","category":"c","description":"d.","confidence":"low","reasoning":"r."}'
        )

        # Pass-1 runs examples in parallel (ThreadPoolExecutor), so a positional
        # side_effect list is order-dependent and flaky. Route by prompt content
        # instead: example 'token0' always errors (exhausts its retries → fallback),
        # 'token1' succeeds, and the pass-2 synthesis prompt returns the JSON.
        def _route(*args, **kwargs):
            prompt = kwargs["messages"][0]["content"]
            if "unifying concept" in prompt or "Return ONLY this JSON" in prompt:
                return synthesis_resp
            if "[token0]" in prompt:
                raise Exception("fail")
            return _make_llm_response("ex 1 summary")

        service._client.post.side_effect = _route
        with patch("time.sleep"):
            result = service.run(rows, max_examples=2)
        # Job still completes; failed example gets fallback text
        assert result["name"] == "n"
        failed_summaries = [s for s in result["pass1_summaries"] if "(summarization failed)" in s["summary"]]
        assert len(failed_summaries) == 1
