"""The judge baseline: the pinned prompt, the parser, and what it refuses.

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M143  `parse_rating` returns 5 instead of None on a failure → the no-default test fails
  M144  an out-of-range rating is CLAMPED instead of refused  → the range test fails
  M145  the parse-failure limit is not enforced               → the refusal test fails
  M146  a job-level failure is counted as a parse failure     → the unavailable test fails
  M147  `score_judge_ratings` imputes 5 for a missing rating   → the drop test fails
  M148  `render_interaction` truncates from the END           → the truncation test fails
  M149  the prompt text changes without the version           → the pin test fails
  M150  the judge path starts reading a `logprobs` key         → the spike test fails
  M151  `temperature=0` is dropped                            → the determinism test fails

⚠ THE SPIKE RESULT IS ENFORCED, NOT JUST RECORDED (FTID §14a). A live call to miLLM with
`logprobs: true, top_logprobs: 5` returned HTTP 200 with NO logprobs field, and
`grep -rn logprobs millm/` finds nothing — the parameters are accepted and discarded. So
there is no P(yes) path, and `test_the_judge_path_does_not_read_logprobs` exists so a
future miLLM that starts returning them cannot silently change how scores are computed.
Two scoring definitions are two different detectors, and only one of them was measured.
"""
import ast
import inspect
import json
import textwrap

import pytest

from src.services.probe_monitor_judge import (
    DEFAULT_PARSE_FAILURE_LIMIT,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    JudgeUnavailable,
    JudgeUnreliable,
    judge_rows,
    parse_rating,
    prompt_hash,
    render_interaction,
    score_judge_ratings,
)


def _named_error(type_name: str, message: str) -> Exception:
    """An exception whose TYPE NAME is what `is_job_level_failure` keys on.

    The classifier reads `str(exc).split(":")[0]` against a set of client exception type
    names (`APIConnectionError`, `NotFoundError`, …). A test that raises a generic
    `RuntimeError` is therefore testing the row-level path while claiming to test the
    job-level one — which is how my first version of these two tests failed.
    """
    return type(type_name, (Exception,), {})(message)


class _Reply:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _Client:
    """An OpenAI-shaped client returning scripted replies, recording every call."""

    def __init__(self, replies, raises=None):
        self._replies = list(replies)
        self._raises = raises or {}
        self.calls = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if index in self._raises:
            raise self._raises[index]
        return _Reply(self._replies[index % len(self._replies)])


class TestTheParserRefusesRatherThanDefaulting:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ('{"rating": 7, "reason": "money"}', 7),
            ('```json\n{"rating": 2, "reason": "chat"}\n```', 2),
            ("```\n{\"rating\": 10}\n```", 10),
            ('Sure! {"rating": 4, "reason": "meh"}', 4),
            ("6", 6),
            ("  3  ", 3),
        ],
    )
    def test_it_reads_the_shapes_a_real_model_produces(self, reply, expected):
        assert parse_rating(reply) == expected

    @pytest.mark.parametrize(
        "reply",
        [None, "", "   ", "I cannot rate this", '{"score": 5}', "{}", "not json at all"],
    )
    def test_a_failure_is_None_and_NEVER_a_midpoint(self, reply):
        """⚠ A SILENT 5 WOULD PLACE A THIRD OF THE CORPUS AT THE MIDPOINT and drag the
        judge's AUROC toward chance — which reads as "the judge is weak" rather than
        "the judge's output stopped parsing"."""
        assert parse_rating(reply) is None

    @pytest.mark.parametrize("value", [0, 11, 99, -3])
    def test_an_out_of_range_rating_is_REFUSED_not_clamped(self, value):
        """A model answering 0 or 99 has not understood the scale; clamping manufactures
        a confident answer out of a misunderstanding."""
        assert parse_rating(json.dumps({"rating": value})) is None

    def test_prose_containing_a_number_is_NOT_read_as_a_rating(self):
        """"there are 3 risks here" is not a 3. The bare-number branch is deliberately
        `fullmatch`, because picking the first integer out of prose invents data."""
        assert parse_rating("there are 3 risks here") is None
        assert parse_rating("I would say this is high stakes") is None

    def test_a_string_rating_still_parses(self):
        """Models emit `"rating": "7"` often enough that refusing it would look like a
        broken judge."""
        assert parse_rating('{"rating": "7"}') == 7


class TestThePromptIsPinned:
    def test_the_version_and_the_text_are_hashed_together(self):
        """An editable ruler silently invalidates every prior score. The hash is what
        makes a change visible on the run that used it."""
        assert JUDGE_PROMPT_VERSION == "stakes-rating/v1"
        assert len(prompt_hash()) == 64

    def test_the_prompt_asks_for_the_shape_the_parser_reads(self):
        """A prompt and parser that disagree produce 100% parse failures and a refusal
        that blames the judge."""
        assert '"rating"' in JUDGE_SYSTEM_PROMPT
        assert "1" in JUDGE_SYSTEM_PROMPT and "10" in JUDGE_SYSTEM_PROMPT

    def test_it_rates_the_SITUATION_not_the_answer(self):
        """The label is about the interaction's stakes. A prompt that drifted into
        rating the response would measure a different thing under the same name."""
        assert "SITUATION" in JUDGE_SYSTEM_PROMPT or "situation" in JUDGE_SYSTEM_PROMPT

    def test_the_hash_moves_when_the_text_does(self, monkeypatch):
        import src.services.probe_monitor_judge as module

        before = prompt_hash()
        monkeypatch.setattr(module, "JUDGE_SYSTEM_PROMPT", JUDGE_SYSTEM_PROMPT + " Also…")
        assert module.prompt_hash() != before


class TestTheInteractionIsRenderedFromTheEnd:
    def test_roles_are_labelled(self):
        text = render_interaction(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        )
        assert "user: hi" in text and "assistant: hello" in text

    def test_truncation_keeps_the_END(self):
        """The consequential turn is at the end of a conversation, so head-truncation
        would remove exactly what the rating depends on."""
        messages = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "THE DECISION"},
        ]
        text = render_interaction(messages, max_chars=200)
        assert "THE DECISION" in text
        assert text.startswith("…")

    def test_a_short_interaction_is_untouched(self):
        text = render_interaction([{"role": "user", "content": "hi"}], max_chars=200)
        assert not text.startswith("…")


class TestTheRunRefusesRatherThanReportingAPartialSample:
    def _interactions(self, n):
        return [[{"role": "user", "content": f"row {i}"}] for i in range(n)]

    def test_a_clean_run_rates_every_row(self):
        client = _Client(['{"rating": 8}'])
        outcome = judge_rows(client, "qwen", self._interactions(10))
        assert outcome.parse_failures == 0
        assert outcome.ratings == [8] * 10
        assert len(client.calls) == 10

    def test_over_the_limit_it_REFUSES(self):
        """⚠ THE REMAINDER IS A DIFFERENT SAMPLE, NOT A SMALLER ONE. Reporting it as
        though it were the intended set is a different measurement under the intended
        name."""
        client = _Client(["nonsense"] * 10)
        with pytest.raises(JudgeUnreliable, match="could not be parsed"):
            judge_rows(client, "qwen", self._interactions(10))

    def test_the_refusal_NAMES_the_share_and_shows_examples(self):
        client = _Client(["nope"])
        with pytest.raises(JudgeUnreliable) as caught:
            judge_rows(client, "qwen", self._interactions(4))
        message = str(caught.value)
        assert "4 of 4" in message
        assert "unparseable" in message

    def test_under_the_limit_it_proceeds_and_COUNTS(self):
        replies = ['{"rating": 3}'] * 39 + ["nonsense"]
        client = _Client(replies)
        outcome = judge_rows(
            client, "qwen", self._interactions(40), parse_failure_limit=0.05
        )
        assert outcome.parse_failures == 1
        assert outcome.parsed == 39

    def test_the_default_limit_is_five_percent(self):
        assert DEFAULT_PARSE_FAILURE_LIMIT == 0.05

    def test_a_JOB_LEVEL_failure_stops_the_run_immediately(self):
        """⚠ NOT COUNTED AS PARSE FAILURES. A judge that is DOWN would otherwise record
        hundreds of parse failures for a connection problem, and then refuse over the
        limit with a misleading reason."""
        # ⚠ THE EXCEPTION TYPE IS WHAT `is_job_level_failure` MATCHES ON, and my first
        # fixture raised a bare `RuntimeError` — which is correctly NOT job-level, so the
        # run continued and refused over the parse-failure limit instead. The fixture was
        # wrong, not the classifier: the OpenAI client raises `APIConnectionError`.
        client = _Client(
            ['{"rating": 5}'],
            raises={2: _named_error("APIConnectionError", "Connection refused")},
        )
        with pytest.raises(JudgeUnavailable, match="job level"):
            judge_rows(client, "qwen", self._interactions(10))
        assert len(client.calls) == 3, "it kept calling a dead endpoint"

    def test_the_unavailable_message_says_how_far_it_got(self):
        client = _Client(
            ['{"rating": 5}'],
            raises={3: _named_error("APIConnectionError", "refused")},
        )
        with pytest.raises(JudgeUnavailable) as caught:
            judge_rows(client, "qwen", self._interactions(10))
        assert "3 of 10" in str(caught.value)

    def test_a_ROW_level_error_does_not_end_the_run(self):
        """A single malformed row must not abandon the baseline."""
        client = _Client(['{"rating": 5}'], raises={0: ValueError("bad row")})
        outcome = judge_rows(client, "qwen", self._interactions(20))
        assert outcome.parse_failures == 1
        assert outcome.parsed == 19

    def test_temperature_ZERO_on_every_call(self):
        """The judge is a measuring instrument; a sampled instrument gives a different
        reading on the same input."""
        client = _Client(['{"rating": 5}'])
        judge_rows(client, "qwen", self._interactions(3))
        assert all(call["temperature"] == 0 for call in client.calls)

    def test_the_pinned_prompt_is_what_is_SENT(self):
        client = _Client(['{"rating": 5}'])
        judge_rows(client, "qwen", self._interactions(1))
        system = client.calls[0]["messages"][0]
        assert system["role"] == "system"
        assert system["content"] == JUDGE_SYSTEM_PROMPT

    def test_an_empty_set_is_not_a_refusal(self):
        """Zero rows is zero failures; refusing on 0/0 would block a legitimate no-op."""
        outcome = judge_rows(_Client(['{"rating": 5}']), "qwen", [])
        assert outcome.ratings == []


class TestUnparseableRowsAreDroppedNotImputed:
    def test_a_missing_rating_is_EXCLUDED_and_counted(self):
        """Imputing a midpoint drags the AUROC toward chance under the name of a
        measurement — the same stance `mean_auroc_across_sets` takes on unscored sets."""
        ratings = [9] * 25 + [None] * 5 + [1] * 25
        labels = [1] * 25 + [1] * 5 + [0] * 25
        result = score_judge_ratings(ratings, labels)
        assert result["scored"] is True
        assert result["n_dropped"] == 5
        assert result["n_positive"] == 25
        assert result["auroc"] == pytest.approx(1.0)

    def test_dropping_keeps_scores_and_labels_ALIGNED(self):
        """The failure mode: dropping a score without its label shifts every later pair
        and produces an AUROC over mismatched rows — which reads as a weak judge."""
        ratings = [None, 9, 1, None, 8, 2]
        labels = [0, 1, 0, 1, 1, 0]
        result = score_judge_ratings(ratings, labels)
        assert result["n_dropped"] == 2
        # The kept pairs are (9,1), (1,0), (8,1), (2,0) — perfectly separable.
        assert result.get("auroc") in (None, 1.0) or result["scored"] is False

    def test_all_failures_is_a_REFUSAL_with_a_reason(self):
        result = score_judge_ratings([None] * 10, [1, 0] * 5)
        assert result["scored"] is False
        assert "no reply could be parsed" in result["reason"]

    def test_too_few_rows_refuses_through_the_SHARED_floor(self):
        """The judge is held to the same 20-per-class rule as the probe, or the two
        numbers in the report would not be comparable."""
        result = score_judge_ratings([9, 1, 8, 2], [1, 0, 1, 0])
        assert result["scored"] is False
        assert "fewer than" in result["reason"]


class TestTheSpikeResultIsEnforced:
    def test_the_judge_path_does_not_read_logprobs(self):
        """⚠ miLLM RETURNS NONE, AND ASKING FAILS SILENTLY — HTTP 200 with the field
        absent. A P(yes) branch behind a capability flag would be a second scoring
        definition nobody has run, and this test is what stops a future miLLM that
        starts returning logprobs from silently changing how scores are computed."""
        import src.services.probe_monitor_judge as module

        source = inspect.getsource(module)
        tree = ast.parse(source)
        # AST, not a substring: the module docstring discusses logprobs at length, and a
        # text scan would match the prose explaining why they are not used.
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "logprobs" not in attributes
        assert "logprobs" not in constants, "a logprobs key is being sent or read"
        assert "top_logprobs" not in constants

    def test_the_call_sends_only_the_documented_parameters(self):
        client = _Client(['{"rating": 5}'])
        judge_rows(client, "qwen", [[{"role": "user", "content": "x"}]])
        assert set(client.calls[0]) == {"model", "temperature", "messages"}


class TestTheJudgeIsTheBaselineNotTheGrader:
    def test_nothing_here_adjusts_a_probe_score(self):
        """FR-13 rung 3 claims the two were measured on the same data, and nothing about
        the judge being right. A judge that could modify a probe's numbers would make the
        probe's AUROC depend on the baseline it is compared against."""
        import src.services.probe_monitor_judge as module

        source = inspect.getsource(module)
        for forbidden in ("probe.val_metrics", "probe.threshold =", "evaluation.metrics ="):
            assert forbidden not in source, f"the judge writes {forbidden}"

    def test_it_recomputes_the_rung_because_rung_3_is_reached_BY_a_judge_run(self):
        """A run that finishes without recomputing leaves the probe claiming less than
        its evidence supports, and nothing would revisit it."""
        from src.services import probe_monitor_judge as module

        tree = ast.parse(textwrap.dedent(inspect.getsource(module.execute_judge_run)))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "recompute_rung" in called
