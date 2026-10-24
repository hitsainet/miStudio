"""A judge reply with no label is a FAILURE, never a name.

LIVE DEFECT, 2026-10-20 (tracker 3.4 part E). A scratch template told the judge
to reply only "OK". The job recorded **20/20 `succeeded`**, 18 of them named
after their own placeholder (`feature_feat_sae_…_000NN`), category
`uncategorized`, empty description. `_parse_dual_label` substituted
`fallback_label` for the missing label and returned it WITHOUT an `error` key,
so `LabelingService.failure_reason` returned None and the writer stored an
adjudicated success: the phantom-complete class the writer's own comments say
was eliminated. It also made the "N failed / why / Try 20 first" states
unreachable live, because nothing else produces a feature-level failure.

These tests go through the real parser, the real `generate_label_from_examples`
(only the HTTP call is stubbed) and the real writer.
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.feature import Feature
from src.models.labeling_job import LabelingJob
from src.services.labeling_judge_health import is_job_level_failure
from src.services.labeling_service import LabelingService
from src.services.openai_labeling_service import (
    UNPARSED_LABEL_REASON,
    OpenAILabelingService,
)

FALLBACK = "feature_feat_sae_x_00007"


def _parser():
    return OpenAILabelingService.__new__(OpenAILabelingService)


def _parse(reply):
    return _parser()._parse_dual_label(reply, FALLBACK)


class TestAReplyWithNoLabel:

    @pytest.mark.parametrize("reply", [
        "OK",                                   # the live reply
        "ok.",
        "",                                     # empty completion
        "I cannot determine a pattern here.",   # prose, no label anywhere
        '{"category": "names"}',                # JSON, label key missing
        '{"category": "names", "specific": ""}',
        "42",                                   # JSON, but not an object
        '["a", "b"]',
    ])
    def test_is_a_failure_with_a_reason(self, reply):
        out = _parse(reply)
        reason = LabelingService.failure_reason(out)
        assert reason is not None, (
            f"{reply!r} parsed as a verdict: {out!r} — the writer would store "
            f"the placeholder as a label"
        )
        assert reason.startswith(UNPARSED_LABEL_REASON)

    def test_the_reason_blames_the_feature_not_the_judge(self):
        """A job-level classification would abort the run and spend nothing —
        and leave the failure states unreachable again."""
        assert not is_job_level_failure(LabelingService.failure_reason(_parse("OK")))

    def test_the_reason_groups_as_one_whatever_the_reply(self):
        """The failure report groups on the text before the first colon
        (`labeling_eligibility._reason_group`). Reply text ahead of it would make
        one group per reply."""
        a = LabelingService.failure_reason(_parse("OK"))
        b = LabelingService.failure_reason(_parse("Sure: here you go"))
        assert a.split(":", 1)[0] == b.split(":", 1)[0] == UNPARSED_LABEL_REASON

    def test_the_reason_says_what_came_back(self):
        assert "'OK'" in LabelingService.failure_reason(_parse("OK"))


class TestARealLabelIsStillAVerdict:
    """The distinction cuts both ways."""

    @pytest.mark.parametrize("reply, name", [
        ('{"category": "semantic", "specific": "court_rulings"}', "court_rulings"),
        ('{"category": "semantic", "label": "court_rulings"}', "court_rulings"),
        ('Here: {"category": "semantic", "specific": "court_rulings"}', "court_rulings"),
        ('category: "code", specific: "python_syntax"', "python_syntax"),
    ])
    def test_parses_without_error(self, reply, name):
        out = _parse(reply)
        assert LabelingService.failure_reason(out) is None, out
        assert out["specific"] == name

    def test_a_declared_refusal_stays_a_verdict(self):
        out = _parse('{"category": "uninterpretable", "specific": "none"}')
        assert LabelingService.failure_reason(out) is None
        assert out["specific"] == "uninterpretable"


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class TestThroughTheRealRequestPath:
    """`generate_label_from_examples` is what bulk labeling and trials call."""

    def _generate(self, reply):
        svc = OpenAILabelingService(api_key="k", model="judge", base_url="http://judge.invalid/v1")
        example = {
            "prefix_tokens": ["the"], "prime_token": "cat", "suffix_tokens": ["sat"],
            "max_activation": 2.0,
        }
        with patch.object(svc, "_call_openai", AsyncMock(return_value=_response(reply))):
            return asyncio.run(svc.generate_label_from_examples(
                examples=[example] * 3,
                template_config={},
                user_prompt_template="{examples_block}",
                system_message="Reply only OK.",
                feature_id="feat_sae_x_00007",
                neuron_index=7,
            ))

    def test_an_ok_reply_comes_back_as_a_failure(self):
        out = self._generate("OK")
        assert LabelingService.failure_reason(out) is not None, out

    def test_a_labelled_reply_comes_back_as_a_verdict(self):
        out = self._generate('{"category": "semantic", "specific": "cats"}')
        assert LabelingService.failure_reason(out) is None, out
        assert out["specific"] == "cats"


class TestTheWriterRecordsItAsFailed:

    def test_an_ok_reply_is_written_as_failed_and_keeps_the_old_name(self):
        writer = LabelingService(MagicMock())
        job = LabelingJob(id="lbl_ok", extraction_job_id="ext_1")
        feature = Feature(
            id="feat_sae_x_00007", name="feature_7", neuron_index=7,
            label_status="pending", label_attempts=0, star_color=None,
        )
        status = writer._persist_label_outcome(
            feature, _parse("OK"), [],
            labeling_job=job, labeled_at=datetime.now(timezone.utc),
            label_source_value="openai", prompt_fingerprint="f" * 64,
            judge_model="judge",
        )
        assert status == LabelingService.LABEL_STATUS_FAILED
        assert feature.label_status == "failed"
        assert feature.label_error.startswith(UNPARSED_LABEL_REASON)
        assert feature.name == "feature_7", "the placeholder was written as a label"
        assert feature.labeled_at is None
        assert feature.label_attempts == 1, "a real per-feature failure spends an attempt"
