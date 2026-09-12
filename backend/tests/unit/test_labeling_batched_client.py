"""Batched labeling client: demux, capability detection, failure granularity.

Each test here targets a failure that would otherwise be SILENT — a mis-ordered
demux attaches every label to the wrong feature while looking healthy, and an
unsupported server returns one choice for a batch of eight without erroring.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.openai_labeling_service import (
    BatchUnsupportedError,
    OpenAILabelingService,
)


def _choice(index, label):
    return SimpleNamespace(
        index=index,
        message=SimpleNamespace(
            content='{"category": "cat", "specific": "%s", "description": ""}' % label
        ),
    )


def _raw(choices, batch_header):
    headers = {}
    if batch_header is not None:
        headers["X-miLLM-Batch"] = str(batch_header)
    raw = MagicMock()
    raw.headers = headers
    raw.parse.return_value = SimpleNamespace(choices=choices)
    return raw


@pytest.fixture
def service():
    with patch.object(OpenAILabelingService, "__init__", lambda self: None):
        svc = OpenAILabelingService()
    svc.model = "test"
    svc.temperature = 0.0
    svc.max_tokens = 64
    svc.top_p = 1.0
    # __init__ is patched out, so every attribute the call paths read must be
    # set here. Default matches the constructor's: reasoning off for labeling.
    svc.chat_template_kwargs = {"enable_thinking": False}
    svc.client = MagicMock()
    import asyncio
    svc._api_semaphore = asyncio.Semaphore(4)
    svc._resolve_user_prompt = MagicMock(side_effect=lambda **k: f"P:{k['feature_id']}")
    return svc


def _reqs(n):
    return [
        {
            "examples": [{"prime_token": "x"}],
            "template_config": {},
            "user_prompt_template": "{examples_block}",
            "system_message": "sys",
            "feature_id": f"f{i}",
        }
        for i in range(n)
    ]


def _wire(service, raw):
    service.client.chat.completions.with_raw_response.create = AsyncMock(
        return_value=raw
    )


class TestDemux:
    @pytest.mark.asyncio
    async def test_labels_follow_choice_index_not_wire_order(self, service):
        """The OpenAI schema does not promise sorted choices.

        Trusting wire order would attach every label to the wrong feature and
        raise nothing at all.
        """
        shuffled = [_choice(2, "third"), _choice(0, "first"), _choice(1, "second")]
        _wire(service, _raw(shuffled, 3))
        out = await service.generate_labels_from_examples_batched(_reqs(3))
        assert [o["specific"] for o in out] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_one_batched_call_serves_the_whole_chunk(self, service):
        _wire(service, _raw([_choice(i, f"l{i}") for i in range(4)], 4))
        await service.generate_labels_from_examples_batched(_reqs(4))
        assert service.client.chat.completions.with_raw_response.create.await_count == 1

    @pytest.mark.asyncio
    async def test_extra_messages_carries_every_conversation_but_the_first(
        self, service
    ):
        _wire(service, _raw([_choice(i, f"l{i}") for i in range(3)], 3))
        await service.generate_labels_from_examples_batched(_reqs(3))
        kwargs = service.client.chat.completions.with_raw_response.create.await_args.kwargs
        assert len(kwargs["extra_body"]["extra_messages"]) == 2
        assert kwargs["messages"][1]["content"] == "P:f0"

    @pytest.mark.asyncio
    async def test_chunks_beyond_batch_size(self, service):
        service.BATCH_SIZE = 2
        calls = []

        async def _create(**kw):
            n = 1 + len(kw["extra_body"]["extra_messages"])
            calls.append(n)
            return _raw([_choice(i, f"l{len(calls)}_{i}") for i in range(n)], n)

        service.client.chat.completions.with_raw_response.create = AsyncMock(
            side_effect=_create
        )
        service.generate_label_from_examples = AsyncMock(
            return_value={"category": "c", "specific": "solo", "description": ""}
        )
        out = await service.generate_labels_from_examples_batched(_reqs(5))
        assert len(out) == 5
        assert calls == [2, 2]  # the trailing single goes down the serial path


class TestCapabilityDetection:
    @pytest.mark.asyncio
    async def test_missing_header_falls_back_to_serial(self, service):
        """A server predating the extension returns ONE choice, no error.

        Without the header check this silently labels 1 of every 8 features.
        """
        _wire(service, _raw([_choice(0, "only")], None))
        service.generate_label_from_examples = AsyncMock(
            return_value={"category": "c", "specific": "serial", "description": ""}
        )
        out = await service.generate_labels_from_examples_batched(_reqs(4))
        assert len(out) == 4
        assert all(o["specific"] == "serial" for o in out)
        assert service.generate_label_from_examples.await_count == 4

    @pytest.mark.asyncio
    async def test_header_disagreeing_with_batch_size_falls_back(self, service):
        _wire(service, _raw([_choice(0, "only")], 1))
        service.generate_label_from_examples = AsyncMock(
            return_value={"category": "c", "specific": "serial", "description": ""}
        )
        out = await service.generate_labels_from_examples_batched(_reqs(4))
        assert service.generate_label_from_examples.await_count == 4
        assert len(out) == 4

    @pytest.mark.asyncio
    async def test_non_contiguous_indices_are_refused(self, service):
        _wire(service, _raw([_choice(0, "a"), _choice(7, "b")], 2))
        with pytest.raises(BatchUnsupportedError):
            await service._generate_chunk_batched(_reqs(2))


class TestFailureGranularity:
    @pytest.mark.asyncio
    async def test_a_batch_failure_costs_speed_not_coverage(self, service):
        """One batched request is ONE failure domain.

        The serial path loses exactly one feature to a timeout; without this
        fallback a batch would lose all eight.
        """
        service.client.chat.completions.with_raw_response.create = AsyncMock(
            side_effect=TimeoutError("upstream timed out")
        )
        service.generate_label_from_examples = AsyncMock(
            return_value={"category": "c", "specific": "recovered", "description": ""}
        )
        out = await service.generate_labels_from_examples_batched(_reqs(8))
        assert len(out) == 8
        assert all(o["specific"] == "recovered" for o in out), (
            "a batch failure wrote error labels instead of retrying serially"
        )

    @pytest.mark.asyncio
    async def test_result_length_always_matches_input(self, service):
        """Callers zip this against their own feature list."""
        _wire(service, _raw([_choice(i, f"l{i}") for i in range(3)], 3))
        out = await service.generate_labels_from_examples_batched(_reqs(3))
        assert len(out) == 3
        assert all(o is not None for o in out)

    @pytest.mark.asyncio
    async def test_empty_input_is_not_a_request(self, service):
        service.client.chat.completions.with_raw_response.create = AsyncMock()
        assert await service.generate_labels_from_examples_batched([]) == []
        service.client.chat.completions.with_raw_response.create.assert_not_awaited()


class TestGuardsAreIndividuallyLoadBearing:
    """Each guard isolated, because together they mask each other.

    The first version of TestCapabilityDetection tripped the header check, the
    size check and the choice-count check with the same fixture, so removing
    any ONE of them left the suite green. These fixtures are shaped so exactly
    one guard can fire.
    """

    @pytest.mark.asyncio
    async def test_full_choice_count_without_a_header_still_falls_back(
        self, service
    ):
        """N well-formed choices and no header is the DANGEROUS case.

        A server that ignores `extra_messages` but honours some other
        multi-choice mechanism (an `n`-style parameter) returns exactly N
        choices with contiguous indices — every one a completion of the FIRST
        prompt. The choice-count and index guards both pass. The capability
        header is the only thing that can tell this from a real batch.
        """
        _wire(service, _raw([_choice(i, "all-from-prompt-0") for i in range(4)], None))
        service.generate_label_from_examples = AsyncMock(
            return_value={"category": "c", "specific": "serial", "description": ""}
        )
        out = await service.generate_labels_from_examples_batched(_reqs(4))
        assert service.generate_label_from_examples.await_count == 4, (
            "accepted 4 choices from a server that never advertised batching; "
            "every label may belong to the first feature"
        )
        assert all(o["specific"] == "serial" for o in out)

    @pytest.mark.asyncio
    async def test_full_choice_count_with_a_smaller_header_falls_back(
        self, service
    ):
        """Header says 2, four choices arrive: only the size guard can fire."""
        _wire(service, _raw([_choice(i, f"l{i}") for i in range(4)], 2))
        service.generate_label_from_examples = AsyncMock(
            return_value={"category": "c", "specific": "serial", "description": ""}
        )
        out = await service.generate_labels_from_examples_batched(_reqs(4))
        assert service.generate_label_from_examples.await_count == 4, (
            "server served fewer conversations than were sent and the "
            "mismatch was accepted"
        )
        assert len(out) == 4
