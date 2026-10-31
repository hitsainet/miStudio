"""Labeling must be able to switch a model's reasoning OFF.

granite-4.2-8b thinks by default. Its template opens a `<think>` tag in the
GENERATION PROMPT, so the completion contains reasoning with no opening tag for
`_strip_think()` to anchor on -- the deliberation is parsed as if it were the
label. The only fix is the template variable, delivered via miLLM's
`chat_template_kwargs` extension inside `extra_body`.

Both call paths must carry it. The batched path is the one bulk labeling
actually uses, and it builds `extra_body` itself for `extra_messages`, so a fix
applied only to the serial path would look correct and change nothing in
production.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.openai_labeling_service import OpenAILabelingService


def _svc(**kw):
    return OpenAILabelingService(api_key="k", model="granite-4.2-8b",
                                 base_url="http://x/v1", **kw)


def test_thinking_is_disabled_by_default():
    assert _svc().chat_template_kwargs == {"enable_thinking": False}


def test_caller_can_override_or_disable_the_extension():
    assert _svc(chat_template_kwargs={"reasoning_effort": "low"}) \
        .chat_template_kwargs == {"reasoning_effort": "low"}
    # {} means "send nothing", distinct from None meaning "use the default".
    assert _svc(chat_template_kwargs={}).chat_template_kwargs == {}


@pytest.mark.asyncio
async def test_serial_path_sends_the_kwargs():
    svc = _svc()
    svc.client = MagicMock()
    svc.client.chat.completions.create = AsyncMock(return_value="ok")
    await svc._call_openai([{"role": "user", "content": "hi"}])
    sent = svc.client.chat.completions.create.call_args.kwargs
    assert sent["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_batched_path_sends_the_kwargs_and_keeps_extra_messages():
    svc = _svc()
    svc.client = MagicMock()
    raw = MagicMock()
    raw.headers = {"X-miLLM-Batch": "true"}
    raw.parse.return_value = "parsed"
    svc.client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw)

    await svc._call_openai_batched([
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ])
    sent = svc.client.chat.completions.with_raw_response.create.call_args.kwargs
    body = sent["extra_body"]
    # Assert the PAYLOAD and that batching survives: overwriting extra_body
    # rather than adding to it would silently disable batching, which no
    # "was it called" assertion would catch.
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["extra_messages"] == [[{"role": "user", "content": "b"}]]


@pytest.mark.asyncio
async def test_empty_kwargs_sends_no_extension_at_all():
    svc = _svc(chat_template_kwargs={})
    svc.client = MagicMock()
    svc.client.chat.completions.create = AsyncMock(return_value="ok")
    await svc._call_openai([{"role": "user", "content": "hi"}])
    sent = svc.client.chat.completions.create.call_args.kwargs
    assert "chat_template_kwargs" not in (sent.get("extra_body") or {})
