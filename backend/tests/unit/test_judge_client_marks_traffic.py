"""Every judge call is marked as judge traffic (038 FR-5; D14; miLLM 036 FR-36.13).

N rows through Part A's real `_client_for` and `ask_judge` against a MockTransport: N requests, each
carrying `X-miLLM-Origin: judge`, and `ask_judge` still sends exactly three parameters. Control C-HEADER:
remove `default_headers=` from `_client_for` → red.
"""
import json

import httpx

from src.services.probe_monitor_judge import JUDGE_ORIGIN, JUDGE_ORIGIN_HEADER, _client_for, ask_judge, stakes_spec


def test_every_judge_call_is_marked():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                         "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": '{"rating": 3}'}}]})

    client = _client_for("http://judge.test/v1", None, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    for i in range(4):
        ask_judge(client, "m", stakes_spec(), [{"role": "user", "content": f"row {i}"}])
    assert len(seen) == 4
    assert [r.headers.get(JUDGE_ORIGIN_HEADER) for r in seen] == [JUDGE_ORIGIN] * 4
    assert (JUDGE_ORIGIN_HEADER, JUDGE_ORIGIN) == ("X-miLLM-Origin", "judge")
    assert set(json.loads(seen[0].content)) == {"model", "temperature", "messages"}
