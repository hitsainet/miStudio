"""The `millm_inference` tools and the lease tools (Feature 034, FR-16, FR-18, FR-19, FR-21, FR-23).

What is pinned here, beyond the caller assertions in `test_reachability.py`:

* **One owner of the sent headers (SC-6).** Every `millm.*` call in the inference modules is
  wrapped in `send(gate, …)` and passes `headers=sent_headers(…)` — asserted on the AST, so a new
  tool that called miLLM directly (and so sent no `X-miLLM-Load-Policy: refuse`) fails here.
* **The fourth result shape.** `409 model_not_resident` / `model_leased` → `refused` with
  `next: "millm_load_model"`; `503` → `refused` with miLLM's `Retry-After`; anything else raises.
* **Caps refuse before the gate, naming the batch tools; a read timeout reaches the agent.**
* **Lease TTL is validated before the gate and refused, never clamped.**

Controls (recorded 2026-10-16, task 4.8; each red, each restore verified):
  * drop `"X-miLLM-Load-Policy": "refuse"` from `SENT_HEADERS` →
    `test_the_two_headers_are_SENT` and `EXPECTED_CALLS[millm_score_chat]` red (SC-6);
  * replace `await send(gate, millm.post(` with `await (millm.post(` in one tool →
    `test_every_millm_call_goes_through_send_with_the_sent_headers` red;
  * `if exc.status_code == 409 and code in LOAD_REFUSALS` → `if False` →
    `test_a_409_model_not_resident_is_REFUSED_with_next` red.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.millm_client import MiLLMResponse
from src.mcp_server.millm_send import SENT_HEADERS, steering_state
from tests.support.external_ctx import make_ctx

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "tools"
#: Modules whose miLLM calls must all go through `send` with the sent headers.
SEND_MODULES = ("millm_inference.py", "millm_batches.py")


class _Millm:
    """A miLLM stand-in whose verbs behave like the real client's: `full=True` returns a
    `MiLLMResponse`; a configured exception is raised instead."""

    def __init__(self, raise_exc=None, headers=None, body=None):
        self.calls = []
        self.raise_exc = raise_exc
        self.headers = headers or {}
        self.body = body if body is not None else {"choices": []}

    async def _verb(self, method, path, json_body=None, *, headers=None, full=False, **kw):
        self.calls.append((method, path, json_body, headers))
        if self.raise_exc is not None:
            raise self.raise_exc
        return MiLLMResponse(200, self.body, self.headers) if full else self.body

    async def post(self, path, json_body=None, **kw):
        return await self._verb("POST", path, json_body, **kw)

    async def get(self, path, **kw):
        return await self._verb("GET", path, **kw)

    async def delete(self, path, **kw):
        return await self._verb("DELETE", path, **kw)


def _tools(millm, gate_open=True):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import millm_inference, millm_models

    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, "ok") if gate_open else (False, "down"))
    gate.invalidate = MagicMock()
    mcp = MCPServer("t")
    ctx = make_ctx(millm_load_approval=False)
    millm_inference.register(mcp, millm, gate, ctx)
    millm_models.register(mcp, millm, gate, ctx)
    return mcp._tool_manager._tools, gate  # noqa: SLF001


def _run(name, millm, gate_open=True, **args):
    tools, gate = _tools(millm, gate_open)
    return asyncio.run(tools[name].fn(**args)), gate


CHAT = {"model": "m", "messages": [{"role": "user", "content": "x"}]}


def _chat(**overrides):
    from src.mcp_server.tools.millm_inference import ChatMessage

    args = dict(CHAT, **overrides)
    args["messages"] = [ChatMessage(**m) for m in args["messages"]]
    return args


# ── one owner of the headers ─────────────────────────────────────────────────────────────────


class TestTheHeadersHaveOneOwner:
    def test_the_two_headers_are_SENT(self):
        millm = _Millm()
        _run("millm_score_chat", millm, **_chat())
        assert millm.calls[0][3] == {"X-miLLM-Load-Policy": "refuse", "X-miLLM-Strict": "true"}
        assert SENT_HEADERS == {"X-miLLM-Load-Policy": "refuse", "X-miLLM-Strict": "true"}

    def test_a_lease_adds_ONLY_its_header(self):
        millm = _Millm()
        _run("millm_score_chat", millm, lease_id="L9", **_chat())
        assert millm.calls[0][3]["X-miLLM-Lease"] == "L9"
        assert "L9" not in repr(millm.calls[0][2])

    @pytest.mark.parametrize("module", SEND_MODULES)
    def test_every_millm_call_goes_through_send_with_the_sent_headers(self, module):
        """AST over the module: each `millm.<verb>(…)` call is the second argument of a
        `send(...)` call and carries `headers=sent_headers(...)`."""
        tree = ast.parse((TOOLS / module).read_text())
        wrapped: set[int] = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "send" and len(node.args) == 2):
                wrapped.add(id(node.args[1]))
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "millm"
        ]
        assert calls, f"{module}: no millm calls found — the scan is checking nothing"
        for call in calls:
            assert id(call) in wrapped, f"{module}:{call.lineno} calls miLLM outside send()"
            headers = [k for k in call.keywords if k.arg == "headers"]
            assert headers and isinstance(headers[0].value, ast.Call) and getattr(
                headers[0].value.func, "id", None) == "sent_headers", (
                f"{module}:{call.lineno} does not pass headers=sent_headers(...)")
            assert any(k.arg == "full" for k in call.keywords), (
                f"{module}:{call.lineno} does not ask for the full response (headers lost)")


# ── the fourth result shape ──────────────────────────────────────────────────────────────────


class TestRefusals:
    def test_a_409_model_not_resident_is_REFUSED_with_next(self):
        exc = BackendError(409, {"error": {"code": "model_not_resident", "message": "not here"}})
        result, _ = _run("millm_score_chat", _Millm(raise_exc=exc), **_chat())
        assert result == {"refused": "model_not_resident", "message": "not here",
                          "next": "millm_load_model", "retry_after_s": None, "status_code": 409}

    def test_a_409_model_leased_is_REFUSED_with_next(self):
        exc = BackendError(409, {"error": {"code": "model_leased", "message": "held"}})
        result, _ = _run("millm_score_completion", _Millm(raise_exc=exc), model="m",
                         prompt=["x"])
        assert result["refused"] == "model_leased" and result["next"] == "millm_load_model"

    def test_a_503_carries_RETRY_AFTER(self):
        exc = BackendError(503, {"error": {"code": "model_loading", "message": "wait"}},
                           headers={"retry-after": "17"})
        result, _ = _run("millm_score_chat", _Millm(raise_exc=exc), **_chat())
        assert result["refused"] == "model_loading" and result["retry_after_s"] == 17
        assert result["status_code"] == 503

    def test_any_other_error_RAISES_with_miLLMs_message(self):
        exc = BackendError(400, {"error": {"code": "unused_fields_refused", "message": "x"}})
        with pytest.raises(BackendError):
            _run("millm_score_chat", _Millm(raise_exc=exc), **_chat())

    def test_a_READ_TIMEOUT_reaches_the_agent_and_resets_the_gate(self):
        exc = BackendError(0, "miLLM request timed out (ReadTimeout) on POST /v1/completions "
                              "— a read timeout means the request may already have been applied")
        result, gate = _run("millm_score_chat", _Millm(raise_exc=exc), **_chat())
        assert result["unavailable"] == "millm"
        assert "may already have been applied" in result["reason"]
        gate.invalidate.assert_called_once_with("millm")


class TestProvenance:
    def test_headers_come_back_VERBATIM_and_only_when_sent(self):
        millm = _Millm(headers={"X-miLLM-Steering": "profile;id=p1", "X-miLLM-Seed": "7"})
        result, _ = _run("millm_score_chat", millm, **_chat())
        assert result["millm_headers"] == {"steering": "profile;id=p1", "seed": "7"}
        assert result["result"] == {"choices": []}

    @pytest.mark.parametrize("raw,state", [
        ("none", "unsteered"),
        ('unknown;reason="engine"', "unknown"),
        ('profile;intensity="0.5"', "steered"),
        ("inline", "steered"),
        ("circuit;id=c1", "steered"),
        ("none, inline", "steered"),
        (None, "not_reported"),
    ])
    def test_ONLY_a_sole_none_reads_as_unsteered(self, raw, state):
        assert steering_state(raw) == state


# ── caps and arguments before the gate ───────────────────────────────────────────────────────


class TestCapsAndArgumentsComeFirst:
    def test_over_the_conversation_cap_refuses_naming_the_batch_tools(self):
        from src.mcp_server.tools.millm_inference import MAX_SCORE_CONVERSATIONS

        millm = _Millm()
        extra = [list(_chat()["messages"])] * MAX_SCORE_CONVERSATIONS
        result, gate = _run("millm_score_chat", millm, gate_open=False,
                            extra_messages=extra, **_chat())
        assert result["error"] == "over_synchronous_cap"
        assert "millm_submit_batch" in result["message"]
        assert not millm.calls and gate.check.await_count == 0

    def test_over_the_prompt_cap_refuses_before_the_gate(self):
        from src.mcp_server.tools.millm_inference import MAX_SCORE_PROMPTS

        millm = _Millm()
        result, gate = _run("millm_score_completion", millm, gate_open=False, model="m",
                            prompt=["x"] * (MAX_SCORE_PROMPTS + 1))
        assert result["error"] == "over_synchronous_cap" and gate.check.await_count == 0

    def test_the_caps_are_STATED_in_the_descriptions(self):
        from src.mcp_server.tools import millm_inference as mod

        tools, _ = _tools(_Millm())
        assert f"At most {mod.MAX_SCORE_CONVERSATIONS} conversations" in (
            tools["millm_score_chat"].description)
        assert f"At most {mod.MAX_SCORE_PROMPTS} prompts" in (
            tools["millm_score_completion"].description)
        for name in ("millm_score_chat", "millm_score_completion"):
            d = tools[name].description
            assert "UNSTEERED" in d and "millm_submit_batch" in d and "refused" in d


# ── leases ───────────────────────────────────────────────────────────────────────────────────


class TestLeases:
    @pytest.mark.parametrize("ttl", [0, 7201, -5])
    def test_a_bad_TTL_is_refused_before_the_gate_and_never_clamped(self, ttl):
        millm = _Millm()
        for name, args in (("millm_acquire_lease", {"model_id": 1, "reason": "r"}),
                           ("millm_renew_lease", {"model_id": 1, "lease_id": "L"})):
            result, gate = _run(name, millm, gate_open=False, ttl_seconds=ttl, **args)
            assert result["error"] == "invalid_ttl" and str(ttl) in result["message"]
            assert gate.check.await_count == 0
        assert not millm.calls

    def test_the_holder_is_the_MCP_identity_and_ttl_travels_when_given(self):
        millm = _Millm(body={"lease_id": "L1", "model_id": 1})
        result, _ = _run("millm_acquire_lease", millm, model_id=1, reason="run",
                         ttl_seconds=600)
        assert millm.calls[0][2] == {"holder": "agent:mistudio-mcp", "reason": "run",
                                     "ttl_seconds": 600}
        assert result["lease"]["lease_id"] == "L1" and "ONCE" in result["hint"]

    def test_renew_sends_the_ttl_in_the_body_and_the_id_in_the_header(self):
        millm = _Millm()
        _run("millm_renew_lease", millm, model_id=1, lease_id="L1", ttl_seconds=60)
        assert millm.calls[0] == ("POST", "/api/models/1/lease/renew", {"ttl_seconds": 60},
                                  {"X-miLLM-Lease": "L1"})

    def test_the_descriptions_say_what_an_agent_must_know(self):
        tools, _ = _tools(_Millm())
        d = tools["millm_acquire_lease"].description
        for needle in ("only time it is ever returned", "One lease per server", "RESIDENT",
                       "restart", "No operator approval"):
            assert needle in d, needle
        assert "Never returns a lease id" in tools["millm_lease_status"].description


# ── phase 5: millm_score_probes and millm_generate ───────────────────────────────────────────


def _probe_tools(millm, gate_open=True):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import millm_probes

    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, "ok") if gate_open else (False, "down"))
    gate.invalidate = MagicMock()
    mcp = MCPServer("t")
    millm_probes.register(mcp, millm, gate, make_ctx())
    return mcp._tool_manager._tools, gate  # noqa: SLF001


def _score(millm, inputs, gate_open=True, **kw):
    from src.mcp_server.tools.millm_probes import ProbeScoreInput

    tools, gate = _probe_tools(millm, gate_open)
    items = [ProbeScoreInput(**i) for i in inputs]
    return asyncio.run(tools["millm_score_probes"].fn(inputs=items, **kw)), gate


class TestScoreProbes:
    @pytest.mark.parametrize("bad", [
        {"text": "a", "messages": [{"role": "user", "content": "b"}]},
        {"token_ids": [1], "text": "a"},
        {},
        {"text": "a", "prompt_tokens": 1},
    ])
    def test_input_shape_is_checked_BEFORE_the_gate(self, bad):
        millm = _Millm()
        result, gate = _score(millm, [{"text": "ok"}, bad], gate_open=False)
        assert result["error"] == "invalid_input" and "inputs[1]" in result["message"]
        assert gate.check.await_count == 0 and not millm.calls

    def test_the_cap_refuses_before_the_gate(self):
        from src.mcp_server.tools.millm_probes import MAX_PROBE_INPUTS

        millm = _Millm()
        result, gate = _score(millm, [{"text": "x"}] * (MAX_PROBE_INPUTS + 1), gate_open=False)
        assert result["error"] == "over_synchronous_cap" and gate.check.await_count == 0

    def test_the_verdict_comes_back_VERBATIM_including_null(self):
        body = {"results": [{"probe_id": "pr_1", "score": 0.5, "threshold": 0.5,
                             "verdict": None, "rung_language": "an association, measured"}],
                "skipped": [{"probe_id": "pr_2", "code": "MODEL_MISMATCH"}]}
        result, _ = _score(_Millm(body=body), [{"token_ids": [1, 2]}])
        assert result["result"] == body

    def test_the_tool_NEVER_compares_a_score_with_a_threshold(self):
        """P-03: the boundary lives in miLLM. A tool that recomputed `score >= threshold` here
        would disagree the day either side changes it."""
        import inspect

        from src.mcp_server.tools import millm_probes

        src = inspect.getsource(millm_probes.register)
        fn = next(n for n in ast.walk(ast.parse(inspect.cleandoc("\n" + src)))
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "millm_score_probes")
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        compares = [n for n in ast.walk(fn) if isinstance(n, ast.Compare)
                    and any(isinstance(o, (ast.GtE, ast.Gt, ast.Lt, ast.LtE)) for o in n.ops)
                    and "threshold" in ast.dump(n)]
        assert not compares and "verdict" not in names

    def test_its_millm_call_goes_through_send_with_the_sent_headers(self):
        millm = _Millm()
        _score(millm, [{"text": "x"}], lease_id="L")
        assert millm.calls[0][1] == "/api/probes/score"
        assert millm.calls[0][3] == {"X-miLLM-Load-Policy": "refuse", "X-miLLM-Strict": "true",
                                     "X-miLLM-Lease": "L"}

    def test_the_description_says_records_nothing_and_null_is_silence(self):
        tools, _ = _probe_tools(_Millm())
        d = " ".join(tools["millm_score_probes"].description.split())
        for needle in ("records nothing", "changes nothing armed", "SAID NOTHING",
                       "never false", "not miStudio's `pm_…` ids", "At most 64 inputs"):
            assert needle in d, needle


class TestGenerate:
    def _gen(self, millm, **kw):
        args = _chat(max_tokens=8)
        args.update(kw)
        return _run("millm_generate", millm, **args)

    @pytest.mark.parametrize("header,state", [
        ("none", "unsteered"),
        ('unknown;reason="no_report"', "unknown"),
        ('profile;id="p1";intensity="1.0"', "steered"),
        ("inline", "steered"),
        ("circuit;id=c1", "steered"),
        (None, "not_reported"),
    ])
    def test_steering_is_reported_from_the_HEADER_only(self, header, state):
        headers = {} if header is None else {"X-miLLM-Steering": header}
        result, _ = self._gen(_Millm(headers=headers))
        assert result["steering_state"] == state
        if header is None:
            assert "steering" not in result["millm_headers"]
        else:
            assert result["millm_headers"]["steering"] == header

    def test_steering_seed_format_and_activations_pass_VERBATIM(self):
        millm = _Millm()
        steering = {"sae_id": "s1", "features": [{"index": 1, "strength": -3.5}]}
        self._gen(millm, seed=7, steering=steering, response_format={"type": "json_object"},
                  return_sae_activations={"layers": [3]})
        body = millm.calls[0][2]
        assert body["steering"] == steering and body["seed"] == 7
        assert body["response_format"] == {"type": "json_object"}
        assert body["return_sae_activations"] == {"layers": [3]}
        assert "profile" not in body

    def test_max_tokens_over_the_cap_refuses_before_the_gate(self):
        from src.mcp_server.tools.millm_inference import MAX_GENERATE_TOKENS

        millm = _Millm()
        args = _chat(max_tokens=MAX_GENERATE_TOKENS + 1)
        result, gate = _run("millm_generate", millm, gate_open=False, **args)
        assert result["error"] == "over_synchronous_cap" and gate.check.await_count == 0


# ── phase 6: the Batch API ───────────────────────────────────────────────────────────────────


class _BatchMillm(_Millm):
    def __init__(self, rows=None, **kw):
        super().__init__(**kw)
        self.rows = rows or []
        self.uploads = []

    async def post(self, path, json_body=None, *, upload=None, **kw):
        if upload is not None:
            self.uploads.append(upload)
        return await self._verb("POST", path, json_body, **kw)

    async def get(self, path, *, jsonl=False, headers=None, full=False, **kw):
        self.calls.append(("GET", path, None, headers))
        if self.raise_exc is not None:
            raise self.raise_exc
        return MiLLMResponse(200, self.rows if jsonl else self.body, self.headers)


def _batch(name, millm, gate_open=True, **args):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import millm_batches

    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, "ok") if gate_open else (False, "down"))
    gate.invalidate = MagicMock()
    mcp = MCPServer("t")
    millm_batches.register(mcp, millm, gate, make_ctx())
    return asyncio.run(mcp._tool_manager._tools[name].fn(**args)), gate  # noqa: SLF001


def _rows(*ids):
    from src.mcp_server.tools.millm_batches import BatchRow

    return [BatchRow(custom_id=i, body={"model": "m", "prompt": i}) for i in ids]


class TestBatches:
    def test_every_line_names_the_batch_endpoint_and_POST(self):
        millm = _BatchMillm()
        _batch("millm_upload_batch_file", millm, endpoint="/v1/chat/completions",
               rows=_rows("a", "b"))
        lines = [__import__("json").loads(x)
                 for x in millm.uploads[0]["content"].decode().splitlines()]
        assert [(l["url"], l["method"], l["custom_id"]) for l in lines] == [
            ("/v1/chat/completions", "POST", "a"), ("/v1/chat/completions", "POST", "b")]
        assert millm.uploads[0]["purpose"] == "batch"

    def test_duplicate_custom_ids_are_refused_before_the_gate(self):
        millm = _BatchMillm()
        result, gate = _batch("millm_upload_batch_file", millm, gate_open=False,
                              endpoint="/v1/completions", rows=_rows("a", "b", "a"))
        assert result["error"] == "duplicate_custom_id" and "'a'" in result["message"]
        assert gate.check.await_count == 0 and not millm.calls

    def test_too_many_rows_are_refused_before_the_gate(self, monkeypatch):
        from src.mcp_server.tools import millm_batches

        monkeypatch.setattr(millm_batches, "MAX_BATCH_ROWS", 2)
        millm = _BatchMillm()
        result, gate = _batch("millm_upload_batch_file", millm, gate_open=False,
                              endpoint="/v1/completions", rows=_rows("a", "b", "c"))
        assert result["error"] == "too_many_rows" and gate.check.await_count == 0

    def test_an_oversized_file_is_refused_before_the_gate(self, monkeypatch):
        from src.mcp_server.tools import millm_batches

        monkeypatch.setattr(millm_batches, "MAX_BATCH_BYTES", 50)
        millm = _BatchMillm()
        result, gate = _batch("millm_upload_batch_file", millm, gate_open=False,
                              endpoint="/v1/completions", rows=_rows("a", "b"))
        assert result["error"] == "file_too_large" and gate.check.await_count == 0

    def test_the_caps_are_the_designed_ones(self):
        from src.mcp_server.tools import millm_batches as m

        assert (m.MAX_BATCH_ROWS, m.MAX_BATCH_BYTES, m.RESULTS_PAGE) == (10_000, 16 * 2**20, 100)

    def test_the_lease_header_travels_when_given(self):
        millm = _BatchMillm()
        _batch("millm_submit_batch", millm, input_file_id="f", endpoint="/v1/completions",
               lease_id="L7")
        assert millm.calls[0][3]["X-miLLM-Lease"] == "L7"

    def test_pack_is_OFF_unless_asked(self):
        """Packing changed 0.57% of top tokens when measured; an unpacked row equals the
        synchronous endpoint, so the tool never turns it on by default."""
        millm = _BatchMillm()
        _batch("millm_submit_batch", millm, input_file_id="f", endpoint="/v1/completions")
        assert "pack" not in millm.calls[0][2]

    def test_results_are_PAGED(self):
        rows = [{"custom_id": f"r{i}"} for i in range(250)]
        millm = _BatchMillm(rows=rows)
        first, _ = _batch("millm_batch_results", millm, file_id="f1")
        assert len(first["rows"]) == 100 and first["total"] == 250
        assert first["next_offset"] == 100
        last, _ = _batch("millm_batch_results", millm, file_id="f1", offset=200)
        assert [r["custom_id"] for r in last["rows"]][0] == "r200"
        assert last["next_offset"] is None

    def test_a_wrong_content_type_reaches_the_agent_as_an_error(self):
        exc = BackendError(200, "miLLM returned text/html from /v1/files/f1/content, not "
                                "application/jsonl")
        with pytest.raises(BackendError, match="not application/jsonl"):
            _batch("millm_batch_results", _BatchMillm(raise_exc=exc), file_id="f1")

    def test_a_page_ENDING_exactly_at_the_last_row_has_no_next(self):
        """Review round 2026-10-16: `end < len(rows)` → `<=` SURVIVED the paging test above (its
        last page overshoots). A next_offset pointing past the end sends an agent to fetch an
        empty page and read it as 'no more results' only by luck.
        Negative control: `<` → `<=` → this test red."""
        millm = _BatchMillm(rows=[{"custom_id": f"r{i}"} for i in range(200)])
        page, _ = _batch("millm_batch_results", millm, file_id="f1", offset=100)
        assert len(page["rows"]) == 100 and page["next_offset"] is None
