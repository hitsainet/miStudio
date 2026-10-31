"""`MiLLMClient` learns headers, full responses, JSON Lines and multipart (Feature 034, FR-21,
FR-22, FR-23) — without a single new call SHAPE in the tool modules.

Two halves:

1. The client: `request_full` returns status, body and headers; errors keep their headers (a
   503's `Retry-After`); every pre-034 path is unchanged; `get(..., jsonl=True)` checks the
   content type before parsing; `post(..., upload=...)` sends multipart.
2. ⚠ The call-site rule miLLM enforces on US. miLLM's
   `tests/unit/test_mcp_tool_paths_are_real.py` reads miStudio's `millm_*` tool modules by regex,
   maps every `millm.<x>(` site to an HTTP verb and checks its literal path against miLLM's
   routers; an unknown verb fails it outright. The first 034 phase-2 commit called
   `millm.request_full(...)` in `millm_models.py` and turned that guard red (found by running
   miLLM's origin/main guard against this branch). `TestTheCallSitesMiLLMReads` mirrors the rule
   here so it is caught in miStudio's own suite.

Controls (recorded 2026-10-16; each red, each restore verified):
  * remove the content-type check in `get_jsonl` → `test_a_wrong_content_type_RAISES` red;
  * `millm.post(` → `millm.request_full("POST", ` in `millm_models.millm_load_model` →
    `test_every_call_site_uses_a_verb_miLLM_can_map` red;
  * drop `headers=response_headers` from the error raise in `request_full` →
    `test_a_503_keeps_its_Retry_After` red.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import re

import httpx
import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.millm_client import JSONL_CONTENT_TYPE, MiLLMClient, MiLLMResponse

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "tools"


def _client(handler) -> MiLLMClient:
    client = MiLLMClient("http://millm")
    client._http = httpx.AsyncClient(  # noqa: SLF001
        base_url="http://millm", transport=httpx.MockTransport(handler)
    )
    return client


class TestRequestFull:
    def test_it_returns_status_body_and_HEADERS(self):
        client = _client(lambda r: httpx.Response(
            200, json={"x": 1}, headers={"X-miLLM-Steering": "none", "X-miLLM-Seed": "7"}))
        response = asyncio.run(client.post("/v1/chat/completions", json_body={}, full=True))
        assert isinstance(response, MiLLMResponse)
        assert response.status == 200 and response.body == {"x": 1}
        assert response.headers["x-millm-steering"] == "none"

    def test_it_SENDS_the_headers_it_is_given(self):
        seen = {}

        def handler(request):
            seen.update(request.headers)
            return httpx.Response(200, json={})

        asyncio.run(_client(handler).post(
            "/v1/completions", json_body={}, headers={"X-miLLM-Load-Policy": "refuse"}))
        assert seen["x-millm-load-policy"] == "refuse"

    def test_the_envelope_is_still_unwrapped(self):
        client = _client(lambda r: httpx.Response(200, json={"success": True, "data": {"a": 1}}))
        assert asyncio.run(client.get("/api/models")) == {"a": 1}
        assert asyncio.run(client.get("/api/models", full=True)).body == {"a": 1}

    def test_a_503_keeps_its_Retry_After(self):
        client = _client(lambda r: httpx.Response(
            503, json={"error": {"code": "model_loading", "message": "loading"}},
            headers={"Retry-After": "12"}))
        with pytest.raises(BackendError) as exc:
            asyncio.run(client.post("/v1/chat/completions", json_body={}, full=True))
        assert exc.value.status_code == 503 and exc.value.headers["retry-after"] == "12"

    def test_an_HTML_200_still_RAISES(self):
        client = _client(lambda r: httpx.Response(200, text="<html/>"))
        with pytest.raises(BackendError, match="non-JSON"):
            asyncio.run(client.get("/api/status"))

    def test_query_params_still_travel(self):
        seen = {}

        def handler(request):
            seen["q"] = dict(request.url.params)
            return httpx.Response(200, json={})

        asyncio.run(_client(handler).get("/api/probes/events", limit=5, probe_id=None))
        assert seen["q"] == {"limit": "5"}

    def test_a_read_timeout_says_it_may_already_have_been_applied(self):
        def handler(request):
            raise httpx.ReadTimeout("slow", request=request)

        with pytest.raises(BackendError) as exc:
            asyncio.run(_client(handler).post("/v1/completions", json_body={}, full=True))
        assert exc.value.status_code == 0 and "may already have been applied" in str(exc.value)

    def test_DELETE_sends_headers(self):
        seen = {}

        def handler(request):
            seen.update(request.headers)
            seen["method"] = request.method
            return httpx.Response(200, json={"success": True, "data": {}})

        asyncio.run(_client(handler).delete("/api/models/1/lease",
                                            headers={"X-miLLM-Lease": "L"}))
        assert seen["method"] == "DELETE" and seen["x-millm-lease"] == "L"


class TestJsonLines:
    def test_it_parses_rows_when_the_type_is_jsonl(self):
        lines = "\n".join(json.dumps({"custom_id": f"r{i}"}) for i in range(3)) + "\n"
        client = _client(lambda r: httpx.Response(
            200, text=lines, headers={"content-type": JSONL_CONTENT_TYPE}))
        rows = asyncio.run(client.get("/v1/files/f1/content", jsonl=True))
        assert [r["custom_id"] for r in rows] == ["r0", "r1", "r2"]

    def test_a_wrong_content_type_RAISES(self):
        client = _client(lambda r: httpx.Response(
            200, text="<html>proxy</html>", headers={"content-type": "text/html"}))
        with pytest.raises(BackendError, match="not application/jsonl"):
            asyncio.run(client.get("/v1/files/f1/content", jsonl=True))

    def test_the_documented_type_is_miLLMs(self):
        """Read at the 034 phase-6 precondition from miLLM origin/main (`files.py`)."""
        assert JSONL_CONTENT_TYPE == "application/jsonl"

    def test_a_404_still_raises_with_the_body(self):
        client = _client(lambda r: httpx.Response(
            404, json={"error": {"code": "file_expired", "message": "gone"}}))
        with pytest.raises(BackendError) as exc:
            asyncio.run(client.get("/v1/files/f1/content", jsonl=True))
        assert exc.value.status_code == 404 and exc.value.detail["error"]["code"] == "file_expired"


class TestMultipart:
    def test_the_upload_is_multipart_with_file_and_purpose(self):
        seen = {}

        def handler(request):
            seen["type"] = request.headers["content-type"]
            seen["body"] = request.content
            seen["lease"] = request.headers.get("x-millm-lease")
            return httpx.Response(200, json={"id": "file-1", "object": "file"})

        response = asyncio.run(_client(handler).post(
            "/v1/files", upload={"filename": "b.jsonl", "content": b'{"a":1}\n',
                                 "purpose": "batch"},
            headers={"X-miLLM-Lease": "L"}, full=True))
        assert seen["type"].startswith("multipart/form-data")
        assert b'name="purpose"' in seen["body"] and b"batch" in seen["body"]
        assert b'filename="b.jsonl"' in seen["body"] and b'{"a":1}' in seen["body"]
        assert seen["lease"] == "L" and response.body["id"] == "file-1"


# ── The call-site rule miLLM's path guard enforces ───────────────────────────────────────────

#: miLLM origin/main `tests/unit/test_mcp_tool_paths_are_real.py`, verbatim patterns.
CALL = re.compile(r"millm\.(get|post|put|delete|raw_get)\(\s*f?[\"']([^\"']+)[\"']", re.MULTILINE)
CALL_SITE = re.compile(r"millm\.([a-z_]+)\(", re.MULTILINE)
NON_HTTP_ATTRS = {"close", "aclose"}
VERB = {"get", "raw_get", "post", "put", "delete"}


def _millm_modules() -> list[pathlib.Path]:
    return sorted(p for p in TOOLS.glob("millm_*.py") if "millm." in p.read_text())


class TestTheCallSitesMiLLMReads:
    def test_it_sees_the_modules(self):
        names = {p.name for p in _millm_modules()}
        assert {"millm_models.py", "millm_inference.py", "millm_probes.py",
                "millm_batches.py"} <= names

    @pytest.mark.parametrize("path", _millm_modules(), ids=lambda p: p.name)
    def test_every_call_site_uses_a_verb_miLLM_can_map(self, path):
        sites = [v for v in CALL_SITE.findall(path.read_text()) if v not in NON_HTTP_ATTRS]
        unknown = sorted({v for v in sites if v not in VERB})
        assert not unknown, (
            f"{path.name} calls millm.{unknown}: miLLM's path guard cannot map that to an HTTP "
            "verb and FAILS. Use millm.get/post/put/delete with headers=/full=/upload=/jsonl=."
        )

    @pytest.mark.parametrize("path", _millm_modules(), ids=lambda p: p.name)
    def test_every_call_site_has_a_LITERAL_path(self, path):
        text = path.read_text()
        sites = [v for v in CALL_SITE.findall(text) if v not in NON_HTTP_ATTRS]
        assert len(CALL.findall(text)) == len(sites), (
            f"{path.name}: some millm.* call has no literal path, so miLLM's guard cannot check it"
        )
