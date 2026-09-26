"""F20 R3-17/18. Failure-path fidelity for the miLLM MCP client.

There was no test file for this client at all, and it is the single component
every one of the 16 circuit tools passes through. An MCP tool's job is to give
an agent an accurate picture; a client that turns one failure mode into another
makes the agent take the wrong recovery action, confidently.

The defect that motivated this file: a 200 response whose body would not parse
as JSON — a misrouted ingress, a proxy splash page — was returned to the caller
as `{}`. Indistinguishable from a genuine empty success. `millm_circuit_status`
would report nothing steering, and an agent would activate a circuit that
contends with one already serving that layer.
"""

import httpx
import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.millm_client import MiLLMClient


def _client(handler) -> MiLLMClient:
    """A client wired to a mock transport, so no server is needed."""
    c = MiLLMClient("http://millm.test")
    c._http = httpx.AsyncClient(
        base_url="http://millm.test",
        transport=httpx.MockTransport(handler),
    )
    return c


class TestANonJSONSuccessIsNotAnEmptyResult:
    """R3-17. The distinction that keeps an agent from acting on a lie."""

    @pytest.mark.asyncio
    async def test_a_200_html_page_RAISES_rather_than_returning_empty(self):
        c = _client(
            lambda r: httpx.Response(200, text="<html>welcome to nginx</html>")
        )
        with pytest.raises(BackendError) as exc:
            await c.get("/api/circuits/active")
        detail = str(exc.value)
        assert "non-JSON" in detail, (
            "the agent must be told the response did not come from the API"
        )
        assert "NOT an empty result" in detail, (
            "the message must forbid the specific wrong reading — that nothing "
            "is configured — because that reading leads to a contended "
            "activation"
        )

    @pytest.mark.asyncio
    async def test_a_genuinely_empty_200_still_returns_empty(self):
        """The guard must not make a real empty response an error, or every
        no-content endpoint starts failing."""
        c = _client(lambda r: httpx.Response(200, content=b""))
        assert await c.get("/api/circuits/active") == {}

    @pytest.mark.asyncio
    async def test_a_valid_envelope_is_unwrapped_unchanged(self):
        c = _client(
            lambda r: httpx.Response(
                200, json={"success": True, "data": {"id": "c1"}}
            )
        )
        assert await c.get("/api/circuits/active") == {"id": "c1"}

    @pytest.mark.asyncio
    async def test_a_refusal_envelope_still_raises_structured(self):
        c = _client(
            lambda r: httpx.Response(
                200,
                json={
                    "success": False,
                    "error": {"code": "COLLISION", "message": "same feature"},
                },
            )
        )
        with pytest.raises(BackendError) as exc:
            await c.get("/api/circuits/active")
        assert "COLLISION" in str(exc.value)


class TestTimeoutsNameTheirPhase:
    """R3-18. `str(e)` on an httpx timeout is frequently empty."""

    @pytest.mark.asyncio
    async def test_a_read_timeout_warns_the_request_may_have_committed(self):
        """The distinction that matters for a retry decision: a connect
        timeout means miLLM was never reached; a read timeout on a POST means
        the import may ALREADY be applied, and retrying double-imports."""

        def handler(request):
            raise httpx.ReadTimeout("", request=request)

        c = _client(handler)
        with pytest.raises(BackendError) as exc:
            await c.post("/api/circuits/import", json_body={"k": "v"})
        detail = str(exc.value)
        assert "ReadTimeout" in detail
        assert "may already have been applied" in detail
        assert "POST" in detail and "/api/circuits/import" in detail

    @pytest.mark.asyncio
    async def test_a_connect_timeout_does_NOT_warn_about_committing(self):
        """Specificity: if both said the same thing, the warning would carry
        no information and would be ignored where it matters."""

        def handler(request):
            raise httpx.ConnectTimeout("", request=request)

        c = _client(handler)
        with pytest.raises(BackendError) as exc:
            await c.post("/api/circuits/import", json_body={"k": "v"})
        detail = str(exc.value)
        assert "ConnectTimeout" in detail
        assert "may already have been applied" not in detail

    @pytest.mark.asyncio
    async def test_an_empty_exception_string_still_names_the_call(self):
        """The original message was literally 'miLLM request timed out: ' —
        no phase, no method, no path."""

        def handler(request):
            raise httpx.ReadTimeout("", request=request)

        c = _client(handler)
        with pytest.raises(BackendError) as exc:
            await c.get("/api/circuits")
        assert str(exc.value).rstrip().endswith("retrying.") or "GET" in str(
            exc.value
        )

    @pytest.mark.asyncio
    async def test_raw_get_timeout_does_not_NameError(self):
        """R3-18 follow-on: `raw_get` has no `method` variable in scope. The
        shared helper was first called there with `method`, which would have
        raised NameError on EVERY export timeout — masking the timeout with an
        unrelated crash."""

        def handler(request):
            raise httpx.ReadTimeout("", request=request)

        c = _client(handler)
        with pytest.raises(BackendError) as exc:
            await c.raw_get("/api/circuits/c1/export")
        assert "GET" in str(exc.value)
