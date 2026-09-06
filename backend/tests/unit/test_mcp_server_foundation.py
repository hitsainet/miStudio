"""Unit tests for the MCP server foundation (Feature 010 Phase 5/6)."""

import anyio
import httpx
import pytest

from src.mcp_server.client import BackendError, MiStudioClient
from src.mcp_server.config import MCPSettings
from src.mcp_server.server import build_server, wrap_tool_with_audit


def make_settings(**overrides) -> MCPSettings:
    defaults = dict(auth_token="secret-token", _env_file=None)
    defaults.update(overrides)
    return MCPSettings(**defaults)


class TestConfig:
    def test_default_categories_exclude_admin(self):
        cats = make_settings().enabled_categories()
        assert "admin" not in cats
        assert {"read", "groups", "steering", "labeling", "experiments",
                "profiles", "circuits", "jlens", "jobs"} == cats
        # Unified MCP categories are opt-in only
        assert not any(c.startswith("millm_") for c in cats)

    def test_unknown_category_rejected(self):
        with pytest.raises(ValueError, match="Unknown MCP tool categories"):
            make_settings(tool_categories="read,bogus").enabled_categories()

    def test_admin_opt_in(self):
        cats = make_settings(tool_categories="read,admin").enabled_categories()
        assert cats == {"read", "admin"}


class TestStartupGuard:
    def test_empty_token_refuses_startup(self):
        with pytest.raises(SystemExit, match="MCP_AUTH_TOKEN is required"):
            build_server(make_settings(auth_token=""))

    def test_anonymous_flag_does_NOT_allow_an_empty_token_over_http(self):
        """MIS-E2E-150. This test PINNED the defect.

        It asserted that `MCP_ALLOW_ANONYMOUS` alone satisfies the startup guard
        — which is exactly the hole. `__main__.py` then adds
        `BearerAuthMiddleware` only when a token is set, otherwise it logs a
        warning and serves on `settings.host`, default `0.0.0.0`. So an operator
        following the manual's own troubleshooting remedy got a LAN-reachable
        UNAUTHENTICATED MCP server exposing `delete_circuit`, GPU steering and
        label write-back.

        Both the manual ("bearer token always required") and the guard's own
        error message ("for local stdio development only") stated the rule.
        Nothing enforced it, and this test certified the unenforced version.
        """
        with pytest.raises(SystemExit, match="stdio"):
            build_server(make_settings(auth_token="", allow_anonymous=True))

    def test_anonymous_flag_allows_an_empty_token_over_stdio(self):
        """The case the flag exists for, and the only one it now covers."""
        mcp, _client = build_server(
            make_settings(auth_token="", allow_anonymous=True), stdio=True
        )
        assert mcp is not None

    def test_stdio_allows_empty_token(self):
        mcp, client = build_server(make_settings(auth_token=""), stdio=True)
        assert mcp is not None


class TestCategoryGating:
    def _tool_names(self, settings: MCPSettings) -> set[str]:
        mcp, _ = build_server(settings)

        async def collect():
            return {t.name for t in await mcp.list_tools()}

        return anyio.run(collect)

    def test_default_catalog(self):
        names = self._tool_names(make_settings())
        assert "search_features" in names
        assert "compute_feature_groups" in names
        assert "steer_sweep" in names
        assert "update_feature_label" in names
        # admin off by default
        assert "delete_extraction" not in names
        # approval-poll tool absent when approval mode off
        assert "get_approval_status" not in names

    def test_read_only_exposure(self):
        names = self._tool_names(make_settings(tool_categories="read"))
        assert "get_feature" in names
        assert "steer_compare" not in names
        assert "update_feature_label" not in names

    def test_admin_and_approval_tools_appear_when_enabled(self):
        names = self._tool_names(
            make_settings(tool_categories="read,steering,admin", steering_approval=True)
        )
        assert "delete_extraction" in names
        assert "get_approval_status" in names

    def test_audit_wrapper_preserves_tools(self):
        mcp, _ = build_server(make_settings())
        wrap_tool_with_audit(mcp)

        async def collect():
            return await mcp.list_tools()

        tools = anyio.run(collect)
        assert len(tools) > 30


class TestClientErrorPassthrough:
    def _client_with_response(self, status_code: int, json_body=None, text=""):
        def handler(request: httpx.Request) -> httpx.Response:
            if json_body is not None:
                return httpx.Response(status_code, json=json_body)
            return httpx.Response(status_code, text=text)

        client = MiStudioClient("http://backend:8000")
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://backend:8000/api/v1"
        )
        return client

    def test_503_detail_passed_verbatim(self):
        client = self._client_with_response(
            503, json_body={"detail": "No model loaded on labeling endpoint"}
        )

        async def call():
            with pytest.raises(BackendError) as exc_info:
                await client.get("/labeling/models/available")
            return exc_info.value

        err = anyio.run(call)
        assert err.status_code == 503
        assert "No model loaded" in str(err)

    def test_structured_409_detail_preserved(self):
        detail = {"code": "PROTECTED_LABEL", "message": "protected", "hint": "override"}
        client = self._client_with_response(409, json_body={"detail": detail})

        async def call():
            with pytest.raises(BackendError) as exc_info:
                await client.patch("/features/x", json_body={})
            return exc_info.value

        err = anyio.run(call)
        assert err.detail == detail
        assert "PROTECTED_LABEL" in str(err)

    def test_success_returns_json(self):
        client = self._client_with_response(200, json_body={"data": [1, 2]})

        async def call():
            return await client.get("/extractions")

        assert anyio.run(call) == {"data": [1, 2]}


class TestBearerMiddleware:
    @pytest.fixture()
    def app_client(self):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        from src.mcp_server.server import BearerAuthMiddleware

        async def ok(request):
            return JSONResponse({"ok": True})

        async def health(request):
            return JSONResponse({"status": "ok"})

        app = Starlette(routes=[Route("/mcp", ok, methods=["GET"]), Route("/health", health)])
        app.add_middleware(BearerAuthMiddleware, token="secret-token")
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    def test_missing_token_401(self, app_client):
        async def call():
            return await app_client.get("/mcp")

        assert anyio.run(call).status_code == 401

    def test_wrong_token_401(self, app_client):
        async def call():
            return await app_client.get("/mcp", headers={"Authorization": "Bearer nope"})

        assert anyio.run(call).status_code == 401

    def test_correct_token_ok(self, app_client):
        async def call():
            return await app_client.get(
                "/mcp", headers={"Authorization": "Bearer secret-token"}
            )

        assert anyio.run(call).status_code == 200

    def test_health_exempt(self, app_client):
        async def call():
            return await app_client.get("/health")

        assert anyio.run(call).status_code == 200


class TestSteeringSlotRelease:
    """Regression: in-flight slots must release on the backend's real terminal
    states (nested status.status of success/failure/revoked), not the
    non-existent flat 'completed' — the old check leaked slots until the
    concurrency guardrail locked out all steering (2026-07-14)."""

    def _run_tool(self, mcp, name, args):
        import anyio

        async def call():
            return await mcp.call_tool(name, args)

        return anyio.run(call)

    def test_terminal_poll_releases_slot(self):
        from unittest.mock import AsyncMock, patch
        from src.mcp_server import tools
        from src.mcp_server.tools import steering as steering_tools
        from src.mcp_server.server import build_server

        mcp, client = build_server(make_settings(tool_categories="steering"))
        steering_tools._inflight.clear()
        steering_tools._inflight.add("task-x")
        with patch.object(
            client, "get",
            AsyncMock(return_value={"task_id": "task-x", "status": {"status": "success"}, "result": {}}),
        ):
            self._run_tool(mcp, "get_steering_result", {"task_id": "task-x"})
        assert "task-x" not in steering_tools._inflight

    def test_non_terminal_poll_keeps_slot(self):
        from unittest.mock import AsyncMock, patch
        from src.mcp_server.tools import steering as steering_tools
        from src.mcp_server.server import build_server

        mcp, client = build_server(make_settings(tool_categories="steering"))
        steering_tools._inflight.clear()
        steering_tools._inflight.add("task-y")
        with patch.object(
            client, "get",
            AsyncMock(return_value={"task_id": "task-y", "status": {"status": "started"}, "result": None}),
        ):
            self._run_tool(mcp, "get_steering_result", {"task_id": "task-y"})
        assert "task-y" in steering_tools._inflight
        steering_tools._inflight.clear()
