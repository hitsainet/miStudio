"""Unit tests for the unified-MCP miLLM integration (miLLM Feature 9,
cross-repo): envelope-unwrapping client, health gate, tool smoke tests with a
mocked client, and the deployment-topology matrix."""

from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
import pytest

from src.mcp_server.client import BackendError
from src.mcp_server.config import MCPSettings
from src.mcp_server.health_gate import HealthGate, gated
from src.mcp_server.millm_client import MiLLMClient
from src.mcp_server.server import build_server


def make_settings(**overrides) -> MCPSettings:
    defaults = dict(auth_token="secret-token", _env_file=None)
    defaults.update(overrides)
    return MCPSettings(**defaults)


def make_client(handler) -> MiLLMClient:
    client = MiLLMClient("http://millm:8000")
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://millm:8000"
    )
    return client


# =============================================================================
# MiLLMClient — envelope unwrap (contract §2)
# =============================================================================


class TestMiLLMClient:
    def test_success_envelope_returns_data_only(self):
        def handler(request):
            return httpx.Response(200, json={
                "success": True, "data": {"clusters": [1, 2]}, "error": None})

        result = anyio.run(make_client(handler).get, "/api/clusters")
        assert result == {"clusters": [1, 2]}  # no {"success": …} leak

    def test_error_envelope_raises_backend_error_with_code(self):
        def handler(request):
            return httpx.Response(422, json={
                "success": False, "data": None,
                "error": {"code": "VALIDATION_ERROR",
                          "message": "meaningless steering",
                          "details": {"d_sae": 100}}})

        with pytest.raises(BackendError) as exc:
            anyio.run(make_client(handler).post, "/api/clusters/x/activate")
        assert exc.value.detail["code"] == "VALIDATION_ERROR"
        assert "meaningless" in exc.value.detail["message"]

    def test_error_envelope_authoritative_even_on_200(self):
        """miLLM's house style returns some caps as 200 + success=false."""
        def handler(request):
            return httpx.Response(200, json={
                "success": False, "data": None,
                "error": {"code": "PAYLOAD_TOO_LARGE", "message": "too big"}})

        with pytest.raises(BackendError) as exc:
            anyio.run(make_client(handler).post, "/api/clusters/import")
        assert exc.value.detail["code"] == "PAYLOAD_TOO_LARGE"

    def test_raw_get_bypasses_envelope(self):
        """Cluster export IS the artifact — no envelope, returned verbatim."""
        def handler(request):
            return httpx.Response(200, json={
                "kind": "mistudio.cluster-definition", "schema_version": "1"})

        result = anyio.run(make_client(handler).raw_get,
                           "/api/clusters/prof_x/export")
        assert result["kind"] == "mistudio.cluster-definition"

    def test_unreachable_raises_backend_error(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        with pytest.raises(BackendError, match="unreachable"):
            anyio.run(make_client(handler).get, "/api/health/detailed")

    def test_repo_id_slash_not_double_encoded(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            return httpx.Response(200, json={"success": True, "data": [],
                                             "error": None})

        anyio.run(make_client(handler).get,
                  "/api/clusters/hub/org/pack/definitions")
        assert seen["path"] == "/api/clusters/hub/org/pack/definitions"


# =============================================================================
# HealthGate (contract §3)
# =============================================================================


class TestHealthGate:
    @staticmethod
    def _stub_http(gate, responder):
        """Replace the gate's long-lived probe client (built in __init__)."""
        stub = MagicMock()
        stub.get = responder
        gate._http = stub

    def test_2xx_is_available_even_degraded(self):
        gate = HealthGate("http://millm:8000")
        self._stub_http(gate, AsyncMock(return_value=httpx.Response(
            200, json={"status": "degraded"})))
        ok, reason = anyio.run(gate.check, "millm")
        assert ok is True and reason == "ok"

    def test_unhealthy_body_refuses(self):
        """Contract: available <=> 2xx AND status != 'unhealthy' (R1 fix)."""
        gate = HealthGate("http://millm:8000")
        self._stub_http(gate, AsyncMock(return_value=httpx.Response(
            200, json={"status": "unhealthy"})))
        ok, reason = anyio.run(gate.check, "millm")
        assert ok is False and "unhealthy" in reason

    def test_3xx_refuses(self):
        """An ingress redirect fronting a dead backend is NOT available."""
        gate = HealthGate("http://millm:8000")
        self._stub_http(gate, AsyncMock(return_value=httpx.Response(
            307, headers={"location": "http://elsewhere"})))
        ok, reason = anyio.run(gate.check, "millm")
        assert ok is False and "307" in reason

    def test_connection_failure_refuses_with_reason(self):
        gate = HealthGate("http://millm:8000")
        self._stub_http(gate, AsyncMock(
            side_effect=httpx.ConnectError("refused")))
        ok, reason = anyio.run(gate.check, "millm")
        assert ok is False
        assert "ConnectError" in reason and "/api/health" in reason

    def test_ttl_caches_probe(self):
        gate = HealthGate("http://millm:8000", ttl_s=60.0)
        calls = []

        async def get(url):
            calls.append(url)
            return httpx.Response(200, json={"status": "healthy"})

        self._stub_http(gate, get)

        async def run():
            await gate.check("millm")
            await gate.check("millm")

        anyio.run(run)
        assert len(calls) == 1  # second check served from cache

    def test_mistudio_product_probes_v1_system_health(self):
        gate = HealthGate("", mistudio_url="http://backend:8000")
        seen = []

        async def get(url):
            seen.append(url)
            return httpx.Response(200, json={"status": "healthy"})

        self._stub_http(gate, get)
        ok, _ = anyio.run(gate.check, "mistudio")
        assert ok is True
        assert seen == ["http://backend:8000/api/v1/system/health"]

    def test_unconfigured_url_refuses(self):
        gate = HealthGate("")
        ok, reason = anyio.run(gate.check, "millm")
        assert ok is False and "MILLM_API_URL" in reason

    def test_gated_decorator_returns_structured_unavailable(self):
        gate = HealthGate("")

        @gated(gate, "millm")
        async def tool():
            return {"data": 1}

        result = anyio.run(tool)
        assert result == {"unavailable": "millm",
                          "reason": "MILLM_API_URL is not configured"}


# =============================================================================
# Tool smoke tests (mocked client; gate open)
# =============================================================================


class _OpenGate(HealthGate):
    def __init__(self):
        super().__init__("http://millm:8000")

    async def check(self, product):
        return True, "ok"


def _register_and_get(module, tool_name):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("test")
    millm = MagicMock(spec=MiLLMClient)
    millm.get = AsyncMock(return_value={"ok": 1})
    millm.post = AsyncMock(return_value={"ok": 1})
    millm.put = AsyncMock(return_value={"ok": 1})
    millm.raw_get = AsyncMock(return_value={"kind": "x"})
    module.register(mcp, millm, _OpenGate())
    tool = mcp._tool_manager._tools[tool_name]  # noqa: SLF001
    return tool, millm


class TestToolSmoke:
    def test_import_xor_validation(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters, "millm_import_cluster")
        both = anyio.run(lambda: tool.run({"definition": {"kind": "x"},
                                           "repo_id": "org/pack"}))
        neither = anyio.run(lambda: tool.run({}))
        assert "exactly one source" in str(both)
        assert "exactly one source" in str(neither)
        millm.post.assert_not_called()

    def test_import_inline_posts_document(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters, "millm_import_cluster")
        anyio.run(lambda: tool.run({"definition": {"kind": "d"},
                                    "activate": True}))
        args, kwargs = millm.post.call_args
        assert args[0] == "/api/clusters/import"
        assert kwargs["json_body"] == {"kind": "d"}
        assert kwargs["activate"] == "true"

    def test_import_hub_requires_filename(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters, "millm_import_cluster")
        result = anyio.run(lambda: tool.run({"repo_id": "org/pack"}))
        assert "filename" in str(result)

    def test_status_hits_detailed_health(self):
        from src.mcp_server.tools import millm_runtime

        tool, millm = _register_and_get(millm_runtime, "millm_status")
        anyio.run(lambda: tool.run({}))
        millm.get.assert_awaited_once_with("/api/health/detailed")

    def test_set_intensity_puts_active_route(self):
        from src.mcp_server.tools import millm_runtime

        tool, millm = _register_and_get(millm_runtime, "millm_set_intensity")
        anyio.run(lambda: tool.run({"intensity": 1.2}))
        millm.put.assert_awaited_once_with(
            "/api/clusters/active/intensity",
            json_body={"intensity": 1.2, "reapply": True})

    def test_export_uses_raw_get(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters, "millm_export_cluster")
        anyio.run(lambda: tool.run({"cluster_id": "prof_x"}))
        millm.raw_get.assert_awaited_once_with("/api/clusters/prof_x/export")

    def test_sensing_tools_routes(self):
        from src.mcp_server.tools import millm_sensing

        tool, millm = _register_and_get(millm_sensing, "millm_sensing_enable")
        anyio.run(lambda: tool.run({"profile_id": "prof_x"}))
        millm.post.assert_awaited_once_with("/api/sensing/prof_x/enable")


# =============================================================================
# Topology matrix (FR-9.4)
# =============================================================================


class TestTopology:
    def _tool_names(self, mcp):
        return {t.name for t in mcp._tool_manager.list_tools()}  # noqa: SLF001

    def test_both_products(self, monkeypatch):
        monkeypatch.setenv("MILLM_API_URL", "http://millm:8000")
        mcp, _ = build_server(make_settings(
            tool_categories="read,millm_runtime,millm_clusters,millm_sensing"))
        names = self._tool_names(mcp)
        assert "millm_status" in names and "millm_import_cluster" in names
        assert any(not n.startswith("millm_") for n in names)  # miStudio too

    def test_mistudio_only_url_unset_skips_categories(self, monkeypatch):
        monkeypatch.delenv("MILLM_API_URL", raising=False)
        mcp, _ = build_server(make_settings(
            tool_categories="read,millm_runtime"))
        names = self._tool_names(mcp)
        assert not any(n.startswith("millm_") for n in names)
        assert len(names) > 0  # miStudio tools unaffected

    def test_millm_down_structured_unavailable(self, monkeypatch):
        """Registered but unreachable: tools answer, with the gate refusal."""
        monkeypatch.setenv("MILLM_API_URL", "http://127.0.0.1:1")  # nothing there
        mcp, _ = build_server(make_settings(tool_categories="millm_runtime"))
        tool = mcp._tool_manager._tools["millm_status"]  # noqa: SLF001
        result = anyio.run(lambda: tool.run({}))
        text = str(result)
        assert "unavailable" in text and "millm" in text

    def test_default_categories_exclude_millm(self):
        cats = make_settings().enabled_categories()
        assert not any(c.startswith("millm_") for c in cats)


class TestRound1Fixes:
    """009 R1 regression pins."""

    def test_activate_profile_sends_required_body(self):
        """R1 #1 (confirmed): a body-less POST 422s on the real route."""
        from src.mcp_server.tools import millm_runtime

        tool, millm = _register_and_get(millm_runtime,
                                        "millm_activate_profile")
        anyio.run(lambda: tool.run({"profile_id": "prof_x"}))
        args, kwargs = millm.post.call_args
        assert args[0] == "/api/profiles/prof_x/activate"
        assert kwargs["json_body"] == {"apply_steering": True}

    def test_set_intensity_null_reapply_means_true(self):
        from src.mcp_server.tools import millm_runtime

        tool, millm = _register_and_get(millm_runtime, "millm_set_intensity")
        anyio.run(lambda: tool.run({"intensity": 1.2, "reapply": None}))
        kwargs = millm.put.call_args.kwargs
        assert kwargs["json_body"]["reapply"] is True

    def test_import_empty_dict_definition_reaches_millm(self):
        """R1 #10: presence, not truthiness — {} goes to the real validator."""
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters,
                                        "millm_import_cluster")
        anyio.run(lambda: tool.run({"definition": {}}))
        millm.post.assert_awaited_once()

    def test_import_on_conflict_passthrough_and_validation(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters,
                                        "millm_import_cluster")
        result = anyio.run(lambda: tool.run({"definition": {"kind": "x"},
                                             "on_conflict": "explode"}))
        assert "on_conflict" in str(result)
        millm.post.assert_not_called()
        anyio.run(lambda: tool.run({"definition": {"kind": "x"},
                                    "on_conflict": "fail"}))
        assert millm.post.call_args.kwargs["on_conflict"] == "fail"

    def test_sensing_events_passes_since(self):
        from src.mcp_server.tools import millm_sensing

        tool, millm = _register_and_get(millm_sensing, "millm_sensing_events")
        anyio.run(lambda: tool.run({"since": "2026-07-16T00:00:00Z"}))
        assert millm.get.call_args.kwargs["since"] == "2026-07-16T00:00:00Z"

    def test_raw_get_envelope_error_is_structured(self):
        def handler(request):
            return httpx.Response(404, json={
                "success": False, "data": None,
                "error": {"code": "PROFILE_NOT_FOUND", "message": "gone"}})

        with pytest.raises(BackendError) as exc:
            anyio.run(make_client(handler).raw_get,
                      "/api/clusters/ghost/export")
        assert exc.value.detail["code"] == "PROFILE_NOT_FOUND"

    def test_health_route_reports_products(self, monkeypatch):
        monkeypatch.setenv("MILLM_API_URL", "http://127.0.0.1:1")
        # keep the 'unit' test off the network (a live backend on :8000
        # made the mistudio probe environment-dependent: 009 R2)
        monkeypatch.setenv("MISTUDIO_API_URL", "http://127.0.0.1:1")
        mcp, _ = build_server(make_settings(tool_categories="read"))

        async def run():
            import httpx as _httpx

            app = mcp.streamable_http_app()
            transport = _httpx.ASGITransport(app=app)
            async with _httpx.AsyncClient(transport=transport,
                                          base_url="http://t") as http:
                return await http.get("/health")

        response = anyio.run(run)
        body = response.json()
        # snapshot semantics (009 R2): /health never blocks — the first hit
        # may report null (not probed yet) and kicks a background refresh
        assert body["products"]["millm"]["available"] in (False, None)
        assert "mistudio" in body["products"]
        # reasons are sanitized categories, never internal URLs
        for product in body["products"].values():
            assert "http" not in (product["reason"] or "")


class TestRound2Fixes:
    """009 R2 pins."""

    def test_snapshot_never_blocks_and_sanitizes(self):
        gate = HealthGate("http://millm-backend.millm.svc:8000")
        available, reason = gate.snapshot("millm")
        assert available is None and reason == "not probed yet"
        assert HealthGate.public_reason(
            "ConnectError: boom (http://internal.svc:8000/api/health)"
        ) == "unreachable"
        assert HealthGate.public_reason("MILLM_API_URL is not configured") \
            == "not configured"

    def test_cache_stamped_after_probe(self):
        """A slow probe must not eat into the TTL window (009 R2)."""
        import time as _time

        gate = HealthGate("http://millm:8000", ttl_s=10.0)

        async def slow_get(url):
            await __import__("anyio").sleep(0.05)
            return httpx.Response(200, json={"status": "healthy"})

        stub = MagicMock()
        stub.get = slow_get
        gate._http = stub
        anyio.run(gate.check, "millm")
        stamped = gate._cache["millm"][0]
        assert _time.monotonic() - stamped < 0.05  # post-probe stamp

    def test_single_flight_under_concurrency(self):
        gate = HealthGate("http://millm:8000", ttl_s=60.0)
        calls = []

        async def get(url):
            calls.append(url)
            await __import__("anyio").sleep(0.02)
            return httpx.Response(200, json={"status": "healthy"})

        stub = MagicMock()
        stub.get = get
        gate._http = stub

        async def run():
            import asyncio

            await asyncio.gather(*[gate.check("millm") for _ in range(8)])

        anyio.run(run)
        assert len(calls) == 1  # no thundering herd (009 R2)

    def test_raw_get_2xx_non_json_is_structured(self):
        def handler(request):
            return httpx.Response(200, text="<html>splash</html>")

        with pytest.raises(BackendError, match="non-JSON"):
            anyio.run(make_client(handler).raw_get, "/api/clusters/x/export")

    def test_raw_get_non_envelope_4xx_stays_structured(self):
        def handler(request):
            return httpx.Response(422, json={"detail": [{"loc": ["query"]}]})

        with pytest.raises(BackendError) as exc:
            anyio.run(make_client(handler).raw_get, "/api/clusters/x/export")
        assert isinstance(exc.value.detail, dict)  # parsed, not re-stringified

    def test_hub_import_forwards_on_conflict(self):
        from src.mcp_server.tools import millm_clusters

        tool, millm = _register_and_get(millm_clusters, "millm_import_cluster")
        anyio.run(lambda: tool.run({"repo_id": "org/pack",
                                    "filename": "a.cluster.json",
                                    "on_conflict": "fail"}))
        body = millm.post.call_args.kwargs["json_body"]
        assert body["on_conflict"] == "fail"

    def test_sensing_since_rejects_naive_timestamps(self):
        from src.mcp_server.tools import millm_sensing

        tool, millm = _register_and_get(millm_sensing, "millm_sensing_events")
        result = anyio.run(lambda: tool.run({"since": "2026-07-16 09:00"}))
        assert "UTC offset" in str(result)
        millm.get.assert_not_called()
        anyio.run(lambda: tool.run({"since": "2026-07-16T09:00:00Z"}))
        millm.get.assert_awaited_once()

    def test_close_backend_clients_hook_exists(self):
        mcp, _ = build_server(make_settings(tool_categories="read"))
        assert callable(getattr(mcp, "close_backend_clients", None))
        anyio.run(mcp.close_backend_clients)


class TestRound3Fixes:
    """009 R3 pins."""

    def test_vendored_contracts_identical_across_repos(self):
        """The cross-product pipe rests on both repos vendoring the SAME
        frozen v1 schema — pin them to EACH OTHER, not just to their own
        mirrors (009 R3: regenerating one side passed its own sync test
        while silently drifting from the other)."""
        from pathlib import Path

        ours = Path(__file__).resolve().parents[3] / "backend" / "src"
        mistudio_schema = None
        for candidate in (
            Path(__file__).resolve().parents[3] / "docs" / "schemas"
            / "cluster-definition-v1.json",
            Path(__file__).resolve().parents[2] / "docs" / "schemas"
            / "cluster-definition-v1.json",
        ):
            if candidate.exists():
                mistudio_schema = candidate
                break
        millm_schema = Path("/home/x-sean/app/miLLM/docs/schemas/"
                            "cluster-definition-v1.json")
        if mistudio_schema is None or not millm_schema.exists():
            pytest.skip("both repos not present in this environment")
        assert mistudio_schema.read_bytes() == millm_schema.read_bytes()

    def test_public_reason_anchored_categories(self):
        assert HealthGate.public_reason(
            "HTTP 503 from http://millm-unhealthy.svc/api/health"
        ) == "error response"  # hostname must not trip the unhealthy branch
        assert HealthGate.public_reason(
            "status 'unhealthy' from http://x/api/health") == "unhealthy"
        assert HealthGate.public_reason("unknown product 'x'") \
            == "unknown product"
        assert HealthGate.public_reason(
            "timed out after 3.0s (http://x/api/health)") == "timed out"

    def test_request_non_envelope_4xx_raises(self):
        """R3 mutation pin: >=400 handling in request() (not just raw_get)."""
        def handler(request):
            return httpx.Response(422, json={"detail": [{"loc": ["query"]}]})

        with pytest.raises(BackendError) as exc:
            anyio.run(make_client(handler).get, "/api/sensing/events")
        assert exc.value.status_code == 422
        assert isinstance(exc.value.detail, dict)

    def test_timeout_labeled_as_timeout(self):
        def handler(request):
            raise httpx.ReadTimeout("boom")

        with pytest.raises(BackendError, match="timed out"):
            anyio.run(make_client(handler).get, "/api/health/detailed")

    def test_gated_mid_ttl_outage_is_structured_and_invalidates(self):
        """EC-9.3: an unreachable raised inside a fresh ok-window returns
        the structured unavailable NOW and expires the stale entry."""
        gate = HealthGate("http://millm:8000", ttl_s=60.0)
        gate._cache["millm"] = (__import__("time").monotonic(), True, "ok")

        @gated(gate, "millm")
        async def tool():
            raise BackendError(0, "miLLM backend unreachable: refused")

        result = anyio.run(tool)
        assert result["unavailable"] == "millm"
        assert "millm" not in gate._cache  # invalidated

    def test_gated_normal_errors_still_raise(self):
        gate = HealthGate("http://millm:8000", ttl_s=60.0)
        gate._cache["millm"] = (__import__("time").monotonic(), True, "ok")

        @gated(gate, "millm")
        async def tool():
            raise BackendError(422, {"code": "VALIDATION_ERROR",
                                     "message": "bad"})

        with pytest.raises(BackendError):
            anyio.run(tool)

    def test_snapshot_background_refresh_lands(self):
        gate = HealthGate("http://millm:8000", ttl_s=60.0)

        async def get(url):
            return httpx.Response(200, json={"status": "healthy"})

        stub = MagicMock()
        stub.get = get
        gate._http = stub

        async def run():
            import asyncio

            available, reason = gate.snapshot("millm")
            assert available is None  # not probed yet
            await asyncio.sleep(0.05)  # let the background refresh land
            return gate.snapshot("millm")

        available, reason = anyio.run(run)
        assert available is True and reason == "ok"

    def test_audit_wrapped_millm_tool_still_callable(self, monkeypatch):
        """R3: the gated+audit+FastMCP triple-wrap path was never executed."""
        from src.mcp_server.server import wrap_tool_with_audit

        monkeypatch.setenv("MILLM_API_URL", "http://127.0.0.1:1")
        mcp, _ = build_server(make_settings(tool_categories="millm_runtime"))
        wrap_tool_with_audit(mcp)
        tool = mcp._tool_manager._tools["millm_status"]  # noqa: SLF001
        result = anyio.run(lambda: tool.run({}))
        assert "unavailable" in str(result)


class TestSensingConfigTool:
    """Enh R2: millm_sensing_config pins (was unpinned; clear-on-omit)."""

    def test_set_min_k_routes(self):
        from src.mcp_server.tools import millm_sensing

        tool, millm = _register_and_get(millm_sensing, "millm_sensing_config")
        anyio.run(lambda: tool.run({"profile_id": "prof_x", "min_k": 2}))
        millm.put.assert_awaited_once_with(
            "/api/sensing/prof_x/config", json_body={"min_k": 2})

    def test_omitting_min_k_refused_not_cleared(self):
        """Calling without min_k must NOT silently wipe the override."""
        from src.mcp_server.tools import millm_sensing

        tool, millm = _register_and_get(millm_sensing, "millm_sensing_config")
        result = anyio.run(lambda: tool.run({"profile_id": "prof_x"}))
        assert "provide" in str(result)
        millm.put.assert_not_called()

    def test_explicit_reset_clears(self):
        from src.mcp_server.tools import millm_sensing

        tool, millm = _register_and_get(millm_sensing, "millm_sensing_config")
        anyio.run(lambda: tool.run({"profile_id": "prof_x", "reset": True}))
        millm.put.assert_awaited_once_with(
            "/api/sensing/prof_x/config", json_body={"min_k": None})


    def test_min_k_plus_reset_refused(self):
        """R3 #9: the contradictory combination must not silently reset."""
        from src.mcp_server.tools import millm_sensing

        tool, millm = _register_and_get(millm_sensing, "millm_sensing_config")
        result = anyio.run(lambda: tool.run({"profile_id": "p", "min_k": 3,
                                             "reset": True}))
        assert "contradictory" in str(result)
        millm.put.assert_not_called()
