"""MIS-E2E-115 / -116: a docs tool that mutated the process, and a gate that raised.

Both break a contract the server states to agents in its own instructions.
"""

import os

import httpx
import pytest


class TestInvalidUrlIsNotAnHttpError:
    """The premise of -116, asserted so the fix cannot be quietly reverted."""

    def test_invalid_url_bypasses_the_httperror_handler(self):
        assert not issubclass(httpx.InvalidURL, httpx.HTTPError), (
            "httpx now derives InvalidURL from HTTPError; the extra handler in "
            "health_gate is redundant but harmless — re-read the fix"
        )

    def test_unsupported_protocol_is_covered_by_httperror(self):
        """Contrast: this one WAS already caught, which is why it was missed."""
        assert issubclass(httpx.UnsupportedProtocol, httpx.HTTPError)


class TestTheGateAnswersInsteadOfRaising:
    @pytest.mark.asyncio
    async def test_a_malformed_url_yields_unavailable_not_an_exception(self):
        from src.mcp_server.health_gate import HealthGate

        # An invalid PORT is what actually raises httpx.InvalidURL on 0.28.1.
        # `http://[not a url` — the first thing I tried — does not: httpx
        # accepts it and the failure surfaces as a ConnectError, which the
        # existing HTTPError handler already caught. Picking a case that
        # genuinely reaches this clause is the difference between testing the
        # fix and testing the handler next to it.
        gate = HealthGate(millm_url="http://host:notaport")
        available, reason = await gate._probe("millm")
        assert available is False
        # Specific, not just non-empty: a catch-all would also return False
        # here, so control C194 (removing the InvalidURL clause) passed until
        # this asserted WHICH answer comes back. The agent reads this string.
        assert "malformed URL" in reason, (
            f"reason was {reason!r}; the InvalidURL clause is not running, so "
            f"a configuration typo is reported as a generic probe failure"
        )

    @pytest.mark.asyncio
    async def test_an_unroutable_host_yields_unavailable(self):
        from src.mcp_server.health_gate import HealthGate

        gate = HealthGate(millm_url="http://millm.invalid")
        available, reason = await gate._probe("millm")
        assert available is False
        assert reason

    @pytest.mark.asyncio
    async def test_an_unconfigured_product_yields_unavailable(self):
        from src.mcp_server.health_gate import HealthGate

        gate = HealthGate(millm_url="")
        available, reason = await gate._probe("millm")
        assert available is False
        assert "not configured" in reason


class TestReadingTheDocsDoesNotChangeTheProcess:
    """MIS-E2E-115: `os.environ.setdefault` outlives the call that made it."""

    def test_the_howto_module_does_not_write_to_the_environment(self):
        import ast
        import inspect

        from src.mcp_server.tools import howto

        tree = ast.parse(inspect.getsource(howto))
        writes = []
        for node in ast.walk(tree):
            # os.environ.setdefault(...) / os.environ.update(...)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func.value
                if (isinstance(target, ast.Attribute) and target.attr == "environ"
                        and node.func.attr in ("setdefault", "update", "pop")):
                    writes.append(node.lineno)
            # os.environ[...] = ...
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute)
                            and t.value.attr == "environ"):
                        writes.append(node.lineno)
        assert not writes, (
            f"howto.py writes to os.environ at {writes}. It is a read-only "
            f"documentation tool; a write there changes what the "
            f"unauthenticated /health endpoint reports for the life of the "
            f"process."
        )

    def test_listing_the_tools_leaves_millm_api_url_untouched(self):
        """The behavioural half — run it and check the environment after."""
        from src.mcp_server.tools.howto import _all_tools

        sentinel = object()
        before = os.environ.get("MILLM_API_URL", sentinel)
        try:
            tools = _all_tools()
            assert tools, "the enumeration returned nothing, so it proves nothing"
        finally:
            after = os.environ.get("MILLM_API_URL", sentinel)
        assert after is before or after == before, (
            f"MILLM_API_URL changed from {before!r} to {after!r} just by "
            f"listing the tools"
        )

    def test_the_override_still_makes_the_millm_tools_visible(self):
        """The fix must not silently drop the categories it was working around."""
        from src.mcp_server.tools.howto import _all_tools

        tools = _all_tools()
        names = {name for entries in tools.values() for name, _ in entries}
        assert any(n.startswith("millm_") for n in names), (
            "no millm_* tools enumerated — the placeholder URL is not reaching "
            "registration, so the docs under-report the surface"
        )

    def test_the_override_does_not_leak_into_a_default_settings_object(self):
        from src.mcp_server.config import MCPSettings

        overridden = MCPSettings(millm_api_url_override="http://millm.invalid")
        assert overridden.millm_api_url == "http://millm.invalid"

        plain = MCPSettings()
        assert plain.millm_api_url != "http://millm.invalid" or \
            os.environ.get("MILLM_API_URL") == "http://millm.invalid", (
            "the override bled into an unrelated settings object"
        )
