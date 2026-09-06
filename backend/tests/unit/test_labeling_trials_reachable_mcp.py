"""Reachability for the labeling-trial MCP tools.

**A capability is not shipped until a test FAILS when its wiring is removed.**

This repo shipped 16 fully-implemented, unit-tested, documented `millm_circuit_*`
tools that were never registered with the server. Every test passed by importing
the module directly, so the suite was green and the docs said ✅ while no agent
could call the feature. This file is the guard against repeating that.

Three shapes, because each catches what the others cannot:

  1. REGISTRY   — the category is wired into CATEGORY_MODULES / VALID_CATEGORIES
  2. SERVER     — the tools reach a REAL build_server, and are ABSENT when the
                  category is disabled (without that half, shape 2 proves nothing)
  3. CALLER     — each tool issues exactly ONE backend call, to the documented
                  path, with the documented PAYLOAD

Mutation controls:
  C39 delete a tool's @mcp.tool() decorator      -> the SERVER tests fail
  C40 remove "labeling" from CATEGORY_MODULES    -> the REGISTRY tests fail
  C41 change a tool's backend path or payload    -> the CALLER tests fail
  C42 add a tool without an EXPECTED_CALLS entry -> the completeness test fails
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server.config import DEFAULT_CATEGORIES, VALID_CATEGORIES, MCPSettings
from src.mcp_server.tools import CATEGORY_MODULES
from src.mcp_server.tools import labeling as labeling_tools

# Written BEFORE the tools existed, so the list is a specification rather than a
# transcription of whatever happens to be registered.
EXPECTED_TRIAL_TOOLS = {
    "list_labeling_templates",
    "run_labeling_trial",
    "get_labeling_trial",
    "list_labeling_trials",
    "compare_labeling_trials",
}


class TestRegistry:
    def test_the_category_is_in_the_module_registry(self):
        assert "labeling" in CATEGORY_MODULES and CATEGORY_MODULES["labeling"]

    def test_the_category_is_selectable(self):
        assert "labeling" in VALID_CATEGORIES

    def test_it_is_on_by_default(self):
        assert "labeling" in {c.strip() for c in DEFAULT_CATEGORIES.split(",")}

    def test_the_module_exposes_register(self):
        assert hasattr(labeling_tools, "register")


class TestBuiltServer:
    """Through the REAL build_server, not a hand-assembled FastMCP."""

    @staticmethod
    def _tools(categories: str) -> set:
        res = build = __import__(
            "src.mcp_server.server", fromlist=["build_server"]).build_server(
            MCPSettings(tool_categories=categories, allow_anonymous=True),
            stdio=True,
        )
        mcp = res[0] if isinstance(res, tuple) else res
        return {t.name for t in asyncio.run(mcp.list_tools())}

    def test_every_trial_tool_reaches_the_built_server(self):
        names = self._tools("labeling")
        missing = EXPECTED_TRIAL_TOOLS - names
        assert not missing, (
            f"{sorted(missing)} are implemented but never reach the server; an "
            f"agent cannot call them no matter what the docs say"
        )

    def test_they_are_absent_when_the_category_is_disabled(self):
        """Without this, the test above proves nothing about wiring — the tools
        could be registered unconditionally and it would still pass."""
        names = self._tools("read")
        assert not (EXPECTED_TRIAL_TOOLS & names), (
            "labeling tools appear even with the category disabled; category "
            "gating is not actually gating"
        )

    def test_the_default_category_set_reaches_them(self):
        assert EXPECTED_TRIAL_TOOLS <= self._tools(DEFAULT_CATEGORIES)


# tool -> (verb, path, kwargs to call it with, expected call payload)
EXPECTED_CALLS = {
    "list_labeling_templates": (
        "get", "/labeling-prompt-templates", {},
        {"search": None, "limit": 50},
    ),
    "run_labeling_trial": (
        "post", "/labeling/trials",
        {"extraction_job_id": "extr_x", "feature_ids": ["f1", "f2"],
         "prompt_template_id": "lpt_1", "name": "baseline"},
        {"json_body": {
            "extraction_job_id": "extr_x", "feature_ids": ["f1", "f2"],
            "labeling_method": "openai_compatible",
            "prompt_template_id": "lpt_1", "name": "baseline"}},
    ),
    "get_labeling_trial": (
        "get", "/labeling/trials/ltr_abc", {"trial_run_id": "ltr_abc"}, {},
    ),
    "list_labeling_trials": (
        "get", "/labeling/trials", {"panel_id": "pnl_x"},
        {"extraction_job_id": None, "panel_id": "pnl_x",
         "prompt_template_id": None, "limit": 50},
    ),
    "compare_labeling_trials": (
        "get", "/labeling/trials/compare/ltr_a/ltr_b",
        {"run_a": "ltr_a", "run_b": "ltr_b"}, {},
    ),
    # The three PRE-EXISTING tools in this module had no payload assertion.
    # Discovered the hard way: a mutation meant for run_labeling_trial matched
    # update_feature_label's identical line first, broke it, and NOTHING went
    # red. A tool whose payload is unasserted can send the wrong arguments
    # forever — which is exactly the failure the payload rule exists to stop.
    "update_feature_label": (
        "patch", "/features/feat_1",
        {"feature_id": "feat_1", "name": "n", "category": "c"},
        {"json_body": {"label_source": "mcp_agent", "override_protected": False,
                       "name": "n", "category": "c"}},
    ),
    "run_enhanced_labeling": (
        "post", "/features/feat_1/label/enhanced",
        {"feature_id": "feat_1"}, {"json_body": {}},
    ),
    "get_enhanced_label": (
        "get", "/features/feat_1/label/enhanced/latest",
        {"feature_id": "feat_1"}, {},
    ),
}


class TestCaller:
    """Asserting the PAYLOAD, not merely that something was called.

    A reachability test that checks only "was called" passes against a call
    sending entirely the wrong arguments — three such mutations survived a green
    suite elsewhere in this repo before payloads were asserted.
    """

    @staticmethod
    def _register():
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("test")
        client = MagicMock()
        for verb in ("get", "post", "patch", "delete"):
            setattr(client, verb, AsyncMock(return_value={}))
        labeling_tools.register(mcp, client, MCPSettings(allow_anonymous=True))
        return mcp, client

    @pytest.mark.parametrize("tool_name", sorted(EXPECTED_CALLS))
    def test_the_tool_issues_its_documented_call(self, tool_name):
        verb, path, kwargs, payload = EXPECTED_CALLS[tool_name]
        mcp, client = self._register()
        fn = asyncio.run(mcp.get_tool(tool_name)).fn if hasattr(mcp, "get_tool") \
            else None
        if fn is None:  # FastMCP versions differ; fall back to the manager
            fn = mcp._tool_manager._tools[tool_name].fn
        asyncio.run(fn(**kwargs))

        called = getattr(client, verb)
        assert called.await_count == 1, (
            f"{tool_name} issued {called.await_count} {verb.upper()} calls; "
            f"asserting on the last would let a wrong first call through"
        )
        for other in ("get", "post", "patch", "delete"):
            if other != verb:
                assert getattr(client, other).await_count == 0, (
                    f"{tool_name} also issued a {other.upper()}"
                )
        assert called.await_args.args[0] == path
        assert called.await_args.kwargs == payload, (
            f"{tool_name} sent {called.await_args.kwargs} but the contract is "
            f"{payload}"
        )

    def test_every_registered_trial_tool_is_covered_here(self):
        """C42. A new tool cannot ship without a caller assertion."""
        mcp, _ = self._register()
        registered = set(mcp._tool_manager._tools)
        uncovered = registered - set(EXPECTED_CALLS)
        assert not uncovered, (
            f"labeling tools with no caller assertion: {sorted(uncovered)}. Every "
            f"tool in this module needs one — an unasserted payload can send the "
            f"wrong arguments indefinitely."
        )


class TestTheToolsSayWhatTheyDo:
    def test_run_labeling_trial_states_that_it_writes_nothing(self):
        """An agent that assumes a trial applied labels would draw exactly the
        wrong conclusion, and the sibling update_feature_label DOES persist."""
        mcp, _ = TestCaller._register()
        doc = mcp._tool_manager._tools["run_labeling_trial"].fn.__doc__ or ""
        assert "NO FEATURE ROW IS WRITTEN" in doc

    def test_compare_states_both_refusals(self):
        mcp, _ = TestCaller._register()
        doc = mcp._tool_manager._tools["compare_labeling_trials"].fn.__doc__ or ""
        assert "different panels" in doc
        assert "comparing nothing is not comparing" in doc.lower()
