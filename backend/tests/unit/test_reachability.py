"""Feature 20 task 4.0 — REACHABILITY ASSURANCE.

The rule this file enforces, stated once:

    A capability is not shipped until a test FAILS when its wiring is removed.

Not "a test exists". Not "the symbol is present". A test that asserts a
mechanism EXISTS passes forever after the mechanism stops being called — which
is exactly how, across this increment, five separate mechanisms shipped that
nothing invoked:

  * `CircuitClaimRegistry.reconcile()` — written, unit-tested, zero callers
  * `ContentionDialog` / `ClaimsStrip` — exported, tested, rendered by no page
  * `all_incumbents` — added to the payload, read by nothing
  * `attached_layers()` — orphaned by the fix that replaced its only consumer
  * `TestRingPruningIsWired` — asserted an entry point existed while nothing
    called it (the name this rule cites by convention)

Three shapes are needed, because each catches what the others cannot:

  1. REGISTRY — the category is in the maps that make it selectable.
  2. BUILT SERVER — the REAL `build_server()` registers it. A hand-called
     `register()` bypasses the gating that actually failed in the audit.
  3. CALLER — each tool issues its documented method and path. Catches a tool
     that registers but calls nothing, or calls the wrong endpoint.
"""

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server.config import DEFAULT_CATEGORIES, VALID_CATEGORIES
from src.mcp_server.tools import MILLM_CATEGORY_MODULES


# ── Shape 1: registry ──────────────────────────────────────────────────────



def _tool_names_in(module) -> set[str]:
    """Tool names a module DECLARES, read from its source text.

    Deliberately not obtained by registering the module: that would ask the
    registration path to vouch for itself, which is the circularity this file
    exists to break. `@mcp.tool()` immediately above `async def NAME` is the
    declaration; whether it reaches the server is the thing under test.
    """
    import inspect
    import re as _re

    src = inspect.getsource(module)
    return set(
        _re.findall(
            r"@mcp\.tool\(\)(?:\s*#[^\n]*)?\s*(?:@[\w.()\"', ]+\s*)*"
            r"(?:#[^\n]*\s*)*async def (millm_[a-z_]+)",
            src,
        )
    )



def _millm_circuits_module():
    """The module under test, resolved once for parametrize-time use."""
    from src.mcp_server.tools import millm_circuits

    return millm_circuits


class TestRegistryReachability:
    """RCH-3, shape 1. A module imported but not registered fails SILENTLY —
    the import succeeds, the tools are never exposed, and nothing errors."""

    def test_the_category_is_in_the_module_registry(self):
        assert "millm_circuits" in MILLM_CATEGORY_MODULES, (
            "the module is imported but not registered — its tools are "
            "unreachable and nothing reports it"
        )

    def test_the_category_is_selectable(self):
        assert "millm_circuits" in VALID_CATEGORIES, (
            "the category cannot be enabled, so registering it is moot"
        )

    def test_it_is_NOT_a_default(self):
        """No `millm_*` category is ever default: they are functional only with
        MILLM_API_URL set, and a default that cannot work is worse than one an
        operator opted into."""
        assert "millm_circuits" not in DEFAULT_CATEGORIES

    def test_every_registered_module_actually_exposes_register(self):
        for category, modules in MILLM_CATEGORY_MODULES.items():
            for module in modules:
                assert hasattr(module, "register"), (
                    f"{category} lists {module.__name__}, which has no "
                    "register() — the server would skip it silently"
                )


# ── Shape 2: built server ──────────────────────────────────────────────────


class TestBuiltServerReachability:
    """RCH-3, shape 2. The REAL `build_server()`, not a hand-called
    `register()`.

    This distinction is the finding: the audit's circuit-category defect was
    that registration existed and the SERVER never picked it up. A test that
    calls `register()` directly proves the module works and says nothing about
    whether anything calls it.
    """

    def _build(self, categories: str, monkeypatch):
        """The REAL build path, with its REAL signature.

        `build_server(settings, stdio=)` returns `(mcp, client)`; calling it
        wrongly is itself the kind of drift this file exists to catch, so the
        helper stays thin rather than wrapping it in a mock.
        """
        from src.mcp_server.config import MCPSettings
        from src.mcp_server.server import build_server

        # NOT MCP_-prefixed: `millm_api_url` reads the bare env var.
        monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
        settings = MCPSettings(
            tool_categories=categories,
            allow_anonymous=True,
        )
        mcp, _client = build_server(settings, stdio=True)
        return {t.name for t in asyncio.run(mcp.list_tools())}

    def test_build_server_exposes_the_circuit_tools(self, monkeypatch):
        names = self._build("millm_circuits", monkeypatch)

        assert "millm_circuit_status" in names, (
            "build_server() did not expose the circuit tools even with the "
            "category enabled — registration exists but nothing reaches it"
        )
        # A representative from each family, so dropping one group is caught.
        for expected in (
            "millm_activate_circuit",
            "millm_import_circuit",
            "millm_export_circuit",
            "millm_circuit_claims",
            "millm_release_circuit_claims",
            "millm_circuit_sensing_events",
        ):
            assert expected in names, f"{expected} is not reachable"

    def test_the_category_is_absent_when_not_enabled(self, monkeypatch):
        """Specificity: if the tools appeared regardless, the test above would
        prove nothing about the wiring."""
        names = self._build("read", monkeypatch)
        assert "millm_circuit_status" not in names

    @pytest.mark.parametrize("category", sorted(MILLM_CATEGORY_MODULES))
    def test_EVERY_millm_category_reaches_the_built_server(
        self, category, monkeypatch
    ):
        """F20 R3-10. The original F20 defect was reproducible TODAY, one
        category over.

        The test above hardcoded `millm_circuits`, so the reachability rule
        this whole file exists to enforce was enforced for exactly one of four
        categories. Adding `if category == "millm_sensing": continue` to the
        registration loop in server.py left the entire suite green while a
        whole tool category was unreachable — the same shape as the defect
        that started this feature (16 tools implemented, unit-tested,
        documented, never registered).

        Driven off the registry rather than a hand-written list, so a category
        added later is covered the moment it exists. A hand-maintained list is
        only as good as the list, and this file already learned that once.
        """
        names = self._build(category, monkeypatch)
        expected = {
            t
            for module in MILLM_CATEGORY_MODULES[category]
            for t in _tool_names_in(module)
        }
        assert expected, (
            f"could not determine any tool names for {category} — the "
            "extraction is broken and this test is checking nothing"
        )
        missing = sorted(expected - names)
        assert not missing, (
            f"category {category!r} is enabled but {missing} did not reach "
            "the built server. The tools exist and are testable by direct "
            "import; no agent can call them."
        )


# ── Shape 3: caller ────────────────────────────────────────────────────────


class RecordingClient:
    """Records the method and path each tool actually issues."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    async def _record(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return {"success": True, "data": {}}

    async def get(self, path, **params):
        return await self._record("GET", path, **params)

    async def post(self, path, json_body=None, **params):
        return await self._record("POST", path, json_body=json_body, **params)

    async def put(self, path, json_body):
        # F20 R1-07: `json_body` is REQUIRED on the real client. Giving it a
        # default here meant a tool calling `put(path)` with no body passed
        # every test and TypeError'd in production — the recorder was more
        # forgiving than the thing it stands in for, which is the one property
        # a stand-in must never have.
        return await self._record("PUT", path, json_body=json_body)

    async def delete(self, path, **params):
        return await self._record("DELETE", path, **params)

    async def raw_get(self, path):
        # R1-08: the real `raw_get` returns the UNENVELOPED body — that is the
        # whole reason the export uses it. Returning an envelope here would let
        # a tool that wrongly unwraps `.data` pass.
        self.calls.append(("RAW_GET", path, {}))
        return {"kind": "mistudio.circuit-definition", "schema_version": "1"}


def _open_gate():
    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, None))
    return gate


def _tools_by_name():
    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.tools import millm_circuits

    mcp = FastMCP("t")
    client = RecordingClient()
    millm_circuits.register(mcp, client, _open_gate())
    manager = mcp._tool_manager
    return manager, client


#: Each tool's DOCUMENTED call: `(method, path, tool_kwargs, expected_payload)`.
#:
#: F20 R1-04: `expected_payload` was added because the assertion below checked
#: only method and path and DISCARDED the recorded kwargs. Three mutations
#: survived 26/26 green:
#:
#:   * dropping `limit=`/`offset=` from the list call
#:   * changing the intensity body key to `{"lambda": …}` — which 422s on
#:     EVERY call in production, because the endpoint requires `intensity`
#:   * disabling the manual gate check in `millm_import_circuit`
#:
#: A harness whose stated rule is "a test FAILS when the wiring is removed"
#: was verifying the address and not the letter.
EXPECTED_CALLS = {
    "millm_circuit_status": ("GET", "/api/circuits/active", {}, {}),
    "millm_list_circuits": ("GET", "/api/circuits", {}, {"limit": 50, "offset": 0}),
    "millm_circuit_claims": ("GET", "/api/circuits/claims", {}, {}),
    "millm_activate_circuit": (
        "POST", "/api/circuits/c1/activate", {"circuit_id": "c1"},
        {"json_body": None, "acknowledge_unvalidated": "false",
         "allow_layer_overlap": "false"},
    ),
    "millm_deactivate_circuit": (
        "POST", "/api/circuits/c1/deactivate", {"circuit_id": "c1"},
        {"json_body": None},
    ),
    "millm_set_circuit_intensity": (
        "PUT", "/api/circuits/active/intensity", {"intensity": 1.0},
        # The body KEY is the whole contract here: the endpoint requires
        # `intensity`, so any other key 422s on every call.
        {"json_body": {"intensity": 1.0, "acknowledge_unvalidated": False}},
    ),
    "millm_delete_circuit": (
        # `acknowledge_serving=True` so this exercises the TRANSMIT path in
        # isolation. Without it the tool first reads /api/circuits/active to
        # decide whether to refuse (R2-20), which is a second call — and the
        # one-call assertion below is load-bearing, so it stays strict here.
        # The guard itself is covered by TestDeletingAServingCircuitIsRefused.
        "DELETE", "/api/circuits/c1",
        {"circuit_id": "c1", "acknowledge_serving": True}, {},
    ),
    "millm_import_circuit": (
        "POST", "/api/circuits/import", {"definition": {"kind": "x"}},
        {"json_body": {"kind": "x"}, "on_conflict": None},
    ),
    "millm_export_circuit": (
        "RAW_GET", "/api/circuits/c1/export", {"circuit_id": "c1"}, {},
    ),
    "millm_release_circuit_claims": (
        "POST", "/api/circuits/claims/release", {"circuit_id": "c1"},
        {"json_body": None, "circuit_id": "c1"},
    ),
    "millm_circuit_sensing_status": (
        "GET", "/api/circuit-sensing/status", {}, {},
    ),
    "millm_circuit_sensing_events": (
        "GET", "/api/circuit-sensing/events", {},
        {"circuit_id": None, "limit": 50, "since": None},
    ),
    "millm_circuit_sensing_event": (
        "GET", "/api/circuit-sensing/events/7", {"event_id": 7}, {},
    ),
    "millm_circuit_sensing_enable": (
        "POST", "/api/circuit-sensing/c1/enable", {"circuit_id": "c1"},
        {"json_body": None},
    ),
    "millm_circuit_sensing_disable": (
        "POST", "/api/circuit-sensing/c1/disable", {"circuit_id": "c1"},
        {"json_body": None},
    ),
    "millm_circuit_sensing_clear": (
        # R1-16: scope is REQUIRED — an unscoped call is refused, not treated
        # as "delete everything".
        "DELETE", "/api/circuit-sensing/events", {"circuit_id": "c1"},
        {"circuit_id": "c1"},
    ),
}


class TestCallerReachability:
    """RCH-4, shape 3. Each tool ISSUES its documented call.

    Registration proves a tool is exposed. This proves it does something, and
    the right something: re-pointing a path or deleting the call body fails
    here and nowhere else.
    """

    @pytest.mark.parametrize("tool_name", sorted(EXPECTED_CALLS))
    def test_the_tool_issues_its_documented_call(self, tool_name):
        manager, client = _tools_by_name()
        method, path, kwargs, payload = EXPECTED_CALLS[tool_name]

        asyncio.run(manager.call_tool(tool_name, kwargs))

        assert client.calls, f"{tool_name} registered but issued NO call"
        assert len(client.calls) == 1, (
            f"{tool_name} issued {len(client.calls)} calls; asserting on the "
            "last would let a wrong first call through"
        )
        actual_method, actual_path, actual_payload = client.calls[0]
        assert (actual_method, actual_path) == (method, path), (
            f"{tool_name} called {actual_method} {actual_path}, documented as "
            f"{method} {path}"
        )
        # R1-04: the PAYLOAD, not just the address. Verifying where a tool
        # points and not what it sends is how a permanently-422ing dial ships
        # with a green suite.
        assert actual_payload == payload, (
            f"{tool_name} sent {actual_payload}, documented as {payload}"
        )

    def test_every_registered_tool_is_covered_here(self):
        """A tool added without an entry above would otherwise be exposed with
        no caller assertion at all — the gap this file exists to close."""
        manager, _ = _tools_by_name()
        registered = {t.name for t in manager.list_tools()}
        missing = registered - set(EXPECTED_CALLS)
        assert not missing, (
            f"tools with no caller assertion: {sorted(missing)} — add them to "
            "EXPECTED_CALLS or they ship unverified"
        )

    def test_export_returns_the_document_UNWRAPPED(self):
        """R1-08. The export IS the portable artifact. A tool that unwrapped
        `.data` would return None for a raw response — and the previous
        assertion checked only the method string, so it could not tell."""
        manager, client = _tools_by_name()
        result = asyncio.run(
            manager.call_tool("millm_export_circuit", {"circuit_id": "c1"})
        )
        assert "mistudio.circuit-definition" in str(result), (
            "the export did not come back as the raw document — unwrapping "
            "`.data` on a raw response yields None"
        )

    def test_export_uses_the_RAW_path(self):
        """The export IS the portable artifact. Re-wrapping it in the envelope
        changes what a round-trip produces, so it must not go through the
        enveloped `get`."""
        manager, client = _tools_by_name()
        asyncio.run(manager.call_tool("millm_export_circuit", {"circuit_id": "c1"}))
        # R2-10: `calls[0]`, not `[-1]`. R1-04 rejected last-call assertions
        # everywhere else and this one was left behind — a tool issuing a wrong
        # first call and a right second would have passed.
        assert len(client.calls) == 1
        assert client.calls[0][0] == "RAW_GET"


# ── Gate degradation ───────────────────────────────────────────────────────


class TestGateDegradation:
    """EC-20.1/20.3. A closed gate returns a STRUCTURED unavailable, never an
    exception and never an unregistration: an agent must be able to tell
    "miLLM is down" from "this tool does not exist"."""

    def test_a_closed_gate_returns_structured_unavailable(self):
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_circuits

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(False, "connection refused"))
        mcp = FastMCP("t")
        client = RecordingClient()
        millm_circuits.register(mcp, client, gate)

        result = asyncio.run(mcp._tool_manager.call_tool("millm_circuit_status", {}))
        payload = result[0] if isinstance(result, tuple) else result
        text = str(payload)
        assert "unavailable" in text and "millm" in text
        assert not client.calls, "a closed gate still issued the HTTP call"

    def test_the_UNGATED_tool_still_checks_the_gate(self):
        """F20 R1-06. `millm_import_circuit` omits `@gated` deliberately (so
        argument validation runs first) and checks the gate BY HAND — and
        nothing tested that hand-rolled check.

        A mutation disabling it SURVIVED the whole suite: the tool would have
        issued its HTTP call with miLLM known-down, turning a clean structured
        "unavailable" into a connection error the agent cannot classify.

        `test_argument_validation_runs_BEFORE_the_gate` below only exercises
        the BAD-argument path, which returns before reaching the gate at all —
        so the one tool that hand-rolls the gate was the one tool whose gate
        was unverified. That is this feature's own named anti-pattern, in the
        feature that exists to enforce against it.
        """
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_circuits

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(False, "connection refused"))
        mcp = FastMCP("t")
        client = RecordingClient()
        millm_circuits.register(mcp, client, gate)

        # VALID arguments, so validation passes and the gate is what must stop
        # it.
        result = asyncio.run(
            mcp._tool_manager.call_tool(
                "millm_import_circuit", {"definition": {"kind": "x"}}
            )
        )
        text = str(result)
        assert "unavailable" in text and "millm" in text, (
            "the hand-rolled gate did not refuse; the agent gets a connection "
            "error instead of a structured unavailable"
        )
        assert not client.calls, (
            "the tool issued its HTTP call with the gate CLOSED — miLLM is "
            "known-down and it called anyway"
        )

    def test_argument_validation_runs_BEFORE_the_gate(self):
        """An agent debugging its own payload must not be told "millm is
        down". A gate failure is about the SERVER; a bad argument is about the
        CALL, and conflating them sends the agent to fix the wrong thing."""
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_circuits

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(False, "connection refused"))
        mcp = FastMCP("t")
        millm_circuits.register(mcp, RecordingClient(), gate)

        result = asyncio.run(
            mcp._tool_manager.call_tool(
                "millm_import_circuit",
                {"definition": {"kind": "x"}, "on_conflict": "nonsense"},
            )
        )
        text = str(result)
        assert "on_conflict" in text, (
            "a bad argument was reported as miLLM being unavailable"
        )


class TestDestructiveScopeIsExplicit:
    """F20 R1-16. `millm_circuit_sensing_clear` defaulted to GLOBAL scope.

    An agent asked to "clear the events for this circuit" that omitted the
    argument wiped every observation in the deployment, irreversibly. The
    destructive path was the quiet one.
    """

    def test_an_unscoped_call_is_REFUSED(self):
        manager, client = _tools_by_name()
        result = asyncio.run(manager.call_tool("millm_circuit_sensing_clear", {}))
        assert "exactly one scope" in str(result)
        assert not client.calls, (
            "an unscoped clear reached the server — every recorded observation "
            "in the deployment would be gone"
        )

    def test_BOTH_scopes_at_once_is_refused(self):
        manager, client = _tools_by_name()
        result = asyncio.run(
            manager.call_tool(
                "millm_circuit_sensing_clear",
                {"circuit_id": "c1", "all_circuits": True},
            )
        )
        assert "exactly one scope" in str(result)
        assert not client.calls

    def test_explicit_all_circuits_IS_allowed(self):
        """The capability still exists — it just cannot be reached by
        omission."""
        manager, client = _tools_by_name()
        asyncio.run(
            manager.call_tool(
                "millm_circuit_sensing_clear", {"all_circuits": True}
            )
        )
        assert client.calls, "the deliberate global clear was blocked too"
        assert client.calls[0][0] == "DELETE"

    def test_the_DESTRUCTIVE_branch_sends_no_circuit_id(self):
        """F20 R2-05: `EXPECTED_CALLS` pins only the scoped call, so the branch
        that deletes EVERY observation in the deployment was asserted by
        nothing.

        Global scope is expressed to the server by OMITTING `circuit_id`. A
        future edit that passed it through would silently narrow the delete —
        or, re-pointed the other way, widen a scoped one. R1-16's fix was
        half-pinned: the refusal was tested, the destructive success path was
        not."""
        manager, client = _tools_by_name()
        asyncio.run(
            manager.call_tool(
                "millm_circuit_sensing_clear", {"all_circuits": True}
            )
        )
        method, path, payload = client.calls[0]
        assert (method, path) == ("DELETE", "/api/circuit-sensing/events")
        assert payload == {"circuit_id": None}, (
            f"the global clear sent {payload} — global scope is expressed by "
            "OMITTING circuit_id, so anything else changes what is deleted"
        )

    def test_scope_validation_runs_BEFORE_the_gate(self):
        """A scope mistake is about the CALL. Reporting it as "millm is down"
        sends the agent to fix the wrong thing."""
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_circuits

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(False, "connection refused"))
        mcp = FastMCP("t")
        millm_circuits.register(mcp, RecordingClient(), gate)

        client = RecordingClient()
        mcp2 = FastMCP("t2")
        millm_circuits.register(mcp2, client, gate)
        result = asyncio.run(
            mcp2._tool_manager.call_tool("millm_circuit_sensing_clear", {})
        )
        assert "exactly one scope" in str(result), (
            "a scope error was reported as miLLM being unavailable"
        )
        # R2-09: the import tool's ordering test asserts this and the
        # DESTRUCTIVE tool's did not — so a mutation returning the scope error
        # AFTER issuing the DELETE would have passed. The tool that can delete
        # everything got the weaker of the two tests.
        assert not client.calls, (
            "the scope error was returned but the DELETE went out first"
        )


class TestTheDescriptionsCarryWhatAnAgentNeeds:
    """F20 R1-17..20. The descriptions ARE the product — an agent reads them
    and relays them to a human.

    Review found the three cardinal semantics each reaching exactly ONE tool,
    while an agent typically calls one tool and sees only that third. And
    several tools stated a problem without its next step, which is how an agent
    ends up polling or guessing.

    These assert on the REGISTERED descriptions, not the source, because that
    is what `list_tools()` actually hands an agent.
    """

    def _descriptions(self):
        manager, _ = _tools_by_name()
        return {t.name: (t.description or "") for t in manager.list_tools()}

    def test_the_rung_GATE_is_stated_where_rungs_are_listed(self):
        """R1-17: an agent must know rung < 2 is refused BEFORE it tries to
        activate, and `millm_list_circuits` is where it sees rungs."""
        d = self._descriptions()["millm_list_circuits"]
        assert "GATED at 2" in d or "gated at 2" in d.lower()
        assert "never moves a rung" in d.lower() or "never raises" in d.lower(), (
            "the observation/validation boundary reaches only the sensing "
            "tools, and this is the tool an agent calls first"
        )

    def test_NO_ACTIVE_CIRCUIT_is_explained_where_it_is_hit(self):
        """R1-18: it is a 200 body, not an error, and the remedy is to
        activate something. An agent told neither will treat it as a fault."""
        d = self._descriptions()["millm_set_circuit_intensity"]
        assert "NO_ACTIVE_CIRCUIT" in d
        assert "AMBIGUOUS_ACTIVE_CIRCUIT" in d, (
            "R2-03 added a refusal for several serving circuits and the tool "
            "never mentioned it"
        )
        # R2-06: these surface as a TOOL ERROR, because the client raises on
        # any `success:false`. Describing them as bodies told the agent to
        # expect a normal result.
        assert "TOOL ERROR" in d

    def test_the_slice_dial_names_what_DOES_govern_intensity(self):
        """Telling an agent the write is inert without naming the alternative
        leaves it retrying the same call."""
        d = self._descriptions()["millm_set_circuit_intensity"]
        assert "cluster" in d.lower() and "0.5 floor" in d

    def test_enable_points_at_the_diagnostic(self):
        """R1-19: 'enabled but no events' has two causes with different
        remedies, and the tool that distinguishes them was never named."""
        d = self._descriptions()["millm_circuit_sensing_enable"]
        assert "millm_circuit_sensing_status" in d
        assert "Do not poll" in d

    def test_unsensable_edges_names_its_actionable_fields(self):
        d = self._descriptions()["millm_circuit_sensing_status"]
        assert "reason" in d and "detail" in d, (
            "the field is described as a list of names, so an agent relays a "
            "count and drops the half that tells the human what to do"
        )

    def test_status_warns_that_a_slice_is_not_the_whole_circuit(self):
        """R1-20: reporting a slice serve with the circuit's own rung claims
        evidence for something that is not being served."""
        d = self._descriptions()["millm_circuit_status"]
        assert "slice_fallback" in d
        assert "does NOT describe" in d or "not describe" in d.lower()

    def test_activate_covers_ALL_THREE_refusal_branches(self):
        """The tool an agent reaches for first must resolve each case, or it
        loops: overridable, collision (never retry), stuck claim (release)."""
        d = self._descriptions()["millm_activate_circuit"]
        assert "overridable" in d
        assert "NEVER retry" in d
        assert "millm_release_circuit_claims" in d
        assert "one model, one fixture" in d, (
            "the measured hazard's caveat must travel with it, or an agent "
            "relays a single-fixture result as a general law"
        )


class TestTheThreeFailureShapesAreDistinguishable:
    """F20 R2-07. `{"error": …}` (your call), `{"unavailable": …}` (server
    down) and a coded tool error (operation refused) need three DIFFERENT
    agent responses — and the vocabulary was documented in no tool description,
    only in a module docstring that `list_tools()` does not transport."""

    def test_the_first_tool_an_agent_meets_explains_them(self):
        manager, _ = _tools_by_name()
        d = {t.name: (t.description or "") for t in manager.list_tools()}
        status = d["millm_circuit_status"]
        assert '{"error"' in status and "unavailable" in status
        assert "REFUSED" in status


class TestDeletingAServingCircuitIsRefused:
    """F20 R2-20. `millm_delete_circuit` was the only irreversible operation in
    the module with no gating of any kind, while its sibling destructive tool
    (`millm_circuit_sensing_clear`) requires an explicit scope opt-in. The
    miLLM route deletes a live circuit unconditionally — it deactivates first
    and proceeds — so NOTHING anywhere in the stack stood between a mistyped
    id and production steering being torn down.

    Failure scenario: an agent asked to "clean up the old circuit" passes a
    stale id. Live steering stops, the definition is destroyed, and the first
    symptom is the served model quietly behaving differently.
    """

    async def _call(self, active_payload, **kwargs):
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_circuits

        class Client(RecordingClient):
            async def get(self, path, **params):
                self.calls.append(("GET", path, params))
                if path == "/api/circuits/active":
                    if isinstance(active_payload, Exception):
                        raise active_payload
                    return active_payload
                return {"success": True, "data": {}}

        mcp = FastMCP("t")
        client = Client()
        millm_circuits.register(mcp, client, _open_gate())
        fn = mcp._tool_manager._tools["millm_delete_circuit"].fn
        return await fn(circuit_id="c1", **kwargs), client

    @pytest.mark.asyncio
    async def test_a_serving_circuit_is_NOT_deleted(self):
        result, client = await self._call({"data": [{"id": "c1"}]})
        assert result["refused"] == "circuit_is_serving"
        assert not any(m == "DELETE" for m, _, _ in client.calls), (
            "the tool REFUSED and deleted anyway — the refusal was cosmetic"
        )

    @pytest.mark.asyncio
    async def test_the_refusal_says_how_to_proceed_and_how_to_keep_it(self):
        """A refusal an agent cannot act on becomes a retry loop or an
        abandoned task. It must name both the override AND the export, because
        the destructive half is unrecoverable."""
        result, _ = await self._call({"data": [{"id": "c1"}]})
        assert "acknowledge_serving=true" in result["reason"]
        assert "millm_export_circuit" in result["reason"]

    @pytest.mark.asyncio
    async def test_a_circuit_that_is_NOT_serving_deletes_without_ceremony(self):
        """The guard must not make ordinary cleanup require a flag."""
        _, client = await self._call({"data": [{"id": "other"}]})
        assert ("DELETE", "/api/circuits/c1", {}) in client.calls

    @pytest.mark.asyncio
    async def test_a_CLEAN_delete_carries_no_warning(self):
        """R3-03's warning must be rare enough to mean something. If every
        delete carried it, it would be scrolled past like every other banner
        and the one that mattered would go unread."""
        result, _ = await self._call({"data": [{"id": "other"}]})
        assert "guard_skipped" not in result
        assert "warning" not in result

    @pytest.mark.asyncio
    async def test_the_override_skips_the_check_entirely(self):
        _, client = await self._call(
            {"data": [{"id": "c1"}]}, acknowledge_serving=True
        )
        assert ("DELETE", "/api/circuits/c1", {}) in client.calls
        assert not any(
            p == "/api/circuits/active" for _, p, _ in client.calls
        ), "the override should not pay for a check whose answer it ignores"

    @pytest.mark.asyncio
    async def test_it_FAILS_OPEN_when_the_serving_state_cannot_be_read(self):
        """Deliberate, and the risky half of this design — so it is pinned.

        If /active is unreachable the delete PROCEEDS. The alternative makes
        cleanup impossible during exactly the outage when an operator most
        needs it. This is a guard against the plausible mistake (a stale id),
        not a lock against a determined caller.
        """
        result, client = await self._call(RuntimeError("boom"))
        assert ("DELETE", "/api/circuits/c1", {}) in client.calls

        # F20 R3-03: and it must SAY SO. R2-20 chose to fail open and then
        # said nothing — the response was byte-identical to a clean delete, so
        # the operator whose steering just stopped could not connect the two.
        # A guard that silently does not run is worse than no guard, because
        # the tool description promises it did.
        assert result["guard_skipped"] == "serving_state_unreadable"
        assert "WITHOUT confirming" in result["warning"]

    @pytest.mark.asyncio
    async def test_an_unrecognised_active_shape_is_UNKNOWN_not_not_serving(self):
        """`/active` returns a LIST (F19: several circuits serve at once). If
        it ever returns something else, the tool must not read that as
        'nothing is serving' — it fails open by the rule above, but via the
        UNKNOWN branch, and the distinction is what keeps a future shape
        change from silently disabling the guard."""
        result, client = await self._call({"data": {"id": "c1"}})
        assert ("DELETE", "/api/circuits/c1", {}) in client.calls
        assert result["guard_skipped"] == "serving_state_unreadable", (
            "an unrecognised /active shape is UNKNOWN, so this delete was "
            "also unguarded and must be reported as such"
        )


class TestEveryToolRespectsTheGate:
    """F20 R3-11/12. The gate was verified on ONE tool of twelve, and the
    hand-rolled variant on ONE of three.

    R1-06 found that `millm_import_circuit` omits `@gated` deliberately (so
    argument validation runs first) and checks the gate by hand, with nothing
    testing that hand-rolled check. The fix was applied to that tool alone. Two
    siblings hand-roll the same check — and they are the module's two
    DESTRUCTIVE tools:

      * `millm_delete_circuit`         — permanent deletion
      * `millm_circuit_sensing_clear`  — irreversible, and global in scope

    Deleting the `gate.check` block from either left the whole suite green
    while the tool issued its destructive call against a miLLM known to be
    down. Likewise `TestGateDegradation` used `millm_circuit_status` as sole
    representative, so eleven `@gated` tools could lose their decorator
    silently — turning a clean structured `unavailable` into an unclassifiable
    connection error.

    Parameterized over the LIVE registry: a tool added later is covered the
    moment it exists.
    """

    #: Minimal valid arguments per tool. A tool absent from here uses `{}`.
    ARGS = {
        "millm_activate_circuit": {"circuit_id": "c1"},
        "millm_deactivate_circuit": {"circuit_id": "c1"},
        "millm_delete_circuit": {"circuit_id": "c1"},
        "millm_export_circuit": {"circuit_id": "c1"},
        "millm_set_circuit_intensity": {"intensity": 1.0},
        "millm_import_circuit": {"definition": {"kind": "x"}},
        "millm_release_circuit_claims": {"circuit_id": "c1"},
        "millm_circuit_sensing_enable": {"circuit_id": "c1"},
        "millm_circuit_sensing_disable": {"circuit_id": "c1"},
        "millm_circuit_sensing_clear": {"circuit_id": "c1"},
        "millm_circuit_sensing_event": {"event_id": 1},
    }

    def _closed_gate_call(self, tool_name):
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_circuits

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(False, "connection refused"))
        mcp = FastMCP("t")
        client = RecordingClient()
        millm_circuits.register(mcp, client, gate)

        result = asyncio.run(
            mcp._tool_manager.call_tool(tool_name, self.ARGS.get(tool_name, {}))
        )
        payload = result[0] if isinstance(result, tuple) else result
        return str(payload), client

    @pytest.mark.parametrize(
        "tool_name",
        sorted(_tool_names_in(_millm_circuits_module())),
    )
    def test_a_closed_gate_issues_NO_http_call(self, tool_name):
        """The property that matters for the destructive tools: with miLLM
        known to be down, nothing is sent. A tool that reports `unavailable`
        AFTER issuing its DELETE has already done the damage."""
        _text, client = self._closed_gate_call(tool_name)
        assert not client.calls, (
            f"{tool_name} issued {client.calls} with the gate CLOSED. For a "
            "destructive tool this means the irreversible call went out while "
            "miLLM was known to be unreachable."
        )

    @pytest.mark.parametrize(
        "tool_name",
        sorted(_tool_names_in(_millm_circuits_module())),
    )
    def test_a_closed_gate_says_so_in_a_way_an_agent_can_classify(
        self, tool_name
    ):
        """An agent must distinguish "miLLM is down" (retry later) from "this
        tool does not exist" (never retry) and from "your arguments are wrong"
        (fix and retry). Silence, or a raw exception, collapses all three."""
        text, _client = self._closed_gate_call(tool_name)
        assert "unavailable" in text and "millm" in text, (
            f"{tool_name} with a closed gate returned {text[:200]!r}, which "
            "does not identify itself as a miLLM availability problem"
        )


class TestTheHowtoToolIsReachableAndHonest:
    """`mistudio_howto` carries the workflow knowledge tool signatures cannot.

    Written after an agent ran the full circuit loop and lost most of a session
    rediscovering things this server already knew — a `/v1` suffix that does
    not belong on `kind`, a nested member shape that reads as missing data, a
    feature join key that returns "0 labelled" for a fully-labelled corpus, and
    a strength scale ~50x off that shipped token soup to production.

    An auditing pass then found the deeper problem: 92 tools across 13
    categories, and SERVER_INSTRUCTIONS named 17 of them. FOUR ENTIRE
    CATEGORIES were unmentioned — including both circuit surfaces (35 tools).
    The whole causal-validation pipeline was reachable over MCP the entire
    time and nothing said so.

    Guidance an agent cannot reach is guidance that does not exist, so this
    asserts the LIVE registry, not the module.
    """

    def _built(self, monkeypatch):
        from src.mcp_server.config import MCPSettings
        from src.mcp_server.server import build_server

        settings = MCPSettings(tool_categories="read", allow_anonymous=True)
        mcp, _client = build_server(settings, stdio=True)
        return {t.name: t for t in asyncio.run(mcp.list_tools())}

    def test_it_reaches_the_built_server_in_a_DEFAULT_category(self, monkeypatch):
        """It ships with `read` deliberately: `read` is in DEFAULT_CATEGORIES,
        so every agent gets the guidance without opting in."""
        from src.mcp_server.config import DEFAULT_CATEGORIES

        assert "read" in DEFAULT_CATEGORIES, (
            "howto ships with `read` precisely because it is a default; if "
            "that changed, move howto to whatever is still default"
        )
        assert "mistudio_howto" in self._built(monkeypatch)

    def test_every_advertised_topic_actually_resolves(self):
        """The topic list is the tool's own index. A name it advertises but
        cannot return is a dead end an agent has no way to recover from."""
        from src.mcp_server.tools.howto import TOPICS

        assert len(TOPICS) >= 7, f"only {len(TOPICS)} topics — content lost?"
        advertised = set(re.findall(r"^  ([a-z_]+) ", TOPICS["overview"], re.M))
        # `tools` is DERIVED (from _all_tools()), not a prose entry in TOPICS —
        # it resolves in the dispatcher. Covered by
        # test_the_index_is_reachable_through_the_tool, which calls it for real.
        missing = sorted(advertised - set(TOPICS) - {"tools"})
        assert not missing, (
            f"the overview advertises {missing}, which resolve to nothing"
        )

    def test_the_discovery_pipeline_topic_names_only_REAL_tools(self):
        """Its value is telling an agent the pipeline exists. Naming a tool
        that does not exist would send it down a path with no exit — worse
        than saying nothing."""
        import inspect

        from src.mcp_server.tools import circuits as circuits_mod
        from src.mcp_server.tools.howto import TOPICS

        source = inspect.getsource(circuits_mod)
        real = set(re.findall(r"async def (\w+)", source))
        named = set(re.findall(r"`(\w+)`", TOPICS["discovery_pipeline"]))
        # Only check names that look like circuit tools; prose backticks other
        # things too (field names, statuses).
        claimed = {n for n in named if "circuit" in n or n in real}
        phantom = sorted(claimed - real)
        assert not phantom, (
            f"discovery_pipeline names {phantom}, which do not exist in the "
            "circuits tool module"
        )

    def test_the_tool_map_names_only_REAL_tools(self):
        """The map's value is telling an agent a capability exists. A phantom
        name sends it down a path with no exit — strictly worse than silence.

        Verified across the WHOLE registry, not one module: the map spans 13
        categories and a per-module check would miss cross-category typos.
        """
        import inspect

        from src.mcp_server.tools import CATEGORY_MODULES, MILLM_CATEGORY_MODULES
        from src.mcp_server.tools.howto import TOPICS

        real = set()
        for mods in {**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}.values():
            for module in mods:
                real |= set(re.findall(r"async def (\w+)", inspect.getsource(module)))
        assert len(real) > 50, f"only found {len(real)} tools — extraction broken"

        prefixes = ("get_", "list_", "run_", "steer_", "millm_", "save_",
                    "delete_", "update_", "find_", "compute_", "export_",
                    "import_", "enter_", "exit_", "cancel_", "start_",
                    "validate_", "promote_", "create_")
        named = set(re.findall(r"\b([a-z][a-z0-9_]{6,})\b", TOPICS["tool_map"]))
        claimed = {n for n in named
                   if (n in real or n.startswith(prefixes))
                   # `millm_circuit_* (13)` is a prose wildcard, not a name.
                   and not n.endswith("_")}
        phantom = sorted(claimed - real)
        assert not phantom, (
            f"tool_map names {phantom}, which are not registered anywhere. An "
            "agent told a tool exists will try to call it."
        )

    def test_EVERY_registered_tool_appears_in_the_generated_index(self):
        """The completeness guarantee, checked REGISTRY vs REGISTRY.

        This test used to build its expectation by regexing module source for
        `@mcp.tool()`. An adversarial pass showed why that is worthless: a tool
        registered by a helper (`mcp.tool()(fn)`) or in a loop is absent from
        the scanner AND from the index, so the two agreed and the guard passed
        while the tool was invisible everywhere. Comparing two views that share
        a blind spot proves nothing.

        Now both sides come from `build_server()` + `list_tools()` — the same
        call an agent makes. If the index and the server disagree about what
        exists, that IS the defect.
        """
        import asyncio as _asyncio
        import os as _os

        from src.mcp_server.config import MCPSettings, VALID_CATEGORIES
        from src.mcp_server.server import build_server
        from src.mcp_server.tools.howto import _all_tools

        _os.environ.setdefault("MILLM_API_URL", "http://millm.test")
        mcp, _c = build_server(
            MCPSettings(tool_categories=",".join(sorted(VALID_CATEGORIES)),
                        allow_anonymous=True),
            stdio=True,
        )
        served = {t.name for t in _asyncio.run(mcp.list_tools())}
        assert len(served) > 80, (
            f"only {len(served)} tools served — this guard is checking nothing"
        )

        index = _all_tools()
        indexed = {n for entries in index.values() for n, _ in entries}

        missing = sorted(served - indexed)
        assert not missing, (
            f"{len(missing)} tool(s) the server SERVES are absent from the "
            f"index and therefore invisible to an agent: {missing}"
        )

        # And the reverse: an index naming a tool the server does not serve
        # sends an agent after something that does not exist.
        phantom = sorted(indexed - served)
        assert not phantom, (
            f"the index names {phantom}, which the server does not serve"
        )

        # Presence is not coverage. A docstring-less tool was previously
        # indexed with an empty summary, so the metric read 93/93 while an
        # agent got a name and nothing else.
        blank = sorted(n for e in index.values() for n, d in e if not d.strip())
        assert not blank, (
            f"{blank} are indexed with an EMPTY summary — they read as covered "
            "and tell an agent nothing. Give them a docstring."
        )

    def test_the_ungated_destructive_tools_are_indexed(self):
        """Named explicitly because they are the ones a shape-sensitive
        extractor drops, and they are the ones that delete things."""
        from src.mcp_server.tools.howto import _all_tools

        indexed = {n for entries in _all_tools().values() for n, _ in entries}
        for tool in ("millm_delete_circuit", "millm_import_circuit",
                     "millm_circuit_sensing_clear"):
            assert tool in indexed, f"{tool} is missing from the index"

    def test_the_index_is_reachable_through_the_tool(self):
        """A derived index nothing returns is the dead code this replaced —
        `_all_tools()` existed for an hour before anything called it."""
        import asyncio as _asyncio

        from unittest.mock import MagicMock as _MM

        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import howto as _howto

        mcp = FastMCP("t")
        _howto.register(mcp, _MM(), _MM())
        fn = mcp._tool_manager._tools["mistudio_howto"].fn

        result = _asyncio.run(fn(topic="tools"))
        assert result.get("tool_count", 0) > 80, (
            "the tools topic returned no index — _all_tools() is unwired"
        )
        assert "circuits" in result["tools"]

        # And the no-arg response must ADVERTISE it, or an agent never asks.
        overview = _asyncio.run(fn())
        assert "tools" in overview["topics"]
        assert overview.get("tool_count", 0) > 80

    def test_EVERY_parameter_has_a_description(self):
        """Surface 3: the schema an agent actually reads when choosing arguments.

        Measured before this guard: 220 parameters across 93 tools, and 220 of
        them had NO description. An agent saw `granularity: string` with no
        valid values, `fdr_q: number` with no range, and `sae_id: string` with
        no hint which of two namespaces it meant. That is the same class of
        failure that had me guessing `kind` wrong and mixing up SAE ids.

        Enum-like strings must list their values, ids must name their
        namespace, and numerics that drive GPU cost must say so — but those are
        judgement calls a test cannot make. What a test CAN enforce is that
        nothing ships blank, which is what stops the surface silently
        regressing to where it started.
        """
        import asyncio as _asyncio
        import os as _os

        from src.mcp_server.config import MCPSettings, VALID_CATEGORIES
        from src.mcp_server.server import build_server

        _os.environ.setdefault("MILLM_API_URL", "http://millm.test")
        settings = MCPSettings(
            tool_categories=",".join(sorted(VALID_CATEGORIES)),
            allow_anonymous=True,
        )
        mcp, _client = build_server(settings, stdio=True)
        tools = _asyncio.run(mcp.list_tools())
        assert len(tools) > 80, (
            f"only {len(tools)} tools built — this guard is checking nothing"
        )

        bare = []
        total = 0
        for tool in tools:
            for pname, schema in (tool.inputSchema or {}).get("properties", {}).items():
                total += 1
                # F4: `"   "` is truthy. An auto-formatter or a stripped
                # placeholder produces exactly that, and it tells an agent
                # nothing while reading as covered.
                if not (schema.get("description") or "").strip():
                    bare.append(f"{tool.name}.{pname}")
        assert total > 150, f"only {total} parameters found — extraction broken"
        assert not bare, (
            f"{len(bare)} of {total} parameters have no description, so an "
            f"agent must guess their meaning: {sorted(bare)[:15]}"
            + (" …" if len(bare) > 15 else "")
            + "\n\nAdd Annotated[T, Field(description=...)]. For enum-like "
            "strings LIST THE VALUES; for ids NAME THE NAMESPACE; for numerics "
            "give the range and any GPU cost."
        )

    def test_the_server_instructions_point_at_it(self):
        """The instructions are the only thing an agent reads before choosing
        a tool. If they do not name howto, howto is undiscoverable."""
        from src.mcp_server.server import SERVER_INSTRUCTIONS

        assert "mistudio_howto" in SERVER_INSTRUCTIONS

        # A mention buried mid-paragraph is not a pointer. The instructions
        # must DIRECT the agent there, in the first few lines, or it will pick
        # a tool before it ever reads about the guidance.
        #
        # Control note: the first version of this test asserted only that the
        # name appeared somewhere, and a mutation that removed the actual
        # call-to-action SURVIVED — the name still occurred further down. A
        # test that cannot fail against the defect it names is not a test.
        head = SERVER_INSTRUCTIONS.split("\n\n")[1] if "\n\n" in SERVER_INSTRUCTIONS else ""
        assert "mistudio_howto" in head and "FIRST" in head, (
            "the instructions mention mistudio_howto but do not tell the "
            "agent to call it first — an agent reads the top and chooses a "
            "tool, so a buried mention is invisible in practice"
        )

    def test_the_instructions_name_the_discovery_pipeline_entry_point(self):
        """The specific omission that cost a session: the instructions
        described the CLUSTER path in detail and said nothing about circuit
        discovery, so an agent had no reason to think it existed."""
        from src.mcp_server.server import SERVER_INSTRUCTIONS

        for tool in ("start_circuit_capture", "validate_circuit_edges",
                     "export_circuit_definition"):
            assert tool in SERVER_INSTRUCTIONS, (
                f"{tool} is registered and reachable but the server "
                "instructions never mention it"
            )


class TestTheDEPLOYMENTEnablesWhatTheCodeRegisters:
    """The reachability rule applied to CONFIGURATION, not just code.

    F20 found 16 tools registered nowhere. This is the same defect one layer
    out: `circuits` IS in DEFAULT_CATEGORIES, and the k8s manifest set an
    explicit MCP_TOOL_CATEGORIES list that omitted it. All 19 circuit tools —
    plus the 16 millm_circuit_* tools — were implemented, registered, unit
    tested, documented in the generated contract, and switched OFF in
    production. An agent asking for them got "no such tool".

    Every guard in this file passes `tool_categories` in explicitly, so none of
    them could ever see this. A capability is not shipped until the DEPLOYMENT
    serves it.
    """

    def _manifests(self) -> dict:
        """`{path: categories}` for EVERY manifest setting MCP_TOOL_CATEGORIES.

        This checked ONE hardcoded path and I picked the wrong one: I fixed
        `k8s/mistudio-deployment.yaml`, watched the guard go green, and the
        change reached nothing — ArgoCD deploys `k8s/base` via kustomize, and
        `mistudio-deployment.yaml` is a legacy standalone manifest that
        `kustomization.yaml` does not reference. The guard confirmed a file
        nobody deploys.

        Discovered rather than named, so a manifest added later is covered and
        a stale one cannot drift out of sight. Every copy must agree: whichever
        one is live, the capability is on.
        """
        import re
        from pathlib import Path

        repo = Path(__file__).resolve().parents[3]
        pattern = re.compile(
            r"name:\s*MCP_TOOL_CATEGORIES\s*\n(?:\s*#[^\n]*\n)*\s*value:\s*\"([^\"]+)\""
        )
        found = {}
        for path in sorted((repo / "k8s").rglob("*.yaml")):
            m = pattern.search(path.read_text())
            if m:
                rel = str(path.relative_to(repo))
                found[rel] = {c.strip() for c in m.group(1).split(",") if c.strip()}
        assert found, (
            "no manifest under k8s/ sets MCP_TOOL_CATEGORIES — either the "
            "layout changed or the env var was renamed, and this guard is "
            "checking nothing"
        )
        return found

    def _deployed_categories(self) -> set:
        """The INTERSECTION across manifests: a category is only reliably
        enabled if every manifest that could be deployed enables it."""
        return set.intersection(*self._manifests().values())

    def test_every_DEFAULT_category_is_actually_deployed(self):
        """A category the code enables by default and the manifest disables is
        a capability that exists everywhere except where it is used."""
        from src.mcp_server.config import DEFAULT_CATEGORIES

        deployed = self._deployed_categories()
        defaults = {c.strip() for c in DEFAULT_CATEGORIES.split(",")}
        missing = sorted(defaults - deployed)
        assert not missing, (
            f"{missing} are in DEFAULT_CATEGORIES but the deployment's "
            f"explicit MCP_TOOL_CATEGORIES omits them, so their tools are "
            "unreachable in production while every other guard reports green."
        )

    def test_the_circuit_categories_are_deployed(self):
        """Named explicitly: these are the 35 tools that were off, and the
        whole causal-discovery pipeline lives in them."""
        deployed = self._deployed_categories()
        for category in ("circuits", "millm_circuits"):
            assert category in deployed, (
                f"{category} is not enabled in the deployment — its tools "
                "cannot be called by any agent"
            )

    def test_every_deployed_category_is_VALID(self):
        """A typo in the manifest silently drops a whole category rather than
        erroring, so the list must be checked against the real names."""
        from src.mcp_server.config import VALID_CATEGORIES

        unknown = sorted(self._deployed_categories() - set(VALID_CATEGORIES))
        assert not unknown, (
            f"the manifest enables {unknown}, which are not valid categories — "
            "the server ignores them and the intended tools stay off"
        )

    def test_admin_stays_OFF(self):
        """Specificity: this guard must not become 'enable everything'.

        `admin` is two irreversible deletes — delete_extraction takes every
        feature, label and activation example with it — and it is deliberately
        absent from DEFAULT_CATEGORIES. If it is ever enabled that should be a
        conscious decision with its own reasoning, not a side effect of a test
        that says 'turn it all on'.
        """
        assert "admin" not in self._deployed_categories(), (
            "admin (irreversible deletes) is enabled in the deployment — if "
            "that is intended, change this test deliberately and say why"
        )


# ── RCH-5: caller coverage across the WHOLE registry ───────────────────────
#
# MIS-E2E-119. `TestCallerReachability` above is complete within its scope, and
# that scope is one module: `_tools_by_name()` registers `millm_circuits` and
# nothing else, so `test_every_registered_tool_is_covered_here` compares 16
# against 16 and passes. The live server registers **116**.
#
# So the one shape that proves a tool does the RIGHT thing — rather than merely
# that it is exposed — covers 14% of the surface, and a tool added today gets
# registration coverage automatically and call coverage only if someone
# remembers. Re-pointing `get_circuit` to `/WRONG-PATH/{id}` left all 31
# reachability tests green (mutation M21).
#
# Closing it properly means a payload fixture per tool, which is a large piece
# of work and not one to fake. What this does instead is convert SILENCE into a
# LISTED DECISION: every registered tool must be either asserted or explicitly
# exempted with a reason, so the gap is visible, counted, and cannot grow
# without someone writing down why.

#: Tools with no caller assertion, and why. Adding a tool to this list is a
#: deliberate act that leaves a record; adding one to the server is not — which
#: is the whole difference this makes.
#:
#: **100 entries, generated from the live registry on 2026-08-23.** That number
#: is the finding, written down: `EXPECTED_CALLS` asserts the payload for 16
#: tools and the server exposes 116. Every line here is a tool whose ADDRESS is
#: verified (shapes 1 and 2 cover all 116) and whose LETTER is not — so
#: re-pointing its path, or changing what it sends, fails no test.
#:
#: Work it down rather than treating it as settled. A tool leaves this list by
#: gaining an `EXPECTED_CALLS` entry; the guard below then fails if it is in
#: both, or if it names a tool that no longer exists.
_CALLER_ASSERTION_EXEMPT: dict[str, str] = {
    # These five ARE payload-asserted, in tests/unit/test_labeling_trials_reachable_mcp.py,
    # which also covers the three pre-existing labeling tools. Exempt here to
    # avoid two competing sources of truth for the same contract, NOT because
    # the assertion is missing.
    "list_labeling_templates": "payload-asserted in test_labeling_trials_reachable_mcp.py",
    "run_labeling_trial": "payload-asserted in test_labeling_trials_reachable_mcp.py",
    "get_labeling_trial": "payload-asserted in test_labeling_trials_reachable_mcp.py",
    "list_labeling_trials": "payload-asserted in test_labeling_trials_reachable_mcp.py",
    "compare_labeling_trials": "payload-asserted in test_labeling_trials_reachable_mcp.py",
    # Payload-asserted in test_jlens_reachable_mcp.py (path AND no-body, both
    # load-bearing: the task id is a path segment, so a fixed path would cancel
    # nothing while looking right). Exempt here only because this harness builds
    # a server with the jlens category disabled.
    "cancel_jlens_task": "payload-asserted in test_jlens_reachable_mcp.py",
    "acquire_jlens_artifact": "no payload fixture yet — MIS-E2E-119 backlog",
    "annotate_jlens_feature": "no payload fixture yet — MIS-E2E-119 backlog",
    "build_circuit_from_discovery": "no payload fixture yet — MIS-E2E-119 backlog",
    "calibrate_circuit_strength": "no payload fixture yet — MIS-E2E-119 backlog",
    "cancel_steering_task": "no payload fixture yet — MIS-E2E-119 backlog",
    "compute_cluster_allocation": "no payload fixture yet — MIS-E2E-119 backlog",
    "compute_feature_groups": "no payload fixture yet — MIS-E2E-119 backlog",
    "compute_jlens_band_report": "no payload fixture yet — MIS-E2E-119 backlog",
    "create_circuit": "no payload fixture yet — MIS-E2E-119 backlog",
    "create_jlens_watchlist": "no payload fixture yet — MIS-E2E-119 backlog",
    "delete_circuit": "no payload fixture yet — MIS-E2E-119 backlog",
    "delete_experiment": "no payload fixture yet — MIS-E2E-119 backlog",
    "delete_extraction": "no payload fixture yet — MIS-E2E-119 backlog",
    "enter_steering_mode": "no payload fixture yet — MIS-E2E-119 backlog",
    "exit_steering_mode": "no payload fixture yet — MIS-E2E-119 backlog",
    "export_circuit_definition": "no payload fixture yet — MIS-E2E-119 backlog",
    "export_circuit_slices": "no payload fixture yet — MIS-E2E-119 backlog",
    "export_cluster_definition": "no payload fixture yet — MIS-E2E-119 backlog",
    "find_features_by_token": "no payload fixture yet — MIS-E2E-119 backlog",
    "find_related_features": "no payload fixture yet — MIS-E2E-119 backlog",
    "fit_jlens_artifact": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_circuit": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_cluster_profile": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_discovery_results": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_enhanced_label": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_experiment": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_extraction_summary": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_feature": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_feature_ablation": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_feature_correlations": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_feature_examples": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_feature_group_members": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_feature_groups": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_feature_logit_lens": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_feature_nlp_analysis": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_feature_token_analysis": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_grouping_status": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_jlens_band_report": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_jlens_gate": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_jlens_interventions": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_jlens_readout": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_jlens_replication_report": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_steering_mode": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_steering_result": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_steering_samples": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_task_status": "no payload fixture yet — MIS-E2E-119 backlog",
    "get_validation_manifest": "no payload fixture yet — MIS-E2E-119 backlog",
    "import_circuit_definition": "no payload fixture yet — MIS-E2E-119 backlog",
    "jlens_cost_estimate": "no payload fixture yet — MIS-E2E-119 backlog",
    "jlens_readout": "no payload fixture yet — MIS-E2E-119 backlog",
    "list_circuit_captures": "no payload fixture yet — MIS-E2E-119 backlog",
    "list_circuits": "no payload fixture yet — MIS-E2E-119 backlog",
    "list_cluster_profiles": "no payload fixture yet — MIS-E2E-119 backlog",
    "list_experiments": "no payload fixture yet — MIS-E2E-119 backlog",
    "list_extractions": "no payload fixture yet — MIS-E2E-119 backlog",
    "list_jlens_artifacts": "no payload fixture yet — MIS-E2E-119 backlog",
    "list_trainings": "no payload fixture yet — MIS-E2E-119 backlog",
    "list_validation_manifests": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_activate_cluster": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_activate_profile": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_deactivate_cluster": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_deactivate_profile": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_export_cluster": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_hub_search": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_import_cluster": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_list_clusters": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_list_profiles": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_sensing_config": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_sensing_disable": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_sensing_enable": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_sensing_events": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_sensing_status": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_set_intensity": "no payload fixture yet — MIS-E2E-119 backlog",
    "millm_status": "no payload fixture yet — MIS-E2E-119 backlog",
    "mistudio_howto": "no payload fixture yet — MIS-E2E-119 backlog",
    "preview_jlens_repo": "no payload fixture yet — MIS-E2E-119 backlog",
    "promote_circuit": "no payload fixture yet — MIS-E2E-119 backlog",
    "publish_jlens_artifact": "no payload fixture yet — MIS-E2E-119 backlog",
    "record_jlens_gate": "no payload fixture yet — MIS-E2E-119 backlog",
    "record_steering_samples": "no payload fixture yet — MIS-E2E-119 backlog",
    "reproduce_calibration": "no payload fixture yet — MIS-E2E-119 backlog",
    "reproduce_validation": "no payload fixture yet — MIS-E2E-119 backlog",
    "restore_jlens_artifact": "no payload fixture yet — MIS-E2E-119 backlog",
    "run_attribution_pass": "no payload fixture yet — MIS-E2E-119 backlog",
    "run_circuit_discovery": "no payload fixture yet — MIS-E2E-119 backlog",
    "run_circuit_faithfulness": "no payload fixture yet — MIS-E2E-119 backlog",
    "run_enhanced_labeling": "no payload fixture yet — MIS-E2E-119 backlog",
    "run_jlens_intervention": "no payload fixture yet — MIS-E2E-119 backlog",
    "save_cluster_profile": "no payload fixture yet — MIS-E2E-119 backlog",
    "save_experiment": "no payload fixture yet — MIS-E2E-119 backlog",
    "search_features": "no payload fixture yet — MIS-E2E-119 backlog",
    "start_circuit_capture": "no payload fixture yet — MIS-E2E-119 backlog",
    "steer_combined": "no payload fixture yet — MIS-E2E-119 backlog",
    "steer_compare": "no payload fixture yet — MIS-E2E-119 backlog",
    "steer_sweep": "no payload fixture yet — MIS-E2E-119 backlog",
    "steering_status": "no payload fixture yet — MIS-E2E-119 backlog",
    "update_circuit": "no payload fixture yet — MIS-E2E-119 backlog",
    "update_feature_label": "no payload fixture yet — MIS-E2E-119 backlog",
    "validate_circuit_edges": "no payload fixture yet — MIS-E2E-119 backlog",
    "validate_jlens_artifact": "no payload fixture yet — MIS-E2E-119 backlog",
}


def _all_registered_tool_names(monkeypatch) -> set[str]:
    """Every tool on the REAL built server, with EVERY category enabled.

    The real build path and its real signature — same as shape 2 above. A
    hand-called `register()` would prove the modules work and say nothing about
    what the server actually exposes, which is the defect this whole file
    exists to catch.
    """
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import CATEGORY_MODULES, MILLM_CATEGORY_MODULES

    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    every_category = ",".join(
        sorted(set(CATEGORY_MODULES) | set(MILLM_CATEGORY_MODULES))
    )
    settings = MCPSettings(tool_categories=every_category, allow_anonymous=True)
    mcp, _client = build_server(settings, stdio=True)
    return {t.name for t in asyncio.run(mcp.list_tools())}


class TestCallerCoverageIsAccounted:
    """Every tool is either payload-asserted or explicitly exempted."""

    def test_the_registry_is_not_empty(self, monkeypatch):
        """Negative control. A discovery step that discovers nothing turns
        every assertion below into a tautology — the exact way three guards in
        this audit failed OPEN."""
        names = _all_registered_tool_names(monkeypatch)
        assert len(names) > 50, (
            f"only {len(names)} tools discovered — the registry walk broke"
        )

    def test_every_tool_is_asserted_or_listed(self, monkeypatch):
        registered = _all_registered_tool_names(monkeypatch)
        accounted = set(EXPECTED_CALLS) | set(_CALLER_ASSERTION_EXEMPT)
        unaccounted = registered - accounted
        assert not unaccounted, (
            f"{len(unaccounted)} tools have neither a payload assertion nor a "
            f"listed exemption. Add an EXPECTED_CALLS entry, or add the name to "
            f"_CALLER_ASSERTION_EXEMPT with the reason — silence is what let "
            f"100 of 116 tools ship with the address verified and not the "
            f"letter:\n  " + "\n  ".join(sorted(unaccounted))
        )

    def test_the_exemption_list_has_no_stale_entries(self, monkeypatch):
        """An exemption for a tool that no longer exists hides a real gap by
        inflating the accounted set."""
        registered = _all_registered_tool_names(monkeypatch)
        stale = set(_CALLER_ASSERTION_EXEMPT) - registered
        assert not stale, f"exemptions for tools that no longer exist: {sorted(stale)}"

    def test_no_tool_is_both_asserted_and_exempted(self):
        """An exemption alongside an assertion is a contradiction, and the kind
        that quietly outlives the assertion it was meant to replace."""
        both = set(EXPECTED_CALLS) & set(_CALLER_ASSERTION_EXEMPT)
        assert not both, (
            f"these tools have a payload assertion AND an exemption: "
            f"{sorted(both)} — remove the exemption"
        )

    def test_no_exemption_carries_an_empty_reason(self):
        blank = [k for k, v in _CALLER_ASSERTION_EXEMPT.items() if not v.strip()]
        assert not blank, f"exemptions with no reason recorded: {blank}"
