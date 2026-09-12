"""
REACHABILITY for the model-management MCP tools (both products).

    A capability is not shipped until a test FAILS when its wiring is removed.

These tools exist because the workflow that needed them was unreachable over
MCP: "unload the current gguf, download and load the IQ4 version, test it."
Every step was a REST route and none was a tool, so the sequence had to be run
with `curl` inside the pod.

The three shapes from `test_reachability.py`, because each catches what the
others cannot:

  1. REGISTRY     — the category is in the maps that make it selectable
  2. BUILT SERVER — the REAL `build_server()` exposes the tools
  3. CALLER       — each tool issues its documented method and path, with the
                    PAYLOAD and the CALL COUNT asserted; "was called" passes
                    against a call sending the wrong arguments

MUTATION CONTROLS (each must turn this file red):
  * remove "models"/"millm_models" from CATEGORY_MODULES or
    MILLM_CATEGORY_MODULES        -> registry + built server fail
  * remove either from VALID_CATEGORIES   -> "selectable" fails
  * remove "models" from DEFAULT_CATEGORIES -> "on by default" fails
  * drop a @mcp.tool() decorator          -> built server fails
  * re-point a path or drop a body key    -> caller fails
  * drop the delete acknowledgement gate  -> the refusal tests fail
  * drop the context read from millm_list_models -> "reports the ACTUAL
    context" fails
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server.config import DEFAULT_CATEGORIES, VALID_CATEGORIES
from src.mcp_server.tools import CATEGORY_MODULES, MILLM_CATEGORY_MODULES

MISTUDIO_TOOLS = {
    "list_models",
    "get_model",
    "get_model_architecture",
    "download_model",
    "cancel_model_download",
    "delete_model",
}

MILLM_TOOLS = {
    "millm_list_models",
    "millm_get_model",
    "millm_preview_model_repo",
    "millm_download_model",
    "millm_cancel_download",
    "millm_load_model",
    "millm_unload_model",
    "millm_delete_model",
}


# ── Shape 1: registry ──────────────────────────────────────────────────────


class TestRegistry:
    def test_both_categories_are_in_the_module_registry(self):
        assert CATEGORY_MODULES.get("models"), "models registered with no modules"
        assert MILLM_CATEGORY_MODULES.get("millm_models"), "millm_models has no modules"

    def test_both_categories_are_selectable(self):
        """Absent here, MCP_TOOL_CATEGORIES=models RAISES at startup instead of
        enabling — which looks exactly like the tools not existing.

        VALID_CATEGORIES is hand-maintained and cannot be derived in config.py
        (tools imports config, so importing back would be a cycle), so this is
        the only thing holding the two lists together.
        """
        assert "models" in VALID_CATEGORIES
        assert "millm_models" in VALID_CATEGORIES

    def test_miStudio_models_is_on_by_default(self):
        """These tools talk only to miStudio's own API. Gating them behind an
        explicit opt-in makes them unreachable for any agent that does not
        already know to ask."""
        assert "models" in {c.strip() for c in DEFAULT_CATEGORIES.split(",")}

    def test_millm_models_is_NOT_on_by_default(self):
        """Every millm_* category needs a second product running. On by default
        it would register tools that answer `unavailable` on a deployment that
        has no miLLM at all."""
        assert "millm_models" not in {c.strip() for c in DEFAULT_CATEGORIES.split(",")}

    def test_the_category_lists_agree(self):
        """The guard config.py's comment points at.

        VALID_CATEGORIES is a second copy of the registry. A category present
        in one and missing from the other is either a tool nobody can enable or
        a category name that enables nothing — both silent.
        """
        registered = set(CATEGORY_MODULES) | set(MILLM_CATEGORY_MODULES)
        assert registered == VALID_CATEGORIES, (
            "VALID_CATEGORIES and the module registries disagree: "
            f"only registered={sorted(registered - VALID_CATEGORIES)}, "
            f"only valid={sorted(VALID_CATEGORIES - registered)}"
        )

    def test_the_modules_expose_register(self):
        for module in CATEGORY_MODULES["models"] + MILLM_CATEGORY_MODULES["millm_models"]:
            assert hasattr(module, "register")


# ── Shape 2: built server ──────────────────────────────────────────────────


class TestBuiltServer:
    """The REAL build_server(), not a hand-called register()."""

    def _build(self, categories: str, monkeypatch):
        from src.mcp_server.config import MCPSettings
        from src.mcp_server.server import build_server

        monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
        settings = MCPSettings(tool_categories=categories, allow_anonymous=True)
        mcp, _client = build_server(settings, stdio=True)
        return {t.name for t in asyncio.run(mcp.list_tools())}

    def test_build_server_exposes_every_miStudio_model_tool(self, monkeypatch):
        names = self._build("models", monkeypatch)
        for expected in MISTUDIO_TOOLS:
            assert expected in names, (
                f"{expected} is not reachable from build_server() — "
                "registration exists but nothing reaches it"
            )

    def test_build_server_exposes_every_miLLM_model_tool(self, monkeypatch):
        names = self._build("millm_models", monkeypatch)
        for expected in MILLM_TOOLS:
            assert expected in names, f"{expected} is not reachable"

    def test_the_tools_are_absent_when_the_categories_are_not_enabled(self, monkeypatch):
        """Specificity: without this, the tests above prove nothing about wiring."""
        names = self._build("read", monkeypatch)
        assert not ((MISTUDIO_TOOLS | MILLM_TOOLS) & names)

    def test_millm_model_tools_are_skipped_with_no_miLLM_configured(self, monkeypatch):
        """A deployment with no miLLM must not advertise tools that can only
        answer `unavailable`."""
        from src.mcp_server.config import MCPSettings
        from src.mcp_server.server import build_server

        monkeypatch.setenv("MILLM_API_URL", "")
        settings = MCPSettings(
            tool_categories="models,millm_models", allow_anonymous=True
        )
        mcp, _client = build_server(settings, stdio=True)
        names = {t.name for t in asyncio.run(mcp.list_tools())}

        assert MISTUDIO_TOOLS <= names, "miStudio's own tools must be unaffected"
        assert not (MILLM_TOOLS & names)

    def test_the_default_configuration_reaches_the_miStudio_tools(self, monkeypatch):
        names = self._build(DEFAULT_CATEGORIES, monkeypatch)
        assert MISTUDIO_TOOLS <= names


class TestTheDeploymentEnablesThem:
    """Registration in code is necessary and NOT sufficient.

    `k8s/base/mcp.yaml` sets MCP_TOOL_CATEGORIES explicitly, and an explicit
    list OVERRIDES DEFAULT_CATEGORIES. That is how 19 circuit tools and 16
    millm_circuit_* tools stayed unreachable in production while registered,
    tested and documented.
    """

    def _deployed_categories(self):
        import pathlib
        import re

        manifest = (
            pathlib.Path(__file__).resolve().parents[3] / "k8s" / "base" / "mcp.yaml"
        )
        assert manifest.exists(), f"manifest not found at {manifest}"
        text = manifest.read_text()
        match = re.search(
            r"name:\s*MCP_TOOL_CATEGORIES\b.*?value:\s*\"([^\"]+)\"", text, re.S
        )
        assert match, (
            "MCP_TOOL_CATEGORIES not found in k8s/base/mcp.yaml — this guard "
            "reads the manifest, and a scrape that matches nothing asserts "
            "nothing"
        )
        return {c.strip() for c in match.group(1).split(",") if c.strip()}

    def test_the_scrape_finds_a_plausible_list(self):
        """Negative control: a source scrape that matches nothing FAILS OPEN,
        which has happened twice in this repo — including inside a reachability
        guard."""
        deployed = self._deployed_categories()
        assert len(deployed) >= 10, f"only found {deployed} — the scrape broke"
        assert "read" in deployed

    @pytest.mark.parametrize("category", ["models", "millm_models"])
    def test_the_deployment_enables_the_category(self, category):
        assert category in self._deployed_categories(), (
            f"{category} is registered in code but absent from "
            "k8s/base/mcp.yaml's explicit MCP_TOOL_CATEGORIES, so every tool "
            "in it is invisible in production"
        )


# ── Shape 3: caller ────────────────────────────────────────────────────────


def _mistudio_tools():
    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import models

    mcp = FastMCP("test")
    client = MagicMock()
    client.get = AsyncMock(return_value={"success": True, "data": {}})
    client.post = AsyncMock(return_value={"success": True, "data": {}})
    client.delete = AsyncMock(return_value={"success": True})
    models.register(mcp, client, MCPSettings(allow_anonymous=True))
    return mcp, client


def _millm_tools(get_return=None):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("test")
    client = MagicMock()
    client.get = AsyncMock(return_value=get_return or {"data": []})
    client.post = AsyncMock(return_value={"data": {}})
    client.delete = AsyncMock(return_value={"success": True})
    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, None))

    from src.mcp_server.tools import millm_models

    millm_models.register(mcp, client, gate)
    return mcp, client


class TestMiStudioCallers:
    def test_list_models_issues_a_GET_with_its_limit(self):
        mcp, client = _mistudio_tools()
        asyncio.run(mcp.call_tool("list_models", {}))

        assert client.get.await_count == 1
        assert client.get.await_args.args[0] == "/models"
        assert client.get.await_args.kwargs == {"limit": 50}

    def test_list_models_forwards_every_filter_it_is_given(self):
        """PAYLOAD, not just address: a dropped filter returns the whole
        catalogue and looks like a working call."""
        mcp, client = _mistudio_tools()
        asyncio.run(
            mcp.call_tool(
                "list_models",
                {"search": "gemma", "status": "ready", "architecture": "gemma3", "limit": 10},
            )
        )

        assert client.get.await_args.kwargs == {
            "limit": 10,
            "search": "gemma",
            "status": "ready",
            "architecture": "gemma3",
        }

    def test_get_model_addresses_the_row(self):
        mcp, client = _mistudio_tools()
        asyncio.run(mcp.call_tool("get_model", {"model_id": "m_abc"}))

        assert client.get.await_count == 1
        assert client.get.await_args.args[0] == "/models/m_abc"

    def test_architecture_addresses_the_architecture_subresource(self):
        mcp, client = _mistudio_tools()
        asyncio.run(mcp.call_tool("get_model_architecture", {"model_id": "m_abc"}))

        assert client.get.await_args.args[0] == "/models/m_abc/architecture"

    def test_download_posts_the_whole_body(self):
        """A dropped `quantization` silently downloads at the API default,
        which is a different model from the one that was asked for."""
        mcp, client = _mistudio_tools()
        asyncio.run(
            mcp.call_tool(
                "download_model",
                {"repo_id": "google/gemma-2-2b", "quantization": "Q8",
                 "trust_remote_code": True},
            )
        )

        assert client.post.await_count == 1
        assert client.post.await_args.args[0] == "/models/download"
        assert client.post.await_args.kwargs == {
            "json_body": {
                "repo_id": "google/gemma-2-2b",
                "quantization": "Q8",
                "trust_remote_code": True,
            }
        }

    def test_a_token_is_sent_only_when_given(self):
        mcp, client = _mistudio_tools()
        asyncio.run(
            mcp.call_tool("download_model", {"repo_id": "a/b", "access_token": "t"})
        )
        assert client.post.await_args.kwargs["json_body"]["access_token"] == "t"

    def test_cancel_addresses_the_cancel_subresource(self):
        mcp, client = _mistudio_tools()
        asyncio.run(mcp.call_tool("cancel_model_download", {"model_id": "m_abc"}))

        assert client.delete.await_count == 1
        assert client.delete.await_args.args[0] == "/models/m_abc/cancel"


class TestMiStudioDeleteIsGated:
    """The gate is the point of the tool, so it is asserted before the call."""

    def test_an_unacknowledged_delete_issues_NO_call(self):
        mcp, client = _mistudio_tools()
        asyncio.run(mcp.call_tool("delete_model", {"model_id": "m_abc"}))

        assert client.delete.await_count == 0, (
            "the model was deleted without acknowledgement — the gate is the "
            "only thing between a stale id and an irreversible delete"
        )

    def test_the_refusal_names_what_would_be_destroyed(self):
        """A refusal that does not say WHICH model leaves the agent no way to
        tell a stale id from the right one."""
        mcp, client = _mistudio_tools()
        client.get = AsyncMock(
            return_value={"data": {"name": "gemma-2-2b", "size_mb": 5120,
                                   "status": "ready"}}
        )
        result = asyncio.run(mcp.call_tool("delete_model", {"model_id": "m_abc"}))

        text = str(result)
        assert "gemma-2-2b" in text
        assert "5.0 GB" in text, f"the size was not reported: {text}"
        assert "delete_not_acknowledged" in text

    def test_the_refusal_still_works_when_the_row_cannot_be_read(self):
        """FAILS OPEN on the lookup, not on the gate: an outage must not make
        the tool unusable, and must not make it delete silently either."""
        mcp, client = _mistudio_tools()
        client.get = AsyncMock(side_effect=RuntimeError("backend down"))

        result = asyncio.run(mcp.call_tool("delete_model", {"model_id": "m_abc"}))

        assert "delete_not_acknowledged" in str(result)
        assert client.delete.await_count == 0

    def test_an_acknowledged_delete_issues_exactly_one_DELETE(self):
        mcp, client = _mistudio_tools()
        asyncio.run(
            mcp.call_tool(
                "delete_model", {"model_id": "m_abc", "acknowledge_permanent": True}
            )
        )

        assert client.delete.await_count == 1
        assert client.delete.await_args.args[0] == "/models/m_abc"
        assert client.get.await_count == 0, (
            "an acknowledged delete should not re-read the row it is about to "
            "destroy"
        )


class TestMiLLMCallers:
    def test_list_addresses_the_models_collection(self):
        mcp, client = _millm_tools()
        asyncio.run(mcp.call_tool("millm_list_models", {}))

        assert client.get.await_count == 1
        assert client.get.await_args.args[0] == "/api/models"

    def test_preview_posts_the_repo_id(self):
        mcp, client = _millm_tools()
        asyncio.run(
            mcp.call_tool("millm_preview_model_repo", {"repo_id": "a/b-GGUF"})
        )

        assert client.post.await_count == 1
        assert client.post.await_args.args[0] == "/api/models/preview"
        assert client.post.await_args.kwargs == {"json_body": {"repo_id": "a/b-GGUF"}}

    def test_download_carries_the_chosen_quantization_and_its_files(self):
        """The whole reason the picker exists. Dropping `gguf_files` downloads
        EVERY quantization in the repo — 121 GB measured, against 2.78 GB for
        the one that was chosen."""
        mcp, client = _millm_tools()
        asyncio.run(
            mcp.call_tool(
                "millm_download_model",
                {
                    "repo_id": "m/gemma-GGUF",
                    "quantization": "Q4",
                    "gguf_label": "IQ4_XS",
                    "gguf_files": ["gemma.IQ4_XS.gguf"],
                },
            )
        )

        assert client.post.await_count == 1
        assert client.post.await_args.args[0] == "/api/models/download"
        assert client.post.await_args.kwargs == {
            "json_body": {
                "source": "huggingface",
                "repo_id": "m/gemma-GGUF",
                "quantization": "Q4",
                "trust_remote_code": False,
                "gguf_label": "IQ4_XS",
                "gguf_files": ["gemma.IQ4_XS.gguf"],
            }
        }

    def test_an_ordinary_download_sends_no_gguf_keys(self):
        """`gguf_files: null` is not the same as absent to the validator, and
        an empty list is REFUSED — it would match nothing and report success
        over an empty directory."""
        mcp, client = _millm_tools()
        asyncio.run(mcp.call_tool("millm_download_model", {"repo_id": "g/gemma-2-2b"}))

        body = client.post.await_args.kwargs["json_body"]
        assert "gguf_files" not in body and "gguf_label" not in body

    def test_load_addresses_the_load_subresource(self):
        mcp, client = _millm_tools()
        asyncio.run(mcp.call_tool("millm_load_model", {"model_id": 44}))

        assert client.post.await_count == 1
        assert client.post.await_args.args[0] == "/api/models/44/load"

    def test_unload_addresses_the_unload_subresource(self):
        mcp, client = _millm_tools()
        asyncio.run(mcp.call_tool("millm_unload_model", {"model_id": 44}))

        assert client.post.await_count == 1
        assert client.post.await_args.args[0] == "/api/models/44/unload"

    def test_cancel_addresses_the_cancel_subresource(self):
        mcp, client = _millm_tools()
        asyncio.run(mcp.call_tool("millm_cancel_download", {"model_id": 44}))

        assert client.post.await_args.args[0] == "/api/models/44/cancel"

    def test_get_addresses_the_row(self):
        mcp, client = _millm_tools(get_return={"data": {"id": 44}})
        asyncio.run(mcp.call_tool("millm_get_model", {"model_id": 44}))

        assert client.get.await_count == 1
        assert client.get.await_args.args[0] == "/api/models/44"


class TestMiLLMListReportsTheActualContext:
    """A GGUF model loads at the largest context that FITS, not the one asked
    for: 8192 requested, 4096 obtained, on both quantizations tried on this
    card. A 4703-token prompt against a 4096 window is the next thing a user
    hits, and nothing else on the system explains it.
    """

    def test_the_resident_model_carries_the_live_context(self):
        mcp, client = _millm_tools()

        async def _get(path, **params):
            if path == "/api/health/inference":
                return {"context_length": 4096}
            return {"data": [{"id": 44, "name": "g:IQ4_XS", "status": "loaded"}]}

        client.get = AsyncMock(side_effect=_get)
        result = asyncio.run(mcp.call_tool("millm_list_models", {}))

        assert "4096" in str(result), (
            "the actual context window was not reported — the requested one is "
            "not what the engine got"
        )

    def test_a_model_merely_on_disk_is_not_given_a_context(self):
        """It is a property of the running engine. Attaching it to a row that
        is not loaded would state a window nothing has measured."""
        mcp, client = _millm_tools()

        async def _get(path, **params):
            if path == "/api/health/inference":
                return {"context_length": 4096}
            return {"data": [{"id": 43, "name": "g:Q4_K_M", "status": "ready"}]}

        client.get = AsyncMock(side_effect=_get)
        result = asyncio.run(mcp.call_tool("millm_list_models", {}))

        assert "context_length" not in str(result)

    def test_nothing_resident_means_no_second_call(self):
        mcp, client = _millm_tools(get_return={"data": [{"id": 1, "status": "ready"}]})
        asyncio.run(mcp.call_tool("millm_list_models", {}))

        assert client.get.await_count == 1, (
            "the engine was asked for a context window with nothing loaded"
        )

    def test_an_unreadable_context_is_reported_as_unknown(self):
        """Silence would read as 'no limit', which is the opposite of true."""
        mcp, client = _millm_tools()

        async def _get(path, **params):
            if path == "/api/health/inference":
                raise RuntimeError("engine not answering")
            return {"data": [{"id": 44, "status": "loaded"}]}

        client.get = AsyncMock(side_effect=_get)
        result = asyncio.run(mcp.call_tool("millm_list_models", {}))

        assert "unknown" in str(result).lower()


class TestDownloadProgressIsHonest:
    """`download_progress` is written in-process and reads null from another
    worker. A client rendering null as 0% shows a stalled bar over a healthy
    download, and the two cases need opposite reactions."""

    def test_unknown_progress_is_distinguished_from_no_progress(self):
        mcp, client = _millm_tools(
            get_return={"data": [{"id": 1, "status": "downloading",
                                  "download_progress": None}]}
        )
        result = asyncio.run(mcp.call_tool("millm_list_models", {}))

        assert "UNKNOWN, not zero" in str(result)

    def test_a_real_percentage_is_left_alone(self):
        mcp, client = _millm_tools(
            get_return={"data": [{"id": 1, "status": "downloading",
                                  "download_progress": 42}]}
        )
        result = asyncio.run(mcp.call_tool("millm_list_models", {}))

        assert "UNKNOWN, not zero" not in str(result)

    def test_a_ready_model_gets_no_progress_note(self):
        mcp, client = _millm_tools(
            get_return={"data": [{"id": 1, "status": "ready",
                                  "download_progress": None}]}
        )
        result = asyncio.run(mcp.call_tool("millm_list_models", {}))

        assert "progress_note" not in str(result)


class TestMiLLMDeleteIsGated:
    def test_an_unacknowledged_delete_issues_NO_delete(self):
        mcp, client = _millm_tools(get_return={"data": {"name": "g", "status": "ready"}})
        asyncio.run(mcp.call_tool("millm_delete_model", {"model_id": 43}))

        assert client.delete.await_count == 0

    def test_the_refusal_names_the_model_its_label_and_its_size(self):
        mcp, client = _millm_tools(
            get_return={"data": {"name": "gemma-GGUF:IQ4_XS", "gguf_label": "IQ4_XS",
                                 "disk_size_mb": 16081, "status": "ready"}}
        )
        result = asyncio.run(mcp.call_tool("millm_delete_model", {"model_id": 44}))

        text = str(result)
        assert "gemma-GGUF:IQ4_XS" in text
        assert "IQ4_XS" in text
        assert "15.7 GB" in text, f"the size was not reported: {text}"

    def test_the_refusal_survives_an_unreadable_row(self):
        mcp, client = _millm_tools()
        client.get = AsyncMock(side_effect=RuntimeError("down"))

        result = asyncio.run(mcp.call_tool("millm_delete_model", {"model_id": 44}))

        assert "delete_not_acknowledged" in str(result)
        assert client.delete.await_count == 0

    def test_an_acknowledged_delete_issues_exactly_one_DELETE(self):
        mcp, client = _millm_tools()
        asyncio.run(
            mcp.call_tool(
                "millm_delete_model",
                {"model_id": 44, "acknowledge_permanent": True},
            )
        )

        assert client.delete.await_count == 1
        assert client.delete.await_args.args[0] == "/api/models/44"


class TestTheGateDegradesGracefully:
    """miLLM being down must yield a structured answer, not an exception —
    tools are never unregistered (contract §3)."""

    @pytest.mark.parametrize("tool_name", sorted(MILLM_TOOLS))
    def test_every_millm_model_tool_reports_unavailable(self, tool_name):
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_models

        mcp = FastMCP("test")
        client = MagicMock()
        client.get = AsyncMock(return_value={"data": []})
        client.post = AsyncMock(return_value={"data": {}})
        client.delete = AsyncMock(return_value={})
        gate = MagicMock()
        gate.check = AsyncMock(return_value=(False, "connection refused"))
        millm_models.register(mcp, client, gate)

        args = {
            "millm_list_models": {},
            "millm_get_model": {"model_id": 1},
            "millm_preview_model_repo": {"repo_id": "a/b"},
            "millm_download_model": {"repo_id": "a/b"},
            "millm_cancel_download": {"model_id": 1},
            "millm_load_model": {"model_id": 1},
            "millm_unload_model": {"model_id": 1},
            "millm_delete_model": {"model_id": 1},
        }[tool_name]

        result = asyncio.run(mcp.call_tool(tool_name, args))

        assert "unavailable" in str(result)
        assert client.delete.await_count == 0
        assert client.post.await_count == 0


class TestAManifestAheadOfItsImageDoesNotCrashTheServer:
    """The incident this deploy actually caused, reproduced.

    `k8s/base/mcp.yaml` and the backend image ship on DIFFERENT SCHEDULES:
    ArgoCD syncs the manifest within minutes, the image takes about nine. For
    that gap the new manifest runs against an image whose VALID_CATEGORIES
    predates it. `enabled_categories()` raised, so the MCP server crashlooped —
    six restarts on 2026-09-07, and only the rolling update kept a healthy pod
    serving. Had the old pod been evicted in that window the server would have
    been down outright, for a config that becomes valid on its own.

    Simulated by asking for a category name that does not exist, which is
    precisely what `models` WAS to the previous image.

    MUTATION CONTROLS:
      * restore the raise in enabled_categories -> "comes up" fails
      * honour unknown names instead of dropping -> "enables nothing" fails
      * drop unknown_categories from /health     -> "is reported" fails
    """

    def test_the_server_comes_up_with_the_categories_it_understands(self, monkeypatch):
        from src.mcp_server.config import MCPSettings
        from src.mcp_server.server import build_server

        monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
        settings = MCPSettings(
            tool_categories="read,models,a_category_from_a_later_release",
            allow_anonymous=True,
        )

        mcp, _client = build_server(settings, stdio=True)
        names = {t.name for t in asyncio.run(mcp.list_tools())}

        assert MISTUDIO_TOOLS <= names, (
            "the categories this build DOES understand must still register"
        )

    def test_the_unknown_category_enables_nothing(self):
        """Fail-closed where it matters. Dropping the crash must not become
        'accept anything'."""
        from src.mcp_server.config import MCPSettings

        cats = MCPSettings(
            tool_categories="read,a_category_from_a_later_release",
            allow_anonymous=True,
        ).enabled_categories()

        assert cats == {"read"}

    def test_the_dropped_name_is_reported_on_health(self, monkeypatch):
        """Silence is how a whole category goes missing unnoticed."""
        import json

        from starlette.testclient import TestClient

        from src.mcp_server.config import MCPSettings
        from src.mcp_server.server import build_server

        monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
        settings = MCPSettings(
            tool_categories="read,models,a_category_from_a_later_release",
            allow_anonymous=True,
        )
        mcp, _client = build_server(settings, stdio=True)

        with TestClient(mcp.streamable_http_app()) as client:
            body = json.loads(client.get("/health").content)

        assert body["unknown_categories"] == ["a_category_from_a_later_release"]
        assert "models" in body["categories"]

    def test_health_stays_quiet_when_every_name_is_understood(self, monkeypatch):
        """Specificity: a key that is always present says nothing."""
        import json

        from starlette.testclient import TestClient

        from src.mcp_server.config import MCPSettings
        from src.mcp_server.server import build_server

        monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
        settings = MCPSettings(tool_categories="read,models", allow_anonymous=True)
        mcp, _client = build_server(settings, stdio=True)

        with TestClient(mcp.streamable_http_app()) as client:
            body = json.loads(client.get("/health").content)

        assert "unknown_categories" not in body
