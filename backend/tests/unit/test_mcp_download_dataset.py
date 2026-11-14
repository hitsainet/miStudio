"""`download_dataset`: a published dataset comes into miStudio over MCP (ENH-006 D116 G1).

The dry run of 2026-11-02 had to `POST /api/v1/datasets/download` with curl, because
`register_probe_dataset` needs a downloaded dataset and only a miDataworks detector-set send
downloaded one. The howto said no step needed a script.

Three shapes, each catching what the others cannot:

1. LIVE REGISTRY: the tool is on the server `build_server` builds with the categories the deployed
   manifest enables, by name. A list of names, not one derived from the decorators, because a
   derived list loses the tool together with its decorator.
2. PAYLOAD AND COUNT: `EXPECTED_CALLS["download_dataset"]` in `test_reachability.py` carries every
   argument; the omitted-argument body is pinned here.
3. THROUGH THE REAL ROUTE: the tool, on the built server, through the real `MiWorkbenchClient`, into the
   real FastAPI route and a real row. Only the Hub and Celery's `.delay` are faked. So "the pin
   carries through" is read off the task's arguments, and "the 409 is not a crash" is read off what
   an agent receives.

Controls are recorded in `0xcc/reviews/mcp_gaps_controls_2026-11-02.md`.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest
from sqlalchemy import func, select

from src.models.dataset import Dataset
from tests.support.fake_hub import COMMIT_A, COMMIT_B, FakeHub, install

TOOL = "download_dataset"


def _settings(categories: str = "probe_monitors,read"):
    from src.mcp_server.config import MCPSettings

    return MCPSettings(tool_categories=categories, allow_anonymous=True, _env_file=None)


def _built(categories: str = "probe_monitors,read"):
    from src.mcp_server.server import build_server

    return build_server(_settings(categories), stdio=True)


def _names(mcp) -> set[str]:
    return {t.name for t in asyncio.run(mcp.list_tools())}


class TestItIsInTheLiveRegistry:
    def test_it_is_on_the_server_the_DEPLOYED_manifest_builds(self):
        """Read from `k8s/base/mcp.yaml`: registered in code and switched off by the manifest is
        how 19 circuit tools were unreachable in production."""
        from tests.unit.test_reachability import TestTheDEPLOYMENTEnablesWhatTheCodeRegisters as T

        deployed = ",".join(sorted(T()._deployed_categories()))
        mcp, _ = _built(deployed)
        assert TOOL in _names(mcp)

    def test_it_is_on_by_default(self):
        """Compose and a bare run use DEFAULT_CATEGORIES."""
        from src.mcp_server.config import DEFAULT_CATEGORIES

        mcp, _ = _built(DEFAULT_CATEGORIES)
        assert TOOL in _names(mcp)

    def test_it_lives_with_the_view_that_needs_it(self):
        """Whichever switch turns `register_probe_dataset` on turns this on too: registering needs
        a downloaded dataset."""
        mcp, _ = _built("probe_monitors")
        assert {TOOL, "register_probe_dataset"} <= _names(mcp)

    def test_the_server_instructions_point_at_it(self):
        """What an agent reads before any tool. They walked straight from `list_datasets` to
        `register_probe_dataset`, as though a dataset could only already be there."""
        mcp, _ = _built()
        assert "download_dataset" in mcp.instructions

    def test_its_schema_is_exactly_the_design(self):
        """No token argument: a credential never passes through an agent transcript."""
        mcp, _ = _built()
        schema = next(t for t in asyncio.run(mcp.list_tools()) if t.name == TOOL).input_schema
        assert set(schema["properties"]) == {"repo_id", "config", "split", "revision"}
        assert schema["required"] == ["repo_id"]


class _Backend:
    """The real client over a transport, so the client's own error handling is in the path."""

    def __init__(self, mcp, studio, respond):
        self.calls: list[tuple[str, str, dict]] = []

        def record(request: httpx.Request) -> httpx.Response:
            self.calls.append((request.method, request.url.path,
                               json.loads(request.content or b"{}")))
            return respond(request)

        studio._http = httpx.AsyncClient(  # noqa: SLF001 - the tools hold this client
            transport=httpx.MockTransport(record), base_url="http://test/api/v1"
        )
        self.mcp = mcp

    def call(self, **arguments):
        from mcp import Client

        async def run():
            async with Client(self.mcp) as agent:
                return await agent.call_tool(TOOL, arguments)

        return asyncio.run(run())


class TestTheBodySent:
    def test_only_what_was_given_is_sent(self):
        """The route reads an absent `revision` as "the default branch's head, recorded". An
        empty string or a null key must not be what makes that so."""
        mcp, studio = _built()
        backend = _Backend(mcp, studio, lambda r: httpx.Response(202, json={"id": "d", "status": "downloading"}))
        result = backend.call(repo_id="org/ds")
        assert not result.is_error
        assert backend.calls == [("POST", "/api/v1/datasets/download", {"repo_id": "org/ds"})]

    def test_the_pin_goes_out_under_the_routes_own_name(self):
        """miDataworks reads the same property name from miStudio's OpenAPI to detect pinning."""
        mcp, studio = _built()
        backend = _Backend(mcp, studio, lambda r: httpx.Response(202, json={"id": "d"}))
        backend.call(repo_id="org/ds", config="en", split="test", revision="v1.0")
        assert backend.calls == [("POST", "/api/v1/datasets/download",
                                  {"repo_id": "org/ds", "config": "en", "split": "test", "revision": "v1.0"})]

    def test_it_never_asks_for_an_approval(self):
        """A download holds no card, so it follows `download_model`: one call, no request filed.
        A second call here would be the probe GPU approval's consume step."""
        mcp, studio = _built()
        backend = _Backend(mcp, studio, lambda r: httpx.Response(202, json={"id": "d"}))
        result = backend.call(repo_id="org/ds")
        assert len(backend.calls) == 1
        assert "approval_request_id" not in result.content[0].text


class TestThroughTheRealRoute:
    """Tool on the built server → real `MiWorkbenchClient` → the real route → a real row."""

    @pytest.fixture
    def dispatch(self):
        with patch("src.workers.dataset_tasks.download_dataset_task.delay") as delay:
            delay.return_value = MagicMock(id="task-1")
            yield delay

    @pytest.fixture
    def agent(self, client):
        """Call a tool the way an agent does: over the protocol, on the built server."""
        from mcp import Client

        mcp, studio = _built()
        studio._http = httpx.AsyncClient(  # noqa: SLF001
            transport=client._transport, base_url="http://test/api/v1",  # noqa: SLF001
            headers=dict(studio._http.headers),  # noqa: SLF001 - keeps the agent header
        )

        async def call(name, **arguments):
            async with Client(mcp) as session:
                return await session.call_tool(name, arguments)

        return call

    @pytest.mark.asyncio
    async def test_it_returns_the_new_row_at_once_and_get_dataset_polls_it(
        self, agent, async_session, monkeypatch, dispatch
    ):
        install(monkeypatch, FakeHub(default=COMMIT_A))
        result = await agent(TOOL, repo_id="org/ds", split="train")
        assert not result.is_error, result.content
        row = json.loads(result.content[0].text)
        assert row["status"] == "downloading"
        assert row["hf_repo_id"] == "org/ds"
        stored = (await async_session.execute(select(Dataset))).scalars().one()
        assert uuid.UUID(str(row["id"])) == stored.id

        polled = await agent("get_dataset", dataset_id=str(row["id"]))
        assert not polled.is_error, polled.content
        assert json.loads(polled.content[0].text)["id"] == row["id"]

    @pytest.mark.asyncio
    async def test_a_pinned_revision_reaches_the_row_AND_the_worker(self, agent, async_session, monkeypatch, dispatch):
        """The tag goes to the Hub; the COMMIT it resolved to goes on the row and to the task, so
        a branch moving between the call and the worker cannot change what is downloaded."""
        hub = install(monkeypatch, FakeHub(refs={"v1.0": COMMIT_B}))
        result = await agent(TOOL, repo_id="org/ds", config="en", split="test", revision="v1.0")
        assert not result.is_error, result.content
        assert [c["revision"] for c in hub.calls] == ["v1.0"]
        assert json.loads(result.content[0].text)["revision"] == COMMIT_B

        stored = (await async_session.execute(select(Dataset))).scalars().one()
        assert stored.revision == COMMIT_B
        dispatch.assert_called_once()
        sent = dispatch.call_args.kwargs
        assert (sent["repo_id"], sent["config"], sent["split"], sent["revision"], sent["access_token"]) == (
            "org/ds", "en", "test", COMMIT_B, None)

    @pytest.mark.asyncio
    async def test_an_existing_dataset_gets_the_routes_answer_and_is_not_a_crash(
        self, agent, async_session, monkeypatch, dispatch
    ):
        """The agent reads the route's own sentence, with the row to use. An exception the SDK
        does not recognise would reach it as the bare "Error executing tool download_dataset" with
        the sentence dropped (`test_mcp_errors_reach_the_agent.py`). Nothing new is created and
        nothing more is queued."""
        install(monkeypatch)
        first = await agent(TOOL, repo_id="org/ds", split="train")
        existing = json.loads(first.content[0].text)["id"]

        again = await agent(TOOL, repo_id="org/ds", split="train")
        assert again.is_error
        text = " ".join(block.text for block in again.content)
        assert text.endswith(f"backend 409: Dataset org/ds (split train) at {COMMIT_A} already exists "
                             f"(dataset {existing})"), text
        assert (await async_session.execute(select(func.count()).select_from(Dataset))).scalar_one() == 1
        dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_ref_that_does_not_resolve_is_refused_with_no_row(self, agent, async_session, monkeypatch, dispatch):
        response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/api/x"))
        from huggingface_hub.errors import RevisionNotFoundError

        install(monkeypatch, FakeHub(raises=lambda: RevisionNotFoundError("no such ref", response=response)))
        result = await agent(TOOL, repo_id="org/ds", revision="nope")
        assert result.is_error
        assert "backend 4" in result.content[0].text
        assert (await async_session.execute(select(func.count()).select_from(Dataset))).scalar_one() == 0
        dispatch.assert_not_called()
